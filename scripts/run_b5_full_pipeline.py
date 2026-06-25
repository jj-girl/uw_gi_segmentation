from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/disk2/hjj/uw_gi_segmentation")
PYTHON = Path("/mnt/disk2/hjj/uwgiseg/bin/python")
RUN_DIR = ROOT / "outputs/b5_full_pipeline"
LOG_PATH = RUN_DIR / "pipeline.log"
STATUS_PATH = RUN_DIR / "status.json"

B5_GLOB = "configs/h200_next_unetpp_b5_folds/h200_next_unetpp_b5_fold*.yaml"
B5_MAIN_CONFIG = "configs/h200_next_unetpp_b5.yaml"
B5_OOF_DIR = Path("outputs/h200_next_unetpp_b5_oof")
B5_OOF_REPORT = B5_OOF_DIR / "h200_next_unetpp_b5_auto_report.md"

STRATEGY_E_GLOB = "configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(message: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        stat = subprocess.check_output(["ps", "-p", str(pid), "-o", "stat="], text=True).strip()
        if not stat or stat.startswith("Z"):
            return False
    except subprocess.CalledProcessError:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return int(text) if text.isdigit() else None


def fold_dir(fold: int) -> Path:
    return ROOT / f"outputs/h200_next_unetpp_b5_fold{fold}"


def latest_metric(fold: int) -> str:
    path = fold_dir(fold) / "train.log"
    if not path.exists():
        return "no train.log"
    lines = path.read_bytes().decode("utf-8", "ignore").replace("\r", "\n").splitlines()
    metrics = [line.strip() for line in lines if line.strip().startswith("epoch=")]
    return metrics[-1] if metrics else "training epoch in progress"


def checkpoint(fold: int) -> Path | None:
    out = fold_dir(fold)
    for name in ["best_postprocess.pt", "best.pt"]:
        path = out / name
        if path.exists():
            return path
    return None


def b5_fold_status() -> list[dict]:
    folds = []
    for fold in range(5):
        pid = read_pid(fold_dir(fold) / "train.pid")
        ckpt = checkpoint(fold)
        is_alive = alive(pid)
        if is_alive:
            state = "running"
        elif ckpt is not None:
            state = "completed"
        elif pid is not None:
            state = "failed_or_stopped_without_checkpoint"
        else:
            state = "not_started"
        folds.append(
            {
                "fold": fold,
                "state": state,
                "pid": pid,
                "checkpoint": None if ckpt is None else str(ckpt.relative_to(ROOT)),
                "latest_metric": latest_metric(fold),
            }
        )
    return folds


def write_status(stage: str, folds: list[dict] | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps({"updated_at": utc_now(), "stage": stage, "b5_folds": folds or b5_fold_status()}, indent=2),
        encoding="utf-8",
    )


def run_step(name: str, command: list[str], log_file: Path, expected: Path | None, force: bool) -> str:
    if expected is not None and (ROOT / expected).exists() and not force:
        log(f"{name}: reusing existing artifact {expected}")
        return "reused"
    log(f"{name}: starting")
    log_file = ROOT / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab") as f:
        f.write(("$ " + " ".join(command) + "\n\n").encode())
        subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    log(f"{name}: finished")
    return "completed"


def wait_for_b5(check_seconds: int) -> None:
    while True:
        folds = b5_fold_status()
        write_status("waiting_for_b5_5fold", folds)
        for item in folds:
            log(f"fold{item['fold']}: {item['state']}; pid={item['pid']}; {item['latest_metric']}")
        if all(item["state"] == "completed" for item in folds):
            log("B5 5-fold checkpoints are ready and no B5 fold is still running")
            return
        time.sleep(check_seconds)


def read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "summary" in data and "summary" in data["summary"]:
        return data["summary"]["summary"]
    return data.get("summary")


def write_report(step_status: dict[str, str]) -> None:
    report_path = RUN_DIR / "final_report.md"
    strategy_eval = read_summary(ROOT / "outputs/maskfix_oof/h200_stage1_eval_config_component_postprocess.json")
    b5_eval = read_summary(ROOT / B5_OOF_DIR / "h200_stage1_eval_config_component_postprocess.json")
    strategy_official = read_summary(ROOT / "outputs/maskfix_oof/maskfix_strategy_e_official_oof_proxy.json")
    b5_official = read_summary(ROOT / B5_OOF_DIR / "h200_next_unetpp_b5_official_oof_proxy.json")
    ensemble_path = ROOT / "outputs/ensemble_strategy_e_b5/strategy_e_b5_weight_search.json"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8")) if ensemble_path.exists() else None

    lines = [
        "# B5 Full Pipeline Report",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Step Status",
        "",
        "| Step | Status |",
        "| --- | --- |",
    ]
    for key, value in step_status.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Local OOF Dice",
            "",
            "| Model | Mean Dice | Positive Dice | Empty FP Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, summary in [("Strategy E", strategy_eval), ("B5", b5_eval)]:
        if summary:
            lines.append(
                f"| {name} | {summary['mean_dice_all_slices']:.10f} | "
                f"{summary['mean_dice_positive_slices']:.10f} | "
                f"{summary['mean_empty_slice_false_positive_rate']:.10f} |"
            )

    lines.extend(
        [
            "",
            "## Official Metric Proxy",
            "",
            "| Model | Combined Proxy | Dice 3D | HD95 mm |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, summary in [("Strategy E", strategy_official), ("B5", b5_official)]:
        if summary:
            lines.append(
                f"| {name} | {summary['mean_combined_proxy']:.10f} | "
                f"{summary['mean_dice_3d']:.10f} | {summary['mean_hd95_mm']:.4f} |"
            )

    if ensemble:
        best = ensemble["best"]
        best_summary = best["summary"]["summary"]
        lines.extend(
            [
                "",
                "## Strategy E + B5 Ensemble",
                "",
                f"- Best Strategy E weight: `{best['weight_a']:.3f}`",
                f"- Best B5 weight: `{best['weight_b']:.3f}`",
                f"- OOF mean Dice: `{best_summary['mean_dice_all_slices']:.10f}`",
                f"- Positive Dice: `{best_summary['mean_dice_positive_slices']:.10f}`",
                f"- Empty FP rate: `{best_summary['mean_empty_slice_false_positive_rate']:.10f}`",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"final report written: {report_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-seconds", type=int, default=300)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log("B5 full pipeline supervisor started")
    step_status: dict[str, str] = {}
    wait_for_b5(args.check_seconds)
    write_status("running_post_training_pipeline")

    step_status["b5_oof_postprocess_search"] = run_step(
        "b5_oof_postprocess_search",
        [
            str(PYTHON),
            "scripts/stage1_auto_pipeline.py",
            "--fold-config-glob",
            B5_GLOB,
            "--main-config",
            B5_MAIN_CONFIG,
            "--checkpoint-name",
            "best_postprocess.pt",
            "--out-dir",
            str(B5_OOF_DIR),
            "--report",
            str(B5_OOF_REPORT),
            "--work-dir",
            str(B5_OOF_DIR / "component_parallel_work"),
            "--gpus",
            args.gpus,
            "--max-workers",
            str(args.max_workers),
        ],
        B5_OOF_DIR / "pipeline_logs/stage1_auto_pipeline.log",
        B5_OOF_DIR / "h200_stage1_eval_config_component_postprocess.json",
        args.force,
    )

    step_status["strategy_e_official_proxy"] = run_step(
        "strategy_e_official_proxy",
        [
            str(PYTHON),
            "scripts/evaluate_official_oof.py",
            "--fold-config-glob",
            STRATEGY_E_GLOB,
            "--checkpoint-name",
            "best_postprocess.pt",
            "--out",
            "outputs/maskfix_oof/maskfix_strategy_e_official_oof_proxy.json",
        ],
        Path("outputs/maskfix_oof/pipeline_logs/official_proxy.log"),
        Path("outputs/maskfix_oof/maskfix_strategy_e_official_oof_proxy.json"),
        args.force,
    )

    step_status["b5_official_proxy"] = run_step(
        "b5_official_proxy",
        [
            str(PYTHON),
            "scripts/evaluate_official_oof.py",
            "--fold-config-glob",
            B5_GLOB,
            "--checkpoint-name",
            "best_postprocess.pt",
            "--out",
            str(B5_OOF_DIR / "h200_next_unetpp_b5_official_oof_proxy.json"),
        ],
        B5_OOF_DIR / "pipeline_logs/official_proxy.log",
        B5_OOF_DIR / "h200_next_unetpp_b5_official_oof_proxy.json",
        args.force,
    )

    step_status["strategy_e_b5_ensemble_weight_search"] = run_step(
        "strategy_e_b5_ensemble_weight_search",
        [
            str(PYTHON),
            "scripts/ensemble_oof_weight_search.py",
            "--model-a-glob",
            STRATEGY_E_GLOB,
            "--model-b-glob",
            B5_GLOB,
            "--model-a-checkpoint",
            "best_postprocess.pt",
            "--model-b-checkpoint",
            "best_postprocess.pt",
            "--weights-b",
            "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
            "--postprocess-source",
            "b",
            "--out",
            "outputs/ensemble_strategy_e_b5/strategy_e_b5_weight_search.json",
        ],
        Path("outputs/ensemble_strategy_e_b5/weight_search.log"),
        Path("outputs/ensemble_strategy_e_b5/strategy_e_b5_weight_search.json"),
        args.force,
    )

    write_report(step_status)
    write_status("complete")
    log("B5 full pipeline supervisor complete")


if __name__ == "__main__":
    main()
