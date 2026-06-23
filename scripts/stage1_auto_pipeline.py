import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.uwgi.dataset import CLASSES  # noqa: E402
from src.uwgi.utils import ensure_dir, load_yaml  # noqa: E402


DEFAULT_FOLD_GLOB = "configs/h200_stage1_folds/h200_stage1_2p5d_unetpp_b3_all_fold*.yaml"
DEFAULT_MAIN_CONFIG = "configs/h200_stage1_2p5d_unetpp_b3_all.yaml"
DEFAULT_REPORT = "outputs/oof/stage1_auto_report.md"


def run_command(command: list[str], log_path: Path, dry_run: bool = False) -> None:
    ensure_dir(log_path.parent)
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        log_file.flush()
        subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT)


def matched_fold_configs(pattern: str) -> list[Path]:
    return sorted(Path().glob(pattern))


def check_checkpoints(config_paths: list[Path], checkpoint_name: str) -> list[dict]:
    rows = []
    for config_path in config_paths:
        cfg = load_yaml(config_path)
        checkpoint = Path(cfg["train"]["output_dir"]) / checkpoint_name
        rows.append(
            {
                "fold": int(cfg["data"]["valid_fold"]),
                "config": str(config_path),
                "checkpoint": str(checkpoint),
                "exists": checkpoint.exists(),
            }
        )
    return sorted(rows, key=lambda item: item["fold"])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_postprocess_config(class_params: dict) -> dict:
    connectivity_values = {int(class_params[name]["connectivity"]) for name in CLASSES}
    return {
        "mask_thresholds": [float(class_params[name]["mask_threshold"]) for name in CLASSES],
        "cls_thresholds": [float(class_params[name]["cls_threshold"]) for name in CLASSES],
        "min_area": [int(class_params[name]["min_area"]) for name in CLASSES],
        "z_min_run": [int(class_params[name]["z_min_run"]) for name in CLASSES],
        "min_volume": [int(class_params[name]["min_volume"]) for name in CLASSES],
        "keep_largest_component": [bool(class_params[name]["keep_largest"]) for name in CLASSES],
        "component_connectivity": sorted(connectivity_values)[-1],
    }


def update_config_postprocess(config_path: Path, class_params: dict) -> bool:
    cfg = load_yaml(config_path)
    desired = build_postprocess_config(class_params)
    current = cfg.get("postprocess", {})
    if all(current.get(key) == value for key, value in desired.items()):
        return False

    post = cfg.setdefault("postprocess", {})
    post.update(desired)
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return True


def maybe_run(
    output_path: Path,
    command: list[str],
    log_path: Path,
    force: bool,
    dry_run: bool,
) -> str:
    if output_path.exists() and not force:
        return "reused"
    run_command(command, log_path, dry_run=dry_run)
    return "ran"


def metric_summary(result: dict) -> dict:
    summary = result["summary"]["summary"] if "summary" in result.get("summary", {}) else result["summary"]
    return {
        "mean_dice_all_slices": float(summary["mean_dice_all_slices"]),
        "mean_dice_positive_slices": float(summary["mean_dice_positive_slices"]),
        "mean_empty_slice_false_positive_rate": float(summary["mean_empty_slice_false_positive_rate"]),
    }


def format_float(value: float) -> str:
    return f"{value:.10f}"


def build_report(
    checkpoint_rows: list[dict],
    threshold_path: Path,
    minarea_z_path: Path,
    component_path: Path,
    eval_path: Path,
    step_status: dict,
    config_paths: list[Path],
) -> str:
    threshold = load_json(threshold_path)
    minarea_z = load_json(minarea_z_path)
    component = load_json(component_path)
    final_eval = load_json(eval_path)

    threshold_mean = float(threshold["mean_dice"])
    minarea_summary = metric_summary(minarea_z)
    component_summary = metric_summary(component)
    final_summary = metric_summary(final_eval)

    lines = [
        "# Stage 1 Auto Pipeline Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Status",
        "",
        "| Step | Status |",
        "| --- | --- |",
    ]
    for name in ["threshold_search", "minarea_z_search", "component_search", "config_update", "config_eval"]:
        lines.append(f"| {name} | {step_status.get(name, 'unknown')} |")

    lines += [
        "",
        "## Checkpoints",
        "",
        "| Fold | Checkpoint | Exists |",
        "| --- | --- | --- |",
    ]
    for row in checkpoint_rows:
        lines.append(f"| {row['fold']} | `{row['checkpoint']}` | {row['exists']} |")

    lines += [
        "",
        "## Metrics",
        "",
        "| Stage | Mean Dice | Positive Dice | Empty FP Rate |",
        "| --- | ---: | ---: | ---: |",
        f"| OOF threshold + cls gate | {format_float(threshold_mean)} | n/a | n/a |",
        (
            "| min_area + z | "
            f"{format_float(minarea_summary['mean_dice_all_slices'])} | "
            f"{format_float(minarea_summary['mean_dice_positive_slices'])} | "
            f"{format_float(minarea_summary['mean_empty_slice_false_positive_rate'])} |"
        ),
        (
            "| 3D component search | "
            f"{format_float(component_summary['mean_dice_all_slices'])} | "
            f"{format_float(component_summary['mean_dice_positive_slices'])} | "
            f"{format_float(component_summary['mean_empty_slice_false_positive_rate'])} |"
        ),
        (
            "| final config eval | "
            f"{format_float(final_summary['mean_dice_all_slices'])} | "
            f"{format_float(final_summary['mean_dice_positive_slices'])} | "
            f"{format_float(final_summary['mean_empty_slice_false_positive_rate'])} |"
        ),
        "",
        "## Recommended Postprocess",
        "",
        "| Organ | Mask Thr | Cls Thr | Min Area | Z Min Run | Min Volume | Keep Largest |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in CLASSES:
        params = component["classes"][name]
        lines.append(
            f"| {name} | {params['mask_threshold']:.2f} | {params['cls_threshold']:.2f} | "
            f"{params['min_area']} | {params['z_min_run']} | {params['min_volume']} | "
            f"{params['keep_largest']} |"
        )

    lines += [
        "",
        "## Artifacts",
        "",
        f"- Thresholds: `{threshold_path}`",
        f"- min_area + z search: `{minarea_z_path}`",
        f"- component search: `{component_path}`",
        f"- final config eval: `{eval_path}`",
        "- Managed configs:",
    ]
    for path in config_paths:
        lines.append(f"  - `{path}`")
    return "\n".join(lines) + "\n"


def append_worklog(report_path: Path, final_eval_path: Path, worklog_path: Path) -> None:
    final_eval = load_json(final_eval_path)
    summary = metric_summary(final_eval)
    text = (
        "\n## Stage 1 自动 OOF Pipeline 固化\n\n"
        f"时间：{datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n\n"
        "- 已新增 `scripts/stage1_auto_pipeline.py`，统一检查 fold checkpoint、复用/运行 OOF 搜索、"
        "同步推荐后处理配置、生成固定报告。\n"
        f"- 报告：`{report_path}`\n"
        f"- 最终 config OOF mean Dice：`{summary['mean_dice_all_slices']:.10f}`\n"
        f"- positive-slice Dice：`{summary['mean_dice_positive_slices']:.10f}`\n"
        f"- empty FP rate：`{summary['mean_empty_slice_false_positive_rate']:.10f}`\n"
        "- 后续新实验应优先复用该 pipeline 做统一比较，避免手工散跑和文档堆叠。\n"
    )
    with worklog_path.open("a", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-config-glob", default=DEFAULT_FOLD_GLOB)
    parser.add_argument("--main-config", default=DEFAULT_MAIN_CONFIG)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--out-dir", default="outputs/oof")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--work-dir", default="outputs/oof/component_parallel_work")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--append-worklog", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    logs_dir = ensure_dir(out_dir / "pipeline_logs")
    threshold_path = out_dir / "h200_stage1_thresholds_with_cls_gate.json"
    minarea_z_path = out_dir / "h200_stage1_postprocess_minarea_z_search.json"
    component_path = out_dir / "h200_stage1_component_parallel_search.json"
    eval_path = out_dir / "h200_stage1_eval_config_component_postprocess.json"
    report_path = Path(args.report)

    config_paths = matched_fold_configs(args.fold_config_glob)
    if not config_paths:
        raise FileNotFoundError(f"No configs matched {args.fold_config_glob}")
    checkpoint_rows = check_checkpoints(config_paths, args.checkpoint_name)
    missing = [row for row in checkpoint_rows if not row["exists"]]
    if missing:
        missing_text = "\n".join(f"fold{row['fold']}: {row['checkpoint']}" for row in missing)
        raise FileNotFoundError(f"Missing checkpoints:\n{missing_text}")

    py = sys.executable
    step_status = {}
    step_status["threshold_search"] = maybe_run(
        threshold_path,
        [
            py,
            "scripts/oof_threshold_search.py",
            "--fold-config-glob",
            args.fold_config_glob,
            "--checkpoint-name",
            args.checkpoint_name,
            "--out",
            str(threshold_path),
        ],
        logs_dir / "threshold_search.log",
        force=args.force,
        dry_run=args.dry_run,
    )
    step_status["minarea_z_search"] = maybe_run(
        minarea_z_path,
        [
            py,
            "-u",
            "scripts/oof_postprocess_search.py",
            "--fold-config-glob",
            args.fold_config_glob,
            "--checkpoint-name",
            args.checkpoint_name,
            "--thresholds-json",
            str(threshold_path),
            "--out",
            str(minarea_z_path),
            "--min-area-grid",
            "0,8,16,24,48,96,192",
            "--z-min-run-grid",
            "1,2,3",
            "--min-volume-grid",
            "0",
            "--connectivity",
            "1",
        ],
        logs_dir / "minarea_z_search.log",
        force=args.force,
        dry_run=args.dry_run,
    )
    step_status["component_search"] = maybe_run(
        component_path,
        [
            py,
            "-u",
            "scripts/oof_component_search_parallel.py",
            "--fold-config-glob",
            args.fold_config_glob,
            "--checkpoint-name",
            args.checkpoint_name,
            "--thresholds-json",
            str(threshold_path),
            "--out",
            str(component_path),
            "--work-dir",
            args.work_dir,
            "--min-area-grid",
            "48,192",
            "--z-min-run-grid",
            "1,2,3",
            "--min-volume-grid",
            "0,64,128,256,512",
            "--keep-largest",
            "--connectivity",
            "1",
            "--gpus",
            args.gpus,
            "--max-workers",
            str(args.max_workers),
        ],
        logs_dir / "component_search.log",
        force=args.force,
        dry_run=args.dry_run,
    )

    component_result = load_json(component_path)
    updated_configs = 0
    for path in [Path(args.main_config), *config_paths]:
        if not args.dry_run:
            updated_configs += int(update_config_postprocess(path, component_result["classes"]))
    if args.dry_run:
        step_status["config_update"] = "dry-run"
    else:
        step_status["config_update"] = f"updated {updated_configs}" if updated_configs else "already current"

    step_status["config_eval"] = maybe_run(
        eval_path,
        [
            py,
            "scripts/evaluate_oof_postprocess.py",
            "--fold-config-glob",
            args.fold_config_glob,
            "--checkpoint-name",
            args.checkpoint_name,
            "--out",
            str(eval_path),
        ],
        logs_dir / "config_eval.log",
        force=args.force,
        dry_run=args.dry_run,
    )

    report = build_report(
        checkpoint_rows=checkpoint_rows,
        threshold_path=threshold_path,
        minarea_z_path=minarea_z_path,
        component_path=component_path,
        eval_path=eval_path,
        step_status=step_status,
        config_paths=[Path(args.main_config), *config_paths],
    )
    if not args.dry_run:
        ensure_dir(report_path.parent)
        report_path.write_text(report, encoding="utf-8")
        if args.append_worklog:
            append_worklog(report_path, eval_path, Path("工作日志.md"))
    print(report)


if __name__ == "__main__":
    main()
