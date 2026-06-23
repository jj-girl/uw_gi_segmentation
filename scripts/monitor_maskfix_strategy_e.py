from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/disk2/hjj/uw_gi_segmentation")
PYTHON = Path("/mnt/disk2/hjj/uwgiseg/bin/python")
LOG_DIR = ROOT / "outputs/maskfix_oof"
LOG_PATH = LOG_DIR / "maskfix_strategy_e_monitor_py.log"
CHECK_SECONDS = int(os.environ.get("CHECK_SECONDS", "120"))
MAX_RUNNING = int(os.environ.get("MAX_RUNNING", "4"))


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fold_dir(fold: int) -> Path:
    return ROOT / f"outputs/h200_maskfix_stage1_strategy_e_fold{fold}"


def fold_config(fold: int) -> Path:
    return ROOT / f"configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold{fold}.yaml"


def pid_file(fold: int) -> Path:
    return fold_dir(fold) / "train.pid"


def read_pid(fold: int) -> int | None:
    path = pid_file(fold)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return int(text) if text.isdigit() else None


def alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def has_best(fold: int) -> bool:
    return (fold_dir(fold) / "best.pt").exists()


def has_best_postprocess(fold: int) -> bool:
    return (fold_dir(fold) / "best_postprocess.pt").exists()


def latest_metric(fold: int) -> str:
    path = fold_dir(fold) / "train.log"
    if not path.exists():
        return "no train.log"
    text = path.read_bytes().decode("utf-8", "ignore")
    metrics = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip().startswith("epoch=")]
    return metrics[-1] if metrics else "training epoch in progress"


def running_count() -> int:
    return sum(alive(read_pid(fold)) for fold in range(5))


def start_fold(fold: int, gpu: str = "1") -> None:
    out_dir = fold_dir(fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "train.log"
    log_file.touch()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = [
        str(PYTHON),
        "-u",
        "-m",
        "src.uwgi.train",
        "--config",
        str(fold_config(fold).relative_to(ROOT)),
    ]
    log(f"starting fold{fold} on CUDA_VISIBLE_DEVICES={gpu}")
    with log_file.open("ab") as f:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    pid_file(fold).write_text(str(process.pid) + "\n", encoding="utf-8")


def main() -> None:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    log(f"python monitor started; check interval={CHECK_SECONDS}s; max running={MAX_RUNNING}")
    while True:
        for fold in range(5):
            pid = read_pid(fold)
            if alive(pid):
                state = f"running pid={pid}"
                if has_best(fold):
                    state += "; best.pt present"
            elif pid is not None:
                state = f"stopped pid={pid}"
                if has_best(fold):
                    state += "; best.pt present"
            else:
                state = "not_started"
            log(f"fold{fold}: {state}; {latest_metric(fold)}")

        if all(has_best(fold) and not alive(read_pid(fold)) for fold in range(5)):
            log("all folds stopped and have best.pt")
            run_oof_pipeline()
            log("monitor exiting")
            return

        for fold in range(5):
            if running_count() >= MAX_RUNNING:
                break
            if not has_best(fold) and not alive(read_pid(fold)):
                start_fold(fold, "1")
                time.sleep(5)

        time.sleep(CHECK_SECONDS)


def run_oof_pipeline() -> None:
    checkpoint_name = "best_postprocess.pt" if all(has_best_postprocess(fold) for fold in range(5)) else "best.pt"
    report = ROOT / "outputs/maskfix_oof/maskfix_strategy_e_auto_report.md"
    cmd = [
        str(PYTHON),
        "scripts/stage1_auto_pipeline.py",
        "--fold-config-glob",
        "configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml",
        "--main-config",
        "configs/h200_maskfix_stage1_strategy_e.yaml",
        "--checkpoint-name",
        checkpoint_name,
        "--out-dir",
        "outputs/maskfix_oof",
        "--report",
        str(report),
        "--work-dir",
        "outputs/maskfix_oof/component_parallel_work",
        "--gpus",
        "0,1",
        "--max-workers",
        "5",
    ]
    log(f"starting OOF pipeline with checkpoint={checkpoint_name}")
    with (LOG_DIR / "maskfix_strategy_e_oof_pipeline.log").open("ab") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    log(f"OOF pipeline finished; report={report}")


if __name__ == "__main__":
    main()
