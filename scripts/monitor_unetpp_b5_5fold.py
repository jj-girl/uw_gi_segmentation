from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("PYTHON", sys.executable))
CONFIG_DIR = ROOT / "configs/h200_next_unetpp_b5_folds"
LOG_DIR = ROOT / "outputs/h200_next_unetpp_b5_5fold"
LOG_PATH = LOG_DIR / "monitor.log"
STATUS_PATH = LOG_DIR / "status.json"
CHECK_SECONDS = int(os.environ.get("CHECK_SECONDS", "180"))
GPU_ID = os.environ.get("GPU_ID", "0")
MAX_RUNNING = int(os.environ.get("MAX_RUNNING", "1"))


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fold_config(fold: int) -> Path:
    return CONFIG_DIR / f"h200_next_unetpp_b5_fold{fold}.yaml"


def fold_dir(fold: int) -> Path:
    return ROOT / f"outputs/h200_next_unetpp_b5_fold{fold}"


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


def checkpoint(fold: int) -> Path | None:
    out = fold_dir(fold)
    for name in ["best_postprocess.pt", "best.pt"]:
        path = out / name
        if path.exists():
            return path
    return None


def latest_metric(fold: int) -> str:
    path = fold_dir(fold) / "train.log"
    if not path.exists():
        return "no train.log"
    text = path.read_bytes().decode("utf-8", "ignore")
    metrics = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip().startswith("epoch=")]
    return metrics[-1] if metrics else "training epoch in progress"


def fold_state(fold: int) -> dict:
    pid = read_pid(fold)
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
    return {
        "fold": fold,
        "state": state,
        "pid": pid,
        "checkpoint": None if ckpt is None else str(ckpt.relative_to(ROOT)),
        "latest_metric": latest_metric(fold),
        "config": str(fold_config(fold).relative_to(ROOT)),
        "log": str((fold_dir(fold) / "train.log").relative_to(ROOT)),
    }


def running_count() -> int:
    return sum(1 for fold in range(5) if fold_state(fold)["state"] == "running")


def start_fold(fold: int) -> None:
    out = fold_dir(fold)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    cmd = [
        str(PYTHON),
        "-u",
        "-m",
        "src.uwgi.train",
        "--config",
        str(fold_config(fold).relative_to(ROOT)),
    ]
    log(f"starting fold{fold} on CUDA_VISIBLE_DEVICES={GPU_ID}")
    with (out / "train.log").open("ab") as f:
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


def write_status(folds: list[dict]) -> None:
    STATUS_PATH.write_text(
        json.dumps({"updated_at": utc_now(), "gpu": GPU_ID, "folds": folds}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log(f"UNet++ B5 5-fold monitor started; gpu={GPU_ID}; max_running={MAX_RUNNING}; interval={CHECK_SECONDS}s")
    while True:
        folds = [fold_state(fold) for fold in range(5)]
        write_status(folds)
        for item in folds:
            log(f"fold{item['fold']}: {item['state']}; pid={item['pid']}; {item['latest_metric']}")

        if all(item["state"] == "completed" for item in folds):
            log("all UNet++ B5 folds completed")
            return

        for item in folds:
            if running_count() >= MAX_RUNNING:
                break
            if item["state"] == "not_started":
                start_fold(int(item["fold"]))
                time.sleep(5)

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
