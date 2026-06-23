#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/hjj/uw_gi_segmentation"
PY="/mnt/disk2/hjj/uwgiseg/bin/python"
LOG_DIR="$ROOT/outputs"
WATCH_LOG="$LOG_DIR/training_strategy_fold0_watchdog.log"
LOCK_DIR="$LOG_DIR/training_strategy_fold0_watchdog.lock"
CHECK_SECONDS="${CHECK_SECONDS:-300}"

mkdir -p "$LOG_DIR"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [ -f "$LOCK_DIR/pid" ] && ps -p "$(cat "$LOCK_DIR/pid")" >/dev/null 2>&1; then
    echo "watchdog already running: pid=$(cat "$LOCK_DIR/pid")"
    exit 0
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi

echo "$$" > "$LOCK_DIR/pid"
trap 'rm -rf "$LOCK_DIR"' EXIT

cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*" | tee -a "$WATCH_LOG"
}

out_dir_for_config() {
  "$PY" - "$1" <<'PY'
import sys
from src.uwgi.utils import load_yaml
print(load_yaml(sys.argv[1])["train"]["output_dir"])
PY
}

is_running() {
  local out_dir="$1"
  local pid_file="$out_dir/train.pid"
  [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1
}

latest_metric() {
  local out_dir="$1"
  "$PY" - "$out_dir" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1]) / "train.log"
if not path.exists():
    print("no train.log")
    raise SystemExit
text = path.read_bytes().decode("utf-8", "ignore")
metrics = [x.strip() for x in re.split(r"[\r\n]+", text) if x.strip().startswith("epoch=")]
print(metrics[-1] if metrics else "no epoch metrics yet")
PY
}

log_status() {
  local name="$1"
  local out_dir="$2"
  local pid_file="$out_dir/train.pid"
  local state="not started"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file")"
    if ps -p "$pid" >/dev/null 2>&1; then
      state="running pid=$pid"
    else
      state="stopped pid=$pid"
    fi
  fi
  log "${name}: ${state}; $(latest_metric "$out_dir" 2>/dev/null || true)"
}

start_exp() {
  local name="$1"
  local cfg="$2"
  local gpu="$3"
  local out_dir
  out_dir="$(out_dir_for_config "$cfg")"
  mkdir -p "$out_dir"
  touch "$out_dir/train.log"

  if is_running "$out_dir"; then
    log "${name} already running; skip start"
    return
  fi
  if [ -f "$out_dir/best.pt" ]; then
    log "${name} already has best.pt; skip start"
    return
  fi

  log "starting ${name} on CUDA_VISIBLE_DEVICES=${gpu}"
  setsid bash -lc "cd '$ROOT' && echo \$\$ > '$out_dir/train.pid' && CUDA_VISIBLE_DEVICES=$gpu exec '$PY' -u -m src.uwgi.train --config '$cfg' >> '$out_dir/train.log' 2>&1" \
    </dev/null >"/tmp/uwgi_${name}_setsid.out" 2>&1 &
  sleep 5
  log_status "$name" "$out_dir"
}

wait_exp() {
  local name="$1"
  local cfg="$2"
  local out_dir
  out_dir="$(out_dir_for_config "$cfg")"
  while is_running "$out_dir"; do
    log_status "$name" "$out_dir"
    sleep "$CHECK_SECONDS"
  done
  log_status "$name" "$out_dir"
}

A_CFG="configs/h200_stage1_strategy_a_folds/h200_stage1_strategy_a_strong_aug_fold0.yaml"
B_CFG="configs/h200_stage1_strategy_b_folds/h200_stage1_strategy_b_organ_balanced_fold0.yaml"
C_CFG="configs/h200_stage1_strategy_c_folds/h200_stage1_strategy_c_organ_balanced_focal_tversky_fold0.yaml"

log "training strategy fold0 watchdog started; check interval=${CHECK_SECONDS}s"
start_exp "strategy_a_strong_aug_fold0" "$A_CFG" 0
start_exp "strategy_b_organ_balanced_fold0" "$B_CFG" 1
wait_exp "strategy_a_strong_aug_fold0" "$A_CFG" &
WAIT_A=$!
wait_exp "strategy_b_organ_balanced_fold0" "$B_CFG" &
WAIT_B=$!
wait "$WAIT_A"
wait "$WAIT_B"

log "A/B fold0 finished; starting strategy C"
start_exp "strategy_c_organ_balanced_focal_tversky_fold0" "$C_CFG" 0
wait_exp "strategy_c_organ_balanced_focal_tversky_fold0" "$C_CFG"
log "training strategy fold0 watchdog complete"
