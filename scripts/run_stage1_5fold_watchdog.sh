#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk2/hjj/uw_gi_segmentation"
PY="/mnt/disk2/hjj/uwgiseg/bin/python"
LOG_DIR="$ROOT/outputs/oof"
WATCH_LOG="$LOG_DIR/stage1_5fold_watchdog.log"
LOCK_DIR="$LOG_DIR/stage1_5fold_watchdog.lock"
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

fold_dir() {
  printf 'outputs/h200_stage1_2p5d_unetpp_b3_all_fold%s' "$1"
}

fold_config() {
  printf 'configs/h200_stage1_folds/h200_stage1_2p5d_unetpp_b3_all_fold%s.yaml' "$1"
}

is_running() {
  local f="$1"
  local pid_file
  pid_file="$(fold_dir "$f")/train.pid"
  [ -f "$pid_file" ] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1
}

latest_metric() {
  local f="$1"
  "$PY" - "$f" <<'PY'
from pathlib import Path
import re
import sys

fold = sys.argv[1]
path = Path(f"outputs/h200_stage1_2p5d_unetpp_b3_all_fold{fold}/train.log")
if not path.exists():
    print("no train.log")
    raise SystemExit
text = path.read_bytes().decode("utf-8", "ignore")
metrics = [x.strip() for x in re.split(r"[\r\n]+", text) if x.strip().startswith("epoch=")]
print(metrics[-1] if metrics else "no epoch metrics yet")
PY
}

log_status() {
  local f pid_file pid state
  for f in "$@"; do
    pid_file="$(fold_dir "$f")/train.pid"
    if [ -f "$pid_file" ]; then
      pid="$(cat "$pid_file")"
      if ps -p "$pid" >/dev/null 2>&1; then
        state="running pid=$pid"
      else
        state="stopped pid=$pid"
      fi
    else
      state="not started"
    fi
    log "fold${f}: ${state}; $(latest_metric "$f" 2>/dev/null || true)"
  done
}

start_fold() {
  local f="$1"
  local gpu="$2"
  local out_dir cfg
  out_dir="$(fold_dir "$f")"
  cfg="$(fold_config "$f")"

  mkdir -p "$out_dir"
  touch "$out_dir/train.log"

  if is_running "$f"; then
    log "fold${f} already running; skip start"
    return
  fi

  if [ -f "$out_dir/best.pt" ]; then
    log "fold${f} already has best.pt; skip start"
    return
  fi

  log "starting fold${f} on CUDA_VISIBLE_DEVICES=${gpu}"
  setsid bash -lc "cd '$ROOT' && echo \$\$ > '$out_dir/train.pid' && CUDA_VISIBLE_DEVICES=$gpu exec '$PY' -u -m src.uwgi.train --config '$cfg' >> '$out_dir/train.log' 2>&1" \
    </dev/null >"/tmp/uwgi_fold${f}_watchdog_setsid.out" 2>&1 &
  sleep 5
  log_status "$f"
}

wait_folds_done() {
  local folds=("$@")
  local any f
  while true; do
    any=0
    for f in "${folds[@]}"; do
      if is_running "$f"; then
        any=1
      fi
    done
    log_status "${folds[@]}"
    if [ "$any" -eq 0 ]; then
      break
    fi
    sleep "$CHECK_SECONDS"
  done
}

require_best_checkpoints() {
  local f ckpt
  for f in 0 1 2 3 4; do
    ckpt="$(fold_dir "$f")/best.pt"
    if [ ! -f "$ckpt" ]; then
      log "missing checkpoint: $ckpt"
      return 1
    fi
  done
}

run_oof() {
  log "all fold checkpoints found; starting 5-fold OOF threshold search"
  "$PY" scripts/oof_threshold_search.py \
    --fold-config-glob 'configs/h200_stage1_folds/h200_stage1_2p5d_unetpp_b3_all_fold*.yaml' \
    --checkpoint-name best.pt \
    --out outputs/oof/h200_stage1_thresholds_with_cls_gate.json \
    --mask-grid 0.25,0.30,0.35,0.40,0.45,0.50 \
    --cls-grid 0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90 \
    >> "$WATCH_LOG" 2>&1

  for z in 1 2 3; do
    log "starting OOF z-axis evaluation: z_min_run=${z}"
    "$PY" scripts/evaluate_oof_postprocess.py \
      --fold-config-glob 'configs/h200_stage1_folds/h200_stage1_2p5d_unetpp_b3_all_fold*.yaml' \
      --checkpoint-name best.pt \
      --out "outputs/oof/h200_stage1_eval_z${z}.json" \
      --z-min-run "$z" \
      >> "$WATCH_LOG" 2>&1
  done
  log "5-fold OOF threshold search and z-axis evaluations complete"
}

log "watchdog started; check interval=${CHECK_SECONDS}s"
log_status 1 2
wait_folds_done 1 2

log "fold1/fold2 are no longer running; starting remaining folds if needed"
start_fold 3 0
start_fold 4 1
wait_folds_done 3 4

require_best_checkpoints
run_oof
