import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.oof_postprocess_search import (  # noqa: E402
    add_to_accumulator,
    candidate_grid,
    empty_accumulator,
    parse_int_grid,
    select_best_candidates,
    update_candidate_scores_for_fold,
    valid_metadata,
)
from scripts.threshold_search import collect_predictions  # noqa: E402
from src.uwgi.dataset import CLASSES  # noqa: E402
from src.uwgi.utils import ensure_dir, get_device, load_yaml  # noqa: E402


def build_candidates(args: argparse.Namespace) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    thresholds = json.loads(Path(args.thresholds_json).read_text(encoding="utf-8"))
    min_area_grid = parse_int_grid(args.min_area_grid)
    z_min_run_grid = parse_int_grid(args.z_min_run_grid)
    min_volume_grid = parse_int_grid(args.min_volume_grid)
    keep_largest_options = [False, True] if args.keep_largest else [False]

    candidates_by_class = {}
    accumulators_by_class = {}
    for class_name in CLASSES:
        class_thresholds = thresholds[class_name]
        candidates_by_class[class_name] = candidate_grid(
            class_name=class_name,
            mask_threshold=float(class_thresholds["mask_threshold"]),
            cls_threshold=(
                None
                if class_thresholds.get("cls_threshold") is None
                else float(class_thresholds["cls_threshold"])
            ),
            min_area_grid=min_area_grid,
            z_min_run_grid=z_min_run_grid,
            min_volume_grid=min_volume_grid,
            keep_largest_options=keep_largest_options,
            connectivity=args.connectivity,
        )
        accumulators_by_class[class_name] = [empty_accumulator() for _ in candidates_by_class[class_name]]
    return candidates_by_class, accumulators_by_class


def merge_accumulators(target: dict[str, list[dict]], source: dict[str, list[dict]]) -> None:
    for class_name in CLASSES:
        for dst, src in zip(target[class_name], source[class_name]):
            for key, value in src.items():
                dst[key] += value


def make_summary(selected: dict[str, dict]) -> dict:
    classes = {class_name: selected[class_name]["metrics"] for class_name in CLASSES}
    positive_values = [
        classes[class_name]["dice_positive_slices"]
        for class_name in CLASSES
        if classes[class_name]["dice_positive_slices"] is not None
    ]
    empty_fp_values = [
        classes[class_name]["empty_slice_false_positive_rate"]
        for class_name in CLASSES
        if classes[class_name]["empty_slice_false_positive_rate"] is not None
    ]
    return {
        "classes": classes,
        "summary": {
            "mean_dice_all_slices": float(np.mean([classes[name]["dice_all_slices"] for name in CLASSES])),
            "mean_dice_positive_slices": float(np.mean(positive_values)) if positive_values else None,
            "mean_empty_slice_false_positive_rate": float(np.mean(empty_fp_values)) if empty_fp_values else None,
        },
    }


def run_worker(args: argparse.Namespace) -> None:
    candidates_by_class, accumulators_by_class = build_candidates(args)
    cfg = load_yaml(args.worker_config)
    if args.worker_device:
        cfg["train"]["device"] = args.worker_device
    checkpoint = Path(cfg["train"]["output_dir"]) / args.checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {args.worker_config}: {checkpoint}")
    device = get_device(cfg["train"]["device"])
    print(f"worker config={args.worker_config} device={device} checkpoint={checkpoint}", flush=True)
    probs, targets, cls_probs = collect_predictions(cfg, checkpoint, device)
    update_candidate_scores_for_fold(
        candidates_by_class,
        accumulators_by_class,
        probs,
        targets,
        cls_probs,
        valid_metadata(cfg),
    )
    result = {
        "config": args.worker_config,
        "fold": int(cfg["data"]["valid_fold"]),
        "accumulators_by_class": accumulators_by_class,
    }
    out = Path(args.worker_out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved worker result: {out}", flush=True)


def launch_workers(args: argparse.Namespace) -> None:
    config_paths = sorted(Path(path) for path in glob.glob(args.fold_config_glob))
    if not config_paths:
        raise FileNotFoundError(f"No configs matched {args.fold_config_glob}")

    work_dir = Path(args.work_dir)
    ensure_dir(work_dir)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    max_workers = min(args.max_workers, len(config_paths))
    pending = list(config_paths)
    running: list[dict] = []
    worker_results = []
    launched = 0

    while pending or running:
        while pending and len(running) < max_workers:
            config_path = pending.pop(0)
            gpu = gpus[launched % len(gpus)]
            launched += 1
            fold = load_yaml(config_path)["data"]["valid_fold"]
            worker_out = work_dir / f"fold{fold}_accumulators.json"
            log_path = work_dir / f"fold{fold}.log"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker-config",
                str(config_path),
                "--worker-out",
                str(worker_out),
                "--worker-device",
                "cuda",
                "--checkpoint-name",
                args.checkpoint_name,
                "--thresholds-json",
                args.thresholds_json,
                "--min-area-grid",
                args.min_area_grid,
                "--z-min-run-grid",
                args.z_min_run_grid,
                "--min-volume-grid",
                args.min_volume_grid,
                "--connectivity",
                str(args.connectivity),
            ]
            if args.keep_largest:
                cmd.append("--keep-largest")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_file = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
            running.append(
                {
                    "process": process,
                    "log_file": log_file,
                    "config": str(config_path),
                    "gpu": gpu,
                    "out": worker_out,
                    "log": log_path,
                }
            )
            print(f"started fold config={config_path} gpu={gpu} pid={process.pid} log={log_path}", flush=True)

        time.sleep(args.poll_seconds)
        still_running = []
        for item in running:
            return_code = item["process"].poll()
            if return_code is None:
                still_running.append(item)
                continue
            item["log_file"].close()
            if return_code != 0:
                raise RuntimeError(
                    f"worker failed config={item['config']} gpu={item['gpu']} "
                    f"return_code={return_code} log={item['log']}"
                )
            worker_results.append(item["out"])
            print(f"finished config={item['config']} gpu={item['gpu']} result={item['out']}", flush=True)
        running = still_running

    aggregate_results(args, worker_results)


def aggregate_results(args: argparse.Namespace, result_paths: list[Path] | None = None) -> None:
    candidates_by_class, accumulators_by_class = build_candidates(args)
    if result_paths is None:
        result_paths = sorted(Path(args.work_dir).glob("fold*_accumulators.json"))
    if not result_paths:
        raise FileNotFoundError(f"No worker results found in {args.work_dir}")
    folds = []
    for path in result_paths:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        merge_accumulators(accumulators_by_class, result["accumulators_by_class"])
        folds.append({"config": result["config"], "fold": result["fold"], "result": str(path)})

    selected = select_best_candidates(candidates_by_class, accumulators_by_class)
    output = {
        "summary": make_summary(selected),
        "classes": selected,
        "folds": sorted(folds, key=lambda item: item["fold"]),
        "num_folds": len(folds),
        "source_thresholds": args.thresholds_json,
        "search_space": {
            "min_area_grid": parse_int_grid(args.min_area_grid),
            "z_min_run_grid": parse_int_grid(args.z_min_run_grid),
            "min_volume_grid": parse_int_grid(args.min_volume_grid),
            "keep_largest_options": [False, True] if args.keep_largest else [False],
            "connectivity": args.connectivity,
        },
    }
    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"saved aggregate result: {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-config-glob")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--thresholds-json", required=True)
    parser.add_argument("--out", default="outputs/oof/h200_stage1_component_parallel_search.json")
    parser.add_argument("--work-dir", default="outputs/oof/component_parallel_work")
    parser.add_argument("--min-area-grid", default="48,192")
    parser.add_argument("--z-min-run-grid", default="1,2,3")
    parser.add_argument("--min-volume-grid", default="0,64,128,256,512")
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--connectivity", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-config")
    parser.add_argument("--worker-out")
    parser.add_argument("--worker-device", default="cuda")
    args = parser.parse_args()

    if args.worker:
        if not args.worker_config or not args.worker_out:
            raise ValueError("--worker requires --worker-config and --worker-out")
        run_worker(args)
        return
    if args.aggregate_only:
        aggregate_results(args)
        return
    if not args.fold_config_glob:
        raise ValueError("--fold-config-glob is required")
    launch_workers(args)


if __name__ == "__main__":
    main()
