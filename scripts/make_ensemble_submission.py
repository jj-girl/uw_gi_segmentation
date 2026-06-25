from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.make_submission import (  # noqa: E402
    build_submission,
    build_submission_metadata,
    postprocess_predictions,
    predict_ensemble,
)
from src.uwgi.utils import ensure_dir, get_device  # noqa: E402


def config_paths(pattern: str) -> list[Path]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No configs matched {pattern}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the final UWGI submission with the selected two-family ensemble: "
            "Strategy E + UNet++ EfficientNet-B5 probability blending."
        )
    )
    parser.add_argument("--model-a-glob", required=True, help="Strategy E fold config glob.")
    parser.add_argument("--model-b-glob", required=True, help="B5 fold config glob.")
    parser.add_argument("--model-a-checkpoint", default="best_postprocess.pt")
    parser.add_argument("--model-b-checkpoint", default="best_postprocess.pt")
    parser.add_argument("--weight-a", type=float, default=0.3, help="Strategy E probability weight.")
    parser.add_argument("--weight-b", type=float, default=0.7, help="B5 probability weight.")
    parser.add_argument("--postprocess-source", choices=["a", "b"], default="b")
    parser.add_argument("--sample-submission", default="data/raw/uw-madison-gi-tract-image-segmentation/sample_submission.csv")
    parser.add_argument("--data-root", default="data/raw/uw-madison-gi-tract-image-segmentation")
    parser.add_argument("--split", default="auto", choices=["auto", "test", "train"])
    parser.add_argument("--out", default="outputs/final_submissions/strategy_e_b5_030_070_submission.csv")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    total = args.weight_a + args.weight_b
    if total <= 0:
        raise ValueError("weight-a + weight-b must be positive")
    weight_a = float(args.weight_a) / total
    weight_b = float(args.weight_b) / total

    model_a_paths = config_paths(args.model_a_glob)
    model_b_paths = config_paths(args.model_b_glob)

    sample_path = Path(args.sample_submission)
    sample = pd.read_csv(sample_path)
    out = Path(args.out)
    ensure_dir(out.parent)
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")

    manifest = {
        "sample_submission": str(sample_path),
        "data_root": args.data_root,
        "model_a_fold_configs": [str(path) for path in model_a_paths],
        "model_b_fold_configs": [str(path) for path in model_b_paths],
        "model_a_checkpoint": args.model_a_checkpoint,
        "model_b_checkpoint": args.model_b_checkpoint,
        "weight_a": weight_a,
        "weight_b": weight_b,
        "postprocess_source": args.postprocess_source,
        "output": str(out),
        "num_sample_rows": int(len(sample)),
        "status": "not_started",
    }

    if sample.empty:
        pd.DataFrame(columns=["id", "class", "predicted"]).to_csv(out, index=False)
        manifest["status"] = "empty_sample_submission"
        manifest["num_images"] = 0
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Sample submission is empty; wrote empty submission: {out}")
        print(f"Saved manifest: {manifest_path}")
        return

    if set(sample.columns) != {"id", "class", "predicted"}:
        raise ValueError(f"Unexpected sample columns: {sample.columns.tolist()}")

    metadata = build_submission_metadata(sample, Path(args.data_root), args.split)
    device = get_device(args.device)

    probs_a, cls_a, cfg_a = predict_ensemble(model_a_paths, args.model_a_checkpoint, metadata, device)
    probs_b, cls_b, cfg_b = predict_ensemble(model_b_paths, args.model_b_checkpoint, metadata, device)
    if probs_a.shape != probs_b.shape:
        raise ValueError(f"Probability shape mismatch: {probs_a.shape} vs {probs_b.shape}")
    probs = weight_a * probs_a + weight_b * probs_b

    if cls_a is None or cls_b is None:
        cls_probs = None
    else:
        if cls_a.shape != cls_b.shape:
            raise ValueError(f"Classification probability shape mismatch: {cls_a.shape} vs {cls_b.shape}")
        cls_probs = weight_a * cls_a + weight_b * cls_b

    postprocess_cfg = cfg_b if args.postprocess_source == "b" else cfg_a
    masks = postprocess_predictions(probs, cls_probs, metadata, postprocess_cfg)
    submission = build_submission(sample, metadata, masks)
    submission.to_csv(out, index=False)

    manifest.update(
        {
            "status": "completed",
            "num_images": int(len(metadata)),
            "num_non_empty_predictions": int((submission["predicted"].fillna("") != "").sum()),
            "postprocess": postprocess_cfg.get("postprocess", {}),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved ensemble submission: {out}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
