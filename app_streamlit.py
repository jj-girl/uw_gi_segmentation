from __future__ import annotations

import glob
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.uwgi.dataset import CLASSES, parse_id
from src.uwgi.inference import predict_logits
from src.uwgi.models import build_model
from src.uwgi.postprocess import postprocess_slice
from src.uwgi.rle import decode_multiclass, rle_encode
from src.uwgi.utils import get_device, load_yaml


DATA_ROOT = ROOT / "data/raw/uw-madison-gi-tract-image-segmentation"
DEFAULT_PRESETS = {
    "Dual-model ensemble": {
        "kind": "ensemble",
        "model_a_name": "Strategy E",
        "model_a_glob": "configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml",
        "model_a_checkpoint": "best_postprocess.pt",
        "model_a_weight": 0.3,
        "model_b_name": "B5",
        "model_b_glob": "configs/h200_next_unetpp_b5_folds/h200_next_unetpp_b5_fold*.yaml",
        "model_b_checkpoint": "best_postprocess.pt",
        "model_b_weight": 0.7,
        "postprocess_source": "b",
        "report": "docs/final_solution_report.md",
        "oof_json": "outputs/ensemble_strategy_e_b5/strategy_e_b5_weight_search.json",
    },
    "B5 primary 5-fold": {
        "kind": "single",
        "glob": "configs/h200_next_unetpp_b5_folds/h200_next_unetpp_b5_fold*.yaml",
        "checkpoint": "best_postprocess.pt",
        "report": "outputs/h200_next_unetpp_b5_oof/h200_next_unetpp_b5_auto_report.md",
        "oof_json": "outputs/h200_next_unetpp_b5_oof/h200_stage1_eval_config_component_postprocess.json",
    },
    "Strategy E auxiliary 5-fold": {
        "kind": "single",
        "glob": "configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml",
        "checkpoint": "best_postprocess.pt",
        "report": "docs/maskfix_strategy_e_auto_pipeline_report.md",
        "oof_json": "outputs/maskfix_oof/h200_stage1_eval_config_component_postprocess.json",
    },
}
CLASS_LABELS = {
    "large_bowel": "Large bowel",
    "small_bowel": "Small bowel",
    "stomach": "Stomach",
}
COLORS = np.array(
    [
        [229, 72, 77],
        [42, 178, 123],
        [64, 126, 230],
    ],
    dtype=np.float32,
)


def configure_page() -> None:
    st.set_page_config(
        page_title="UWGI Medical Segmentation Workbench",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.2rem;
            max-width: 1480px;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 0.72rem 0.85rem;
            background: #f8fafc;
        }
        div[data-testid="stMetric"] label {
            color: #334155;
        }
        .uwgi-section {
            border-top: 1px solid #d8dee9;
            padding-top: 0.85rem;
            margin-top: 0.4rem;
        }
        .uwgi-caption {
            color: #475569;
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .uwgi-chip {
            display: inline-block;
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            margin-right: 0.35rem;
            margin-bottom: 0.3rem;
            font-size: 0.78rem;
            color: #334155;
            background: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_metadata(data_root: str) -> pd.DataFrame:
    path = Path(data_root) / "metadata_folds.csv"
    if path.exists():
        meta = pd.read_csv(path)
        corrected = meta.copy()
        parts = corrected["image_path"].map(lambda value: Path(value).stem.split("_"))
        corrected["width"] = parts.map(lambda item: int(item[-4]))
        corrected["height"] = parts.map(lambda item: int(item[-3]))
        return corrected
    raise FileNotFoundError(f"Missing metadata file: {path}")


@st.cache_data(show_spinner=False)
def load_labels(data_root: str) -> pd.DataFrame:
    path = Path(data_root) / "train.csv"
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(f"Missing labels file: {path}")


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict | None:
    json_path = Path(path)
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_report(path: str) -> str:
    report_path = Path(path)
    if not report_path.exists():
        return ""
    return report_path.read_text(encoding="utf-8")


def matched_configs(pattern: str) -> list[Path]:
    return sorted(Path(path) for path in glob.glob(pattern))


def configs_by_fold(pattern: str) -> dict[int, Path]:
    result = {}
    for path in matched_configs(pattern):
        cfg = load_yaml(path)
        result[int(cfg["data"]["valid_fold"])] = path
    return result


def checkpoint_path(config_path: Path, checkpoint_name: str) -> Path:
    cfg = load_yaml(config_path)
    return Path(cfg["train"]["output_dir"]) / checkpoint_name


def summarize_checkpoint_status(
    config_paths: list[Path],
    checkpoint_name: str,
    model_name: str = "model",
) -> pd.DataFrame:
    rows = []
    for path in config_paths:
        cfg = load_yaml(path)
        checkpoint = checkpoint_path(path, checkpoint_name)
        rows.append(
            {
                "model": model_name,
                "fold": int(cfg["data"]["valid_fold"]),
                "config": str(path),
                "checkpoint": str(checkpoint),
                "ready": checkpoint.exists(),
            }
        )
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_trained_model(config_path: str, checkpoint: str, device_name: str):
    cfg = load_yaml(config_path)
    device = get_device(device_name)
    model = build_model(
        cfg["model"]["name"],
        in_channels=int(cfg["model"]["in_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        encoder_weights=None,
        classification_head=bool(cfg["model"].get("classification_head", False)),
    ).to(device)
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("ema_model") or ckpt["model"])
    model.eval()
    return model, cfg


def scan_files(scan_dir: Path) -> dict[int, Path]:
    files = {}
    for path in scan_dir.glob("slice_*.png"):
        try:
            slice_id = int(path.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        files[slice_id] = path
    if not files:
        raise FileNotFoundError(f"No slice files found under {scan_dir}")
    return files


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    image = image.astype(np.float32)
    max_value = float(image.max())
    return image / max_value if max_value > 0 else image


def load_image_stack(row: pd.Series, image_size: int, slice_window: int) -> torch.Tensor:
    scan_dir = Path(row.image_path).parent
    files = scan_files(scan_dir)
    half = slice_window // 2
    channels = []
    for offset in range(-half, half + 1):
        target = int(row.slice) + offset
        path = files[target] if target in files else files[min(files, key=lambda key: abs(key - target))]
        image = load_grayscale(path)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        channels.append(image)
    stack = np.stack(channels, axis=0).astype(np.float32)
    return torch.from_numpy(stack).unsqueeze(0)


def load_center_image(row: pd.Series, image_size: int) -> np.ndarray:
    image = load_grayscale(Path(row.image_path))
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def load_truth_mask(row: pd.Series, labels: pd.DataFrame, image_size: int) -> np.ndarray:
    rows = labels[labels["id"] == row.id]
    mask = decode_multiclass(rows, (int(row.height), int(row.width)), CLASSES)
    resized = [
        cv2.resize(channel, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        for channel in mask
    ]
    return np.stack(resized, axis=0).astype(np.uint8)


def to_rgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    return np.repeat((image * 255).astype(np.uint8)[:, :, None], 3, axis=2)


def colorize(mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    canvas = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.float32)
    occupied = np.zeros(mask.shape[1:], dtype=bool)
    for channel_mask, color in zip(mask, COLORS):
        active = channel_mask > 0
        canvas[active] = color
        occupied |= active
    canvas[~occupied] = 0
    return np.clip(canvas * alpha, 0, 255).astype(np.uint8)


def overlay(base: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    rgb = to_rgb(base).astype(np.float32)
    color = colorize(mask, alpha=1.0).astype(np.float32)
    active = mask.max(axis=0) > 0
    result = rgb.copy()
    result[active] = rgb[active] * (1.0 - alpha) + color[active] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = float((pred * target).sum())
    denominator = float(pred.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return (2.0 * intersection + 1e-7) / (denominator + 1e-7)


@torch.no_grad()
def predict_selected_slice(
    config_paths: list[Path],
    checkpoint_name: str,
    row: pd.Series,
    device_name: str,
    use_tta: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    first_cfg = load_yaml(config_paths[0])
    image_size = int(first_cfg["data"]["image_size"])
    slice_window = int(first_cfg["data"].get("slice_window", first_cfg["model"]["in_channels"]))
    images = load_image_stack(row, image_size=image_size, slice_window=slice_window)
    prob_sum = None
    cls_sum = None
    for config_path in config_paths:
        checkpoint = checkpoint_path(config_path, checkpoint_name)
        model, cfg = load_trained_model(str(config_path), str(checkpoint), device_name)
        device = next(model.parameters()).device
        logits, cls_logits = predict_logits(model, images.to(device), tta=use_tta)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        prob_sum = probs if prob_sum is None else prob_sum + probs
        if cls_logits is not None:
            cls_probs = torch.sigmoid(cls_logits)[0].detach().cpu().numpy()
            cls_sum = cls_probs if cls_sum is None else cls_sum + cls_probs
    probs = prob_sum / len(config_paths)
    cls_probs = cls_sum / len(config_paths) if cls_sum is not None else None
    return probs, cls_probs, first_cfg


@torch.no_grad()
def predict_weighted_ensemble_slice(
    model_a_configs: list[Path],
    model_b_configs: list[Path],
    model_a_checkpoint: str,
    model_b_checkpoint: str,
    weight_a: float,
    weight_b: float,
    postprocess_source: str,
    row: pd.Series,
    device_name: str,
    use_tta: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    total = weight_a + weight_b
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    weight_a = weight_a / total
    weight_b = weight_b / total

    probs_a, cls_a, cfg_a = predict_selected_slice(
        model_a_configs,
        checkpoint_name=model_a_checkpoint,
        row=row,
        device_name=device_name,
        use_tta=use_tta,
    )
    probs_b, cls_b, cfg_b = predict_selected_slice(
        model_b_configs,
        checkpoint_name=model_b_checkpoint,
        row=row,
        device_name=device_name,
        use_tta=use_tta,
    )
    if probs_a.shape != probs_b.shape:
        raise ValueError(f"Probability shape mismatch: {probs_a.shape} vs {probs_b.shape}")
    probs = weight_a * probs_a + weight_b * probs_b

    if cls_a is None or cls_b is None:
        cls_probs = None
    else:
        if cls_a.shape != cls_b.shape:
            raise ValueError(f"Classification probability shape mismatch: {cls_a.shape} vs {cls_b.shape}")
        cls_probs = weight_a * cls_a + weight_b * cls_b

    cfg = cfg_b if postprocess_source == "b" else cfg_a
    return probs, cls_probs, cfg


def build_postprocess_controls(default_cfg: dict) -> dict:
    post_cfg = default_cfg.get("postprocess", {})
    defaults = {
        "mask_thresholds": post_cfg.get("mask_thresholds", [0.5] * len(CLASSES)),
        "cls_thresholds": post_cfg.get("cls_thresholds", [0.5] * len(CLASSES)),
        "min_area": post_cfg.get("min_area", [0] * len(CLASSES)),
    }
    custom = st.sidebar.toggle("Adjust postprocess", value=False)
    if not custom:
        return defaults
    values = {"mask_thresholds": [], "cls_thresholds": [], "min_area": []}
    for idx, name in enumerate(CLASSES):
        with st.sidebar.expander(CLASS_LABELS[name], expanded=idx == 0):
            values["mask_thresholds"].append(
                st.slider(
                    "Mask threshold",
                    0.05,
                    0.95,
                    float(defaults["mask_thresholds"][idx]),
                    0.05,
                    key=f"mask_thr_{name}",
                )
            )
            values["cls_thresholds"].append(
                st.slider(
                    "Classification gate",
                    0.0,
                    0.99,
                    float(defaults["cls_thresholds"][idx]),
                    0.01,
                    key=f"cls_thr_{name}",
                )
            )
            values["min_area"].append(
                st.number_input(
                    "Min area",
                    min_value=0,
                    max_value=4096,
                    value=int(defaults["min_area"][idx]),
                    step=8,
                    key=f"min_area_{name}",
                )
            )
    return values


def select_case_controls(meta: pd.DataFrame) -> pd.Series:
    cases = sorted(meta["case"].unique())
    case = st.sidebar.selectbox("Case", cases, index=0)
    case_meta = meta[meta["case"] == case]
    days = sorted(case_meta["day"].unique(), key=lambda item: int(str(item).replace("day", "")))
    day = st.sidebar.selectbox("Day", days, index=0)
    volume_meta = case_meta[case_meta["day"] == day].sort_values("slice").reset_index(drop=True)
    positive_only = st.sidebar.toggle("Show positive slices only", value=False)
    visible = volume_meta[volume_meta["has_mask"] == 1].reset_index(drop=True) if positive_only else volume_meta
    if visible.empty:
        visible = volume_meta
    slice_values = visible["slice"].astype(int).tolist()
    selected_slice = st.sidebar.select_slider("Slice", options=slice_values, value=slice_values[len(slice_values) // 2])
    row = visible[visible["slice"].astype(int) == int(selected_slice)].iloc[0]
    return row


def render_header(row: pd.Series, bundle_name: str, fold_count: int, checkpoint_label: str) -> None:
    st.title("UWGI Dual-Model Segmentation Workbench")
    st.caption("2.5D MRI segmentation for large bowel, small bowel, and stomach.")
    st.success("Using the selected model bundle with OOF-tuned postprocess settings.")
    chips = [
        bundle_name,
        f"case {row.case}",
        f"day {row.day}",
        f"slice {int(row.slice):04d}",
        f"{fold_count} fold(s)",
        checkpoint_label,
    ]
    st.markdown("".join(f"<span class='uwgi-chip'>{chip}</span>" for chip in chips), unsafe_allow_html=True)


def render_summary_metrics(
    pred_mask: np.ndarray,
    truth_mask: np.ndarray,
    cls_probs: np.ndarray | None,
) -> None:
    cols = st.columns(6)
    mean_dice = float(np.mean([dice_score(pred_mask[idx], truth_mask[idx]) for idx in range(len(CLASSES))]))
    pred_area = int(pred_mask.sum())
    truth_area = int(truth_mask.sum())
    cols[0].metric("Mean Dice", f"{mean_dice:.4f}")
    cols[1].metric("Predicted Area", f"{pred_area:,}")
    cols[2].metric("Truth Area", f"{truth_area:,}")
    for idx, name in enumerate(CLASSES):
        value = cls_probs[idx] if cls_probs is not None else np.nan
        cols[idx + 3].metric(CLASS_LABELS[name], "n/a" if np.isnan(value) else f"{value:.3f}")


def render_prediction_tabs(
    base: np.ndarray,
    probs: np.ndarray,
    pred_mask: np.ndarray,
    truth_mask: np.ndarray,
    row: pd.Series,
    alpha: float,
) -> None:
    tab_overview, tab_classes, tab_export = st.tabs(["Overview", "Class Detail", "Export"])
    with tab_overview:
        left, middle, right = st.columns(3)
        left.image(to_rgb(base), caption="MRI slice", width="stretch")
        middle.image(overlay(base, truth_mask, alpha), caption="Ground truth overlay", width="stretch")
        right.image(overlay(base, pred_mask, alpha), caption="Prediction overlay", width="stretch")
    with tab_classes:
        rows = []
        for idx, name in enumerate(CLASSES):
            rows.append(
                {
                    "organ": CLASS_LABELS[name],
                    "dice": dice_score(pred_mask[idx], truth_mask[idx]),
                    "predicted_pixels": int(pred_mask[idx].sum()),
                    "truth_pixels": int(truth_mask[idx].sum()),
                    "mean_probability": float(probs[idx].mean()),
                    "max_probability": float(probs[idx].max()),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        class_cols = st.columns(3)
        for idx, name in enumerate(CLASSES):
            panel = class_cols[idx]
            panel.image((np.clip(probs[idx], 0, 1) * 255).astype(np.uint8), caption=f"{CLASS_LABELS[name]} probability", width="stretch")
            panel.image(overlay(base, pred_mask[idx : idx + 1], alpha), caption=f"{CLASS_LABELS[name]} mask", width="stretch")
    with tab_export:
        export_rows = []
        for idx, name in enumerate(CLASSES):
            export_rows.append(
                {
                    "id": f"{row.id}_{name}",
                    "class": name,
                    "predicted": rle_encode(pred_mask[idx]),
                }
            )
        export_df = pd.DataFrame(export_rows)
        st.dataframe(export_df, hide_index=True, width="stretch")
        st.download_button(
            "Download current RLE CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{row.id}_prediction.csv",
            mime="text/csv",
        )


def render_case_context(meta: pd.DataFrame, row: pd.Series) -> None:
    volume = meta[(meta["case"] == row.case) & (meta["day"] == row.day)].sort_values("slice")
    counts = volume[["has_large_bowel", "has_small_bowel", "has_stomach"]].sum()
    st.markdown("<div class='uwgi-section'></div>", unsafe_allow_html=True)
    st.subheader("Case Context")
    cols = st.columns(4)
    cols[0].metric("Slices", f"{len(volume):,}")
    cols[1].metric("Large bowel positive", f"{int(counts['has_large_bowel']):,}")
    cols[2].metric("Small bowel positive", f"{int(counts['has_small_bowel']):,}")
    cols[3].metric("Stomach positive", f"{int(counts['has_stomach']):,}")
    chart = volume[["slice", "has_large_bowel", "has_small_bowel", "has_stomach"]].rename(
        columns={
            "has_large_bowel": "Large bowel",
            "has_small_bowel": "Small bowel",
            "has_stomach": "Stomach",
        }
    )
    st.bar_chart(chart.set_index("slice"), height=170)


def render_oof_panel(preset: dict) -> None:
    result = read_json(preset["oof_json"])
    if not result:
        return
    st.markdown("<div class='uwgi-section'></div>", unsafe_allow_html=True)
    st.subheader("OOF Validation")
    if preset.get("kind") == "ensemble" and "best" in result:
        best = result["best"]
        summary = best.get("summary", {}).get("summary", {})
        classes = best.get("summary", {}).get("classes", {})
        st.caption(
            f"Best weights: Strategy E {best.get('weight_a', 0):.3f}, "
            f"B5 {best.get('weight_b', 0):.3f}"
        )
    else:
        summary = result.get("summary", {}).get("summary", result.get("summary", {}))
        classes = result.get("classes") or result.get("summary", {}).get("classes", {})
    cols = st.columns(3)
    cols[0].metric("Mean Dice", f"{summary.get('mean_dice_all_slices', 0):.4f}")
    cols[1].metric("Positive Dice", f"{summary.get('mean_dice_positive_slices', 0):.4f}")
    cols[2].metric("Empty FP Rate", f"{summary.get('mean_empty_slice_false_positive_rate', 0):.4f}")
    if classes:
        table = []
        for name in CLASSES:
            item = classes.get(name, {})
            table.append(
                {
                    "organ": CLASS_LABELS[name],
                    "dice_all_slices": item.get("dice_all_slices"),
                    "dice_positive_slices": item.get("dice_positive_slices"),
                    "empty_slice_false_positive_rate": item.get("empty_slice_false_positive_rate"),
                }
            )
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


def main() -> None:
    configure_page()
    meta = load_metadata(str(DATA_ROOT))
    labels = load_labels(str(DATA_ROOT))

    preset_name = st.sidebar.selectbox("Model bundle", list(DEFAULT_PRESETS), index=0)
    preset = DEFAULT_PRESETS[preset_name]

    if preset.get("kind") == "ensemble":
        model_a_by_fold = configs_by_fold(preset["model_a_glob"])
        model_b_by_fold = configs_by_fold(preset["model_b_glob"])
        if not model_a_by_fold:
            st.error(f"No configs matched: {preset['model_a_glob']}")
            st.stop()
        if not model_b_by_fold:
            st.error(f"No configs matched: {preset['model_b_glob']}")
            st.stop()

        model_a_checkpoint = st.sidebar.text_input(
            f"{preset['model_a_name']} checkpoint",
            preset["model_a_checkpoint"],
        )
        model_b_checkpoint = st.sidebar.text_input(
            f"{preset['model_b_name']} checkpoint",
            preset["model_b_checkpoint"],
        )
        model_a_status = summarize_checkpoint_status(
            list(model_a_by_fold.values()),
            model_a_checkpoint,
            preset["model_a_name"],
        )
        model_b_status = summarize_checkpoint_status(
            list(model_b_by_fold.values()),
            model_b_checkpoint,
            preset["model_b_name"],
        )
        checkpoint_status = pd.concat([model_a_status, model_b_status], ignore_index=True)

        paired_folds = sorted(set(model_a_by_fold) & set(model_b_by_fold))
        ready_folds = [
            fold
            for fold in paired_folds
            if checkpoint_path(model_a_by_fold[fold], model_a_checkpoint).exists()
            and checkpoint_path(model_b_by_fold[fold], model_b_checkpoint).exists()
        ]
        if not ready_folds:
            st.error("No paired ready checkpoints were found for this dual-model bundle.")
            st.dataframe(checkpoint_status, hide_index=True, width="stretch")
            st.stop()

        default_folds = ready_folds if len(ready_folds) <= 2 else [ready_folds[0]]
        selected_folds = st.sidebar.multiselect("Paired folds for inference", ready_folds, default=default_folds)
        if not selected_folds:
            st.warning("Select at least one paired fold.")
            st.stop()
        selected_model_a_configs = [model_a_by_fold[fold] for fold in selected_folds]
        selected_model_b_configs = [model_b_by_fold[fold] for fold in selected_folds]
        postprocess_cfg_path = selected_model_b_configs[0] if preset.get("postprocess_source") == "b" else selected_model_a_configs[0]
        default_cfg = load_yaml(postprocess_cfg_path)
        checkpoint_label = (
            f"{preset['model_a_name']} {model_a_checkpoint} + "
            f"{preset['model_b_name']} {model_b_checkpoint}"
        )
        fold_count = len(selected_folds)
    else:
        config_paths = matched_configs(preset["glob"])
        if not config_paths:
            st.error(f"No configs matched: {preset['glob']}")
            st.stop()

        checkpoint_name = st.sidebar.text_input("Checkpoint", preset["checkpoint"])
        checkpoint_status = summarize_checkpoint_status(config_paths, checkpoint_name, preset_name)
        ready_configs = [Path(row.config) for row in checkpoint_status.itertuples(index=False) if row.ready]
        if not ready_configs:
            st.error("No ready checkpoints were found for this bundle.")
            st.dataframe(checkpoint_status, hide_index=True, width="stretch")
            st.stop()

        fold_options = [int(load_yaml(path)["data"]["valid_fold"]) for path in ready_configs]
        default_folds = fold_options if len(fold_options) <= 2 else [fold_options[0]]
        selected_folds = st.sidebar.multiselect("Folds for inference", fold_options, default=default_folds)
        selected_configs = [
            path
            for path in ready_configs
            if int(load_yaml(path)["data"]["valid_fold"]) in set(selected_folds)
        ]
        if not selected_configs:
            st.warning("Select at least one fold.")
            st.stop()
        default_cfg = load_yaml(selected_configs[0])
        checkpoint_label = checkpoint_name
        fold_count = len(selected_configs)

    device_choice = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
    use_tta = st.sidebar.toggle("Horizontal flip TTA", value=bool(default_cfg.get("inference", {}).get("tta", True)))
    alpha = st.sidebar.slider("Overlay opacity", 0.2, 0.8, 0.48, 0.02)
    row = select_case_controls(meta)
    post_cfg = build_postprocess_controls(default_cfg)

    render_header(row, preset_name, fold_count, checkpoint_label)
    status_col, action_col = st.columns([0.72, 0.28])
    with status_col:
        st.dataframe(checkpoint_status, hide_index=True, width="stretch")
    run = action_col.button("Run Segmentation", type="primary", width="stretch")

    if not run and "last_prediction" not in st.session_state:
        st.info("Choose a case and run segmentation to load the selected model bundle.")
        render_case_context(meta, row)
        render_oof_panel(preset)
        return

    if run:
        with st.spinner("Running model inference..."):
            if preset.get("kind") == "ensemble":
                probs, cls_probs, cfg = predict_weighted_ensemble_slice(
                    selected_model_a_configs,
                    selected_model_b_configs,
                    model_a_checkpoint=model_a_checkpoint,
                    model_b_checkpoint=model_b_checkpoint,
                    weight_a=float(preset["model_a_weight"]),
                    weight_b=float(preset["model_b_weight"]),
                    postprocess_source=str(preset["postprocess_source"]),
                    row=row,
                    device_name=device_choice,
                    use_tta=use_tta,
                )
            else:
                probs, cls_probs, cfg = predict_selected_slice(
                    selected_configs,
                    checkpoint_name=checkpoint_name,
                    row=row,
                    device_name=device_choice,
                    use_tta=use_tta,
                )
            image_size = int(cfg["data"]["image_size"])
            base = load_center_image(row, image_size)
            truth_mask = load_truth_mask(row, labels, image_size)
            st.session_state["last_prediction"] = {
                "preset_name": preset_name,
                "row_id": row.id,
                "base": base,
                "probs": probs,
                "cls_probs": cls_probs,
                "truth_mask": truth_mask,
            }

    prediction = st.session_state["last_prediction"]
    if prediction["row_id"] != row.id:
        st.warning("The displayed prediction belongs to the previous selected slice. Run segmentation again for the current slice.")
    if prediction.get("preset_name") != preset_name:
        st.warning("The displayed prediction belongs to a different model bundle. Run segmentation again for the selected bundle.")

    pred_mask = postprocess_slice(
        prediction["probs"],
        cls_probs=prediction["cls_probs"],
        mask_thresholds=post_cfg["mask_thresholds"],
        cls_thresholds=post_cfg["cls_thresholds"],
        min_area=post_cfg["min_area"],
    ).astype(np.uint8)
    render_summary_metrics(pred_mask, prediction["truth_mask"], prediction["cls_probs"])
    render_prediction_tabs(
        prediction["base"],
        prediction["probs"],
        pred_mask,
        prediction["truth_mask"],
        row,
        alpha,
    )
    render_case_context(meta, row)
    render_oof_panel(preset)

    report = read_report(preset["report"])
    if report:
        with st.expander("Pipeline Report", expanded=False):
            st.markdown(report)


if __name__ == "__main__":
    main()
