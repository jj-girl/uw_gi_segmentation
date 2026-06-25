# Streamlit Medical Segmentation Workbench

The Streamlit app is a local visual workbench for the final UWGI system. It
supports the selected dual-model ensemble and both standalone model families for
side-by-side inspection.

## Run

```bash
/mnt/disk2/hjj/uwgiseg/bin/streamlit run app_streamlit.py --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## Model Bundles

The sidebar exposes three compatible bundles:

| Bundle | Role |
| --- | --- |
| `Dual-model ensemble` | Default. Uses `0.30 Strategy E + 0.70 B5` with B5 postprocess settings. |
| `B5 primary 5-fold` | Standalone B5 model for baseline inspection. |
| `Strategy E auxiliary 5-fold` | Standalone Strategy E model for comparison and debugging. |

The dual-model bundle requires paired ready checkpoints for both model families:

```text
outputs/h200_next_unetpp_b5_fold*/best_postprocess.pt
outputs/h200_maskfix_stage1_strategy_e_fold*/best_postprocess.pt
```

## Features

- Browse `case/day/slice` from the local UW-Madison GI dataset.
- Run selected-fold 2.5D inference with optional horizontal flip TTA.
- Compare dual-model ensemble, B5-only, and Strategy-E-only predictions.
- Show MRI, ground-truth overlay, prediction overlay, class probability maps,
  per-organ Dice, predicted area, and RLE export.
- Inspect case-level positive-slice distribution and OOF validation results.
- Adjust mask thresholds, classification gates, and min-area postprocess values
  from the sidebar.

## Notes

- Select one paired fold for quick UI interaction or all five folds for the
  full dual-model visualization.
- The app uses corrected mask decoding and dimensions.
- The local Kaggle test set is hidden, so the app focuses on validation and
  train-set case review.
- Switching model bundles keeps the old prediction visible until `Run
  Segmentation` is clicked again; the app shows a warning when the displayed
  prediction belongs to a different bundle.
