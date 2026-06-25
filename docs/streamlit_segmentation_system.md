# Streamlit Medical Segmentation Workbench

The Streamlit app is a local visual workbench for the final UWGI system. It
supports the selected dual-branch fusion model and both standalone model families for
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
| `Dual-branch fusion` | Default. Uses B5 as the primary branch and the B3 auxiliary branch as the complementary branch, with B5 postprocess settings. |
| `B5 primary 5-fold` | Standalone B5 branch for baseline inspection. |
| `B3 auxiliary 5-fold` | Standalone B3 auxiliary branch for comparison and debugging. |

The dual-branch bundle requires paired ready checkpoints for both model families:

```text
outputs/h200_next_unetpp_b5_fold*/best_postprocess.pt
outputs/h200_maskfix_stage1_strategy_e_fold*/best_postprocess.pt
```

## Features

- Browse `case/day/slice` from the local UW-Madison GI dataset.
- Run selected-fold 2.5D inference with optional horizontal flip TTA.
- Compare dual-branch fusion, B5-only, and B3-only predictions.
- Show MRI, ground-truth overlay, prediction overlay, class probability maps,
  per-organ Dice, predicted area, and RLE export.
- Inspect case-level positive-slice distribution and OOF validation results.
- Adjust mask thresholds, classification gates, and min-area postprocess values
  from the sidebar.

## Notes

- Select one paired fold for quick UI interaction or all five folds for the
  full dual-branch visualization.
- The app uses corrected mask decoding and dimensions.
- The local Kaggle test set is hidden, so the app focuses on validation and
  train-set case review.
- Switching model bundles keeps the old prediction visible until `Run
  Segmentation` is clicked again; the app shows a warning when the displayed
  prediction belongs to a different bundle.
