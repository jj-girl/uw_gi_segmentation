# Streamlit Medical Segmentation Workbench

This Streamlit app wraps the current maskfix UWGI model bundle as a local
medical image segmentation workbench.

## Run

```bash
/mnt/disk2/hjj/uwgiseg/bin/streamlit run app_streamlit.py --server.port 8501
```

Then open:

```text
http://localhost:8501
```

## Current Default Model

- Configs: `configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml`
- Checkpoint: `best_postprocess.pt`
- OOF report: `docs/maskfix_strategy_e_auto_pipeline_report.md`

## Features

- Browse `case/day/slice` from the local UW-Madison GI dataset.
- Run single-slice 2.5D inference with optional horizontal flip TTA.
- Show MRI, ground-truth overlay, prediction overlay, class probability maps,
  per-organ Dice, predicted area, and RLE export.
- Inspect case-level positive-slice distribution and OOF validation results.
- Adjust mask thresholds, classification gates, and min-area postprocess values
  from the sidebar.

## Notes

- Select one fold for quick UI interaction or all five folds for ensemble
  visualization.
- The app uses corrected mask decoding and dimensions.
- The local Kaggle test set is hidden, so the app focuses on validation and
  train-set case review.
