# E Strategy Submission Pipeline

## Current Baseline

- Configs: `configs/h200_stage1_strategy_e_folds/h200_stage1_strategy_e_postprocess_aware_fold*.yaml`
- Checkpoint: `best_postprocess.pt`
- Ensemble: 5-fold arithmetic mean of segmentation probabilities and classification probabilities.
- Postprocess: selected `min_area + z-axis`
  - `mask_thresholds: [0.25, 0.25, 0.25]`
  - `cls_thresholds: [0.90, 0.90, 0.90]`
  - `min_area: [16, 96, 192]`
  - `z_min_run: [3, 3, 2]`
  - `min_volume: [0, 0, 0]`
  - `keep_largest_component: [false, false, false]`

## Command

```bash
/mnt/disk2/hjj/uwgiseg/bin/python scripts/make_submission.py \
  --fold-config-glob 'configs/h200_stage1_strategy_e_folds/h200_stage1_strategy_e_postprocess_aware_fold*.yaml' \
  --checkpoint-name best_postprocess.pt \
  --sample-submission data/raw/uw-madison-gi-tract-image-segmentation/sample_submission.csv \
  --data-root data/raw/uw-madison-gi-tract-image-segmentation \
  --out outputs/submissions/strategy_e_minarea_z_submission.csv
```

The script writes a companion manifest next to the CSV.

## Local Dataset Note

The downloaded Kaggle data has an empty `sample_submission.csv`, because test images are hidden. On this machine the command therefore writes a valid empty CSV and manifest with `status=empty_sample_submission`.

For a real Kaggle submission, run the same script in an environment where the hidden test `sample_submission.csv` and `test/` images are available, or adapt a Kaggle notebook to call this script with the same config/checkpoint bundle.

## Validation Done

- `python -m compileall -q src scripts`
- Empty local sample path:
  - output: `outputs/submissions/strategy_e_minarea_z_submission.csv`
  - manifest: `outputs/submissions/strategy_e_minarea_z_submission.manifest.json`
- Local train-format inference check:
  - output: `outputs/submissions/local_train_sample_strategy_e_minarea_z_submission.csv`
  - positive-slice format check: `outputs/submissions/local_positive_strategy_e_minarea_z_submission.csv`

The local checks verify model loading, 5-fold ensemble inference, postprocess application, resizing, and RLE CSV writing. They do not produce a leaderboard score because hidden test data is unavailable locally.
