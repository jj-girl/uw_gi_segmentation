# Maskfix Submission Pipeline

## Current Bundle

- Configs: `configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml`
- Checkpoint: `best_postprocess.pt`
- Ensemble: 5-fold arithmetic mean of segmentation probabilities and classification probabilities.
- Postprocess:
  - `mask_thresholds: [0.25, 0.25, 0.25]`
  - `cls_thresholds: [0.70, 0.80, 0.20]`
  - `min_area: [48, 48, 48]`
  - `z_min_run: [2, 3, 1]`
  - `min_volume: [512, 512, 0]`
  - `keep_largest_component: [false, false, true]`

## Command

```bash
/mnt/disk2/hjj/uwgiseg/bin/python scripts/make_submission.py \
  --fold-config-glob 'configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml' \
  --checkpoint-name best_postprocess.pt \
  --sample-submission data/raw/uw-madison-gi-tract-image-segmentation/sample_submission.csv \
  --data-root data/raw/uw-madison-gi-tract-image-segmentation \
  --out outputs/maskfix_submissions/maskfix_strategy_e_submission.csv
```

The script writes a companion manifest next to the CSV.

## Local Dataset Note

The downloaded Kaggle data has an empty `sample_submission.csv`, because test
images are hidden. On this machine the command writes a valid empty CSV and
manifest with `status=empty_sample_submission`.

For a real Kaggle submission, run the same command in an environment where the
hidden test `sample_submission.csv` and `test/` images are available.

## Validation Done

- `python -m compileall -q src scripts app_streamlit.py`
- Empty local sample path:
  - output: `outputs/maskfix_submissions/maskfix_strategy_e_submission.csv`
  - manifest: `outputs/maskfix_submissions/maskfix_strategy_e_submission.manifest.json`
- Local train-format smoke inference:
  - output: `outputs/maskfix_submissions/local_train_sample_maskfix_strategy_e_submission.csv`
  - manifest: `outputs/maskfix_submissions/local_train_sample_maskfix_strategy_e_submission.manifest.json`

The local train-format check verifies model loading, 5-fold ensemble inference,
postprocess application, resizing, and RLE CSV writing. It does not provide a
leaderboard score.
