# UW-Madison GI Tract Image Segmentation

PyTorch project for multi-class abdominal MRI segmentation on the Kaggle
UW-Madison GI Tract Image Segmentation dataset.

## Current Valid Result

The current valid model family is the mask-coordinate-fixed 5-fold Strategy E
run:

- Configs: `configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml`
- Checkpoints: `outputs/h200_maskfix_stage1_strategy_e_fold*/best_postprocess.pt`
- OOF report: `docs/maskfix_strategy_e_auto_pipeline_report.md`

Final OOF postprocessed result:

| Stage | Mean Dice | Positive Dice | Empty FP Rate |
| --- | ---: | ---: | ---: |
| final config eval | 0.9188102194 | 0.7634351159 | 0.0170830851 |

Recommended postprocess:

```yaml
mask_thresholds: [0.25, 0.25, 0.25]
cls_thresholds: [0.70, 0.80, 0.20]
min_area: [48, 48, 48]
z_min_run: [2, 3, 1]
min_volume: [512, 512, 0]
keep_largest_component: [false, false, true]
component_connectivity: 1
```

Older pre-maskfix metrics, checkpoints, submissions, and OOF artifacts are
invalid because masks were decoded with the wrong coordinate convention. Old
result files were moved to `outputs/invalid_pre_maskfix_artifacts/` for audit
only.

## Important Mask Fix

Two data bugs were fixed on 2026-06-23:

- RLE masks now use row-major reshape/flatten.
- Scan filename dimensions are parsed as `width, height`, matching `cv2` image
  shape `(height, width)`.

Because of this, all models trained before the fix must be treated as invalid.

## Project Layout

```text
configs/      training configs
data/         Kaggle data and generated metadata
docs/         pipeline and UI documentation
outputs/      checkpoints, OOF reports, submissions, archived invalid outputs
scripts/      data, training, OOF, submission, and monitoring utilities
src/uwgi/     package source
```

## Environment

Project virtual environment:

```bash
/mnt/disk2/hjj/uwgiseg/bin/python
```

Install dependencies:

```bash
/mnt/disk2/hjj/uwgiseg/bin/pip install -r requirements.txt
/mnt/disk2/hjj/uwgiseg/bin/pip install segmentation-models-pytorch timm
```

## Data

Expected extracted Kaggle layout:

```text
data/raw/uw-madison-gi-tract-image-segmentation/
  train.csv
  sample_submission.csv
  train/
    case*/
      case*_day*/
        scans/*.png
```

Prepare data:

```bash
/mnt/disk2/hjj/uwgiseg/bin/python scripts/prepare_data.py \
  --zip data/raw/uw-madison-gi-tract-image-segmentation.zip
```

## Train Current Model

Run one fold:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/disk2/hjj/uwgiseg/bin/python -u -m src.uwgi.train \
  --config configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold0.yaml
```

The completed 5-fold run used:

- UNet++ EfficientNet-B3
- 2.5D 5-slice input at `384x384`
- classification head
- Dice + BCE + classification BCE
- EMA
- postprocess-aware checkpoint selection

## OOF And Postprocess

Re-run the current OOF/postprocess pipeline:

```bash
/mnt/disk2/hjj/uwgiseg/bin/python scripts/stage1_auto_pipeline.py \
  --fold-config-glob 'configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml' \
  --main-config configs/h200_maskfix_stage1_strategy_e.yaml \
  --checkpoint-name best_postprocess.pt \
  --out-dir outputs/maskfix_oof \
  --report outputs/maskfix_oof/maskfix_strategy_e_auto_report.md \
  --work-dir outputs/maskfix_oof/component_parallel_work \
  --gpus 0,1 \
  --max-workers 5
```

## Submission

Generate the current maskfix submission:

```bash
/mnt/disk2/hjj/uwgiseg/bin/python scripts/make_submission.py \
  --fold-config-glob 'configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml' \
  --checkpoint-name best_postprocess.pt \
  --sample-submission data/raw/uw-madison-gi-tract-image-segmentation/sample_submission.csv \
  --data-root data/raw/uw-madison-gi-tract-image-segmentation \
  --out outputs/maskfix_submissions/maskfix_strategy_e_submission.csv
```

The local Kaggle test set is hidden, so the local `sample_submission.csv` is
empty. The command still writes a valid empty CSV and manifest.

## Streamlit Workbench

Run the local visualization workbench:

```bash
/mnt/disk2/hjj/uwgiseg/bin/streamlit run app_streamlit.py --server.port 8501
```

The workbench defaults to the current maskfix 5-fold model bundle.
