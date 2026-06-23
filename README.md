# UW-Madison GI Tract Image Segmentation

Modern medical image segmentation baseline for the Kaggle UW-Madison GI Tract Image Segmentation competition.

This project is the new direction for the prior pneumothorax segmentation coursework: a cleaner, reproducible PyTorch project for multi-class abdominal MRI segmentation.

## Task

Segment three gastrointestinal organs from 2D MRI slices:

- large bowel
- small bowel
- stomach

The Kaggle labels are run-length encoded masks in `train.csv`.

## Project Layout

```text
uw_gi_segmentation/
  configs/              training configs
  data/
    raw/                original Kaggle files
    processed/          optional cached/preprocessed files
  outputs/              checkpoints, logs, predictions
  scripts/              data migration and utility scripts
  src/uwgi/             package source
```

## Data

Download the competition data from Kaggle and place the zip under either:

- `C:\Users\Xinjing\Downloads`
- `data/raw`

Then run:

```bash
python scripts/prepare_data.py --zip "C:\Users\Xinjing\Downloads\uw-madison-gi-tract-image-segmentation.zip"
```

If Kaggle API is configured, you can also run:

```bash
python scripts/prepare_data.py --download
```

Expected raw structure after extraction:

```text
data/raw/uw-madison-gi-tract-image-segmentation/
  train.csv
  sample_submission.csv
  train/
    case*/
      case*_day*/
        scans/*.png
```

## Baseline Train

```bash
python -m src.uwgi.train --config configs/unet_baseline.yaml
```

## Recommended Model Roadmap

1. `UNetSmall`: local baseline, no external model zoo dependency.
2. `smp_unet_resnet34`: stable and strong medical segmentation baseline.
3. `smp_unetplusplus_efficientnet_b3`: stronger encoder and decoder, good second model.
4. `smp_deeplabv3plus_resnet50`: atrous/spatial context baseline.
5. `2.5D UNet`: use adjacent MRI slices as channels for sequence context.
