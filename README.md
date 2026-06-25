# UW-Madison GI Tract Image Segmentation

PyTorch solution workspace for the Kaggle UW-Madison GI Tract Image
Segmentation competition.

## Final Landing State

The final local solution is:

- Primary model: `UNet++ EfficientNet-B5`, 5 folds.
- Auxiliary model: `maskfix Strategy E`, 5 folds.
- Final inference: probability blend with `0.70 * B5 + 0.30 * Strategy E`.
- Postprocess source: B5 fold configs.

This is the best validated local OOF option currently available. B5 is the main
model because it beats Strategy E by itself; Strategy E is retained only because
its errors are not fully correlated with B5 and the weighted blend improves OOF.

## Final Metrics

### Local OOF Dice

| Model | Mean Dice | Positive Dice | Empty FP Rate |
| --- | ---: | ---: | ---: |
| Strategy E | 0.9188102194 | 0.7634351159 | 0.0170830851 |
| B5 | 0.9210980579 | 0.7687032270 | 0.0156804169 |
| 0.30 Strategy E + 0.70 B5 | 0.9226132123 | 0.7754965725 | 0.0159891921 |

### Official-Metric Proxy

The exact Kaggle private evaluator is not public in this repository. The local
proxy reports 3D Dice, HD95 in mm, and a normalized Hausdorff score combined
with the competition-style `0.4 Dice / 0.6 Hausdorff` weighting.

| Model | Combined Proxy | 3D Dice | HD95 mm |
| --- | ---: | ---: | ---: |
| Strategy E | 0.9091636213 | 0.8039622003 | 17.9245 |
| B5 | 0.9115916074 | 0.8086270730 | 17.0627 |

Small bowel remains the weakest organ, but B5 improves its official proxy over
Strategy E:

| Model | Small Bowel 3D Dice | Small Bowel HD95 mm | Small Bowel Combined Proxy |
| --- | ---: | ---: | ---: |
| Strategy E | 0.7271530957 | 22.3210 | 0.8762803556 |
| B5 | 0.7316217186 | 21.5878 | 0.8785468738 |

## Key Artifacts

| Purpose | Path |
| --- | --- |
| Final report | `outputs/b5_full_pipeline/final_report.md` |
| Final pipeline status | `outputs/b5_full_pipeline/status.json` |
| B5 OOF report | `outputs/h200_next_unetpp_b5_oof/h200_next_unetpp_b5_auto_report.md` |
| B5 OOF eval JSON | `outputs/h200_next_unetpp_b5_oof/h200_stage1_eval_config_component_postprocess.json` |
| B5 official proxy JSON | `outputs/h200_next_unetpp_b5_oof/h200_next_unetpp_b5_official_oof_proxy.json` |
| Strategy E official proxy JSON | `outputs/maskfix_oof/maskfix_strategy_e_official_oof_proxy.json` |
| Ensemble weight search JSON | `outputs/ensemble_strategy_e_b5/strategy_e_b5_weight_search.json` |

## Active Model Files

### B5 Primary Model

```text
configs/h200_next_unetpp_b5.yaml
configs/h200_next_unetpp_b5_folds/h200_next_unetpp_b5_fold*.yaml
outputs/h200_next_unetpp_b5_fold*/best_postprocess.pt
```

### Strategy E Auxiliary Model

```text
configs/h200_maskfix_stage1_strategy_e.yaml
configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml
outputs/h200_maskfix_stage1_strategy_e_fold*/best_postprocess.pt
```

Do not use old pre-maskfix, baseline, DeepLabV3+, small_bowel_aware, or
nnunet_route artifacts as active project state.

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

## Reproduce Final Evaluation Pipeline

The completed end-to-end supervisor waits for B5 folds, runs B5 OOF
postprocess search, runs official proxy for both model lines, runs ensemble
weight search, and writes the final report:

```bash
/mnt/disk2/hjj/uwgiseg/bin/python -u scripts/run_b5_full_pipeline.py \
  --check-seconds 300 \
  --gpus 0 \
  --max-workers 5
```

The completed status is:

```text
outputs/b5_full_pipeline/status.json
```

## Generate Final Submission

Use the final two-family ensemble submission script:

```bash
/mnt/disk2/hjj/uwgiseg/bin/python scripts/make_ensemble_submission.py \
  --model-a-glob 'configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml' \
  --model-b-glob 'configs/h200_next_unetpp_b5_folds/h200_next_unetpp_b5_fold*.yaml' \
  --model-a-checkpoint best_postprocess.pt \
  --model-b-checkpoint best_postprocess.pt \
  --weight-a 0.3 \
  --weight-b 0.7 \
  --postprocess-source b \
  --sample-submission data/raw/uw-madison-gi-tract-image-segmentation/sample_submission.csv \
  --data-root data/raw/uw-madison-gi-tract-image-segmentation \
  --out outputs/final_submissions/strategy_e_b5_030_070_submission.csv
```

The local Kaggle data has an empty `sample_submission.csv` because test images
are hidden. On this machine the command writes a valid empty CSV and manifest.
Run the same command in the Kaggle test environment or an environment with the
hidden `test/` images to generate the real submission.

## Important Mask Fix

Two data bugs were fixed on 2026-06-23:

- RLE masks now use row-major reshape/flatten.
- Scan filename dimensions are parsed as `width, height`, matching `cv2` image
  shape `(height, width)`.

Models trained before this fix are invalid and should not be used for model
selection or submission.

## Documentation

- Final solution report: `docs/final_solution_report.md`
- Current system state: `docs/current_system_state.md`
- Submission pipeline: `docs/submission_pipeline.md`
- B5 full pipeline report: `outputs/b5_full_pipeline/final_report.md`
