# Final Solution Report

Generated: 2026-06-25

## Executive Summary

The final local solution uses `UNet++ EfficientNet-B5` as the primary model and
keeps `maskfix Strategy E` as a frozen auxiliary model for inference-time
diversity. The selected blend is:

```text
0.70 * B5 + 0.30 * Strategy E
```

This is better than either standalone model on local OOF Dice. B5 remains the
main model line because it also wins the official-metric proxy as a standalone
model.

## Why Ensemble Helps

Strategy E is weaker on average than B5, but its errors are not identical. The
weighted probability blend improves the final mask probabilities in a subset of
cases, especially around uncertain boundaries, small structures, and slice-level
presence decisions.

The measured gain over pure B5 is modest but real in OOF:

```text
0.9226132123 - 0.9210980579 = +0.0015151545
```

This is enough to keep Strategy E in the final inference path, while still
treating B5 as the model-development baseline.

## Final Metrics

### Local OOF Dice

| Model | Mean Dice | Positive Dice | Empty FP Rate |
| --- | ---: | ---: | ---: |
| Strategy E | 0.9188102194 | 0.7634351159 | 0.0170830851 |
| B5 | 0.9210980579 | 0.7687032270 | 0.0156804169 |
| 0.30 Strategy E + 0.70 B5 | 0.9226132123 | 0.7754965725 | 0.0159891921 |

### Ensemble Per-Organ OOF

| Organ | Dice All Slices | Positive Dice | Empty FP Rate |
| --- | ---: | ---: | ---: |
| large_bowel | 0.9134744122 | 0.8022088017 | 0.0223260006 |
| small_bowel | 0.8993136864 | 0.6964538586 | 0.0174390914 |
| stomach | 0.9550515384 | 0.8278270572 | 0.0082024842 |

### Official-Metric Proxy

The exact Kaggle private evaluator is not included in this repository. The
local proxy reports 3D Dice, HD95 in millimeters, and a normalized Hausdorff
score combined using a competition-style `0.4 Dice / 0.6 Hausdorff` weighting.

| Model | Combined Proxy | 3D Dice | HD95 mm |
| --- | ---: | ---: | ---: |
| Strategy E | 0.9091636213 | 0.8039622003 | 17.9245 |
| B5 | 0.9115916074 | 0.8086270730 | 17.0627 |

Small bowel remains the weak organ:

| Model | Small Bowel 3D Dice | Small Bowel HD95 mm | Small Bowel Combined Proxy |
| --- | ---: | ---: | ---: |
| Strategy E | 0.7271530957 | 22.3210 | 0.8762803556 |
| B5 | 0.7316217186 | 21.5878 | 0.8785468738 |

## Final Artifacts

### Model Configs

```text
configs/h200_next_unetpp_b5.yaml
configs/h200_next_unetpp_b5_folds/h200_next_unetpp_b5_fold*.yaml
configs/h200_maskfix_stage1_strategy_e.yaml
configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml
```

### Checkpoints

```text
outputs/h200_next_unetpp_b5_fold*/best_postprocess.pt
outputs/h200_maskfix_stage1_strategy_e_fold*/best_postprocess.pt
```

### Evaluation Outputs

```text
outputs/b5_full_pipeline/final_report.md
outputs/b5_full_pipeline/status.json
outputs/h200_next_unetpp_b5_oof/h200_stage1_eval_config_component_postprocess.json
outputs/h200_next_unetpp_b5_oof/h200_next_unetpp_b5_official_oof_proxy.json
outputs/maskfix_oof/maskfix_strategy_e_official_oof_proxy.json
outputs/ensemble_strategy_e_b5/strategy_e_b5_weight_search.json
```

### Final Submission Script

```text
scripts/make_ensemble_submission.py
```

## Reproduction Commands

### Full Evaluation Pipeline

```bash
/mnt/disk2/hjj/uwgiseg/bin/python -u scripts/run_b5_full_pipeline.py \
  --check-seconds 300 \
  --gpus 0 \
  --max-workers 5
```

### Final Submission

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

## Operational Guidance

- Treat B5 as the main model for future improvements.
- Keep Strategy E frozen as an ensemble contributor.
- Do not spend more time tuning Strategy E unless a new auxiliary model is being
  compared against it.
- Prioritize small-bowel improvements only if they also improve whole-model OOF
  or official proxy; isolated small-bowel gains can easily increase false
  positives.
- Use `scripts/make_submission.py` only for single-family ablations, not final
  submission.

## Known Limitations

- The official proxy is not the exact Kaggle hidden evaluator.
- The local sample submission is empty because Kaggle test images are hidden.
- Ensemble inference is slower and uses more memory than pure B5.
- The ensemble gain is small, so leaderboard variance is still possible.
