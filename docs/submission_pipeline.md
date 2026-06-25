# Final Submission Pipeline

## Current Bundle

Final submission uses a two-family weighted probability ensemble:

| Family | Config Glob | Checkpoint | Weight |
| --- | --- | --- | ---: |
| Strategy E | `configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold*.yaml` | `best_postprocess.pt` | 0.30 |
| B5 | `configs/h200_next_unetpp_b5_folds/h200_next_unetpp_b5_fold*.yaml` | `best_postprocess.pt` | 0.70 |

Postprocess parameters come from the B5 fold configs.

## Command

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

The script writes a companion manifest next to the CSV.

## Local Dataset Note

The downloaded Kaggle data has an empty `sample_submission.csv`, because test
images are hidden. On this machine the command writes a valid empty CSV and a
manifest with `status=empty_sample_submission`.

For a real Kaggle submission, run the same command where the hidden test
`sample_submission.csv` and `test/` images are available.

## Validation Evidence

Final weight search:

```text
outputs/ensemble_strategy_e_b5/strategy_e_b5_weight_search.json
```

Best local OOF result:

| Strategy E Weight | B5 Weight | Mean Dice | Positive Dice | Empty FP Rate |
| ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.70 | 0.9226132123 | 0.7754965725 | 0.0159891921 |

Official proxy artifacts:

```text
outputs/maskfix_oof/maskfix_strategy_e_official_oof_proxy.json
outputs/h200_next_unetpp_b5_oof/h200_next_unetpp_b5_official_oof_proxy.json
```

## Legacy Single-Family Submission

`scripts/make_submission.py` is still useful for single-family smoke tests or
ablation submissions. It should not be used for the final selected submission,
because it cannot blend B5 with Strategy E.
