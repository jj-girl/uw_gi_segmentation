# Stage 1 Auto Pipeline Report

> Invalidated on 2026-06-23: this report was generated before the RLE
> decode and image-dimension mask fix. Do not use these metrics as valid
> results. Use the new `outputs/maskfix_oof` report after retraining.

Generated: 2026-06-23 01:51:41 UTC
Revised: 2026-06-23, final selection set to `min_area + z-axis`.

## Status

| Step | Status |
| --- | --- |
| threshold_search | ran |
| minarea_z_search | ran |
| component_search | ran |
| config_update | updated 6 to selected `min_area + z-axis` params |
| config_eval | component candidate evaluated; not selected |

## Checkpoints

| Fold | Checkpoint | Exists |
| --- | --- | --- |
| 0 | `outputs/h200_stage1_strategy_e_postprocess_aware_fold0/best_postprocess.pt` | True |
| 1 | `outputs/h200_stage1_strategy_e_postprocess_aware_fold1/best_postprocess.pt` | True |
| 2 | `outputs/h200_stage1_strategy_e_postprocess_aware_fold2/best_postprocess.pt` | True |
| 3 | `outputs/h200_stage1_strategy_e_postprocess_aware_fold3/best_postprocess.pt` | True |
| 4 | `outputs/h200_stage1_strategy_e_postprocess_aware_fold4/best_postprocess.pt` | True |

## Metrics

| Stage | Mean Dice | Positive Dice | Empty FP Rate |
| --- | ---: | ---: | ---: |
| OOF threshold + cls gate | 0.8184999148 | n/a | n/a |
| selected: min_area + z | 0.8188877215 | 0.4273936246 | 0.0150056935 |
| component candidate | 0.8188721092 | 0.4270471362 | 0.0148688635 |

Decision: use `min_area + z-axis`; component filtering reduces empty false positives slightly but lowers mean Dice by `0.0000156`.

## Recommended Postprocess

| Organ | Mask Thr | Cls Thr | Min Area | Z Min Run | Min Volume | Keep Largest |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| large_bowel | 0.25 | 0.90 | 16 | 3 | 0 | False |
| small_bowel | 0.25 | 0.90 | 96 | 3 | 0 | False |
| stomach | 0.25 | 0.90 | 192 | 2 | 0 | False |

## Artifacts

- Thresholds: `outputs/oof_strategy_e/h200_stage1_thresholds_with_cls_gate.json`
- selected min_area + z search: `outputs/oof_strategy_e/h200_stage1_postprocess_minarea_z_search.json`
- component candidate search: `outputs/oof_strategy_e/h200_stage1_component_parallel_search.json`
- component candidate eval: `outputs/oof_strategy_e/h200_stage1_eval_config_component_postprocess.json`
- Managed configs:
  - `configs/h200_stage1_strategy_e_postprocess_aware.yaml`
  - `configs/h200_stage1_strategy_e_folds/h200_stage1_strategy_e_postprocess_aware_fold0.yaml`
  - `configs/h200_stage1_strategy_e_folds/h200_stage1_strategy_e_postprocess_aware_fold1.yaml`
  - `configs/h200_stage1_strategy_e_folds/h200_stage1_strategy_e_postprocess_aware_fold2.yaml`
  - `configs/h200_stage1_strategy_e_folds/h200_stage1_strategy_e_postprocess_aware_fold3.yaml`
  - `configs/h200_stage1_strategy_e_folds/h200_stage1_strategy_e_postprocess_aware_fold4.yaml`
