# Maskfix Strategy E Auto Pipeline Report

Generated: 2026-06-23 18:28:30 UTC

## Status

| Step | Status |
| --- | --- |
| threshold_search | ran |
| minarea_z_search | ran |
| component_search | ran |
| config_update | updated 6 |
| config_eval | ran |

## Checkpoints

| Fold | Checkpoint | Exists |
| --- | --- | --- |
| 0 | `outputs/h200_maskfix_stage1_strategy_e_fold0/best_postprocess.pt` | True |
| 1 | `outputs/h200_maskfix_stage1_strategy_e_fold1/best_postprocess.pt` | True |
| 2 | `outputs/h200_maskfix_stage1_strategy_e_fold2/best_postprocess.pt` | True |
| 3 | `outputs/h200_maskfix_stage1_strategy_e_fold3/best_postprocess.pt` | True |
| 4 | `outputs/h200_maskfix_stage1_strategy_e_fold4/best_postprocess.pt` | True |

## Metrics

| Stage | Mean Dice | Positive Dice | Empty FP Rate |
| --- | ---: | ---: | ---: |
| OOF threshold + cls gate | 0.9173107277 | n/a | n/a |
| min_area + z | 0.9185714246 | 0.7642339701 | 0.0175445504 |
| 3D component search | 0.9188102194 | 0.7634351159 | 0.0170830851 |
| final config eval | 0.9188102194 | 0.7634351159 | 0.0170830851 |

## Recommended Postprocess

| Organ | Mask Thr | Cls Thr | Min Area | Z Min Run | Min Volume | Keep Largest |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| large_bowel | 0.25 | 0.70 | 48 | 2 | 512 | False |
| small_bowel | 0.25 | 0.80 | 48 | 3 | 512 | False |
| stomach | 0.25 | 0.20 | 48 | 1 | 0 | True |

## Artifacts

- Thresholds: `outputs/maskfix_oof/h200_stage1_thresholds_with_cls_gate.json`
- min_area + z search: `outputs/maskfix_oof/h200_stage1_postprocess_minarea_z_search.json`
- component search: `outputs/maskfix_oof/h200_stage1_component_parallel_search.json`
- final config eval: `outputs/maskfix_oof/h200_stage1_eval_config_component_postprocess.json`
- Managed configs:
  - `configs/h200_maskfix_stage1_strategy_e.yaml`
  - `configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold0.yaml`
  - `configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold1.yaml`
  - `configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold2.yaml`
  - `configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold3.yaml`
  - `configs/h200_maskfix_stage1_strategy_e_folds/h200_maskfix_stage1_strategy_e_fold4.yaml`
