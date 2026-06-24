# Current System State

Last updated: 2026-06-24

## Kept Model Lines

1. `maskfix Strategy E`
   - Main validated 5-fold model.
   - Configs: `configs/h200_maskfix_stage1_strategy_e.yaml`, `configs/h200_maskfix_stage1_strategy_e_folds/`
   - Outputs: `outputs/h200_maskfix_stage1_strategy_e_fold0` through `outputs/h200_maskfix_stage1_strategy_e_fold4`
   - OOF/report artifacts: `outputs/maskfix_oof/`

2. `UNet++ EfficientNet-B5`
   - Accepted as second model line after fold0 small-bowel proxy comparison.
   - Configs: `configs/h200_next_unetpp_b5_folds/`
   - Outputs: `outputs/h200_next_unetpp_b5_fold0` through `outputs/h200_next_unetpp_b5_fold4`
   - Monitor: `scripts/monitor_unetpp_b5_5fold.py`
   - Monitor logs/status: `outputs/h200_next_unetpp_b5_5fold/`

## Fold0 Small Bowel Decision

| Model | Combined proxy | Dice 3D | HD95 mm |
| --- | ---: | ---: | ---: |
| `UNet++ B5` | `0.881990` | `0.738322` | `20.417` |
| `small_bowel_aware` | `0.877970` | `0.732717` | `23.140` |
| `Strategy E` fold0 | `0.877919` | `0.731494` | `22.469` |

Decision: keep and expand `UNet++ B5`; remove `DeepLabV3+` and `small_bowel_aware` trial outputs/configs.

## Cleanup Policy

Removed experiment families include:

- pre-maskfix and old stage1/stage2/stage3 experiments
- `DeepLabV3+ ResNet50` candidate
- `small_bowel_aware` candidate
- baseline OOF and interrupted baseline monitor outputs
- `nnunet_route` experiments
- transient `next_steps_auto` outputs
- old debug/cache directories

Raw data, source code, main Strategy E artifacts, and active B5 artifacts are kept.
