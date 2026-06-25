# Current System State

Last updated: 2026-06-25

## Final Decision

The project is now centered on a two-family inference ensemble:

1. `UNet++ EfficientNet-B5`
   - Role: primary model.
   - Weight in final blend: `0.70`.
   - Reason: best standalone model by local OOF Dice and official-metric proxy.

2. `maskfix Strategy E`
   - Role: auxiliary diversity model.
   - Weight in final blend: `0.30`.
   - Reason: lower standalone score than B5, but useful non-identical errors
     improve the blended OOF result.

## Final Evidence

### Local OOF Dice

| Model | Mean Dice | Positive Dice | Empty FP Rate |
| --- | ---: | ---: | ---: |
| Strategy E | 0.9188102194 | 0.7634351159 | 0.0170830851 |
| B5 | 0.9210980579 | 0.7687032270 | 0.0156804169 |
| 0.30 Strategy E + 0.70 B5 | 0.9226132123 | 0.7754965725 | 0.0159891921 |

### Official-Metric Proxy

| Model | Combined Proxy | 3D Dice | HD95 mm |
| --- | ---: | ---: | ---: |
| Strategy E | 0.9091636213 | 0.8039622003 | 17.9245 |
| B5 | 0.9115916074 | 0.8086270730 | 17.0627 |

The official proxy is local and approximate. It uses 3D Dice, HD95 in mm, and a
normalized Hausdorff proxy with `0.4/0.6` Dice/Hausdorff weighting.

## Active Paths

### Configs

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

### Final Evaluation Artifacts

```text
outputs/b5_full_pipeline/final_report.md
outputs/b5_full_pipeline/status.json
outputs/h200_next_unetpp_b5_oof/h200_stage1_eval_config_component_postprocess.json
outputs/h200_next_unetpp_b5_oof/h200_next_unetpp_b5_official_oof_proxy.json
outputs/maskfix_oof/maskfix_strategy_e_official_oof_proxy.json
outputs/ensemble_strategy_e_b5/strategy_e_b5_weight_search.json
```

### Final Submission Entrypoint

```text
scripts/make_ensemble_submission.py
```

## Completed Pipeline

`scripts/run_b5_full_pipeline.py` completed all post-training stages:

| Step | Status |
| --- | --- |
| B5 5-fold training/checkpoint verification | complete |
| B5 OOF/postprocess search | complete |
| Strategy E official proxy | complete |
| B5 official proxy | complete |
| Strategy E + B5 weight search | complete |
| Final report generation | complete |

## Cleanup Policy

Removed or inactive experiment families include:

- pre-maskfix experiments
- old stage1/stage2/stage3 variants
- baseline artifacts
- `DeepLabV3+ ResNet50`
- `small_bowel_aware`
- `nnunet_route`
- transient `next_steps_auto` outputs

Raw data, source code, final B5 artifacts, final Strategy E artifacts, OOF
reports, official proxy outputs, and final ensemble search outputs are kept.

## Operating Rule

Use B5 as the model development baseline. Keep Strategy E only as a frozen
ensemble contributor unless a future experiment proves a better auxiliary model.
