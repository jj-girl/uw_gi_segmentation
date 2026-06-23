# Training Strategy Experiments

> Invalidated on 2026-06-23 for numeric conclusions: experiments listed here
> were run before the RLE decode and image-dimension fix. The strategy taxonomy
> is still useful, but the reported comparisons must be recomputed from
> `h200_maskfix_*` runs.

目标：在不覆盖当前 Stage 1 baseline 的前提下，逐轴测试训练策略是否更适合 UWGI 数据。

## 当前实验轴

| Config | 变量 | 目的 |
| --- | --- | --- |
| `configs/h200_stage1_strategy_a_strong_aug.yaml` | all slices + 强增强 | 测试形变、强度扰动、z context dropout 是否提升泛化 |
| `configs/h200_stage1_strategy_b_organ_balanced.yaml` | organ-balanced sampler + 强增强 | 降低空切片/器官不均衡带来的训练偏置 |
| `configs/h200_stage1_strategy_c_organ_balanced_focal_tversky.yaml` | B + focal tversky | 测试不对称 loss 对漏检和小器官的影响 |
| `configs/h200_stage1_strategy_e_postprocess_aware.yaml` | baseline 训练 + postprocess-aware checkpoint | 让训练期保存指标更接近最终后处理指标 |
| `configs/h200_stage1_strategy_f_volume_norm.yaml` | baseline 训练 + volume percentile normalization | 测试 case/day volume 级强度归一化是否改善 2.5D 上下文 |

## 评估规则

- 单 fold 只能看趋势，不作为最终结论。
- 稳定候选必须跑 5-fold OOF，并通过 `scripts/stage1_auto_pipeline.py` 的同一套报告比较。
- 不用 positive-only 作为主路线；它会削弱空切片判别。
- A/B/C 单 fold 已低于 baseline，不扩 5-fold；下一步优先看 E 的 `best_postprocess.pt`。
- F 只改预处理 normalization，不改 sampler/loss/增强；可和 E 在另一张 GPU 上并行跑。

## 启动 fold0 对照

```bash
CHECK_SECONDS=300 bash scripts/run_training_strategy_fold0_watchdog.sh
```

watchdog 会先并行启动 A/B 的 fold0，完成后再启动 C 的 fold0。状态日志：
`outputs/training_strategy_fold0_watchdog.log`。

## 参考

- Albumentations segmentation 示例强调 image/mask 必须同步几何增强。
  https://albumentations.ai/docs/3-basic-usage/semantic-segmentation/
- SMP 是竞赛分割中常用的模型库。
  https://github.com/qubvel-org/segmentation_models.pytorch
