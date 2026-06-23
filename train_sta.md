# 推荐训练策略

> 2026-06-23 注意：本文档中的历史实验结论和性能数字来自 mask 坐标修复前的训练，
> 不再作为有效结果使用。训练策略框架可参考，最终指标必须以 `h200_maskfix_*`
> 重训和 `outputs/maskfix_oof` 重新评估为准。

## 核心原则

本项目的训练策略以正式全量实验为准。模型评估必须使用：

- 全量训练数据
- 按 case 分组的 fold 划分
- 真实分布的验证集
- 稳定的消融实验

训练目标是在避免过拟合和欠拟合的前提下，逐步提升模型性能。

## 数据划分策略

### 必须按 case 分 fold

UW GI 数据是连续 MRI slice。如果随机按 slice 划分，训练集和验证集会出现同一个 case 的相邻切片，导致数据泄漏。

推荐：

- 使用 `GroupKFold`
- group 字段为 `case`
- 默认 5-fold

### 验证集保持真实分布

验证集不要做：

- positive/negative 平衡
- positive-only 过滤
- 人为采样

否则验证 Dice 会失真，无法代表真实推理表现。

### 正式训练数据约束

- 不设置 `limit_train_samples`
- 不设置 `limit_valid_samples`
- 使用全量 fold 数据

## 四阶段训练路线

## 阶段 1：可靠主模型训练

目的：建立一个可信的 2.5D 主模型，不追求极限。

推荐配置：

- 数据：全量 train
- 划分：5-fold group split by case
- 输入：2.5D 5-slice
- 模型：`UNet++ EfficientNet-B3`
- image size：`320` 或 `384`
- sampling：`all_slices`
- loss：`Dice + BCE + classification BCE`
- optimizer：AdamW
- lr：`2e-4`
- weight decay：`1e-5`
- scheduler：cosine
- epoch：`15-20`
- augmentation：
  - horizontal flip
  - shift/scale/rotate
  - brightness/contrast
  - 先不要使用过强畸变增强
- checkpoint：保存 best valid Dice

预期：

- 建立稳定 baseline。
- 判断 2.5D 和分类头是否有效。

## 阶段 2：均衡采样微调

目的：缓解空切片过多导致模型倾向预测空 mask 的问题。

推荐配置：

- 数据：全量 train
- 输入：2.5D 5-slice
- 模型：同阶段 1
- 初始化：加载阶段 1 的 best checkpoint
- sampling：`balanced_positive_negative`
- loss：`Dice + Focal + classification BCE`
- lr：`1e-4`
- epoch：`10-15`
- validation：保持真实分布

注意：

- balanced sampling 不是删除 negative slice。
- 它只是提高 positive slice 在训练 batch 中出现的频率。
- 验证集不能 balanced。

预期：

- 减少模型过度保守。
- 提高 positive slice 上的器官召回。

## 阶段 3：正样本分割精修

目的：让模型更专注于器官边界和小器官区域。

推荐配置：

- 数据：positive slices only
- 输入：2.5D 5-slice
- 模型：
  - `UNet++ EfficientNet-B3`
  - 或 `UNet++ EfficientNet-B5`
  - 或 `DeepLabV3+ ResNet50`
- sampling：`positive_slices_only`
- loss：
  - `Dice + Focal`
  - 或 `Focal Tversky`
- classification head：
  - 可以关闭
  - 或保留但降低 `cls_weight`
- lr：`1e-4`
- epoch：`10-20`
- validation：仍然使用真实分布

预期：

- 提升边界质量。
- 改善 small bowel 等较难类别。

## 阶段 4：推理融合与后处理

目的：提升最终预测稳定性，减少误检和碎片 mask。

推荐方法：

- 使用 classification head 做 organ-level gate。
- 每个器官单独搜索 mask threshold。
- 小连通域过滤。
- 最小面积过滤。
- z-axis 连续性修正。
- horizontal flip TTA。
- 资源允许时加入 scale TTA。

可选 ensemble：

- 阶段 1 主模型
- 阶段 2 balanced fine-tune 模型
- 阶段 3 positive-only refinement 模型

预期：

- 提升验证 Dice。
- 预测结果更平滑、更符合 MRI 序列连续性。

## 避免过拟合

建议：

- 按 case 分 fold。
- validation 保持真实分布。
- 使用 weight decay。
- 使用适度数据增强。
- 使用 early stopping。
- 使用 EMA 或 SWA。
- 不用 validation 数据调训练采样。
- 不用小样本子集 Dice 判断最终性能。

可采用 early stopping：

- 如果连续 5 个 epoch valid Dice 不提升，则停止训练。

过拟合迹象：

- train Dice 持续上升，valid Dice 停滞或下降。
- 可视化中模型只记住部分器官形态。
- 对某些 case 表现极好，对其他 case 明显失败。

## 避免欠拟合

建议：

- 使用 2.5D，而不是纯 2D。
- 正式 image size 不低于 `320`，推荐 `384`。
- 使用 ImageNet pretrained encoder。
- backbone 从 EfficientNet-B3 起步。
- epoch 不少于 `15-20`。
- 如果显存允许，尝试 EfficientNet-B5 或 ConvNeXt。
- 使用 balanced fine-tuning 或 positive-only refinement。

欠拟合迹象：

- train Dice 和 valid Dice 都很低。
- loss 下降缓慢。
- 预测 mask 长期接近全空。
- 分类 head 无法区分 positive/negative slice。

## 推荐实验矩阵

| 实验 | 输入 | 模型 | 采样 | Loss | 目的 |
|---|---|---|---|---|---|
| E1 | 2D | UNet++ B3 | all | Dice+BCE | 基线 |
| E2 | 2.5D-5ch | UNet++ B3 | all | Dice+BCE+Cls | 主模型 |
| E3 | 2.5D-5ch | UNet++ B3 | balanced | Dice+Focal+Cls | 抗空切片 |
| E4 | 2.5D-5ch | UNet++ B3/B5 | positive-only | Focal Tversky | 分割精修 |
| E5 | E2+E3/E4 | ensemble/TTA/postprocess | - | - | 最终方案 |

## 建议性能目标

在合理 GPU 条件下：

- 单 fold 主模型：`0.80-0.84`
- balanced/fine-tune 后：`0.83-0.86`
- TTA + threshold + 后处理：`0.85+`
- 多 fold ensemble：向 `0.86-0.88` 靠近

## 最推荐执行顺序

1. 检查 full-data metadata 和 fold 分布。
2. 跑 E2：2.5D-5ch + classification head + all slices。
3. 从 E2 checkpoint 继续跑 E3：balanced sampling fine-tune。
4. 加 threshold search 与 postprocess。
5. 跑 E4：positive-only refinement。
6. 比较 B3、B5、DeepLabV3+。
7. 资源允许时做 TTA、3-fold、5-fold ensemble。

## 最终报告中的训练策略表述

本项目采用分阶段训练策略：首先使用全量切片训练 2.5D 分类-分割联合模型，建立稳定主模型；随后通过正负样本均衡采样缓解空切片主导问题；进一步使用正样本切片训练分割精修模型，以提升器官边界质量；最后结合分类 gate、器官级阈值搜索、连通域过滤、z 轴连续性修正和 TTA 提升推理稳定性。
