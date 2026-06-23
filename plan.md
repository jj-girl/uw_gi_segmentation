# UW GI 医学图像分割项目升级计划

## 总体目标

将当前的 `2D UNet++ EfficientNet-B3` 基线，升级为一个更完整、更现代、更适合课程设计展示的医学图像分割系统：

**2.5D 多切片输入 + 分类-分割联合学习 + 正样本/均衡采样训练 + ROI 裁剪预处理 + 强 backbone 对比 + 损失函数与训练策略优化 + 后处理 + 消融实验。**

最终项目需要同时服务于：

- 模型训练与性能提升
- 可视化演示
- 课程报告
- 答辩 PPT
- 错误分析与创新点说明

## 当前基础版本

- 数据集：UW-Madison GI Tract Image Segmentation
- 任务：腹部 MRI 多类别器官分割
- 分割类别：
  - large bowel
  - small bowel
  - stomach
- 当前主模型：`smp_unetplusplus_efficientnet-b3`
- 当前已支持输入：
  - 2D 单切片：`(1, H, W)`
  - 2.5D 三切片：`(3, H, W)`
  - 2.5D 五切片：`(5, H, W)`
- 输出：3 通道二值 mask：`(3, H, W)`
- 当前正式数据：85 个 case，38496 张 MRI slice
- 当前用途：全量 fold 训练与正式模型评估

## 当前已识别的问题

1. 纯 2D 模型无法利用相邻切片的三维上下文。
2. 空切片较多，单纯分割模型容易被 negative slice 主导。
3. 分类与分割任务尚未在正式实验中充分联合验证。
4. ROI/crop 预处理仍是轻量版本，还没有做 detector crop。
5. 当前使用的 EfficientNet-B3 只是中等规模 backbone，正式实验可尝试更强结构。
6. 训练策略仍需系统化：scheduler、EMA/SWA、TTA、阈值搜索、后处理、ensemble 都需要逐步完善。
7. 正式实验必须使用全量数据和真实验证分布。

## 指标参考

Kaggle UW GI 排行榜参考：

- 第 1 名约 `0.8859`
- 前 20 名约 `0.878-0.886`

课程设计中更现实的目标：

- `0.70-0.78`：流程跑通，但模型较弱。
- `0.78-0.82`：普通 baseline。
- `0.82-0.85`：较好的单模型结果。
- `0.85-0.87`：较强课程项目，有明显优化和消融实验支撑。
- `0.87+`：接近 Kaggle 强方案，通常需要 2.5D/3D、多模型、TTA、后处理、fold ensemble。

## 阶段 1：可靠基线

- [ ] 保留 `2D UNet++ EfficientNet-B3` 作为对照 baseline。
- [ ] 记录配置、数据划分、训练 loss、验证 Dice、预测可视化。
- [ ] 建立实验记录表模板。
- [ ] 确保后续每个优化都有可比较的基线。

交付物：

- baseline 指标
- baseline 可视化
- 报告/PPT 中的基线模型说明

## 阶段 2：2.5D 多切片输入

- [x] 在数据集中加入 `slice_window` 参数。
- [x] 支持 `slice_window=3` 和 `slice_window=5`。
- [x] 将模型输入从 `1 channel` 扩展到 `3/5 channels`。
- [x] 使用中心 slice 的 mask 作为训练标签。
- [x] 边界 slice 使用最近切片复制补齐。
- [x] 新增 H200 正式训练配置。

实验设计：

- 2D vs 3-slice 2.5D
- 2D vs 5-slice 2.5D

预期价值：

- 引入 MRI 序列上下文。
- 作为第一个核心创新点。

## 阶段 3：分类-分割联合学习

- [x] 为 segmentation model 增加可选 classification head。
- [x] 预测 3 个器官级存在性标签：
  - large bowel exists
  - small bowel exists
  - stomach exists
- [x] 分类标签定义为：对应器官 mask 是否存在正像素。
- [x] 联合损失：
  - segmentation loss
  - classification BCE loss
- [x] 在 H200 正式训练配置中启用分类头。
- [x] 新增分类 gate 后处理接口。

实验设计：

- 仅分割 vs 分割+分类头。
- 统一 mask threshold vs 分类 gate + mask threshold。

预期价值：

- 减少空切片误检。
- 对齐 Kaggle top solution 中常见的两阶段/分类分割策略。

## 阶段 4：正样本与均衡采样训练

- [x] 增加训练采样模式：
  - `all_slices`
  - `positive_slices_only`
  - `balanced_positive_negative`
- [x] 增加 positive/negative balanced sampler。
- [x] 增加可选分层抽样限制，保留为调试能力但正式训练不启用。
- [ ] 分类分支正式训练时使用 all slices。
- [ ] 分割精修模型主要使用 positive slices。
- [ ] 比较不同采样策略对验证 Dice 的影响。

实验设计：

- all slices
- balanced positive/negative sampling
- positive-only segmentation refinement

预期价值：

- 避免空切片主导分割学习。
- 增强模型对器官边界和小器官区域的学习。

## 阶段 5：ROI / Crop 预处理

先实现轻量 crop，再考虑 detector crop。

- [x] 增加 center crop。
- [x] 增加 foreground/body-region threshold crop。
- [ ] 保存 crop metadata，支持将预测结果映射回原图。
- [ ] 比较 full image vs cropped image。

后续可选：

- [ ] 增加 detector-based crop，例如 YOLO 风格 ROI 检测。

预期价值：

- 减少无关背景。
- 缓解手臂、高亮伪影对归一化和分割的影响。
- 提升模型对腹部主体区域的关注。

## 阶段 6：更强模型架构

通过 SMP/timm 保持模型工厂可扩展。

候选模型：

- [ ] `UNet++ EfficientNet-B3`
- [ ] `UNet++ EfficientNet-B5`
- [ ] `DeepLabV3+ ResNet50`
- [ ] `DeepLabV3+ EfficientNet-B4`
- [ ] `UperNet ConvNeXt-Tiny/Base`
- [ ] 可选：`SegFormer` 或 `Swin` 系模型

实验设计：

- 同数据、同 fold、同 image size 下比较不同架构。
- 同架构下比较不同输入分辨率。

预期价值：

- 提升特征表达能力。
- 给报告/PPT 提供结构对比和模型选择依据。

## 阶段 7：损失函数与训练策略优化

- [x] 实现损失函数：
  - Dice + BCE
  - Dice + Focal
  - Tversky
  - Focal Tversky
- [ ] 实现 Dice + Lovasz。
- [x] 增加 cosine learning-rate scheduler。
- [x] 增加 gradient accumulation。
- [x] 增加 EMA。
- [ ] 增加 SWA。
- [x] 保留 AMP 支持，用于 GPU 训练。

实验设计：

- 不同 loss 对比。
- scheduler on/off 对比。
- EMA/SWA on/off 对比。

预期价值：

- 提升收敛稳定性。
- 增强消融实验说服力。

## 阶段 8：阈值搜索与后处理

- [x] 支持每个器官独立 threshold。
- [x] 增加小连通域过滤。
- [x] 增加最小面积过滤。
- [x] 增加 z-axis 连续性修正 helper：
  - 去除孤立 positive slice
  - 保留连续阳性片段

实验设计：

- raw prediction vs threshold search
- threshold search vs threshold + connected-component filtering
- 加入/不加入 z-axis 连续性修正

预期价值：

- 减少碎片 mask。
- 降低空切片误检。
- 让可视化结果更平滑、更符合医学图像序列连续性。

## 阶段 9：TTA 与 Ensemble

- [x] 增加 horizontal flip TTA。
- [ ] 增加 scale TTA。
- [ ] 增加模型 ensemble：
  - 2.5D UNet++ B3
  - DeepLabV3+
  - positive-slice refinement model
- [ ] 增加 fold ensemble：
  - 先做 3-fold
  - 如果 GPU/时间允许再扩展到 5-fold

预期价值：

- 提升最终稳定性和分数。
- 作为 competition-style final solution。

## 阶段 10：报告和 PPT 重构

报告建议结构：

1. 研究背景与任务定义
2. 数据集与评价指标
3. 基线模型
4. 2.5D 多切片建模
5. 分类-分割联合学习
6. 正样本/均衡采样策略
7. ROI 裁剪预处理
8. 损失函数与训练策略
9. 后处理方法
10. 实验结果与消融分析
11. 可视化结果与错误案例分析
12. 总结与未来工作

PPT 建议结构：

1. 项目背景
2. 数据集与任务
3. 原始 baseline 局限
4. 现代医学分割框架设计
5. 模型架构
6. 训练策略
7. 实验与消融结果
8. 预测可视化
9. 错误分析
10. 应用前景与总结

## 推荐执行顺序

1. 确认可靠 baseline。
2. 完成 2.5D 数据集与配置。
3. 完成分类头与联合 loss。
4. 完成 positive-only / balanced sampling。
5. 完成 loss 和 scheduler。
6. 完成阈值搜索与后处理。
7. 完成 ROI/crop 对比。
8. 完成更强架构对比。
9. 完成 TTA 和 ensemble。
10. 重构报告和 PPT。

## 第一阶段里程碑

构建并比较：

- `2D UNet++ EfficientNet-B3`
- `2.5D 3-slice UNet++ EfficientNet-B3`
- `2.5D 5-slice UNet++ EfficientNet-B3`
- `2.5D 5-slice UNet++ EfficientNet-B3 + classification head`

最低交付物：

- 训练配置
- 验证 Dice 表格
- 预测可视化图片
- 简短错误分析
