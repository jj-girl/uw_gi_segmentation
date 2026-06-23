# 2.5D + 3D 双分支融合方案调研与设计

## 是否有必要做

结论：

**不是第一阶段必须做，但非常适合作为高阶创新点和冲分模块。**

原因：

1. UW GI 是连续 MRI slice 任务，器官在 z 轴上具有连续结构。
2. 2.5D 模型可以利用局部相邻切片，但本质上仍然是 2D 分割。
3. 3D 模型可以显式建模体数据连续性，有助于减少孤立误检和切片间不连续。
4. Kaggle 1st place solution 明确使用了 `2.5D/2D segmentation group + 3D group`，并进行 logits 融合。
5. Kaggle 1st place 讨论中提到，将 2.5D 模型与公开 3D UNet 模型融合可带来约 `+0.003` 的提升。

因此：

- 如果目标是课程设计完整性：2.5D + 分类头 + 后处理已经足够。
- 如果目标是提升项目档次和创新性：建议加入 3D 分支作为高级模块。
- 如果目标是接近 Kaggle top 方案：2.5D + 3D 融合非常值得做。

参考：

- Kaggle 1st place solution: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/337197
- Kaggle 2nd place solution: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/337400
- Kaggle 3rd place solution: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/337468

## 2.5D 和 3D 的分工

### 2.5D 分支

职责：

- 负责高分辨率细节。
- 使用 ImageNet 预训练 encoder。
- 对器官边界、小结构、纹理细节更敏感。

推荐模型：

- `UNet++ EfficientNet-B3/B5`
- `DeepLabV3+ ResNet50/EfficientNet-B4`
- `UperNet ConvNeXt`

输入：

- `(5, H, W)`，即 `s-2, s-1, s, s+1, s+2`

输出：

- 当前 slice 的 3 类 mask logits。

### 3D 分支

职责：

- 建模 z 轴连续性。
- 抑制孤立 slice 误检。
- 增强体数据结构一致性。

推荐模型：

- MONAI `SegResNet`
- MONAI `DynUNet`
- 轻量 3D UNet

输入：

- 一个 case/day 的 volume patch，例如 `(C=1, D=32, H=256, W=256)`

输出：

- 体数据 mask logits，例如 `(3, D, H, W)`

## 具体实现路线

## 阶段 A：先不训练 3D，只做 2.5D 强基线

先完成：

- 2.5D 5-slice
- classification head
- balanced sampling
- threshold search
- postprocess

原因：

- 2.5D 是主要性能来源之一。
- 训练更稳定。
- 实现成本低。
- 适合先产出可展示结果。

## 阶段 B：构建 3D 数据集

需要新增：

- `UWGI3DVolumeDataset`
- 以 `case + day` 为单位组装 volume
- 将每个 slice 的 mask 堆叠成 3D mask
- 支持 depth crop / sliding window
- 支持 resize 到统一 `H, W`

数据形式：

```text
image: (1, D, H, W)
mask:  (3, D, H, W)
```

建议初始参数：

- depth：`32`
- image size：`256`
- stride：`16`
- batch size：`1-2`

## 阶段 C：训练 3D SegResNet

推荐配置：

- 模型：MONAI `SegResNet`
- loss：Dice + CE 或 Dice + Focal
- optimizer：AdamW
- lr：`1e-4`
- epoch：`100+`，视数据量和时间调整
- AMP：开启
- patch-based training：开启

注意：

- 3D 模型训练明显更慢。
- 需要 careful validation。
- 不建议一开始就做 3D，否则会拖慢主线开发。

## 阶段 D：2.5D + 3D logits 融合

推理时：

1. 2.5D 模型对每个 slice 输出 logits。
2. 3D 模型对整个 volume 或 sliding window 输出 logits。
3. 将 3D logits resize/align 到 2.5D 输出尺寸。
4. 做加权融合：

```python
final_logits = 0.6 * logits_2p5d + 0.4 * logits_3d
```

权重可搜索：

- `0.7 / 0.3`
- `0.6 / 0.4`
- `0.5 / 0.5`

然后再做：

- sigmoid
- classification gate
- threshold search
- connected component filtering
- z-axis continuity postprocess

## 预期性能收益

合理预期：

- 如果 2.5D 单模型已经较强，3D 融合可能带来 `+0.002` 到 `+0.006` 的 Dice 提升。
- 如果 2.5D 模型较弱，3D 分支不一定能明显提升，甚至可能因为训练不充分拖累融合。
- 3D 分支最主要的收益是提升序列连续性和减少孤立误检。

## 风险

1. 实现复杂度明显增加。
2. 3D 数据组织和 sliding window 推理容易出 bug。
3. 3D 模型训练成本高。
4. 如果时间不足，3D 分支可能无法训练到有效水平。
5. 报告中如果没有充分实验支撑，3D 会显得像堆模块。

## 是否推荐本项目做

推荐程度：

- 课程项目最低可交付：不必做 3D。
- 想体现较强创新性：建议做轻量 3D 分支。
- 想冲接近 Kaggle 高分：建议做 2.5D + 3D logits 融合。

最稳妥方案：

1. 先完成 2.5D 强基线。
2. 用 H200 训练 2.5D stage1/stage2。
3. 如果 2.5D 验证 Dice 达到 `0.83+`，再开启 3D 分支。
4. 将 3D 作为高级增强模块，而不是主线依赖。

## 报告中的创新点表述

可以写为：

> 本项目在 2.5D 多切片分割模型的基础上，引入 3D 体数据分支，用于建模 MRI 序列在 z 轴方向上的器官连续性。2.5D 分支侧重于高分辨率边界细节，3D 分支侧重于体结构一致性。最终通过 logits 加权融合，结合器官级分类 gate 与 z-axis 连续性后处理，提高多器官分割结果的稳定性和解剖一致性。

