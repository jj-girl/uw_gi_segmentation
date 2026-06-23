# H200 服务器迁移与训练指南

## 迁移前注意事项

不要把本地 `.kaggle/access_token` 上传到服务器仓库或共享目录。

建议上传：

- `configs/`
- `scripts/`
- `src/`
- `README.md`
- `plan.md`
- `train_sta.md`
- `requirements.txt`
- `pyproject.toml`
- `h200_guide.md`

不建议上传：

- `.venv/`
- `.kaggle/`
- `outputs/`
- 大量临时缓存

数据可以上传 Kaggle zip，也可以在服务器上重新下载。

## 环境建议

推荐环境：

```bash
conda create -n uwgi python=3.11 -y
conda activate uwgi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install segmentation-models-pytorch timm albumentations kaggle
```

如果服务器已有 CUDA/PyTorch 统一环境，优先使用服务器管理员提供的 PyTorch 版本。

## 数据准备

如果服务器上已有 Kaggle zip：

```bash
python scripts/prepare_data.py \
  --zip /path/to/uw-madison-gi-tract-image-segmentation.zip \
  --out data/raw/uw-madison-gi-tract-image-segmentation
```

如果服务器能直接使用 Kaggle API，则在服务器单独配置 token 后运行：

```bash
python scripts/prepare_data.py \
  --download \
  --out data/raw/uw-madison-gi-tract-image-segmentation
```

检查数据：

```bash
python scripts/inspect_data.py --root data/raw/uw-madison-gi-tract-image-segmentation
```

## 第一阶段：主模型训练

目标：训练可靠的 2.5D 5-slice 分类-分割联合模型。

```bash
python -m src.uwgi.train \
  --config configs/h200_stage1_2p5d_unetpp_b3_all.yaml
```

输出：

- `outputs/h200_stage1_2p5d_unetpp_b3_all_fold0/best.pt`
- `outputs/h200_stage1_2p5d_unetpp_b3_all_fold0/last.pt`
- `outputs/h200_stage1_2p5d_unetpp_b3_all_fold0/metrics.csv`

中断恢复：

```bash
python -m src.uwgi.train \
  --config configs/h200_stage1_2p5d_unetpp_b3_all.yaml \
  --resume outputs/h200_stage1_2p5d_unetpp_b3_all_fold0/last.pt
```

## 第二阶段：均衡采样微调

目标：从第一阶段 best checkpoint 初始化，使用 balanced sampling 缓解空切片主导问题。

```bash
python -m src.uwgi.train \
  --config configs/h200_stage2_2p5d_unetpp_b3_balanced.yaml
```

注意：

- `init_from_checkpoint` 表示只加载模型权重。
- `resume_from_checkpoint` 表示完整恢复训练状态。
- 微调阶段不要错误使用 resume，否则学习率和 epoch 会沿用旧训练状态。

## 第三阶段：正样本精修

目标：使用 positive-only slices 训练分割精修模型，提高边界质量。

```bash
python -m src.uwgi.train \
  --config configs/h200_stage3_2p5d_unetpp_b5_positive_refine.yaml
```

## 评估与阈值搜索

固定配置评估：

```bash
python scripts/evaluate_checkpoint.py \
  --config configs/h200_stage2_2p5d_unetpp_b3_balanced.yaml \
  --checkpoint outputs/h200_stage2_2p5d_unetpp_b3_balanced_fold0/best.pt
```

阈值搜索：

```bash
python scripts/threshold_search.py \
  --config configs/h200_stage2_2p5d_unetpp_b3_balanced.yaml \
  --checkpoint outputs/h200_stage2_2p5d_unetpp_b3_balanced_fold0/best.pt \
  --out outputs/h200_stage2_2p5d_unetpp_b3_balanced_fold0/thresholds.json
```

## 推荐训练顺序

1. 先跑 fold0 的 stage1。
2. 查看 `metrics.csv`，确认没有明显过拟合或欠拟合。
3. 用 stage1 best checkpoint 跑 stage2。
4. 跑 threshold search。
5. 生成可视化。
6. 如果 fold0 有效，再扩展到 3-fold 或 5-fold。
7. 最后考虑 stage3 refinement 和模型融合。

## 资源建议

H200 显存较大，可以尝试：

- `image_size=384`
- batch size `16`
- EfficientNet-B3/B5
- AMP 开启
- EMA 开启

如果显存非常充裕，可以进一步尝试：

- `image_size=416`
- `UNet++ EfficientNet-B5`
- `DeepLabV3+ EfficientNet-B4`
