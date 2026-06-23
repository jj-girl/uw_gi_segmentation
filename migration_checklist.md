# H200 迁移前工程清单

## 必须迁移

- `src/`
- `scripts/`
- `configs/`
- `README.md`
- `plan.md`
- `train_sta.md`
- `h200_guide.md`
- `fusion_2p5d_3d.md`
- `requirements.txt`
- `pyproject.toml`

## 不要迁移

- `.venv/`
- `.kaggle/`
- `.kaggle/access_token`
- `outputs/`
- `data/raw/` 中的本地临时解压目录
- `__pycache__/`

## 数据迁移建议

如果服务器网络稳定，优先在 H200 上重新下载 Kaggle 数据。

如果服务器网络不稳定，可以只迁移原始 zip：

```text
data/raw/uw-madison-gi-tract-image-segmentation.zip
```

到服务器后再执行：

```bash
python scripts/prepare_data.py \
  --zip data/raw/uw-madison-gi-tract-image-segmentation.zip \
  --out data/raw/uw-madison-gi-tract-image-segmentation
```

## 迁移后第一轮验证

```bash
python -m py_compile \
  src/uwgi/train.py \
  src/uwgi/visualize.py \
  scripts/threshold_search.py \
  scripts/evaluate_checkpoint.py
```

```bash
python scripts/inspect_data.py \
  --root data/raw/uw-madison-gi-tract-image-segmentation
```

## 推荐训练顺序

1. `configs/h200_stage1_2p5d_unetpp_b3_all.yaml`
2. `configs/h200_stage2_2p5d_unetpp_b3_balanced.yaml`
3. `configs/h200_stage3_2p5d_unetpp_b5_positive_refine.yaml`

第一轮不要直接上 3D。先让 2.5D 主模型在全量数据上达到稳定验证 Dice，再决定是否加入 3D 分支融合。

## 3D 分支依赖

如果后续实现 `SegResNet/DynUNet`，服务器环境需要额外安装：

```bash
pip install monai
```

3D 分支建议作为第二阶段创新模块，而不是迁移后的第一件事。
