# Stage 1 Auto Pipeline

目标：把 Stage 1 的 OOF 评估、后处理搜索、配置同步和报告生成收敛到一个入口，减少手工散跑。

## 入口

```bash
/mnt/disk2/hjj/uwgiseg/bin/python scripts/stage1_auto_pipeline.py
```

默认行为：

- 检查 5 个 fold 的 `best.pt` 是否存在。
- 已存在的搜索结果会直接复用。
- 推荐后处理参数会同步到主配置和 fold 配置；参数未变化时不重写文件。
- 生成固定报告：`outputs/oof/stage1_auto_report.md`。
- 不自动追加 `工作日志.md`，需要记录时加 `--append-worklog`。

## 复跑

```bash
/mnt/disk2/hjj/uwgiseg/bin/python scripts/stage1_auto_pipeline.py --force
```

常用参数：

- `--gpus 0,1`：component 搜索使用的 GPU 列表。
- `--max-workers 5`：并行任务数。
- `--checkpoint-name best.pt`：选择 fold checkpoint。
- `--append-worklog`：把本次最终指标追加到 `工作日志.md`。

## 主要产物

- `outputs/oof/h200_stage1_thresholds_with_cls_gate.json`
- `outputs/oof/h200_stage1_postprocess_minarea_z_search.json`
- `outputs/oof/h200_stage1_component_parallel_search.json`
- `outputs/oof/h200_stage1_eval_config_component_postprocess.json`
- `outputs/oof/stage1_auto_report.md`

当前原则：报告文件保存完整状态，工作日志只记录阶段性结论，避免重复粘贴大段指标。

## 参考路线

- nnU-Net：自动配置、训练、后处理和 ensemble 的完整分割 pipeline。
  https://github.com/MIC-DKFZ/nnUNet
- nnU-Net postprocessing：基于验证集判断 connected-component 后处理是否保留。
  https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/postprocessing/remove_connected_components.py
- MONAI Auto3DSeg：先分析数据，再生成算法、训练 checkpoint、排名并 ensemble。
  https://github.com/Project-MONAI/tutorials/blob/main/auto3dseg/README.md
