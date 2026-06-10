# RhythmMamba 原论文与 UBFC-rPPG 复现实验参数对比

本文档整理 `/root/RhythmMamba-main/RhythmMamba.pdf` 原论文中与 UBFC-rPPG 相关的实验设置，并对比我们当前在 UBFC 数据集上的第一次、第二次复现实验参数和结果。

## 总体结论

我们这次 UBFC-rPPG 复现的核心实验协议和论文的 UBFC intra-dataset 设置基本一致，尤其是 `30 subjects train / 12 subjects test`、160 帧切段、裁脸 resize、最后 epoch 测试、FFT/频谱 HR 评估这一条线是对上的。

差异主要在工程环境、具体开源配置超参、预处理实现绕过方案，以及我们额外做了 DataLoader / CPU-GPU 协同优化。

## 主要相同点

| 项目 | 论文 | 我们的 UBFC 复现 |
|---|---|---|
| 数据集 | UBFC-rPPG，42 个视频 / 42 subjects | UBFC-rPPG |
| intra 协议 | 前 30 个样本训练，后 12 个样本测试 | `BEGIN: 0.0 END: 0.72` 训练，`0.72-1.0` 测试，实际 30 / 12 subjects |
| 验证集 | UBFC intra 无 validation | 配置里 VALID 指向 test split，但 `USE_LAST_EPOCH: True`，不做 best-val 选择 |
| checkpoint | 最后一个 epoch | 最后一个 epoch，Epoch29 |
| 切段长度 | 160 frames | `CHUNK_LENGTH: 160` |
| 输入尺寸 | 论文效率实验写 128x128；预处理写 crop + resize | `RESIZE H/W: 128` |
| 增强 | random upsampling、downsampling、horizontal flipping | `AUG: 1`，trainer 中做时间重采样 / 翻转 |
| 后处理 | Butterworth 0.75-2.5 Hz + Welch / PSD HR | `FFT` 评估，代码中 bandpass 0.75-2.5 Hz + Welch |
| loss | temporal negative Pearson + frequency CE | `Hybrid_Loss = 0.2 * Pearson loss + 1.0 * Frequency loss` |

相关代码位置：

| 文件 | 作用 |
|---|---|
| `configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT_RUN2.yaml` | 二次复现 UBFC 配置 |
| `dataset/data_loader/UBFCrPPGLoader.py` | UBFC-rPPG 数据读取和 split 逻辑 |
| `neural_methods/trainer/RhythmMambaTrainer.py` | RhythmMamba 训练、优化器、增强和测试流程 |
| `neural_methods/loss/TorchLossComputer.py` | Hybrid loss |
| `evaluation/post_process.py` | FFT / Welch / bandpass 后处理 |

## 主要不同点

| 项目 | 论文 | 我们 |
|---|---|---|
| 硬件 | NVIDIA RTX 3090 | RTX 5070 12GB，WSL2 |
| 训练超参公开程度 | PDF 正文没有明确写 batch size、epoch、lr | 按开源配置跑：batch 8、epoch 30、lr 3e-4、AdamW、OneCycleLR |
| 预处理实现 | 论文只描述通用 rPPG-toolbox 流程 | 我们用串行预处理 + ffmpeg CLI 读取 UBFC，绕过 OpenCV VideoCapture 问题 |
| DataLoader | 论文未描述 | 二次复现加入 `pin_memory`、`persistent_workers`、`prefetch_factor`、`non_blocking` |
| 评价范围 | 论文同时做 intra 和 cross-dataset | 我们当前只复现 UBFC intra，没有跑 UBFC cross-dataset |
| 表格指标 | 论文 Table 1 对 UBFC intra 主要报 MAE、RMSE、Pearson | 我们额外输出 MAPE、SNR 和标准误 |

## UBFC Split 细节

论文写的是 UBFC intra-dataset 采用前 30 个样本训练、后 12 个样本测试。

我们的配置为：

```yaml
TRAIN:
  DATA:
    BEGIN: 0.0
    END: 0.72
TEST:
  DATA:
    BEGIN: 0.72
    END: 1.0
```

UBFC loader 中 `split_raw_data()` 会按照排序后的 `subject*` 目录列表进行百分比切分：

```python
file_num = len(data_dirs)
choose_range = range(int(begin * file_num), int(end * file_num))
```

由于 UBFC 原始 subject 编号本身有缺号，CSV 中看到的 subject 编号不是简单的 `1-30` 和 `31-42`。但实际唯一 subject 数仍然是：

| split | clip 数 | unique subject 数 |
|---|---:|---:|
| train | 342 clips | 30 subjects |
| test | 141 clips | 12 subjects |

因此从协议层面看，我们与论文 UBFC intra 设置是对齐的。

## 指标对比

| 指标 | 论文 UBFC intra Ours | 第一次复现 | 第二次复现 Run2 |
|---|---:|---:|---:|
| MAE | 0.50 bpm | 0.54 +/- 0.17 bpm | 0.540054 +/- 0.168391 bpm |
| RMSE | 0.75 bpm | 0.79 +/- 0.36 bpm | 0.794938 +/- 0.361862 bpm |
| Pearson | 0.99 | 0.997 +/- 0.023 | 0.997358 +/- 0.022973 |
| MAPE | 论文 Table 1 未列 | 0.56% +/- 0.16% | 0.555209% +/- 0.164368% |
| SNR | 论文 Table 1 未列 | 8.32 +/- 1.45 dB | 8.524459 +/- 1.437678 dB |

## 解释与判断

我们现在的结果已经非常接近论文 UBFC intra 的复现结果。MAE / RMSE 比论文略高一点点，但 Pearson 非常稳定，说明预测心率与真实心率的整体趋势高度一致。

二次复现的 DataLoader 优化主要改善吞吐和训练稳定性，符合预期，不太会显著改变最终 HR 指标。如果下一步目标是继续提升 MAE / RMSE / Pearson，优先方向应该转向增强策略、loss 权重、后处理窗口、模型正则或 checkpoint 选择策略，而不是继续只调 DataLoader。

