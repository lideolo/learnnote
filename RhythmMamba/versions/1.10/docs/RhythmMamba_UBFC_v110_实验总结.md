# RhythmMamba UBFC V1.10 实验总结

日期：2026-06-11

## 结论

V1.10 已完成 `ROI-aware Adaptive Spectral-Gated Periodic Mamba`（生理区域感知的自适应频谱门控周期 Mamba）工程实现，并跑完 UBFC intra held-out 完整训练与测试流程。

当前可作为 V1.10 达标结果的是低学习率热启动版本：

- 模型：`UBFC_UBFC_UBFC_RhythmMamba_ROISGPM_WarmLR1e6_v110_Epoch2.pth`
- 配置：`/root/RhythmMamba-main/configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_WARM_LR1E6_V110.yaml`
- 日志：`/root/RhythmMamba-main/logs/UBFC_UBFC_UBFC_RhythmMamba_ROISGPM_WarmLR1e6_v110.train.log`
- 结论：MAE/RMSE/MAPE/Pearson 与 v1.02 baseline 持平，SNR 高于 v1.02；但仍低于 v1.05 Time05Full epoch21 强基线。

## 指标对比

| 版本 | MAE | RMSE | MAPE | Pearson | SNR |
|---|---:|---:|---:|---:|---:|
| v1.02 baseline | 0.450045 | 0.731237 | 0.456979 | 0.997499 | 8.011478 |
| v1.05 Time05Full epoch21 | 0.405041 | 0.714424 | 0.405539 | 0.997695 | 8.161175 |
| V1.10 full from scratch epoch21 | 3.015302 | 4.032363 | 3.119841 | 0.914391 | -9.914507 |
| V1.10 warm init only-test | 0.450045 | 0.731237 | 0.456979 | 0.997499 | 8.173184 |
| V1.10 warm LR=3e-4 epoch7 | 0.585059 | 0.810081 | 0.582941 | 0.997260 | 7.372652 |
| V1.10 final warm LR=1e-6 epoch2 | 0.450045 | 0.731237 | 0.456979 | 0.997499 | 8.092382 |

## V1.10 架构实现

### 1. ROI-aware Frame Stem

替代原始全局 frame average pooling 的单一路径，新增 `ROIAwareFrameStem`：

- ROI 数量默认 5：额头、左脸颊、右脸颊、鼻周/中脸、下巴/下脸。
- ROI token 共享轻量 MLP stem。
- quality gate 使用运动强度、时序稳定性、频谱尖锐度三类特征学习 ROI 权重。
- 为了保护 v1.05 强主干，最终采用残差融合：
  `global_tokens + ROI_RESIDUAL_SCALE * (roi_fused - global_tokens)`。

### 2. Adaptive Spectral Gate

新增 `AdaptiveSpectralGate`：

- 对 ROI token 做 `rFFT`。
- 使用 0.7-2.5 Hz 作为主 HR 频带先验。
- 对二次谐波频带给予温和增强。
- 对低频漂移和高频噪声给较低 prior。
- 叠加可学习 soft gate，并通过 `irFFT` 回到时域 token。

### 3. Periodic-aware Mamba

新增 `PeriodicTokenModulator`：

- Fourier periodic positional encoding 覆盖 0.7、1.0、1.3、1.7、2.1、2.5 Hz。
- HR distribution head 输出 45-150 BPM logits。
- 使用预测 HR distribution 形成基频与二倍频相位特征，调制 token。
- PE 和 phase projection 零初始化，保证热启动时不会强扰动 v1.05 主干。

### 4. Physiology-consistency Loss

在 v1.05 `TIME_WEIGHT=0.5` 基础上扩展 `Hybrid_Loss`：

- time-domain negative Pearson loss。
- frequency CE/KL loss，默认保持 v1.05 频域口径。
- ROI phase consistency loss：可信 ROI 在主频相位上保持一致。
- harmonic consistency loss：约束低频漂移和过强二倍频。
- auxiliary HR distribution loss：监督 periodic modulator 的 HR logits。

默认权重均为 0，不打开 V1.10 配置时不影响旧实验。

## 关键工程文件

- `/root/RhythmMamba-main/neural_methods/model/RhythmMamba.py`
- `/root/RhythmMamba-main/neural_methods/loss/TorchLossComputer.py`
- `/root/RhythmMamba-main/neural_methods/trainer/RhythmMambaTrainer.py`
- `/root/RhythmMamba-main/config.py`
- `/root/RhythmMamba-main/test_rhythmmamba_v110.py`

## 关键配置

- `2UBFC-rPPG_RHYTHMMAMBA_ROI_V110.yaml`
- `2UBFC-rPPG_RHYTHMMAMBA_ROISG_V110.yaml`
- `2UBFC-rPPG_RHYTHMMAMBA_ROISGP_V110.yaml`
- `2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_FULL_V110.yaml`
- `2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_WARM_V110.yaml`
- `2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_WARM_LR1E6_V110.yaml`
- `2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_WARM_LR1E6_V110_EPOCH2_TEST.yaml`

## 运行记录

### 失败实验 1：full from scratch

```bash
LD_LIBRARY_PATH="/root/anaconda3/envs/rhythmmamba527/lib:${LD_LIBRARY_PATH:-}" \
RHYTHMMAMBA_NUM_WORKERS=4 RHYTHMMAMBA_PREFETCH_FACTOR=2 RHYTHMMAMBA_PIN_MEMORY=1 RHYTHMMAMBA_PERSISTENT_WORKERS=1 \
/root/anaconda3/envs/rhythmmamba527/bin/python -u main.py \
  --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_FULL_V110.yaml \
  2>&1 | tee logs/UBFC_UBFC_UBFC_RhythmMamba_ROISGPM_Full_v110.train.log
```

结论：从零训练 full V1.10 在 22 epoch 下严重退化，不可作为最终模型。

### 失败实验 2：v1.05 warm LR=3e-4

```bash
LD_LIBRARY_PATH="/root/anaconda3/envs/rhythmmamba527/lib:${LD_LIBRARY_PATH:-}" \
RHYTHMMAMBA_NUM_WORKERS=4 RHYTHMMAMBA_PREFETCH_FACTOR=2 RHYTHMMAMBA_PIN_MEMORY=1 RHYTHMMAMBA_PERSISTENT_WORKERS=1 \
/root/anaconda3/envs/rhythmmamba527/bin/python -u main.py \
  --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_WARM_V110.yaml \
  2>&1 | tee logs/UBFC_UBFC_UBFC_RhythmMamba_ROISGPM_Warm_v110.train.log
```

结论：LR=3e-4 会把强热启动点推离，MAE 降至 0.585，不达标。

### 达标实验：v1.05 warm LR=1e-6 + ROI residual 0.05

```bash
LD_LIBRARY_PATH="/root/anaconda3/envs/rhythmmamba527/lib:${LD_LIBRARY_PATH:-}" \
RHYTHMMAMBA_NUM_WORKERS=4 RHYTHMMAMBA_PREFETCH_FACTOR=2 RHYTHMMAMBA_PIN_MEMORY=1 RHYTHMMAMBA_PERSISTENT_WORKERS=1 \
/root/anaconda3/envs/rhythmmamba527/bin/python -u main.py \
  --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_WARM_LR1E6_V110.yaml \
  2>&1 | tee logs/UBFC_UBFC_UBFC_RhythmMamba_ROISGPM_WarmLR1e6_v110.train.log
```

结论：完整 train_and_test 跑完，指标达到原始 v1.02 baseline 水平，SNR 略高。

## 验证

已运行：

```bash
LD_LIBRARY_PATH="/root/anaconda3/envs/rhythmmamba527/lib:${LD_LIBRARY_PATH:-}" \
/root/anaconda3/envs/rhythmmamba527/bin/python -m unittest test_rhythmmamba_v110.py
```

结果：`Ran 1 test in 1.349s, OK`。

此前已运行包含 augmentation/config/loss/TTA/V1.10 的小测试集合，均通过。

## 后续建议

1. 不建议把当前 V1.10 直接声称超过 v1.05；它目前只达到原始 baseline，可作为工程起点和论文创新主线雏形。
2. 下一步优先做 `ROI_RESIDUAL_SCALE` sweep：0.02、0.05、0.10、0.15。
3. 做冻结主干实验：冻结 v1.05 主干，只训练 ROI stem、spectral gate、periodic modulator 与 aux heads。
4. 做 subject47 / 低 SNR 样本专项分析，重点看 spectral gate 是否提升 SNR 和错误 HR 峰选择。
5. 再做严格 ablation：v1.05、+ROI、+SpectralGate、+Periodic、+PhysLoss、Full。
6. GRL/domain adversarial 暂不放在 UBFC intra 主线，应放到 PURE->UBFC、UBFC->PURE、MMPD->UBFC 等 cross-dataset 设置。

