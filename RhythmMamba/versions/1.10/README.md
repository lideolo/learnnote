# RhythmMamba V1.10

主题：ROI-aware Adaptive Spectral-Gated Periodic Mamba  
中文：生理区域感知的自适应频谱门控周期 Mamba

## Final Checkpoint

`checkpoints/UBFC_UBFC_UBFC_RhythmMamba_ROISGPM_WarmLR1e6_v110_Epoch2.pth`

SHA256:

```text
efd6d3ce93711561c24d83dff23465e34ee15a5d31295a3819cab3c32b0eca95
```

## Best V1.10 Result

配置：`configs/2UBFC-rPPG_RHYTHMMAMBA_ROISGPM_WARM_LR1E6_V110.yaml`  
日志：`logs/UBFC_UBFC_UBFC_RhythmMamba_ROISGPM_WarmLR1e6_v110.train.log`

| Metric | Value |
|---|---:|
| MAE | 0.45004500450044915 |
| RMSE | 0.7312365800752446 |
| MAPE | 0.45697906362933527 |
| Pearson | 0.9974990351874917 |
| SNR | 8.092382434457273 |

该结果与 v1.02 baseline 的 MAE/RMSE/MAPE/Pearson 持平，SNR 更高；未超过 v1.05 Time05Full epoch21 强基线。

## Contents

- `code/`: V1.10 相关源码快照。
- `configs/`: ROI、ROI+SpectralGate、ROI+SpectralGate+Periodic、Full、Warm、WarmLR1e6 配置。
- `logs/`: full-from-scratch、warm-init、warm LR=3e-4、warm LR=1e-6 实验日志。
- `tests/`: V1.10 单元测试。
- `docs/`: 中文实验总结。
- `checkpoints/`: 最终达标 V1.10 checkpoint。

## Notes

- 从零 full V1.10 训练失败，不能作为论文主结果。
- LR=3e-4 热启动会破坏强初始化，指标低于原始 baseline。
- 当前达标版本使用 v1.05 checkpoint 非严格热启动、`ROI_RESIDUAL_SCALE=0.05`、`LR=1e-6`、3 epoch。

