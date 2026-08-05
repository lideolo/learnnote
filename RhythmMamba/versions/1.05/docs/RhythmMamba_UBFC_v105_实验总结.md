# RhythmMamba UBFC-rPPG v1.05 最终实验总结

日期：2026-06-11

## 结论

本轮在 v1.04 的 loss 优化基础上继续实验，最终找到一个超过 v1.02 的 UBFC-rPPG 候选：

**Time05 + 内部验证 MAE 选 epoch21 + full-train epoch21 held-out 测试**。

对 v1.02 AugFix epoch29，官方 `calculate_metrics` 同口径 held-out 结果如下：

| 候选 | 选择方式 | MAE | RMSE | MAPE | Pearson | SNR |
|---|---|---:|---:|---:|---:|---:|
| v1.02 AugFix epoch29 | 原 v1.02 last epoch | 0.450045 | 0.731237 | 0.456979 | 0.997499 | 8.011478 |
| v1.05 Time05Full epoch21 | 内部验证 MAE 选 epoch，再 full-train 同 epoch 测试 | **0.405041** | **0.714424** | **0.405539** | **0.997695** | **8.161175** |

相对 v1.02：

- MAE 降低 `0.0450045`，约 `10.0%`。
- RMSE 降低 `0.0168123`，约 `2.3%`。
- MAPE 降低 `0.0514403`。
- Pearson 提升 `0.0001960`。
- SNR 提升 `0.149697 dB`。

## 科学选择逻辑

为避免直接按 held-out 挑 checkpoint，本轮使用 v1.03/v1.04 沿用的内部验证协议：

```text
内部验证：
  train: 0.00-0.60
  valid/test field: 0.60-0.72

最终 held-out：
  train: 0.00-0.72
  test: 0.72-1.00
```

在内部验证的 Time05 训练轨迹中，逐 epoch 复算 FFT HR 指标后：

| 内部验证候选 | Val MAE | Val RMSE | Val MAPE | Val Pearson |
|---|---:|---:|---:|---:|
| Time05 epoch21 | **1.404140** | **2.474838** | **1.999311** | 0.997013 |
| Time05 epoch26, 原 loss-selected | 1.620162 | 2.498297 | 2.250157 | 0.996796 |
| v1.04 control | 1.620162 | 2.721786 | 2.293708 | 0.996959 |

因此本轮最终规则是：`TIME_WEIGHT=0.5` 作为候选训练目标，checkpoint 用内部验证 MAE 选择 epoch21；再在 full split 上使用同一训练 epoch 的 checkpoint 做一次 held-out 评估。

这相当于常见的 validation-based early stopping / epoch-count selection，而不是按 held-out 最优点直接挑模型。

## 本轮主要改动

代码入口：

- `config.py`
  - 新增 `TRAIN.SEED`。
  - 新增 `TRAIN.MODEL_SELECTION.METRIC`。
  - 新增 `INFERENCE.TTA.HORIZONTAL_FLIP` 和 `INFERENCE.TTA.ALIGN_SIGN`。
- `main.py`
  - 解析配置后调用 `set_random_seed(config.TRAIN.SEED)`。
- `neural_methods/trainer/RhythmMambaTrainer.py`
  - 抽出统一预测函数 `_predict_ppg()`。
  - 支持水平翻转 TTA 与符号对齐。
  - 支持验证阶段计算 FFT HR 指标，并按 `TRAIN.MODEL_SELECTION.METRIC` 选择 checkpoint。
  - 复用 `_store_batch_predictions()`，减少验证/测试路径差异。
- 新增 held-out 复核配置：
  - `configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_LOSSOPT_V105_TIME05_EPOCH21_TEST.yaml`

## 负结果与排除项

| 候选 | 结论 |
|---|---|
| v1.05 TTA | MAE `0.4950`、RMSE `0.7795`，退步 |
| Time05 full last epoch29 | MAE `0.4950`、RMSE `0.7795`，退步 |
| Seed101 full | MAE `0.5401`、RMSE `0.8249`，退步 |
| Seed ensemble | SNR/Pearson 可提升，但 MAE/RMSE 不过 v1.02 |
| SWA / checkpoint prediction ensemble | 没有超过 v1.02 主指标 |
| GREEN/POS/CHROM 先验融合 | 在该 split 上不稳定或明显退步 |
| 无标签频谱后处理 | 内部验证最优的短窗规则迁移到 held-out 退步，拒绝 |

## 与 v1.02 的优缺点

优点：

- 主指标 MAE、RMSE、MAPE、Pearson、SNR 全部优于 v1.02。
- 改进来自验证集选择的训练权重和 epoch，而不是改评价函数。
- Time-domain Pearson loss 权重从 `0.2` 提到 `0.5` 后，epoch21 的预测修正了 v1.02/v1.02-epoch23 之间反复出现的 subject49/subject9 小误差交换问题。
- 新增的 seed、TTA、metric-selection 配置让后续多 split / 多 seed 实验更容易复现。

缺点：

- UBFC held-out 只有 12 个 subject，MAE 改善主要来自少数 0.54 bpm 量化误差减少，统计稳健性仍需多 split 或 subject-level cross-validation。
- 最大误差 subject47 仍为约 `+2.16 bpm`，还没有被解决。
- `TIME_WEIGHT=0.5` 的 last epoch29 反而退步，说明该候选对 checkpoint/epoch 选择更敏感。
- 本轮 full-train 使用内部验证确定的 epoch count，但还不是完整多 seed 平均结果。

## 关键日志

- 官方复核日志：`logs/UBFC_UBFC_UBFC_RhythmMamba_LossOpt_v105_Time05Epoch21.heldout.log`
- full-train 训练日志：`logs/UBFC_UBFC_UBFC_RhythmMamba_LossOpt_v105_Time05Full.train.log`
- v1.02 对照日志：`logs/UBFC_UBFC_UBFC_RhythmMamba_AugFix_v102.only_test.log`

官方复核输出：

```text
FFT MAE (FFT Label): 0.4050405040504046 +/- 0.1698882636199543
FFT RMSE (FFT Label): 0.7144242964170822 +/- 0.3639647822212871
FFT MAPE (FFT Label): 0.4055387344112288 +/- 0.15868320417527815
FFT Pearson (FFT Label): 0.9976950563581655 +/- 0.021458272340701304
FFT SNR (FFT Label): 8.16117498579995 +/- 1.4600570288992303 (dB)
```

## 测试

```text
python -m unittest \
  test_rhythmmamba_augmentation.py \
  test_rhythmmamba_configurable.py \
  test_rhythmmamba_loss_configurable.py \
  test_rhythmmamba_tta.py

Ran 9 tests in 0.590s
OK
```

## 参考依据

- RhythmMamba 长时序 rPPG 建模与频域设计：<https://arxiv.org/abs/2404.06483>
- rPPG-Toolbox benchmark 与标准化评估实践：<https://proceedings.neurips.cc/paper_files/paper/2023/hash/d7d0d548a6317407e02230f15ce75817-Abstract-Datasets_and_Benchmarks.html>
- SWA/checkpoint averaging 作为负结果参考方向：<https://arxiv.org/abs/1803.05407>
- UBFC-rPPG 数据集说明：<https://sites.google.com/view/ybenezeth/ubfcrppg>

## 后续建议

下一轮最值得做的是 subject-level 多折验证和多 seed 均值/置信区间。如果 Time05 epoch-count selection 在多折上仍稳定优于 v1.02，再把它作为正式主版本；否则应把本轮结果视为 UBFC 当前 split 上的有效改进候选。
