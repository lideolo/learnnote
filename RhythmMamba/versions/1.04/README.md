# RhythmMamba UBFC-rPPG LossOpt v1.04 实验总结

## 结论

本轮 v1.04 选择的优化方向是训练目标/频域损失优化。实验基于 v1.02 AugFix 代码路径，并参考 v1.03 的内部验证协议，目标是在不直接调 held-out UBFC 测试集的前提下改进 MAE/RMSE/Pearson。

最终结论：v1.04 的 loss 候选没有同时超过内部验证 control 的 MAE、RMSE、Pearson，因此不推进 held-out 测试，不替代当前最佳稳定版本。当前 UBFC-rPPG held-out 最佳稳定版本仍是 v1.02 AugFix。

## 原实验状态

| 版本 | 改动性质 | Held-out MAE | Held-out RMSE | Held-out Pearson | 状态 |
|---|---|---:|---:|---:|---|
| v1.01 | DataLoader/CPU-GPU 优化后的 baseline | 0.540054 | 0.794938 | 0.997358 | baseline |
| v1.02 | 修复 RhythmMamba 训练增强 bug | 0.450045 | 0.731237 | 0.997499 | 当前最佳 |
| v1.03 | Mamba 模块参数筛选 | 0.495050 | 0.747671 | 0.997219 | 优于 v1.01，但不如 v1.02 |

v1.02 的关键收益来自训练增强实现修复：

- temporal upsampling/downsampling 只修改当前 sample，不再覆盖整个 batch。
- horizontal flip 使用每个 sample 自己的随机值，不再使用最后一个 sample 的随机值决定整批。
- 训练协议、数据切分、模型默认参数、FFT 评估保持不变。

v1.03 做了 Mamba 参数可配置化和小范围验证筛选，但最终 held-out 没超过 v1.02，因此没有晋升。

## v1.04 修改内容

### 1. 将 Hybrid_Loss 参数化

修改文件：

- `config.py`
- `neural_methods/loss/TorchLossComputer.py`
- `neural_methods/trainer/RhythmMambaTrainer.py`

新增默认配置：

```yaml
TRAIN:
  LOSS:
    TIME_WEIGHT: 0.2
    FREQ_CE_WEIGHT: 1.0
    FREQ_KL_WEIGHT: 0.0
    FREQ_STD: 3.0
```

默认值保持原 v1.02/v1.03 的 loss 行为：

```text
loss = 0.2 * NegPearson + 1.0 * FrequencyCE
```

新增后可测试：

```text
loss = TIME_WEIGHT * NegPearson
     + FREQ_CE_WEIGHT * FrequencyCE
     + FREQ_KL_WEIGHT * FrequencyKL
```

### 2. 使用已存在但原先未启用的频域 KL 项

原 `Frequency_loss` 已经返回 `loss_distribution_kl`，但 `Hybrid_Loss` 丢弃了它。v1.04 没有改写频域算法，只把 KL 权重做成可配置候选，便于科学筛选。

### 3. 修正 loss 计算中的设备硬编码

`TorchLossComputer` 内部部分张量原先硬编码 `.cuda()` 或 `torch.device('cuda')`。v1.04 改为跟随输入 tensor 的 device：

- 训练 CUDA 路径语义不变。
- loss 单元测试可以在 CPU 上验证权重组合逻辑。
- 降低以后在不同设备上调试的阻力。

### 4. 新增回归测试

新增：

- `test_rhythmmamba_loss_configurable.py`

验证点：

- 默认权重等价原始 Hybrid_Loss 公式。
- 自定义 `TIME_WEIGHT/FREQ_CE_WEIGHT/FREQ_KL_WEIGHT/FREQ_STD` 会实际进入 loss 计算。

同时保留并通过：

- `test_rhythmmamba_augmentation.py`
- `test_rhythmmamba_configurable.py`

## 防过拟合实验协议

v1.04 沿用 v1.03 的内部验证协议：

```text
内部验证筛选：
  Train: 0.00-0.60
  Valid/Test field: 0.60-0.72

Held-out 测试：
  Train: 0.00-0.72
  Test: 0.72-1.00
```

规则：

- 候选选择只看内部验证 split。
- 只有候选在内部验证 MAE/RMSE/Pearson 上有足够证据优于 control，才进入 held-out 测试。
- 本轮没有候选满足三指标共同过关，因此没有运行 held-out 测试。

## v1.04 候选结果

内部验证 control 使用 v1.02 默认 loss、v1.03 记录的 `ValDefault`：

| 候选 | Loss 设置 | Best epoch | Min val loss | Val MAE | Val RMSE | Val Pearson | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| Control | `time=0.2, CE=1.0, KL=0.0, std=3` | 26 | 4.646198 | 1.620162 | 2.721786 | 0.996959 | 对照 |
| KL005STD3 | `time=0.2, CE=1.0, KL=0.05, std=3` | 25 | 4.673183 | 1.728173 | 2.774848 | 0.996413 | 拒绝，三指标变差 |
| KL005STD5 | `time=0.2, CE=1.0, KL=0.05, std=5` | 19 | 4.653902 | 1.620162 | 2.721786 | 0.996959 | 与 control HR 持平，loss/SNR 略差 |
| Time05 | `time=0.5, CE=1.0, KL=0.0, std=3` | 26 | 4.696975 | 1.620162 | 2.498297 | 0.996796 | RMSE 改善，但 Pearson 下降 |

Time05 是混合结果：

- 优点：内部验证 RMSE 从 `2.721786` 降到 `2.498297`。
- 不足：MAE 没有实质改善，Pearson 从 `0.996959` 降到 `0.996796`。
- 决策：不进入 held-out 测试，因为目标同时关注 MAE/RMSE/Pearson，且 UBFC 验证样本较少，单指标改善容易偶然。

## 修改后的优缺点

### 优点

- loss 相关实验可以通过 YAML 配置完成，不需要每次改 Python 源码。
- 默认行为保持兼容，不影响 v1.02/v1.03 默认训练。
- 可复现记录更完整：每个候选都有独立配置和日志。
- 更符合防过拟合原则：没有因为 Time05 的单项 RMSE 改善就直接触碰 held-out 测试。
- 设备硬编码减少，loss 逻辑更容易单元测试。

### 缺点

- KL 候选增加了训练计算量，尤其 `FREQ_KL_WEIGHT > 0` 时训练速度更慢。
- KL 候选没有带来内部验证 HR 指标提升。
- Time05 只改善 RMSE，Pearson 略降，不适合作为稳定提升。
- 本轮没有产生新的 held-out 最佳模型。

## 新问题与已解决问题

### 已解决

- `Hybrid_Loss` 原先不能通过配置调整权重，v1.04 已解决。
- `Frequency_loss` 返回的 KL 项原先无法在 RhythmMamba 中实验，v1.04 已支持。
- loss 计算中部分 CUDA 硬编码不利于 CPU 测试，v1.04 已改为跟随输入 device。

### 新发现的问题

- 频域 KL 对 UBFC 内部验证不稳定：较尖锐的 `std=3` 让 MAE/RMSE/Pearson 都变差。
- 更平滑的 `std=5` 可以恢复 HR 指标到 control 水平，但没有带来收益。
- 提高 time-domain Pearson 权重可能改善 RMSE，但会轻微牺牲 Pearson，说明该方向需要更谨慎的多指标筛选。

## 资源约束执行情况

训练时遵守用户要求：

- 显存预留目标：至少 2GB。
- 系统可用内存预留目标：至少 3GB。

实际观察：

- 训练峰值显存通常约 `8.1-8.3GB / 12.2GB`，保留约 `3.6-3.9GB`。
- 系统 available memory 多数抽样在 `7GB-10GB+`，满足 3GB 预留。
- 后续候选将 DataLoader workers 从 2 降到 1，以降低内存压力。

## 测试结果

```text
test_rhythmmamba_augmentation.py: Ran 3 tests, OK
test_rhythmmamba_configurable.py: Ran 2 tests, OK
test_rhythmmamba_loss_configurable.py: Ran 2 tests, OK
```

## 最终建议

当前不建议把 v1.04 loss 候选晋升为最佳模型。UBFC-rPPG 当前最佳稳定结果仍保留 v1.02 AugFix：

```text
MAE:     0.45004500450044915
RMSE:    0.7312365800752446
Pearson: 0.9974990351874917
```

后续若继续优化，建议优先考虑：

1. 用更稳健的多折内部验证或 subject-level 交叉验证筛 loss 权重。
2. 做轻量训练策略优化，例如 early stopping 或 HR 指标驱动的 checkpoint selection。
3. 再小范围评估 `TIME_WEIGHT`，但必须同时约束 Pearson 不下降。

