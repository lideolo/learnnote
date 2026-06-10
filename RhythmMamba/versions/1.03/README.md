# RhythmMamba UBFC-rPPG Module Optimization 1.03

Version name: `1.03`

This version records a controlled RhythmMamba module-parameter optimization
experiment on top of the 1.02 AugFix code path.

Important conclusion: this module experiment is reproducible, but it is **not
promoted as the best stable model**, because the final held-out test did not
beat 1.02. The current best stable UBFC-rPPG version remains `1.02`.

## Anti-Overfitting Protocol

To avoid tuning directly on the held-out UBFC test split:

1. Candidate selection used an internal validation split only.
2. The held-out `0.72-1.00` split was evaluated once, after selecting the
   candidate from validation.
3. Candidate count was deliberately small and pre-registered.

Splits:

```text
Validation screening:
  Train: 0.00-0.60
  Valid: 0.60-0.72
  Test field in screening configs: 0.60-0.72

Final held-out evaluation:
  Train: 0.00-0.72
  Test:  0.72-1.00
```

## Code Change

`RhythmMamba` module parameters are now configurable while preserving the
original defaults:

```yaml
MODEL:
  RHYTHMMAMBA:
    DEPTH: 24
    EMBED_DIM: 96
    MLP_RATIO: 2
    DROP_PATH_RATE: 0.1
    MAMBA_D_STATE: 48
    MAMBA_D_CONV: 4
    MAMBA_EXPAND: 2
    MULTI_TEMPORAL_PATHS: 3
```

The selected final candidate changed only:

```yaml
DROP_PATH_RATE: 0.15
MAMBA_D_STATE: 32
```

## Tests

```bash
cd /root/RhythmMamba-main
export LD_LIBRARY_PATH="/root/anaconda3/envs/rhythmmamba527/lib:${LD_LIBRARY_PATH:-}"
/root/anaconda3/envs/rhythmmamba527/bin/python test_rhythmmamba_augmentation.py
/root/anaconda3/envs/rhythmmamba527/bin/python test_rhythmmamba_configurable.py
```

Results:

```text
test_rhythmmamba_augmentation.py: Ran 3 tests, OK
test_rhythmmamba_configurable.py: Ran 2 tests, OK
```

## Candidate Screening

All candidates used the 1.02 augmentation fix and the same train/validation
protocol.

| Candidate | Module change | Best epoch | Min validation loss | Validation MAE | Validation RMSE | Validation Pearson | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| Default | 1.02 defaults | 26 | 4.6461975659642905 | 1.6201620162016197 | 2.7217864630107944 | 0.9969593871067349 | Control |
| State32-DP015 | `MAMBA_D_STATE=32`, `DROP_PATH_RATE=0.15` | 26 | 4.643556288310459 | 1.6201620162016197 | 2.7217864630107944 | 0.9969593871067349 | Selected by lowest validation loss with no HR metric regression vs Default |
| Path2-DP015 | `MULTI_TEMPORAL_PATHS=2`, `DROP_PATH_RATE=0.15` | 16 | 4.644200989178249 | 1.7281728172817268 | 2.732481146500126 | 0.9963329073936371 | Rejected, HR metrics worse |

## Final Held-Out Test

Selected candidate:

```text
UBFC_UBFC_UBFC_RhythmMamba_MambaOpt_v103_State32DP015_Epoch29.pth
```

Independent only-test result:

```text
MAE:     0.4950495049504949 +/- 0.16174511440779998 bpm
RMSE:    0.7476709147541034 +/- 0.3596106542782038 bpm
MAPE:    0.4992802480625001 +/- 0.15015857799470292 %
Pearson: 0.9972186154848613 +/- 0.023569117358221602
SNR:     8.30902829062464 +/- 1.3943895481952011 dB
```

Comparison:

| Version | MAE | RMSE | Pearson | Status |
|---|---:|---:|---:|---|
| 1.01 baseline | 0.5400540054005383 | 0.7949379717666831 | 0.9973577707377148 | Baseline |
| 1.02 AugFix | 0.45004500450044915 | 0.7312365800752446 | 0.9974990351874917 | Current best |
| 1.03 State32-DP015 | 0.4950495049504949 | 0.7476709147541034 | 0.9972186154848613 | Not promoted |

1.03 improved MAE/RMSE over 1.01, but it regressed versus 1.02 and slightly
regressed Pearson versus both 1.01 and 1.02. Therefore it is kept as a
traceable module-optimization experiment, not as the best stable checkpoint.

## Resource Notes

Validation and final training used:

```bash
export RHYTHMMAMBA_NUM_WORKERS=2
export RHYTHMMAMBA_PREFETCH_FACTOR=2
export RHYTHMMAMBA_PIN_MEMORY=1
export RHYTHMMAMBA_PERSISTENT_WORKERS=1
```

Observed GPU memory stayed below about `8.6 GB / 12.2 GB`, leaving more than
`2 GB` free. System memory spot checks stayed well above `3 GB` available.

## Stored Files

```text
configs/
  2UBFC-rPPG_RHYTHMMAMBA_MAMBAOPT_V103_VAL_DEFAULT.yaml
  2UBFC-rPPG_RHYTHMMAMBA_MAMBAOPT_V103_VAL_STATE32_DP015.yaml
  2UBFC-rPPG_RHYTHMMAMBA_MAMBAOPT_V103_VAL_PATH2_DP015.yaml
  2UBFC-rPPG_RHYTHMMAMBA_MAMBAOPT_V103_STATE32_DP015.yaml
  2UBFC-rPPG_RHYTHMMAMBA_MAMBAOPT_V103_STATE32_DP015_TEST.yaml

code/
  config.py
  neural_methods/model/RhythmMamba.py
  neural_methods/trainer/RhythmMambaTrainer.py

tests/
  test_rhythmmamba_augmentation.py
  test_rhythmmamba_configurable.py

logs/
  validation candidate train and monitor logs
  final train, monitor, and only-test logs

patches/
  rhythmmamba_module_config_v103.patch

project_status_at_capture.txt
```
