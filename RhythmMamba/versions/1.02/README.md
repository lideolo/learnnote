# RhythmMamba UBFC-rPPG Augmentation Fix 1.02

Version name: `1.02`

This version is based on the 1.01 UBFC-rPPG baseline and changes only the
RhythmMamba training-time augmentation implementation. The training protocol,
preprocessed data, model hyperparameters, loss, optimizer, scheduler, and FFT
evaluation settings remain aligned with 1.01.

## Change

`neural_methods/trainer/RhythmMambaTrainer.py`

- Fixed temporal upsampling and downsampling augmentation to write only the
  current sample instead of overwriting the whole batch.
- Fixed horizontal flipping to use each sample's own random decision instead of
  using the last sample's random value for the whole batch.
- Kept the original augmentation policy and thresholds:
  - high HR `> 90`: temporal upsampling
  - low HR `< 75`: temporal downsampling
  - otherwise: unchanged

Regression coverage:

```bash
cd /root/RhythmMamba-main
/root/anaconda3/envs/rhythmmamba527/bin/python test_rhythmmamba_augmentation.py
```

Result:

```text
Ran 3 tests in 0.061s
OK
```

## Reproduction

Environment:

```bash
conda activate rhythmmamba527
export LD_LIBRARY_PATH="/root/anaconda3/envs/rhythmmamba527/lib:${LD_LIBRARY_PATH:-}"
```

Training:

```bash
cd /root/RhythmMamba-main
export RHYTHMMAMBA_NUM_WORKERS=4
export RHYTHMMAMBA_PREFETCH_FACTOR=2
export RHYTHMMAMBA_PIN_MEMORY=1
export RHYTHMMAMBA_PERSISTENT_WORKERS=1

/root/anaconda3/envs/rhythmmamba527/bin/python main.py \
  --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_AUGFIX_V102.yaml \
  > logs/UBFC_UBFC_UBFC_RhythmMamba_AugFix_v102.train.log 2>&1
```

Only-test verification:

```bash
/root/anaconda3/envs/rhythmmamba527/bin/python main.py \
  --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_AUGFIX_V102_TEST.yaml \
  > logs/UBFC_UBFC_UBFC_RhythmMamba_AugFix_v102.only_test.log 2>&1
```

Checkpoint:

```text
/experiment0/user/PreTrainedModels/UBFC-rPPG_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/UBFC_UBFC_UBFC_RhythmMamba_AugFix_v102_Epoch29.pth
```

## Resource Notes

The training run was monitored while preserving user resources.

Observed GPU memory stayed around `8.1-8.4 GB / 12.2 GB`, leaving more than
`2 GB` free. System memory spot checks stayed above `3 GB` available.

Final post-run resources:

```text
GPU memory: 1871 / 12227 MB
System memory available: 12098 MB
```

## Metrics

1.01 baseline Run2:

```text
MAE:     0.5400540054005383 +/- 0.1683914215469826 bpm
RMSE:    0.7949379717666831 +/- 0.3618622940730566 bpm
MAPE:    0.5552086596956083 +/- 0.16436819264736727 %
Pearson: 0.9973577707377148 +/- 0.022972760280592922
SNR:     8.524458794506911 +/- 1.4376775617814308 dB
```

1.02 AugFix v102, independently verified by only-test:

```text
MAE:     0.45004500450044915 +/- 0.16637468506464817 bpm
RMSE:    0.7312365800752446 +/- 0.3618622940730564 bpm
MAPE:    0.45697906362933527 +/- 0.15533549755317777 %
Pearson: 0.9974990351874917 +/- 0.02235100624138243
SNR:     8.011477550113215 +/- 1.3630708974080006 dB
```

Delta versus 1.01:

```text
MAE:     -0.09000900090008918 bpm
RMSE:    -0.0637013916914385 bpm
Pearson: +0.00014126444977685947
```

## Stored Files

```text
configs/
  2UBFC-rPPG_RHYTHMMAMBA_AUGFIX_V102.yaml
  2UBFC-rPPG_RHYTHMMAMBA_AUGFIX_V102_TEST.yaml

code/
  neural_methods/trainer/RhythmMambaTrainer.py

tests/
  test_rhythmmamba_augmentation.py

logs/
  UBFC_UBFC_UBFC_RhythmMamba_AugFix_v102.train.log
  UBFC_UBFC_UBFC_RhythmMamba_AugFix_v102.only_test.log
  UBFC_UBFC_UBFC_RhythmMamba_AugFix_v102.monitor.log

patches/
  augmentation_fix_v102.patch

project_status_at_capture.txt
```
