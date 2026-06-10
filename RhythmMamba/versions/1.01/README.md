# RhythmMamba UBFC-rPPG Reproduction Baseline 1.01

Version name: `1.01`

This directory stores the baseline reproducible state for the RTX 5070 RhythmMamba UBFC-rPPG reproduction and the second DataLoader / CPU-GPU coordination run.

## Scope

This version records:

- UBFC-rPPG intra-dataset reproduction settings.
- The successful RTX 5070 / WSL2 environment assumptions.
- The serial preprocessing + ffmpeg CLI workaround for UBFC video decoding.
- The second reproduction with DataLoader and CPU-GPU transfer optimization.
- The final metrics and logs.

This directory intentionally does not store raw datasets, preprocessed `.npy` files, or model checkpoints.

## Runtime Environment

Use the conda environment:

```bash
conda activate rhythmmamba527
```

Recorded environment:

```text
Python: 3.11.15
PyTorch: 2.12.0+cu130
CUDA: 13.0
GPU: RTX 5070 12GB
OS: WSL2 Ubuntu 22.04
```

Important runtime library path:

```bash
export LD_LIBRARY_PATH="/root/anaconda3/envs/rhythmmamba527/lib:${LD_LIBRARY_PATH:-}"
```

## Data Paths

Raw UBFC-rPPG dataset:

```text
/root/Datasets/UBFC/
```

Preprocessed cache:

```text
/root/RhythmMamba_Preprocessed/UBFC/
```

Model checkpoint directory:

```text
/experiment0/user/PreTrainedModels/UBFC-rPPG_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/
```

Final Run2 checkpoint:

```text
/experiment0/user/PreTrainedModels/UBFC-rPPG_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2_Epoch29.pth
```

## UBFC Protocol

The reproduction follows the RhythmMamba paper's UBFC intra-dataset protocol:

```text
Train: first 30 subjects
Test: remaining 12 subjects
```

In config:

```yaml
TRAIN.DATA.BEGIN: 0.0
TRAIN.DATA.END: 0.72
TEST.DATA.BEGIN: 0.72
TEST.DATA.END: 1.0
```

Actual cached clip counts:

```text
Train: 342 clips
Test: 141 clips
```

## Core Training Settings

```yaml
MODEL.NAME: RhythmMamba
TRAIN.BATCH_SIZE: 8
TRAIN.EPOCHS: 30
TRAIN.LR: 3e-4
TRAIN.AUG: 1
TEST.USE_LAST_EPOCH: True
INFERENCE.BATCH_SIZE: 2
INFERENCE.EVALUATION_METHOD: FFT
```

Preprocessing:

```yaml
FS: 30
DATA_FORMAT: NDCHW
DATA_TYPE: ['Standardized']
LABEL_TYPE: Standardized
CHUNK_LENGTH: 160
RESIZE: 128x128
CROP_FACE: True
USE_LARGE_FACE_BOX: True
LARGE_BOX_COEF: 1.5
DO_DYNAMIC_DETECTION: False
```

## DataLoader / CPU-GPU Optimization

Run2 keeps the model, loss, data split, learning rate, epochs, batch size, and evaluation method unchanged. It only optimizes data loading and host-to-device transfer:

```text
num_workers=4
pin_memory=True
persistent_workers=True
prefetch_factor=2
non_blocking=True
optimizer.zero_grad(set_to_none=True)
```

Environment variables supported by `main.py`:

```bash
export RHYTHMMAMBA_NUM_WORKERS=4
export RHYTHMMAMBA_PREFETCH_FACTOR=2
export RHYTHMMAMBA_PIN_MEMORY=1
export RHYTHMMAMBA_PERSISTENT_WORKERS=1
```

Split-specific overrides are also supported:

```text
RHYTHMMAMBA_TRAIN_NUM_WORKERS
RHYTHMMAMBA_VALID_NUM_WORKERS
RHYTHMMAMBA_TEST_NUM_WORKERS
RHYTHMMAMBA_UNSUPERVISED_NUM_WORKERS
```

## Reproduction Commands

From the project root:

```bash
cd /root/RhythmMamba-main
conda activate rhythmmamba527
export LD_LIBRARY_PATH="/root/anaconda3/envs/rhythmmamba527/lib:${LD_LIBRARY_PATH:-}"
```

Run the second reproduction:

```bash
bash tools/run_ubfc_dataloader_opt_run2.sh
```

Run only the final test:

```bash
/root/anaconda3/envs/rhythmmamba527/bin/python main.py \
  --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT_RUN2_TEST.yaml \
  > logs/UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2.only_test.log 2>&1
```

Monitor log:

```bash
tail -f logs/UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2.monitor.log
```

## Final Metrics

First reproduction:

```text
MAE:     0.54 +/- 0.17 bpm
RMSE:    0.79 +/- 0.36 bpm
MAPE:    0.56% +/- 0.16%
Pearson: 0.997 +/- 0.023
SNR:     8.32 +/- 1.45 dB
```

Second reproduction Run2:

```text
MAE:     0.5400540054005383 +/- 0.1683914215469826 bpm
RMSE:    0.7949379717666831 +/- 0.3618622940730566 bpm
MAPE:    0.5552086596956083 +/- 0.16436819264736727%
Pearson: 0.9973577707377148 +/- 0.022972760280592922
SNR:     8.524458794506911 +/- 1.4376775617814308 dB
```

## Stored Files

```text
configs/
  2UBFC-rPPG_RHYTHMMAMBA_REAL.yaml
  2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT.yaml
  2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT_RUN2.yaml
  2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT_RUN2_TEST.yaml

code/
  main/main.py
  neural_methods/trainer/RhythmMambaTrainer.py
  dataset/data_loader/UBFCrPPGLoader.py

tools/
  monitor_training.py
  run_ubfc_dataloader_opt_run2.sh

docs/
  RTX5070_RhythmMamba_UBFC复现完整说明_260528.md
  UBFC实验参数与原论文对比.md

logs/
  UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2.monitor.log
  UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2.only_test.log

patches/
  dataloader_cpu_gpu_opt.patch

project_status_at_capture.txt
```

## Notes

- The `glibc / libstdc++` issue observed in this reproduction mainly affects UBFC raw video decoding and preprocessing through OpenCV `VideoCapture`.
- After preprocessing, training uses cached `.npy` files and does not read raw `.avi` files.
- Run2's DataLoader optimization improves throughput and transfer behavior, but does not materially change the final HR metrics.

