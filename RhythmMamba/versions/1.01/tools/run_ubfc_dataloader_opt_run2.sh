#!/usr/bin/env bash
set -euo pipefail

cd /root/RhythmMamba-main

export LD_LIBRARY_PATH="/root/anaconda3/envs/rhythmmamba527/lib:${LD_LIBRARY_PATH:-}"
export RHYTHMMAMBA_NUM_WORKERS="${RHYTHMMAMBA_NUM_WORKERS:-4}"
export RHYTHMMAMBA_PREFETCH_FACTOR="${RHYTHMMAMBA_PREFETCH_FACTOR:-2}"
export RHYTHMMAMBA_PIN_MEMORY="${RHYTHMMAMBA_PIN_MEMORY:-1}"
export RHYTHMMAMBA_PERSISTENT_WORKERS="${RHYTHMMAMBA_PERSISTENT_WORKERS:-1}"

RUN_NAME="UBFC_UBFC_UBFC_RhythmMamba_DataloaderOpt_Run2"
TRAIN_LOG="logs/${RUN_NAME}.train.log"
MONITOR_LOG="logs/${RUN_NAME}.monitor.log"
CKPT_DIR="/experiment0/user/PreTrainedModels/UBFC-rPPG_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

/root/anaconda3/envs/rhythmmamba527/bin/python main.py \
  --config_file configs/train_configs/intra/2UBFC-rPPG_RHYTHMMAMBA_REAL_DATALOADER_OPT_RUN2.yaml \
  > "${TRAIN_LOG}" 2>&1 &

TRAIN_PID="$!"
echo "Training PID: ${TRAIN_PID}"
echo "Training log: ${TRAIN_LOG}"
echo "Monitor log: ${MONITOR_LOG}"

/root/anaconda3/envs/rhythmmamba527/bin/python tools/monitor_training.py \
  --pid "${TRAIN_PID}" \
  --train-log "${TRAIN_LOG}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --model-prefix "${RUN_NAME}" \
  --out "${MONITOR_LOG}" \
  --interval 5
