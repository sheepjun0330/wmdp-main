#!/usr/bin/env bash
set -u

# =========================
# Basic config
# =========================
GPU_ID=7
DEVICE=cuda:0
BATCH_SIZE=32

LM_EVAL_BIN=lm-eval

# =========================
# Evaluation tasks
# =========================
TASKS="mmlu,wmdp"

# =========================
# Model paths to evaluate
# (여러 개 넣어도 됨)
# =========================
MODELS=(
wmdp/models/zephyr_rmu_alm_sam_sam_5e-6
)
# =========================
# Run
# =========================
for MODEL_PATH in "${MODELS[@]}"; do
  EXP_NAME=$(basename "${MODEL_PATH}")
  LOG_FILE="logs/lmeval_${EXP_NAME}.log"

  echo "======================================"
  echo "Evaluating: ${MODEL_PATH}"
  echo "Log file  : ${LOG_FILE}"
  echo "======================================"

  CUDA_VISIBLE_DEVICES=${GPU_ID} \
  ${LM_EVAL_BIN} \
    --model hf \
    --model_args pretrained=${MODEL_PATH},dtype=auto \
    --tasks ${TASKS} \
    --batch_size ${BATCH_SIZE} \
    --device ${DEVICE} \
    > "${LOG_FILE}" 2>&1 || true
done
