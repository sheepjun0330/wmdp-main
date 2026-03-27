#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TRAIN_GPU_ID="${TRAIN_GPU_ID:-4,5}"
EVAL_GPU_ID="${EVAL_GPU_ID:-4,5}"
DEVICE="${DEVICE:-cuda:0}"
TASKS="${TASKS:-mmlu,wmdp}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
RUN_EVAL="${RUN_EVAL:-1}"
DOMAIN="${DOMAIN:-bio}"

case "${DOMAIN}" in
  bio)
    RETAIN_CORPORA_DEFAULT="wikitext"
    FORGET_CORPORA_DEFAULT="bio-forget-corpus"
    DOMAIN_TAG="bio"
    ;;
  cyber)
    RETAIN_CORPORA_DEFAULT="wikitext"
    FORGET_CORPORA_DEFAULT="cyber-forget-corpus"
    DOMAIN_TAG="cyber"
    ;;
  both)
    RETAIN_CORPORA_DEFAULT="wikitext,wikitext"
    FORGET_CORPORA_DEFAULT="bio-forget-corpus,cyber-forget-corpus"
    DOMAIN_TAG="bio_cyber"
    ;;
  *)
    echo "[ERROR] DOMAIN must be one of: bio, cyber, both" >&2
    exit 1
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/models/new/zephyr_rmu_alm_sam_sam_1e-5_${DOMAIN_TAG}}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${ROOT_DIR}/models/new/eval_logs}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${ROOT_DIR}/models/new/eval_results}"

mkdir -p "$(dirname "${OUTPUT_DIR}")" "${EVAL_LOG_DIR}" "${EVAL_RESULTS_DIR}"

CUDA_VISIBLE_DEVICES="${TRAIN_GPU_ID}" uv run python -m rmu.unlearn \
  --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
  --max_num_batches 150 \
  --batch_size 4 \
  --retain_corpora "${RETAIN_CORPORA:-${RETAIN_CORPORA_DEFAULT}}" \
  --forget_corpora "${FORGET_CORPORA:-${FORGET_CORPORA_DEFAULT}}" \
  --steering_coeffs 6.5,6.5 \
  --alpha 1200,1200 \
  --lr 5e-6 \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --verbose \
  --dual_mode alm_sam_sam_joint2 \
  --tau 0.01 \
  --lagran_lambda_init 1.0 \
  --lagran_lambda_lr 1e-3 \
  --forget_rho 1e-5 \
  --retain_rho 1e-5 \
  --use_wandb \
  --wandb_project rmu-unlearn \
  --wandb_run_name "zephyr_rmu_alm_sam_sam1e-5_${DOMAIN_TAG}_tau0.01"

if [[ "${RUN_EVAL}" == "1" ]]; then
  GPU_ID="${EVAL_GPU_ID}" \
  DEVICE="${DEVICE}" \
  BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  TASKS="${TASKS}" \
  LOG_DIR="${EVAL_LOG_DIR}" \
  RESULTS_DIR="${EVAL_RESULTS_DIR}" \
  bash "${ROOT_DIR}/eval.sh" "${OUTPUT_DIR}"
fi
