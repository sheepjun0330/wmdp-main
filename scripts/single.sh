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
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-HuggingFaceH4/zephyr-7b-beta}"
MAX_NUM_BATCHES="${MAX_NUM_BATCHES:-150}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEERING_COEFFS="${STEERING_COEFFS:-6.5,6.5}"
ALPHA="${ALPHA:-1200,1200}"
BETA="${BETA:-0.1}"
GAMMA="${GAMMA:-1.0}"
SEED="${SEED:-42}"
DUAL_MODE="${DUAL_MODE:-alm_sam_sam_joint2}"
TAU="${TAU:-0.01}"
LAGRAN_LAMBDA_INIT="${LAGRAN_LAMBDA_INIT:-1.0}"
LAGRAN_LAMBDA_LR="${LAGRAN_LAMBDA_LR:-1e-3}"
FORGET_RHO="${FORGET_RHO:-1e-4}"
RETAIN_RHO="${RETAIN_RHO:-1e-4}"
FORGET_LR="${FORGET_LR:-5e-6}"
RETAIN_LR="${RETAIN_LR:-1e-5}"
JOINT_LR="${JOINT_LR:-5e-6}"
ALM_RHO="${ALM_RHO:-1.0}"
FORGET_SCALE="${FORGET_SCALE:-1.0}"
RETAIN_LAMBDA="${RETAIN_LAMBDA:-2.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
USE_WANDB="${USE_WANDB:-1}"

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

RUN_NAME="${RUN_NAME:-alm_sam_sam_joint2_alm_on_seed42_flr5e-6_rlr1e-5_jlr5e-6_frho1e-4_rrho1e-4_tau0.01_llr1e-3_initL1_almrho1.0_fscale1.0_rlam2.0_wd0.0}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/models/new2/${RUN_NAME}}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${ROOT_DIR}/models/new2/eval_logs}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${ROOT_DIR}/models/new2/eval_results}"
WANDB_PROJECT="${WANDB_PROJECT:-rmu-unlearn}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"

mkdir -p "$(dirname "${OUTPUT_DIR}")" "${EVAL_LOG_DIR}" "${EVAL_RESULTS_DIR}"

cmd=(
  uv run python -m rmu.unlearn
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --max_num_batches "${MAX_NUM_BATCHES}"
  --batch_size "${BATCH_SIZE}"
  --retain_corpora "${RETAIN_CORPORA:-${RETAIN_CORPORA_DEFAULT}}"
  --forget_corpora "${FORGET_CORPORA:-${FORGET_CORPORA_DEFAULT}}"
  --steering_coeffs "${STEERING_COEFFS}"
  --alpha "${ALPHA}"
  --beta "${BETA}"
  --gamma "${GAMMA}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
  --verbose
  --dual_mode "${DUAL_MODE}"
  --tau "${TAU}"
  --lagran_lambda_init "${LAGRAN_LAMBDA_INIT}"
  --lagran_lambda_lr "${LAGRAN_LAMBDA_LR}"
  --forget_rho "${FORGET_RHO}"
  --retain_rho "${RETAIN_RHO}"
)

if [[ -n "${FORGET_LR}" ]]; then
  cmd+=(--forget_lr "${FORGET_LR}")
fi
if [[ -n "${RETAIN_LR}" ]]; then
  cmd+=(--retain_lr "${RETAIN_LR}")
fi
if [[ -n "${JOINT_LR}" ]]; then
  cmd+=(--joint_lr "${JOINT_LR}")
fi
if [[ -n "${ALM_RHO}" ]]; then
  cmd+=(--alm_rho "${ALM_RHO}")
fi
if [[ -n "${FORGET_SCALE}" ]]; then
  cmd+=(--forget_scale "${FORGET_SCALE}")
fi
if [[ -n "${RETAIN_LAMBDA}" ]]; then
  cmd+=(--retain_lambda "${RETAIN_LAMBDA}")
fi
if [[ -n "${WEIGHT_DECAY}" ]]; then
  cmd+=(--weight_decay "${WEIGHT_DECAY}")
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  cmd+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_run_name "${WANDB_RUN_NAME}")
fi

CUDA_VISIBLE_DEVICES="${TRAIN_GPU_ID}" "${cmd[@]}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  GPU_ID="${EVAL_GPU_ID}" \
  DEVICE="${DEVICE}" \
  BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  TASKS="${TASKS}" \
  LOG_DIR="${EVAL_LOG_DIR}" \
  RESULTS_DIR="${EVAL_RESULTS_DIR}" \
  bash "${ROOT_DIR}/eval.sh" "${OUTPUT_DIR}"
fi
