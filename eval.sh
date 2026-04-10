#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

if [[ -n "${GPU_ID:-}" ]]; then
  GPU_ID="${GPU_ID}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_ID="${CUDA_VISIBLE_DEVICES}"
else
  GPU_ID="7"
fi
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TASKS="${TASKS:-mmlu,wmdp}"
LM_EVAL_BIN="${LM_EVAL_BIN:-${ROOT_DIR}/.venv/bin/lm-eval}"
DTYPE="${DTYPE:-auto}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/eval_results}"
SKIP_EXISTING_RESULTS="${SKIP_EXISTING_RESULTS:-1}"

DEFAULT_MODELS=(
  "${ROOT_DIR}/models/zephyr_rmu_alm_sam_sam_5e-6"
)

if [[ ! -x "${LM_EVAL_BIN}" ]]; then
  LM_EVAL_BIN="lm-eval"
fi

MODELS=()

add_model() {
  local model_path="$1"
  if [[ "${model_path}" != /* ]]; then
    model_path="${ROOT_DIR}/${model_path}"
  fi
  MODELS+=("${model_path}")
}

if [[ $# -gt 0 ]]; then
  for model_path in "$@"; do
    add_model "${model_path}"
  done
elif [[ -n "${MODELS_CSV:-}" ]]; then
  IFS=',' read -r -a input_models <<< "${MODELS_CSV}"
  for model_path in "${input_models[@]}"; do
    add_model "${model_path}"
  done
else
  for model_path in "${DEFAULT_MODELS[@]}"; do
    add_model "${model_path}"
  done
fi

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

overall_status=0

for MODEL_PATH in "${MODELS[@]}"; do
  EXP_NAME="$(basename "${MODEL_PATH}")"
  LOG_FILE="${LOG_DIR}/lmeval_${EXP_NAME}.log"
  RESULT_PATH="${RESULTS_DIR}/${EXP_NAME}.json"

  echo "======================================"
  echo "Evaluating: ${MODEL_PATH}"
  echo "GPU ID    : ${GPU_ID}"
  echo "Device    : ${DEVICE}"
  echo "Tasks     : ${TASKS}"
  echo "Log file  : ${LOG_FILE}"
  echo "JSON file : ${RESULT_PATH}"
  echo "======================================"

  if [[ "${SKIP_EXISTING_RESULTS}" == "1" ]] && [[ -s "${RESULT_PATH}" ]]; then
    echo "[INFO] Skip existing eval result: ${RESULT_PATH}"
    continue
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${LM_EVAL_BIN}" \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=${DTYPE}" \
    --tasks "${TASKS}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --output_path "${RESULT_PATH}" \
    > "${LOG_FILE}" 2>&1 || overall_status=$?
done

exit "${overall_status}"
