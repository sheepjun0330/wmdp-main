#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${METHOD:-}" ]]; then
  echo "[ERROR] METHOD must be set before sourcing _grid_common.sh" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

MODEL="${MODEL:-HuggingFaceH4/zephyr-7b-beta}"
GPU_ID="${GPU_ID:-0}"
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

RETAIN_CORPORA="${RETAIN_CORPORA:-${RETAIN_CORPORA_DEFAULT}}"
FORGET_CORPORA="${FORGET_CORPORA:-${FORGET_CORPORA_DEFAULT}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NUM_BATCHES="${MAX_NUM_BATCHES:-150}"
EPOCHS="${EPOCHS:-1}"
LAYER_ID="${LAYER_ID:-7}"
LAYER_IDS="${LAYER_IDS:-5,6,7}"
PARAM_IDS="${PARAM_IDS:-6}"
ALPHA="${ALPHA:-1200,1200}"
BETA="${BETA:-0.1}"
GAMMA="${GAMMA:-1.0}"
STEERING_COEFFS="${STEERING_COEFFS:-6.5,6.5}"
VERBOSE_FLAG="${VERBOSE_FLAG:-0}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-rmu-unlearn}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${ROOT_DIR}/models/grid_search/${DOMAIN_TAG}}"
ALM_ON="${ALM_ON:-1}"
ALM_OFF_LAM_INIT="${ALM_OFF_LAM_INIT:-1}"
ALM_OFF_LAM_INITS=("${ALM_OFF_LAM_INITS[@]:-${ALM_OFF_LAM_INIT}}")
ALM_OFF_KEEP_RHO="${ALM_OFF_KEEP_RHO:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${ROOT_DIR}/eval.sh}"
EVAL_GPU_ID="${EVAL_GPU_ID:-${GPU_ID}}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
EVAL_TASKS="${EVAL_TASKS:-mmlu,wmdp}"
DELETE_MODEL_AFTER_EVAL="${DELETE_MODEL_AFTER_EVAL:-1}"
EVAL_LOG_DIRNAME="${EVAL_LOG_DIRNAME:-eval_logs}"
EVAL_RESULTS_DIRNAME="${EVAL_RESULTS_DIRNAME:-eval_results}"
SKIP_EXISTING_RESULTS="${SKIP_EXISTING_RESULTS:-1}"
SKIP_EXISTING_MODELS="${SKIP_EXISTING_MODELS:-1}"
EVAL_EVERY_N="${EVAL_EVERY_N:-1}"

FORGET_LRS=("${FORGET_LRS[@]:-5e-6}")
RETAIN_LRS=("${RETAIN_LRS[@]:-5e-6}")
JOINT_LRS=("${JOINT_LRS[@]:-5e-6}")
BETAS=("${BETAS[@]:-${BETA}}")
GAMMAS=("${GAMMAS[@]:-${GAMMA}}")
FORGET_RHOS=("${FORGET_RHOS[@]:-1e-5}")
RETAIN_RHOS=("${RETAIN_RHOS[@]:-1e-5}")
TAUS=("${TAUS[@]:-0.01}")
LAMBDA_LRS=("${LAMBDA_LRS[@]:-1e-3}")
LAM_INITS=("${LAM_INITS[@]:-1}")
ALM_RHOS=("${ALM_RHOS[@]:-1.0}")
FORGET_SCALES=("${FORGET_SCALES[@]:-1.0}")
RETAIN_LAMBDAS=("${RETAIN_LAMBDAS[@]:-2.0}")
WEIGHT_DECAYS=("${WEIGHT_DECAYS[@]:-0.0}")
SEEDS=("${SEEDS[@]:-42}")
LOCK_RHOS="${LOCK_RHOS:-0}"
LOCK_LRS="${LOCK_LRS:-0}"
USE_JOINT_LR_AXIS="${USE_JOINT_LR_AXIS:-1}"
USE_RETAIN_RHO_AXIS="${USE_RETAIN_RHO_AXIS:-1}"
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-1.0}"

if [[ "${ALM_ON}" == "0" ]]; then
  TAUS=(0)
  LAMBDA_LRS=(0)
  LAM_INITS=("${ALM_OFF_LAM_INITS[@]}")
  if [[ "${ALM_OFF_KEEP_RHO}" != "1" ]]; then
    ALM_RHOS=(0.0)
  fi
fi

if [[ "${ALM_ON}" == "1" ]]; then
  ALM_TAG="alm_on"
  ALM_DIR="ALM_ON"
else
  if [[ "${#ALM_OFF_LAM_INITS[@]}" -eq 1 ]]; then
    ALM_OFF_LAM_TAG="${ALM_OFF_LAM_INITS[0]//./p}"
    ALM_TAG="alm_off_fix${ALM_OFF_LAM_TAG}"
  else
    ALM_TAG="alm_off_sweep"
  fi
  ALM_DIR="ALM_OFF"
fi

mkdir -p "${BASE_OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

is_nonempty_file() {
  local path="$1"
  [[ -s "${path}" ]]
}

has_saved_model() {
  local model_dir="$1"

  [[ -f "${model_dir}/config.json" ]] || return 1
  [[ -n "$(find "${model_dir}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit 2>/dev/null)" ]]
}

slugify_name_value() {
  local value="$1"
  value="${value//,/x}"
  value="${value// /}"
  printf '%s' "${value}"
}

append_run_part() {
  local label="$1"
  local value="$2"
  RUN_PARTS+=("${label}$(slugify_name_value "${value}")")
}

if ! [[ "${EVAL_EVERY_N}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] EVAL_EVERY_N must be a positive integer" >&2
  exit 1
fi

PENDING_EVAL_MODEL_DIRS=()
PENDING_EVAL_RUN_NAMES=()
PENDING_EVAL_RESULT_FILES=()

queue_eval_run() {
  local run_name="$1"
  local model_dir="$2"
  local result_file="$3"

  PENDING_EVAL_RUN_NAMES+=("${run_name}")
  PENDING_EVAL_MODEL_DIRS+=("${model_dir}")
  PENDING_EVAL_RESULT_FILES+=("${result_file}")
  echo "[INFO] Queued for eval (${#PENDING_EVAL_MODEL_DIRS[@]}/${EVAL_EVERY_N}): ${run_name}"
}

run_pending_evals() {
  local pending_count="${#PENDING_EVAL_MODEL_DIRS[@]}"
  local eval_status=0

  if [[ "${RUN_EVAL}" != "1" || "${pending_count}" -eq 0 ]]; then
    return
  fi

  echo "[INFO] Running eval batch for ${pending_count} model(s)"
  if GPU_ID="${EVAL_GPU_ID}" \
    DEVICE="${EVAL_DEVICE}" \
    BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    TASKS="${EVAL_TASKS}" \
    LOG_DIR="${EVAL_LOG_DIR}" \
    RESULTS_DIR="${EVAL_RESULTS_DIR}" \
    bash "${EVAL_SCRIPT}" "${PENDING_EVAL_MODEL_DIRS[@]}"; then
    eval_status=0
  else
    eval_status=$?
  fi

  if [[ ${eval_status} -ne 0 ]]; then
    echo "[WARN] Eval batch failed with status ${eval_status} (continue)"
  fi

  for idx in "${!PENDING_EVAL_MODEL_DIRS[@]}"; do
    local run_name="${PENDING_EVAL_RUN_NAMES[idx]}"
    local model_dir="${PENDING_EVAL_MODEL_DIRS[idx]}"
    local result_file="${PENDING_EVAL_RESULT_FILES[idx]}"

    if is_nonempty_file "${result_file}"; then
      echo "[INFO] Eval completed: ${run_name}"
      if [[ "${DELETE_MODEL_AFTER_EVAL}" == "1" ]]; then
        rm -rf "${model_dir}"
        echo "[INFO] Deleted model dir after eval: ${model_dir}"
      fi
    else
      echo "[WARN] Eval result missing: ${run_name} (model kept)"
    fi
  done

  PENDING_EVAL_MODEL_DIRS=()
  PENDING_EVAL_RUN_NAMES=()
  PENDING_EVAL_RESULT_FILES=()
}

for seed in "${SEEDS[@]}"; do
  for f_lr in "${FORGET_LRS[@]}"; do
    if [[ "${LOCK_LRS}" == "1" ]]; then
      r_lr_values=("${f_lr}")
    else
      r_lr_values=("${RETAIN_LRS[@]}")
    fi
    for r_lr in "${r_lr_values[@]}"; do
      if [[ "${USE_JOINT_LR_AXIS}" == "1" ]]; then
        j_lr_values=("${JOINT_LRS[@]}")
      else
        j_lr_values=("${JOINT_LRS[0]}")
      fi
      for j_lr in "${j_lr_values[@]}"; do
        for tau in "${TAUS[@]}"; do
          for f_rho in "${FORGET_RHOS[@]}"; do
            if [[ "${USE_RETAIN_RHO_AXIS}" != "1" ]]; then
              r_rho_values=("${RETAIN_RHOS[0]}")
            elif [[ "${LOCK_RHOS}" == "1" ]]; then
              r_rho_values=("${f_rho}")
            else
              r_rho_values=("${RETAIN_RHOS[@]}")
            fi
            for r_rho in "${r_rho_values[@]}"; do
              for lam_lr in "${LAMBDA_LRS[@]}"; do
                for lam_init in "${LAM_INITS[@]}"; do
                  for alm_rho in "${ALM_RHOS[@]}"; do
                    for forget_scale in "${FORGET_SCALES[@]}"; do
                      for retain_lambda in "${RETAIN_LAMBDAS[@]}"; do
                        for gamma in "${GAMMAS[@]}"; do
                          for beta in "${BETAS[@]}"; do
                            for wd in "${WEIGHT_DECAYS[@]}"; do
                              RUN_PARTS=("${METHOD}" "${ALM_TAG}" "seed${seed}")
                              append_run_part "alpha" "${ALPHA}"
                              append_run_part "sc" "${STEERING_COEFFS}"
                              append_run_part "flr" "${f_lr}"
                              append_run_part "rlr" "${r_lr}"

                              if [[ "${USE_JOINT_LR_AXIS}" == "1" || "${j_lr}" != "${r_lr}" ]]; then
                                append_run_part "jlr" "${j_lr}"
                              fi

                              append_run_part "frho" "${f_rho}"

                              if [[ "${r_rho}" != "${f_rho}" ]]; then
                                append_run_part "rrho" "${r_rho}"
                              fi

                              if [[ "${ALM_ON}" == "1" ]]; then
                                append_run_part "tau" "${tau}"
                                append_run_part "llr" "${lam_lr}"
                                append_run_part "initL" "${lam_init}"
                                append_run_part "almrho" "${alm_rho}"
                              fi

                              if [[ "${#BETAS[@]}" -gt 1 || "${beta}" != "0.1" ]]; then
                                append_run_part "beta" "${beta}"
                              fi

                              if [[ "${#GAMMAS[@]}" -gt 1 || "${gamma}" != "1.0" ]]; then
                                append_run_part "gamma" "${gamma}"
                              fi

                              if [[ "${#FORGET_SCALES[@]}" -gt 1 || "${forget_scale}" != "1.0" ]]; then
                                append_run_part "fscale" "${forget_scale}"
                              fi

                              if [[ "${#RETAIN_LAMBDAS[@]}" -gt 1 || "${retain_lambda}" != "2.0" ]]; then
                                append_run_part "rlam" "${retain_lambda}"
                              fi

                              if [[ "${#WEIGHT_DECAYS[@]}" -gt 1 || "${wd}" != "0.0" ]]; then
                                append_run_part "wd" "${wd}"
                              fi

                              RUN_NAME="$(IFS=_; printf '%s' "${RUN_PARTS[*]}")"
                                METHOD_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${METHOD}"
                                ALM_OUTPUT_DIR="${METHOD_OUTPUT_DIR}/${ALM_DIR}"
                                MODEL_OUTPUT_DIR="${ALM_OUTPUT_DIR}/${RUN_NAME}"
                                EVAL_LOG_DIR="${ALM_OUTPUT_DIR}/${EVAL_LOG_DIRNAME}"
                                EVAL_RESULTS_DIR="${ALM_OUTPUT_DIR}/${EVAL_RESULTS_DIRNAME}"
                                EVAL_LOG_FILE="${EVAL_LOG_DIR}/lmeval_${RUN_NAME}.log"
                                EVAL_RESULT_FILE="${EVAL_RESULTS_DIR}/${RUN_NAME}.json"

                                mkdir -p "${ALM_OUTPUT_DIR}" "${EVAL_LOG_DIR}" "${EVAL_RESULTS_DIR}"

                                echo "================================================="
                                echo "RUN ${RUN_NAME}"
                                echo "METHOD     = ${METHOD}"
                                echo "ALM_ON     = ${ALM_ON}"
                                echo "ALM_DIR    = ${ALM_DIR}"
                                if [[ "${ALM_ON}" == "0" ]]; then
                                  echo "ALM_MODE   = fixed_forget_coeff"
                                  echo "FIXED_COEF = ${ALM_OFF_LAM_INIT}"
                                fi
                                echo "GPU        = ${GPU_ID}"
                                echo "MODEL_DIR  = ${MODEL_OUTPUT_DIR}"
                                echo "EVAL_LOG   = ${EVAL_LOG_FILE}"
                                echo "EVAL_JSON  = ${EVAL_RESULT_FILE}"
                                echo "CLIP_NORM  = ${CLIP_GRAD_NORM}"
                                echo "BETA       = ${beta}"
                                echo "GAMMA      = ${gamma}"
                                echo "================================================="

                                if [[ "${SKIP_EXISTING_RESULTS}" == "1" ]] && is_nonempty_file "${EVAL_RESULT_FILE}"; then
                                  echo "[INFO] Skip completed run: ${RUN_NAME}"
                                  continue
                                fi

                                should_run_train=1
                                if [[ "${SKIP_EXISTING_MODELS}" == "1" ]] && has_saved_model "${MODEL_OUTPUT_DIR}"; then
                                  should_run_train=0
                                  echo "[INFO] Reusing existing model dir: ${MODEL_OUTPUT_DIR}"
                                fi

                                cmd=(
                                  "${PYTHON_BIN}" -m rmu.unlearn
                                  --model_name_or_path "${MODEL}"
                                  --retain_corpora "${RETAIN_CORPORA}"
                                  --forget_corpora "${FORGET_CORPORA}"
                                  --dual_mode "${METHOD}"
                                  --alpha "${ALPHA}"
                                  --beta "${beta}"
                                  --gamma "${gamma}"
                                  --steering_coeffs "${STEERING_COEFFS}"
                                  --batch_size "${BATCH_SIZE}"
                                  --max_num_batches "${MAX_NUM_BATCHES}"
                                  --epochs "${EPOCHS}"
                                  --layer_id "${LAYER_ID}"
                                  --layer_ids "${LAYER_IDS}"
                                  --param_ids "${PARAM_IDS}"
                                  --seed "${seed}"
                                  --output_dir "${MODEL_OUTPUT_DIR}"
                                  --tau "${tau}"
                                  --lagran_lambda_init "${lam_init}"
                                  --lagran_lambda_lr "${lam_lr}"
                                  --forget_lr "${f_lr}"
                                  --retain_lr "${r_lr}"
                                  --joint_lr "${j_lr}"
                                  --forget_rho "${f_rho}"
                                  --retain_rho "${r_rho}"
                                  --alm_rho "${alm_rho}"
                                  --forget_scale "${forget_scale}"
                                  --retain_lambda "${retain_lambda}"
                                  --weight_decay "${wd}"
                                  --clip_grad_norm "${CLIP_GRAD_NORM}"
                                )

                                if [[ "${VERBOSE_FLAG}" == "1" ]]; then
                                  cmd+=(--verbose)
                                fi
                                if [[ "${USE_WANDB}" == "1" ]]; then
                                  cmd+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_run_name "${RUN_NAME}")
                                fi

                                train_status=0
                                if [[ "${should_run_train}" == "1" ]]; then
                                  if "${cmd[@]}"; then
                                    train_status=0
                                  else
                                    train_status=$?
                                  fi
                                fi

                                if [[ ${train_status} -eq 0 ]]; then
                                  if [[ "${RUN_EVAL}" == "1" ]]; then
                                    queue_eval_run "${RUN_NAME}" "${MODEL_OUTPUT_DIR}" "${EVAL_RESULT_FILE}"
                                    if [[ "${#PENDING_EVAL_MODEL_DIRS[@]}" -ge "${EVAL_EVERY_N}" ]]; then
                                      run_pending_evals
                                    fi
                                  fi
                                else
                                  echo "[WARN] Failed: ${RUN_NAME} (continue)"
                                fi
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

run_pending_evals
