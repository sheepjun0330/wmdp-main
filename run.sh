#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Basic config
# =========================================
GPU_ID=0,7
SCRIPT="rmu/unlearn.py"
MODEL="HuggingFaceH4/zephyr-7b-beta"
#HuggingFaceH4/zephyr-7b-beta

RETAIN_CORPORA="wikitext,wikitext"
FORGET_CORPORA="bio-forget-corpus,cyber-forget-corpus"

BATCH_SIZE=4
MAX_NUM_BATCHES=80
LAYER_ID=7
LAYER_IDS="5,6,7"
PARAM_IDS="6"
SEED=42
ALPHA="100,100"

METHOD="alm_sam_sam_joint2"

# =========================================
# Fixed params (except LR)
# =========================================
RHO=5e-3
LAM_INIT=1
ADAM_EPS=1e-12
APPLY_WD_ONCE=1

# =========================================
# Sweep params
# =========================================
LRS=(5e-5)
TAUS=(50)
LAM_LRS=(1)
EPOCH=(1)

# =========================================
# Run
# =========================================
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for LR in "${LRS[@]}"; do
  for TAU in "${TAUS[@]}"; do
    for LAM_LR in "${LAM_LRS[@]}"; do

      DATE_TAG=$(date +"%Y%m%d-%H%M%S")

      OUTPUT_DIR="EVALS/froz/${MODEL##*/}_${METHOD}_lr${LR}_rho${RHO}_tau${TAU}_lam${LAM_INIT}_llr${LAM_LR}_alpha${ALPHA//,/}_layer${LAYER_ID}_b${MAX_NUM_BATCHES}_seed${SEED}_epoch${EPOCH}_revise_alpha"

      echo "================================================="
      echo "METHOD     = ${METHOD}"
      echo "LR         = ${LR}"
      echo "TAU        = ${TAU}"
      echo "LAM_LR     = ${LAM_LR}"
      echo "OUTPUT_DIR = ${OUTPUT_DIR}"
      echo "================================================="

      python3 "${SCRIPT}" \
        --model_name_or_path "${MODEL}" \
        --retain_corpora "${RETAIN_CORPORA}" \
        --forget_corpora "${FORGET_CORPORA}" \
        --dual_mode "${METHOD}" \
        --alpha "${ALPHA}" \
        --batch_size "${BATCH_SIZE}" \
        --max_num_batches "${MAX_NUM_BATCHES}" \
        --layer_id "${LAYER_ID}" \
        --layer_ids "${LAYER_IDS}" \
        --param_ids "${PARAM_IDS}" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_DIR}" \
        \
        --tau "${TAU}" \
        --lagran_lambda_init "${LAM_INIT}" \
        --lagran_lambda_lr "${LAM_LR}" \
        \
        --lr "${LR}" \
        --retain_rho "${RHO}" \
        --forget_rho "${RHO}" \
        \
        --adam_epsilon "${ADAM_EPS}" \
        --epochs "${EPOCH}"
    done
  done
done
