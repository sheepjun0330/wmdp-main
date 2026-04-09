#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

METHOD="alm_sam_sam_joint2"
DOMAIN="${DOMAIN:-both}"

# Keep the search centered around the current "both" setup.
FORGET_LRS_STR="${FORGET_LRS_STR:-2.5e-6 5e-6 7.5e-6}"
RETAIN_LRS_STR="${RETAIN_LRS_STR:-5e-5 1e-4}"
ALPHAS_STR="${ALPHAS_STR:-800,800}"
STEERING_GRID_STR="${STEERING_GRID_STR:-5.5,5.5 10,10}"
TAUS_STR="${TAUS_STR:-0.01 0.05 0.1}"
SEEDS_STR="${SEEDS_STR:-42}"
EVAL_EVERY_N="${EVAL_EVERY_N:-2}"

read -r -a FORGET_LRS_GRID <<< "${FORGET_LRS_STR}"
read -r -a RETAIN_LRS_GRID <<< "${RETAIN_LRS_STR}"
read -r -a ALPHA_GRID <<< "${ALPHAS_STR}"
read -r -a STEERING_GRID <<< "${STEERING_GRID_STR}"
read -r -a TAU_GRID <<< "${TAUS_STR}"
read -r -a SEEDS <<< "${SEEDS_STR}"

FORGET_RHOS=(1e-4)
RETAIN_RHOS=(1e-4)
LOCK_RHOS=1
LAMBDA_LRS=(1e-3)
LAM_INITS=(1)
ALM_RHOS=(1.0)
BETAS=(0.1)
GAMMAS=(1.0)
FORGET_SCALES=(1.0)
RETAIN_LAMBDAS=(2.0)
WEIGHT_DECAYS=(0.0)

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${ROOT_DIR}/models/grid_search_nearby2/${DOMAIN}}"

echo "[INFO] joint2 nearby sweep"
echo "[INFO] DOMAIN=${DOMAIN}"
echo "[INFO] FORGET_LRS=${FORGET_LRS_STR}"
echo "[INFO] RETAIN_LRS=${RETAIN_LRS_STR}"
echo "[INFO] ALPHAS=${ALPHAS_STR}"
echo "[INFO] STEERING_GRID=${STEERING_GRID_STR}"
echo "[INFO] TAUS=${TAUS_STR}"
echo "[INFO] SEEDS=${SEEDS_STR}"
echo "[INFO] EVAL_EVERY_N=${EVAL_EVERY_N}"

for alpha in "${ALPHA_GRID[@]}"; do
  for steering in "${STEERING_GRID[@]}"; do
    for tau in "${TAU_GRID[@]}"; do
      for forget_lr in "${FORGET_LRS_GRID[@]}"; do
        for retain_lr in "${RETAIN_LRS_GRID[@]}"; do
          echo "[INFO] Launch alpha=${alpha} steering=${steering} tau=${tau} forget_lr=${forget_lr} retain_lr=${retain_lr}"
          (
            ALPHA="${alpha}"
            STEERING_COEFFS="${steering}"
            TAUS=("${tau}")
            FORGET_LRS=("${forget_lr}")
            RETAIN_LRS=("${retain_lr}")
            JOINT_LRS=("${retain_lr}")
            USE_JOINT_LR_AXIS=0
            source "${SCRIPT_DIR}/_grid_common.sh"
          )
        done
      done
    done
  done
done
