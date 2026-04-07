#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
METHOD="alm_sam_sam_joint2"
DOMAIN="${DOMAIN:-bio}"

FORGET_LRS=(2.5e-6 5e-6)
RETAIN_LRS=(5e-6 1e-5)
JOINT_LRS=(5e-6)
FORGET_RHOS=(1e-6 1e-5 1e-4)
RETAIN_RHOS=("${FORGET_RHOS[@]}")
LOCK_RHOS=1
TAUS=(0.001 0.01)

case "${DOMAIN}" in
  bio)
    FORGET_LRS=(5e-6 7.5e-6)
    RETAIN_LRS=(5e-6 7.5e-6 1e-5)
    TAUS=(0.001 0.005 0.01)
    LAMBDA_LRS=(1e-4 5e-4 1e-3)
    ;;
  cyber)
    LAMBDA_LRS=(1e-3 1e-2)
    ;;
  both)
    LAMBDA_LRS=(1e-4 1e-3 1e-2)
    ;;
  *)
    echo "[ERROR] DOMAIN must be one of: bio, cyber, both" >&2
    exit 1
    ;;
esac

LAM_INITS=(1)
ALM_RHOS=(1.0)
SEEDS=(42)
BASE_OUTPUT_DIR="${ROOT_DIR}/models/grid_search_detail/${DOMAIN}"
source "${SCRIPT_DIR}/_grid_common.sh"
