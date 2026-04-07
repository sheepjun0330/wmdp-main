#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
METHOD="base_adamw_alm_joint"
DOMAIN="${DOMAIN:-bio}"

ALM_ON=0
ALM_OFF_LAM_INITS=(0)

FORGET_LRS=(5e-6 7.5e-6)
RETAIN_LRS=(5e-6 7.5e-6)
LOCK_LRS=0
JOINT_LRS=(5e-6)
USE_JOINT_LR_AXIS=0

FORGET_RHOS=(1e-5)
RETAIN_RHOS=(1e-5)
USE_RETAIN_RHO_AXIS=0

case "${DOMAIN}" in
  bio)
    BASE_OUTPUT_DIR="${ROOT_DIR}/models/grid_search_detail/${DOMAIN}"
    ;;
  cyber|both)
    BASE_OUTPUT_DIR="${ROOT_DIR}/models/grid_search_detail/${DOMAIN}"
    ;;
  *)
    echo "[ERROR] DOMAIN must be one of: bio, cyber, both" >&2
    exit 1
    ;;
esac

SEEDS=(42)
source "${SCRIPT_DIR}/_grid_common.sh"
