#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DOMAIN="${DOMAIN:-bio}"

case "${DOMAIN}" in
  bio) DOMAIN_TAG="bio" ;;
  cyber) DOMAIN_TAG="cyber" ;;
  both) DOMAIN_TAG="bio_cyber" ;;
  *)
    echo "[ERROR] DOMAIN must be one of: bio, cyber, both" >&2
    exit 1
    ;;
esac

METHOD="robust_unlearn_sam_alm"
ALM_ON=0
ALM_OFF_LAM_INITS=(1.0)
MAX_NUM_BATCHES=125
FORGET_LRS=(2.5e-6 5e-6 1e-5)
RETAIN_LRS=("${FORGET_LRS[@]}")
LOCK_LRS=1
JOINT_LRS=(5e-6)
FORGET_RHOS=(1e-3 1e-2 1e-1)
RETAIN_RHOS=("${FORGET_RHOS[@]}")
LOCK_RHOS=1
USE_JOINT_LR_AXIS=0
USE_RETAIN_RHO_AXIS=0
USE_WANDB=1
CLIP_GRAD_NORM=0
ALPHA="1,1"
BETAS=(0.01 0.05)
GAMMAS=(1.0 2.5)
FORGET_SCALES=(1.0 2.0)
RETAIN_LAMBDAS=(1.0)
BASE_OUTPUT_DIR="${ROOT_DIR}/models/grid_search_robust_paper_npo_sam_stronger_forget/${DOMAIN_TAG}"
SEEDS=(42)
source "${SCRIPT_DIR}/_grid_common.sh"
