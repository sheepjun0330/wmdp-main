#!/usr/bin/env bash
set -euo pipefail

METHOD="alm_sam_sam_joint2"

# Single-case defaults for a quick manual test.
# Override any of these from the shell if needed, e.g.
# HF_TOKEN=... GPU_ID=5 MAX_NUM_BATCHES=1 bash scripts/test_alm_sam_sam_joint2.sh
FORGET_LRS=("${FORGET_LRS[@]:-1e-5}")
RETAIN_LRS=("${RETAIN_LRS[@]:-1e-5}")
JOINT_LRS=("${JOINT_LRS[@]:-1e-5}")
FORGET_RHOS=("${FORGET_RHOS[@]:-1e-3}")
RETAIN_RHOS=("${RETAIN_RHOS[@]:-1e-3}")
TAUS=("${TAUS[@]:-10}")
LAMBDA_LRS=("${LAMBDA_LRS[@]:-0}")
LAM_INITS=("${LAM_INITS[@]:-1}")
ALM_RHOS=("${ALM_RHOS[@]:-1.0}")
SEEDS=("${SEEDS[@]:-0}")

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_grid_common.sh"
