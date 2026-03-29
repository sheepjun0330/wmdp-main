#!/usr/bin/env bash
METHOD="base_adamw_alm_joint"
FORGET_LRS=(5e-6 1e-5)
RETAIN_LRS=("${FORGET_LRS[@]}")
JOINT_LRS=(5e-6 1e-5)
TAUS=(0.01)
LAMBDA_LRS=(1e-4 1e-3)
LAM_INITS=(1)
ALM_RHOS=(0.5 1.0 2.0)
SEEDS=(42)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_grid_common.sh"
