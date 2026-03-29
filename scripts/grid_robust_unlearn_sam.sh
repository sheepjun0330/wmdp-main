#!/usr/bin/env bash
METHOD="robust_unlearn_sam_alm"
ALM_ON=0
ALM_OFF_LAM_INITS=(1.0 2.5)
FORGET_LRS=(5e-6)
RETAIN_LRS=(2.5e-6 5e-6 1e-5)
JOINT_LRS=(5e-6)
FORGET_RHOS=(1e-3 1e-2 1e-1)
RETAIN_RHOS=("${FORGET_RHOS[@]}")
LOCK_RHOS=1
RETAIN_LAMBDAS=(2.0)
SEEDS=(42)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_grid_common.sh"
