#!/usr/bin/env bash
METHOD="sharp_minmax_nomask_alm"
FORGET_LRS=(5e-6 1e-5)
RETAIN_LRS=(5e-6 1e-5)
JOINT_LRS=(1e-5)
FORGET_RHOS=(1e-3 5e-3)
RETAIN_RHOS=(1e-3 5e-3)
TAUS=(1 10 50)
LAMBDA_LRS=(0 1e-3 1e-2)
LAM_INITS=(1)
ALM_RHOS=(1.0)
SEEDS=(0 1)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_grid_common.sh"
