#!/usr/bin/env bash
METHOD="sharp_minmax_nomask_alm"
ALM_ON=0
ALM_OFF_LAM_INITS=(2.0 4.0 8.0 16.0 32.0)
FORGET_LRS=(5e-6)
RETAIN_LRS=(5e-6)
JOINT_LRS=(1e-2 3e-2 1e-1)
FORGET_RHOS=(1e-4 1e-3)
RETAIN_RHOS=(1e-4 1e-3)
SEEDS=(42)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_grid_common.sh"
