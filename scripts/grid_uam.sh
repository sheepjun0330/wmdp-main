#!/usr/bin/env bash
METHOD="uam_alm"
ALM_ON=0
ALM_OFF_KEEP_RHO=1
ALM_OFF_LAM_INIT=0
FORGET_LRS=(5e-5)
RETAIN_LRS=(5e-5)
JOINT_LRS=(5e-5)
ALM_RHOS=(5e-6 5e-4 5e-3)
UAM_GAMMAS=(2.0)
UAM_EPSS=(1e-12)
SEEDS=(42)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_grid_common.sh"
