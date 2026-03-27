#!/usr/bin/env bash
METHOD="uam_alm"
FORGET_LRS=(5e-6 1e-5)
RETAIN_LRS=(5e-6 1e-5)
JOINT_LRS=(5e-6)
TAUS=(0.001 0.01 0.1)
LAMBDA_LRS=(1e-4 1e-3 1e-2)
LAM_INITS=(1)
ALM_RHOS=(0.5 1.0 2.0)
UAM_GAMMAS=(1.0 2.0)
UAM_EPSS=(1e-12)
SEEDS=(42)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_grid_common.sh"
