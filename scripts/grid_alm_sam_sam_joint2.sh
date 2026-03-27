#!/usr/bin/env bash
METHOD="alm_sam_sam_joint2"
FORGET_LRS=(5e-6 1e-5)
RETAIN_LRS=(5e-6 1e-5)
JOINT_LRS=(5e-6)
FORGET_RHOS=(1e-6 1e-5 1e-4)
RETAIN_RHOS=(1e-6 1e-5 1e-4)
TAUS=(0.001 0.01 0.1)
LAMBDA_LRS=(1e-4 1e-3 1e-2)
LAM_INITS=(1)
ALM_RHOS=(1.0)
SEEDS=(42)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_grid_common.sh"
