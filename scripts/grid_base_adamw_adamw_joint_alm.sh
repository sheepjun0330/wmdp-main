#!/usr/bin/env bash
METHOD="base_adamw_adamw_joint_alm"
FORGET_LRS=(5e-6 1e-5)
RETAIN_LRS=(5e-6 1e-5)
JOINT_LRS=(5e-6 1e-5)
TAUS=(1 10 50)
LAMBDA_LRS=(0 1e-3)
LAM_INITS=(1)
FORGET_SCALES=(0.5 1.0 2.0)
ALM_RHOS=(1.0)
SEEDS=(0 1)
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_grid_common.sh"
