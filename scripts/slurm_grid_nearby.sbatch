#!/usr/bin/env bash
#SBATCH -J grid-nearby
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TARGET_SCRIPT="${1:-${SCRIPT_DIR}/grid_alm_sam_sam_joint2_nearby.sh}"
shift || true

if [[ "${TARGET_SCRIPT}" != /* ]]; then
  TARGET_SCRIPT="${ROOT_DIR}/${TARGET_SCRIPT}"
fi

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "[ERROR] Target script not found: ${TARGET_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/logs"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES}}"
fi
export EVAL_GPU_ID="${EVAL_GPU_ID:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0,1}}}"
export EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"

echo "[INFO] Slurm job id      : ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Slurm job name    : ${SLURM_JOB_NAME:-unknown}"
echo "[INFO] Slurm nodelist    : ${SLURM_JOB_NODELIST:-unknown}"
echo "[INFO] CUDA devices      : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] Training GPU IDs  : ${GPU_ID:-unset}"
echo "[INFO] Eval GPU IDs      : ${EVAL_GPU_ID}"
echo "[INFO] Target script     : ${TARGET_SCRIPT}"

bash "${TARGET_SCRIPT}" "$@"
