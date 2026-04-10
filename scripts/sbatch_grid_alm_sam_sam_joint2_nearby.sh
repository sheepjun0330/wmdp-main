#!/usr/bin/env bash
#SBATCH -J nearby-joint2
#SBATCH -p RTX6000ADA
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -o logs/slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${ROOT_DIR}/logs"

cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Keep caches on writable storage when the home cache is restricted.
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/.cache/slurm/${SLURM_JOB_ID:-manual}}"
mkdir -p "${CACHE_ROOT}/uv" "${CACHE_ROOT}/hf" "${CACHE_ROOT}/hf-datasets" "${CACHE_ROOT}/transformers"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CACHE_ROOT}/uv}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/hf-datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/transformers}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES}}"
fi
export EVAL_GPU_ID="${EVAL_GPU_ID:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0,1}}}"
export EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"

if [[ -d "${ROOT_DIR}/.venv" ]]; then
  # Optional activation keeps python/pip aligned even when we do not invoke uv run.
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv/bin/activate"
fi

if [[ "${RUN_UV_SYNC:-0}" == "1" ]]; then
  uv sync --locked --no-dev
fi

echo "[INFO] job_id=${SLURM_JOB_ID:-unknown}"
echo "[INFO] partition=${SLURM_JOB_PARTITION:-unknown}"
echo "[INFO] nodelist=${SLURM_JOB_NODELIST:-unknown}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] train_gpu_ids=${GPU_ID:-unset}"
echo "[INFO] eval_gpu_ids=${EVAL_GPU_ID}"
echo "[INFO] uv_cache_dir=${UV_CACHE_DIR}"
echo "[INFO] transformers_cache=${TRANSFORMERS_CACHE}"
echo "[INFO] python=$(command -v python)"
echo "[INFO] uv=$(command -v uv)"

bash "${SCRIPT_DIR}/grid_alm_sam_sam_joint2_nearby.sh"
