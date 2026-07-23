#!/bin/bash
#SBATCH --job-name=t2m_summary_plot
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/net/monsoon/kylehall/heat_extremes/t2m_verification/logs/%x_%j.out
#SBATCH --error=/net/monsoon/kylehall/heat_extremes/t2m_verification/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/net/monsoon/kylehall/heat_extremes/t2m_verification}"
CONDA_BASE="${CONDA_BASE:-/home/kylehall/miniconda3}"
CONDA_ENV_NAME="${HEAT_EXTREMES_ENV:-heat-extremes}"

mkdir -p "${PIPELINE_ROOT}/logs"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# Conda's MKL activation hook dereferences unset variables, which conflicts
# with the job script's nounset setting. Restore nounset immediately after.
set +u
conda activate "${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
set -u
cd "${REPOSITORY_ROOT}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
overwrite_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    overwrite_args+=(--overwrite)
fi

python -u "${SCRIPT_DIR}/aggregate_and_plot_t2m_metrics.py" \
    --metric-store "${PIPELINE_ROOT}/t2m_case_metrics.zarr" \
    --output "${PIPELINE_ROOT}/t2m_metric_summary.zarr" \
    --figure-directory "${PIPELINE_ROOT}/figures" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    "${overwrite_args[@]}"
