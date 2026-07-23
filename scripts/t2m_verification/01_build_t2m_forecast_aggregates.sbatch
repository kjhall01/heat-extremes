#!/bin/bash
#SBATCH --job-name=t2m_aifs_daily
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/net/monsoon/kylehall/heat_extremes/t2m_verification/logs/%x_%j.out
#SBATCH --error=/net/monsoon/kylehall/heat_extremes/t2m_verification/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="${PIPELINE_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the t2m_verification directory.}}"
REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/net/monsoon/kylehall/heat_extremes/t2m_verification}"
FORECAST_DAYS="${FORECAST_DAYS:-1 3 5 7 9}"
START_YEAR="${START_YEAR:-2022}"
END_YEAR="${END_YEAR:-2025}"
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
read -r -a forecast_day_args <<< "${FORECAST_DAYS}"
overwrite_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    overwrite_args+=(--overwrite)
fi

python -u "${SCRIPT_DIR}/build_t2m_forecast_aggregates.py" \
    --start-year "${START_YEAR}" \
    --end-year "${END_YEAR}" \
    --forecast-days "${forecast_day_args[@]}" \
    --output "${PIPELINE_ROOT}/aifs_t2m_daily.zarr" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    "${overwrite_args[@]}"
