#!/bin/bash
#SBATCH --job-name=t2m_case_metrics
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/monsoon/kylehall/heat_extremes/t2m_verification/logs/%x_%j.out
#SBATCH --error=/net/monsoon/kylehall/heat_extremes/t2m_verification/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="${PIPELINE_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:?Submit from the t2m_verification directory.}}"
REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/net/monsoon/kylehall/heat_extremes/t2m_verification}"
COVERAGE_PERCENTILES="${COVERAGE_PERCENTILES:-90}"
THRESHOLDS_CELSIUS="${THRESHOLDS_CELSIUS:-35}"
POE_DECISION_THRESHOLDS="${POE_DECISION_THRESHOLDS:-0.5}"
TIME_BATCH_SIZE="${TIME_BATCH_SIZE:-8}"
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
read -r -a percentile_args <<< "${COVERAGE_PERCENTILES}"
read -r -a threshold_args <<< "${THRESHOLDS_CELSIUS}"
read -r -a decision_threshold_args <<< "${POE_DECISION_THRESHOLDS}"
overwrite_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    overwrite_args+=(--overwrite)
fi

python -u "${SCRIPT_DIR}/calculate_t2m_metrics.py" \
    --model-store "${PIPELINE_ROOT}/aifs_t2m_daily.zarr" \
    --observation-store "${PIPELINE_ROOT}/era5_t2m_daily_matched.zarr" \
    --output "${PIPELINE_ROOT}/t2m_case_metrics.zarr" \
    --coverage-percentiles "${percentile_args[@]}" \
    --thresholds-celsius "${threshold_args[@]}" \
    --poe-decision-thresholds "${decision_threshold_args[@]}" \
    --time-batch-size "${TIME_BATCH_SIZE}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    "${overwrite_args[@]}"
