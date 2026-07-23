#!/bin/bash
#SBATCH --job-name=precip_case_metrics
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/net/monsoon/kylehall/heat_extremes/precipitation_verification/logs/%x_%j.out
#SBATCH --error=/net/monsoon/kylehall/heat_extremes/precipitation_verification/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/net/monsoon/kylehall/heat_extremes/precipitation_verification}"
COVERAGE_PERCENTILES="${COVERAGE_PERCENTILES:-90}"
PRECIP_THRESHOLDS_MILLIMETERS="${PRECIP_THRESHOLDS_MILLIMETERS:-}"
POE_DECISION_THRESHOLDS="${POE_DECISION_THRESHOLDS:-0.5}"
TIME_BATCH_SIZE="${TIME_BATCH_SIZE:-8}"
CONDA_BASE="${CONDA_BASE:-/home/kylehall/miniconda3}"
CONDA_ENV_NAME="${HEAT_EXTREMES_ENV:-heat-extremes}"

mkdir -p "${PIPELINE_ROOT}/logs"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
cd "${REPOSITORY_ROOT}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
read -r -a percentile_args <<< "${COVERAGE_PERCENTILES}"
read -r -a decision_threshold_args <<< "${POE_DECISION_THRESHOLDS}"
threshold_args=()
metric_args=()
if [[ -n "${PRECIP_THRESHOLDS_MILLIMETERS}" ]]; then
    read -r -a threshold_args <<< "${PRECIP_THRESHOLDS_MILLIMETERS}"
    metric_args+=(--thresholds-millimeters "${threshold_args[@]}")
fi
overwrite_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    overwrite_args+=(--overwrite)
fi

python -u "${SCRIPT_DIR}/calculate_precipitation_metrics.py" \
    --model-store "${PIPELINE_ROOT}/aifs_daily_precipitation.zarr" \
    --observation-store "${PIPELINE_ROOT}/era5_daily_precipitation_matched.zarr" \
    --output "${PIPELINE_ROOT}/precipitation_case_metrics.zarr" \
    --coverage-percentiles "${percentile_args[@]}" \
    --poe-decision-thresholds "${decision_threshold_args[@]}" \
    --time-batch-size "${TIME_BATCH_SIZE}" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    "${metric_args[@]}" \
    "${overwrite_args[@]}"
