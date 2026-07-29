#!/bin/bash
# Metadata-only historical-archive preflight for model daily-temperature q95.
# No forecast-temperature chunks are read by this job.
#SBATCH --job-name=model_clim_preflight
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:-/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results}"

if [[ ! -x "${PYTHON}" || ! -f "${REPOSITORY_ROOT}/scripts/verification/preflight_model_temperature_climatology.py" ]]; then
    echo "Python executable or climatology preflight script is unavailable" >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Results root: ${RESULT_ROOT}"

srun --cpu-bind=none "${PYTHON}" -u \
    "${REPOSITORY_ROOT}/scripts/verification/preflight_model_temperature_climatology.py" \
    --repository-root "${REPOSITORY_ROOT}" \
    --results-root "${RESULT_ROOT}" \
    "$@"

echo "Finish: $(date --iso-8601=seconds)"
