#!/bin/bash
# Generic aggregate-only plotting job; never opens forecast or ERA5 stores.
#SBATCH --job-name=heat_verify_plot
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:?Set HEAT_VERIFICATION_RESULTS_ROOT to the configured results root}"
MODEL_NAME="${VERIFICATION_MODEL_NAME:?Set VERIFICATION_MODEL_NAME to model.name}"
FIGURE_DIRECTORY="${VERIFICATION_FIGURE_DIRECTORY:-${RESULT_ROOT}/${MODEL_NAME}/figures}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Configured Python executable is missing: ${PYTHON}" >&2
    exit 2
fi
mkdir -p "${RESULT_ROOT}/logs" "${FIGURE_DIRECTORY}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "Model: ${MODEL_NAME}"
echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Figure output: ${FIGURE_DIRECTORY}"
srun "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/plot_verification.py" \
    --result-dirs "${RESULT_ROOT}/${MODEL_NAME}" \
    --output-directory "${FIGURE_DIRECTORY}" \
    --reliability-forecast-days 0 5 10 13
echo "Finish: $(date --iso-8601=seconds)"
