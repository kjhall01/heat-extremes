#!/bin/bash
# Generic aggregation job; submit after the configured array succeeds.
#SBATCH --job-name=heat_verify_aggregate
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
CONFIG="${VERIFICATION_CONFIG:?Set VERIFICATION_CONFIG to the model YAML}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
MODEL_NAME="${VERIFICATION_MODEL_NAME:?Set VERIFICATION_MODEL_NAME to model.name}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Configured Python executable is missing: ${PYTHON}" >&2
    exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "Model: ${MODEL_NAME}"
echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
srun "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/aggregate_verification.py" --config "${CONFIG}"
echo "Finish: $(date --iso-8601=seconds)"
