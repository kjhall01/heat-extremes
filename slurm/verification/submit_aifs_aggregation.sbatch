#!/bin/bash
#SBATCH --job-name=aifs_verify_aggregate
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results/logs/%x_%j.out
#SBATCH --error=/net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results/logs/%x_%j.err

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
CONFIG="${VERIFICATION_CONFIG:-${REPOSITORY_ROOT}/configs/verification/aifs_ens_v2.yaml}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Configured Python executable is missing: ${PYTHON}" >&2
    exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "Model: aifs_ens_v2"
echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"

srun "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/aggregate_verification.py" --config "${CONFIG}"

echo "Finish: $(date --iso-8601=seconds)"
