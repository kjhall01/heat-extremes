#!/bin/bash
#SBATCH --job-name=aifs-heat
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --array=0-20%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-47%4
#SBATCH --output=logs/aifs_heat_%A_%a.out
#SBATCH --error=logs/aifs_heat_%A_%a.err


set -euo pipefail

# Map array index 0..47 to 2022-01 .. 2025-12.
YEAR=$((2022 + SLURM_ARRAY_TASK_ID / 12))
MONTH=$((1 + SLURM_ARRAY_TASK_ID % 12))


PYTHON_SCRIPT="/home/kylehall/heat-extremes/scripts/process_aifs_heat_month.py"

PYTHON="${HOME}/miniconda3/envs/heat-extremes/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable not found: ${PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "Processing script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Processing: ${YEAR}-$(printf '%02d' "${MONTH}")"
echo "Node: $(hostname)"
echo "Python: ${PYTHON}"
echo "Script: ${PYTHON_SCRIPT}"
echo "Started: $(date --iso-8601=seconds)"

srun "${PYTHON}" "${PYTHON_SCRIPT}" \
    --year "${YEAR}" \
    --month "${MONTH}" \
    --max-days 15 \
    --zarr-format 3

echo "Finished: $(date --iso-8601=seconds)"