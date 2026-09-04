#!/bin/bash
# One model/month global-Z500 RMSE partial from registered raw reforecast data.
#SBATCH --job-name=z500_score_task
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
MANIFEST="${REFORECAST_MANIFEST:?Set REFORECAST_MANIFEST to the generated inventory JSON}"
OUTPUT_DIRECTORY="${GLOBAL_Z500_OUTPUT_DIRECTORY:?Set GLOBAL_Z500_OUTPUT_DIRECTORY}"
ERA5_Z500_STORE="${ERA5_Z500_STORE:?Set ERA5_Z500_STORE to a consolidated ERA5 pressure-level Zarr}"
ERA5_Z500_VARIABLE="${ERA5_Z500_VARIABLE:-z}"

read -r -a FORECAST_DAYS <<< "${GLOBAL_Z500_FORECAST_DAYS:-0 3 6 9 12}"
read -r -a LEAD_HOURS <<< "${GLOBAL_Z500_LEAD_HOURS:-0 72 144 216 288}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "Task ${SLURM_ARRAY_TASK_ID}; start $(date --iso-8601=seconds); host $(hostname)"
srun --cpu-bind=none "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/compute_global_z500_task.py" \
    --manifest "${MANIFEST}" --task-index "${SLURM_ARRAY_TASK_ID}" \
    --era5-z500-store "${ERA5_Z500_STORE}" --era5-z500-variable "${ERA5_Z500_VARIABLE}" \
    --output-directory "${OUTPUT_DIRECTORY}" \
    --preflight "${OUTPUT_DIRECTORY}/z500_preflight.json" \
    --forecast-days "${FORECAST_DAYS[@]}" --lead-hours "${LEAD_HOURS[@]}"
