#!/bin/bash
# Metadata-only Z500 availability audit.  It must succeed before compute jobs.
#SBATCH --job-name=z500_score_preflight
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
MANIFEST="${REFORECAST_MANIFEST:?Set REFORECAST_MANIFEST to the generated inventory JSON}"
OUTPUT_DIRECTORY="${GLOBAL_Z500_OUTPUT_DIRECTORY:?Set GLOBAL_Z500_OUTPUT_DIRECTORY}"
ERA5_Z500_STORE="${ERA5_Z500_STORE:?Set ERA5_Z500_STORE to a consolidated ERA5 pressure-level Zarr}"
ERA5_Z500_VARIABLE="${ERA5_Z500_VARIABLE:-z}"

read -r -a MODELS <<< "${GLOBAL_Z500_MODELS:-aifs_ens_v2 ifs_ens aifs_v2 aurora_e2s graphcast_e2s}"
read -r -a FORECAST_DAYS <<< "${GLOBAL_Z500_FORECAST_DAYS:-0 3 6 9 12}"
read -r -a LEAD_HOURS <<< "${GLOBAL_Z500_LEAD_HOURS:-0 72 144 216 288}"

mkdir -p "${OUTPUT_DIRECTORY}"
srun "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/preflight_global_z500_scorecard.py" \
    --manifest "${MANIFEST}" --era5-z500-store "${ERA5_Z500_STORE}" \
    --era5-z500-variable "${ERA5_Z500_VARIABLE}" --output "${OUTPUT_DIRECTORY}/z500_preflight.json" \
    --models "${MODELS[@]}" --forecast-days "${FORECAST_DAYS[@]}" --lead-hours "${LEAD_HOURS[@]}"
