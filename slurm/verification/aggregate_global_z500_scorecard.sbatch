#!/bin/bash
# Aggregate global-Z500 partials and join global T2M report-scorecard values.
#SBATCH --job-name=z500_score_aggregate
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
HEAT_SCORECARD="${HEAT_REPORT_SCORECARD:-}"
read -r -a MODELS <<< "${GLOBAL_Z500_MODELS:-aifs_ens_v2 ifs_ens aifs_v2 aurora_e2s graphcast_e2s}"

ARGS=(--output-directory "${OUTPUT_DIRECTORY}" --manifest "${MANIFEST}" --models "${MODELS[@]}")
if [[ -n "${HEAT_SCORECARD}" ]]; then
    ARGS+=(--heat-scorecard "${HEAT_SCORECARD}")
fi
srun "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/aggregate_global_z500_scorecard.py" "${ARGS[@]}"
