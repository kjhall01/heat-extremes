#!/bin/bash
# Submit compute array -> exact aggregation -> aggregate-only figures.
# Run from the repository root or set REPOSITORY_ROOT explicitly.

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SLURM_DIRECTORY="${REPOSITORY_ROOT}/slurm/verification"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:-/net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results}"

if [[ ! -f "${SLURM_DIRECTORY}/submit_aifs_verification_array.sbatch" ]]; then
    echo "Verification Slurm scripts are missing under ${SLURM_DIRECTORY}" >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}/logs"
EXPORT_ARGUMENT="--export=ALL,REPOSITORY_ROOT=${REPOSITORY_ROOT}"

compute_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" "${SLURM_DIRECTORY}/submit_aifs_verification_array.sbatch")"
aggregate_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterok:${compute_job}" "${SLURM_DIRECTORY}/submit_aifs_aggregation.sbatch")"
plot_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterok:${aggregate_job}" "${SLURM_DIRECTORY}/submit_aifs_plotting.sbatch")"

printf 'Submitted AIFS verification array: %s\n' "${compute_job}"
printf 'Submitted exact aggregation: %s (afterok:%s)\n' "${aggregate_job}" "${compute_job}"
printf 'Submitted aggregate-only plotting: %s (afterok:%s)\n' "${plot_job}" "${aggregate_job}"
printf 'Monitor: squeue -u "%s" -j %s,%s,%s\n' "${USER}" "${compute_job}" "${aggregate_job}" "${plot_job}"
