#!/usr/bin/env bash
# Submit one aligned Nigeria-plus-global report scorecard.
#
# Run from any directory on the cluster with:
#   bash slurm/verification/submit_both_jobs.sh
#
# Optional environment overrides:
#   HEAT_VERIFICATION_RESULTS_ROOT=/path/to/verification_results
#   REPORT_SCORECARD_FORECAST_DAYS="0 3 6 9"
#   REPORT_SCORECARD_MODELS="aifs_ens_v2 aifs_v2 ..."
#   REPORT_SCORECARD_REGIONS="nigeria global"
#   REPORT_SCORECARD_YEARS="2022 2023 2024 2025"
#   REPORT_SCORECARD_MONTHS="6 7 8 9"

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIRECTORY}/submit_report_scorecard.sbatch"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:-/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results}"
# IFS currently has complete scorecard cases through day 9, not day 12.
FORECAST_DAYS_TEXT="${REPORT_SCORECARD_FORECAST_DAYS:-0 3 6 9}"
REGIONS_TEXT="${REPORT_SCORECARD_REGIONS:-nigeria global}"
OUTPUT_DIRECTORY="${REPORT_SCORECARD_OUTPUT_DIRECTORY:-${RESULT_ROOT}/_report_scorecard}"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is unavailable; run this from a Slurm login node." >&2
    exit 2
fi
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
    echo "Report-scorecard Slurm script is missing: ${SBATCH_SCRIPT}" >&2
    exit 2
fi
if [[ ! -d "${RESULT_ROOT}" ]]; then
    echo "Verification result root is missing: ${RESULT_ROOT}" >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}/logs"

export_values="ALL,REPOSITORY_ROOT=${REPOSITORY_ROOT},HEAT_VERIFICATION_RESULTS_ROOT=${RESULT_ROOT}"
export_values+=",REPORT_SCORECARD_REGIONS=${REGIONS_TEXT},REPORT_SCORECARD_OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY}"
export_values+=",REPORT_SCORECARD_FORECAST_DAYS=${FORECAST_DAYS_TEXT}"
job_id="$(sbatch --parsable \
    --export="${export_values}" \
    --output="${RESULT_ROOT}/logs/report_scorecard_%j.out" \
    --error="${RESULT_ROOT}/logs/report_scorecard_%j.err" \
    "${SBATCH_SCRIPT}")"
job_number="${job_id%%;*}"

printf 'Submitted aligned report scorecard: %s\n' "${job_id}"
printf 'Regions: %s\n' "${REGIONS_TEXT}"
printf 'Figure:  %s\n' "${OUTPUT_DIRECTORY}/heat_report_scorecard.png"
printf 'Logs:    %s\n' "${RESULT_ROOT}/logs/report_scorecard_${job_number}.out/.err"
printf 'Monitor: squeue -u "%s" -j %s\n' "${USER}" "${job_number}"
