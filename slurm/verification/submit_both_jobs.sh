#!/usr/bin/env bash
# Submit the title-free Nigeria and global report scorecards together.
#
# Run from any directory on the cluster with:
#   bash slurm/verification/submit_both_jobs.sh
#
# Optional environment overrides:
#   HEAT_VERIFICATION_RESULTS_ROOT=/path/to/verification_results
#   REPORT_SCORECARD_FORECAST_DAYS="0 3 6 9"
#   REPORT_SCORECARD_MODELS="aifs_ens_v2 ifs_ens ..."
#   REPORT_SCORECARD_YEARS="2022 2023 2024 2025"
#   REPORT_SCORECARD_MONTHS="6 7 8 9"

set -euo pipefail

SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIRECTORY}/../.." && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIRECTORY}/submit_report_scorecard.sbatch"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:-/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results}"
# IFS currently has complete scorecard cases through day 9, not day 12.
FORECAST_DAYS_TEXT="${REPORT_SCORECARD_FORECAST_DAYS:-0 3 6 9}"

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

submit_scorecard() {
    local region="$1"
    local output_directory="$2"
    local job_id
    local export_values

    export_values="ALL,REPOSITORY_ROOT=${REPOSITORY_ROOT},HEAT_VERIFICATION_RESULTS_ROOT=${RESULT_ROOT}"
    export_values+=",REPORT_SCORECARD_REGIONS=${region},REPORT_SCORECARD_OUTPUT_DIRECTORY=${output_directory}"
    export_values+=",REPORT_SCORECARD_FORECAST_DAYS=${FORECAST_DAYS_TEXT}"
    job_id="$(sbatch --parsable \
        --export="${export_values}" \
        --output="${RESULT_ROOT}/logs/report_scorecard_${region}_%j.out" \
        --error="${RESULT_ROOT}/logs/report_scorecard_${region}_%j.err" \
        "${SBATCH_SCRIPT}")"
    printf '%s\n' "${job_id}"
}

nigeria_job="$(submit_scorecard nigeria "${RESULT_ROOT}/_report_scorecard")"
global_job="$(submit_scorecard global "${RESULT_ROOT}/_report_scorecard_global")"
nigeria_job_number="${nigeria_job%%;*}"
global_job_number="${global_job%%;*}"

printf 'Submitted Nigeria report scorecard: %s\n' "${nigeria_job}"
printf 'Submitted global report scorecard:  %s\n' "${global_job}"
printf 'Nigeria figure: %s\n' "${RESULT_ROOT}/_report_scorecard/heat_report_scorecard.png"
printf 'Global figure:  %s\n' "${RESULT_ROOT}/_report_scorecard_global/heat_report_scorecard.png"
printf 'Nigeria logs:   %s\n' "${RESULT_ROOT}/logs/report_scorecard_nigeria_${nigeria_job_number}.out/.err"
printf 'Global logs:    %s\n' "${RESULT_ROOT}/logs/report_scorecard_global_${global_job_number}.out/.err"
printf 'Monitor: squeue -u "%s" -j %s,%s\n' "${USER}" "${nigeria_job_number}" "${global_job_number}"
