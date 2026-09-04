#!/bin/bash
# Submit the bounded Z500 global-scorecard chain from an existing inventory.
#
# Usage (after the report heat-scorecard job is submitted or complete):
#   ERA5_Z500_STORE=/net/.../era5_pressure_levels.zarr \
#   bash slurm/verification/submit_global_z500_scorecard.sh

set -eo pipefail

usage() {
    cat <<'EOF'
Usage: submit_global_z500_scorecard.sh [options]

Required environment variable:
  ERA5_Z500_STORE       consolidated Zarr with ERA5 geopotential at 500 hPa

Options:
  --manifest FILE       default: <result-root>/inventory/reforecast_inventory.json
  --result-root DIR     default: heat_extremes_reforecast_verification/verification_results
  --output-directory DIR
                         default: <result-root>/_global_z500_scorecard
  --models "NAME ..."   default: "aifs_ens_v2 ifs_ens aifs_v2 aurora_e2s graphcast_e2s"
  --forecast-days "N ..."
                         default: "0 3 6 9 12"
  --lead-hours "H ..."  instantaneous Z500 hours paired with labels; default: "0 72 144 216 288"
  --max-concurrent N     default: 1
  --heat-scorecard FILE default: <result-root>/_report_scorecard/heat_report_scorecard.csv
EOF
}

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:-/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results}"
MANIFEST=""
OUTPUT_DIRECTORY=""
MODELS_TEXT="aifs_ens_v2 ifs_ens aifs_v2 aurora_e2s graphcast_e2s"
FORECAST_DAYS_TEXT="0 3 6 9 12"
LEAD_HOURS_TEXT="0 72 144 216 288"
MAX_CONCURRENT=1
HEAT_SCORECARD=""

while (( $# )); do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --output-directory) OUTPUT_DIRECTORY="$2"; shift 2 ;;
        --models) MODELS_TEXT="$2"; shift 2 ;;
        --forecast-days) FORECAST_DAYS_TEXT="$2"; shift 2 ;;
        --lead-hours) LEAD_HOURS_TEXT="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --heat-scorecard) HEAT_SCORECARD="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

: "${ERA5_Z500_STORE:?Set ERA5_Z500_STORE to a consolidated ERA5 pressure-level Zarr}"
if [[ ! -f "${REPOSITORY_ROOT}/slurm/verification/preflight_global_z500_scorecard.sbatch" ]]; then
    echo "Run from the repository root or set REPOSITORY_ROOT" >&2
    exit 2
fi
MANIFEST="${MANIFEST:-${RESULT_ROOT}/inventory/reforecast_inventory.json}"
OUTPUT_DIRECTORY="${OUTPUT_DIRECTORY:-${RESULT_ROOT}/_global_z500_scorecard}"
HEAT_SCORECARD="${HEAT_SCORECARD:-${RESULT_ROOT}/_report_scorecard/heat_report_scorecard.csv}"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Inventory manifest is missing: ${MANIFEST}" >&2
    exit 2
fi
if (( MAX_CONCURRENT < 1 )); then
    echo "--max-concurrent must be positive" >&2
    exit 2
fi

PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
TASK_COUNT="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["task_count"])' "${MANIFEST}")"
if (( TASK_COUNT == 0 )); then
    echo "Inventory has no tasks: ${MANIFEST}" >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}/logs" "${OUTPUT_DIRECTORY}"
EXPORT_ARGUMENT="--export=ALL,REPOSITORY_ROOT=${REPOSITORY_ROOT},REFORECAST_MANIFEST=${MANIFEST},GLOBAL_Z500_OUTPUT_DIRECTORY=${OUTPUT_DIRECTORY},GLOBAL_Z500_MODELS=${MODELS_TEXT},GLOBAL_Z500_FORECAST_DAYS=${FORECAST_DAYS_TEXT},GLOBAL_Z500_LEAD_HOURS=${LEAD_HOURS_TEXT},HEAT_REPORT_SCORECARD=${HEAT_SCORECARD}"
preflight_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" \
    --output="${RESULT_ROOT}/logs/z500_preflight_%j.out" --error="${RESULT_ROOT}/logs/z500_preflight_%j.err" \
    "${REPOSITORY_ROOT}/slurm/verification/preflight_global_z500_scorecard.sbatch")"
compute_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterok:${preflight_job}" \
    --array="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}" \
    --output="${RESULT_ROOT}/logs/z500_compute_%A_%a.out" --error="${RESULT_ROOT}/logs/z500_compute_%A_%a.err" \
    "${REPOSITORY_ROOT}/slurm/verification/compute_global_z500_scorecard.sbatch")"
aggregate_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterany:${compute_job}" \
    --output="${RESULT_ROOT}/logs/z500_aggregate_%j.out" --error="${RESULT_ROOT}/logs/z500_aggregate_%j.err" \
    "${REPOSITORY_ROOT}/slurm/verification/aggregate_global_z500_scorecard.sbatch")"

printf 'Submitted Z500 preflight: %s\n' "${preflight_job}"
printf 'Submitted Z500 model/month array: %s (afterok:%s)\n' "${compute_job}" "${preflight_job}"
printf 'Submitted Z500 aggregation: %s (afterany:%s)\n' "${aggregate_job}" "${compute_job}"
