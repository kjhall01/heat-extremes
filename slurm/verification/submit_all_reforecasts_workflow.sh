#!/bin/bash
# Inventory standard reforecast directories, then submit independent
# model/month case-cache tasks, cache-backed metric tasks, then tolerant
# all-model aggregation and plotting.

set -eo pipefail

usage() {
    cat <<'EOF'
Usage: submit_all_reforecasts_workflow.sh [options]

Options:
  --reforecast-root DIRECTORY   default: /net/monsoon/marchakitus/reforecast
  --metadata-csv FILE           default: Rossby Model Storage Locations - Sheet1.csv in repository root
  --result-root DIRECTORY       default: /net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results
  --models "NAME ..."          optionally run only named normalized/display models
  --manifest FILE               inventory manifest path; use a model-specific path for an isolated run
  --config-directory DIRECTORY  generated model-config directory; defaults beside the manifest
  --years "YYYY ..."            default: "2022 2023 2024 2025"
  --months "MM ..."             default: "6 7 8 9"
  --max-forecast-day N          inclusive zero-based lead cap; default: 14
  --max-concurrent N            default: 1
  --regions "NAME ..."          score this region subset after case caching (default: all configured)
  --probability-bins "P ..."    reliability-bin edges (default: config values)
  --decision-thresholds "P ..." probability cutoffs for POD/FAR (default: config value, 0.5)
  --overwrite                   replace only configured partial result partitions
  --inventory-only              write/report inventory but do not submit jobs
EOF
}

REFORECAST_ROOT="/net/monsoon/marchakitus/reforecast"
METADATA_CSV=""
RESULT_ROOT="/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results"
MODELS_TEXT=""
MANIFEST=""
CONFIG_DIRECTORY=""
YEARS_TEXT="2022 2023 2024 2025"
MONTHS_TEXT="6 7 8 9"
MAX_FORECAST_DAY=14
MAX_CONCURRENT=1
REGIONS_TEXT=""
PROBABILITY_BINS_TEXT=""
DECISION_THRESHOLDS_TEXT=""
OVERWRITE_VALUE=0
INVENTORY_ONLY=0
while (( $# )); do
    case "$1" in
        --reforecast-root) REFORECAST_ROOT="$2"; shift 2 ;;
        --metadata-csv) METADATA_CSV="$2"; shift 2 ;;
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --models) MODELS_TEXT="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --config-directory) CONFIG_DIRECTORY="$2"; shift 2 ;;
        --years) YEARS_TEXT="$2"; shift 2 ;;
        --months) MONTHS_TEXT="$2"; shift 2 ;;
        --max-forecast-day) MAX_FORECAST_DAY="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --regions) REGIONS_TEXT="$2"; shift 2 ;;
        --probability-bins) PROBABILITY_BINS_TEXT="$2"; shift 2 ;;
        --decision-thresholds) DECISION_THRESHOLDS_TEXT="$2"; shift 2 ;;
        --overwrite) OVERWRITE_VALUE=1; shift ;;
        --inventory-only) INVENTORY_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
SLURM_DIRECTORY="${REPOSITORY_ROOT}/slurm/verification"
if [[ ! -x "${PYTHON}" || ! -d "${REFORECAST_ROOT}" || ! -f "${SLURM_DIRECTORY}/submit_reforecast_model_task.sbatch" ]]; then
    echo "Python, reforecast root, or Slurm scripts are unavailable" >&2
    exit 2
fi
REFORECAST_ROOT="$(cd "${REFORECAST_ROOT}" && pwd)"
# Do not require the new result directory (or even its immediate parent) to
# exist before this launcher creates it.  The Python writers still confine all
# overwrite/delete operations to this explicit root.
RESULT_ROOT="${RESULT_ROOT%/}"
MANIFEST="${MANIFEST:-${RESULT_ROOT}/inventory/reforecast_inventory.json}"
CONFIG_DIRECTORY="${CONFIG_DIRECTORY:-${RESULT_ROOT}/inventory/configs}"
METADATA_CSV="${METADATA_CSV:-${REPOSITORY_ROOT}/Rossby Model Storage Locations - Sheet1.csv}"
mkdir -p "${RESULT_ROOT}/logs" "$(dirname "${MANIFEST}")" "${CONFIG_DIRECTORY}"

MODEL_ARGS=()
if [[ -n "${MODELS_TEXT}" ]]; then
    read -r -a MODELS <<< "${MODELS_TEXT}"
    MODEL_ARGS=(--models "${MODELS[@]}")
fi

"${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/inventory_reforecast_models.py" \
    --reforecast-root "${REFORECAST_ROOT}" \
    --metadata-csv "${METADATA_CSV}" \
    --results-root "${RESULT_ROOT}" \
    --repository-root "${REPOSITORY_ROOT}" \
    --years ${YEARS_TEXT} --months ${MONTHS_TEXT} \
    --max-forecast-day "${MAX_FORECAST_DAY}" \
    --manifest "${MANIFEST}" --config-directory "${CONFIG_DIRECTORY}" \
    "${MODEL_ARGS[@]}"

TASK_COUNT="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["task_count"])' "${MANIFEST}")"
if (( TASK_COUNT == 0 )); then
    echo "Inventory found no requested model/month partitions; nothing submitted" >&2
    exit 0
fi
if (( INVENTORY_ONLY )); then
    echo "Inventory-only: ${MANIFEST} (${TASK_COUNT} tasks)"
    exit 0
fi
if (( MAX_CONCURRENT < 1 )); then
    echo "--max-concurrent must be positive" >&2
    exit 2
fi

EXPORT_ARGUMENT="--export=ALL,REPOSITORY_ROOT=${REPOSITORY_ROOT},REFORECAST_MANIFEST=${MANIFEST},HEAT_VERIFICATION_RESULTS_ROOT=${RESULT_ROOT},OVERWRITE=${OVERWRITE_VALUE},VERIFICATION_REGIONS=${REGIONS_TEXT},VERIFICATION_PROBABILITY_BINS=${PROBABILITY_BINS_TEXT},VERIFICATION_DECISION_THRESHOLDS=${DECISION_THRESHOLDS_TEXT}"
cache_job="$(sbatch --parsable "${EXPORT_ARGUMENT},VERIFICATION_STAGE=case_cache" --array="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}" \
    --output="${RESULT_ROOT}/logs/reforecast_cache_%A_%a.out" --error="${RESULT_ROOT}/logs/reforecast_cache_%A_%a.err" \
    "${SLURM_DIRECTORY}/submit_reforecast_model_task.sbatch")"
metrics_job="$(sbatch --parsable "${EXPORT_ARGUMENT},VERIFICATION_STAGE=cached_metrics" --dependency="afterany:${cache_job}" \
    --array="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}" \
    --output="${RESULT_ROOT}/logs/reforecast_metrics_%A_%a.out" --error="${RESULT_ROOT}/logs/reforecast_metrics_%A_%a.err" \
    "${SLURM_DIRECTORY}/submit_reforecast_model_task.sbatch")"
# afterany is deliberate: isolated task/model failures are recorded, while
# successful model partitions still aggregate and plot.
aggregate_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterany:${metrics_job}" \
    --output="${RESULT_ROOT}/logs/reforecast_aggregate_%j.out" --error="${RESULT_ROOT}/logs/reforecast_aggregate_%j.err" \
    "${SLURM_DIRECTORY}/submit_reforecast_aggregation.sbatch")"
plot_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterany:${aggregate_job}" \
    --output="${RESULT_ROOT}/logs/reforecast_plot_%j.out" --error="${RESULT_ROOT}/logs/reforecast_plot_%j.err" \
    "${SLURM_DIRECTORY}/submit_reforecast_plotting.sbatch")"

printf 'Inventory: %s (%s tasks)\n' "${MANIFEST}" "${TASK_COUNT}"
printf 'Submitted reforecast case-cache array: %s\n' "${cache_job}"
printf 'Submitted cache-backed metric array: %s (afterany:%s)\n' "${metrics_job}" "${cache_job}"
printf 'Submitted tolerant aggregation: %s (afterany:%s)\n' "${aggregate_job}" "${metrics_job}"
printf 'Submitted aggregate-only all-model plotting: %s (afterany:%s)\n' "${plot_job}" "${aggregate_job}"
printf 'Monitor: squeue -u "%s" -j %s,%s,%s,%s\n' "${USER}" "${cache_job}" "${metrics_job}" "${aggregate_job}" "${plot_job}"
