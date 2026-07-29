#!/bin/bash
# Build final global q95 schemas, then submit longitude-band compute -> finalize.
# The preflight report is required and raw source stores are never modified.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_model_temperature_q95_workflow.sh [options]

Options:
  --result-root DIRECTORY      default: verification_results root on /net/monsoon
  --preflight-report FILE      default: <result-root>/model_climatology/preflight/model_temperature_q95_preflight.json
  --output-directory DIRECTORY default: <result-root>/model_climatology/model_temperature_q95_2000_2020
  --models "NAME ..."          default: all ready models in the preflight report
  --years "YYYY ..."           default: "2000 ... 2020"
  --months "MM ..."            default: "1 ... 12"
  --max-forecast-day N         inclusive zero-based cap; default: 12
  --window-days N              odd calendar-day window; default: 15
  --percentile P               default: 95
  --max-concurrent N           simultaneous longitude tasks; default: 1
  --overwrite                  replace existing final q95 products, never raw stores
  --initialize-only            create/reuse final schemas and manifest but do not submit jobs
EOF
}

RESULT_ROOT="/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results"
PREFLIGHT_REPORT=""
OUTPUT_DIRECTORY=""
MODELS_TEXT=""
YEARS_TEXT="2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020"
MONTHS_TEXT="1 2 3 4 5 6 7 8 9 10 11 12"
MAX_FORECAST_DAY=12
WINDOW_DAYS=15
PERCENTILE=95
MAX_CONCURRENT=1
OVERWRITE_VALUE=0
INITIALIZE_ONLY=0

while (( $# )); do
    case "$1" in
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --preflight-report) PREFLIGHT_REPORT="$2"; shift 2 ;;
        --output-directory) OUTPUT_DIRECTORY="$2"; shift 2 ;;
        --models) MODELS_TEXT="$2"; shift 2 ;;
        --years) YEARS_TEXT="$2"; shift 2 ;;
        --months) MONTHS_TEXT="$2"; shift 2 ;;
        --max-forecast-day) MAX_FORECAST_DAY="$2"; shift 2 ;;
        --window-days) WINDOW_DAYS="$2"; shift 2 ;;
        --percentile) PERCENTILE="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --overwrite) OVERWRITE_VALUE=1; shift ;;
        --initialize-only) INITIALIZE_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
SLURM_DIRECTORY="${REPOSITORY_ROOT}/slurm/verification"
BUILD_SCRIPT="${REPOSITORY_ROOT}/scripts/verification/build_model_temperature_q95_workflow.py"
if [[ ! -x "${PYTHON}" || ! -f "${BUILD_SCRIPT}" || ! -f "${SLURM_DIRECTORY}/submit_model_temperature_q95_band.sbatch" ]]; then
    echo "Python, q95 workflow scripts, or Slurm scripts are unavailable" >&2
    exit 2
fi
if (( MAX_CONCURRENT < 1 || MAX_FORECAST_DAY < 0 || MAX_FORECAST_DAY > 14 )); then
    echo "--max-concurrent must be positive and --max-forecast-day must be within 0--14" >&2
    exit 2
fi

read -r -a YEARS <<< "${YEARS_TEXT}"
read -r -a MONTHS <<< "${MONTHS_TEXT}"
if (( ${#YEARS[@]} == 0 || ${#MONTHS[@]} == 0 )); then
    echo "At least one year and month are required" >&2
    exit 2
fi
RESULT_ROOT="${RESULT_ROOT%/}"
PREFLIGHT_REPORT="${PREFLIGHT_REPORT:-${RESULT_ROOT}/model_climatology/preflight/model_temperature_q95_preflight.json}"
if [[ -z "${OUTPUT_DIRECTORY}" ]]; then
    OUTPUT_DIRECTORY="${RESULT_ROOT}/model_climatology/model_temperature_q95_${YEARS[0]}_${YEARS[$(( ${#YEARS[@]} - 1 ))]}"
fi
MANIFEST="${OUTPUT_DIRECTORY}/workflow_manifest.json"
mkdir -p "${RESULT_ROOT}/logs"

MODEL_ARGS=()
if [[ -n "${MODELS_TEXT}" ]]; then
    read -r -a MODELS <<< "${MODELS_TEXT}"
    MODEL_ARGS=(--models "${MODELS[@]}")
fi
OVERWRITE_ARGS=()
if (( OVERWRITE_VALUE )); then
    OVERWRITE_ARGS+=(--overwrite)
fi

"${PYTHON}" -u "${BUILD_SCRIPT}" \
    --results-root "${RESULT_ROOT}" \
    --preflight-report "${PREFLIGHT_REPORT}" \
    --output-directory "${OUTPUT_DIRECTORY}" \
    --years "${YEARS[@]}" --months "${MONTHS[@]}" \
    --max-forecast-day "${MAX_FORECAST_DAY}" \
    --window-days "${WINDOW_DAYS}" --percentile "${PERCENTILE}" \
    "${MODEL_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

TASK_COUNT="$("${PYTHON}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["task_count"])' "${MANIFEST}")"
if (( TASK_COUNT == 0 )); then
    echo "No q95 longitude tasks were initialized" >&2
    exit 2
fi
if (( INITIALIZE_ONLY )); then
    echo "Initialized only: ${MANIFEST} (${TASK_COUNT} global longitude tasks)"
    exit 0
fi

EXPORT_ARGUMENT="--export=ALL,REPOSITORY_ROOT=${REPOSITORY_ROOT},MODEL_Q95_MANIFEST=${MANIFEST},MODEL_Q95_OVERWRITE=${OVERWRITE_VALUE}"
compute_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --array="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}" \
    --output="${RESULT_ROOT}/logs/model_q95_%A_%a.out" --error="${RESULT_ROOT}/logs/model_q95_%A_%a.err" \
    "${SLURM_DIRECTORY}/submit_model_temperature_q95_band.sbatch")"
finalize_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterok:${compute_job}" \
    --output="${RESULT_ROOT}/logs/model_q95_finalize_%j.out" --error="${RESULT_ROOT}/logs/model_q95_finalize_%j.err" \
    "${SLURM_DIRECTORY}/finalize_model_temperature_q95_workflow.sbatch")"

printf 'Manifest: %s (%s global longitude tasks)\n' "${MANIFEST}" "${TASK_COUNT}"
printf 'Submitted final-global-q95 array: %s\n' "${compute_job}"
printf 'Submitted q95 finalizer: %s (afterok:%s)\n' "${finalize_job}" "${compute_job}"
printf 'Monitor: squeue -u "%s" -j %s,%s\n' "${USER}" "${compute_job}" "${finalize_job}"
