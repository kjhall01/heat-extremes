#!/bin/bash
# Build global q95 schemas, then submit daily staging -> parallel lead q95 -> finalize.
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
  --max-concurrent N           simultaneous q95 Slurm tasks; default: 24
  --stage-max-concurrent N     simultaneous raw-read staging tasks; default: 12
  --quantile-tasks-per-job N   q95 calculations per Slurm task; default: 2
  --submit-leads-only          do not submit staging; use existing staged band stores
  --staging-job-id ID          optional afterok dependency for --submit-leads-only
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
MAX_CONCURRENT=24
STAGE_MAX_CONCURRENT=12
QUANTILE_TASKS_PER_JOB=2
OVERWRITE_VALUE=0
INITIALIZE_ONLY=0
SUBMIT_LEADS_ONLY=0
STAGING_JOB_ID=""

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
        --stage-max-concurrent) STAGE_MAX_CONCURRENT="$2"; shift 2 ;;
        --quantile-tasks-per-job) QUANTILE_TASKS_PER_JOB="$2"; shift 2 ;;
        --submit-leads-only) SUBMIT_LEADS_ONLY=1; shift ;;
        --staging-job-id) STAGING_JOB_ID="$2"; shift 2 ;;
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
if [[ ! -x "${PYTHON}" || ! -f "${BUILD_SCRIPT}" || ! -f "${SLURM_DIRECTORY}/submit_model_temperature_q95_stage.sbatch" || ! -f "${SLURM_DIRECTORY}/submit_model_temperature_q95_lead.sbatch" ]]; then
    echo "Python, q95 workflow scripts, or Slurm scripts are unavailable" >&2
    exit 2
fi
if (( MAX_CONCURRENT < 1 || MAX_CONCURRENT > 24 || STAGE_MAX_CONCURRENT < 1 || STAGE_MAX_CONCURRENT > 24 || QUANTILE_TASKS_PER_JOB < 1 || MAX_FORECAST_DAY < 0 || MAX_FORECAST_DAY > 14 )); then
    echo "concurrency values must be within the general-QOS limit of 24 and --max-forecast-day must be within 0--14" >&2
    exit 2
fi
if (( INITIALIZE_ONLY && SUBMIT_LEADS_ONLY )); then
    echo "--initialize-only and --submit-leads-only cannot be used together" >&2
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
    --quantile-tasks-per-job "${QUANTILE_TASKS_PER_JOB}" \
    "${MODEL_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

read -r STAGING_TASK_COUNT QUANTILE_TASK_COUNT QUANTILE_JOB_COUNT < <("${PYTHON}" -c 'import json, sys; payload=json.load(open(sys.argv[1])); print(payload["staging_task_count"], payload["quantile_task_count"], payload["quantile_job_count"])' "${MANIFEST}")
if (( STAGING_TASK_COUNT == 0 || QUANTILE_TASK_COUNT == 0 || QUANTILE_JOB_COUNT == 0 )); then
    echo "No q95 staging or lead tasks were initialized" >&2
    exit 2
fi
if (( INITIALIZE_ONLY )); then
    echo "Initialized only: ${MANIFEST} (${STAGING_TASK_COUNT} staging tasks; ${QUANTILE_TASK_COUNT} lead calculations in ${QUANTILE_JOB_COUNT} Slurm tasks)"
    exit 0
fi

EXPORT_ARGUMENT="--export=ALL,REPOSITORY_ROOT=${REPOSITORY_ROOT},MODEL_Q95_MANIFEST=${MANIFEST},MODEL_Q95_OVERWRITE=${OVERWRITE_VALUE}"
LEAD_DEPENDENCY=()
stage_job=""
if (( SUBMIT_LEADS_ONLY )); then
    if [[ -n "${STAGING_JOB_ID}" ]]; then
        LEAD_DEPENDENCY=(--dependency="afterok:${STAGING_JOB_ID}")
    fi
else
    stage_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --array="0-$((STAGING_TASK_COUNT - 1))%${STAGE_MAX_CONCURRENT}" \
        --output="${RESULT_ROOT}/logs/model_q95_stage_%A_%a.out" --error="${RESULT_ROOT}/logs/model_q95_stage_%A_%a.err" \
        "${SLURM_DIRECTORY}/submit_model_temperature_q95_stage.sbatch")"
    LEAD_DEPENDENCY=(--dependency="afterok:${stage_job}")
fi
quantile_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" "${LEAD_DEPENDENCY[@]}" --array="0-$((QUANTILE_JOB_COUNT - 1))%${MAX_CONCURRENT}" \
    --output="${RESULT_ROOT}/logs/model_q95_lead_%A_%a.out" --error="${RESULT_ROOT}/logs/model_q95_lead_%A_%a.err" \
    "${SLURM_DIRECTORY}/submit_model_temperature_q95_lead.sbatch")"
finalize_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterok:${quantile_job}" \
    --output="${RESULT_ROOT}/logs/model_q95_finalize_%j.out" --error="${RESULT_ROOT}/logs/model_q95_finalize_%j.err" \
    "${SLURM_DIRECTORY}/finalize_model_temperature_q95_workflow.sbatch")"

printf 'Manifest: %s (%s staging tasks; %s lead calculations in %s Slurm tasks)\n' "${MANIFEST}" "${STAGING_TASK_COUNT}" "${QUANTILE_TASK_COUNT}" "${QUANTILE_JOB_COUNT}"
if [[ -n "${stage_job}" ]]; then
    printf 'Submitted raw-read staging array: %s\n' "${stage_job}"
elif [[ -n "${STAGING_JOB_ID}" ]]; then
    printf 'Using existing staging array: %s\n' "${STAGING_JOB_ID}"
else
    printf 'Using already-complete staged band stores\n'
fi
printf 'Submitted lead-q95 array: %s (throttle=%s)\n' "${quantile_job}" "${MAX_CONCURRENT}"
printf 'Submitted q95 finalizer: %s (afterok:%s)\n' "${finalize_job}" "${quantile_job}"
if [[ -n "${stage_job}" ]]; then
    printf 'Monitor: squeue -u "%s" -j %s,%s,%s\n' "${USER}" "${stage_job}" "${quantile_job}" "${finalize_job}"
elif [[ -n "${STAGING_JOB_ID}" ]]; then
    printf 'Monitor: squeue -u "%s" -j %s,%s,%s\n' "${USER}" "${STAGING_JOB_ID}" "${quantile_job}" "${finalize_job}"
else
    printf 'Monitor: squeue -u "%s" -j %s,%s\n' "${USER}" "${quantile_job}" "${finalize_job}"
fi
