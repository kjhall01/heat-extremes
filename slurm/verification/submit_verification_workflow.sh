#!/bin/bash
# Submit a configuration-driven compute array -> aggregation -> plotting chain.
# Example is documented in docs/verification.md.  This script does not infer
# data paths from the source store; those stay in the supplied YAML.

set -eo pipefail

usage() {
    cat <<'EOF'
Usage: submit_verification_workflow.sh --config FILE --model NAME --result-root DIRECTORY \
       --years "YYYY ..." --months "MM ..." [--max-concurrent N] [--overwrite]
EOF
}

CONFIG=""
MODEL_NAME=""
RESULT_ROOT=""
YEARS_TEXT=""
MONTHS_TEXT=""
MAX_CONCURRENT=2
OVERWRITE_VALUE=0
while (( $# )); do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --model) MODEL_NAME="$2"; shift 2 ;;
        --result-root) RESULT_ROOT="$2"; shift 2 ;;
        --years) YEARS_TEXT="$2"; shift 2 ;;
        --months) MONTHS_TEXT="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --overwrite) OVERWRITE_VALUE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ -z "${CONFIG}" || -z "${MODEL_NAME}" || -z "${RESULT_ROOT}" || -z "${YEARS_TEXT}" || -z "${MONTHS_TEXT}" ]]; then
    usage >&2
    exit 2
fi

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SLURM_DIRECTORY="${REPOSITORY_ROOT}/slurm/verification"
CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
RESULT_ROOT="$(cd "$(dirname "${RESULT_ROOT}")" && pwd)/$(basename "${RESULT_ROOT}")"
if [[ ! -f "${CONFIG}" || ! -f "${SLURM_DIRECTORY}/submit_verification_array.sbatch" ]]; then
    echo "Config or generic verification Slurm scripts not found" >&2
    exit 2
fi
read -r -a YEARS <<< "${YEARS_TEXT}"
read -r -a MONTHS <<< "${MONTHS_TEXT}"
TASK_COUNT=$(( ${#YEARS[@]} * ${#MONTHS[@]} ))
if (( TASK_COUNT < 1 || MAX_CONCURRENT < 1 )); then
    echo "At least one year/month and a positive --max-concurrent value are required" >&2
    exit 2
fi
mkdir -p "${RESULT_ROOT}/logs"
EXPORT_ARGUMENT="--export=ALL,REPOSITORY_ROOT=${REPOSITORY_ROOT},VERIFICATION_CONFIG=${CONFIG},HEAT_VERIFICATION_RESULTS_ROOT=${RESULT_ROOT},VERIFICATION_MODEL_NAME=${MODEL_NAME},VERIFICATION_YEARS=${YEARS_TEXT},VERIFICATION_MONTHS=${MONTHS_TEXT},OVERWRITE=${OVERWRITE_VALUE}"

compute_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --array="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}" \
    --output="${RESULT_ROOT}/logs/heat_verify_%A_%a.out" --error="${RESULT_ROOT}/logs/heat_verify_%A_%a.err" \
    "${SLURM_DIRECTORY}/submit_verification_array.sbatch")"
aggregate_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterok:${compute_job}" \
    --output="${RESULT_ROOT}/logs/heat_verify_aggregate_%j.out" --error="${RESULT_ROOT}/logs/heat_verify_aggregate_%j.err" \
    "${SLURM_DIRECTORY}/submit_verification_aggregation.sbatch")"
plot_job="$(sbatch --parsable "${EXPORT_ARGUMENT}" --dependency="afterok:${aggregate_job}" \
    --output="${RESULT_ROOT}/logs/heat_verify_plot_%j.out" --error="${RESULT_ROOT}/logs/heat_verify_plot_%j.err" \
    "${SLURM_DIRECTORY}/submit_verification_plotting.sbatch")"

printf 'Submitted compute array: %s\n' "${compute_job}"
printf 'Submitted aggregation: %s (afterok:%s)\n' "${aggregate_job}" "${compute_job}"
printf 'Submitted plotting: %s (afterok:%s)\n' "${plot_job}" "${aggregate_job}"
printf 'Monitor: squeue -u "%s" -j %s,%s,%s\n' "${USER}" "${compute_job}" "${aggregate_job}" "${plot_job}"
