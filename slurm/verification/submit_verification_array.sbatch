#!/bin/bash
# Generic configuration-driven verification array.  The workflow launcher
# supplies array bounds, logs, and required environment values.
#SBATCH --job-name=heat_verify
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --array=0-15%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
CONFIG="${VERIFICATION_CONFIG:?Set VERIFICATION_CONFIG to the model YAML}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:?Set HEAT_VERIFICATION_RESULTS_ROOT to the configured results root}"
MODEL_NAME="${VERIFICATION_MODEL_NAME:?Set VERIFICATION_MODEL_NAME to model.name}"
YEARS_TEXT="${VERIFICATION_YEARS:?Set VERIFICATION_YEARS, for example '2022 2023'}"
MONTHS_TEXT="${VERIFICATION_MONTHS:?Set VERIFICATION_MONTHS, for example '6 7 8 9'}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Configured Python executable is missing: ${PYTHON}" >&2
    exit 2
fi
read -r -a YEARS <<< "${YEARS_TEXT}"
read -r -a MONTHS <<< "${MONTHS_TEXT}"
TASK_COUNT=$(( ${#YEARS[@]} * ${#MONTHS[@]} ))
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TASK_COUNT )); then
    echo "Array task ${SLURM_ARRAY_TASK_ID} is outside ${TASK_COUNT} configured partitions" >&2
    exit 2
fi

YEAR="${YEARS[$((SLURM_ARRAY_TASK_ID / ${#MONTHS[@]}))]}"
MONTH="${MONTHS[$((SLURM_ARRAY_TASK_ID % ${#MONTHS[@]}))]}"
PARTITION="$(printf '%04d-%02d' "${YEAR}" "${MONTH}")"
OUTPUT_PATH="${RESULT_ROOT}/${MODEL_NAME}/partial/${PARTITION}"
mkdir -p "${RESULT_ROOT}/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "Model: ${MODEL_NAME}"
echo "Partition: ${PARTITION}"
echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Output: ${OUTPUT_PATH}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi
srun "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/compute_verification_partition.py" \
    --config "${CONFIG}" --year "${YEAR}" --month "${MONTH}" --resume "${OVERWRITE_ARGS[@]}"

echo "Finish: $(date --iso-8601=seconds)"
