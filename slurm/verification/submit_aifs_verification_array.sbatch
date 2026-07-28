#!/bin/bash
# Resource settings: adjust these conservative initial values after a smoke run.
#SBATCH --job-name=aifs_verify
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --array=0-15%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results/logs/%x_%A_%a.out
#SBATCH --error=/net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results/logs/%x_%A_%a.err

set -eo pipefail

# Submit from the repository root or set REPOSITORY_ROOT explicitly.  Do not
# infer this from BASH_SOURCE: Slurm executes scripts from a spool directory.
REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
CONFIG="${VERIFICATION_CONFIG:-${REPOSITORY_ROOT}/configs/verification/aifs_ens_v2.yaml}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:-/net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/verification_results}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Configured Python executable is missing: ${PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${REPOSITORY_ROOT}/scripts/verification/compute_verification_partition.py" ]]; then
    echo "Repository does not contain verification script: ${REPOSITORY_ROOT}" >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

MONTHS=(6 7 8 9)
YEAR=$((2022 + SLURM_ARRAY_TASK_ID / 4))
MONTH="${MONTHS[$((SLURM_ARRAY_TASK_ID % 4))]}"
PARTITION="$(printf '%04d-%02d' "${YEAR}" "${MONTH}")"
OUTPUT_PATH="${RESULT_ROOT}/aifs_ens_v2/partial/${PARTITION}"

echo "Model: aifs_ens_v2"
echo "Partition: ${PARTITION}"
echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Output: ${OUTPUT_PATH}"
echo "Python: ${PYTHON}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi

srun "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/compute_verification_partition.py" \
    --config "${CONFIG}" --year "${YEAR}" --month "${MONTH}" --resume "${OVERWRITE_ARGS[@]}"

echo "Finish: $(date --iso-8601=seconds)"
