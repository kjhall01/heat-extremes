#!/bin/bash
# One model/month task from a metadata-only reforecast inventory.
# The Python task captures source/model errors per partition so independent
# models continue and downstream aggregation can use completed work.
#SBATCH --job-name=reforecast_verify
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
MANIFEST="${REFORECAST_MANIFEST:?Set REFORECAST_MANIFEST to the generated inventory JSON}"
RESULT_ROOT="${HEAT_VERIFICATION_RESULTS_ROOT:?Set HEAT_VERIFICATION_RESULTS_ROOT to the inventory results root}"
VERIFICATION_STAGE="${VERIFICATION_STAGE:-case_cache}"

if [[ ! -x "${PYTHON}" || ! -f "${MANIFEST}" ]]; then
    echo "Python executable or inventory manifest is missing" >&2
    exit 2
fi
mkdir -p "${RESULT_ROOT}/logs"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Manifest: ${MANIFEST}"
echo "Stage: ${VERIFICATION_STAGE}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi
REGION_ARGS=()
if [[ -n "${VERIFICATION_REGIONS:-}" ]]; then
    read -r -a REQUESTED_REGIONS <<< "${VERIFICATION_REGIONS}"
    REGION_ARGS=(--regions "${REQUESTED_REGIONS[@]}")
fi
BIN_ARGS=()
if [[ -n "${VERIFICATION_PROBABILITY_BINS:-}" ]]; then
    read -r -a REQUESTED_BINS <<< "${VERIFICATION_PROBABILITY_BINS}"
    BIN_ARGS=(--probability-bins "${REQUESTED_BINS[@]}")
fi
DECISION_ARGS=()
if [[ -n "${VERIFICATION_DECISION_THRESHOLDS:-}" ]]; then
    read -r -a REQUESTED_THRESHOLDS <<< "${VERIFICATION_DECISION_THRESHOLDS}"
    DECISION_ARGS=(--decision-thresholds "${REQUESTED_THRESHOLDS[@]}")
fi
# Some cluster partitions provide a non-contiguous CPU allocation whose
# inherited affinity mask cannot be satisfied by srun's default binding.
# The task is internally thread-limited above, so disable the invalid binding
# request while retaining the Slurm CPU allocation.
srun --cpu-bind=none "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/compute_reforecast_task.py" \
    --manifest "${MANIFEST}" --task-index "${SLURM_ARRAY_TASK_ID}" --stage "${VERIFICATION_STAGE}" \
    "${REGION_ARGS[@]}" "${BIN_ARGS[@]}" "${DECISION_ARGS[@]}" "${OVERWRITE_ARGS[@]}"

echo "Finish: $(date --iso-8601=seconds)"
