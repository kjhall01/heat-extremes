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

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi
srun "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/compute_reforecast_task.py" \
    --manifest "${MANIFEST}" --task-index "${SLURM_ARRAY_TASK_ID}" "${OVERWRITE_ARGS[@]}"

echo "Finish: $(date --iso-8601=seconds)"
