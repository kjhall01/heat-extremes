#!/bin/bash
# One independent model-band-forecast-day q95 task from staged daily data.
#SBATCH --job-name=model_q95_lead
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
MANIFEST="${MODEL_Q95_MANIFEST:?Set MODEL_Q95_MANIFEST to the generated q95 workflow manifest}"

if [[ ! -x "${PYTHON}" || ! -f "${MANIFEST}" || ! -f "${REPOSITORY_ROOT}/scripts/verification/compute_model_temperature_q95_lead.py" ]]; then
    echo "Python executable, q95 manifest, or q95 lead script is unavailable" >&2
    exit 2
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OVERWRITE_ARGS=()
if [[ "${MODEL_Q95_OVERWRITE:-0}" == "1" ]]; then OVERWRITE_ARGS+=(--overwrite); fi

echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Phase: staged daily means to one q95 forecast day"
srun --cpu-bind=none "${PYTHON}" -u \
    "${REPOSITORY_ROOT}/scripts/verification/compute_model_temperature_q95_lead.py" \
    --manifest "${MANIFEST}" --task-index "${SLURM_ARRAY_TASK_ID}" "${OVERWRITE_ARGS[@]}"
echo "Finish: $(date --iso-8601=seconds)"
