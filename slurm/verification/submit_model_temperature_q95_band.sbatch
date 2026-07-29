#!/bin/bash
# One non-overlapping longitude region of one final global model q95 Zarr.
# Raw reforecast stores are opened read-only. A bounded, per-band temporary
# daily-mean Zarr is removed after its final q95 output is complete.
# The launcher supplies the precise array size and conservative concurrency.
#SBATCH --job-name=model_q95
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --array=0-23%1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
MANIFEST="${MODEL_Q95_MANIFEST:?Set MODEL_Q95_MANIFEST to the generated q95 workflow manifest}"

if [[ ! -x "${PYTHON}" || ! -f "${MANIFEST}" || ! -f "${REPOSITORY_ROOT}/scripts/verification/compute_model_temperature_q95_band.py" ]]; then
    echo "Python executable, q95 manifest, or q95 compute script is unavailable" >&2
    exit 2
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OVERWRITE_ARGS=()
if [[ "${MODEL_Q95_OVERWRITE:-0}" == "1" ]]; then
    OVERWRITE_ARGS+=(--overwrite)
fi

echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
echo "Manifest: ${MANIFEST}"
echo "Raw source policy: read only"
echo "Daily staging policy: temporary per-band store, removed after successful q95 write"

# Some nodes inherit a non-contiguous affinity mask that makes Slurm's default
# CPU binding fail before Python starts. The task is explicitly thread-limited.
srun --cpu-bind=none "${PYTHON}" -u \
    "${REPOSITORY_ROOT}/scripts/verification/compute_model_temperature_q95_band.py" \
    --manifest "${MANIFEST}" --task-index "${SLURM_ARRAY_TASK_ID}" "${OVERWRITE_ARGS[@]}"

echo "Finish: $(date --iso-8601=seconds)"
