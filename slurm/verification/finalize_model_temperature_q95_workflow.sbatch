#!/bin/bash
# Final metadata-only commit of all global q95 products after the array succeeds.
#SBATCH --job-name=model_q95_finalize
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
MANIFEST="${MODEL_Q95_MANIFEST:?Set MODEL_Q95_MANIFEST to the generated q95 workflow manifest}"

if [[ ! -x "${PYTHON}" || ! -f "${MANIFEST}" ]]; then
    echo "Python executable or q95 manifest is unavailable" >&2
    exit 2
fi

echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
srun --cpu-bind=none "${PYTHON}" -u \
    "${REPOSITORY_ROOT}/scripts/verification/finalize_model_temperature_q95_workflow.py" \
    --manifest "${MANIFEST}"
echo "Finish: $(date --iso-8601=seconds)"
