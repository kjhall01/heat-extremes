#!/bin/bash
# Render aggregate-only all-model comparison figures from the inventory.
#SBATCH --job-name=reforecast_plot
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -eo pipefail

REPOSITORY_ROOT="${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:?Set REPOSITORY_ROOT or submit from repository root}}"
PYTHON="${HEAT_EXTREMES_PYTHON:-/home/kylehall/miniconda3/envs/heat-extremes/bin/python}"
MANIFEST="${REFORECAST_MANIFEST:?Set REFORECAST_MANIFEST to the generated inventory JSON}"
if [[ ! -x "${PYTHON}" || ! -f "${MANIFEST}" ]]; then
    echo "Python executable or inventory manifest is missing" >&2
    exit 2
fi
echo "Hostname: $(hostname)"
echo "Start: $(date --iso-8601=seconds)"
srun "${PYTHON}" -u "${REPOSITORY_ROOT}/scripts/verification/plot_reforecast_inventory.py" --manifest "${MANIFEST}"
echo "Finish: $(date --iso-8601=seconds)"
