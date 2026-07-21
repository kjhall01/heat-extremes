#!/bin/bash
#SBATCH --job-name=era5_arco_cache
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --array=0-26%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=/net/monsoon/kylehall/ERA5/logs/%x_%A_%a.out
#SBATCH --error=/net/monsoon/kylehall/ERA5/logs/%x_%A_%a.err

set -euo pipefail

OUTPUT_ROOT="${OUTPUT_ROOT:-/net/monsoon/kylehall/ERA5/era5_arco_6h_surface}"
CONDA_ENV_NAME="${ERA5_ARCO_CACHE_ENV:-era5-arco-cache}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:?Submit the job from the era5_arco_cache directory.}"

# Avoid accidental nested threading inside numerical libraries.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

YEAR=$((2000 + SLURM_ARRAY_TASK_ID))
CACHE_ARGS=(
    --year "$YEAR"
    --output-root "$OUTPUT_ROOT"
    --workers "$SLURM_CPUS_PER_TASK"
)

# Current-year ARCO data is necessarily incomplete. Let the Python cache job
# stop at its latest complete 6-hour timestep; older requested years stay full.
if [[ "$YEAR" -eq "$(date +%Y)" ]]; then
    CACHE_ARGS+=(--allow-partial-year)
    if [[ "${OVERWRITE_CURRENT_YEAR:-0}" == "1" ]]; then
        CACHE_ARGS+=(--overwrite)
    fi
fi

CONDA_BASE="/home/kylehall/miniconda3"
CONDA_ENV="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Missing Conda initialization: ${CONDA_BASE}" >&2
    exit 2
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "[$(date --iso-8601=seconds)] Host: $(hostname)"
echo "[$(date --iso-8601=seconds)] Year: ${YEAR}"
echo "[$(date --iso-8601=seconds)] Python: $(which python)"
echo "[$(date --iso-8601=seconds)] Output: ${OUTPUT_ROOT}"

child_pid=""

forward_usr1() {
    echo "[$(date --iso-8601=seconds)] Forwarding Slurm USR1 warning to Python."
    if [[ -n "${child_pid:-}" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -USR1 -- "-$child_pid" 2>/dev/null || kill -USR1 "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
    exit 99
}

forward_term() {
    echo "[$(date --iso-8601=seconds)] Forwarding TERM to Python."
    if [[ -n "${child_pid:-}" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" || true
    fi
    exit 143
}

trap forward_usr1 USR1
trap forward_term TERM

setsid python -u "$SCRIPT_DIR/cache_era5_arco_year.py" \
    "${CACHE_ARGS[@]}" &
child_pid=$!

return_code=0
wait "$child_pid" || return_code=$?

echo "[$(date --iso-8601=seconds)] Python exited with code ${return_code}"

# DSI uses exit code 99 as the intentional requeue request.
if [[ "$return_code" -eq 99 ]]; then
    exit 99
fi

exit "$return_code"
