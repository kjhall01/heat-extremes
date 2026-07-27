#!/bin/bash
#SBATCH --job-name=deterministic_verification
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# Batch (non-interactive) run of deterministic_verification_metrics.ipynb.
#
# Follows https://clinic.ds.uchicago.edu/tutorials/slurm.html: that page
# documents *interactive* Slurm use (srun, attaching VS Code to a compute
# node) in detail, but only mentions sbatch/batch jobs in passing ("if you
# become bothered by long running jobs crashing ... you will need to
# leverage sbatch") without a script template. So the directives below
# borrow that page's own conventions rather than inventing unrelated ones:
#   - partition "general", matching its srun example:
#       srun -p general --gres=gpu:1 --pty --mem 1000 -t 90:00 /bin/bash
#   - no --gres=gpu:... requested: this workload (xarray/dask over
#     zarr/netCDF) is CPU- and memory-bound, not GPU-bound.
#   - output goes to /net/scratch (see bottom of script) per the tutorial's
#     storage guidance: that's ephemeral ("any data put here may be deleted
#     at any time"), so copy anything worth keeping to /net/projects or
#     elsewhere afterward.
#
# --cpus-per-task/--mem above are Slurm's allocation to this job; the run
# parameters below (year range, region, variable, relative-threshold
# percentile, dask cluster sizing) are what the notebook actually does
# *with* that allocation. _build_notebook.py's Step 0 reads each of
# TEST_YEAR_START/TEST_YEAR_END/REGION_BOUNDS/VARIABLE/RELATIVE_PERCENTILE/
# N_WORKERS/THREADS_PER_WORKER/MEMORY_LIMIT from the environment (falling
# back to its own hardcoded defaults if unset), so this is the one place to
# change them for a batch run -- no need to hand-edit _build_notebook.py or
# the .ipynb.
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p logs

# --- Run parameters: edit these, or override without editing this file via
#     e.g. `sbatch --export=ALL,TEST_YEAR_START=2023,TEST_YEAR_END=2023,REGION_BOUNDS=global,VARIABLE=t2m_max_6h,RELATIVE_PERCENTILE=0.99 run_notebook.slurm`
#     (`:  "${VAR:=default}"` only sets VAR if it isn't already in the
#     environment, so anything passed via --export takes precedence). ---
: "${TEST_YEAR_START:=2022}"
: "${TEST_YEAR_END:=2022}"    # inclusive; set > TEST_YEAR_START for a multi-year run
: "${REGION_BOUNDS:=global}"   # south,north,west,east (degrees); or "global" for the full grid
: "${VARIABLE:=t_wb_2m_mean_6h}" # t2m_mean_6h, t2m_max_6h, t2m_min_6h, total_precipitation, or the
                             # wet-bulb-temperature equivalents t_wb_2m_mean_6h/t_wb_2m_max_6h/
                             # t_wb_2m_min_6h (Stull 2011 approximation from 2m_temperature +
                             # 2m_dewpoint_temperature -- see wetbulb.py).
                             # Absolute (35 degC) applies to every temperature variable, including
                             # wet-bulb, but not total_precipitation. Relative (percentile) applies
                             # only to t2m_max_6h/t2m_min_6h/total_precipitation -- there's no
                             # precomputed climatology file for the daily mean or for any wet-bulb
                             # variable, so those always run absolute-only.
: "${RELATIVE_PERCENTILE:=0.95}" # 0.95, 0.99, or 0.999 -- only used when VARIABLE is
                                 # t2m_max_6h/t2m_min_6h/total_precipitation; ignored for
                                 # t2m_mean_6h and every t_wb_2m_*_6h wet-bulb variable
export TEST_YEAR_START TEST_YEAR_END REGION_BOUNDS VARIABLE RELATIVE_PERCENTILE

# Dask cluster sizing -- kept in sync with --cpus-per-task/--mem above by
# default (N_WORKERS defaults to the CPUs this job was granted), rather than
# the notebook's own hardcoded fallback, which doesn't know what Slurm gave
# it. Lower MEMORY_LIMIT is a *per-worker* cap; N_WORKERS * MEMORY_LIMIT
# should stay comfortably under --mem above.
: "${N_WORKERS:=${SLURM_CPUS_PER_TASK:=8}}"
: "${THREADS_PER_WORKER:=1}"
: "${MEMORY_LIMIT:=32GiB}"
export N_WORKERS THREADS_PER_WORKER MEMORY_LIMIT

cd "$SLURM_SUBMIT_DIR"

# Regenerate the notebook from _build_notebook.py before running it, so the
# executed notebook always reflects the current script (and the run
# parameters exported above) rather than a stale .ipynb -- see that script's
# own module docstring: hand-editing the .ipynb directly gets overwritten
# next time it runs. Settings not exposed above (absolute_threshold, etc.)
# still have their defaults in _build_notebook.py's Step 0 cell -- edit that
# file directly for those, not the notebook.
python _build_notebook.py

# /net/scratch, not the repo directory: per the tutorial, this is ephemeral
# shared scratch space, available from any node -- appropriate for a job
# output that you'll immediately want to inspect, then either discard or
# copy somewhere persistent (e.g. /net/projects/...).
SCRATCH_DIR="/net/scratch/${USER}/deterministic_verification"
mkdir -p "$SCRATCH_DIR"
OUTPUT_NOTEBOOK="${SCRATCH_DIR}/deterministic_verification_metrics_${SLURM_JOB_ID}.ipynb"

# --ExecutePreprocessor.timeout=-1: don't let nbconvert kill a long-running
# cell (e.g. Step 4's batched pass over many initializations) on its own
# unrelated default timeout -- let Slurm's --time above be the only limit.
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output "$OUTPUT_NOTEBOOK" \
  deterministic_verification_metrics.ipynb

echo "Executed notebook written to $OUTPUT_NOTEBOOK"
echo "That is on /net/scratch -- ephemeral -- copy it to /net/projects/..."
echo "(or elsewhere persistent) if you want to keep it."
