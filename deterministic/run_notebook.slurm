#!/bin/bash
#SBATCH --job-name=west_afric_95_aifs_tmin
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
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
: "${TEST_YEAR_END:=2025}"    # inclusive; set > TEST_YEAR_START for a multi-year run

# Region: pick ONE of REGION (a name) or REGION_BOUNDS (raw numbers) --
# _build_notebook.py's Step 0 raises if both end up set. REGION has no
# default of its own (stays empty unless you pass one via --export); when
# it's empty, REGION_BOUNDS falls back to its own Delhi default below, same
# as before this option existed. Set REGION instead (e.g.
# `--export=ALL,REGION=west_africa`) and leave REGION_BOUNDS unset -- don't
# set both.
: "${REGION:=west_africa}"  # global, tropics, nh_extratropics, sh_extratropics, conus, europe,
                 # east_asia, sea, south_asia, west_africa, east_africa, mena, lac
                 # (see _build_notebook.py's REGIONS dict for the exact box each one is)
if [ -z "$REGION" ]; then
    : "${REGION_BOUNDS:=27.5,29.5,76.5,78.5}"   # south,north,west,east (degrees); or "global" for the full grid
else
    : "${REGION_BOUNDS:=}"   # keep empty when REGION is set -- empty is treated the same as unset
fi
export REGION

: "${MODEL:=aifs_ens_mean}" # aifs_ens_mean (AIFS-ENS-v2, 26-member ensemble mean -- default),
                            # aifs_single (AIFS-single-v2's own single deterministic run),
                            # graphcast, or aurora (the two newer models -- see graphcast.py/
                            # aurora.py for real-vs-assumed caveats, substantially more
                            # unconfirmed than AIFS's own loaders). Restricts which VARIABLE
                            # values are even valid (_build_notebook.py's model_variables dict):
                            # neither graphcast nor aurora has 2m_dewpoint_temperature (no
                            # wet-bulb variables for either), and aurora additionally has no
                            # total_precipitation at all.

: "${VARIABLE:=t2m_min_6h}" # t2m_mean_6h, t2m_max_6h, t2m_min_6h, total_precipitation, or the
                             # wet-bulb-temperature equivalents t_wb_2m_mean_6h/t_wb_2m_max_6h/
                             # t_wb_2m_min_6h (Stull 2011 approximation from 2m_temperature +
                             # 2m_dewpoint_temperature -- see wetbulb.py). Not every value is
                             # valid for every MODEL above -- see that setting's comment.
                             # Absolute (35 degC) applies to every temperature variable, including
                             # wet-bulb, but not total_precipitation. Relative (percentile) applies
                             # only to t2m_max_6h/t2m_min_6h/total_precipitation -- there's no
                             # precomputed climatology file for the daily mean or for any wet-bulb
                             # variable, so those always run absolute-only.
: "${RELATIVE_PERCENTILE:=0.95}" # 0.95, 0.99, or 0.999 -- only used when VARIABLE is
                                 # t2m_max_6h/t2m_min_6h/total_precipitation; ignored for
                                 # t2m_mean_6h and every t_wb_2m_*_6h wet-bulb variable
export TEST_YEAR_START TEST_YEAR_END REGION_BOUNDS MODEL VARIABLE RELATIVE_PERCENTILE

# Dask cluster sizing -- kept in sync with --cpus-per-task/--mem above by
# default (N_WORKERS defaults to the CPUs this job was granted), rather than
# the notebook's own hardcoded fallback, which doesn't know what Slurm gave
# it. Lower MEMORY_LIMIT is a *per-worker* cap; N_WORKERS * MEMORY_LIMIT
# should stay comfortably under --mem above.
: "${N_WORKERS:=${SLURM_CPUS_PER_TASK:-6}}"
: "${THREADS_PER_WORKER:=1}"
: "${MEMORY_LIMIT:=8GiB}"
export N_WORKERS THREADS_PER_WORKER MEMORY_LIMIT

# /net/scratch, not the repo directory: per the tutorial, this is ephemeral
# shared scratch space, available from any node -- appropriate both for the
# executed-notebook output below AND for dask's own spill-to-disk directory
# (LOCAL_DIRECTORY): when a worker approaches MEMORY_LIMIT it writes some of
# its data to local_directory to free RAM, and dask's own default for that
# is just the current working directory -- which, without this, would be
# wherever this script's `cd "$SLURM_SUBMIT_DIR"` below points (typically a
# small-quota home/project filesystem, not built for GBs of spill files).
# Hit this for real: a global/multi-year run spilled enough to fill that
# quota and crashed with "OSError: [Errno 28] No space left on device" from
# zict's disk-write path. Giving dask an explicit, large scratch directory
# instead avoids that -- see the LocalCluster(..., local_directory=...) in
# _build_notebook.py's Step 0 cluster cell.
SCRATCH_DIR="/net/scratch/${USER}/deterministic_verification"
mkdir -p "$SCRATCH_DIR"
LOCAL_DIRECTORY="${SCRATCH_DIR}/dask-worker-space_${SLURM_JOB_ID:-interactive}"
mkdir -p "$LOCAL_DIRECTORY"
export LOCAL_DIRECTORY

cd "$SLURM_SUBMIT_DIR"

# Regenerate the notebook from _build_notebook.py before running it, so the
# executed notebook always reflects the current script (and the run
# parameters exported above) rather than a stale .ipynb -- see that script's
# own module docstring: hand-editing the .ipynb directly gets overwritten
# next time it runs. Settings not exposed above (absolute_threshold, etc.)
# still have their defaults in _build_notebook.py's Step 0 cell -- edit that
# file directly for those, not the notebook.
python _build_notebook.py

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
