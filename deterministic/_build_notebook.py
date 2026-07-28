"""Builds deterministic_verification_metrics.ipynb via nbformat.

Global POD / FAR / SEDI maps at lead days 1, 3, 5, 7, 9, for a selectable
forecast model (`model_source`/`MODEL` in Step 0: `aifs_ens_mean` -- AIFS-ENS-v2's
26-member ensemble collapsed to its across-member mean via
`aifs_singlev2.ensemble_mean()`, the default; `aifs_single` -- AIFS-single-v2's
own single deterministic run; `graphcast`/`aurora` -- the two newer
deterministic reforecast archives, `graphcast.py`/`aurora.py`) vs ERA5
daily-mean T2M, under an absolute 35 degC threshold. Whichever model is
selected, everything downstream still only ever sees one deterministic-like
field per case -- see Step 1. RMSE and the relative/climatology-percentile
threshold are both commented out for now (not deleted) so only the
absolute-threshold path actually runs. This notebook cannot be executed in
this sandbox -- it has no access to /net/monsoon or the real model/ERA5 data
stores -- so it is built but NOT run here; run it in the environment where
those are available.

Restructured into explicit, numbered steps (0-7) so each stage can be run
and inspected independently, per request: 0. setup, 1. load data, 2.
preprocess + align, 3. examine the variable of interest, 4. calculate
H/M/F/C, 5. calculate POD/FAR/SEDI, 6. sanity checks, 7. plotting.

Reconciled with direct edits made to the .ipynb outside this script
(test_year_start/test_year_end instead of a single test_year) -- this
script is the source of truth going forward; re-run it to regenerate the
notebook rather than hand-editing the .ipynb, or those edits will be lost
next time this script runs. Two issues found in the hand-edited version are
fixed here: `calculate_scores` had started returning a tuple
(`model_var, scores`), which `mean_in_time_batches` cannot accept (it needs
a Dataset/DataArray back); and a histogram cell referenced `model_var`,
which only exists inside `calculate_scores`'s local scope, not at notebook
level -- both are superseded by the new Step 3 below, which computes a
small `model_var`/`era5_var` sample at notebook scope specifically for
inspection.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Deterministic-model verification: POD / FAR / SEDI maps

Global maps of **POD**, **FAR**, and **SEDI** for a selectable forecast
model against ERA5, at lead days **1, 3, 5, 7, 9**, over
`test_year_start`-`test_year_end`.

**Model source is selectable** (`model_source` in Step 0, or the `MODEL`
environment variable): `aifs_ens_mean` (default) -- AIFS-ENS-v2's 26-member
ensemble (`/net/monsoon/marchakitus/AIFS/v2p0/combined/forecasts_AIFS_ENS_v2`,
opened via `aifs_singlev2.open_aifs_ensv2()`), collapsed to its across-member
mean via `aifs_singlev2.ensemble_mean()`; `aifs_single` -- AIFS-single-v2's
own single deterministic run (`open_aifs_singlev2()`, no ensemble dimension
to begin with); `graphcast`/`aurora` -- the two newer deterministic
reforecast archives (`open_graphcast()`/`open_aurora()`), with substantially
more unconfirmed-against-real-data caveats than AIFS's own loaders (see
`graphcast.py`/`aurora.py`'s docstrings) -- neither has
`2m_dewpoint_temperature` (so no wet-bulb `variable` options), and Aurora
additionally has no `total_precipitation` at all. Note the ensemble mean is
a smoothed field -- expect lower POD for extreme thresholds than a true
single-realization run would give, since averaging 26 members together
suppresses each member's own extremes (see `aifs_singlev2.ensemble_mean`'s
docstring).

**`test_year_start`/`test_year_end` and `region_bounds` (see the Step 0
config cell) are the two settings to change between runs** -- no separate
demo/full switch: set `region_bounds` to a small box (e.g. the Delhi, India
default) and/or a narrow year range for a fast run, or `region_bounds = None`
(or the environment variable `REGION_BOUNDS=global`) with a wide year range
for the real global/full-year thing. A small region keeps every step cheap
regardless of how many initializations it covers (Step 3 takes advantage of
this -- see there), so there's no need for anything beyond these two
settings to get a fast, representative run.

**Region can also be picked by name** via `REGION` (Step 0's `REGIONS` dict,
or the `REGION` environment variable) instead of typing out raw numbers in
`REGION_BOUNDS`: `global`, `tropics`, `nh_extratropics`, `sh_extratropics`
(these three restrict latitude only, full longitude width),
`conus`, `europe`, `east_asia`, `sea`, `south_asia`, `west_africa`,
`east_africa`, `mena`, `lac`. Set only one of `REGION`/`REGION_BOUNDS` --
`REGION_BOUNDS` still works unchanged for a custom box not in that list.

All of these (plus `model_source`, the dask cluster's
`n_workers`/`threads_per_worker`/`memory_limit`/`local_directory`,
`variable`, and `relative_percentile` below) can also be set from
**outside** this notebook via environment variables (`TEST_YEAR_START`,
`TEST_YEAR_END`, `REGION`, `REGION_BOUNDS`, `MODEL`, `N_WORKERS`,
`THREADS_PER_WORKER`, `MEMORY_LIMIT`, `LOCAL_DIRECTORY`, `VARIABLE`,
`RELATIVE_PERCENTILE`) instead of editing the Step 0 cells directly -- see
`run_notebook.slurm`, which sets these before regenerating and executing
this notebook so a batch run's parameters live in the Slurm submit script,
not in this file. `LOCAL_DIRECTORY` in particular matters for
a large run: it's where dask spills data to disk under memory pressure, and
`run_notebook.slurm` points it at `/net/scratch` rather than dask's own
default (the current working directory) -- see that script and the Step 0
cluster cell for why (a real "No space left on device" spill failure).

*RMSE is commented out for now* (uncomment `squared_error`/
`rmse_from_mean_squared_error`/the `rmse_*` lines in Steps 4, 5, and 7 to
bring it back) -- both other threshold definitions below are active
wherever they apply to the selected `variable`.

**Extreme-event variable is selectable** (`variable` in Step 0, or the
`VARIABLE` environment variable): `t2m_mean_6h` (daily mean, the default),
`t2m_max_6h` (daily max), `t2m_min_6h` (daily min) 2m air temperature,
`total_precipitation` (daily total), or the wet-bulb-temperature equivalents
`t_wb_2m_mean_6h`/`t_wb_2m_max_6h`/`t_wb_2m_min_6h` (daily mean/max/min
wet-bulb temperature, via Stull (2011)'s empirical approximation from
`2m_temperature` + `2m_dewpoint_temperature` -- see `wetbulb.py`) -- all
seven are computed regardless of which is selected (Step 2). Which threshold
definition(s) actually run depends on `variable`
(`has_absolute_threshold`/`has_relative_climatology` in Step 0, both derived
from `variable`, not separate switches):

- **Absolute** (`variable > 35`degC): temperature variables only, including
  wet-bulb (`t2m_mean_6h`/`t2m_max_6h`/`t2m_min_6h`/`t_wb_2m_mean_6h`/
  `t_wb_2m_max_6h`/`t_wb_2m_min_6h` -- 35 degC wet-bulb happens to also be
  the commonly-cited human heat-stress survivability limit, so the same
  number does double duty). **Skipped** for `total_precipitation` -- 35
  doesn't mean anything in precipitation's units (meters/day).
- **Relative** (`variable` above `relative_percentile` -- 0.95, 0.99, or
  0.999, default 0.95, also settable via the `RELATIVE_PERCENTILE`
  environment variable -- of its 1979-2018 daily climatology at that grid
  cell and day of year): automatic whenever a precomputed climatology file
  exists for `variable` -- `t2m_max_6h`, `t2m_min_6h`, or
  `total_precipitation` (from `/net/monsoon/aasch/percentiles/`, via
  `climatology.open_percentile_climatology`). **Skipped** for
  `t2m_mean_6h` and every `t_wb_2m_*_6h` variable -- there's no precomputed
  daily-mean (or wet-bulb) climatology file, so those cases run
  absolute-only. `total_precipitation` is the reverse case: relative-only,
  absolute skipped.

**Steps below, each runnable and inspectable on its own:**

0. Setup -- imports, config, dask cluster.
1. Load data -- raw model and ERA5 datasets, untouched.
2. Preprocess and align -- units, daily aggregation, spatial grid alignment.
3. Examine the variable of interest -- a small `model_var`/`era5_var` sample,
   before running anything expensive, to sanity-check the data makes sense.
4. Calculate H, M, F, C -- batched over initializations.
5. Calculate POD, FAR, SEDI.
6. Sanity checks -- range/consistency checks on the results themselves.
7. Plotting.

Self-contained: does not import the `heatextremes` package (a shared,
read-only dependency here) -- see `deterministic_metrics.py` and
`era5_loader.py` for the small amount of logic copied locally instead.

**Not executed in this environment:** this notebook needs `/net/monsoon/...`
and the real AIFS-ENS-v2/ERA5 data stores, neither of which is reachable
here, so cells below were written, unit-tested against synthetic arrays (see
the accompanying module tests), and syntax-checked, but not run end-to-end
against real data. In particular:

- `aifs_singlev2.open_aifs_ensv2()`'s `start_year`/`end_year` filename-based
  filtering is adapted from `open_aifs_singlev2()` (itself adapted from the
  original, reference ensemble loader) but not verified against the real
  store.
- The climatology computation (one quantile reduction per day of year, over
  22 years of global 0.25-degree daily data) is expensive; consider
  persisting `threshold_by_doy` to disk after computing it once, rather
  than recomputing it on every run."""
))

# --- Step 0: Setup ---------------------------------------------------------

cells.append(nbf.v4.new_markdown_cell(
"""## Step 0: Setup

Imports, configuration, and the dask cluster. The settings you'll actually
change between runs are `test_year_start`/`test_year_end` (inclusive; set
them equal for a single year, or apart for a multi-year run), `region_bounds`
(a `(south, north, west, east)` box in degrees, or `None` for the full global
grid -- or pick one by name via `REGION`/`REGIONS` instead, e.g.
`west_africa`, `conus`, `tropics`; see `REGIONS` below), `model_source`
(`aifs_ens_mean` default, `aifs_single`, `graphcast`, or `aurora` -- see
`model_variables` below for which `variable` options each one supports),
and `variable` (`t2m_mean_6h`, `t2m_max_6h`, `t2m_min_6h`,
`total_precipitation`, or the wet-bulb-temperature equivalents
`t_wb_2m_mean_6h`/`t_wb_2m_max_6h`/`t_wb_2m_min_6h`, absolute-threshold only
-- see `wetbulb.py`)."""
))

cells.append(nbf.v4.new_code_cell(
"""import csv
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from zarr.errors import ZarrUserWarning

from era5_loader import (
    open_cached_era5,
    daily_era5_aggregates,
    daily_era5_wet_bulb_aggregates,  # local, no heatextremes dependency
)

from aifs_singlev2 import (
    open_aifs_singlev2,  # AIFS-single-v2's own single deterministic run -- MODEL="aifs_single"
    open_aifs_ensv2,  # AIFS-ENS-v2 (26-member ensemble) -- MODEL="aifs_ens_mean" (the default)
    ensemble_mean,  # collapses the 26-member "number" dimension to its mean right after loading (Step 1)
    daily_aifs_aggregates_calendar_aligned,
    daily_aifs_precipitation,  # total_precipitation is already daily in the real store -- see its docstring
    daily_aifs_wet_bulb_calendar_aligned,  # t_wb_2m_mean_6h/t_wb_2m_max_6h/t_wb_2m_min_6h -- see wetbulb.py
)
from graphcast import open_graphcast  # MODEL="graphcast" -- no dewpoint, no confirmed precipitation convention
from aurora import open_aurora  # MODEL="aurora" -- no dewpoint, no precipitation at all
from climatology import open_percentile_climatology, threshold_at_verification_time  # local relative-threshold helpers
from deterministic_metrics import (
    extreme_indicators,
    false_alarm_ratio,
    mean_in_time_batches,  # local copy, no heatextremes dependency -- see deterministic_metrics.py
    probability_of_detection,
    probability_of_false_detection,
    verification_time,  # used in Step 3 to find the ERA5 dates a model sample verifies against
    # rmse_from_mean_squared_error,  # RMSE commented out for now -- see calculate_scores/finish-metrics cells
    # squared_error,  # RMSE commented out for now -- see calculate_scores/finish-metrics cells
    symmetric_extremal_dependence_index,
)

def _env(name: str, default=None):
    \"\"\"Like os.environ.get, but treats an unset or empty-string variable
    the same way (falls back to default either way).

    Plain os.environ.get(name, default) only substitutes default when the
    key is entirely absent -- a variable that's exported but set to an
    empty string (e.g. left blank in an interactive shell, or a Slurm
    --export value that expanded to nothing) comes back as an empty string
    instead, silently bypassing the default -- int() on an empty string
    then raises a confusing ValueError downstream. The shell side of this
    notebook (run_notebook.slurm's parameter-expansion defaults) already
    treats unset-or-empty the same way; this makes the Python side
    consistent with it.
    \"\"\"
    value = os.environ.get(name)
    return default if not value else value


# --- The settings you'll actually change between runs ---
# Each reads an environment variable first, falling back to the hardcoded
# default if that variable isn't set -- so a Slurm submit script can control
# these without editing this file (see run_notebook.slurm, which sets
# TEST_YEAR_START/TEST_YEAR_END/REGION_BOUNDS before generating this
# notebook). Editing the defaults below still works fine for interactive use.
test_year_start = int(_env("TEST_YEAR_START", 2022))
test_year_end = int(_env("TEST_YEAR_END", 2022))  # inclusive; > test_year_start for multi-year

# Region to restrict both model and ERA5 to, as (south, north, west, east) in
# degrees -- e.g. the Delhi, India default below. A small region keeps every
# step cheap regardless of how many initializations it covers (Step 3 takes
# advantage of this -- see there, and Step 2 for where this gets applied); a
# large or global region will be slow, especially Step 3's inspection and
# Step 4's batched pass.
#
# Named regions -- broad climate/geographic boxes, selectable by name via
# REGION instead of typing out raw numbers in REGION_BOUNDS. An entry with no
# "longitude" key (tropics/nh_extratropics/sh_extratropics) means no
# longitude restriction at all -- the full grid width, latitude-banded only.
# "global" (empty dict) means no restriction on either dimension, same as
# region_bounds = None below. Longitudes are already in this pipeline's
# normalize_longitude() convention ([-180, 180), not 0-360).
REGIONS = {
    "global": {},
    "tropics": {"latitude": (-23.5, 23.5)},
    "nh_extratropics": {"latitude": (23.5, 90.0)},
    "sh_extratropics": {"latitude": (-90.0, -23.5)},
    "conus": {"latitude": (24.0, 50.0), "longitude": (-125.0, -66.0)},
    "europe": {"latitude": (35.0, 72.0), "longitude": (-10.0, 40.0)},
    "east_asia": {"latitude": (20.0, 55.0), "longitude": (100.0, 150.0)},
    # Mainland and maritime SEA: Thailand, Malaysia, Indonesia, Philippines.
    "sea": {"latitude": (-12.0, 25.0), "longitude": (90.0, 140.0)},
    # India, Bangladesh, Pakistan, Nepal, and nearby South Asian land areas.
    "south_asia": {"latitude": (5.0, 35.0), "longitude": (65.0, 95.0)},
    "west_africa": {"latitude": (0.0, 25.0), "longitude": (-20.0, 20.0)},
    "east_africa": {"latitude": (-15.0, 20.0), "longitude": (25.0, 55.0)},
    # Middle East and North Africa.
    "mena": {"latitude": (12.0, 42.0), "longitude": (-20.0, 65.0)},
    # Latin America and the Caribbean; includes Mexico and Caribbean islands.
    "lac": {"latitude": (-60.0, 32.0), "longitude": (-120.0, -55.0)},
}

# REGION (a name from REGIONS above) and REGION_BOUNDS (a raw
# "south,north,west,east" box, or the literal "global") are two ways to say
# the same thing -- set at most one. REGION_BOUNDS still works completely
# unchanged for a custom box that isn't one of the named regions.
# region_name is kept (None unless REGION was used) so region_label below can
# use the readable name instead of formatted numbers.
_region_env = _env("REGION")
_region_bounds_env = _env("REGION_BOUNDS")
if _region_env is not None and _region_bounds_env is not None:
    raise ValueError(
        "Set only one of REGION (a named region) or REGION_BOUNDS (a custom "
        f"box) -- got REGION={_region_env!r} and REGION_BOUNDS={_region_bounds_env!r}."
    )

region_name = None
if _region_env is not None:
    if _region_env not in REGIONS:
        raise ValueError(f"REGION must be one of {sorted(REGIONS)}, got: {_region_env!r}")
    region_name = _region_env
    _region_spec = REGIONS[region_name]
    if not _region_spec:
        region_bounds = None
    else:
        _lat = _region_spec.get("latitude")
        _lon = _region_spec.get("longitude")
        region_bounds = (_lat[0], _lat[1], _lon[0] if _lon else None, _lon[1] if _lon else None)
elif _region_bounds_env is None:
    region_bounds = (27.5, 29.5, 76.5, 78.5)  # Delhi, India; set to None for global
elif _region_bounds_env.strip().lower() == "global":
    region_bounds = None
else:
    region_bounds = tuple(float(value) for value in _region_bounds_env.split(","))
    if len(region_bounds) != 4:
        raise ValueError(
            "REGION_BOUNDS must be 'south,north,west,east' or 'global', "
            f"got: {_region_bounds_env!r}"
        )

# Which model source to use, selectable via MODEL. "aifs_ens_mean" (default)
# is AIFS-ENS-v2's 26-member ensemble collapsed to its across-member mean
# (Step 1, via ensemble_mean()); "aifs_single" is AIFS-single-v2's own single
# deterministic run; "graphcast"/"aurora" are the two newer deterministic
# reforecast archives (see graphcast.py/aurora.py's module docstrings for
# real-vs-assumed caveats -- substantially more unconfirmed than AIFS's own
# loaders, since neither store's directory listing has actually been seen).
#
# MODEL_VARIABLES below restricts which VARIABLE values are even offered per
# model, since not every model has every raw variable this pipeline can
# derive a threshold for:
# - Neither GraphCast nor Aurora has 2m_dewpoint_temperature, so every
#   wet-bulb variable (t_wb_2m_*) is unavailable for both.
# - Aurora additionally has no total_precipitation at all (confirmed).
# - GraphCast's total_precipitation ("tp") is deliberately left out here too,
#   even though the raw variable exists: its real day-boundary/aggregation
#   convention hasn't been confirmed against real data the way AIFS's has
#   (see aifs_singlev2.daily_aifs_precipitation's docstring for how much
#   back-and-forth even that confirmed case took) -- add
#   "total_precipitation" to graphcast's set below once you've verified that
#   against the real store, not before.
model_variables = {
    "aifs_single": {
        "t2m_mean_6h", "t2m_max_6h", "t2m_min_6h", "total_precipitation",
        "t_wb_2m_mean_6h", "t_wb_2m_max_6h", "t_wb_2m_min_6h",
    },
    "aifs_ens_mean": {
        "t2m_mean_6h", "t2m_max_6h", "t2m_min_6h", "total_precipitation",
        "t_wb_2m_mean_6h", "t_wb_2m_max_6h", "t_wb_2m_min_6h",
    },
    "graphcast": {"t2m_mean_6h", "t2m_max_6h", "t2m_min_6h"},
    "aurora": {"t2m_mean_6h", "t2m_max_6h", "t2m_min_6h"},
}
model_source = _env("MODEL", "aifs_ens_mean")
if model_source not in model_variables:
    raise ValueError(f"MODEL must be one of {sorted(model_variables)}, got: {model_source!r}")

# Which daily-aggregated quantity to run the whole exercise against. Every
# one allowed for model_source is computed by daily_era5_aggregates()/
# daily_era5_wet_bulb_aggregates()/daily_aifs_aggregates_calendar_aligned()/
# daily_aifs_precipitation()/daily_aifs_wet_bulb_calendar_aligned() (Step 2)
# regardless of this setting, so switching it doesn't need new data -- just
# re-running from Step 2 onward (or the whole notebook) with a different
# value.
_allowed_variables = model_variables[model_source]
variable = _env("VARIABLE", "t2m_mean_6h")
if variable not in _allowed_variables:
    raise ValueError(
        f"VARIABLE must be one of {sorted(_allowed_variables)} for MODEL={model_source!r}, "
        f"got: {variable!r}"
    )

# Wet-bulb variables (2m air temperature + dewpoint combined via Stull
# (2011)'s empirical approximation -- see wetbulb.py) need a couple of
# special cases below, since they're derived from two raw variables rather
# than resampled directly from one: Step 2 skips the in-place Kelvin-to-degC
# conversion of 2m_temperature for these (wetbulb.wet_bulb_temperature needs
# the RAW Kelvin value and does its own conversion internally -- converting
# 2m_temperature first would double-convert it), and compute_model_var
# (also Step 2) dispatches to daily_aifs_wet_bulb_calendar_aligned instead of
# the plain temperature aggregator.
wet_bulb_variables = {"t_wb_2m_mean_6h", "t_wb_2m_max_6h", "t_wb_2m_min_6h"}

forecast_days = 10  # daily aggregation covers lead days up to (and including) day 9
lead_days_to_plot = [1, 3, 5, 7, 9]

# Absolute threshold: 35 degC, temperature-only (including wet-bulb
# temperature -- 35 degC wet-bulb is itself the commonly-cited human
# heat-stress survivability limit, so the same number does double duty
# here) -- doesn't apply to total_precipitation (units: meters/day), so it's
# automatically skipped for that variable (has_absolute_threshold below),
# not applied with a meaningless number. Note daily *minimum* T2M (or
# wet-bulb T) exceeding 35 degC is a much more extreme, rarer event than
# daily mean or max doing so (it means the temperature never dropped below
# 35 degC all day/night) -- Step 3's sanity check will make that obvious if
# so.
absolute_threshold = 35.0  # degC
has_absolute_threshold = variable != "total_precipitation"

# Relative (climatology-percentile) threshold config. Uses Aaron Schwartz's
# precomputed 1979-2018 daily percentile climatology (see climatology.py's
# CLIMATOLOGY_PATHS) instead of computing one from scratch. Only available
# for t2m_max_6h/t2m_min_6h/total_precipitation -- there's no precomputed
# file for the daily mean, nor for any of the wet-bulb variables -- so it's
# automatically skipped when variable is t2m_mean_6h or any t_wb_2m_*_6h.
# No separate on/off switch needed: has_relative_climatology below is
# derived from variable, not a config flag of its own.
_allowed_percentiles = {0.95, 0.99, 0.999}  # the quantiles actually present in the precomputed files
relative_percentile = float(_env("RELATIVE_PERCENTILE", 0.95))
if relative_percentile not in _allowed_percentiles:
    raise ValueError(
        f"RELATIVE_PERCENTILE must be one of {sorted(_allowed_percentiles)}, got: {relative_percentile}"
    )
has_relative_climatology = variable in {"t2m_max_6h", "t2m_min_6h", "total_precipitation"}

# Human-readable tag for this region, used in cache/output filenames below so
# that changing region_bounds (without also changing test_year_start/end)
# can't silently reuse a cache file computed for a different region. Uses the
# REGIONS name directly when REGION was used (e.g. "west_africa", not a wall
# of formatted numbers); falls back to formatted numbers for a custom
# REGION_BOUNDS box, or "global" for either region_name == "global" or a bare
# region_bounds = None. Handles the latitude-only named regions
# (tropics/nh_extratropics/sh_extratropics -- longitude entries are None)
# separately, since the 4-number format string can't take a None.
if region_name is not None:
    region_label = region_name
elif region_bounds is None:
    region_label = "global"
elif region_bounds[2] is None or region_bounds[3] is None:
    region_label = "region_lat_{:.2f}_{:.2f}".format(region_bounds[0], region_bounds[1])
else:
    region_label = "region_{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(*region_bounds)

# Human-readable tag for which threshold(s) were actually computed (absolute,
# relative, or both -- t2m_mean_6h gets absolute-only, total_precipitation
# gets relative-only, t2m_max_6h/t2m_min_6h get both), used in the same
# filenames -- so switching variable/relative_percentile never gets mistaken
# for a previous, different result.
_threshold_parts = []
if has_absolute_threshold:
    _threshold_parts.append("abs")
if has_relative_climatology:
    _threshold_parts.append(f"rel{relative_percentile}")
threshold_label = "_".join(_threshold_parts) if _threshold_parts else "none"

# Where the expensive per-lead-map result is cached, so re-running this notebook
# doesn't redo the full batching pass over all initializations. Filename includes
# the year range, region_label, variable, and threshold_label, so switching any
# of these never gets mistaken for (or overwrites) a previous, different result.
# No longer named "absolute_by_lead_map": it holds relative_* columns too
# whenever has_relative_climatology is True, and may hold only relative_*
# columns (no absolute_*) for total_precipitation.
#
# Prefix is model_source itself (e.g. "aifs_ens_mean", "graphcast") rather
# than a fixed string -- deliberately, not cosmetically: with MODEL now
# selectable, a fixed prefix would let a stale cache file from a *different*
# model source get silently reused for this one, since
# test_year_start/end/region_label/variable/threshold_label alone don't
# capture which model produced the cached numbers.
by_lead_map_cache_path = Path(
    f"{model_source}_by_lead_map_{test_year_start}_{test_year_end}_{region_label}_{variable}_{threshold_label}.nc"
)

# One row appended per run that actually computes (not loads from cache) -- see
# log_run_timing() in Step 4 -- so wall-clock time can be compared across cluster
# configurations / regions / year ranges over time.
run_timing_log_path = Path("run_timings.csv")"""
))

cells.append(nbf.v4.new_code_cell(
"""from dask.distributed import Client, LocalCluster

# Named (not inlined) so Step 4 can log them alongside each run's wall-clock time,
# for comparing configurations later. Each also reads an environment variable
# first (see run_notebook.slurm, which derives these from the job's own
# --cpus-per-task/--mem so they can't silently drift out of sync with what
# Slurm actually granted).
n_workers = int(_env("N_WORKERS", 6))              # match the CPUs you requested
threads_per_worker = int(_env("THREADS_PER_WORKER", 1))
memory_limit = _env("MEMORY_LIMIT", "3GiB")  # tune to your node memory -- global, multi-decade
                                                        # data needs more than the small single-region demo
                                                        # this notebook started from

# Where dask workers spill data to disk when they approach memory_limit
# (zict/zict.file under the hood). Left as dask's own default (None -- which
# resolves to the current working directory) only for interactive/notebook
# use; run_notebook.slurm always sets LOCAL_DIRECTORY to a /net/scratch path
# instead, since dask's default put spill files in the repo directory --
# actually hit "OSError: [Errno 28] No space left on device" from a global
# run spilling enough to fill that filesystem's (small) quota. If you're
# running this interactively on a big region/year range, set LOCAL_DIRECTORY
# yourself to somewhere with real space before starting the cluster.
local_directory = _env("LOCAL_DIRECTORY", None)

cluster = LocalCluster(
    n_workers=n_workers,
    threads_per_worker=threads_per_worker,
    processes=True,
    memory_limit=memory_limit,
    local_directory=local_directory,
)
client = Client(cluster)
client"""
))

# --- Step 1: Load data ------------------------------------------------------

cells.append(nbf.v4.new_markdown_cell(
"""## Step 1: Load data

Raw model and ERA5 datasets, untouched -- no unit conversion, no
aggregation, no region subsetting yet. Each cell's own output lets you
confirm what actually came off disk before Step 2 changes anything.

ERA5 loading is copied from `ensemble_verification_metrics.md`, restricted
to `test_year_start`-`test_year_end` **plus one extra year** past the end
(`end_year=test_year_end + 1`): ERA5 is only needed here to verify forecasts
in the requested range, not to build a climatology baseline too -- the
relative threshold's climatology (Step 2) comes from a separate precomputed
file (1979-2018 baseline), not from the ERA5 loaded here, so no extra
baseline years need loading for it either. The `+1` buffer is still needed
so late-year initializations verifying a few days into the following
January (lead days up to 9) can still find a matching ERA5 observation --
dropping it would silently produce a few extra NaN cases for the last
initializations of `test_year_end`, at the higher lead days, rather than an
error. **The forecast model is selectable** (`model_source` in Step 0, or the
`MODEL` environment variable): `aifs_ens_mean` (default) opens the
**AIFS-ENS-v2** 26-member ensemble via `open_aifs_ensv2()` and immediately
collapses it to its across-member mean via `ensemble_mean()`, dropping the
`"number"` dimension entirely; `aifs_single` opens **AIFS-single-v2**'s own
single deterministic run via `open_aifs_singlev2()` directly (no ensemble
dimension to begin with); `graphcast`/`aurora` open the newer GraphCast/
Aurora reforecast archives via `open_graphcast()`/`open_aurora()` (see
`graphcast.py`/`aurora.py` for the substantial unconfirmed-against-real-data
caveats on those two -- more so than AIFS's own loaders). Whichever is
chosen, every loader parses each store's filename (`init_YYYYMMDDT00.zarr`)
and only opens the matching stores, rather than opening the whole archive's
metadata and subsetting afterward -- and by the time `model` (this cell's
own output) is produced, every branch has already reduced to one
deterministic-like field per case, so nothing downstream needs to know or
care which model actually produced it.

**Not every `variable` (Step 0) is available for every `model_source`** --
`model_variables` in Step 0 restricts this: GraphCast/Aurora have no
`2m_dewpoint_temperature` (no wet-bulb variables), and Aurora has no
`total_precipitation` at all (GraphCast's is excluded too, pending real-store
confirmation of its aggregation convention) -- picking an unsupported
combination raises a clear error in Step 0 rather than failing confusingly
later."""
))

cells.append(nbf.v4.new_code_cell(
"""era5 = open_cached_era5(
    start_year=test_year_start,
    end_year=test_year_end + 1,  # buffer for late-test_year_end initializations verifying into next January
    chunks={"time": 244, "latitude": 90, "longitude": 180},
)
era5"""
))

cells.append(nbf.v4.new_code_cell(
"""# need this warning nonsense bbecause zarr versions are hard
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Numcodecs codecs are not in the Zarr version 3 specification.*",
        category=ZarrUserWarning,
    )
    # Dispatches on model_source (Step 0, MODEL env var). Every branch parses
    # each store's init_YYYYMMDDT00.zarr filename before opening, so only
    # [test_year_start, test_year_end] gets opened, not the whole archive's
    # metadata -- same approach for all four models.
    if model_source == "aifs_ens_mean":
        model = open_aifs_ensv2(start_year=test_year_start, end_year=test_year_end)
        # Collapse the 26-member ensemble to its across-member mean right
        # away -- everything after this line (including every other cell in
        # this notebook) treats `model` as a single deterministic-like
        # field, same as every other model_source option gives directly.
        model = ensemble_mean(model)
    elif model_source == "aifs_single":
        model = open_aifs_singlev2(start_year=test_year_start, end_year=test_year_end)
    elif model_source == "graphcast":
        # variables=["2t"]: restrict IO to just 2m temperature -- the only
        # variable model_variables (Step 0) allows for this model_source, so
        # nothing else GraphCast has (mslp, pressure-level q/t/u/v/z, tp) is
        # actually needed here. See graphcast.py for why tp is excluded.
        model = open_graphcast(start_year=test_year_start, end_year=test_year_end, variables=["2t"])
    elif model_source == "aurora":
        # Same reasoning as graphcast above -- only 2m temperature is used.
        model = open_aurora(start_year=test_year_start, end_year=test_year_end, variables=["2t"])
    else:
        raise ValueError(f"Unhandled MODEL: {model_source!r}")  # defensive -- Step 0 already validated this

model"""
))

# --- Step 2: Preprocess and align ------------------------------------------

cells.append(nbf.v4.new_markdown_cell(
"""## Step 2: Preprocess and align

Longitude normalization and ERA5 daily aggregation are copied from
`ensemble_verification_metrics.md`. If `region_bounds` is set (see Step 0),
`subset_region` (also copied from `ensemble_verification_metrics.md`)
restricts both `model` and `era5` to that box instead of the full global
grid. `model_test_year` below always covers every initialization in
`test_year_start`-`test_year_end` regardless of `region_bounds`; a small
region keeps that cheap even over the full year range, so there's no need
to also throw away initializations to get a fast run. Set
`region_bounds = None` in the Step 0 config cell for the real global run.

**Units:** both ERA5 and AIFS-ENS-v2 store `2m_temperature` (and
`2m_dewpoint_temperature`) natively in Kelvin (AIFS inherits this from being
trained on ERA5) -- neither `open_cached_era5`/`daily_era5_aggregates` nor
`open_aifs_ensv2`/`ensemble_mean` convert it (averaging commutes with the
Kelvin offset, so taking the ensemble mean first vs. converting units first
would give the same answer either way -- this notebook happens to take the
mean first, in Step 1). Converted to degC below, right after
regridding/subsetting and before daily aggregation, so it's comparable
against `absolute_threshold` (degC) everywhere downstream -- **except when
`variable` is one of the wet-bulb variables** (`wet_bulb_variables` in Step
0): `wetbulb.wet_bulb_temperature` needs the RAW Kelvin values and does its
own conversion internally, so the in-place `2m_temperature -= 273.15` step
below is skipped entirely in that case (for both `model` and `era5`) to
avoid double-converting it.

**What "align" means here -- spatial only, not time.** `xr.align(...,
exclude={"time", "prediction_timedelta"})` below only aligns the
`latitude`/`longitude` grids between `model` and `era5` (so both cover the
same spatial extent with matching coordinates); their *time* axes are
deliberately left alone. `model` stays indexed by `(time, prediction_timedelta)`
(initialization + lead time) and `era5` stays indexed by plain daily `time`
-- there is no step here that forces them onto a shared time axis. Instead,
the metric functions in Step 4 look up the correct ERA5 value for each
(initialization, lead time) case at its verification time
(`init_time + lead_time`) individually, via `deterministic_metrics.verification_time`/
`match_observations`. That's intentional, not a missing step: pre-aligning
time here isn't possible anyway, since one ERA5 day corresponds to many
different (initialization, lead time) pairs from different forecast runs."""
))

cells.append(nbf.v4.new_code_cell(
"""def normalize_longitude(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    \"\"\"Convert a longitude coordinate from 0--360 to [-180, 180) and sort it.\"\"\"
    if "longitude" not in data.coords:
        raise ValueError("All inputs must have a longitude coordinate")
    normalized_longitude = (data.longitude + 180) % 360 - 180
    if normalized_longitude.to_index().has_duplicates:
        raise ValueError("Longitude normalization produced duplicate coordinates")
    return data.assign_coords(longitude=normalized_longitude).sortby("longitude")


def subset_region(
    data: xr.Dataset | xr.DataArray,
    bounds: tuple[float, float, float | None, float | None],
) -> xr.Dataset | xr.DataArray:
    \"\"\"Select a region after longitude normalization. (Only used when region_bounds is set.)

    `west`/`east` may be None -- the latitude-only named regions
    (tropics/nh_extratropics/sh_extratropics in REGIONS, Step 0) restrict
    latitude only, leaving the full longitude range untouched.
    \"\"\"
    south, north, west, east = bounds
    latitude_slice = (
        slice(south, north)
        if data.latitude.values[0] < data.latitude.values[-1]
        else slice(north, south)
    )
    if west is None or east is None:
        return data.sel(latitude=latitude_slice)
    return data.sel(latitude=latitude_slice, longitude=slice(west, east))


model = normalize_longitude(model)
era5 = normalize_longitude(era5)

if region_bounds is not None:
    model = subset_region(model, region_bounds)
    era5 = subset_region(era5, region_bounds)

# Both ERA5 and AIFS-ENS-v2 store 2m_temperature natively in Kelvin (AIFS is
# trained on ERA5, so it inherits ERA5's units) -- convert to degC here, before
# daily aggregation, so absolute_threshold (in degC) is comparable downstream.
# Subtracting a constant commutes with both mean() and max(), so converting
# before vs. after daily_era5_aggregates()/daily_aifs_aggregates_calendar_aligned()
# is equivalent.
#
# Wet-bulb variables are the exception: wetbulb.wet_bulb_temperature needs
# RAW Kelvin 2m_temperature/2m_dewpoint_temperature and does its own
# conversion internally (see daily_era5_wet_bulb_aggregates/
# daily_aifs_wet_bulb_calendar_aligned's docstrings) -- converting
# 2m_temperature in place first would double-convert it. So for those
# variables, skip the in-place conversion for *both* model and era5 (model's
# conversion is skipped too, even though model's own aggregation happens
# later in compute_model_var below, since model_batch there needs to still
# be raw Kelvin when daily_aifs_wet_bulb_calendar_aligned runs on it), and
# call daily_era5_wet_bulb_aggregates instead of daily_era5_aggregates.
KELVIN_TO_CELSIUS_OFFSET = 273.15
if variable in wet_bulb_variables:
    era5 = daily_era5_wet_bulb_aggregates(era5)
else:
    model["2m_temperature"] = model["2m_temperature"] - KELVIN_TO_CELSIUS_OFFSET
    model["2m_temperature"].attrs["units"] = "degC"
    era5["2m_temperature"] = era5["2m_temperature"] - KELVIN_TO_CELSIUS_OFFSET
    era5["2m_temperature"].attrs["units"] = "degC"
    era5 = daily_era5_aggregates(era5)

# Spatial alignment only -- see markdown above for why time is deliberately
# left unaligned (matched later, per-case, at verification time in Step 4).
# (No "number"/ensemble-member dimension here: AIFS-ENS-v2's 26 members were
# already collapsed to their mean in Step 1, via ensemble_mean() -- by this
# point model is deterministic-like again, same shape as AIFS-single-v2 gave.)
excluded_dimensions = {"time", "prediction_timedelta"}
model, era5 = xr.align(
    model, era5, join="inner", exclude=excluded_dimensions, copy=False
)

era5_var = era5[variable]

# model was already restricted to [test_year_start, test_year_end] at load time
# (open_aifs_ensv2's start_year/end_year), so this is just a defensive no-op,
# not a real filter. Demo mode restricts region only (above), not initialization
# count -- model_test_year covers every initialization in range either way.
model_test_year = model.sel(time=slice(f"{test_year_start}-01-01", f"{test_year_end}-12-31"))
era5_var"""
))

cells.append(nbf.v4.new_markdown_cell(
"""### Relative (climatology-percentile) threshold

Loads Aaron Schwartz's precomputed 1979-2018 daily percentile climatology
(`/net/monsoon/aasch/percentiles/percentiles_1979-2018_{2m_temperature_max,
2m_temperature_min,total_precipitation}.nc`, via
`climatology.open_percentile_climatology`) -- no from-scratch computation
needed, since this is already sitting on disk. (`climatology.
local_climatology_quantile` still exists for computing one from scratch if
you ever need a variable without a precomputed file -- see that module's
docstring -- but it's a genuinely expensive fallback: one quantile reduction
per day of year, over however many baseline years/however much of the globe
it covers.)

**Only runs when `has_relative_climatology` is true** (Step 0: `variable` is
`t2m_max_6h`, `t2m_min_6h`, or `total_precipitation`) -- there's no
precomputed climatology file for the daily mean, so this cell, and every
`relative_*` step downstream, is automatically skipped when
`variable = "t2m_mean_6h"`.

`relative_percentile` (Step 0 config, or the `RELATIVE_PERCENTILE`
environment variable) selects which of the file's three precomputed
quantiles (0.95, 0.99, 0.999) to use as the threshold.

Longitude is normalized and, if `region_bounds` is set, the same region is
applied, exactly as for `model`/`era5` above -- the climatology file's native
grid (0-360 longitude, descending latitude) otherwise wouldn't line up.

**Units:** the precomputed *temperature* climatologies are stored in Kelvin
(confirmed -- same native units as ERA5/AIFS `2m_temperature`), so those are
converted to degC here too, right after the region step, same offset and
pattern as `model`/`era5` above -- without this, `threshold_by_doy` would sit
around 300+ while `model_var`/`era5_var` sit around 30-40, and every
comparison in `extreme_indicators` would silently always be false (zero hits
everywhere, not a crash). The *precipitation* climatology is already in
meters (confirmed -- same as ERA5/model's native `total_precipitation`
units), so it's used as-is, no conversion.

`xr.align` against `model` then trims it to exactly the same spatial extent
(defensive: normally a no-op once both are on the same 0.25-degree grid).
`.persist()` materializes the result once -- it's small (one day-of-year x
lat x lon array) and already on disk, but without `.persist()` it would
still be re-read from disk inside every one of Step 4's per-initialization
batches rather than once up front.

**`compute_model_var`**, defined at the end of this cell, is the one place
that knows how to get `variable`'s daily-aggregated `DataArray` out of a raw
model batch regardless of which of the four variables is selected -- Step 3
and Step 4 both call it rather than duplicating this branch."""
))

cells.append(nbf.v4.new_code_cell(
"""if has_relative_climatology:
    threshold_by_doy = open_percentile_climatology(variable, relative_percentile)
    threshold_by_doy = normalize_longitude(threshold_by_doy)
    if region_bounds is not None:
        threshold_by_doy = subset_region(threshold_by_doy, region_bounds)
    if variable != "total_precipitation":
        # Precomputed temperature climatology is stored in Kelvin (same native
        # units as ERA5/AIFS 2m_temperature) -- convert to degC here too, same
        # offset/order as model/era5 above, so it's comparable to model_var/
        # era5_var (both already in degC) inside extreme_indicators. The
        # precipitation climatology is already in meters (confirmed -- same as
        # ERA5/model's native total_precipitation units), so no conversion.
        threshold_by_doy = threshold_by_doy - KELVIN_TO_CELSIUS_OFFSET
        threshold_by_doy.attrs["units"] = "degC"
    # Defensive trim to match model/era5's exact grid (see markdown above) --
    # discard the aligned copy of model, it should be unchanged.
    _, threshold_by_doy = xr.align(
        model, threshold_by_doy, join="inner", exclude=excluded_dimensions, copy=False
    )
    threshold_by_doy = threshold_by_doy.persist()
    print(f"threshold_by_doy ({variable}, {relative_percentile} percentile):", threshold_by_doy.sizes)
else:
    threshold_by_doy = None
    print(f"No precomputed climatology for variable={variable!r} -- relative threshold skipped.")


def compute_model_var(model_batch: xr.Dataset) -> xr.DataArray:
    \"\"\"Return the daily-aggregated `variable` DataArray for one batch of model
    initializations (used by both Step 3's sample and Step 4's calculate_scores).

    Temperature variables (t2m_mean_6h/t2m_max_6h/t2m_min_6h) go through
    daily_aifs_aggregates_calendar_aligned's 6-hourly resampling.
    total_precipitation is already daily in the real store (under
    prediction_timedelta_daily, confirmed against a real Dataset repr --
    see aifs_singlev2.daily_aifs_precipitation), so it's read directly
    instead, no resampling needed. Wet-bulb variables
    (t_wb_2m_mean_6h/t_wb_2m_max_6h/t_wb_2m_min_6h) go through
    daily_aifs_wet_bulb_calendar_aligned, which computes wet-bulb temperature
    from model_batch's still-raw-Kelvin 2m_temperature/2m_dewpoint_temperature
    (see the wet_bulb_variables branch above, which deliberately leaves
    model's 2m_temperature unconverted for this reason) before aggregating.
    \"\"\"
    if variable == "total_precipitation":
        return daily_aifs_precipitation(model_batch, max_days=forecast_days)["total_precipitation"]
    if variable in wet_bulb_variables:
        return daily_aifs_wet_bulb_calendar_aligned(model_batch, max_days=forecast_days)[variable]
    return daily_aifs_aggregates_calendar_aligned(model_batch, max_days=forecast_days)[variable]


threshold_by_doy"""
))

# --- Step 3: Examine the variable of interest -------------------------------

cells.append(nbf.v4.new_markdown_cell(
"""## Step 3: Examine the variable of interest

Before running the expensive batched pass in Step 4, sanity-check that
`variable` (currently whichever of `t2m_mean_6h`/`t2m_max_6h`/`t2m_min_6h`/
`total_precipitation`/`t_wb_2m_mean_6h`/`t_wb_2m_max_6h`/`t_wb_2m_min_6h` is
configured in Step 0) looks physically reasonable --
right order of magnitude, right units (degC not Kelvin for temperature;
meters for precipitation), and actually crosses whichever threshold(s) are
active at least sometimes (otherwise Step 5's POD/FAR/SEDI will be all-NaN
from zero exceedance events, not a bug). This is especially worth checking
for `t2m_min_6h`: "daily min > 35 degC" is a much rarer, more extreme event
than the same threshold on daily mean or max (it means the temperature never
dropped below 35 degC all day/night), so seeing few or no exceedances there
is plausible and not necessarily a bug -- but zero exceedances anywhere in
JJAS would still be worth a second look. For `total_precipitation` there's
no absolute threshold to compare against at all (Step 0:
`has_absolute_threshold` is false) -- only the printed relative-threshold
stats and the histogram are relevant there.

**Restricted to June-September (JJAS)** via `select_jjas` below -- the
season the absolute 35 degC threshold is actually meant to catch in northern
India, and also India's monsoon season, so it's the right restriction for
`total_precipitation` too. Picking an arbitrary initialization (e.g.
whichever comes first in `test_year_start`) risks landing in winter, where
`variable` looking nowhere near a summer heat/monsoon extreme is *correct*,
not a bug -- Delhi's January daily mean temperature is typically 10-15 degC,
and January precipitation is near zero. Filtering to JJAS first avoids
drawing the wrong conclusion from an unrepresentative sample.

**How much of JJAS gets examined depends on whether `region_bounds` is set:**

- If `region_bounds` is set, `model_test_year` is already restricted to that
  (small) region in Step 2, not to a small number of initializations -- so
  it's cheap to examine *every* JJAS initialization across the whole
  `test_year_start`-`test_year_end` range, not just one. (A very large
  custom region may still be expensive even though it's non-None -- this
  assumes a region small enough to be cheap, like the Delhi default.)
- If `region_bounds` is `None`, `model_test_year` covers the global grid, so
  this instead looks at only one representative JJAS initialization, for
  the same memory reason as before (daily-aggregating the whole archive
  just for a sanity-check histogram would recreate the exact problem Step
  4's batching avoids).

`era5_var` is restricted to the dates the examined sample(s) actually verify
against (`verification_time` from `deterministic_metrics.py`), so the two
are a fair, small, apples-to-apples comparison either way."""
))

cells.append(nbf.v4.new_code_cell(
"""def select_jjas(data: xr.Dataset | xr.DataArray, time_dim: str = "time") -> xr.Dataset | xr.DataArray:
    \"\"\"Restrict to initializations in June-September (JJAS) -- the season
    absolute heat extremes like `absolute_threshold` actually occur in
    northern India. Keeps every year present in `data`, not just one.\"\"\"
    month = data[time_dim].dt.month
    return data.isel({time_dim: month.isin([6, 7, 8, 9]).values})


model_test_year_jjas = select_jjas(model_test_year)
if model_test_year_jjas.sizes["time"] == 0:
    raise ValueError(
        f"No JJAS (Jun-Sep) initializations found in {test_year_start}-{test_year_end} -- "
        "check test_year_start/test_year_end."
    )

if region_bounds is not None:
    # Region-restricted already (Step 2), so examining every JJAS initialization
    # in range -- not just one -- is still cheap.
    model_sample = model_test_year_jjas
else:
    # Global run: keep this cheap by looking at one representative JJAS
    # initialization, not the whole archive.
    model_sample = model_test_year_jjas.isel(time=slice(0, 1))

model_var_sample = compute_model_var(model_sample)

sample_verification_times = verification_time(model_var_sample)
era5_var_sample = era5_var.sel(
    time=slice(sample_verification_times.min().values, sample_verification_times.max().values)
)

_unit_label = "degC" if variable != "total_precipitation" else "m"

print("model_var_sample:", model_var_sample.sizes)
print("era5_var_sample:", era5_var_sample.sizes)
print()
print(f"model max ({variable}):", float(model_var_sample.max()), _unit_label)
print(f"era5 max ({variable}):", float(era5_var_sample.max()), _unit_label)
if has_absolute_threshold:
    print(f"absolute_threshold:", absolute_threshold, _unit_label)

if has_relative_climatology:
    relative_threshold_sample = threshold_at_verification_time(threshold_by_doy, model_var_sample)
    print(
        f"relative threshold ({relative_percentile} percentile, 1979-2018) over this sample -- "
        f"mean: {float(relative_threshold_sample.mean()):.4f} {_unit_label}, "
        f"min: {float(relative_threshold_sample.min()):.4f}, "
        f"max: {float(relative_threshold_sample.max()):.4f}"
    )"""
))

cells.append(nbf.v4.new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(model_var_sample.values.flatten(), bins=20)
if has_absolute_threshold:
    axes[0].axvline(absolute_threshold, color="red", linestyle="--", label=f"{absolute_threshold} {_unit_label}")
    axes[0].legend()
axes[0].set_title(f"model {variable} (JJAS, {model_sample.sizes['time']} init(s))")
axes[1].hist(era5_var_sample.values.flatten(), bins=20)
if has_absolute_threshold:
    axes[1].axvline(absolute_threshold, color="red", linestyle="--", label=f"{absolute_threshold} {_unit_label}")
    axes[1].legend()
axes[1].set_title(f"era5 {variable} (matching dates)")
fig.tight_layout()"""
))

# --- Step 4: Calculate H, M, F, C -------------------------------------------

cells.append(nbf.v4.new_markdown_cell(
"""## Step 4: Calculate H, M, F, C

`calculate_scores` returns, per initialization batch: unreduced
hit/miss/false-alarm/correct-negative indicators for the absolute threshold
(prefixed `absolute_*`) whenever `has_absolute_threshold` is true, plus the
same for the relative threshold (prefixed `relative_*`) whenever
`has_relative_climatology` is true (Step 2) -- merged into one `Dataset` via
`xr.merge` when both are active (just one or the other, unmerged, when only
one applies -- e.g. relative-only for `total_precipitation`, absolute-only
for `t2m_mean_6h`). RMSE's `squared_error` term is still commented out
(uncomment it, and the `rmse_*` lines in Step 5/7, to bring that back).

`mean_in_time_batches` reduces only over `"time"` (the initializations),
keeping `prediction_timedelta`, `latitude`, and `longitude` -- i.e. one
global map per lead time, not a single lead-time-averaged map. It processes
one initialization at a time (`batch_size=1`) specifically so the full
multi-year, global, 6-hourly archive never needs to fit in memory at once --
see the earlier discussion of why this batching exists at all.

**Cached to disk** at `by_lead_map_cache_path` after the first run, since
this batching pass is still the expensive part of the notebook even with
climatology skipped. Re-running this cell (or the whole notebook) loads the
cached file instead of recomputing it. To force a fresh recomputation --
e.g. after changing `absolute_threshold`, `variable`, or anything upstream
-- delete that file, or change `by_lead_map_cache_path` in Step 0.

**Timed and logged to `run_timing_log_path`** whenever it actually computes
(not when loading from cache): one row per run recording wall-clock time
alongside the cluster configuration (`n_workers`, `threads_per_worker`,
`memory_limit`) and run parameters, so you can build up a record comparing
configurations over multiple runs rather than a single one-off number.

`prediction_timedelta` is stored as a plain integer lead-day count rather
than relying on `xarray`/`netCDF4`'s `timedelta64` CF encoding: that
round-trip raised `ValueError: failed to prevent overwriting existing key
'dtype' in attrs on variable 'prediction_timedelta'` when actually tested
(xarray 2025.6.1 / netCDF4 1.7.4) -- converting to/from integer days on
either side of `to_netcdf`/`open_dataset` sidesteps it entirely."""
))

cells.append(nbf.v4.new_code_cell(
"""def calculate_scores(model_batch: xr.Dataset) -> xr.Dataset:
    # compute_model_var (Step 2): calendar-aligned 6-hourly resampling for
    # temperature variables, or a direct read for total_precipitation
    # (already daily in the real store) -- see that function's docstring.
    model_var = compute_model_var(model_batch)

    score_pieces = []

    if has_absolute_threshold:
        absolute_indicators = extreme_indicators(model_var, era5_var, absolute_threshold).rename(
            {name: f"absolute_{name}" for name in ("hits", "misses", "false_alarms", "correct_negatives")}
        )
        score_pieces.append(absolute_indicators)

    if has_relative_climatology:
        relative_threshold = threshold_at_verification_time(threshold_by_doy, model_var)
        relative_indicators = extreme_indicators(model_var, era5_var, relative_threshold).rename(
            {name: f"relative_{name}" for name in ("hits", "misses", "false_alarms", "correct_negatives")}
        )
        score_pieces.append(relative_indicators)

    scores = xr.merge(score_pieces) if len(score_pieces) > 1 else score_pieces[0]

    # scores["squared_error"] = squared_error(model_var, era5_var)  # RMSE commented out for now
    return scores.drop_vars("verification_time", errors="ignore")


def log_run_timing(elapsed_seconds: float) -> None:
    \"\"\"Append one row to run_timing_log_path recording this run's wall-clock
    time alongside the cluster configuration and run parameters, so runs can
    be compared over time (e.g. different regions, year ranges, or n_workers).\"\"\"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_source": model_source,
        "region_label": region_label,
        "test_year_start": test_year_start,
        "test_year_end": test_year_end,
        "n_initializations": model_test_year.sizes["time"],
        "n_lead_days": len(lead_days_to_plot),
        "variable": variable,
        "absolute_threshold": absolute_threshold if has_absolute_threshold else None,
        "relative_percentile": relative_percentile if has_relative_climatology else None,
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "memory_limit": memory_limit,
        "elapsed_seconds": round(elapsed_seconds, 1),
    }
    write_header = not run_timing_log_path.exists()
    with open(run_timing_log_path, "a", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged run timing to {run_timing_log_path}: {row}")


def select_by_lead_day(scores: xr.Dataset, lead_days: list[int]) -> xr.Dataset:
    \"\"\"Select the nearest available prediction_timedelta bin to each nominal lead
    day, then relabel the coordinate to the clean nominal day (see Step 5's markdown:
    with calendar-aligned bins this should always match exactly; nearest+tolerance
    is a defensive fallback, not a fix for a known offset). Defined here (not in
    Step 5, where it's also used) since this cell's own H/M/F/C plot below needs
    it too -- defined once, used in both places.\"\"\"
    lead_timedeltas = np.array(lead_days).astype("timedelta64[D]").astype("timedelta64[ns]")
    selected = scores.sel(
        prediction_timedelta=lead_timedeltas, method="nearest", tolerance=np.timedelta64(12, "h")
    )
    return selected.assign_coords(prediction_timedelta=lead_timedeltas)


def plot_metric_grid(
    data_by_row: dict[str, tuple[xr.DataArray, dict]],
    lead_days: list[int],
    suptitle: str,
    region_bounds: tuple[float, float, float | None, float | None] | None = None,
):
    \"\"\"data_by_row: {row_label: (DataArray with a prediction_timedelta dim, plot_kwargs)}.

    region_bounds: (south, north, west, east) -- same convention as
    subset_region() (west/east may be None for a latitude-only named region).
    Pass the notebook's region_bounds here to zoom the map to
    that box instead of the whole globe -- when the data only covers a small
    region, set_global() would draw the entire world with the actual (tiny)
    data patch invisible at that scale. Leave as None for the global run.

    Defined here (not in Step 7, where it's also used for POD/FAR/SEDI) since
    this cell's own H/M/F/C plot below needs it first -- defined once, used
    in both places.
    \"\"\"
    n_rows = len(data_by_row)
    n_cols = len(lead_days)
    figure, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    for row, (row_label, (data, plot_kwargs)) in enumerate(data_by_row.items()):
        for col, lead in enumerate(lead_days):
            axis = axes[row, col]
            lead_timedelta = np.timedelta64(lead, "D")
            data.sel(prediction_timedelta=lead_timedelta).plot(
                ax=axis, x="longitude", y="latitude", transform=ccrs.PlateCarree(),
                add_colorbar=(col == n_cols - 1), cbar_kwargs={"label": row_label} if col == n_cols - 1 else None,
                **plot_kwargs,
            )
            if region_bounds is not None:
                south, north, west, east = region_bounds
                # west/east are None for the latitude-only named regions
                # (tropics/nh_extratropics/sh_extratropics) -- zoom to the
                # full longitude width for those, restricted latitude band only.
                west = -180.0 if west is None else west
                east = 180.0 if east is None else east
                axis.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
            else:
                axis.set_global()
            axis.coastlines(linewidth=0.5)
            axis.set_title(f"{row_label}, lead={lead}d" if row == 0 else f"lead={lead}d")
    figure.suptitle(suptitle)
    figure.tight_layout()
    return figure


if by_lead_map_cache_path.exists():
    by_lead_map = xr.open_dataset(by_lead_map_cache_path)
    # Cached prediction_timedelta was saved as plain integer lead-days (see markdown
    # above for why) -- convert back to timedelta64 to match the rest of the notebook.
    by_lead_map = by_lead_map.assign_coords(
        prediction_timedelta=(by_lead_map["prediction_timedelta"].values * np.timedelta64(1, "D")).astype(
            "timedelta64[ns]"
        )
    )
    # Fail loudly here, not later as a cryptic KeyError from .sel(): a cache written
    # by an earlier run with different forecast_days/lead_days_to_plot (e.g. before
    # this config changed, even under the same region_label/variable) can
    # silently be missing a lead day this run wants.
    cached_leads = set(by_lead_map["prediction_timedelta"].values)
    wanted_leads = {np.timedelta64(day, "D").astype("timedelta64[ns]") for day in lead_days_to_plot}
    missing_leads = sorted(wanted_leads - cached_leads)
    if missing_leads:
        raise ValueError(
            f"{by_lead_map_cache_path} is stale: missing lead day(s) {missing_leads} "
            f"(cached file only has {sorted(cached_leads)}). It was likely written by a "
            "run with different forecast_days/lead_days_to_plot. Delete this file "
            "(by_lead_map_cache_path.unlink()) and re-run this cell."
        )
else:
    start_time = time.perf_counter()

    initialization_batch_size = 1
    summaries = mean_in_time_batches(
        model_test_year,
        calculate_scores,
        reductions={"by_lead_map": ("time",)},
        batch_size=initialization_batch_size,
    )
    by_lead_map = summaries["by_lead_map"]

    elapsed_seconds = time.perf_counter() - start_time
    log_run_timing(elapsed_seconds)

    # Save prediction_timedelta as plain integer lead-days, not timedelta64 -- see
    # markdown above for the netCDF encoding error this sidesteps.
    by_lead_map_to_save = by_lead_map.assign_coords(
        prediction_timedelta=(by_lead_map["prediction_timedelta"] / np.timedelta64(1, "D")).astype(int)
    )
    by_lead_map_to_save.to_netcdf(by_lead_map_cache_path)

by_lead_map"""
))

cells.append(nbf.v4.new_markdown_cell(
"""### H / M / F / C maps

Before reducing `by_lead_map`'s hit/miss/false-alarm/correct-negative
indicators down to POD/FAR/SEDI (Step 5), plot the four raw categories
themselves -- each cell of `by_lead_map` is the *fraction* of initializations
classified as that category (a mean of mutually-exclusive 0/1 indicators, so
all four sum to ~1 at every valid grid cell/lead day -- Step 6 checks this
explicitly), which is worth seeing directly: e.g. a region with almost no
Hits or False alarms just means the event essentially never happens there,
which changes how much to read into that region's POD/FAR/SEDI at all.

Same up-to-two-figures pattern as Step 7's POD/FAR/SEDI plots (one for the
absolute threshold whenever `has_absolute_threshold` is true, one for the
relative threshold whenever `has_relative_climatology` is true, each skipped
with a printed message when it doesn't apply to `variable`) and the same
`plot_metric_grid`/`select_by_lead_day` helpers (defined above, in this
step, precisely so this cell could use them too) -- just 4 rows (H, M, F, C)
instead of 3 (POD, FAR, SEDI). Also saved as a PNG, same naming convention as
Step 7's figures plus an `hmfc` tag so they don't collide with those."""
))

cells.append(nbf.v4.new_code_cell(
"""_hmfc_row_labels = {
    "hits": "Hits (H)",
    "misses": "Misses (M)",
    "false_alarms": "False alarms (F)",
    "correct_negatives": "Correct negatives (C)",
}

if has_absolute_threshold:
    absolute_hmfc_by_lead = select_by_lead_day(
        by_lead_map[[f"absolute_{name}" for name in _hmfc_row_labels]], lead_days_to_plot
    )
    absolute_hmfc_figure = plot_metric_grid(
        {
            label: (absolute_hmfc_by_lead[f"absolute_{name}"], {"cmap": "viridis", "vmin": 0, "vmax": 1})
            for name, label in _hmfc_row_labels.items()
        },
        lead_days_to_plot,
        f"{variable} > {absolute_threshold} degC -- H/M/F/C, {test_year_start}-{test_year_end}",
        region_bounds=region_bounds,
    )
    absolute_hmfc_figure.savefig(
        f"{model_source}_absolute_hmfc_maps_{test_year_start}_{test_year_end}_{region_label}_{variable}.png",
        dpi=150,
    )
else:
    print(f"No absolute threshold for variable={variable!r} -- skipping absolute H/M/F/C maps.")"""
))

cells.append(nbf.v4.new_code_cell(
"""if has_relative_climatology:
    relative_hmfc_by_lead = select_by_lead_day(
        by_lead_map[[f"relative_{name}" for name in _hmfc_row_labels]], lead_days_to_plot
    )
    relative_hmfc_figure = plot_metric_grid(
        {
            label: (relative_hmfc_by_lead[f"relative_{name}"], {"cmap": "viridis", "vmin": 0, "vmax": 1})
            for name, label in _hmfc_row_labels.items()
        },
        lead_days_to_plot,
        f"{variable} > {relative_percentile} percentile climatology -- H/M/F/C, {test_year_start}-{test_year_end}",
        region_bounds=region_bounds,
    )
    relative_hmfc_figure.savefig(
        f"{model_source}_relative_hmfc_maps_{test_year_start}_{test_year_end}_{region_label}_{variable}_rel{relative_percentile}.png",
        dpi=150,
    )
else:
    print(f"No relative threshold for variable={variable!r} (no precomputed climatology) -- skipping relative H/M/F/C maps.")"""
))

# --- Step 5: Calculate POD, FAR, SEDI ---------------------------------------

cells.append(nbf.v4.new_markdown_cell(
"""## Step 5: Calculate POD, FAR, SEDI

`by_lead_map` holds batch-averaged (not summed) indicators; because these
are means of 0/1 indicators rather than raw counts, the rate functions
(`probability_of_detection`, etc.) give identical results to computing them
from summed contingency counts directly (see `deterministic_metrics.py`'s
module docstring). `finish_event_scores` is called for `"absolute"` whenever
`has_absolute_threshold` is true, and again for `"relative"` whenever
`has_relative_climatology` is true -- both, one, or (in principle) neither,
depending on `variable` (Step 0). (RMSE is still commented out -- uncomment
along with `squared_error` above to bring it back.)

**Lead-day bins are calendar-aligned** (0-24h, 24-48h, ... since
initialization), via `aifs_singlev2.daily_aifs_aggregates_calendar_aligned`
(see its docstring): AIFS-single-v2 has no t+0h step, so the plain
`resample(...).mean()` used by `daily_aifs_aggregates` anchors bins to
whichever step comes first in the archive -- 6h, 30h, 54h, ... -- not to
lead time zero. `daily_aifs_aggregates_calendar_aligned` bins explicitly via
`floor(step / 1 day) + 1` instead, so `prediction_timedelta` labels land
exactly on `1 days`, `2 days`, `3 days`, .... The `+1` (end-of-window, not
start-of-window) and the calendar alignment itself both match the real
store's own `prediction_timedelta_daily` coordinate -- confirmed directly
against a real AIFS-single-v2 `Dataset` repr, which indexes
`total_precipitation` by `prediction_timedelta_daily` (`1 days` through
`50 days`) as a genuinely separate, already-daily coordinate from
`2m_temperature`'s 6-hourly `prediction_timedelta` (confirmed by the same
repr's `Data variables:` section, which is what actually settled this after
some earlier back-and-forth about whether precipitation shared temperature's
6-hourly dimension -- it doesn't). This was confirmed against
AIFS-single-v2's store specifically, not AIFS-ENS-v2's -- assumed to hold
for the ensemble store too, since it's presumably the same underlying model
run with an added `"number"` dimension, but not independently re-confirmed
after switching model source here. One caveat this doesn't remove: **day 1
is a partial bin** for temperature specifically (3 of the usual 4 six-hourly
samples -- 6h, 12h, 18h -- since there's no 0h sample to fill the first
slot); every day after that is a complete 4-sample bin. This caveat doesn't
apply to `total_precipitation`, which needs no aggregation here at all.

`select_by_lead_day` (defined in Step 4, alongside `plot_metric_grid` --
both needed there already, for the H/M/F/C maps, so this step just reuses
them) still selects with `method="nearest"` (12-hour `tolerance`) rather
than an exact match, as a defensive fallback -- with calendar-aligned bins
this should always match exactly, but if a lead day is ever genuinely
missing (e.g. a short rollout), this raises a clear error instead of a
silent mismatch."""
))

cells.append(nbf.v4.new_code_cell(
"""def finish_event_scores(means: xr.Dataset, prefix: str) -> xr.Dataset:
    \"\"\"POD/miss-rate/FAR/POFD/SEDI for one threshold definition's indicator means.\"\"\"
    counts = means[[f"{prefix}_{name}" for name in ("hits", "misses", "false_alarms", "correct_negatives")]]
    counts = counts.rename({f"{prefix}_{name}": name for name in ("hits", "misses", "false_alarms", "correct_negatives")})
    pod = probability_of_detection(counts)
    return xr.Dataset(
        {
            "pod": pod,
            "miss_rate": 1 - pod,
            "far": false_alarm_ratio(counts),
            "pofd": probability_of_false_detection(counts),
            "sedi": symmetric_extremal_dependence_index(counts),
        }
    )


# select_by_lead_day is defined in Step 4 (this cell reuses it -- also needed
# there for the H/M/F/C plot).
# rmse_map = rmse_from_mean_squared_error(by_lead_map["squared_error"]).rename("rmse")  # RMSE commented out for now
if has_absolute_threshold:
    absolute_scores = finish_event_scores(by_lead_map, "absolute")
    absolute_by_lead = select_by_lead_day(absolute_scores, lead_days_to_plot)
else:
    absolute_scores = None
    absolute_by_lead = None

# rmse_by_lead = select_by_lead_day(rmse_map, lead_days_to_plot)  # RMSE commented out for now

if has_relative_climatology:
    relative_scores = finish_event_scores(by_lead_map, "relative")
    relative_by_lead = select_by_lead_day(relative_scores, lead_days_to_plot)
else:
    relative_scores = None
    relative_by_lead = None

absolute_by_lead if has_absolute_threshold else relative_by_lead"""
))

# --- Step 6: Sanity checks ---------------------------------------------------

cells.append(nbf.v4.new_markdown_cell(
"""## Step 6: Sanity checks

Cheap checks on the results themselves, before reading too much into the
maps in Step 7:

1. **The four categories should sum to ~1** at every valid cell (`by_lead_map`
   holds means of mutually-exclusive, exhaustive 0/1 indicators).
2. **Range check**: POD/FAR/POFD in `[0, 1]`, SEDI in `[-1, 1]`.
3. **Skill should not improve with lead time.** POD/SEDI should trend down
   (or stay flat) from the shortest to the longest lead day plotted, FAR up
   or flat -- a strong reversal is a red flag, not just an odd result.
4. **NaN fraction**: how much of the map is actually defined. If POD/FAR are
   only defined over a tiny sliver, the maps in Step 7 are mostly
   sample-size noise, not signal."""
))

cells.append(nbf.v4.new_code_cell(
"""# Run every check for whichever threshold definition(s) are active for this
# variable -- same checks, just per threshold definition, since
# by_lead_map/*_scores/*_by_lead hold each independently.
scores_by_prefix = {}
by_lead_by_prefix = {}
if has_absolute_threshold:
    scores_by_prefix["absolute"] = absolute_scores
    by_lead_by_prefix["absolute"] = absolute_by_lead
if has_relative_climatology:
    scores_by_prefix["relative"] = relative_scores
    by_lead_by_prefix["relative"] = relative_by_lead

# 1. Four categories sum to ~1
for prefix in scores_by_prefix:
    category_total = sum(
        by_lead_map[f"{prefix}_{name}"] for name in ("hits", "misses", "false_alarms", "correct_negatives")
    )
    print(f"[{prefix}] category sum -- min:", float(category_total.min()), "max:", float(category_total.max()), "(expect ~1.0, or NaN)")
print()

# 2. Range check
for prefix, scores in scores_by_prefix.items():
    for name in ("pod", "far", "pofd", "sedi"):
        values = scores[name]
        print(f"[{prefix}] {name}: min={float(values.min()):.3f} max={float(values.max()):.3f}")
print()

# 3. Trend across lead days (only informative with more than one lead day plotted)
print("spatial-mean by lead day (days:", lead_days_to_plot, "):")
for prefix, by_lead in by_lead_by_prefix.items():
    for name in ("pod", "far", "sedi"):
        means = by_lead[name].mean(dim=["latitude", "longitude"], skipna=True)
        print(f"  [{prefix}] {name}:", [round(float(v), 3) for v in means.values])
print()

# 4. NaN fraction
for prefix, by_lead in by_lead_by_prefix.items():
    valid_fraction = by_lead["pod"].notnull().mean(dim=["latitude", "longitude"])
    print(f"[{prefix}] fraction of map with a defined POD, by lead day:", [round(float(v), 3) for v in valid_fraction.values])"""
))

# --- Step 7: Plotting --------------------------------------------------------

cells.append(nbf.v4.new_markdown_cell(
"""## Step 7: Plotting

Up to two identically-laid-out figures (POD / FAR / SEDI, 3 rows x N lead
days each): one for the absolute threshold whenever `has_absolute_threshold`
is true, one for the relative threshold whenever `has_relative_climatology`
is true -- each skipped, with a printed message instead, when it doesn't
apply to the selected `variable` (e.g. absolute is skipped for
`total_precipitation`; relative is skipped for `t2m_mean_6h`). Same
`plot_metric_grid` helper as Step 4's H/M/F/C maps (defined there, reused
here, not redefined). RMSE's figure is still commented out for now
(`plot_metric_grid` is still usable for it, just not called).

Each figure is also saved as a PNG (filename includes `region_label`,
`variable`, and -- for the relative figure -- the percentile used, so plots
from different regions, year ranges, variables, or percentiles never
overwrite each other), so you have the maps as a persistent file without
re-running the notebook."""
))

cells.append(nbf.v4.new_code_cell(
"""# plot_metric_grid is defined in Step 4 (needed there first, for the H/M/F/C
# maps) -- this step just reuses it, same as select_by_lead_day in Step 5.

# RMSE commented out for now -- uncomment (and the rmse_map/rmse_by_lead lines in Step 5) to bring it back:
# rmse_figure = plot_metric_grid(
#     {"RMSE": (rmse_by_lead, {"cmap": "magma_r", "vmin": 0})},
#     lead_days_to_plot,
#     f"{model_source} vs ERA5, {variable}, {test_year_start}-{test_year_end} -- RMSE",
#     region_bounds=region_bounds,
# )
# rmse_figure.savefig(f"{model_source}_rmse_maps_{test_year_start}_{test_year_end}_{region_label}_{variable}.png", dpi=150)"""
))

cells.append(nbf.v4.new_code_cell(
"""if has_absolute_threshold:
    absolute_figure = plot_metric_grid(
        {
            "POD": (absolute_by_lead["pod"], {"cmap": "viridis", "vmin": 0, "vmax": 1}),
            "FAR": (absolute_by_lead["far"], {"cmap": "viridis", "vmin": 0, "vmax": 1}),
            "SEDI": (absolute_by_lead["sedi"], {"cmap": "RdBu_r", "vmin": -1, "vmax": 1}),
        },
        lead_days_to_plot,
        f"{variable} > {absolute_threshold} degC, {test_year_start}-{test_year_end}",
        region_bounds=region_bounds,
    )
    absolute_figure.savefig(
        f"{model_source}_absolute_maps_{test_year_start}_{test_year_end}_{region_label}_{variable}.png",
        dpi=150,
    )
else:
    print(f"No absolute threshold for variable={variable!r} -- skipping absolute maps.")"""
))

cells.append(nbf.v4.new_code_cell(
"""if has_relative_climatology:
    relative_figure = plot_metric_grid(
        {
            "POD": (relative_by_lead["pod"], {"cmap": "viridis", "vmin": 0, "vmax": 1}),
            "FAR": (relative_by_lead["far"], {"cmap": "viridis", "vmin": 0, "vmax": 1}),
            "SEDI": (relative_by_lead["sedi"], {"cmap": "RdBu_r", "vmin": -1, "vmax": 1}),
        },
        lead_days_to_plot,
        f"{variable} > {relative_percentile} percentile of 1979-2018 climatology, {test_year_start}-{test_year_end}",
        region_bounds=region_bounds,
    )
    relative_figure.savefig(
        f"{model_source}_relative_maps_{test_year_start}_{test_year_end}_{region_label}_{variable}_rel{relative_percentile}.png",
        dpi=150,
    )
else:
    print(f"No relative threshold for variable={variable!r} (no precomputed climatology) -- skipping relative maps.")"""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("deterministic_verification_metrics.ipynb", "w") as f:
    nbf.write(nb, f)

print("wrote notebook with", len(cells), "cells")
