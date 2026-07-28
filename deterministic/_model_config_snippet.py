import os
from pathlib import Path

def _env(name: str, default=None):
    """Like os.environ.get, but treats an unset or empty-string variable
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
    """
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