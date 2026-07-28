"""Loader for the deterministic AIFS-single-v2 reforecast.

`open_aifs_singlev2()` is adapted from the `open_aifs_ensv2()` used in
`ensemble_verification_metrics.md` (reproduced below, unmodified, as
`open_aifs_ensv2_reference`, for comparison) to point at the AIFS-single-v2
store instead of AIFS-ENS-v2, per "Rossby Model Storage Locations -
Sheet1.csv":

    /net/monsoon/marchakitus/reforecast/forecasts_AIFS_v2

The only substantive change is dropping the `"number"` (ensemble member)
chunk spec, since AIFS-single-v2 is a genuinely deterministic run with no
ensemble-member dimension -- everything else (glob pattern, concat_dim,
variable selection/renaming) is copied as-is.

CAVEAT: this has not been run against the real store (no access to
/net/monsoon from this environment). The short variable names (`2d`, `2t`,
`tp`) are assumed to match the ensemble store's layout; confirm against the
real path before trusting this, and adjust the chunks/rename map if the
single-run store is organized differently. The glob/filename pattern
(`init_YYYYMMDDT00.zarr`, one store per initialization) *is* confirmed --
it was read directly from `ls` output on the real directory.

`daily_aifs_aggregates()` is reproduced unmodified: it only touches the
named `variable`/`step_dim`, never the ensemble-member dimension, so it
works unchanged whether or not a `"number"` dimension is present.
"""

import re
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

import wetbulb  # local, no heatextremes dependency -- see wetbulb.py

# Confirmed real filename pattern (from `ls` on the actual store):
# init_20000101T00.zarr, init_20000107T00.zarr, ...
_INIT_STORE_FILENAME = re.compile(r"^init_(\d{4})(\d{2})(\d{2})T\d{2}\.zarr$")


def _init_year(path: Path) -> int:
    """Parse the initialization year from an `init_YYYYMMDDT00.zarr` store name."""
    match = _INIT_STORE_FILENAME.match(path.name)
    if not match:
        raise ValueError(
            f"Unrecognized AIFS-single-v2 store filename: {path.name!r} "
            "(expected 'init_YYYYMMDDT00.zarr')"
        )
    return int(match.group(1))


def open_aifs_singlev2(
    start_year: int | None = None,
    end_year: int | None = None,
) -> xr.Dataset:
    """Open the deterministic AIFS-single-v2 reforecast.

    Deterministic counterpart to `open_aifs_ensv2_reference` below: same
    store layout, variable selection, and renaming, but no ensemble
    `"number"` dimension/chunk, and pointed at the AIFS-single-v2 path from
    "Rossby Model Storage Locations - Sheet1.csv"
    (`/net/monsoon/marchakitus/reforecast/forecasts_AIFS_v2`) instead of
    AIFS-ENS-v2's path.

    Parameters
    ----------
    start_year, end_year : int, optional
        If given, only initializations with `start_year <= year <= end_year`
        are opened (each bound is independently optional; pass just one to
        get an open-ended range). The year is parsed from each store's
        filename (`init_YYYYMMDDT00.zarr`) *before* calling
        `xr.open_mfdataset`, so -- unlike filtering with `.sel(time=...)`
        after opening the full archive -- this actually avoids opening
        zarr metadata for the stores you don't need. Omit both (the
        default) to open every available initialization, as before.
    """
    root = Path("/net/monsoon/marchakitus/reforecast/forecasts_AIFS_v2")
    paths = sorted(root.glob("*.zarr"))
    for path in paths:
        assert path.is_dir(), f"{path} does not exist?"

    if start_year is not None or end_year is not None:
        paths = [
            path
            for path in paths
            if (start_year is None or _init_year(path) >= start_year)
            and (end_year is None or _init_year(path) <= end_year)
        ]
        if not paths:
            raise FileNotFoundError(
                f"No AIFS-single-v2 initialization stores found under {root} "
                f"for start_year={start_year}, end_year={end_year}"
            )

    wanted = ["2d", "2t", "tp"]
    with ProgressBar():
        ds = xr.open_mfdataset(
            paths,
            engine="zarr",
            combine="nested",
            concat_dim="time",
            preprocess=lambda x: x[wanted],
            chunks={
                "time": 1,  # unavoidable: one time per store
                "prediction_timedelta": 24,
                "latitude": 180,
                "longitude": 180,
            },
            parallel=True,
            data_vars="all",
            coords="minimal",
            compat="override",
            join="override",
            combine_attrs="override",
            consolidated=None,
        )
    return ds.rename(
        {
            "2d": "2m_dewpoint_temperature",
            "2t": "2m_temperature",
            "tp": "total_precipitation",
            "lat": "latitude",
            "lon": "longitude",
        }
    )


def open_aifs_ensv2_reference() -> xr.Dataset:
    """Unmodified copy of the ensemble loader, kept only for comparison.

    This is exactly `open_aifs_ensv2()` from `ensemble_verification_metrics.md`
    (AIFS-ENS-v2, with the ensemble `"number"` chunk) -- not meant to be
    called from the deterministic notebook; see `open_aifs_singlev2` above.
    """
    root = Path("/net/monsoon/marchakitus/AIFS/v2p0/combined/forecasts_AIFS_ENS_v2")
    paths = sorted(root.glob("*.zarr"))
    for path in paths:
        assert path.is_dir(), f"{path} does not exist?"
    wanted = ["2d", "2t", "tp"]
    with ProgressBar():
        ds = xr.open_mfdataset(
            paths,
            engine="zarr",
            combine="nested",
            concat_dim="time",
            preprocess=lambda x: x[wanted],
            chunks={
                "time": 1,  # unavoidable: one time per store
                "number": 26,  # combine all ensemble members
                "prediction_timedelta": 24,
                "latitude": 180,
                "longitude": 180,
            },
            parallel=True,
            data_vars="all",
            coords="minimal",
            compat="override",
            join="override",
            combine_attrs="override",
            consolidated=None,
        )
    return ds.rename(
        {
            "2d": "2m_dewpoint_temperature",
            "2t": "2m_temperature",
            "tp": "total_precipitation",
            "lat": "latitude",
            "lon": "longitude",
        }
    )


def open_aifs_ensv2(
    start_year: int | None = None,
    end_year: int | None = None,
) -> xr.Dataset:
    """Open the AIFS-ENS-v2 ensemble reforecast (26 members), year-filtered.

    Adapted from `open_aifs_ensv2_reference` above the same way
    `open_aifs_singlev2` adapts it for the deterministic store: adds
    `start_year`/`end_year` filename-based filtering (same
    `_init_year`/`_INIT_STORE_FILENAME` helpers used there) so a restricted
    year range doesn't need opening all ~25 years of zarr metadata first.
    Everything else -- glob pattern, `wanted` variables, chunking (including
    the `"number"` ensemble-member chunk, unlike `open_aifs_singlev2`, which
    drops it), renaming -- is unchanged from `open_aifs_ensv2_reference`.

    CAVEAT: the `start_year`/`end_year` filtering assumes this store's
    filenames follow the same `init_YYYYMMDDT00.zarr` convention confirmed
    for AIFS-single-v2's store (confirmed via `ls` on *that* path
    specifically, not this ensemble path). If AIFS-ENS-v2's real filenames
    differ, `_init_year` raises a clear `ValueError` rather than silently
    doing the wrong thing -- so this is safe to try, just not independently
    confirmed the way the deterministic store's naming was.

    Used by the deterministic-verification notebook's `ensemble_mean()`
    (below): the notebook calls `open_aifs_ensv2(...)`, then immediately
    `ensemble_mean(...)` to collapse the `"number"` dimension away, so
    everything downstream (`daily_aifs_aggregates_calendar_aligned`,
    `daily_aifs_precipitation`, `daily_aifs_wet_bulb_calendar_aligned`,
    `calculate_scores`, etc.) still only ever sees a single deterministic-like
    field per (time, prediction_timedelta, latitude, longitude) -- none of
    that code needed to change to support this.
    """
    root = Path("/net/monsoon/marchakitus/AIFS/v2p0/combined/forecasts_AIFS_ENS_v2")
    paths = sorted(root.glob("*.zarr"))
    for path in paths:
        assert path.is_dir(), f"{path} does not exist?"

    if start_year is not None or end_year is not None:
        paths = [
            path
            for path in paths
            if (start_year is None or _init_year(path) >= start_year)
            and (end_year is None or _init_year(path) <= end_year)
        ]
        if not paths:
            raise FileNotFoundError(
                f"No AIFS-ENS-v2 initialization stores found under {root} "
                f"for start_year={start_year}, end_year={end_year}"
            )

    wanted = ["2d", "2t", "tp"]
    with ProgressBar():
        ds = xr.open_mfdataset(
            paths,
            engine="zarr",
            combine="nested",
            concat_dim="time",
            preprocess=lambda x: x[wanted],
            chunks={
                "time": 1,  # unavoidable: one time per store
                "number": 26,  # combine all ensemble members
                "prediction_timedelta": 24,
                "latitude": 180,
                "longitude": 180,
            },
            parallel=True,
            data_vars="all",
            coords="minimal",
            compat="override",
            join="override",
            combine_attrs="override",
            consolidated=None,
        )
    return ds.rename(
        {
            "2d": "2m_dewpoint_temperature",
            "2t": "2m_temperature",
            "tp": "total_precipitation",
            "lat": "latitude",
            "lon": "longitude",
        }
    )


def ensemble_mean(ds: xr.Dataset, member_dim: str = "number") -> xr.Dataset:
    """Collapse an ensemble dataset to its across-member mean.

    Reduces AIFS-ENS-v2's 26-member ensemble down to one deterministic-like
    field per (time, prediction_timedelta, latitude, longitude) -- the
    "ensemble mean" forecast -- dropping `member_dim` entirely, so the result
    can flow through the exact same deterministic-verification pipeline
    originally written for AIFS-single-v2's single run
    (`daily_aifs_aggregates_calendar_aligned`, `daily_aifs_precipitation`,
    `daily_aifs_wet_bulb_calendar_aligned`, `calculate_scores`, etc.) with no
    changes to any of that code -- none of it needs to know an ensemble was
    ever involved.

    Note the ensemble mean is a smoothed, less-extreme field than any
    individual member or a true single-realization deterministic run:
    averaging 26 members together cancels out each member's own
    unpredictable small-scale detail, which tends to suppress the tails of
    the distribution. So expect systematically lower POD (and likely lower
    FAR too) for extreme-threshold events than AIFS-single-v2 gave -- that's
    an expected property of verifying an ensemble mean against extremes, not
    a bug.
    """
    if member_dim not in ds.dims:
        raise ValueError(f"Dataset must have a {member_dim!r} dimension")
    return ds.mean(dim=member_dim, skipna=True)


def daily_aifs_aggregates(
    ds: xr.Dataset,
    max_days: int | None = None,
    variable: str = "2m_temperature",
    step_dim: str = "prediction_timedelta",
    output_step_dim: str = "prediction_timedelta",
) -> xr.Dataset:
    """Return daily mean and maximum temperature from 6-hourly forecasts.

    Unmodified from the ensemble notebook: operates only on `variable` /
    `step_dim`, so it is unaffected by whether a `"number"` (ensemble
    member) dimension is present, and needs no changes for AIFS-single-v2.
    """
    if variable not in ds:
        raise KeyError(f"Dataset is missing {variable}")
    if step_dim not in ds.dims:
        raise ValueError(f"Dataset must have a {step_dim} dimension")
    if step_dim not in ds.coords:
        raise ValueError(f"Dataset must have a {step_dim} coordinate")
    temperature = ds[variable]
    if max_days is not None:
        if max_days < 1:
            raise ValueError("max_days must be at least 1")
        temperature = temperature.where(
            temperature[step_dim] < np.timedelta64(max_days, "D"),
            drop=True,
        )
    daily = xr.Dataset(
        {
            "t2m_mean_6h": temperature.resample(**{step_dim: "1D"}).mean(),
            "t2m_max_6h": temperature.resample(**{step_dim: "1D"}).max(),
        }
    )
    if step_dim != output_step_dim:
        daily = daily.rename({step_dim: output_step_dim})
    return daily


def daily_aifs_aggregates_calendar_aligned(
    ds: xr.Dataset,
    max_days: int | None = None,
    variable: str = "2m_temperature",
    step_dim: str = "prediction_timedelta",
    output_step_dim: str = "prediction_timedelta",
) -> xr.Dataset:
    """Like `daily_aifs_aggregates`, but bins to calendar-day boundaries since
    initialization (0-24h, 24-48h, 48-72h, ...) rather than to whichever step
    happens to come first in the archive, using the *same* end-of-window
    labeling convention as the store's own `prediction_timedelta_daily`
    coordinate (confirmed directly against a real AIFS-single-v2 `Dataset`
    repr: `total_precipitation` is natively indexed by
    `prediction_timedelta_daily`, running `1 days` through `50 days` -- i.e.
    label `N days` means the window `[(N-1)*24h, N*24h)`, not
    `[N*24h, (N+1)*24h)`). `2m_temperature` itself is only stored at raw
    6-hourly resolution in the real store (indexed by `prediction_timedelta`,
    not `prediction_timedelta_daily` -- confirmed by the same repr), so it
    still needs aggregating here -- this just makes sure the labels mean the
    same thing as they do for precipitation in the same store (see
    `daily_aifs_precipitation` below, which reads `total_precipitation`
    directly off its own daily coordinate -- no resampling needed there).

    AIFS-single-v2 has no t+0h step (its first output is t+6h), so plain
    `resample(...).mean()` -- what `daily_aifs_aggregates` uses -- anchors its
    bins to the first *available* step instead of to lead time zero, giving
    bins at 6h-30h, 30h-54h, 54h-78h, etc. Confirmed directly: none of
    xarray's built-in `resample(origin=...)` options ('epoch', 'start',
    'start_day', 'end', 'end_day') force calendar alignment for a timedelta64
    coordinate -- every one of them reduces to anchoring on the data's own
    first/last timestamp, the same as passing no `origin` at all. So this
    bins explicitly via `floor(step / 1 day) + 1` instead of `resample`.

    Caveat: because there is no t+0h step, day `1` (window `[0h, 24h)`) is
    necessarily a partial bin -- only 3 of the usual 4 six-hourly samples
    (6h, 12h, 18h; missing the 0h sample that doesn't exist). Day `2` onward
    are complete 4-sample bins (e.g. day 2 = 24h, 30h, 36h, 42h). This is a
    genuine property of the archive, not something fixable in code -- and
    it's specific to `2m_temperature`'s 6-hourly storage, so it doesn't apply
    to `total_precipitation` (already daily, no aggregation happening here
    at all for that variable).

    Not "unmodified from the ensemble notebook" like `daily_aifs_aggregates`
    above -- this is new, written to get calendar-aligned lead-day labels for
    the deterministic notebook's `.sel(prediction_timedelta=...)` calls.

    Also returns `t2m_min_6h` (daily minimum) alongside the original
    `t2m_mean_6h`/`t2m_max_6h`, so the notebook's `variable` config can select
    daily min as well as mean/max -- see `era5_loader.daily_era5_aggregates`,
    which got the same addition for the same reason.
    """
    if variable not in ds:
        raise KeyError(f"Dataset is missing {variable}")
    if step_dim not in ds.dims:
        raise ValueError(f"Dataset must have a {step_dim} dimension")
    if step_dim not in ds.coords:
        raise ValueError(f"Dataset must have a {step_dim} coordinate")
    temperature = ds[variable]
    if max_days is not None:
        if max_days < 1:
            raise ValueError("max_days must be at least 1")
        temperature = temperature.where(
            temperature[step_dim] < np.timedelta64(max_days, "D"),
            drop=True,
        )

    # +1: end-of-window labeling, matching prediction_timedelta_daily's own
    # convention -- floor(step/1day) alone would label [0h,24h) as "0 days"
    # (start-of-window), but the real store calls that window "1 days".
    lead_day = (temperature[step_dim] // np.timedelta64(1, "D")).astype(int) + 1
    temperature = temperature.assign_coords(lead_day=(step_dim, lead_day.data))
    grouped = temperature.groupby("lead_day")

    daily = xr.Dataset(
        {
            "t2m_mean_6h": grouped.mean(),
            "t2m_max_6h": grouped.max(),
            "t2m_min_6h": grouped.min(),
        }
    )
    daily = daily.assign_coords(
        lead_day=daily["lead_day"].values.astype("timedelta64[D]").astype("timedelta64[ns]")
    ).rename({"lead_day": output_step_dim})
    return daily


def daily_aifs_precipitation(
    ds: xr.Dataset,
    max_days: int | None = None,
    step_dim: str = "prediction_timedelta_daily",
    output_step_dim: str = "prediction_timedelta",
) -> xr.Dataset:
    """Return `total_precipitation` already at daily resolution -- no aggregation needed.

    CONFIRMED against a real AIFS-single-v2 `Dataset` repr (`Dimensions:`
    `time: 91, prediction_timedelta: 200, latitude: 721, longitude: 1440,
    prediction_timedelta_daily: 50`; `Data variables:` showing
    `2m_temperature (time, prediction_timedelta, latitude, longitude)` but
    `total_precipitation (time, prediction_timedelta_daily, latitude,
    longitude)`): unlike `2m_temperature`/`2m_dewpoint_temperature` (only
    ever stored 6-hourly, on `prediction_timedelta`, hence
    `daily_aifs_aggregates_calendar_aligned` above), `total_precipitation` is
    natively daily, on its own separate `prediction_timedelta_daily`
    coordinate (`1 days` through `50 days` -- no `0 days`, i.e. no partial
    first-day ambiguity to resolve here the way there is for temperature).

    (Earlier versions of this function/docstring went back and forth on this
    -- briefly "corrected" to assume precipitation shared temperature's
    6-hourly dimension and needed summing -- before this real `Dataset` repr
    settled it. This version, reading `total_precipitation` directly off its
    own daily coordinate with no resampling, is the one confirmed against
    actual data.)

    This function just selects and (optionally) trims that existing daily
    variable and renames its dimension to `output_step_dim`, so downstream
    code (`calculate_scores`'s `compute_model_var`, in particular) can treat
    `total_precipitation` the same way as the temperature variables --
    indexed by a dimension named `prediction_timedelta`, regardless of which
    one is actually selected.
    """
    if "total_precipitation" not in ds:
        raise KeyError("Dataset is missing total_precipitation")
    if step_dim not in ds.dims:
        raise ValueError(f"Dataset must have a {step_dim} dimension")
    if step_dim not in ds.coords:
        raise ValueError(f"Dataset must have a {step_dim} coordinate")
    precipitation = ds["total_precipitation"]
    if max_days is not None:
        if max_days < 1:
            raise ValueError("max_days must be at least 1")
        precipitation = precipitation.where(
            precipitation[step_dim] <= np.timedelta64(max_days, "D"),
            drop=True,
        )
    daily = xr.Dataset({"total_precipitation": precipitation})
    if step_dim != output_step_dim:
        daily = daily.rename({step_dim: output_step_dim})
    return daily


def daily_aifs_wet_bulb_calendar_aligned(
    ds: xr.Dataset,
    max_days: int | None = None,
    step_dim: str = "prediction_timedelta",
    output_step_dim: str = "prediction_timedelta",
) -> xr.Dataset:
    """Daily mean/max/min wet-bulb temperature (Stull 2011), calendar-aligned.

    Same day-boundary convention as `daily_aifs_aggregates_calendar_aligned`
    above (`+1`/end-of-window labeling, confirmed against the real store's
    `prediction_timedelta_daily`; day 1 is a genuine partial 3-sample bin --
    see that function's docstring for the full justification, which applies
    unchanged here since wet-bulb temperature is derived from the same
    6-hourly `2m_temperature`/`2m_dewpoint_temperature` fields, not a
    separately-stored daily variable the way `total_precipitation` is).

    Takes the RAW `ds` -- both `2m_temperature` and `2m_dewpoint_temperature`
    still in Kelvin, as they come straight off `open_aifs_singlev2`, *not*
    already Kelvin-converted the way `_build_notebook.py`'s Step 2 cell
    converts `2m_temperature` for the plain temperature variables.
    `wetbulb.wet_bulb_temperature(..., input_units="K")` does its own
    Kelvin-to-Celsius conversion internally, so an already-converted
    `2m_temperature` here would be double-converted. See
    `compute_model_var` in `_build_notebook.py`'s Step 2 cell: it skips the
    in-place `2m_temperature -= 273.15` step whenever a wet-bulb variable is
    selected, specifically so `model_batch` stays raw by the time this
    function sees it.

    Only an absolute threshold is supported for the resulting
    `t_wb_2m_mean_6h`/`t_wb_2m_max_6h`/`t_wb_2m_min_6h` in the notebook --
    see `wetbulb.py`'s module docstring and `era5_loader.daily_era5_wet_bulb_aggregates`
    for why there's no relative/climatology-percentile option for this
    variable yet.
    """
    required = {"2m_temperature", "2m_dewpoint_temperature"}
    missing = required - set(ds.data_vars)
    if missing:
        raise KeyError(f"Dataset is missing required variables: {sorted(missing)}")
    if step_dim not in ds.dims:
        raise ValueError(f"Dataset must have a {step_dim} dimension")
    if step_dim not in ds.coords:
        raise ValueError(f"Dataset must have a {step_dim} coordinate")

    wet_bulb = wetbulb.wet_bulb_temperature(
        ds["2m_temperature"], ds["2m_dewpoint_temperature"], input_units="K"
    )
    if max_days is not None:
        if max_days < 1:
            raise ValueError("max_days must be at least 1")
        wet_bulb = wet_bulb.where(
            wet_bulb[step_dim] < np.timedelta64(max_days, "D"),
            drop=True,
        )

    # +1: end-of-window labeling -- see daily_aifs_aggregates_calendar_aligned
    # above for the full justification (identical here).
    lead_day = (wet_bulb[step_dim] // np.timedelta64(1, "D")).astype(int) + 1
    wet_bulb = wet_bulb.assign_coords(lead_day=(step_dim, lead_day.data))
    grouped = wet_bulb.groupby("lead_day")

    daily = xr.Dataset(
        {
            "t_wb_2m_mean_6h": grouped.mean(),
            "t_wb_2m_max_6h": grouped.max(),
            "t_wb_2m_min_6h": grouped.min(),
        }
    )
    daily = daily.assign_coords(
        lead_day=daily["lead_day"].values.astype("timedelta64[D]").astype("timedelta64[ns]")
    ).rename({"lead_day": output_step_dim})
    return daily
