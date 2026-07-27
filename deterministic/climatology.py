"""Local (per grid-cell, day-of-year) climatological percentile thresholds.

Used to define a "relative" extreme-event threshold -- e.g. "daily max T2M
above the 99th percentile of local climatology" -- as an alternative to a
fixed absolute threshold (e.g. 35 degrees C), for use with
``deterministic_metrics.extreme_indicators`` /
``deterministic_metrics.contingency_counts``.

Two ways to get a ``threshold_by_doy`` array (the ``dayofyear`` + spatial
dims input ``threshold_at_verification_time`` below expects):

- ``open_percentile_climatology``: load one of Aaron Schwartz's precomputed
  1979-2018 daily percentile climatologies (``/net/monsoon/aasch/percentiles/``)
  and select a quantile -- fast, no computation, but only available for
  ``t2m_max_6h``/``t2m_min_6h``/``total_precipitation`` (there's no
  precomputed file for the daily mean). This is what the
  deterministic-verification notebook actually uses.
- ``local_climatology_quantile``: compute one from scratch from a raw daily
  baseline record -- expensive (one quantile reduction per day of year, over
  however many years/however much of the globe the baseline covers), but
  works for any variable, including ones without a precomputed file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from deterministic_metrics import verification_time  # local, no heatextremes dependency (see that module)


CLIMATOLOGY_PATHS = {
    "t2m_max_6h": Path("/net/monsoon/aasch/percentiles/percentiles_1979-2018_2m_temperature_max.nc"),
    "t2m_min_6h": Path("/net/monsoon/aasch/percentiles/percentiles_1979-2018_2m_temperature_min.nc"),
    "total_precipitation": Path("/net/monsoon/aasch/percentiles/percentiles_1979-2018_total_precipitation.nc"),
}
# The data variable holding the actual threshold values differs by file: both
# temperature files use "2m_temperature" internally (confirmed against the
# real files), but the precipitation file uses "total_precipitation".
CLIMATOLOGY_VARIABLE_NAMES = {
    "t2m_max_6h": "2m_temperature",
    "t2m_min_6h": "2m_temperature",
    "total_precipitation": "total_precipitation",
}


def open_percentile_climatology(
    variable: str,
    percentile: float,
    path: str | Path | None = None,
) -> xr.DataArray:
    """Load a precomputed daily percentile climatology and select one quantile.

    Each file has dims ``(dayofyear, latitude, longitude, quantile)`` --
    confirmed against the real files -- with ``latitude`` descending
    (90 to -90) and ``longitude`` in the 0-360 convention, same as ERA5's
    native layout. Callers should run the result through the same
    ``normalize_longitude`` (and, if applicable, ``subset_region``) used on
    ``model``/``era5`` before aligning the three together, so they end up on
    a matching grid.

    Parameters
    ----------
    variable : str
        Which reforecast/ERA5 variable this threshold is for -- must be
        ``"t2m_max_6h"``, ``"t2m_min_6h"``, or ``"total_precipitation"``;
        there is no precomputed file for ``"t2m_mean_6h"``.
    percentile : float
        One of the quantiles actually present in the file (currently 0.95,
        0.99, or 0.999 in the real files) -- matched with a small floating
        point tolerance, not looked up by exact float equality.
    path : str or Path, optional
        Override the default path for ``variable`` (``CLIMATOLOGY_PATHS``).
        The data-variable-name lookup (``CLIMATOLOGY_VARIABLE_NAMES``) still
        goes by ``variable``, not by the overridden path.

    Returns
    -------
    xr.DataArray
        Dims ``(dayofyear, latitude, longitude)`` -- the ``quantile``
        dimension is selected down to a scalar and dropped.
    """
    if variable not in CLIMATOLOGY_PATHS:
        raise ValueError(
            f"No precomputed percentile climatology for variable={variable!r}. "
            f"Only {sorted(CLIMATOLOGY_PATHS)} have one (1979-2018 baseline, from "
            "/net/monsoon/aasch/percentiles/) -- there is no equivalent file for "
            "t2m_mean_6h. Pass an explicit path= to use a different file (still "
            "need a recognized variable= for the internal data-variable-name lookup)."
        )
    if path is None:
        path = CLIMATOLOGY_PATHS[variable]
    path = Path(path)
    climatology_variable_name = CLIMATOLOGY_VARIABLE_NAMES[variable]

    dataset = xr.open_dataset(path, chunks={})
    if climatology_variable_name not in dataset:
        raise KeyError(
            f"{path} is missing expected variable {climatology_variable_name!r} "
            f"(has: {sorted(dataset.data_vars)})"
        )
    threshold_by_doy = dataset[climatology_variable_name]

    available_quantiles = threshold_by_doy["quantile"].values
    matches = np.isclose(available_quantiles, percentile)
    if not matches.any():
        raise ValueError(
            f"percentile={percentile} not present in {path}'s quantile coordinate "
            f"({sorted(available_quantiles)})"
        )
    matched_quantile = available_quantiles[matches][0]
    return threshold_by_doy.sel(quantile=matched_quantile).drop_vars("quantile")


def local_climatology_quantile(
    daily_baseline: xr.DataArray,
    quantile: float,
    window_days: int = 15,
    dayofyear_dim: str = "dayofyear",
) -> xr.DataArray:
    """Per-grid-cell, per-day-of-year climatological quantile from a baseline record.

    For every day of year ``d`` (1 through 365/366), pools every value in
    ``daily_baseline`` whose day of year is within ``window_days`` days of
    ``d`` -- wrapping circularly across the year boundary (e.g. for
    ``window_days=15``, day-of-year 3 pools days 353-366 and 1-18 across
    every year present in ``daily_baseline``) -- and returns the requested
    quantile of that pooled sample, independently at every grid cell.

    This is a standard, though computationally expensive, definition of
    "local climatology": e.g. with ``quantile=0.975`` and
    ``window_days=15``, the threshold for day-of-year 200 is the 97.5th
    percentile of every daily value from day-of-year 185 through 215, across
    all baseline years, at that grid cell.

    Parameters
    ----------
    daily_baseline : xr.DataArray
        Daily-resolution values (e.g. ERA5 daily-mean T2M) spanning the
        baseline period, with a ``time`` dimension/coordinate plus spatial
        dimensions (e.g. ``latitude``, ``longitude``). Restrict ``time`` to
        the desired baseline years (e.g. excluding the test year) before
        calling this -- it is not done here.
    quantile : float
        Quantile in [0, 1] (e.g. 0.975 for the 97.5th percentile).
    window_days : int
        Half-width, in days, of the circular day-of-year pooling window.
    dayofyear_dim : str
        Name of the output day-of-year dimension/coordinate.

    Returns
    -------
    xr.DataArray
        Dims: (``dayofyear_dim``, plus ``daily_baseline``'s non-time dims).
        ``dayofyear_dim`` runs 1 through the maximum day-of-year present in
        ``daily_baseline`` (366 if any leap year is included).

    Notes
    -----
    This computes one quantile reduction per day of year (up to 366) over
    the full baseline record and grid -- expensive for a multi-decade,
    global, high-resolution baseline. Run it once (e.g. on a dask cluster)
    and persist the result (``.to_zarr(...)`` / ``.to_netcdf(...)``) rather
    than recomputing it in every notebook run.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if "time" not in daily_baseline.dims:
        raise ValueError("daily_baseline must have a 'time' dimension")

    dayofyear = daily_baseline["time"].dt.dayofyear
    n_doy = int(dayofyear.max().item())
    daily_baseline = daily_baseline.assign_coords({dayofyear_dim: ("time", dayofyear.values)})

    thresholds = []
    for target_day in range(1, n_doy + 1):
        distance = np.abs(daily_baseline[dayofyear_dim] - target_day)
        circular_distance = np.minimum(distance, n_doy - distance)
        window = daily_baseline.where(circular_distance <= window_days, drop=True)
        thresholds.append(
            window.quantile(quantile, dim="time", skipna=True).drop_vars(
                "quantile", errors="ignore"
            )
        )

    return xr.concat(
        thresholds,
        dim=pd.Index(range(1, n_doy + 1), name=dayofyear_dim),
    )


def threshold_at_verification_time(
    threshold_by_doy: xr.DataArray,
    forecast: xr.DataArray,
    dayofyear_dim: str = "dayofyear",
) -> xr.DataArray:
    """Look up a day-of-year climatological threshold at each forecast case's verification time.

    ``forecast`` must have ``time`` (initialization) and
    ``prediction_timedelta`` (lead time) dimensions; the verification time
    for each case is ``time + prediction_timedelta``
    (``deterministic_metrics.verification_time``). The result
    is broadcastable against ``forecast`` for use as the ``threshold``
    argument to ``deterministic_metrics.extreme_indicators`` /
    ``contingency_counts``.

    Parameters
    ----------
    threshold_by_doy : xr.DataArray
        Output of :func:`local_climatology_quantile`.
    forecast : xr.DataArray
        Forecast array whose cases need a threshold looked up.
    dayofyear_dim : str
        Name of the day-of-year dimension/coordinate on ``threshold_by_doy``.

    Returns
    -------
    xr.DataArray
        Dims: (``time``, ``prediction_timedelta``, plus
        ``threshold_by_doy``'s non-dayofyear dims).
    """
    valid_time = verification_time(forecast)
    verification_dayofyear = valid_time.dt.dayofyear
    return threshold_by_doy.sel({dayofyear_dim: verification_dayofyear})
