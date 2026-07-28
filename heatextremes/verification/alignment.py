"""Grid and valid-date alignment used by the verified heat notebooks."""

from __future__ import annotations

import numpy as np
import xarray as xr


def normalize_longitude_like(
    source: xr.DataArray,
    target_longitude: xr.DataArray,
    *,
    longitude_name: str = "longitude",
) -> xr.DataArray:
    """Normalize and sort source longitude in the target convention."""
    target_uses_360 = bool(
        float(target_longitude.min()) >= 0.0 and float(target_longitude.max()) > 180.0
    )
    longitude = (
        source[longitude_name] % 360.0
        if target_uses_360
        else ((source[longitude_name] + 180.0) % 360.0) - 180.0
    )
    normalized = source.assign_coords({longitude_name: longitude}).sortby(longitude_name)
    if normalized.indexes[longitude_name].has_duplicates:
        raise ValueError("Longitude normalization produced duplicate coordinates")
    return normalized


def map_to_forecast_grid(
    source: xr.DataArray,
    forecast: xr.DataArray,
    *,
    method: str,
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
) -> xr.DataArray:
    """Return a lazy source field on exactly the forecast spatial grid."""
    source = normalize_longitude_like(source, forecast[longitude_name], longitude_name=longitude_name)
    same_latitude = np.array_equal(source[latitude_name].values, forecast[latitude_name].values)
    same_longitude = np.array_equal(source[longitude_name].values, forecast[longitude_name].values)
    if same_latitude and same_longitude:
        return source
    return source.interp(
        {latitude_name: forecast[latitude_name], longitude_name: forecast[longitude_name]},
        method=method,
    )


def match_observation_by_valid_date(
    observation: xr.DataArray,
    valid_date: xr.DataArray,
    *,
    initialization_name: str = "initialization",
    forecast_day_name: str = "forecast_day",
    time_name: str = "time",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
) -> xr.DataArray:
    """Vectorize exact ERA5 lookup without stacking forecast cases.

    ``valid_date`` may be ``(initialization, forecast_day, longitude)`` or a
    single-lead ``(initialization, longitude)`` field.  Keeping the vectorized
    indexer multidimensional avoids the MultiIndex loss caused by the previous
    stack/select/unstack experiment.
    """
    required_observation = {time_name, latitude_name, longitude_name}
    if not required_observation.issubset(observation.dims):
        raise ValueError(f"Observation must have dimensions {sorted(required_observation)}")
    if initialization_name not in valid_date.dims:
        raise ValueError(f"valid_date must have {initialization_name!r}")
    if longitude_name not in valid_date.dims:
        raise ValueError(f"valid_date must have {longitude_name!r}")
    if not observation.indexes[time_name].is_unique:
        raise ValueError("Observation time coordinate must be unique")

    lookup_time = "_observation_time"
    lookup_initialization = "_lookup_initialization"
    observation_for_lookup = observation.rename({time_name: lookup_time})
    dates_for_lookup = valid_date.rename({initialization_name: lookup_initialization})
    matched = observation_for_lookup.sel(
        {lookup_time: dates_for_lookup, longitude_name: dates_for_lookup[longitude_name]}
    )
    matched = matched.rename({lookup_initialization: initialization_name})
    desired = [initialization_name]
    if forecast_day_name in matched.dims:
        desired.append(forecast_day_name)
    desired.extend([latitude_name, longitude_name])
    return matched.transpose(*desired)


def assert_exact_case_alignment(*arrays: xr.DataArray) -> None:
    """Raise a useful error when canonical scored fields differ in coordinates."""
    if len(arrays) < 2:
        return
    xr.align(*arrays, join="exact", copy=False)
