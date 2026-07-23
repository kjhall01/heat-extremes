"""Helpers for 6-hourly forecast-day temperature verification.

Forecast day 1 deliberately means the first *completed* 24-hour forecast
window: leads 6, 12, 18, and 24 hours.  This is different from grouping a
timedelta coordinate by ``.dt.days``, which places 6--18 hours in day 0 and
24 hours in day 1.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import xarray as xr


FORECAST_DAY_DIM = "forecast_day"
SIX_HOURLY_SAMPLE_DIM = "six_hourly_sample"
_SIX_HOURLY_OFFSETS = np.array([6, 12, 18, 24], dtype="int64")


def forecast_day_leads(forecast_days: int | Iterable[int]) -> xr.DataArray:
    """Return the four 6-hourly leads belonging to each requested forecast day.

    Day ``n`` contains ``(n - 1) * 24 + (6, 12, 18, 24)`` hours.  Thus day 1
    is 6--24 h, day 3 is 54--72 h, and so on.  The returned array has
    dimensions ``forecast_day`` and ``six_hourly_sample``.
    """
    days = _as_forecast_days(forecast_days)
    lead_hours = (days[:, None] - 1) * 24 + _SIX_HOURLY_OFFSETS[None, :]
    return xr.DataArray(
        lead_hours.astype("timedelta64[h]").astype("timedelta64[ns]"),
        dims=(FORECAST_DAY_DIM, SIX_HOURLY_SAMPLE_DIM),
        coords={
            FORECAST_DAY_DIM: days,
            SIX_HOURLY_SAMPLE_DIM: np.arange(1, _SIX_HOURLY_OFFSETS.size + 1),
        },
        name="prediction_timedelta",
        attrs={
            "description": "6-hourly leads in each completed 24-hour forecast day",
        },
    )


def daily_forecast_day_leads(forecast_days: int | Iterable[int]) -> xr.DataArray:
    """Return the end-of-day lead for each requested daily forecast total."""
    days = _as_forecast_days(forecast_days)
    return xr.DataArray(
        (days * 24).astype("timedelta64[h]").astype("timedelta64[ns]"),
        dims=FORECAST_DAY_DIM,
        coords={FORECAST_DAY_DIM: days},
        name="prediction_timedelta_daily",
        attrs={"description": "end lead of each daily forecast accumulation"},
    )


def select_daily_forecast_totals(
    total_precipitation: xr.DataArray,
    forecast_days: int | Iterable[int],
    *,
    step_dim: str = "prediction_timedelta_daily",
) -> xr.DataArray:
    """Select already accumulated daily precipitation totals by forecast day.

    AIFS ENS v2 precipitation is supplied on ``prediction_timedelta_daily``
    with labels 1--50 days.  Its values are already daily accumulations, so
    they are selected rather than summed again.
    """
    _validate_step_coordinate(total_precipitation, step_dim)
    leads = daily_forecast_day_leads(forecast_days)
    requested_leads = leads.values
    available = total_precipitation.indexes[step_dim]
    missing = requested_leads[~np.isin(requested_leads, available.values)]
    if missing.size:
        missing_days = ", ".join(
            str(int(value / np.timedelta64(1, "D"))) for value in missing
        )
        raise ValueError(f"forecast is missing requested daily precipitation leads: {missing_days}")

    selected = total_precipitation.sel({step_dim: leads}).rename("total_precipitation")
    return _assign_forecast_day_window_coordinates(selected, first_sample_hour=0)


def aggregate_6hourly_forecast_days(
    temperature: xr.DataArray,
    forecast_days: int | Iterable[int],
    *,
    step_dim: str = "prediction_timedelta",
) -> xr.Dataset:
    """Aggregate forecast temperature over requested completed forecast days.

    The input must contain the exact four 6-hourly leads for every requested
    day.  A missing data value at any of those samples makes that output case
    missing, rather than silently turning it into a partial-day aggregate.
    """
    _validate_step_coordinate(temperature, step_dim)
    leads = forecast_day_leads(forecast_days)
    available = temperature.indexes[step_dim]
    requested_leads = leads.values.ravel()
    missing = requested_leads[~np.isin(requested_leads, available.values)]
    if missing.size:
        missing_hours = ", ".join(
            str(int(value / np.timedelta64(1, "h"))) for value in missing
        )
        raise ValueError(
            f"forecast is missing requested 6-hourly leads (hours): {missing_hours}"
        )

    selected = temperature.sel({step_dim: leads})
    return _aggregate_6hourly_samples(selected, required_samples=leads.sizes[SIX_HOURLY_SAMPLE_DIM])


def sample_observations_at_forecast_day_leads(
    observations: xr.DataArray,
    initialization_times: xr.DataArray,
    forecast_days: int | Iterable[int],
    *,
    time_dim: str = "time",
) -> xr.DataArray:
    """Select observations at the four valid times for each forecast day.

    Unavailable observation timestamps are retained as missing values.  This
    permits the subsequent aggregation to mark an incomplete day as missing
    while preserving the forecast initialization and forecast-day axes.
    """
    _validate_observation_time(observations, initialization_times, time_dim)
    leads = forecast_day_leads(forecast_days)
    valid_times = initialization_times + leads
    positions = observations.indexes[time_dim].get_indexer(valid_times.values.ravel())
    positions = positions.reshape(valid_times.shape)
    # Do not attach the initialization-time coordinate to this vectorized
    # indexer: it conflicts with the multi-dimensional observation-time
    # coordinate created by ``isel``.
    indexer = xr.DataArray(positions, dims=valid_times.dims)
    found = indexer >= 0
    safe_indexer = indexer.where(found, 0).astype(np.intp)
    sampled = observations.isel({time_dim: safe_indexer}).where(found)

    # Vectorized indexing creates a multi-dimensional time coordinate.  The
    # initialization coordinate is the useful axis after sampling; retain the
    # full valid-time matrix separately for provenance.
    return sampled.drop_vars(time_dim).assign_coords(
        {
            time_dim: initialization_times,
            "verification_time": valid_times,
        }
    )


def aggregate_6hourly_observation_samples(temperature: xr.DataArray) -> xr.Dataset:
    """Aggregate sampled ERA5 temperature over a completed forecast day."""
    if SIX_HOURLY_SAMPLE_DIM not in temperature.dims:
        raise ValueError(f"temperature must have a {SIX_HOURLY_SAMPLE_DIM!r} dimension")
    return _aggregate_6hourly_samples(
        temperature,
        required_samples=temperature.sizes[SIX_HOURLY_SAMPLE_DIM],
    )


def aggregate_6hourly_precipitation_samples(
    total_precipitation: xr.DataArray,
) -> xr.DataArray:
    """Sum four end-labelled 6-hourly precipitation totals into a full day.

    ERA5 cache values at time ``T`` measure accumulation over ``(T - 6 h, T]``.
    Sampling a forecast day at leads 6, 12, 18, and 24 h and summing them
    therefore gives the matching accumulation over the forecast-day window.
    Any unavailable sample makes the total missing rather than partial.
    """
    if SIX_HOURLY_SAMPLE_DIM not in total_precipitation.dims:
        raise ValueError(
            f"total_precipitation must have a {SIX_HOURLY_SAMPLE_DIM!r} dimension"
        )
    required_samples = total_precipitation.sizes[SIX_HOURLY_SAMPLE_DIM]
    result = total_precipitation.sum(
        SIX_HOURLY_SAMPLE_DIM,
        skipna=True,
        min_count=required_samples,
    ).rename("total_precipitation")
    result = result.assign_attrs(
        {
            **total_precipitation.attrs,
            "aggregation_source": (
                f"sum of {required_samples} 6-hourly ERA5 accumulations over a forecast day"
            ),
        }
    )
    return _assign_forecast_day_window_coordinates(result, first_sample_hour=0)


def normalize_longitude(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Convert longitudes to ``[-180, 180)`` and sort them."""
    if "longitude" not in data.coords:
        raise ValueError("data must have a longitude coordinate")
    normalized_longitude = (data.longitude + 180) % 360 - 180
    if normalized_longitude.to_index().has_duplicates:
        raise ValueError("longitude normalization produced duplicate coordinates")
    return data.assign_coords(longitude=normalized_longitude).sortby("longitude")


def _aggregate_6hourly_samples(
    temperature: xr.DataArray,
    *,
    required_samples: int,
) -> xr.Dataset:
    valid_day = temperature.count(SIX_HOURLY_SAMPLE_DIM) == required_samples
    common_attrs = dict(temperature.attrs)
    common_attrs["aggregation_source"] = (
        f"{required_samples} 6-hourly samples over a completed forecast day"
    )

    aggregates = {
        "t2m_min_6h": temperature.min(SIX_HOURLY_SAMPLE_DIM, skipna=True),
        "t2m_mean_6h": temperature.mean(SIX_HOURLY_SAMPLE_DIM, skipna=True),
        "t2m_max_6h": temperature.max(SIX_HOURLY_SAMPLE_DIM, skipna=True),
    }
    result = xr.Dataset(
        {
            name: values.where(valid_day).assign_attrs(
                {
                    **common_attrs,
                    "long_name": (
                        "2 metre temperature "
                        f"{name.removeprefix('t2m_').removesuffix('_6h')}"
                    ),
                }
            )
            for name, values in aggregates.items()
        }
    )
    return _assign_forecast_day_window_coordinates(result, first_sample_hour=6)


def _assign_forecast_day_window_coordinates(
    data: xr.Dataset | xr.DataArray,
    *,
    first_sample_hour: int,
) -> xr.Dataset | xr.DataArray:
    if FORECAST_DAY_DIM not in data.dims:
        return data
    forecast_days = data[FORECAST_DAY_DIM].values.astype("int64")
    return data.assign_coords(
        {
            "forecast_lead_start": (
                FORECAST_DAY_DIM,
                ((forecast_days - 1) * 24 + first_sample_hour)
                .astype("timedelta64[h]")
                .astype("timedelta64[ns]"),
            ),
            "forecast_lead_end": (
                FORECAST_DAY_DIM,
                (forecast_days * 24).astype("timedelta64[h]").astype("timedelta64[ns]"),
            ),
        }
    )


def _as_forecast_days(forecast_days: int | Iterable[int]) -> np.ndarray:
    if isinstance(forecast_days, Integral) and not isinstance(forecast_days, bool):
        values = (int(forecast_days),)
    else:
        try:
            values = tuple(forecast_days)
        except TypeError as error:
            raise TypeError("forecast_days must be an integer or iterable of integers") from error
    if not values or any(
        not isinstance(value, Integral) or isinstance(value, bool) or value < 1
        for value in values
    ):
        raise ValueError("forecast_days must contain one or more positive integers")
    if len(set(values)) != len(values):
        raise ValueError("forecast_days must not contain duplicates")
    return np.asarray(values, dtype="int64")


def _validate_step_coordinate(temperature: xr.DataArray, step_dim: str) -> None:
    if step_dim not in temperature.dims:
        raise ValueError(f"temperature must have a {step_dim!r} dimension")
    if step_dim not in temperature.coords or temperature[step_dim].dims != (step_dim,):
        raise ValueError(f"temperature {step_dim!r} coordinate must be one-dimensional")


def _validate_observation_time(
    observations: xr.DataArray,
    initialization_times: xr.DataArray,
    time_dim: str,
) -> None:
    if time_dim not in observations.dims or time_dim not in observations.coords:
        raise ValueError(f"observations must have a {time_dim!r} dimension and coordinate")
    if observations[time_dim].dims != (time_dim,):
        raise ValueError(f"observations {time_dim!r} coordinate must be one-dimensional")
    if not observations.indexes[time_dim].is_unique:
        raise ValueError("observations time coordinate must contain unique timestamps")
    if initialization_times.dims != (time_dim,):
        raise ValueError(f"initialization_times must have exactly the {time_dim!r} dimension")
