"""Valid-time selection helpers for forecast verification."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np
import xarray as xr

from heatextremes.metrics._verification import verification_time


def valid_time_mask(
    model_ensemble: xr.DataArray,
    target_hour: int | None = None,
    target_day_offsets: int | Iterable[int] | None = None,
) -> xr.DataArray:
    """Return a mask selecting forecasts by valid UTC hour and calendar day.

    ``target_day_offsets`` is measured from each initialization's calendar
    date. For example, ``target_hour=0`` and ``target_day_offsets=5`` select
    forecasts valid at 00Z on the date five calendar days after each
    initialization, regardless of the initialization hour.
    """
    valid_time = verification_time(model_ensemble)
    selection = xr.ones_like(valid_time, dtype=bool)

    if target_hour is not None:
        if not isinstance(target_hour, Integral) or not 0 <= target_hour < 24:
            raise ValueError("target_hour must be an integer from 0 through 23")
        selection = selection & (valid_time.dt.hour == target_hour)

    if target_day_offsets is not None:
        day_offsets = _as_day_offsets(target_day_offsets)
        initialization_date = model_ensemble.time.dt.floor("D")
        verification_date = valid_time.dt.floor("D")
        days_from_initialization = (
            verification_date - initialization_date
        ) / np.timedelta64(1, "D")
        selection = selection & days_from_initialization.isin(day_offsets)

    return selection.rename("valid_time_selection")


def _as_day_offsets(target_day_offsets: int | Iterable[int]) -> tuple[int, ...]:
    """Validate and normalize one or more calendar-day offsets."""
    if isinstance(target_day_offsets, Integral):
        return (int(target_day_offsets),)

    try:
        values = tuple(target_day_offsets)
    except TypeError as error:
        raise TypeError("target_day_offsets must be an integer or iterable of integers") from error

    if not values or not all(isinstance(value, Integral) for value in values):
        raise ValueError("target_day_offsets must contain at least one integer")
    return tuple(int(value) for value in values)
