from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.metrics import valid_time_mask


def test_valid_time_mask_selects_calendar_day_and_hour_per_initialization() -> None:
    model = xr.DataArray(
        np.ones((2, 3, 1)),
        dims=("time", "prediction_timedelta", "number"),
        coords={
            "time": np.array(["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[ns]"),
            "prediction_timedelta": np.array(
                [
                    np.timedelta64(4, "D") + np.timedelta64(18, "h"),
                    np.timedelta64(5, "D"),
                    np.timedelta64(5, "D") + np.timedelta64(6, "h"),
                ],
                dtype="timedelta64[ns]",
            ),
            "number": [0],
        },
    )

    selected = valid_time_mask(model, target_hour=0, target_day_offsets=5)

    np.testing.assert_array_equal(
        selected.values,
        np.array([[False, True, False], [True, False, False]]),
    )


def test_valid_time_mask_accepts_multiple_day_offsets() -> None:
    model = xr.DataArray(
        np.ones((1, 3, 1)),
        dims=("time", "prediction_timedelta", "number"),
        coords={
            "time": np.array(["2024-01-01T06"], dtype="datetime64[ns]"),
            "prediction_timedelta": np.array(
                [
                    np.timedelta64(18, "h"),
                    np.timedelta64(1, "D") + np.timedelta64(18, "h"),
                    np.timedelta64(2, "D") + np.timedelta64(18, "h"),
                ],
                dtype="timedelta64[ns]",
            ),
            "number": [0],
        },
    )

    selected = valid_time_mask(model, target_hour=0, target_day_offsets=[1, 2])

    np.testing.assert_array_equal(selected.values, np.array([[True, True, False]]))
