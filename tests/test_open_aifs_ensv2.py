from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.open_aifs_ensv2 import daily_aifs_aggregates


def test_daily_aifs_aggregates_returns_daily_forecast_temperature() -> None:
    ds = xr.Dataset(
        {
            "2m_temperature": (
                ("time", "prediction_timedelta", "number"),
                np.arange(8, dtype="float32")[None, :, None],
            )
        },
        coords={
            "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "prediction_timedelta": (
                (np.arange(8).astype("timedelta64[h]") * 6).astype("timedelta64[ns]")
            ),
            "number": [0],
        },
    )

    daily = daily_aifs_aggregates(ds)

    np.testing.assert_allclose(daily["t2m_mean_6h"].values, [[[1.5], [5.5]]])
    np.testing.assert_allclose(daily["t2m_max_6h"].values, [[[3.0], [7.0]]])
    np.testing.assert_array_equal(
        daily.prediction_timedelta.values,
        np.array([np.timedelta64(0, "D"), np.timedelta64(1, "D")], dtype="timedelta64[ns]"),
    )


def test_daily_aifs_aggregates_limits_the_number_of_complete_days() -> None:
    ds = xr.Dataset(
        {
            "2m_temperature": (
                ("time", "prediction_timedelta"),
                np.arange(8, dtype="float32")[None, :],
            )
        },
        coords={
            "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "prediction_timedelta": (
                (np.arange(8).astype("timedelta64[h]") * 6).astype("timedelta64[ns]")
            ),
        },
    )

    daily = daily_aifs_aggregates(ds, max_days=1)

    assert daily.sizes["prediction_timedelta"] == 1
    np.testing.assert_allclose(daily["t2m_mean_6h"].values, [[1.5]])
