from __future__ import annotations

import numpy as np
import xarray as xr

from pathlib import Path

from heatextremes.open_aifs_ensv2 import _forecast_store_year, daily_aifs_aggregates


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

    np.testing.assert_allclose(daily["t2m_min_6h"].values, [[[0.0], [4.0]]])
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
    np.testing.assert_allclose(daily["t2m_min_6h"].values, [[0.0]])
    np.testing.assert_allclose(daily["t2m_mean_6h"].values, [[1.5]])


def test_forecast_store_year_reads_a_year_from_a_store_name() -> None:
    assert _forecast_store_year(Path("aifs_2024-06-01T00:00:00.zarr")) == 2024
    assert _forecast_store_year(Path("not-a-date.zarr")) is None
