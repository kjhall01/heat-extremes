from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.forecast_days import (
    aggregate_6hourly_forecast_days,
    aggregate_6hourly_observation_samples,
    aggregate_6hourly_precipitation_samples,
    daily_forecast_day_leads,
    forecast_day_leads,
    sample_observations_at_forecast_day_leads,
    select_daily_forecast_totals,
)


def test_forecast_day_leads_uses_completed_24_hour_windows() -> None:
    result = forecast_day_leads([1, 3, 5])

    np.testing.assert_array_equal(
        result.values / np.timedelta64(1, "h"),
        np.array([[6, 12, 18, 24], [54, 60, 66, 72], [102, 108, 114, 120]]),
    )


def test_aggregate_6hourly_forecast_days_excludes_initialization_step() -> None:
    temperature = xr.DataArray(
        np.arange(13, dtype="float32")[None, :, None],
        dims=("time", "prediction_timedelta", "number"),
        coords={
            "time": np.array(["2024-01-01T00"], dtype="datetime64[ns]"),
            "prediction_timedelta": (np.arange(13) * 6).astype("timedelta64[h]").astype(
                "timedelta64[ns]"
            ),
            "number": [0],
        },
        attrs={"units": "K"},
    )

    result = aggregate_6hourly_forecast_days(temperature, [1, 3])

    np.testing.assert_allclose(result["t2m_min_6h"].values, [[[1.0], [9.0]]])
    np.testing.assert_allclose(result["t2m_mean_6h"].values, [[[2.5], [10.5]]])
    np.testing.assert_allclose(result["t2m_max_6h"].values, [[[4.0], [12.0]]])
    np.testing.assert_array_equal(result.forecast_day.values, [1, 3])
    np.testing.assert_array_equal(
        result.forecast_lead_start.values / np.timedelta64(1, "h"), [6, 54]
    )
    assert result["t2m_max_6h"].attrs["units"] == "K"


def test_sampled_observations_follow_each_initialization_and_require_four_values() -> None:
    observations = xr.DataArray(
        np.arange(16, dtype="float32"),
        dims="time",
        coords={
            "time": np.arange(
                np.datetime64("2024-01-01T00"),
                np.datetime64("2024-01-05T00"),
                np.timedelta64(6, "h"),
            ).astype("datetime64[ns]")
        },
    )
    initialization_times = xr.DataArray(
        np.array(["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[ns]"),
        dims="time",
        coords={"time": np.array(["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[ns]")},
    )

    sampled = sample_observations_at_forecast_day_leads(
        observations, initialization_times, [1]
    )
    result = aggregate_6hourly_observation_samples(sampled)

    np.testing.assert_allclose(result["t2m_min_6h"].values, [[1.0], [2.0]])
    np.testing.assert_allclose(result["t2m_mean_6h"].values, [[2.5], [3.5]])
    np.testing.assert_allclose(result["t2m_max_6h"].values, [[4.0], [5.0]])

    missing = aggregate_6hourly_observation_samples(sampled.isel(time=0).where(False))
    assert np.isnan(missing["t2m_mean_6h"].item())


def test_daily_precipitation_forecast_totals_and_era5_sums_share_forecast_days() -> None:
    daily_model = xr.DataArray(
        np.arange(1, 6, dtype="float32")[None, :, None],
        dims=("time", "prediction_timedelta_daily", "number"),
        coords={
            "time": np.array(["2024-01-01T00"], dtype="datetime64[ns]"),
            "prediction_timedelta_daily": (np.arange(1, 6) * 24)
            .astype("timedelta64[h]")
            .astype("timedelta64[ns]"),
            "number": [0],
        },
        name="tp",
        attrs={"units": "m"},
    )

    selected = select_daily_forecast_totals(daily_model, [1, 3])

    np.testing.assert_allclose(selected.values, [[[1.0], [3.0]]])
    np.testing.assert_array_equal(daily_forecast_day_leads([1, 3]) / np.timedelta64(1, "D"), [1, 3])
    np.testing.assert_array_equal(
        selected.forecast_lead_start / np.timedelta64(1, "h"), [0, 48]
    )

    sampled_era5 = xr.DataArray(
        np.ones((1, 2, 4), dtype="float32"),
        dims=("time", "forecast_day", "six_hourly_sample"),
        coords={
            "time": selected.time,
            "forecast_day": selected.forecast_day,
            "six_hourly_sample": [1, 2, 3, 4],
        },
        attrs={"units": "m"},
    )
    matched = aggregate_6hourly_precipitation_samples(sampled_era5)

    np.testing.assert_allclose(matched.values, [[4.0, 4.0]])
    np.testing.assert_array_equal(matched.forecast_lead_start / np.timedelta64(1, "h"), [0, 48])
