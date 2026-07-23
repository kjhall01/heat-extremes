from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.metrics import central_ensemble_coverage, coverage


def test_coverage_uses_verification_time_and_broadcasts_other_dimensions() -> None:
    initialization_times = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    lead_times = np.array(
        [np.timedelta64(0, "D"), np.timedelta64(1, "D")], dtype="timedelta64[ns]"
    )
    ensemble_values = np.broadcast_to(
        np.array([0.0, 10.0, 20.0, 30.0])[None, None, :, None], (2, 2, 4, 2)
    )
    model_ensemble = xr.DataArray(
        ensemble_values,
        dims=("time", "prediction_timedelta", "number", "latitude"),
        coords={
            "time": initialization_times,
            "prediction_timedelta": lead_times,
            "number": [0, 1, 2, 3],
            "latitude": [45.0, 46.0],
        },
    ).chunk({"number": 2})
    observations = xr.DataArray(
        [15.0, 30.0, 15.0],
        dims="time",
        coords={
            "time": np.array(
                ["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
            )
        },
    )

    result = coverage(model_ensemble, observations, percentile=90)

    expected = xr.DataArray(
        np.array([[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]),
        dims=("time", "prediction_timedelta", "latitude"),
        coords={
            "time": initialization_times,
            "prediction_timedelta": lead_times,
            "latitude": [45.0, 46.0],
        },
        name="coverage",
    )
    xr.testing.assert_equal(result.drop_vars("verification_time"), expected)
    xr.testing.assert_equal(
        result["verification_time"].drop_vars("verification_time"),
        xr.DataArray(
            np.array(
                [
                    ["2024-01-01", "2024-01-02"],
                    ["2024-01-02", "2024-01-03"],
                ],
                dtype="datetime64[ns]",
            ),
            dims=("time", "prediction_timedelta"),
            coords={
                "time": initialization_times,
                "prediction_timedelta": lead_times,
            },
        ),
    )


def test_coverage_is_nan_when_a_verification_observation_is_unavailable() -> None:
    model_ensemble = xr.DataArray(
        [[[0.0, 10.0], [0.0, 10.0]]],
        dims=("time", "prediction_timedelta", "number"),
        coords={
            "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "prediction_timedelta": np.array(
                [np.timedelta64(0, "D"), np.timedelta64(2, "D")], dtype="timedelta64[ns]"
            ),
            "number": [0, 1],
        },
    )
    observations = xr.DataArray(
        [5.0],
        dims="time",
        coords={"time": np.array(["2024-01-01"], dtype="datetime64[ns]")},
    )

    result = coverage(model_ensemble, observations)

    np.testing.assert_equal(result.values, [[1.0, np.nan]])


def test_direct_coverage_accepts_daily_forecast_day_axes() -> None:
    model = xr.DataArray(
        [[[0.0, 10.0, 20.0, 30.0]]],
        dims=("time", "forecast_day", "number"),
        coords={
            "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "forecast_day": [1],
            "number": [0, 1, 2, 3],
        },
    )
    observations = xr.DataArray(
        [[15.0]],
        dims=("time", "forecast_day"),
        coords={"time": model.time, "forecast_day": model.forecast_day},
    )

    result = central_ensemble_coverage(model, observations, percentile=90)

    np.testing.assert_equal(result.values, [[1.0]])
