from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.metrics import (
    brier_score_of_exceedance,
    probability_of_exceedance,
    probability_of_exceedance_brier_score,
    probability_of_exceedance_contingency,
)


def test_poe_brier_score_matches_valid_time_and_broadcasts_dataarray_threshold() -> None:
    initialization_times = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
    lead_times = np.array(
        [np.timedelta64(0, "D"), np.timedelta64(1, "D")], dtype="timedelta64[ns]"
    )
    model_ensemble = xr.DataArray(
        np.broadcast_to(
            np.array([0.0, 10.0, 20.0, 30.0])[None, None, :, None], (2, 2, 4, 2)
        ),
        dims=("time", "prediction_timedelta", "number", "latitude"),
        coords={
            "time": initialization_times,
            "prediction_timedelta": lead_times,
            "number": [0, 1, 2, 3],
            "latitude": [45.0, 46.0],
        },
    ).chunk({"number": 2})
    observations = xr.DataArray(
        [25.0, 20.0, 30.0],
        dims="time",
        coords={
            "time": np.array(
                ["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"
            )
        },
    )
    threshold = xr.DataArray([20.0, 40.0], dims="latitude", coords={"latitude": [45.0, 46.0]})

    result = probability_of_exceedance_brier_score(model_ensemble, observations, threshold)

    expected = xr.DataArray(
        np.array(
            [
                [[0.5625, 0.0], [0.0625, 0.0]],
                [[0.0625, 0.0], [0.5625, 0.0]],
            ]
        ),
        dims=("time", "prediction_timedelta", "latitude"),
        coords={
            "time": initialization_times,
            "prediction_timedelta": lead_times,
            "latitude": [45.0, 46.0],
        },
        name="probability_of_exceedance_brier_score",
    )
    xr.testing.assert_equal(result.drop_vars("verification_time"), expected)


def test_poe_brier_score_accepts_a_scalar_threshold_and_masks_missing_observations() -> None:
    model_ensemble = xr.DataArray(
        [[[0.0, 10.0, 20.0, 30.0], [0.0, 10.0, 20.0, 30.0]]],
        dims=("time", "prediction_timedelta", "number"),
        coords={
            "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
            "prediction_timedelta": np.array(
                [np.timedelta64(0, "D"), np.timedelta64(2, "D")], dtype="timedelta64[ns]"
            ),
            "number": [0, 1, 2, 3],
        },
    )
    observations = xr.DataArray(
        [20.0],
        dims="time",
        coords={"time": np.array(["2024-01-01"], dtype="datetime64[ns]")},
    )

    result = probability_of_exceedance_brier_score(model_ensemble, observations, threshold=20)

    np.testing.assert_equal(result.values, [[0.0625, np.nan]])


def test_direct_poe_and_brier_score_accept_daily_forecast_day_axes() -> None:
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
        [[25.0]],
        dims=("time", "forecast_day"),
        coords={"time": model.time, "forecast_day": model.forecast_day},
    )

    probability = probability_of_exceedance(model, threshold=20.0)
    score = brier_score_of_exceedance(model, observations, threshold=20.0)

    np.testing.assert_equal(probability.values, [[0.25]])
    np.testing.assert_equal(score.values, [[0.5625]])


def test_poe_contingency_uses_member_probability_decision_threshold() -> None:
    model = xr.DataArray(
        [
            [[0.0, 10.0, 20.0, 30.0]],
            [[30.0, 30.0, 30.0, 30.0]],
            [[0.0, 0.0, 0.0, 0.0]],
            [[30.0, 30.0, 30.0, 30.0]],
        ],
        dims=("time", "forecast_day", "number"),
        coords={
            "time": np.arange(
                np.datetime64("2024-01-01"), np.datetime64("2024-01-05"), np.timedelta64(1, "D")
            ),
            "forecast_day": [1],
            "number": [0, 1, 2, 3],
        },
    )
    observations = xr.DataArray(
        [[25.0], [25.0], [10.0], [10.0]],
        dims=("time", "forecast_day"),
        coords={"time": model.time, "forecast_day": model.forecast_day},
    )

    result = probability_of_exceedance_contingency(
        model, observations, threshold=20.0, decision_threshold=0.5
    )

    # 25% PoE with an observed event is a miss; then hit, true negative, FP.
    np.testing.assert_equal(result.values[:, 0], [2, 1, 0, 3])
    assert result.dtype == np.dtype("int8")
