from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.metrics import (
    coverage,
    mean_in_time_batches,
    probability_of_exceedance_brier_score,
)


def test_mean_in_time_batches_matches_direct_nan_aware_means() -> None:
    model = xr.DataArray(
        np.broadcast_to(
            np.array([0.0, 10.0, 20.0, 30.0])[None, None, :, None], (4, 2, 4, 2)
        ),
        dims=("time", "prediction_timedelta", "number", "latitude"),
        coords={
            "time": np.arange(
                np.datetime64("2024-01-01"), np.datetime64("2024-01-05"), np.timedelta64(1, "D")
            ).astype("datetime64[ns]"),
            "prediction_timedelta": np.array(
                [np.timedelta64(0, "D"), np.timedelta64(1, "D")], dtype="timedelta64[ns]"
            ),
            "number": [0, 1, 2, 3],
            "latitude": [45.0, 46.0],
        },
    ).chunk({"time": 1})
    observations = xr.DataArray(
        [15.0, 20.0, 30.0, 15.0],
        dims="time",
        coords={
            "time": np.arange(
                np.datetime64("2024-01-01"), np.datetime64("2024-01-05"), np.timedelta64(1, "D")
            ).astype("datetime64[ns]")
        },
    )

    def calculate(model_batch: xr.DataArray) -> xr.Dataset:
        return xr.Dataset(
            {
                "coverage": coverage(model_batch, observations),
                "brier_score": probability_of_exceedance_brier_score(
                    model_batch, observations, threshold=20.0
                ),
            }
        )

    summaries = mean_in_time_batches(
        model,
        calculate,
        reductions={
            "map": ("time", "prediction_timedelta"),
            "lead": ("time", "latitude"),
        },
        batch_size=2,
    )
    direct = calculate(model)

    xr.testing.assert_allclose(
        summaries["map"], direct.mean(("time", "prediction_timedelta"), skipna=True).compute()
    )
    xr.testing.assert_allclose(
        summaries["lead"], direct.mean(("time", "latitude"), skipna=True).compute()
    )
