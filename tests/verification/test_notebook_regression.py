"""Tiny regression fixture for notebook 03's weighted temperature formula."""

from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.verification.deterministic import deterministic_temperature_statistics


def test_notebook03_weighted_rmse_bias_fixture() -> None:
    # This is the direct scalar formula from 03_full_aifs_heat_verification:
    # sum(field * cos(latitude)) / sum(cos(latitude)), then sqrt for RMSE.
    forecast = xr.DataArray(
        np.array([[[3.0], [6.0]]]),
        dims=("initialization", "latitude", "longitude"),
        coords={
            "initialization": np.array(["2024-07-01"], dtype="datetime64[ns]"),
            "latitude": [0.0, 60.0],
            "longitude": [0.0],
        },
    )
    observation = xr.zeros_like(forecast)
    hot = xr.DataArray(
        np.array([[[True], [False]]]), dims=forecast.dims, coords=forecast.coords
    )
    implementation = deterministic_temperature_statistics(forecast, observation, hot).compute()
    weights = np.cos(np.deg2rad(np.array([0.0, 60.0])))
    direct_rmse = np.sqrt((3.0**2 * weights[0] + 6.0**2 * weights[1]) / weights.sum())
    direct_bias = (3.0 * weights[0] + 6.0 * weights[1]) / weights.sum()
    result_rmse = implementation.sel(metric="rmse", subset="all")
    result_bias = implementation.sel(metric="bias", subset="all")
    assert np.isclose(np.sqrt(result_rmse.numerator / result_rmse.denominator), direct_rmse)
    assert np.isclose(result_bias.numerator / result_bias.denominator, direct_bias)
