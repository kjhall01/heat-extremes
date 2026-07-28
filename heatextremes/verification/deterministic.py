"""Sufficient statistics for ensemble-mean temperature verification."""

from __future__ import annotations

import xarray as xr

from .alignment import assert_exact_case_alignment
from .weighting import DEFAULT_REDUCTION_DIMS, weighted_sufficient_statistics


SUBSETS = ("all", "observed_hot", "observed_nonhot")


def deterministic_temperature_statistics(
    forecast_ensemble_mean: xr.DataArray,
    observation: xr.DataArray,
    observed_hot: xr.DataArray,
    *,
    reduction_dims: tuple[str, ...] = DEFAULT_REDUCTION_DIMS,
) -> xr.Dataset:
    """Return additive RMSE/MAE/bias sufficient statistics by observed subset."""
    assert_exact_case_alignment(forecast_ensemble_mean, observation, observed_hot)
    error = forecast_ensemble_mean - observation
    subsets: dict[str, xr.DataArray | None] = {
        "all": None,
        "observed_hot": observed_hot.astype(bool),
        "observed_nonhot": ~observed_hot.astype(bool),
    }
    pieces: list[xr.Dataset] = []
    for subset, condition in subsets.items():
        for metric, field in (
            ("rmse", error**2),
            ("mae", abs(error)),
            ("bias", error),
        ):
            statistic = weighted_sufficient_statistics(
                field, condition=condition, reduction_dims=reduction_dims
            ).expand_dims(metric=[metric], subset=[subset])
            pieces.append(statistic)
    return xr.combine_by_coords(pieces, combine_attrs="override")
