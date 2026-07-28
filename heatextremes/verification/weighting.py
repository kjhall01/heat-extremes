"""Cosine-latitude weighted sufficient-statistic helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr


DEFAULT_REDUCTION_DIMS = ("initialization", "latitude", "longitude")


def cosine_latitude_weights(
    field: xr.DataArray,
    *,
    latitude_name: str = "latitude",
) -> xr.DataArray:
    """Return non-negative cosine latitude weights for a canonical field."""
    if latitude_name not in field.coords:
        raise ValueError(f"Field is missing latitude coordinate {latitude_name!r}")
    return np.cos(np.deg2rad(field[latitude_name])).clip(min=0.0).rename("cosine_latitude_weight")


def weighted_sufficient_statistics(
    field: xr.DataArray,
    *,
    condition: xr.DataArray | None = None,
    reduction_dims: tuple[str, ...] = DEFAULT_REDUCTION_DIMS,
) -> xr.Dataset:
    """Return additive weighted numerator/denominator/count statistics.

    Conditional metrics are intentionally aggregated over all cases at once:
    ``sum(weight * field * condition) / sum(weight * condition)``.  The
    caller never receives an average of per-initialization scores.
    """
    dimensions = tuple(dimension for dimension in reduction_dims if dimension in field.dims)
    weights = cosine_latitude_weights(field).broadcast_like(field)
    valid = field.notnull() & weights.notnull()
    if condition is not None:
        field, condition = xr.align(field, condition, join="exact", copy=False)
        valid = valid & condition.fillna(False).astype(bool)
    weighted = field.where(valid, 0.0) * weights.where(valid, 0.0)
    return xr.Dataset(
        {
            "numerator": weighted.sum(dimensions, skipna=True),
            "denominator": weights.where(valid).sum(dimensions, skipna=True),
            "unweighted_support": valid.sum(dimensions),
        }
    )


def weighted_support(
    valid_condition: xr.DataArray,
    *,
    reduction_dims: tuple[str, ...] = DEFAULT_REDUCTION_DIMS,
) -> xr.Dataset:
    """Return weighted and unweighted support for a boolean condition."""
    one = xr.ones_like(valid_condition, dtype=float)
    return weighted_sufficient_statistics(
        one,
        condition=valid_condition,
        reduction_dims=reduction_dims,
    )
