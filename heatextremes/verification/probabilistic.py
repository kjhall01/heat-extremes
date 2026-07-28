"""Additive probability-forecast scores."""

from __future__ import annotations

import xarray as xr

from .alignment import assert_exact_case_alignment
from .weighting import DEFAULT_REDUCTION_DIMS, weighted_sufficient_statistics


def probability_event_statistics(
    probability: xr.DataArray,
    observed_event: xr.DataArray,
    *,
    reduction_dims: tuple[str, ...] = DEFAULT_REDUCTION_DIMS,
) -> xr.Dataset:
    """Return Brier, frequency, and probability-bias sufficient statistics."""
    assert_exact_case_alignment(probability, observed_event)
    observed = observed_event.astype(float)
    pieces: list[xr.Dataset] = []
    for metric, field in (
        ("brier_score", (probability - observed) ** 2),
        ("mean_forecast_probability", probability),
        ("observed_event_frequency", observed),
        ("probability_frequency_bias", probability - observed),
    ):
        pieces.append(
            weighted_sufficient_statistics(field, reduction_dims=reduction_dims).expand_dims(
                metric=[metric]
            )
        )
    result = xr.combine_by_coords(pieces, combine_attrs="override")
    support = weighted_sufficient_statistics(
        xr.ones_like(observed), reduction_dims=reduction_dims
    )
    event = weighted_sufficient_statistics(observed, reduction_dims=reduction_dims)
    non_event = weighted_sufficient_statistics(1.0 - observed, reduction_dims=reduction_dims)
    result["weighted_support"] = result["denominator"]
    result["event_weighted_support"] = event["numerator"] + xr.zeros_like(result["denominator"])
    result["non_event_weighted_support"] = non_event["numerator"] + xr.zeros_like(
        result["denominator"]
    )
    result["unweighted_support"] = support["unweighted_support"] + xr.zeros_like(
        result["denominator"], dtype="int64"
    )
    return result
