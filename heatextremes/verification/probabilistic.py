"""Additive probability scores and binary decision diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from .alignment import assert_exact_case_alignment
from .weighting import DEFAULT_REDUCTION_DIMS, cosine_latitude_weights, weighted_sufficient_statistics


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


def probability_decision_statistics(
    probability: xr.DataArray,
    observed_event: xr.DataArray,
    decision_thresholds: Sequence[float],
    *,
    reduction_dims: tuple[str, ...] = DEFAULT_REDUCTION_DIMS,
) -> xr.Dataset:
    """Return additive POD/FAR statistics for probability decision cutoffs.

    A forecast is positive when ``probability >= decision_threshold``.  The
    output stores contingency-table totals, not monthly ratios, so POD and FAR
    remain exact after aggregation across arbitrary cache partitions.
    """
    assert_exact_case_alignment(probability, observed_event)
    thresholds = tuple(float(value) for value in decision_thresholds)
    if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Probability decision thresholds must be within [0, 1]")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Probability decision thresholds must be unique")

    observed = observed_event.astype(bool)
    dimensions = tuple(dimension for dimension in reduction_dims if dimension in probability.dims)
    weights = cosine_latitude_weights(probability).broadcast_like(probability)
    valid = probability.notnull() & observed_event.notnull() & weights.notnull()

    def support(mask: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
        selected = valid & mask
        return (
            weights.where(selected, 0.0).sum(dimensions, skipna=True),
            selected.sum(dimensions),
        )

    event_weight, _ = support(observed)
    non_event_weight, _ = support(~observed)
    total_weight, total_count = support(xr.ones_like(observed, dtype=bool))
    pieces: list[xr.Dataset] = []
    for threshold in thresholds:
        positive = probability >= threshold
        hit_weight, hit_count = support(positive & observed)
        miss_weight, miss_count = support(~positive & observed)
        false_alarm_weight, false_alarm_count = support(positive & ~observed)
        correct_negative_weight, correct_negative_count = support(~positive & ~observed)
        common = {
            "weighted_support": total_weight,
            "unweighted_support": total_count,
            "event_weighted_support": event_weight,
            "non_event_weighted_support": non_event_weight,
            "weighted_hits": hit_weight,
            "weighted_misses": miss_weight,
            "weighted_false_alarms": false_alarm_weight,
            "weighted_correct_negatives": correct_negative_weight,
            "unweighted_hits": hit_count,
            "unweighted_misses": miss_count,
            "unweighted_false_alarms": false_alarm_count,
            "unweighted_correct_negatives": correct_negative_count,
        }
        for metric, numerator, denominator in (
            ("pod", hit_weight, hit_weight + miss_weight),
            ("far", false_alarm_weight, hit_weight + false_alarm_weight),
        ):
            pieces.append(
                xr.Dataset({"numerator": numerator, "denominator": denominator, **common}).expand_dims(
                    decision_threshold=[threshold], metric=[metric]
                )
            )
    return xr.combine_by_coords(pieces, combine_attrs="override")
