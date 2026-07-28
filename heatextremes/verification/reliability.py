"""Additive probability-reliability bin summaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .alignment import assert_exact_case_alignment
from .weighting import DEFAULT_REDUCTION_DIMS, cosine_latitude_weights


def probability_reliability_statistics(
    probability: xr.DataArray,
    observed_event: xr.DataArray,
    bin_edges: Sequence[float],
    *,
    reduction_dims: tuple[str, ...] = DEFAULT_REDUCTION_DIMS,
) -> xr.Dataset:
    """Return additive reliability-bin counts and forecast/event sums.

    Bins are lower-inclusive and upper-exclusive except the final bin, which
    includes a probability of one.  This makes the configured [0, 1] range
    exhaustive without moving exact-one forecasts into an artificial bin.
    """
    assert_exact_case_alignment(probability, observed_event)
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
        raise ValueError("Reliability bin edges must start at 0 and end at 1")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Reliability bin edges must be strictly increasing")

    observed = observed_event.astype(float)
    dimensions = tuple(dimension for dimension in reduction_dims if dimension in probability.dims)
    weights = cosine_latitude_weights(probability).broadcast_like(probability)
    valid = probability.notnull() & observed.notnull() & weights.notnull()
    pieces: list[xr.Dataset] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (probability >= lower) & (
            probability <= upper if index == len(edges) - 2 else probability < upper
        )
        selected = valid & in_bin
        selected_weight = weights.where(selected, 0.0)
        piece = xr.Dataset(
            {
                "weighted_count": selected_weight.sum(dimensions, skipna=True),
                "unweighted_count": selected.sum(dimensions),
                "weighted_probability_sum": (probability.where(selected, 0.0) * selected_weight).sum(
                    dimensions, skipna=True
                ),
                "weighted_observation_sum": (observed.where(selected, 0.0) * selected_weight).sum(
                    dimensions, skipna=True
                ),
            }
        ).expand_dims(bin=[index]).assign_coords(
            bin_lower=("bin", [lower]),
            bin_upper=("bin", [upper]),
        )
        pieces.append(piece)
    return xr.combine_by_coords(pieces, combine_attrs="override")
