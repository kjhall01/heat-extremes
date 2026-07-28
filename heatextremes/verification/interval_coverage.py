"""Central ensemble-interval coverage sufficient statistics."""

from __future__ import annotations

from collections.abc import Mapping

import xarray as xr

from .alignment import assert_exact_case_alignment
from .weighting import DEFAULT_REDUCTION_DIMS, weighted_sufficient_statistics


INTERVAL_SUBSETS = ("all", "observed_hot", "observed_nonhot")


def interval_coverage_statistics(
    lower: xr.DataArray,
    upper: xr.DataArray,
    observation: xr.DataArray,
    observed_hot: xr.DataArray,
    *,
    nominal_coverage: float,
    reduction_dims: tuple[str, ...] = DEFAULT_REDUCTION_DIMS,
) -> xr.Dataset:
    """Return inclusive central-interval coverage and width statistics."""
    assert_exact_case_alignment(lower, upper, observation, observed_hot)
    valid_bounds = lower.notnull() & upper.notnull() & observation.notnull()
    covered = ((lower <= observation) & (observation <= upper)).where(valid_bounds).astype(float)
    width = (upper - lower).where(valid_bounds)
    subsets: Mapping[str, xr.DataArray | None] = {
        "all": None,
        "observed_hot": observed_hot.astype(bool),
        "observed_nonhot": ~observed_hot.astype(bool),
    }
    pieces: list[xr.Dataset] = []
    for subset, condition in subsets.items():
        coverage = weighted_sufficient_statistics(
            covered, condition=condition, reduction_dims=reduction_dims
        )
        widths = weighted_sufficient_statistics(
            width, condition=condition, reduction_dims=reduction_dims
        )
        piece = xr.Dataset(
            {
                "numerator": coverage["numerator"],
                "denominator": coverage["denominator"],
                "unweighted_numerator": covered.where(
                    valid_bounds if condition is None else valid_bounds & condition.fillna(False)
                ).sum(tuple(dim for dim in reduction_dims if dim in covered.dims), skipna=True),
                "unweighted_support": coverage["unweighted_support"],
                "width_numerator": widths["numerator"],
            }
        ).expand_dims(nominal_coverage=[float(nominal_coverage)], subset=[subset])
        pieces.append(piece)
    return xr.combine_by_coords(pieces, combine_attrs="override")
