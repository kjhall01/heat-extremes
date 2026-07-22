"""Helpers for calculating verification metrics in manageable time batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import xarray as xr


def mean_in_time_batches(
    data: xr.Dataset | xr.DataArray,
    calculate: Callable[[xr.Dataset | xr.DataArray], xr.Dataset | xr.DataArray],
    reductions: Mapping[str, tuple[str, ...]],
    batch_size: int = 1,
    time_dim: str = "time",
) -> dict[str, xr.Dataset]:
    """Calculate NaN-aware means without retaining full metric arrays.

    ``calculate`` receives one time batch and returns one or more metric data
    variables. Each requested reduction is summed and counted immediately;
    only those small partial statistics are brought into memory.
    """
    if time_dim not in data.dims:
        raise ValueError(f"data must have a {time_dim!r} dimension")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not reductions:
        raise ValueError("reductions must not be empty")

    totals: dict[str, xr.Dataset] = {}
    counts: dict[str, xr.Dataset] = {}
    for start in range(0, data.sizes[time_dim], batch_size):
        result = calculate(data.isel({time_dim: slice(start, start + batch_size)}))
        if isinstance(result, xr.DataArray):
            result = result.to_dataset(name=result.name or "value")

        statistics = []
        for name, dimensions in reductions.items():
            statistics.extend(
                (
                    result.sum(dimensions, skipna=True).rename(
                        {variable: f"{name}_sum_{variable}" for variable in result.data_vars}
                    ),
                    result.count(dimensions).rename(
                        {variable: f"{name}_count_{variable}" for variable in result.data_vars}
                    ),
                )
            )
        statistics = xr.merge(statistics).compute()

        for name in reductions:
            sum_names = {f"{name}_sum_{variable}": variable for variable in result.data_vars}
            count_names = {
                f"{name}_count_{variable}": variable for variable in result.data_vars
            }
            batch_total = statistics[list(sum_names)].rename(sum_names)
            batch_count = statistics[list(count_names)].rename(count_names)
            if name not in totals:
                totals[name] = batch_total
                counts[name] = batch_count
                continue
            totals[name], batch_total = xr.align(totals[name], batch_total, join="exact")
            counts[name], batch_count = xr.align(counts[name], batch_count, join="exact")
            totals[name] = totals[name] + batch_total
            counts[name] = counts[name] + batch_count

    return {name: totals[name] / counts[name] for name in reductions}
