#!/usr/bin/env python
"""Aggregate a case-wise T2M metric store and make forecast-day line plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import dask
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar


AGGREGATION_LABELS = {
    "t2m_min_6h": "Daily minimum",
    "t2m_mean_6h": "Daily mean",
    "t2m_max_6h": "Daily maximum",
    "total_precipitation": "Daily precipitation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_complete_store(args.metric_store)
    _prepare_output(args.output, args.overwrite)
    args.figure_directory.mkdir(parents=True, exist_ok=True)

    metrics = xr.open_zarr(args.metric_store, consolidated=True, chunks={"time": 16})
    summary = _summarize(metrics)
    print(f"Writing {args.output}")
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        with ProgressBar():
            summary.to_zarr(args.output, mode="w", consolidated=False)
    zarr.consolidate_metadata(str(args.output))
    (args.output / "_SUCCESS").touch()

    _plot_global_means(summary, args.figure_directory)
    print("Summary store and figures complete")


def _summarize(metrics: xr.Dataset) -> xr.Dataset:
    summaries: dict[str, xr.DataArray] = {}
    for name, values in metrics.data_vars.items():
        if "time" not in values.dims:
            raise ValueError(f"Metric variable must have a time dimension: {name}")
        if name.startswith("poe_contingency_"):
            summaries.update(_contingency_rate_summaries(name, values))
            continue
        summaries[f"{name}_map_mean"] = values.mean("time", skipna=True)
        summaries[f"{name}_map_valid_cases"] = values.count("time")
        global_dimensions = tuple(
            dimension
            for dimension in ("time", "latitude", "longitude")
            if dimension in values.dims
        )
        summaries[f"{name}_global_mean"] = values.mean(global_dimensions, skipna=True)
        summaries[f"{name}_global_valid_values"] = values.count(global_dimensions)
    return xr.Dataset(summaries).assign_attrs(metrics.attrs)


def _contingency_rate_summaries(
    name: str,
    values: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """Return conditional hit, miss, and false-positive rates from coded cases."""
    suffix = name.removeprefix("poe_contingency_")
    map_dimensions = ("time",)
    global_dimensions = tuple(
        dimension
        for dimension in ("time", "latitude", "longitude")
        if dimension in values.dims
    )
    observed_event = (values == 1) | (values == 2)
    observed_non_event = (values == 0) | (values == 3)
    definitions = {
        "poe_hit_rate": ((values == 1), observed_event),
        "poe_miss_rate": ((values == 2), observed_event),
        "poe_false_positive_rate": ((values == 3), observed_non_event),
    }
    summaries: dict[str, xr.DataArray] = {}
    for metric, (numerator, denominator) in definitions.items():
        summaries[f"{metric}_{suffix}_map_mean"] = _rate(
            numerator, denominator, map_dimensions
        )
        summaries[f"{metric}_{suffix}_global_mean"] = _rate(
            numerator, denominator, global_dimensions
        )
    summaries[f"poe_valid_cases_{suffix}_map_cases"] = (values >= 0).sum("time")
    summaries[f"poe_valid_cases_{suffix}_global_cases"] = (values >= 0).sum(global_dimensions)
    return summaries


def _rate(
    numerator: xr.DataArray,
    denominator: xr.DataArray,
    dimensions: tuple[str, ...],
) -> xr.DataArray:
    numerator_total = numerator.sum(dimensions)
    denominator_total = denominator.sum(dimensions)
    return xr.where(denominator_total > 0, numerator_total / denominator_total, np.nan)


def _plot_global_means(summary: xr.Dataset, figure_directory: Path) -> None:
    groups = {
        "coverage": ("coverage_", "Central interval coverage", "Coverage"),
        "poe": ("poe_", "Member-count probability of exceedance", "Probability"),
        "brier": ("brier_score_", "Brier score", "Brier score"),
        "hit_rate": ("poe_hit_rate_", "PoE hit rate", "Rate"),
        "miss_rate": ("poe_miss_rate_", "PoE miss rate", "Rate"),
        "false_positive_rate": (
            "poe_false_positive_rate_",
            "PoE false-positive rate",
            "Rate",
        ),
    }
    for filename, (prefix, title, ylabel) in groups.items():
        variables = sorted(
            name
            for name in summary.data_vars
            if name.startswith(prefix) and name.endswith("_global_mean")
            and (filename != "poe" or _is_probability_metric(name))
        )
        if not variables:
            continue
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        for name in variables:
            metric_name = name.removesuffix("_global_mean")
            aggregation = _aggregation_name(metric_name)
            values = summary[name].compute()
            axis.plot(
                values.forecast_day,
                values,
                marker="o",
                label=_plot_label(metric_name, aggregation),
            )
        axis.set(
            xlabel="Forecast day",
            ylabel=ylabel,
            title=title,
            ylim=(0, 1),
            xticks=summary.forecast_day.values,
        )
        axis.grid(True, alpha=0.35)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_directory / f"{filename}_by_forecast_day.png", dpi=180)
        plt.close(figure)


def _aggregation_name(metric_name: str) -> str:
    for aggregation in AGGREGATION_LABELS:
        if metric_name.endswith(aggregation):
            return aggregation
    return metric_name


def _plot_label(metric_name: str, aggregation: str) -> str:
    aggregation_label = AGGREGATION_LABELS.get(aggregation, aggregation)
    metric_label = metric_name.removesuffix(f"_{aggregation}").replace("_", " ")
    return f"{aggregation_label} ({metric_label})"


def _is_probability_metric(name: str) -> bool:
    return not name.startswith(
        (
            "poe_hit_rate_",
            "poe_miss_rate_",
            "poe_false_positive_rate_",
        )
    )


def _require_complete_store(store: Path) -> None:
    if not (store / "_SUCCESS").exists():
        raise FileNotFoundError(f"Input store is missing its _SUCCESS marker: {store}")


def _prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        state = "complete" if (output / "_SUCCESS").exists() else "incomplete"
        raise FileExistsError(f"Output store already exists ({state}): {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    (output / "_SUCCESS").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
