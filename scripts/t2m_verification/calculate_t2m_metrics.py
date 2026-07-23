#!/usr/bin/env python
"""Write case-wise coverage, ensemble PoE, and Brier scores for T2M Zarr stores."""

from __future__ import annotations

import argparse
from pathlib import Path

import dask
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar

from heatextremes.metrics import (
    brier_score_of_exceedance,
    central_ensemble_coverage,
    probability_of_exceedance,
    probability_of_exceedance_contingency,
)


TEMPERATURE_AGGREGATIONS = ("t2m_min_6h", "t2m_mean_6h", "t2m_max_6h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-store", type=Path, required=True)
    parser.add_argument("--observation-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-percentiles", type=float, nargs="+", default=[90.0])
    parser.add_argument("--thresholds-celsius", type=float, nargs="+", default=[35.0])
    parser.add_argument("--poe-decision-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--temperature-unit", choices=("K", "C"), default="K")
    parser.add_argument("--time-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.time_batch_size < 1:
        raise ValueError("time-batch-size must be at least 1")
    _require_complete_store(args.model_store)
    _require_complete_store(args.observation_store)
    _prepare_output(args.output, args.overwrite)

    model = xr.open_zarr(
        args.model_store,
        consolidated=True,
        chunks={"time": args.time_batch_size, "forecast_day": 1, "number": -1},
    )
    observations = xr.open_zarr(
        args.observation_store,
        consolidated=True,
        chunks={"time": args.time_batch_size, "forecast_day": 1},
    )
    aggregations = _common_aggregations(model, observations)
    print(f"Calculating metrics for {model.sizes['time']} initializations in batches of "
          f"{args.time_batch_size}")

    with dask.config.set(scheduler="threads", num_workers=args.workers):
        with ProgressBar():
            for start in range(0, model.sizes["time"], args.time_batch_size):
                stop = min(start + args.time_batch_size, model.sizes["time"])
                print(f"Metric batch {start}:{stop}")
                model_batch, observation_batch = xr.align(
                    model.isel(time=slice(start, stop)),
                    observations.isel(time=slice(start, stop)),
                    join="exact",
                    copy=False,
                )
                metrics = _calculate_metrics(
                    model_batch,
                    observation_batch,
                    aggregations,
                    coverage_percentiles=args.coverage_percentiles,
                    thresholds_celsius=args.thresholds_celsius,
                    poe_decision_thresholds=args.poe_decision_thresholds,
                    temperature_unit=args.temperature_unit,
                )
                mode = "w" if start == 0 else "a"
                write_args = {"mode": mode, "consolidated": False}
                if start:
                    write_args["append_dim"] = "time"
                metrics.to_zarr(args.output, **write_args)

    zarr.consolidate_metadata(str(args.output))
    (args.output / "_SUCCESS").touch()
    print("Metric store complete")


def _calculate_metrics(
    model: xr.Dataset,
    observations: xr.Dataset,
    aggregations: tuple[str, ...],
    *,
    coverage_percentiles: list[float],
    thresholds_celsius: list[float],
    poe_decision_thresholds: list[float],
    temperature_unit: str,
) -> xr.Dataset:
    results: dict[str, xr.DataArray] = {}
    for aggregation in aggregations:
        ensemble = model[aggregation]
        observed = observations[aggregation]
        for percentile in coverage_percentiles:
            name = f"coverage_p{_metric_number(percentile)}_{aggregation}"
            results[name] = central_ensemble_coverage(
                ensemble, observed, percentile=percentile
            ).assign_attrs({"central_percentile": float(percentile)})
        for threshold_celsius in thresholds_celsius:
            threshold = _threshold_in_data_units(threshold_celsius, temperature_unit)
            label = _metric_number(threshold_celsius)
            probability = probability_of_exceedance(ensemble, threshold)
            results[f"poe_{label}c_{aggregation}"] = probability.assign_attrs(
                {
                    "temperature_threshold_celsius": float(threshold_celsius),
                    "temperature_threshold_data_units": threshold,
                }
            )
            results[f"brier_score_{label}c_{aggregation}"] = brier_score_of_exceedance(
                ensemble, observed, threshold
            ).assign_attrs(
                {
                    "temperature_threshold_celsius": float(threshold_celsius),
                    "temperature_threshold_data_units": threshold,
                }
            )
            for decision_threshold in poe_decision_thresholds:
                decision_label = _metric_number(decision_threshold)
                results[
                    f"poe_contingency_p{decision_label}_{label}c_{aggregation}"
                ] = probability_of_exceedance_contingency(
                    ensemble,
                    observed,
                    threshold,
                    decision_threshold=decision_threshold,
                ).assign_attrs(
                    {
                        "temperature_threshold_celsius": float(threshold_celsius),
                        "temperature_threshold_data_units": threshold,
                        "decision_threshold": float(decision_threshold),
                    }
                )
    return xr.Dataset(results).assign_attrs(
        {
            "forecast_day_definition": "day n = leads (n-1)*24 + [6, 12, 18, 24] hours",
            "temperature_unit": temperature_unit,
            "coverage_percentiles": ",".join(
                _metric_number(value) for value in coverage_percentiles
            ),
            "thresholds_celsius": ",".join(_metric_number(value) for value in thresholds_celsius),
            "poe_decision_thresholds": ",".join(
                _metric_number(value) for value in poe_decision_thresholds
            ),
        }
    )


def _threshold_in_data_units(threshold_celsius: float, temperature_unit: str) -> float:
    if temperature_unit == "K":
        return threshold_celsius + 273.15
    return threshold_celsius


def _metric_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _common_aggregations(model: xr.Dataset, observations: xr.Dataset) -> tuple[str, ...]:
    aggregations = tuple(
        name
        for name in TEMPERATURE_AGGREGATIONS
        if name in model.data_vars and name in observations.data_vars
    )
    if aggregations != TEMPERATURE_AGGREGATIONS:
        raise KeyError(
            "Model and observation stores must both contain "
            f"{list(TEMPERATURE_AGGREGATIONS)}; found {list(aggregations)}"
        )
    return aggregations


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
