#!/usr/bin/env python
"""Write case-wise coverage and optional PoE diagnostics for daily precipitation."""

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


VARIABLE = "total_precipitation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-store", type=Path, required=True)
    parser.add_argument("--observation-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-percentiles", type=float, nargs="+", default=[90.0])
    parser.add_argument("--thresholds-millimeters", type=float, nargs="*", default=[])
    parser.add_argument("--poe-decision-thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--precipitation-unit", choices=("m", "mm"), default="m")
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
    _validate_inputs(model, observations)
    print(
        f"Calculating precipitation metrics for {model.sizes['time']} initializations "
        f"in batches of {args.time_batch_size}"
    )

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
                    model_batch[VARIABLE],
                    observation_batch[VARIABLE],
                    coverage_percentiles=args.coverage_percentiles,
                    thresholds_millimeters=args.thresholds_millimeters,
                    poe_decision_thresholds=args.poe_decision_thresholds,
                    precipitation_unit=args.precipitation_unit,
                )
                mode = "w" if start == 0 else "a"
                write_args = {"mode": mode, "consolidated": False}
                if start:
                    write_args["append_dim"] = "time"
                metrics.to_zarr(args.output, **write_args)

    zarr.consolidate_metadata(str(args.output))
    (args.output / "_SUCCESS").touch()
    print("Precipitation metric store complete")


def _calculate_metrics(
    model: xr.DataArray,
    observations: xr.DataArray,
    *,
    coverage_percentiles: list[float],
    thresholds_millimeters: list[float],
    poe_decision_thresholds: list[float],
    precipitation_unit: str,
) -> xr.Dataset:
    results: dict[str, xr.DataArray] = {}
    for percentile in coverage_percentiles:
        results[f"coverage_p{_metric_number(percentile)}_{VARIABLE}"] = (
            central_ensemble_coverage(model, observations, percentile=percentile).assign_attrs(
                {"central_percentile": float(percentile)}
            )
        )
    for threshold_millimeters in thresholds_millimeters:
        threshold = _threshold_in_data_units(threshold_millimeters, precipitation_unit)
        label = _metric_number(threshold_millimeters)
        results[f"poe_{label}mm_{VARIABLE}"] = probability_of_exceedance(
            model, threshold
        ).assign_attrs(
            {
                "precipitation_threshold_millimeters": float(threshold_millimeters),
                "precipitation_threshold_data_units": threshold,
            }
        )
        results[f"brier_score_{label}mm_{VARIABLE}"] = brier_score_of_exceedance(
            model, observations, threshold
        ).assign_attrs(
            {
                "precipitation_threshold_millimeters": float(threshold_millimeters),
                "precipitation_threshold_data_units": threshold,
            }
        )
        for decision_threshold in poe_decision_thresholds:
            decision_label = _metric_number(decision_threshold)
            results[
                f"poe_contingency_p{decision_label}_{label}mm_{VARIABLE}"
            ] = probability_of_exceedance_contingency(
                model,
                observations,
                threshold,
                decision_threshold=decision_threshold,
            ).assign_attrs(
                {
                    "precipitation_threshold_millimeters": float(threshold_millimeters),
                    "precipitation_threshold_data_units": threshold,
                    "decision_threshold": float(decision_threshold),
                }
            )
    return xr.Dataset(results).assign_attrs(
        {
            "forecast_day_definition": "daily totals ending at forecast lead n days",
            "precipitation_unit": precipitation_unit,
            "coverage_percentiles": ",".join(
                _metric_number(value) for value in coverage_percentiles
            ),
            "thresholds_millimeters": ",".join(
                _metric_number(value) for value in thresholds_millimeters
            ),
            "poe_decision_thresholds": ",".join(
                _metric_number(value) for value in poe_decision_thresholds
            ),
        }
    )


def _threshold_in_data_units(threshold_millimeters: float, precipitation_unit: str) -> float:
    if precipitation_unit == "m":
        return threshold_millimeters / 1000.0
    return threshold_millimeters


def _metric_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _validate_inputs(model: xr.Dataset, observations: xr.Dataset) -> None:
    missing = [
        store_name
        for store_name, store in (("model", model), ("observations", observations))
        if VARIABLE not in store
    ]
    if missing:
        raise KeyError(f"Missing {VARIABLE!r} in: {', '.join(missing)}")


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
