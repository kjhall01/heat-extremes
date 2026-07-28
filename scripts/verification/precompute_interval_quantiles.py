#!/usr/bin/env python3
"""Save only selected AIFS ensemble-temperature quantiles, one month/lead at a time.

This optional preprocessing step exists because the normal compact AIFS heat
products retain probabilities and ensemble mean only.  It never writes member
temperatures and is not needed for deterministic or probability metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.config import load_config
from heatextremes.verification.io import assert_safe_result_path, write_json_atomic, write_netcdf_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--forecast-days", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _heat_month_module(repository_root: Path):
    source = repository_root / "scripts" / "process_aifs_heat_month.py"
    spec = importlib.util.spec_from_file_location("existing_aifs_heat_month", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import existing local-day implementation from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quantile_levels(interval_levels: tuple[float, ...]) -> np.ndarray:
    return np.asarray(
        sorted(
            {
                value
                for coverage in interval_levels
                for value in ((1.0 - coverage) / 2.0, 1.0 - (1.0 - coverage) / 2.0)
            }
        ),
        dtype=float,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.assert_partition_selected(args.year, args.month)
    pattern = config.data["paths"].get("interval_quantile_file_pattern")
    if not pattern:
        raise ValueError(
            "Set paths.interval_quantile_file_pattern beneath verification_results_root before preprocessing"
        )
    days = tuple(args.forecast_days or config.forecast_days)
    invalid = sorted(set(days) - set(config.forecast_days))
    if invalid:
        raise ValueError(f"Requested forecast days are not configured: {invalid}")
    outputs = {
        day: Path(
            pattern.format(
                model=config.model_name,
                year=args.year,
                month=args.month,
                forecast_day=day,
            )
        )
        for day in days
    }
    if args.dry_run:
        print(json.dumps({"quantile_levels": _quantile_levels(config.interval_levels).tolist(), "outputs": {str(day): str(path) for day, path in outputs.items()}}, indent=2))
        return
    for output in outputs.values():
        assert_safe_result_path(output, config.result_root)
        if output.exists() and not args.overwrite:
            print(f"Skipping existing quantile product: {output}")

    repository_root = Path(__file__).resolve().parents[2]
    existing = _heat_month_module(repository_root)
    raw = existing.open_aifs_month(
        Path(config.data["paths"]["raw_aifs_root"]),
        args.year,
        args.month,
    )
    source_variable = str(config.data["variables"].get("raw_member_temperature", "2t"))
    if source_variable not in raw:
        raise KeyError(f"Raw AIFS monthly store does not contain {source_variable!r}")
    quantile_levels = _quantile_levels(config.interval_levels)
    try:
        for day, output in outputs.items():
            if output.exists() and not args.overwrite:
                continue
            # The source function is the verified notebook/monthly-product
            # implementation.  Selecting before quantile reduction leaves a
            # one-lead Dask graph; no member cube is written.
            # Use the same 15-day source window as the compact monthly
            # processor.  Some longitude/init-hour bands need samples beyond
            # a UTC calendar day to form the first local day; selecting the
            # requested day before the quantile reduction keeps the evaluated
            # graph bounded to one output lead.
            daily = existing.local_solar_daily_mean_forecast(
                raw[source_variable], max_days=max(config.forecast_days) + 1
            ).sel(forecast_day=day)
            quantiles = daily.quantile(quantile_levels, dim="number").rename(
                "ensemble_temperature_quantile"
            )
            quantiles = quantiles.rename({"time": "initialization"}).transpose(
                "initialization", "quantile", "latitude", "longitude"
            )
            product = quantiles.to_dataset().assign_attrs(
                {
                    "source": "AIFS ENS v2 member-level local-solar daily mean",
                    "forecast_day": int(day),
                    "quantile_definition": "member quantiles required for configured central intervals",
                    "member_temperatures_saved": "false",
                }
            )
            print(f"Computing and writing selected quantiles for day {day}: {output}", flush=True)
            write_netcdf_atomic(product.compute(), output)
            write_json_atomic(
                {
                    "model": config.model_name,
                    "year": args.year,
                    "month": args.month,
                    "forecast_day": int(day),
                    "quantiles": quantile_levels.tolist(),
                    "status": "complete",
                },
                output.with_suffix(".json"),
            )
    finally:
        raw.close()


if __name__ == "__main__":
    main()
