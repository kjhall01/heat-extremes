#!/usr/bin/env python
"""Write ERA5 daily precipitation totals matched to an AIFS forecast-day store."""

from __future__ import annotations

import argparse
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from heatextremes import (
    aggregate_6hourly_precipitation_samples,
    normalize_longitude,
    open_cached_era5,
    sample_observations_at_forecast_day_leads,
)
from heatextremes.open_cached_era5 import DEFAULT_ROOT as DEFAULT_ERA5_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-store", type=Path, required=True)
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_complete_store(args.model_store)
    _prepare_output(args.output, args.overwrite)

    model = xr.open_zarr(args.model_store, consolidated=True)
    forecast_days = model.forecast_day.values.astype(int)
    start_year, end_year = _needed_era5_years(model.time.values, forecast_days)
    print(f"Opening ERA5 cache for {start_year}--{end_year}")
    era5 = open_cached_era5(
        root=args.era5_root,
        start_year=start_year,
        end_year=end_year,
        chunks={"time": 36, "latitude": 180, "longitude": 180},
    )
    sampled = sample_observations_at_forecast_day_leads(
        era5["total_precipitation"],
        model.time,
        forecast_days,
    )
    daily = aggregate_6hourly_precipitation_samples(sampled).to_dataset()
    daily = normalize_longitude(daily).reindex(
        latitude=model.latitude,
        longitude=model.longitude,
    ).chunk({"time": 4, "forecast_day": 1, "latitude": 180, "longitude": 180})
    daily.attrs.update(
        {
            "source": "ECMWF ERA5 ARCO 6-hourly cache",
            "forecast_day_definition": "sum ERA5 accumulations ending at leads 6, 12, 18, 24 h",
            "model_store": str(args.model_store),
        }
    )

    print(f"Writing {args.output}")
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        with ProgressBar():
            daily.to_zarr(args.output, mode="w", consolidated=True)
    (args.output / "_SUCCESS").touch()
    print("Matched ERA5 daily precipitation store complete")


def _needed_era5_years(
    initialization_times: np.ndarray,
    forecast_days: np.ndarray,
) -> tuple[int, int]:
    latest_lead = np.timedelta64(int(forecast_days.max()) * 24, "h")
    start = np.datetime64(initialization_times.min())
    end = np.datetime64(initialization_times.max()) + latest_lead
    return int(str(start)[:4]), int(str(end)[:4])


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
