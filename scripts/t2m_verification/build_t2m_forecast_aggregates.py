#!/usr/bin/env python
"""Write AIFS T2M min/mean/max for selected completed forecast days to Zarr."""

from __future__ import annotations

import argparse
from pathlib import Path

import dask
from dask.diagnostics import ProgressBar

from heatextremes import (
    aggregate_6hourly_forecast_days,
    normalize_longitude,
    open_aifs_ensv2,
)
from heatextremes.open_aifs_ensv2 import DEFAULT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--forecast-days", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument("--forecast-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("end-year must be greater than or equal to start-year")
    _prepare_output(args.output, args.overwrite)

    years = range(args.start_year, args.end_year + 1)
    print(f"Opening AIFS ENS v2 forecast stores for {args.start_year}--{args.end_year}")
    model = open_aifs_ensv2(
        root=args.forecast_root,
        years=years,
        variables=["2t"],
        chunks={
            "time": 4,
            "number": -1,
            "prediction_timedelta": 24,
            "latitude": 180,
            "longitude": 180,
        },
    )
    daily = aggregate_6hourly_forecast_days(model["2m_temperature"], args.forecast_days)
    daily = normalize_longitude(daily)
    daily = daily.chunk(
        {
            "time": 4,
            "forecast_day": 1,
            "number": -1,
            "latitude": 180,
            "longitude": 180,
        }
    )
    daily.attrs.update(
        {
            "source": "AIFS ENS v2",
            "forecast_day_definition": "day n = leads (n-1)*24 + [6, 12, 18, 24] hours",
            "forecast_year_start": args.start_year,
            "forecast_year_end": args.end_year,
        }
    )

    print(f"Writing {args.output}")
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        with ProgressBar():
            daily.to_zarr(args.output, mode="w", consolidated=True)
    (args.output / "_SUCCESS").touch()
    print("Forecast aggregate store complete")


def _prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        state = "complete" if (output / "_SUCCESS").exists() else "incomplete"
        raise FileExistsError(f"Output store already exists ({state}): {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    (output / "_SUCCESS").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
