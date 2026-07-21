#!/usr/bin/env python3
"""Validate the collection of yearly WeatherBench2 ERA5 cache stores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
)

DEFAULT_ROOT = Path("/net/monsoon/kylehall/ERA5/wb2_era5_6h_surface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2023)
    return parser.parse_args()


def directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    total_bytes = 0

    print(
        f"{'year':>6} {'status':>10} {'times':>8} "
        f"{'start':>20} {'end':>20} {'GiB':>9}"
    )

    for year in range(args.start_year, args.end_year + 1):
        path = args.root / f"wb2_era5_6h_surface_{year:04d}.zarr"
        success = path / "_SUCCESS"

        if not path.exists() or not success.exists():
            failures.append(f"{year}: missing store or _SUCCESS marker")
            print(f"{year:>6} {'MISSING':>10}")
            continue

        try:
            ds = xr.open_zarr(path, consolidated=True, chunks={})
            missing = set(VARIABLES) - set(ds.data_vars)
            if missing:
                raise ValueError(f"missing variables {sorted(missing)}")

            if ds.sizes["latitude"] != 721 or ds.sizes["longitude"] != 1440:
                raise ValueError(
                    f"wrong grid {ds.sizes['latitude']}x{ds.sizes['longitude']}"
                )

            if ds.sizes["time"] > 1:
                hours = np.diff(ds.time.values) / np.timedelta64(1, "h")
                if not np.all(hours == 6):
                    raise ValueError("time coordinate is not uniformly 6-hourly")

            if int(ds.time.dt.year.min()) != year or int(ds.time.dt.year.max()) != year:
                raise ValueError("time coordinate contains the wrong year")

            size_bytes = directory_size(path)
            total_bytes += size_bytes

            start = np.datetime_as_string(ds.time.values[0], unit="h")
            end = np.datetime_as_string(ds.time.values[-1], unit="h")
            print(
                f"{year:>6} {'OK':>10} {ds.sizes['time']:>8} "
                f"{start:>20} {end:>20} {size_bytes / 1024**3:>9.2f}"
            )
        except Exception as exc:
            failures.append(f"{year}: {exc}")
            print(f"{year:>6} {'FAILED':>10}  {exc}")

    print(f"\nTotal on-disk size: {total_bytes / 1024**3:.2f} GiB")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\nAll requested yearly stores passed validation.")


if __name__ == "__main__":
    main()
