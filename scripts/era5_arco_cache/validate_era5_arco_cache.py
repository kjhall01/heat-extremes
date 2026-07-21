#!/usr/bin/env python3
"""Validate the collection of yearly ECMWF ERA5 ARCO cache stores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
)

DEFAULT_ROOT = Path("/net/monsoon/kylehall/ERA5/era5_arco_6h_surface")
STORE_PREFIX = "era5_arco_6h_surface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument(
        "--allow-partial-final-year",
        action="store_true",
        help="Permit the requested final year to end at a complete 6-hour timestep before Dec 31.",
    )
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
        path = args.root / f"{STORE_PREFIX}_{year:04d}.zarr"
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

            full_end = np.datetime64(f"{year + 1:04d}-01-01T00:00:00") - np.timedelta64(
                6, "h"
            )
            actual_end = ds.time.values[-1]
            if actual_end != full_end:
                if not (args.allow_partial_final_year and year == args.end_year):
                    raise ValueError(
                        "store is partial; pass --allow-partial-final-year to accept it"
                    )
                if actual_end > full_end:
                    raise ValueError("time coordinate extends beyond the requested year")
                if bool(ds.attrs.get("cache_complete_year", True)):
                    raise ValueError("partial store is not marked cache_complete_year=false")

            expected_time = np.arange(
                np.datetime64(f"{year:04d}-01-01T00:00:00"),
                actual_end + np.timedelta64(6, "h"),
                np.timedelta64(6, "h"),
            )
            np.testing.assert_array_equal(ds.time.values, expected_time)

            if ds["total_precipitation"].attrs.get("units") != "m":
                raise ValueError("total_precipitation is not stored in metres")

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
