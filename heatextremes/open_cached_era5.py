"""Helpers for opening the yearly ECMWF ERA5 ARCO cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


DEFAULT_ROOT = Path("/net/monsoon/kylehall/ERA5/era5_arco_6h_surface")
STORE_PREFIX = "era5_arco_6h_surface"


def open_cached_era5(
    root: str | Path = DEFAULT_ROOT,
    start_year: int = 2000,
    end_year: int | None = None,
    chunks: dict | None = None
) -> xr.Dataset:
    """Open completed ECMWF ERA5 ARCO cache stores lazily along time.

    The cache stores 0.25-degree fields at 00, 06, 12, and 18 UTC. Its
    variables are ``2m_temperature``, ``2m_dewpoint_temperature``,
    ``surface_pressure``, and ``total_precipitation``. The precipitation value
    at time ``T`` is the total over ``(T - 6 hours, T]`` in metres.

    The most recent published store may be a partial current-year snapshot. If
    ``end_year`` is omitted, every published store from ``start_year`` through
    the latest available store is opened. Explicitly requesting an end year
    verifies that every intervening yearly store exists.
    """
    dask_chunks = {} if chunks is None else chunks

    root = Path(root)
    if end_year is not None and end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year")

    discovered = {
        int(path.name.removeprefix(f"{STORE_PREFIX}_").removesuffix(".zarr")): path
        for path in root.glob(f"{STORE_PREFIX}_[0-9][0-9][0-9][0-9].zarr")
        if (path / "_SUCCESS").exists()
    }
    if end_year is None:
        available = [year for year in discovered if year >= start_year]
        if not available:
            raise FileNotFoundError(
                f"No completed ERA5 ARCO cache stores found under {root} from {start_year} onward"
            )
        end_year = max(available)

    years = range(start_year, end_year + 1)
    paths = [root / f"{STORE_PREFIX}_{year:04d}.zarr" for year in years]

    missing = [
        path for path in paths
        if not path.exists() or not (path / "_SUCCESS").exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing incomplete/yearly stores:\n{formatted}")

    datasets = [
        xr.open_zarr(path, consolidated=True, chunks=dask_chunks)
        for path in paths
    ]

    return xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    )


def daily_era5_aggregates(ds: xr.Dataset) -> xr.Dataset:
    """Return UTC-day temperature summaries and correctly aligned precipitation totals.

    ``total_precipitation`` values in the cache are labelled by the end of each
    6-hour accumulation. Shifting them back six hours before resampling places
    each accumulation in the UTC day it measures. A trailing incomplete day is
    returned as missing rather than as a partial precipitation total.
    """
    required = {"2m_temperature", "total_precipitation"}
    missing = required - set(ds.data_vars)
    if missing:
        raise KeyError(f"Dataset is missing required variables: {sorted(missing)}")

    precipitation = ds["total_precipitation"].assign_coords(
        time=ds.time - np.timedelta64(6, "h")
    )
    return xr.Dataset(
        {
            "t2m_min_6h": ds["2m_temperature"].resample(time="1D").min(),
            "t2m_mean_6h": ds["2m_temperature"].resample(time="1D").mean(),
            "t2m_max_6h": ds["2m_temperature"].resample(time="1D").max(),
            "total_precipitation": precipitation.resample(time="1D").sum(min_count=4),
        }
    )


if __name__ == "__main__":
    ds = open_cached_era5()
    print(ds)

    print(daily_era5_aggregates(ds))
