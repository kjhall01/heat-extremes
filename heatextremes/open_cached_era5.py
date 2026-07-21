"""Helper for opening the yearly WeatherBench2 ERA5 cache as one dataset."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


DEFAULT_ROOT = Path("/net/monsoon/kylehall/ERA5/wb2_era5_6h_surface")


def open_cached_era5(
    root: str | Path = DEFAULT_ROOT,
    start_year: int = 2001,
    end_year: int = 2023,
) -> xr.Dataset:
    """
    Open completed yearly stores and concatenate lazily along time.

    Native cache chunks are retained:
      time=244, latitude=45, longitude=90
    except for edge chunks.
    """
    root = Path(root)
    paths = [
        root / f"wb2_era5_6h_surface_{year:04d}.zarr"
        for year in range(start_year, end_year + 1)
    ]

    missing = [
        path for path in paths
        if not path.exists() or not (path / "_SUCCESS").exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing incomplete/yearly stores:\n{formatted}")

    datasets = [
        xr.open_zarr(path, consolidated=True, chunks={})
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


if __name__ == "__main__":
    ds = open_cached_era5()
    print(ds)

    # Example UTC-day aggregates from the four 6-hourly samples.
    daily = xr.Dataset(
        {
            "t2m_mean_6h": ds["2m_temperature"].resample(time="1D").mean(),
            "t2m_max_6h": ds["2m_temperature"].resample(time="1D").max(),
        }
    )
    print(daily)
