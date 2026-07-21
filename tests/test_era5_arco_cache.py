"""Unit tests for the ECMWF ERA5 ARCO cache transformation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "era5_arco_cache"
    / "cache_era5_arco_year.py"
)
SPEC = importlib.util.spec_from_file_location("era5_arco_cache", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


def make_arco_source(year: int = 2024) -> xr.Dataset:
    """Build a tiny complete hourly ARCO-shaped source for a leap year."""
    start = np.datetime64(f"{year - 1:04d}-12-31T19:00:00")
    end = np.datetime64(f"{year + 1:04d}-01-01T00:00:00")
    time = np.arange(start, end, np.timedelta64(1, "h")).astype("datetime64[ns]")
    values = da.arange(time.size, chunks=24, dtype="float32")[:, None, None]
    values = values + da.zeros((1, 721, 1440), chunks=(1, 45, 90), dtype="float32")
    coordinates = {
        "time": time,
        "latitude": np.linspace(-90, 90, 721),
        "longitude": np.arange(1440) / 4 - 180,
    }
    return xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), values),
            "d2m": (("time", "latitude", "longitude"), values + 1),
            "sp": (("time", "latitude", "longitude"), values + 2),
            "tp": (("time", "latitude", "longitude"), values),
        },
        coords=coordinates,
    )


def test_select_year_maps_variables_and_aggregates_hourly_precipitation() -> None:
    source = make_arco_source()

    result = cache.select_year(source, 2024, source_url="https://example.test/arco.zarr")

    assert tuple(result.data_vars) == cache.VARIABLES
    assert result.sizes == {"time": 1464, "latitude": 721, "longitude": 1440}
    assert result.attrs["cache_source_url"] == "https://example.test/arco.zarr"
    assert result["total_precipitation"].attrs["units"] == "m"

    # The Jan-01 00 UTC accumulation contains Dec-31 19 through Jan-01 00.
    np.testing.assert_allclose(
        result["total_precipitation"].isel(time=0, latitude=0, longitude=0), 15
    )
    # The next accumulation contains the next six hourly inputs (6 through 11).
    np.testing.assert_allclose(
        result["total_precipitation"].isel(time=1, latitude=0, longitude=0), 51
    )
    # Instantaneous fields are sampled at their matching 6-hour valid times.
    np.testing.assert_allclose(result["2m_temperature"].isel(time=0, latitude=0, longitude=0), 5)
    np.testing.assert_allclose(result["2m_temperature"].isel(time=1, latitude=0, longitude=0), 11)


def test_select_year_requires_all_arco_variables() -> None:
    source = make_arco_source().drop_vars("tp")

    with pytest.raises(KeyError, match="tp"):
        cache.select_year(source, 2024)


def test_select_year_can_stop_at_the_latest_complete_six_hour_timestep() -> None:
    source = make_arco_source().sel(time=slice(None, "2024-01-02T08:00:00"))

    result = cache.select_year(source, 2024, allow_partial_year=True)

    assert result.sizes["time"] == 6
    assert result.time.values[-1] == np.datetime64("2024-01-02T06:00:00")
    assert result.attrs["cache_complete_year"] is False
    assert result.attrs["cache_time_coverage_end"] == "2024-01-02T06:00:00"


def test_select_year_rejects_missing_hourly_precipitation() -> None:
    source = make_arco_source()
    source = source.isel(time=np.arange(source.sizes["time"]) != 10)

    with pytest.raises(ValueError, match="does not contain every 1-hourly timestamp"):
        cache.select_year(source, 2024)
