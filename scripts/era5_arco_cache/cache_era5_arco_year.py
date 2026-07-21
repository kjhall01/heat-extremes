#!/usr/bin/env python3
"""Cache selected ECMWF ERA5 ARCO surface fields to a local 6-hourly Zarr store.

The source is ECMWF's authenticated, time-chunked ERA5 ARCO Zarr archive. Each
year is written to a temporary directory, validated, and atomically renamed to
its final path. Re-running a completed year is therefore safe.

Instantaneous fields are sampled at 00, 06, 12, and 18 UTC. Total
precipitation is an hourly accumulation in ERA5 ARCO, so it is summed into
6-hour accumulations ending at those timestamps. It is *not* subsampled.

Designed for a Slurm array with one year per task.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import dask
import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc


SOURCE = (
    "https://arco.datastores.ecmwf.int/cadl-arco-time-002/arco/"
    "reanalysis_era5_single_levels/sfc/timeChunked.zarr"
)

SOURCE_TO_CACHE_VARIABLE = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "tp": "total_precipitation",
}

INSTANTANEOUS_SOURCE_VARIABLES = ("t2m", "d2m", "sp")
VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
)

DEFAULT_OUTPUT_ROOT = Path("/net/monsoon/kylehall/ERA5/era5_arco_6h_surface")
STORE_PREFIX = "era5_arco_6h_surface"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def handle_usr1(signum: int, frame: object) -> None:
    """
    Exit promptly when Slurm warns of preemption.

    The current output is only a staging directory. On requeue, the script
    removes that staging directory and safely restarts the year.
    """
    log(f"Received signal {signum}; exiting with code 99 for Slurm requeue.")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument(
        "--cdsapirc",
        type=Path,
        help="Optional path to a CDS API configuration file containing 'key: <token>'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        help="Number of threaded Dask workers used for ARCO reads and local writes.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=244,
        help="Target Zarr time chunk. 244 steps = 61 days at 6-hour cadence.",
    )
    parser.add_argument("--latitude-chunk", type=int, default=45)
    parser.add_argument("--longitude-chunk", type=int, default=90)
    parser.add_argument(
        "--allow-partial-year",
        action="store_true",
        help=(
            "Cache through the latest complete 6-hour ARCO timestep when the requested "
            "year is not yet complete."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an already completed yearly store.",
    )
    return parser.parse_args()


def get_cdsapi_key(cdsapirc: Path | None = None) -> str:
    """Return the CDS API token from CDSAPI_KEY or a CDS API configuration file."""
    token = os.environ.get("CDSAPI_KEY")
    if token:
        return token

    config_path = cdsapirc or Path.home() / ".cdsapirc"
    if config_path.is_file():
        for line in config_path.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "key" and value.strip():
                return value.strip()

    raise RuntimeError(
        "No CDS API key found. Set CDSAPI_KEY or create ~/.cdsapirc with "
        "'key: <your CDS API token>'. Accept the ERA5 licence in the CDS first."
    )


def open_source(source: str, cdsapi_key: str) -> xr.Dataset:
    """Open the ECMWF time-chunked ARCO store without loading data values."""
    log(f"Opening ECMWF ERA5 ARCO source: {source}")
    return xr.open_zarr(
        source,
        consolidated=True,
        chunks={},
        storage_options={"headers": {"Authorization": f"Bearer {cdsapi_key}"}},
    )


def year_start(year: int) -> np.datetime64:
    return np.datetime64(f"{year:04d}-01-01T00:00:00")


def full_year_end(year: int, cadence_hours: int) -> np.datetime64:
    return np.datetime64(f"{year + 1:04d}-01-01T00:00:00") - np.timedelta64(
        cadence_hours, "h"
    )


def expected_times(
    year: int,
    cadence_hours: int,
    *,
    end_time: np.datetime64 | None = None,
) -> np.ndarray:
    start = np.datetime64(f"{year:04d}-01-01T00:00:00")
    end = full_year_end(year, cadence_hours) if end_time is None else end_time
    return np.arange(
        start,
        end + np.timedelta64(cadence_hours, "h"),
        np.timedelta64(cadence_hours, "h"),
    )


def assert_regular_time(
    time_values: np.ndarray,
    *,
    cadence_hours: int,
    name: str,
) -> None:
    if time_values.size == 0:
        raise ValueError(f"{name} has no time steps")
    if time_values.size > 1:
        hours = np.diff(time_values) / np.timedelta64(1, "h")
        if not np.all(hours == cadence_hours):
            bad = np.where(hours != cadence_hours)[0][:10]
            raise ValueError(
                f"{name} is not uniformly {cadence_hours}-hourly at indices {bad.tolist()}"
            )


def assert_expected_time(
    time_values: np.ndarray,
    *,
    year: int,
    cadence_hours: int,
    name: str,
    end_time: np.datetime64 | None = None,
) -> None:
    expected = expected_times(year, cadence_hours, end_time=end_time)
    if not np.array_equal(time_values, expected):
        raise ValueError(
            f"{name} does not contain every {cadence_hours}-hourly timestamp for {year}"
        )


def latest_complete_six_hour_time(source: xr.Dataset, year: int) -> np.datetime64:
    """Return the latest source timestamp that closes a 6-hour interval in ``year``."""
    source_time = source.time.sel(
        time=slice(year_start(year), np.datetime64(f"{year + 1:04d}-01-01T00:00:00"))
    ).values
    if source_time.size == 0:
        raise ValueError(f"No ARCO data found for {year}")
    assert_regular_time(source_time, cadence_hours=1, name="ARCO source time coordinate")

    latest_hour = min(source_time[-1], full_year_end(year, cadence_hours=1))
    elapsed_hours = int((latest_hour - year_start(year)) / np.timedelta64(1, "h"))
    latest_complete = year_start(year) + np.timedelta64((elapsed_hours // 6) * 6, "h")
    if latest_complete < year_start(year):
        raise ValueError(f"ARCO has not yet published a complete 6-hour interval for {year}")
    return latest_complete


def six_hourly_precipitation(
    source: xr.Dataset,
    year: int,
    *,
    end_time: np.datetime64,
) -> xr.DataArray:
    """Sum ERA5's hourly precipitation into 6-hour accumulations ending at time."""
    start_time = year_start(year)
    source_start = start_time - np.timedelta64(5, "h")

    hourly = source["tp"].sel(time=slice(source_start, end_time))
    assert_regular_time(hourly.time.values, cadence_hours=1, name="ARCO total precipitation")

    total = hourly.resample(time="6h", label="right", closed="right").sum(skipna=False)
    total = total.sel(time=slice(start_time, end_time))
    assert_expected_time(
        total.time.values,
        year=year,
        cadence_hours=6,
        name="6-hourly ARCO total precipitation",
        end_time=end_time,
    )

    total = total.rename("total_precipitation")
    total.attrs = dict(source["tp"].attrs)
    total.attrs.update(
        {
            "long_name": "Total precipitation over the preceding 6 hours",
            "units": "m",
            "cell_methods": "time: sum",
            "comment": (
                "Each value at time T is the sum of hourly ERA5 total precipitation "
                "accumulations over (T - 6 hours, T]."
            ),
        }
    )
    return total


def select_year(
    source: xr.Dataset,
    year: int,
    *,
    source_url: str = SOURCE,
    allow_partial_year: bool = False,
) -> xr.Dataset:
    """Create the canonical 6-hourly local cache for a full or current partial year."""
    start = year_start(year)
    end = full_year_end(year, cadence_hours=6)
    if allow_partial_year:
        end = latest_complete_six_hour_time(source, year)
    missing = [name for name in SOURCE_TO_CACHE_VARIABLE if name not in source]
    if missing:
        raise KeyError(f"ARCO source is missing expected variables: {missing}")

    instantaneous = source[list(INSTANTANEOUS_SOURCE_VARIABLES)].sel(
        time=slice(start, end)
    )
    assert_expected_time(
        instantaneous.time.values,
        year=year,
        cadence_hours=1,
        name="ARCO instantaneous fields",
        end_time=end,
    )

    precipitation = six_hourly_precipitation(source, year, end_time=end)
    instantaneous = instantaneous.sel(time=precipitation.time).rename(
        {name: SOURCE_TO_CACHE_VARIABLE[name] for name in INSTANTANEOUS_SOURCE_VARIABLES}
    )
    result = xr.merge([instantaneous, precipitation], compat="override", join="exact")
    result = result[list(VARIABLES)].transpose("time", "latitude", "longitude")

    if result.sizes["latitude"] != 721 or result.sizes["longitude"] != 1440:
        raise ValueError(
            "Unexpected source grid: "
            f"{result.sizes['latitude']} x {result.sizes['longitude']}"
        )

    assert_expected_time(
        result.time.values,
        year=year,
        cadence_hours=6,
        name="selected ARCO cache data",
        end_time=end,
    )

    result.attrs = dict(source.attrs)
    result.attrs.update(
        {
            "cache_source": "ECMWF ERA5 ARCO time-chunked surface Zarr",
            "cache_source_url": source_url,
            "cache_subset": ", ".join(VARIABLES),
            "cache_year": int(year),
            "cache_complete_year": bool(end == full_year_end(year, cadence_hours=6)),
            "cache_time_coverage_end": np.datetime_as_string(end, unit="s"),
            "cache_grid": "0.25 degree, 1440x721; latitude [-90, 90], longitude [-180, 180)",
            "cache_temporal_resolution": "6 hourly",
            "total_precipitation_definition": "6-hour accumulation ending at each timestamp",
        }
    )
    return result


def make_encoding(
    time_chunk: int,
    latitude_chunk: int,
    longitude_chunk: int,
) -> dict[str, dict[str, object]]:
    compressor = Blosc(
        cname="zstd",
        clevel=3,
        shuffle=Blosc.BITSHUFFLE,
    )
    return {
        variable: {
            "dtype": "float32",
            "compressor": compressor,
            "chunks": (time_chunk, latitude_chunk, longitude_chunk),
        }
        for variable in VARIABLES
    }


def output_paths(output_root: Path, year: int) -> tuple[Path, Path]:
    final_path = output_root / f"{STORE_PREFIX}_{year:04d}.zarr"
    job_token = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    staging_path = output_root / (
        f".{STORE_PREFIX}_{year:04d}.zarr.partial-{job_token}"
    )
    return final_path, staging_path


def write_year(
    year_ds: xr.Dataset,
    staging_path: Path,
    *,
    workers: int,
    time_chunk: int,
    latitude_chunk: int,
    longitude_chunk: int,
) -> None:
    n_time = year_ds.sizes["time"]
    encoding = make_encoding(time_chunk, latitude_chunk, longitude_chunk)

    dask.config.set(
        scheduler="threads",
        num_workers=workers,
        **{
            "array.slicing.split_large_chunks": True,
            "temporary-directory": os.environ.get("TMPDIR", "/tmp"),
        },
    )

    for block_number, start in enumerate(range(0, n_time, time_chunk)):
        stop = min(start + time_chunk, n_time)
        block = year_ds.isel(time=slice(start, stop))

        logical_gib = block.nbytes / 1024**3
        log(
            f"Loading block {block_number + 1}: time[{start}:{stop}] "
            f"({logical_gib:.2f} GiB logical)"
        )

        with ProgressBar(minimum=1.0):
            block = block.load()

        # The source fields are already float32, but enforce the cache contract.
        for variable in VARIABLES:
            block[variable] = block[variable].astype("float32", copy=False)

        # Do not carry the source Zarr's time=1/full-globe encoding into the
        # destination. The first write below establishes the local encoding.
        for name in block.variables:
            block[name].encoding = {}

        block = block.chunk(
            {
                "time": min(time_chunk, block.sizes["time"]),
                "latitude": latitude_chunk,
                "longitude": longitude_chunk,
            }
        )

        if start == 0:
            log(f"Creating staging store: {staging_path}")
            with ProgressBar(minimum=1.0):
                block.to_zarr(
                    staging_path,
                    mode="w",
                    consolidated=False,
                    encoding=encoding,
                    safe_chunks=False,
                )
        else:
            log(f"Appending block {block_number + 1} to staging store")
            with ProgressBar(minimum=1.0):
                block.to_zarr(
                    staging_path,
                    mode="a",
                    append_dim="time",
                    consolidated=False,
                    safe_chunks=False,
                )

        # Release the roughly 3 GiB in-memory block before the next download.
        del block

    log("Consolidating Zarr metadata")
    zarr.consolidate_metadata(str(staging_path))


def validate_store(
    source_year: xr.Dataset,
    store_path: Path,
) -> None:
    log("Validating dimensions, coordinates, cadence, variables, and samples")
    cached = xr.open_zarr(store_path, consolidated=True, chunks={})

    expected_sizes = {
        "time": source_year.sizes["time"],
        "latitude": source_year.sizes["latitude"],
        "longitude": source_year.sizes["longitude"],
    }
    actual_sizes = {name: cached.sizes[name] for name in expected_sizes}

    if actual_sizes != expected_sizes:
        raise ValueError(
            f"Cached dimensions differ from source: {actual_sizes} != {expected_sizes}"
        )

    if set(VARIABLES) - set(cached.data_vars):
        raise ValueError(
            f"Cached store is missing variables: {set(VARIABLES) - set(cached.data_vars)}"
        )

    np.testing.assert_array_equal(cached.time.values, source_year.time.values)
    np.testing.assert_array_equal(cached.latitude.values, source_year.latitude.values)
    np.testing.assert_array_equal(cached.longitude.values, source_year.longitude.values)

    cache_year = int(source_year.time.dt.year[0])
    np.testing.assert_array_equal(
        cached.time.values,
        expected_times(cache_year, cadence_hours=6, end_time=source_year.time.values[-1]),
    )

    # Compare a small spread of points and both ends of the yearly time range.
    time_indices = sorted(set([0, cached.sizes["time"] // 2, cached.sizes["time"] - 1]))
    latitude_indices = [0, cached.sizes["latitude"] // 2, cached.sizes["latitude"] - 1]
    longitude_indices = [0, cached.sizes["longitude"] // 2, cached.sizes["longitude"] - 1]

    source_sample = source_year[list(VARIABLES)].isel(
        time=time_indices,
        latitude=latitude_indices,
        longitude=longitude_indices,
    ).load()
    cached_sample = cached[list(VARIABLES)].isel(
        time=time_indices,
        latitude=latitude_indices,
        longitude=longitude_indices,
    ).load()

    xr.testing.assert_allclose(cached_sample, source_sample)
    log("Validation passed")


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGUSR1, handle_usr1)

    if args.year < 1941:
        raise ValueError(
            "This 6-hour precipitation cache requires data from the preceding day; "
            "choose a year from 1941 onward."
        )

    if args.workers < 1:
        raise ValueError("--workers must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    final_path, staging_path = output_paths(args.output_root, args.year)
    success_marker = final_path / "_SUCCESS"

    if final_path.exists():
        if success_marker.exists() and not args.overwrite:
            log(f"Completed store already exists; skipping: {final_path}")
            return
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to replace existing store without --overwrite: {final_path}"
            )
        log(f"Removing existing store because --overwrite was supplied: {final_path}")
        shutil.rmtree(final_path)

    if staging_path.exists():
        log(f"Removing stale staging store: {staging_path}")
        shutil.rmtree(staging_path)

    source = open_source(args.source, get_cdsapi_key(args.cdsapirc))
    year_ds = select_year(
        source,
        args.year,
        source_url=args.source,
        allow_partial_year=args.allow_partial_year,
    )

    log(
        f"Selected year {args.year}: "
        f"{year_ds.sizes['time']} times x "
        f"{year_ds.sizes['latitude']} lat x "
        f"{year_ds.sizes['longitude']} lon x "
        f"{len(VARIABLES)} variables; "
        f"{year_ds.nbytes / 1024**3:.2f} GiB logical"
    )
    log(
        f"Coverage through {np.datetime_as_string(year_ds.time.values[-1], unit='h')}; "
        f"complete_year={year_ds.attrs['cache_complete_year']}"
    )

    try:
        write_year(
            year_ds,
            staging_path,
            workers=args.workers,
            time_chunk=args.time_chunk,
            latitude_chunk=args.latitude_chunk,
            longitude_chunk=args.longitude_chunk,
        )
        validate_store(year_ds, staging_path)

        log(f"Atomically promoting staging store to final path: {final_path}")
        os.replace(staging_path, final_path)

        success_marker.write_text(
            f"year={args.year}\n"
            f"source={args.source}\n"
            "source_format=ECMWF ERA5 ARCO timeChunked.zarr\n"
            f"variables={','.join(VARIABLES)}\n"
            f"complete_year={year_ds.attrs['cache_complete_year']}\n"
            f"time_coverage_end={year_ds.attrs['cache_time_coverage_end']}\n"
            f"completed_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        )
        log(f"Finished successfully: {final_path}")
    except BaseException:
        log(
            "Year did not complete. The final path was not published; "
            f"staging remains at {staging_path}"
        )
        raise


if __name__ == "__main__":
    main()
