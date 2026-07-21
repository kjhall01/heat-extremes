#!/usr/bin/env python3
"""
Cache one year of selected WeatherBench2 ERA5 surface variables to local Zarr.

The source is the public WeatherBench2 0.25-degree, 6-hourly ERA5 store.
Each year is written to a temporary directory, validated, and atomically renamed
to its final path. Re-running a completed year is therefore safe.

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
import gcsfs
import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc


SOURCE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

VARIABLES = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
)

DEFAULT_OUTPUT_ROOT = Path("/net/monsoon/kylehall/ERA5/wb2_era5_6h_surface")


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
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        help="Number of threaded Dask workers used for GCS reads and local writes.",
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
        "--overwrite",
        action="store_true",
        help="Replace an already completed yearly store.",
    )
    return parser.parse_args()


def open_source(source: str) -> xr.Dataset:
    log(f"Opening public GCS source: {source}")
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(source)
    return xr.open_zarr(
        mapper,
        consolidated=True,
        chunks={},  # Preserve native WB2 chunks: one full global field per time.
    )


def select_year(source: xr.Dataset, year: int) -> xr.Dataset:
    start = f"{year:04d}-01-01T00:00:00"
    end = f"{year:04d}-12-31T23:59:59"

    missing = [name for name in VARIABLES if name not in source]
    if missing:
        raise KeyError(f"Source is missing expected variables: {missing}")

    result = (
        source[list(VARIABLES)]
        .sel(time=slice(start, end))
        .transpose("time", "latitude", "longitude")
    )

    if result.sizes["time"] == 0:
        raise ValueError(f"No source data found for year {year}")

    if result.sizes["latitude"] != 721 or result.sizes["longitude"] != 1440:
        raise ValueError(
            "Unexpected source grid: "
            f"{result.sizes['latitude']} x {result.sizes['longitude']}"
        )

    time_values = result.time.values
    if time_values.size > 1:
        hours = np.diff(time_values) / np.timedelta64(1, "h")
        if not np.all(hours == 6):
            bad = np.where(hours != 6)[0][:10]
            raise ValueError(f"Non-6-hourly source cadence at indices {bad.tolist()}")

    result.attrs = dict(source.attrs)
    result.attrs.update(
        {
            "cache_source": SOURCE,
            "cache_subset": ", ".join(VARIABLES),
            "cache_year": int(year),
            "cache_grid": "0.25 degree, 1440x721 with poles",
            "cache_temporal_resolution": "6 hourly",
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
    final_path = output_root / f"wb2_era5_6h_surface_{year:04d}.zarr"
    job_token = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    staging_path = output_root / (
        f".wb2_era5_6h_surface_{year:04d}.zarr.partial-{job_token}"
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

    if cached.sizes["time"] > 1:
        hours = np.diff(cached.time.values) / np.timedelta64(1, "h")
        if not np.all(hours == 6):
            raise ValueError("Cached time coordinate is not uniformly 6-hourly")

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

    if not (1959 <= args.year <= 2023):
        raise ValueError("The selected WeatherBench2 store covers 1959 through 2023-01-10")

    if args.workers < 1:
        raise ValueError("--workers must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    final_path, staging_path = output_paths(args.output_root, args.year)
    success_marker = final_path / "_SUCCESS"

    if final_path.exists() and success_marker.exists() and not args.overwrite:
        log(f"Completed store already exists; skipping: {final_path}")
        return

   # if final_path.exists():
   #     if args.overwrite:
   #         log(f"Removing existing final store because --overwrite was supplied: {final_path}")
   #     else:
   #         log(f"Removing incomplete final store without _SUCCESS marker: {final_path}")
   #     shutil.rmtree(final_path)

   # if staging_path.exists():
   #     log(f"Removing stale staging store from an earlier attempt: {staging_path}")
   #     shutil.rmtree(staging_path)

    source = open_source(args.source)
    year_ds = select_year(source, args.year)

    log(
        f"Selected year {args.year}: "
        f"{year_ds.sizes['time']} times x "
        f"{year_ds.sizes['latitude']} lat x "
        f"{year_ds.sizes['longitude']} lon x "
        f"{len(VARIABLES)} variables; "
        f"{year_ds.nbytes / 1024**3:.2f} GiB logical"
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
            f"variables={','.join(VARIABLES)}\n"
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
