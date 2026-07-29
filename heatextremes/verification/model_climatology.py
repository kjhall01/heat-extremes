"""Metadata preflight helpers for lead-dependent model temperature climatologies.

The preflight deliberately reads only Zarr metadata and coordinate arrays.  It
uses the same local-solar day construction as the standard reforecast adapter
to estimate the sample support available for a circular calendar-day window.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar

from .io import now_utc, remove_result_path, write_json_atomic
from .models.standard_reforecast import _processing_helpers
from .reforecast_inventory import ReforecastModelInventory, store_year_month


DAY_OF_YEAR_COUNT = 366
MODEL_Q95_VARIABLE = "t2m_daily_mean_model_q95"
SAMPLE_COUNT_VARIABLE = "q95_sample_count"
MODEL_Q95_PRODUCT_VERSION = 1
MODEL_Q95_OUTPUT_CHUNKS = {
    "forecast_day": 1,
    "dayofyear": DAY_OF_YEAR_COUNT,
    "latitude": 90,
    "longitude": 90,
}


@dataclass(frozen=True)
class RawStorePreflight:
    """Metadata findings for one raw reforecast Zarr store."""

    path: Path
    initialization_count: int
    initialization_start: str | None
    initialization_end: str | None
    available_forecast_days: tuple[int, ...]
    grid_shape: tuple[int, int]
    member_count: int | None
    step_hours: tuple[int, ...]
    valid_day_counts: np.ndarray


def circular_window_counts(daily_counts: np.ndarray, *, window_days: int) -> np.ndarray:
    """Sum day-of-year counts over a centered circular calendar window."""
    if window_days < 1 or window_days % 2 == 0:
        raise ValueError("window_days must be a positive odd integer")
    if daily_counts.shape[-1] != DAY_OF_YEAR_COUNT:
        raise ValueError(f"daily_counts must end in {DAY_OF_YEAR_COUNT} day-of-year bins")
    radius = window_days // 2
    return sum(np.roll(daily_counts, shift, axis=-1) for shift in range(-radius, radius + 1))


def _canonical_temperature(dataset: xr.Dataset, variable: str) -> xr.DataArray:
    if variable not in dataset:
        raise KeyError(f"Raw store is missing temperature variable {variable!r}")
    temperature = dataset[variable]
    renames = {
        old: new
        for old, new in (("lat", "latitude"), ("lon", "longitude"))
        if old in temperature.dims or old in temperature.coords
    }
    if renames:
        temperature = temperature.rename(renames)
    required = {"time", "prediction_timedelta", "latitude", "longitude"}
    missing = sorted(required.difference(temperature.dims))
    if missing:
        raise ValueError(f"Raw temperature is missing dimensions: {missing}")
    return temperature.sortby("time").sortby("longitude")


def _representative_longitude_indices(
    temperature: xr.DataArray, helpers: Any
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Choose one longitude per verified local-solar offset band."""
    longitudes = np.asarray(temperature["longitude"].values, dtype=float)
    indices: list[int] = []
    labels: list[str] = []
    for start, stop, offset in helpers.longitude_band_specs(temperature["longitude"]):
        in_band = np.flatnonzero((longitudes >= start) & (longitudes < stop))
        if not in_band.size:
            continue
        indices.append(int(in_band[len(in_band) // 2]))
        labels.append(f"utc{offset:+03d}")
    if not indices:
        raise ValueError("No longitude bands were available for local-solar preflight")
    return np.asarray(indices, dtype=int), tuple(labels)


def _day_of_year_counts(
    valid_dates: xr.DataArray,
    *,
    forecast_days: tuple[int, ...],
    band_labels: tuple[str, ...],
) -> np.ndarray:
    """Count valid local dates by lead, longitude band, and day of year."""
    selected = valid_dates.sel(forecast_day=list(forecast_days))
    values = np.asarray(selected.values)
    if values.ndim != 3:
        raise ValueError(f"Expected valid_date(time, forecast_day, longitude), got {selected.dims}")
    if values.shape[2] != len(band_labels):
        raise ValueError("Representative-longitude selection did not preserve all local-solar bands")
    counts = np.zeros((len(forecast_days), len(band_labels), DAY_OF_YEAR_COUNT), dtype=np.int32)
    for lead_index in range(len(forecast_days)):
        for band_index in range(len(band_labels)):
            dates = values[:, lead_index, band_index]
            dates = dates[~pd.isnull(dates)]
            if len(dates):
                day_of_year = pd.DatetimeIndex(dates).dayofyear.to_numpy(dtype=int)
                counts[lead_index, band_index] += np.bincount(
                    day_of_year - 1, minlength=DAY_OF_YEAR_COUNT
                )
    return counts


def inspect_raw_store(
    path: Path,
    *,
    variable: str,
    ensemble: bool,
    forecast_days: Iterable[int],
) -> RawStorePreflight:
    """Inspect one raw store without loading temperature chunks."""
    requested_days = tuple(sorted(set(int(day) for day in forecast_days)))
    if not requested_days:
        raise ValueError("forecast_days must not be empty")
    helpers = _processing_helpers()
    dataset = xr.open_zarr(path, consolidated=None, chunks={})
    try:
        temperature = _canonical_temperature(dataset, variable)
        if ensemble and "number" not in temperature.dims:
            raise ValueError("Expected ensemble dimension 'number' is absent")
        if not ensemble and temperature.sizes.get("number", 1) > 1:
            raise ValueError("Deterministic model has multiple raw ensemble members")

        step_hours = np.asarray(
            temperature["prediction_timedelta"].values / np.timedelta64(1, "h"), dtype=np.int64
        )
        if not len(step_hours) or np.any(np.diff(step_hours) != 6):
            raise ValueError("prediction_timedelta must be a complete six-hourly sequence")

        longitude_indices, band_labels = _representative_longitude_indices(temperature, helpers)
        # Only coordinate arrays are needed after this selection.  The helper's
        # temperature mean remains lazy; valid_date is derived from coordinates.
        coordinate_sample = temperature.isel(
            latitude=slice(0, 1), longitude=longitude_indices
        )
        # Match the metadata inventory: construct the full supported
        # local-solar horizon first, then verify the requested subset.  A
        # longitude band whose local midnight is late in the UTC horizon can
        # otherwise lack four samples when a shorter cap is applied up front.
        daily = helpers.local_solar_daily_mean_forecast(
            coordinate_sample, max_days=helpers.MAX_FORECAST_DAYS
        )
        available_days = tuple(int(day) for day in daily["forecast_day"].values)
        absent = sorted(set(requested_days).difference(available_days))
        if absent:
            raise ValueError(f"Missing required local-solar forecast days: {absent}")
        valid_counts = _day_of_year_counts(
            daily["valid_date"], forecast_days=requested_days, band_labels=band_labels
        )
        initialization = pd.DatetimeIndex(temperature["time"].values)
        return RawStorePreflight(
            path=path,
            initialization_count=len(initialization),
            initialization_start=initialization.min().isoformat() if len(initialization) else None,
            initialization_end=initialization.max().isoformat() if len(initialization) else None,
            available_forecast_days=available_days,
            grid_shape=(temperature.sizes["latitude"], temperature.sizes["longitude"]),
            member_count=int(temperature.sizes["number"]) if "number" in temperature.dims else None,
            step_hours=tuple(int(value) for value in step_hours),
            valid_day_counts=valid_counts,
        )
    finally:
        dataset.close()


def preflight_model(
    inventory: ReforecastModelInventory,
    *,
    years: Iterable[int],
    months: Iterable[int],
    forecast_days: Iterable[int],
    window_days: int,
    minimum_samples: int,
) -> dict[str, Any]:
    """Audit historical source coverage and q95 sample support for one model."""
    requested_years = tuple(sorted(set(int(year) for year in years)))
    requested_months = tuple(sorted(set(int(month) for month in months)))
    requested_days = tuple(sorted(set(int(day) for day in forecast_days)))
    requested_partitions = tuple(
        (year, month) for year in requested_years for month in requested_months
    )
    stores_by_partition: dict[tuple[int, int], list[Path]] = {
        partition: [] for partition in requested_partitions
    }
    unparsed_stores: list[str] = []
    for path in sorted(inventory.directory.glob(inventory.source_store_glob)):
        partition = store_year_month(path)
        if partition is None:
            unparsed_stores.append(path.name)
        elif partition in stores_by_partition:
            stores_by_partition[partition].append(path)

    valid_counts: np.ndarray | None = None
    errors: list[dict[str, str]] = []
    initialization_total = 0
    initialization_starts: list[str] = []
    initialization_ends: list[str] = []
    grid_shapes: Counter[tuple[int, int]] = Counter()
    member_counts: Counter[int | None] = Counter()
    checked_stores = 0

    for partition, stores in stores_by_partition.items():
        for path in stores:
            try:
                result = inspect_raw_store(
                    path,
                    variable=inventory.source_temperature_variable,
                    ensemble=inventory.ensemble,
                    forecast_days=requested_days,
                )
            except Exception as error:
                errors.append(
                    {
                        "partition": f"{partition[0]:04d}-{partition[1]:02d}",
                        "store": path.name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            checked_stores += 1
            initialization_total += result.initialization_count
            if result.initialization_start:
                initialization_starts.append(result.initialization_start)
            if result.initialization_end:
                initialization_ends.append(result.initialization_end)
            grid_shapes[result.grid_shape] += 1
            member_counts[result.member_count] += 1
            valid_counts = (
                result.valid_day_counts.copy()
                if valid_counts is None
                else valid_counts + result.valid_day_counts
            )

    missing_partitions = [
        f"{year:04d}-{month:02d}"
        for year, month in requested_partitions
        if not stores_by_partition[(year, month)]
    ]
    if valid_counts is None:
        support = np.zeros((len(requested_days), 0, DAY_OF_YEAR_COUNT), dtype=np.int32)
    else:
        support = circular_window_counts(valid_counts, window_days=window_days)

    support_summary: list[dict[str, Any]] = []
    for index, forecast_day in enumerate(requested_days):
        values = support[index].ravel()
        support_summary.append(
            {
                "forecast_day": forecast_day,
                "minimum": int(values.min()) if values.size else 0,
                "p05": float(np.percentile(values, 5)) if values.size else 0.0,
                "median": float(np.median(values)) if values.size else 0.0,
                "maximum": int(values.max()) if values.size else 0,
                "cells_below_minimum": int((values < minimum_samples).sum()),
                "total_band_day_cells": int(values.size),
            }
        )

    all_leads_supported = bool(support_summary) and all(
        item["minimum"] >= minimum_samples for item in support_summary
    )
    layout_consistent = len(grid_shapes) <= 1 and len(member_counts) <= 1
    status = (
        "ready"
        if not errors and not missing_partitions and all_leads_supported and layout_consistent
        else "review_required"
    )
    return {
        "model": inventory.name,
        "display_name": inventory.display_name,
        "status": status,
        "raw_directory": str(inventory.directory),
        "source_temperature_variable": inventory.source_temperature_variable,
        "source_store_glob": inventory.source_store_glob,
        "ensemble": inventory.ensemble,
        "forecast_days": list(requested_days),
        "requested_partitions": len(requested_partitions),
        "present_partitions": len(requested_partitions) - len(missing_partitions),
        "missing_partitions": missing_partitions,
        "discovered_stores": sum(len(paths) for paths in stores_by_partition.values()),
        "checked_stores": checked_stores,
        "metadata_errors": errors,
        "unparsed_store_names": unparsed_stores,
        "initialization_count": initialization_total,
        "initialization_start": min(initialization_starts) if initialization_starts else None,
        "initialization_end": max(initialization_ends) if initialization_ends else None,
        "grid_shapes": {f"{latitude}x{longitude}": count for (latitude, longitude), count in grid_shapes.items()},
        "member_counts": {str(count): stores for count, stores in member_counts.items()},
        "consistent_grid_and_member_layout": layout_consistent,
        "q95_window_days": window_days,
        "q95_minimum_samples": minimum_samples,
        "q95_support_by_lead": support_summary,
    }


# ---------------------------------------------------------------------------
# Final, global model-climatology products
#
# The helpers below deliberately use longitude bands only as *work units*.
# Each array task writes its non-overlapping longitude region directly into one
# global Zarr store.  There is never a daily-temperature cache or a collection
# of band products to merge afterwards.


def _payload_digest(payload: dict[str, Any]) -> str:
    """Return a stable signature for a small JSON-serializable manifest."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _selected_raw_stores(
    directory: Path,
    *,
    years: Iterable[int],
    months: Iterable[int],
    source_store_glob: str = "*.zarr",
) -> list[Path]:
    """Find the exact historical raw stores recorded by a preflight report."""
    wanted = {(int(year), int(month)) for year in years for month in months}
    paths = [
        path
        for path in sorted(directory.glob(source_store_glob))
        if path.is_dir() and store_year_month(path) in wanted
    ]
    if not paths:
        raise FileNotFoundError(
            f"No dated raw Zarr stores under {directory} for the requested climatology period"
        )
    return paths


def _longitude_band_records(longitude: xr.DataArray) -> list[dict[str, int | float]]:
    """Translate verified local-solar bands into contiguous global indices."""
    helpers = _processing_helpers()
    values = np.asarray(longitude.values, dtype=float)
    records: list[dict[str, int | float]] = []
    for start, stop, offset_hours in helpers.longitude_band_specs(longitude):
        selected = np.flatnonzero((values >= start) & (values < stop))
        if not selected.size:
            continue
        expected = np.arange(selected[0], selected[-1] + 1)
        if not np.array_equal(selected, expected):
            raise ValueError(
                f"Longitude band [{start}, {stop}) is not contiguous on the source grid"
            )
        records.append(
            {
                "index": len(records),
                "longitude_start_index": int(selected[0]),
                "longitude_stop_index": int(selected[-1] + 1),
                "longitude_start": float(start),
                "longitude_stop": float(stop),
                "local_solar_offset_hours": int(offset_hours),
            }
        )
    if not records:
        raise ValueError("No local-solar longitude bands overlap the source grid")
    covered = np.concatenate(
        [
            np.arange(record["longitude_start_index"], record["longitude_stop_index"])
            for record in records
        ]
    )
    if not np.array_equal(covered, np.arange(values.size)):
        raise ValueError(
            "Local-solar longitude bands do not cover the complete source longitude grid"
        )
    return records


def _coordinate_schema(
    path: Path, *, variable: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int | float]]]:
    """Read one source store's small coordinate arrays, never temperature data."""
    dataset = xr.open_zarr(path, consolidated=None, chunks={})
    try:
        temperature = _canonical_temperature(dataset, variable)
        latitude = np.asarray(temperature["latitude"].values)
        longitude = np.asarray(temperature["longitude"].values)
        bands = _longitude_band_records(temperature["longitude"])
        return latitude, longitude, bands
    finally:
        dataset.close()


def q95_product_store(model_directory: Path) -> Path:
    """Return the one global Zarr product path for a model."""
    return model_directory / "q95.zarr"


def _set_zarr_dimensions(array: zarr.Array, dimensions: tuple[str, ...]) -> None:
    array.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)


def _open_output_group(path: Path, *, mode: str) -> zarr.Group:
    """Open a Zarr v2 output group under either supported zarr-python API."""
    version_keyword = (
        "zarr_format"
        if "zarr_format" in inspect.signature(zarr.open_group).parameters
        else "zarr_version"
    )
    try:
        return zarr.open_group(str(path), mode=mode, **{version_keyword: 2})
    except (TypeError, ValueError):
        if mode == "w":
            raise
        # A failed first attempt under zarr-python 3 may have left a tiny v3
        # group behind. Open it only so bootstrap can recognize and replace
        # the incomplete schema with the intended v2 product.
        return zarr.open_group(str(path), mode=mode)


def _global_store_has_schema(group: zarr.Group) -> bool:
    """Reject a schema left behind by an interrupted initialization."""
    required = {
        "forecast_day",
        "dayofyear",
        "latitude",
        "longitude",
        MODEL_Q95_VARIABLE,
        SAMPLE_COUNT_VARIABLE,
    }
    return required.issubset(group.array_keys())


def _create_global_q95_store(
    store: Path,
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    forecast_days: tuple[int, ...],
    manifest_sha256: str,
    model_name: str,
    years: tuple[int, ...],
    months: tuple[int, ...],
    window_days: int,
    percentile: float,
    longitude_chunk: int,
) -> None:
    """Create metadata and empty final arrays without materializing NaN chunks."""
    store.parent.mkdir(parents=True, exist_ok=True)
    group = _open_output_group(store, mode="w")
    group.attrs.update(
        {
            "title": "Lead-dependent local-solar daily-mean model temperature climatology",
            "product_version": MODEL_Q95_PRODUCT_VERSION,
            "model": model_name,
            "manifest_sha256": manifest_sha256,
            "climatology_years": list(years),
            "climatology_months": list(months),
            "percentile": float(percentile),
            "calendar_window_days": int(window_days),
            "daily_time_basis": "six-hour UTC-offset longitude-band local-solar day",
            "source": "raw model reforecasts; read only",
        }
    )
    coordinate_specs = {
        "forecast_day": (np.asarray(forecast_days, dtype=np.int16), ("forecast_day",)),
        "dayofyear": (np.arange(1, DAY_OF_YEAR_COUNT + 1, dtype=np.int16), ("dayofyear",)),
        "latitude": (np.asarray(latitude), ("latitude",)),
        "longitude": (np.asarray(longitude), ("longitude",)),
    }
    for name, (values, dimensions) in coordinate_specs.items():
        # zarr-python 3 requires shape even when data are supplied. Creating
        # then assigning these tiny coordinate arrays also works in zarr 2.
        array = group.create_dataset(
            name,
            shape=values.shape,
            chunks=values.shape,
            dtype=values.dtype,
        )
        array[...] = values
        _set_zarr_dimensions(array, dimensions)

    q95_shape = (len(forecast_days), DAY_OF_YEAR_COUNT, len(latitude), len(longitude))
    q95_chunks = (
        min(MODEL_Q95_OUTPUT_CHUNKS["forecast_day"], q95_shape[0]),
        min(MODEL_Q95_OUTPUT_CHUNKS["dayofyear"], q95_shape[1]),
        min(MODEL_Q95_OUTPUT_CHUNKS["latitude"], q95_shape[2]),
        min(int(longitude_chunk), q95_shape[3]),
    )
    q95 = group.create_dataset(
        MODEL_Q95_VARIABLE,
        shape=q95_shape,
        chunks=q95_chunks,
        dtype=np.float32,
        fill_value=np.nan,
    )
    _set_zarr_dimensions(q95, ("forecast_day", "dayofyear", "latitude", "longitude"))
    q95.attrs.update(
        {
            "long_name": "Model local-solar daily-mean 2 m temperature calendar-day percentile",
            "units": "K",
            "percentile": float(percentile),
            "calendar_window_days": int(window_days),
        }
    )
    sample_count = group.create_dataset(
        SAMPLE_COUNT_VARIABLE,
        shape=(len(forecast_days), DAY_OF_YEAR_COUNT, len(longitude)),
        chunks=(q95_chunks[0], q95_chunks[1], q95_chunks[3]),
        dtype=np.uint16,
        fill_value=0,
    )
    _set_zarr_dimensions(sample_count, ("forecast_day", "dayofyear", "longitude"))
    sample_count.attrs.update(
        {
            "long_name": (
                "Number of raw local-solar daily-mean cases used in each percentile window"
            ),
            "calendar_window_days": int(window_days),
        }
    )


def build_q95_workflow_manifest(
    preflight: dict[str, Any],
    *,
    result_root: Path,
    output_directory: Path,
    years: Iterable[int],
    months: Iterable[int],
    forecast_days: Iterable[int],
    window_days: int,
    percentile: float,
    models: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create one empty global output per ready model and a compact task manifest.

    The preflight is intentionally authoritative: this function will not run a
    second metadata audit or touch source values.  It only reads one model
    store's coordinate arrays to define the global Zarr schema.
    """
    requested_years = tuple(sorted(set(int(year) for year in years)))
    requested_months = tuple(sorted(set(int(month) for month in months)))
    requested_days = tuple(sorted(set(int(day) for day in forecast_days)))
    if not requested_years or not requested_months or not requested_days:
        raise ValueError("years, months, and forecast_days must not be empty")
    if any(day < 0 or day >= 15 for day in requested_days):
        raise ValueError("forecast_days must remain within the verified 0--14 local-solar range")
    if window_days < 1 or window_days % 2 == 0:
        raise ValueError("window_days must be a positive odd integer")
    if not 0 < percentile < 100:
        raise ValueError("percentile must be strictly between 0 and 100")

    preflight_years = tuple(int(year) for year in preflight.get("requested_years", ()))
    preflight_months = tuple(int(month) for month in preflight.get("requested_months", ()))
    if preflight_years and preflight_years != requested_years:
        raise ValueError(
            "The supplied preflight report covers different years; rerun the metadata-only "
            "preflight "
            "for this climatology period."
        )
    if preflight_months and preflight_months != requested_months:
        raise ValueError(
            "The supplied preflight report covers different months; rerun the metadata-only "
            "preflight "
            "for this climatology period."
        )
    if int(preflight.get("q95_window_days", window_days)) != window_days:
        raise ValueError("The preflight report used a different q95 window length")

    selected = set(models or ())
    reports = [
        report
        for report in preflight.get("models", [])
        if not selected or report["model"] in selected
    ]
    unknown = selected.difference(report["model"] for report in reports)
    if unknown:
        raise ValueError(
            f"Requested models are absent from the preflight report: {sorted(unknown)}"
        )
    if not reports:
        raise ValueError("No models were selected from the preflight report")
    unready = [report["model"] for report in reports if report.get("status") != "ready"]
    if unready:
        raise ValueError(f"Refusing live q95 computation for preflight-unready models: {unready}")

    result_root = result_root.expanduser().resolve(strict=False)
    output_directory = output_directory.expanduser().resolve(strict=False)
    if output_directory == result_root or result_root not in output_directory.parents:
        raise ValueError("output_directory must be a dedicated child of result_root")

    model_specs: list[dict[str, Any]] = []
    for report in sorted(reports, key=lambda item: item["model"]):
        raw_directory = Path(report["raw_directory"])
        source_glob = str(report.get("source_store_glob", "*.zarr"))
        source_paths = _selected_raw_stores(
            raw_directory,
            years=requested_years,
            months=requested_months,
            source_store_glob=source_glob,
        )
        expected_store_count = int(report.get("checked_stores", len(source_paths)))
        if len(source_paths) != expected_store_count:
            raise ValueError(
                f"{report['model']}: found {len(source_paths)} selected raw stores, but the "
                "ready preflight "
                f"checked {expected_store_count}. Refusing a different source set."
            )
        latitude, longitude, bands = _coordinate_schema(
            source_paths[0], variable=str(report["source_temperature_variable"])
        )
        common_band_width = math.gcd(
            *(
                int(band["longitude_stop_index"]) - int(band["longitude_start_index"])
                for band in bands
            )
        )
        # Every array region must begin and end on Zarr chunk boundaries. This
        # retains reasonably small chunks on the production 0.25° grid while
        # also supporting smaller/reduced test grids exactly.
        longitude_chunk = math.gcd(MODEL_Q95_OUTPUT_CHUNKS["longitude"], common_band_width)
        model_directory = output_directory / str(report["model"])
        model_specs.append(
            {
                "model": str(report["model"]),
                "display_name": str(report.get("display_name", report["model"])),
                "raw_directory": str(raw_directory),
                "source_temperature_variable": str(report["source_temperature_variable"]),
                "ensemble": bool(report["ensemble"]),
                "source_stores": [str(path) for path in source_paths],
                "source_store_count": len(source_paths),
                "latitude_count": int(latitude.size),
                "longitude_count": int(longitude.size),
                "output_longitude_chunk": int(longitude_chunk),
                "longitude_bands": bands,
                "product_directory": str(model_directory),
                "product_store": str(q95_product_store(model_directory)),
                # Stored only in the small workflow manifest. Coordinates are
                # not duplicated in the output or in band temporary files.
                "latitude": latitude.tolist(),
                "longitude": longitude.tolist(),
            }
        )

    manifest_core: dict[str, Any] = {
        "product": "lead_dependent_model_temperature_q95",
        "product_version": MODEL_Q95_PRODUCT_VERSION,
        "result_root": str(result_root),
        "output_directory": str(output_directory),
        "years": list(requested_years),
        "months": list(requested_months),
        "forecast_days": list(requested_days),
        "window_days": int(window_days),
        "percentile": float(percentile),
        "models": model_specs,
    }
    manifest_sha256 = _payload_digest(manifest_core)

    for spec in model_specs:
        model_directory = Path(spec["product_directory"])
        store = q95_product_store(model_directory)
        if overwrite and model_directory.exists():
            remove_result_path(model_directory, result_root)
        if store.exists():
            group = _open_output_group(store, mode="r")
            if group.attrs.get("manifest_sha256") != manifest_sha256:
                raise ValueError(
                    f"Existing q95 product has a different manifest: {store}. "
                    "Use --overwrite to replace it."
                )
            if not _global_store_has_schema(group):
                if (model_directory / "completion.json").is_file():
                    raise ValueError(
                        f"Completed q95 product has an invalid schema: {store}; "
                        "use --overwrite to replace it."
                    )
                remove_result_path(model_directory, result_root)
                _create_global_q95_store(
                    store,
                    latitude=np.asarray(spec["latitude"]),
                    longitude=np.asarray(spec["longitude"]),
                    forecast_days=requested_days,
                    manifest_sha256=manifest_sha256,
                    model_name=str(spec["model"]),
                    years=requested_years,
                    months=requested_months,
                    window_days=window_days,
                    percentile=percentile,
                    longitude_chunk=int(spec["output_longitude_chunk"]),
                )
        else:
            _create_global_q95_store(
                store,
                latitude=np.asarray(spec["latitude"]),
                longitude=np.asarray(spec["longitude"]),
                forecast_days=requested_days,
                manifest_sha256=manifest_sha256,
                model_name=str(spec["model"]),
                years=requested_years,
                months=requested_months,
                window_days=window_days,
                percentile=percentile,
                longitude_chunk=int(spec["output_longitude_chunk"]),
            )

    tasks = [
        {"model": spec["model"], "band_index": int(band["index"])}
        for spec in model_specs
        for band in spec["longitude_bands"]
    ]
    manifest = {
        **manifest_core,
        "manifest_sha256": manifest_sha256,
        "created_at": now_utc(),
        "task_count": len(tasks),
        "tasks": tasks,
        "raw_source_write_policy": "read_only",
        "daily_intermediate_storage": "none; daily means exist only in task memory",
    }
    return manifest


def _manifest_model(manifest: dict[str, Any], model_name: str) -> dict[str, Any]:
    for model in manifest["models"]:
        if model["model"] == model_name:
            return model
    raise KeyError(f"Model {model_name!r} is absent from the q95 workflow manifest")


def q95_band_marker(model_directory: Path, band_index: int) -> Path:
    return model_directory / "band_completion" / f"band_{band_index:02d}.json"


def _valid_band_marker(
    marker: Path,
    *,
    manifest_sha256: str,
    band_index: int,
) -> bool:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "complete"
        and payload.get("manifest_sha256") == manifest_sha256
        and payload.get("band_index") == band_index
    )


def _open_raw_ensemble_mean(model: dict[str, Any]) -> tuple[xr.Dataset, xr.DataArray]:
    """Open raw source stores read-only and reduce ensemble members immediately."""
    variable = str(model["source_temperature_variable"])

    def preprocess(dataset: xr.Dataset) -> xr.Dataset:
        if variable not in dataset:
            raise KeyError(f"Raw forecast store is missing temperature variable {variable!r}")
        return dataset[[variable]]

    raw = xr.open_mfdataset(
        [str(path) for path in model["source_stores"]],
        engine="zarr",
        combine="nested",
        concat_dim="time",
        preprocess=preprocess,
        chunks={"time": 1},
        parallel=True,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
        combine_attrs="override",
        consolidated=None,
    )
    temperature = _canonical_temperature(raw, variable).sortby("prediction_timedelta")
    temperature = temperature.chunk(
        {
            "time": 1,
            "prediction_timedelta": 24,
            "latitude": MODEL_Q95_OUTPUT_CHUNKS["latitude"],
            "longitude": MODEL_Q95_OUTPUT_CHUNKS["longitude"],
            **({"number": 26} if "number" in temperature.dims else {}),
        }
    )
    if bool(model["ensemble"]):
        if "number" not in temperature.dims:
            raw.close()
            raise ValueError(
                "Preflight marked this model as ensemble, but raw stores lack 'number'"
            )
        temperature = temperature.mean("number", skipna=True)
    elif "number" in temperature.dims:
        if temperature.sizes["number"] != 1:
            raw.close()
            raise ValueError(
                "Preflight marked this model deterministic, but raw stores have multiple members"
            )
        temperature = temperature.isel(number=0, drop=True)
    return raw, temperature.astype(np.float32)


def _local_solar_daily_mean_band(
    temperature: xr.DataArray,
    *,
    band: dict[str, Any],
    forecast_days: tuple[int, ...],
) -> xr.DataArray:
    """Build one local-solar longitude band's daily means without temporary files."""
    helpers = _processing_helpers()
    # Build the full project-standard 15-day local horizon before selecting
    # requested leads. This avoids a late-offset band losing its final complete
    # local day when a shorter raw UTC horizon is selected up front.
    temperature = temperature.where(
        temperature["prediction_timedelta"] < np.timedelta64(helpers.MAX_FORECAST_DAYS, "D"),
        drop=True,
    ).isel(
        longitude=slice(int(band["longitude_start_index"]), int(band["longitude_stop_index"]))
    )
    if not temperature.sizes.get("longitude", 0):
        raise ValueError(f"Longitude band {band['index']} selected no source grid cells")

    hourly_results: list[xr.DataArray] = []
    for init_hour in np.unique(temperature["time"].dt.hour.values):
        selected = temperature.sel(time=temperature["time"].dt.hour == int(init_hour))
        if selected.sizes["time"]:
            hourly_results.append(
                helpers.aggregate_init_hour_band_mean(
                    selected,
                    init_hour=int(init_hour),
                    offset_hours=int(band["local_solar_offset_hours"]),
                )
            )
    if not hourly_results:
        raise ValueError(f"Longitude band {band['index']} has no initialization hours")
    daily = xr.concat(
        hourly_results,
        dim="time",
        join="outer",
        coords="minimal",
        compat="override",
    ).sortby("time")
    available = {int(day) for day in daily["forecast_day"].values}
    absent = sorted(set(forecast_days).difference(available))
    if absent:
        raise ValueError(
            f"Longitude band {band['index']} lacks requested local-solar forecast day(s): {absent}"
        )
    return daily.sel(forecast_day=list(forecast_days)).astype(np.float32)


def _window_sample_counts(
    valid_date: xr.DataArray,
    *,
    forecast_days: tuple[int, ...],
    longitude_count: int,
    window_days: int,
) -> np.ndarray:
    """Return exact calendar-window case counts, repeated across one UTC-offset band."""
    representative = valid_date.isel(longitude=0)
    counts = np.zeros((len(forecast_days), DAY_OF_YEAR_COUNT), dtype=np.int32)
    for lead_index, forecast_day in enumerate(forecast_days):
        values = np.asarray(representative.sel(forecast_day=forecast_day).values)
        values = values[~pd.isnull(values)]
        if values.size:
            day_of_year = pd.DatetimeIndex(values).dayofyear.to_numpy(dtype=int)
            counts[lead_index] = np.bincount(day_of_year - 1, minlength=DAY_OF_YEAR_COUNT)
    windowed = circular_window_counts(counts, window_days=window_days)
    if windowed.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError("q95 sample counts exceed uint16 output capacity")
    return np.broadcast_to(
        windowed[..., np.newaxis].astype(np.uint16),
        (len(forecast_days), DAY_OF_YEAR_COUNT, longitude_count),
    ).copy()


def _calendar_window_quantile_numpy(
    values: np.ndarray,
    day_of_year: np.ndarray,
    *,
    window_days: int,
    quantile: float,
) -> np.ndarray:
    """Exact circular day-of-year quantile for one spatial NumPy block."""
    values = np.asarray(values, dtype=np.float32)
    day_of_year = np.asarray(day_of_year, dtype=np.int16)
    output = np.full(values.shape[:-1] + (DAY_OF_YEAR_COUNT,), np.nan, dtype=np.float32)
    radius = window_days // 2
    for target_day in range(1, DAY_OF_YEAR_COUNT + 1):
        circular_distance = np.abs(
            (
                (day_of_year.astype(np.int32) - target_day + DAY_OF_YEAR_COUNT // 2)
                % DAY_OF_YEAR_COUNT
            )
            - DAY_OF_YEAR_COUNT // 2
        )
        selected = circular_distance <= radius
        if selected.any():
            with np.errstate(all="ignore"):
                output[..., target_day - 1] = np.nanquantile(
                    values[..., selected], quantile, axis=-1, method="linear"
                ).astype(np.float32)
    return output


def calendar_window_quantile(
    daily_temperature: xr.DataArray,
    valid_date: xr.DataArray,
    *,
    window_days: int,
    percentile: float,
) -> xr.DataArray:
    """Lazily calculate a global-compatible q95 field for one forecast lead."""
    if daily_temperature.dims != ("time", "latitude", "longitude"):
        daily_temperature = daily_temperature.transpose("time", "latitude", "longitude")
    if valid_date.dims != ("time",):
        valid_date = valid_date.transpose("time")
    if not np.array_equal(daily_temperature["time"].values, valid_date["time"].values):
        raise ValueError("Daily temperature and valid_date initialization coordinates differ")
    day_of_year = xr.DataArray(
        valid_date.dt.dayofyear.fillna(1).astype(np.int16).values,
        dims=("time",),
        coords={"time": daily_temperature["time"]},
    )
    quantiles = xr.apply_ufunc(
        _calendar_window_quantile_numpy,
        daily_temperature,
        day_of_year,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["dayofyear"]],
        kwargs={"window_days": int(window_days), "quantile": float(percentile) / 100.0},
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={
            "output_sizes": {"dayofyear": DAY_OF_YEAR_COUNT},
            "allow_rechunk": False,
        },
    )
    return quantiles.assign_coords(
        dayofyear=np.arange(1, DAY_OF_YEAR_COUNT + 1, dtype=np.int16)
    ).transpose(
        "dayofyear", "latitude", "longitude"
    )


def compute_q95_band(
    manifest: dict[str, Any],
    *,
    model_name: str,
    band_index: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compute one non-overlapping longitude region of one final global Zarr.

    Source stores are opened read-only.  Daily means are persisted only in the
    worker's memory while its 13 lead q95 reductions are written directly into
    the final global product.
    """
    model = _manifest_model(manifest, model_name)
    model_directory = Path(model["product_directory"])
    store = Path(model["product_store"])
    manifest_sha256 = str(manifest["manifest_sha256"])
    bands = {int(item["index"]): item for item in model["longitude_bands"]}
    if band_index not in bands:
        raise KeyError(f"{model_name} has no longitude band {band_index}")
    band = bands[band_index]
    marker = q95_band_marker(model_directory, band_index)
    if not overwrite and _valid_band_marker(
        marker, manifest_sha256=manifest_sha256, band_index=band_index
    ):
        return {"model": model_name, "band_index": band_index, "status": "already_complete"}
    if not store.is_dir():
        raise FileNotFoundError(f"Global q95 product was not initialized: {store}")
    group = _open_output_group(store, mode="r")
    if group.attrs.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Global q95 product does not match this workflow manifest")
    if overwrite:
        marker.unlink(missing_ok=True)

    forecast_days = tuple(int(day) for day in manifest["forecast_days"])
    raw: xr.Dataset | None = None
    daily: xr.DataArray | None = None
    try:
        print(
            f"Opening {model_name} raw source read-only for longitude band {band_index} "
            f"[{band['longitude_start']}, {band['longitude_stop']})…",
            flush=True,
        )
        raw, temperature = _open_raw_ensemble_mean(model)
        expected_longitude = np.asarray(model["longitude"])[
            int(band["longitude_start_index"]) : int(band["longitude_stop_index"])
        ]
        actual_longitude = np.asarray(temperature["longitude"].values)[
            int(band["longitude_start_index"]) : int(band["longitude_stop_index"])
        ]
        if not np.array_equal(actual_longitude, expected_longitude):
            raise ValueError("Raw longitude coordinates differ from the global product schema")
        daily = _local_solar_daily_mean_band(
            temperature, band=band, forecast_days=forecast_days
        ).chunk(
            {
                "time": -1,
                "forecast_day": 1,
                "latitude": MODEL_Q95_OUTPUT_CHUNKS["latitude"],
                "longitude": MODEL_Q95_OUTPUT_CHUNKS["longitude"],
            }
        )
        sample_counts = _window_sample_counts(
            daily["valid_date"],
            forecast_days=forecast_days,
            longitude_count=daily.sizes["longitude"],
            window_days=int(manifest["window_days"]),
        )
        print(
            f"Persisting in-memory daily ensemble means for {model_name} band {band_index}; "
            "no daily intermediates are written to disk…",
            flush=True,
        )
        with ProgressBar():
            daily = daily.persist()

        longitude_region = slice(
            int(band["longitude_start_index"]), int(band["longitude_stop_index"])
        )
        for lead_index, forecast_day in enumerate(forecast_days):
            print(
                f"Computing {model_name} band {band_index}, forecast day {forecast_day} "
                f"({lead_index + 1}/{len(forecast_days)})…",
                flush=True,
            )
            lead_daily = daily.sel(forecast_day=forecast_day, drop=True)
            lead_dates = (
                daily["valid_date"]
                .sel(forecast_day=forecast_day)
                .isel(longitude=0, drop=True)
            )
            q95 = calendar_window_quantile(
                lead_daily,
                lead_dates,
                window_days=int(manifest["window_days"]),
                percentile=float(manifest["percentile"]),
            ).expand_dims(forecast_day=[forecast_day])
            # Deliberately omit coordinate variables: this task owns only its
            # latitude/all-leads, longitude-region data chunks. It must never
            # contend with another task over global coordinate metadata.
            payload = xr.Dataset(
                {
                    MODEL_Q95_VARIABLE: (
                        ("forecast_day", "dayofyear", "latitude", "longitude"),
                        q95.data,
                    ),
                    SAMPLE_COUNT_VARIABLE: (
                        ("forecast_day", "dayofyear", "longitude"),
                        sample_counts[lead_index : lead_index + 1],
                    ),
                }
            )
            with ProgressBar():
                payload.to_zarr(
                    store,
                    mode="r+",
                    region={
                        "forecast_day": slice(lead_index, lead_index + 1),
                        "longitude": longitude_region,
                    },
                    consolidated=False,
                )

        write_json_atomic(
            {
                "status": "complete",
                "created_at": now_utc(),
                "manifest_sha256": manifest_sha256,
                "model": model_name,
                "band_index": band_index,
                "longitude_start_index": int(band["longitude_start_index"]),
                "longitude_stop_index": int(band["longitude_stop_index"]),
                "source_store_count": int(model["source_store_count"]),
                "daily_intermediate_storage": "none",
            },
            marker,
        )
        return {"model": model_name, "band_index": band_index, "status": "complete"}
    finally:
        # Persisted Dask blocks are process-local. Release references before
        # the Slurm task exits so a failed/retried task never leaves a cache on
        # the shared filesystem.
        del daily
        gc.collect()
        if raw is not None:
            raw.close()


def finalize_q95_workflow(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate all band markers, consolidate each global Zarr, then commit it."""
    manifest_sha256 = str(manifest["manifest_sha256"])
    results: list[dict[str, Any]] = []
    for model in manifest["models"]:
        model_directory = Path(model["product_directory"])
        store = Path(model["product_store"])
        expected_bands = [int(item["index"]) for item in model["longitude_bands"]]
        incomplete = [
            band_index
            for band_index in expected_bands
            if not _valid_band_marker(
                q95_band_marker(model_directory, band_index),
                manifest_sha256=manifest_sha256,
                band_index=band_index,
            )
        ]
        if incomplete:
            raise RuntimeError(
                f"{model['model']}: cannot finalize; longitude bands are incomplete: {incomplete}"
            )
        print(f"Consolidating final global q95 metadata for {model['model']}…", flush=True)
        zarr.consolidate_metadata(str(store))
        dataset = xr.open_zarr(store, consolidated=True, chunks={})
        try:
            expected_shape = (
                len(manifest["forecast_days"]),
                DAY_OF_YEAR_COUNT,
                int(model["latitude_count"]),
                int(model["longitude_count"]),
            )
            if dataset[MODEL_Q95_VARIABLE].shape != expected_shape:
                raise ValueError(
                    f"{model['model']}: q95 shape "
                    f"{dataset[MODEL_Q95_VARIABLE].shape} != {expected_shape}"
                )
            if dataset[SAMPLE_COUNT_VARIABLE].shape != (
                expected_shape[0], expected_shape[1], expected_shape[3]
            ):
                raise ValueError(f"{model['model']}: sample-count shape is invalid")
        finally:
            dataset.close()
        write_json_atomic(
            {
                "status": "complete",
                "created_at": now_utc(),
                "manifest_sha256": manifest_sha256,
                "model": model["model"],
                "product_store": store.name,
                "variables": [MODEL_Q95_VARIABLE, SAMPLE_COUNT_VARIABLE],
                "dimensions": {
                    "forecast_day": len(manifest["forecast_days"]),
                    "dayofyear": DAY_OF_YEAR_COUNT,
                    "latitude": int(model["latitude_count"]),
                    "longitude": int(model["longitude_count"]),
                },
                "daily_intermediate_storage": "none",
            },
            model_directory / "completion.json",
        )
        results.append({"model": model["model"], "status": "complete", "store": str(store)})
    return results
