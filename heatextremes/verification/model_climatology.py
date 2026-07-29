"""Metadata preflight helpers for lead-dependent model temperature climatologies.

The preflight deliberately reads only Zarr metadata and coordinate arrays.  It
uses the same local-solar day construction as the standard reforecast adapter
to estimate the sample support available for a circular calendar-day window.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

from .models.standard_reforecast import _processing_helpers
from .reforecast_inventory import ReforecastModelInventory, store_year_month


DAY_OF_YEAR_COUNT = 366


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
        "ensemble": inventory.ensemble,
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
