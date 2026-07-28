"""Restartable, casewise Zarr verification-cache products.

The cache is deliberately source-independent.  A raw or compact model adapter
constructs one canonical, local-solar-day forecast lead; this module commits
only the fields required to calculate verification metrics later.  It never
stores the source model's native time-resolution fields or ensemble members.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from .config import VerificationConfig
from .events import CANONICAL_EVENTS
from .io import assert_safe_result_path, git_commit, now_utc, remove_result_path, write_json_atomic


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CaseCacheResult:
    """Outcome of one case-cache month build."""

    partition_directory: Path
    skipped: bool
    completed_forecast_days: tuple[int, ...]


def case_cache_partition_directory(config: VerificationConfig, year: int, month: int) -> Path:
    """Return the configured, model-scoped case-cache partition directory."""
    return config.model_result_dir / "case_cache" / f"{year:04d}-{month:02d}"


def case_cache_lead_path(
    config: VerificationConfig,
    year: int,
    month: int,
    forecast_day: int,
) -> Path:
    """Return the independent, atomically written Zarr product for one lead."""
    return case_cache_partition_directory(config, year, month) / f"forecast_day_{forecast_day:03d}.zarr"


def _cache_metadata(path: Path) -> dict[str, Any] | None:
    """Read only root Zarr metadata; never open array chunks for validation."""
    metadata_path = path / ".zattrs"
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def case_cache_store_is_valid(
    path: Path,
    config: VerificationConfig,
    *,
    year: int,
    month: int,
    forecast_day: int,
) -> bool:
    """Check that one committed Zarr cache store matches this scientific input."""
    metadata = _cache_metadata(path)
    return bool(
        path.is_dir()
        and (path / ".zmetadata").is_file()
        and metadata is not None
        and metadata.get("verification_case_cache_schema") == CACHE_SCHEMA_VERSION
        and metadata.get("case_cache_hash") == config.case_cache_hash
        and metadata.get("model") == config.model_name
        and metadata.get("year") == year
        and metadata.get("month") == month
        and metadata.get("forecast_day") == forecast_day
    )


def case_cache_completion_is_valid(
    directory: Path,
    config: VerificationConfig,
    *,
    year: int,
    month: int,
) -> bool:
    """Validate a completed cache partition without opening data arrays."""
    marker = directory / "completion.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = payload.get("expected_stores")
    if (
        payload.get("status") != "complete"
        or payload.get("case_cache_hash") != config.case_cache_hash
        or payload.get("model") != config.model_name
        or payload.get("year") != year
        or payload.get("month") != month
        or not isinstance(expected, list)
    ):
        return False
    if sorted(expected) != [f"forecast_day_{day:03d}.zarr" for day in sorted(config.forecast_days)]:
        return False
    return all(
        case_cache_store_is_valid(
            directory / f"forecast_day_{forecast_day:03d}.zarr",
            config,
            year=year,
            month=month,
            forecast_day=forecast_day,
        )
        for forecast_day in config.forecast_days
    )


def _cache_chunks(config: VerificationConfig, dataset: xr.Dataset) -> dict[str, int]:
    configured = config.data.get("case_cache", {}).get(
        "chunks", config.data.get("chunking", {}).get("cache", {})
    )
    if not isinstance(configured, Mapping):
        raise ValueError("case_cache.chunks must be a mapping of dimension names to positive sizes")
    chunks = {str(name): int(size) for name, size in configured.items() if name in dataset.dims}
    if any(size < 1 for size in chunks.values()):
        raise ValueError("case_cache chunk sizes must be positive")
    return chunks


def _canonical_cache_dataset(lead, config: VerificationConfig, *, year: int, month: int) -> xr.Dataset:
    """Serialize the canonical lead fields needed by future metric jobs only."""
    probabilities = xr.concat(
        [lead.event_probabilities[event].astype(np.float32) for event in CANONICAL_EVENTS],
        dim=xr.IndexVariable("event", list(CANONICAL_EVENTS)),
    ).rename("forecast_probability")
    observed = xr.concat(
        [lead.observed_events[event].astype(np.float32) for event in CANONICAL_EVENTS],
        dim=xr.IndexVariable("event", list(CANONICAL_EVENTS)),
    )
    valid_event = probabilities.notnull() & observed.notnull()
    valid_temperature = lead.ensemble_mean_temperature.notnull() & lead.observation_temperature.notnull()
    dataset = xr.Dataset(
        {
            "forecast_temperature": lead.ensemble_mean_temperature.where(valid_temperature).astype(np.float32),
            "observation_temperature": lead.observation_temperature.where(valid_temperature).astype(np.float32),
            "temperature_case_valid": valid_temperature.astype(bool),
            "forecast_probability": probabilities.where(valid_event).astype(np.float32),
            "observed_event": observed.where(valid_event, 0.0).astype(np.uint8),
            "event_case_valid": valid_event.astype(bool),
        }
    )
    if lead.interval_quantiles is not None:
        dataset["forecast_temperature_quantile"] = lead.interval_quantiles.astype(np.float32)
    if lead.valid_date is not None:
        # Valid date can vary by longitude for a local-solar-day product, so
        # preserve its compact native coordinate dimensions rather than
        # broadcasting a datetime value over latitude.
        dataset = dataset.assign_coords(valid_date=lead.valid_date)
    dataset = dataset.assign_coords(forecast_day=np.int16(lead.forecast_day))
    dataset.attrs = {
        "verification_case_cache_schema": CACHE_SCHEMA_VERSION,
        "case_cache_hash": config.case_cache_hash,
        "model": config.model_name,
        "display_name": config.model_display_name,
        "year": int(year),
        "month": int(month),
        "forecast_day": int(lead.forecast_day),
        "ensemble": bool(config.data["model"].get("ensemble", False)),
        "interval_quantiles_available": lead.interval_quantiles is not None,
        "temperature_units": str(config.data.get("observations", {}).get("temperature_units", "unknown")),
        "events": list(CANONICAL_EVENTS),
        "description": (
            "Canonical local-solar-day forecast and aligned ERA5 verification cases. "
            "Derived metric tables may be recomputed from this cache without opening raw model data."
        ),
    }
    chunks = _cache_chunks(config, dataset)
    return dataset.chunk(chunks) if chunks else dataset


def _write_zarr_atomic(dataset: xr.Dataset, path: Path, result_root: Path) -> None:
    """Commit a consolidated Zarr v2 directory by atomic same-filesystem rename."""
    target, _ = assert_safe_result_path(path, result_root)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite committed case-cache store: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        # The project pins zarr<3.  Stating the format explicitly prevents an
        # environment upgrade from silently creating an incompatible cache.
        format_keyword = (
            "zarr_format"
            if "zarr_format" in inspect.signature(xr.Dataset.to_zarr).parameters
            else "zarr_version"
        )
        dataset.to_zarr(temporary, mode="w", consolidated=True, **{format_keyword: 2})
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def dry_run_case_cache_partition(
    config: VerificationConfig,
    year: int,
    month: int,
    *,
    forecast_days: Sequence[int] | None = None,
) -> dict[str, object]:
    """Resolve case-cache inputs and products without opening array stores."""
    config.assert_partition_selected(year, month)
    from .models import get_model_adapter

    adapter = get_model_adapter(config)
    days = tuple(int(day) for day in (forecast_days or config.forecast_days))
    invalid = sorted(set(days) - set(config.forecast_days))
    if invalid:
        raise ValueError(f"Requested forecast days are not configured: {invalid}")
    return {
        "model": config.model_name,
        "partition": f"{year:04d}-{month:02d}",
        "inputs": {name: str(path) for name, path in adapter.dry_run_paths(year, month).items()},
        "case_cache_directory": str(case_cache_partition_directory(config, year, month)),
        "case_cache_stores": [str(case_cache_lead_path(config, year, month, day)) for day in days],
        "zarr_format": 2,
    }


def compute_case_cache_partition(
    config: VerificationConfig,
    year: int,
    month: int,
    *,
    forecast_days: Sequence[int] | None = None,
    overwrite: bool = False,
    resume: bool = True,
    repository_root: Path | None = None,
) -> CaseCacheResult:
    """Build per-lead canonical Zarr cases for one month with restart safety."""
    config.assert_partition_selected(year, month)
    partition = f"{year:04d}-{month:02d}"
    directory = case_cache_partition_directory(config, year, month)
    assert_safe_result_path(directory, config.result_root)
    days = tuple(int(day) for day in (forecast_days or config.forecast_days))
    invalid = sorted(set(days) - set(config.forecast_days))
    if invalid:
        raise ValueError(f"Requested forecast days are not configured: {invalid}")
    if case_cache_completion_is_valid(directory, config, year=year, month=month) and not overwrite:
        return CaseCacheResult(directory, True, tuple(config.forecast_days))
    if directory.exists() and overwrite:
        print(f"Removing configured case-cache partition before overwrite: {directory}", flush=True)
        remove_result_path(directory, config.result_root)
    elif directory.exists() and not resume:
        raise FileExistsError(f"Case-cache partition exists: {directory}; use --resume or --overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("case_cache_hash") != config.case_cache_hash:
            raise RuntimeError(
                "Existing case-cache partition has a different scientific cache hash; use --overwrite"
            )
    else:
        write_json_atomic(
            {
                "status": "in_progress",
                "model": config.model_name,
                "year": year,
                "month": month,
                "case_cache_hash": config.case_cache_hash,
                "created_at": now_utc(),
            },
            progress_path,
        )

    from .models import get_model_adapter

    adapter = get_model_adapter(config)
    opened = adapter.open_partition(year, month)
    try:
        for forecast_day in days:
            store = case_cache_lead_path(config, year, month, forecast_day)
            if case_cache_store_is_valid(
                store, config, year=year, month=month, forecast_day=forecast_day
            ):
                print(f"{partition}: case-cache forecast day {forecast_day} already committed", flush=True)
                continue
            if store.exists():
                # A final path without matching metadata is necessarily an
                # interrupted/obsolete output.  It is still confined to this
                # exact configured cache partition.
                print(f"{partition}: removing invalid case-cache store: {store}", flush=True)
                remove_result_path(store, config.result_root)
            print(f"{partition}: writing bounded case-cache forecast day {forecast_day}", flush=True)
            with ProgressBar():
                lead = adapter.lead(opened, forecast_day)
                dataset = _canonical_cache_dataset(lead, config, year=year, month=month)
                _write_zarr_atomic(dataset, store, config.result_root)
            del lead, dataset
            gc.collect()
    finally:
        adapter.close_partition(opened)

    full_partition = set(days) == set(config.forecast_days)
    if full_partition:
        expected = [f"forecast_day_{day:03d}.zarr" for day in config.forecast_days]
        write_json_atomic(
            {
                "status": "complete",
                "model": config.model_name,
                "year": year,
                "month": month,
                "partition": partition,
                "case_cache_hash": config.case_cache_hash,
                "git_commit": git_commit(repository_root or Path.cwd()),
                "creation_timestamp": now_utc(),
                "forecast_days": list(config.forecast_days),
                "expected_stores": expected,
                "zarr_format": 2,
            },
            directory / "completion.json",
        )
    else:
        write_json_atomic(
            {
                "status": "in_progress",
                "model": config.model_name,
                "year": year,
                "month": month,
                "case_cache_hash": config.case_cache_hash,
                "updated_at": now_utc(),
                "completed_request": {"forecast_days": list(days)},
                "message": "Subset request completed; full configured case cache remains resumable.",
            },
            progress_path,
        )
    return CaseCacheResult(directory, False, days)
