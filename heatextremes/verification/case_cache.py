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
from collections.abc import Collection, Mapping, Sequence
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


def _metadata_matches_case_identity(
    metadata: Mapping[str, Any] | None,
    config: VerificationConfig,
    *,
    year: int,
    month: int,
    forecast_day: int,
) -> bool:
    """Validate cache identity without trusting its configuration fingerprint."""
    return bool(
        metadata is not None
        and metadata.get("verification_case_cache_schema") == CACHE_SCHEMA_VERSION
        and metadata.get("model") == config.model_name
        and metadata.get("year") == year
        and metadata.get("month") == month
        and metadata.get("forecast_day") == forecast_day
        and tuple(metadata.get("events", ())) == CANONICAL_EVENTS
        and metadata.get("ensemble") == bool(config.data["model"].get("ensemble", False))
        and metadata.get("temperature_units")
        == str(config.data.get("observations", {}).get("temperature_units", "unknown"))
    )


def _shortened_standard_horizon_can_adopt(config: VerificationConfig) -> bool:
    """Limit broad legacy adoption to the known 0--14 raw-model transition."""
    days = config.forecast_days
    return bool(
        config.data["model"].get("adapter") == "standard_reforecast_raw"
        and days == tuple(range(len(days)))
        and len(days) < 15
    )


def case_cache_store_is_valid(
    path: Path,
    config: VerificationConfig,
    *,
    year: int,
    month: int,
    forecast_day: int,
    accepted_hashes: Collection[str] | None = None,
) -> bool:
    """Check that one committed Zarr cache store matches this scientific input."""
    metadata = _cache_metadata(path)
    hashes = set(config.compatible_case_cache_hashes if accepted_hashes is None else accepted_hashes)
    return bool(
        path.is_dir()
        and (path / ".zmetadata").is_file()
        and _metadata_matches_case_identity(
            metadata,
            config,
            year=year,
            month=month,
            forecast_day=forecast_day,
        )
        and metadata.get("case_cache_hash") in hashes
    )


def _adoptable_legacy_store_hashes(
    directory: Path,
    config: VerificationConfig,
    *,
    year: int,
    month: int,
    forecast_days: Sequence[int],
) -> frozenset[str]:
    """Return older store fingerprints only for a complete safe-shortening set.

    A cache lead is immutable once atomically committed.  This deliberately
    narrow bridge exists for the transition from the original fixed 0--14
    raw-model configuration to a shorter source-verified horizon.  All new
    requested lead stores must already exist and agree on canonical schema,
    model, date, event definitions, ensemble capability, and units.  Any
    missing/incompatible store returns an empty set, preserving the normal
    hash refusal instead of silently mixing scientific inputs.
    """
    if not _shortened_standard_horizon_can_adopt(config):
        return frozenset()
    hashes: set[str] = set()
    for forecast_day in forecast_days:
        store = case_cache_lead_path(config, year, month, forecast_day)
        metadata = _cache_metadata(store)
        if not (
            store.is_dir()
            and (store / ".zmetadata").is_file()
            and _metadata_matches_case_identity(
                metadata,
                config,
                year=year,
                month=month,
                forecast_day=forecast_day,
            )
        ):
            return frozenset()
        cache_hash = metadata.get("case_cache_hash") if metadata is not None else None
        if not isinstance(cache_hash, str) or not cache_hash:
            return frozenset()
        hashes.add(cache_hash)
    return frozenset(hashes)


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
    adopted_hashes = payload.get("adopted_legacy_store_hashes", [])
    if (
        payload.get("status") != "complete"
        or payload.get("case_cache_hash") != config.case_cache_hash
        or payload.get("model") != config.model_name
        or payload.get("year") != year
        or payload.get("month") != month
        or not isinstance(expected, list)
        or not isinstance(adopted_hashes, list)
        or any(not isinstance(value, str) for value in adopted_hashes)
    ):
        return False
    if adopted_hashes and not _shortened_standard_horizon_can_adopt(config):
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
            accepted_hashes=set(config.compatible_case_cache_hashes).union(adopted_hashes),
        )
        for forecast_day in config.forecast_days
    )


def _write_completion_marker(
    directory: Path,
    config: VerificationConfig,
    *,
    year: int,
    month: int,
    repository_root: Path | None,
    adopted_legacy_store_hashes: Collection[str] = (),
) -> None:
    """Mark a complete configured cache partition after all stores validate."""
    partition = f"{year:04d}-{month:02d}"
    expected = [f"forecast_day_{day:03d}.zarr" for day in config.forecast_days]
    write_json_atomic(
        {
            "status": "complete",
            "model": config.model_name,
            "year": year,
            "month": month,
            "partition": partition,
            "case_cache_hash": config.case_cache_hash,
            "compatible_store_hashes": sorted(config.compatible_case_cache_hashes),
            "adopted_legacy_store_hashes": sorted(set(adopted_legacy_store_hashes)),
            "git_commit": git_commit(repository_root or Path.cwd()),
            "creation_timestamp": now_utc(),
            "forecast_days": list(config.forecast_days),
            "expected_stores": expected,
            "zarr_format": 2,
        },
        directory / "completion.json",
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


def _without_scalar_forecast_day(values: xr.DataArray) -> xr.DataArray:
    """Remove a source scalar lead label before combining canonical fields.

    Event-onset construction references both the previous and target lead. A
    source adapter may therefore leave a scalar ``forecast_day`` coordinate
    from the previous-day operand even though the returned array has no lead
    dimension. The cache is already one store per target lead, so replace all
    such incidental scalar labels with the explicit target label below.
    """
    if "forecast_day" in values.coords and "forecast_day" not in values.dims:
        return values.reset_coords("forecast_day", drop=True)
    return values


def _without_source_auxiliary_coordinates(values: xr.DataArray) -> xr.DataArray:
    """Keep only dimension coordinates when combining case fields.

    Forecast temperature and each event field may carry a different lazy
    ``valid_date`` auxiliary coordinate after onset construction. The case
    cache writes the authoritative target-lead valid-date coordinate once, so
    retaining the source-specific copies would make xarray reject an otherwise
    compatible Dataset merge.
    """
    removable = [name for name in values.coords if name not in values.dims]
    return values.drop_vars(removable) if removable else values


def _canonical_cache_dataset(lead, config: VerificationConfig, *, year: int, month: int) -> xr.Dataset:
    """Serialize the canonical lead fields needed by future metric jobs only."""
    probabilities = xr.concat(
        [
            _without_source_auxiliary_coordinates(lead.event_probabilities[event]).astype(np.float32)
            for event in CANONICAL_EVENTS
        ],
        dim=xr.IndexVariable("event", list(CANONICAL_EVENTS)),
    ).rename("forecast_probability")
    observed = xr.concat(
        [
            _without_source_auxiliary_coordinates(lead.observed_events[event]).astype(np.float32)
            for event in CANONICAL_EVENTS
        ],
        dim=xr.IndexVariable("event", list(CANONICAL_EVENTS)),
    )
    forecast_temperature = _without_source_auxiliary_coordinates(lead.ensemble_mean_temperature)
    observation_temperature = _without_source_auxiliary_coordinates(lead.observation_temperature)
    valid_event = probabilities.notnull() & observed.notnull()
    valid_temperature = forecast_temperature.notnull() & observation_temperature.notnull()
    dataset = xr.Dataset(
        {
            "forecast_temperature": forecast_temperature.where(valid_temperature).astype(np.float32),
            "observation_temperature": observation_temperature.where(valid_temperature).astype(np.float32),
            "temperature_case_valid": valid_temperature.astype(bool),
            "forecast_probability": probabilities.where(valid_event).astype(np.float32),
            "observed_event": observed.where(valid_event, 0.0).astype(np.uint8),
            "event_case_valid": valid_event.astype(bool),
        }
    )
    if lead.interval_quantiles is not None:
        dataset["forecast_temperature_quantile"] = _without_source_auxiliary_coordinates(
            lead.interval_quantiles
        ).astype(np.float32)
    if lead.valid_date is not None:
        # Valid date can vary by longitude for a local-solar-day product, so
        # preserve its compact native coordinate dimensions rather than
        # broadcasting a datetime value over latitude.
        dataset = dataset.assign_coords(valid_date=_without_scalar_forecast_day(lead.valid_date))
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
    # Source Zarr stores can carry backend-specific encodings (notably the
    # Zarr v3 ``serializer`` encoding).  These are source I/O instructions,
    # not scientific metadata, and passing one through to an explicit Zarr v2
    # output makes zarr reject the write.  Let this cache's own chunks/default
    # v2 codec define every output array instead.
    for variable in dataset.variables.values():
        variable.encoding.clear()
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
    adopted_legacy_hashes: frozenset[str] = frozenset()
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress_hash = progress.get("case_cache_hash")
        if progress_hash not in config.compatible_case_cache_hashes:
            adopted_legacy_hashes = _adoptable_legacy_store_hashes(
                directory,
                config,
                year=year,
                month=month,
                forecast_days=days,
            )
            if not adopted_legacy_hashes:
                raise RuntimeError(
                    "Existing case-cache partition has a different scientific cache hash; use --overwrite"
                )
        if progress_hash != config.case_cache_hash:
            print(
                f"{partition}: adopting committed case-cache stores from a prior lead range",
                flush=True,
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

    # Lead stores are independent and atomically committed. When a prior job
    # failed only because it requested an unavailable final lead, all newly
    # configured leads may already be present. Complete the status marker
    # directly: this avoids reopening raw forecasts or repeating any compute.
    requested_stores_ready = all(
        case_cache_store_is_valid(
            case_cache_lead_path(config, year, month, forecast_day),
            config,
            year=year,
            month=month,
            forecast_day=forecast_day,
            accepted_hashes=set(config.compatible_case_cache_hashes).union(adopted_legacy_hashes),
        )
        for forecast_day in days
    )
    if requested_stores_ready:
        if set(days) == set(config.forecast_days):
            _write_completion_marker(
                directory,
                config,
                year=year,
                month=month,
                repository_root=repository_root,
                adopted_legacy_store_hashes=adopted_legacy_hashes,
            )
        return CaseCacheResult(directory, False, days)

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
        _write_completion_marker(
            directory,
            config,
            year=year,
            month=month,
            repository_root=repository_root,
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
