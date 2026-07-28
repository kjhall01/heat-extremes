"""Metadata-only inventory and configuration generation for raw reforecasts."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_DATE_PATTERNS = (
    r"(?P<year>(?:19|20)\d{2})[-_](?P<month>0[1-9]|1[0-2])[-_]\d{2}",
    r"(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])\d{2}",
    r"(?P<year>(?:19|20)\d{2})[-_](?P<month>0[1-9]|1[0-2])",
    r"(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])",
)


@dataclass(frozen=True)
class ReforecastModelInventory:
    """One standard-format raw-model directory and its selected months."""

    name: str
    directory: Path
    partitions: tuple[tuple[int, int], ...]
    store_count: int
    unparsed_store_names: tuple[str, ...]
    display_name: str
    ensemble: bool
    source_temperature_variable: str
    source_store_glob: str


def store_year_month(path: Path) -> tuple[int, int] | None:
    """Parse a date from an ``init_*.zarr`` name without opening the store."""
    for pattern in _DATE_PATTERNS:
        match = re.search(pattern, path.name)
        if match is not None:
            return int(match.group("year")), int(match.group("month"))
    return None


def model_name_from_directory(directory: Path) -> str:
    """Use the suffix of ``forecasts_*`` as a stable result-directory name."""
    raw = directory.name.removeprefix("forecasts_")
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if not normalized:
        raise ValueError(f"Could not derive a model name from {directory.name!r}")
    return normalized


def inventory_reforecast_root(
    root: Path,
    *,
    years: Iterable[int],
    months: Iterable[int],
) -> list[ReforecastModelInventory]:
    """Discover compatible model directories when no metadata registry is used."""
    if not root.is_dir():
        raise FileNotFoundError(f"Reforecast root does not exist or is not a directory: {root}")
    wanted = {(int(year), int(month)) for year in years for month in months}
    found: list[ReforecastModelInventory] = []
    names: set[str] = set()
    for directory in sorted(path for path in root.glob("forecasts_*") if path.is_dir()):
        model_name = model_name_from_directory(directory)
        if model_name in names:
            raise ValueError(f"Reforecast model-name collision after normalization: {model_name}")
        names.add(model_name)
        found.append(
            _inventory_directory(
                name=model_name,
                display_name=model_name.replace("_", " ").title(),
                directory=directory,
                wanted=wanted,
                ensemble=True,
                source_temperature_variable="2t",
                source_store_glob="init_*.zarr",
            )
        )
    return found


def _inventory_directory(
    *,
    name: str,
    display_name: str,
    directory: Path,
    wanted: set[tuple[int, int]],
    ensemble: bool,
    source_temperature_variable: str,
    source_store_glob: str,
) -> ReforecastModelInventory:
    stores = sorted(path for path in directory.glob(source_store_glob) if path.is_dir())
    grouped: dict[tuple[int, int], int] = defaultdict(int)
    unparsed: list[str] = []
    for store in stores:
        partition = store_year_month(store)
        if partition is None:
            unparsed.append(store.name)
        elif partition in wanted:
            grouped[partition] += 1
    return ReforecastModelInventory(
        name=name,
        directory=directory,
        partitions=tuple(sorted(grouped)),
        store_count=sum(grouped.values()),
        unparsed_store_names=tuple(unparsed),
        display_name=display_name,
        ensemble=ensemble,
        source_temperature_variable=source_temperature_variable,
        source_store_glob=source_store_glob,
    )


def _metadata_paths(row: dict[str, str]) -> list[Path]:
    """Split a multiline metadata Path cell into absolute filesystem paths."""
    return [Path(value.strip()) for value in row.get("Path", "").splitlines() if value.strip().startswith("/")]


def _temperature_variable(row: dict[str, str]) -> str | None:
    variables = {value.strip().lower() for value in row.get("Variables", "").splitlines()}
    if "2t" in variables:
        return "2t"
    if "2m_temperature" in variables:
        return "2m_temperature"
    return None


def _is_ensemble(row: dict[str, str]) -> bool | None:
    raw = row.get("N Members", "").strip()
    if raw.lower() in {"", "n/a", "na", "none", "-"}:
        return False
    try:
        return int(raw) > 1
    except ValueError:
        return None


def inventory_metadata_csv(
    metadata_csv: Path,
    *,
    root: Path,
    years: Iterable[int],
    months: Iterable[int],
    excluded_model_names: Iterable[str] = ("gencast",),
    allowed_external_model_names: Iterable[str] = ("aifs-ens-v2",),
) -> tuple[list[ReforecastModelInventory], list[dict[str, str]]]:
    """Inventory models listed in the CSV registry, not arbitrary directories.

    The registry decides deterministic versus ensemble capability from
    ``N Members`` and permits a configured raw temperature source of either
    ``2t`` or ``2m_temperature``. AIFS-ENS-v2 is allowed outside the generic
    root because it is the established pilot source with ``*.zarr`` rather
    than ``init_*.zarr`` names. The source stores are never opened here.
    """
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Model metadata CSV is missing: {metadata_csv}")
    if not root.is_dir():
        raise FileNotFoundError(f"Reforecast root does not exist or is not a directory: {root}")
    root = root.resolve()
    wanted = {(int(year), int(month)) for year in years for month in months}
    excluded = {name.casefold() for name in excluded_model_names}
    allowed_external = {name.casefold() for name in allowed_external_model_names}
    inventories: list[ReforecastModelInventory] = []
    skipped: list[dict[str, str]] = []
    seen_directories: set[Path] = set()
    seen_names: set[str] = set()
    with metadata_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            display_name = row.get("Model", "").strip()
            if not display_name:
                skipped.append({"model": "<blank>", "reason": "blank Model column"})
                continue
            if display_name.casefold() in excluded:
                skipped.append({"model": display_name, "reason": "explicitly excluded"})
                continue
            candidates = []
            for candidate in _metadata_paths(row):
                try:
                    candidate.resolve(strict=False).relative_to(root)
                except ValueError:
                    if display_name.casefold() not in allowed_external:
                        continue
                    candidates.append(candidate)
                else:
                    candidates.append(candidate)
            if not candidates:
                skipped.append({"model": display_name, "reason": "no metadata path beneath selected reforecast root"})
                continue
            variable = _temperature_variable(row)
            if variable is None:
                skipped.append({"model": display_name, "reason": "no supported surface-temperature variable (2t or 2m_temperature)"})
                continue
            ensemble = _is_ensemble(row)
            if ensemble is None:
                skipped.append({"model": display_name, "reason": "unparseable N Members metadata"})
                continue
            for directory in candidates:
                resolved = directory.resolve(strict=False)
                if resolved in seen_directories:
                    skipped.append({"model": display_name, "reason": f"duplicate metadata directory: {directory}"})
                    continue
                if not directory.is_dir():
                    skipped.append({"model": display_name, "reason": f"metadata directory does not exist: {directory}"})
                    continue
                name = model_name_from_directory(directory)
                if name in seen_names:
                    skipped.append({"model": display_name, "reason": f"normalized model-name collision: {name}"})
                    continue
                seen_directories.add(resolved)
                seen_names.add(name)
                inventories.append(
                    _inventory_directory(
                        name=name,
                        display_name=display_name,
                        directory=directory,
                        wanted=wanted,
                        ensemble=ensemble,
                        source_temperature_variable=variable,
                        source_store_glob=("*.zarr" if display_name.casefold() in allowed_external else "init_*.zarr"),
                    )
                )
    return inventories, skipped


def raw_reforecast_config(
    inventory: ReforecastModelInventory,
    *,
    result_root: Path,
    region_file: Path,
) -> dict[str, object]:
    """Return a config for a standard deterministic or ensemble raw source."""
    partitions = [{"year": year, "month": month} for year, month in inventory.partitions]
    years = sorted({year for year, _ in inventory.partitions})
    months = sorted({month for _, month in inventory.partitions})
    return {
        "model": {
            "name": inventory.name,
            "display_name": inventory.display_name,
            "adapter": "standard_reforecast_raw",
            "ensemble": inventory.ensemble,
            "member_temperature_available": inventory.ensemble,
        },
        "paths": {
            "raw_reforecast_root": str(inventory.directory),
            "threshold_store": "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/thresholds/t2m_daily_mean_percentiles_1991_2020.zarr",
            "era5_daily_temperature_store": "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/daily/t2m_daily_mean.zarr",
            "era5_hazard_store": "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/hazards/t2m_daily_mean_q95_hazards.zarr",
            "verification_results_root": str(result_root),
        },
        "variables": {
            "raw_source_temperature": inventory.source_temperature_variable,
            "threshold_temperature": "t2m_daily_mean_calendar_day_percentile",
        },
        "observations": {
            "daily_temperature_variable": "t2m_daily_mean",
            "event_variables": {
                "hot_day_q95": "hot_day_q95",
                "heatwave_start_q95_2d": "heatwave_start_q95_2d",
                "heatwave_start_q95_3d": "heatwave_start_q95_3d",
            },
            "temperature_units": "K",
        },
        "events": {"percentile": 95.0},
        "selection": {
            "years": years,
            "months": months,
            "partitions": partitions,
            "forecast_days": list(range(15)),
            "map_forecast_days": [0, 5, 10, 13],
        },
        "chunking": {
            "raw": {
                "time": 1,
                "prediction_timedelta": 24,
                "latitude": 180,
                "longitude": 180,
                **({"number": 26} if inventory.ensemble else {}),
            },
            "thresholds": {"latitude": 180, "longitude": 180},
            "observations": {"time": 31, "latitude": 180, "longitude": 180},
        },
        "metrics": {
            "probability_bins": [round(value / 10, 1) for value in range(11)],
            "interval_levels": [0.5, 0.8, 0.9, 0.95],
        },
        "regions": {"file": str(region_file)},
        "output": {"table_format": "auto"},
    }
