"""Metadata-only inventory and configuration generation for raw reforecasts."""

from __future__ import annotations

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
    """Inventory directory names and initialization-store names only."""
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
        stores = sorted(path for path in directory.glob("init_*.zarr") if path.is_dir())
        grouped: dict[tuple[int, int], int] = defaultdict(int)
        unparsed: list[str] = []
        for store in stores:
            partition = store_year_month(store)
            if partition is None:
                unparsed.append(store.name)
            elif partition in wanted:
                grouped[partition] += 1
        found.append(
            ReforecastModelInventory(
                name=model_name,
                directory=directory,
                partitions=tuple(sorted(grouped)),
                store_count=sum(grouped.values()),
                unparsed_store_names=tuple(unparsed),
            )
        )
    return found


def raw_reforecast_config(
    inventory: ReforecastModelInventory,
    *,
    result_root: Path,
    region_file: Path,
) -> dict[str, object]:
    """Return a config for a standard 2t/member/lead reforecast source."""
    partitions = [{"year": year, "month": month} for year, month in inventory.partitions]
    years = sorted({year for year, _ in inventory.partitions})
    months = sorted({month for _, month in inventory.partitions})
    return {
        "model": {
            "name": inventory.name,
            "display_name": inventory.name.replace("_", " ").title(),
            "adapter": "standard_reforecast_raw",
            "ensemble": True,
            "member_temperature_available": True,
        },
        "paths": {
            "raw_reforecast_root": str(inventory.directory),
            "threshold_store": "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/thresholds/t2m_daily_mean_percentiles_1991_2020.zarr",
            "era5_daily_temperature_store": "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/daily/t2m_daily_mean.zarr",
            "era5_hazard_store": "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/hazards/t2m_daily_mean_q95_hazards.zarr",
            "verification_results_root": str(result_root),
        },
        "variables": {
            "raw_source_temperature": "2t",
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
                "number": 26,
                "prediction_timedelta": 24,
                "latitude": 180,
                "longitude": 180,
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
