"""Configuration loading and deterministic partition planning."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_MONTHS = (6, 7, 8, 9)


def _expand(value: Any) -> Any:
    """Expand environment variables in recursively nested configuration values."""
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass(frozen=True)
class Partition:
    """One restartable verification partition."""

    year: int
    month: int

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


@dataclass(frozen=True)
class VerificationConfig:
    """Resolved YAML configuration with small, typed convenience accessors."""

    path: Path
    data: dict[str, Any]
    config_hash: str
    case_cache_hash: str
    compatible_case_cache_hashes: frozenset[str]

    @property
    def model_name(self) -> str:
        return str(self.data["model"]["name"])

    @property
    def model_display_name(self) -> str:
        return str(self.data["model"].get("display_name", self.model_name))

    @property
    def result_root(self) -> Path:
        return Path(self.data["paths"]["verification_results_root"])

    @property
    def model_result_dir(self) -> Path:
        return self.result_root / self.model_name

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["selection"]["years"])

    @property
    def months(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["selection"].get("months", DEFAULT_MONTHS))

    @property
    def forecast_days(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["selection"]["forecast_days"])

    @property
    def map_forecast_days(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.data["selection"].get("map_forecast_days", ()))

    @property
    def probability_bins(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.data["metrics"]["probability_bins"])

    @property
    def interval_levels(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.data["metrics"]["interval_levels"])

    @property
    def probability_decision_thresholds(self) -> tuple[float, ...]:
        values = self.data["metrics"].get("probability_decision_thresholds", (0.5,))
        return tuple(float(value) for value in values)

    @property
    def table_format(self) -> str:
        return str(self.data.get("output", {}).get("table_format", "auto"))

    @property
    def region_file(self) -> Path:
        configured = Path(self.data["regions"]["file"])
        return configured if configured.is_absolute() else self.path.parent / configured

    def partitions(self) -> tuple[Partition, ...]:
        explicit = self.data["selection"].get("partitions")
        if explicit is not None:
            parsed = {
                Partition(int(item["year"]), int(item["month"]))
                for item in explicit
            }
            if not parsed:
                raise ValueError("selection.partitions cannot be empty")
            invalid = [item for item in parsed if not 1 <= item.month <= 12]
            if invalid:
                raise ValueError(f"selection.partitions contains invalid months: {invalid}")
            return tuple(sorted(parsed, key=lambda item: (item.year, item.month)))
        return tuple(Partition(year, month) for year in self.years for month in self.months)

    def assert_partition_selected(self, year: int, month: int) -> None:
        if Partition(year, month) not in self.partitions():
            raise ValueError(
                f"{year:04d}-{month:02d} is not in configured partitions "
                f"({[item.label for item in self.partitions()]})"
            )


def _case_cache_signature(data: Mapping[str, Any], forecast_days: list[int]) -> dict[str, Any]:
    """Return the inputs that can change a canonical cache lead's values."""
    return {
        "case_cache_schema": 1,
        "model": data["model"],
        # The output root, region definitions, probability bins, and map
        # selection do not change a canonical forecast/observation case.  By
        # excluding them, users can make new regional/decision-threshold
        # products from one durable case cache.
        "paths": {
            key: value
            for key, value in data["paths"].items()
            if key != "verification_results_root"
        },
        "variables": data.get("variables", {}),
        "observations": data.get("observations", {}),
        "events": data.get("events", {}),
        "forecast_days": forecast_days,
        "interval_levels": list(data["metrics"]["interval_levels"]),
    }


def _hash_case_cache_signature(signature: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def load_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> VerificationConfig:
    """Read, resolve, and minimally validate a verification YAML file.

    ``HEAT_VERIFICATION_RESULTS_ROOT`` is intentionally supported as an
    environment override because it is often different on login and compute
    nodes.  Other paths can use ordinary ``${VARIABLE}`` syntax in YAML.
    """
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Verification configuration must be a YAML mapping")

    data = _expand(loaded)
    if overrides:
        data = _deep_update(data, _expand(dict(overrides)))
    environment_result_root = os.environ.get("HEAT_VERIFICATION_RESULTS_ROOT")
    if environment_result_root:
        data.setdefault("paths", {})["verification_results_root"] = environment_result_root
    environment_paths = {
        "HEAT_AIFS_RAW_ROOT": "raw_aifs_root",
        "HEAT_AIFS_COMPACT_MONTHLY_STORE_PATTERN": "compact_monthly_store_pattern",
        "HEAT_ERA5_DAILY_TEMPERATURE_STORE": "era5_daily_temperature_store",
        "HEAT_ERA5_HAZARD_STORE": "era5_hazard_store",
        "HEAT_INTERVAL_QUANTILE_FILE_PATTERN": "interval_quantile_file_pattern",
    }
    for environment_name, config_name in environment_paths.items():
        value = os.environ.get(environment_name)
        if value:
            data.setdefault("paths", {})[config_name] = value

    _validate_config(data)
    region_config = Path(data["regions"]["file"])
    region_path = region_config if region_config.is_absolute() else config_path.parent / region_config
    if not region_path.is_file():
        raise FileNotFoundError(f"Configured region file is missing: {region_path}")
    # Region definitions affect every score. Include their content in the
    # compatibility hash so a new region cannot silently resume an old partial.
    canonical = json.dumps(
        {
            "verification_metric_schema": 2,
            "config": data,
            "region_file_sha256": hashlib.sha256(region_path.read_bytes()).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    configured_days = [int(value) for value in data["selection"]["forecast_days"]]
    case_cache_hash = _hash_case_cache_signature(_case_cache_signature(data, configured_days))
    compatible_hashes = {case_cache_hash}
    # Before per-model lead discovery, standard raw reforecast configs always
    # used 0--14. A shorter discovered range is scientifically identical for
    # a successfully committed source lead when the raw product itself ends
    # before day 14. Accept that exact historical signature so a failed final
    # lead can be resumed without discarding the good preceding lead stores.
    historical_days = list(range(15))
    is_zero_based_prefix = configured_days == list(range(len(configured_days)))
    if (
        data["model"].get("adapter") == "standard_reforecast_raw"
        and is_zero_based_prefix
        and len(configured_days) < len(historical_days)
    ):
        compatible_hashes.add(
            _hash_case_cache_signature(_case_cache_signature(data, historical_days))
        )
    return VerificationConfig(
        path=config_path,
        data=data,
        config_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        case_cache_hash=case_cache_hash,
        compatible_case_cache_hashes=frozenset(compatible_hashes),
    )


def _validate_config(data: Mapping[str, Any]) -> None:
    required = {
        "model": ("name", "adapter"),
        "paths": ("verification_results_root",),
        "selection": ("years", "forecast_days"),
        "metrics": ("probability_bins", "interval_levels"),
        "regions": ("file",),
    }
    for section, names in required.items():
        if section not in data or not isinstance(data[section], Mapping):
            raise ValueError(f"Configuration is missing mapping {section!r}")
        missing = [name for name in names if name not in data[section]]
        if missing:
            raise ValueError(f"Configuration section {section!r} is missing {missing}")

    months = data["selection"].get("months", DEFAULT_MONTHS)
    if not months or any(int(month) < 1 or int(month) > 12 for month in months):
        raise ValueError("selection.months must contain calendar months 1 through 12")
    if not data["selection"]["years"]:
        raise ValueError("selection.years must not be empty")
    if not data["selection"]["forecast_days"]:
        raise ValueError("selection.forecast_days must not be empty")
    forecast_days = [int(value) for value in data["selection"]["forecast_days"]]
    if any(day < 0 or day >= 15 for day in forecast_days):
        raise ValueError(
            "selection.forecast_days must be within 0 through 14 "
            "(the supported 15-day forecast window)"
        )

    bins = [float(value) for value in data["metrics"]["probability_bins"]]
    if len(bins) < 2 or bins[0] != 0.0 or bins[-1] != 1.0 or any(
        right <= left for left, right in zip(bins, bins[1:])
    ):
        raise ValueError("metrics.probability_bins must be strictly increasing from 0 to 1")
    levels = [float(value) for value in data["metrics"]["interval_levels"]]
    if any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("metrics.interval_levels must be in the open interval (0, 1)")
    decision_thresholds = [float(value) for value in data["metrics"].get("probability_decision_thresholds", [0.5])]
    if not decision_thresholds or any(value < 0.0 or value > 1.0 for value in decision_thresholds):
        raise ValueError("metrics.probability_decision_thresholds must be within [0, 1]")
    if len(set(decision_thresholds)) != len(decision_thresholds):
        raise ValueError("metrics.probability_decision_thresholds must be unique")
