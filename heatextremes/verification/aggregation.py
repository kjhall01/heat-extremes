"""Exact aggregation of partition-level sufficient statistics."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import Partition, VerificationConfig
from .io import (
    TABLE_STEMS,
    completed_output_names,
    concatenate_tables,
    completion_is_valid,
    find_table,
    now_utc,
    read_table,
    write_json_atomic,
    write_netcdf_atomic,
    write_table_atomic,
    assert_safe_result_path,
)


@dataclass(frozen=True)
class Discovery:
    completed: tuple[Path, ...]
    missing: tuple[str, ...]


def discover_partitions(
    result_dir: Path,
    expected: Iterable[Partition],
) -> Discovery:
    """Discover complete expected partitions and report absent/incomplete ones."""
    completed: list[Path] = []
    missing: list[str] = []
    for partition in expected:
        path = result_dir / "partial" / partition.label
        if completion_is_valid(path):
            completed.append(path)
        else:
            missing.append(partition.label)
    return Discovery(tuple(completed), tuple(missing))


def verify_partition_metadata(partitions: Iterable[Path]) -> dict[str, object]:
    """Ensure all partitions have compatible model/configuration metadata."""
    baseline: dict[str, object] | None = None
    seen: set[tuple[str, int, int]] = set()
    for path in partitions:
        payload = json.loads((path / "completion.json").read_text(encoding="utf-8"))
        key = (str(payload["model"]), int(payload["year"]), int(payload["month"]))
        if key in seen:
            raise ValueError(f"Duplicate completed verification partition: {key}")
        seen.add(key)
        signature = {
            key: payload.get(key)
            for key in (
                "model",
                "configuration_hash",
                "forecast_days",
                "probability_bins",
                "probability_decision_thresholds",
                "interval_levels",
            )
        }
        if baseline is None:
            baseline = signature
        elif signature != baseline:
            raise ValueError(
                f"Incompatible completion metadata in {path}: {signature} != {baseline}"
            )
    return baseline or {}


def _sum_grouped(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    grouped = frame.groupby(keys, dropna=False, as_index=False)[available].sum(min_count=1)
    return grouped


def _with_default_columns(frame: pd.DataFrame, defaults: dict[str, object]) -> pd.DataFrame:
    """Accept early partial-schema tables while preserving current tidy columns."""
    result = frame.copy()
    for column, value in defaults.items():
        if column not in result:
            result[column] = value
    return result


def aggregate_deterministic(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _with_default_columns(frame, {"event": None})
    keys = ["model", "region", "forecast_day", "subset", "event", "metric"]
    result = _sum_grouped(
        frame, keys, ["numerator", "denominator", "weighted_support", "unweighted_support"]
    )
    ratio = result["numerator"] / result["denominator"]
    result["value"] = np.where(result["metric"].eq("rmse"), np.sqrt(ratio), ratio)
    return result


def aggregate_probability(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _with_default_columns(frame, {"subset": "all", "decision_threshold": np.nan})
    keys = [
        "model",
        "region",
        "forecast_day",
        "subset",
        "event",
        "metric",
        "decision_threshold",
    ]
    result = _sum_grouped(
        frame,
        keys,
        [
            "numerator",
            "denominator",
            "weighted_support",
            "unweighted_support",
            "event_weighted_support",
            "non_event_weighted_support",
            "weighted_hits",
            "weighted_misses",
            "weighted_false_alarms",
            "weighted_correct_negatives",
            "unweighted_hits",
            "unweighted_misses",
            "unweighted_false_alarms",
            "unweighted_correct_negatives",
        ],
    )
    result["value"] = result["numerator"] / result["denominator"]
    return result


def aggregate_reliability(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _with_default_columns(frame, {"subset": "all", "metric": "reliability_bin"})
    keys = [
        "model",
        "region",
        "forecast_day",
        "subset",
        "event",
        "metric",
        "bin",
        "bin_lower",
        "bin_upper",
    ]
    result = _sum_grouped(
        frame,
        keys,
        [
            "weighted_count",
            "unweighted_count",
            "weighted_probability_sum",
            "weighted_observation_sum",
        ],
    )
    result["mean_forecast_probability"] = (
        result["weighted_probability_sum"] / result["weighted_count"]
    )
    result["observed_frequency"] = result["weighted_observation_sum"] / result["weighted_count"]
    return result


def aggregate_interval(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _with_default_columns(frame, {"event": None, "metric": "interval_coverage"})
    available = frame[frame["status"].eq("available")].copy()
    unavailable = frame[~frame["status"].eq("available")].copy()
    keys = [
        "model",
        "region",
        "forecast_day",
        "subset",
        "event",
        "metric",
        "nominal_coverage",
        "status",
        "reason",
    ]
    output: list[pd.DataFrame] = []
    if not available.empty:
        result = _sum_grouped(
            available,
            keys,
            [
                "numerator",
                "denominator",
                "unweighted_numerator",
                "unweighted_support",
                "width_numerator",
            ],
        )
        result["empirical_weighted_coverage"] = result["numerator"] / result["denominator"]
        result["empirical_unweighted_coverage"] = (
            result["unweighted_numerator"] / result["unweighted_support"]
        )
        result["mean_interval_width"] = result["width_numerator"] / result["denominator"]
        output.append(result)
    if not unavailable.empty:
        unavailable = unavailable.drop_duplicates(keys)
        output.append(unavailable)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def aggregate_spatial(partitions: Iterable[Path]) -> xr.Dataset | None:
    """Add partial spatial numerators/denominators without reopening raw stores."""
    total: xr.Dataset | None = None
    for path in partitions:
        map_path = path / "maps.nc"
        if not map_path.is_file():
            continue
        with xr.open_dataset(map_path) as opened:
            current = opened.load()
        if total is None:
            total = current
        else:
            total, current = xr.align(total, current, join="exact")
            total = total + current
    if total is None:
        return None
    denominator = total["temperature_denominator"]
    total["temperature_bias"] = total["temperature_bias_numerator"] / denominator
    total["temperature_rmse"] = np.sqrt(total["temperature_squared_error_numerator"] / denominator)
    hot_denominator = total["hot_day_denominator"]
    total["hot_day_brier_score"] = total["hot_day_brier_numerator"] / hot_denominator
    total["mean_hot_day_probability"] = total["hot_day_probability_numerator"] / hot_denominator
    total["observed_hot_day_frequency"] = total["hot_day_observation_numerator"] / hot_denominator
    total["hot_day_probability_frequency_bias"] = (
        total["mean_hot_day_probability"] - total["observed_hot_day_frequency"]
    )
    return total


def aggregate_result_directory(
    config: VerificationConfig,
    *,
    result_dir: Path | None = None,
    years: set[int] | None = None,
    months: set[int] | None = None,
    regions: set[str] | None = None,
    forecast_days: set[int] | None = None,
    allow_missing: bool = False,
) -> tuple[Path, Discovery]:
    """Aggregate one model's partial products and write atomic final products."""
    target = result_dir or config.model_result_dir
    assert_safe_result_path(target, config.result_root)
    expected = tuple(
        partition
        for partition in config.partitions()
        if (years is None or partition.year in years) and (months is None or partition.month in months)
    )
    discovery = discover_partitions(target, expected)
    if discovery.missing and not allow_missing:
        raise RuntimeError(
            "Missing expected partitions; rerun after they complete or pass --allow-missing: "
            + ", ".join(discovery.missing)
        )
    if not discovery.completed:
        raise RuntimeError("No completed partitions were found")
    metadata = verify_partition_metadata(discovery.completed)
    if metadata.get("configuration_hash") != config.config_hash:
        raise ValueError(
            "Completed partition configuration hash does not match the supplied aggregation config; "
            "use the exact compute configuration or recompute with --overwrite"
        )

    output = target / "aggregated"
    assert_safe_result_path(output, config.result_root)
    output.mkdir(parents=True, exist_ok=True)
    filters = {"region": regions, "forecast_day": forecast_days}
    aggregators = {
        "deterministic": aggregate_deterministic,
        "probability": aggregate_probability,
        "probability_reliability": aggregate_reliability,
        "interval_coverage": aggregate_interval,
    }
    output_names = {
        "deterministic": "deterministic_by_lead_region",
        "probability": "probability_by_lead_region",
        "probability_reliability": "probability_reliability_by_lead_region_bin",
        "interval_coverage": "interval_coverage_by_lead_region",
    }
    table_format = config.table_format
    from .io import resolve_table_format, table_path

    resolved_format = resolve_table_format(table_format)
    for stem, aggregate in aggregators.items():
        paths = [find_table(partition, stem) for partition in discovery.completed]
        frame = concatenate_tables([path for path in paths if path is not None])
        for column, wanted in filters.items():
            if wanted is not None and column in frame:
                frame = frame[frame[column].isin(wanted)]
        final = aggregate(frame)
        write_table_atomic(final, table_path(output, output_names[stem], resolved_format))

    spatial = aggregate_spatial(discovery.completed)
    if spatial is not None:
        write_netcdf_atomic(spatial, output / "spatial_metrics.nc")
    write_json_atomic(
        {
            "status": "complete",
            "created_at": now_utc(),
            "partial_partitions": [path.name for path in discovery.completed],
            "missing_partitions": list(discovery.missing),
            "metadata_signature": metadata,
            "filters": {
                "years": sorted(years) if years else None,
                "months": sorted(months) if months else None,
                "regions": sorted(regions) if regions else None,
                "forecast_days": sorted(forecast_days) if forecast_days else None,
            },
        },
        output / "metadata.json",
    )
    return output, discovery
