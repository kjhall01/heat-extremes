#!/usr/bin/env python3
"""Report completeness of legacy monthly and modern case-cache intermediates.

The audit is metadata-only: it reads directory entries, Zarr metadata files,
and completion markers but never opens temperature data chunks.  A modern
forecast-day store is counted as complete only when its partition's
``completion.json`` commits the requested lead range and the store has a
``.zmetadata`` file.  This lets a completed 0--14 partition satisfy an audit
limited to days 0--12, while still flagging partial or uncommitted writes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import xarray as xr


DEFAULT_RESULTS_ROOT = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results"
)
DEFAULT_MONTHLY_ROOT = Path("/net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/monthly")
LEGACY_AIFS_MODEL = "aifs_ens_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--monthly-root", type=Path, default=DEFAULT_MONTHLY_ROOT)
    parser.add_argument("--models", nargs="+", help="Models to inspect; defaults to the manifest plus legacy AIFS.")
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument(
        "--max-forecast-day",
        type=int,
        choices=range(15),
        default=12,
        help="Inclusive zero-based lead cap (default: 12, i.e. days 0 through 12).",
    )
    parser.add_argument("--csv", type=Path, help="Optional path for the summary table as CSV.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-model missing-slice detail.")
    return parser.parse_args()


def read_manifest(results_root: Path) -> dict[str, Any]:
    path = results_root / "inventory" / "reforecast_inventory.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise ValueError(f"Inventory manifest is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Inventory manifest must be a JSON object: {path}")
    return payload


def expected_partitions(
    manifest: dict[str, Any], model: str, years: Iterable[int], months: Iterable[int]
) -> tuple[str, ...]:
    for item in manifest.get("models", []):
        if not isinstance(item, dict) or item.get("model") != model:
            continue
        selected = item.get("selected_partitions", [])
        partitions = {
            f"{int(partition['year']):04d}-{int(partition['month']):02d}"
            for partition in selected
            if isinstance(partition, dict) and {"year", "month"} <= set(partition)
        }
        if partitions:
            return tuple(sorted(partitions))
    return tuple(f"{year:04d}-{month:02d}" for year in sorted(set(years)) for month in sorted(set(months)))


def completion_commits_required_leads(
    directory: Path, *, model: str, partition: str, required_stores: set[str]
) -> bool:
    """Return whether the completion marker commits at least the requested stores."""
    try:
        marker = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if not isinstance(marker, dict):
        return False
    expected_stores = marker.get("expected_stores")
    return bool(
        marker.get("status") == "complete"
        and marker.get("model") == model
        and marker.get("partition") == partition
        and isinstance(expected_stores, list)
        and required_stores.issubset(set(expected_stores))
    )


def summarize_details(slices: list[tuple[str, int]]) -> str:
    """Render missing or uncommitted lead slices compactly by month."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for partition, forecast_day in slices:
        grouped[partition].append(forecast_day)
    return "; ".join(
        f"{partition}: " + ", ".join(str(day) for day in sorted(days))
        for partition, days in sorted(grouped.items())
    )


def audit_modern_model(
    results_root: Path, model: str, partitions: tuple[str, ...], forecast_days: tuple[int, ...]
) -> tuple[dict[str, Any], list[tuple[str, int]], list[tuple[str, int]]]:
    required_stores = {f"forecast_day_{day:03d}.zarr" for day in forecast_days}
    complete: list[tuple[str, int]] = []
    missing: list[tuple[str, int]] = []
    uncommitted_or_invalid: list[tuple[str, int]] = []

    for partition in partitions:
        directory = results_root / model / "case_cache" / partition
        committed = completion_commits_required_leads(
            directory, model=model, partition=partition, required_stores=required_stores
        )
        for forecast_day in forecast_days:
            store = directory / f"forecast_day_{forecast_day:03d}.zarr"
            if not store.is_dir():
                missing.append((partition, forecast_day))
            elif not (store / ".zmetadata").is_file() or not committed:
                uncommitted_or_invalid.append((partition, forecast_day))
            else:
                complete.append((partition, forecast_day))

    expected = len(partitions) * len(forecast_days)
    return (
        {
            "model": model,
            "storage": "modern case cache",
            "expected_months": len(partitions),
            "expected_leads": f"{forecast_days[0]}–{forecast_days[-1]}",
            "complete_case_cache_slices": len(complete),
            "missing_slices": len(missing),
            "uncommitted_or_invalid_slices": len(uncommitted_or_invalid),
            "cache_coverage_%": round(100 * len(complete) / expected, 1) if expected else float("nan"),
        },
        missing,
        uncommitted_or_invalid,
    )


def audit_legacy_aifs(
    monthly_root: Path, partitions: tuple[str, ...], forecast_days: tuple[int, ...]
) -> tuple[dict[str, Any], list[tuple[str, int]], list[tuple[str, int]]]:
    complete: list[tuple[str, int]] = []
    missing: list[tuple[str, int]] = []
    invalid: list[tuple[str, int]] = []

    for partition in partitions:
        year, month = partition.split("-")
        store = monthly_root / year / f"aifs_ens_v2_heat_{year}{month}.zarr"
        if not store.is_dir():
            missing.extend((partition, forecast_day) for forecast_day in forecast_days)
            continue
        try:
            # Only the small coordinate array is read; forecast fields remain lazy.
            dataset = xr.open_zarr(store, consolidated=False, chunks={})
            available_days = {int(day) for day in dataset["forecast_day"].values}
        except Exception:
            invalid.extend((partition, forecast_day) for forecast_day in forecast_days)
            continue
        for forecast_day in forecast_days:
            (complete if forecast_day in available_days else missing).append((partition, forecast_day))

    expected = len(partitions) * len(forecast_days)
    return (
        {
            "model": LEGACY_AIFS_MODEL,
            "storage": "legacy monthly intermediates",
            "expected_months": len(partitions),
            "expected_leads": f"{forecast_days[0]}–{forecast_days[-1]}",
            "complete_case_cache_slices": len(complete),
            "missing_slices": len(missing),
            "uncommitted_or_invalid_slices": len(invalid),
            "cache_coverage_%": round(100 * len(complete) / expected, 1) if expected else float("nan"),
        },
        missing,
        invalid,
    )


def main() -> None:
    args = parse_args()
    if not args.years or not args.months:
        raise ValueError("--years and --months must not be empty")
    manifest = read_manifest(args.results_root)
    manifest_models = [
        str(item["model"])
        for item in manifest.get("models", [])
        if isinstance(item, dict) and isinstance(item.get("model"), str)
    ]
    models = args.models or [LEGACY_AIFS_MODEL, *manifest_models]
    models = list(dict.fromkeys(models))
    if not models:
        raise ValueError("No models found; pass --models explicitly or create the inventory manifest first")

    forecast_days = tuple(range(args.max_forecast_day + 1))
    rows: list[dict[str, Any]] = []
    details: dict[str, tuple[list[tuple[str, int]], list[tuple[str, int]]]] = {}
    for model in models:
        partitions = expected_partitions(manifest, model, args.years, args.months)
        if model.casefold() == LEGACY_AIFS_MODEL:
            row, missing, invalid = audit_legacy_aifs(args.monthly_root, partitions, forecast_days)
        else:
            row, missing, invalid = audit_modern_model(args.results_root, model, partitions, forecast_days)
        rows.append(row)
        details[model] = (missing, invalid)

    table = pd.DataFrame(rows).sort_values("model")
    print(table.to_string(index=False))
    if not args.quiet:
        for model, (missing, invalid) in details.items():
            if missing:
                print(f"\n{model} missing: {summarize_details(missing)}")
            if invalid:
                print(f"\n{model} uncommitted or invalid: {summarize_details(invalid)}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"\nSummary CSV: {args.csv}")


if __name__ == "__main__":
    main()
