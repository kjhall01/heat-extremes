#!/usr/bin/env python3
"""Preflight raw reforecast archives for model lead-dependent q95 climatologies.

This command checks historical Zarr metadata and coordinate arrays only.  It
does not read a single 2 m temperature data chunk or write model products.
The JSON and CSV reports state whether each selected model has complete source
coverage, a consistent raw layout, and enough samples for a circular
calendar-day percentile window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.io import now_utc, write_json_atomic, write_table_atomic
from heatextremes.verification.model_climatology import preflight_model
from heatextremes.verification.reforecast_inventory import inventory_metadata_csv


DEFAULT_REFORECAST_ROOT = Path("/net/monsoon/marchakitus/reforecast")
DEFAULT_RESULTS_ROOT = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reforecast-root", type=Path, default=DEFAULT_REFORECAST_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--models", nargs="+", help="Optional normalized model names to inspect.")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2000, 2021)))
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--max-forecast-day", type=int, choices=range(15), default=12)
    parser.add_argument("--window-days", type=int, default=15)
    parser.add_argument(
        "--minimum-q95-samples",
        type=int,
        default=35,
        help="Flag model/lead support below this count; the default is appropriate for weekly 2000–20 data.",
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print the report but do not write JSON or CSV.")
    return parser.parse_args()


def summary_row(report: dict[str, Any]) -> dict[str, Any]:
    support = report["q95_support_by_lead"]
    return {
        "model": report["model"],
        "status": report["status"],
        "source_stores": report["discovered_stores"],
        "checked_stores": report["checked_stores"],
        "present_partitions": report["present_partitions"],
        "requested_partitions": report["requested_partitions"],
        "missing_partitions": len(report["missing_partitions"]),
        "metadata_errors": len(report["metadata_errors"]),
        "initializations": report["initialization_count"],
        "minimum_q95_samples": min((item["minimum"] for item in support), default=0),
        "median_q95_samples": min((item["median"] for item in support), default=0.0),
        "low_support_band_day_cells": sum(item["cells_below_minimum"] for item in support),
    }


def main() -> None:
    args = parse_args()
    if not args.years or not args.months:
        raise ValueError("--years and --months must not be empty")
    if args.window_days < 1 or args.window_days % 2 == 0:
        raise ValueError("--window-days must be a positive odd integer")
    if args.minimum_q95_samples < 1:
        raise ValueError("--minimum-q95-samples must be positive")
    if any(not 1 <= month <= 12 for month in args.months):
        raise ValueError("--months must be between 1 and 12")

    metadata_csv = args.metadata_csv or (args.repository_root / "Rossby Model Storage Locations - Sheet1.csv")
    inventories, skipped_models = inventory_metadata_csv(
        metadata_csv,
        root=args.reforecast_root,
        years=args.years,
        months=args.months,
        max_forecast_day=args.max_forecast_day,
    )
    selected = set(args.models or ())
    if selected:
        inventories = [inventory for inventory in inventories if inventory.name in selected]
        unknown = selected.difference(inventory.name for inventory in inventories)
        if unknown:
            raise ValueError(f"Requested models were not available from the metadata registry: {sorted(unknown)}")
    if not inventories:
        raise ValueError("No compatible raw reforecast models were selected")

    requested_days = tuple(range(args.max_forecast_day + 1))
    reports = []
    for inventory in inventories:
        print(f"Preflighting {inventory.name}…", flush=True)
        reports.append(
            preflight_model(
                inventory,
                years=args.years,
                months=args.months,
                forecast_days=requested_days,
                window_days=args.window_days,
                minimum_samples=args.minimum_q95_samples,
            )
        )

    payload = {
        "status": "model_temperature_climatology_preflight_complete",
        "created_at": now_utc(),
        "reforecast_root": str(args.reforecast_root),
        "requested_years": sorted(set(args.years)),
        "requested_months": sorted(set(args.months)),
        "forecast_days": list(requested_days),
        "q95_window_days": args.window_days,
        "minimum_q95_samples": args.minimum_q95_samples,
        "skipped_metadata_models": skipped_models,
        "models": reports,
    }
    summary = pd.DataFrame(summary_row(report) for report in reports).sort_values("model")
    print(summary.to_string(index=False))

    if args.dry_run:
        return
    output_directory = args.output_directory or (args.results_root / "model_climatology" / "preflight")
    write_json_atomic(payload, output_directory / "model_temperature_q95_preflight.json")
    write_table_atomic(summary, output_directory / "model_temperature_q95_preflight.csv")
    print(f"JSON report: {output_directory / 'model_temperature_q95_preflight.json'}")
    print(f"CSV report: {output_directory / 'model_temperature_q95_preflight.csv'}")


if __name__ == "__main__":
    main()
