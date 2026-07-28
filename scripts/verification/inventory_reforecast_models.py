#!/usr/bin/env python3
"""Inventory standard raw reforecast directories and generate model configs.

This scans directory/store names plus Zarr metadata and coordinates for one
store per selected month. It never loads model-temperature chunks. All
manifests and generated configuration files are written beneath the explicitly
supplied results root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.io import assert_safe_result_path, now_utc, write_json_atomic
from heatextremes.verification.reforecast_inventory import inventory_metadata_csv, raw_reforecast_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reforecast-root",
        type=Path,
        default=Path("/net/monsoon/marchakitus/reforecast"),
        help="Directory containing forecasts_<model>/init_*.zarr directories.",
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        help="Authoritative model registry; defaults to Rossby Model Storage Locations - Sheet1.csv in the repository root.",
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--months", type=int, nargs="+", default=[6, 7, 8, 9])
    parser.add_argument(
        "--max-forecast-day",
        type=int,
        choices=range(15),
        default=14,
        help="Inclusive zero-based local-solar lead cap (default: 14, i.e. days 0 through 14).",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-directory", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_yaml_atomic(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    root = args.reforecast_root.resolve()
    results_root = args.results_root.resolve(strict=False)
    manifest = args.manifest.resolve(strict=False)
    config_directory = args.config_directory.resolve(strict=False)
    assert_safe_result_path(manifest, results_root)
    assert_safe_result_path(config_directory / "placeholder.yaml", results_root)
    region_file = args.repository_root.resolve() / "configs" / "verification" / "regions.yaml"
    if not region_file.is_file():
        raise FileNotFoundError(f"Region configuration is missing: {region_file}")
    metadata_csv = args.metadata_csv or (args.repository_root.resolve() / "Rossby Model Storage Locations - Sheet1.csv")
    inventories, skipped_models = inventory_metadata_csv(
        metadata_csv.resolve(),
        root=root,
        years=args.years,
        months=args.months,
        max_forecast_day=args.max_forecast_day,
    )
    records: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    for item in inventories:
        if item.lead_inventory_errors:
            skipped_models.append(
                {
                    "model": item.display_name,
                    "reason": "could not establish a safe common local-solar forecast-day range: "
                    + "; ".join(item.lead_inventory_errors),
                }
            )
            continue
        config_path = config_directory / f"{item.name}.yaml"
        selected = [{"year": year, "month": month} for year, month in item.partitions]
        records.append(
            {
                "model": item.name,
                "display_name": item.display_name,
                "source_directory": str(item.directory),
                "source_temperature_variable": item.source_temperature_variable,
                "source_store_glob": item.source_store_glob,
                "ensemble": item.ensemble,
                "store_count": item.store_count,
                "selected_partitions": selected,
                "forecast_days": list(item.forecast_days),
                "lead_inventory_errors": list(item.lead_inventory_errors),
                "unparsed_store_names": list(item.unparsed_store_names),
                "config": str(config_path),
            }
        )
        for year, month in item.partitions:
            tasks.append(
                {
                    "model": item.name,
                    "config": str(config_path),
                    "year": year,
                    "month": month,
                }
            )
        if not args.dry_run and item.partitions:
            assert_safe_result_path(config_path, results_root)
            _write_yaml_atomic(raw_reforecast_config(item, result_root=results_root, region_file=region_file), config_path)
    payload = {
        "status": "inventory_complete",
        "created_at": now_utc(),
        "reforecast_root": str(root),
        "results_root": str(results_root),
        "requested_years": sorted(set(args.years)),
        "requested_months": sorted(set(args.months)),
        "requested_max_forecast_day": args.max_forecast_day,
        "metadata_csv": str(metadata_csv.resolve()),
        "models": records,
        "skipped_metadata_models": skipped_models,
        "tasks": tasks,
        "task_count": len(tasks),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    write_json_atomic(payload, manifest)
    print(f"Inventoried {len(records)} model directories and {len(tasks)} selected model-month tasks")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
