#!/usr/bin/env python3
"""Initialize final-only global model q95 products from a ready preflight.

This command reads the preflight JSON plus one raw store's coordinate arrays
per model.  It does not read forecast-temperature chunks.  The resulting
manifest drives the Slurm longitude-band array: every task writes straight into
its model's one global q95.zarr product, with no persisted daily intermediate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.io import write_json_atomic
from heatextremes.verification.model_climatology import build_q95_workflow_manifest


DEFAULT_RESULTS_ROOT = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--preflight-report",
        type=Path,
        help=(
            "Default: <results-root>/model_climatology/preflight/"
            "model_temperature_q95_preflight.json"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Default: <results-root>/model_climatology/model_temperature_q95_2000_2020",
    )
    parser.add_argument(
        "--manifest", type=Path, help="Default: <output-directory>/workflow_manifest.json"
    )
    parser.add_argument("--models", nargs="+", help="Optional ready model names to include.")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2000, 2021)))
    parser.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    parser.add_argument("--max-forecast-day", type=int, choices=range(15), default=12)
    parser.add_argument("--window-days", type=int, default=15)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing final q95 product(s) only; raw stores are always read-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.years or not args.months:
        raise ValueError("--years and --months must not be empty")
    if any(not 1 <= month <= 12 for month in args.months):
        raise ValueError("--months must be between 1 and 12")
    years = tuple(sorted(set(args.years)))
    months = tuple(sorted(set(args.months)))
    period_label = f"{years[0]}_{years[-1]}"
    output_directory = args.output_directory or (
        args.results_root / "model_climatology" / f"model_temperature_q95_{period_label}"
    )
    preflight_report = args.preflight_report or (
        args.results_root
        / "model_climatology"
        / "preflight"
        / "model_temperature_q95_preflight.json"
    )
    manifest_path = args.manifest or (output_directory / "workflow_manifest.json")
    if not preflight_report.is_file():
        raise FileNotFoundError(f"Ready preflight report is missing: {preflight_report}")
    with preflight_report.open(encoding="utf-8") as handle:
        preflight = json.load(handle)

    manifest = build_q95_workflow_manifest(
        preflight,
        result_root=args.results_root,
        output_directory=output_directory,
        years=years,
        months=months,
        forecast_days=range(args.max_forecast_day + 1),
        window_days=args.window_days,
        percentile=args.percentile,
        models=args.models,
        overwrite=args.overwrite,
    )
    write_json_atomic(manifest, manifest_path)
    print(f"Manifest: {manifest_path}")
    print(f"Global products: {output_directory}/<model>/q95.zarr")
    print(f"Array tasks: {manifest['task_count']}")
    print("Raw source stores: read only")
    print("Daily intermediate storage: none (in-memory only within each array task)")


if __name__ == "__main__":
    main()
