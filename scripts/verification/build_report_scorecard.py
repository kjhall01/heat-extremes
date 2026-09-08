#!/usr/bin/env python3
"""Build the report-facing heat scorecard from completed canonical case caches.

The script does not open raw model forecast stores except for the historical
AIFS ENS v2 compact monthly product, which has not yet been migrated to the
canonical cache.  It writes raw (not bias-corrected) deterministic q95
POD/FAR and native-probability Brier diagnostics side by side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.report_scorecard import (
    DEFAULT_AIFS_MONTHLY_ROOT,
    DEFAULT_ERA5_DAILY_TEMPERATURE_STORE,
    DEFAULT_ERA5_HAZARD_STORE,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_THRESHOLD_STORE,
    DEFAULT_THRESHOLD_VARIABLE,
    build_report_scorecard,
)
from heatextremes.verification.regions import load_regions, select_regions


DEFAULT_MODELS = ["aifs_ens_v2", "ifs_ens", "aifs_v2", "aurora_e2s", "graphcast_e2s"]
# The report-facing default is the publishable Nigeria panel.  Other regional
# or global scorecards remain available through --regions when they are needed.
DEFAULT_REGIONS = ["nigeria"]
DEFAULT_FORECAST_DAYS = [0, 3, 6, 9, 12]
DEFAULT_YEARS = [2022, 2023, 2024, 2025]
DEFAULT_MONTHS = [6, 7, 8, 9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--region-file", type=Path, default=Path("configs/verification/regions.yaml"))
    parser.add_argument("--forecast-days", type=int, nargs="+", default=DEFAULT_FORECAST_DAYS)
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--months", type=int, nargs="+", default=DEFAULT_MONTHS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--legacy-aifs-monthly-root", type=Path, default=DEFAULT_AIFS_MONTHLY_ROOT)
    parser.add_argument("--era5-daily-temperature-store", type=Path, default=DEFAULT_ERA5_DAILY_TEMPERATURE_STORE)
    parser.add_argument("--era5-hazard-store", type=Path, default=DEFAULT_ERA5_HAZARD_STORE)
    parser.add_argument("--threshold-store", type=Path, default=DEFAULT_THRESHOLD_STORE)
    parser.add_argument("--threshold-variable", default=DEFAULT_THRESHOLD_VARIABLE)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument(
        "--no-frequency-change-maps",
        action="store_false",
        dest="include_frequency_change_maps",
        help="Skip the observed hot-day-frequency map column when regenerating tables only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    region_file = args.region_file
    if not region_file.is_absolute():
        region_file = Path(__file__).resolve().parents[2] / region_file
    regions = select_regions(load_regions(region_file), args.regions)
    scorecard = build_report_scorecard(
        models=args.models,
        regions=regions,
        output_directory=args.output_directory,
        results_root=args.results_root,
        legacy_aifs_monthly_root=args.legacy_aifs_monthly_root,
        era5_daily_temperature_store=args.era5_daily_temperature_store,
        era5_hazard_store=args.era5_hazard_store,
        threshold_store=args.threshold_store,
        threshold_variable=args.threshold_variable,
        threshold_percentile=args.threshold_percentile,
        years=args.years,
        months=args.months,
        forecast_days=args.forecast_days,
        include_frequency_change_maps=args.include_frequency_change_maps,
    )
    print(f"Wrote {len(scorecard)} scorecard rows to {args.output_directory}")


if __name__ == "__main__":
    main()
