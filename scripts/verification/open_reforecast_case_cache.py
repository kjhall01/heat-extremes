#!/usr/bin/env python3
"""Open one model's intermediate Zarr stores lazily and report gaps.

Example:
    python scripts/verification/open_reforecast_case_cache.py \
      --model-name aurora_e2s

``aifs_ens_v2`` instead opens the legacy compact monthly intermediates used by
``03_full_aifs_heat_verification.ipynb``.  Other model names open the modern
reforecast case cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.case_cache_reader import (
    DEFAULT_AIFS_MONTHLY_ROOT,
    DEFAULT_ERA5_DAILY_TEMPERATURE_STORE,
    DEFAULT_ERA5_HAZARD_STORE,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_VERIFICATION_MONTHS,
    open_model_intermediates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Model result-directory name.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--monthly-root",
        type=Path,
        default=DEFAULT_AIFS_MONTHLY_ROOT,
        help="Legacy compact AIFS monthly-store root (used only for aifs_ens_v2).",
    )
    parser.add_argument(
        "--era5-daily-temperature-store",
        type=Path,
        default=DEFAULT_ERA5_DAILY_TEMPERATURE_STORE,
        help="ERA5 daily-temperature store used to canonically align legacy AIFS.",
    )
    parser.add_argument(
        "--era5-hazard-store",
        type=Path,
        default=DEFAULT_ERA5_HAZARD_STORE,
        help="ERA5 hazard store used to canonically align legacy AIFS.",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        nargs="+",
        help="Expected zero-based forecast-day labels (default: manifest or 0 through 14).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="Legacy AIFS years to open (ignored for modern cache models).",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=DEFAULT_VERIFICATION_MONTHS,
        help="Legacy AIFS months to open; default is JJAS, 6 7 8 9 (ignored for modern models).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = open_model_intermediates(
        args.model_name,
        results_root=args.results_root,
        monthly_root=args.monthly_root,
        era5_daily_temperature_store=args.era5_daily_temperature_store,
        era5_hazard_store=args.era5_hazard_store,
        forecast_days=args.forecast_days,
        years=args.years,
        months=args.months,
    )
    print(dataset)


if __name__ == "__main__":
    main()
