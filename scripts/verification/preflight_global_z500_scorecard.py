#!/usr/bin/env python3
"""Inspect Z500 source/ERA5 metadata before submitting global scorecard tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.global_z500 import (
    DEFAULT_FORECAST_DAYS,
    DEFAULT_LEAD_HOURS,
    preflight_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--era5-z500-store", type=Path, required=True)
    parser.add_argument("--era5-z500-variable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--forecast-days", type=int, nargs="+", default=DEFAULT_FORECAST_DAYS)
    parser.add_argument("--lead-hours", type=int, nargs="+", default=DEFAULT_LEAD_HOURS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = preflight_manifest(
        args.manifest,
        era5_z500_store=args.era5_z500_store,
        era5_z500_variable=args.era5_z500_variable,
        models=args.models,
        forecast_days=args.forecast_days,
        lead_hours=args.lead_hours,
        output_path=args.output,
    )
    available = [item["model"] for item in payload["models"] if item["status"] == "available"]
    unavailable = [item["model"] for item in payload["models"] if item["status"] != "available"]
    print(f"Wrote {args.output}; available={available}; unavailable={unavailable}")


if __name__ == "__main__":
    main()
