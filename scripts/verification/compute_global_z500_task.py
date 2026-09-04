#!/usr/bin/env python3
"""Compute one model/month additive global-Z500 RMSE partial."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.global_z500 import (
    DEFAULT_FORECAST_DAYS,
    DEFAULT_LEAD_HOURS,
    compute_manifest_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--era5-z500-store", type=Path, required=True)
    parser.add_argument("--era5-z500-variable", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--forecast-days", type=int, nargs="+", default=DEFAULT_FORECAST_DAYS)
    parser.add_argument("--lead-hours", type=int, nargs="+", default=DEFAULT_LEAD_HOURS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = compute_manifest_task(
        args.manifest,
        task_index=args.task_index,
        era5_z500_store=args.era5_z500_store,
        era5_z500_variable=args.era5_z500_variable,
        output_directory=args.output_directory,
        forecast_days=args.forecast_days,
        lead_hours=args.lead_hours,
        preflight_path=args.preflight,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
