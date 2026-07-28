#!/usr/bin/env python3
"""Compute one bounded, restartable month of forecast verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.config import load_config
from heatextremes.verification.runner import compute_partition, dry_run_partition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--regions", nargs="+")
    parser.add_argument("--forecast-days", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(
            json.dumps(
                dry_run_partition(
                    config,
                    args.year,
                    args.month,
                    regions=args.regions,
                    forecast_days=args.forecast_days,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = compute_partition(
        config,
        args.year,
        args.month,
        regions=args.regions,
        forecast_days=args.forecast_days,
        overwrite=args.overwrite,
        resume=args.resume,
        repository_root=Path(__file__).resolve().parents[2],
    )
    print(
        f"{'Skipped' if result.skipped else 'Completed'} {result.partition_directory}; "
        f"forecast_days={list(result.completed_forecast_days)}"
    )


if __name__ == "__main__":
    main()
