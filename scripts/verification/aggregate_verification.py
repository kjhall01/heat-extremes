#!/usr/bin/env python3
"""Aggregate exact verification metrics from completed monthly partials."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.aggregation import aggregate_result_directory
from heatextremes.verification.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        required=True,
        help="One configuration per model result directory (one is the usual case).",
    )
    parser.add_argument(
        "--result-dirs",
        type=Path,
        nargs="+",
        help="One or more model result directories; defaults to the configured model directory.",
    )
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--months", type=int, nargs="+")
    parser.add_argument("--regions", nargs="+")
    parser.add_argument("--forecast-days", type=int, nargs="+")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [load_config(path) for path in args.config]
    if args.result_dirs and len(args.result_dirs) != len(configs):
        raise ValueError("--result-dirs must have exactly one entry per --config")
    result_dirs = args.result_dirs or [config.model_result_dir for config in configs]
    for config, result_dir in zip(configs, result_dirs, strict=True):
        output, discovery = aggregate_result_directory(
            config,
            result_dir=result_dir,
            years=set(args.years) if args.years else None,
            months=set(args.months) if args.months else None,
            regions=set(args.regions) if args.regions else None,
            forecast_days=set(args.forecast_days) if args.forecast_days else None,
            allow_missing=args.allow_missing,
        )
        print(f"Wrote {output}; completed={len(discovery.completed)} missing={list(discovery.missing)}")


if __name__ == "__main__":
    main()
