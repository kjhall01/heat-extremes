#!/usr/bin/env python3
"""Make verification figures from aggregated products only; never opens raw data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.plotting import make_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--regions", nargs="+")
    parser.add_argument("--reliability-forecast-days", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_all_plots(
        args.result_dirs,
        args.output_directory,
        reliability_forecast_days=args.reliability_forecast_days,
        regions=args.regions,
    )
    print(f"Wrote aggregate-only figures to {args.output_directory}")


if __name__ == "__main__":
    main()
