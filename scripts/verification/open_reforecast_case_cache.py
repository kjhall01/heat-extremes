#!/usr/bin/env python3
"""Open one model's reforecast case-cache stores lazily and report gaps.

Example:
    python scripts/verification/open_reforecast_case_cache.py \
      --model-name aurora_e2s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.case_cache_reader import (
    DEFAULT_RESULTS_ROOT,
    open_model_case_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Model result-directory name.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--forecast-days",
        type=int,
        nargs="+",
        help="Expected zero-based forecast-day labels (default: manifest or 0 through 14).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = open_model_case_cache(
        args.model_name,
        results_root=args.results_root,
        forecast_days=args.forecast_days,
    )
    print(dataset)


if __name__ == "__main__":
    main()
