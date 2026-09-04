#!/usr/bin/env python3
"""Aggregate global Z500 partials and join the global T2M report scorecard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.global_z500 import aggregate_global_z500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--heat-scorecard", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--models", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    z500, combined = aggregate_global_z500(
        args.output_directory,
        heat_scorecard_path=args.heat_scorecard,
        manifest_path=args.manifest,
        models=args.models,
    )
    print(f"Wrote {len(z500)} Z500 rows and {len(combined)} combined global-scorecard rows")


if __name__ == "__main__":
    main()
