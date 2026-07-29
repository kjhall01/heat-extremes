#!/usr/bin/env python3
"""Compute one manifest-listed longitude region of a global model q95 product."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.model_climatology import compute_q95_band


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute this final global longitude region."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    tasks = manifest.get("tasks", [])
    if not 0 <= args.task_index < len(tasks):
        raise IndexError(f"task index {args.task_index} is outside 0 through {len(tasks) - 1}")
    task = tasks[args.task_index]
    result = compute_q95_band(
        manifest,
        model_name=str(task["model"]),
        band_index=int(task["band_index"]),
        overwrite=args.overwrite,
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
