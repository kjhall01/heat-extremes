#!/usr/bin/env python3
"""Compute one small group of staged model-band-lead q95 fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.model_climatology import compute_q95_lead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lead_tasks = manifest.get("quantile_tasks", [])
    job_tasks = manifest.get("quantile_job_tasks", [])
    if not 0 <= args.task_index < len(job_tasks):
        raise IndexError(f"task index {args.task_index} is outside 0 through {len(job_tasks) - 1}")
    for lead_task_index in job_tasks[args.task_index]["lead_task_indices"]:
        task = lead_tasks[int(lead_task_index)]
        print(
            compute_q95_lead(
                manifest,
                model_name=str(task["model"]),
                band_index=int(task["band_index"]),
                forecast_day=int(task["forecast_day"]),
                overwrite=args.overwrite,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
