#!/usr/bin/env python3
"""Compute one task from a raw-reforecast inventory manifest.

Expected source problems are recorded per model/month and return success to
Slurm so one malformed model cannot block independent models or aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.config import load_config
from heatextremes.verification.io import assert_safe_result_path, now_utc, write_json_atomic
from heatextremes.verification.runner import compute_partition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--raise-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if not 0 <= args.task_index < len(tasks):
        raise IndexError(f"Task index {args.task_index} is outside manifest task_count={len(tasks)}")
    task = tasks[args.task_index]
    config = load_config(task["config"])
    label = f"{int(task['year']):04d}-{int(task['month']):02d}"
    print(f"Model: {config.model_name}\nPartition: {label}\nManifest task: {args.task_index}", flush=True)
    try:
        result = compute_partition(
            config,
            int(task["year"]),
            int(task["month"]),
            overwrite=args.overwrite,
            resume=True,
            repository_root=Path(__file__).resolve().parents[2],
        )
    except Exception as error:
        failure = config.model_result_dir / "failures" / f"{label}.json"
        assert_safe_result_path(failure, config.result_root)
        write_json_atomic(
            {
                "status": "failed",
                "model": config.model_name,
                "partition": label,
                "task_index": args.task_index,
                "created_at": now_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            failure,
        )
        print(f"ISOLATED FAILURE: {config.model_name} {label}: {error}", file=sys.stderr, flush=True)
        if args.raise_on_error:
            raise
        return
    print(f"{'Skipped' if result.skipped else 'Completed'} {result.partition_directory}", flush=True)


if __name__ == "__main__":
    main()
