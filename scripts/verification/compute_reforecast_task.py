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
from heatextremes.verification.case_cache import compute_case_cache_partition
from heatextremes.verification.io import (
    assert_safe_result_path,
    now_utc,
    remove_result_path,
    write_json_atomic,
)
from heatextremes.verification.runner import compute_partition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--stage",
        choices=("case_cache", "cached_metrics", "raw_metrics"),
        default="case_cache",
        help="The all-model workflow uses case_cache then cached_metrics.",
    )
    parser.add_argument("--regions", nargs="+")
    parser.add_argument("--probability-bins", type=float, nargs="+")
    parser.add_argument("--decision-thresholds", type=float, nargs="+")
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
    metric_overrides: dict[str, object] = {}
    if args.probability_bins:
        metric_overrides["probability_bins"] = args.probability_bins
    if args.decision_thresholds:
        metric_overrides["probability_decision_thresholds"] = args.decision_thresholds
    overrides = {"metrics": metric_overrides} if metric_overrides else None
    config = load_config(task["config"], overrides=overrides)
    label = f"{int(task['year']):04d}-{int(task['month']):02d}"
    failure = config.model_result_dir / "failures" / args.stage / f"{label}.json"
    print(
        f"Model: {config.model_name}\nPartition: {label}\nStage: {args.stage}\nManifest task: {args.task_index}",
        flush=True,
    )
    try:
        if args.stage == "case_cache":
            result = compute_case_cache_partition(
                config,
                int(task["year"]),
                int(task["month"]),
                overwrite=args.overwrite,
                resume=True,
                repository_root=Path(__file__).resolve().parents[2],
            )
        else:
            result = compute_partition(
                config,
                int(task["year"]),
                int(task["month"]),
                overwrite=args.overwrite,
                resume=True,
                repository_root=Path(__file__).resolve().parents[2],
                input_source="case_cache" if args.stage == "cached_metrics" else "raw",
                regions=args.regions,
            )
    except Exception as error:
        assert_safe_result_path(failure, config.result_root)
        write_json_atomic(
            {
                "status": "failed",
                "model": config.model_name,
                "partition": label,
                "stage": args.stage,
                "task_index": args.task_index,
                "created_at": now_utc(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
            failure,
        )
        print(
            f"ISOLATED FAILURE: {config.model_name} {label} ({args.stage}): {error}",
            file=sys.stderr,
            flush=True,
        )
        if args.raise_on_error:
            raise
        return
    # A resumed partition can repair a previous isolated failure. Remove only
    # this explicitly scoped marker so status inspection is not stale.
    remove_result_path(failure, config.result_root)
    print(f"{'Skipped' if result.skipped else 'Completed'} {result.partition_directory}", flush=True)


if __name__ == "__main__":
    main()
