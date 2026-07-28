#!/usr/bin/env python3
"""Aggregate every model that has completed partitions in an inventory."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.aggregation import aggregate_result_directory
from heatextremes.verification.config import load_config
from heatextremes.verification.io import assert_safe_result_path, now_utc, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result_root = Path(manifest["results_root"])
    statuses: list[dict[str, object]] = []
    for record in manifest["models"]:
        if not record["selected_partitions"]:
            statuses.append({"model": record["model"], "status": "skipped", "reason": "no selected initializations"})
            continue
        config = load_config(record["config"])
        try:
            output, discovery = aggregate_result_directory(config, allow_missing=True)
            statuses.append(
                {
                    "model": config.model_name,
                    "status": "complete",
                    "output": str(output),
                    "completed_partitions": [item.name for item in discovery.completed],
                    "missing_partitions": list(discovery.missing),
                }
            )
            print(f"Aggregated {config.model_name}: completed={len(discovery.completed)}", flush=True)
        except Exception as error:
            statuses.append(
                {
                    "model": record["model"],
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"ISOLATED AGGREGATION FAILURE: {record['model']}: {error}", file=sys.stderr, flush=True)
    status_path = result_root / "inventory" / "aggregation_status.json"
    assert_safe_result_path(status_path, result_root)
    write_json_atomic({"created_at": now_utc(), "models": statuses}, status_path)
    print(f"Aggregation status: {status_path}")


if __name__ == "__main__":
    main()
