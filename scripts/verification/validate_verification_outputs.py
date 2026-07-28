#!/usr/bin/env python3
"""Validate expected monthly partial outputs without opening forecast data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.aggregation import discover_partitions, verify_partition_metadata
from heatextremes.verification.config import load_config
from heatextremes.verification.io import TABLE_STEMS, find_table, read_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result_dir = args.result_dir or config.model_result_dir
    discovery = discover_partitions(result_dir, config.partitions())
    failures: list[str] = []
    if discovery.completed:
        signature = verify_partition_metadata(discovery.completed)
        print(f"Metadata signature: {signature}")
        if signature.get("configuration_hash") != config.config_hash:
            failures.append("completion configuration hash differs from supplied configuration")
        expected_days = set(config.forecast_days)
        for directory in discovery.completed:
            for stem in TABLE_STEMS:
                path = find_table(directory, stem)
                if path is None:
                    failures.append(f"{directory.name}: missing {stem} table")
                    continue
                frame = read_table(path)
                if "forecast_day" not in frame or set(frame["forecast_day"]) != expected_days:
                    failures.append(f"{directory.name}: {stem} does not contain exactly configured forecast days")
    print(f"Completed partitions ({len(discovery.completed)}):")
    for path in discovery.completed:
        print(f"  OK      {path.name}")
    print(f"Missing partitions ({len(discovery.missing)}):")
    for label in discovery.missing:
        print(f"  MISSING {label}")
    if failures:
        print("Validation failures:")
        for failure in failures:
            print(f"  FAILED  {failure}")
    if discovery.missing or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
