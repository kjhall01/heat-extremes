#!/usr/bin/env python3
"""Make aggregate-only comparison figures for successful inventory models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.io import assert_safe_result_path, now_utc, write_json_atomic
from heatextremes.verification.plotting import make_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result_root = Path(manifest["results_root"])
    status_path = result_root / "inventory" / "aggregation_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"Aggregate status is missing: {status_path}")
    statuses = json.loads(status_path.read_text(encoding="utf-8"))["models"]
    result_dirs = [Path(item["output"]).parent for item in statuses if item["status"] == "complete"]
    output = result_root / "_all_models" / "figures"
    assert_safe_result_path(output, result_root)
    plot_status = result_root / "inventory" / "plot_status.json"
    assert_safe_result_path(plot_status, result_root)
    if not result_dirs:
        write_json_atomic(
            {"created_at": now_utc(), "status": "skipped", "reason": "no models aggregated successfully"},
            plot_status,
        )
        print("No successful model aggregates; no figures made")
        return
    make_all_plots(
        result_dirs,
        output,
        reliability_forecast_days=[0, 5, 10, 13],
        allowed_output_roots=[result_root],
    )
    write_json_atomic(
        {"created_at": now_utc(), "status": "complete", "result_dirs": [str(path) for path in result_dirs], "output": str(output)},
        plot_status,
    )
    print(f"Wrote all-model aggregate-only figures to {output}")


if __name__ == "__main__":
    main()
