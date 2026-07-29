#!/usr/bin/env python3
"""Consolidate and commit completed global model q95 products."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from heatextremes.verification.model_climatology import finalize_q95_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for result in finalize_q95_workflow(manifest):
        print(f"Finalized: {result['model']} -> {result['store']}")


if __name__ == "__main__":
    main()
