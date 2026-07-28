from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from heatextremes.verification.case_cache import (
    CACHE_SCHEMA_VERSION,
    case_cache_completion_is_valid,
    case_cache_lead_path,
    compute_case_cache_partition,
)
from heatextremes.verification.config import load_config


def test_shorter_discovered_lead_range_adopts_prior_committed_leads(tmp_path: Path) -> None:
    """A final unavailable lead must not make earlier committed work unusable."""
    regions = tmp_path / "regions.yaml"
    regions.write_text("regions:\n  global: {}\n", encoding="utf-8")
    config_path = tmp_path / "short_horizon.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "name": "short_horizon",
                    "adapter": "standard_reforecast_raw",
                    "ensemble": False,
                },
                "paths": {
                    "raw_reforecast_root": str(tmp_path / "not_opened"),
                    "verification_results_root": str(tmp_path / "results"),
                },
                "selection": {
                    "years": [2022],
                    "months": [6],
                    "forecast_days": list(range(14)),
                },
                "metrics": {"probability_bins": [0.0, 1.0], "interval_levels": [0.5]},
                "regions": {"file": str(regions)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    shortened = load_config(config_path)
    historical = load_config(
        config_path,
        overrides={"selection": {"forecast_days": list(range(15))}},
    )
    assert historical.case_cache_hash in shortened.compatible_case_cache_hashes

    directory = shortened.model_result_dir / "case_cache" / "2022-06"
    directory.mkdir(parents=True)
    (directory / "progress.json").write_text(
        json.dumps({"status": "in_progress", "case_cache_hash": historical.case_cache_hash}),
        encoding="utf-8",
    )
    for forecast_day in shortened.forecast_days:
        cached = xr.Dataset(
            {"forecast_temperature": ("initialization", np.asarray([300.0], dtype=np.float32))},
            coords={"initialization": [np.datetime64("2022-06-01")]},
        )
        cached.attrs.update(
            {
                "verification_case_cache_schema": CACHE_SCHEMA_VERSION,
                "case_cache_hash": historical.case_cache_hash,
                "model": shortened.model_name,
                "year": 2022,
                "month": 6,
                "forecast_day": forecast_day,
            }
        )
        cached.to_zarr(
            case_cache_lead_path(shortened, 2022, 6, forecast_day), mode="w", consolidated=True
        )

    # All usable stores predate the shorter configuration. A successful
    # adoption must create completion metadata without attempting to open the
    # deliberately nonexistent raw directory.
    result = compute_case_cache_partition(shortened, 2022, 6, repository_root=tmp_path)
    assert not result.skipped
    assert case_cache_completion_is_valid(directory, shortened, year=2022, month=6)
