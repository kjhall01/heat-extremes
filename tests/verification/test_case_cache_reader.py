from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from heatextremes.verification.case_cache_reader import open_model_case_cache


def _write_case_store(path: Path, *, forecast_day: int, value: float) -> None:
    dataset = xr.Dataset(
        {
            "forecast_temperature": (
                ("initialization", "latitude", "longitude"),
                np.full((1, 1, 1), value, dtype=np.float32),
            ),
            "temperature_case_valid": (
                ("initialization", "latitude", "longitude"),
                np.ones((1, 1, 1), dtype=bool),
            ),
        },
        coords={
            "initialization": [np.datetime64("2022-06-01")],
            "latitude": [0.0],
            "longitude": [0.0],
            "forecast_day": forecast_day,
        },
    ).chunk({"initialization": 1})
    dataset.to_zarr(path, mode="w", consolidated=True)


def test_lazy_case_cache_reader_fills_missing_leads_and_reports_them(
    tmp_path: Path,
    capsys,
) -> None:
    results_root = tmp_path / "results"
    partition = results_root / "example_model" / "case_cache" / "2022-06"
    partition.mkdir(parents=True)
    _write_case_store(partition / "forecast_day_000.zarr", forecast_day=0, value=280.0)
    _write_case_store(partition / "forecast_day_002.zarr", forecast_day=2, value=282.0)
    inventory = results_root / "inventory"
    inventory.mkdir()
    (inventory / "reforecast_inventory.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model": "example_model",
                        "selected_partitions": [
                            {"year": 2022, "month": 6},
                            {"year": 2022, "month": 7},
                        ],
                        "forecast_days": [0, 1, 2, 99],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = open_model_case_cache("example_model", results_root=results_root)

    output = capsys.readouterr().out
    assert "Filled with NaNs: 2022-06/forecast_day_001" in output
    assert "Missing month(s): 2022-07" in output
    assert dataset["forecast_temperature"].chunks is not None
    assert dataset.forecast_day.values.tolist() == [0, 1, 2]
    assert np.isnan(dataset["forecast_temperature"].sel(forecast_day=1).compute()).all()
    assert dataset["forecast_temperature"].sel(forecast_day=2).compute().item() == 282.0
