from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from heatextremes.verification.case_cache_reader import (
    open_model_case_cache,
    open_model_intermediates,
)


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
            "event_case_valid": (
                ("event", "initialization", "latitude", "longitude"),
                np.ones((1, 1, 1, 1), dtype=bool),
            ),
        },
        coords={
            "initialization": [np.datetime64("2022-06-01")],
            "latitude": [0.0],
            "longitude": [0.0],
            "forecast_day": forecast_day,
            "event": ["hot_day_q95"],
            "valid_date": (
                ("initialization", "longitude"),
                np.asarray([["2022-06-01"]], dtype="datetime64[ns]"),
            ),
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
    assert "[1/2] Opening 2022-06" in output
    assert "Modern cache example_model: ready; data remain lazy" in output
    assert "Filled with NaNs: 2022-06/forecast_day_001" in output
    assert "Missing month(s): 2022-07" in output
    assert dataset["forecast_temperature"].chunks is not None
    assert dataset.forecast_day.values.tolist() == [0, 1, 2]
    assert np.isnan(dataset["forecast_temperature"].sel(forecast_day=1).compute()).all()
    assert np.isnat(dataset["valid_date"].sel(forecast_day=1).compute()).all()
    assert dataset["event_case_valid"].dtype == bool
    assert not dataset["event_case_valid"].sel(forecast_day=1).compute().any()
    assert dataset["forecast_temperature"].sel(forecast_day=2).compute().item() == 282.0


def test_aifs_ens_v2_uses_legacy_monthly_intermediates(tmp_path: Path, capsys) -> None:
    monthly_root = tmp_path / "monthly"
    store = monthly_root / "2022" / "aifs_ens_v2_heat_202206.zarr"
    store.parent.mkdir(parents=True)
    xr.Dataset(
        {
            "t2m_daily_mean_ensemble_mean": (
                ("time", "forecast_day", "latitude", "longitude"),
                np.asarray([[[[280.0]], [[282.0]]]], dtype=np.float32),
            ),
            "hot_day_q95_probability": (
                ("time", "forecast_day", "latitude", "longitude"),
                np.asarray([[[[0.2]], [[0.4]]]], dtype=np.float32),
            ),
            "heatwave_start_q95_2d_probability": (
                ("time", "forecast_day", "latitude", "longitude"),
                np.asarray([[[[0.1]], [[0.3]]]], dtype=np.float32),
            ),
            "heatwave_start_q95_3d_probability": (
                ("time", "forecast_day", "latitude", "longitude"),
                np.asarray([[[[0.05]], [[0.15]]]], dtype=np.float32),
            ),
        },
        coords={
            "time": [np.datetime64("2022-06-01")],
            "forecast_day": [0, 2],
            "latitude": [0.0],
            "longitude": [0.0],
            "valid_date": (
                ("time", "forecast_day", "longitude"),
                np.asarray([[["2022-06-01"], ["2022-06-02"]]], dtype="datetime64[ns]"),
            ),
        },
    ).chunk({"time": 1}).to_zarr(store, mode="w", consolidated=False)
    daily_store = tmp_path / "era5_daily.zarr"
    xr.Dataset(
        {"t2m_daily_mean": (("time", "latitude", "longitude"), np.asarray([[[281.0]], [[283.0]]]))},
        coords={
            "time": np.asarray(["2022-06-01", "2022-06-02"], dtype="datetime64[ns]"),
            "latitude": [0.0],
            "longitude": [0.0],
        },
    ).to_zarr(daily_store, mode="w", consolidated=True)
    hazard_store = tmp_path / "era5_hazards.zarr"
    xr.Dataset(
        {
            event: (("time", "latitude", "longitude"), np.asarray([[[0]], [[1]]], dtype=np.uint8))
            for event in ("hot_day_q95", "heatwave_start_q95_2d", "heatwave_start_q95_3d")
        },
        coords={
            "time": np.asarray(["2022-06-01", "2022-06-02"], dtype="datetime64[ns]"),
            "latitude": [0.0],
            "longitude": [0.0],
        },
    ).to_zarr(hazard_store, mode="w", consolidated=True)

    dataset = open_model_intermediates(
        "aifs_ens_v2",
        monthly_root=monthly_root,
        era5_daily_temperature_store=daily_store,
        era5_hazard_store=hazard_store,
        forecast_days=[0, 1, 2],
    )

    assert "Legacy AIFS monthly report" in capsys.readouterr().out
    assert dataset.attrs["intermediate_reader_source"] == "legacy_aifs_monthly"
    assert dataset["forecast_temperature"].chunks is not None
    assert set(dataset.data_vars) == {
        "forecast_temperature",
        "observation_temperature",
        "temperature_case_valid",
        "forecast_probability",
        "observed_event",
        "event_case_valid",
    }
    np.testing.assert_array_equal(
        dataset.initialization.values,
        np.asarray(["2022-06-01"], dtype="datetime64[ns]"),
    )
    assert dataset.forecast_day.values.tolist() == [0, 1, 2]
    assert np.isnan(dataset["forecast_temperature"].sel(forecast_day=1).compute()).all()
    assert np.isnan(dataset["observation_temperature"].sel(forecast_day=1).compute()).all()
    assert dataset["observation_temperature"].sel(forecast_day=2).compute().item() == 283.0


def test_aifs_legacy_reader_filters_stores_before_opening(tmp_path: Path) -> None:
    monthly_root = tmp_path / "monthly"
    for label in ("202206", "202212"):
        store = monthly_root / label[:4] / f"aifs_ens_v2_heat_{label}.zarr"
        store.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {
                "t2m_daily_mean_ensemble_mean": (
                    ("time", "forecast_day", "latitude", "longitude"),
                    np.full((1, 1, 1, 1), 280.0, dtype=np.float32),
                ),
                "hot_day_q95_probability": (
                    ("time", "forecast_day", "latitude", "longitude"),
                    np.full((1, 1, 1, 1), 0.2, dtype=np.float32),
                ),
                "heatwave_start_q95_2d_probability": (
                    ("time", "forecast_day", "latitude", "longitude"),
                    np.full((1, 1, 1, 1), 0.1, dtype=np.float32),
                ),
                "heatwave_start_q95_3d_probability": (
                    ("time", "forecast_day", "latitude", "longitude"),
                    np.full((1, 1, 1, 1), 0.05, dtype=np.float32),
                ),
            },
            coords={
                "time": [np.datetime64(f"{label[:4]}-{label[4:]}-01")],
                "forecast_day": [0],
                "latitude": [0.0],
                "longitude": [0.0],
                "valid_date": (
                    ("time", "forecast_day", "longitude"),
                    np.asarray([[[f"{label[:4]}-{label[4:]}-01"]]], dtype="datetime64[ns]"),
                ),
            },
        ).to_zarr(store, mode="w", consolidated=False)
    daily_store = tmp_path / "daily.zarr"
    xr.Dataset(
        {"t2m_daily_mean": (("time", "latitude", "longitude"), np.full((2, 1, 1), 281.0))},
        coords={
            "time": np.asarray(["2022-06-01", "2022-12-01"], dtype="datetime64[ns]"),
            "latitude": [0.0],
            "longitude": [0.0],
        },
    ).to_zarr(daily_store, mode="w", consolidated=True)
    hazard_store = tmp_path / "hazards.zarr"
    xr.Dataset(
        {event: (("time", "latitude", "longitude"), np.zeros((2, 1, 1))) for event in ("hot_day_q95", "heatwave_start_q95_2d", "heatwave_start_q95_3d")},
        coords={
            "time": np.asarray(["2022-06-01", "2022-12-01"], dtype="datetime64[ns]"),
            "latitude": [0.0],
            "longitude": [0.0],
        },
    ).to_zarr(hazard_store, mode="w", consolidated=True)

    dataset = open_model_intermediates(
        "aifs_ens_v2",
        monthly_root=monthly_root,
        era5_daily_temperature_store=daily_store,
        era5_hazard_store=hazard_store,
        forecast_days=[0],
        months=[6],
    )

    np.testing.assert_array_equal(
        dataset.initialization.dt.month.values,
        np.asarray([6]),
    )
