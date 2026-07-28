from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from heatextremes.verification.config import load_config
from heatextremes.verification.models import get_model_adapter


def _write_store(dataset: xr.Dataset, path: Path) -> None:
    dataset.to_zarr(path, mode="w", consolidated=True)


def test_standard_raw_adapter_builds_one_canonical_lead_without_compact_output(tmp_path: Path) -> None:
    source = tmp_path / "forecasts_example"
    source.mkdir()
    longitude = np.asarray([0.0, 90.0, 180.0, 270.0])
    steps = np.arange(60, dtype="timedelta64[h]") * 6
    raw = xr.Dataset(
        {
            "2t": (
                ("time", "number", "prediction_timedelta", "latitude", "longitude"),
                np.stack(
                    [
                        np.full((1, steps.size, 1, longitude.size), 300.0),
                        np.full((1, steps.size, 1, longitude.size), 302.0),
                    ],
                    axis=1,
                ),
            )
        },
        coords={
            "time": [np.datetime64("2022-06-01T00")],
            "number": [0, 1],
            "prediction_timedelta": steps,
            "latitude": [0.0],
            "longitude": longitude,
        },
    )
    _write_store(raw, source / "init_2022-06-01.zarr")

    days = np.arange(1, 367, dtype=np.int16)
    threshold = xr.Dataset(
        {
            "t2m_daily_mean_calendar_day_percentile": (
                ("percentiles", "dayofyear", "latitude", "longitude"),
                np.full((1, days.size, 1, longitude.size), 301.0),
            )
        },
        coords={"percentiles": [95.0], "dayofyear": days, "latitude": [0.0], "longitude": longitude},
    )
    threshold_path = tmp_path / "threshold.zarr"
    _write_store(threshold, threshold_path)

    valid_times = np.arange(np.datetime64("2022-05-25"), np.datetime64("2022-07-31"), dtype="datetime64[D]")
    daily = xr.Dataset(
        {"t2m_daily_mean": (("time", "latitude", "longitude"), np.full((valid_times.size, 1, longitude.size), 300.0))},
        coords={"time": valid_times, "latitude": [0.0], "longitude": longitude},
    )
    hazards = xr.Dataset(
        {
            event: (("time", "latitude", "longitude"), np.zeros((valid_times.size, 1, longitude.size)))
            for event in ("hot_day_q95", "heatwave_start_q95_2d", "heatwave_start_q95_3d")
        },
        coords={"time": valid_times, "latitude": [0.0], "longitude": longitude},
    )
    daily_path = tmp_path / "daily.zarr"
    hazards_path = tmp_path / "hazards.zarr"
    _write_store(daily, daily_path)
    _write_store(hazards, hazards_path)
    region_file = tmp_path / "regions.yaml"
    region_file.write_text("regions:\n  global: {}\n", encoding="utf-8")

    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "example", "adapter": "standard_reforecast_raw", "ensemble": True},
                "paths": {
                    "raw_reforecast_root": str(source),
                    "threshold_store": str(threshold_path),
                    "era5_daily_temperature_store": str(daily_path),
                    "era5_hazard_store": str(hazards_path),
                    "verification_results_root": str(tmp_path / "results"),
                },
                "variables": {"raw_source_temperature": "2t"},
                "observations": {
                    "daily_temperature_variable": "t2m_daily_mean",
                    "event_variables": {
                        "hot_day_q95": "hot_day_q95",
                        "heatwave_start_q95_2d": "heatwave_start_q95_2d",
                        "heatwave_start_q95_3d": "heatwave_start_q95_3d",
                    },
                },
                "selection": {"years": [2022], "months": [6], "forecast_days": list(range(15))},
                "metrics": {"probability_bins": [0.0, 0.5, 1.0], "interval_levels": [0.5]},
                "regions": {"file": str(region_file)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    adapter = get_model_adapter(config)
    opened = adapter.open_partition(2022, 6)
    try:
        lead = adapter.lead(opened, 0)
        assert np.isclose(lead.ensemble_mean_temperature.mean().compute().item(), 301.0)
        assert np.isclose(lead.event_probabilities["hot_day_q95"].mean().compute().item(), 0.5)
        assert lead.interval_quantiles is not None
    finally:
        adapter.close_partition(opened)

    deterministic_source = tmp_path / "forecasts_deterministic"
    deterministic_source.mkdir()
    # A singleton ``number`` axis is accepted as deterministic too.
    _write_store(raw.isel(number=[0]), deterministic_source / "init_2022-06-01.zarr")
    deterministic_payload = copy.deepcopy(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    deterministic_payload["model"] = {
        "name": "deterministic",
        "adapter": "standard_reforecast_raw",
        "ensemble": False,
    }
    deterministic_payload["paths"]["raw_reforecast_root"] = str(deterministic_source)
    deterministic_path = tmp_path / "deterministic.yaml"
    deterministic_path.write_text(yaml.safe_dump(deterministic_payload, sort_keys=False), encoding="utf-8")
    deterministic_adapter = get_model_adapter(load_config(deterministic_path))
    deterministic_opened = deterministic_adapter.open_partition(2022, 6)
    try:
        deterministic_lead = deterministic_adapter.lead(deterministic_opened, 0)
        assert np.isclose(deterministic_lead.ensemble_mean_temperature.mean().compute().item(), 300.0)
        assert np.isclose(deterministic_lead.event_probabilities["hot_day_q95"].mean().compute().item(), 0.0)
        assert deterministic_lead.interval_quantiles is None
    finally:
        deterministic_adapter.close_partition(deterministic_opened)
