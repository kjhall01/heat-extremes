from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT_DIRECTORY = Path(__file__).parents[1] / "scripts" / "t2m_verification"
PRECIPITATION_SCRIPT_DIRECTORY = (
    Path(__file__).parents[1] / "scripts" / "precipitation_verification"
)


def _load_script(name: str, directory: Path = SCRIPT_DIRECTORY):
    path = directory / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _daily_temperature_datasets() -> tuple[xr.Dataset, xr.Dataset]:
    dimensions = ("time", "forecast_day", "number", "latitude", "longitude")
    coordinates = {
        "time": np.array(["2024-01-01"], dtype="datetime64[ns]"),
        "forecast_day": [1],
        "number": [0, 1, 2, 3],
        "latitude": [45.0],
        "longitude": [-75.0],
    }
    ensemble_values = np.array([0.0, 10.0, 20.0, 30.0])[None, None, :, None, None]
    model = xr.Dataset(
        {
            name: (dimensions, ensemble_values)
            for name in ("t2m_min_6h", "t2m_mean_6h", "t2m_max_6h")
        },
        coords=coordinates,
    )
    observations = xr.Dataset(
        {
            name: (
                ("time", "forecast_day", "latitude", "longitude"),
                np.array(25.0)[None, None, None, None],
            )
            for name in model.data_vars
        },
        coords={key: value for key, value in coordinates.items() if key != "number"},
    )
    return model, observations


def test_calculate_t2m_metrics_writes_each_aggregation_and_metric() -> None:
    script = _load_script("calculate_t2m_metrics.py")
    model, observations = _daily_temperature_datasets()

    result = script._calculate_metrics(
        model,
        observations,
        script.TEMPERATURE_AGGREGATIONS,
        coverage_percentiles=[90.0],
        thresholds_celsius=[35.0],
        poe_decision_thresholds=[0.5],
        temperature_unit="C",
    )

    assert set(result.data_vars) == {
        f"{metric}_{aggregation}"
        for metric in (
            "coverage_p90",
            "poe_35c",
            "brier_score_35c",
            "poe_contingency_p0p5_35c",
        )
        for aggregation in script.TEMPERATURE_AGGREGATIONS
    }
    assert result["coverage_p90_t2m_min_6h"].item() == 1.0
    assert result["poe_35c_t2m_mean_6h"].item() == 0.0
    assert result["brier_score_35c_t2m_max_6h"].item() == 0.0
    assert result["poe_contingency_p0p5_35c_t2m_max_6h"].item() == 0


def test_summary_keeps_maps_counts_and_forecast_day_global_means() -> None:
    script = _load_script("aggregate_and_plot_t2m_metrics.py")
    metrics = xr.Dataset(
        {
            "coverage_p90_t2m_max_6h": (
                ("time", "forecast_day", "latitude", "longitude"),
                np.array([0.0, 1.0])[:, None, None, None],
            )
        },
        coords={
            "time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "forecast_day": [1],
            "latitude": [45.0],
            "longitude": [-75.0],
        },
    )

    summary = script._summarize(metrics)

    assert summary["coverage_p90_t2m_max_6h_map_mean"].item() == 0.5
    assert summary["coverage_p90_t2m_max_6h_map_valid_cases"].item() == 2
    assert summary["coverage_p90_t2m_max_6h_global_mean"].item() == 0.5


def test_summary_calculates_conditional_contingency_rates() -> None:
    script = _load_script("aggregate_and_plot_t2m_metrics.py")
    contingency = xr.DataArray(
        np.array([1, 2, 3, 0], dtype="int8")[:, None, None, None],
        dims=("time", "forecast_day", "latitude", "longitude"),
        coords={
            "time": np.arange(
                np.datetime64("2024-01-01"),
                np.datetime64("2024-01-05"),
                np.timedelta64(1, "D"),
            ),
            "forecast_day": [1],
            "latitude": [45.0],
            "longitude": [-75.0],
        },
    )

    summary = script._summarize(
        contingency.to_dataset(name="poe_contingency_p0p5_35c_t2m_max_6h")
    )

    assert summary["poe_hit_rate_p0p5_35c_t2m_max_6h_global_mean"].item() == 0.5
    assert summary["poe_miss_rate_p0p5_35c_t2m_max_6h_global_mean"].item() == 0.5
    assert summary["poe_false_positive_rate_p0p5_35c_t2m_max_6h_global_mean"].item() == 0.5


def test_precipitation_metric_script_converts_millimeter_thresholds() -> None:
    script = _load_script(
        "calculate_precipitation_metrics.py",
        PRECIPITATION_SCRIPT_DIRECTORY,
    )
    model, observations = _daily_temperature_datasets()
    model = model[["t2m_max_6h"]].rename({"t2m_max_6h": "total_precipitation"})
    observations = observations[["t2m_max_6h"]].rename(
        {"t2m_max_6h": "total_precipitation"}
    )

    result = script._calculate_metrics(
        model.total_precipitation,
        observations.total_precipitation,
        coverage_percentiles=[90.0],
        thresholds_millimeters=[20.0],
        poe_decision_thresholds=[0.5],
        precipitation_unit="mm",
    )

    assert result["coverage_p90_total_precipitation"].item() == 1.0
    assert result["poe_20mm_total_precipitation"].item() == 0.25
    assert result["brier_score_20mm_total_precipitation"].item() == 0.5625
    assert result["poe_contingency_p0p5_20mm_total_precipitation"].item() == 2


def test_metric_script_appends_time_batches_to_a_completed_zarr_store(
    tmp_path: Path, monkeypatch
) -> None:
    script = _load_script("calculate_t2m_metrics.py")
    model, observations = _daily_temperature_datasets()
    second_model = model.assign_coords(time=np.array(["2024-01-02"], dtype="datetime64[ns]"))
    second_observations = observations.assign_coords(
        time=np.array(["2024-01-02"], dtype="datetime64[ns]")
    )
    model = xr.concat([model, second_model], dim="time")
    observations = xr.concat([observations, second_observations], dim="time")
    model_store = tmp_path / "model.zarr"
    observation_store = tmp_path / "observations.zarr"
    output_store = tmp_path / "metrics.zarr"
    model.to_zarr(model_store, consolidated=True)
    observations.to_zarr(observation_store, consolidated=True)
    (model_store / "_SUCCESS").touch()
    (observation_store / "_SUCCESS").touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calculate_t2m_metrics.py",
            "--model-store",
            str(model_store),
            "--observation-store",
            str(observation_store),
            "--output",
            str(output_store),
            "--temperature-unit",
            "C",
            "--time-batch-size",
            "1",
            "--workers",
            "1",
        ],
    )

    script.main()

    result = xr.open_zarr(output_store, consolidated=True)
    assert result.sizes["time"] == 2
    assert (output_store / "_SUCCESS").exists()
