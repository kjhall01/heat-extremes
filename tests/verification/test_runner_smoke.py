from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from heatextremes.verification.aggregation import aggregate_result_directory
from heatextremes.verification.config import load_config
from heatextremes.verification.io import completion_is_valid, find_table, read_table
from heatextremes.verification.plotting import make_all_plots
from heatextremes.verification.runner import compute_partition


def _write_synthetic_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    latitude = [0.0, 60.0]
    longitude = [0.0, 10.0]
    initialization = np.array(["2022-06-01", "2022-06-02"], dtype="datetime64[ns]")
    valid_dates = np.array(
        [
            [["2022-06-01", "2022-06-02"]],
            [["2022-06-02", "2022-06-03"]],
        ],
        dtype="datetime64[ns]",
    )
    dimensions = ("time", "forecast_day", "latitude", "longitude")
    coordinates = {
        "time": initialization,
        "forecast_day": [0],
        "latitude": latitude,
        "longitude": longitude,
        "valid_date": (("time", "forecast_day", "longitude"), valid_dates),
    }
    compact = xr.Dataset(
        {
            "t2m_daily_mean_ensemble_mean": (dimensions, np.full((2, 1, 2, 2), 2.0)),
            "hot_day_q95_probability": (dimensions, np.full((2, 1, 2, 2), 0.25)),
            "heatwave_start_q95_2d_probability": (dimensions, np.full((2, 1, 2, 2), 0.5)),
            "heatwave_start_q95_3d_probability": (dimensions, np.full((2, 1, 2, 2), 0.75)),
        },
        coords=coordinates,
    )
    compact_path = tmp_path / "monthly" / "2022" / "aifs_ens_v2_heat_202206.zarr"
    compact_path.parent.mkdir(parents=True)
    compact.to_zarr(compact_path, consolidated=False)

    time = np.array(["2022-06-01", "2022-06-02", "2022-06-03"], dtype="datetime64[ns]")
    observations = xr.Dataset(
        {"t2m_daily_mean": (("time", "latitude", "longitude"), np.ones((3, 2, 2)))},
        coords={"time": time, "latitude": latitude, "longitude": longitude},
    )
    daily_path = tmp_path / "daily.zarr"
    observations.to_zarr(daily_path, consolidated=True)
    hazards = xr.Dataset(
        {
            "hot_day_q95": (("time", "latitude", "longitude"), np.zeros((3, 2, 2), dtype=bool)),
            "heatwave_start_q95_2d": (("time", "latitude", "longitude"), np.zeros((3, 2, 2), dtype=bool)),
            "heatwave_start_q95_3d": (("time", "latitude", "longitude"), np.zeros((3, 2, 2), dtype=bool)),
        },
        coords={"time": time, "latitude": latitude, "longitude": longitude},
    )
    hazard_path = tmp_path / "hazards.zarr"
    hazards.to_zarr(hazard_path, consolidated=True)
    return compact_path, daily_path, hazard_path


def test_small_partition_resumes_and_aggregates_without_raw_member_cube(tmp_path: Path) -> None:
    compact_path, daily_path, hazard_path = _write_synthetic_sources(tmp_path)
    regions_path = tmp_path / "regions.yaml"
    regions_path.write_text("regions:\n  global: {}\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "synthetic_aifs", "adapter": "aifs_ens_v2", "ensemble": True},
                "paths": {
                    "compact_monthly_store_pattern": str(compact_path).replace("202206", "{year}{month:02d}"),
                    "era5_daily_temperature_store": str(daily_path),
                    "era5_hazard_store": str(hazard_path),
                    "verification_results_root": str(tmp_path / "results"),
                    "interval_quantile_file_pattern": None,
                },
                "variables": {
                    "ensemble_mean_temperature": "t2m_daily_mean_ensemble_mean",
                    "event_probabilities": {
                        "hot_day_q95": "hot_day_q95_probability",
                        "heatwave_start_q95_2d": "heatwave_start_q95_2d_probability",
                        "heatwave_start_q95_3d": "heatwave_start_q95_3d_probability",
                    },
                },
                "observations": {
                    "daily_temperature_variable": "t2m_daily_mean",
                    "event_variables": {
                        "hot_day_q95": "hot_day_q95",
                        "heatwave_start_q95_2d": "heatwave_start_q95_2d",
                        "heatwave_start_q95_3d": "heatwave_start_q95_3d",
                    },
                },
                "selection": {"years": [2022], "months": [6], "forecast_days": [0], "map_forecast_days": [0]},
                "chunking": {"compact": {"time": 1}, "observations": {"time": 1}},
                "metrics": {"probability_bins": [0.0, 0.5, 1.0], "interval_levels": [0.5]},
                "regions": {"file": str(regions_path)},
                "output": {"table_format": "csv"},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    first = compute_partition(config, 2022, 6, repository_root=tmp_path)
    assert not first.skipped
    assert completion_is_valid(first.partition_directory)
    assert (first.partition_directory / "maps.nc").is_file()
    interval = read_table(find_table(first.partition_directory, "interval_coverage"))
    assert set(interval["status"]) == {"unavailable"}

    # Simulate interruption after data were committed but before the completion marker.
    (first.partition_directory / "completion.json").unlink()
    resumed = compute_partition(config, 2022, 6, repository_root=tmp_path)
    assert not resumed.skipped
    assert completion_is_valid(first.partition_directory)

    aggregate_directory, discovery = aggregate_result_directory(config)
    assert discovery.missing == ()
    deterministic = read_table(find_table(aggregate_directory, "deterministic_by_lead_region"))
    assert np.isclose(deterministic[deterministic.metric.eq("bias")]["value"].iloc[0], 1.0)
    figure_directory = config.model_result_dir / "figures"
    make_all_plots([config.model_result_dir], figure_directory, reliability_forecast_days=[0])
    assert (figure_directory / "rmse_by_lead_global.png").is_file()
