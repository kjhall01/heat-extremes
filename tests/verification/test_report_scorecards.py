from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from heatextremes.verification.global_z500 import (
    aggregate_global_z500,
    global_z500_partition_statistics,
)
from heatextremes.verification.regions import Region
from heatextremes.verification.report_scorecard import score_lead


def test_report_scorecard_keeps_deterministic_and_probabilistic_metrics_distinct() -> None:
    initialization = np.array(["2022-06-01"], dtype="datetime64[ns]")
    forecast_day = [0]
    latitude = [0.0]
    longitude = [0.0, 10.0]
    coordinates = {
        "initialization": initialization,
        "forecast_day": forecast_day,
        "latitude": latitude,
        "longitude": longitude,
    }
    shape = (1, 1, 1, 2)
    dataset = xr.Dataset(
        {
            "forecast_temperature": (
                ("initialization", "forecast_day", "latitude", "longitude"), np.full(shape, 2.0)
            ),
            "observation_temperature": (
                ("initialization", "forecast_day", "latitude", "longitude"), np.full(shape, 1.0)
            ),
            "temperature_case_valid": (
                ("initialization", "forecast_day", "latitude", "longitude"), np.ones(shape, dtype=bool)
            ),
            "forecast_probability": (
                ("event", "initialization", "forecast_day", "latitude", "longitude"),
                np.array([[[[[0.8, 0.3]]]]]),
            ),
            "observed_event": (
                ("event", "initialization", "forecast_day", "latitude", "longitude"),
                np.array([[[[[1, 0]]]]]),
            ),
            "event_case_valid": (
                ("event", "initialization", "forecast_day", "latitude", "longitude"), np.ones((1, *shape), dtype=bool)
            ),
        },
        coords={
            **coordinates,
            "event": ["hot_day_q95"],
            "valid_date": (
                ("initialization", "forecast_day", "longitude"),
                np.array([[[np.datetime64("2022-06-01"), np.datetime64("2022-06-01")]]]),
            ),
        },
    )
    threshold = xr.DataArray(
        np.full((1, 1, 2), 1.5),
        dims=("dayofyear", "latitude", "longitude"),
        coords={"dayofyear": [152], "latitude": latitude, "longitude": longitude},
    )

    result = score_lead(dataset, 0, threshold, Region("global")).compute()

    assert result["rmse_all"].item() == pytest.approx(1.0)
    assert result["rmse_hot"].item() == pytest.approx(1.0)
    assert result["pod_deterministic"].item() == pytest.approx(1.0)
    assert result["far_deterministic"].item() == pytest.approx(0.5)
    assert result["brier_score_probabilistic"].item() == pytest.approx((0.2**2 + 0.3**2) / 2)


def test_global_z500_statistics_use_requested_instantaneous_lead_hours() -> None:
    time = np.array(["2022-06-01", "2022-06-02"], dtype="datetime64[ns]")
    lead_hours = [0, 72]
    forecast = xr.DataArray(
        np.full((2, 2, 1, 1), 2.0),
        dims=("time", "prediction_timedelta", "latitude", "longitude"),
        coords={
            "time": time,
            "prediction_timedelta": np.asarray(lead_hours) * np.timedelta64(1, "h"),
            "latitude": [0.0],
            "longitude": [0.0],
        },
    )
    era5 = xr.DataArray(
        np.ones((5, 1, 1)),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(
                ["2022-06-01", "2022-06-02", "2022-06-04", "2022-06-05", "2022-06-06"],
                dtype="datetime64[ns]",
            ),
            "latitude": [0.0],
            "longitude": [0.0],
        },
    )

    result = global_z500_partition_statistics(
        forecast, era5, forecast_days=[0, 3], lead_hours=lead_hours
    )

    assert result["forecast_day"].tolist() == [0, 3]
    assert result["lead_hours"].tolist() == lead_hours
    assert result["z500_squared_error_numerator"].tolist() == pytest.approx([2.0, 2.0])
    assert result["z500_weight_denominator"].tolist() == pytest.approx([2.0, 2.0])
    assert result["z500_cases"].tolist() == [2, 2]


def test_global_z500_aggregate_excludes_a_model_with_any_unavailable_partition(tmp_path: Path) -> None:
    output = tmp_path / "scorecard"
    graphcast = output / "partial" / "graphcast_e2s"
    ifs = output / "partial" / "ifs_ens"
    graphcast.mkdir(parents=True)
    ifs.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["graphcast_e2s"],
            "display_name": ["GraphCast"],
            "partition": ["2022-06"],
            "status": ["available"],
            "reason": [None],
            "forecast_day": [0],
            "lead_hours": [0],
            "z500_squared_error_numerator": [4.0],
            "z500_weight_denominator": [4.0],
            "z500_cases": [4],
        }
    ).to_csv(graphcast / "2022-06.csv", index=False)
    pd.DataFrame(
        {
            "model": ["ifs_ens"],
            "display_name": ["ECMWF IFS ENS"],
            "partition": ["2022-06"],
            "status": ["unavailable"],
            "reason": ["No Z500 variable"],
            "forecast_day": [0],
            "lead_hours": [0],
        }
    ).to_csv(ifs / "2022-06.csv", index=False)
    manifest = tmp_path / "inventory.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [{"model": "graphcast_e2s"}, {"model": "ifs_ens"}],
                "tasks": [
                    {"model": "graphcast_e2s", "year": 2022, "month": 6},
                    {"model": "ifs_ens", "year": 2022, "month": 6},
                ],
            }
        )
    )

    z500, combined = aggregate_global_z500(
        output, manifest_path=manifest, models=["graphcast_e2s", "ifs_ens"]
    )

    assert z500["model"].tolist() == ["graphcast_e2s"]
    assert z500["z500_rmse"].tolist() == pytest.approx([1.0])
    assert combined["model"].tolist() == ["graphcast_e2s"]
    unavailable = pd.read_csv(output / "global_z500_unavailable.csv")
    assert unavailable["model"].tolist() == ["ifs_ens"]
