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
from heatextremes.verification.report_scorecard import (
    _relative_performance,
    hot_day_frequency_change,
    plot_scorecard,
    score_lead,
)


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


def test_report_plot_combines_frequency_map_and_absolute_metric_cells(tmp_path: Path) -> None:
    rows = []
    for model, label, adjustment in (("aifs_ens_v2", "AIFS ENS v2 mean", 0.0), ("graphcast_e2s", "GraphCast", 0.1)):
        for forecast_day in (0, 3):
            rows.append(
                {
                    "region": "nigeria",
                    "model": model,
                    "model_label": label,
                    "forecast_day": forecast_day,
                    "rmse_hot": 1.0 + adjustment,
                    "pod_deterministic": 0.5 - adjustment,
                    "far_deterministic": 0.2 + adjustment,
                    "brier_score_probabilistic": 0.1 + adjustment,
                }
            )
    frequency_change = xr.DataArray(
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        dims=("latitude", "longitude"),
        coords={"latitude": [4.0, 14.0], "longitude": [2.0, 15.0]},
    )
    path = tmp_path / "scorecard.png"

    plot_scorecard(
        pd.DataFrame(rows),
        path,
        frequency_change_maps={"nigeria": frequency_change},
        regions={"nigeria": Region("nigeria", 4.0, 14.0, 2.0, 15.0)},
        reference_model="aifs_ens_v2",
    )

    assert path.is_file()


def test_scorecard_colours_orient_all_metrics_as_performance_against_ifs() -> None:
    lower_values = pd.DataFrame(
        {0: [2.0, 1.0], 3: [4.0, 2.0]}, index=["ifs_ens", "graphcast_e2s"]
    )
    higher_values = pd.DataFrame(
        {0: [0.4, 0.6], 3: [0.2, 0.3]}, index=["ifs_ens", "graphcast_e2s"]
    )

    lower_is_better = _relative_performance(lower_values, "ifs_ens", higher_is_better=False)
    higher_is_better = _relative_performance(higher_values, "ifs_ens", higher_is_better=True)

    # For RMSE/FAR/Brier IFS/model - 1 makes GraphCast's lower raw values
    # positive (blue/better); for POD model/IFS - 1 has the same orientation.
    assert lower_is_better.loc["graphcast_e2s"].tolist() == pytest.approx([1.0, 1.0])
    assert higher_is_better.loc["graphcast_e2s"].tolist() == pytest.approx([0.5, 0.5])


def test_frequency_change_map_uses_observed_hot_day_rates() -> None:
    temperature = xr.DataArray(
        np.array([2.0, 1.0, 2.0, 2.0])[:, None, None],
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["1991-06-01", "1991-06-02", "2022-06-01", "2022-06-02"], dtype="datetime64[ns]"),
            "latitude": [10.0],
            "longitude": [5.0],
        },
    )
    threshold = xr.DataArray(
        np.array([[[1.5]], [[1.5]]]),
        dims=("dayofyear", "latitude", "longitude"),
        coords={"dayofyear": [152, 153], "latitude": [10.0], "longitude": [5.0]},
    )

    result = hot_day_frequency_change(
        temperature,
        threshold,
        Region("box", 5.0, 15.0, 0.0, 10.0),
        validation_years=[2022],
        climatology_years=[1991],
        months=[6],
    )

    # Validation hot-day frequency is 1.0; climatology is 0.5.
    assert result.item() == pytest.approx(100.0)
