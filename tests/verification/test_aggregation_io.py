from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from heatextremes.verification.aggregation import aggregate_deterministic, aggregate_probability
from heatextremes.verification.io import (
    assert_safe_result_path,
    completion_is_valid,
    write_json_atomic,
    write_table_atomic,
)
from heatextremes.verification.models.base import ModelCapabilities
from heatextremes.verification.regions import Region
from heatextremes.verification.runner import _unavailable_interval_rows


def test_rmse_aggregation_uses_total_squared_error_not_monthly_mean() -> None:
    partial = pd.DataFrame(
        {
            "model": ["m", "m"],
            "region": ["global", "global"],
            "forecast_day": [0, 0],
            "subset": ["all", "all"],
            "metric": ["rmse", "rmse"],
            "numerator": [4.0, 36.0],
            "denominator": [1.0, 9.0],
            "weighted_support": [1.0, 9.0],
            "unweighted_support": [1, 9],
        }
    )
    result = aggregate_deterministic(partial)
    assert result["value"].item() == 2.0


def test_probability_aggregation_is_exact_for_brier_score() -> None:
    partial = pd.DataFrame(
        {
            "model": ["m", "m"],
            "region": ["global", "global"],
            "forecast_day": [0, 0],
            "event": ["hot_day_q95", "hot_day_q95"],
            "metric": ["brier_score", "brier_score"],
            "numerator": [1.0, 3.0],
            "denominator": [2.0, 6.0],
            "weighted_support": [2.0, 6.0],
            "unweighted_support": [2, 6],
            "event_weighted_support": [1.0, 1.0],
            "non_event_weighted_support": [1.0, 5.0],
        }
    )
    assert aggregate_probability(partial)["value"].item() == 0.5


def test_result_path_guard_refuses_root_and_outside_paths(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    with pytest.raises(ValueError):
        assert_safe_result_path(result_root, result_root)
    with pytest.raises(ValueError):
        assert_safe_result_path(tmp_path / "other", result_root)
    assert assert_safe_result_path(result_root / "model" / "partial", result_root)[0].name == "partial"


def test_completion_marker_requires_every_committed_file(tmp_path: Path) -> None:
    table = tmp_path / "deterministic.csv"
    write_table_atomic(pd.DataFrame({"value": [1.0]}), table)
    write_json_atomic(
        {"status": "complete", "expected_output_files": [table.name]}, tmp_path / "completion.json"
    )
    assert completion_is_valid(tmp_path)
    table.unlink()
    assert not completion_is_valid(tmp_path)


def test_deterministic_models_explicitly_lack_ensemble_interval_capability() -> None:
    capabilities = ModelCapabilities(ensemble=False, member_temperature=False, interval_quantiles=False)
    assert not capabilities.ensemble
    assert not capabilities.interval_quantiles
    rows = _unavailable_interval_rows(
        model="deterministic",
        partition="2022-06",
        forecast_day=0,
        regions={"global": Region("global")},
        levels=[0.9],
        reason="deterministic model does not provide ensemble intervals",
    )
    assert set(rows["status"]) == {"unavailable"}
    assert all("does not provide" in value for value in rows["reason"])
