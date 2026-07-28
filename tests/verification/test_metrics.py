from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.verification.deterministic import deterministic_temperature_statistics
from heatextremes.verification.interval_coverage import interval_coverage_statistics
from heatextremes.verification.probabilistic import (
    probability_decision_statistics,
    probability_event_statistics,
)
from heatextremes.verification.reliability import probability_reliability_statistics
from heatextremes.verification.weighting import weighted_sufficient_statistics


def _field(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("initialization", "latitude", "longitude"),
        coords={
            "initialization": np.array(["2024-06-01", "2024-06-02"], dtype="datetime64[ns]"),
            "latitude": [0.0, 60.0],
            "longitude": [0.0],
        },
    )


def test_cosine_latitude_weighted_mean_and_support() -> None:
    field = _field(np.array([[[2.0], [4.0]], [[2.0], [4.0]]]))
    statistic = weighted_sufficient_statistics(field).compute()
    # Each latitude occurs twice; cos(0)=1 and cos(60)=0.5.
    assert statistic["numerator"].item() == 8.0
    assert statistic["denominator"].item() == 3.0
    assert statistic["unweighted_support"].item() == 4


def test_hot_conditioning_uses_observed_mask_and_not_forecast_classification() -> None:
    forecast = _field(np.array([[[12.0], [0.0]], [[10.0], [0.0]]]))
    observed = _field(np.zeros((2, 2, 1)))
    observed_hot = _field(np.array([[[True], [False]], [[True], [False]]]))
    statistic = deterministic_temperature_statistics(forecast, observed, observed_hot).compute()
    hot_rmse = statistic.sel(metric="rmse", subset="observed_hot")
    nonhot_rmse = statistic.sel(metric="rmse", subset="observed_nonhot")
    assert hot_rmse["numerator"].item() == 244.0
    assert hot_rmse["denominator"].item() == 2.0
    assert nonhot_rmse["numerator"].item() == 0.0
    assert np.isclose(nonhot_rmse["denominator"].item(), 1.0)


def test_empty_conditional_subset_has_zero_support_and_nan_value_after_division() -> None:
    forecast = _field(np.ones((2, 2, 1)))
    observed = _field(np.zeros((2, 2, 1)))
    hot = _field(np.zeros((2, 2, 1), dtype=bool))
    result = deterministic_temperature_statistics(forecast, observed, hot).compute()
    selected = result.sel(metric="mae", subset="observed_hot")
    assert selected["denominator"].item() == 0.0
    assert np.isnan((selected["numerator"] / selected["denominator"]).item())


def test_brier_and_probability_frequency_sufficient_statistics() -> None:
    probability = _field(np.array([[[0.0], [1.0]], [[1.0], [0.0]]]))
    event = _field(np.array([[[0], [1]], [[0], [0]]], dtype=bool))
    result = probability_event_statistics(probability, event).compute()
    brier = result.sel(metric="brier_score")
    frequency = result.sel(metric="observed_event_frequency")
    # Errors are 0, 0, 1, 0; latitude weights are 1, .5, 1, .5.
    assert brier["numerator"].item() == 1.0
    assert brier["denominator"].item() == 3.0
    assert np.isclose(frequency["numerator"].item(), 0.5)


def test_probability_decision_statistics_store_exact_pod_and_far_totals() -> None:
    probability = _field(np.array([[[0.0], [1.0]], [[1.0], [0.0]]]))
    event = _field(np.array([[[0], [1]], [[0], [0]]], dtype=bool))
    result = probability_decision_statistics(probability, event, [0.5]).compute()
    pod = result.sel(metric="pod", decision_threshold=0.5)
    far = result.sel(metric="far", decision_threshold=0.5)
    # Weighted hits=.5, misses=0, and false alarms=1.
    assert np.isclose(pod["numerator"].item(), 0.5)
    assert np.isclose(pod["denominator"].item(), 0.5)
    assert np.isclose(far["numerator"].item(), 1.0)
    assert np.isclose(far["denominator"].item(), 1.5)


def test_reliability_bins_store_additive_sums() -> None:
    probability = _field(np.array([[[0.0], [0.2]], [[0.8], [1.0]]]))
    event = _field(np.array([[[0], [1]], [[1], [0]]], dtype=bool))
    result = probability_reliability_statistics(probability, event, [0.0, 0.5, 1.0]).compute()
    first = result.sel(bin=0)
    last = result.sel(bin=1)
    assert first["unweighted_count"].item() == 2
    assert first["weighted_count"].item() == 1.5
    assert np.isclose(first["weighted_probability_sum"].item(), 0.1)
    assert last["unweighted_count"].item() == 2  # Includes p == 1 in final bin.
    assert last["weighted_observation_sum"].item() == 1.0


def test_interval_coverage_is_inclusive_and_reports_width() -> None:
    lower = _field(np.array([[[0.0], [0.0]], [[0.0], [0.0]]]))
    upper = _field(np.array([[[1.0], [1.0]], [[1.0], [1.0]]]))
    observation = _field(np.array([[[0.0], [1.0]], [[1.1], [0.5]]]))
    hot = _field(np.array([[[True], [True]], [[False], [False]]]))
    result = interval_coverage_statistics(
        lower, upper, observation, hot, nominal_coverage=0.9
    ).compute()
    all_cases = result.sel(subset="all", nominal_coverage=0.9)
    assert all_cases["numerator"].item() == 2.0
    assert all_cases["denominator"].item() == 3.0
    assert all_cases["unweighted_numerator"].item() == 3.0
    assert all_cases["width_numerator"].item() == 3.0
