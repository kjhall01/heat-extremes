"""Probability-of-exceedance Brier score metric."""

from __future__ import annotations

from numbers import Real

import numpy as np
import xarray as xr

from heatextremes.metrics._verification import (
    ENSEMBLE_MEMBER_DIM,
    match_observations,
    validate_forecast_and_observations,
    verification_time,
)


def probability_of_exceedance_brier_score(
    model_ensemble: xr.DataArray,
    observations: xr.DataArray,
    threshold: float | int | xr.DataArray,
) -> xr.DataArray:
    """Return the per-case Brier score for an exceedance event.

    The probability of exceedance is the fraction of valid ensemble members
    strictly greater than ``threshold``. The observed event is likewise
    defined as the matched observation being strictly greater than the
    threshold. Observations are matched at ``time + prediction_timedelta``.

    ``threshold`` may be a scalar or a DataArray. A DataArray threshold is
    aligned and broadcast against the model and observations using xarray's
    standard rules, allowing, for example, a spatially varying threshold.
    The ensemble-member ``number`` dimension is reduced in the returned
    per-case scores.
    """
    validate_forecast_and_observations(model_ensemble, observations)
    threshold = _as_threshold_dataarray(threshold)

    valid_time = verification_time(model_ensemble)
    observed_at_verification_time = match_observations(
        observations, valid_time, model_ensemble
    )
    result = brier_score_of_exceedance(
        model_ensemble,
        observed_at_verification_time,
        threshold,
    )
    return result.rename("probability_of_exceedance_brier_score").assign_coords(
        verification_time=valid_time
    )


def probability_of_exceedance(
    model_ensemble: xr.DataArray,
    threshold: float | int | xr.DataArray,
) -> xr.DataArray:
    """Return the member-count probability that a forecast exceeds a threshold."""
    if ENSEMBLE_MEMBER_DIM not in model_ensemble.dims:
        raise ValueError(f"model_ensemble must have a {ENSEMBLE_MEMBER_DIM!r} dimension")
    threshold = _as_threshold_dataarray(threshold)
    valid_members = model_ensemble.notnull() & threshold.notnull()
    return (
        (model_ensemble > threshold)
        .where(valid_members)
        .mean(ENSEMBLE_MEMBER_DIM, skipna=True)
        .rename("probability_of_exceedance")
        .assign_attrs(
            {
                "long_name": "ensemble member-count probability of exceedance",
                "event_definition": "value > threshold",
            }
        )
    )


def brier_score_of_exceedance(
    model_ensemble: xr.DataArray,
    observations: xr.DataArray,
    threshold: float | int | xr.DataArray,
) -> xr.DataArray:
    """Return Brier scores when forecast cases and observations are aligned.

    Unlike :func:`probability_of_exceedance_brier_score`, this function does
    not look observations up by valid time.  It is intended for daily or other
    pre-aggregated verification fields with matching coordinates.
    """
    threshold = _as_threshold_dataarray(threshold)
    probability = probability_of_exceedance(model_ensemble, threshold)
    observed_event = (observations > threshold).where(
        observations.notnull() & threshold.notnull()
    )
    result = (probability - observed_event) ** 2
    result = result.where(probability.notnull() & observed_event.notnull())
    model_dimensions = tuple(
        dimension for dimension in model_ensemble.dims if dimension != ENSEMBLE_MEMBER_DIM
    )
    extra_dimensions = tuple(
        dimension for dimension in result.dims if dimension not in model_dimensions
    )
    return result.transpose(*model_dimensions, *extra_dimensions).rename(
        "brier_score_of_exceedance"
    ).assign_attrs(
        {
            "long_name": "probability-of-exceedance Brier score",
            "description": (
                "squared error between the ensemble fraction strictly above "
                "the threshold and the observed exceedance event"
            ),
            "event_definition": "value > threshold",
        }
    )


def probability_of_exceedance_contingency(
    model_ensemble: xr.DataArray,
    observations: xr.DataArray,
    threshold: float | int | xr.DataArray,
    decision_threshold: float = 0.5,
) -> xr.DataArray:
    """Classify PoE forecasts into compact contingency-table outcomes.

    The forecast event occurs when the member-count probability of exceedance
    is greater than or equal to ``decision_threshold``.  Output codes are
    ``0`` for true negative, ``1`` for hit, ``2`` for miss, ``3`` for false
    positive, and ``-1`` for a missing forecast or observation.  Keeping one
    signed-byte categorical field is substantially smaller than storing four
    case-wise indicator arrays.
    """
    try:
        decision_threshold = float(decision_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("decision_threshold must be a number from 0 through 1") from error
    if not np.isfinite(decision_threshold) or not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be a finite number from 0 through 1")

    threshold = _as_threshold_dataarray(threshold)
    probability = probability_of_exceedance(model_ensemble, threshold)
    observed_event = (observations > threshold).where(
        observations.notnull() & threshold.notnull()
    )
    forecast_event = probability >= decision_threshold
    valid = probability.notnull() & observed_event.notnull()
    result = xr.where(
        observed_event,
        xr.where(forecast_event, 1, 2),
        xr.where(forecast_event, 3, 0),
    ).where(valid, -1).astype("int8")
    model_dimensions = tuple(
        dimension for dimension in model_ensemble.dims if dimension != ENSEMBLE_MEMBER_DIM
    )
    extra_dimensions = tuple(
        dimension for dimension in result.dims if dimension not in model_dimensions
    )
    return result.transpose(*model_dimensions, *extra_dimensions).rename(
        "probability_of_exceedance_contingency"
    ).assign_attrs(
        {
            "long_name": "probability-of-exceedance contingency outcome",
            "event_definition": "value > threshold",
            "forecast_event_definition": "probability_of_exceedance >= decision_threshold",
            "decision_threshold": decision_threshold,
            "flag_values": np.array([-1, 0, 1, 2, 3], dtype="int8"),
            "flag_meanings": "missing true_negative hit miss false_positive",
        }
    )


def _as_threshold_dataarray(threshold: float | int | xr.DataArray) -> xr.DataArray:
    """Return a scalar or spatial threshold as a DataArray."""
    if isinstance(threshold, xr.DataArray):
        return threshold
    if isinstance(threshold, Real):
        return xr.DataArray(threshold)
    raise TypeError("threshold must be a number or xarray.DataArray")
