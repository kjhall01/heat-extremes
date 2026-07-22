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
    probability_of_exceedance = _probability_of_exceedance(model_ensemble, threshold)
    observed_event = (observed_at_verification_time > threshold).where(
        observed_at_verification_time.notnull() & threshold.notnull()
    )

    score = (probability_of_exceedance - observed_event) ** 2
    valid_values = probability_of_exceedance.notnull() & observed_event.notnull()
    result = score.where(valid_values)

    model_dimensions = tuple(
        dimension for dimension in model_ensemble.dims if dimension != ENSEMBLE_MEMBER_DIM
    )
    extra_dimensions = tuple(
        dimension for dimension in result.dims if dimension not in model_dimensions
    )
    result = result.transpose(*model_dimensions, *extra_dimensions)
    return result.rename("probability_of_exceedance_brier_score").assign_coords(
        verification_time=valid_time
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


def _as_threshold_dataarray(threshold: float | int | xr.DataArray) -> xr.DataArray:
    """Return a scalar or spatial threshold as a DataArray."""
    if isinstance(threshold, xr.DataArray):
        return threshold
    if isinstance(threshold, Real):
        return xr.DataArray(threshold)
    raise TypeError("threshold must be a number or xarray.DataArray")


def _probability_of_exceedance(
    model_ensemble: xr.DataArray,
    threshold: xr.DataArray,
) -> xr.DataArray:
    """Calculate member-count probability while excluding missing members."""
    valid_members = model_ensemble.notnull() & threshold.notnull()
    return (model_ensemble > threshold).where(valid_members).mean(
        ENSEMBLE_MEMBER_DIM, skipna=True
    )
