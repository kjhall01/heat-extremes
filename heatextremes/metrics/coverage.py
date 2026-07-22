"""Ensemble coverage metrics."""

from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.metrics._verification import (
    ENSEMBLE_MEMBER_DIM,
    INITIALIZATION_TIME_DIM,
    LEAD_TIME_DIM,
    match_observations,
    validate_forecast_and_observations,
    verification_time,
)


def coverage(
    model_ensemble: xr.DataArray,
    observations: xr.DataArray,
    percentile: float = 90.0,
) -> xr.DataArray:
    """Return whether observations fall within a central ensemble interval.

    ``model_ensemble`` is expected to be indexed by initialization ``time``,
    ``prediction_timedelta``, and ensemble-member ``number``.  The observation
    for each forecast is selected at ``time + prediction_timedelta``.  The
    returned values are 1.0 when that observation lies in the central
    ``percentile``% ensemble interval and 0.0 when it lies outside. Missing
    observations or undefined ensemble bounds are returned as ``NaN``.

    The ``number`` dimension is reduced; all other model dimensions are
    retained. Any additional dimensions on either input are aligned and
    broadcast according to xarray's normal rules.

    Parameters
    ----------
    model_ensemble
        Ensemble forecasts with ``time``, ``prediction_timedelta``, and
        ``number`` dimensions.
    observations
        Verification observations with a unique ``time`` coordinate.
    percentile
        Width of the central ensemble interval in percent. For example, 90
        evaluates whether observations are within the 5th--95th percentile
        range.

    Returns
    -------
    xr.DataArray
        A per-forecast coverage indicator, with a two-dimensional
        ``verification_time`` coordinate indexed by initialization time and
        lead time.
    """
    percentile = _validate_inputs(model_ensemble, observations, percentile)

    valid_time = verification_time(model_ensemble)
    observed_at_verification_time = match_observations(
        observations, valid_time, model_ensemble
    )

    lower_quantile = (100.0 - percentile) / 200.0
    upper_quantile = 1.0 - lower_quantile
    ensemble_for_quantiles = _single_member_chunk(model_ensemble)
    interval = ensemble_for_quantiles.quantile(
        [lower_quantile, upper_quantile], dim=ENSEMBLE_MEMBER_DIM
    )
    lower_bound = interval.isel(quantile=0, drop=True)
    upper_bound = interval.isel(quantile=1, drop=True)

    is_covered = (lower_bound <= observed_at_verification_time) & (
        observed_at_verification_time <= upper_bound
    )
    has_valid_values = (
        observed_at_verification_time.notnull()
        & lower_bound.notnull()
        & upper_bound.notnull()
    )
    result = is_covered.where(has_valid_values).astype(float)

    model_dimensions = tuple(
        dimension for dimension in model_ensemble.dims if dimension != ENSEMBLE_MEMBER_DIM
    )
    extra_dimensions = tuple(
        dimension for dimension in result.dims if dimension not in model_dimensions
    )
    result = result.transpose(*model_dimensions, *extra_dimensions)
    return result.rename("coverage").assign_coords(
        verification_time=valid_time
    ).assign_attrs(
        {
            "long_name": "central ensemble interval coverage indicator",
            "description": (
                "1.0 when the verification observation is within the central "
                "ensemble interval, 0.0 when it is outside"
            ),
            "central_percentile": percentile,
        }
    )


def _single_member_chunk(model_ensemble: xr.DataArray) -> xr.DataArray:
    """Make the reduction axis contiguous when forecasts use Dask chunks."""
    member_chunks = model_ensemble.chunksizes.get(ENSEMBLE_MEMBER_DIM, ())
    if len(member_chunks) > 1:
        return model_ensemble.chunk({ENSEMBLE_MEMBER_DIM: -1})
    return model_ensemble


def _validate_inputs(
    model_ensemble: xr.DataArray,
    observations: xr.DataArray,
    percentile: float,
) -> float:
    """Validate required dimensions, coordinates, and percentile."""
    try:
        percentile = float(percentile)
    except (TypeError, ValueError) as error:
        raise ValueError("percentile must be a number between 0 and 100") from error

    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be a finite number between 0 and 100")

    validate_forecast_and_observations(model_ensemble, observations)

    return percentile
