"""Shared validation and observation matching for forecast verification metrics."""

from __future__ import annotations

import numpy as np
import xarray as xr


INITIALIZATION_TIME_DIM = "time"
LEAD_TIME_DIM = "prediction_timedelta"
ENSEMBLE_MEMBER_DIM = "number"


def validate_forecast_and_observations(
    model_ensemble: xr.DataArray,
    observations: xr.DataArray,
) -> None:
    """Validate the common forecast and observation structure."""
    for dimension in (
        INITIALIZATION_TIME_DIM,
        LEAD_TIME_DIM,
        ENSEMBLE_MEMBER_DIM,
    ):
        if dimension not in model_ensemble.dims:
            raise ValueError(f"model_ensemble must have a {dimension!r} dimension")

    for coordinate in (INITIALIZATION_TIME_DIM, LEAD_TIME_DIM):
        if coordinate not in model_ensemble.coords:
            raise ValueError(f"model_ensemble must have a {coordinate!r} coordinate")
        if model_ensemble[coordinate].dims != (coordinate,):
            raise ValueError(
                f"model_ensemble {coordinate!r} coordinate must be one-dimensional"
            )

    if INITIALIZATION_TIME_DIM not in observations.dims:
        raise ValueError("observations must have a 'time' dimension")
    if INITIALIZATION_TIME_DIM not in observations.coords:
        raise ValueError("observations must have a 'time' coordinate")
    if observations[INITIALIZATION_TIME_DIM].dims != (INITIALIZATION_TIME_DIM,):
        raise ValueError("observations 'time' coordinate must be one-dimensional")
    if not observations.indexes[INITIALIZATION_TIME_DIM].is_unique:
        raise ValueError("observations 'time' coordinate must contain unique timestamps")


def verification_time(model_ensemble: xr.DataArray) -> xr.DataArray:
    """Calculate the valid time for every initialization and lead time."""
    return model_ensemble[INITIALIZATION_TIME_DIM] + model_ensemble[LEAD_TIME_DIM]


def match_observations(
    observations: xr.DataArray,
    verification_times: xr.DataArray,
    model_ensemble: xr.DataArray,
) -> xr.DataArray:
    """Vectorize exact observation-time lookup, retaining missing timestamps."""
    observation_index = observations.indexes[INITIALIZATION_TIME_DIM]
    positions = observation_index.get_indexer(verification_times.values.ravel())
    positions = positions.reshape(verification_times.shape)

    indexer = xr.DataArray(positions, dims=verification_times.dims)
    found_observation = indexer >= 0
    safe_indexer = indexer.where(found_observation, 0).astype(np.intp)
    matched = observations.isel({INITIALIZATION_TIME_DIM: safe_indexer}).where(found_observation)

    # Vectorized indexing replaces the initialization-time coordinate with a
    # two-dimensional verification-time coordinate. Restore the forecast axes;
    # verification timestamps are attached to the returned metric instead.
    return matched.drop_vars(INITIALIZATION_TIME_DIM).assign_coords(
        {
            INITIALIZATION_TIME_DIM: model_ensemble[INITIALIZATION_TIME_DIM],
            LEAD_TIME_DIM: model_ensemble[LEAD_TIME_DIM],
        }
    )
