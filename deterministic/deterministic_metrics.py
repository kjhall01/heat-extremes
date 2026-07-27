"""Verification metrics for deterministic (single-valued) heat-extreme forecasts.

Self-contained: does not import the ``heatextremes`` package. It complements
the ensemble/probabilistic metrics in ``heatextremes.metrics`` (``coverage``,
``probability_of_exceedance_brier_score``) for the case where the forecast is
a single deterministic value per grid-cell / day / lead time rather than an
ensemble, and its verification-time matching (``verification_time``,
``match_observations``) and batching utility (``mean_in_time_batches``) below
are copied from ``heatextremes.metrics._verification``/``heatextremes.metrics.batches``,
unchanged, rather than imported -- ``heatextremes`` is a shared, read-only
dependency here, so this module owns its own copy of the small amount of
logic it needs instead of depending on that package at all.

Two families of metrics are provided:

* A continuous-error metric, ``rmse``, following the point-forecast
  verification framework of Gneiting (2011), "Making and Evaluating Point
  Forecasts" (https://www.bundesbank.de/resource/blob/635562/...).
* Threshold-exceedance ("extreme event") metrics built on a 2x2 contingency
  table -- hits, misses, false alarms, correct negatives -- namely the
  probability of detection (POD), false-alarm ratio (FAR), probability of
  false detection (POFD), and the Symmetric Extremal Dependence Index
  (SEDI) of Ferro and Stephenson (2011), "Deterministic Forecasts of
  Extreme Events and Their Verification"
  (https://journals.ametsoc.org/view/journals/wefo/26/5/waf-d-10-05030_1.xml).

Calling convention, matched to ``heatextremes.metrics.coverage`` /
``probability_of_exceedance_brier_score``
------------------------------------------------------------------------
``forecast`` is indexed by initialization ``time`` and ``prediction_timedelta``
(and any spatial dims); ``observations`` is indexed by a plain, unique
``time`` coordinate (e.g. a daily ERA5 series) -- *not* pre-matched to the
forecast's initialization/lead-time grid. Every function here uses
``verification_time`` and ``match_observations`` (below, copied from
``heatextremes.metrics._verification``) internally to look up, for every
(initialization, lead time) pair, the observation valid at
``time + prediction_timedelta``, exactly like ``coverage()`` and
``probability_of_exceedance_brier_score()`` do. Passing already-matched,
identically-shaped arrays as ``forecast``/``observations`` is a bug: do not
do that.

``dim`` arguments (a dimension name or sequence of dimension names) specify
which axis/axes to reduce over, mirroring xarray's own ``.mean(dim=...)``
convention. Pass ``dim=None`` (the default) to reduce over every dimension
and return a scalar; pass e.g. ``dim="time"`` to reduce only over
initializations, keeping ``prediction_timedelta``, ``latitude``, and
``longitude`` intact.

NaNs (missing forecast or observation values) are excluded pairwise: if
either the forecast or the matched observation is NaN at a point, that
point does not contribute to any of the four contingency-table categories,
nor to the RMSE.

For large forecast archives that do not fit in memory as a single array
(e.g. a full multi-decade reforecast), ``squared_error`` and
``extreme_indicators`` below are unreduced, elementwise building blocks
meant to be passed through a batching/running-mean utility such as
``mean_in_time_batches`` (below, copied from ``heatextremes.metrics.batches``
-- the same tool used for the ensemble coverage/Brier-score notebook),
confirmed to compute a genuine NaN-aware running mean (sum / count per
batch) -- and then combined into
RMSE/POD/FAR/POFD/SEDI afterward from the batch-reduced means. Because the
four contingency indicators are 0/1 per point, a *mean* of them over some
dimension is algebraically identical (up to the total count cancelling out
of every ratio) to computing ``contingency_counts`` and dividing; the rate
functions (``probability_of_detection``, ``false_alarm_ratio``,
``probability_of_false_detection``, ``symmetric_extremal_dependence_index``)
work on either.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from numbers import Real
from typing import Hashable, Iterable, Union

import numpy as np
import xarray as xr

DimsLike = Union[Hashable, Iterable[Hashable], None]

# --- Verification-time matching (self-contained; no heatextremes dependency) ---
#
# Copied from heatextremes.metrics._verification, not imported from it: the
# heatextremes package is a shared/read-only dependency that can't be modified
# here, so this module has its own copy of the small amount of matching logic
# it needs, rather than depending on that package at all. Logic is unchanged
# from the original -- only the import has been removed.

INITIALIZATION_TIME_DIM = "time"
LEAD_TIME_DIM = "prediction_timedelta"


def verification_time(forecast: xr.DataArray) -> xr.DataArray:
    """Calculate the valid time for every initialization and lead time."""
    return forecast[INITIALIZATION_TIME_DIM] + forecast[LEAD_TIME_DIM]


def match_observations(
    observations: xr.DataArray,
    verification_times: xr.DataArray,
    forecast: xr.DataArray,
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
            INITIALIZATION_TIME_DIM: forecast[INITIALIZATION_TIME_DIM],
            LEAD_TIME_DIM: forecast[LEAD_TIME_DIM],
        }
    )


# --- Batching utility (self-contained; no heatextremes dependency) ---
#
# Copied from heatextremes.metrics.batches.mean_in_time_batches, unchanged,
# for the same reason as above: needed for large archives that don't fit in
# memory as a single array (see module docstring), without depending on the
# heatextremes package to get it.


def mean_in_time_batches(
    data: xr.Dataset | xr.DataArray,
    calculate: Callable[[xr.Dataset | xr.DataArray], xr.Dataset | xr.DataArray],
    reductions: Mapping[str, tuple[str, ...]],
    batch_size: int = 1,
    time_dim: str = "time",
) -> dict[str, xr.Dataset]:
    """Calculate NaN-aware means without retaining full metric arrays.

    ``calculate`` receives one time batch and returns one or more metric data
    variables. Each requested reduction is summed and counted immediately;
    only those small partial statistics are brought into memory.
    """
    if time_dim not in data.dims:
        raise ValueError(f"data must have a {time_dim!r} dimension")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not reductions:
        raise ValueError("reductions must not be empty")

    totals: dict[str, xr.Dataset] = {}
    counts: dict[str, xr.Dataset] = {}
    for start in range(0, data.sizes[time_dim], batch_size):
        result = calculate(data.isel({time_dim: slice(start, start + batch_size)}))
        if isinstance(result, xr.DataArray):
            result = result.to_dataset(name=result.name or "value")

        statistics = []
        for name, dimensions in reductions.items():
            statistics.extend(
                (
                    result.sum(dimensions, skipna=True).rename(
                        {variable: f"{name}_sum_{variable}" for variable in result.data_vars}
                    ),
                    result.count(dimensions).rename(
                        {variable: f"{name}_count_{variable}" for variable in result.data_vars}
                    ),
                )
            )
        statistics = xr.merge(statistics).compute()

        for name in reductions:
            sum_names = {f"{name}_sum_{variable}": variable for variable in result.data_vars}
            count_names = {
                f"{name}_count_{variable}": variable for variable in result.data_vars
            }
            batch_total = statistics[list(sum_names)].rename(sum_names)
            batch_count = statistics[list(count_names)].rename(count_names)
            if name not in totals:
                totals[name] = batch_total
                counts[name] = batch_count
                continue
            totals[name], batch_total = xr.align(totals[name], batch_total, join="exact")
            counts[name], batch_count = xr.align(counts[name], batch_count, join="exact")
            totals[name] = totals[name] + batch_total
            counts[name] = counts[name] + batch_count

    return {name: totals[name] / counts[name] for name in reductions}


def validate_deterministic_forecast_and_observations(
    forecast: xr.DataArray,
    observations: xr.DataArray,
) -> None:
    """Validate forecast/observation structure for a deterministic (non-ensemble) forecast.

    Same checks as ``heatextremes.metrics._verification.validate_forecast_and_observations``,
    minus the ensemble-member (``number``) dimension requirement, since a
    deterministic forecast has no ensemble-member axis.
    """
    for dimension in (INITIALIZATION_TIME_DIM, LEAD_TIME_DIM):
        if dimension not in forecast.dims:
            raise ValueError(f"forecast must have a {dimension!r} dimension")

    for coordinate in (INITIALIZATION_TIME_DIM, LEAD_TIME_DIM):
        if coordinate not in forecast.coords:
            raise ValueError(f"forecast must have a {coordinate!r} coordinate")
        if forecast[coordinate].dims != (coordinate,):
            raise ValueError(f"forecast {coordinate!r} coordinate must be one-dimensional")

    if INITIALIZATION_TIME_DIM not in observations.dims:
        raise ValueError(f"observations must have a {INITIALIZATION_TIME_DIM!r} dimension")
    if INITIALIZATION_TIME_DIM not in observations.coords:
        raise ValueError(f"observations must have a {INITIALIZATION_TIME_DIM!r} coordinate")
    if observations[INITIALIZATION_TIME_DIM].dims != (INITIALIZATION_TIME_DIM,):
        raise ValueError(
            f"observations {INITIALIZATION_TIME_DIM!r} coordinate must be one-dimensional"
        )
    if not observations.indexes[INITIALIZATION_TIME_DIM].is_unique:
        raise ValueError(
            f"observations {INITIALIZATION_TIME_DIM!r} coordinate must contain unique timestamps"
        )


def _as_threshold_dataarray(threshold: float | int | xr.DataArray) -> xr.DataArray:
    """Return a scalar or spatially/temporally varying threshold as a DataArray.

    Copied from the pattern in ``heatextremes.metrics.poe_brier_score``: a
    DataArray threshold (e.g. a per-grid-cell, per-day-of-year climatological
    percentile, already looked up at each case's verification time -- see
    ``climatology.threshold_at_verification_time``) is aligned/broadcast
    against forecast and observations using xarray's standard rules.
    """
    if isinstance(threshold, xr.DataArray):
        return threshold
    if isinstance(threshold, Real):
        return xr.DataArray(threshold)
    raise TypeError("threshold must be a number or xarray.DataArray")


def match_forecast_to_observations(
    forecast: xr.DataArray,
    observations: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return (verification_time, matched_observations) for this forecast.

    Public, not a private helper: this is the exact matching step
    `squared_error`/`extreme_indicators` do internally before comparing
    anything, exposed so it can be called and inspected on its own -- e.g.
    to look at what `observations` value each forecast case actually got
    matched against, before trusting a downstream RMSE/hit/miss/etc.
    Validates forecast/observations structure first (see
    `validate_deterministic_forecast_and_observations`).
    """
    validate_deterministic_forecast_and_observations(forecast, observations)
    valid_time = verification_time(forecast)
    observed_at_verification_time = match_observations(observations, valid_time, forecast)
    return valid_time, observed_at_verification_time


def squared_error(forecast: xr.DataArray, observations: xr.DataArray) -> xr.DataArray:
    """Per-case squared error ``(forecast - observed) ** 2``, unreduced.

    ``observations`` is matched to each (initialization time, lead time)
    case at its verification time (``time + prediction_timedelta``) first --
    see the module docstring. Building block for :func:`rmse`. Average this
    over the desired dimension(s) -- directly, or via a batching utility
    such as ``heatextremes.metrics.mean_in_time_batches`` -- and pass the
    result to :func:`rmse_from_mean_squared_error` to finish computing RMSE.
    (Do not average ``sqrt(squared_error)`` batch-by-batch: that computes
    MAE, not RMSE, since square root does not commute with averaging.)
    """
    valid_time, observed = match_forecast_to_observations(forecast, observations)
    error = forecast - observed
    result = (error**2).where(forecast.notnull() & observed.notnull())
    return result.rename("squared_error").assign_coords(verification_time=valid_time)


def rmse_from_mean_squared_error(mean_squared_error: xr.DataArray) -> xr.DataArray:
    """RMSE from an already-averaged squared error (see :func:`squared_error`)."""
    return np.sqrt(mean_squared_error)


def rmse(
    forecast: xr.DataArray,
    observations: xr.DataArray,
    dim: DimsLike = None,
) -> xr.DataArray:
    """Root-mean-square error between a deterministic forecast and observations.

    RMSE = sqrt( mean( (forecast - matched_observation) ** 2 ) )

    Reference: Gneiting, T. (2011), "Making and Evaluating Point Forecasts."

    Parameters
    ----------
    forecast : xr.DataArray
        Dims include initialization ``time`` and ``prediction_timedelta``
        (plus e.g. ``latitude``, ``longitude``).
    observations : xr.DataArray
        Dim/coordinate: a plain, unique ``time`` (e.g. daily observations) --
        matched internally to each forecast case's verification time.
    dim : str or sequence of str, optional
        Dimension(s) to average over. Defaults to all dimensions
        (returns a scalar). Pass e.g. ``"time"`` to keep lead time and
        space, or ``("time", "latitude", "longitude")`` to keep only
        lead time.

    Returns
    -------
    xr.DataArray
    """
    return rmse_from_mean_squared_error(
        squared_error(forecast, observations).mean(dim=dim, skipna=True)
    )


def extreme_indicators(
    forecast: xr.DataArray,
    observations: xr.DataArray,
    threshold: float | xr.DataArray,
) -> xr.Dataset:
    """Per-case 0/1 hit/miss/false-alarm/correct-negative indicators, unreduced.

    ``observations`` is matched to each (initialization time, lead time)
    case at its verification time first -- see the module docstring.
    ``threshold`` may be a scalar (e.g. a fixed absolute threshold) or an
    already-broadcastable DataArray (e.g. a spatially-varying or
    per-case climatological threshold; see
    ``climatology.threshold_at_verification_time`` for the latter).

    Building block for :func:`contingency_counts`. Points where the
    forecast, the matched observation, or the threshold is NaN are NaN in
    every indicator (so they drop out of a subsequent
    ``.sum(..., skipna=True)`` or ``.mean(..., skipna=True)``).

    Summing these over some dimension reproduces :func:`contingency_counts`;
    *averaging* them instead (e.g. via a batching utility such as
    ``heatextremes.metrics.mean_in_time_batches``, which reduces with a
    running mean rather than a running sum) still works with
    :func:`probability_of_detection`, :func:`false_alarm_ratio`,
    :func:`probability_of_false_detection`, and
    :func:`symmetric_extremal_dependence_index` unchanged, because the total
    count cancels out of every ratio (e.g.
    ``mean_hits / (mean_hits + mean_misses) == hits / (hits + misses)``).
    """
    valid_time, observed = match_forecast_to_observations(forecast, observations)
    threshold = _as_threshold_dataarray(threshold)

    valid = forecast.notnull() & observed.notnull() & threshold.notnull()
    forecast_extreme = forecast > threshold
    observed_extreme = observed > threshold

    indicators = xr.Dataset(
        {
            "hits": (forecast_extreme & observed_extreme).astype(float),
            "misses": (~forecast_extreme & observed_extreme).astype(float),
            "false_alarms": (forecast_extreme & ~observed_extreme).astype(float),
            "correct_negatives": (~forecast_extreme & ~observed_extreme).astype(float),
        }
    )
    return indicators.where(valid).assign_coords(verification_time=valid_time)


def contingency_counts(
    forecast: xr.DataArray,
    observations: xr.DataArray,
    threshold: float | xr.DataArray,
    dim: DimsLike = None,
) -> xr.Dataset:
    """Hit / miss / false-alarm / correct-negative counts for threshold exceedance.

    Following the standard 2x2 contingency table for a binary event
    ``value > threshold``:

    * hits (H): forecast_extreme and observed_extreme
    * misses (M): not forecast_extreme and observed_extreme
    * false_alarms (F): forecast_extreme and not observed_extreme
    * correct_negatives (C): not forecast_extreme and not observed_extreme

    ``observations`` is matched to each forecast case at its verification
    time first (see the module docstring); points where either the
    forecast, matched observation, or threshold is NaN are excluded from
    every category before counting.

    Parameters
    ----------
    forecast : xr.DataArray
        Dims include initialization ``time`` and ``prediction_timedelta``.
    observations : xr.DataArray
        Plain, unique ``time``-indexed observations (matched internally).
    threshold : float or xr.DataArray
        Exceedance threshold defining the binary extreme event
        (``value > threshold``); scalar or already-broadcastable DataArray.
    dim : str or sequence of str, optional
        Dimension(s) to sum counts over. Defaults to all dimensions
        (returns scalar counts). Pass e.g. ``"time"`` to count across
        initializations only, keeping lead time / space for a per-lead
        or per-gridpoint contingency table.

    Returns
    -------
    xr.Dataset
        Variables ``hits``, ``misses``, ``false_alarms``, ``correct_negatives``.
    """
    return extreme_indicators(forecast, observations, threshold).sum(dim=dim, skipna=True)


def probability_of_detection(counts: xr.Dataset) -> xr.DataArray:
    """Probability of detection / hit rate: POD = H / (H + M).

    ``1 - POD`` is the miss rate: the fraction of observed extremes the
    forecast failed to flag.

    Parameters
    ----------
    counts : xr.Dataset
        Output of :func:`contingency_counts` (or :func:`extreme_indicators`
        reduced/averaged some other way -- see module docstring).
    """
    h, m = counts["hits"], counts["misses"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return h / (h + m)


def false_alarm_ratio(counts: xr.Dataset) -> xr.DataArray:
    """False-alarm ratio: FAR = F / (H + F).

    Fraction of forecast extremes that were not observed. Not to be
    confused with the false-alarm *rate* (POFD) used in SEDI.

    Parameters
    ----------
    counts : xr.Dataset
        Output of :func:`contingency_counts`.
    """
    h, f = counts["hits"], counts["false_alarms"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return f / (h + f)


def probability_of_false_detection(counts: xr.Dataset) -> xr.DataArray:
    """False-alarm rate: POFD = F / (F + C).

    Fraction of non-events that were incorrectly forecast as extreme.
    This is the "F" (false-alarm rate) term used in the SEDI formula of
    Ferro and Stephenson (2011) -- distinct from the false-alarm *ratio*
    returned by :func:`false_alarm_ratio`.

    Parameters
    ----------
    counts : xr.Dataset
        Output of :func:`contingency_counts`.
    """
    f, c = counts["false_alarms"], counts["correct_negatives"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return f / (f + c)


def symmetric_extremal_dependence_index(counts: xr.Dataset) -> xr.DataArray:
    """Symmetric Extremal Dependence Index (SEDI).

    SEDI = [log(F) - log(H) - log(1-F) + log(1-H)] / [log(F) + log(H) + log(1-F) + log(1-H)]

    where H is the hit rate (POD) and F is the false-alarm rate (POFD) --
    *not* the false-alarm ratio. SEDI ranges from -1 (worst) to 1
    (perfect), is 0 for random/constant forecasts, and remains
    well-behaved for rare events, unlike POD/FAR/CSI.

    Reference: Ferro, C. A. T., and D. B. Stephenson (2011), "Deterministic
    Forecasts of Extreme Events and Their Verification," Wea. Forecasting,
    26, 699-713.

    Notes
    -----
    SEDI is undefined (NaN/inf) when H or F is exactly 0 or 1, since the
    formula involves ``log(H)``, ``log(F)``, ``log(1-H)``, and
    ``log(1-F)``. This happens with small sample sizes or a threshold so
    extreme/lenient that one contingency-table cell is always empty.
    Divide-by-zero and log-of-zero warnings are suppressed (the result is
    NaN or +/-inf as appropriate) rather than raised, matching how
    ``xarray``/``numpy`` handle these cases elsewhere in this codebase.

    Parameters
    ----------
    counts : xr.Dataset
        Output of :func:`contingency_counts`.
    """
    hit_rate = probability_of_detection(counts)
    false_alarm_rate = probability_of_false_detection(counts)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_f = np.log(false_alarm_rate)
        log_h = np.log(hit_rate)
        log_1mf = np.log(1 - false_alarm_rate)
        log_1mh = np.log(1 - hit_rate)

        numerator = log_f - log_h - log_1mf + log_1mh
        denominator = log_f + log_h + log_1mf + log_1mh
        sedi = numerator / denominator

    return sedi


def deterministic_scores(
    forecast: xr.DataArray,
    observations: xr.DataArray,
    threshold: float | xr.DataArray,
    dim: DimsLike = None,
) -> xr.Dataset:
    """Convenience wrapper: compute RMSE plus the full contingency-table suite.

    Mirrors the ``calculate_scores`` pattern used for the ensemble metrics
    (coverage / probability-of-exceedance Brier score): one call that
    returns everything needed for a per-lead-time or per-map summary. Not
    suitable for large archives that need batched/running-mean reduction --
    for that, call :func:`squared_error` and :func:`extreme_indicators`
    directly inside a batching utility (see module docstring), then finish
    with :func:`rmse_from_mean_squared_error` and the rate functions.

    Parameters
    ----------
    forecast : xr.DataArray
        Dims include initialization ``time`` and ``prediction_timedelta``.
    observations : xr.DataArray
        Plain, unique ``time``-indexed observations (matched internally at
        each case's verification time).
    threshold : float or xr.DataArray
        Exceedance threshold for the contingency-table metrics.
    dim : str or sequence of str, optional
        Dimension(s) to reduce over for both RMSE and the contingency
        counts (e.g. ``"time"`` to reduce over initializations only).

    Returns
    -------
    xr.Dataset
        Variables: ``rmse``, ``hits``, ``misses``, ``false_alarms``,
        ``correct_negatives``, ``pod``, ``miss_rate``, ``far``, ``pofd``,
        ``sedi``.
    """
    counts = contingency_counts(forecast, observations, threshold, dim=dim)
    pod = probability_of_detection(counts)

    scores = counts.assign(
        rmse=rmse(forecast, observations, dim=dim),
        pod=pod,
        miss_rate=1 - pod,
        far=false_alarm_ratio(counts),
        pofd=probability_of_false_detection(counts),
        sedi=symmetric_extremal_dependence_index(counts),
    )
    return scores
