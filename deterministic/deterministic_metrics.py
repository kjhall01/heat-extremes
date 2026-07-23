"""Verification metrics for deterministic (single-valued) heat-extreme forecasts.

These metrics complement the ensemble/probabilistic metrics in
``heatextremes.metrics`` (``coverage``, ``probability_of_exceedance_brier_score``)
for the case where the forecast is a single deterministic value per
grid-cell / day / lead time rather than an ensemble.

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

All functions operate on aligned ``xarray.DataArray``/``Dataset`` objects and
accept an optional ``dim`` argument (a dimension name or sequence of
dimension names) specifying which axis/axes to reduce over, mirroring
xarray's own ``.mean(dim=...)`` convention. Pass ``dim=None`` (the default)
to reduce over every dimension and return a scalar; pass e.g.
``dim="time"`` to reduce only over initializations, keeping
``prediction_timedelta``, ``latitude``, and ``longitude`` intact -- the same
per-lead-time / per-map pattern used elsewhere in this project.

NaNs (missing forecast or observation values) are excluded pairwise: if
either the forecast or the observation is NaN at a point, that point does
not contribute to any of the four contingency-table categories, nor to the
RMSE.
"""

from __future__ import annotations

from typing import Hashable, Iterable, Union

import numpy as np
import xarray as xr

DimsLike = Union[Hashable, Iterable[Hashable], None]


def rmse(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    dim: DimsLike = None,
) -> xr.DataArray:
    """Root-mean-square error between a deterministic forecast and observations.

    RMSE = sqrt( mean( (forecast - observed) ** 2 ) )

    Reference: Gneiting, T. (2011), "Making and Evaluating Point Forecasts."

    Parameters
    ----------
    forecast, observed : xr.DataArray
        Aligned forecast and observation arrays (e.g. dims ``time``,
        ``prediction_timedelta``, ``latitude``, ``longitude``).
    dim : str or sequence of str, optional
        Dimension(s) to average over. Defaults to all dimensions
        (returns a scalar). Pass e.g. ``"time"`` to keep lead time and
        space, or ``("time", "latitude", "longitude")`` to keep only
        lead time.

    Returns
    -------
    xr.DataArray
    """
    error = forecast - observed
    return np.sqrt((error**2).mean(dim=dim, skipna=True))


def contingency_counts(
    forecast: xr.DataArray,
    observed: xr.DataArray,
    threshold: float,
    dim: DimsLike = None,
) -> xr.Dataset:
    """Hit / miss / false-alarm / correct-negative counts for threshold exceedance.

    Following the standard 2x2 contingency table for a binary event
    ``value > threshold``:

    * hits (H): forecast_extreme and observed_extreme
    * misses (M): not forecast_extreme and observed_extreme
    * false_alarms (F): forecast_extreme and not observed_extreme
    * correct_negatives (C): not forecast_extreme and not observed_extreme

    Points where either ``forecast`` or ``observed`` is NaN are excluded
    from every category before counting.

    Parameters
    ----------
    forecast, observed : xr.DataArray
        Aligned forecast and observation arrays.
    threshold : float
        Exceedance threshold defining the binary extreme event
        (``value > threshold``).
    dim : str or sequence of str, optional
        Dimension(s) to sum counts over. Defaults to all dimensions
        (returns scalar counts). Pass e.g. ``"time"`` to count across
        initializations only, keeping lead time / space for a per-lead
        or per-gridpoint contingency table.

    Returns
    -------
    xr.Dataset
        Variables ``hits``, ``misses``, ``false_alarms``, ``correct_negatives``
        (integer counts).
    """
    valid = forecast.notnull() & observed.notnull()
    forecast_extreme = (forecast > threshold) & valid
    observed_extreme = (observed > threshold) & valid

    hits = (forecast_extreme & observed_extreme).sum(dim=dim)
    misses = (~forecast_extreme & observed_extreme & valid).sum(dim=dim)
    false_alarms = (forecast_extreme & ~observed_extreme & valid).sum(dim=dim)
    correct_negatives = (~forecast_extreme & ~observed_extreme & valid).sum(dim=dim)

    return xr.Dataset(
        {
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_negatives": correct_negatives,
        }
    )


def probability_of_detection(counts: xr.Dataset) -> xr.DataArray:
    """Probability of detection / hit rate: POD = H / (H + M).

    ``1 - POD`` is the miss rate: the fraction of observed extremes the
    forecast failed to flag.

    Parameters
    ----------
    counts : xr.Dataset
        Output of :func:`contingency_counts`.
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
    observed: xr.DataArray,
    threshold: float,
    dim: DimsLike = None,
) -> xr.Dataset:
    """Convenience wrapper: compute RMSE plus the full contingency-table suite.

    Mirrors the ``calculate_scores`` pattern used for the ensemble metrics
    (coverage / probability-of-exceedance Brier score): one call that
    returns everything needed for a per-lead-time or per-map summary.

    Parameters
    ----------
    forecast, observed : xr.DataArray
        Aligned forecast and observation arrays.
    threshold : float
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
    counts = contingency_counts(forecast, observed, threshold, dim=dim)
    pod = probability_of_detection(counts)

    scores = counts.assign(
        rmse=rmse(forecast, observed, dim=dim),
        pod=pod,
        miss_rate=1 - pod,
        far=false_alarm_ratio(counts),
        pofd=probability_of_false_detection(counts),
        sedi=symmetric_extremal_dependence_index(counts),
    )
    return scores
