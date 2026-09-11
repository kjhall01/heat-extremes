"""Report-facing heat scorecards built from canonical verification cases.

This module is deliberately separate from the normal aggregate figures.  It
reproduces the deterministic temperature-versus-ERA5-q95 definitions used in
``model_scorecards.ipynb`` while also reporting a hot-day Brier score.  All
event scores use the same deterministic q95 exceedance of each model's
ensemble-mean/deterministic temperature; no member-fraction probability is
used in this report product.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

from .alignment import map_to_forecast_grid
from .case_cache_reader import (
    DEFAULT_AIFS_MONTHLY_ROOT,
    DEFAULT_ERA5_DAILY_TEMPERATURE_STORE,
    DEFAULT_ERA5_HAZARD_STORE,
    DEFAULT_RESULTS_ROOT,
    LEGACY_AIFS_MODEL_NAMES,
    open_model_intermediates,
)
from .io import now_utc, write_json_atomic, write_table_atomic
from .regions import Region, region_mask
from .weighting import cosine_latitude_weights


DEFAULT_THRESHOLD_STORE = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/"
    "thresholds/t2m_daily_mean_percentiles_1991_2020.zarr"
)
DEFAULT_THRESHOLD_VARIABLE = "t2m_daily_mean_calendar_day_percentile"
DEFAULT_MODEL_LABELS = {
    "aifs_ens_v2": "AIFSv2 (ensemble mean)",
    "aifs_v2": "AIFSv2 (deterministic)",
    "aurora_e2s": "Aurora",
    "graphcast_e2s": "GraphCast",
    "ifs_ens": "IFS (ensemble mean)",
}
DEFAULT_MODEL_ORDER = (
    "aifs_ens_v2",
    "aifs_v2",
    "aurora_e2s",
    "graphcast_e2s",
    "ifs_ens",
)

# A scorecard does not average monthly ratios: every value below is one
# cosine-latitude weighted reduction over its common initialization cases.
_REDUCTION_DIMS = ("initialization", "latitude", "longitude")


def select_percentile(field: xr.DataArray, percentile: float) -> xr.DataArray:
    """Select a percentile independent of the source coordinate spelling."""
    for name in ("percentiles", "percentile", "quantile"):
        if name in field.dims or name in field.coords:
            return field.sel({name: percentile}, method="nearest", drop=True)
    return field


def threshold_for_calendar_days(threshold: xr.DataArray, dayofyear: xr.DataArray) -> xr.DataArray:
    """Select local calendar-day thresholds, handling 0/1-based stores and leap day."""
    available = np.asarray(threshold["dayofyear"].values, dtype=np.int16)
    if not len(available):
        raise ValueError("The q95 threshold store has an empty dayofyear coordinate")
    lookup = dayofyear - 1 if available.min() == 0 else dayofyear
    # ``nearest`` is only material for a 365-day no-leap product on 29
    # February.  It avoids inventing an unverified interpolation convention.
    return threshold.sel(dayofyear=lookup, method="nearest")


def local_day_threshold(lead: xr.Dataset, threshold: xr.DataArray) -> xr.DataArray:
    """Map ERA5 q95 to a forecast grid and its longitude-specific local date."""
    forecast = lead["forecast_temperature"]
    if "valid_date" not in lead.coords:
        raise KeyError("Canonical verification cases are missing valid_date")
    mapped = map_to_forecast_grid(threshold, forecast, method="linear")
    valid_date = lead["valid_date"]
    dayofyear = valid_date.dt.dayofyear.fillna(1).astype(np.int16)
    return threshold_for_calendar_days(mapped, dayofyear).where(valid_date.notnull()).transpose(
        *forecast.dims
    )


def _dimensions(values: xr.DataArray) -> tuple[str, ...]:
    return tuple(name for name in _REDUCTION_DIMS if name in values.dims)


def _weighted_support(valid: xr.DataArray, weights: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    dimensions = _dimensions(valid)
    weighted = weights.where(valid, 0.0).sum(dimensions, skipna=True)
    count = valid.sum(dimensions, skipna=True)
    return weighted, count


def _weighted_mean(
    values: xr.DataArray, valid: xr.DataArray, weights: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    support, count = _weighted_support(valid, weights)
    numerator = values.where(valid, 0.0).fillna(0.0) * weights.where(valid, 0.0)
    return numerator.sum(_dimensions(values), skipna=True) / support, support, count


def score_lead(
    dataset: xr.Dataset,
    forecast_day: int,
    threshold: xr.DataArray,
    region: Region,
) -> xr.Dataset:
    """Compute one report scorecard row from a canonical model lead.

    ``pod_deterministic`` and ``far_deterministic`` classify the model's
    ensemble-mean/deterministic temperature against the ERA5 1991--2020 q95
    threshold. ``brier_score_binary`` is the Brier score of that same 0/1
    q95-exceedance forecast, with no separate probability decision cutoff.
    """
    lead = dataset.sel(forecast_day=forecast_day)
    forecast = lead["forecast_temperature"]
    observation = lead["observation_temperature"]
    observed_hot = lead["observed_event"].sel(event="hot_day_q95") > 0.5
    temperature_valid = lead["temperature_case_valid"].fillna(False).astype(bool)
    event_valid = lead["event_case_valid"].sel(event="hot_day_q95").fillna(False).astype(bool)
    region_valid = region_mask(forecast, region)
    q95 = local_day_threshold(lead, threshold)
    weights = cosine_latitude_weights(forecast).broadcast_like(forecast)

    temperature_valid = temperature_valid & region_valid & observation.notnull()
    deterministic_valid = temperature_valid & event_valid & q95.notnull()
    hot_temperature_valid = temperature_valid & event_valid & observed_hot

    error = forecast - observation
    rmse_all_mean_square, all_weighted_support, all_cases = _weighted_mean(
        error**2, temperature_valid, weights
    )
    rmse_hot_mean_square, hot_weighted_support, hot_cases = _weighted_mean(
        error**2, hot_temperature_valid, weights
    )
    forecast_hot = forecast > q95
    binary_brier_score, brier_weighted_support, brier_cases = _weighted_mean(
        (forecast_hot.astype(float) - observed_hot.astype(float)) ** 2,
        deterministic_valid,
        weights,
    )
    hit = deterministic_valid & forecast_hot & observed_hot
    miss = deterministic_valid & ~forecast_hot & observed_hot
    false_alarm = deterministic_valid & forecast_hot & ~observed_hot
    hit_support, hit_cases = _weighted_support(hit, weights)
    miss_support, miss_cases = _weighted_support(miss, weights)
    false_alarm_support, false_alarm_cases = _weighted_support(false_alarm, weights)
    binary_support, binary_cases = _weighted_support(deterministic_valid, weights)

    return xr.Dataset(
        {
            "rmse_all": np.sqrt(rmse_all_mean_square),
            "rmse_hot": np.sqrt(rmse_hot_mean_square),
            "pod_deterministic": hit_support / (hit_support + miss_support),
            "far_deterministic": false_alarm_support / (hit_support + false_alarm_support),
            "brier_score_binary": binary_brier_score,
            "all_weighted_support": all_weighted_support,
            "hot_weighted_support": hot_weighted_support,
            "binary_weighted_support": binary_support,
            "brier_weighted_support": brier_weighted_support,
            "all_cases": all_cases,
            "hot_cases": hot_cases,
            "binary_cases": binary_cases,
            "brier_cases": brier_cases,
            "hits": hit_cases,
            "misses": miss_cases,
            "false_alarms": false_alarm_cases,
        }
    )


def _common_initializations(datasets: Mapping[str, xr.Dataset]) -> np.ndarray:
    names = list(datasets)
    common = datasets[names[0]]["initialization"].values
    for name in names[1:]:
        common = np.intersect1d(common, datasets[name]["initialization"].values)
    if not len(common):
        raise ValueError("Selected models have no common initialization dates")
    return common


def _assert_requested_case_coverage(
    dataset: xr.Dataset,
    model: str,
    *,
    years: Sequence[int],
    months: Sequence[int],
) -> None:
    """Refuse a report figure that would silently contain missing cache slices."""
    missing: list[str] = []
    for attribute in (
        "intermediate_reader_missing_partitions",
        "intermediate_reader_missing_slices",
        "intermediate_reader_incomplete_slices",
    ):
        raw = dataset.attrs.get(attribute, "[]")
        try:
            values = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            values = [raw]
        if values:
            missing.append(f"{attribute}={values}")
    available_months = {
        (int(value.year), int(value.month))
        for value in pd.to_datetime(dataset["initialization"].values)
    }
    expected_months = {(int(year), int(month)) for year in years for month in months}
    absent_months = sorted(expected_months - available_months)
    if absent_months:
        missing.append("no initialization in " + ", ".join(f"{year:04d}-{month:02d}" for year, month in absent_months))
    if missing:
        raise RuntimeError(
            f"Refusing incomplete report scorecard input for {model}: " + "; ".join(missing)
        )


def open_scorecard_datasets(
    models: Sequence[str],
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    legacy_aifs_monthly_root: str | Path = DEFAULT_AIFS_MONTHLY_ROOT,
    era5_daily_temperature_store: str | Path = DEFAULT_ERA5_DAILY_TEMPERATURE_STORE,
    era5_hazard_store: str | Path = DEFAULT_ERA5_HAZARD_STORE,
    years: Sequence[int] = (2022, 2023, 2024, 2025),
    months: Sequence[int] = (6, 7, 8, 9),
    forecast_days: Sequence[int] = (0, 3, 6, 9, 12),
) -> tuple[dict[str, xr.Dataset], int]:
    """Open only the selected canonical leads and align all model case dates."""
    datasets: dict[str, xr.Dataset] = {}
    for model in models:
        kwargs: dict[str, object] = {
            "results_root": results_root,
            "monthly_root": legacy_aifs_monthly_root,
            "era5_daily_temperature_store": era5_daily_temperature_store,
            "era5_hazard_store": era5_hazard_store,
            "forecast_days": forecast_days,
        }
        if model.casefold() in LEGACY_AIFS_MODEL_NAMES:
            kwargs.update(years=years, months=months)
        print(f"Opening scorecard cases: {model}", flush=True)
        dataset = open_model_intermediates(model, **kwargs)
        dataset = dataset.sel(initialization=dataset.initialization.dt.month.isin(months))
        _assert_requested_case_coverage(dataset, model, years=years, months=months)
        datasets[model] = dataset
    common = _common_initializations(datasets)
    return {name: dataset.sel(initialization=common) for name, dataset in datasets.items()}, len(common)


def compute_scorecard(
    datasets: Mapping[str, xr.Dataset],
    *,
    threshold: xr.DataArray,
    regions: Mapping[str, Region],
    forecast_days: Sequence[int],
    labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return one exact score row per model, region, and forecast day."""
    labels = labels or DEFAULT_MODEL_LABELS
    rows: list[dict[str, object]] = []
    for region_name, region in regions.items():
        for model, dataset in datasets.items():
            available_days = {int(value) for value in dataset["forecast_day"].values}
            for forecast_day in forecast_days:
                if forecast_day not in available_days:
                    raise ValueError(f"{model} does not expose requested forecast day {forecast_day}")
                print(f"Scoring {region_name}: {model}, forecast day {forecast_day}", flush=True)
                with ProgressBar():
                    result = score_lead(dataset, forecast_day, threshold, region).compute()
                rows.append(
                    {
                        "region": region_name,
                        "model": model,
                        "model_label": labels.get(model, model),
                        "forecast_day": int(forecast_day),
                        **{name: float(result[name]) for name in result.data_vars},
                    }
                )
    return pd.DataFrame(rows).sort_values(["region", "model", "forecast_day"]).reset_index(drop=True)


def scorecard_direction_table(
    scorecard: pd.DataFrame,
    *,
    candidate: str = "graphcast_e2s",
    comparator: str = "ifs_ens",
) -> pd.DataFrame:
    """Make an explicit GraphCast-versus-IFS direction-check table.

    A negative difference is better for errors/False Alarm Ratio/Brier Score; positive is
    better for Probability of Detection. Values are not declared statistically
    significant here.
    """
    metric_directions = {
        "rmse_hot": "lower",
        "pod_deterministic": "higher",
        "far_deterministic": "lower",
        "brier_score_binary": "lower",
    }
    wanted = scorecard[scorecard["model"].isin([candidate, comparator])]
    if set(wanted["model"]) != {candidate, comparator}:
        return pd.DataFrame(
            columns=[
                "region", "forecast_day", "metric", "candidate", "comparator", "candidate_minus_comparator",
                "better_direction", "candidate_is_better",
            ]
        )
    rows: list[dict[str, object]] = []
    for (region, forecast_day), frame in wanted.groupby(["region", "forecast_day"]):
        values = frame.set_index("model")
        for metric, direction in metric_directions.items():
            difference = float(values.loc[candidate, metric] - values.loc[comparator, metric])
            rows.append(
                {
                    "region": region,
                    "forecast_day": int(forecast_day),
                    "metric": metric,
                    "candidate": candidate,
                    "comparator": comparator,
                    "candidate_minus_comparator": difference,
                    "better_direction": direction,
                    "candidate_is_better": difference > 0 if direction == "higher" else difference < 0,
                }
            )
    return pd.DataFrame(rows)


_PLOT_METRICS = (
    ("rmse_hot", "Hot-Day RMSE (K)", False),
    ("pod_deterministic", "Probability of Detection", True),
    ("far_deterministic", "False Alarm Ratio", False),
    ("brier_score_binary", "Brier Score", False),
)


def _relative_performance(values: pd.DataFrame, reference_model: str, higher_is_better: bool) -> pd.DataFrame:
    reference = values.loc[reference_model]
    relative = values.divide(reference, axis="columns") if higher_is_better else values.rdiv(reference, axis="columns")
    return relative.replace([np.inf, -np.inf], np.nan) - 1.0


def _format_metric(value: float, metric: str) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.3f}" if metric == "brier_score_binary" else f"{value:.2f}"


def plot_scorecard(
    scorecard: pd.DataFrame,
    path: Path,
    *,
    reference_model: str = "ifs_ens",
) -> None:
    """Write a polished, table-only regional or global scorecard figure.

    Cell colour is a signed relative score versus IFS ensemble mean; absolute
    metric values are printed in every cell. Red is worse, white is equal, and
    blue is better.  For RMSE, FAR, and Brier score the colour value is
    ``IFS / model - 1``; for POD it is ``model / IFS - 1``.  This orients all
    four metrics so positive values mean better performance.  The legend uses
    those unitless values directly rather than labelling the transformed score
    as a percentage.
    """
    region_names = list(scorecard["region"].drop_duplicates())
    if reference_model not in set(scorecard["model"]):
        raise ValueError(
            f"Cannot plot a comparison scorecard without baseline model {reference_model!r}. "
            "Include it in --models or pass an explicit reference_model."
        )
    plot_metrics = _PLOT_METRICS
    # Each shade represents a signed, unitless comparison to the reference,
    # oriented so positive means better. Keep the generous +/-1 scale: a
    # compact +/-0.5 scale made ordinary differences look saturated and the
    # cell labels hard to read in the report version.
    cell_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cell_cmap = plt.colormaps["RdBu"].copy()
    cell_cmap.set_bad("#e5e7eb")
    figure = plt.figure(
        figsize=(
            5.4 + 3.25 * len(plot_metrics),
            1.90 + 3.15 * len(region_names),
        ),
        facecolor="white",
    )
    grid = figure.add_gridspec(
        len(region_names) + 1,
        len(plot_metrics),
        height_ratios=[*([1.0] * len(region_names)), 0.045],
        wspace=0.45,
        hspace=0.62,
    )
    region_label_axes: list[tuple[str, Axes]] = []
    for row, region_name in enumerate(region_names):
        regional = scorecard[scorecard["region"].eq(region_name)]
        available_models = list(regional["model"].drop_duplicates())
        models = [model for model in DEFAULT_MODEL_ORDER if model in available_models]
        models.extend(model for model in available_models if model not in models)
        model_labels = (
            regional.drop_duplicates("model").set_index("model").loc[models, "model_label"].tolist()
        )
        for metric_index, (metric, label, higher_is_better) in enumerate(plot_metrics):
            axis = figure.add_subplot(grid[row, metric_index])
            if metric_index == 0:
                region_label_axes.append((region_name, axis))
            values = regional.pivot(index="model", columns="forecast_day", values=metric).reindex(index=models)
            relative = _relative_performance(values, reference_model, higher_is_better)
            axis.imshow(
                np.ma.masked_invalid(relative.to_numpy()),
                cmap=cell_cmap,
                norm=cell_norm,
                interpolation="none",
                aspect="auto",
                origin="upper",
            )
            axis.set_xlim(-0.5, len(values.columns) - 0.5)
            axis.set_ylim(len(models) - 0.5, -0.5)
            for model_index, model in enumerate(models):
                for lead_index, forecast_day in enumerate(values.columns):
                    value = values.loc[model, forecast_day]
                    relative_value = relative.loc[model, forecast_day]
                    axis.text(
                        lead_index,
                        model_index,
                        _format_metric(float(value), metric),
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if np.isfinite(relative_value)
                            and (relative_value < -0.35 or relative_value > 0.35)
                            else "#263238"
                        ),
                        fontsize=10.5,
                        fontweight="bold" if model == reference_model else "normal",
                    )
            axis.set_xticks(
                range(len(values.columns)), labels=[str(value) for value in values.columns], fontsize=10
            )
            if row == 0:
                axis.set_title(label, fontsize=12, fontweight="semibold", pad=13)
            if metric_index == 0:
                axis.set_yticks(
                    range(len(models)),
                    labels=model_labels,
                    fontsize=9,
                    rotation=0,
                    va="center",
                    ha="right",
                )
                for tick, model in zip(axis.get_yticklabels(), models):
                    tick.set_fontweight("bold" if model == reference_model else "normal")
            else:
                axis.set_yticks([])
            if row == len(region_names) - 1:
                axis.set_xlabel("Forecast day", fontsize=10, labelpad=7)
            axis.tick_params(axis="x", length=0, pad=3)
            axis.tick_params(axis="y", length=0, pad=5)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.set_xticks(np.arange(-0.5, len(values.columns), 1), minor=True)
            axis.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=1.5)
            axis.tick_params(which="minor", bottom=False, left=False)
    colorbar_axis = figure.add_subplot(grid[-1, :])
    colorbar = figure.colorbar(
        ScalarMappable(norm=cell_norm, cmap=cell_cmap), cax=colorbar_axis, orientation="horizontal"
    )
    colorbar.set_ticks([-0.75, 0.0, 0.75])
    colorbar.set_ticklabels(["−0.75", "0", "+0.75"])
    colorbar.outline.set_visible(False)
    colorbar.set_label(
        f"Signed relative score vs {DEFAULT_MODEL_LABELS.get(reference_model, reference_model)} "
        "(red = worse; blue = better)",
        labelpad=6,
        fontsize=9,
        color="#374151",
    )
    colorbar.ax.tick_params(labelsize=8, length=0, colors="#374151")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(
        top=0.88,
        bottom=0.11,
        left=0.20,
        right=0.96,
    )
    for region_name, axis in region_label_axes:
        bounds = axis.get_position()
        figure.text(
            0.055,
            (bounds.y0 + bounds.y1) / 2,
            region_name.replace("_", " ").title(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="semibold",
            color="#374151",
        )
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)


def build_report_scorecard(
    *,
    models: Sequence[str],
    regions: Mapping[str, Region],
    output_directory: str | Path,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    legacy_aifs_monthly_root: str | Path = DEFAULT_AIFS_MONTHLY_ROOT,
    era5_daily_temperature_store: str | Path = DEFAULT_ERA5_DAILY_TEMPERATURE_STORE,
    era5_hazard_store: str | Path = DEFAULT_ERA5_HAZARD_STORE,
    threshold_store: str | Path = DEFAULT_THRESHOLD_STORE,
    threshold_variable: str = DEFAULT_THRESHOLD_VARIABLE,
    threshold_percentile: float = 95.0,
    years: Sequence[int] = (2022, 2023, 2024, 2025),
    months: Sequence[int] = (6, 7, 8, 9),
    forecast_days: Sequence[int] = (0, 3, 6, 9, 12),
) -> pd.DataFrame:
    """Build the CSV, direction check, PNG, and scientific provenance record."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    datasets, common_initialization_count = open_scorecard_datasets(
        models,
        results_root=results_root,
        legacy_aifs_monthly_root=legacy_aifs_monthly_root,
        era5_daily_temperature_store=era5_daily_temperature_store,
        era5_hazard_store=era5_hazard_store,
        years=years,
        months=months,
        forecast_days=forecast_days,
    )
    threshold_source = xr.open_zarr(threshold_store, consolidated=True, chunks="auto")
    try:
        if threshold_variable not in threshold_source:
            raise KeyError(f"Threshold store is missing {threshold_variable!r}")
        threshold = select_percentile(threshold_source[threshold_variable], threshold_percentile)
        scorecard = compute_scorecard(
            datasets, threshold=threshold, regions=regions, forecast_days=forecast_days
        )
    finally:
        threshold_source.close()
        for dataset in datasets.values():
            dataset.close()

    write_table_atomic(scorecard, output / "heat_report_scorecard.csv")
    direction = scorecard_direction_table(scorecard)
    write_table_atomic(direction, output / "graphcast_vs_ifs_direction_check.csv")
    plot_scorecard(
        scorecard,
        output / "heat_report_scorecard.png",
    )
    write_json_atomic(
        {
            "created_at": now_utc(),
            "models": list(models),
            "regions": list(regions),
            "years": [int(value) for value in years],
            "months": [int(value) for value in months],
            "forecast_days": [int(value) for value in forecast_days],
            "common_initialization_count": common_initialization_count,
            "threshold_store": str(threshold_store),
            "threshold_variable": threshold_variable,
            "threshold_percentile": float(threshold_percentile),
            "temperature_units": "K",
            "bias_correction": "none",
            "deterministic_definition": (
                "Raw forecast ensemble-mean/deterministic temperature > ERA5 1991-2020 local "
                "calendar-day q95. Probability of Detection is hits / (hits + misses); False Alarm "
                "Ratio is false alarms / (hits + false alarms)."
            ),
            "brier_definition": (
                "Brier Score is the cosine-latitude-weighted mean of (p - o)^2 for the ERA5 q95 "
                "hot-day event, with p=1 when raw forecast ensemble-mean/deterministic local-solar "
                "daily T2M exceeds the ERA5 1991-2020 local calendar-day q95 and p=0 otherwise. "
                "It is a binary Brier score (weighted event error), not a member-fraction probability score."
            ),
            "figure_layout": "One aligned table-only scorecard for the selected regions; no map panel.",
            "plot_reference_model": "ifs_ens",
            "plot_relative_performance_definition": (
                "Heatmap colour is signed relative performance versus ECMWF IFS ENS: "
                "for RMSE, FAR, and Brier score it is IFS / model - 1; for POD it is "
                "model / IFS - 1. Thus red is worse, white is equal, and blue is better."
            ),
            "comparison_note": (
                "All model comparisons use the exact intersection of selected JJAS initialization dates. "
                "No significance testing is implied."
            ),
        },
        output / "heat_report_scorecard_metadata.json",
    )
    return scorecard
