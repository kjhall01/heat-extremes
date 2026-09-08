"""Report-facing heat scorecards built from canonical verification cases.

This module is deliberately separate from the normal aggregate figures.  It
reproduces the deterministic temperature-versus-ERA5-q95 definitions used in
``model_scorecards.ipynb`` while also reporting the native event-probability
Brier score.  The distinction matters: POD/FAR here are for a deterministic
forecast made from each model's mean temperature; Brier score is for its
probabilistic hot-day forecast.
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

try:  # Cartopy is included in the project environment but optional for tests/lightweight installs.
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ModuleNotFoundError:  # pragma: no cover - exercised only without the optional plotting dependency.
    ccrs = None
    cfeature = None

from .alignment import map_to_forecast_grid
from .case_cache_reader import (
    DEFAULT_AIFS_MONTHLY_ROOT,
    DEFAULT_ERA5_DAILY_TEMPERATURE_STORE,
    DEFAULT_ERA5_HAZARD_STORE,
    DEFAULT_RESULTS_ROOT,
    LEGACY_AIFS_MODEL_NAMES,
    open_model_intermediates,
)
from .io import now_utc, write_json_atomic, write_netcdf_atomic, write_table_atomic
from .regions import Region, region_mask
from .weighting import cosine_latitude_weights


DEFAULT_THRESHOLD_STORE = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/"
    "thresholds/t2m_daily_mean_percentiles_1991_2020.zarr"
)
DEFAULT_THRESHOLD_VARIABLE = "t2m_daily_mean_calendar_day_percentile"
DEFAULT_MODEL_LABELS = {
    "aifs_ens_v2": "AIFS ENS v2 mean",
    "ifs_ens": "ECMWF IFS ENS",
    "aifs_v2": "AIFS v2",
    "aurora_e2s": "Aurora",
    "graphcast_e2s": "GraphCast",
}

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
    threshold. ``brier_score_probabilistic`` uses the model's native
    hot-day exceedance probability, with no decision cutoff.
    """
    lead = dataset.sel(forecast_day=forecast_day)
    forecast = lead["forecast_temperature"]
    observation = lead["observation_temperature"]
    probability = lead["forecast_probability"].sel(event="hot_day_q95")
    observed_hot = lead["observed_event"].sel(event="hot_day_q95") > 0.5
    temperature_valid = lead["temperature_case_valid"].fillna(False).astype(bool)
    event_valid = lead["event_case_valid"].sel(event="hot_day_q95").fillna(False).astype(bool)
    region_valid = region_mask(forecast, region)
    q95 = local_day_threshold(lead, threshold)
    weights = cosine_latitude_weights(forecast).broadcast_like(forecast)

    temperature_valid = temperature_valid & region_valid & observation.notnull()
    deterministic_valid = temperature_valid & event_valid & q95.notnull()
    probability_valid = event_valid & region_valid & probability.notnull()
    hot_temperature_valid = temperature_valid & event_valid & observed_hot

    error = forecast - observation
    rmse_all_mean_square, all_weighted_support, all_cases = _weighted_mean(
        error**2, temperature_valid, weights
    )
    rmse_hot_mean_square, hot_weighted_support, hot_cases = _weighted_mean(
        error**2, hot_temperature_valid, weights
    )
    brier_score, probability_weighted_support, probability_cases = _weighted_mean(
        (probability - observed_hot.astype(float)) ** 2, probability_valid, weights
    )

    forecast_hot = forecast > q95
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
            "brier_score_probabilistic": brier_score,
            "all_weighted_support": all_weighted_support,
            "hot_weighted_support": hot_weighted_support,
            "binary_weighted_support": binary_support,
            "probability_weighted_support": probability_weighted_support,
            "all_cases": all_cases,
            "hot_cases": hot_cases,
            "binary_cases": binary_cases,
            "probability_cases": probability_cases,
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

    A negative difference is better for errors/FAR/Brier; positive is better
    for POD.  Values are not declared statistically significant here.
    """
    metric_directions = {
        "rmse_all": "lower",
        "rmse_hot": "lower",
        "pod_deterministic": "higher",
        "far_deterministic": "lower",
        "brier_score_probabilistic": "lower",
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
    ("rmse_hot", "Hot-day\nRMSE (K)", False),
    ("pod_deterministic", "Deterministic\nPOD", True),
    ("far_deterministic", "Deterministic\nFAR", False),
    ("brier_score_probabilistic", "Probabilistic\nBrier", False),
)

_GLOBAL_PLOT_METRICS = (
    # The global rung requested for the report is standard all-day T2M RMSE,
    # not error conditional on a local hot day as in the regional heat panel.
    ("rmse_all", "Global T2M\nRMSE (K)", False),
    ("pod_deterministic", "Deterministic\nPOD", True),
    ("far_deterministic", "Deterministic\nFAR", False),
    ("brier_score_probabilistic", "Probabilistic\nBrier", False),
)


def hot_day_frequency_change(
    daily_temperature: xr.DataArray,
    threshold: xr.DataArray,
    region: Region,
    *,
    validation_years: Sequence[int],
    climatology_years: Sequence[int],
    months: Sequence[int],
) -> xr.DataArray:
    """Observed JJAS q95 incidence change for the map portion of the scorecard."""
    selected_years = sorted(set(validation_years).union(climatology_years))
    selected = daily_temperature.where(
        daily_temperature.time.dt.year.isin(selected_years)
        & daily_temperature.time.dt.month.isin(months),
        drop=True,
    )
    selected = selected.where(region_mask(selected, region), drop=True)
    mapped_threshold = map_to_forecast_grid(threshold, selected, method="linear")
    q95 = threshold_for_calendar_days(mapped_threshold, selected.time.dt.dayofyear)
    hot = selected > q95
    validation = hot.where(hot.time.dt.year.isin(validation_years)).mean("time", skipna=True)
    climatology = hot.where(hot.time.dt.year.isin(climatology_years)).mean("time", skipna=True)
    return (100 * (validation / climatology - 1)).rename("hot_day_frequency_change")


def compute_frequency_change_maps(
    *,
    daily_temperature_store: str | Path,
    threshold: xr.DataArray,
    regions: Mapping[str, Region],
    validation_years: Sequence[int],
    climatology_years: Sequence[int],
    months: Sequence[int],
) -> dict[str, xr.DataArray]:
    """Compute bounded regional map inputs once; the global row is metrics-only."""
    source = xr.open_zarr(daily_temperature_store, consolidated=True, chunks="auto")
    try:
        if "t2m_daily_mean" not in source:
            raise KeyError("ERA5 daily-temperature store is missing 't2m_daily_mean'")
        daily_temperature = source["t2m_daily_mean"]
        maps: dict[str, xr.DataArray] = {}
        for name, region in regions.items():
            # A map for the global scorecard is both less useful to a report
            # designer and disproportionately expensive. Its global metrics
            # still appear in the scorecard row below the regional panels.
            if region.latitude_min is None and region.longitude_min is None:
                continue
            print(f"Computing observed hot-day frequency-change map: {name}", flush=True)
            with ProgressBar():
                maps[name] = hot_day_frequency_change(
                    daily_temperature,
                    threshold,
                    region,
                    validation_years=validation_years,
                    climatology_years=climatology_years,
                    months=months,
                ).compute()
        return maps
    finally:
        source.close()


def _relative_performance(values: pd.DataFrame, reference_model: str, higher_is_better: bool) -> pd.DataFrame:
    reference = values.loc[reference_model]
    relative = values.divide(reference, axis="columns") if higher_is_better else values.rdiv(reference, axis="columns")
    return relative.replace([np.inf, -np.inf], np.nan) - 1.0


def _format_metric(value: float, metric: str) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.3f}" if metric == "brier_score_probabilistic" else f"{value:.2f}"


def plot_scorecard(
    scorecard: pd.DataFrame,
    path: Path,
    *,
    frequency_change_maps: Mapping[str, xr.DataArray] | None = None,
    regions: Mapping[str, Region],
    reference_model: str = "ifs_ens",
) -> None:
    """Write the regional map-plus-scorecard or global table-only figure.

    Cell colour encodes relative performance versus ECMWF IFS ENS; absolute
    values are printed in every cell. Red is worse, white is equal, and blue
    is better. For RMSE, FAR, and Brier score the plotted value is
    ``IFS / model - 1``; for POD it is ``model / IFS - 1``. This orients all
    four metrics so positive values mean better performance. The layout
    intentionally follows ``model_scorecards.ipynb`` so the report PNG is
    usable as a stand-alone figure rather than a compact diagnostic.
    """
    region_names = list(scorecard["region"].drop_duplicates())
    if reference_model not in set(scorecard["model"]):
        raise ValueError(
            f"Cannot plot a comparison scorecard without baseline model {reference_model!r}. "
            "Include it in --models or pass an explicit reference_model."
        )
    frequency_change_maps = frequency_change_maps or {}
    # The global product deliberately has no map: a world-scale incidence map
    # is expensive and adds no useful context to a global metric scorecard.
    # Reclaim that space for the metric cells rather than leaving a placeholder.
    show_map_column = any(region_name in frequency_change_maps for region_name in region_names)
    metric_start_column = 1 if show_map_column else 0
    plot_metrics = _GLOBAL_PLOT_METRICS if region_names == ["global"] else _PLOT_METRICS
    # Each shade represents the percent departure from the reference model,
    # after orienting every metric so positive is better.  Keep the notebook's
    # generous +/-100% scale: a compact +/-50% scale made ordinary differences
    # look saturated and made the text hard to read in the report version.
    cell_norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cell_cmap = plt.colormaps["RdBu"].copy()
    cell_cmap.set_bad("#e5e7eb")
    map_norm = TwoSlopeNorm(vmin=-100.0, vcenter=0.0, vmax=500.0)
    map_cmap = plt.colormaps["RdBu_r"].copy()
    map_cmap.set_bad("#e5e7eb")
    figure = plt.figure(
        figsize=(
            (8.5 if show_map_column else 4.9) + 3.35 * len(plot_metrics),
            1.35 + 3.15 * len(region_names),
        ),
        facecolor="white",
    )
    grid = figure.add_gridspec(
        len(region_names),
        len(plot_metrics) + 1 + int(show_map_column),
        width_ratios=[
            *([1.50] if show_map_column else []),
            *([1.52] * len(plot_metrics)),
            0.12,
        ],
        wspace=0.45,
        hspace=0.45,
    )
    for row, region_name in enumerate(region_names):
        if show_map_column:
            region = regions[region_name]
            map_axis = (
                figure.add_subplot(grid[row, 0], projection=ccrs.PlateCarree())
                if ccrs is not None
                else figure.add_subplot(grid[row, 0])
            )
            frequency_map = frequency_change_maps.get(region_name)
            if frequency_map is None:
                map_axis.text(
                    0.5,
                    0.5,
                    "Map\ndisabled",
                    transform=map_axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
            else:
                map_kwargs = {"transform": ccrs.PlateCarree()} if ccrs is not None else {}
                image = map_axis.pcolormesh(
                    frequency_map.longitude,
                    frequency_map.latitude,
                    frequency_map,
                    cmap=map_cmap,
                    norm=map_norm,
                    shading="auto",
                    rasterized=True,
                    **map_kwargs,
                )
                if ccrs is not None:
                    # Coastlines give geographic orientation; country boundaries
                    # make clear that Nigeria is a reporting box, rather than a
                    # political-boundary average.
                    map_axis.coastlines(color="#4a4a4a", linewidth=0.65)
                    if cfeature is not None:
                        map_axis.add_feature(
                            cfeature.BORDERS.with_scale("50m"),
                            edgecolor="#4a4a4a",
                            linewidth=0.55,
                            zorder=3,
                        )
                    map_axis.spines["geo"].set_visible(True)
                    map_axis.spines["geo"].set_color("#1f2937")
                    map_axis.spines["geo"].set_linewidth(0.8)
                if region.latitude_min is not None and region.longitude_min is not None:
                    if ccrs is not None:
                        map_axis.set_extent(
                            [
                                region.longitude_min,
                                region.longitude_max,
                                region.latitude_min,
                                region.latitude_max,
                            ],
                            crs=ccrs.PlateCarree(),
                        )
                    else:
                        map_axis.set(
                            xlim=(region.longitude_min, region.longitude_max),
                            ylim=(region.latitude_min, region.latitude_max),
                        )
                map_axis.set_facecolor("#f5f7fa")
                map_axis.set_title(
                    f"{region_name.replace('_', ' ').title()}\nobserved hot-day change",
                    fontsize=12,
                    fontweight="semibold",
                    pad=16,
                )
                if ccrs is None:
                    map_axis.set(xlabel="Longitude", ylabel="Latitude")
                elif region.latitude_min is not None and region.longitude_min is not None:
                    gridlines = map_axis.gridlines(
                        draw_labels=True,
                        linewidth=0.35,
                        color="#6b7280",
                        alpha=0.5,
                        linestyle=":",
                        x_inline=False,
                        y_inline=False,
                    )
                    gridlines.top_labels = False
                    gridlines.right_labels = False
                    gridlines.xlabel_style = {"size": 8, "color": "#374151"}
                    gridlines.ylabel_style = {"size": 8, "color": "#374151"}
                map_colorbar = figure.colorbar(
                    image,
                    ax=map_axis,
                    orientation="horizontal",
                    pad=0.12,
                    ticks=[-80, -40, 0, 200, 400],
                )
                map_colorbar.outline.set_visible(False)
                map_colorbar.set_label("Extreme-incidence rate change (%)", fontsize=8.5, color="#374151")
                map_colorbar.ax.tick_params(labelsize=8, length=0, colors="#374151")
        regional = scorecard[scorecard["region"].eq(region_name)]
        models = list(regional["model"].drop_duplicates())
        model_labels = (
            regional.drop_duplicates("model").set_index("model").loc[models, "model_label"].tolist()
        )
        for metric_index, (metric, label, higher_is_better) in enumerate(plot_metrics):
            column = metric_start_column + metric_index
            axis = figure.add_subplot(grid[row, column])
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
                        fontsize=10,
                        fontweight="bold" if model == reference_model else "normal",
                    )
            axis.set_xticks(
                range(len(values.columns)), labels=[str(value) for value in values.columns], fontsize=10
            )
            if row == 0:
                axis.set_title(label, fontsize=12, fontweight="semibold", pad=16)
            if metric_index == 0:
                axis.set_yticks(
                    range(len(models)),
                    labels=model_labels,
                    fontsize=8.5,
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
    colorbar_axis = figure.add_subplot(grid[:, -1])
    colorbar = figure.colorbar(ScalarMappable(norm=cell_norm, cmap=cell_cmap), cax=colorbar_axis)
    colorbar.set_ticks([-0.75, 0.0, 0.75])
    colorbar.set_ticklabels(["75% worse", "equal", "75% better"])
    colorbar.outline.set_visible(False)
    colorbar.set_label(
        f"Performance relative to {DEFAULT_MODEL_LABELS.get(reference_model, reference_model)}\n"
        "(red = worse; blue = better)",
        rotation=270,
        labelpad=29,
        fontsize=9,
        color="#374151",
    )
    colorbar.ax.tick_params(labelsize=8, length=0, colors="#374151")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.subplots_adjust(
        # Metric headings are two lines; leave room for both without adding a
        # figure-level title above the panels.
        top=0.86,
        bottom=0.13,
        left=0.055 if show_map_column else 0.13,
        right=0.96,
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
    include_frequency_change_maps: bool = True,
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
        frequency_change_maps = (
            compute_frequency_change_maps(
                daily_temperature_store=era5_daily_temperature_store,
                threshold=threshold,
                regions=regions,
                validation_years=years,
                climatology_years=tuple(range(1991, 2021)),
                months=months,
            )
            if include_frequency_change_maps
            else {}
        )
    finally:
        threshold_source.close()
        for dataset in datasets.values():
            dataset.close()

    write_table_atomic(scorecard, output / "heat_report_scorecard.csv")
    direction = scorecard_direction_table(scorecard)
    write_table_atomic(direction, output / "graphcast_vs_ifs_direction_check.csv")
    for region_name, frequency_change_map in frequency_change_maps.items():
        write_netcdf_atomic(
            frequency_change_map.to_dataset(),
            output / f"observed_hot_day_frequency_change_{region_name}.nc",
        )
    plot_scorecard(
        scorecard,
        output / "heat_report_scorecard.png",
        frequency_change_maps=frequency_change_maps,
        regions=regions,
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
                "Forecast ensemble-mean temperature > ERA5 1991-2020 local calendar-day q95; "
                "POD and FAR use this binary forecast."
            ),
            "probabilistic_definition": (
                "Native model hot-day exceedance probability; Brier score is reported without a decision cutoff."
            ),
            "map_definition": (
                "Map panels show 100 * (2022-2025 JJAS observed local-calendar-day q95 hot-day frequency / "
                "1991-2020 JJAS frequency - 1), using ERA5 only. The global scorecard is map-free."
            ),
            "global_plot_metric_note": (
                "A global-only figure shows all-day global T2M RMSE (rmse_all), whereas regional heat "
                "figures show observed-hot-day-conditional T2M RMSE (rmse_hot)."
            ),
            "map_files": [
                f"observed_hot_day_frequency_change_{region_name}.nc"
                for region_name in frequency_change_maps
            ],
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
