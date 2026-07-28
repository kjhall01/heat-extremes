"""Figures made exclusively from aggregated verification products."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .io import assert_safe_result_path, find_table, read_table, write_table_atomic


AGGREGATE_TABLES = {
    "deterministic": "deterministic_by_lead_region",
    "probability": "probability_by_lead_region",
    "reliability": "probability_reliability_by_lead_region_bin",
    "interval": "interval_coverage_by_lead_region",
}


def _aggregate_table(result_dirs: Sequence[Path], name: str) -> pd.DataFrame:
    stem = AGGREGATE_TABLES[name]
    frames: list[pd.DataFrame] = []
    for result_dir in result_dirs:
        path = find_table(result_dir / "aggregated", stem)
        if path is None:
            raise FileNotFoundError(f"Missing aggregate product {stem} under {result_dir}")
        frames.append(read_table(path))
    return pd.concat(frames, ignore_index=True)


def _save_figure_data(frame: pd.DataFrame, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_table_atomic(frame.reset_index(drop=True), directory / f"{stem}.csv")


def _finish(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_temperature_metric(
    deterministic: pd.DataFrame,
    *,
    metric: str,
    output_directory: Path,
    regions: Iterable[str] | None = None,
) -> None:
    """Plot all/observed-hot/observed-non-hot temperature scores by lead."""
    frame = deterministic[deterministic["metric"].eq(metric)].copy()
    if regions:
        frame = frame[frame["region"].isin(regions)]
    for region, regional in frame.groupby("region", dropna=False):
        figure, axis = plt.subplots(figsize=(8, 4.8))
        for (model, subset), values in regional.groupby(["model", "subset"], dropna=False):
            values = values.sort_values("forecast_day")
            axis.plot(values["forecast_day"], values["value"], marker="o", label=f"{model} — {subset}")
        axis.set(xlabel="Forecast day", ylabel=metric.upper(), title=f"{metric.upper()} by lead — {region}")
        axis.grid(alpha=0.35)
        axis.legend(fontsize="small")
        stem = f"{metric}_by_lead_{region}"
        _save_figure_data(regional, output_directory / "data", stem)
        _finish(figure, output_directory / f"{stem}.png")


def plot_brier_scores(probability: pd.DataFrame, output_directory: Path) -> None:
    frame = probability[probability["metric"].eq("brier_score")].copy()
    for region, regional in frame.groupby("region", dropna=False):
        figure, axis = plt.subplots(figsize=(8, 4.8))
        for (model, event), values in regional.groupby(["model", "event"], dropna=False):
            values = values.sort_values("forecast_day")
            axis.plot(values["forecast_day"], values["value"], marker="o", label=f"{model} — {event}")
        axis.set(xlabel="Forecast day", ylabel="Brier score", title=f"Heat-event Brier score — {region}")
        axis.grid(alpha=0.35)
        axis.legend(fontsize="small")
        stem = f"brier_by_lead_{region}"
        _save_figure_data(regional, output_directory / "data", stem)
        _finish(figure, output_directory / f"{stem}.png")


def plot_probability_frequency(probability: pd.DataFrame, output_directory: Path) -> None:
    source = probability[probability["event"].eq("hot_day_q95")].copy()
    for region, regional in source.groupby("region", dropna=False):
        pivot = regional.pivot_table(
            index=["model", "forecast_day"], columns="metric", values="value"
        ).reset_index()
        needed = {"mean_forecast_probability", "observed_event_frequency"}
        if not needed.issubset(pivot):
            continue
        figure, axis = plt.subplots(figsize=(8, 4.8))
        for model, values in pivot.groupby("model"):
            values = values.sort_values("forecast_day")
            axis.plot(
                values["forecast_day"], values["mean_forecast_probability"], marker="o", label=f"{model} forecast"
            )
            axis.plot(
                values["forecast_day"], values["observed_event_frequency"], marker="x", linestyle="--", label=f"{model} observed"
            )
        axis.set(xlabel="Forecast day", ylabel="Cosine-latitude weighted frequency", title=f"Hot-day forecast probability vs observed frequency — {region}")
        axis.grid(alpha=0.35)
        axis.legend(fontsize="small")
        stem = f"hot_day_probability_frequency_{region}"
        _save_figure_data(pivot, output_directory / "data", stem)
        _finish(figure, output_directory / f"{stem}.png")


def plot_reliability(
    reliability: pd.DataFrame,
    output_directory: Path,
    *,
    forecast_days: Sequence[int] | None = None,
) -> None:
    source = reliability.copy()
    if forecast_days:
        source = source[source["forecast_day"].isin(forecast_days)]
    for (region, event), values in source.groupby(["region", "event"], dropna=False):
        figure, axis = plt.subplots(figsize=(5.8, 5.5))
        axis.plot([0, 1], [0, 1], "--", color="black", label="Perfect reliability")
        for (model, forecast_day), curve in values.groupby(["model", "forecast_day"], dropna=False):
            curve = curve.sort_values("bin")
            axis.plot(
                curve["mean_forecast_probability"],
                curve["observed_frequency"],
                marker="o",
                label=f"{model} day {forecast_day}",
            )
        axis.set(
            xlabel="Mean forecast probability",
            ylabel="Observed event frequency",
            xlim=(0, 1),
            ylim=(0, 1),
            title=f"Reliability — {event}, {region}",
        )
        axis.grid(alpha=0.35)
        axis.legend(fontsize="x-small")
        stem = f"reliability_{event}_{region}"
        _save_figure_data(values, output_directory / "data", stem)
        _finish(figure, output_directory / f"{stem}.png")


def plot_interval_coverage(interval: pd.DataFrame, output_directory: Path) -> None:
    source = interval[interval["status"].eq("available")].copy()
    if source.empty:
        return
    for (region, subset), values in source.groupby(["region", "subset"], dropna=False):
        figure, axis = plt.subplots(figsize=(8, 4.8))
        for (model, forecast_day), curve in values.groupby(["model", "forecast_day"], dropna=False):
            curve = curve.sort_values("nominal_coverage")
            axis.plot(
                curve["nominal_coverage"],
                curve["empirical_weighted_coverage"],
                marker="o",
                label=f"{model} day {forecast_day}",
            )
        axis.plot([0, 1], [0, 1], "--", color="black", label="Nominal")
        axis.set(
            xlabel="Nominal interval coverage",
            ylabel="Empirical weighted coverage",
            xlim=(0, 1),
            ylim=(0, 1),
            title=f"Interval coverage — {subset}, {region}",
        )
        axis.grid(alpha=0.35)
        axis.legend(fontsize="x-small")
        stem = f"interval_coverage_nominal_{subset}_{region}"
        _save_figure_data(values, output_directory / "data", stem)
        _finish(figure, output_directory / f"{stem}.png")

        figure, axis = plt.subplots(figsize=(8, 4.8))
        for (model, nominal), curve in values.groupby(["model", "nominal_coverage"], dropna=False):
            curve = curve.sort_values("forecast_day")
            axis.plot(
                curve["forecast_day"],
                curve["empirical_weighted_coverage"],
                marker="o",
                label=f"{model} nominal {nominal:g}",
            )
        axis.set(
            xlabel="Forecast day",
            ylabel="Empirical weighted coverage",
            ylim=(0, 1),
            title=f"Interval coverage by lead — {subset}, {region}",
        )
        axis.grid(alpha=0.35)
        axis.legend(fontsize="x-small")
        stem = f"interval_coverage_lead_{subset}_{region}"
        _save_figure_data(values, output_directory / "data", stem)
        _finish(figure, output_directory / f"{stem}.png")


def plot_regional_comparison(deterministic: pd.DataFrame, output_directory: Path) -> None:
    source = deterministic[
        deterministic["metric"].eq("rmse") & deterministic["subset"].eq("observed_hot")
    ].copy()
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for (model, region), values in source.groupby(["model", "region"], dropna=False):
        values = values.sort_values("forecast_day")
        axis.plot(values["forecast_day"], values["value"], marker="o", label=f"{model} — {region}")
    axis.set(xlabel="Forecast day", ylabel="Observed-hot-day RMSE", title="Regional hot-day RMSE")
    axis.grid(alpha=0.35)
    axis.legend(fontsize="x-small", ncol=2)
    _save_figure_data(source, output_directory / "data", "regional_hot_day_rmse")
    _finish(figure, output_directory / "regional_hot_day_rmse.png")


def plot_spatial_maps(result_dirs: Sequence[Path], output_directory: Path) -> None:
    """Plot aggregate spatial fields; input is already aggregate-only NetCDF."""
    variables = (
        "temperature_bias",
        "temperature_rmse",
        "hot_day_brier_score",
        "mean_hot_day_probability",
        "observed_hot_day_frequency",
        "hot_day_probability_frequency_bias",
    )
    (output_directory / "data").mkdir(parents=True, exist_ok=True)
    for result_dir in result_dirs:
        path = result_dir / "aggregated" / "spatial_metrics.nc"
        if not path.is_file():
            continue
        model = result_dir.name
        with xr.open_dataset(path) as dataset:
            for variable in variables:
                if variable not in dataset:
                    continue
                for forecast_day in dataset["forecast_day"].values:
                    field = dataset[variable].sel(forecast_day=forecast_day).load()
                    figure, axis = plt.subplots(figsize=(10, 4.5))
                    if "bias" in variable:
                        limit = float(np.nanmax(np.abs(field.values)))
                        image = axis.pcolormesh(
                            field.longitude, field.latitude, field, cmap="RdBu_r", vmin=-limit, vmax=limit
                        )
                    else:
                        image = axis.pcolormesh(field.longitude, field.latitude, field, cmap="viridis")
                    axis.set(xlabel="Longitude", ylabel="Latitude", title=f"{model}: {variable}, day {forecast_day}")
                    figure.colorbar(image, ax=axis, label=variable)
                    stem = f"map_{model}_{variable}_day{forecast_day}"
                    write_table_atomic(
                        field.to_dataframe(name=variable).reset_index(),
                        output_directory / "data" / f"{stem}.csv",
                    )
                    _finish(figure, output_directory / f"{stem}.png")


def make_all_plots(
    result_dirs: Sequence[Path],
    output_directory: Path,
    *,
    reliability_forecast_days: Sequence[int] | None = None,
    regions: Sequence[str] | None = None,
    allowed_output_roots: Sequence[Path] | None = None,
) -> None:
    """Create the complete aggregate-only plotting suite."""
    # Figures are overwriteable products, so keep them beneath one of the
    # explicitly supplied model-result directories.  This makes a typo in a
    # standalone plotting command unable to replace an unrelated file.
    if not result_dirs:
        raise ValueError("At least one model result directory is required")
    allowed = False
    for result_dir in (allowed_output_roots or result_dirs):
        try:
            assert_safe_result_path(output_directory, result_dir)
        except ValueError:
            continue
        allowed = True
        break
    if not allowed:
        raise ValueError(
            "Figure output must be a child of one of the supplied model result directories: "
            + ", ".join(str(path) for path in result_dirs)
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    deterministic = _aggregate_table(result_dirs, "deterministic")
    probability = _aggregate_table(result_dirs, "probability")
    reliability = _aggregate_table(result_dirs, "reliability")
    interval = _aggregate_table(result_dirs, "interval")
    for metric in ("rmse", "bias", "mae"):
        plot_temperature_metric(deterministic, metric=metric, output_directory=output_directory, regions=regions)
    plot_brier_scores(probability, output_directory)
    plot_probability_frequency(probability, output_directory)
    plot_reliability(reliability, output_directory, forecast_days=reliability_forecast_days)
    plot_interval_coverage(interval, output_directory)
    plot_regional_comparison(deterministic, output_directory)
    plot_spatial_maps(result_dirs, output_directory)
