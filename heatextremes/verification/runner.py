"""Bounded monthly partition computation for heat verification."""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

from .config import VerificationConfig
from .alignment import map_to_forecast_grid
from .deterministic import deterministic_temperature_statistics
from .events import CANONICAL_EVENTS
from .interval_coverage import INTERVAL_SUBSETS, interval_coverage_statistics
from .io import (
    assert_safe_result_path,
    completed_output_names,
    completion_is_valid,
    find_table,
    git_commit,
    now_utc,
    read_table,
    remove_result_path,
    resolve_table_format,
    table_path,
    write_json_atomic,
    write_netcdf_atomic,
    write_table_atomic,
)
from .models import get_model_adapter
from .probabilistic import probability_decision_statistics, probability_event_statistics
from .regions import Region, load_regions, region_mask, select_regions
from .reliability import probability_reliability_statistics


@dataclass(frozen=True)
class ComputeResult:
    partition_directory: Path
    skipped: bool
    completed_forecast_days: tuple[int, ...]


def partition_directory(config: VerificationConfig, year: int, month: int) -> Path:
    return config.model_result_dir / "partial" / f"{year:04d}-{month:02d}"


def dry_run_partition(
    config: VerificationConfig,
    year: int,
    month: int,
    *,
    regions: Sequence[str] | None = None,
    forecast_days: Sequence[int] | None = None,
    input_source: str = "raw",
) -> dict[str, object]:
    """Resolve a partition plan without opening a data store."""
    config.assert_partition_selected(year, month)
    if input_source == "raw":
        adapter = get_model_adapter(config)
    elif input_source == "case_cache":
        from .models import get_case_cache_adapter

        adapter = get_case_cache_adapter(config)
    else:
        raise ValueError(f"Unsupported verification input source: {input_source}")
    configured_regions = select_regions(load_regions(config.region_file), list(regions) if regions else None)
    days = tuple(forecast_days) if forecast_days else config.forecast_days
    invalid = sorted(set(days) - set(config.forecast_days))
    if invalid:
        raise ValueError(f"Requested forecast days are not configured: {invalid}")
    return {
        "model": config.model_name,
        "partition": f"{year:04d}-{month:02d}",
        "inputs": {name: str(path) for name, path in adapter.dry_run_paths(year, month).items()},
        "output_directory": str(partition_directory(config, year, month)),
        "forecast_days": list(days),
        "regions": list(configured_regions),
        "interval_coverage": (
            "determined from case-cache lead products"
            if input_source == "case_cache"
            else (
                "configured quantiles"
                if adapter.capabilities.interval_quantiles
                else "unavailable from compact store"
            )
        ),
        "input_source": input_source,
    }


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _load_existing(directory: Path, stem: str) -> pd.DataFrame:
    path = find_table(directory, stem)
    return read_table(path) if path is not None else pd.DataFrame()


def _replace_lead_rows(existing: pd.DataFrame, new: pd.DataFrame, forecast_day: int) -> pd.DataFrame:
    if existing.empty:
        return new.reset_index(drop=True)
    replace = existing["forecast_day"].eq(forecast_day)
    if "region" in existing and "region" in new:
        replace &= existing["region"].isin(set(new["region"]))
    retained = existing[~replace]
    return pd.concat([retained, new], ignore_index=True)


def _lead_has_regions(frame: pd.DataFrame, forecast_day: int, regions: Mapping[str, Region]) -> bool:
    if frame.empty or "forecast_day" not in frame or "region" not in frame:
        return False
    available = set(frame.loc[frame["forecast_day"].eq(forecast_day), "region"])
    return set(regions).issubset(available)


def _completion_configuration_hash(directory: Path) -> str | None:
    marker = directory / "completion.json"
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("configuration_hash")
    except (OSError, json.JSONDecodeError):
        return None


def _completion_matches_request(
    directory: Path,
    *,
    configuration_hash: str,
    forecast_days: Sequence[int],
    regions: Mapping[str, Region],
) -> bool:
    """Return whether a completion marker describes this metric selection."""
    try:
        payload = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("configuration_hash") == configuration_hash
        and {int(value) for value in payload.get("forecast_days", ())} == set(forecast_days)
        and set(payload.get("regions", ())) == set(regions)
    )


def _frame_from_statistics(
    dataset: xr.Dataset,
    *,
    model: str,
    partition: str,
) -> pd.DataFrame:
    """Turn an already-computed, compact statistics dataset into tidy rows.

    Computation is deliberately performed once for all metric families in
    :func:`_lead_statistics`.  Calling ``Dataset.compute`` here would submit
    independent graphs and prevent Dask from sharing the bounded per-lead
    forecast reads between regions and metrics.
    """
    frame = dataset.to_dataframe().reset_index()
    frame.insert(0, "model", model)
    frame.insert(1, "initialization_period", partition)
    return frame


def _spatial_sufficient_statistics(lead) -> xr.Dataset:
    """Compute unweighted initialization sums for compact map aggregation."""
    forecast = lead.ensemble_mean_temperature
    observed = lead.observation_temperature
    error = forecast - observed
    temperature_valid = error.notnull()
    hot_probability = lead.event_probabilities["hot_day_q95"]
    hot_observed = lead.observed_events["hot_day_q95"].astype(float)
    hot_valid = hot_probability.notnull() & hot_observed.notnull()
    return xr.Dataset(
        {
            "temperature_bias_numerator": error.where(temperature_valid, 0.0).sum("initialization"),
            "temperature_squared_error_numerator": (error**2).where(
                temperature_valid, 0.0
            ).sum("initialization"),
            "temperature_denominator": temperature_valid.sum("initialization"),
            "hot_day_brier_numerator": ((hot_probability - hot_observed) ** 2).where(
                hot_valid, 0.0
            ).sum("initialization"),
            "hot_day_probability_numerator": hot_probability.where(hot_valid, 0.0).sum(
                "initialization"
            ),
            "hot_day_observation_numerator": hot_observed.where(hot_valid, 0.0).sum(
                "initialization"
            ),
            "hot_day_denominator": hot_valid.sum("initialization"),
        }
    ).expand_dims(forecast_day=[lead.forecast_day])


def _interval_bounds(quantiles: xr.DataArray, nominal_coverage: float) -> tuple[xr.DataArray, xr.DataArray]:
    if "quantile" not in quantiles.dims:
        raise ValueError("Configured interval quantiles must have a 'quantile' dimension")
    lower_quantile = (1.0 - nominal_coverage) / 2.0
    upper_quantile = 1.0 - lower_quantile
    values = np.asarray(quantiles["quantile"].values, dtype=float)
    for value in (lower_quantile, upper_quantile):
        if not np.any(np.isclose(values, value, rtol=0.0, atol=1e-10)):
            raise KeyError(
                f"Interval quantile product lacks required q={value:g} for nominal {nominal_coverage:g}"
            )
    return (
        quantiles.sel(quantile=lower_quantile, method="nearest"),
        quantiles.sel(quantile=upper_quantile, method="nearest"),
    )


def _unavailable_interval_rows(
    *,
    model: str,
    partition: str,
    forecast_day: int,
    regions: Mapping[str, Region],
    levels: Sequence[float],
    reason: str,
) -> pd.DataFrame:
    rows = [
        {
            "model": model,
            "initialization_period": partition,
            "region": region,
            "forecast_day": forecast_day,
            "subset": subset,
            "event": None,
            "metric": "interval_coverage",
            "nominal_coverage": level,
            "status": "unavailable",
            "reason": reason,
            "numerator": np.nan,
            "denominator": np.nan,
            "unweighted_numerator": np.nan,
            "unweighted_support": np.nan,
            "width_numerator": np.nan,
        }
        for region in regions
        for level in levels
        for subset in INTERVAL_SUBSETS
    ]
    return pd.DataFrame(rows)


def _lead_statistics(
    lead,
    *,
    model: str,
    partition: str,
    regions: Mapping[str, Region],
    probability_bins: Sequence[float],
    probability_decision_thresholds: Sequence[float],
    interval_levels: Sequence[float],
    interval_unavailable_reason: str | None = None,
    land_mask: xr.DataArray | None = None,
    include_spatial: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, xr.Dataset | None]:
    """Compute all regional sufficient statistics for one bounded lead.

    All regions and metric families are assembled into *one* Dask computation.
    This lets Dask share raw forecast/observation reads among regional masks
    without retaining a global forecast or matched-observation cube after the
    reduction completes.
    """
    deterministic_pieces: list[xr.Dataset] = []
    probability_pieces: list[xr.Dataset] = []
    decision_pieces: list[xr.Dataset] = []
    reliability_pieces: list[xr.Dataset] = []
    interval_pieces: list[xr.Dataset] = []

    compatible_land_mask = (
        map_to_forecast_grid(land_mask.astype(float), lead.ensemble_mean_temperature, method="nearest").astype(
            bool
        )
        if land_mask is not None
        else None
    )
    for region_name, region in regions.items():
        mask = region_mask(lead.ensemble_mean_temperature, region, land_mask=compatible_land_mask)
        forecast = lead.ensemble_mean_temperature.where(mask)
        observed_temperature = lead.observation_temperature.where(mask)
        observed_hot = lead.observed_hot.where(mask)
        deterministic_pieces.append(
            deterministic_temperature_statistics(forecast, observed_temperature, observed_hot).expand_dims(
                region=[region_name]
            )
        )
        for event in CANONICAL_EVENTS:
            probability = lead.event_probabilities[event].where(mask)
            observed_event = lead.observed_events[event].where(mask)
            probability_pieces.append(
                probability_event_statistics(probability, observed_event).expand_dims(
                    region=[region_name], event=[event]
                )
            )
            decision_pieces.append(
                probability_decision_statistics(
                    probability, observed_event, probability_decision_thresholds
                ).expand_dims(region=[region_name], event=[event])
            )
            reliability_pieces.append(
                probability_reliability_statistics(probability, observed_event, probability_bins).expand_dims(
                    region=[region_name], event=[event]
                )
            )
        if lead.interval_quantiles is not None:
            for level in interval_levels:
                lower, upper = _interval_bounds(lead.interval_quantiles, level)
                interval_pieces.append(
                    interval_coverage_statistics(
                        lower.where(mask),
                        upper.where(mask),
                        observed_temperature,
                        observed_hot,
                        nominal_coverage=level,
                    ).expand_dims(region=[region_name])
                )

    deterministic_statistics = xr.combine_by_coords(deterministic_pieces, combine_attrs="override")
    probability_statistics = xr.combine_by_coords(probability_pieces, combine_attrs="override")
    decision_statistics = xr.combine_by_coords(decision_pieces, combine_attrs="override")
    reliability_statistics = xr.combine_by_coords(reliability_pieces, combine_attrs="override")
    interval_statistics = (
        xr.combine_by_coords(interval_pieces, combine_attrs="override") if interval_pieces else None
    )

    # A single call is important: separate ``.compute()`` calls would cause
    # repeated upstream reads for each metric family or region.  This graph is
    # still bounded by one forecast day and is released by the caller before
    # advancing to the next day.
    computations: list[xr.Dataset] = [
        deterministic_statistics,
        probability_statistics,
        reliability_statistics,
        decision_statistics,
    ]
    interval_index: int | None = None
    if interval_statistics is not None:
        interval_index = len(computations)
        computations.append(interval_statistics)
    spatial_index: int | None = None
    if include_spatial:
        spatial_index = len(computations)
        computations.append(_spatial_sufficient_statistics(lead))
    computed = dask.compute(*computations)

    deterministic = _frame_from_statistics(
        computed[0],
        model=model,
        partition=partition,
    )
    deterministic["forecast_day"] = lead.forecast_day
    deterministic["event"] = None
    deterministic["weighted_support"] = deterministic["denominator"]
    ratio = deterministic["numerator"] / deterministic["denominator"]
    deterministic["value"] = np.where(deterministic["metric"].eq("rmse"), np.sqrt(ratio), ratio)

    probability = _frame_from_statistics(
        computed[1],
        model=model,
        partition=partition,
    )
    probability["forecast_day"] = lead.forecast_day
    probability["subset"] = "all"
    probability["decision_threshold"] = np.nan
    probability["value"] = probability["numerator"] / probability["denominator"]

    decisions = _frame_from_statistics(
        computed[3],
        model=model,
        partition=partition,
    )
    decisions["forecast_day"] = lead.forecast_day
    decisions["subset"] = "all"
    decisions["value"] = decisions["numerator"] / decisions["denominator"]
    probability = pd.concat([probability, decisions], ignore_index=True, sort=False)

    reliability = _frame_from_statistics(
        computed[2],
        model=model,
        partition=partition,
    )
    reliability["forecast_day"] = lead.forecast_day
    reliability["subset"] = "all"
    reliability["metric"] = "reliability_bin"
    reliability["mean_forecast_probability"] = (
        reliability["weighted_probability_sum"] / reliability["weighted_count"]
    )
    reliability["observed_frequency"] = (
        reliability["weighted_observation_sum"] / reliability["weighted_count"]
    )

    if interval_index is not None:
        interval = _frame_from_statistics(
            computed[interval_index],
            model=model,
            partition=partition,
        )
        interval["forecast_day"] = lead.forecast_day
        interval["event"] = None
        interval["metric"] = "interval_coverage"
        interval["status"] = "available"
        interval["reason"] = ""
        interval["empirical_weighted_coverage"] = interval["numerator"] / interval["denominator"]
        interval["empirical_unweighted_coverage"] = (
            interval["unweighted_numerator"] / interval["unweighted_support"]
        )
        interval["mean_interval_width"] = interval["width_numerator"] / interval["denominator"]
    else:
        reason = interval_unavailable_reason or (
            "configured interval-quantile preprocessing product is absent"
            if lead.interval_quantiles is None
            else "no interval levels were configured"
        )
        interval = _unavailable_interval_rows(
            model=model,
            partition=partition,
            forecast_day=lead.forecast_day,
            regions=regions,
            levels=interval_levels,
            reason=reason,
        )
    spatial = computed[spatial_index] if spatial_index is not None else None
    return deterministic, probability, interval, reliability, spatial


def _map_lead_completed(path: Path, forecast_day: int) -> bool:
    if not path.is_file():
        return False
    with xr.open_dataset(path) as dataset:
        return forecast_day in dataset["forecast_day"].values


def _upsert_map(path: Path, statistic: xr.Dataset) -> None:
    if path.is_file():
        with xr.open_dataset(path) as previous:
            retained = previous.load().drop_sel(forecast_day=statistic["forecast_day"].values)
        statistic = xr.concat([retained, statistic], dim="forecast_day").sortby("forecast_day")
    write_netcdf_atomic(statistic, path)


def _write_root_metadata(config: VerificationConfig, repository_root: Path) -> None:
    path = config.model_result_dir / "run_metadata.json"
    if path.exists():
        return
    write_json_atomic(
        {
            "model": config.model_name,
            "display_name": config.model_display_name,
            "configuration_hash": config.config_hash,
            "configuration_path": str(config.path),
            "git_commit": git_commit(repository_root),
            "created_at": now_utc(),
            "scientific_definition": "verified notebook 03_full_aifs_heat_verification",
        },
        path,
    )


def _open_optional_land_mask(config: VerificationConfig) -> tuple[xr.DataArray | None, xr.Dataset | None]:
    """Open a configured compatible land mask lazily, if one was supplied."""
    region_config = config.data.get("regions", {})
    store = region_config.get("land_mask_store")
    if not store:
        return None, None
    path = Path(store)
    variable = region_config.get("land_mask_variable")
    if path.suffix == ".zarr" or path.is_dir():
        dataset = xr.open_zarr(path, consolidated=True, chunks={})
    else:
        dataset = xr.open_dataset(path, chunks={})
    if variable is None:
        if len(dataset.data_vars) != 1:
            dataset.close()
            raise ValueError("regions.land_mask_variable is required when the mask store has multiple variables")
        variable = next(iter(dataset.data_vars))
    if variable not in dataset:
        dataset.close()
        raise KeyError(f"Configured land mask variable is missing: {variable!r}")
    return dataset[variable], dataset


def compute_partition(
    config: VerificationConfig,
    year: int,
    month: int,
    *,
    regions: Sequence[str] | None = None,
    forecast_days: Sequence[int] | None = None,
    overwrite: bool = False,
    resume: bool = True,
    repository_root: Path | None = None,
    input_source: str = "raw",
) -> ComputeResult:
    """Compute one JJAS month, checkpointing compact results after every lead."""
    config.assert_partition_selected(year, month)
    partition = f"{year:04d}-{month:02d}"
    regions_to_score = select_regions(load_regions(config.region_file), list(regions) if regions else None)
    days = tuple(int(day) for day in (forecast_days or config.forecast_days))
    invalid = sorted(set(days) - set(config.forecast_days))
    if invalid:
        raise ValueError(f"Requested forecast days are not configured: {invalid}")
    directory = partition_directory(config, year, month)
    assert_safe_result_path(directory, config.result_root)
    if completion_is_valid(directory) and not overwrite:
        completed_hash = _completion_configuration_hash(directory)
        if completed_hash != config.config_hash:
            raise RuntimeError(
                f"Completed partition uses configuration hash {completed_hash}; "
                "pass --overwrite to replace it with the current configuration"
            )
        if _completion_matches_request(
            directory,
            configuration_hash=config.config_hash,
            forecast_days=days,
            regions=regions_to_score,
        ):
            return ComputeResult(directory, True, days)
    if directory.exists() and overwrite:
        print(f"Removing configured result partition before overwrite: {directory}", flush=True)
        remove_result_path(directory, config.result_root)
    elif directory.exists() and not resume:
        raise FileExistsError(f"Partial result directory exists: {directory}; use --resume or --overwrite")
    directory.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("configuration_hash") != config.config_hash:
            raise RuntimeError(
                "Existing incomplete partition has a different configuration hash; use --overwrite"
            )
    else:
        write_json_atomic(
            {
                "status": "in_progress",
                "model": config.model_name,
                "year": year,
                "month": month,
                "configuration_hash": config.config_hash,
                "created_at": now_utc(),
            },
            progress_path,
        )
    _write_root_metadata(config, repository_root or Path.cwd())

    table_format = resolve_table_format(config.table_format)

    frames = {stem: _load_existing(directory, stem) for stem in ("deterministic", "probability", "interval_coverage", "probability_reliability")}
    map_path = directory / "maps.nc"
    if input_source == "raw":
        adapter = get_model_adapter(config)
    elif input_source == "case_cache":
        from .models import get_case_cache_adapter

        adapter = get_case_cache_adapter(config)
    else:
        raise ValueError(f"Unsupported verification input source: {input_source}")
    opened = adapter.open_partition(year, month)
    land_mask, land_mask_dataset = _open_optional_land_mask(config)
    try:
        for forecast_day in days:
            table_complete = all(
                _lead_has_regions(frames[stem], forecast_day, regions_to_score) for stem in frames
            )
            map_complete = (
                forecast_day not in config.map_forecast_days or _map_lead_completed(map_path, forecast_day)
            )
            if table_complete and map_complete:
                print(f"{partition}: forecast day {forecast_day} already checkpointed", flush=True)
                continue

            print(f"{partition}: calculating bounded forecast day {forecast_day}", flush=True)
            # Every expensive Dask reduction remains scoped to this one lead.
            # The progress bar is intentionally visible in Slurm stdout so a
            # tailed log distinguishes active reduction from scheduler delay.
            with ProgressBar():
                lead = adapter.lead(opened, forecast_day)
                print(
                    f"{partition}: forecast day {forecast_day}, assembling all "
                    f"{len(regions_to_score)} regional reductions in one shared graph",
                    flush=True,
                )
                deterministic, probability, interval, reliability, map_statistic = _lead_statistics(
                    lead,
                    model=config.model_name,
                    partition=partition,
                    regions=regions_to_score,
                    probability_bins=config.probability_bins,
                    probability_decision_thresholds=config.probability_decision_thresholds,
                    interval_levels=config.interval_levels,
                    interval_unavailable_reason=(
                        "deterministic model does not provide ensemble intervals"
                        if not adapter.capabilities.ensemble
                        else None
                    ),
                    land_mask=land_mask,
                    include_spatial=forecast_day in config.map_forecast_days,
                )
            for stem, new in (
                ("deterministic", deterministic),
                ("probability", probability),
                ("interval_coverage", interval),
                ("probability_reliability", reliability),
            ):
                frames[stem] = _replace_lead_rows(frames[stem], new, forecast_day)
                write_table_atomic(frames[stem], table_path(directory, stem, table_format))

            if map_statistic is not None:
                _upsert_map(map_path, map_statistic)
            adapter.release_lead(opened)
            del lead, deterministic, probability, interval, reliability, map_statistic
            gc.collect()
    finally:
        adapter.close_partition(opened)
        if land_mask_dataset is not None:
            land_mask_dataset.close()

    full_partition = set(days) == set(config.forecast_days)
    if full_partition:
        include_maps = bool(config.map_forecast_days)
        expected = completed_output_names(directory, include_maps=include_maps)
        write_json_atomic(
            {
                "status": "complete",
                "model": config.model_name,
                "year": year,
                "month": month,
                "partition": partition,
                "configuration_hash": config.config_hash,
                "git_commit": git_commit(repository_root or Path.cwd()),
                "creation_timestamp": now_utc(),
                "forecast_days": list(config.forecast_days),
                "regions": list(regions_to_score),
                "probability_bins": list(config.probability_bins),
                "probability_decision_thresholds": list(config.probability_decision_thresholds),
                "interval_levels": list(config.interval_levels),
                "expected_output_files": expected,
            },
            directory / "completion.json",
        )
    else:
        write_json_atomic(
            {
                "status": "in_progress",
                "model": config.model_name,
                "year": year,
                "month": month,
                "configuration_hash": config.config_hash,
                "updated_at": now_utc(),
                "completed_request": {"forecast_days": list(days), "regions": list(regions_to_score)},
                "message": "Forecast-day subset completed; full configured forecast-day range remains resumable.",
            },
            progress_path,
        )
    return ComputeResult(directory, False, days)
