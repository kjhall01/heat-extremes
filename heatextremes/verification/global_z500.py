"""Bounded global Z500 RMSE verification for report scorecards.

The heat case cache deliberately contains only daily T2M and event fields, so
Z500 is evaluated from the registered raw reforecast stores in a separate
workflow.  It writes additive squared-error sums by model/month/lead; the
aggregation stage, never a mean of monthly RMSEs, derives the final score.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar

from .alignment import map_to_forecast_grid
from .io import now_utc, write_json_atomic, write_table_atomic
from .reforecast_inventory import store_year_month
from .weighting import cosine_latitude_weights


DEFAULT_FORECAST_VARIABLES = ("z_500", "geopotential_500", "z500", "geopotential")
DEFAULT_FORECAST_DAYS = (0, 3, 6, 9, 12)
# These preserve the report's stored 0/3/6/9/12 *lead labels* for an
# instantaneous field.  They are intentionally configurable: local-solar
# daily T2M has longitude-dependent valid dates and therefore no one exact
# instantaneous Z500 target time.
DEFAULT_LEAD_HOURS = (0, 72, 144, 216, 288)


def canonicalize_spatial_coordinates(values: xr.DataArray) -> xr.DataArray:
    """Rename the two common source spellings without changing their values."""
    renames = {
        old: new
        for old, new in (("lat", "latitude"), ("lon", "longitude"))
        if old in values.dims or old in values.coords
    }
    return values.rename(renames) if renames else values


def select_z500_variable(
    dataset: xr.Dataset,
    candidates: Sequence[str] = DEFAULT_FORECAST_VARIABLES,
) -> tuple[xr.DataArray, str]:
    """Find a native Z500/geopotential field and isolate 500 hPa if needed."""
    name = next((candidate for candidate in candidates if candidate in dataset), None)
    if name is None:
        raise KeyError(f"No Z500 variable; tried {list(candidates)}")
    values = canonicalize_spatial_coordinates(dataset[name])
    for level_name in ("level", "pressure_level", "isobaricInhPa"):
        if level_name in values.dims or level_name in values.coords:
            values = values.sel({level_name: 500}, method="nearest", drop=True)
            break
    if "number" in values.dims:
        values = values.mean("number", skipna=True)
    required = {"time", "prediction_timedelta", "latitude", "longitude"}
    missing = required - set(values.dims)
    if missing:
        raise ValueError(f"Z500 field {name!r} is missing dimensions {sorted(missing)}")
    return values, name


def select_era5_z500_variable(dataset: xr.Dataset, name: str) -> xr.DataArray:
    """Select a configured ERA5 geopotential field at 500 hPa."""
    if name not in dataset:
        raise KeyError(f"ERA5 Z500 store is missing configured variable {name!r}")
    values = canonicalize_spatial_coordinates(dataset[name])
    for level_name in ("level", "pressure_level", "isobaricInhPa"):
        if level_name in values.dims or level_name in values.coords:
            values = values.sel({level_name: 500}, method="nearest", drop=True)
            break
    required = {"time", "latitude", "longitude"}
    missing = required - set(values.dims)
    if missing:
        raise ValueError(f"ERA5 Z500 field {name!r} is missing dimensions {sorted(missing)}")
    return values


def normalized_units(values: xr.DataArray) -> str | None:
    """Normalize cosmetic spelling differences, retaining a conservative comparison."""
    raw = values.attrs.get("units")
    if raw is None or not str(raw).strip():
        return None
    return "".join(str(raw).lower().split()).replace("**", "^")


def selected_partition_stores(record: dict[str, object], year: int, month: int) -> list[Path]:
    directory = Path(str(record["source_directory"]))
    source_glob = str(record["source_store_glob"])
    return [
        path
        for path in sorted(directory.glob(source_glob))
        if path.is_dir() and store_year_month(path) == (year, month)
    ]


def preflight_record(
    record: dict[str, object],
    *,
    forecast_days: Sequence[int],
    lead_hours: Sequence[int],
    candidates: Sequence[str] = DEFAULT_FORECAST_VARIABLES,
) -> dict[str, object]:
    """Inspect one source's metadata only and report its Z500 capability."""
    partitions = record.get("selected_partitions", [])
    if not partitions:
        return {"model": record["model"], "status": "unavailable", "reason": "no selected partitions"}
    source_variables: set[str] = set()
    source_units: set[str] = set()
    initializations: set[str] = set()
    stores_seen: list[str] = []
    for partition in partitions:
        year, month = int(partition["year"]), int(partition["month"])
        stores = selected_partition_stores(record, year, month)
        if not stores:
            return {
                "model": record["model"],
                "display_name": record.get("display_name", record["model"]),
                "status": "unavailable",
                "reason": f"no source stores for {year:04d}-{month:02d}",
            }
        for store in stores:
            source = xr.open_zarr(store, consolidated=False, chunks={})
            try:
                values, variable = select_z500_variable(source, candidates)
                available_hours = {
                    int(value / np.timedelta64(1, "h")) for value in values["prediction_timedelta"].values
                }
                missing_hours = [hour for hour in lead_hours if hour not in available_hours]
                if missing_hours:
                    return {
                        "model": record["model"],
                        "display_name": record.get("display_name", record["model"]),
                        "status": "unavailable",
                        "source_store": str(store),
                        "missing_lead_hours": missing_hours,
                        "reason": "source lacks one or more requested Z500 lead hours",
                    }
                source_variables.add(variable)
                if normalized_units(values) is not None:
                    source_units.add(normalized_units(values) or "")
                initializations.update(
                    np.datetime_as_string(np.datetime64(value), unit="ns") for value in values["time"].values
                )
                stores_seen.append(str(store))
            except (KeyError, ValueError) as error:
                return {
                    "model": record["model"],
                    "display_name": record.get("display_name", record["model"]),
                    "status": "unavailable",
                    "reason": str(error),
                    "source_store": str(store),
                }
            finally:
                source.close()
    if len(source_variables) != 1:
        return {
            "model": record["model"],
            "display_name": record.get("display_name", record["model"]),
            "status": "unavailable",
            "reason": f"source variable changes across partitions: {sorted(source_variables)}",
        }
    if len(source_units) > 1:
        return {
            "model": record["model"],
            "display_name": record.get("display_name", record["model"]),
            "status": "unavailable",
            "reason": f"source units change across partitions: {sorted(source_units)}",
        }
    return {
        "model": record["model"],
        "display_name": record.get("display_name", record["model"]),
        "status": "available",
        "source_variable": next(iter(source_variables)),
        "source_units": next(iter(source_units), None),
        "forecast_days": [int(value) for value in forecast_days],
        "lead_hours": [int(value) for value in lead_hours],
        "initializations": sorted(initializations),
        "source_store_count": len(stores_seen),
    }


def preflight_manifest(
    manifest_path: Path,
    *,
    era5_z500_store: Path,
    era5_z500_variable: str,
    models: Sequence[str],
    forecast_days: Sequence[int],
    lead_hours: Sequence[int],
    output_path: Path,
) -> dict[str, object]:
    """Write a machine-readable availability report before expensive Z500 jobs."""
    if len(forecast_days) != len(lead_hours):
        raise ValueError("forecast_days and lead_hours must have the same length")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {str(record["model"]): record for record in manifest.get("models", [])}
    era5_source = xr.open_zarr(era5_z500_store, consolidated=True, chunks={})
    try:
        era5 = select_era5_z500_variable(era5_source, era5_z500_variable)
        era5_metadata = {
            "store": str(era5_z500_store),
            "variable": era5_z500_variable,
            "units": era5.attrs.get("units"),
            "time_start": str(era5["time"].values[0]),
            "time_end": str(era5["time"].values[-1]),
        }
    finally:
        era5_source.close()
    model_reports = []
    for model in models:
        if model not in records:
            model_reports.append(
                {"model": model, "status": "unavailable", "reason": "model is absent from inventory manifest"}
            )
            continue
        model_reports.append(
            preflight_record(
                records[model], forecast_days=forecast_days, lead_hours=lead_hours
            )
        )
    available_initializations = [
        set(item["initializations"])
        for item in model_reports
        if item["status"] == "available" and item.get("initializations")
    ]
    common_initializations = (
        sorted(set.intersection(*available_initializations)) if available_initializations else []
    )
    if available_initializations and not common_initializations:
        raise RuntimeError("Z500-capable models have no common initialization dates")
    payload = {
        "created_at": now_utc(),
        "manifest": str(manifest_path),
        "era5": era5_metadata,
        "forecast_days": [int(value) for value in forecast_days],
        "lead_hours": [int(value) for value in lead_hours],
        "models": model_reports,
        "common_initializations": common_initializations,
    }
    write_json_atomic(payload, output_path)
    return payload


def _open_partition_z500(
    record: dict[str, object],
    year: int,
    month: int,
    lead_hours: Sequence[int],
    initializations: Sequence[str] | None = None,
) -> tuple[xr.DataArray, str]:
    stores = selected_partition_stores(record, year, month)
    if not stores:
        raise FileNotFoundError(f"No source stores for {record['model']} {year:04d}-{month:02d}")
    pieces: list[xr.DataArray] = []
    source_variable = ""
    for store in stores:
        source = xr.open_zarr(store, consolidated=False, chunks="auto")
        values, source_variable = select_z500_variable(source)
        requested = np.asarray(lead_hours) * np.timedelta64(1, "h")
        pieces.append(values.sel(prediction_timedelta=requested))
    result = xr.concat(pieces, dim="time").sortby("time")
    if initializations is not None:
        selected = np.asarray(initializations, dtype="datetime64[ns]")
        result = result.sel(time=np.intersect1d(result["time"].values, selected))
    return result, source_variable


def global_z500_partition_statistics(
    forecast: xr.DataArray,
    era5: xr.DataArray,
    *,
    forecast_days: Sequence[int],
    lead_hours: Sequence[int],
) -> pd.DataFrame:
    """Return additive global squared-error sums, one row per requested lead."""
    rows: list[dict[str, object]] = []
    for forecast_day, lead_hour in zip(forecast_days, lead_hours, strict=True):
        lead = forecast.sel(prediction_timedelta=np.timedelta64(lead_hour, "h"))
        lead = lead.rename({"time": "initialization"})
        valid_times = lead["initialization"] + np.timedelta64(lead_hour, "h")
        needed_times = np.unique(valid_times.values)
        # Select time before interpolation so a four-year ERA5 store never
        # enters the horizontal regridding graph.
        selected_era5 = era5.sel(time=needed_times)
        mapped = map_to_forecast_grid(selected_era5, lead, method="linear")
        observation = mapped.sel(time=xr.DataArray(valid_times.values, dims="initialization"))
        observation = observation.transpose(*lead.dims)
        error = lead - observation
        weights = cosine_latitude_weights(error).broadcast_like(error)
        valid = error.notnull() & weights.notnull()
        dimensions = tuple(name for name in ("initialization", "latitude", "longitude") if name in error.dims)
        numerator = ((error**2).where(valid, 0.0) * weights.where(valid, 0.0)).sum(dimensions, skipna=True)
        denominator = weights.where(valid, 0.0).sum(dimensions, skipna=True)
        cases = valid.sum(dimensions, skipna=True)
        with ProgressBar():
            result = xr.Dataset(
                {"squared_error_numerator": numerator, "weight_denominator": denominator, "cases": cases}
            ).compute()
        rows.append(
            {
                "forecast_day": int(forecast_day),
                "lead_hours": int(lead_hour),
                "z500_squared_error_numerator": float(result["squared_error_numerator"]),
                "z500_weight_denominator": float(result["weight_denominator"]),
                "z500_cases": int(result["cases"]),
            }
        )
    return pd.DataFrame(rows)


def compute_manifest_task(
    manifest_path: Path,
    *,
    task_index: int,
    era5_z500_store: Path,
    era5_z500_variable: str,
    output_directory: Path,
    forecast_days: Sequence[int],
    lead_hours: Sequence[int],
    preflight_path: Path | None = None,
) -> Path:
    """Compute one model/month task, or write an explicit unavailable status."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if not 0 <= task_index < len(tasks):
        raise IndexError(f"task_index={task_index} outside task_count={len(tasks)}")
    task = tasks[task_index]
    records = {str(record["model"]): record for record in manifest["models"]}
    model = str(task["model"])
    year, month = int(task["year"]), int(task["month"])
    path = output_directory / "partial" / model / f"{year:04d}-{month:02d}.csv"
    if model not in records:
        raise KeyError(f"Task model {model!r} is absent from manifest records")
    record = records[model]
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path else None
    reports = {item["model"]: item for item in preflight.get("models", [])} if preflight else {}
    report = reports.get(model) or preflight_record(
        record, forecast_days=forecast_days, lead_hours=lead_hours
    )
    common = {
        "model": model,
        "display_name": record.get("display_name", model),
        "partition": f"{year:04d}-{month:02d}",
        "status": report["status"],
        "reason": report.get("reason"),
        "source_variable": report.get("source_variable"),
        "source_units": report.get("source_units"),
    }
    if report["status"] != "available":
        frame = pd.DataFrame(
            [{**common, "forecast_day": day, "lead_hours": hour} for day, hour in zip(forecast_days, lead_hours)]
        )
        write_table_atomic(frame, path)
        return path

    era5_source = xr.open_zarr(era5_z500_store, consolidated=True, chunks="auto")
    try:
        era5 = select_era5_z500_variable(era5_source, era5_z500_variable)
        forecast, source_variable = _open_partition_z500(
            record,
            year,
            month,
            lead_hours,
            initializations=preflight.get("common_initializations") if preflight else None,
        )
        if not forecast.sizes.get("time", 0):
            frame = pd.DataFrame(
                [
                    {
                        **common,
                        "forecast_day": day,
                        "lead_hours": hour,
                        "status": "unavailable",
                        "reason": "no common initialization dates in this partition",
                        "source_variable": source_variable,
                    }
                    for day, hour in zip(forecast_days, lead_hours)
                ]
            )
        else:
            source_units, era5_units = normalized_units(forecast), normalized_units(era5)
            if source_units and era5_units and source_units != era5_units:
                frame = pd.DataFrame(
                    [
                        {
                            **common,
                            "forecast_day": day,
                            "lead_hours": hour,
                            "status": "unavailable",
                            "reason": f"unit mismatch: forecast={source_units}, ERA5={era5_units}",
                            "source_variable": source_variable,
                        }
                        for day, hour in zip(forecast_days, lead_hours)
                    ]
                )
            else:
                frame = global_z500_partition_statistics(
                    forecast, era5, forecast_days=forecast_days, lead_hours=lead_hours
                )
                frame = frame.assign(**{**common, "status": "available", "source_variable": source_variable})
                frame["era5_variable"] = era5_z500_variable
                frame["era5_units"] = era5.attrs.get("units")
    finally:
        era5_source.close()
    write_table_atomic(frame, path)
    return path


def aggregate_global_z500(
    output_directory: Path,
    *,
    heat_scorecard_path: Path | None = None,
    manifest_path: Path | None = None,
    models: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate exact Z500 RMSE and join global raw-T2M RMSE when available."""
    files = sorted((output_directory / "partial").glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No Z500 partial CSVs under {output_directory / 'partial'}")
    partial = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_models = set(models) if models else {str(item["model"]) for item in manifest["models"]}
        expected = {
            (str(task["model"]), f"{int(task['year']):04d}-{int(task['month']):02d}")
            for task in manifest["tasks"]
            if str(task["model"]) in selected_models
        }
        actual = set(zip(partial["model"].astype(str), partial["partition"].astype(str), strict=False))
        missing = sorted(expected - actual)
        if missing:
            details = ", ".join(f"{model}/{partition}" for model, partition in missing)
            raise RuntimeError(f"Refusing incomplete Z500 aggregate; missing task output(s): {details}")
    unavailable_models = set(partial.loc[~partial["status"].eq("available"), "model"])
    available = partial[
        partial["status"].eq("available") & ~partial["model"].isin(unavailable_models)
    ].copy()
    if available.empty:
        z500 = pd.DataFrame(columns=["model", "display_name", "forecast_day", "lead_hours", "z500_rmse"])
    else:
        grouped = available.groupby(["model", "display_name", "forecast_day", "lead_hours"], as_index=False).agg(
            z500_squared_error_numerator=("z500_squared_error_numerator", "sum"),
            z500_weight_denominator=("z500_weight_denominator", "sum"),
            z500_cases=("z500_cases", "sum"),
        )
        grouped["z500_rmse"] = np.sqrt(
            grouped["z500_squared_error_numerator"] / grouped["z500_weight_denominator"]
        )
        z500 = grouped.sort_values(["model", "forecast_day"])
    write_table_atomic(z500, output_directory / "global_z500_rmse.csv")

    unavailable = partial[~partial["status"].eq("available")].copy()
    availability_columns = ["model", "display_name", "partition", "forecast_day", "lead_hours", "status", "reason"]
    write_table_atomic(unavailable.reindex(columns=availability_columns), output_directory / "global_z500_unavailable.csv")

    combined = z500.copy()
    if heat_scorecard_path is not None and heat_scorecard_path.is_file():
        heat = pd.read_csv(heat_scorecard_path)
        heat = heat[heat["region"].eq("global")][["model", "forecast_day", "rmse_all", "rmse_hot"]].rename(
            columns={"rmse_all": "t2m_rmse_all", "rmse_hot": "t2m_rmse_hot"}
        )
        combined = combined.merge(heat, on=["model", "forecast_day"], how="outer", validate="one_to_one")
    write_table_atomic(combined, output_directory / "global_model_scorecard.csv")
    write_json_atomic(
        {
            "created_at": now_utc(),
            "partial_count": len(files),
            "available_z500_rows": len(z500),
            "unavailable_z500_rows": len(unavailable),
            "unavailable_models": sorted(str(value) for value in unavailable_models),
            "z500_definition": (
                "Cosine-latitude weighted global instantaneous Z500 RMSE at the explicit lead_hours column; "
                "monthly squared errors and weights are summed before the square root."
            ),
            "t2m_definition": (
                "When present, rmse_all/rmse_hot are local-solar daily mean T2M raw-scorecard values; "
                "they are not instantaneous-field metrics."
            ),
        },
        output_directory / "global_model_scorecard_metadata.json",
    )
    return z500, combined
