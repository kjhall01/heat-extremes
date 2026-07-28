#!/usr/bin/env python3
"""
Process one month of AIFS ENS v2 forecasts into compact heat-hazard products.

The script:
  1. Opens AIFS initialization stores for one year/month.
  2. Computes approximate local-solar daily-mean 2 m temperature.
  3. Applies fixed ERA5 1991–2020 calendar-day q95 thresholds.
  4. Computes ensemble probabilities for:
       - q95 daily-mean hot day
       - onset of a >=2-day q95 event
       - onset of a >=3-day q95 event
  5. Writes only compact ensemble products, not member-resolved temperatures.

The output contains:
  - t2m_daily_mean_ensemble_mean
  - hot_day_q95_probability
  - heatwave_start_q95_2d_probability
  - heatwave_start_q95_3d_probability
  - valid_date(time, forecast_day, longitude)

Example
-------
python process_aifs_heat_month.py --year 2024 --month 7
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


DEFAULT_AIFS_ROOT = Path(
    "/net/monsoon/marchakitus/AIFS/v2p0/combined/forecasts_AIFS_ENS_v2"
)
DEFAULT_THRESHOLD_STORE = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/"
    "thresholds/t2m_daily_mean_percentiles_1991_2020.zarr"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/monthly"
)

SOURCE_VARIABLE = "2t"
TEMPERATURE_NAME = "2m_temperature"
THRESHOLD_NAME = "t2m_daily_mean_calendar_day_percentile"

DEFAULT_OPEN_CHUNKS = {
    "time": 1,
    "number": 26,
    "prediction_timedelta": 24,
    "latitude": 180,
    "longitude": 180,
}

DEFAULT_OUTPUT_CHUNKS = {
    "time": 1,
    "forecast_day": 8,
    "latitude": 180,
    "longitude": 180,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact monthly AIFS ENS v2 heat-hazard products."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    parser.add_argument("--max-days", type=int, default=15)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--aifs-root", type=Path, default=DEFAULT_AIFS_ROOT)
    parser.add_argument(
        "--threshold-store",
        type=Path,
        default=DEFAULT_THRESHOLD_STORE,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing monthly output store.",
    )
    parser.add_argument(
        "--zarr-format",
        type=int,
        choices=(2, 3),
        default=3,
        help=(
            "Output Zarr format. Default 3 avoids the zarr-python v3 "
            "serializer error seen when forcing format 2."
        ),
    )
    return parser.parse_args()


def store_year_month(path: Path) -> tuple[int, int] | None:
    """Extract YYYY and MM from common timestamp forms in a store name."""
    name = path.name

    patterns = (
        r"(?P<year>(?:19|20)\d{2})[-_](?P<month>0[1-9]|1[0-2])[-_]\d{2}",
        r"(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])\d{2}",
        r"(?P<year>(?:19|20)\d{2})[-_](?P<month>0[1-9]|1[0-2])",
        r"(?P<year>(?:19|20)\d{2})(?P<month>0[1-9]|1[0-2])",
    )

    for pattern in patterns:
        match = re.search(pattern, name)
        if match is not None:
            return int(match.group("year")), int(match.group("month"))

    return None


def select_monthly_paths(root: Path, year: int, month: int) -> list[Path]:
    paths = []
    unparsed = []

    for path in sorted(root.glob("*.zarr")):
        parsed = store_year_month(path)
        if parsed is None:
            unparsed.append(path.name)
            continue
        if parsed == (year, month):
            paths.append(path)

    if not paths:
        message = (
            f"No AIFS ENS v2 stores found under {root} for {year}-{month:02d}."
        )
        if unparsed:
            message += (
                f" Could not parse dates from {len(unparsed)} store names; "
                f"examples: {unparsed[:3]}"
            )
        raise FileNotFoundError(message)

    return paths


def open_aifs_month(
    root: Path,
    year: int,
    month: int,
    *,
    chunks: dict[str, int] | None = None,
    source_variable: str = SOURCE_VARIABLE,
    require_member_dimension: bool = True,
) -> xr.Dataset:
    """Open only standard-format stores initialized during one calendar month.

    ``source_variable`` defaults to the historical AIFS name but is explicit
    so the verified local-day construction can be reused for compatible
    reforecast directories without copying this processor.
    """
    paths = select_monthly_paths(root, year, month)

    def preprocess(ds: xr.Dataset) -> xr.Dataset:
        if source_variable not in ds:
            raise KeyError(f"Forecast store is missing {source_variable!r}.")
        return ds[[source_variable]]

    ds = xr.open_mfdataset(
        [str(path) for path in paths],
        engine="zarr",
        combine="nested",
        concat_dim="time",
        preprocess=preprocess,
        chunks=DEFAULT_OPEN_CHUNKS if chunks is None else chunks,
        parallel=True,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
        combine_attrs="override",
        consolidated=None,
    )

    rename = {
        source_variable: TEMPERATURE_NAME,
        "lat": "latitude",
        "lon": "longitude",
    }
    ds = ds.rename(
        {
            old: new
            for old, new in rename.items()
            if old in ds.variables or old in ds.dims
        }
    )

    required = {
        "time",
        "prediction_timedelta",
        "latitude",
        "longitude",
    }
    if require_member_dimension:
        required.add("number")
    missing = required.difference(ds.dims)
    if missing:
        raise ValueError(f"Opened AIFS data are missing dimensions: {sorted(missing)}")

    ds = ds.sortby("time").sortby("prediction_timedelta")
    if np.any(np.diff(ds["longitude"].values) <= 0):
        ds = ds.sortby("longitude")

    # Verify filename selection against the actual initialization coordinate.
    wrong_month = (
        (ds["time"].dt.year != year) | (ds["time"].dt.month != month)
    ).any()
    if bool(wrong_month):
        raise ValueError(
            "At least one opened initialization does not match the requested "
            f"month {year}-{month:02d}."
        )

    ds.attrs.update(
        processing_year=year,
        processing_month=month,
        source_store_count=len(paths),
    )
    return ds


def longitude_band_specs(longitudes: xr.DataArray):
    values = np.asarray(longitudes.values)
    if np.any(np.diff(values) <= 0):
        raise ValueError("Longitude must be strictly increasing.")

    if values.min() >= 0 and values.max() > 180:
        return (
            (0.0, 45.0, 0),
            (45.0, 135.0, 6),
            (135.0, 180.0, 12),
            (180.0, 225.0, -12),
            (225.0, 315.0, -6),
            (315.0, 360.0, 0),
        )

    if values.min() < 0 and values.max() <= 180:
        return (
            (-180.0, -135.0, -12),
            (-135.0, -45.0, -6),
            (-45.0, 45.0, 0),
            (45.0, 135.0, 6),
            (135.0, 180.0, 12),
        )

    raise ValueError("Expected longitude convention [0, 360) or [-180, 180).")


def select_half_open(
    da: xr.DataArray,
    start: float,
    stop: float,
    *,
    lon_dim: str = "longitude",
) -> xr.DataArray:
    return da.sel({lon_dim: slice(start, np.nextafter(stop, -np.inf))})


def aggregate_init_hour_band_mean(
    temperature: xr.DataArray,
    *,
    init_hour: int,
    offset_hours: int,
    init_dim: str = "time",
    step_dim: str = "prediction_timedelta",
    lon_dim: str = "longitude",
) -> xr.DataArray:
    """Aggregate one init-hour group and longitude band to local daily means."""
    steps = temperature[step_dim]
    if not np.issubdtype(steps.dtype, np.timedelta64):
        raise TypeError(f"{step_dim} must contain timedeltas.")

    step_hours = (steps / np.timedelta64(1, "h")).astype(np.int64)
    local_hour = (init_hour + step_hours + offset_hours) % 24
    midnight_indices = np.flatnonzero(local_hour.values == 0)

    if midnight_indices.size == 0:
        raise ValueError(
            f"No local-midnight step for init hour {init_hour}, "
            f"offset {offset_hours}."
        )

    aligned = temperature.isel({step_dim: slice(int(midnight_indices[0]), None)})
    n_samples = (aligned.sizes[step_dim] // 4) * 4
    if n_samples == 0:
        raise ValueError("No complete four-sample local day is available.")

    aligned = aligned.isel({step_dim: slice(0, n_samples)})
    first_steps = aligned[step_dim].isel({step_dim: slice(0, None, 4)})

    daily = aligned.coarsen(
        {step_dim: 4},
        boundary="exact",
        coord_func={step_dim: "min"},
    ).mean()

    n_days = daily.sizes[step_dim]
    daily = daily.rename({step_dim: "forecast_day"})
    daily = daily.assign_coords(
        forecast_day=np.arange(n_days, dtype=np.int16)
    )

    # Valid local date differs by initialization and longitude band.
    valid_date_2d = (
        temperature[init_dim]
        + first_steps
        + np.timedelta64(offset_hours, "h")
    ).dt.floor("D")
    valid_date_2d = valid_date_2d.rename({step_dim: "forecast_day"})
    valid_date_2d = valid_date_2d.assign_coords(
        forecast_day=daily["forecast_day"]
    )

    valid_date_3d = valid_date_2d.expand_dims(
        {lon_dim: daily[lon_dim]}
    ).transpose(init_dim, "forecast_day", lon_dim)

    daily = daily.assign_coords(valid_date=valid_date_3d)
    return daily


def local_solar_daily_mean_forecast(
    temperature: xr.DataArray,
    *,
    max_days: int,
) -> xr.DataArray:
    """Compute member-resolved local-solar daily-mean forecasts lazily."""
    if max_days < 1:
        raise ValueError("max_days must be at least one.")

    temperature = temperature.where(
        temperature["prediction_timedelta"] < np.timedelta64(max_days, "D"),
        drop=True,
    )

    init_hours = np.unique(temperature["time"].dt.hour.values)
    band_results = []

    for start, stop, offset in longitude_band_specs(temperature["longitude"]):
        band = select_half_open(temperature, start, stop)
        if band.sizes.get("longitude", 0) == 0:
            continue

        hour_results = []
        for hour in init_hours:
            hour_mask = band["time"].dt.hour == int(hour)
            hour_band = band.sel(time=hour_mask)
            if hour_band.sizes["time"] == 0:
                continue

            hour_results.append(
                aggregate_init_hour_band_mean(
                    hour_band,
                    init_hour=int(hour),
                    offset_hours=offset,
                )
            )

        band_daily = xr.concat(
            hour_results,
            dim="time",
            join="outer",
            coords="minimal",
            compat="override",
        ).sortby("time")

        band_results.append(band_daily)

    daily = xr.concat(
        band_results,
        dim="longitude",
        join="outer",
        coords="minimal",
        compat="override",
    ).sortby("longitude")

    daily.name = "t2m_daily_mean"
    daily.attrs.update(
        long_name="AIFS ENS v2 approximate local-solar daily-mean 2 m temperature",
        source_sampling="six-hourly",
        local_day_method="six-hour UTC-offset longitude bands",
    )
    return daily


def normalize_longitude_like(
    source: xr.DataArray,
    target_longitude: xr.DataArray,
) -> xr.DataArray:
    target_uses_360 = (
        float(target_longitude.min()) >= 0
        and float(target_longitude.max()) > 180
    )

    if target_uses_360:
        longitude = source["longitude"] % 360
    else:
        longitude = ((source["longitude"] + 180) % 360) - 180

    return source.assign_coords(longitude=longitude).sortby("longitude")


def map_threshold_to_forecast_grid(
    threshold: xr.DataArray,
    forecast: xr.DataArray,
) -> xr.DataArray:
    threshold = normalize_longitude_like(threshold, forecast["longitude"])

    same_lat = np.array_equal(
        threshold["latitude"].values,
        forecast["latitude"].values,
    )
    same_lon = np.array_equal(
        threshold["longitude"].values,
        forecast["longitude"].values,
    )

    if same_lat and same_lon:
        return threshold

    return threshold.interp(
        latitude=forecast["latitude"],
        longitude=forecast["longitude"],
        method="linear",
    )


def threshold_for_valid_dates(
    threshold: xr.DataArray,
    valid_date: xr.DataArray,
) -> xr.DataArray:
    """
    Vectorized day-of-year selection for valid_date(time, forecast_day, longitude).
    """
    dayofyear = valid_date.dt.dayofyear
    valid = valid_date.notnull()

    # Replace missing day-of-year indices temporarily, then restore NaNs.
    safe_dayofyear = dayofyear.fillna(1).astype(np.int16)
    selected = threshold.sel(dayofyear=safe_dayofyear)
    selected = selected.where(valid)

    return selected.transpose(
        "time",
        "forecast_day",
        "latitude",
        "longitude",
    )


def member_event_start(
    hot: xr.DataArray,
    valid: xr.DataArray,
    *,
    min_duration: int,
    day_dim: str = "forecast_day",
) -> xr.DataArray:
    """
    Mark event onset member-by-member and mask unverifiable trajectory edges.

    Day zero is NaN because the previous forecast day is unavailable.
    The final min_duration-1 days are NaN because future days are unavailable.
    """
    if min_duration < 1:
        raise ValueError("min_duration must be at least one.")

    hot_bool = hot.fillna(False).astype(bool)
    valid_bool = valid.fillna(False).astype(bool)

    qualifies = xr.concat(
        [
            hot_bool.shift({day_dim: -lag}, fill_value=False)
            for lag in range(min_duration)
        ],
        dim="_duration_check",
    ).all("_duration_check")

    future_valid = xr.concat(
        [
            valid_bool.shift({day_dim: -lag}, fill_value=False)
            for lag in range(min_duration)
        ],
        dim="_duration_check",
    ).all("_duration_check")

    previous_hot = hot_bool.shift({day_dim: 1}, fill_value=False)
    previous_valid = valid_bool.shift({day_dim: 1}, fill_value=False)

    onset = qualifies & ~previous_hot
    onset_valid = future_valid & previous_valid

    return onset.astype(np.float32).where(onset_valid)


def build_compact_products(
    daily: xr.DataArray,
    threshold: xr.DataArray,
) -> xr.Dataset:
    q95 = threshold_for_valid_dates(threshold, daily["valid_date"])

    valid = daily.notnull() & q95.notnull()
    member_hot = (daily > q95).astype(np.float32).where(valid)

    products = {
        "t2m_daily_mean_ensemble_mean": daily.mean(
            "number", skipna=True
        ).astype(np.float32),
        "hot_day_q95_probability": member_hot.mean(
            "number", skipna=True
        ).astype(np.float32),
    }

    for duration in (2, 3):
        member_start = member_event_start(
            member_hot,
            valid,
            min_duration=duration,
        )
        products[f"heatwave_start_q95_{duration}d_probability"] = (
            member_start.mean("number", skipna=True).astype(np.float32)
        )

    ds = xr.Dataset(products)
    ds = ds.assign_coords(valid_date=daily["valid_date"])
    ds.attrs.update(
        title="AIFS ENS v2 compact daily-mean heat-hazard forecasts",
        threshold="ERA5 1991-2020 calendar-day q95",
        event_definition="daily mean T2M above q95",
        local_day_method="six-hour UTC-offset longitude bands",
        ensemble_size=int(daily.sizes["number"]),
    )
    return ds


def clear_chunk_encoding(obj: xr.Dataset) -> xr.Dataset:
    obj = obj.copy()
    for name in obj.variables:
        obj[name].encoding.pop("chunks", None)
        obj[name].encoding.pop("preferred_chunks", None)
        # Avoid carrying source-store codecs/serializers into a new store.
        obj[name].encoding.pop("serializer", None)
        obj[name].encoding.pop("compressor", None)
        obj[name].encoding.pop("compressors", None)
        obj[name].encoding.pop("filters", None)
    return obj


def write_output(
    ds: xr.Dataset,
    path: Path,
    *,
    overwrite: bool,
    zarr_format: int,
) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    ds = clear_chunk_encoding(ds)

    # Zarr v3 metadata consolidation is not assumed here.
    consolidated = False if zarr_format == 3 else True

    with ProgressBar():
        ds.to_zarr(
            path,
            mode="w",
            consolidated=consolidated,
            zarr_format=zarr_format,
        )


def main() -> None:
    args = parse_args()

    output_path = (
        args.output_root
        / f"{args.year}"
        / f"aifs_ens_v2_heat_{args.year}{args.month:02d}.zarr"
    )

    print(f"Opening AIFS initializations for {args.year}-{args.month:02d}")
    aifs = open_aifs_month(
        args.aifs_root,
        args.year,
        args.month,
    )
    print(aifs)

    print("Constructing lazy local-solar daily-mean forecasts")
    daily = local_solar_daily_mean_forecast(
        aifs[TEMPERATURE_NAME],
        max_days=args.max_days,
    )

    print("Opening and mapping ERA5 q95 threshold")
    threshold = xr.open_zarr(
        args.threshold_store,
        consolidated=True,
        chunks={},
    )[THRESHOLD_NAME].sel(
        percentiles=args.percentile,
        drop=True,
    )
    threshold = map_threshold_to_forecast_grid(threshold, daily)

    print("Building compact ensemble products")
    products = build_compact_products(daily, threshold)

    chunk_spec = {
        dim: size
        for dim, size in DEFAULT_OUTPUT_CHUNKS.items()
        if dim in products.dims
    }
    products = products.chunk(chunk_spec)

    metadata = {
        "year": args.year,
        "month": args.month,
        "max_days": args.max_days,
        "percentile": args.percentile,
        "aifs_root": str(args.aifs_root),
        "threshold_store": str(args.threshold_store),
        "output_path": str(output_path),
        "zarr_format": args.zarr_format,
        "initializations": int(aifs.sizes["time"]),
        "ensemble_members": int(aifs.sizes["number"]),
        "initialization_hours": [
            int(value)
            for value in np.unique(aifs["time"].dt.hour.values)
        ],
    }
    products.attrs["run_metadata_json"] = json.dumps(metadata, sort_keys=True)

    print(products)
    print("Output chunks:")
    for name, array in products.data_vars.items():
        print(f"  {name}: {array.chunks}")

    print(f"Writing {output_path}")
    write_output(
        products,
        output_path,
        overwrite=args.overwrite,
        zarr_format=args.zarr_format,
    )

    metadata_path = output_path.parent / (
        f"aifs_ens_v2_heat_{args.year}{args.month:02d}.json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Completed successfully.")


if __name__ == "__main__":
    main()
