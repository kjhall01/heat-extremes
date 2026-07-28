"""Lazy, gap-aware access to legacy and modern model intermediate Zarr stores.

``aifs_ens_v2`` uses the existing compact monthly stores consumed by the
``03_full_aifs_heat_verification.ipynb`` notebook.  Other models use the
modern, canonical case-cache layout written by the reforecast workflow.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from .alignment import map_to_forecast_grid, match_observation_by_valid_date
from .events import CANONICAL_EVENTS


DEFAULT_RESULTS_ROOT = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results"
)
DEFAULT_AIFS_MONTHLY_ROOT = Path("/net/monsoon/kylehall/ERA5/heat_extremes_aifs_ens_v2/monthly")
DEFAULT_ERA5_DAILY_TEMPERATURE_STORE = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/daily/t2m_daily_mean.zarr"
)
DEFAULT_ERA5_HAZARD_STORE = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_climatology/hazards/t2m_daily_mean_q95_hazards.zarr"
)
LEGACY_AIFS_MODEL_NAMES = frozenset({"aifs_ens_v2", "aifs-ens-v2"})
SUPPORTED_FORECAST_DAYS = tuple(range(15))
DEFAULT_VERIFICATION_MONTHS = (6, 7, 8, 9)
_PARTITION_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])")
_STORE_PATTERN = re.compile(r"forecast_day_(?P<forecast_day>\d{3})\.zarr")
_AIFS_MONTHLY_STORE_PATTERN = re.compile(r"aifs_ens_v2_heat_(?P<year>\d{4})(?P<month>0[1-9]|1[0-2])\.zarr")


@dataclass(frozen=True)
class CaseCacheAvailability:
    """Discovered coverage of one model's case-cache outputs."""

    model_name: str
    expected_partitions: tuple[str, ...]
    missing_partitions: tuple[str, ...]
    missing_slices: tuple[tuple[str, int], ...]
    incomplete_slices: tuple[tuple[str, int], ...]
    manifest_used: bool

    def format_report(self) -> str:
        """Return a concise, user-facing coverage report."""
        lines = [
            f"Case-cache report for {self.model_name}: "
            f"{len(self.expected_partitions)} expected month(s), "
            f"{len(self.missing_slices)} missing lead slice(s)."
        ]
        if not self.manifest_used:
            lines.append("No inventory manifest found; whole missing months cannot be inferred.")
        if self.missing_partitions:
            lines.append("Missing month(s): " + ", ".join(self.missing_partitions))
        if self.missing_slices:
            details = ", ".join(
                f"{partition}/forecast_day_{forecast_day:03d}"
                for partition, forecast_day in self.missing_slices
            )
            lines.append("Filled with NaNs: " + details)
        if self.incomplete_slices:
            details = ", ".join(
                f"{partition}/forecast_day_{forecast_day:03d}"
                for partition, forecast_day in self.incomplete_slices
            )
            lines.append("Incomplete store(s), treated as missing: " + details)
        return "\n".join(lines)


def open_model_intermediates(
    model_name: str,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    monthly_root: str | Path = DEFAULT_AIFS_MONTHLY_ROOT,
    era5_daily_temperature_store: str | Path = DEFAULT_ERA5_DAILY_TEMPERATURE_STORE,
    era5_hazard_store: str | Path = DEFAULT_ERA5_HAZARD_STORE,
    forecast_days: Sequence[int] | None = None,
    years: Sequence[int] | None = None,
    months: Sequence[int] | None = DEFAULT_VERIFICATION_MONTHS,
    chunks: Mapping[str, int] | str = "auto",
    verbose: bool = True,
) -> xr.Dataset:
    """Open a model's intermediate stores lazily and fill missing leads with NaNs.

    ``aifs_ens_v2`` (and the hyphenated alias) opens the legacy compact
    monthly intermediates used by ``03_full_aifs_heat_verification.ipynb``.
    Every other model opens its modern canonical case cache.

    Both routes return the same canonical fields: ``forecast_temperature``,
    ``observation_temperature``, ``temperature_case_valid``,
    ``forecast_probability``, ``observed_event``, and ``event_case_valid``.
    The legacy route matches ERA5 temperature and event fields lazily using
    the compact store's ``valid_date`` coordinate.  ``results_root`` is used
    only by modern case-cache models; the monthly and ERA5 paths are used only
    by AIFS ENS v2.  For AIFS, ``months`` defaults to JJAS so it matches the
    verification workflow without opening irrelevant monthly stores; pass
    ``months=None`` to include every discovered month.
    """
    if model_name.casefold() in LEGACY_AIFS_MODEL_NAMES:
        return _open_legacy_aifs_monthly_intermediates(
            model_name,
            monthly_root=monthly_root,
            era5_daily_temperature_store=era5_daily_temperature_store,
            era5_hazard_store=era5_hazard_store,
            forecast_days=forecast_days,
            years=years,
            months=months,
            chunks=chunks,
            verbose=verbose,
        )
    return _open_modern_case_cache(
        model_name,
        results_root=results_root,
        forecast_days=forecast_days,
        chunks=chunks,
        verbose=verbose,
    )


def open_model_case_cache(
    model_name: str,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    monthly_root: str | Path = DEFAULT_AIFS_MONTHLY_ROOT,
    era5_daily_temperature_store: str | Path = DEFAULT_ERA5_DAILY_TEMPERATURE_STORE,
    era5_hazard_store: str | Path = DEFAULT_ERA5_HAZARD_STORE,
    forecast_days: Sequence[int] | None = None,
    years: Sequence[int] | None = None,
    months: Sequence[int] | None = DEFAULT_VERIFICATION_MONTHS,
    chunks: Mapping[str, int] | str = "auto",
    verbose: bool = True,
) -> xr.Dataset:
    """Backward-compatible name for :func:`open_model_intermediates`."""
    return open_model_intermediates(
        model_name,
        results_root=results_root,
        monthly_root=monthly_root,
        era5_daily_temperature_store=era5_daily_temperature_store,
        era5_hazard_store=era5_hazard_store,
        forecast_days=forecast_days,
        years=years,
        months=months,
        chunks=chunks,
        verbose=verbose,
    )


def _open_modern_case_cache(
    model_name: str,
    *,
    results_root: str | Path,
    forecast_days: Sequence[int] | None,
    chunks: Mapping[str, int] | str,
    verbose: bool,
) -> xr.Dataset:
    """Open a modern model's case-cache stores lazily and fill missing leads.

    The inventory manifest supplies expected months and forecast days when it
    exists.  Otherwise existing ``case_cache/YYYY-MM`` directories are used
    and the project-standard days 0--14 are expected.  Missing leads within a
    month that has at least one readable store are inserted as all-NaN slices.
    A wholly absent expected month is reported but omitted because there is no
    trustworthy initialization coordinate from which to construct its shape.

    """
    root = Path(results_root).expanduser()
    cache_root = root / model_name / "case_cache"
    expected_partitions, manifest_days, manifest_used = _manifest_expectations(root, model_name)
    discovered_partitions = _partition_directories(cache_root)
    if not expected_partitions:
        expected_partitions = tuple(sorted(discovered_partitions))
    expected_days = _expected_forecast_days(forecast_days, manifest_days)
    _log(
        f"Modern cache {model_name}: {len(expected_partitions)} month(s), "
        f"forecast days {list(expected_days)}, root={cache_root}",
        verbose,
    )

    missing_partitions = tuple(
        partition for partition in expected_partitions if partition not in discovered_partitions
    )
    missing_slices: list[tuple[str, int]] = []
    incomplete_slices: list[tuple[str, int]] = []
    monthly_datasets: list[xr.Dataset] = []

    for index, partition in enumerate(expected_partitions, start=1):
        directory = discovered_partitions.get(partition)
        if directory is None:
            _log(f"[{index}/{len(expected_partitions)}] {partition}: missing month directory", verbose)
            continue
        _log(f"[{index}/{len(expected_partitions)}] Opening {partition}: {directory}", verbose)
        try:
            monthly, missing, incomplete = _open_partition(
                directory,
                expected_days,
                chunks=chunks,
            )
        except Exception:
            _log(f"[{index}/{len(expected_partitions)}] {partition}: FAILED while opening", verbose)
            raise
        missing_slices.extend((partition, forecast_day) for forecast_day in missing)
        incomplete_slices.extend((partition, forecast_day) for forecast_day in incomplete)
        if monthly is not None:
            monthly_datasets.append(monthly)
            _log(
                f"[{index}/{len(expected_partitions)}] {partition}: ready "
                f"({len(expected_days) - len(missing)}/{len(expected_days)} lead stores)",
                verbose,
            )
        else:
            _log(f"[{index}/{len(expected_partitions)}] {partition}: no readable lead stores", verbose)

    report = CaseCacheAvailability(
        model_name=model_name,
        expected_partitions=expected_partitions,
        missing_partitions=missing_partitions,
        missing_slices=tuple(missing_slices),
        incomplete_slices=tuple(incomplete_slices),
        manifest_used=manifest_used,
    )
    _log(report.format_report(), verbose)
    if not monthly_datasets:
        raise FileNotFoundError(
            f"No readable case-cache stores found for model {model_name!r} beneath {cache_root}"
        )

    dataset = _concat_months(monthly_datasets)
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs.update(
        intermediate_reader_source="modern_case_cache",
        intermediate_reader_model=model_name,
        intermediate_reader_results_root=str(root),
        intermediate_reader_expected_forecast_days=json.dumps(list(expected_days)),
        intermediate_reader_missing_partitions=json.dumps(list(missing_partitions)),
        intermediate_reader_missing_slices=json.dumps(list(missing_slices)),
        intermediate_reader_incomplete_slices=json.dumps(list(incomplete_slices)),
    )
    _log(
        f"Modern cache {model_name}: ready; data remain lazy until compute/load/plot.",
        verbose,
    )
    return dataset


def _open_legacy_aifs_monthly_intermediates(
    model_name: str,
    *,
    monthly_root: str | Path,
    era5_daily_temperature_store: str | Path,
    era5_hazard_store: str | Path,
    forecast_days: Sequence[int] | None,
    years: Sequence[int] | None,
    months: Sequence[int] | None,
    chunks: Mapping[str, int] | str,
    verbose: bool,
) -> xr.Dataset:
    """Open and canonically align the compact monthly AIFS stores lazily."""
    root = Path(monthly_root).expanduser()
    requested_years = _validate_years(years)
    requested_months = _validate_months(months)
    stores = _legacy_aifs_monthly_stores(
        root,
        years=requested_years,
        months=requested_months,
    )
    if not stores:
        raise FileNotFoundError(f"No AIFS ENS v2 monthly stores found beneath {root}")
    expected_days = _expected_forecast_days(forecast_days, None)
    _log(
        f"Legacy AIFS monthly: {len(stores)} store(s), forecast days {list(expected_days)}, root={root}",
        verbose,
    )
    missing_slices: list[tuple[str, int]] = []
    monthly_datasets: list[xr.Dataset] = []

    for index, (partition, path) in enumerate(stores, start=1):
        _log(f"[{index}/{len(stores)}] Opening {partition}: {path}", verbose)
        try:
            dataset = xr.open_zarr(path, consolidated=False, chunks=chunks)
        except Exception:
            _log(f"[{index}/{len(stores)}] {partition}: FAILED while opening", verbose)
            raise
        if "forecast_day" not in dataset.dims:
            raise ValueError(f"Legacy AIFS monthly store lacks forecast_day: {path}")
        available_days = {int(day) for day in dataset["forecast_day"].values}
        missing = tuple(day for day in expected_days if day not in available_days)
        missing_slices.extend((partition, forecast_day) for forecast_day in missing)
        # ``reindex`` is lazy for data arrays and fills absent compact-product
        # leads with NaN, including their valid-date coordinate.
        monthly_datasets.append(dataset.reindex(forecast_day=expected_days))
        _log(
            f"[{index}/{len(stores)}] {partition}: ready "
            f"({len(expected_days) - len(missing)}/{len(expected_days)} leads)",
            verbose,
        )

    if not monthly_datasets:
        raise FileNotFoundError(f"No matching AIFS ENS v2 monthly stores found beneath {root}")
    compact = xr.concat(
        monthly_datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
        combine_attrs="override",
    ).sortby("time")
    compact = _canonicalize_dimensions(compact)
    _log("Matching legacy AIFS fields to ERA5 temperature and events lazily...", verbose)
    dataset = _canonicalize_legacy_aifs(
        compact,
        daily_temperature_store=era5_daily_temperature_store,
        hazard_store=era5_hazard_store,
        chunks=chunks,
    )
    _log(
        f"Legacy AIFS monthly report for {model_name}: {len(monthly_datasets)} month(s), "
        f"{len(missing_slices)} missing lead slice(s).",
        verbose,
    )
    if missing_slices:
        details = ", ".join(
            f"{partition}/forecast_day_{forecast_day:03d}"
            for partition, forecast_day in missing_slices
        )
        _log("Filled with NaNs: " + details, verbose)
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs.update(
        intermediate_reader_source="legacy_aifs_monthly",
        intermediate_reader_model=model_name,
        intermediate_reader_monthly_root=str(root),
        intermediate_reader_years=json.dumps(list(requested_years) if requested_years else None),
        intermediate_reader_months=json.dumps(list(requested_months) if requested_months else None),
        intermediate_reader_era5_daily_temperature_store=str(era5_daily_temperature_store),
        intermediate_reader_era5_hazard_store=str(era5_hazard_store),
        intermediate_reader_expected_forecast_days=json.dumps(list(expected_days)),
        intermediate_reader_missing_slices=json.dumps(list(missing_slices)),
    )
    _log(
        f"Legacy AIFS monthly {model_name}: ready; data remain lazy until compute/load/plot.",
        verbose,
    )
    return dataset


def _log(message: str, verbose: bool) -> None:
    if verbose:
        print(f"[intermediate-reader] {message}", flush=True)


def _legacy_aifs_monthly_stores(
    root: Path,
    *,
    years: tuple[int, ...] | None,
    months: tuple[int, ...] | None,
) -> list[tuple[str, Path]]:
    """Discover only the requested legacy monthly stores before opening Zarr."""
    stores: list[tuple[str, Path]] = []
    for path in sorted(root.glob("20??/aifs_ens_v2_heat_20????.zarr")):
        match = _AIFS_MONTHLY_STORE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        year, month = int(match.group("year")), int(match.group("month"))
        if (years is not None and year not in years) or (months is not None and month not in months):
            continue
        stores.append((f"{year:04d}-{month:02d}", path))
    return stores


def _validate_years(years: Sequence[int] | None) -> tuple[int, ...] | None:
    if years is None:
        return None
    parsed = tuple(sorted(set(int(year) for year in years)))
    if not parsed:
        raise ValueError("years must not be empty")
    return parsed


def _validate_months(months: Sequence[int] | None) -> tuple[int, ...] | None:
    if months is None:
        return None
    parsed = tuple(sorted(set(int(month) for month in months)))
    if not parsed or any(month < 1 or month > 12 for month in parsed):
        raise ValueError("months must contain calendar month numbers 1 through 12")
    return parsed


def _canonicalize_legacy_aifs(
    compact: xr.Dataset,
    *,
    daily_temperature_store: str | Path,
    hazard_store: str | Path,
    chunks: Mapping[str, int] | str,
) -> xr.Dataset:
    """Match legacy AIFS compact fields to the canonical case-cache schema."""
    needed = {
        "t2m_daily_mean_ensemble_mean",
        "hot_day_q95_probability",
        "heatwave_start_q95_2d_probability",
        "heatwave_start_q95_3d_probability",
    }
    missing = sorted(needed - set(compact.data_vars))
    if missing:
        raise KeyError(f"Legacy AIFS monthly stores are missing required variables: {missing}")
    if "valid_date" not in compact.coords:
        raise KeyError("Legacy AIFS monthly stores are missing required valid_date coordinates")

    forecast_temperature = compact["t2m_daily_mean_ensemble_mean"]
    valid_date = compact["valid_date"]
    daily_temperature = xr.open_zarr(
        daily_temperature_store,
        consolidated=True,
        chunks=chunks,
    )["t2m_daily_mean"]
    mapped_temperature = map_to_forecast_grid(daily_temperature, forecast_temperature, method="linear")
    observation_temperature = _match_with_missing_valid_dates(mapped_temperature, valid_date)

    probability_names = {
        "hot_day_q95": "hot_day_q95_probability",
        "heatwave_start_q95_2d": "heatwave_start_q95_2d_probability",
        "heatwave_start_q95_3d": "heatwave_start_q95_3d_probability",
    }
    probabilities = xr.concat(
        [compact[probability_names[event]] for event in CANONICAL_EVENTS],
        dim=xr.IndexVariable("event", list(CANONICAL_EVENTS)),
    ).rename("forecast_probability")
    hazards = xr.open_zarr(hazard_store, consolidated=True, chunks=chunks)
    observed_events: list[xr.DataArray] = []
    for event in CANONICAL_EVENTS:
        if event not in hazards:
            raise KeyError(f"ERA5 hazard store is missing required event variable: {event}")
        mapped_hazard = map_to_forecast_grid(
            hazards[event].astype(np.float32),
            forecast_temperature,
            method="nearest",
        )
        observed_events.append(_match_with_missing_valid_dates(mapped_hazard, valid_date))
    observed_event = xr.concat(
        observed_events,
        dim=xr.IndexVariable("event", list(CANONICAL_EVENTS)),
    )

    temperature_case_valid = forecast_temperature.notnull() & observation_temperature.notnull()
    event_case_valid = probabilities.notnull() & observed_event.notnull()
    return xr.Dataset(
        {
            "forecast_temperature": forecast_temperature.where(temperature_case_valid).astype(np.float32),
            "observation_temperature": observation_temperature.where(temperature_case_valid).astype(np.float32),
            "temperature_case_valid": temperature_case_valid.astype(bool),
            "forecast_probability": probabilities.where(event_case_valid).astype(np.float32),
            "observed_event": observed_event.where(event_case_valid, 0.0).astype(np.uint8),
            "event_case_valid": event_case_valid.astype(bool),
        },
        coords={"valid_date": valid_date},
        attrs=dict(compact.attrs),
    )


def _match_with_missing_valid_dates(
    observation: xr.DataArray,
    valid_date: xr.DataArray,
) -> xr.DataArray:
    """Match ERA5 while retaining all-NaN output for missing legacy leads."""
    valid = valid_date.notnull()
    fallback_date = observation["time"].isel(time=0)
    safe_valid_date = valid_date.where(valid, fallback_date)
    return match_observation_by_valid_date(observation, safe_valid_date).where(valid)


def _canonicalize_dimensions(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    renames = {
        "time": "initialization",
        "lat": "latitude",
        "lon": "longitude",
    }
    applicable = {
        old: new
        for old, new in renames.items()
        if old in data.dims or old in data.coords or (isinstance(data, xr.Dataset) and old in data.data_vars)
    }
    return data.rename(applicable)


def _manifest_expectations(
    results_root: Path,
    model_name: str,
) -> tuple[tuple[str, ...], tuple[int, ...] | None, bool]:
    manifest = results_root / "inventory" / "reforecast_inventory.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (), None, False
    models = payload.get("models", [])
    if not isinstance(models, list):
        return (), None, False
    for model in models:
        if not isinstance(model, dict) or model.get("model") != model_name:
            continue
        partitions: list[str] = []
        for item in model.get("selected_partitions", []):
            if isinstance(item, dict) and "year" in item and "month" in item:
                partitions.append(f"{int(item['year']):04d}-{int(item['month']):02d}")
        days = model.get("forecast_days")
        discovered_days = tuple(int(day) for day in days) if isinstance(days, list) else None
        return tuple(sorted(set(partitions))), discovered_days, True
    return (), None, False


def _expected_forecast_days(
    requested: Sequence[int] | None,
    manifest_days: tuple[int, ...] | None,
) -> tuple[int, ...]:
    source_days = tuple(
        int(day)
        for day in (requested if requested is not None else manifest_days or SUPPORTED_FORECAST_DAYS)
    )
    invalid = sorted(set(source_days) - set(SUPPORTED_FORECAST_DAYS))
    if requested is not None and invalid:
        raise ValueError(f"forecast_days must be within 0 through 14; got {invalid}")
    # Older manifests may retain the prior over-long source horizon.  Ignore
    # those labels during automatic discovery rather than making inspection of
    # an otherwise usable cache fail.
    days = tuple(day for day in source_days if day in SUPPORTED_FORECAST_DAYS)
    if not days:
        raise ValueError("forecast_days must not be empty")
    return tuple(sorted(set(days)))


def _partition_directories(cache_root: Path) -> dict[str, Path]:
    if not cache_root.is_dir():
        return {}
    return {
        path.name: path
        for path in cache_root.iterdir()
        if path.is_dir() and _PARTITION_PATTERN.fullmatch(path.name)
    }


def _open_partition(
    directory: Path,
    expected_days: Sequence[int],
    *,
    chunks: Mapping[str, int] | str,
) -> tuple[xr.Dataset | None, tuple[int, ...], tuple[int, ...]]:
    stores = _lead_stores(directory)
    readable: dict[int, xr.Dataset] = {}
    incomplete: list[int] = []
    for forecast_day in expected_days:
        store = stores.get(forecast_day)
        if store is None:
            continue
        if not (store / ".zmetadata").is_file():
            incomplete.append(forecast_day)
            continue
        readable[forecast_day] = _as_forecast_day_slice(
            xr.open_zarr(store, consolidated=True, chunks=chunks),
            forecast_day,
        )
    missing = tuple(forecast_day for forecast_day in expected_days if forecast_day not in readable)
    if not readable:
        return None, missing, tuple(incomplete)

    template = next(iter(readable.values()))
    slices = [
        readable.get(forecast_day, _nan_forecast_day_slice(template, forecast_day))
        for forecast_day in expected_days
    ]
    return (
        xr.concat(
            slices,
            dim="forecast_day",
            data_vars="all",
            coords="all",
            compat="override",
            join="outer",
            combine_attrs="drop_conflicts",
        ),
        missing,
        tuple(incomplete),
    )


def _lead_stores(directory: Path) -> dict[int, Path]:
    stores: dict[int, Path] = {}
    for path in directory.iterdir():
        match = _STORE_PATTERN.fullmatch(path.name)
        if path.is_dir() and match is not None:
            stores[int(match.group("forecast_day"))] = path
    return stores


def _as_forecast_day_slice(dataset: xr.Dataset, forecast_day: int) -> xr.Dataset:
    if "forecast_day" in dataset.dims:
        if dataset.sizes["forecast_day"] != 1:
            raise ValueError("A case-cache store must contain exactly one forecast_day")
        dataset = dataset.isel(forecast_day=0, drop=True)
    elif "forecast_day" in dataset.coords:
        dataset = dataset.drop_vars("forecast_day")
    return dataset.expand_dims(forecast_day=[forecast_day])


def _nan_forecast_day_slice(template: xr.Dataset, forecast_day: int) -> xr.Dataset:
    """Build a lazy all-NaN lead slice from a same-month template store."""
    variables: dict[str, xr.DataArray] = {}
    for name, values in template.data_vars.items():
        # The cache schema only contains numeric/bool science fields.  A float
        # placeholder represents missing values consistently for every field,
        # including validity masks and the uint8 observed-event encoding.
        values = values.reset_coords(
            [coordinate for coordinate in values.coords if coordinate not in values.dims],
            drop=True,
        )
        variables[name] = xr.full_like(values, fill_value=np.nan, dtype=np.float32)
    coordinates = {
        name: _missing_coordinate(values)
        for name, values in template.coords.items()
        if name not in template.dims and name != "forecast_day"
    }
    placeholder = xr.Dataset(variables, coords=coordinates)
    return placeholder.assign_coords(
        {dimension: template[dimension] for dimension in template.dims if dimension != "forecast_day"}
    ).assign_coords(forecast_day=[forecast_day])


def _missing_coordinate(values: xr.DataArray) -> xr.DataArray:
    """Return a missing-valued copy of an auxiliary coordinate, lazily."""
    if np.issubdtype(values.dtype, np.datetime64):
        return xr.full_like(values, fill_value=np.datetime64("NaT"), dtype=values.dtype)
    if np.issubdtype(values.dtype, np.timedelta64):
        return xr.full_like(values, fill_value=np.timedelta64("NaT"), dtype=values.dtype)
    if np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
        return xr.full_like(values, fill_value=np.nan, dtype=np.float32)
    return xr.full_like(values, fill_value=None, dtype=object)


def _concat_months(monthly_datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    if len(monthly_datasets) == 1:
        return monthly_datasets[0]
    for dimension in ("initialization", "time"):
        if all(dimension in dataset.dims for dataset in monthly_datasets):
            return xr.concat(
                monthly_datasets,
                dim=dimension,
                data_vars="all",
                coords="all",
                compat="override",
                join="outer",
                combine_attrs="drop_conflicts",
            )
    raise ValueError(
        "Case-cache months have no shared initialization/time dimension and cannot be concatenated"
    )
