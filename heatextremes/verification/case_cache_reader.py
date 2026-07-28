"""Lazy, gap-aware access to model case-cache Zarr stores.

The reforecast workflow writes one independent Zarr store per model, month,
and forecast day.  This module makes that layout convenient to analyse as one
lazy :class:`xarray.Dataset` without hiding unfinished work: absent lead
stores in an otherwise available month are represented by all-NaN slices and
reported when the dataset is opened.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


DEFAULT_RESULTS_ROOT = Path(
    "/net/monsoon/kylehall/ERA5/heat_extremes_reforecast_verification/verification_results"
)
SUPPORTED_FORECAST_DAYS = tuple(range(15))
_PARTITION_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])")
_STORE_PATTERN = re.compile(r"forecast_day_(?P<forecast_day>\d{3})\.zarr")


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


def open_model_case_cache(
    model_name: str,
    *,
    results_root: str | Path = DEFAULT_RESULTS_ROOT,
    forecast_days: Sequence[int] | None = None,
    chunks: Mapping[str, int] | str = "auto",
) -> xr.Dataset:
    """Open a model's case-cache stores lazily and fill missing leads with NaNs.

    The inventory manifest supplies expected months and forecast days when it
    exists.  Otherwise existing ``case_cache/YYYY-MM`` directories are used
    and the project-standard days 0--14 are expected.  Missing leads within a
    month that has at least one readable store are inserted as all-NaN slices.
    A wholly absent expected month is reported but omitted because there is no
    trustworthy initialization coordinate from which to construct its shape.

    Parameters
    ----------
    model_name:
        Result-directory/model name, for example ``aurora_e2s``.
    results_root:
        Verification results root containing the model directory and optional
        ``inventory/reforecast_inventory.json`` manifest.
    forecast_days:
        Override the discovered expected forecast days.  Values must be in the
        supported 0--14 range.
    chunks:
        Passed to :func:`xarray.open_zarr`; the default ``"auto"`` produces
        Dask-backed arrays and therefore does not read data values on open.

    Returns
    -------
    xarray.Dataset
        Cases concatenated over ``forecast_day`` and their native time
        dimension.  Coverage details are printed and also stored in dataset
        attributes as JSON.
    """
    root = Path(results_root).expanduser()
    cache_root = root / model_name / "case_cache"
    expected_partitions, manifest_days, manifest_used = _manifest_expectations(root, model_name)
    discovered_partitions = _partition_directories(cache_root)
    if not expected_partitions:
        expected_partitions = tuple(sorted(discovered_partitions))
    expected_days = _expected_forecast_days(forecast_days, manifest_days)

    missing_partitions = tuple(
        partition for partition in expected_partitions if partition not in discovered_partitions
    )
    missing_slices: list[tuple[str, int]] = []
    incomplete_slices: list[tuple[str, int]] = []
    monthly_datasets: list[xr.Dataset] = []

    for partition in expected_partitions:
        directory = discovered_partitions.get(partition)
        if directory is None:
            continue
        monthly, missing, incomplete = _open_partition(
            directory,
            expected_days,
            chunks=chunks,
        )
        missing_slices.extend((partition, forecast_day) for forecast_day in missing)
        incomplete_slices.extend((partition, forecast_day) for forecast_day in incomplete)
        if monthly is not None:
            monthly_datasets.append(monthly)

    report = CaseCacheAvailability(
        model_name=model_name,
        expected_partitions=expected_partitions,
        missing_partitions=missing_partitions,
        missing_slices=tuple(missing_slices),
        incomplete_slices=tuple(incomplete_slices),
        manifest_used=manifest_used,
    )
    print(report.format_report(), flush=True)
    if not monthly_datasets:
        raise FileNotFoundError(
            f"No readable case-cache stores found for model {model_name!r} beneath {cache_root}"
        )

    dataset = _concat_months(monthly_datasets)
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs.update(
        case_cache_reader_model=model_name,
        case_cache_reader_results_root=str(root),
        case_cache_reader_expected_forecast_days=json.dumps(list(expected_days)),
        case_cache_reader_missing_partitions=json.dumps(list(missing_partitions)),
        case_cache_reader_missing_slices=json.dumps(list(missing_slices)),
        case_cache_reader_incomplete_slices=json.dumps(list(incomplete_slices)),
    )
    return dataset


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
    placeholder = xr.Dataset(variables)
    return placeholder.assign_coords(
        {dimension: template[dimension] for dimension in template.dims if dimension != "forecast_day"}
    ).assign_coords(forecast_day=[forecast_day])


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
