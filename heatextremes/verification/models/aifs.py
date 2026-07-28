"""Adapter for the repository's compact AIFS ENS v2 heat-hazard products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import xarray as xr

from ..alignment import assert_exact_case_alignment, map_to_forecast_grid, match_observation_by_valid_date
from ..config import VerificationConfig
from .base import CanonicalLead, ModelAdapter, ModelCapabilities


CANONICAL_EVENTS = (
    "hot_day_q95",
    "heatwave_start_q95_2d",
    "heatwave_start_q95_3d",
)


@dataclass
class _AIFSPartition:
    compact: xr.Dataset
    daily_temperature: xr.DataArray
    hazards: xr.Dataset
    quantile_cache: dict[int, xr.DataArray]
    quantile_sources: list[xr.Dataset]


class AIFSEnsV2Adapter(ModelAdapter):
    """Translate existing AIFS monthly products to canonical fields.

    The compact monthly product does not contain member temperatures.  It is
    still an ensemble forecast for probability metrics, but interval coverage
    is exposed only when the optional selected-quantile preprocessing files
    have been configured and are present.
    """

    def __init__(self, config: VerificationConfig):
        self.config = config
        model = config.data["model"]
        interval_pattern = config.data["paths"].get("interval_quantile_file_pattern")
        self.capabilities = ModelCapabilities(
            ensemble=bool(model.get("ensemble", True)),
            member_temperature=bool(model.get("member_temperature_available", False)),
            interval_quantiles=bool(interval_pattern),
        )

    def _format_path(self, template: str, year: int, month: int, **extra: object) -> Path:
        fields = {
            "year": year,
            "month": month,
            "model": self.config.model_name,
            **extra,
        }
        return Path(template.format(**fields))

    def dry_run_paths(self, year: int, month: int) -> Mapping[str, Path]:
        paths = self.config.data["paths"]
        return {
            "compact_forecast": self._format_path(paths["compact_monthly_store_pattern"], year, month),
            "era5_daily": Path(paths["era5_daily_temperature_store"]),
            "era5_hazards": Path(paths["era5_hazard_store"]),
        }

    def open_partition(self, year: int, month: int) -> _AIFSPartition:
        paths = self.dry_run_paths(year, month)
        compact_path = paths["compact_forecast"]
        if not compact_path.is_dir():
            raise FileNotFoundError(f"AIFS compact monthly store is missing: {compact_path}")
        chunks = dict(self.config.data.get("chunking", {}).get("compact", {}))
        compact = xr.open_zarr(compact_path, consolidated=False, chunks=chunks or None)
        compact = self._canonicalize(compact)
        mappings = self.config.data["variables"]
        needed = [mappings["ensemble_mean_temperature"]] + [
            mappings["event_probabilities"][event] for event in CANONICAL_EVENTS
        ]
        missing = [name for name in needed if name not in compact]
        if missing:
            compact.close()
            raise KeyError(f"Compact AIFS store is missing configured variables: {missing}")
        if "valid_date" not in compact.coords:
            compact.close()
            raise KeyError("Compact AIFS store is missing required valid_date coordinate")

        observations = self.config.data["observations"]
        daily = xr.open_zarr(
            paths["era5_daily"],
            consolidated=True,
            chunks=dict(self.config.data.get("chunking", {}).get("observations", {})) or None,
        )[observations["daily_temperature_variable"]]
        hazards = xr.open_zarr(
            paths["era5_hazards"],
            consolidated=True,
            chunks=dict(self.config.data.get("chunking", {}).get("observations", {})) or None,
        )
        # Keep ERA5's lookup axis named ``time``.  The alignment helper then
        # replaces it with canonical ``initialization`` from valid_date.
        return _AIFSPartition(compact, daily, hazards, {}, [])

    @staticmethod
    def _canonicalize(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        renames = {
            "time": "initialization",
            "lat": "latitude",
            "lon": "longitude",
            "number": "member",
        }
        applicable = {
            old: new
            for old, new in renames.items()
            if old in data.dims or old in data.coords or (isinstance(data, xr.Dataset) and old in data.data_vars)
        }
        return data.rename(applicable)

    def _interval_quantiles(self, opened: _AIFSPartition, forecast_day: int) -> xr.DataArray | None:
        pattern = self.config.data["paths"].get("interval_quantile_file_pattern")
        if not pattern:
            return None
        if forecast_day in opened.quantile_cache:
            return opened.quantile_cache[forecast_day]
        initialization = opened.compact["initialization"]
        # The monthly compact coordinate supplies year/month reliably and is cheap to inspect.
        year = int(initialization.dt.year.values[0])
        month = int(initialization.dt.month.values[0])
        path = self._format_path(pattern, year, month, forecast_day=forecast_day)
        if not path.is_file():
            return None
        variable = self.config.data["variables"].get(
            "interval_quantile_temperature", "ensemble_temperature_quantile"
        )
        dataset = xr.open_dataset(
            path, chunks=dict(self.config.data.get("chunking", {}).get("quantiles", {})) or None
        )
        if variable not in dataset:
            dataset.close()
            raise KeyError(f"Interval-quantile file {path} is missing {variable!r}")
        # Keep only the small NetCDF handle alive.  Values remain lazy until
        # the current lead's bounded reduction is computed.
        values = self._canonicalize(dataset[variable])
        opened.quantile_sources.append(dataset)
        opened.quantile_cache[forecast_day] = values
        return values

    def lead(self, opened: _AIFSPartition, forecast_day: int) -> CanonicalLead:
        compact = opened.compact
        if forecast_day not in compact["forecast_day"].values:
            raise KeyError(f"Forecast day {forecast_day} is not available in compact AIFS product")
        mappings = self.config.data["variables"]
        forecast = compact[mappings["ensemble_mean_temperature"]].sel(forecast_day=forecast_day)
        valid_date = compact["valid_date"].sel(forecast_day=forecast_day)
        daily = map_to_forecast_grid(opened.daily_temperature, forecast, method="linear")
        observation_temperature = match_observation_by_valid_date(daily, valid_date)

        observed_events: dict[str, xr.DataArray] = {}
        probabilities: dict[str, xr.DataArray] = {}
        for event in CANONICAL_EVENTS:
            probabilities[event] = compact[mappings["event_probabilities"][event]].sel(
                forecast_day=forecast_day
            )
            hazard = opened.hazards[self.config.data["observations"]["event_variables"][event]]
            hazard = map_to_forecast_grid(hazard.astype("float32"), forecast, method="nearest")
            observed_events[event] = match_observation_by_valid_date(hazard, valid_date) > 0.5

        observed_hot = observed_events["hot_day_q95"]
        arrays = [forecast, observation_temperature, observed_hot, *probabilities.values()]
        assert_exact_case_alignment(*arrays)
        return CanonicalLead(
            forecast_day=forecast_day,
            ensemble_mean_temperature=forecast,
            observation_temperature=observation_temperature,
            observed_hot=observed_hot,
            event_probabilities=probabilities,
            observed_events=observed_events,
            interval_quantiles=self._interval_quantiles(opened, forecast_day),
        )

    def close_partition(self, opened: _AIFSPartition) -> None:
        for dataset in (opened.compact, opened.daily_temperature, opened.hazards):
            close = getattr(dataset, "close", None)
            if close is not None:
                close()
        opened.quantile_cache.clear()
        for dataset in opened.quantile_sources:
            dataset.close()
        opened.quantile_sources.clear()
