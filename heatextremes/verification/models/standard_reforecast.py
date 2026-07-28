"""Adapter for standard raw ensemble reforecast directories.

The source is read directly from ``init_*.zarr`` stores.  It reuses the
verified local-solar daily-temperature and threshold definitions from the
historical AIFS monthly processor, but never writes a reconstructed forecast
or matched-ERA5 cube.  Only the current forecast lead is reduced and scored.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr

from ..alignment import assert_exact_case_alignment, map_to_forecast_grid, match_observation_by_valid_date
from ..config import VerificationConfig
from ..events import CANONICAL_EVENTS
from .base import CanonicalLead, ModelAdapter, ModelCapabilities


def _processing_helpers():
    """Load verified functions without converting the historical script to a package."""
    source = Path(__file__).resolve().parents[3] / "scripts" / "process_aifs_heat_month.py"
    spec = importlib.util.spec_from_file_location("verified_heat_month_processor", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load verified heat processor: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _RawReforecastPartition:
    raw: xr.Dataset
    daily: xr.DataArray
    threshold_dataset: xr.Dataset
    threshold: xr.DataArray
    daily_observation: xr.DataArray
    hazards: xr.Dataset


class StandardReforecastAdapter(ModelAdapter):
    """Map compatible raw ``init_*.zarr`` stores to canonical lead fields."""

    def __init__(self, config: VerificationConfig):
        self.config = config
        model = config.data["model"]
        ensemble = bool(model.get("ensemble", True))
        self.capabilities = ModelCapabilities(
            ensemble=ensemble,
            member_temperature=ensemble,
            interval_quantiles=ensemble,
        )
        self._helpers = _processing_helpers()

    def dry_run_paths(self, year: int, month: int) -> Mapping[str, Path]:
        paths = self.config.data["paths"]
        return {
            "raw_reforecast_directory": Path(paths["raw_reforecast_root"]),
            "thresholds": Path(paths["threshold_store"]),
            "era5_daily": Path(paths["era5_daily_temperature_store"]),
            "era5_hazards": Path(paths["era5_hazard_store"]),
        }

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

    def open_partition(self, year: int, month: int) -> _RawReforecastPartition:
        paths = self.config.data["paths"]
        source_root = Path(paths["raw_reforecast_root"])
        source_variable = str(self.config.data["variables"].get("raw_source_temperature", "2t"))
        raw = self._helpers.open_aifs_month(
            source_root,
            year,
            month,
            chunks=dict(self.config.data.get("chunking", {}).get("raw", {})) or None,
            source_variable=source_variable,
            require_member_dimension=bool(self.config.data["model"].get("ensemble", True)),
        )
        member_dimension = "number"
        has_members = member_dimension in raw.dims
        configured_ensemble = bool(self.config.data["model"].get("ensemble", True))
        if configured_ensemble and not has_members:
            raw.close()
            raise ValueError("model.ensemble is true but the raw store has no 'number' member dimension")
        if not configured_ensemble and has_members:
            if raw.sizes[member_dimension] == 1:
                # Some deterministic producers retain a singleton member axis.
                # Treat it as deterministic rather than rejecting a valid
                # 0/1 event forecast or inventing an ensemble diagnostic.
                raw = raw.isel({member_dimension: 0}, drop=True)
                has_members = False
            else:
                raw.close()
                raise ValueError(
                    "model.ensemble is false but the raw store contains multiple 'number' members; "
                    "correct the metadata/configuration"
                )
        self.capabilities = ModelCapabilities(
            ensemble=has_members,
            member_temperature=has_members,
            interval_quantiles=has_members,
        )

        max_days = max(self.config.forecast_days) + 1
        daily = self._helpers.local_solar_daily_mean_forecast(
            raw[self._helpers.TEMPERATURE_NAME], max_days=max_days
        ).rename({"time": "initialization", "number": "member"} if has_members else {"time": "initialization"})

        threshold_dataset = xr.open_zarr(
            Path(paths["threshold_store"]),
            consolidated=True,
            chunks=dict(self.config.data.get("chunking", {}).get("thresholds", {})) or None,
        )
        threshold_variable = str(
            self.config.data["variables"].get(
                "threshold_temperature", "t2m_daily_mean_calendar_day_percentile"
            )
        )
        if threshold_variable not in threshold_dataset:
            raw.close()
            threshold_dataset.close()
            raise KeyError(f"Threshold store is missing {threshold_variable!r}")
        percentile = float(self.config.data.get("events", {}).get("percentile", 95.0))
        threshold = threshold_dataset[threshold_variable]
        percentile_dimensions = [name for name in ("percentiles", "percentile") if name in threshold.dims]
        if len(percentile_dimensions) > 1:
            raw.close()
            threshold_dataset.close()
            raise ValueError(
                f"Threshold {threshold_variable!r} has ambiguous percentile dimensions: {percentile_dimensions}"
            )
        if percentile_dimensions:
            # The verified climatology notebook writes ``percentiles``.
            # Accept the singular spelling too for compatible future stores.
            threshold = threshold.sel({percentile_dimensions[0]: percentile}, method="nearest", drop=True)
        threshold = self._helpers.map_threshold_to_forecast_grid(threshold, daily)

        observations = self.config.data["observations"]
        daily_observation = xr.open_zarr(
            Path(paths["era5_daily_temperature_store"]),
            consolidated=True,
            chunks=dict(self.config.data.get("chunking", {}).get("observations", {})) or None,
        )[observations["daily_temperature_variable"]]
        hazards = xr.open_zarr(
            Path(paths["era5_hazard_store"]),
            consolidated=True,
            chunks=dict(self.config.data.get("chunking", {}).get("observations", {})) or None,
        )
        return _RawReforecastPartition(raw, daily, threshold_dataset, threshold, daily_observation, hazards)

    @staticmethod
    def _threshold_for_dates(threshold: xr.DataArray, valid_date: xr.DataArray) -> xr.DataArray:
        safe_dayofyear = valid_date.dt.dayofyear.fillna(1).astype(np.int16)
        selected = threshold.sel(dayofyear=safe_dayofyear).where(valid_date.notnull())
        return selected.transpose("initialization", "forecast_day", "latitude", "longitude")

    @staticmethod
    def _member_mean(values: xr.DataArray) -> xr.DataArray:
        return values.mean("member", skipna=True) if "member" in values.dims else values

    def _hot_members(self, opened: _RawReforecastPartition, days: list[int]) -> tuple[xr.DataArray, xr.DataArray]:
        daily = opened.daily.sel(forecast_day=days)
        threshold = self._threshold_for_dates(opened.threshold, daily["valid_date"])
        valid = daily.notnull() & threshold.notnull()
        return (daily > threshold).astype(np.float32).where(valid), valid

    def _event_probability(
        self,
        opened: _RawReforecastPartition,
        forecast_day: int,
        duration: int,
        template: xr.DataArray,
    ) -> xr.DataArray:
        available = {int(value) for value in opened.daily["forecast_day"].values}
        required = [forecast_day - 1, *range(forecast_day, forecast_day + duration)]
        if any(day not in available for day in required):
            return xr.full_like(template, np.nan, dtype=np.float32)
        hot, valid = self._hot_members(opened, required)
        future_days = list(range(forecast_day, forecast_day + duration))
        qualifies = hot.sel(forecast_day=future_days).fillna(False).astype(bool).all("forecast_day")
        future_valid = valid.sel(forecast_day=future_days).fillna(False).astype(bool).all("forecast_day")
        previous_hot = hot.sel(forecast_day=forecast_day - 1).fillna(False).astype(bool)
        previous_valid = valid.sel(forecast_day=forecast_day - 1).fillna(False).astype(bool)
        onset = (qualifies & ~previous_hot).astype(np.float32).where(future_valid & previous_valid)
        return self._member_mean(onset).astype(np.float32)

    def lead(self, opened: _RawReforecastPartition, forecast_day: int) -> CanonicalLead:
        if forecast_day not in opened.daily["forecast_day"].values:
            raise KeyError(f"Forecast day {forecast_day} is not available in raw reforecast product")
        daily = opened.daily.sel(forecast_day=forecast_day)
        forecast = self._member_mean(daily)
        valid_date = daily["valid_date"]
        daily_observation = map_to_forecast_grid(opened.daily_observation, forecast, method="linear")
        observation_temperature = match_observation_by_valid_date(daily_observation, valid_date)

        hot_members, _ = self._hot_members(opened, [forecast_day])
        probabilities: dict[str, xr.DataArray] = {
            "hot_day_q95": self._member_mean(hot_members.sel(forecast_day=forecast_day)).astype(np.float32),
            "heatwave_start_q95_2d": self._event_probability(opened, forecast_day, 2, forecast),
            "heatwave_start_q95_3d": self._event_probability(opened, forecast_day, 3, forecast),
        }
        observed_events: dict[str, xr.DataArray] = {}
        for event in CANONICAL_EVENTS:
            variable = self.config.data["observations"]["event_variables"][event]
            hazard = map_to_forecast_grid(opened.hazards[variable].astype(np.float32), forecast, method="nearest")
            observed_events[event] = match_observation_by_valid_date(hazard, valid_date) > 0.5

        interval_quantiles = None
        if self.capabilities.interval_quantiles:
            levels = sorted(
                {
                    quantile
                    for coverage in self.config.interval_levels
                    for quantile in ((1.0 - coverage) / 2.0, 1.0 - (1.0 - coverage) / 2.0)
                }
            )
            interval_quantiles = daily.quantile(levels, dim="member").transpose(
                "initialization", "quantile", "latitude", "longitude"
            )
        arrays = [forecast, observation_temperature, observed_events["hot_day_q95"], *probabilities.values()]
        assert_exact_case_alignment(*arrays)
        return CanonicalLead(
            forecast_day=forecast_day,
            ensemble_mean_temperature=forecast,
            observation_temperature=observation_temperature,
            observed_hot=observed_events["hot_day_q95"],
            event_probabilities=probabilities,
            observed_events=observed_events,
            interval_quantiles=interval_quantiles,
            valid_date=valid_date,
        )

    def close_partition(self, opened: _RawReforecastPartition) -> None:
        for source in (opened.raw, opened.threshold_dataset, opened.daily_observation, opened.hazards):
            close = getattr(source, "close", None)
            if close is not None:
                close()
