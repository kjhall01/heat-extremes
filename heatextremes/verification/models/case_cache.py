"""Adapter for source-independent, casewise verification-cache Zarr stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import xarray as xr

from ..alignment import assert_exact_case_alignment
from ..case_cache import case_cache_completion_is_valid, case_cache_lead_path, case_cache_partition_directory
from ..config import VerificationConfig
from ..events import CANONICAL_EVENTS
from .base import CanonicalLead, ModelAdapter, ModelCapabilities


@dataclass
class _CaseCachePartition:
    directory: Path
    year: int
    month: int
    sources: list[xr.Dataset] = field(default_factory=list)


class CaseCacheAdapter(ModelAdapter):
    """Read canonical cases without opening native forecasts or ERA5 sources."""

    def __init__(self, config: VerificationConfig):
        self.config = config
        ensemble = bool(config.data["model"].get("ensemble", False))
        self.capabilities = ModelCapabilities(
            ensemble=ensemble,
            member_temperature=False,
            # Cache metadata are intentionally not opened during dry run.
            # Individual lead stores determine this capability at runtime.
            interval_quantiles=False,
        )

    def dry_run_paths(self, year: int, month: int) -> Mapping[str, Path]:
        return {"case_cache_partition": case_cache_partition_directory(self.config, year, month)}

    def open_partition(self, year: int, month: int) -> _CaseCachePartition:
        directory = case_cache_partition_directory(self.config, year, month)
        if not case_cache_completion_is_valid(directory, self.config, year=year, month=month):
            raise FileNotFoundError(
                "Verified case-cache partition is missing, incomplete, or incompatible: " f"{directory}"
            )
        return _CaseCachePartition(directory, year, month)

    def lead(self, opened: _CaseCachePartition, forecast_day: int) -> CanonicalLead:
        path = case_cache_lead_path(
            self.config,
            opened.year,
            opened.month,
            forecast_day,
        )
        chunks = dict(self.config.data.get("case_cache", {}).get("read_chunks", {})) or None
        dataset = xr.open_zarr(path, consolidated=True, chunks=chunks)
        opened.sources.append(dataset)
        required = {
            "forecast_temperature",
            "observation_temperature",
            "temperature_case_valid",
            "forecast_probability",
            "observed_event",
            "event_case_valid",
        }
        missing = sorted(required - set(dataset.data_vars))
        if missing:
            raise KeyError(f"Case-cache store {path} is missing variables: {missing}")
        events = set(str(value) for value in dataset["event"].values)
        missing_events = sorted(set(CANONICAL_EVENTS) - events)
        if missing_events:
            raise KeyError(f"Case-cache store {path} is missing events: {missing_events}")
        temperature_valid = dataset["temperature_case_valid"].astype(bool)
        forecast = dataset["forecast_temperature"].where(temperature_valid)
        observation = dataset["observation_temperature"].where(temperature_valid)
        event_valid = dataset["event_case_valid"].astype(bool)
        probabilities: dict[str, xr.DataArray] = {}
        observed_events: dict[str, xr.DataArray] = {}
        for event in CANONICAL_EVENTS:
            valid = event_valid.sel(event=event)
            probabilities[event] = dataset["forecast_probability"].sel(event=event).where(valid)
            observed_events[event] = dataset["observed_event"].sel(event=event).where(valid) > 0.5
        interval_quantiles = (
            dataset["forecast_temperature_quantile"]
            if "forecast_temperature_quantile" in dataset
            else None
        )
        self.capabilities = ModelCapabilities(
            ensemble=bool(dataset.attrs.get("ensemble", False)),
            member_temperature=False,
            interval_quantiles=interval_quantiles is not None,
        )
        arrays = [forecast, observation, observed_events["hot_day_q95"], *probabilities.values()]
        assert_exact_case_alignment(*arrays)
        return CanonicalLead(
            forecast_day=forecast_day,
            ensemble_mean_temperature=forecast,
            observation_temperature=observation,
            observed_hot=observed_events["hot_day_q95"],
            event_probabilities=probabilities,
            observed_events=observed_events,
            interval_quantiles=interval_quantiles,
            valid_date=dataset.coords.get("valid_date"),
        )

    def close_partition(self, opened: _CaseCachePartition) -> None:
        for source in opened.sources:
            source.close()
        opened.sources.clear()

    def release_lead(self, opened: _CaseCachePartition) -> None:
        if opened.sources:
            opened.sources.pop().close()
