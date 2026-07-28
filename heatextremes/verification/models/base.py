"""Source-independent canonical forecast adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import xarray as xr


@dataclass(frozen=True)
class ModelCapabilities:
    """Features that must be explicitly available rather than assumed."""

    ensemble: bool
    member_temperature: bool
    interval_quantiles: bool


@dataclass
class CanonicalLead:
    """Canonical fields for one bounded forecast lead.

    All scored arrays use ``initialization, latitude, longitude`` dimensions.
    ``event_probabilities`` and ``observed_events`` are keyed by canonical
    event names and may be supplied by ensemble or deterministic adapters.
    """

    forecast_day: int
    ensemble_mean_temperature: xr.DataArray
    observation_temperature: xr.DataArray
    observed_hot: xr.DataArray
    event_probabilities: Mapping[str, xr.DataArray]
    observed_events: Mapping[str, xr.DataArray]
    interval_quantiles: xr.DataArray | None = None
    valid_date: xr.DataArray | None = None


class ModelAdapter(ABC):
    """Small interface separating data-source details from metric implementations."""

    capabilities: ModelCapabilities

    @abstractmethod
    def dry_run_paths(self, year: int, month: int) -> Mapping[str, Path]:
        """Resolve the exact paths a partition would use without opening arrays."""

    @abstractmethod
    def open_partition(self, year: int, month: int):
        """Open a bounded month of compact source products lazily."""

    @abstractmethod
    def lead(self, opened_partition, forecast_day: int) -> CanonicalLead:
        """Construct one canonical lead with exactly aligned forecast/observations."""

    @abstractmethod
    def close_partition(self, opened_partition) -> None:
        """Close source datasets and release file handles."""

    def release_lead(self, opened_partition) -> None:
        """Release an optional per-lead resource after bounded reduction.

        Most source adapters keep one month-level store open and need no
        action.  Cache adapters open one independent Zarr store per lead and
        override this hook to avoid retaining unnecessary file handles.
        """
