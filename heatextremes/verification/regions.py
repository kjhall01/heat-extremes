"""Configuration-driven geographic masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr
import yaml


@dataclass(frozen=True)
class Region:
    """A rectangular region expressed in conventional -180..180 longitudes."""

    name: str
    latitude_min: float | None = None
    latitude_max: float | None = None
    longitude_min: float | None = None
    longitude_max: float | None = None
    land_only: bool = False


def load_regions(path: str | Path) -> dict[str, Region]:
    """Load named region definitions from YAML."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    entries = raw.get("regions", raw)
    if not isinstance(entries, Mapping):
        raise ValueError("Region YAML must contain a 'regions' mapping")

    result: dict[str, Region] = {}
    for name, value in entries.items():
        value = value or {}
        if not isinstance(value, Mapping):
            raise ValueError(f"Region {name!r} must be a mapping")
        latitude = value.get("latitude")
        longitude = value.get("longitude")
        if latitude is not None and (not isinstance(latitude, list) or len(latitude) != 2):
            raise ValueError(f"Region {name!r} latitude must be [south, north]")
        if longitude is not None and (not isinstance(longitude, list) or len(longitude) != 2):
            raise ValueError(f"Region {name!r} longitude must be [west, east]")
        result[str(name)] = Region(
            name=str(name),
            latitude_min=None if latitude is None else float(latitude[0]),
            latitude_max=None if latitude is None else float(latitude[1]),
            longitude_min=None if longitude is None else float(longitude[0]),
            longitude_max=None if longitude is None else float(longitude[1]),
            land_only=bool(value.get("land_only", False)),
        )
    if not result:
        raise ValueError("No regions were configured")
    return result


def canonical_longitude(longitude: xr.DataArray) -> xr.DataArray:
    """Return longitude values normalized to ``[-180, 180)`` without reordering data."""
    return ((longitude + 180.0) % 360.0) - 180.0


def region_mask(
    reference: xr.DataArray | xr.Dataset,
    region: Region,
    *,
    land_mask: xr.DataArray | None = None,
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
) -> xr.DataArray:
    """Build a latitude/longitude mask for a region.

    This mask-based implementation works for either latitude orientation and
    avoids assuming the source longitude convention.  A west bound greater
    than the east bound selects a seam-crossing region.
    """
    if latitude_name not in reference.coords or longitude_name not in reference.coords:
        raise ValueError("Reference must provide latitude and longitude coordinates")
    latitude = reference[latitude_name]
    longitude = canonical_longitude(reference[longitude_name])
    mask = xr.ones_like(latitude, dtype=bool) * xr.ones_like(longitude, dtype=bool)
    if region.latitude_min is not None:
        mask = mask & (latitude >= region.latitude_min) & (latitude <= region.latitude_max)
    if region.longitude_min is not None:
        west, east = region.longitude_min, region.longitude_max
        if west <= east:
            mask = mask & (longitude >= west) & (longitude <= east)
        else:
            mask = mask & ((longitude >= west) | (longitude <= east))
    if region.land_only:
        if land_mask is None:
            raise ValueError(
                f"Region {region.name!r} requests land_only but no compatible land mask is configured"
            )
        land_mask, mask = xr.align(land_mask.astype(bool), mask, join="exact")
        mask = mask & land_mask
    return mask.rename("region_mask")


def select_regions(
    all_regions: Mapping[str, Region], names: list[str] | None,
) -> dict[str, Region]:
    """Return all configured regions or a validated named subset."""
    if not names:
        return dict(all_regions)
    unknown = sorted(set(names) - set(all_regions))
    if unknown:
        raise KeyError(f"Unknown regions: {unknown}; available={sorted(all_regions)}")
    return {name: all_regions[name] for name in names}
