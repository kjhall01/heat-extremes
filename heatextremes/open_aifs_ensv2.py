import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


DEFAULT_ROOT = Path("/net/monsoon/marchakitus/AIFS/v2p0/combined/forecasts_AIFS_ENS_v2")
DEFAULT_VARIABLES = ("2d", "2t", "tp")


def open_aifs_ensv2(
    root: str | Path = DEFAULT_ROOT,
    *,
    years: Iterable[int] | None = None,
    variables: Iterable[str] = DEFAULT_VARIABLES,
    chunks: dict[str, int] | None = None,
):
    """Open AIFS ENS v2 forecast stores, optionally restricted to years.

    Each source Zarr store represents one initialization time.  ``years`` is
    matched against the four-digit year embedded in a store name, avoiding the
    need to open every available year before selecting a verification period.
    """
    root = Path(root)
    paths = sorted(root.glob("*.zarr"))
    if years is not None:
        wanted_years = {int(year) for year in years}
        paths = [path for path in paths if _forecast_store_year(path) in wanted_years]
    if not paths:
        selected = "all years" if years is None else f"years {sorted(wanted_years)}"
        raise FileNotFoundError(f"No AIFS ENS v2 Zarr stores found under {root} for {selected}")
    for path in paths:
        assert path.is_dir(), f'{path} does not exist?'
    wanted = list(variables)
    dask_chunks = (
        {
            "time": 1,  # unavoidable: one time per store
            "number": 26,  # combine all ensemble members
            "prediction_timedelta": 24,
            "latitude": 180,
            "longitude": 180,
        }
        if chunks is None
        else chunks
    )

    with ProgressBar():
        ds = xr.open_mfdataset(
            paths,
            engine="zarr",
            combine="nested",
            concat_dim="time",
            preprocess=lambda x: x[wanted],
            chunks=dask_chunks,
            parallel=True,
            data_vars="all",
            coords="minimal",
            compat="override",
            join="override",
            combine_attrs="override",
            consolidated=None,
        )

    rename = {
        "2d": "2m_dewpoint_temperature",
        "2t": "2m_temperature",
        "tp": "total_precipitation",
        "lat": "latitude",
        "lon": "longitude",
    }
    return ds.rename({source: target for source, target in rename.items() if source in ds})


def _forecast_store_year(path: Path) -> int | None:
    """Extract the first plausible four-digit year from an AIFS store name."""
    match = re.search(r"(?:19|20)\d{2}", path.name)
    return None if match is None else int(match.group())


def daily_aifs_aggregates(
    ds: xr.Dataset,
    max_days: int | None = None,
    variable: str = "2m_temperature",
    step_dim: str = "prediction_timedelta",
    output_step_dim: str = "prediction_timedelta",
) -> xr.Dataset:
    """Return daily minimum, mean, and maximum temperature from 6-hourly forecasts."""
    if variable not in ds:
        raise KeyError(f"Dataset is missing {variable}")
    if step_dim not in ds.dims:
        raise ValueError(f"Dataset must have a {step_dim} dimension")
    if step_dim not in ds.coords:
        raise ValueError(f"Dataset must have a {step_dim} coordinate")

    temperature = ds[variable]
    if max_days is not None:
        if max_days < 1:
            raise ValueError("max_days must be at least 1")
        temperature = temperature.where(
            temperature[step_dim] < np.timedelta64(max_days, "D"),
            drop=True,
        )

    daily = xr.Dataset(
        {
            "t2m_min_6h": temperature.resample(**{step_dim: "1D"}).min(),
            "t2m_mean_6h": temperature.resample(**{step_dim: "1D"}).mean(),
            "t2m_max_6h": temperature.resample(**{step_dim: "1D"}).max(),
        }
    )

    if step_dim != output_step_dim:
        daily = daily.rename({step_dim: output_step_dim})

    return daily
