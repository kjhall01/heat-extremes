from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


def open_aifs_ensv2():
    root = Path("/net/monsoon/marchakitus/AIFS/v2p0/combined/forecasts_AIFS_ENS_v2")
    paths = sorted(root.glob("*.zarr"))
    for path in paths:
        assert path.is_dir(), f'{path} does not exist?'
    wanted = ["2d", "2t", 'tp']

    with ProgressBar():
        ds = xr.open_mfdataset(
            paths,
            engine="zarr",
            combine="nested",
            concat_dim="time",
            preprocess=lambda x: x[wanted],
            chunks={
                "time": 1,  # unavoidable: one time per store
                "number": 26,  # combine all ensemble members
                "step": 24,
                "latitude": 180,
                "longitude": 180,
            },
            parallel=True,
            data_vars="all",
            coords="minimal",
            compat="override",
            join="override",
            combine_attrs="override",
            consolidated=None,
        )

    return ds.rename(
        {
            "2d": "2m_dewpoint_temperature",
            "2t": "2m_temperature",
            'tp': 'total_precipitation',
            "lat": "latitude",
            "lon": "longitude",
        }
    )


def daily_aifs_aggregates(
    ds: xr.Dataset,
    max_days: int | None = None,
    variable: str = "2m_temperature",
    step_dim: str = "prediction_timedelta",
    output_step_dim: str = "prediction_timedelta",
) -> xr.Dataset:
    """Return daily mean and maximum temperature from 6-hourly forecasts."""
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
            "t2m_mean_6h": temperature.resample(**{step_dim: "1D"}).mean(),
            "t2m_max_6h": temperature.resample(**{step_dim: "1D"}).max(),
        }
    )

    if step_dim != output_step_dim:
        daily = daily.rename({step_dim: output_step_dim})

    return daily