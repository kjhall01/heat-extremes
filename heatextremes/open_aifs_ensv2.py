from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


def open_aifs_ensv2():
    root = Path("/net/monsoon/marchakitus/AIFS/v2p0/combined/forecasts_AIFS_ENS_v2")
    paths = sorted(root.glob("*.zarr"))
    for path in paths:
        assert path.is_dir(), f'{path} does not exist?'
    wanted = ["2d", "2t"]

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
            "lat": "latitude",
            "lon": "longitude",
        }
    )


def daily_aifs_aggregates(ds: xr.Dataset, max_days: int | None = None) -> xr.Dataset:
    """Return daily mean and maximum 2m temperature from 6-hourly forecasts.

    Forecast steps are assumed to be at 00, 06, 12, and 18 UTC, starting at
    zero. The output uses ``prediction_timedelta`` so it can be passed directly
    to the verification metrics.
    """
    if "2m_temperature" not in ds:
        raise KeyError("Dataset is missing 2m_temperature")
    if "step" not in ds.dims:
        raise ValueError("Dataset must have a step dimension")
    if max_days is not None:
        if max_days < 1:
            raise ValueError("max_days must be at least 1")
        ds = ds.where(ds.step < np.timedelta64(max_days, "D"), drop=True)

    temperature = ds["2m_temperature"]
    return xr.Dataset(
        {
            "t2m_mean_6h": temperature.resample(step="1D").mean(),
            "t2m_max_6h": temperature.resample(step="1D").max(),
        }
    ).rename({"step": "prediction_timedelta"})
