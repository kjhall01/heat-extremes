from pathlib import Path

import xarray as xr
from dask.diagnostics import ProgressBar


def open_aifs_ensv2():
    root = Path("/net/monsoon/marchakitus/AIFS/v2p0/combined/forecasts_AIFS_ENS_v2")
    paths = sorted(root.glob("*.zarr"))
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
