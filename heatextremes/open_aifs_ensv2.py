import xarray as xr 
import gcsfs
from dask.diagnostics import ProgressBar
from pathlib import Path
import numpy as np

def open_aifs_ensv2():
    fs = gcsfs.GCSFileSystem(token="anon")
    era5_store = fs.get_mapper(
        "weatherbench2/datasets/era5/"
        "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    )

    paths = sorted(Path("/net/monsoon/marchakitus/AIFS/v2p0/combined/forecasts_AIFS_ENS_v2").glob("*.zarr"))
    wanted = ['2d', '2t']

    with ProgressBar():
        ds = xr.open_mfdataset(
            paths,
            engine="zarr",
            combine="nested",
            concat_dim="time",
            preprocess=lambda x: x[wanted],
            chunks={
                "time": 1,       # unavoidable: one time per store
                "number": 26,    # combine all ensemble members
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

    ds = ds.rename({'2d': '2m_dewpoint_temperature', '2t': '2m_temperature', 'lat': 'latitude', 'lon': 'longitude' })
    return ds 