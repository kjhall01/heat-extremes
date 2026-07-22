from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.open_cached_era5 import daily_era5_aggregates


def test_daily_era5_aggregates_aligns_end_labelled_precipitation() -> None:
    time = np.arange(
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-01-02T06:00:00"),
        np.timedelta64(6, "h"),
    ).astype("datetime64[ns]")
    ds = xr.Dataset(
        {
            "2m_temperature": ("time", np.arange(time.size, dtype="float32")),
            "total_precipitation": ("time", np.ones(time.size, dtype="float32")),
        },
        coords={"time": time},
    )

    daily = daily_era5_aggregates(ds)

    assert daily["t2m_min_6h"].sel(time="2024-01-01").item() == 0
    assert daily["total_precipitation"].sel(time="2024-01-01").item() == 4
    assert np.isnan(daily["total_precipitation"].sel(time="2024-01-02").item())
