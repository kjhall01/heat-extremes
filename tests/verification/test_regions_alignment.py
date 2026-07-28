from __future__ import annotations

import numpy as np
import xarray as xr

from heatextremes.verification.alignment import match_observation_by_valid_date
from heatextremes.verification.regions import Region, region_mask


def test_region_mask_handles_descending_latitude_and_seam_crossing_longitudes() -> None:
    reference = xr.DataArray(
        np.ones((3, 4)),
        dims=("latitude", "longitude"),
        coords={"latitude": [60.0, 30.0, 0.0], "longitude": [0.0, 20.0, 340.0, 350.0]},
    )
    region = Region("seam", latitude_min=20.0, latitude_max=65.0, longitude_min=330.0, longitude_max=10.0)
    result = region_mask(reference, region)
    expected = np.array([[True, False, True, True], [True, False, True, True], [False, False, False, False]])
    np.testing.assert_array_equal(result.values, expected)


def test_vectorized_valid_date_alignment_retains_forecast_axes() -> None:
    observation = xr.DataArray(
        np.arange(12, dtype=float).reshape(3, 2, 2),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.array(["2024-06-01", "2024-06-02", "2024-06-03"], dtype="datetime64[ns]"),
            "latitude": [0.0, 1.0],
            "longitude": [0.0, 10.0],
        },
    )
    valid_date = xr.DataArray(
        np.array(
            [
                [["2024-06-01", "2024-06-02"], ["2024-06-02", "2024-06-03"]],
                [["2024-06-02", "2024-06-03"], ["2024-06-03", "2024-06-03"]],
            ],
            dtype="datetime64[ns]",
        ),
        dims=("initialization", "forecast_day", "longitude"),
        coords={
            "initialization": np.array(["2024-05-31", "2024-06-01"], dtype="datetime64[ns]"),
            "forecast_day": [0, 1],
            "longitude": [0.0, 10.0],
        },
    )
    matched = match_observation_by_valid_date(observation, valid_date)
    assert matched.dims == ("initialization", "forecast_day", "latitude", "longitude")
    assert matched.sel(initialization="2024-05-31", forecast_day=0, latitude=0, longitude=10).item() == 5.0
