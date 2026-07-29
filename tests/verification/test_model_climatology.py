from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from heatextremes.verification.model_climatology import (
    DAY_OF_YEAR_COUNT,
    circular_window_counts,
    inspect_raw_store,
    preflight_model,
)
from heatextremes.verification.reforecast_inventory import ReforecastModelInventory


def test_circular_window_counts_wraps_year_boundary() -> None:
    counts = np.zeros(DAY_OF_YEAR_COUNT, dtype=np.int32)
    counts[0] = 2
    counts[-1] = 3

    windowed = circular_window_counts(counts, window_days=3)

    assert windowed[0] == 5
    assert windowed[-1] == 5


def test_raw_store_preflight_uses_only_local_solar_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "init_2000-01-01.zarr"
    raw = xr.Dataset(
        {
            "2t": (
                ("time", "prediction_timedelta", "lat", "lon"),
                np.full((1, 60, 1, 6), 280.0, dtype=np.float32),
            )
        },
        coords={
            "time": [np.datetime64("2000-01-01T00")],
            "prediction_timedelta": np.arange(60) * np.timedelta64(6, "h"),
            "lat": [0.0],
            "lon": [20.0, 80.0, 150.0, 200.0, 260.0, 320.0],
        },
    )
    raw.to_zarr(path, mode="w", consolidated=True)

    result = inspect_raw_store(path, variable="2t", ensemble=False, forecast_days=[0])

    assert result.initialization_count == 1
    assert 0 in result.available_forecast_days
    assert result.grid_shape == (1, 6)
    assert result.member_count is None
    assert result.valid_day_counts.shape == (1, 6, DAY_OF_YEAR_COUNT)

    report = preflight_model(
        ReforecastModelInventory(
            name="synthetic",
            directory=tmp_path,
            partitions=((2000, 1),),
            store_count=1,
            unparsed_store_names=(),
            display_name="Synthetic",
            ensemble=False,
            source_temperature_variable="2t",
            source_store_glob="init_*.zarr",
        ),
        years=[2000],
        months=[1],
        forecast_days=[0],
        window_days=15,
        minimum_samples=1,
    )
    assert report["discovered_stores"] == 1
    assert report["checked_stores"] == 1
    assert report["metadata_errors"] == []
