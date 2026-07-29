from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from heatextremes.verification.model_climatology import (
    DAY_OF_YEAR_COUNT,
    MODEL_Q95_VARIABLE,
    SAMPLE_COUNT_VARIABLE,
    build_q95_workflow_manifest,
    calendar_window_quantile,
    compute_q95_band,
    circular_window_counts,
    finalize_q95_workflow,
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


def test_calendar_window_quantile_is_circular_and_uses_global_day_labels() -> None:
    daily = xr.DataArray(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1),
        dims=("time", "latitude", "longitude"),
        coords={
            "time": np.asarray(["2000-12-31", "2001-01-01", "2001-01-02"], dtype="datetime64[ns]"),
            "latitude": [0.0],
            "longitude": [0.0],
        },
    ).chunk({"time": -1})
    valid_date = xr.DataArray(daily.time.values, dims=("time",), coords={"time": daily.time})

    result = calendar_window_quantile(daily, valid_date, window_days=3, percentile=95.0).compute()

    # Day one wraps to Dec 31 and includes all three values.
    assert result.dims == ("dayofyear", "latitude", "longitude")
    assert np.isclose(result.sel(dayofyear=1).item(), 2.9)


def test_q95_band_tasks_write_one_final_global_store_without_daily_files(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    raw_directory.mkdir()
    longitude = np.asarray([20.0, 80.0, 150.0, 200.0, 260.0, 320.0])
    for index, initialization in enumerate(("2000-01-01T00", "2000-01-08T00")):
        raw = xr.Dataset(
            {
                "2t": (
                    ("time", "prediction_timedelta", "lat", "lon"),
                    np.full((1, 60, 1, longitude.size), 280.0 + 2 * index, dtype=np.float32),
                )
            },
            coords={
                "time": [np.datetime64(initialization)],
                "prediction_timedelta": np.arange(60) * np.timedelta64(6, "h"),
                "lat": [0.0],
                "lon": longitude,
            },
        )
        raw.to_zarr(
            raw_directory / f"init_2000-01-{1 + 7 * index:02d}.zarr",
            mode="w",
            consolidated=True,
        )

    preflight = {
        "requested_years": [2000],
        "requested_months": [1],
        "q95_window_days": 15,
        "models": [
            {
                "model": "synthetic",
                "display_name": "Synthetic",
                "status": "ready",
                "raw_directory": str(raw_directory),
                "source_temperature_variable": "2t",
                "ensemble": False,
                "checked_stores": 2,
            }
        ],
    }
    results_root = tmp_path / "results"
    manifest = build_q95_workflow_manifest(
        preflight,
        result_root=results_root,
        output_directory=results_root / "model_climatology" / "synthetic_q95",
        years=[2000],
        months=[1],
        forecast_days=[0],
        window_days=15,
        percentile=95.0,
    )

    assert manifest["task_count"] == 6
    for task in manifest["tasks"]:
        assert compute_q95_band(
            manifest, model_name=task["model"], band_index=task["band_index"]
        )["status"] == "complete"
    completed = finalize_q95_workflow(manifest)

    product_directory = Path(manifest["models"][0]["product_directory"])
    dataset = xr.open_zarr(product_directory / "q95.zarr", consolidated=True, chunks={})
    try:
        assert completed[0]["status"] == "complete"
        assert dataset[MODEL_Q95_VARIABLE].shape == (1, DAY_OF_YEAR_COUNT, 1, 6)
        assert dataset[SAMPLE_COUNT_VARIABLE].shape == (1, DAY_OF_YEAR_COUNT, 6)
        assert int(dataset[SAMPLE_COUNT_VARIABLE].max()) == 2
    finally:
        dataset.close()
    assert (product_directory / "completion.json").is_file()
    assert not list(product_directory.glob("*daily*"))
