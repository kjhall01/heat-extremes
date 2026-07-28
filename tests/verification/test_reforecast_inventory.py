from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from heatextremes.verification.reforecast_inventory import (
    inventory_metadata_csv,
    inventory_reforecast_root,
    raw_reforecast_config,
)


def test_inventory_scans_only_standard_init_store_names(tmp_path: Path) -> None:
    root = tmp_path / "reforecast"
    aurora = root / "forecasts_Aurora-E2S"
    other = root / "forecasts_other"
    aurora.mkdir(parents=True)
    other.mkdir()
    (aurora / "init_2022-06-01.zarr").mkdir()
    (aurora / "init_2022-06-15.zarr").mkdir()
    (aurora / "init_2022-10-01.zarr").mkdir()
    (aurora / "init_unparseable.zarr").mkdir()
    (other / "init_2023-07-01.zarr").mkdir()

    inventory = inventory_reforecast_root(root, years=[2022, 2023], months=[6, 7])
    assert [item.name for item in inventory] == ["aurora_e2s", "other"]
    assert inventory[0].partitions == ((2022, 6),)
    assert inventory[0].store_count == 2
    assert inventory[0].source_store_glob == "init_*.zarr"
    assert inventory[1].partitions == ((2023, 7),)

    config = raw_reforecast_config(
        inventory[0], result_root=tmp_path / "results", region_file=tmp_path / "regions.yaml"
    )
    assert config["model"]["adapter"] == "standard_reforecast_raw"
    assert config["selection"]["partitions"] == [{"year": 2022, "month": 6}]


def test_inventory_uses_common_local_solar_leads_from_store_metadata(tmp_path: Path) -> None:
    root = tmp_path / "reforecast"
    source = root / "forecasts_short_horizon"
    source.mkdir(parents=True)
    longitude = np.asarray([0.0, 90.0, 180.0, 270.0])

    def write_store(path: Path, step_count: int) -> None:
        raw = xr.Dataset(
            {
                "2t": (
                    ("time", "prediction_timedelta", "lat", "lon"),
                    np.full((1, step_count, 1, longitude.size), 300.0),
                )
            },
            coords={
                "time": [np.datetime64("2022-06-01T00")],
                "prediction_timedelta": np.arange(step_count) * np.timedelta64(6, "h"),
                "lat": [0.0],
                "lon": longitude,
            },
        )
        raw.to_zarr(path, mode="w", consolidated=True)

    # The second store has one fewer complete local day. The generated model
    # config must therefore use the common range rather than blindly using
    # the historical 0--14 AIFS range.
    write_store(source / "init_2022-06-01.zarr", step_count=60)
    write_store(source / "init_2022-07-01.zarr", step_count=56)

    inventory = inventory_reforecast_root(root, years=[2022], months=[6, 7])
    assert len(inventory) == 1
    discovered_days = inventory[0].forecast_days
    assert discovered_days
    assert max(discovered_days) < 14

    config = raw_reforecast_config(
        inventory[0], result_root=tmp_path / "results", region_file=tmp_path / "regions.yaml"
    )
    assert config["selection"]["forecast_days"] == list(discovered_days)


def test_metadata_registry_sets_deterministic_capability_and_excludes_gencast(tmp_path: Path) -> None:
    root = tmp_path / "reforecast"
    graphcast = root / "forecasts_graphcast_e2s"
    ensemble = root / "forecasts_example_ens"
    aifs_ens = tmp_path / "forecasts_AIFS_ENS_v2"
    graphcast.mkdir(parents=True)
    ensemble.mkdir()
    aifs_ens.mkdir()
    (graphcast / "init_2022-06-01.zarr").mkdir()
    (ensemble / "init_2022-06-01.zarr").mkdir()
    (aifs_ens / "forecast_2022-06-01.zarr").mkdir()
    metadata = tmp_path / "models.csv"
    metadata.write_text(
        "Model,N Members,Variables,Path\n"
        f"GraphCast,N/A,2t,{graphcast}\n"
        f"Example ENS,25,2t,{ensemble}\n"
        f"AIFS-ENS-v2,25,2t,{aifs_ens}\n"
        f"gencast,52,2m_temperature,{root / 'forecasts_gencast'}\n",
        encoding="utf-8",
    )

    inventory, skipped = inventory_metadata_csv(metadata, root=root, years=[2022], months=[6])
    assert [item.name for item in inventory] == ["graphcast_e2s", "example_ens", "aifs_ens_v2"]
    assert {item.name: item.ensemble for item in inventory} == {
        "graphcast_e2s": False,
        "example_ens": True,
        "aifs_ens_v2": True,
    }
    assert inventory[-1].source_store_glob == "*.zarr"
    assert any(item["model"] == "gencast" and item["reason"] == "explicitly excluded" for item in skipped)
