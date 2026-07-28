from __future__ import annotations

from pathlib import Path

from heatextremes.verification.reforecast_inventory import inventory_reforecast_root, raw_reforecast_config


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
    assert inventory[1].partitions == ((2023, 7),)

    config = raw_reforecast_config(
        inventory[0], result_root=tmp_path / "results", region_file=tmp_path / "regions.yaml"
    )
    assert config["model"]["adapter"] == "standard_reforecast_raw"
    assert config["selection"]["partitions"] == [{"year": 2022, "month": 6}]
