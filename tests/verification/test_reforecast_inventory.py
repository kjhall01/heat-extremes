from __future__ import annotations

from pathlib import Path

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
    assert inventory[1].partitions == ((2023, 7),)

    config = raw_reforecast_config(
        inventory[0], result_root=tmp_path / "results", region_file=tmp_path / "regions.yaml"
    )
    assert config["model"]["adapter"] == "standard_reforecast_raw"
    assert config["selection"]["partitions"] == [{"year": 2022, "month": 6}]


def test_metadata_registry_sets_deterministic_capability_and_excludes_gencast(tmp_path: Path) -> None:
    root = tmp_path / "reforecast"
    graphcast = root / "forecasts_graphcast_e2s"
    ensemble = root / "forecasts_example_ens"
    graphcast.mkdir(parents=True)
    ensemble.mkdir()
    (graphcast / "init_2022-06-01.zarr").mkdir()
    (ensemble / "init_2022-06-01.zarr").mkdir()
    metadata = tmp_path / "models.csv"
    metadata.write_text(
        "Model,N Members,Variables,Path\n"
        f"GraphCast,N/A,2t,{graphcast}\n"
        f"Example ENS,25,2t,{ensemble}\n"
        f"gencast,52,2m_temperature,{root / 'forecasts_gencast'}\n",
        encoding="utf-8",
    )

    inventory, skipped = inventory_metadata_csv(metadata, root=root, years=[2022], months=[6])
    assert [item.name for item in inventory] == ["graphcast_e2s", "example_ens"]
    assert {item.name: item.ensemble for item in inventory} == {"graphcast_e2s": False, "example_ens": True}
    assert any(item["model"] == "gencast" and item["reason"] == "explicitly excluded" for item in skipped)
