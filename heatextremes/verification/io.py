"""Safe, restart-oriented output utilities for verification products."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import xarray as xr


TABLE_STEMS = (
    "deterministic",
    "probability",
    "interval_coverage",
    "probability_reliability",
)


def resolve_table_format(requested: str) -> str:
    """Select Parquet when its engine is installed, otherwise CSV."""
    if requested not in {"auto", "parquet", "csv"}:
        raise ValueError("table format must be auto, parquet, or csv")
    parquet_available = importlib.util.find_spec("pyarrow") is not None
    if requested == "parquet" and not parquet_available:
        raise RuntimeError("Parquet was requested but pyarrow is unavailable")
    return "parquet" if requested == "parquet" or (requested == "auto" and parquet_available) else "csv"


def table_path(directory: Path, stem: str, table_format: str) -> Path:
    return directory / f"{stem}.{ 'parquet' if table_format == 'parquet' else 'csv'}"


def find_table(directory: Path, stem: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def read_table(path: Path) -> pd.DataFrame:
    """Read a supported tidy verification table."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix: {path}")


def write_table_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Atomically write a small table; never leave a partially committed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif path.suffix == ".csv":
            frame.to_csv(temporary, index=False, float_format="%.17g")
        else:
            raise ValueError(f"Unsupported table suffix: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Atomically write UTF-8 JSON with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_netcdf_atomic(dataset: xr.Dataset, path: Path) -> None:
    """Atomically write a genuinely spatial product as NetCDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{path.suffix}")
    try:
        dataset.to_netcdf(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def assert_safe_result_path(path: Path, result_root: Path) -> tuple[Path, Path]:
    """Verify that a destructive target is a child, never the results root."""
    raw = str(path)
    if any(token in raw for token in ("{", "}", "${", "$")):
        raise ValueError(f"Refusing unresolved result path: {path}")
    root = result_root.expanduser().resolve(strict=False)
    target = path.expanduser().resolve(strict=False)
    if target == root:
        raise ValueError("Refusing to delete or overwrite the verification-results root itself")
    if root not in target.parents:
        raise ValueError(f"Refusing path outside results root: {target} (root={root})")
    return target, root


def remove_result_path(path: Path, result_root: Path) -> None:
    """Remove only a resolved, explicit child of the configured results root."""
    target, _ = assert_safe_result_path(path, result_root)
    if not target.exists():
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def git_commit(repository_root: Path) -> str | None:
    """Return the current commit if Git metadata is available."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def completion_is_valid(directory: Path) -> bool:
    """Check that a completion marker names files that are still present."""
    marker = directory / "completion.json"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return payload.get("status") == "complete" and all(
            (directory / item).is_file() for item in payload.get("expected_output_files", [])
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def completed_output_names(directory: Path, *, include_maps: bool = True) -> list[str]:
    """Return the actual table file names used by a completed partition."""
    names: list[str] = []
    for stem in TABLE_STEMS:
        path = find_table(directory, stem)
        if path is None:
            raise FileNotFoundError(f"Missing required partial table {stem} in {directory}")
        names.append(path.name)
    if include_maps and (directory / "maps.nc").is_file():
        names.append("maps.nc")
    return names


def concatenate_tables(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [read_table(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
