"""Storage helpers for forecast datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xarray as xr

from severewx.utils.paths import DataPaths, dated_filename


def forecast_path(paths: DataPaths, date: str, cycle: str) -> Path:
    return paths.processed / dated_filename("forecast", date, "nc", cycle)


def raw_grib_path(paths: DataPaths, date: str, cycle: str, lead: int) -> Path:
    return paths.raw / f"gfs_{date}_{cycle}_f{int(lead):03d}.grib2"


def raw_grib_metadata_path(paths: DataPaths, date: str, cycle: str, lead: int) -> Path:
    return paths.raw / f"gfs_{date}_{cycle}_f{int(lead):03d}.json"


def staged_raw_archive_dir(paths: DataPaths, date: str, cycle: str, archive_root: str | None = None) -> Path:
    if archive_root:
        return Path(archive_root).resolve() / date / cycle
    return (paths.raw / "staged_gfs_archive" / date / cycle).resolve()


def historical_feature_path(paths: DataPaths, date: str, cycle: str) -> Path:
    year = date[:4]
    return paths.feature_archive / year / date / cycle / "features.parquet"


def historical_metadata_path(paths: DataPaths, date: str, cycle: str) -> Path:
    year = date[:4]
    return paths.archive_metadata / year / date / f"{cycle}.json"


def save_forecast_dataset(dataset: xr.Dataset, paths: DataPaths, date: str, cycle: str) -> Path:
    output_path = forecast_path(paths, date, cycle)
    dataset.to_netcdf(output_path)
    return output_path


def load_forecast_dataset(paths: DataPaths, date: str, cycle: str) -> xr.Dataset:
    return xr.load_dataset(forecast_path(paths, date, cycle))


def load_json_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_metadata(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
