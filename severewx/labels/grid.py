"""Daily gridded label generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from severewx.config import AppSettings
from severewx.labels.neighborhood import neighborhood_hits


HAZARDS = ("tornado", "hail", "wind")


def make_label_grid(settings: AppSettings) -> xr.Dataset:
    lat = np.arange(
        settings.get("grid.lat_min"),
        settings.get("grid.lat_max") + settings.get("grid.lat_step"),
        settings.get("grid.lat_step"),
        dtype=np.float32,
    )
    lon = np.arange(
        settings.get("grid.lon_min"),
        settings.get("grid.lon_max") + settings.get("grid.lon_step"),
        settings.get("grid.lon_step"),
        dtype=np.float32,
    )
    return xr.Dataset(coords={"lat": lat, "lon": lon})


def build_daily_labels(reports: pd.DataFrame, settings: AppSettings) -> xr.Dataset:
    base = make_label_grid(settings)
    radius = float(settings.get("labels.neighborhood_km", 40.0))
    lon2d, lat2d = np.meshgrid(base["lon"].values, base["lat"].values)
    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for hazard in HAZARDS:
        subset = reports.loc[reports["hazard"] == hazard]
        hits = neighborhood_hits(
            lat2d,
            lon2d,
            subset["lat"].to_numpy(dtype=float),
            subset["lon"].to_numpy(dtype=float),
            radius_km=radius,
        )
        data_vars[hazard] = (("lat", "lon"), hits.astype(np.int8))
    any_hits = np.maximum.reduce([data_vars[hazard][1] for hazard in HAZARDS]).astype(np.int8)
    base = base.assign(data_vars)
    base["any"] = (("lat", "lon"), any_hits)
    return base


def build_label_cube(reports: pd.DataFrame, start: str, end: str, settings: AppSettings) -> xr.Dataset:
    dates = pd.date_range(start, end, freq="D")
    daily_datasets: list[xr.Dataset] = []
    for valid_date in dates:
        subset = reports.loc[reports["date"] == valid_date.date().isoformat()]
        dataset = build_daily_labels(subset, settings)
        dataset = dataset.expand_dims(date=[np.datetime64(valid_date.date())])
        daily_datasets.append(dataset)
    return xr.concat(daily_datasets, dim="date")
