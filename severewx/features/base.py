"""Base feature helpers."""

from __future__ import annotations

import pandas as pd
import xarray as xr


def flatten_feature_dataset(dataset: xr.Dataset) -> pd.DataFrame:
    frame = dataset.to_dataframe().reset_index()
    for coord in ("time", "date"):
        if coord in frame.columns:
            frame[coord] = pd.to_datetime(frame[coord])
    return frame


def spatial_mean(data: xr.DataArray, radius: int = 1) -> xr.DataArray:
    rolled = []
    for y_shift in range(-radius, radius + 1):
        for x_shift in range(-radius, radius + 1):
            rolled.append(data.shift(lat=y_shift, lon=x_shift))
    stacked = xr.concat(rolled, dim="neighbor")
    return stacked.mean("neighbor", skipna=True).fillna(data)


def clip_and_fill(data: xr.DataArray, minimum: float | None = None, maximum: float | None = None) -> xr.DataArray:
    if minimum is not None:
        data = xr.where(data < minimum, minimum, data)
    if maximum is not None:
        data = xr.where(data > maximum, maximum, data)
    return data.fillna(0.0)
