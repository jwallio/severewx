"""Kinematic severe-weather features."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .base import clip_and_fill


def add_kinematic_features(dataset: xr.Dataset) -> xr.Dataset:
    wind10m_speed = np.hypot(dataset["u10"], dataset["v10"])
    wind850_speed = np.hypot(dataset["u850"], dataset["v850"])
    wind500_speed = np.hypot(dataset["u500"], dataset["v500"])
    shear_0_6km = np.hypot(dataset["u500"] - dataset["u10"], dataset["v500"] - dataset["v10"])
    ll_shear_proxy = np.hypot(dataset["u850"] - dataset["u10"], dataset["v850"] - dataset["v10"])
    deep_layer_speed_diff = clip_and_fill(wind500_speed - wind10m_speed, minimum=0.0)
    shear_ratio = clip_and_fill(ll_shear_proxy / xr.where(shear_0_6km < 1.0, 1.0, shear_0_6km), minimum=0.0, maximum=1.5)
    helicity_proxy = clip_and_fill((wind850_speed * ll_shear_proxy * (1.0 + shear_ratio)) / 6.0, minimum=0.0, maximum=600.0)
    dataset["wind10m_speed"] = wind10m_speed.astype("float32")
    dataset["wind850_speed"] = wind850_speed.astype("float32")
    dataset["wind500_speed"] = wind500_speed.astype("float32")
    dataset["shear_0_6km"] = shear_0_6km.astype("float32")
    dataset["ll_shear_proxy"] = ll_shear_proxy.astype("float32")
    dataset["llj_speed"] = wind850_speed.astype("float32")
    dataset["deep_layer_speed_diff"] = deep_layer_speed_diff.astype("float32")
    dataset["shear_ratio"] = shear_ratio.astype("float32")
    dataset["helicity_proxy"] = helicity_proxy.astype("float32")
    return dataset
