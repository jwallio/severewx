"""Synoptic forcing features."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .base import clip_and_fill


def _gradient_magnitude(data: xr.DataArray) -> xr.DataArray:
    lat_grad = data.differentiate("lat")
    lon_grad = data.differentiate("lon")
    return np.hypot(lat_grad, lon_grad)


def add_forcing_features(dataset: xr.Dataset) -> xr.Dataset:
    height_gradient = _gradient_magnitude(dataset["z500"])
    mslp_gradient = _gradient_magnitude(dataset["mslp"])
    upper_jet = np.hypot(dataset["u500"], dataset["v500"])
    llj = dataset["llj_speed"] if "llj_speed" in dataset else np.hypot(dataset["u850"], dataset["v850"])
    moisture_transport = dataset["moisture_transport"] if "moisture_transport" in dataset else 0.0
    forcing_proxy = clip_and_fill(height_gradient * 10.0 + mslp_gradient * 0.01 + upper_jet * 0.1 + llj * 0.15, minimum=0.0)
    synoptic_support = clip_and_fill(forcing_proxy + upper_jet * 0.35 + llj * 0.20, minimum=0.0)
    forcing_instability_overlap = clip_and_fill((forcing_proxy / 90.0) * (dataset["effective_instability"] / 1500.0), minimum=0.0, maximum=2.0)
    qg_support_proxy = clip_and_fill(height_gradient * llj * 0.25 + moisture_transport * 0.03, minimum=0.0)
    dataset["forcing_proxy"] = forcing_proxy.astype("float32")
    dataset["synoptic_support"] = synoptic_support.astype("float32")
    dataset["forcing_instability_overlap"] = forcing_instability_overlap.astype("float32")
    dataset["qg_support_proxy"] = qg_support_proxy.astype("float32")
    return dataset
