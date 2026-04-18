"""Thermodynamic features."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .base import clip_and_fill


def add_thermodynamic_features(dataset: xr.Dataset) -> xr.Dataset:
    dewpoint_depression = clip_and_fill(dataset["t2m"] - dataset["td2m"], minimum=0.0, maximum=30.0)
    mixed_layer_moisture = clip_and_fill(dataset["td2m"] - 273.15, minimum=-30.0, maximum=30.0)
    moisture_transport = clip_and_fill(np.hypot(dataset["u850"], dataset["v850"]) * mixed_layer_moisture, minimum=0.0)
    moisture_quality = clip_and_fill((dataset["td2m"] - 287.0) / 6.0, minimum=0.0, maximum=1.5)
    lcl_height_proxy = clip_and_fill(125.0 * dewpoint_depression, minimum=50.0, maximum=2500.0)
    low_lcl_support = clip_and_fill(1.0 - (lcl_height_proxy - 800.0) / 1200.0, minimum=0.0, maximum=1.2)
    dataset["dewpoint_depression"] = dewpoint_depression.astype("float32")
    dataset["mixed_layer_moisture"] = mixed_layer_moisture.astype("float32")
    dataset["moisture_transport"] = moisture_transport.astype("float32")
    dataset["moisture_quality"] = moisture_quality.astype("float32")
    dataset["lcl_height_proxy"] = lcl_height_proxy.astype("float32")
    dataset["low_lcl_support"] = low_lcl_support.astype("float32")
    if "pwat" not in dataset:
        dataset["pwat"] = xr.zeros_like(dataset["cape"]).astype("float32")
    return dataset
