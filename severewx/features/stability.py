"""Stability and lapse-rate proxies."""

from __future__ import annotations

import xarray as xr

from .base import clip_and_fill


def add_stability_features(dataset: xr.Dataset) -> xr.Dataset:
    t500 = dataset["t500"] if "t500" in dataset else dataset["t700"] - 18.0
    lapse_rate_700_500 = clip_and_fill((dataset["t700"] - t500) / 3.0, minimum=0.0, maximum=12.0)
    effective_instability = clip_and_fill(dataset["cape"] / (1.0 + abs(dataset["cin"]) / 50.0), minimum=0.0)
    capped_instability = clip_and_fill(dataset["cape"] / (1.0 + abs(dataset["cin"]) / 100.0), minimum=0.0)
    dataset["lapse_rate_700_500"] = lapse_rate_700_500.astype("float32")
    dataset["effective_instability"] = effective_instability.astype("float32")
    dataset["capped_instability"] = capped_instability.astype("float32")
    return dataset
