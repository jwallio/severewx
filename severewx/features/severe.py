"""Severe-weather composite features."""

from __future__ import annotations

import xarray as xr

from .base import clip_and_fill


def add_severe_features(dataset: xr.Dataset) -> xr.Dataset:
    cape_term = clip_and_fill(dataset["cape"] / 1800.0, minimum=0.0, maximum=3.0)
    shear_term = clip_and_fill(dataset["shear_0_6km"] / 20.0, minimum=0.0, maximum=3.0)
    ll_shear_term = clip_and_fill(dataset["ll_shear_proxy"] / 12.0, minimum=0.0, maximum=2.0)
    helicity_term = clip_and_fill(dataset["helicity_proxy"] / 180.0, minimum=0.0, maximum=3.0)
    cin_penalty = clip_and_fill(1.0 - abs(dataset["cin"]) / 175.0, minimum=0.0, maximum=1.0)
    low_lcl_term = dataset["low_lcl_support"]
    moisture_term = dataset["moisture_quality"]
    lapse_term = clip_and_fill(dataset["lapse_rate_700_500"] / 7.0, minimum=0.0, maximum=1.5)
    forcing_term = clip_and_fill(dataset["forcing_proxy"] / 85.0, minimum=0.0, maximum=1.5)

    cape_shear_overlap = clip_and_fill(cape_term * shear_term, minimum=0.0, maximum=4.0)
    tornado_favored_overlap = clip_and_fill(
        cape_term * shear_term * ll_shear_term * moisture_term * low_lcl_term * cin_penalty,
        minimum=0.0,
        maximum=6.0,
    )
    hail_favored_overlap = clip_and_fill(cape_term * lapse_term * shear_term * (1.0 + 0.35 * forcing_term), minimum=0.0, maximum=6.0)
    wind_favored_overlap = clip_and_fill(shear_term * forcing_term * (dataset["moisture_transport"] / 400.0), minimum=0.0, maximum=6.0)

    scp_proxy = clip_and_fill(cape_term * shear_term * helicity_term, minimum=0.0, maximum=8.0)
    stp_proxy = clip_and_fill(cape_term * shear_term * ll_shear_term * low_lcl_term * cin_penalty * moisture_term, minimum=0.0, maximum=8.0)
    sig_tor_support = clip_and_fill(
        0.45 * stp_proxy + 0.25 * tornado_favored_overlap + 0.15 * clip_and_fill(dataset["qg_support_proxy"] / 120.0, minimum=0.0, maximum=1.5) + 0.15 * ll_shear_term,
        minimum=0.0,
        maximum=2.5,
    )

    dataset["cape_shear_overlap"] = cape_shear_overlap.astype("float32")
    dataset["tornado_favored_overlap"] = tornado_favored_overlap.astype("float32")
    dataset["hail_favored_overlap"] = hail_favored_overlap.astype("float32")
    dataset["wind_favored_overlap"] = wind_favored_overlap.astype("float32")
    dataset["scp_proxy"] = scp_proxy.astype("float32")
    dataset["stp_proxy"] = stp_proxy.astype("float32")
    dataset["sig_tor_support"] = sig_tor_support.astype("float32")
    return dataset
