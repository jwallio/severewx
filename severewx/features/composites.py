"""Composed feature pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from severewx.config import AppSettings

from .analogs import compute_analog_features, load_analog_archive
from .base import spatial_mean
from .forcing import add_forcing_features
from .kinematics import add_kinematic_features
from .severe import add_severe_features
from .stability import add_stability_features
from .thermodynamics import add_thermodynamic_features


def add_prior_run_delta(dataset: xr.Dataset, prior_run: xr.Dataset | None) -> xr.Dataset:
    if prior_run is None or "cape" not in prior_run:
        dataset["prior_run_delta"] = xr.zeros_like(dataset["cape"]).astype(np.float32)
        return dataset
    aligned_prior = prior_run.reindex_like(dataset, method=None)
    delta = np.abs(dataset["cape"] - aligned_prior["cape"]).fillna(0.0)
    dataset["prior_run_delta"] = delta.astype(np.float32)
    return dataset


def add_neighborhood_features(dataset: xr.Dataset, radius: int) -> xr.Dataset:
    dataset["neighborhood_mean_any"] = spatial_mean(dataset["cape"] / 2500.0 + dataset["shear_0_6km"] / 40.0, radius=radius)
    dataset["spatial_coverage"] = spatial_mean(dataset["sig_tor_support"], radius=max(radius, 2))
    dataset["tornado_corridor_index"] = spatial_mean(dataset["tornado_favored_overlap"], radius=max(radius, 2))
    dataset["outbreak_corridor_index"] = spatial_mean(
        dataset["sig_tor_support"] + 0.5 * dataset["forcing_instability_overlap"] + 0.25 * dataset["wind_favored_overlap"],
        radius=max(radius, 2),
    )
    dataset["hazard_contrast"] = (
        dataset["tornado_favored_overlap"] - 0.5 * dataset["hail_favored_overlap"] - 0.35 * dataset["wind_favored_overlap"]
    ).astype(np.float32)
    return dataset


def build_feature_dataset(
    forecast: xr.Dataset,
    settings: AppSettings,
    prior_run: xr.Dataset | None = None,
    analog_archive_path: Path | None = None,
) -> xr.Dataset:
    dataset = forecast.copy()
    dataset = add_kinematic_features(dataset)
    dataset = add_thermodynamic_features(dataset)
    dataset = add_stability_features(dataset)
    dataset = add_forcing_features(dataset)
    dataset = add_severe_features(dataset)
    dataset = add_prior_run_delta(dataset, prior_run)
    dataset = add_neighborhood_features(dataset, int(settings.get("features.spatial_radius_gridpoints", 1)))
    dataset = compute_analog_features(dataset, load_analog_archive(analog_archive_path), settings)
    return dataset
