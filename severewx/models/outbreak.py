"""Outbreak-risk training and scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from severewx.config import AppSettings
from severewx.models.model_io import save_model_artifact
from severewx.models.train import _make_backend


OUTBREAK_FEATURES = [
    "lead_day",
    "max_sig_tor_support",
    "max_tornado_favored_overlap",
    "max_outbreak_corridor_index",
    "max_tornado_corridor_index",
    "max_synoptic_support",
    "max_forcing_proxy",
    "max_cape",
    "max_shear_0_6km",
    "mean_analog_outbreak_support",
    "mean_analog_sigtor_support",
    "mean_spatial_coverage",
]


@dataclass(slots=True)
class TrainedOutbreakModel:
    artifact: dict[str, Any]


def outbreak_training_frame(gridpoint_frame: pd.DataFrame) -> pd.DataFrame:
    label_columns = ["any_outbreak", "tornado_outbreak", "significant_tornado_support"]
    frame = gridpoint_frame
    if "spatial_coverage" not in frame.columns:
        fallback = frame["outbreak_corridor_index"] if "outbreak_corridor_index" in frame.columns else pd.Series(0.0, index=frame.index)
        frame = frame.assign(spatial_coverage=fallback)
    grouped = frame.groupby(["date", "lead_day"], as_index=False).agg(
        max_sig_tor_support=("sig_tor_support", "max"),
        max_tornado_favored_overlap=("tornado_favored_overlap", "max"),
        max_outbreak_corridor_index=("outbreak_corridor_index", "max"),
        max_tornado_corridor_index=("tornado_corridor_index", "max"),
        max_synoptic_support=("synoptic_support", "max"),
        max_forcing_proxy=("forcing_proxy", "max"),
        max_cape=("cape", "max"),
        max_shear_0_6km=("shear_0_6km", "max"),
        mean_analog_outbreak_support=("analog_outbreak_support", "mean"),
        mean_analog_sigtor_support=("analog_sigtor_support", "mean"),
        mean_spatial_coverage=("spatial_coverage", "mean"),
        any_outbreak=(label_columns[0], "max"),
        tornado_outbreak=(label_columns[1], "max"),
        significant_tornado_support=(label_columns[2], "max"),
    )
    grouped = grouped.loc[grouped[label_columns].notna().all(axis=1)].copy()
    grouped["outbreak_target"] = grouped[label_columns].max(axis=1).astype(int)
    return grouped


def train_outbreak_model(gridpoint_frame: pd.DataFrame, settings: AppSettings) -> TrainedOutbreakModel:
    frame = outbreak_training_frame(gridpoint_frame)
    model, backend = _make_backend(settings)
    model.fit(frame[OUTBREAK_FEATURES], frame["outbreak_target"])
    artifact = {"hazard": "outbreak", "backend": backend, "model": model, "columns": OUTBREAK_FEATURES}
    return TrainedOutbreakModel(artifact=artifact)


def save_outbreak_model(gridpoint_frame: pd.DataFrame, settings: AppSettings, paths: Any) -> dict[str, Any]:
    trained = train_outbreak_model(gridpoint_frame, settings)
    save_model_artifact(trained.artifact, paths, "outbreak")
    return trained.artifact


def apply_outbreak_model(model_artifact: dict[str, Any], feature_frame: pd.DataFrame) -> pd.DataFrame:
    aggregate = outbreak_training_frame(feature_frame.assign(any_outbreak=0, tornado_outbreak=0, significant_tornado_support=0))
    probabilities = model_artifact["model"].predict_proba(aggregate[model_artifact["columns"]])[:, 1]
    aggregate["outbreak_day_prob"] = probabilities
    merged = feature_frame.merge(aggregate[["date", "lead_day", "outbreak_day_prob"]], on=["date", "lead_day"], how="left")
    local_support = (
        0.25 * merged["any_prob"]
        + 0.25 * merged["tornado_prob"]
        + 0.20 * merged["sig_tor_support"].clip(0.0, 1.0)
        + 0.10 * merged["analog_outbreak_support"].clip(0.0, 1.0)
        + 0.10 * merged["outbreak_corridor_index"].clip(0.0, 1.0)
        + 0.10 * merged["tornado_corridor_index"].clip(0.0, 1.0)
    )
    merged["outbreak_risk"] = np.clip(merged["outbreak_day_prob"] * np.clip(local_support, 0, 1.5), 0, 1)
    return merged
