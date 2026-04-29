"""Hazard prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from severewx.bustrisk.rules import add_bust_risk
from severewx.config import AppSettings
from severewx.confidence.score import add_confidence
from severewx.features.composites import build_feature_dataset
from severewx.features.base import flatten_feature_dataset
from severewx.models.model_io import load_model_artifact
from severewx.models.outbreak import apply_outbreak_model
from severewx.models.tornado_concern import apply_tornado_concern_model


def predict_hazard_frame(
    forecast: xr.Dataset,
    settings: AppSettings,
    paths: Any,
    prior_run: xr.Dataset | None = None,
    analog_archive_path: Path | None = None,
) -> pd.DataFrame:
    features = build_feature_dataset(forecast, settings=settings, prior_run=prior_run, analog_archive_path=analog_archive_path)
    frame = flatten_feature_dataset(features)
    frame["date"] = pd.to_datetime(frame["time"]).dt.date.astype(str)
    first_time = pd.to_datetime(frame["time"].min())
    frame["lead_day"] = ((pd.to_datetime(frame["time"]) - first_time).dt.total_seconds() // 86400).astype(int) + 1

    for hazard in ("tornado", "hail", "wind", "any"):
        artifact = load_model_artifact(paths, hazard)
        columns = [column for column in artifact["columns"] if column in frame.columns]
        raw = artifact["model"].predict_proba(frame[columns])[:, 1]
        calibrated = artifact["calibrator"].apply(frame.assign(raw_pred=raw), "raw_pred")
        frame[f"{hazard}_prob"] = np.clip(calibrated, 0.0, 1.0)
        frame[f"{hazard}_calibration_quality"] = 1.0 - artifact["metrics"]["brier_score"]

    outbreak_artifact = load_model_artifact(paths, "outbreak")
    frame = apply_outbreak_model(outbreak_artifact, frame)
    frame = apply_tornado_concern_model(frame, paths, settings=settings)
    frame = add_confidence(frame, settings)
    frame = add_bust_risk(frame, settings)
    return frame


def prediction_frame_to_dataset(frame: pd.DataFrame) -> xr.Dataset:
    dataset = frame.set_index(["time", "lat", "lon"]).to_xarray()
    return dataset.sortby(["time", "lat", "lon"])
