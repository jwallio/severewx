"""Hazard model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from severewx.calibration.calibrate import fit_group_calibrator
from severewx.config import AppSettings
from severewx.models.model_io import save_model_artifact
from severewx.verify.metrics import brier_score, safe_roc_auc


def _make_backend(settings: AppSettings) -> tuple[Any, str]:
    backend = str(settings.get("models.backend", "lightgbm")).lower()
    if backend == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                random_state=int(settings.get("models.random_state", 42)),
            )
            return model, "lightgbm"
        except ImportError:
            pass
    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, random_state=int(settings.get("models.random_state", 42)))
    return model, "hist_gradient_boosting"


def _split_train_valid(frame: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(frame["date"].unique())
    if len(unique_dates) < 3:
        return frame.copy(), frame.copy()
    n_valid = max(1, int(round(len(unique_dates) * validation_fraction)))
    valid_dates = set(unique_dates[-n_valid:])
    train = frame.loc[~frame["date"].isin(valid_dates)].copy()
    valid = frame.loc[frame["date"].isin(valid_dates)].copy()
    return train, valid


def _feature_columns(settings: AppSettings) -> list[str]:
    columns = list(settings.get("models.hazard_columns", []))
    return ["lead_day", "lat", "lon", *columns]


@dataclass(slots=True)
class TrainedHazardModel:
    artifact: dict[str, Any]


def train_hazard_model(hazard: str, frame: pd.DataFrame, settings: AppSettings) -> TrainedHazardModel:
    if hazard not in frame.columns:
        raise ValueError(f"missing target column for hazard {hazard}")
    if len(frame) < int(settings.get("models.min_training_rows", 50)):
        raise ValueError(f"not enough rows to train {hazard}")
    columns = [column for column in _feature_columns(settings) if column in frame.columns]
    train_frame, valid_frame = _split_train_valid(frame, float(settings.get("models.validation_fraction", 0.2)))
    model, backend = _make_backend(settings)
    model.fit(train_frame[columns], train_frame[hazard])
    raw_valid = model.predict_proba(valid_frame[columns])[:, 1] if hasattr(model, "predict_proba") else model.predict(valid_frame[columns])
    calibrator = fit_group_calibrator(valid_frame.assign(raw_pred=raw_valid), "raw_pred", hazard, settings)
    calibrated = calibrator.apply(valid_frame.assign(raw_pred=raw_valid), "raw_pred")
    importances = feature_importance(model, columns)
    artifact = {
        "hazard": hazard,
        "backend": backend,
        "model": model,
        "columns": columns,
        "feature_importance": importances,
        "metrics": {
            "brier_score": brier_score(valid_frame[hazard].to_numpy(), calibrated),
            "roc_auc": safe_roc_auc(valid_frame[hazard].to_numpy(), calibrated),
        },
        "calibrator": calibrator,
    }
    return TrainedHazardModel(artifact=artifact)


def feature_importance(model: Any, columns: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "feature_names_in_") and hasattr(model, "predict_proba"):
        values = np.ones(len(columns))
    else:
        values = np.ones(len(columns))
    total = float(np.sum(values)) or 1.0
    return {column: float(value / total) for column, value in zip(columns, values, strict=False)}


def train_and_save_hazard_model(hazard: str, frame: pd.DataFrame, settings: AppSettings, paths: Any) -> dict[str, Any]:
    trained = train_hazard_model(hazard, frame, settings)
    save_model_artifact(trained.artifact, paths, hazard)
    return trained.artifact
