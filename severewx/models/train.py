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


def _hazard_sample_weight(hazard: str, target: pd.Series, settings: AppSettings) -> tuple[np.ndarray | None, dict[str, Any]]:
    config = settings.get("models.hazard_sample_weighting", {}) or {}
    enabled = bool(config.get("enabled", False))
    per_hazard_max = config.get("per_hazard_max_positive_weight", {}) or {}
    max_positive_weight = float(per_hazard_max.get(hazard, config.get("max_positive_weight", 1.0)))
    metadata: dict[str, Any] = {
        "enabled": enabled,
        "mode": str(config.get("mode", "none")),
        "positive_weight": 1.0,
        "hazard": hazard,
        "max_positive_weight": max_positive_weight,
    }
    if not enabled:
        return None, metadata

    values = pd.Series(target).fillna(0).to_numpy(dtype=float)
    positive_mask = values > 0.5
    positive_count = int(positive_mask.sum())
    negative_count = int((~positive_mask).sum())
    metadata["positive_count"] = positive_count
    metadata["negative_count"] = negative_count
    if positive_count == 0 or negative_count == 0:
        return None, metadata

    mode = str(config.get("mode", "balanced_binary")).lower()
    negative_weight = float(config.get("negative_weight", 1.0))
    if mode == "balanced_binary":
        raw_positive_weight = negative_count / positive_count
    else:
        raw_positive_weight = float(config.get("positive_weight", 1.0))
    positive_weight = float(
        np.clip(
            raw_positive_weight,
            float(config.get("min_positive_weight", 1.0)),
            max_positive_weight,
        )
    )
    metadata["mode"] = mode
    metadata["positive_weight"] = positive_weight
    sample_weight = np.where(positive_mask, positive_weight, negative_weight).astype(float)
    return sample_weight, metadata


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
    sample_weight, weighting_metadata = _hazard_sample_weight(hazard, train_frame[hazard], settings)
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(train_frame[columns], train_frame[hazard], **fit_kwargs)
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
        "sample_weighting": weighting_metadata,
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
