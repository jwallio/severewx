"""Confidence scoring.

The intent here is to make confidence reflect signal quality, not just raw probability.

Formula outline:
- `signal_quality_score`: agreement + stability + calibration + analog support + outbreak coherence
- `confidence_score`: a blended product-confidence score that still respects probability magnitude
  but is capped by poor signal quality
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from severewx.config import AppSettings
from severewx.confidence.agreement import add_agreement_columns


def _tier(values: pd.Series, moderate: float, high: float) -> pd.Series:
    return pd.Series(np.where(values >= high, "high", np.where(values >= moderate, "moderate", "low")), index=values.index)


def add_confidence(frame: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    weights = settings.get("confidence.weights", {})
    frame = add_agreement_columns(frame)

    probability = frame[["tornado_prob", "hail_prob", "wind_prob", "any_prob"]].max(axis=1).clip(0.0, 1.0)
    stability = (1.0 - np.clip(frame["prior_run_delta"] / 450.0, 0.0, 1.0)).clip(0.0, 1.0)
    calibration = frame[
        ["tornado_calibration_quality", "hail_calibration_quality", "wind_calibration_quality", "any_calibration_quality"]
    ].mean(axis=1).clip(0.0, 1.0)
    analog = (
        0.35 * frame["analog_similarity"].clip(0.0, 1.0)
        + 0.35 * frame.get("analog_outbreak_support", pd.Series(0.0, index=frame.index)).clip(0.0, 1.0)
        + 0.30 * frame.get("analog_mode_coherence", pd.Series(0.0, index=frame.index)).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    outbreak = frame["outbreak_risk"].clip(0.0, 1.0)
    expected_outbreak_support = (
        0.45 * frame["tornado_prob"]
        + 0.20 * frame["any_prob"]
        + 0.20 * frame["sig_tor_support"].clip(0.0, 1.0)
        + 0.15 * frame["tornado_corridor_index"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    outbreak_coherence = (1.0 - np.abs(outbreak - expected_outbreak_support)).clip(0.0, 1.0)

    frame["signal_quality_score"] = (
        weights.get("stability", 0.15) * stability
        + weights.get("calibration", 0.15) * calibration
        + weights.get("analog", 0.15) * analog
        + weights.get("agreement", 0.15) * frame["agreement_score"]
        + weights.get("outbreak_coherence", 0.15) * outbreak_coherence
        + weights.get("outbreak", 0.10) * outbreak
    )
    frame["signal_quality_score"] = frame["signal_quality_score"].clip(0.0, 1.0)
    frame["outbreak_coherence"] = outbreak_coherence

    frame["confidence_score"] = (
        0.55 * frame["signal_quality_score"]
        + weights.get("probability", 0.15) * probability
        + weights.get("signal_quality", 0.15) * frame["signal_quality_score"]
        + 0.15 * np.minimum(probability, frame["agreement_score"])
    ).clip(0.0, 1.0)

    thresholds = settings.get("confidence.thresholds", {})
    moderate = float(thresholds.get("moderate", 0.4))
    high = float(thresholds.get("high", 0.7))
    frame["signal_quality_tier"] = _tier(frame["signal_quality_score"], moderate, high)
    frame["confidence_tier"] = _tier(frame["confidence_score"], moderate, high)
    return frame
