"""Bust-risk heuristics.

The first-pass bust layer is rule-based on purpose. It is designed to penalize:
- cap-hold scenarios
- poor low-level moisture return
- forcing / instability mismatch
- contamination / broad precipitation interference proxies
- hazard disagreement
- weak analog support
- weak outbreak-support coherence
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from severewx.config import AppSettings


def _tier(values: pd.Series, moderate: float, high: float) -> pd.Series:
    return pd.Series(np.where(values >= high, "high", np.where(values >= moderate, "moderate", "low")), index=values.index)


def add_bust_risk(frame: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    weights = settings.get("bustrisk.weights", {})
    outbreak_threshold = float(settings.get("bustrisk.outbreak_threshold", 0.45))

    cap_hold = (
        np.clip((np.abs(frame["cin"]) - 50.0) / 140.0, 0.0, 1.0)
        * np.clip(1.0 - frame["forcing_proxy"] / 85.0, 0.0, 1.0)
        * np.clip(1.0 - frame["ll_shear_proxy"] / 14.0, 0.0, 1.0)
    )
    moisture_failure = (
        np.clip((287.5 - frame["td2m"]) / 8.0, 0.0, 1.0)
        * np.clip(1.0 - frame["moisture_transport"] / 350.0, 0.0, 1.0)
    )
    timing_mismatch = (
        np.clip(frame["prior_run_delta"] / 400.0, 0.0, 1.0)
        * np.clip(np.abs(frame["forcing_instability_overlap"] - frame["outbreak_corridor_index"]) / 1.2, 0.0, 1.0)
    )
    spatial_coverage = frame.get("spatial_coverage", pd.Series(0.5, index=frame.index))
    contamination = (
        np.clip(frame["pwat"] / 45.0, 0.0, 1.0)
        * np.clip(1.0 - frame["lapse_rate_700_500"] / 7.2, 0.0, 1.0)
        * np.clip(spatial_coverage, 0.0, 1.0)
    )
    mode_risk = np.clip(
        (frame["wind_prob"] + frame["hail_prob"] - frame["tornado_prob"] - 0.8 * frame["sig_tor_support"]) + 0.2,
        0.0,
        1.0,
    )
    disagreement = frame.get("disagreement_score", pd.Series(0.5, index=frame.index)).clip(0.0, 1.0)
    analog_penalty = 1.0 - frame.get("analog_outbreak_support", pd.Series(0.0, index=frame.index)).clip(0.0, 1.0)

    expected_outbreak = (
        0.45 * frame["tornado_prob"]
        + 0.20 * frame["any_prob"]
        + 0.20 * frame["sig_tor_support"].clip(0.0, 1.0)
        + 0.15 * frame["tornado_corridor_index"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    outbreak_penalty = np.clip(np.abs(frame["outbreak_risk"].clip(0.0, 1.0) - expected_outbreak), 0.0, 1.0)
    outbreak_penalty = np.where(frame["outbreak_risk"] >= outbreak_threshold, outbreak_penalty, outbreak_penalty * 0.6)

    frame["cap_hold_component"] = cap_hold
    frame["moisture_failure_component"] = moisture_failure
    frame["timing_mismatch_component"] = timing_mismatch
    frame["contamination_component"] = contamination
    frame["mode_risk_component"] = mode_risk
    frame["analog_penalty_component"] = analog_penalty
    frame["outbreak_penalty_component"] = outbreak_penalty

    frame["bust_risk_score"] = (
        weights.get("cap_hold", 0.2) * cap_hold
        + weights.get("moisture_failure", 0.15) * moisture_failure
        + weights.get("timing_mismatch", 0.15) * timing_mismatch
        + weights.get("contamination", 0.15) * contamination
        + weights.get("mode_risk", 0.1) * mode_risk
        + weights.get("disagreement", 0.1) * disagreement
        + weights.get("analog_penalty", 0.1) * analog_penalty
        + weights.get("outbreak_penalty", 0.05) * outbreak_penalty
    ).clip(0.0, 1.0)

    thresholds = settings.get("bustrisk.thresholds", {})
    frame["bust_risk_tier"] = _tier(
        frame["bust_risk_score"],
        float(thresholds.get("moderate", 0.4)),
        float(thresholds.get("high", 0.7)),
    )
    return frame
