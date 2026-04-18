"""Verification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def frequency_bias(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    forecast_yes = np.sum(np.asarray(y_prob) >= threshold)
    observed_yes = np.sum(np.asarray(y_true) > 0)
    if observed_yes == 0:
        return float("nan")
    return float(forecast_yes / observed_yes)


def threshold_summary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true) > 0
    y_fcst = np.asarray(y_prob) >= threshold
    hits = float(np.sum(y_true & y_fcst))
    misses = float(np.sum(y_true & ~y_fcst))
    false_alarms = float(np.sum(~y_true & y_fcst))
    hit_rate = hits / (hits + misses) if (hits + misses) else float("nan")
    false_alarm_rate = false_alarms / (hits + false_alarms) if (hits + false_alarms) else float("nan")
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
    }
