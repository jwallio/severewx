"""Verification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

SPC_TORNADO_THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    if not np.any(mask):
        return float("nan")
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    return float(np.mean((y_prob - y_true) ** 2))


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if y_true.size == 0:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def frequency_bias(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    forecast_yes = np.sum(y_prob >= threshold)
    observed_yes = np.sum(y_true > 0)
    if observed_yes == 0:
        return float("nan")
    return float(forecast_yes / observed_yes)


def threshold_summary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    raw_true = np.asarray(y_true, dtype=float)
    raw_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(raw_true) & np.isfinite(raw_prob)
    y_true = raw_true[mask] > 0
    y_fcst = raw_prob[mask] >= threshold
    hits = float(np.sum(y_true & y_fcst))
    misses = float(np.sum(y_true & ~y_fcst))
    false_alarms = float(np.sum(~y_true & y_fcst))
    hit_rate = hits / (hits + misses) if (hits + misses) else float("nan")
    false_alarm_rate = false_alarms / (hits + false_alarms) if (hits + false_alarms) else float("nan")
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) else float("nan")
    forecast_yes = hits + false_alarms
    observed_yes = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "forecast_yes": forecast_yes,
        "observed_yes": observed_yes,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "critical_success_index": csi,
    }


def threshold_summaries(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: tuple[float, ...] = SPC_TORNADO_THRESHOLDS,
) -> list[dict[str, float]]:
    summaries: list[dict[str, float]] = []
    for threshold in thresholds:
        summary = threshold_summary(y_true, y_prob, threshold=threshold)
        summary["threshold"] = float(threshold)
        summaries.append(summary)
    return summaries


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(n_bins):
        left = edges[index]
        right = edges[index + 1]
        if index == n_bins - 1:
            in_bin = (y_prob >= left) & (y_prob <= right)
        else:
            in_bin = (y_prob >= left) & (y_prob < right)
        count = int(np.sum(in_bin))
        rows.append(
            {
                "bin_start": float(left),
                "bin_end": float(right),
                "count": float(count),
                "mean_forecast": float(np.mean(y_prob[in_bin])) if count else float("nan"),
                "observed_frequency": float(np.mean(y_true[in_bin])) if count else float("nan"),
            }
        )
    return rows
