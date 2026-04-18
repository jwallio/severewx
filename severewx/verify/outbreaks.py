"""Outbreak verification helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def outbreak_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float = 0.5) -> dict[str, float]:
    observed = y_true.to_numpy(dtype=int)
    forecast = (y_prob.to_numpy(dtype=float) >= threshold).astype(int)
    hits = float(np.sum((observed == 1) & (forecast == 1)))
    misses = float(np.sum((observed == 1) & (forecast == 0)))
    false_alarms = float(np.sum((observed == 0) & (forecast == 1)))
    correct_nulls = float(np.sum((observed == 0) & (forecast == 0)))
    return {
        "outbreak_hits": hits,
        "outbreak_misses": misses,
        "false_outbreak_alarms": false_alarms,
        "correct_nulls": correct_nulls,
        "outbreak_hit_rate": hits / (hits + misses) if (hits + misses) else float("nan"),
        "outbreak_false_alarm_ratio": false_alarms / (hits + false_alarms) if (hits + false_alarms) else float("nan"),
    }


def case_review_table(per_day: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "valid_date",
        "lead_day",
        "forecast_outbreak_prob",
        "forecast_tornado_prob",
        "forecast_sig_tor_support",
        "observed_any_outbreak",
        "observed_tornado_outbreak",
        "observed_significant_tornado_support",
        "observed_category",
    ]
    available = [column for column in columns if column in per_day.columns]
    return per_day[available].sort_values(["observed_tornado_outbreak", "forecast_outbreak_prob"], ascending=[False, False]).reset_index(drop=True)
