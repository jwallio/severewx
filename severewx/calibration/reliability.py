"""Reliability summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(y_prob, bins, right=True)
    rows: list[dict[str, float | int]] = []
    for bin_index in range(1, n_bins + 1):
        mask = indices == bin_index
        if not np.any(mask):
            rows.append({"bin": bin_index, "count": 0, "mean_prob": 0.0, "observed_freq": 0.0})
            continue
        rows.append(
            {
                "bin": bin_index,
                "count": int(mask.sum()),
                "mean_prob": float(y_prob[mask].mean()),
                "observed_freq": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)
