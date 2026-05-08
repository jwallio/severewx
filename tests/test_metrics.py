import math

import numpy as np

from severewx.verify.metrics import brier_score, frequency_bias, reliability_bins, safe_roc_auc, threshold_summaries, threshold_summary


def test_metrics_basic_values() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.6, 0.4])
    assert round(brier_score(y_true, y_prob), 4) == 0.0925
    assert round(frequency_bias(y_true, y_prob, threshold=0.5), 4) == 1.0
    assert round(safe_roc_auc(y_true, y_prob), 4) == 1.0
    summary = threshold_summary(y_true, y_prob, threshold=0.5)
    assert summary["hits"] == 2.0
    assert summary["false_alarms"] == 0.0
    assert summary["critical_success_index"] == 1.0


def test_metrics_ignore_nonfinite_pairs() -> None:
    y_true = np.array([0, 1, 1, 0, 1], dtype=float)
    y_prob = np.array([0.1, 0.8, np.nan, np.inf, 0.6], dtype=float)

    assert round(brier_score(y_true, y_prob), 4) == 0.07
    assert round(safe_roc_auc(y_true, y_prob), 4) == 1.0
    assert round(frequency_bias(y_true, y_prob, threshold=0.5), 4) == 1.0
    summary = threshold_summary(y_true, y_prob, threshold=0.5)
    assert summary["hits"] == 2.0
    assert summary["false_alarms"] == 0.0


def test_spc_threshold_summaries_and_reliability_bins() -> None:
    y_true = np.array([0, 1, 1, 0, 0])
    y_prob = np.array([0.01, 0.06, 0.2, 0.35, 0.0])

    summaries = threshold_summaries(y_true, y_prob)

    assert [row["threshold"] for row in summaries] == [0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6]
    assert summaries[0]["hits"] == 2.0
    assert summaries[0]["false_alarms"] == 1.0
    assert summaries[-1]["misses"] == 2.0

    bins = reliability_bins(y_true, y_prob, n_bins=2)
    assert len(bins) == 2
    assert bins[0]["count"] == 5.0
    assert math.isnan(bins[1]["observed_frequency"])
