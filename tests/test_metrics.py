import numpy as np

from severewx.verify.metrics import brier_score, frequency_bias, safe_roc_auc, threshold_summary


def test_metrics_basic_values() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.6, 0.4])
    assert round(brier_score(y_true, y_prob), 4) == 0.0925
    assert round(frequency_bias(y_true, y_prob, threshold=0.5), 4) == 1.0
    assert round(safe_roc_auc(y_true, y_prob), 4) == 1.0
    summary = threshold_summary(y_true, y_prob, threshold=0.5)
    assert summary["hits"] == 2.0
    assert summary["false_alarms"] == 0.0
