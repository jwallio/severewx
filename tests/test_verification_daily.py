import json

import numpy as np
import pandas as pd
import xarray as xr

from severewx.verify.daily import _align_prediction_to_label_grid, verify_daily_probabilities


def test_verify_daily_probabilities_writes_run_level_summary(tmp_path) -> None:
    times = pd.to_datetime(["2026-04-09T00:00:00", "2026-04-09T06:00:00", "2026-04-10T00:00:00"])
    prediction = xr.Dataset(
        {
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.7]], [[0.6]], [[0.2]]])),
            "hail_prob": (("time", "lat", "lon"), np.array([[[0.4]], [[0.3]], [[0.3]]])),
            "wind_prob": (("time", "lat", "lon"), np.array([[[0.5]], [[0.4]], [[0.2]]])),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.8]], [[0.7]], [[0.3]]])),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.75]], [[0.72]], [[0.2]]])),
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[0.8]], [[0.7]], [[0.2]]])),
            "confidence_score": (("time", "lat", "lon"), np.array([[[0.7]], [[0.68]], [[0.3]]])),
            "signal_quality_score": (("time", "lat", "lon"), np.array([[[0.72]], [[0.7]], [[0.28]]])),
            "bust_risk_score": (("time", "lat", "lon"), np.array([[[0.25]], [[0.3]], [[0.6]]])),
        },
        coords={"time": times, "lat": [35.0], "lon": [-97.0]},
    )
    labels = xr.Dataset(
        {
            "tornado": (("date", "lat", "lon"), np.array([[[1]], [[0]]])),
            "hail": (("date", "lat", "lon"), np.array([[[0]], [[0]]])),
            "wind": (("date", "lat", "lon"), np.array([[[0]], [[0]]])),
            "any": (("date", "lat", "lon"), np.array([[[1]], [[0]]])),
        },
        coords={"date": np.array([np.datetime64("2026-04-09"), np.datetime64("2026-04-10")]), "lat": [35.0], "lon": [-97.0]},
    )
    outbreaks = pd.DataFrame(
        {
            "date": ["2026-04-09", "2026-04-10"],
            "any_outbreak": [1, 0],
            "tornado_outbreak": [1, 0],
            "significant_tornado_support": [1, 0],
            "category": ["significant_tornado_outbreak_day", "null_day"],
        }
    )
    output = tmp_path / "verification.json"
    verify_daily_probabilities(
        prediction,
        labels,
        output,
        target_date="2026-04-09",
        outbreak_table=outbreaks,
        training_data_summary={"source_preference": "historical_feature_archive", "row_count": 10, "training_quality_tier": "real-heavy", "guardrails_passed": True},
        evaluation_metadata={"ingest_summary": {"source": "nomads"}},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_summary"]["init_date"] == "2026-04-09"
    assert payload["run_summary"]["training_data_source"] == "historical_feature_archive"
    assert payload["run_summary"]["training_quality_tier"] == "real-heavy"
    assert payload["run_summary"]["evaluation_source"] == "nomads"
    assert len(payload["per_day"]) == 2
    assert payload["tornado_outbreak_case_review"][0]["observed_category"] == "significant_tornado_outbreak_day"


def test_verify_daily_probabilities_aligns_mismatched_label_grid(tmp_path) -> None:
    times = pd.to_datetime(["2026-04-26T00:00:00"])
    prediction = xr.Dataset(
        {
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.7, 0.6], [0.5, 0.4]]], dtype=float)),
            "hail_prob": (("time", "lat", "lon"), np.array([[[0.2, 0.2], [0.2, 0.2]]], dtype=float)),
            "wind_prob": (("time", "lat", "lon"), np.array([[[0.3, 0.3], [0.3, 0.3]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.8, 0.7], [0.6, 0.5]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.4, 0.4], [0.4, 0.4]]], dtype=float)),
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[0.3, 0.3], [0.3, 0.3]]], dtype=float)),
            "confidence_score": (("time", "lat", "lon"), np.array([[[0.5, 0.5], [0.5, 0.5]]], dtype=float)),
            "signal_quality_score": (("time", "lat", "lon"), np.array([[[0.45, 0.45], [0.45, 0.45]]], dtype=float)),
            "bust_risk_score": (("time", "lat", "lon"), np.array([[[0.2, 0.2], [0.2, 0.2]]], dtype=float)),
        },
        coords={"time": times, "lat": [34.0, 36.0], "lon": [-99.0, -97.0]},
    )
    labels = xr.Dataset(
        {
            "tornado": (("date", "lat", "lon"), np.array([[[1]]], dtype=int)),
            "hail": (("date", "lat", "lon"), np.array([[[0]]], dtype=int)),
            "wind": (("date", "lat", "lon"), np.array([[[0]]], dtype=int)),
            "any": (("date", "lat", "lon"), np.array([[[1]]], dtype=int)),
        },
        coords={"date": np.array([np.datetime64("2026-04-26")]), "lat": [35.0], "lon": [-98.0]},
    )
    output = tmp_path / "verification_mismatch.json"
    verify_daily_probabilities(prediction, labels, output, target_date="2026-04-26")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["run_summary"]["n_valid_days"] == 1
    assert len(payload["lead_day_summary"]) == 4


def test_align_prediction_to_label_grid_downsamples_forecast_to_coarse_labels() -> None:
    prediction = xr.Dataset(
        {
            "tornado_prob": (("lat", "lon"), np.arange(25, dtype=float).reshape(5, 5)),
            "hail_prob": (("lat", "lon"), np.full((5, 5), 0.2, dtype=float)),
            "wind_prob": (("lat", "lon"), np.full((5, 5), 0.3, dtype=float)),
            "any_prob": (("lat", "lon"), np.full((5, 5), 0.4, dtype=float)),
            "outbreak_risk": (("lat", "lon"), np.full((5, 5), 0.5, dtype=float)),
            "sig_tor_support": (("lat", "lon"), np.full((5, 5), 0.6, dtype=float)),
            "confidence_score": (("lat", "lon"), np.full((5, 5), 0.7, dtype=float)),
            "signal_quality_score": (("lat", "lon"), np.full((5, 5), 0.8, dtype=float)),
            "bust_risk_score": (("lat", "lon"), np.full((5, 5), 0.1, dtype=float)),
        },
        coords={"lat": [20.0, 20.25, 20.5, 20.75, 21.0], "lon": [-100.0, -99.75, -99.5, -99.25, -99.0]},
    )
    labels = xr.Dataset(
        {
            "tornado": (("lat", "lon"), np.array([[1, 0], [0, 1]], dtype=int)),
            "hail": (("lat", "lon"), np.zeros((2, 2), dtype=int)),
            "wind": (("lat", "lon"), np.zeros((2, 2), dtype=int)),
            "any": (("lat", "lon"), np.array([[1, 0], [0, 1]], dtype=int)),
        },
        coords={"lat": [20.0, 21.0], "lon": [-100.0, -99.0]},
    )

    aligned = _align_prediction_to_label_grid(prediction, labels)

    assert aligned.sizes["lat"] == 2
    assert aligned.sizes["lon"] == 2
    assert aligned["tornado_prob"].values.tolist() == [[0.0, 4.0], [20.0, 24.0]]
