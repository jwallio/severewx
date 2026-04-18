import pandas as pd

from severewx.config import load_settings
from severewx.confidence.score import add_confidence


def test_confidence_score_and_tier_exist() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "tornado_prob": [0.7, 0.2],
            "hail_prob": [0.5, 0.1],
            "wind_prob": [0.6, 0.2],
            "any_prob": [0.8, 0.3],
            "prior_run_delta": [80.0, 350.0],
            "analog_similarity": [0.8, 0.2],
            "analog_outbreak_support": [0.75, 0.15],
            "analog_mode_coherence": [0.7, 0.1],
            "outbreak_risk": [0.75, 0.15],
            "sig_tor_support": [0.8, 0.2],
            "tornado_corridor_index": [0.7, 0.1],
            "tornado_calibration_quality": [0.8, 0.7],
            "hail_calibration_quality": [0.8, 0.7],
            "wind_calibration_quality": [0.8, 0.7],
            "any_calibration_quality": [0.8, 0.7],
        }
    )
    scored = add_confidence(frame, settings)
    assert scored["confidence_score"].between(0, 1).all()
    assert scored["signal_quality_score"].between(0, 1).all()
    assert set(scored["confidence_tier"]) <= {"low", "moderate", "high"}
    assert scored.loc[0, "confidence_score"] > scored.loc[1, "confidence_score"]
