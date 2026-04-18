import pandas as pd

from severewx.bustrisk.rules import add_bust_risk
from severewx.config import load_settings


def test_bust_risk_increases_in_bad_environment() -> None:
    settings = load_settings()
    frame = pd.DataFrame(
        {
            "cin": [-180.0, -20.0],
            "td2m": [281.0, 291.0],
            "moisture_transport": [120.0, 430.0],
            "prior_run_delta": [400.0, 50.0],
            "forcing_proxy": [20.0, 90.0],
            "forcing_instability_overlap": [0.1, 0.9],
            "outbreak_corridor_index": [0.9, 0.85],
            "pwat": [42.0, 20.0],
            "lapse_rate_700_500": [5.5, 7.5],
            "ll_shear_proxy": [6.0, 17.0],
            "any_prob": [0.7, 0.6],
            "wind_prob": [0.5, 0.4],
            "hail_prob": [0.5, 0.3],
            "tornado_prob": [0.1, 0.5],
            "sig_tor_support": [0.1, 0.8],
            "tornado_corridor_index": [0.2, 0.85],
            "analog_outbreak_support": [0.1, 0.9],
            "disagreement_score": [0.8, 0.2],
            "outbreak_risk": [0.6, 0.7],
        }
    )
    scored = add_bust_risk(frame, settings)
    assert scored.loc[0, "bust_risk_score"] > scored.loc[1, "bust_risk_score"]
    assert set(scored["bust_risk_tier"]) <= {"low", "moderate", "high"}
