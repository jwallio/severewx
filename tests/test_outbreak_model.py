import pandas as pd
import pytest

from severewx.models.outbreak import outbreak_training_frame


def test_outbreak_training_frame_falls_back_when_spatial_coverage_missing() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-04-09", "2026-04-09"],
            "lead_day": [1, 1],
            "sig_tor_support": [0.7, 0.9],
            "tornado_favored_overlap": [0.8, 1.1],
            "outbreak_corridor_index": [0.4, 0.8],
            "tornado_corridor_index": [0.5, 0.6],
            "synoptic_support": [60.0, 70.0],
            "forcing_proxy": [45.0, 50.0],
            "cape": [1400.0, 1800.0],
            "shear_0_6km": [24.0, 30.0],
            "analog_outbreak_support": [0.3, 0.6],
            "analog_sigtor_support": [0.2, 0.5],
            "any_outbreak": [0, 1],
            "tornado_outbreak": [0, 1],
            "significant_tornado_support": [0, 0],
        }
    )

    grouped = outbreak_training_frame(frame)

    assert len(grouped) == 1
    assert grouped.loc[0, "mean_spatial_coverage"] == pytest.approx(0.6)
    assert grouped.loc[0, "outbreak_target"] == 1


def test_outbreak_training_frame_drops_unlabeled_groups() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-04-09", "2026-04-09", "2026-04-10", "2026-04-10"],
            "lead_day": [1, 1, 1, 1],
            "sig_tor_support": [0.7, 0.9, 0.3, 0.2],
            "tornado_favored_overlap": [0.8, 1.1, 0.4, 0.3],
            "outbreak_corridor_index": [0.4, 0.8, 0.2, 0.2],
            "tornado_corridor_index": [0.5, 0.6, 0.2, 0.2],
            "synoptic_support": [60.0, 70.0, 45.0, 40.0],
            "forcing_proxy": [45.0, 50.0, 30.0, 28.0],
            "cape": [1400.0, 1800.0, 900.0, 850.0],
            "shear_0_6km": [24.0, 30.0, 16.0, 14.0],
            "analog_outbreak_support": [0.3, 0.6, 0.1, 0.1],
            "analog_sigtor_support": [0.2, 0.5, 0.1, 0.1],
            "spatial_coverage": [0.4, 0.7, 0.2, 0.2],
            "any_outbreak": [0.0, 1.0, float("nan"), float("nan")],
            "tornado_outbreak": [0.0, 1.0, float("nan"), float("nan")],
            "significant_tornado_support": [0.0, 0.0, float("nan"), float("nan")],
        }
    )

    grouped = outbreak_training_frame(frame)

    assert grouped["date"].tolist() == ["2026-04-09"]
    assert grouped["outbreak_target"].isna().sum() == 0
    assert grouped["outbreak_target"].dtype.kind in {"i", "u"}
