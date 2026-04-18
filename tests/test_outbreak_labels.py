import pandas as pd

from severewx.config import load_settings
from severewx.labels.outbreaks import build_outbreak_table, classify_outbreak_day


def test_outbreak_classification_flags_significant_tornado_support() -> None:
    settings = load_settings()
    reports = pd.DataFrame(
        {
            "date": ["2024-04-27"] * 18,
            "time": ["2100"] * 18,
            "lat": [35.0] * 18,
            "lon": [-97.0] * 18,
            "hazard": ["tornado"] * 18,
            "magnitude": [2.0] * 18,
            "significant": [1] * 6 + [0] * 12,
            "source_name": ["spc_csv"] * 18,
            "is_real_source": [True] * 18,
            "ingest_timestamp": ["2026-04-13T00:00:00Z"] * 18,
        }
    )
    daily = classify_outbreak_day(reports, settings)
    assert daily["tornado_outbreak"] == 1
    assert daily["significant_tornado_support"] == 1
    assert daily["category"] == "significant_tornado_outbreak_day"
    assert daily["dominant_region"] != "none"
    table = build_outbreak_table(reports, "2024-04-27", "2024-04-27", settings)
    assert int(table.loc[0, "any_outbreak"]) == 1
    assert bool(table.loc[0, "report_source_is_real"]) is True
    assert table.loc[0, "report_source_name"] == "spc_csv"


def test_outbreak_classification_distinguishes_active_non_outbreak_from_null() -> None:
    settings = load_settings()
    active_reports = pd.DataFrame(
        {
            "date": ["2024-05-01"] * 9,
            "time": ["2200"] * 9,
            "lat": [33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0],
            "lon": [-97.0, -97.0, -96.5, -96.5, -96.0, -96.0, -95.5, -95.5, -95.0],
            "hazard": ["hail", "hail", "wind", "hail", "wind", "hail", "wind", "hail", "wind"],
            "magnitude": [1.0, 1.25, 55.0, 1.5, 58.0, 1.2, 60.0, 1.0, 52.0],
            "significant": [0] * 9,
        }
    )
    active = classify_outbreak_day(active_reports, settings)
    assert active["active_non_outbreak_severe"] == 1
    assert active["category"] == "active_non_outbreak_severe_day"

    null = classify_outbreak_day(active_reports.iloc[0:0], settings)
    assert null["any_outbreak"] == 0
    assert null["category"] == "null_day"
