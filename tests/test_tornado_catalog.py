import json
import sys
from pathlib import Path

import pandas as pd

from severewx.cli.build_tornado_catalog import main as build_tornado_catalog_main
from severewx.labels.tornado_catalog import (
    TornadoCatalogThresholds,
    build_daily_tornado_summary,
    build_tornado_candidate_dates,
    load_tornado_features,
)


def _feature(
    feature_id: str,
    start: str,
    f_scale: int,
    states: list[str],
    path_length: float | None = None,
) -> dict:
    properties = {
        "id": feature_id,
        "datetime-start": start,
        "f-scale": f_scale,
        "scale": "EF" if start >= "2007-01-01" else "F",
        "states": states,
    }
    if path_length is not None:
        properties["path-length"] = path_length
    return {"type": "Feature", "geometry": None, "properties": properties}


def test_build_daily_tornado_summary_aggregates_unique_ids_and_states() -> None:
    features = [
        _feature("a", "2024-03-14T00:30:00Z", 3, ["OK"], 10.0),
        _feature("a", "2024-03-14T00:30:00Z", 3, ["OK"], 10.0),
        _feature("b", "2024-03-14T03:00:00Z", 1, ["KS"], 5.5),
        _feature("c", "2024-03-15T04:00:00Z", 0, ["TX"], None),
    ]
    features[0]["properties"]["f-scale"] = [2, 3]
    summary = build_daily_tornado_summary(features)

    march14 = summary.loc[summary["date"] == "2024-03-14"].iloc[0]
    assert int(march14["tornado_count"]) == 2
    assert int(march14["significant_tornado_count"]) == 1
    assert int(march14["max_f_scale"]) == 3
    assert march14["states_affected"] == "KS;OK"
    assert int(march14["states_affected_count"]) == 2
    assert float(march14["total_path_length_mi"]) == 15.5
    assert march14["outbreak_class"] == "moderate_tornado_day"


def test_candidate_dates_prioritize_stronger_classes_first() -> None:
    summary = pd.DataFrame(
        [
            {"date": "2024-03-14", "tornado_count": 12, "significant_tornado_count": 3, "max_f_scale": 4, "states_affected": "AL;MS", "states_affected_count": 2, "total_path_length_mi": 120.0, "outbreak_class": "significant_tornado_outbreak", "outbreak_rank": 4},
            {"date": "2024-03-15", "tornado_count": 8, "significant_tornado_count": 2, "max_f_scale": 3, "states_affected": "OK", "states_affected_count": 1, "total_path_length_mi": 80.0, "outbreak_class": "tornado_outbreak", "outbreak_rank": 3},
            {"date": "2024-03-13", "tornado_count": 5, "significant_tornado_count": 1, "max_f_scale": 2, "states_affected": "TX", "states_affected_count": 1, "total_path_length_mi": 40.0, "outbreak_class": "moderate_tornado_day", "outbreak_rank": 2},
            {"date": "2024-03-12", "tornado_count": 1, "significant_tornado_count": 0, "max_f_scale": 1, "states_affected": "IA", "states_affected_count": 1, "total_path_length_mi": 2.0, "outbreak_class": "low_tornado_day", "outbreak_rank": 1},
        ]
    )

    candidates = build_tornado_candidate_dates(summary)
    assert candidates["date"].tolist() == ["2024-03-14", "2024-03-15", "2024-03-13"]


def test_build_tornado_catalog_cli_writes_outputs(tmp_path, monkeypatch) -> None:
    root = tmp_path
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    features = [
        _feature("a", "2024-03-14T00:30:00Z", 3, ["OK"], 10.0),
        _feature("b", "2024-03-14T03:00:00Z", 1, ["KS"], 5.5),
        _feature("c", "2024-03-15T04:00:00Z", 0, ["TX"], None),
    ]
    input_path = data_dir / "tor.json"
    input_path.write_text(json.dumps(features), encoding="utf-8")
    config_path = root / "config.yaml"
    config_path.write_text("paths:\n  root: .\n", encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setenv("SEVEREWX_CONFIG", str(config_path))
    monkeypatch.setattr(sys, "argv", ["build_tornado_catalog", "--input", str(input_path)])

    build_tornado_catalog_main()

    summary_path = root / "data" / "interim" / "tornado_daily_summary.csv"
    candidate_path = root / "data" / "interim" / "tornado_candidate_dates.csv"
    summary = pd.read_csv(summary_path)
    candidates = pd.read_csv(candidate_path)
    assert summary_path.exists()
    assert candidate_path.exists()
    assert summary["date"].tolist() == ["2024-03-14", "2024-03-15"]
    assert candidates["date"].tolist() == ["2024-03-14"]
