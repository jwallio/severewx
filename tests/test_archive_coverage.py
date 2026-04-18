import json

import pandas as pd

from severewx.archive.coverage import archive_coverage_summary, write_archive_coverage_summary
from severewx.config import load_settings
from severewx.ingest.storage import historical_feature_path, historical_metadata_path
from severewx.utils.paths import build_paths


def test_archive_coverage_summary_reports_lead_and_region_counts(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)

    feature_file = historical_feature_path(paths, "2026-04-09", "00")
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2026-04-09", "2026-04-10"],
            "lead_day": [1, 2],
            "region": ["south_plains", "southeast"],
            "forecast_source_kind": ["local_staged_gfs", "synthetic"],
            "archive_source": ["historical_feature_archive", ""],
            "archive_chunk_status": ["local_real", "synthetic_degraded"],
            "category": ["tornado_outbreak_day", "null_day"],
            "tornado": [1, 0],
            "hail": [0, 0],
            "wind": [0, 1],
            "any": [1, 1],
        }
    ).to_parquet(feature_file, index=False)

    metadata_file = historical_metadata_path(paths, "2026-04-09", "00")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(json.dumps({"init_date": "2026-04-09", "cycle": "00", "real_data": True, "archive_status": "local_real"}), encoding="utf-8")

    summary = archive_coverage_summary(paths, training_data_summary={"real_archive_rows": 1, "synthetic_rows": 1}, settings=settings)
    assert summary["archive_chunk_count"] == 1
    assert any(row["lead_day"] == 1 for row in summary["coverage_by_lead_day"])
    assert any(row["region"] == "south_plains" for row in summary["coverage_by_region"])
    assert summary["training_real_row_fraction"] == 0.5
    assert summary["archive_status_counts"]["local_real"] == 1
    assert summary["archive_status_reporting_counts"]["local_real"] == 1
    assert any(row["archive_chunk_status"] == "local_real" for row in summary["coverage_by_archive_status"])
    assert "archive_guardrails" in summary

    json_path, csv_path = write_archive_coverage_summary(paths, training_data_summary={"real_archive_rows": 1, "synthetic_rows": 1}, settings=settings)
    assert json_path.exists()
    assert csv_path.exists()
