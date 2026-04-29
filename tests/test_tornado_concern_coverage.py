import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd
import xarray as xr

from severewx.cli import tornado_concern_coverage as coverage_cli
from severewx.cli.tornado_concern_eval import (
    build_tornado_concern_candidate_rows,
    summarize_tornado_concern_coverage,
    write_tornado_concern_candidate_csv,
)


def _write_forecast(path: Path, date: str) -> None:
    xr.Dataset(coords={"time": pd.to_datetime([f"{date}T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(path)


def _write_verification(path: Path, date: str, category: str, source: str = "local_staged_gfs") -> None:
    payload = {
        "run_summary": {"init_date": date, "evaluation_source": source},
        "evaluation_metadata": {"ingest_summary": {"source": source}},
        "per_day": [{"valid_date": date, "observed_category": category}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_ingest_summary(path: Path, source: str = "local_staged_gfs", source_mode: str = "real", real_ingest_available: bool = True) -> None:
    path.write_text(
        json.dumps({"source": source, "source_mode": source_mode, "real_ingest_available": real_ingest_available}),
        encoding="utf-8",
    )


def _build_fixture_dirs(tmp_path: Path) -> SimpleNamespace:
    outputs = tmp_path / "outputs"
    verification = outputs / "verification"
    processed = tmp_path / "processed"
    labels = tmp_path / "labels"
    interim = tmp_path / "interim"
    outputs.mkdir()
    verification.mkdir(parents=True)
    processed.mkdir()
    labels.mkdir()
    interim.mkdir()

    _write_forecast(outputs / "forecast_products_2024-03-14_00.nc", "2024-03-14")
    _write_forecast(outputs / "forecast_products_2024-05-06_00.nc", "2024-05-06")
    _write_verification(verification / "2024-03-14_verification.json", "2024-03-14", "significant_tornado_outbreak_day")

    pd.DataFrame(
        [
            {"date": "2024-03-14", "tornado_outbreak": 1, "significant_tornado_support": 1},
            {"date": "2024-04-01", "tornado_outbreak": 1, "significant_tornado_support": 0},
            {"date": "2024-06-01", "tornado_outbreak": 1, "significant_tornado_support": 0},
        ]
    ).to_parquet(labels / "outbreaks_2024-03-01_2025-06-30.parquet", index=False)
    pd.DataFrame(
        [
            {"date": "2024-05-06", "hazard": "tornado"},
            {"date": "2024-05-06", "hazard": "tornado"},
            {"date": "2024-05-06", "hazard": "tornado"},
            {"date": "2024-05-07", "hazard": "hail"},
        ]
    ).to_parquet(labels / "spc_reports_2024-03-01_2025-06-30.parquet", index=False)

    _write_ingest_summary(interim / "ingest_summary_2024-03-14_00.json")
    _write_ingest_summary(interim / "ingest_summary_2024-04-01_00.json")
    _write_ingest_summary(interim / "ingest_summary_2024-05-06_00.json")

    return SimpleNamespace(outputs=outputs, verification=verification, labels=labels, interim=interim, processed=processed)


def test_candidate_union_and_priority_tiers(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)

    frame = build_tornado_concern_candidate_rows(paths.outputs, paths.verification, labels_dir=paths.labels, interim_dir=paths.interim)

    assert list(frame["date"]) == ["2024-03-14", "2024-04-01", "2024-06-01", "2024-05-06"]
    assert list(frame["priority_tier"]) == ["sig_tor", "outbreak", "outbreak", "all_tornado"]
    row_0314 = frame.loc[frame["date"] == "2024-03-14"].iloc[0]
    row_0401 = frame.loc[frame["date"] == "2024-04-01"].iloc[0]
    row_0506 = frame.loc[frame["date"] == "2024-05-06"].iloc[0]
    assert bool(row_0314["candidate_source_verification_metadata"])
    assert int(row_0314["significant_tornado_support"]) == 1
    assert int(row_0401["tornado_outbreak"]) == 1
    assert bool(row_0506["candidate_source_spc_reports"])
    assert int(row_0506["tornado_report_count"]) == 3


def test_date_range_and_priority_filters(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)

    ranged = build_tornado_concern_candidate_rows(
        paths.outputs,
        paths.verification,
        labels_dir=paths.labels,
        interim_dir=paths.interim,
        start="2024-04-01",
        end="2024-05-06",
    )
    assert list(ranged["date"]) == ["2024-04-01", "2024-05-06"]

    assert list(
        build_tornado_concern_candidate_rows(
            paths.outputs,
            paths.verification,
            labels_dir=paths.labels,
            interim_dir=paths.interim,
            priority_tier="sig_tor",
        )["date"]
    ) == ["2024-03-14"]
    assert list(
        build_tornado_concern_candidate_rows(
            paths.outputs,
            paths.verification,
            labels_dir=paths.labels,
            interim_dir=paths.interim,
            priority_tier="outbreak",
        )["date"]
    ) == ["2024-04-01", "2024-06-01"]
    assert list(
        build_tornado_concern_candidate_rows(
            paths.outputs,
            paths.verification,
            labels_dir=paths.labels,
            interim_dir=paths.interim,
            priority_tier="all_tornado",
        )["date"]
    ) == ["2024-05-06"]


def test_failure_reasons_are_deterministic(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)
    frame = build_tornado_concern_candidate_rows(paths.outputs, paths.verification, labels_dir=paths.labels, interim_dir=paths.interim)

    reasons = dict(zip(frame["date"], frame["failure_reason"], strict=False))
    assert reasons["2024-03-14"] == ""
    assert reasons["2024-04-01"] == "missing_forecast_artifacts"
    assert reasons["2024-05-06"] == "missing_verification_artifacts"
    assert reasons["2024-06-01"] == "not_real_ingest_confirmed"


def test_candidate_csv_writing_has_required_columns_and_order(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)
    frame = build_tornado_concern_candidate_rows(paths.outputs, paths.verification, labels_dir=paths.labels, interim_dir=paths.interim)
    output_csv = tmp_path / "ranked.csv"

    write_tornado_concern_candidate_csv(output_csv, frame)
    written = pd.read_csv(output_csv)

    assert list(written["date"]) == ["2024-03-14", "2024-04-01", "2024-06-01", "2024-05-06"]
    for column in [
        "date",
        "priority_tier",
        "has_spc_tornado_reports",
        "tornado_report_count",
        "tornado_outbreak",
        "significant_tornado_support",
        "candidate_source_verification_metadata",
        "candidate_source_outbreak_parquet",
        "candidate_source_spc_reports",
        "verification_real_ingest",
        "verification_metadata_present",
        "interim_ingest_summary_present",
        "interim_source",
        "interim_source_mode",
        "interim_real_ingest_available_flag",
        "interim_ingest_real_confirmed",
        "real_ingest_confirmed",
        "forecast_metadata_present",
        "forecast_metadata_source",
        "forecast_metadata_source_mode",
        "forecast_metadata_real_ingest_available",
        "archive_metadata_present",
        "archive_status",
        "archive_forecast_source",
        "archive_real_data",
        "staged_input_dir_present",
        "staged_input_file_count",
        "forecast_artifacts_present",
        "verification_artifacts_present",
        "fully_evaluable_initial",
        "final_ready",
        "failure_reason",
        "real_ingest_failure_detail",
    ]:
        assert column in written.columns


def test_summarize_tornado_concern_coverage_counts_generated_and_failed_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2024-03-14",
                "priority_tier": "sig_tor",
                "real_ingest_confirmed": True,
                "forecast_artifacts_present": True,
                "forecast_artifacts_present_initial": True,
                "forecast_generated": False,
                "forecast_attempted": False,
                "verification_artifacts_present": True,
                "verification_artifacts_present_initial": True,
                "verification_generated": False,
                "verification_attempted": False,
                "fully_evaluable_initial": True,
                "final_ready": True,
                "failure_reason": "",
            },
            {
                "date": "2024-04-01",
                "priority_tier": "outbreak",
                "real_ingest_confirmed": True,
                "forecast_artifacts_present": True,
                "forecast_artifacts_present_initial": False,
                "forecast_generated": True,
                "forecast_attempted": True,
                "verification_artifacts_present": False,
                "verification_artifacts_present_initial": False,
                "verification_generated": False,
                "verification_attempted": True,
                "fully_evaluable_initial": False,
                "final_ready": False,
                "failure_reason": "verification_failed",
            },
            {
                "date": "2024-05-06",
                "priority_tier": "all_tornado",
                "real_ingest_confirmed": False,
                "forecast_artifacts_present": False,
                "forecast_artifacts_present_initial": False,
                "forecast_generated": False,
                "forecast_attempted": False,
                "verification_artifacts_present": False,
                "verification_artifacts_present_initial": False,
                "verification_generated": False,
                "verification_attempted": False,
                "fully_evaluable_initial": False,
                "final_ready": False,
                "failure_reason": "not_real_ingest_confirmed",
            },
        ]
    )

    summary = summarize_tornado_concern_coverage(frame)
    assert summary["candidate_dates_total"] == 3
    assert summary["sig_tor_candidates_total"] == 1
    assert summary["outbreak_candidates_total"] == 1
    assert summary["all_tornado_candidates_total"] == 1
    assert summary["real_ingest_confirmed_total"] == 2
    assert summary["forecast_artifacts_generated"] == 1
    assert summary["verification_artifacts_failed"] == 1
    assert summary["fully_evaluable_dates_total"] == 1
    assert summary["failed_dates_total"] == 2


def test_tornado_concern_coverage_cli_writes_csv_and_markdown(tmp_path: Path, monkeypatch) -> None:
    paths = _build_fixture_dirs(tmp_path)
    output_csv = tmp_path / "coverage.csv"
    output_md = tmp_path / "coverage.md"

    monkeypatch.setattr(coverage_cli, "load_settings", lambda: object())
    monkeypatch.setattr(coverage_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tornado_concern_coverage",
            "--start",
            "2024-03-01",
            "--end",
            "2024-06-30",
            "--priority-tier",
            "all",
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ],
    )

    coverage_cli.main()

    written = pd.read_csv(output_csv)
    markdown = output_md.read_text(encoding="utf-8")
    assert list(written["date"]) == ["2024-03-14", "2024-04-01", "2024-05-06", "2024-06-01"]
    assert "## Counts" in markdown
    assert "## Breakdown By Priority Tier" in markdown
    assert "## Breakdown By Failure Reason" in markdown
    assert "## Real-Ingest Failure Detail Breakdown" in markdown
    assert "## Top Blocked Dates" in markdown


def test_real_ingest_failure_detail_verification_metadata_not_real(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)
    (paths.verification / "2024-04-02_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-04-02", "evaluation_source": "synthetic_fallback"},
                "evaluation_metadata": {"ingest_summary": {"source": "synthetic_fallback"}},
                "per_day": [{"valid_date": "2024-04-02", "observed_category": "significant_tornado_outbreak_day"}],
            }
        ),
        encoding="utf-8",
    )
    frame = build_tornado_concern_candidate_rows(paths.outputs, paths.verification, labels_dir=paths.labels, interim_dir=paths.interim)
    row = frame.loc[frame["date"] == "2024-04-02"].iloc[0]
    assert row["failure_reason"] == "not_real_ingest_confirmed"
    assert row["real_ingest_failure_detail"] == "verification_metadata_not_real"


def test_real_ingest_failure_detail_interim_summary_synthetic_source(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)
    pd.DataFrame([{"date": "2024-04-10", "hazard": "tornado"}]).to_parquet(
        paths.labels / "spc_reports_extra.parquet", index=False
    )
    _write_ingest_summary(
        paths.interim / "ingest_summary_2024-04-10_00.json",
        source="synthetic_fallback",
        source_mode="synthetic",
        real_ingest_available=False,
    )
    frame = build_tornado_concern_candidate_rows(paths.outputs, paths.verification, labels_dir=paths.labels, interim_dir=paths.interim)
    row = frame.loc[frame["date"] == "2024-04-10"].iloc[0]
    assert row["real_ingest_failure_detail"] == "interim_summary_synthetic_source"


def test_real_ingest_failure_detail_forecast_metadata_synthetic_source(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)
    pd.DataFrame([{"date": "2024-04-11", "hazard": "tornado"}]).to_parquet(
        paths.labels / "spc_reports_forecast_meta.parquet", index=False
    )
    (paths.outputs / "forecast_metadata_2024-04-11_00.json").write_text(
        json.dumps({"ingest_summary": {"source": "synthetic_fallback", "source_mode": "synthetic", "real_ingest_available": False}}),
        encoding="utf-8",
    )
    frame = build_tornado_concern_candidate_rows(paths.outputs, paths.verification, labels_dir=paths.labels, interim_dir=paths.interim)
    row = frame.loc[frame["date"] == "2024-04-11"].iloc[0]
    assert bool(row["forecast_metadata_present"])
    assert row["real_ingest_failure_detail"] == "forecast_metadata_synthetic_source"


def test_real_ingest_failure_detail_archive_status_synthetic_degraded(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)
    pd.DataFrame([{"date": "2024-04-12", "hazard": "tornado"}]).to_parquet(
        paths.labels / "spc_reports_archive_meta.parquet", index=False
    )
    archive_dir = paths.processed / "archive_metadata" / "2024" / "2024-04-12"
    archive_dir.mkdir(parents=True)
    (archive_dir / "00.json").write_text(
        json.dumps({"archive_status": "synthetic_degraded", "forecast_source": "synthetic_fallback", "real_data": False}),
        encoding="utf-8",
    )
    frame = build_tornado_concern_candidate_rows(paths.outputs, paths.verification, labels_dir=paths.labels, interim_dir=paths.interim)
    row = frame.loc[frame["date"] == "2024-04-12"].iloc[0]
    assert bool(row["archive_metadata_present"])
    assert row["real_ingest_failure_detail"] == "archive_status_synthetic_degraded"


def test_real_ingest_failure_detail_staged_inputs_present_without_real_provenance(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)
    pd.DataFrame([{"date": "2024-04-13", "hazard": "tornado"}]).to_parquet(
        paths.labels / "spc_reports_staged.parquet", index=False
    )
    staged_dir = tmp_path / "raw" / "staged_gfs" / "2024-04-13" / "00"
    staged_dir.mkdir(parents=True)
    (staged_dir / "gfs.t00z.pgrb2.0p25.f000.grib2").write_text("x", encoding="utf-8")
    frame = build_tornado_concern_candidate_rows(paths.outputs, paths.verification, labels_dir=paths.labels, interim_dir=paths.interim)
    row = frame.loc[frame["date"] == "2024-04-13"].iloc[0]
    assert bool(row["staged_input_dir_present"])
    assert int(row["staged_input_file_count"]) == 1
    assert row["real_ingest_failure_detail"] == "staged_inputs_present_without_real_provenance"


def test_real_ingest_failure_detail_no_local_real_ingest_evidence(tmp_path: Path) -> None:
    paths = _build_fixture_dirs(tmp_path)
    pd.DataFrame([{"date": "2024-04-14", "hazard": "tornado"}]).to_parquet(
        paths.labels / "spc_reports_none.parquet", index=False
    )
    frame = build_tornado_concern_candidate_rows(paths.outputs, paths.verification, labels_dir=paths.labels, interim_dir=paths.interim)
    row = frame.loc[frame["date"] == "2024-04-14"].iloc[0]
    assert row["real_ingest_failure_detail"] == "no_local_real_ingest_evidence"
