import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from severewx.cli import recover_tornado_concern_real_ingest as recovery_cli
from severewx.cli.tornado_concern_eval import (
    derive_tornado_concern_recovery_status,
    select_tornado_concern_recovery_candidates,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-04-01",
                "priority_tier": "sig_tor",
                "priority_sort_key": 0,
                "failure_reason": "not_real_ingest_confirmed",
                "real_ingest_failure_detail": "no_local_real_ingest_evidence",
                "staged_input_dir_present": False,
                "staged_input_file_count": 0,
                "forecast_artifacts_present": False,
                "verification_artifacts_present": False,
                "forecast_metadata_present": False,
                "real_ingest_confirmed": False,
                "final_ready": False,
            },
            {
                "date": "2024-04-02",
                "priority_tier": "outbreak",
                "priority_sort_key": 1,
                "failure_reason": "not_real_ingest_confirmed",
                "real_ingest_failure_detail": "no_local_real_ingest_evidence",
                "staged_input_dir_present": False,
                "staged_input_file_count": 0,
                "forecast_artifacts_present": False,
                "verification_artifacts_present": False,
                "forecast_metadata_present": False,
                "real_ingest_confirmed": False,
                "final_ready": False,
            },
            {
                "date": "2024-04-03",
                "priority_tier": "all_tornado",
                "priority_sort_key": 2,
                "failure_reason": "not_real_ingest_confirmed",
                "real_ingest_failure_detail": "verification_metadata_not_real",
                "staged_input_dir_present": False,
                "staged_input_file_count": 0,
                "forecast_artifacts_present": False,
                "verification_artifacts_present": False,
                "forecast_metadata_present": False,
                "real_ingest_confirmed": False,
                "final_ready": False,
            },
        ]
    )


def test_select_tornado_concern_recovery_candidates_filters_and_limits() -> None:
    selected = select_tornado_concern_recovery_candidates(
        _candidate_rows(),
        priority_tier="all",
        failure_detail="no_local_real_ingest_evidence",
        max_dates=2,
    )
    assert list(selected["date"]) == ["2024-04-01", "2024-04-02"]


def test_derive_tornado_concern_recovery_status_variants() -> None:
    initial = {"real_ingest_failure_detail": "no_local_real_ingest_evidence"}
    final_ready = {"final_ready": True, "real_ingest_confirmed": True, "real_ingest_failure_detail": ""}
    status, _ = derive_tornado_concern_recovery_status(initial, final_ready)
    assert status == "recovered_ready"

    final_real_only = {"final_ready": False, "real_ingest_confirmed": True, "real_ingest_failure_detail": "missing_verification_artifacts"}
    status, _ = derive_tornado_concern_recovery_status(initial, final_real_only)
    assert status == "recovered_real_ingest_only"

    final_blocked = {"final_ready": False, "real_ingest_confirmed": False, "real_ingest_failure_detail": "no_local_real_ingest_evidence"}
    status, _ = derive_tornado_concern_recovery_status(initial, final_blocked)
    assert status == "still_blocked_no_real_evidence"


def test_recovery_cli_dry_run_writes_csv_and_markdown(tmp_path: Path, monkeypatch) -> None:
    output_csv = tmp_path / "recovery.csv"
    output_md = tmp_path / "recovery.md"
    paths = SimpleNamespace(outputs=tmp_path / "outputs", verification=tmp_path / "verification", labels=tmp_path / "labels", interim=tmp_path / "interim")
    paths.outputs.mkdir()
    paths.verification.mkdir()
    paths.labels.mkdir()
    paths.interim.mkdir()

    monkeypatch.setattr(recovery_cli, "load_settings", lambda: object())
    monkeypatch.setattr(recovery_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(recovery_cli, "build_tornado_concern_candidate_rows", lambda *args, **kwargs: _candidate_rows())
    monkeypatch.setattr(recovery_cli, "_run_cli", lambda command: (_ for _ in ()).throw(AssertionError("dry run must not invoke subprocess")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_tornado_concern_real_ingest",
            "--dry-run",
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ],
    )

    recovery_cli.main()

    written = pd.read_csv(output_csv)
    markdown = output_md.read_text(encoding="utf-8")
    assert list(written["date"]) == ["2024-04-01", "2024-04-02"]
    assert set(written["recovery_status"]) == {"dry_run"}
    assert "## Recovery Status Counts" in markdown
    assert "## Priority Tier Counts" in markdown


def test_recovery_cli_executes_recovery_and_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    output_csv = tmp_path / "recovery.csv"
    output_md = tmp_path / "recovery.md"
    paths = SimpleNamespace(outputs=tmp_path / "outputs", verification=tmp_path / "verification", labels=tmp_path / "labels", interim=tmp_path / "interim")
    paths.outputs.mkdir()
    paths.verification.mkdir()
    paths.labels.mkdir()
    paths.interim.mkdir()

    initial = _candidate_rows().iloc[[0]].copy()
    refreshed = initial.copy()
    refreshed.loc[:, "staged_input_dir_present"] = True
    refreshed.loc[:, "staged_input_file_count"] = 10
    refreshed.loc[:, "forecast_artifacts_present"] = True
    refreshed.loc[:, "forecast_metadata_present"] = True
    refreshed.loc[:, "verification_artifacts_present"] = True
    refreshed.loc[:, "real_ingest_confirmed"] = True
    refreshed.loc[:, "final_ready"] = True
    refreshed.loc[:, "failure_reason"] = ""
    refreshed.loc[:, "real_ingest_failure_detail"] = ""
    calls: list[str] = []

    def fake_builder(*args, **kwargs):
        if kwargs.get("start") == "2024-04-01":
            if calls:
                return refreshed
            return initial
        return _candidate_rows()

    def fake_run(command):
        calls.append(" ".join(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(recovery_cli, "load_settings", lambda: object())
    monkeypatch.setattr(recovery_cli, "build_paths", lambda _settings: paths)
    monkeypatch.setattr(recovery_cli, "build_tornado_concern_candidate_rows", fake_builder)
    monkeypatch.setattr(recovery_cli, "_run_cli", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover_tornado_concern_real_ingest",
            "--priority-tier",
            "sig_tor",
            "--max-dates",
            "1",
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ],
    )

    recovery_cli.main()

    written = pd.read_csv(output_csv)
    assert list(written["date"]) == ["2024-04-01"]
    assert written.iloc[0]["recovery_status"] == "recovered_ready"
    assert "severewx.cli.stage_historical_gfs" in calls[0]
    assert "severewx.cli.run_forecast" in calls[1]
    assert "severewx.cli.verify_day" in calls[2]
