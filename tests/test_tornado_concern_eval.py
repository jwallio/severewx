import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from severewx.cli.tornado_concern_eval import (
    audit_real_case_init_dates,
    BASELINE_VS_MASKED_CORE_STEEPER_VARIANTS,
    BROADER_VALIDATION_RAW_CORE_VARIANTS,
    CHECKPOINT_COMPONENT_VARIANTS,
    COMPONENT_VARIANTS,
    CORE_VARIANTS,
    RAW_CORE_VARIANTS,
    SOURCE_VARIANTS,
    _apply_component_variant_to_frame,
    _apply_core_variant_to_frame,
    _apply_raw_core_variant_to_frame,
    _apply_source_variant_to_frame,
    SCORE_VARIANTS,
    _apply_score_variant_to_frame,
    _daily_tornado_concern_components,
    apply_tornado_preference_mode,
    discover_real_tornado_relevant_candidate_dates,
    evaluate_tornado_concern_for_artifacts,
    missing_real_case_forecast_artifacts,
    select_real_case_init_dates,
    summarize_failure_patterns,
    summarize_component_variants_from_artifacts,
    summarize_broader_validation_checkpoint_from_artifacts,
    summarize_baseline_vs_masked_core_steeper_from_artifacts,
    summarize_component_checkpoint_from_artifacts,
    summarize_core_variants_from_artifacts,
    summarize_raw_core_variants_from_artifacts,
    summarize_real_case_backfill,
    summarize_score_variants,
    summarize_source_variants_from_artifacts,
    summarize_case_windows,
    write_init_dates_file,
)


def test_evaluate_tornado_concern_for_artifacts_scores_outbreak_day_higher(tmp_path: Path) -> None:
    times = pd.to_datetime(["2024-04-26T00:00:00", "2024-04-27T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[2.4, 2.1], [0.6, 0.4]], [[1.4, 1.2], [0.4, 0.3]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.19, 0.18], [0.05, 0.04]], [[0.15, 0.14], [0.05, 0.03]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[3.2, 2.9], [0.5, 0.4]], [[2.0, 1.8], [0.4, 0.3]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.002, 0.0015], [0.0, 0.0]], [[0.0005, 0.0004], [0.0, 0.0]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.04, 0.03], [0.0, 0.0]], [[0.01, 0.008], [0.0, 0.0]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.6, 0.5], [0.1, 0.1]], [[0.2, 0.2], [0.05, 0.05]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0, 36.0], "lon": [-98.0, -97.0]},
    )
    prediction_path = tmp_path / "forecast_products_2024-04-26_00.nc"
    prediction.to_netcdf(prediction_path)

    verification_payload = {
        "run_summary": {
            "init_date": "2024-04-26",
            "evaluation_source": "local_staged_gfs",
        },
        "evaluation_metadata": {
            "ingest_summary": {
                "source": "local_staged_gfs",
            }
        },
        "per_day": [
            {
                "valid_date": "2024-04-26",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
            },
            {
                "valid_date": "2024-04-27",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
            },
        ]
    }
    verification_path = tmp_path / "2024-04-26_verification.json"
    verification_path.write_text(json.dumps(verification_payload), encoding="utf-8")

    result = evaluate_tornado_concern_for_artifacts(prediction_path, verification_path, init_date="2024-04-26")

    assert list(result["init_date"]) == ["2024-04-26", "2024-04-26"]
    assert list(result["valid_date"]) == ["2024-04-26", "2024-04-27"]
    assert bool(result["is_real_ingest"].all())
    assert "tornado_concern_score" in result.columns
    assert "learned_tornado_concern_prob" in result.columns
    assert "support_top" in result.columns
    assert "context_top" in result.columns
    assert float(result.loc[result["valid_date"] == "2024-04-26", "learned_tornado_concern_prob"].iloc[0]) == 0.6
    assert float(result.loc[result["valid_date"] == "2024-04-26", "tornado_concern_score"].iloc[0]) > float(
        result.loc[result["valid_date"] == "2024-04-27", "tornado_concern_score"].iloc[0]
    )

    summary = summarize_case_windows(result)
    assert list(summary["init_date"]) == ["2024-04-26"]
    assert summary.loc[0, "top_valid_date"] == "2024-04-26"
    assert float(summary.loc[0, "top_learned_tornado_concern_prob"]) == 0.6


def test_select_real_case_init_dates_prefers_real_tornado_relevant_windows(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    verification_dir = tmp_path / "verification"
    labels_dir = tmp_path / "labels"
    interim_dir = tmp_path / "interim"
    outputs_dir.mkdir()
    verification_dir.mkdir()
    labels_dir.mkdir()
    interim_dir.mkdir()

    xr.Dataset(coords={"time": pd.to_datetime(["2024-03-14T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(
        outputs_dir / "forecast_products_2024-03-14_00.nc"
    )
    xr.Dataset(coords={"time": pd.to_datetime(["2024-04-26T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(
        outputs_dir / "forecast_products_2024-04-26_00.nc"
    )
    xr.Dataset(coords={"time": pd.to_datetime(["2024-05-06T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(
        outputs_dir / "forecast_products_2024-05-06_00.nc"
    )
    xr.Dataset(coords={"time": pd.to_datetime(["2024-04-02T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(
        outputs_dir / "forecast_products_2024-04-02_00.nc"
    )

    (verification_dir / "2024-03-14_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-03-14", "evaluation_source": "local_staged_gfs"},
                "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                "per_day": [{"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day"}],
            }
        ),
        encoding="utf-8",
    )
    (verification_dir / "2024-04-26_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
                "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                "per_day": [{"valid_date": "2024-04-28", "observed_category": "tornado_outbreak_day"}],
            }
        ),
        encoding="utf-8",
    )
    (verification_dir / "2024-04-02_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-04-02", "evaluation_source": "synthetic_fallback"},
                "evaluation_metadata": {"ingest_summary": {"source": "synthetic_fallback"}},
                "per_day": [{"valid_date": "2024-04-02", "observed_category": "significant_tornado_outbreak_day"}],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"date": "2024-05-06", "tornado_outbreak": 1, "significant_tornado_support": 0},
            {"date": "2024-04-10", "tornado_outbreak": 1, "significant_tornado_support": 1},
        ]
    ).to_parquet(labels_dir / "outbreaks_2024-03-01_2025-06-30.parquet", index=False)
    (interim_dir / "ingest_summary_2024-05-06_00.json").write_text(
        json.dumps({"source": "local_staged_gfs", "source_mode": "real", "real_ingest_available": True}),
        encoding="utf-8",
    )
    (interim_dir / "ingest_summary_2024-04-10_00.json").write_text(
        json.dumps({"source": "local_staged_gfs", "source_mode": "real", "real_ingest_available": True}),
        encoding="utf-8",
    )

    assert select_real_case_init_dates(outputs_dir, verification_dir, labels_dir=labels_dir, interim_dir=interim_dir) == [
        "2024-03-14",
        "2024-04-26",
        "2024-05-06",
    ]


def test_discover_real_tornado_relevant_candidate_dates_includes_spc_report_evidence(tmp_path: Path) -> None:
    verification_dir = tmp_path / "verification"
    labels_dir = tmp_path / "labels"
    interim_dir = tmp_path / "interim"
    verification_dir.mkdir()
    labels_dir.mkdir()
    interim_dir.mkdir()

    pd.DataFrame(
        [
            {"date": "2024-03-14", "tornado_outbreak": 1, "significant_tornado_support": 1},
            {"date": "2024-05-06", "tornado_outbreak": 0, "significant_tornado_support": 0},
        ]
    ).to_parquet(labels_dir / "outbreaks_2024-03-01_2025-06-30.parquet", index=False)
    pd.DataFrame(
        [
            {"date": "2024-05-06", "hazard": "tornado"},
            {"date": "2024-05-07", "hazard": "hail"},
        ]
    ).to_parquet(labels_dir / "spc_reports_2024-03-01_2025-06-30.parquet", index=False)
    (interim_dir / "ingest_summary_2024-03-14_00.json").write_text(
        json.dumps({"source": "local_staged_gfs", "source_mode": "real", "real_ingest_available": True}),
        encoding="utf-8",
    )
    (interim_dir / "ingest_summary_2024-05-06_00.json").write_text(
        json.dumps({"source": "local_staged_gfs", "source_mode": "real", "real_ingest_available": True}),
        encoding="utf-8",
    )

    assert discover_real_tornado_relevant_candidate_dates(verification_dir, labels_dir=labels_dir, interim_dir=interim_dir) == [
        "2024-03-14",
        "2024-05-06",
    ]


def test_select_real_case_init_dates_excludes_missing_forecast_and_non_tornado_windows(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    verification_dir = tmp_path / "verification"
    labels_dir = tmp_path / "labels"
    interim_dir = tmp_path / "interim"
    outputs_dir.mkdir()
    verification_dir.mkdir()
    labels_dir.mkdir()
    interim_dir.mkdir()

    xr.Dataset(coords={"time": pd.to_datetime(["2024-04-26T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(
        outputs_dir / "forecast_products_2024-04-26_00.nc"
    )

    (verification_dir / "2024-04-26_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
                "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                "per_day": [{"valid_date": "2024-04-29", "observed_category": "active_non_outbreak_severe_day"}],
            }
        ),
        encoding="utf-8",
    )
    (verification_dir / "2024-05-06_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-05-06", "evaluation_source": "local_staged_gfs"},
                "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                "per_day": [{"valid_date": "2024-05-06", "observed_category": "tornado_outbreak_day"}],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"date": "2024-04-26", "tornado_outbreak": 0, "significant_tornado_support": 0},
            {"date": "2024-05-06", "tornado_outbreak": 1, "significant_tornado_support": 1},
            {"date": "2024-05-07", "tornado_outbreak": 1, "significant_tornado_support": 0},
        ]
    ).to_parquet(labels_dir / "outbreaks_2024-03-01_2025-06-30.parquet", index=False)
    (interim_dir / "ingest_summary_2024-05-06_00.json").write_text(
        json.dumps({"source": "synthetic_fallback", "source_mode": "synthetic", "real_ingest_available": False}),
        encoding="utf-8",
    )
    (interim_dir / "ingest_summary_2024-05-07_00.json").write_text(
        json.dumps({"source": "local_staged_gfs", "source_mode": "real", "real_ingest_available": True}),
        encoding="utf-8",
    )

    assert select_real_case_init_dates(outputs_dir, verification_dir, labels_dir=labels_dir, interim_dir=interim_dir) == []


def test_audit_real_case_init_dates_reports_stage_counts_and_examples(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    verification_dir = tmp_path / "verification"
    labels_dir = tmp_path / "labels"
    interim_dir = tmp_path / "interim"
    outputs_dir.mkdir()
    verification_dir.mkdir()
    labels_dir.mkdir()
    interim_dir.mkdir()

    xr.Dataset(coords={"time": pd.to_datetime(["2024-03-14T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(
        outputs_dir / "forecast_products_2024-03-14_00.nc"
    )
    xr.Dataset(coords={"time": pd.to_datetime(["2024-04-26T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(
        outputs_dir / "forecast_products_2024-04-26_00.nc"
    )
    xr.Dataset(coords={"time": pd.to_datetime(["2024-05-06T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(
        outputs_dir / "forecast_products_2024-05-06_00.nc"
    )

    (verification_dir / "2024-03-14_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-03-14", "evaluation_source": "local_staged_gfs"},
                "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                "per_day": [{"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day"}],
            }
        ),
        encoding="utf-8",
    )
    (verification_dir / "2024-04-02_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-04-02", "evaluation_source": "synthetic_fallback"},
                "evaluation_metadata": {"ingest_summary": {"source": "synthetic_fallback"}},
                "per_day": [{"valid_date": "2024-04-02", "observed_category": "significant_tornado_outbreak_day"}],
            }
        ),
        encoding="utf-8",
    )
    (verification_dir / "2024-04-26_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
                "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                "per_day": [{"valid_date": "2024-04-26", "observed_category": "tornado_outbreak_day"}],
            }
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {"date": "2024-03-14", "tornado_outbreak": 1, "significant_tornado_support": 1},
            {"date": "2024-04-26", "tornado_outbreak": 1, "significant_tornado_support": 0},
            {"date": "2024-05-06", "tornado_outbreak": 1, "significant_tornado_support": 0},
            {"date": "2024-05-07", "tornado_outbreak": 1, "significant_tornado_support": 1},
        ]
    ).to_parquet(labels_dir / "outbreaks_2024-03-01_2025-06-30.parquet", index=False)

    (interim_dir / "ingest_summary_2024-05-06_00.json").write_text(
        json.dumps({"source": "local_staged_gfs", "source_mode": "real", "real_ingest_available": True}),
        encoding="utf-8",
    )
    (interim_dir / "ingest_summary_2024-05-07_00.json").write_text(
        json.dumps({"source": "local_staged_gfs", "source_mode": "real", "real_ingest_available": True}),
        encoding="utf-8",
    )

    audit = audit_real_case_init_dates(outputs_dir, verification_dir, labels_dir=labels_dir, interim_dir=interim_dir, sample_size=3)

    assert audit["candidate_dates_from_verification_jsons"] == 3
    assert audit["candidate_dates_from_outbreaks_parquet"] == 4
    assert audit["dates_with_local_forecast_artifact_present"] == 3
    assert audit["dates_with_real_ingest_confirmed"] == 4
    assert audit["dates_with_tornado_relevant_label_or_report_evidence"] == 5
    assert audit["final_selected_dates"] == 3
    assert audit["missing_forecast_examples"] == ["2024-04-02", "2024-05-07"]
    assert audit["not_real_ingest_examples"] == ["2024-04-02"]
    assert audit["final_selected_examples"] == ["2024-03-14", "2024-04-26", "2024-05-06"]


def test_missing_real_case_forecast_artifacts_reports_expected_pattern_and_command(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    verification_dir = tmp_path / "verification"
    labels_dir = tmp_path / "labels"
    interim_dir = tmp_path / "interim"
    outputs_dir.mkdir()
    verification_dir.mkdir()
    labels_dir.mkdir()
    interim_dir.mkdir()

    xr.Dataset(coords={"time": pd.to_datetime(["2024-03-14T00:00:00"]), "lat": [35.0], "lon": [-98.0]}).to_netcdf(
        outputs_dir / "forecast_products_2024-03-14_00.nc"
    )

    (verification_dir / "2024-03-14_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-03-14", "evaluation_source": "local_staged_gfs"},
                "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                "per_day": [{"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day"}],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"date": "2024-03-14", "tornado_outbreak": 1, "significant_tornado_support": 1},
            {"date": "2024-05-06", "tornado_outbreak": 1, "significant_tornado_support": 0},
            {"date": "2024-05-07", "tornado_outbreak": 1, "significant_tornado_support": 1},
        ]
    ).to_parquet(labels_dir / "outbreaks_2024-03-01_2025-06-30.parquet", index=False)
    (interim_dir / "ingest_summary_2024-05-06_00.json").write_text(
        json.dumps({"source": "local_staged_gfs", "source_mode": "real", "real_ingest_available": True}),
        encoding="utf-8",
    )
    (interim_dir / "ingest_summary_2024-05-07_00.json").write_text(
        json.dumps({"source": "synthetic_fallback", "source_mode": "synthetic", "real_ingest_available": False}),
        encoding="utf-8",
    )

    rows = missing_real_case_forecast_artifacts(outputs_dir, verification_dir, labels_dir=labels_dir, interim_dir=interim_dir)

    assert rows == [
        {
            "init_date": "2024-05-06",
            "expected_artifact_pattern": str(outputs_dir / "forecast_products_2024-05-06_*.nc"),
            "generation_command": "python -m severewx.cli.run_forecast --date 2024-05-06 --cycle 00",
        }
    ]


def test_summarize_real_case_backfill_separates_already_present_from_generated() -> None:
    summary = summarize_real_case_backfill(
        [
            {
                "init_date": "2024-03-14",
                "forecast_present_initial": True,
                "verification_present_initial": True,
                "fully_evaluable_initial": True,
                "forecast_generated": False,
                "verification_generated": False,
                "forecast_attempted": False,
                "verification_attempted": False,
                "fully_evaluable": True,
                "blocker": "",
            },
            {
                "init_date": "2024-05-26",
                "forecast_present_initial": False,
                "verification_present_initial": False,
                "fully_evaluable_initial": False,
                "forecast_generated": True,
                "verification_generated": True,
                "forecast_attempted": True,
                "verification_attempted": True,
                "fully_evaluable": True,
                "blocker": "",
            },
            {
                "init_date": "2024-05-27",
                "forecast_present_initial": False,
                "verification_present_initial": False,
                "fully_evaluable_initial": False,
                "forecast_generated": False,
                "verification_generated": False,
                "forecast_attempted": True,
                "verification_attempted": False,
                "forecast_present": False,
                "verification_present": False,
                "fully_evaluable": False,
                "blocker": "forecast_generation_failed:missing_input",
            },
            {
                "init_date": "2024-05-28",
                "forecast_present_initial": True,
                "verification_present_initial": False,
                "fully_evaluable_initial": False,
                "forecast_generated": False,
                "verification_generated": False,
                "forecast_attempted": False,
                "verification_attempted": True,
                "forecast_present": True,
                "verification_present": False,
                "fully_evaluable": False,
                "blocker": "verification_generation_failed:missing_labels",
            },
        ]
    )

    assert summary == {
        "candidate_dates_total": 4,
        "forecast_artifacts_already_present": 2,
        "forecast_artifacts_generated": 1,
        "forecast_artifacts_failed": 1,
        "verification_artifacts_already_present": 1,
        "verification_artifacts_generated": 1,
        "verification_artifacts_failed": 1,
        "fully_evaluable_dates_total": 2,
        "newly_added_ready_dates": 1,
        "failed_dates_total": 2,
        "failed_dates_examples": [
            "2024-05-27:forecast_generation_failed:missing_input",
            "2024-05-28:verification_generation_failed:missing_labels",
        ],
        "failed_dates_examples_with_reason": [
            "2024-05-27:forecast_generation_failed:missing_input",
            "2024-05-28:verification_generation_failed:missing_labels",
        ],
    }


def test_write_init_dates_file_writes_ready_only_lines(tmp_path: Path) -> None:
    path = tmp_path / "verification" / "tornado_concern_ready_dates.txt"
    write_init_dates_file(path, ["2024-03-14", "2024-04-26"])
    assert path.read_text(encoding="utf-8") == "2024-03-14\n2024-04-26\n"


def test_tornado_concern_eval_skip_missing_cli_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-04-26T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[2.4]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.19]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[3.2]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.002]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.04]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-04-26_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [
            {
                "valid_date": "2024-04-26",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
            }
        ],
    }
    (verification_dir / "2024-04-26_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["tornado_concern_eval", "--dates", "2024-04-26", "2024-04-27", "--skip-missing"])

    module.main()
    captured = capsys.readouterr()
    assert "warning: skipping 2024-04-27 because forecast or verification artifacts are missing" in captured.out
    assert "2024-04-26" in captured.out


def test_tornado_concern_eval_writes_ranked_csv_and_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-04-26T00:00:00", "2024-04-27T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[2.4]], [[1.2]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.19]], [[0.12]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[3.2]], [[1.4]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.002]], [[0.0005]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.04]], [[0.01]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.6]], [[0.2]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-04-26_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [
            {
                "valid_date": "2024-04-26",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
            },
            {
                "valid_date": "2024-04-27",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
            },
        ],
    }
    (verification_dir / "2024-04-26_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    csv_path = tmp_path / "reports" / "tornado_concern_eval.csv"
    md_path = tmp_path / "reports" / "tornado_concern_eval.md"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "tornado_concern_eval",
            "--dates",
            "2024-04-26",
            "--output-csv",
            str(csv_path),
            "--output-md",
            str(md_path),
        ],
    )

    module.main()

    ranked = pd.read_csv(csv_path)
    assert csv_path.exists()
    assert md_path.exists()
    assert {
        "init_date",
        "valid_date",
        "day_rank_within_init",
        "window_max_rank",
        "observed_category",
        "observed_tornado_outbreak",
        "observed_significant_tornado_support",
        "is_real_ingest",
        "source_variant_name",
        "component_variant_name",
        "tornado_concern_score",
        "raw_tornado_prob_max",
        "learned_tornado_concern_prob",
        "core_minus_context",
        "scp_x_core",
        "overlap_x_core",
        "support_x_overlap",
        "overlap_compactness_proxy",
        "overlap_purity_proxy",
        "core_tornado_alignment_proxy",
        "support_tornado_alignment_proxy",
        "broad_contamination_proxy",
        "scp_support_ratio",
        "overlap_to_context_ratio",
        "core_to_context_ratio",
    }.issubset(ranked.columns)
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Tornado Concern Evaluation" in markdown
    assert "## Window Summary" in markdown
    assert "## Failure Pattern Counts" in markdown
    assert "## Source Variant Summary" in markdown
    assert "## Component Variant Summary" in markdown
    assert "## Ranked Daily Rows" in markdown
    assert "## Flagged Real-Window Failures" not in markdown
    assert "## Conservative Tornado Preference Diagnostics" in markdown


def test_tornado_concern_penalizes_hail_like_high_support_low_overlap_regime() -> None:
    tornado_like = xr.Dataset(
        {
            "sig_tor_support": (("lat", "lon"), np.array([[2.2, 2.0], [1.8, 1.7]], dtype=float)),
            "outbreak_risk": (("lat", "lon"), np.array([[0.18, 0.17], [0.16, 0.15]], dtype=float)),
            "tornado_favored_overlap": (("lat", "lon"), np.array([[3.1, 2.9], [2.7, 2.6]], dtype=float)),
            "hail_favored_overlap": (("lat", "lon"), np.array([[1.8, 1.7], [1.6, 1.5]], dtype=float)),
            "wind_favored_overlap": (("lat", "lon"), np.array([[1.3, 1.2], [1.1, 1.0]], dtype=float)),
            "any_prob": (("lat", "lon"), np.array([[0.10, 0.08], [0.06, 0.05]], dtype=float)),
            "tornado_prob": (("lat", "lon"), np.array([[0.005, 0.004], [0.003, 0.003]], dtype=float)),
        }
    )
    hail_like = xr.Dataset(
        {
            "sig_tor_support": (("lat", "lon"), np.array([[2.2, 2.0], [1.8, 1.7]], dtype=float)),
            "outbreak_risk": (("lat", "lon"), np.array([[0.18, 0.17], [0.16, 0.15]], dtype=float)),
            "tornado_favored_overlap": (("lat", "lon"), np.array([[1.4, 1.3], [1.2, 1.1]], dtype=float)),
            "hail_favored_overlap": (("lat", "lon"), np.array([[3.0, 2.8], [2.6, 2.4]], dtype=float)),
            "wind_favored_overlap": (("lat", "lon"), np.array([[1.5, 1.4], [1.3, 1.2]], dtype=float)),
            "any_prob": (("lat", "lon"), np.array([[0.10, 0.08], [0.06, 0.05]], dtype=float)),
            "tornado_prob": (("lat", "lon"), np.array([[0.001, 0.001], [0.0008, 0.0008]], dtype=float)),
        }
    )

    tornado_score = _daily_tornado_concern_components(tornado_like)["tornado_concern_score"]
    hail_score = _daily_tornado_concern_components(hail_like)["tornado_concern_score"]

    assert tornado_score > 0.0
    assert hail_score > 0.0
    assert tornado_score > hail_score


def test_tornado_concern_scores_tornado_day_above_hail_day_in_2024_03_14_style_window() -> None:
    tornado_day = xr.Dataset(
        {
            "sig_tor_support": (("lat", "lon"), np.array([[1.7, 1.5], [1.3, 1.2]], dtype=float)),
            "outbreak_risk": (("lat", "lon"), np.array([[0.22, 0.20], [0.18, 0.17]], dtype=float)),
            "tornado_favored_overlap": (("lat", "lon"), np.array([[2.1, 1.9], [1.7, 1.6]], dtype=float)),
            "hail_favored_overlap": (("lat", "lon"), np.array([[2.3, 2.2], [2.0, 1.9]], dtype=float)),
            "wind_favored_overlap": (("lat", "lon"), np.array([[1.4, 1.3], [1.2, 1.1]], dtype=float)),
            "scp_proxy": (("lat", "lon"), np.array([[0.95, 0.88], [0.80, 0.74]], dtype=float)),
            "synoptic_support": (("lat", "lon"), np.array([[620.0, 600.0], [580.0, 560.0]], dtype=float)),
            "any_prob": (("lat", "lon"), np.array([[0.04, 0.03], [0.02, 0.02]], dtype=float)),
            "tornado_prob": (("lat", "lon"), np.array([[0.003, 0.0025], [0.002, 0.0018]], dtype=float)),
        }
    )
    hail_day = xr.Dataset(
        {
            "sig_tor_support": (("lat", "lon"), np.array([[1.8, 1.7], [1.5, 1.4]], dtype=float)),
            "outbreak_risk": (("lat", "lon"), np.array([[0.24, 0.23], [0.21, 0.20]], dtype=float)),
            "tornado_favored_overlap": (("lat", "lon"), np.array([[1.9, 1.8], [1.6, 1.5]], dtype=float)),
            "hail_favored_overlap": (("lat", "lon"), np.array([[3.1, 3.0], [2.8, 2.7]], dtype=float)),
            "wind_favored_overlap": (("lat", "lon"), np.array([[1.5, 1.4], [1.3, 1.2]], dtype=float)),
            "scp_proxy": (("lat", "lon"), np.array([[0.62, 0.58], [0.52, 0.48]], dtype=float)),
            "synoptic_support": (("lat", "lon"), np.array([[760.0, 740.0], [720.0, 700.0]], dtype=float)),
            "any_prob": (("lat", "lon"), np.array([[0.05, 0.04], [0.03, 0.03]], dtype=float)),
            "tornado_prob": (("lat", "lon"), np.array([[0.001, 0.001], [0.0008, 0.0008]], dtype=float)),
        }
    )

    tornado_score = _daily_tornado_concern_components(tornado_day)["tornado_concern_score"]
    hail_score = _daily_tornado_concern_components(hail_day)["tornado_concern_score"]

    assert tornado_score > 0.0
    assert hail_score > 0.0
    assert tornado_score > hail_score


def test_failure_summary_flags_hail_over_tornado_real_window() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-14",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "ingest_source": "local_staged_gfs",
                "is_real_ingest": True,
                "tornado_concern_score": 0.10,
                "learned_tornado_concern_prob": 0.3,
            },
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-15",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "ingest_source": "local_staged_gfs",
                "is_real_ingest": True,
                "tornado_concern_score": 0.20,
                "learned_tornado_concern_prob": 0.4,
            },
        ]
    )

    window_summary = summarize_case_windows(frame)
    failure_summary = summarize_failure_patterns(window_summary)

    assert bool(window_summary.loc[0, "hail_outranks_tornado_failure"])
    assert bool(window_summary.loc[0, "top_day_category_mismatch"])
    assert int(
        failure_summary.loc[failure_summary["pattern"] == "hail_outranks_tornado_failure", "count_real_windows"].iloc[0]
    ) == 1


def test_ranked_output_contains_richer_component_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-03-14T00:00:00", "2024-03-15T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[1.8]], [[1.9]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.22]], [[0.23]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[2.1]], [[1.9]]], dtype=float)),
            "hail_favored_overlap": (("time", "lat", "lon"), np.array([[[2.2]], [[3.0]]], dtype=float)),
            "wind_favored_overlap": (("time", "lat", "lon"), np.array([[[1.2]], [[1.3]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.9]], [[0.6]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[620.0]], [[760.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.003]], [[0.001]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.04]], [[0.05]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.2]], [[0.8]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-03-14_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-03-14", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [
            {
                "valid_date": "2024-03-14",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
            },
            {
                "valid_date": "2024-03-15",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
            },
        ],
    }
    (verification_dir / "2024-03-14_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    csv_path = tmp_path / "reports" / "ranked.csv"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["tornado_concern_eval", "--dates", "2024-03-14", "--output-csv", str(csv_path)])
    module.main()
    ranked = pd.read_csv(csv_path)
    assert {
        "raw_base_score_before_variant",
        "variant_name",
        "variant_base_score",
        "variant_tornado_term",
        "variant_broad_term",
        "variant_learned_term",
        "variant_penalty_term",
        "variant_final_score_before_preference",
        "tornado_signal",
        "broad_signal",
        "discriminator",
        "base_score",
        "discriminator_multiplier",
        "final_score_minus_window_top",
        "ranking_tornado_concern_score",
    }.issubset(ranked.columns)


def test_markdown_includes_flagged_real_window_failure_details(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-03-14T00:00:00", "2024-03-15T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[1.35]], [[1.90]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.18]], [[0.27]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[1.65]], [[1.85]]], dtype=float)),
            "hail_favored_overlap": (("time", "lat", "lon"), np.array([[[2.0]], [[3.25]]], dtype=float)),
            "wind_favored_overlap": (("time", "lat", "lon"), np.array([[[1.1]], [[1.4]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.45]], [[0.55]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[520.0]], [[820.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.002]], [[0.001]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.03]], [[0.06]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.2]], [[0.8]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-03-14_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-03-14", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [
            {
                "valid_date": "2024-03-14",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
            },
            {
                "valid_date": "2024-03-15",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
            },
        ],
    }
    (verification_dir / "2024-03-14_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    md_path = tmp_path / "reports" / "flagged.md"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["tornado_concern_eval", "--dates", "2024-03-14", "--output-md", str(md_path)])
    module.main()

    markdown = md_path.read_text(encoding="utf-8")
    assert "## Failure Pattern Counts" in markdown
    assert "## Score Variant Summary" in markdown
    assert "## Source Field Audit For Flagged Real Windows" in markdown
    assert "## Bad-Window Component Audit" in markdown
    assert "## Flagged Real-Window Failures" in markdown
    assert "### 2024-03-14" in markdown
    assert "top_ranked_day" in markdown
    assert "top_minus_best_tornado_core_top" in markdown
    assert "top_minus_best_tornado_tornado_signal" in markdown
    assert "top_minus_best_tornado_score" in markdown
    assert "failure_driver" in markdown


def test_conservative_mode_can_flip_close_hail_over_tornado_ordering() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-14",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "is_real_ingest": True,
                "tornado_concern_score": 0.090,
                "core_top": 0.020,
                "scp_top": 0.50,
                "tornado_signal": 0.010,
                "discriminator": 0.20,
            },
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-15",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "tornado_concern_score": 0.100,
                "core_top": 0.010,
                "scp_top": 0.40,
                "tornado_signal": 0.007,
                "discriminator": 0.15,
            },
        ]
    )

    adjusted, changes = apply_tornado_preference_mode(frame, mode="conservative")
    summary = summarize_case_windows(adjusted)

    assert not changes.empty
    assert summary.loc[0, "top_valid_date"] == "2024-03-14"


def test_conservative_mode_does_not_flip_clear_non_tornado_top_day() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-01",
                "valid_date": "2024-04-01",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "tornado_concern_score": 0.25,
                "core_top": 0.020,
                "scp_top": 0.35,
                "tornado_signal": 0.020,
                "discriminator": 0.20,
            },
            {
                "init_date": "2024-04-01",
                "valid_date": "2024-04-02",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "is_real_ingest": True,
                "tornado_concern_score": 0.10,
                "core_top": 0.021,
                "scp_top": 0.42,
                "tornado_signal": 0.021,
                "discriminator": 0.24,
            },
        ]
    )

    adjusted, changes = apply_tornado_preference_mode(frame, mode="conservative")
    summary = summarize_case_windows(adjusted)

    assert changes.empty
    assert summary.loc[0, "top_valid_date"] == "2024-04-01"


def test_conservative_mode_does_not_create_non_outbreak_over_outbreak_failure() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-26",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "is_real_ingest": True,
                "tornado_concern_score": 0.11,
                "core_top": 0.020,
                "scp_top": 0.48,
                "tornado_signal": 0.012,
                "discriminator": 0.22,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "tornado_concern_score": 0.10,
                "core_top": 0.010,
                "scp_top": 0.20,
                "tornado_signal": 0.005,
                "discriminator": 0.10,
            },
        ]
    )

    adjusted, _ = apply_tornado_preference_mode(frame, mode="conservative")
    summary = summarize_case_windows(adjusted)

    assert not bool(summary.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_compact_tornado_mode_can_flip_broad_top_to_compact_tornado_signal() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-20",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "tornado_concern_score": 0.200,
                "tornado_signal": 0.006,
                "discriminator": 0.100,
                "core_compactness_proxy": 0.20,
                "core_purity_proxy": 0.30,
                "core_tornado_alignment_proxy": 0.55,
                "overlap_purity_proxy": 0.50,
                "broad_contamination_proxy": 0.55,
            },
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-19",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "is_real_ingest": True,
                "tornado_concern_score": 0.160,
                "tornado_signal": 0.012,
                "discriminator": 0.200,
                "core_compactness_proxy": 0.42,
                "core_purity_proxy": 0.58,
                "core_tornado_alignment_proxy": 0.78,
                "overlap_purity_proxy": 0.50,
                "broad_contamination_proxy": 0.30,
            },
        ]
    )

    adjusted, changes = apply_tornado_preference_mode(frame, mode="compact_tornado")
    summary = summarize_case_windows(adjusted)

    assert not changes.empty
    assert summary.loc[0, "top_valid_date"] == "2024-05-19"


def test_compact_tornado_mode_does_not_flip_when_top_is_not_broad_or_weak() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-06-02",
                "valid_date": "2024-06-03",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "tornado_concern_score": 0.200,
                "tornado_signal": 0.006,
                "discriminator": 0.100,
                "core_compactness_proxy": 0.20,
                "core_purity_proxy": 0.30,
                "core_tornado_alignment_proxy": 0.78,
                "overlap_purity_proxy": 0.50,
                "broad_contamination_proxy": 0.20,
            },
            {
                "init_date": "2024-06-02",
                "valid_date": "2024-06-02",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "is_real_ingest": True,
                "tornado_concern_score": 0.160,
                "tornado_signal": 0.012,
                "discriminator": 0.200,
                "core_compactness_proxy": 0.42,
                "core_purity_proxy": 0.58,
                "core_tornado_alignment_proxy": 0.90,
                "overlap_purity_proxy": 0.50,
                "broad_contamination_proxy": 0.12,
            },
        ]
    )

    adjusted, changes = apply_tornado_preference_mode(frame, mode="compact_tornado")
    summary = summarize_case_windows(adjusted)

    assert changes.empty
    assert summary.loc[0, "top_valid_date"] == "2024-06-03"


def test_score_variant_cli_is_accepted_and_baseline_remains_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-04-26T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[2.0]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.18]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[2.4]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.85]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[640.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.003]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.04]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.55]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-04-26_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [{"valid_date": "2024-04-26", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1}],
    }
    (verification_dir / "2024-04-26_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    default_csv = tmp_path / "default.csv"
    emphasis_csv = tmp_path / "emphasis.csv"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("sys.argv", ["tornado_concern_eval", "--dates", "2024-04-26", "--output-csv", str(default_csv)])
    module.main()
    default_ranked = pd.read_csv(default_csv)

    monkeypatch.setattr(
        "sys.argv",
        ["tornado_concern_eval", "--dates", "2024-04-26", "--score-variant", "tornado_emphasis", "--output-csv", str(emphasis_csv)],
    )
    module.main()
    emphasis_ranked = pd.read_csv(emphasis_csv)

    assert SCORE_VARIANTS == ["baseline", "tornado_emphasis", "capped_broad", "learned_gated", "hybrid", "lead_time_calibrated"]
    assert set(default_ranked["variant_name"]) == {"baseline"}
    assert set(emphasis_ranked["variant_name"]) == {"tornado_emphasis"}


def test_tornado_emphasis_can_flip_close_structural_hail_over_tornado_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-14",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "raw_base_score_before_variant": 0.20,
                "base_score": 0.20,
                "tornado_signal": 0.0100,
                "broad_signal": 0.2800,
                "discriminator": 0.0345,
                "learned_tornado_concern_prob": 0.20,
            },
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-15",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "raw_base_score_before_variant": 0.21,
                "base_score": 0.21,
                "tornado_signal": 0.0050,
                "broad_signal": 0.3500,
                "discriminator": 0.0141,
                "learned_tornado_concern_prob": 0.20,
            },
        ]
    )

    baseline = summarize_case_windows(_apply_score_variant_to_frame(frame, "baseline"))
    emphasized = summarize_case_windows(_apply_score_variant_to_frame(frame, "tornado_emphasis"))

    assert baseline.loc[0, "top_valid_date"] == "2024-03-15"
    assert emphasized.loc[0, "top_valid_date"] == "2024-03-14"


def test_capped_broad_reduces_broad_context_dominance() -> None:
    frame = pd.DataFrame(
        [
            {"raw_base_score_before_variant": 0.20, "base_score": 0.20, "tornado_signal": 0.010, "broad_signal": 0.20, "discriminator": 0.0476, "learned_tornado_concern_prob": 0.10},
            {"raw_base_score_before_variant": 0.20, "base_score": 0.20, "tornado_signal": 0.010, "broad_signal": 0.60, "discriminator": 0.0164, "learned_tornado_concern_prob": 0.10},
        ]
    )

    baseline = _apply_score_variant_to_frame(frame, "baseline")
    capped = _apply_score_variant_to_frame(frame, "capped_broad")
    baseline_gap = float(baseline.loc[0, "tornado_concern_score"] - baseline.loc[1, "tornado_concern_score"])
    capped_gap = float(capped.loc[0, "tornado_concern_score"] - capped.loc[1, "tornado_concern_score"])

    assert capped_gap < baseline_gap


def test_learned_gated_reduces_learned_term_override_when_core_disagrees() -> None:
    frame = pd.DataFrame(
        [
            {
                "raw_base_score_before_variant": 0.22,
                "base_score": 0.22,
                "tornado_signal": 0.004,
                "broad_signal": 0.32,
                "discriminator": 0.012,
                "learned_tornado_concern_prob": 0.95,
            }
        ]
    )

    baseline = _apply_score_variant_to_frame(frame, "baseline")
    gated = _apply_score_variant_to_frame(frame, "learned_gated")

    assert float(gated.loc[0, "variant_penalty_term"]) < 1.0
    assert float(gated.loc[0, "tornado_concern_score"]) < float(baseline.loc[0, "tornado_concern_score"])


def test_hybrid_keeps_outbreak_above_non_outbreak_guardrail_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "raw_base_score_before_variant": 0.18,
                "base_score": 0.18,
                "tornado_signal": 0.011,
                "broad_signal": 0.24,
                "discriminator": 0.0438,
                "learned_tornado_concern_prob": 0.45,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "raw_base_score_before_variant": 0.16,
                "base_score": 0.16,
                "tornado_signal": 0.004,
                "broad_signal": 0.36,
                "discriminator": 0.0110,
                "learned_tornado_concern_prob": 0.92,
                "is_real_ingest": True,
            },
        ]
    )

    hybrid = summarize_case_windows(_apply_score_variant_to_frame(frame, "hybrid"))

    assert hybrid.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(hybrid.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_variant_summary_reports_all_variants() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-14",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "is_real_ingest": True,
                "raw_base_score_before_variant": 0.20,
                "base_score": 0.20,
                "tornado_signal": 0.0100,
                "broad_signal": 0.2800,
                "discriminator": 0.0345,
                "learned_tornado_concern_prob": 0.20,
            },
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-15",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "is_real_ingest": True,
                "raw_base_score_before_variant": 0.21,
                "base_score": 0.21,
                "tornado_signal": 0.0050,
                "broad_signal": 0.3500,
                "discriminator": 0.0141,
                "learned_tornado_concern_prob": 0.20,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "is_real_ingest": True,
                "raw_base_score_before_variant": 0.18,
                "base_score": 0.18,
                "tornado_signal": 0.0110,
                "broad_signal": 0.2400,
                "discriminator": 0.0438,
                "learned_tornado_concern_prob": 0.45,
            },
        ]
    )

    summary = summarize_score_variants(frame)

    assert set(summary["variant"]) == set(SCORE_VARIANTS)
    assert {"top_day_for_2024_03_14_window", "top_day_for_2024_04_26_window"}.issubset(summary.columns)


def test_component_variant_cli_is_accepted_and_baseline_remains_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-04-26T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[2.0]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.18]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[2.4]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.85]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[640.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.003]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.04]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.55]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-04-26_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [{"valid_date": "2024-04-26", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1}],
    }
    (verification_dir / "2024-04-26_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    default_csv = tmp_path / "default_component.csv"
    hybrid_csv = tmp_path / "hybrid_component.csv"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("sys.argv", ["tornado_concern_eval", "--dates", "2024-04-26", "--output-csv", str(default_csv)])
    module.main()
    default_ranked = pd.read_csv(default_csv)

    monkeypatch.setattr(
        "sys.argv",
        ["tornado_concern_eval", "--dates", "2024-04-26", "--component-variant", "hybrid", "--output-csv", str(hybrid_csv)],
    )
    module.main()
    hybrid_ranked = pd.read_csv(hybrid_csv)

    assert COMPONENT_VARIANTS == [
        "baseline",
        "core_tornado_gated",
        "overlap_emphasis",
        "scp_core_hybrid",
        "hybrid",
        "tornado_hail_separation_v1",
        "tornado_signal_continuity_v1",
    ]
    assert set(default_ranked["component_variant_name"]) == {"baseline"}
    assert set(hybrid_ranked["component_variant_name"]) == {"hybrid"}


def test_component_audit_fields_appear_in_ranked_output() -> None:
    frame = pd.DataFrame(
        [
            {
                "support_top": 0.15,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.80,
                "joint_area": 0.05,
                "core_top": 0.020,
                "scp_top": 0.42,
                "penalty_top": 0.10,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.1,
            }
        ]
    )

    adjusted = _apply_component_variant_to_frame(frame, "baseline")

    assert {
        "component_variant_name",
        "component_core_term",
        "component_overlap_term",
        "component_scp_term",
        "component_penalty_term",
        "core_minus_context",
        "scp_x_core",
        "overlap_x_core",
        "support_x_overlap",
    }.issubset(adjusted.columns)


def test_component_checkpoint_reports_baseline_vs_hail_separation_side_by_side(tmp_path: Path) -> None:
    times = pd.to_datetime(["2024-05-21T00:00:00", "2024-05-22T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[2.1]], [[1.1]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.24]], [[0.30]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[3.1]], [[1.35]]], dtype=float)),
            "hail_favored_overlap": (("time", "lat", "lon"), np.array([[[1.2]], [[3.2]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.55]], [[0.20]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.005]], [[0.003]]], dtype=float)),
            "hail_prob": (("time", "lat", "lon"), np.array([[[0.02]], [[0.12]]], dtype=float)),
            "wind_prob": (("time", "lat", "lon"), np.array([[[0.01]], [[0.03]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.05]], [[0.16]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.30]], [[0.30]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction_path = tmp_path / "forecast_products_2024-05-19_00.nc"
    prediction.to_netcdf(prediction_path)
    verification_path = tmp_path / "2024-05-19_verification.json"
    verification_path.write_text(
        json.dumps(
            {
                "run_summary": {"init_date": "2024-05-19", "evaluation_source": "local_staged_gfs"},
                "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                "per_day": [
                    {"valid_date": "2024-05-21", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1},
                    {"valid_date": "2024-05-22", "observed_category": "hail_outbreak_day", "observed_tornado_outbreak": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_component_checkpoint_from_artifacts(
        [("2024-05-19", prediction_path, verification_path)],
        requested_case_count=2,
        skipped_missing_count=1,
    )

    assert list(summary["component_variant"]) == CHECKPOINT_COMPONENT_VARIANTS
    assert set(summary["ready_case_count"]) == {1}
    assert set(summary["skipped_missing_count"]) == {1}
    assert {
        "hail_outranks_tornado_failure_real",
        "non_outbreak_outranks_outbreak_failure_real",
        "top_day_category_mismatch_real",
        "tornado_day_ranked_first_real",
        "sig_tor_day_ranked_first_real",
        "mean_top_minus_best_tornado_margin_real",
        "median_top_minus_best_tornado_margin_real",
    }.issubset(summary.columns)


def test_core_tornado_gated_can_reduce_generic_core_hail_over_tornado_miss() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-14",
                "observed_category": "significant_tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.12,
                "normalized_synoptic_support": 0.75,
                "context_top": 0.95,
                "joint_area": 0.015,
                "core_top": 0.020,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.20,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.25,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-03-14",
                "valid_date": "2024-03-15",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.22,
                "normalized_synoptic_support": 0.92,
                "context_top": 1.00,
                "joint_area": 0.018,
                "core_top": 0.040,
                "scp_top": 0.20,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 1.55,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    baseline = _apply_score_variant_to_frame(_apply_component_variant_to_frame(frame, "baseline"), "baseline")
    gated = _apply_score_variant_to_frame(_apply_component_variant_to_frame(frame, "core_tornado_gated"), "baseline")

    tornado_baseline = baseline.loc[baseline["valid_date"] == "2024-03-14"].iloc[0]
    hail_baseline = baseline.loc[baseline["valid_date"] == "2024-03-15"].iloc[0]
    tornado_gated = gated.loc[gated["valid_date"] == "2024-03-14"].iloc[0]
    hail_gated = gated.loc[gated["valid_date"] == "2024-03-15"].iloc[0]

    assert float(tornado_gated["component_core_term"]) > float(hail_gated["component_core_term"])
    assert float(hail_gated["tornado_concern_score"] / hail_baseline["tornado_concern_score"]) < float(
        tornado_gated["tornado_concern_score"] / tornado_baseline["tornado_concern_score"]
    )


def test_tornado_hail_separation_v1_penalizes_hail_dominant_weak_tornado_row() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-05-22",
                "support_top": 0.20,
                "normalized_synoptic_support": 0.90,
                "context_top": 0.95,
                "joint_area": 0.020,
                "core_top": 0.030,
                "scp_top": 0.20,
                "penalty_top": 0.18,
                "sig_tor_support_max": 1.10,
                "outbreak_risk_max": 0.30,
                "tornado_overlap_max": 1.35,
                "hail_overlap_max": 3.20,
            },
            {
                "valid_date": "2024-05-21",
                "support_top": 0.18,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.75,
                "joint_area": 0.016,
                "core_top": 0.026,
                "scp_top": 0.55,
                "penalty_top": 0.02,
                "sig_tor_support_max": 2.10,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 3.10,
                "hail_overlap_max": 1.20,
            },
        ]
    )

    baseline = _apply_score_variant_to_frame(_apply_component_variant_to_frame(frame, "baseline"), "baseline")
    separated = _apply_score_variant_to_frame(_apply_component_variant_to_frame(frame, "tornado_hail_separation_v1"), "baseline")
    hail_baseline = baseline.loc[baseline["valid_date"] == "2024-05-22"].iloc[0]
    hail_separated = separated.loc[separated["valid_date"] == "2024-05-22"].iloc[0]
    tornado_baseline = baseline.loc[baseline["valid_date"] == "2024-05-21"].iloc[0]
    tornado_separated = separated.loc[separated["valid_date"] == "2024-05-21"].iloc[0]

    assert float(hail_separated["component_hail_dominance"]) > 0.50
    assert float(hail_separated["component_penalty_term"]) < 0.75
    assert float(hail_separated["tornado_concern_score"]) < float(hail_baseline["tornado_concern_score"])
    assert float(tornado_separated["component_tornado_agreement"]) > float(hail_separated["component_tornado_agreement"])
    assert float(tornado_separated["tornado_concern_score"]) >= 0.95 * float(tornado_baseline["tornado_concern_score"])


def test_tornado_signal_continuity_v1_is_optional_and_preserves_baseline_default() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-04-26",
                "support_top": 0.12,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.72,
                "joint_area": 0.010,
                "core_top": 0.012,
                "scp_top": 0.54,
                "penalty_top": 0.03,
                "sig_tor_support_max": 2.00,
                "outbreak_risk_max": 0.26,
                "tornado_overlap_max": 3.00,
            }
        ]
    )

    baseline = _apply_component_variant_to_frame(frame, "baseline")
    continuity = _apply_component_variant_to_frame(frame, "tornado_signal_continuity_v1")

    assert set(baseline["component_variant_name"]) == {"baseline"}
    assert set(continuity["component_variant_name"]) == {"tornado_signal_continuity_v1"}
    assert float(continuity.loc[0, "component_core_term"]) > 1.0
    assert float(continuity.loc[0, "base_score"]) > float(baseline.loc[0, "base_score"])


def test_tornado_signal_continuity_v1_does_not_boost_broad_hail_wind_context() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-05-22",
                "support_top": 0.24,
                "normalized_synoptic_support": 0.96,
                "context_top": 1.00,
                "joint_area": 0.025,
                "core_top": 0.030,
                "scp_top": 0.10,
                "penalty_top": 0.16,
                "sig_tor_support_max": 0.70,
                "outbreak_risk_max": 0.20,
                "tornado_overlap_max": 0.90,
                "hail_overlap_max": 3.40,
                "wind_overlap_max": 2.80,
            }
        ]
    )

    baseline = _apply_component_variant_to_frame(frame, "baseline")
    continuity = _apply_component_variant_to_frame(frame, "tornado_signal_continuity_v1")

    assert float(continuity.loc[0, "component_core_term"]) == 1.0
    assert float(continuity.loc[0, "component_overlap_term"]) == 1.0
    assert float(continuity.loc[0, "component_scp_term"]) == 1.0
    assert float(continuity.loc[0, "base_score"]) <= float(baseline.loc[0, "base_score"])


def test_tornado_signal_continuity_v1_does_not_weaken_hail_separation_behavior() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-05-22",
                "support_top": 0.20,
                "normalized_synoptic_support": 0.90,
                "context_top": 0.95,
                "joint_area": 0.020,
                "core_top": 0.030,
                "scp_top": 0.20,
                "penalty_top": 0.18,
                "sig_tor_support_max": 1.10,
                "outbreak_risk_max": 0.30,
                "tornado_overlap_max": 1.35,
                "hail_overlap_max": 3.20,
            }
        ]
    )

    baseline = _apply_component_variant_to_frame(frame, "baseline")
    separated = _apply_component_variant_to_frame(frame, "tornado_hail_separation_v1")

    assert float(separated.loc[0, "component_penalty_term"]) < 1.0
    assert float(separated.loc[0, "base_score"]) < float(baseline.loc[0, "base_score"])


def test_overlap_emphasis_can_improve_overlap_driven_tornado_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-26",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.14,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.80,
                "joint_area": 0.030,
                "core_top": 0.018,
                "scp_top": 0.35,
                "penalty_top": 0.06,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.18,
                "tornado_overlap_max": 2.6,
                "learned_tornado_concern_prob": 0.20,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-27",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.16,
                "normalized_synoptic_support": 0.74,
                "context_top": 0.86,
                "joint_area": 0.028,
                "core_top": 0.024,
                "scp_top": 0.34,
                "penalty_top": 0.06,
                "sig_tor_support_max": 1.4,
                "outbreak_risk_max": 0.20,
                "tornado_overlap_max": 1.4,
                "learned_tornado_concern_prob": 0.20,
                "is_real_ingest": True,
            },
        ]
    )

    baseline = _apply_score_variant_to_frame(_apply_component_variant_to_frame(frame, "baseline"), "baseline")
    overlap = _apply_score_variant_to_frame(_apply_component_variant_to_frame(frame, "overlap_emphasis"), "baseline")

    tornado_baseline = baseline.loc[baseline["valid_date"] == "2024-04-26"].iloc[0]
    hail_baseline = baseline.loc[baseline["valid_date"] == "2024-04-27"].iloc[0]
    tornado_overlap = overlap.loc[overlap["valid_date"] == "2024-04-26"].iloc[0]
    hail_overlap = overlap.loc[overlap["valid_date"] == "2024-04-27"].iloc[0]

    assert float(tornado_overlap["component_overlap_term"]) > float(hail_overlap["component_overlap_term"])
    assert float(tornado_overlap["tornado_concern_score"] - tornado_baseline["tornado_concern_score"]) > float(
        hail_overlap["tornado_concern_score"] - hail_baseline["tornado_concern_score"]
    )


def test_scp_core_hybrid_can_improve_weak_separation_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-05-01",
                "valid_date": "2024-05-01",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.11,
                "normalized_synoptic_support": 0.72,
                "context_top": 0.76,
                "joint_area": 0.020,
                "core_top": 0.016,
                "scp_top": 0.52,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.7,
                "outbreak_risk_max": 0.18,
                "tornado_overlap_max": 1.8,
                "learned_tornado_concern_prob": 0.18,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-05-01",
                "valid_date": "2024-05-02",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.14,
                "normalized_synoptic_support": 0.76,
                "context_top": 0.88,
                "joint_area": 0.025,
                "core_top": 0.026,
                "scp_top": 0.26,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.3,
                "outbreak_risk_max": 0.21,
                "tornado_overlap_max": 1.6,
                "learned_tornado_concern_prob": 0.18,
                "is_real_ingest": True,
            },
        ]
    )

    baseline = _apply_score_variant_to_frame(_apply_component_variant_to_frame(frame, "baseline"), "baseline")
    hybrid = _apply_score_variant_to_frame(_apply_component_variant_to_frame(frame, "scp_core_hybrid"), "baseline")

    tornado_baseline = baseline.loc[baseline["valid_date"] == "2024-05-01"].iloc[0]
    hail_baseline = baseline.loc[baseline["valid_date"] == "2024-05-02"].iloc[0]
    tornado_hybrid = hybrid.loc[hybrid["valid_date"] == "2024-05-01"].iloc[0]
    hail_hybrid = hybrid.loc[hybrid["valid_date"] == "2024-05-02"].iloc[0]

    assert float(tornado_hybrid["component_scp_term"]) > float(hail_hybrid["component_scp_term"])
    assert float(tornado_hybrid["tornado_concern_score"] - tornado_baseline["tornado_concern_score"]) > float(
        hail_hybrid["tornado_concern_score"] - hail_baseline["tornado_concern_score"]
    )


def test_component_hybrid_keeps_outbreak_above_non_outbreak_guardrail_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.14,
                "normalized_synoptic_support": 0.68,
                "context_top": 0.78,
                "joint_area": 0.040,
                "core_top": 0.022,
                "scp_top": 0.42,
                "penalty_top": 0.05,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.5,
                "learned_tornado_concern_prob": 0.25,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.08,
                "normalized_synoptic_support": 0.88,
                "context_top": 0.95,
                "joint_area": 0.018,
                "core_top": 0.021,
                "scp_top": 0.18,
                "penalty_top": 0.05,
                "sig_tor_support_max": 1.1,
                "outbreak_risk_max": 0.23,
                "tornado_overlap_max": 1.4,
                "learned_tornado_concern_prob": 0.25,
                "is_real_ingest": True,
            },
        ]
    )

    hybrid = summarize_case_windows(_apply_score_variant_to_frame(_apply_component_variant_to_frame(frame, "hybrid"), "baseline"))

    assert hybrid.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(hybrid.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_source_variant_cli_is_accepted_and_baseline_remains_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-04-26T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[2.0]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.18]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[2.4]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.85]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[640.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.003]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.04]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.55]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-04-26_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [{"valid_date": "2024-04-26", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1}],
    }
    (verification_dir / "2024-04-26_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    default_csv = tmp_path / "default_source.csv"
    hybrid_csv = tmp_path / "hybrid_source.csv"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("sys.argv", ["tornado_concern_eval", "--dates", "2024-04-26", "--output-csv", str(default_csv)])
    module.main()
    default_ranked = pd.read_csv(default_csv)

    monkeypatch.setattr(
        "sys.argv",
        ["tornado_concern_eval", "--dates", "2024-04-26", "--source-variant", "hybrid", "--output-csv", str(hybrid_csv)],
    )
    module.main()
    hybrid_ranked = pd.read_csv(hybrid_csv)

    assert SOURCE_VARIANTS == [
        "baseline",
        "compact_overlap",
        "tornado_purity_gate",
        "contamination_penalty",
        "hybrid",
        "failure_targeted_contamination",
    ]
    assert set(default_ranked["source_variant_name"]) == {"baseline"}
    assert set(hybrid_ranked["source_variant_name"]) == {"hybrid"}


def test_source_audit_fields_appear_in_ranked_output() -> None:
    frame = pd.DataFrame(
        [
            {
                "support_top": 0.15,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.80,
                "joint_area": 0.001,
                "core_top": 0.020,
                "scp_top": 0.42,
                "penalty_top": 0.10,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.1,
            }
        ]
    )

    adjusted = _apply_source_variant_to_frame(frame, "baseline")

    assert {
        "source_variant_name",
        "overlap_compactness_proxy",
        "overlap_purity_proxy",
        "core_tornado_alignment_proxy",
        "support_tornado_alignment_proxy",
        "broad_contamination_proxy",
        "scp_support_ratio",
        "overlap_to_context_ratio",
        "core_to_context_ratio",
    }.issubset(adjusted.columns)


def test_compact_overlap_can_reduce_broad_overlap_hail_over_tornado_miss() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-03-14",
                "support_top": 0.12,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0003,
                "core_top": 0.018,
                "scp_top": 0.44,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.20,
                "tornado_overlap_max": 1.9,
            },
            {
                "valid_date": "2024-03-15",
                "support_top": 0.18,
                "normalized_synoptic_support": 0.86,
                "context_top": 0.98,
                "joint_area": 0.0032,
                "core_top": 0.026,
                "scp_top": 0.24,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.5,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 2.4,
            },
        ]
    )

    baseline = _apply_source_variant_to_frame(frame, "baseline")
    compact = _apply_source_variant_to_frame(frame, "compact_overlap")
    hail_base = baseline.loc[baseline["valid_date"] == "2024-03-15"].iloc[0]
    tornado_base = baseline.loc[baseline["valid_date"] == "2024-03-14"].iloc[0]
    hail_compact = compact.loc[compact["valid_date"] == "2024-03-15"].iloc[0]
    tornado_compact = compact.loc[compact["valid_date"] == "2024-03-14"].iloc[0]

    assert float(tornado_compact["overlap_compactness_proxy"]) > float(hail_compact["overlap_compactness_proxy"])
    assert float(hail_compact["source_overlap_term"]) < float(tornado_compact["source_overlap_term"])
    assert float(hail_compact["raw_base_score_before_variant"] / hail_base["raw_base_score_before_variant"]) < float(
        tornado_compact["raw_base_score_before_variant"] / tornado_base["raw_base_score_before_variant"]
    )


def test_tornado_purity_gate_can_improve_low_purity_generic_core_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-03-14",
                "support_top": 0.11,
                "normalized_synoptic_support": 0.72,
                "context_top": 0.86,
                "joint_area": 0.0010,
                "core_top": 0.018,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.20,
                "tornado_overlap_max": 2.0,
            },
            {
                "valid_date": "2024-03-15",
                "support_top": 0.18,
                "normalized_synoptic_support": 0.90,
                "context_top": 1.00,
                "joint_area": 0.0015,
                "core_top": 0.030,
                "scp_top": 0.18,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.3,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 2.1,
            },
        ]
    )

    baseline = _apply_source_variant_to_frame(frame, "baseline")
    purity = _apply_source_variant_to_frame(frame, "tornado_purity_gate")
    hail_base = baseline.loc[baseline["valid_date"] == "2024-03-15"].iloc[0]
    tornado_base = baseline.loc[baseline["valid_date"] == "2024-03-14"].iloc[0]
    hail_purity = purity.loc[purity["valid_date"] == "2024-03-15"].iloc[0]
    tornado_purity = purity.loc[purity["valid_date"] == "2024-03-14"].iloc[0]

    assert float(tornado_purity["core_tornado_alignment_proxy"]) > float(hail_purity["core_tornado_alignment_proxy"])
    assert float(hail_purity["source_core_term"]) < float(tornado_purity["source_core_term"])
    assert float(hail_purity["raw_base_score_before_variant"] / hail_base["raw_base_score_before_variant"]) < float(
        tornado_purity["raw_base_score_before_variant"] / tornado_base["raw_base_score_before_variant"]
    )


def test_contamination_penalty_can_reduce_contamination_driven_false_top_day() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-04-26",
                "support_top": 0.12,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0011,
                "core_top": 0.018,
                "scp_top": 0.46,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.20,
                "tornado_overlap_max": 2.0,
            },
            {
                "valid_date": "2024-04-29",
                "support_top": 0.22,
                "normalized_synoptic_support": 0.98,
                "context_top": 1.00,
                "joint_area": 0.0028,
                "core_top": 0.026,
                "scp_top": 0.20,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.3,
                "outbreak_risk_max": 0.26,
                "tornado_overlap_max": 2.2,
            },
        ]
    )

    baseline = _apply_source_variant_to_frame(frame, "baseline")
    contamination = _apply_source_variant_to_frame(frame, "contamination_penalty")
    false_top_base = baseline.loc[baseline["valid_date"] == "2024-04-29"].iloc[0]
    false_top_penalty = contamination.loc[contamination["valid_date"] == "2024-04-29"].iloc[0]

    assert float(false_top_penalty["broad_contamination_proxy"]) > 0.0
    assert float(false_top_penalty["source_penalty_term"]) < 1.0
    assert float(false_top_penalty["raw_base_score_before_variant"]) < float(false_top_base["raw_base_score_before_variant"])


def test_source_hybrid_does_not_let_non_outbreak_severe_day_outrank_outbreak_day() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.13,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0011,
                "core_top": 0.019,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.22,
                "normalized_synoptic_support": 0.98,
                "context_top": 1.00,
                "joint_area": 0.0028,
                "core_top": 0.025,
                "scp_top": 0.20,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.3,
                "outbreak_risk_max": 0.25,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    hybrid = summarize_case_windows(_apply_score_variant_to_frame(_apply_component_variant_to_frame(_apply_source_variant_to_frame(frame, "hybrid"), "baseline"), "baseline"))

    assert hybrid.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(hybrid.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_source_variant_summary_reports_all_variants(tmp_path: Path) -> None:
    times = pd.to_datetime(["2024-03-14T00:00:00", "2024-03-15T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[1.35]], [[1.90]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.18]], [[0.27]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[1.65]], [[1.85]]], dtype=float)),
            "hail_favored_overlap": (("time", "lat", "lon"), np.array([[[2.0]], [[3.25]]], dtype=float)),
            "wind_favored_overlap": (("time", "lat", "lon"), np.array([[[1.1]], [[1.4]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.45]], [[0.55]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[520.0]], [[820.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.002]], [[0.001]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.03]], [[0.06]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.2]], [[0.8]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction_path = tmp_path / "forecast_products_2024-03-14_00.nc"
    prediction.to_netcdf(prediction_path)
    verification_payload = {
        "run_summary": {"init_date": "2024-03-14", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [
            {"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 1},
            {"valid_date": "2024-03-15", "observed_category": "hail_outbreak_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
        ],
    }
    verification_path = tmp_path / "2024-03-14_verification.json"
    verification_path.write_text(json.dumps(verification_payload), encoding="utf-8")

    summary = summarize_source_variants_from_artifacts([("2024-03-14", prediction_path, verification_path)], component_variant="baseline", score_variant="baseline")

    assert set(summary["source_variant"]) == set(SOURCE_VARIANTS)
    assert {"source_root_cause_for_2024_03_14", "top_day_for_2024_03_14_window"}.issubset(summary.columns)


def test_core_variant_cli_is_accepted_and_baseline_remains_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-04-26T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[2.0]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.18]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[2.4]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.85]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[640.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.003]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.04]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.55]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-04-26_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [{"valid_date": "2024-04-26", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1}],
    }
    (verification_dir / "2024-04-26_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    default_csv = tmp_path / "default_core.csv"
    hybrid_csv = tmp_path / "hybrid_core.csv"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("sys.argv", ["tornado_concern_eval", "--dates", "2024-04-26", "--output-csv", str(default_csv)])
    module.main()
    default_ranked = pd.read_csv(default_csv)

    monkeypatch.setattr(
        "sys.argv",
        ["tornado_concern_eval", "--dates", "2024-04-26", "--core-variant", "hybrid", "--output-csv", str(hybrid_csv)],
    )
    module.main()
    hybrid_ranked = pd.read_csv(hybrid_csv)

    assert CORE_VARIANTS == ["baseline", "compact_core", "percentile_core", "aligned_core", "hybrid"]
    assert set(default_ranked["core_variant_name"]) == {"baseline"}
    assert set(hybrid_ranked["core_variant_name"]) == {"hybrid"}


def test_core_audit_fields_appear_in_ranked_output() -> None:
    frame = pd.DataFrame(
        [
            {
                "support_top": 0.15,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.80,
                "joint_area": 0.0010,
                "core_top": 0.020,
                "scp_top": 0.42,
                "penalty_top": 0.10,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.1,
            }
        ]
    )

    adjusted = _apply_core_variant_to_frame(frame, "baseline")

    assert {
        "core_variant_name",
        "effective_core_top",
        "effective_core_to_context_ratio",
        "core_compactness_proxy",
        "core_percentile_proxy",
        "core_alignment_proxy",
        "core_diffuseness_proxy",
        "core_purity_proxy",
        "core_support_interaction",
        "core_overlap_interaction",
        "core_scp_interaction",
        "core_penalty_term",
    }.issubset(adjusted.columns)


def test_compact_core_can_reduce_diffuse_core_hail_over_tornado_miss() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-03-14",
                "support_top": 0.12,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.78,
                "joint_area": 0.0003,
                "core_top": 0.016,
                "scp_top": 0.46,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.20,
                "tornado_overlap_max": 2.1,
            },
            {
                "valid_date": "2024-03-15",
                "support_top": 0.17,
                "normalized_synoptic_support": 0.88,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.026,
                "scp_top": 0.22,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.3,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 2.0,
            },
        ]
    )

    baseline = _apply_core_variant_to_frame(frame, "baseline")
    compact = _apply_core_variant_to_frame(frame, "compact_core")
    hail_base = baseline.loc[baseline["valid_date"] == "2024-03-15"].iloc[0]
    tornado_base = baseline.loc[baseline["valid_date"] == "2024-03-14"].iloc[0]
    hail_compact = compact.loc[compact["valid_date"] == "2024-03-15"].iloc[0]
    tornado_compact = compact.loc[compact["valid_date"] == "2024-03-14"].iloc[0]

    assert float(tornado_compact["core_compactness_proxy"]) > float(hail_compact["core_compactness_proxy"])
    assert float(tornado_compact["effective_core_top"] / tornado_base["core_top"]) > float(hail_compact["effective_core_top"] / hail_base["core_top"])


def test_percentile_core_can_reduce_max_spike_false_top_day() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-05-01",
                "support_top": 0.13,
                "normalized_synoptic_support": 0.74,
                "context_top": 0.82,
                "joint_area": 0.0008,
                "core_top": 0.019,
                "scp_top": 0.44,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.21,
                "tornado_overlap_max": 2.0,
            },
            {
                "valid_date": "2024-05-02",
                "support_top": 0.08,
                "normalized_synoptic_support": 0.92,
                "context_top": 1.00,
                "joint_area": 0.0030,
                "core_top": 0.030,
                "scp_top": 0.18,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 1.5,
            },
        ]
    )

    baseline = _apply_core_variant_to_frame(frame, "baseline")
    percentile = _apply_core_variant_to_frame(frame, "percentile_core")
    spiky_base = baseline.loc[baseline["valid_date"] == "2024-05-02"].iloc[0]
    spiky_percentile = percentile.loc[percentile["valid_date"] == "2024-05-02"].iloc[0]

    assert float(spiky_percentile["core_percentile_proxy"]) < 1.0
    assert float(spiky_percentile["effective_core_top"]) < float(spiky_base["core_top"])


def test_aligned_core_can_improve_tornado_support_aligned_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-04-26",
                "support_top": 0.10,
                "normalized_synoptic_support": 0.72,
                "context_top": 0.78,
                "joint_area": 0.0009,
                "core_top": 0.014,
                "scp_top": 0.50,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.18,
                "tornado_overlap_max": 2.0,
            },
            {
                "valid_date": "2024-04-27",
                "support_top": 0.14,
                "normalized_synoptic_support": 0.86,
                "context_top": 0.92,
                "joint_area": 0.0022,
                "core_top": 0.019,
                "scp_top": 0.20,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 1.6,
            },
        ]
    )

    baseline = _apply_core_variant_to_frame(frame, "baseline")
    aligned = _apply_core_variant_to_frame(frame, "aligned_core")
    tornado_base = baseline.loc[baseline["valid_date"] == "2024-04-26"].iloc[0]
    tornado_aligned = aligned.loc[aligned["valid_date"] == "2024-04-26"].iloc[0]
    hail_aligned = aligned.loc[aligned["valid_date"] == "2024-04-27"].iloc[0]

    assert float(tornado_aligned["core_alignment_proxy"]) > float(hail_aligned["core_alignment_proxy"])
    assert float(tornado_aligned["effective_core_top"] - tornado_base["core_top"]) > 0.0


def test_core_hybrid_does_not_let_non_outbreak_severe_day_outrank_outbreak_day() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.13,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0009,
                "core_top": 0.018,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.18,
                "normalized_synoptic_support": 0.98,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.024,
                "scp_top": 0.18,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.25,
                "tornado_overlap_max": 1.6,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    hybrid = summarize_case_windows(
        _apply_score_variant_to_frame(
            _apply_component_variant_to_frame(_apply_source_variant_to_frame(_apply_core_variant_to_frame(frame, "hybrid"), "baseline"), "baseline"),
            "baseline",
        )
    )

    assert hybrid.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(hybrid.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_markdown_contains_core_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-03-14T00:00:00", "2024-03-15T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[1.35]], [[1.90]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.18]], [[0.27]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[1.65]], [[1.85]]], dtype=float)),
            "hail_favored_overlap": (("time", "lat", "lon"), np.array([[[2.0]], [[3.25]]], dtype=float)),
            "wind_favored_overlap": (("time", "lat", "lon"), np.array([[[1.1]], [[1.4]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.45]], [[0.55]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[520.0]], [[820.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.002]], [[0.001]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.03]], [[0.06]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.2]], [[0.8]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-03-14_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-03-14", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [
            {"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 1},
            {"valid_date": "2024-03-15", "observed_category": "hail_outbreak_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
        ],
    }
    (verification_dir / "2024-03-14_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    md_path = tmp_path / "reports" / "core_sections.md"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["tornado_concern_eval", "--dates", "2024-03-14", "--output-md", str(md_path)])
    module.main()

    markdown = md_path.read_text(encoding="utf-8")
    assert "## Core Variant Summary" in markdown
    assert "## Core Construction Audit For Flagged Real Windows" in markdown


def test_core_variant_summary_reports_all_variants(tmp_path: Path) -> None:
    times = pd.to_datetime(["2024-03-14T00:00:00", "2024-03-15T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[1.35]], [[1.90]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.18]], [[0.27]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[1.65]], [[1.85]]], dtype=float)),
            "hail_favored_overlap": (("time", "lat", "lon"), np.array([[[2.0]], [[3.25]]], dtype=float)),
            "wind_favored_overlap": (("time", "lat", "lon"), np.array([[[1.1]], [[1.4]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.45]], [[0.55]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[520.0]], [[820.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.002]], [[0.001]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.03]], [[0.06]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.2]], [[0.8]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction_path = tmp_path / "forecast_products_2024-03-14_00.nc"
    prediction.to_netcdf(prediction_path)
    verification_payload = {
        "run_summary": {"init_date": "2024-03-14", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [
            {"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 1},
            {"valid_date": "2024-03-15", "observed_category": "hail_outbreak_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
        ],
    }
    verification_path = tmp_path / "2024-03-14_verification.json"
    verification_path.write_text(json.dumps(verification_payload), encoding="utf-8")

    summary = summarize_core_variants_from_artifacts([("2024-03-14", prediction_path, verification_path)], source_variant="baseline", component_variant="baseline", score_variant="baseline")

    assert set(summary["core_variant"]) == set(CORE_VARIANTS)
    assert {"core_root_cause_for_2024_03_14", "top_day_for_2024_03_14_window"}.issubset(summary.columns)


def test_broader_validation_checkpoint_summary_reports_requested_metrics(tmp_path: Path) -> None:
    def _write_case(init_date: str, times: list[str], sig: list[float], outbreak: list[float], overlap: list[float], scp: list[float], syn: list[float], tor: list[float], any_prob: list[float], learned: list[float], per_day: list[dict[str, object]]) -> tuple[str, Path, Path]:
        prediction = xr.Dataset(
            {
                "sig_tor_support": (("time", "lat", "lon"), np.array(sig, dtype=float).reshape(len(times), 1, 1)),
                "outbreak_risk": (("time", "lat", "lon"), np.array(outbreak, dtype=float).reshape(len(times), 1, 1)),
                "tornado_favored_overlap": (("time", "lat", "lon"), np.array(overlap, dtype=float).reshape(len(times), 1, 1)),
                "scp_proxy": (("time", "lat", "lon"), np.array(scp, dtype=float).reshape(len(times), 1, 1)),
                "synoptic_support": (("time", "lat", "lon"), np.array(syn, dtype=float).reshape(len(times), 1, 1)),
                "tornado_prob": (("time", "lat", "lon"), np.array(tor, dtype=float).reshape(len(times), 1, 1)),
                "any_prob": (("time", "lat", "lon"), np.array(any_prob, dtype=float).reshape(len(times), 1, 1)),
                "tornado_concern_prob": (("time", "lat", "lon"), np.array(learned, dtype=float).reshape(len(times), 1, 1)),
            },
            coords={"time": pd.to_datetime(times), "lat": [35.0], "lon": [-98.0]},
        )
        prediction_path = tmp_path / f"forecast_products_{init_date}_00.nc"
        prediction.to_netcdf(prediction_path)
        verification_path = tmp_path / f"{init_date}_verification.json"
        verification_path.write_text(
            json.dumps(
                {
                    "run_summary": {"init_date": init_date, "evaluation_source": "local_staged_gfs"},
                    "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                    "per_day": per_day,
                }
            ),
            encoding="utf-8",
        )
        return init_date, prediction_path, verification_path

    artifact_pairs = [
        _write_case(
            "2024-03-14",
            ["2024-03-14T00:00:00", "2024-03-15T00:00:00"],
            [1.35, 1.90],
            [0.18, 0.27],
            [1.65, 1.85],
            [0.45, 0.55],
            [520.0, 820.0],
            [0.002, 0.001],
            [0.03, 0.06],
            [0.2, 0.8],
            [
                {"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 1},
                {"valid_date": "2024-03-15", "observed_category": "hail_outbreak_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
            ],
        ),
        _write_case(
            "2024-04-26",
            ["2024-04-28T00:00:00", "2024-04-29T00:00:00"],
            [2.20, 1.15],
            [0.24, 0.18],
            [2.75, 1.50],
            [0.72, 0.18],
            [760.0, 930.0],
            [0.003, 0.001],
            [0.05, 0.04],
            [0.65, 0.25],
            [
                {"valid_date": "2024-04-28", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 0},
                {"valid_date": "2024-04-29", "observed_category": "active_non_outbreak_severe_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
            ],
        ),
    ]

    summary = summarize_broader_validation_checkpoint_from_artifacts(
        artifact_pairs,
        core_variant="baseline",
        source_variant="baseline",
        component_variant="baseline",
        score_variant="baseline",
    )

    assert list(summary["raw_core_variant"]) == BROADER_VALIDATION_RAW_CORE_VARIANTS
    assert {
        "real_ingest_windows",
        "hail_outranks_tornado_failure_real",
        "non_outbreak_outranks_outbreak_failure_real",
        "top_day_category_mismatch_real",
        "tornado_day_ranked_first_real",
        "sig_tor_day_ranked_first_real",
        "mean_top_minus_best_tornado_margin_real",
        "median_top_minus_best_tornado_margin_real",
    }.issubset(summary.columns)


def test_markdown_summary_contains_broader_validation_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    cases = {
        "2024-03-14": {
            "times": ["2024-03-14T00:00:00", "2024-03-15T00:00:00"],
            "sig": [1.35, 1.90],
            "outbreak": [0.18, 0.27],
            "overlap": [1.65, 1.85],
            "scp": [0.45, 0.55],
            "syn": [520.0, 820.0],
            "tor": [0.002, 0.001],
            "any_prob": [0.03, 0.06],
            "learned": [0.2, 0.8],
            "per_day": [
                {"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 1},
                {"valid_date": "2024-03-15", "observed_category": "hail_outbreak_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
            ],
        },
        "2024-04-26": {
            "times": ["2024-04-28T00:00:00", "2024-04-29T00:00:00"],
            "sig": [2.20, 1.15],
            "outbreak": [0.24, 0.18],
            "overlap": [2.75, 1.50],
            "scp": [0.72, 0.18],
            "syn": [760.0, 930.0],
            "tor": [0.003, 0.001],
            "any_prob": [0.05, 0.04],
            "learned": [0.65, 0.25],
            "per_day": [
                {"valid_date": "2024-04-28", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 0},
                {"valid_date": "2024-04-29", "observed_category": "active_non_outbreak_severe_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
            ],
        },
    }

    for init_date, case in cases.items():
        prediction = xr.Dataset(
            {
                "sig_tor_support": (("time", "lat", "lon"), np.array(case["sig"], dtype=float).reshape(2, 1, 1)),
                "outbreak_risk": (("time", "lat", "lon"), np.array(case["outbreak"], dtype=float).reshape(2, 1, 1)),
                "tornado_favored_overlap": (("time", "lat", "lon"), np.array(case["overlap"], dtype=float).reshape(2, 1, 1)),
                "scp_proxy": (("time", "lat", "lon"), np.array(case["scp"], dtype=float).reshape(2, 1, 1)),
                "synoptic_support": (("time", "lat", "lon"), np.array(case["syn"], dtype=float).reshape(2, 1, 1)),
                "tornado_prob": (("time", "lat", "lon"), np.array(case["tor"], dtype=float).reshape(2, 1, 1)),
                "any_prob": (("time", "lat", "lon"), np.array(case["any_prob"], dtype=float).reshape(2, 1, 1)),
                "tornado_concern_prob": (("time", "lat", "lon"), np.array(case["learned"], dtype=float).reshape(2, 1, 1)),
            },
            coords={"time": pd.to_datetime(case["times"]), "lat": [35.0], "lon": [-98.0]},
        )
        prediction.to_netcdf(outputs_dir / f"forecast_products_{init_date}_00.nc")
        (verification_dir / f"{init_date}_verification.json").write_text(
            json.dumps(
                {
                    "run_summary": {"init_date": init_date, "evaluation_source": "local_staged_gfs"},
                    "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                    "per_day": case["per_day"],
                }
            ),
            encoding="utf-8",
        )

    markdown_path = tmp_path / "checkpoint.md"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["tornado_concern_eval", "--dates", "2024-03-14", "2024-04-26", "--output-md", str(markdown_path)],
    )
    module.main()

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Broader Validation Checkpoint" in markdown
    assert "masked_core_steeper" in markdown


def test_baseline_vs_masked_core_steeper_summary_reports_requested_metrics(tmp_path: Path) -> None:
    def _write_case(init_date: str, times: list[str], sig: list[float], outbreak: list[float], overlap: list[float], scp: list[float], syn: list[float], tor: list[float], any_prob: list[float], learned: list[float], per_day: list[dict[str, object]]) -> tuple[str, Path, Path]:
        prediction = xr.Dataset(
            {
                "sig_tor_support": (("time", "lat", "lon"), np.array(sig, dtype=float).reshape(len(times), 1, 1)),
                "outbreak_risk": (("time", "lat", "lon"), np.array(outbreak, dtype=float).reshape(len(times), 1, 1)),
                "tornado_favored_overlap": (("time", "lat", "lon"), np.array(overlap, dtype=float).reshape(len(times), 1, 1)),
                "scp_proxy": (("time", "lat", "lon"), np.array(scp, dtype=float).reshape(len(times), 1, 1)),
                "synoptic_support": (("time", "lat", "lon"), np.array(syn, dtype=float).reshape(len(times), 1, 1)),
                "tornado_prob": (("time", "lat", "lon"), np.array(tor, dtype=float).reshape(len(times), 1, 1)),
                "any_prob": (("time", "lat", "lon"), np.array(any_prob, dtype=float).reshape(len(times), 1, 1)),
                "tornado_concern_prob": (("time", "lat", "lon"), np.array(learned, dtype=float).reshape(len(times), 1, 1)),
            },
            coords={"time": pd.to_datetime(times), "lat": [35.0], "lon": [-98.0]},
        )
        prediction_path = tmp_path / f"forecast_products_{init_date}_00.nc"
        prediction.to_netcdf(prediction_path)
        verification_path = tmp_path / f"{init_date}_verification.json"
        verification_path.write_text(
            json.dumps(
                {
                    "run_summary": {"init_date": init_date, "evaluation_source": "local_staged_gfs"},
                    "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                    "per_day": per_day,
                }
            ),
            encoding="utf-8",
        )
        return init_date, prediction_path, verification_path

    artifact_pairs = [
        _write_case(
            "2024-03-14",
            ["2024-03-14T00:00:00", "2024-03-15T00:00:00"],
            [1.35, 1.90],
            [0.18, 0.27],
            [1.65, 1.85],
            [0.45, 0.55],
            [520.0, 820.0],
            [0.002, 0.001],
            [0.03, 0.06],
            [0.2, 0.8],
            [
                {"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 1},
                {"valid_date": "2024-03-15", "observed_category": "hail_outbreak_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
            ],
        ),
        _write_case(
            "2024-04-26",
            ["2024-04-28T00:00:00", "2024-04-29T00:00:00"],
            [2.20, 1.15],
            [0.24, 0.18],
            [2.75, 1.50],
            [0.72, 0.18],
            [760.0, 930.0],
            [0.003, 0.001],
            [0.05, 0.04],
            [0.65, 0.25],
            [
                {"valid_date": "2024-04-28", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 0},
                {"valid_date": "2024-04-29", "observed_category": "active_non_outbreak_severe_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
            ],
        ),
    ]

    summary = summarize_baseline_vs_masked_core_steeper_from_artifacts(
        artifact_pairs,
        core_variant="baseline",
        source_variant="baseline",
        component_variant="baseline",
        score_variant="baseline",
    )

    assert list(summary["raw_core_variant"]) == BASELINE_VS_MASKED_CORE_STEEPER_VARIANTS
    assert {
        "real_ingest_windows",
        "hail_outranks_tornado_failure_real",
        "non_outbreak_outranks_outbreak_failure_real",
        "top_day_category_mismatch_real",
        "tornado_day_ranked_first_real",
        "sig_tor_day_ranked_first_real",
        "mean_top_minus_best_tornado_margin_real",
        "median_top_minus_best_tornado_margin_real",
    }.issubset(summary.columns)


def test_markdown_summary_contains_baseline_vs_masked_core_steeper_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    cases = {
        "2024-03-14": {
            "times": ["2024-03-14T00:00:00", "2024-03-15T00:00:00"],
            "sig": [1.35, 1.90],
            "outbreak": [0.18, 0.27],
            "overlap": [1.65, 1.85],
            "scp": [0.45, 0.55],
            "syn": [520.0, 820.0],
            "tor": [0.002, 0.001],
            "any_prob": [0.03, 0.06],
            "learned": [0.2, 0.8],
            "per_day": [
                {"valid_date": "2024-03-14", "observed_category": "significant_tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 1},
                {"valid_date": "2024-03-15", "observed_category": "hail_outbreak_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
            ],
        },
        "2024-04-26": {
            "times": ["2024-04-28T00:00:00", "2024-04-29T00:00:00"],
            "sig": [2.20, 1.15],
            "outbreak": [0.24, 0.18],
            "overlap": [2.75, 1.50],
            "scp": [0.72, 0.18],
            "syn": [760.0, 930.0],
            "tor": [0.003, 0.001],
            "any_prob": [0.05, 0.04],
            "learned": [0.65, 0.25],
            "per_day": [
                {"valid_date": "2024-04-28", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1, "observed_significant_tornado_support": 0},
                {"valid_date": "2024-04-29", "observed_category": "active_non_outbreak_severe_day", "observed_tornado_outbreak": 0, "observed_significant_tornado_support": 0},
            ],
        },
    }

    for init_date, case in cases.items():
        prediction = xr.Dataset(
            {
                "sig_tor_support": (("time", "lat", "lon"), np.array(case["sig"], dtype=float).reshape(2, 1, 1)),
                "outbreak_risk": (("time", "lat", "lon"), np.array(case["outbreak"], dtype=float).reshape(2, 1, 1)),
                "tornado_favored_overlap": (("time", "lat", "lon"), np.array(case["overlap"], dtype=float).reshape(2, 1, 1)),
                "scp_proxy": (("time", "lat", "lon"), np.array(case["scp"], dtype=float).reshape(2, 1, 1)),
                "synoptic_support": (("time", "lat", "lon"), np.array(case["syn"], dtype=float).reshape(2, 1, 1)),
                "tornado_prob": (("time", "lat", "lon"), np.array(case["tor"], dtype=float).reshape(2, 1, 1)),
                "any_prob": (("time", "lat", "lon"), np.array(case["any_prob"], dtype=float).reshape(2, 1, 1)),
                "tornado_concern_prob": (("time", "lat", "lon"), np.array(case["learned"], dtype=float).reshape(2, 1, 1)),
            },
            coords={"time": pd.to_datetime(case["times"]), "lat": [35.0], "lon": [-98.0]},
        )
        prediction.to_netcdf(outputs_dir / f"forecast_products_{init_date}_00.nc")
        (verification_dir / f"{init_date}_verification.json").write_text(
            json.dumps(
                {
                    "run_summary": {"init_date": init_date, "evaluation_source": "local_staged_gfs"},
                    "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
                    "per_day": case["per_day"],
                }
            ),
            encoding="utf-8",
        )

    markdown_path = tmp_path / "checkpoint.md"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["tornado_concern_eval", "--dates", "2024-03-14", "2024-04-26", "--output-md", str(markdown_path)],
    )
    module.main()

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Baseline vs Masked Core Steeper Validation" in markdown
    assert "masked_core_steeper" in markdown


def test_raw_core_variant_cli_is_accepted_and_baseline_remains_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from severewx.cli import tornado_concern_eval as module

    outputs_dir = tmp_path / "data" / "outputs"
    verification_dir = outputs_dir / "verification"
    outputs_dir.mkdir(parents=True)
    verification_dir.mkdir(parents=True)

    times = pd.to_datetime(["2024-04-26T00:00:00"])
    prediction = xr.Dataset(
        {
            "sig_tor_support": (("time", "lat", "lon"), np.array([[[2.0]]], dtype=float)),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.18]]], dtype=float)),
            "tornado_favored_overlap": (("time", "lat", "lon"), np.array([[[2.4]]], dtype=float)),
            "scp_proxy": (("time", "lat", "lon"), np.array([[[0.85]]], dtype=float)),
            "synoptic_support": (("time", "lat", "lon"), np.array([[[640.0]]], dtype=float)),
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.003]]], dtype=float)),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.04]]], dtype=float)),
            "tornado_concern_prob": (("time", "lat", "lon"), np.array([[[0.55]]], dtype=float)),
        },
        coords={"time": times, "lat": [35.0], "lon": [-98.0]},
    )
    prediction.to_netcdf(outputs_dir / "forecast_products_2024-04-26_00.nc")
    verification_payload = {
        "run_summary": {"init_date": "2024-04-26", "evaluation_source": "local_staged_gfs"},
        "evaluation_metadata": {"ingest_summary": {"source": "local_staged_gfs"}},
        "per_day": [{"valid_date": "2024-04-26", "observed_category": "tornado_outbreak_day", "observed_tornado_outbreak": 1}],
    }
    (verification_dir / "2024-04-26_verification.json").write_text(json.dumps(verification_payload), encoding="utf-8")

    default_csv = tmp_path / "default_raw_core.csv"
    masked_csv = tmp_path / "masked_raw_core.csv"
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("sys.argv", ["tornado_concern_eval", "--dates", "2024-04-26", "--output-csv", str(default_csv)])
    module.main()
    default_ranked = pd.read_csv(default_csv)

    monkeypatch.setattr(
        "sys.argv",
        ["tornado_concern_eval", "--dates", "2024-04-26", "--raw-core-variant", "masked_core_only", "--output-csv", str(masked_csv)],
    )
    module.main()
    masked_ranked = pd.read_csv(masked_csv)

    assert RAW_CORE_VARIANTS == [
        "baseline",
        "masked_core_only",
        "masked_core_stronger",
        "masked_core_calibrated",
        "masked_core_two_stage",
        "masked_core_lower_floor",
        "masked_core_steeper",
        "footprint_discriminating",
    ]
    assert set(default_ranked["raw_core_variant_name"]) == {"baseline"}
    assert set(masked_ranked["raw_core_variant_name"]) == {"masked_core_only"}


def test_masked_core_only_can_improve_support_masked_hail_over_tornado_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-03-14",
                "support_top": 0.11,
                "normalized_synoptic_support": 0.72,
                "context_top": 0.82,
                "joint_area": 0.0006,
                "core_top": 0.010,
                "scp_top": 0.52,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.20,
                "tornado_overlap_max": 2.1,
                "learned_tornado_concern_prob": 0.20,
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "observed_category": "tornado_outbreak_day",
                "is_real_ingest": True,
            },
            {
                "valid_date": "2024-03-15",
                "support_top": 0.18,
                "normalized_synoptic_support": 0.86,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.016,
                "scp_top": 0.16,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 1.5,
                "learned_tornado_concern_prob": 0.20,
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "observed_category": "hail_outbreak_day",
                "is_real_ingest": True,
            },
        ]
    )

    baseline = _apply_score_variant_to_frame(
        _apply_component_variant_to_frame(
            _apply_source_variant_to_frame(_apply_core_variant_to_frame(_apply_raw_core_variant_to_frame(frame, "baseline"), "baseline"), "baseline"),
            "baseline",
        ),
        "baseline",
    )
    masked = _apply_score_variant_to_frame(
        _apply_component_variant_to_frame(
            _apply_source_variant_to_frame(_apply_core_variant_to_frame(_apply_raw_core_variant_to_frame(frame, "masked_core_only"), "baseline"), "baseline"),
            "baseline",
        ),
        "baseline",
    )

    hail_base = baseline.loc[baseline["valid_date"] == "2024-03-15"].iloc[0]
    tornado_base = baseline.loc[baseline["valid_date"] == "2024-03-14"].iloc[0]
    hail_masked = masked.loc[masked["valid_date"] == "2024-03-15"].iloc[0]
    tornado_masked = masked.loc[masked["valid_date"] == "2024-03-14"].iloc[0]

    assert float(hail_masked["raw_core_mask_factor"]) < float(tornado_masked["raw_core_mask_factor"])
    assert float(hail_masked["tornado_concern_score"] - tornado_masked["tornado_concern_score"]) < float(
        hail_base["tornado_concern_score"] - tornado_base["tornado_concern_score"]
    )


def test_masked_core_stronger_is_at_least_as_strong_a_mask_in_low_support_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-03-15",
                "support_top": 0.18,
                "normalized_synoptic_support": 0.86,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.016,
                "scp_top": 0.16,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 1.5,
            }
        ]
    )

    masked = _apply_raw_core_variant_to_frame(frame, "masked_core_only").iloc[0]
    stronger = _apply_raw_core_variant_to_frame(frame, "masked_core_stronger").iloc[0]

    assert float(stronger["raw_core_mask_factor"]) <= float(masked["raw_core_mask_factor"])
    assert float(stronger["raw_core_final_value"]) <= float(masked["raw_core_final_value"])


def test_masked_core_calibrated_is_at_least_as_strong_as_stronger_in_scp_sig_support_favored_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-03-14",
                "support_top": 0.11,
                "normalized_synoptic_support": 0.72,
                "context_top": 0.82,
                "joint_area": 0.0024,
                "core_top": 0.012,
                "scp_top": 0.40,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.8,
                "outbreak_risk_max": 0.20,
                "tornado_overlap_max": 1.45,
            }
        ]
    )

    stronger = _apply_raw_core_variant_to_frame(frame, "masked_core_stronger").iloc[0]
    calibrated = _apply_raw_core_variant_to_frame(frame, "masked_core_calibrated").iloc[0]

    assert float(calibrated["raw_core_mask_factor"]) >= float(stronger["raw_core_mask_factor"])
    assert float(calibrated["raw_core_final_value"]) >= float(stronger["raw_core_final_value"])


def test_masked_core_two_stage_is_at_least_as_strong_as_calibrated_in_weak_support_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-03-15",
                "support_top": 0.12,
                "normalized_synoptic_support": 0.84,
                "context_top": 0.98,
                "joint_area": 0.0028,
                "core_top": 0.014,
                "scp_top": 0.20,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.25,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 1.55,
            }
        ]
    )

    calibrated = _apply_raw_core_variant_to_frame(frame, "masked_core_calibrated").iloc[0]
    two_stage = _apply_raw_core_variant_to_frame(frame, "masked_core_two_stage").iloc[0]

    assert float(two_stage["raw_core_mask_factor"]) <= float(calibrated["raw_core_mask_factor"])
    assert float(two_stage["raw_core_final_value"]) <= float(calibrated["raw_core_final_value"])


def test_masked_core_lower_floor_applies_smaller_mask_than_two_stage_in_weak_support_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-03-15",
                "support_top": 0.12,
                "normalized_synoptic_support": 0.84,
                "context_top": 0.98,
                "joint_area": 0.0028,
                "core_top": 0.014,
                "scp_top": 0.20,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.25,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 1.55,
            }
        ]
    )

    two_stage = _apply_raw_core_variant_to_frame(frame, "masked_core_two_stage").iloc[0]
    lower_floor = _apply_raw_core_variant_to_frame(frame, "masked_core_lower_floor").iloc[0]

    assert float(lower_floor["raw_core_mask_factor"]) <= float(two_stage["raw_core_mask_factor"])
    assert float(lower_floor["raw_core_final_value"]) <= float(two_stage["raw_core_final_value"])


def test_masked_core_steeper_applies_smaller_mask_than_lower_floor_in_weak_support_case() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-03-15",
                "support_top": 0.12,
                "normalized_synoptic_support": 0.84,
                "context_top": 0.98,
                "joint_area": 0.0028,
                "core_top": 0.014,
                "scp_top": 0.20,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.25,
                "outbreak_risk_max": 0.24,
                "tornado_overlap_max": 1.55,
            }
        ]
    )

    lower_floor = _apply_raw_core_variant_to_frame(frame, "masked_core_lower_floor").iloc[0]
    steeper = _apply_raw_core_variant_to_frame(frame, "masked_core_steeper").iloc[0]

    assert float(steeper["raw_core_mask_factor"]) <= float(lower_floor["raw_core_mask_factor"])
    assert float(steeper["raw_core_final_value"]) <= float(lower_floor["raw_core_final_value"])


def test_footprint_discriminating_penalizes_broad_high_end_footprint() -> None:
    frame = pd.DataFrame(
        [
            {
                "valid_date": "2024-05-27",
                "support_top": 1.00,
                "normalized_synoptic_support": 0.68,
                "context_top": 1.00,
                "joint_area": 0.016,
                "core_top": 0.82,
                "scp_top": 0.97,
                "penalty_top": 0.77,
                "sig_tor_support_max": 2.50,
                "outbreak_risk_max": 0.35,
                "tornado_overlap_max": 6.00,
            },
            {
                "valid_date": "2024-05-26",
                "support_top": 1.00,
                "normalized_synoptic_support": 0.54,
                "context_top": 1.00,
                "joint_area": 0.003,
                "core_top": 0.62,
                "scp_top": 0.96,
                "penalty_top": 0.89,
                "sig_tor_support_max": 2.50,
                "outbreak_risk_max": 0.36,
                "tornado_overlap_max": 6.00,
            },
        ]
    )

    baseline = _apply_raw_core_variant_to_frame(frame, "baseline")
    discriminating = _apply_raw_core_variant_to_frame(frame, "footprint_discriminating")

    broad = discriminating.loc[discriminating["valid_date"].eq("2024-05-27")].iloc[0]
    compact = discriminating.loc[discriminating["valid_date"].eq("2024-05-26")].iloc[0]
    broad_base = baseline.loc[baseline["valid_date"].eq("2024-05-27")].iloc[0]
    compact_base = baseline.loc[baseline["valid_date"].eq("2024-05-26")].iloc[0]

    assert float(broad["raw_core_mask_factor"]) < 1.0
    assert float(compact["raw_core_mask_factor"]) == 1.0
    assert float(broad["raw_core_final_value"]) < float(broad_base["raw_core_final_value"])
    assert float(compact["raw_core_final_value"]) == float(compact_base["raw_core_final_value"])


def test_masked_core_only_guardrail_keeps_outbreak_above_non_outbreak() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.13,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0008,
                "core_top": 0.018,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.18,
                "normalized_synoptic_support": 0.98,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.022,
                "scp_top": 0.16,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.25,
                "tornado_overlap_max": 1.5,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    masked = summarize_case_windows(
        _apply_score_variant_to_frame(
            _apply_component_variant_to_frame(
                _apply_source_variant_to_frame(_apply_core_variant_to_frame(_apply_raw_core_variant_to_frame(frame, "masked_core_only"), "baseline"), "baseline"),
                "baseline",
            ),
            "baseline",
        )
    )

    assert masked.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(masked.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_masked_core_stronger_guardrail_keeps_outbreak_above_non_outbreak() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.13,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0008,
                "core_top": 0.018,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.18,
                "normalized_synoptic_support": 0.98,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.022,
                "scp_top": 0.16,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.25,
                "tornado_overlap_max": 1.5,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    stronger = summarize_case_windows(
        _apply_score_variant_to_frame(
            _apply_component_variant_to_frame(
                _apply_source_variant_to_frame(_apply_core_variant_to_frame(_apply_raw_core_variant_to_frame(frame, "masked_core_stronger"), "baseline"), "baseline"),
                "baseline",
            ),
            "baseline",
        )
    )

    assert stronger.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(stronger.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_masked_core_calibrated_guardrail_keeps_outbreak_above_non_outbreak() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.13,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0008,
                "core_top": 0.018,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.18,
                "normalized_synoptic_support": 0.98,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.022,
                "scp_top": 0.16,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.25,
                "tornado_overlap_max": 1.5,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    calibrated = summarize_case_windows(
        _apply_score_variant_to_frame(
            _apply_component_variant_to_frame(
                _apply_source_variant_to_frame(_apply_core_variant_to_frame(_apply_raw_core_variant_to_frame(frame, "masked_core_calibrated"), "baseline"), "baseline"),
                "baseline",
            ),
            "baseline",
        )
    )

    assert calibrated.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(calibrated.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_masked_core_two_stage_guardrail_keeps_outbreak_above_non_outbreak() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.13,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0008,
                "core_top": 0.018,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.18,
                "normalized_synoptic_support": 0.98,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.022,
                "scp_top": 0.16,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.25,
                "tornado_overlap_max": 1.5,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    two_stage = summarize_case_windows(
        _apply_score_variant_to_frame(
            _apply_component_variant_to_frame(
                _apply_source_variant_to_frame(_apply_core_variant_to_frame(_apply_raw_core_variant_to_frame(frame, "masked_core_two_stage"), "baseline"), "baseline"),
                "baseline",
            ),
            "baseline",
        )
    )

    assert two_stage.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(two_stage.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_masked_core_lower_floor_guardrail_keeps_outbreak_above_non_outbreak() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.13,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0008,
                "core_top": 0.018,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.18,
                "normalized_synoptic_support": 0.98,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.022,
                "scp_top": 0.16,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.25,
                "tornado_overlap_max": 1.5,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    lower_floor = summarize_case_windows(
        _apply_score_variant_to_frame(
            _apply_component_variant_to_frame(
                _apply_source_variant_to_frame(_apply_core_variant_to_frame(_apply_raw_core_variant_to_frame(frame, "masked_core_lower_floor"), "baseline"), "baseline"),
                "baseline",
            ),
            "baseline",
        )
    )

    assert lower_floor.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(lower_floor.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_masked_core_steeper_guardrail_keeps_outbreak_above_non_outbreak() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-28",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 1,
                "support_top": 0.13,
                "normalized_synoptic_support": 0.70,
                "context_top": 0.82,
                "joint_area": 0.0008,
                "core_top": 0.018,
                "scp_top": 0.48,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.9,
                "outbreak_risk_max": 0.22,
                "tornado_overlap_max": 2.2,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-04-26",
                "valid_date": "2024-04-29",
                "observed_category": "active_non_outbreak_severe_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.18,
                "normalized_synoptic_support": 0.98,
                "context_top": 1.00,
                "joint_area": 0.0032,
                "core_top": 0.022,
                "scp_top": 0.16,
                "penalty_top": 0.08,
                "sig_tor_support_max": 1.2,
                "outbreak_risk_max": 0.25,
                "tornado_overlap_max": 1.5,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    steeper = summarize_case_windows(
        _apply_score_variant_to_frame(
            _apply_component_variant_to_frame(
                _apply_source_variant_to_frame(_apply_core_variant_to_frame(_apply_raw_core_variant_to_frame(frame, "masked_core_steeper"), "baseline"), "baseline"),
                "baseline",
            ),
            "baseline",
        )
    )

    assert steeper.loc[0, "top_valid_date"] == "2024-04-28"
    assert not bool(steeper.loc[0, "non_outbreak_outranks_outbreak_failure"])


def test_failure_targeted_contamination_penalizes_broad_weak_structure() -> None:
    frame = pd.DataFrame(
        [
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-22",
                "observed_category": "hail_outbreak_day",
                "observed_tornado_outbreak": 0,
                "observed_significant_tornado_support": 0,
                "support_top": 0.22,
                "normalized_synoptic_support": 0.90,
                "context_top": 1.00,
                "joint_area": 0.004,
                "core_top": 0.18,
                "effective_core_top": 0.18,
                "scp_top": 0.28,
                "penalty_top": 0.05,
                "sig_tor_support_max": 1.45,
                "outbreak_risk_max": 0.55,
                "tornado_overlap_max": 1.55,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
            {
                "init_date": "2024-05-19",
                "valid_date": "2024-05-21",
                "observed_category": "tornado_outbreak_day",
                "observed_tornado_outbreak": 1,
                "observed_significant_tornado_support": 0,
                "support_top": 0.18,
                "normalized_synoptic_support": 0.65,
                "context_top": 0.82,
                "joint_area": 0.0007,
                "core_top": 0.12,
                "effective_core_top": 0.12,
                "scp_top": 0.55,
                "penalty_top": 0.05,
                "sig_tor_support_max": 2.20,
                "outbreak_risk_max": 0.65,
                "tornado_overlap_max": 3.20,
                "learned_tornado_concern_prob": 0.30,
                "is_real_ingest": True,
            },
        ]
    )

    baseline = _apply_source_variant_to_frame(frame, "baseline")
    targeted = _apply_source_variant_to_frame(frame, "failure_targeted_contamination")

    hail_baseline = baseline.loc[baseline["valid_date"].eq("2024-05-22")].iloc[0]
    hail_targeted = targeted.loc[targeted["valid_date"].eq("2024-05-22")].iloc[0]
    tornado_baseline = baseline.loc[baseline["valid_date"].eq("2024-05-21")].iloc[0]
    tornado_targeted = targeted.loc[targeted["valid_date"].eq("2024-05-21")].iloc[0]

    assert float(hail_targeted["source_penalty_term"]) < float(tornado_targeted["source_penalty_term"])
    assert float(hail_targeted["raw_base_score_before_variant"]) < float(hail_baseline["raw_base_score_before_variant"])
    assert float(tornado_targeted["raw_base_score_before_variant"]) > 0.70 * float(tornado_baseline["raw_base_score_before_variant"])
