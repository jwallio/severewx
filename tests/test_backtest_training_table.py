import json
from pathlib import Path

from severewx.backtest.training_table import build_training_frame, write_training_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(path: Path) -> Path:
    _write_json(
        path,
        {
            "manifest_id": "training_table_test",
            "require_case_metadata": True,
            "target_product": {
                "field": "tornado_environment_outlook_hybrid_consensus",
                "artifact_source": "consensus",
                "map_domain": "conus",
                "map_style": "outlook",
                "display_preset": "auto",
                "lead_days": [1, 2, 3],
                "synthetic_fallback_allowed": False,
                "default_sources": ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent", "ecmwf_recent"],
            },
            "cases": [
                {
                    "case_id": "case_tune",
                    "date": "2024-04-02",
                    "cycle": "00",
                    "fold": "tune",
                    "region": "plains",
                    "season": "spring",
                    "regime": "dryline_supercell",
                    "tags": ["tornado_relevant", "event"],
                    "expected_signal": "event",
                },
                {
                    "case_id": "case_test",
                    "date": "2024-06-01",
                    "cycle": "00",
                    "fold": "test",
                    "region": "midwest",
                    "season": "summer",
                    "regime": "northwest_flow",
                    "tags": ["hard_negative"],
                    "expected_signal": "non_tornado_severe_day",
                },
            ],
        },
    )
    return path


def _run_dir(path: Path) -> Path:
    products = []
    scores = []
    rows = []
    for case_id, date, observed in [("case_tune", "2024-04-02", 1), ("case_test", "2024-06-01", 0)]:
        verification = path / "verification" / f"{case_id}.json"
        _write_json(
            verification,
            {
                "per_day": [
                    {
                        "valid_date": date,
                        "lead_day": 1,
                        "forecast_tornado_prob": 0.20 if observed else 0.08,
                        "forecast_confidence": 0.7,
                        "forecast_signal_quality": 0.6,
                        "observed_tornado_outbreak": observed,
                        "observed_any_outbreak": observed,
                        "observed_tornado_report_count": 4 if observed else 0,
                        "observed_category": "tornado_outbreak_day" if observed else "non_tornado_severe_day",
                        "observed_report_source_is_real": True,
                        "tornado_brier": 0.01,
                    }
                ]
            },
        )
        metadata = path / "products" / f"{case_id}.json"
        _write_json(metadata, {"publication_status": "public_candidate", "readiness_status": "public_candidate"})
        products.append(
            {
                "case_id": case_id,
                "date": date,
                "cycle": "00",
                "day": 1,
                "valid_date": date,
                "metadata_path": str(metadata),
                "publication_status": "public_candidate",
                "public_ready": True,
                "max_tornado_concern_prob": 0.20 if observed else 0.08,
            }
        )
        scores.append(
            {
                "case_id": case_id,
                "date": date,
                "cycle": "00",
                "verification_path": str(verification),
                "verification_status": "available",
            }
        )
        for source in ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent", "ecmwf_recent"]:
            rows.append(
                {
                    "case_id": case_id,
                    "date": date,
                    "cycle": "00",
                    "source": source,
                    "status": "available",
                    "source_mode": "real",
                }
            )
    _write_json(
        path / "source_availability.json",
        {
            "manifest_id": "training_table_test",
            "run_options": {"sources": ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent", "ecmwf_recent"]},
            "rows": rows,
            "products": products,
            "scores": scores,
        },
    )
    return path


def test_build_training_frame_combines_manifest_sources_and_verified_labels(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    run_dir = _run_dir(tmp_path / "run")

    frame = build_training_frame(manifest, [run_dir])

    assert len(frame) == 2
    assert set(frame["fold"]) == {"tune", "test"}
    assert set(frame["lead_day"]) == {1}
    assert frame["verified"].all()
    assert set(frame["source_count"]) == {5}
    assert set(frame["source_availability_tier"]) == {"recent_full_stack"}
    assert frame.loc[frame["case_id"] == "case_tune", "observed_tornado_outbreak"].iloc[0] == 1


def test_write_training_artifacts_outputs_skill_and_calibration(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    run_dir = _run_dir(tmp_path / "run")
    output_dir = tmp_path / "training"

    result = write_training_artifacts(manifest_path=manifest, run_dirs=[run_dir], output_dir=output_dir)

    assert result.table_csv.exists()
    assert result.summary_json.exists()
    assert result.calibration_json.exists()
    table = json.loads(result.table_json.read_text(encoding="utf-8"))
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    calibration = json.loads(result.calibration_json.read_text(encoding="utf-8"))
    assert len(table) == 2
    assert "calibrated_tornado_probability" in table[0]
    assert summary["segments"]["fold"]["tune"]["row_count"] == 1
    assert summary["segments"]["source_availability_tier"]["recent_full_stack"]["row_count"] == 2
    assert calibration["status"] in {"candidate_fit", "insufficient_tune_bin_rows"}
