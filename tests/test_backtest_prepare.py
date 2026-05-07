import json

from severewx.backtest.manifest import load_backtest_manifest
from severewx.cli import prepare_backtest_cases as prep
from severewx.config import load_settings
from severewx.models.forecast_consensus import ConsensusSource
from severewx.utils.paths import build_paths


def test_default_backtest_manifest_defines_consensus_target() -> None:
    manifest = load_backtest_manifest("backtests/tornado_environment_consensus_v1.json")

    assert manifest.manifest_id == "tornado_environment_consensus_v1"
    assert manifest.target_product["field"] == "tornado_environment_outlook_hybrid_consensus"
    assert manifest.target_product["artifact_source"] == "consensus"
    assert manifest.target_product["map_domain"] == "conus"
    assert manifest.target_product["lead_days"] == [1, 2, 3]
    assert manifest.target_product["synthetic_fallback_allowed"] is False
    assert manifest.target_product["default_sources"] == ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent"]
    assert len(manifest.cases) >= 10
    assert len({case.case_id for case in manifest.cases}) == len(manifest.cases)


def test_prepare_backtest_cases_dry_run_writes_deterministic_ledgers(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)

    manifest, rows, outputs = prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        dry_run=True,
        limit=2,
    )

    assert manifest.manifest_id == "tornado_environment_consensus_v1"
    assert len(rows) == 8
    assert {row["status"] for row in rows} == {"planned"}
    assert {row["reason"] for row in rows} == {"dry_run"}
    assert outputs[0].exists()
    assert outputs[1].exists()
    assert outputs[2].exists()
    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["case_count"] == len(manifest.cases)
    assert len(payload["case_summary"]) == 2
    assert payload["case_summary"][0]["ready_for_scoring"] is False


def test_prepare_backtest_cases_records_available_and_unavailable_sources(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    paths = build_paths(settings)

    def fake_run_source_product(date, cycle, source, settings, *, skip_render):
        metadata_path = prep.source_metadata_path(paths.outputs, date, cycle, source)
        product_path = prep.source_product_path(paths.outputs, date, cycle, source)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if source == "hrrr_recent":
            product_path.write_text("fake-netcdf", encoding="utf-8")
            metadata_path.write_text(
                json.dumps(
                    {
                        "ingest_summary": {
                            "source": source,
                            "source_mode": "real",
                            "source_model": "hrrr",
                            "available_fields": ["cape", "mslp"],
                            "missing_requested_leads": [60, 72],
                        }
                    }
                ),
                encoding="utf-8",
            )
            return ConsensusSource(source, product_path, metadata_path)
        return None

    monkeypatch.setattr(prep, "_run_source_product", fake_run_source_product)

    _, rows, outputs = prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        sources_value="hrrr_recent,rap_recent",
        limit=1,
    )

    assert [row["status"] for row in rows] == ["available", "unavailable"]
    assert rows[0]["source_mode"] == "real"
    assert rows[0]["source_model"] == "hrrr"
    assert rows[0]["missing_requested_leads"] == "60,72"
    assert rows[1]["reason"] == "source_run_failed_or_unavailable"
    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert payload["case_summary"][0]["available_sources"] == ["hrrr_recent"]
    assert payload["case_summary"][0]["unavailable_sources"] == [
        {"source": "rap_recent", "reason": "source_run_failed_or_unavailable"}
    ]
    assert payload["case_summary"][0]["ready_for_scoring"] is True
