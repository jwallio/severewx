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


def test_v2_backtest_manifest_has_locked_folds_and_case_counts() -> None:
    manifest = load_backtest_manifest("backtests/tornado_environment_consensus_v2.json")

    assert manifest.manifest_id == "tornado_environment_consensus_v2"
    assert manifest.target_product["verification_target"] == "tornado_reports_25mi"
    assert manifest.target_product["spc_probability_bins"] == [0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6]
    assert manifest.target_product["default_sources"] == ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent", "ecmwf_recent"]
    assert manifest.target_product["optional_sources"] == ["open_meteo_recent"]
    assert "noaa_ecmwf_openmeteo" in manifest.target_product["source_comparison_groups"]
    assert manifest.target_product["synthetic_fallback_allowed"] is False
    assert len(manifest.cases) == 250
    assert {"train", "tune", "test"} <= {case.fold for case in manifest.cases}
    assert sum(1 for case in manifest.cases if "tornado_relevant" in case.tags) >= 100
    assert sum(1 for case in manifest.cases if "hard_negative" in case.tags) >= 100
    assert sum(1 for case in manifest.cases if "null" in case.tags) >= 50
    assert all(case.region != "unknown" and case.season != "unknown" and case.regime != "unknown" for case in manifest.cases)
    assert len({(case.date, case.cycle) for case in manifest.cases}) == len(manifest.cases)


def test_v2_pilot_manifest_targets_recent_hydration_cases() -> None:
    manifest = load_backtest_manifest("backtests/tornado_environment_consensus_v2_pilot.json")

    assert manifest.manifest_id == "tornado_environment_consensus_v2_pilot"
    assert manifest.target_product["verification_target"] == "tornado_reports_25mi"
    assert manifest.target_product["synthetic_fallback_allowed"] is False
    assert manifest.target_product["default_sources"] == ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent", "ecmwf_recent"]
    assert manifest.target_product["optional_sources"] == ["open_meteo_recent"]
    assert len(manifest.cases) == 15
    assert {"train", "tune", "test"} <= {case.fold for case in manifest.cases}
    assert min(case.date for case in manifest.cases) >= "2024-01-01"


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
    assert outputs[3].exists()
    assert outputs[4].exists()
    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["case_count"] == len(manifest.cases)
    assert len(payload["case_summary"]) == 2
    assert payload["case_summary"][0]["ready_for_scoring"] is False
    summary = json.loads(outputs[3].read_text(encoding="utf-8"))
    assert summary["readiness_status"] == "planned_only"
    assert "source_ablation" in summary
    assert summary["source_comparison"] == {}
    assert summary["metric_summary"]["spc_probability_bins"] == [0.02, 0.05, 0.1, 0.15, 0.3, 0.45, 0.6]


def test_prepare_backtest_cases_supports_manifest_batches(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    manifest = load_backtest_manifest("backtests/tornado_environment_consensus_v2.json")

    _, rows, outputs = prep.prepare_backtest_cases(
        manifest_path="backtests/tornado_environment_consensus_v2.json",
        output_dir=tmp_path / "backtests",
        dry_run=True,
        batch_size=3,
        batch_number=2,
    )

    selected_case_ids = list(dict.fromkeys(row["case_id"] for row in rows))
    assert selected_case_ids == [case.case_id for case in manifest.cases[3:6]]
    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert payload["selected_case_count"] == 3
    assert payload["run_options"]["batch_size"] == 3
    assert payload["run_options"]["batch_number"] == 2
    summary = json.loads(outputs[3].read_text(encoding="utf-8"))
    assert summary["selected_cases"] == selected_case_ids
    assert "noaa_only" in summary["source_comparison"]
    assert "noaa_ecmwf_openmeteo" in summary["source_comparison"]
    assert summary["source_comparison"]["noaa_only"]["score_status"] == "availability_only"
    assert "max_day1_3_tornado_concern" not in summary["source_comparison"]["noaa_only"]["cases"][0]


def test_prepare_backtest_cases_supports_manifest_source_groups(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)

    _, rows, outputs = prep.prepare_backtest_cases(
        manifest_path="backtests/tornado_environment_consensus_v2_pilot.json",
        output_dir=tmp_path / "backtests",
        sources_value="group:noaa_ecmwf",
        dry_run=True,
        limit=1,
    )

    assert [row["source"] for row in rows] == ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent", "ecmwf_recent"]
    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert payload["run_options"]["sources"] == ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent", "ecmwf_recent"]


def test_prepare_backtest_cases_rejects_unknown_manifest_source_group(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)

    try:
        prep.prepare_backtest_cases(
            manifest_path="backtests/tornado_environment_consensus_v2_pilot.json",
            output_dir=tmp_path / "backtests",
            sources_value="group:not_real",
            dry_run=True,
            limit=1,
        )
    except ValueError as exc:
        assert "unknown source comparison group: not_real" in str(exc)
        assert "noaa_ecmwf" in str(exc)
    else:
        raise AssertionError("unknown source comparison group should fail")


def test_write_hydration_status_reports_available_missing_and_partial(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    paths = build_paths(settings)
    manifest = load_backtest_manifest(prep.DEFAULT_MANIFEST)
    case = manifest.cases[0]
    product_path = prep.source_product_path(paths.outputs, case.date, case.cycle, "hrrr_recent")
    metadata_path = prep.source_metadata_path(paths.outputs, case.date, case.cycle, "hrrr_recent")
    product_path.parent.mkdir(parents=True, exist_ok=True)
    product_path.write_text("fake-netcdf", encoding="utf-8")
    metadata_path.write_text(json.dumps({"ingest_summary": {"source_mode": "real", "source_model": "hrrr"}}), encoding="utf-8")
    partial_product_path = prep.source_product_path(paths.outputs, case.date, case.cycle, "rap_recent")
    partial_product_path.write_text("fake-netcdf", encoding="utf-8")

    json_path, csv_path = prep.write_hydration_status(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "status",
        sources_value="hrrr_recent,rap_recent,nam_recent",
        limit=1,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    by_source = {row["source"]: row for row in payload["rows"]}
    assert csv_path.exists()
    assert payload["available"] == 1
    assert payload["partial"] == 1
    assert payload["missing"] == 1
    assert by_source["hrrr_recent"]["status"] == "available"
    assert by_source["rap_recent"]["status"] == "partial"
    assert by_source["nam_recent"]["status"] == "missing"


def test_prepare_backtest_cases_coverage_only_disables_expensive_steps(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    paths = build_paths(settings)

    def fake_run_source_product(date, cycle, source, settings, *, skip_render):
        product_path = prep.source_product_path(paths.outputs, date, cycle, source)
        metadata_path = prep.source_metadata_path(paths.outputs, date, cycle, source)
        product_path.parent.mkdir(parents=True, exist_ok=True)
        product_path.write_text("fake-netcdf", encoding="utf-8")
        metadata_path.write_text(json.dumps({"ingest_summary": {"source": source, "source_mode": "real"}}), encoding="utf-8")
        return ConsensusSource(source, product_path, metadata_path)

    def fail_expensive_step(*args, **kwargs):
        raise AssertionError("coverage-only should not run expensive steps")

    monkeypatch.setattr(prep, "_run_source_product", fake_run_source_product)
    monkeypatch.setattr(prep, "build_forecast_consensus", fail_expensive_step)
    monkeypatch.setattr(prep, "build_product_bundle", fail_expensive_step)
    monkeypatch.setattr(prep, "_verify_case", fail_expensive_step)

    _, rows, outputs = prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        sources_value="hrrr_recent",
        limit=1,
        coverage_only=True,
        build_consensus=True,
        build_products=True,
        verify=True,
        score=True,
    )

    assert len(rows) == 1
    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert payload["products"] == []
    assert payload["scores"] == []
    assert payload["run_options"]["coverage_only"] is True
    assert payload["run_options"]["build_consensus"] is False


def test_prepare_backtest_cases_allows_lead_hour_override(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    seen_leads = None

    def fake_run_source_product(date, cycle, source, settings, *, skip_render):
        nonlocal seen_leads
        seen_leads = settings.get("ingest.leads")
        paths = build_paths(settings)
        product_path = prep.source_product_path(paths.outputs, date, cycle, source)
        metadata_path = prep.source_metadata_path(paths.outputs, date, cycle, source)
        product_path.parent.mkdir(parents=True, exist_ok=True)
        product_path.write_text("fake-netcdf", encoding="utf-8")
        metadata_path.write_text(json.dumps({"ingest_summary": {"source": source, "source_mode": "real"}}), encoding="utf-8")
        return ConsensusSource(source, product_path, metadata_path)

    monkeypatch.setattr(prep, "_run_source_product", fake_run_source_product)

    _, _, outputs = prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        sources_value="hrrr_recent",
        limit=1,
        coverage_only=True,
        lead_hours=[0, 24, 48],
    )

    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert seen_leads == [0, 24, 48]
    assert payload["run_options"]["lead_hours"] == [0, 24, 48]


def test_prepare_backtest_cases_checkpoints_after_each_case(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    paths = build_paths(settings)
    calls = 0

    def fake_run_source_product(date, cycle, source, settings, *, skip_render):
        nonlocal calls
        calls += 1
        product_path = prep.source_product_path(paths.outputs, date, cycle, source)
        metadata_path = prep.source_metadata_path(paths.outputs, date, cycle, source)
        product_path.parent.mkdir(parents=True, exist_ok=True)
        product_path.write_text("fake-netcdf", encoding="utf-8")
        metadata_path.write_text(json.dumps({"ingest_summary": {"source": source, "source_mode": "real"}}), encoding="utf-8")
        if calls == 2:
            checkpoint = tmp_path / "backtests" / "source_availability.json"
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            assert payload["selected_case_count"] == 2
            assert len(payload["case_summary"]) == 1
        return ConsensusSource(source, product_path, metadata_path)

    monkeypatch.setattr(prep, "_run_source_product", fake_run_source_product)

    prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        sources_value="hrrr_recent",
        limit=2,
        coverage_only=True,
    )

    final_payload = json.loads((tmp_path / "backtests" / "source_availability.json").read_text(encoding="utf-8"))
    assert len(final_payload["case_summary"]) == 2


def test_prepare_backtest_cases_checkpoints_after_each_source(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    paths = build_paths(settings)
    calls = 0

    def fake_run_source_product(date, cycle, source, settings, *, skip_render):
        nonlocal calls
        calls += 1
        product_path = prep.source_product_path(paths.outputs, date, cycle, source)
        metadata_path = prep.source_metadata_path(paths.outputs, date, cycle, source)
        product_path.parent.mkdir(parents=True, exist_ok=True)
        product_path.write_text("fake-netcdf", encoding="utf-8")
        metadata_path.write_text(json.dumps({"ingest_summary": {"source": source, "source_mode": "real"}}), encoding="utf-8")
        if calls == 2:
            checkpoint = tmp_path / "backtests" / "source_availability.json"
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            assert [row["source"] for row in payload["rows"]] == ["hrrr_recent"]
        return ConsensusSource(source, product_path, metadata_path)

    monkeypatch.setattr(prep, "_run_source_product", fake_run_source_product)

    prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        sources_value="hrrr_recent,rap_recent",
        limit=1,
        coverage_only=True,
    )

    final_payload = json.loads((tmp_path / "backtests" / "source_availability.json").read_text(encoding="utf-8"))
    assert [row["source"] for row in final_payload["rows"]] == ["hrrr_recent", "rap_recent"]


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


def test_prepare_backtest_cases_resume_reuses_existing_source_artifacts(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    paths = build_paths(settings)
    manifest = load_backtest_manifest(prep.DEFAULT_MANIFEST)
    case = manifest.cases[0]
    product_path = prep.source_product_path(paths.outputs, case.date, case.cycle, "hrrr_recent")
    metadata_path = prep.source_metadata_path(paths.outputs, case.date, case.cycle, "hrrr_recent")
    product_path.parent.mkdir(parents=True, exist_ok=True)
    product_path.write_text("fake-netcdf", encoding="utf-8")
    metadata_path.write_text(json.dumps({"ingest_summary": {"source_mode": "real", "source_model": "hrrr"}}), encoding="utf-8")

    def fail_run_source_product(*args, **kwargs):
        raise AssertionError("resume should not rerun existing source artifacts")

    monkeypatch.setattr(prep, "_run_source_product", fail_run_source_product)

    _, rows, outputs = prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        sources_value="hrrr_recent",
        limit=1,
        resume=True,
    )

    assert rows[0]["status"] == "available"
    assert rows[0]["reason"] == "reused_existing"
    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert payload["run_options"]["resume"] is True


def test_prepare_backtest_cases_resume_rebuilds_consensus_when_sources_change(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    paths = build_paths(settings)
    manifest = load_backtest_manifest(prep.DEFAULT_MANIFEST)
    case = manifest.cases[0]
    artifact_paths = prep._case_artifact_paths(paths, tmp_path / "backtests", manifest, case, ["hrrr_recent"])
    stale_consensus_path = prep.consensus_product_path(artifact_paths.outputs, case.date, case.cycle)
    stale_metadata_path = prep.consensus_metadata_path(artifact_paths.outputs, case.date, case.cycle)
    stale_consensus_path.parent.mkdir(parents=True, exist_ok=True)
    stale_consensus_path.write_text("stale-consensus", encoding="utf-8")
    stale_metadata_path.write_text(json.dumps({"included_sources": ["hrrr_recent", "rap_recent"]}), encoding="utf-8")

    def fake_run_source_product(date, cycle, source, settings, *, skip_render):
        product_path = prep.source_product_path(paths.outputs, date, cycle, source)
        metadata_path = prep.source_metadata_path(paths.outputs, date, cycle, source)
        product_path.parent.mkdir(parents=True, exist_ok=True)
        product_path.write_text("fake-netcdf", encoding="utf-8")
        metadata_path.write_text(json.dumps({"ingest_summary": {"source": source, "source_mode": "real"}}), encoding="utf-8")
        return ConsensusSource(source, product_path, metadata_path)

    rebuilt = False

    def fake_build_consensus(*, date, cycle, sources, output_path, metadata_path, field_name, unavailable_sources):
        nonlocal rebuilt
        rebuilt = True
        output_path.write_text("rebuilt-consensus", encoding="utf-8")
        metadata_path.write_text(json.dumps({"included_sources": [source.name for source in sources]}), encoding="utf-8")
        return output_path, metadata_path

    monkeypatch.setattr(prep, "_run_source_product", fake_run_source_product)
    monkeypatch.setattr(prep, "build_forecast_consensus", fake_build_consensus)

    _, rows, _ = prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        sources_value="hrrr_recent",
        limit=1,
        resume=True,
        build_consensus=True,
    )

    assert rebuilt is True
    consensus_rows = [row for row in rows if row["source"] == "forecast_consensus"]
    assert consensus_rows[0]["reason"] == ""
    assert stale_consensus_path.read_text(encoding="utf-8") == "rebuilt-consensus"


def test_prepare_backtest_cases_can_build_consensus_products_and_scores(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    paths = build_paths(settings)

    def fake_run_source_product(date, cycle, source, settings, *, skip_render):
        product_path = prep.source_product_path(paths.outputs, date, cycle, source)
        metadata_path = prep.source_metadata_path(paths.outputs, date, cycle, source)
        product_path.parent.mkdir(parents=True, exist_ok=True)
        product_path.write_text("fake-netcdf", encoding="utf-8")
        metadata_path.write_text(json.dumps({"ingest_summary": {"source": source, "source_mode": "real"}}), encoding="utf-8")
        return ConsensusSource(source, product_path, metadata_path)

    def fake_build_consensus(*, date, cycle, sources, output_path, metadata_path, field_name, unavailable_sources):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("fake-consensus", encoding="utf-8")
        metadata_path.write_text(json.dumps({"included_sources": [source.name for source in sources]}), encoding="utf-8")
        return output_path, metadata_path

    def fake_build_product_bundle(**kwargs):
        valid_date = kwargs["valid_date"]
        outdir = kwargs["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)
        metadata_path = outdir / f"{valid_date}.json"
        image_path = outdir / f"{valid_date}.png"
        metadata_path.write_text("{}", encoding="utf-8")
        image_path.write_text("png", encoding="utf-8")
        return {
            "metadata_path": str(metadata_path),
            "main_image_path": str(image_path),
            "publication_status": "public_candidate",
            "public_ready": True,
            "max_tornado_concern_prob": 0.25,
        }

    def fake_verify(paths, case):
        output = paths.verification / "backtests" / f"{case.case_id}_verification.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"per_day": [{"observed_tornado_outbreak": 1, "observed_any_outbreak": 1, "tornado_brier": 0.1}]}),
            encoding="utf-8",
        )
        return output, ""

    monkeypatch.setattr(prep, "_run_source_product", fake_run_source_product)
    monkeypatch.setattr(prep, "build_forecast_consensus", fake_build_consensus)
    monkeypatch.setattr(prep, "build_product_bundle", fake_build_product_bundle)
    monkeypatch.setattr(prep, "_verify_case", fake_verify)

    _, rows, outputs = prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        sources_value="hrrr_recent",
        limit=1,
        build_consensus=True,
        build_products=True,
        verify=True,
        score=True,
    )

    assert rows[-1]["source"] == "forecast_consensus"
    assert rows[-1]["status"] == "available"
    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert len(payload["products"]) == 3
    assert payload["products"][0]["public_ready"] is True
    assert payload["scores"][0]["verification_status"] == "available"
    assert payload["scores"][0]["observed_tornado_outbreak_any"] is True
    assert payload["scores"][0]["max_day1_3_tornado_concern"] == 0.25
    summary = json.loads(outputs[3].read_text(encoding="utf-8"))
    assert summary["readiness_status"] == "public_ready_candidate"
    assert summary["metric_summary"]["thresholds"][0]["threshold"] == 0.02
    assert summary["metric_summary"]["thresholds"][-1]["false_alarm_rate"] is None
    assert summary["learned_weight_report"]["status"] == "candidate_only"
    assert summary["spc_comparison"]["status"] == "not_configured"


def test_prepare_backtest_cases_records_product_failures_without_aborting(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    monkeypatch.setattr(prep, "load_settings", lambda: settings)
    paths = build_paths(settings)

    def fake_run_source_product(date, cycle, source, settings, *, skip_render):
        product_path = prep.source_product_path(paths.outputs, date, cycle, source)
        metadata_path = prep.source_metadata_path(paths.outputs, date, cycle, source)
        product_path.parent.mkdir(parents=True, exist_ok=True)
        product_path.write_text("fake-netcdf", encoding="utf-8")
        metadata_path.write_text(json.dumps({"ingest_summary": {"source": source, "source_mode": "real"}}), encoding="utf-8")
        return ConsensusSource(source, product_path, metadata_path)

    def fake_build_consensus(*, date, cycle, sources, output_path, metadata_path, field_name, unavailable_sources):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("fake-consensus", encoding="utf-8")
        metadata_path.write_text(json.dumps({"included_sources": [source.name for source in sources]}), encoding="utf-8")
        return output_path, metadata_path

    product_calls = 0

    def fake_build_product_bundle(**kwargs):
        nonlocal product_calls
        product_calls += 1
        if product_calls == 2:
            raise ValueError("missing valid time")
        outdir = kwargs["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)
        metadata_path = outdir / f"{kwargs['valid_date']}.json"
        image_path = outdir / f"{kwargs['valid_date']}.png"
        metadata_path.write_text("{}", encoding="utf-8")
        image_path.write_text("png", encoding="utf-8")
        return {
            "metadata_path": str(metadata_path),
            "main_image_path": str(image_path),
            "publication_status": "public_candidate",
            "public_ready": True,
            "max_tornado_concern_prob": 0.15,
        }

    monkeypatch.setattr(prep, "_run_source_product", fake_run_source_product)
    monkeypatch.setattr(prep, "build_forecast_consensus", fake_build_consensus)
    monkeypatch.setattr(prep, "build_product_bundle", fake_build_product_bundle)
    monkeypatch.setattr(prep, "_verify_case", lambda paths, case: (None, "missing_labels"))

    _, _, outputs = prep.prepare_backtest_cases(
        manifest_path=prep.DEFAULT_MANIFEST,
        output_dir=tmp_path / "backtests",
        sources_value="hrrr_recent",
        limit=1,
        build_consensus=True,
        build_products=True,
        score=True,
    )

    payload = json.loads(outputs[1].read_text(encoding="utf-8"))
    assert len(payload["products"]) == 3
    assert [product["product_status"] for product in payload["products"]] == ["available", "failed", "available"]
    assert payload["products"][1]["publication_status"] == "failed_guardrails"
    assert "missing valid time" in payload["products"][1]["reason"]
