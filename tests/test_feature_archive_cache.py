import json

import pandas as pd

from severewx.archive.coverage import archive_coverage_summary
from severewx.archive.feature_cache import (
    build_cached_feature_archive_for_cycle,
    cached_feature_archive_metadata_path,
    cached_feature_archive_path,
)
from severewx.config import load_settings
from severewx.ingest.storage import historical_feature_path, historical_metadata_path, staged_raw_archive_dir, write_json_metadata
from severewx.models.dataset import load_training_table
from severewx.utils.paths import build_paths


def _archived_frame(source_kind: str = "local_staged_gfs", archive_status: str = "local_real") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-04-09"],
            "init_date": ["2026-04-09"],
            "init_cycle": ["00"],
            "lead_day": [1],
            "lat": [35.0],
            "lon": [-97.0],
            "region": ["south_plains"],
            "forecast_source_kind": [source_kind],
            "archive_source": ["historical_feature_archive"],
            "archive_chunk_status": [archive_status],
            "archive_data_kind": [archive_status],
            "cape": [1500.0],
            "cin": [-50.0],
            "t2m": [295.0],
            "td2m": [289.0],
            "mslp": [100900.0],
            "wind10m_speed": [12.0],
            "shear_0_6km": [28.0],
            "ll_shear_proxy": [14.0],
            "llj_speed": [20.0],
            "helicity_proxy": [180.0],
            "pwat": [30.0],
            "moisture_transport": [300.0],
            "low_lcl_support": [0.9],
            "moisture_quality": [0.7],
            "lapse_rate_700_500": [6.8],
            "forcing_proxy": [55.0],
            "forcing_instability_overlap": [0.9],
            "scp_proxy": [3.0],
            "stp_proxy": [1.8],
            "cape_shear_overlap": [1.5],
            "tornado_favored_overlap": [1.2],
            "hail_favored_overlap": [0.9],
            "wind_favored_overlap": [0.8],
            "outbreak_corridor_index": [0.7],
            "tornado_corridor_index": [0.8],
            "analog_similarity": [0.6],
            "analog_tornado_similarity": [0.7],
            "analog_hail_similarity": [0.2],
            "analog_wind_similarity": [0.2],
            "analog_tornado_rate": [0.5],
            "analog_hail_rate": [0.2],
            "analog_wind_rate": [0.2],
            "analog_outbreak_support": [0.6],
            "analog_sigtor_support": [0.5],
            "analog_mode_coherence": [0.4],
            "prior_run_delta": [100.0],
            "neighborhood_mean_any": [0.6],
            "synoptic_support": [75.0],
            "sig_tor_support": [0.8],
            "spatial_coverage": [0.55],
            "tornado": [1],
            "hail": [0],
            "wind": [0],
            "any": [1],
            "tornado_outbreak": [1],
            "hail_outbreak": [0],
            "wind_outbreak": [0],
            "any_outbreak": [1],
            "significant_tornado_support": [1],
            "category": ["significant_tornado_outbreak_day"],
        }
    )


def _write_source_archive(
    paths,
    date: str,
    cycle: str,
    archive_status: str = "local_real",
    raw_paths: list[str] | None = None,
    source_kind: str | None = None,
) -> None:
    archive_file = historical_feature_path(paths, date, cycle)
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    resolved_source_kind = source_kind or ("local_staged_gfs" if archive_status == "local_real" else "synthetic_fallback")
    _archived_frame(source_kind=resolved_source_kind, archive_status=archive_status).to_parquet(archive_file, index=False)
    staged_validation = {
        "status": "complete" if archive_status == "local_real" else "partial",
        "source_files": [
            {"path": path, "usable": True, "lead_hour": index * 6}
            for index, path in enumerate(raw_paths or [])
        ],
    }
    write_json_metadata(
        historical_metadata_path(paths, date, cycle),
        {
            "init_date": date,
            "cycle": cycle,
            "archive_status": archive_status,
            "forecast_source": resolved_source_kind,
            "forecast_source_origin": "local" if resolved_source_kind == "local_staged_gfs" else "synthetic",
            "label_linkage_status": "linked",
            "outbreak_label_status": "linked",
            "real_data": archive_status == "local_real",
            "staged_validation": staged_validation,
            "ingest_summary": {"processing_scope": {"requested_leads": [0, 6]}},
        },
    )


def test_cached_feature_archive_paths_are_deterministic(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    assert cached_feature_archive_path(paths, "2026-04-09", "00").as_posix().endswith("cached_feature_archive/2026/2026-04-09/00/training_features.parquet")
    assert cached_feature_archive_metadata_path(paths, "2026-04-09", "00").as_posix().endswith("archive_metadata/2026/2026-04-09/00_feature_cache.json")


def test_cached_feature_archive_build_is_resumable_and_writes_metadata(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    _write_source_archive(paths, "2026-04-09", "00")
    source_path = historical_feature_path(paths, "2026-04-09", "00")
    source_frame = pd.read_parquet(source_path)
    source_frame["unused_debug_column"] = 123
    source_frame.to_parquet(source_path, index=False)

    first = build_cached_feature_archive_for_cycle("2026-04-09", "00", settings, paths)
    second = build_cached_feature_archive_for_cycle("2026-04-09", "00", settings, paths)
    assert first.status == "built"
    assert second.status == "skipped"
    metadata = json.loads(cached_feature_archive_metadata_path(paths, "2026-04-09", "00").read_text(encoding="utf-8"))
    cached_frame = pd.read_parquet(cached_feature_archive_path(paths, "2026-04-09", "00"))
    assert metadata["forecast_source_origin"] == "local"
    assert metadata["cache_status"] == "cached_feature_archive_ready"
    assert metadata["row_count"] == 1
    assert "unused_debug_column" not in cached_frame.columns
    assert float(cached_frame.loc[0, "spatial_coverage"]) == 0.55


def test_cached_feature_archive_refreshes_when_source_status_changes_to_local_real(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    _write_source_archive(paths, "2026-04-09", "00", archive_status="synthetic_degraded", source_kind="synthetic_fallback")

    first = build_cached_feature_archive_for_cycle("2026-04-09", "00", settings, paths)
    first_frame = pd.read_parquet(cached_feature_archive_path(paths, "2026-04-09", "00"))
    assert first.status == "built"
    assert first_frame["forecast_source_kind"].tolist() == ["synthetic_fallback"]

    _write_source_archive(paths, "2026-04-09", "00", archive_status="local_real", source_kind="local_staged_gfs")
    second = build_cached_feature_archive_for_cycle("2026-04-09", "00", settings, paths)
    metadata = json.loads(cached_feature_archive_metadata_path(paths, "2026-04-09", "00").read_text(encoding="utf-8"))
    second_frame = pd.read_parquet(cached_feature_archive_path(paths, "2026-04-09", "00"))

    assert second.status == "built"
    assert metadata["source_archive_status"] == "local_real"
    assert metadata["forecast_source"] == "local_staged_gfs"
    assert metadata["forecast_source_origin"] == "local"
    assert metadata["real_data"] is True
    assert second_frame["forecast_source_kind"].tolist() == ["local_staged_gfs"]


def test_training_prefers_cached_feature_archive_when_present(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_overall"] = 1
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_hazard"] = {"tornado": 1, "any": 1}
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_lead_day"] = {1: 1}
    settings.raw["models"]["training_guardrails"]["maximum_synthetic_fraction"] = 1.0
    paths = build_paths(settings)
    _write_source_archive(paths, "2026-04-09", "00")
    build_cached_feature_archive_for_cycle("2026-04-09", "00", settings, paths)

    outbreak_table = pd.DataFrame(
        {
            "date": ["2026-04-09"],
            "tornado_outbreak": [1],
            "hail_outbreak": [0],
            "wind_outbreak": [0],
            "any_outbreak": [1],
            "significant_tornado_support": [1],
            "regional_cluster_count": [12],
            "report_count": [18],
            "category": ["significant_tornado_outbreak_day"],
            "report_source_name": ["spc_csv"],
            "report_source_is_real": [True],
            "report_ingest_timestamp": ["2026-04-13T00:00:00Z"],
        }
    )
    bundle = load_training_table(paths, settings, requested_hazards=["tornado", "any", "outbreak"], outbreak_table=outbreak_table)
    assert bundle.metadata["source_preference"] == "cached_feature_archive"
    assert bundle.metadata["training_cache_used"] is True
    assert bundle.metadata["guardrails_passed"] is True
    assert bundle.metadata["training_allowed_without_override"] is True


def test_archive_coverage_reports_cached_feature_archive(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    _write_source_archive(paths, "2026-04-09", "00")
    build_cached_feature_archive_for_cycle("2026-04-09", "00", settings, paths)
    summary = archive_coverage_summary(paths, settings=settings)
    assert summary["cached_feature_archive_chunk_count"] == 1
    assert summary["cached_feature_archive_row_count"] == 1
    assert summary["cached_feature_archive_real_row_count"] == 1
    assert summary["cached_feature_archive_real_fraction"] == 1.0


def test_cached_feature_archive_retains_raw_when_source_not_verified(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw.setdefault("archive", {}).setdefault("raw_file_handling", {})["mode"] = "delete_raw_after_verified_cache"
    paths = build_paths(settings)
    raw_file = paths.raw / "staged_gfs" / "2026-04-09" / "00" / "gfs.t00z.pgrb2.0p25.f000.nc"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("placeholder", encoding="utf-8")
    _write_source_archive(paths, "2026-04-09", "00", archive_status="partial_real", raw_paths=[str(raw_file)])

    build_cached_feature_archive_for_cycle("2026-04-09", "00", settings, paths)
    metadata = json.loads(cached_feature_archive_metadata_path(paths, "2026-04-09", "00").read_text(encoding="utf-8"))
    assert raw_file.exists()
    assert metadata["raw_lifecycle"]["action"] == "retained"
    assert metadata["raw_lifecycle"]["eligible"] is False


def test_cached_feature_archive_can_move_raw_after_verified_success(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw.setdefault("archive", {}).setdefault("raw_file_handling", {})["mode"] = "move_raw_to_archive"
    paths = build_paths(settings)
    raw_file = paths.raw / "staged_gfs" / "2026-04-09" / "00" / "gfs.t00z.pgrb2.0p25.f000.nc"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("placeholder", encoding="utf-8")
    _write_source_archive(paths, "2026-04-09", "00", archive_status="local_real", raw_paths=[str(raw_file)])

    build_cached_feature_archive_for_cycle("2026-04-09", "00", settings, paths)
    destination = staged_raw_archive_dir(paths, "2026-04-09", "00") / raw_file.name
    metadata = json.loads(cached_feature_archive_metadata_path(paths, "2026-04-09", "00").read_text(encoding="utf-8"))
    assert not raw_file.exists()
    assert destination.exists()
    assert metadata["raw_lifecycle"]["action"] == "moved"
    assert metadata["raw_lifecycle"]["raw_files_moved"] == 1


def test_cached_feature_archive_can_delete_raw_after_verified_success(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw.setdefault("archive", {}).setdefault("raw_file_handling", {})["mode"] = "delete_raw_after_verified_cache"
    paths = build_paths(settings)
    raw_file = paths.raw / "staged_gfs" / "2026-04-09" / "00" / "gfs.t00z.pgrb2.0p25.f000.nc"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("placeholder", encoding="utf-8")
    _write_source_archive(paths, "2026-04-09", "00", archive_status="local_real", raw_paths=[str(raw_file)])

    build_cached_feature_archive_for_cycle("2026-04-09", "00", settings, paths)
    metadata = json.loads(cached_feature_archive_metadata_path(paths, "2026-04-09", "00").read_text(encoding="utf-8"))
    assert not raw_file.exists()
    assert metadata["raw_lifecycle"]["action"] == "deleted"
    assert metadata["raw_lifecycle"]["raw_files_deleted"] == 1
