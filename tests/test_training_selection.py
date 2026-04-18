import pandas as pd

from severewx.archive.feature_cache import cached_feature_archive_path
from severewx.config import load_settings
from severewx.ingest.storage import historical_feature_path
from severewx.models.dataset import load_training_table
from severewx.utils.paths import build_paths


def _real_outbreak_table(date: str, *, positive: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date],
            "tornado_outbreak": [1 if positive else 0],
            "hail_outbreak": [0],
            "wind_outbreak": [0],
            "any_outbreak": [1 if positive else 0],
            "significant_tornado_support": [1 if positive else 0],
            "regional_cluster_count": [12 if positive else 4],
            "report_count": [18],
            "category": ["significant_tornado_outbreak_day" if positive else "non_outbreak_severe_day"],
            "report_source_name": ["spc_csv"],
            "report_source_is_real": [True],
            "report_ingest_timestamp": ["2026-04-13T00:00:00Z"],
        }
    )


def _synthetic_outbreak_table(date: str) -> pd.DataFrame:
    frame = _real_outbreak_table(date, positive=True)
    frame["report_source_name"] = "synthetic_fallback"
    frame["report_source_is_real"] = False
    return frame


def test_training_prefers_feature_archive_when_present(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_overall"] = 1
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_hazard"] = {"tornado": 1, "any": 1}
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_lead_day"] = {1: 1}
    settings.raw["models"]["training_guardrails"]["maximum_synthetic_fraction"] = 1.0
    paths = build_paths(settings)
    archive_file = historical_feature_path(paths, "2026-04-09", "00")
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "date": ["2026-04-09"],
            "init_date": ["2026-04-09"],
            "init_cycle": ["00"],
            "lead_day": [1],
            "lat": [35.0],
            "lon": [-97.0],
            "region": ["south_plains"],
            "forecast_source_kind": ["local_staged_gfs"],
            "archive_source": ["historical_feature_archive"],
            "archive_chunk_status": ["local_real"],
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
            "spatial_coverage": [0.5],
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
    frame.to_parquet(archive_file, index=False)
    bundle = load_training_table(
        paths,
        settings,
        requested_hazards=["tornado", "any", "outbreak"],
        outbreak_table=_real_outbreak_table("2026-04-09", positive=True),
    )
    assert bundle.metadata["source_preference"] == "historical_feature_archive"
    assert len(bundle.frame) >= 1
    assert "archive_preferred_satisfied" in bundle.metadata
    assert "coverage_by_hazard_and_lead_day" in bundle.metadata
    assert bundle.metadata["guardrails_passed"] is True
    assert bundle.metadata["real_row_counts_by_source_kind"]["local_staged_gfs"] == 1
    assert bundle.metadata["archive_row_counts_by_status"]["local_real"] == 1


def test_training_guardrails_mark_archive_as_degraded_when_real_coverage_is_too_low(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    archive_file = historical_feature_path(paths, "2026-04-09", "00")
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2026-04-09"],
            "init_date": ["2026-04-09"],
            "init_cycle": ["00"],
            "lead_day": [1],
            "lat": [35.0],
            "lon": [-97.0],
            "region": ["south_plains"],
            "forecast_source_kind": ["nomads"],
            "archive_source": ["historical_feature_archive"],
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
            "spatial_coverage": [0.5],
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
    ).to_parquet(archive_file, index=False)
    bundle = load_training_table(paths, settings, requested_hazards=["tornado", "any"])
    assert bundle.metadata["source_preference"] == "historical_feature_archive_degraded"
    assert bundle.metadata["guardrails_passed"] is False
    assert bundle.metadata["training_allowed_without_override"] is False
    assert bundle.metadata["guardrail_failures"]


def test_training_marks_archive_degraded_when_outbreak_targets_have_zero_positives(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_overall"] = 1
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_hazard"] = {"tornado": 1, "any": 1}
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_lead_day"] = {1: 1}
    settings.raw["models"]["training_guardrails"]["maximum_synthetic_fraction"] = 1.0
    paths = build_paths(settings)
    archive_file = historical_feature_path(paths, "2026-04-09", "00")
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2026-04-09"],
            "init_date": ["2026-04-09"],
            "init_cycle": ["00"],
            "lead_day": [1],
            "lat": [35.0],
            "lon": [-97.0],
            "region": ["south_plains"],
            "forecast_source_kind": ["local_staged_gfs"],
            "archive_source": ["historical_feature_archive"],
            "archive_chunk_status": ["local_real"],
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
            "spatial_coverage": [0.5],
            "tornado": [1],
            "hail": [0],
            "wind": [0],
            "any": [1],
            "tornado_outbreak": [0],
            "hail_outbreak": [0],
            "wind_outbreak": [0],
            "any_outbreak": [0],
            "significant_tornado_support": [0],
            "category": ["non_outbreak_severe_day"],
        }
    ).to_parquet(archive_file, index=False)
    bundle = load_training_table(
        paths,
        settings,
        requested_hazards=["tornado", "any", "outbreak"],
        outbreak_table=_real_outbreak_table("2026-04-09", positive=False),
    )
    assert bundle.metadata["guardrails_passed"] is True
    assert bundle.metadata["training_allowed_without_override"] is False
    assert bundle.metadata["source_preference"] == "historical_feature_archive_degraded"
    assert "zero positives across strict outbreak targets" in bundle.metadata["degraded_mode_reason"]


def test_training_rejects_synthetic_outbreak_labels_for_full_strict_mode(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_overall"] = 1
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_hazard"] = {"tornado": 1, "any": 1}
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_lead_day"] = {1: 1}
    settings.raw["models"]["training_guardrails"]["maximum_synthetic_fraction"] = 1.0
    paths = build_paths(settings)
    archive_file = historical_feature_path(paths, "2026-04-09", "00")
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2026-04-09"],
            "init_date": ["2026-04-09"],
            "init_cycle": ["00"],
            "lead_day": [1],
            "lat": [35.0],
            "lon": [-97.0],
            "region": ["south_plains"],
            "forecast_source_kind": ["local_staged_gfs"],
            "archive_source": ["historical_feature_archive"],
            "archive_chunk_status": ["local_real"],
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
    ).to_parquet(archive_file, index=False)
    bundle = load_training_table(
        paths,
        settings,
        requested_hazards=["tornado", "any", "outbreak"],
        outbreak_table=_synthetic_outbreak_table("2026-04-09"),
    )
    assert bundle.metadata["guardrails_passed"] is True
    assert bundle.metadata["training_allowed_without_override"] is False
    assert bundle.metadata["source_preference"] == "historical_feature_archive_degraded"
    assert "synthetic/fallback SPC reports" in bundle.metadata["degraded_mode_reason"]


def test_training_overlays_current_outbreak_labels_onto_stale_cached_archive(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_overall"] = 1
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_hazard"] = {"tornado": 1, "any": 1}
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_lead_day"] = {1: 1}
    settings.raw["models"]["training_guardrails"]["maximum_synthetic_fraction"] = 1.0
    paths = build_paths(settings)
    cache_file = cached_feature_archive_path(paths, "2026-04-09", "00")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2026-04-09"],
            "init_date": ["2026-04-09"],
            "init_cycle": ["00"],
            "lead_day": [1],
            "lat": [35.0],
            "lon": [-97.0],
            "region": ["south_plains"],
            "forecast_source_kind": ["local_staged_gfs"],
            "archive_source": ["historical_feature_archive"],
            "archive_chunk_status": ["local_real"],
            "tornado": [1],
            "hail": [0],
            "wind": [0],
            "any": [1],
            "tornado_outbreak": [0],
            "hail_outbreak": [0],
            "wind_outbreak": [0],
            "any_outbreak": [0],
            "significant_tornado_support": [0],
            "category": ["non_outbreak_severe_day"],
        }
    ).to_parquet(cache_file, index=False)
    bundle = load_training_table(
        paths,
        settings,
        requested_hazards=["tornado", "any", "outbreak"],
        outbreak_table=_real_outbreak_table("2026-04-09", positive=True),
    )
    assert bundle.metadata["source_preference"] == "cached_feature_archive"
    assert bundle.metadata["training_allowed_without_override"] is True
    assert float(bundle.frame["tornado_outbreak"].sum()) > 0.0
    assert float(bundle.frame["any_outbreak"].sum()) > 0.0
    assert float(bundle.frame["significant_tornado_support"].sum()) > 0.0


def test_hazard_only_training_ignores_outbreak_label_gating(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_overall"] = 1
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_hazard"] = {"tornado": 1, "any": 1}
    settings.raw["models"]["training_guardrails"]["minimum_real_rows_by_lead_day"] = {1: 1}
    settings.raw["models"]["training_guardrails"]["maximum_synthetic_fraction"] = 1.0
    paths = build_paths(settings)
    archive_file = historical_feature_path(paths, "2026-04-09", "00")
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2026-04-09"],
            "init_date": ["2026-04-09"],
            "init_cycle": ["00"],
            "lead_day": [1],
            "lat": [35.0],
            "lon": [-97.0],
            "region": ["south_plains"],
            "forecast_source_kind": ["local_staged_gfs"],
            "archive_source": ["historical_feature_archive"],
            "archive_chunk_status": ["local_real"],
            "tornado": [1],
            "hail": [0],
            "wind": [0],
            "any": [1],
        }
    ).to_parquet(archive_file, index=False)
    bundle = load_training_table(paths, settings, requested_hazards=["tornado", "any"])
    assert bundle.metadata["source_preference"] == "historical_feature_archive"
    assert bundle.metadata["training_allowed_without_override"] is True
