import json

import pandas as pd
import xarray as xr

from severewx.archive.history import build_historical_archive_for_cycle
from severewx.config import load_settings
from severewx.ingest.nomads import SyntheticForecastSource
from severewx.ingest.storage import historical_feature_path, historical_metadata_path, save_forecast_dataset, write_json_metadata
from severewx.utils.paths import build_paths


def test_historical_archive_paths_are_deterministic(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    assert historical_feature_path(paths, "2026-04-09", "00").as_posix().endswith("feature_archive/2026/2026-04-09/00/features.parquet")
    assert historical_metadata_path(paths, "2026-04-09", "00").as_posix().endswith("archive_metadata/2026/2026-04-09/00.json")


def test_historical_archive_build_is_resumable_and_writes_metadata(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["ingest"]["source"] = "synthetic"
    paths = build_paths(settings)
    label_cube = xr.Dataset(
        {
            "tornado": (("date", "lat", "lon"), [[[0]]]),
            "hail": (("date", "lat", "lon"), [[[0]]]),
            "wind": (("date", "lat", "lon"), [[[0]]]),
            "any": (("date", "lat", "lon"), [[[0]]]),
        },
        coords={"date": [pd.Timestamp("2026-04-09")], "lat": [35.0], "lon": [-97.0]},
    )
    outbreak_table = pd.DataFrame({"date": ["2026-04-09"], "category": ["null_day"], "any_outbreak": [0], "tornado_outbreak": [0], "significant_tornado_support": [0]})

    first = build_historical_archive_for_cycle("2026-04-09", "00", settings, paths, label_cube=label_cube, outbreak_table=outbreak_table)
    second = build_historical_archive_for_cycle("2026-04-09", "00", settings, paths, label_cube=label_cube, outbreak_table=outbreak_table)
    third = build_historical_archive_for_cycle("2026-04-09", "00", settings, paths, label_cube=label_cube, outbreak_table=outbreak_table, force=True)
    assert first.status == "synthetic_degraded"
    assert second.status == "skipped"
    assert third.status == "synthetic_degraded"

    feature_file = historical_feature_path(paths, "2026-04-09", "00")
    metadata_file = historical_metadata_path(paths, "2026-04-09", "00")
    assert feature_file.exists()
    assert metadata_file.exists()
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert payload["init_date"] == "2026-04-09"
    assert "available_fields" in payload
    assert "label_positive_counts" in payload
    assert "label_linkage_status" in payload
    assert payload["archive_status"] == "synthetic_degraded"


def test_historical_archive_build_supports_local_staged_gfs(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["ingest"]["source"] = "local_staged_gfs"
    settings.raw["ingest"]["allow_synthetic_fallback"] = False
    settings.raw["ingest"]["leads"] = [0, 6]
    settings.raw["ingest"]["local_staged_gfs"]["file_patterns"] = [
        "{root}/data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.nc"
    ]
    paths = build_paths(settings)
    staged_root = tmp_path / "data" / "raw" / "staged_gfs" / "2026-04-09" / "00"
    staged_root.mkdir(parents=True, exist_ok=True)
    synthetic, _ = SyntheticForecastSource().fetch_cycle("2026-04-09", "00", settings, paths)
    for lead_index, lead in enumerate(settings.raw["ingest"]["leads"]):
        synthetic.isel(time=lead_index).drop_vars("time").to_netcdf(staged_root / f"gfs.t00z.pgrb2.0p25.f{int(lead):03d}.nc")

    label_cube = xr.Dataset(
        {
            "tornado": (("date", "lat", "lon"), [[[1]]]),
            "hail": (("date", "lat", "lon"), [[[0]]]),
            "wind": (("date", "lat", "lon"), [[[0]]]),
            "any": (("date", "lat", "lon"), [[[1]]]),
        },
        coords={"date": [pd.Timestamp("2026-04-09")], "lat": [20.0], "lon": [-130.0]},
    )
    outbreak_table = pd.DataFrame({"date": ["2026-04-09"], "category": ["tornado_outbreak_day"], "any_outbreak": [1], "tornado_outbreak": [1], "significant_tornado_support": [1]})

    result = build_historical_archive_for_cycle("2026-04-09", "00", settings, paths, label_cube=label_cube, outbreak_table=outbreak_table)
    metadata = json.loads(historical_metadata_path(paths, "2026-04-09", "00").read_text(encoding="utf-8"))
    assert result.status == "local_real"
    assert metadata["archive_status"] == "local_real"
    assert metadata["forecast_source"] == "local_staged_gfs"
    assert metadata["forecast_source_origin"] == "local"
    assert metadata["staged_validation"]["status"] == "complete"
    assert metadata["staged_validation"]["available_leads"] == [0, 6]


def test_degraded_archive_chunk_is_retried_when_real_source_is_configured(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["ingest"]["source"] = "nomads"
    paths = build_paths(settings)

    feature_file = historical_feature_path(paths, "2026-04-09", "00")
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2026-04-09"], "lead_day": [1], "lat": [35.0], "lon": [-97.0]}).to_parquet(feature_file, index=False)
    metadata_file = historical_metadata_path(paths, "2026-04-09", "00")
    write_json_metadata(metadata_file, {"init_date": "2026-04-09", "cycle": "00", "archive_status": "synthetic_degraded", "real_data": False})

    label_cube = xr.Dataset(
        {
            "tornado": (("date", "lat", "lon"), [[[0]]]),
            "hail": (("date", "lat", "lon"), [[[0]]]),
            "wind": (("date", "lat", "lon"), [[[0]]]),
            "any": (("date", "lat", "lon"), [[[0]]]),
        },
        coords={"date": [pd.Timestamp("2026-04-09")], "lat": [35.0], "lon": [-97.0]},
    )
    outbreak_table = pd.DataFrame({"date": ["2026-04-09"], "category": ["null_day"], "any_outbreak": [0], "tornado_outbreak": [0], "significant_tornado_support": [0]})

    def fake_ingest(date: str, cycle: str, settings, **kwargs):
        dataset, summary = SyntheticForecastSource().fetch_cycle(date, cycle, settings, paths)
        save_forecast_dataset(dataset, paths, date, cycle)
        write_json_metadata(paths.interim / f"ingest_summary_{date}_{cycle}.json", summary)
        return paths.processed / f"forecast_{date}_{cycle}.nc"

    monkeypatch.setattr("severewx.archive.history.ingest_forecast_cycle", fake_ingest)
    result = build_historical_archive_for_cycle("2026-04-09", "00", settings, paths, label_cube=label_cube, outbreak_table=outbreak_table)
    assert result.status == "synthetic_degraded"
    assert result.status != "skipped"
