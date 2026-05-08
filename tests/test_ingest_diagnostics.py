import numpy as np
import pandas as pd
import xarray as xr

from severewx.config import load_settings
from severewx.ingest import nomads as nomads_ingest
from severewx.ingest.nomads import FIELD_FILTERS, SyntheticForecastSource
from severewx.ingest.normalize import normalize_dataset
from severewx.ingest.storage import raw_grib_metadata_path
from severewx.utils.paths import build_paths


def test_normalize_dataset_derives_optional_fields_and_diagnostics() -> None:
    settings = load_settings()
    raw = xr.Dataset(
        {
            "t2m": (("lat", "lon"), np.full((2, 2), 295.0)),
            "mslp": (("lat", "lon"), np.full((2, 2), 100900.0)),
            "u10": (("lat", "lon"), np.full((2, 2), 10.0)),
            "v10": (("lat", "lon"), np.full((2, 2), 2.0)),
            "cape": (("lat", "lon"), np.full((2, 2), 1500.0)),
            "u850": (("lat", "lon"), np.full((2, 2), 20.0)),
            "v850": (("lat", "lon"), np.full((2, 2), 6.0)),
            "t700": (("lat", "lon"), np.full((2, 2), 275.0)),
            "z500": (("lat", "lon"), np.full((2, 2), 5700.0)),
            "u500": (("lat", "lon"), np.full((2, 2), 30.0)),
            "v500": (("lat", "lon"), np.full((2, 2), 10.0)),
        },
        coords={"lat": [35.0, 36.0], "lon": [-98.0, -97.0]},
    )
    normalized, diagnostics = normalize_dataset(
        raw,
        valid_time=pd.Timestamp("2026-04-09T00:00:00"),
        required_fields=list(settings.get("ingest.required_fields", [])),
        requested_leads=[0, 6],
    )
    assert "td2m" in normalized
    assert "cin" in normalized
    assert diagnostics["missing_required_fields"] == []
    assert "td2m" in diagnostics["fallbacks_used"]


def test_normalize_dataset_handles_scalar_time_and_valid_time_without_conflict() -> None:
    settings = load_settings()
    raw = xr.Dataset(
        {
            "TMP_2_m_above_ground": (("latitude", "longitude"), np.full((2, 2), 295.0)),
            "PRMSL_mean_sea_level": (("latitude", "longitude"), np.full((2, 2), 100900.0)),
            "UGRD_10_m_above_ground": (("latitude", "longitude"), np.full((2, 2), 10.0)),
            "VGRD_10_m_above_ground": (("latitude", "longitude"), np.full((2, 2), 2.0)),
            "CAPE_surface": (("latitude", "longitude"), np.full((2, 2), 1500.0)),
            "UGRD_850mb": (("latitude", "longitude"), np.full((2, 2), 20.0)),
            "VGRD_850mb": (("latitude", "longitude"), np.full((2, 2), 6.0)),
            "TMP_700mb": (("latitude", "longitude"), np.full((2, 2), 275.0)),
            "HGT_500mb": (("latitude", "longitude"), np.full((2, 2), 5700.0)),
            "UGRD_500mb": (("latitude", "longitude"), np.full((2, 2), 30.0)),
            "VGRD_500mb": (("latitude", "longitude"), np.full((2, 2), 10.0)),
        },
        coords={
            "latitude": [35.0, 36.0],
            "longitude": [262.0, 263.0],
            "time": np.datetime64("2024-03-14T00:00:00"),
            "valid_time": np.datetime64("2024-03-14T06:00:00"),
        },
    )
    normalized, diagnostics = normalize_dataset(
        raw,
        required_fields=list(settings.get("ingest.required_fields", [])),
        requested_leads=[0],
    )
    assert normalized.dims["time"] == 1
    assert "valid_time" not in normalized.coords
    assert pd.Timestamp(normalized["time"].values[0]) == pd.Timestamp("2024-03-14T06:00:00")
    assert normalized["lon"].values.tolist() == [-98.0, -97.0]
    assert diagnostics["coord_issues"] == []


def test_normalize_dataset_strips_auxiliary_scalar_coords_from_fields() -> None:
    settings = load_settings()
    raw = xr.Dataset(
        {
            "TMP_2_m_above_ground": xr.DataArray(
                np.full((2, 2), 295.0),
                dims=("latitude", "longitude"),
                coords={
                    "latitude": [35.0, 36.0],
                    "longitude": [262.0, 263.0],
                    "step": np.timedelta64(6, "h"),
                },
            ),
            "PRMSL_mean_sea_level": xr.DataArray(
                np.full((2, 2), 100900.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "UGRD_10_m_above_ground": xr.DataArray(
                np.full((2, 2), 10.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "VGRD_10_m_above_ground": xr.DataArray(
                np.full((2, 2), 2.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "CAPE_surface": xr.DataArray(
                np.full((2, 2), 1500.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "UGRD_850mb": xr.DataArray(
                np.full((2, 2), 20.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "VGRD_850mb": xr.DataArray(
                np.full((2, 2), 6.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "TMP_700mb": xr.DataArray(
                np.full((2, 2), 275.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "HGT_500mb": xr.DataArray(
                np.full((2, 2), 5700.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "UGRD_500mb": xr.DataArray(
                np.full((2, 2), 30.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "VGRD_500mb": xr.DataArray(
                np.full((2, 2), 10.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
        },
        coords={"time": np.datetime64("2024-04-26T06:00:00")},
    )
    normalized, diagnostics = normalize_dataset(
        raw,
        required_fields=list(settings.get("ingest.required_fields", [])),
        requested_leads=[0],
    )
    assert "step" not in normalized.coords
    assert diagnostics["coord_issues"] == []


def test_synthetic_ingest_summary_reports_cache_and_mode(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    dataset, summary = SyntheticForecastSource().fetch_cycle("2026-04-09", "00", settings, paths)
    assert dataset.attrs["source"] == "synthetic"
    assert summary["source_mode"] == "synthetic"
    assert summary["cache_overview"]["cache_hits"] == 0
    assert not raw_grib_metadata_path(paths, "2026-04-09", "00", 0).exists()


def test_ingest_can_fail_over_from_nomads_to_remote_staged_gfs(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["paths"]["data"] = str(tmp_path / "data")
    settings.raw["paths"]["raw"] = str(tmp_path / "data" / "raw")
    settings.raw["paths"]["interim"] = str(tmp_path / "data" / "interim")
    settings.raw["paths"]["outputs"] = str(tmp_path / "data" / "outputs")
    settings.raw["ingest"]["source"] = "nomads"
    settings.raw["ingest"]["failover_sources"] = ["aws_recent"]
    settings.raw["ingest"]["allow_synthetic_fallback"] = False
    settings.raw["ingest"]["leads"] = [0, 6]
    settings.raw["ingest"]["allow_partial_cycle"] = False
    paths = build_paths(settings)

    def fail_nomads(self, date, cycle, settings, paths):
        raise RuntimeError("nomads 500")

    def fake_stage_historical_gfs(
        *,
        start,
        end,
        cycles,
        leads,
        output_root,
        paths: object,
        settings,
        source_strategy,
        timeout,
        retries,
        backoff_seconds,
    ):
        assert start == "2026-04-09"
        assert end == "2026-04-09"
        assert cycles == ["00"]
        assert leads == [0, 6]
        assert output_root == paths.raw / "staged_gfs"
        assert source_strategy == "aws_recent"
        return {
            "report_path": str(paths.interim / "staged_gfs_download_2026-04-09_2026-04-09_00.json"),
            "successful_downloads": 2,
            "skipped_existing_files": 0,
            "failed_downloads": 0,
            "successful_downloads_by_source": {"aws_recent": 2},
        }

    def fake_local_staged_fetch(self, date, cycle, settings, paths):
        dataset, summary = SyntheticForecastSource().fetch_cycle(date, cycle, settings, paths)
        summary["source"] = "local_staged_gfs"
        summary["source_mode"] = "real"
        dataset.attrs["source"] = "local_staged_gfs"
        return dataset, summary

    monkeypatch.setattr(nomads_ingest.NomadsForecastSource, "fetch_cycle", fail_nomads)
    monkeypatch.setattr(nomads_ingest, "stage_historical_gfs", fake_stage_historical_gfs)
    monkeypatch.setattr(nomads_ingest.LocalStagedGFSForecastSource, "fetch_cycle", fake_local_staged_fetch)

    output = nomads_ingest.ingest_forecast_cycle("2026-04-09", "00", settings=settings)
    summary = pd.read_json(paths.interim / "ingest_summary_2026-04-09_00.json", typ="series")

    assert output.exists()
    assert summary["source"] == "aws_recent"
    assert summary["source_mode"] == "real"
    assert summary["real_ingest_available"]
    assert summary["attempted_sources"] == ["nomads", "aws_recent"]
    assert summary["provider_failures"] == [{"source": "nomads", "error": "nomads 500"}]
    assert summary["remote_stage_strategy"] == "aws_recent"


def test_ingest_can_use_hrrr_as_remote_staged_forecast_source(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["paths"]["data"] = str(tmp_path / "data")
    settings.raw["paths"]["raw"] = str(tmp_path / "data" / "raw")
    settings.raw["paths"]["interim"] = str(tmp_path / "data" / "interim")
    settings.raw["paths"]["outputs"] = str(tmp_path / "data" / "outputs")
    settings.raw["ingest"]["source"] = "hrrr_recent"
    settings.raw["ingest"]["allow_synthetic_fallback"] = False
    settings.raw["ingest"]["leads"] = [0, 6]
    paths = build_paths(settings)

    def fake_stage_historical_gfs(
        *,
        start,
        end,
        cycles,
        leads,
        output_root,
        paths: object,
        settings,
        source_strategy,
        timeout,
        retries,
        backoff_seconds,
    ):
        assert start == "2026-05-03"
        assert end == "2026-05-03"
        assert cycles == ["00"]
        assert leads == [0, 6]
        assert output_root == paths.raw / "staged_forecasts" / "hrrr_recent"
        assert source_strategy == "hrrr_recent"
        return {
            "report_path": str(paths.interim / "staged_gfs_download_2026-05-03_2026-05-03_00.json"),
            "successful_downloads": 2,
            "skipped_existing_files": 0,
            "failed_downloads": 0,
            "successful_downloads_by_source": {"hrrr_recent": 2},
        }

    def fake_local_staged_fetch(self, date, cycle, settings, paths):
        assert self.source_name == "local_staged_hrrr_recent"
        assert self.stage_root == paths.raw / "staged_forecasts" / "hrrr_recent"
        assert self.stage_source_name == "hrrr_recent"
        dataset, summary = SyntheticForecastSource().fetch_cycle(date, cycle, settings, paths)
        summary["source"] = self.source_name
        summary["source_mode"] = "real"
        dataset.attrs["source"] = self.source_name
        return dataset, summary

    monkeypatch.setattr(nomads_ingest, "stage_historical_gfs", fake_stage_historical_gfs)
    monkeypatch.setattr(nomads_ingest.LocalStagedGFSForecastSource, "fetch_cycle", fake_local_staged_fetch)

    output = nomads_ingest.ingest_forecast_cycle("2026-05-03", "00", settings=settings)
    summary = pd.read_json(paths.interim / "ingest_summary_2026-05-03_00.json", typ="series")

    assert output.exists()
    assert summary["source"] == "hrrr_recent"
    assert summary["source_model"] == "hrrr"
    assert summary["source_mode"] == "real"
    assert summary["source_origin"] == "remote"
    assert summary["real_ingest_available"]
    assert summary["remote_stage_strategy"] == "hrrr_recent"
    assert summary["remote_stage_output_source_name"] == "hrrr_recent"
    assert summary["remote_stage_successful_downloads_by_source"] == {"hrrr_recent": 2}


def test_source_specific_staged_paths_take_precedence_over_gfs_defaults(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["ingest"]["local_staged_gfs"]["file_patterns"] = [
        "{root}/data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.grib2"
    ]
    paths = build_paths(settings)
    source = nomads_ingest.LocalStagedGFSForecastSource(
        source_name="local_staged_nam_recent",
        stage_root=paths.raw / "staged_forecasts" / "nam_recent",
        stage_source_name="nam_recent",
    )

    candidates = source._candidate_paths("2026-05-03", "00", 60, settings)

    assert candidates[0] == paths.raw / "staged_forecasts" / "nam_recent" / "2026-05-03" / "00" / "nam.t00z.awphys60.tm00.grib2"
    assert candidates[1] == paths.raw / "staged_forecasts" / "nam_recent" / "2026-05-03" / "00" / "nam.t00z.awphys60.tm00.nc"
    assert "staged_gfs" in str(candidates[2])


def test_ingest_can_use_ecmwf_as_open_meteo_backed_remote_source(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["paths"]["data"] = str(tmp_path / "data")
    settings.raw["paths"]["raw"] = str(tmp_path / "data" / "raw")
    settings.raw["paths"]["interim"] = str(tmp_path / "data" / "interim")
    settings.raw["paths"]["outputs"] = str(tmp_path / "data" / "outputs")
    settings.raw["ingest"]["source"] = "ecmwf_recent"
    settings.raw["ingest"]["allow_synthetic_fallback"] = False
    settings.raw["ingest"]["leads"] = [0, 6]
    paths = build_paths(settings)

    def fake_stage_historical_gfs(
        *,
        start,
        end,
        cycles,
        leads,
        output_root,
        paths: object,
        settings,
        source_strategy,
        timeout,
        retries,
        backoff_seconds,
    ):
        assert output_root == paths.raw / "staged_forecasts" / "ecmwf_recent"
        assert source_strategy == "ecmwf_recent"
        return {
            "report_path": str(paths.interim / "staged_gfs_download_2026-05-03_2026-05-03_00.json"),
            "successful_downloads": 2,
            "skipped_existing_files": 0,
            "failed_downloads": 0,
            "successful_downloads_by_source": {"ecmwf_recent": 2},
            "results": [
                {"lead_hour": 0, "status": "downloaded"},
                {"lead_hour": 24, "status": "downloaded"},
            ],
        }

    def fake_local_staged_fetch(self, date, cycle, settings, paths):
        assert self.source_name == "local_staged_ecmwf_recent"
        assert self.stage_root == paths.raw / "staged_forecasts" / "ecmwf_recent"
        assert self.stage_source_name == "ecmwf_recent"
        assert settings.get("ingest.leads") == [0, 24]
        dataset, summary = SyntheticForecastSource().fetch_cycle(date, cycle, settings, paths)
        summary["source"] = self.source_name
        summary["source_mode"] = "real"
        dataset.attrs["source"] = self.source_name
        return dataset, summary

    monkeypatch.setattr(nomads_ingest, "stage_historical_gfs", fake_stage_historical_gfs)
    monkeypatch.setattr(nomads_ingest.LocalStagedGFSForecastSource, "fetch_cycle", fake_local_staged_fetch)

    output = nomads_ingest.ingest_forecast_cycle("2026-05-03", "00", settings=settings)
    summary = pd.read_json(paths.interim / "ingest_summary_2026-05-03_00.json", typ="series")

    assert output.exists()
    assert summary["source"] == "ecmwf_recent"
    assert summary["source_model"] == "ecmwf_ifs"
    assert summary["remote_stage_strategy"] == "ecmwf_recent"
    assert summary["remote_stage_output_source_name"] == "ecmwf_recent"


def test_grib_filters_disambiguate_surface_cape_and_cin() -> None:
    assert FIELD_FILTERS["cape"]["typeOfLevel"] == "surface"
    assert FIELD_FILTERS["cin"]["typeOfLevel"] == "surface"


def test_mslp_filter_supports_hrrr_and_rap_short_names() -> None:
    assert FIELD_FILTERS["mslp"] == [{"shortName": "prmsl"}, {"shortName": "mslma"}, {"shortName": "mslet"}]


def test_curvilinear_remote_grib_grid_remaps_to_configured_lat_lon() -> None:
    settings = load_settings()
    settings.raw["grid"]["lat_min"] = 35.0
    settings.raw["grid"]["lat_max"] = 36.0
    settings.raw["grid"]["lat_step"] = 1.0
    settings.raw["grid"]["lon_min"] = -98.0
    settings.raw["grid"]["lon_max"] = -97.0
    settings.raw["grid"]["lon_step"] = 1.0
    raw = xr.Dataset(
        {
            "t2m": (("y", "x"), np.array([[290.0, 291.0], [292.0, 293.0]], dtype=np.float32)),
            "cape": (("y", "x"), np.array([[1000.0, 1100.0], [1200.0, 1300.0]], dtype=np.float32)),
        },
        coords={
            "lat": (("y", "x"), np.array([[35.0, 35.0], [36.0, 36.0]], dtype=np.float32)),
            "lon": (("y", "x"), np.array([[262.0, 263.0], [262.0, 263.0]], dtype=np.float32)),
        },
    )

    remapped = nomads_ingest._rectilinearize_curvilinear_dataset(raw, settings)

    assert remapped["lat"].values.tolist() == [35.0, 36.0]
    assert remapped["lon"].values.tolist() == [-98.0, -97.0]
    assert remapped["t2m"].dims == ("lat", "lon")
    assert remapped["t2m"].values.tolist() == [[290.0, 291.0], [292.0, 293.0]]
