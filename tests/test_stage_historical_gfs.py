import json
import sys
from pathlib import Path

import requests
import xarray as xr
import yaml

from severewx.cli.stage_historical_gfs import main as stage_historical_gfs_main
from severewx.config import load_settings
from severewx.ingest.stage_gfs import (
    NOAA_GRIB_SOURCE_SPECS,
    build_source_attempts,
    build_source_urls,
    resolve_source_names,
    stage_historical_gfs,
    staged_forecast_output_path,
    staged_gfs_output_path,
)
from severewx.utils.paths import build_paths


class _FakeResponse:
    def __init__(self, url: str, status_code: int, content: bytes | None = None, json_payload=None) -> None:
        self.url = url
        self.status_code = status_code
        self._content = content or b""
        self._json_payload = json_payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for {self.url}", response=self)

    def iter_content(self, chunk_size: int = 1024 * 1024):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def json(self):
        return self._json_payload

    def close(self) -> None:
        return None


class _FakeSession:
    def __init__(self, responses: dict[str, tuple[int, bytes | None, object | None]]) -> None:
        self.responses = responses
        self.requested_urls: list[str] = []

    def get(self, url: str, timeout: int = 90, stream: bool = True, params=None):  # noqa: ARG002
        resolved_url = requests.Request("GET", url, params=params).prepare().url if params is not None else url
        self.requested_urls.append(resolved_url)
        status_code, content, json_payload = self.responses.get(resolved_url, (404, b"", None))
        return _FakeResponse(resolved_url, status_code, content, json_payload)


def _settings_for_stage(tmp_path: Path):
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    return settings


def test_staged_gfs_output_path_and_default_url_template() -> None:
    output_path = staged_gfs_output_path("data/raw/staged_gfs", "2026-04-09", "00", 6)
    assert output_path.as_posix().endswith("/data/raw/staged_gfs/2026-04-09/00/gfs.t00z.pgrb2.0p25.f006.grib2")

    urls = build_source_urls("2026-04-09", "00", 6, "data/raw/staged_gfs", url_templates=["https://example/{date_nodash}/{cycle}/{lead_padded}"])
    assert urls == ["https://example/20260409/00/006"]


def test_source_selection_prefers_ncei_for_older_dates_and_aws_for_recent_dates() -> None:
    assert resolve_source_names("2011-04-27", source_strategy="auto", recent_window_days=45)[0] == "ncei_historical"
    assert resolve_source_names("2026-04-09", source_strategy="auto", recent_window_days=45)[0] == "aws_recent"

    attempts = build_source_attempts("2011-04-27", "00", 0, "data/raw/staged_gfs", source_strategy="ncei_historical")
    assert attempts[0]["source_name"] == "ncei_historical"
    assert "model-gfs-004-files" in attempts[0]["url"]


def test_model_specific_recent_source_templates_and_output_names() -> None:
    assert {"hrrr_recent", "rap_recent", "nam_recent", "gefs_mean_recent", "gefs_control_recent"}.issubset(NOAA_GRIB_SOURCE_SPECS)

    hrrr_attempts = build_source_attempts("2026-05-03", "00", 6, "data/raw/staged_forecasts/hrrr_recent", source_strategy="hrrr_recent")
    assert hrrr_attempts == [
        {
            "source_name": "hrrr_recent",
            "url": "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20260503/conus/hrrr.t00z.wrfsfcf06.grib2",
        }
    ]
    assert staged_forecast_output_path("data/raw/staged_forecasts/hrrr_recent", "2026-05-03", "00", 6, source_name="hrrr_recent").as_posix().endswith(
        "/data/raw/staged_forecasts/hrrr_recent/2026-05-03/00/hrrr.t00z.wrfsfcf06.grib2"
    )

    nam_attempts = build_source_attempts("2026-05-03", "12", 84, "data/raw/staged_forecasts/nam_recent", source_strategy="nam_recent")
    assert nam_attempts[0]["url"] == "https://noaa-nam-pds.s3.amazonaws.com/nam.20260503/nam.t12z.awphys84.tm00.grib2"

    gefs_attempts = build_source_attempts("2026-05-03", "00", 72, "data/raw/staged_forecasts/gefs_mean_recent", source_strategy="gefs_mean_recent")
    assert gefs_attempts[0]["url"] == "https://noaa-gefs-pds.s3.amazonaws.com/gefs.20260503/00/atmos/pgrb2sp25/geavg.t00z.pgrb2s.0p25.f072"


def test_stage_historical_gfs_downloads_and_skips_existing(tmp_path: Path) -> None:
    settings = _settings_for_stage(tmp_path)
    paths = build_paths(settings)
    stage_root = paths.raw / "staged_gfs"
    existing_path = stage_root / "2011-04-27" / "00" / "gfs.t00z.pgrb2.0p25.f000.grib2"
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_bytes(b"existing")

    ncei_url = "https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-004-files/201104/20110427/gfs_3_20110427_0000_006.grb2"
    aws_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20110427/00/atmos/gfs.t00z.pgrb2.0p25.f006"
    session = _FakeSession({ncei_url: (200, b"pilot-bytes", None)})

    report = stage_historical_gfs(
        "2011-04-27",
        "2011-04-27",
        ["00"],
        [0, 6],
        stage_root,
        paths,
        session=session,
        source_strategy="ncei_historical",
        retries=1,
        backoff_seconds=0,
    )

    downloaded_path = stage_root / "2011-04-27" / "00" / "gfs.t00z.pgrb2.0p25.f006.grib2"
    assert report["attempted_downloads"] == 1
    assert report["successful_downloads"] == 1
    assert report["skipped_existing_files"] == 1
    assert report["failed_downloads"] == 0
    assert report["successful_downloads_by_source"] == {"ncei_historical": 1}
    assert downloaded_path.read_bytes() == b"pilot-bytes"
    assert aws_url not in session.requested_urls
    assert Path(report["report_path"]).exists()


def test_stage_historical_gfs_supports_recent_non_gfs_sources(tmp_path: Path) -> None:
    settings = _settings_for_stage(tmp_path)
    paths = build_paths(settings)
    stage_root = paths.raw / "staged_forecasts" / "rap_recent"
    rap_url = "https://noaa-rap-pds.s3.amazonaws.com/rap.20260503/rap.t00z.awip32f06.grib2"
    session = _FakeSession({rap_url: (200, b"rap-bytes", None)})

    report = stage_historical_gfs(
        "2026-05-03",
        "2026-05-03",
        ["00"],
        [6],
        stage_root,
        paths,
        session=session,
        source_strategy="rap_recent",
        retries=1,
        backoff_seconds=0,
    )

    downloaded_path = stage_root / "2026-05-03" / "00" / "rap.t00z.awip32f06.grib2"
    assert report["output_source_name"] == "rap_recent"
    assert report["successful_downloads"] == 1
    assert report["successful_downloads_by_source"] == {"rap_recent": 1}
    assert downloaded_path.read_bytes() == b"rap-bytes"


def test_stage_historical_gfs_cli_writes_report(tmp_path: Path, monkeypatch) -> None:
    settings = _settings_for_stage(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(settings.raw), encoding="utf-8")
    monkeypatch.setenv("SEVEREWX_CONFIG", str(config_path))

    target_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260409/00/atmos/gfs.t00z.pgrb2.0p25.f000"
    session = _FakeSession({target_url: (200, b"cli-bytes", None)})
    monkeypatch.setattr("severewx.ingest.stage_gfs.requests.Session", lambda: session)
    monkeypatch.setattr(
        sys,
        "argv",
        ["stage_historical_gfs", "--start", "2026-04-09", "--cycles", "00", "--leads", "0", "--source-strategy", "aws_recent", "--retries", "1", "--backoff-seconds", "0"],
    )

    stage_historical_gfs_main()

    paths = build_paths(settings)
    report_path = paths.interim / "staged_gfs_download_2026-04-09_2026-04-09_00.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    staged_path = paths.raw / "staged_gfs" / "2026-04-09" / "00" / "gfs.t00z.pgrb2.0p25.f000.grib2"
    assert staged_path.exists()
    assert payload["successful_downloads"] == 1
    assert payload["failed_downloads"] == 0
    assert payload["source_names"] == ["aws_recent"]


def test_stage_historical_gfs_open_meteo_recent_writes_netcdf(tmp_path: Path) -> None:
    settings = _settings_for_stage(tmp_path)
    paths = build_paths(settings)
    stage_root = paths.raw / "staged_gfs"
    api_url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast?"
        "start_date=2026-04-09&end_date=2026-04-09&models=gfs_seamless&timezone=GMT"
        "&latitude=20.0&longitude=-130.0"
        "&hourly=temperature_2m&hourly=dew_point_2m&hourly=pressure_msl&hourly=wind_speed_10m&hourly=wind_direction_10m"
        "&hourly=cape&hourly=convective_inhibition&hourly=temperature_700hPa&hourly=geopotential_height_500hPa"
        "&hourly=wind_speed_850hPa&hourly=wind_direction_850hPa&hourly=wind_speed_500hPa&hourly=wind_direction_500hPa"
        "&hourly=temperature_500hPa&hourly=total_column_integrated_water_vapour"
    )
    payload = [
        {
            "latitude": 20.0,
            "longitude": -130.0,
            "hourly": {
                "time": ["2026-04-09T00:00"],
                "temperature_2m": [20.0],
                "dew_point_2m": [15.0],
                "pressure_msl": [1000.0],
                "wind_speed_10m": [36.0],
                "wind_direction_10m": [180.0],
                "cape": [1200.0],
                "convective_inhibition": [-25.0],
                "temperature_700hPa": [-8.0],
                "geopotential_height_500hPa": [5700.0],
                "wind_speed_850hPa": [54.0],
                "wind_direction_850hPa": [225.0],
                "wind_speed_500hPa": [72.0],
                "wind_direction_500hPa": [270.0],
                "temperature_500hPa": [-18.0],
                "total_column_integrated_water_vapour": [32.0],
            },
        }
    ]
    session = _FakeSession({api_url: (200, None, payload)})
    settings.raw["grid"]["lat_min"] = 20.0
    settings.raw["grid"]["lat_max"] = 20.0
    settings.raw["grid"]["lon_min"] = -130.0
    settings.raw["grid"]["lon_max"] = -130.0
    settings.raw.setdefault("ingest", {}).setdefault("open_meteo_recent", {})["grid_step_degrees"] = 2.0

    report = stage_historical_gfs(
        "2026-04-09",
        "2026-04-09",
        ["00"],
        [0],
        stage_root,
        paths,
        settings=settings,
        session=session,
        source_strategy="open_meteo_recent",
        retries=1,
        backoff_seconds=0,
    )

    staged_path = stage_root / "2026-04-09" / "00" / "gfs.t00z.pgrb2.0p25.f000.nc"
    assert report["successful_downloads"] == 1
    assert report["successful_downloads_by_source"] == {"open_meteo_recent": 1}
    assert staged_path.exists()


def test_stage_historical_gfs_open_meteo_recent_allows_null_hourly_values(tmp_path: Path) -> None:
    settings = _settings_for_stage(tmp_path)
    paths = build_paths(settings)
    stage_root = paths.raw / "staged_gfs"
    api_url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast?"
        "start_date=2024-04-26&end_date=2024-04-26&models=gfs_seamless&timezone=GMT"
        "&latitude=20.0&longitude=-130.0"
        "&hourly=temperature_2m&hourly=dew_point_2m&hourly=pressure_msl&hourly=wind_speed_10m&hourly=wind_direction_10m"
        "&hourly=cape&hourly=convective_inhibition&hourly=temperature_700hPa&hourly=geopotential_height_500hPa"
        "&hourly=wind_speed_850hPa&hourly=wind_direction_850hPa&hourly=wind_speed_500hPa&hourly=wind_direction_500hPa"
        "&hourly=temperature_500hPa&hourly=total_column_integrated_water_vapour"
    )
    payload = [
        {
            "latitude": 20.0,
            "longitude": -130.0,
            "hourly": {
                "time": ["2024-04-26T00:00"],
                "temperature_2m": [20.0],
                "dew_point_2m": [None],
                "pressure_msl": [1000.0],
                "wind_speed_10m": [36.0],
                "wind_direction_10m": [180.0],
                "cape": [None],
                "convective_inhibition": [-25.0],
                "temperature_700hPa": [-8.0],
                "geopotential_height_500hPa": [5700.0],
                "wind_speed_850hPa": [54.0],
                "wind_direction_850hPa": [225.0],
                "wind_speed_500hPa": [72.0],
                "wind_direction_500hPa": [270.0],
                "temperature_500hPa": [-18.0],
                "total_column_integrated_water_vapour": [32.0],
            },
        }
    ]
    session = _FakeSession({api_url: (200, None, payload)})
    settings.raw["grid"]["lat_min"] = 20.0
    settings.raw["grid"]["lat_max"] = 20.0
    settings.raw["grid"]["lon_min"] = -130.0
    settings.raw["grid"]["lon_max"] = -130.0
    settings.raw.setdefault("ingest", {}).setdefault("open_meteo_recent", {})["grid_step_degrees"] = 2.0

    report = stage_historical_gfs(
        "2024-04-26",
        "2024-04-26",
        ["00"],
        [0],
        stage_root,
        paths,
        settings=settings,
        session=session,
        source_strategy="open_meteo_recent",
        retries=1,
        backoff_seconds=0,
    )

    staged_path = stage_root / "2024-04-26" / "00" / "gfs.t00z.pgrb2.0p25.f000.nc"
    assert report["successful_downloads"] == 1
    assert staged_path.exists()
    dataset = xr.open_dataset(staged_path)
    try:
        assert float(dataset["cape"].values[0, 0, 0]) != float(dataset["cape"].values[0, 0, 0])  # NaN
        assert float(dataset["td2m"].values[0, 0, 0]) != float(dataset["td2m"].values[0, 0, 0])  # NaN
    finally:
        dataset.close()
