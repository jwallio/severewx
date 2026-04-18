"""Small historical GFS downloader/stager for local staged archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr

from severewx.config import AppSettings
from severewx.ingest.storage import write_json_metadata
from severewx.utils.dates import cycle_datetime
from severewx.utils.dates import iter_dates
from severewx.utils.logging import configure_logging
from severewx.utils.paths import DataPaths


LOGGER = configure_logging()

AWS_RECENT_URL_TEMPLATES = [
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date_nodash}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f{lead_padded}",
]

NCEI_HISTORICAL_URL_TEMPLATES = [
    "https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-004-files/{year}{month}/{date_nodash}/gfs_3_{date_nodash}_{cycle}00_{lead_padded}.grb2",
    "https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-004-files-old/{year}{month}/{date_nodash}/gfs_3_{date_nodash}_{cycle}00_{lead_padded}.grb2",
]

DEFAULT_RECENT_WINDOW_DAYS = 45
OPEN_METEO_GFS_AVAILABLE_SINCE = pd.Timestamp("2021-03-23", tz=UTC)
OPEN_METEO_RECENT_ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_GRID_STEP_DEGREES = 4.0
OPEN_METEO_CHUNK_SIZE = 25
OPEN_METEO_HOURLY_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "cape",
    "convective_inhibition",
    "temperature_700hPa",
    "geopotential_height_500hPa",
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "wind_speed_500hPa",
    "wind_direction_500hPa",
    "temperature_500hPa",
    "total_column_integrated_water_vapour",
]


@dataclass(slots=True)
class StageTarget:
    date: str
    cycle: str
    lead: int
    output_path: Path


def staged_gfs_output_path(output_root: Path | str, date: str, cycle: str, lead: int) -> Path:
    root = Path(output_root).resolve()
    return root / date / cycle / f"gfs.t{cycle}z.pgrb2.0p25.f{int(lead):03d}.grib2"


def _report_path(paths: DataPaths, start: str, end: str, cycles: list[str]) -> Path:
    cycle_part = "-".join(cycles)
    return paths.interim / f"staged_gfs_download_{start}_{end}_{cycle_part}.json"


def _format_tokens(date: str, cycle: str, lead: int, output_root: Path) -> dict[str, Any]:
    year, month, day = date.split("-")
    return {
        "date": date,
        "date_nodash": date.replace("-", ""),
        "year": year,
        "month": month,
        "day": day,
        "cycle": cycle,
        "lead": int(lead),
        "lead_padded": f"{int(lead):03d}",
        "output_root": str(output_root),
    }


def _source_templates(source_name: str) -> list[str]:
    normalized = str(source_name).lower()
    if normalized == "ncei_historical":
        return list(NCEI_HISTORICAL_URL_TEMPLATES)
    if normalized == "aws_recent":
        return list(AWS_RECENT_URL_TEMPLATES)
    if normalized == "open_meteo_recent":
        return [OPEN_METEO_RECENT_ENDPOINT]
    raise ValueError(f"unsupported source name: {source_name}")


def _auto_source_names(date: str, recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS) -> list[str]:
    timestamp = pd.Timestamp(date, tz=UTC)
    age_days = (pd.Timestamp.now(tz=UTC).normalize() - timestamp).days
    if age_days <= recent_window_days:
        return ["aws_recent", "open_meteo_recent", "ncei_historical"]
    if timestamp >= OPEN_METEO_GFS_AVAILABLE_SINCE:
        return ["ncei_historical", "open_meteo_recent", "aws_recent"]
    return ["ncei_historical", "aws_recent"]


def resolve_source_names(
    date: str,
    source_strategy: str = "auto",
    recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
) -> list[str]:
    normalized = str(source_strategy).lower()
    if normalized == "auto":
        return _auto_source_names(date, recent_window_days=recent_window_days)
    if normalized in {"ncei_historical", "aws_recent", "open_meteo_recent"}:
        return [normalized]
    raise ValueError(f"unsupported source strategy: {source_strategy}")


def build_source_attempts(
    date: str,
    cycle: str,
    lead: int,
    output_root: Path | str,
    *,
    url_templates: list[str] | None = None,
    source_strategy: str = "auto",
    recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    root = Path(output_root).resolve()
    tokens = _format_tokens(date, cycle, lead, root)
    attempts: list[dict[str, Any]] = []
    if url_templates:
        for template in url_templates:
            attempts.append({"source_name": "custom_url_templates", "url": template.format(**tokens)})
        return attempts
    for source_name in resolve_source_names(date, source_strategy=source_strategy, recent_window_days=recent_window_days):
        if source_name == "open_meteo_recent":
            attempts.append({"source_name": source_name, "url": OPEN_METEO_RECENT_ENDPOINT})
            continue
        for template in _source_templates(source_name):
            attempts.append({"source_name": source_name, "url": template.format(**tokens)})
    return attempts


def build_source_urls(
    date: str,
    cycle: str,
    lead: int,
    output_root: Path | str,
    url_templates: list[str] | None = None,
) -> list[str]:
    return [attempt["url"] for attempt in build_source_attempts(date, cycle, lead, output_root, url_templates=url_templates)]


def iter_stage_targets(
    start: str,
    end: str,
    cycles: list[str],
    leads: list[int],
    output_root: Path | str,
) -> list[StageTarget]:
    targets: list[StageTarget] = []
    for valid_date in iter_dates(start, end):
        date_value = valid_date.isoformat()
        for cycle in cycles:
            for lead in leads:
                targets.append(
                    StageTarget(
                        date=date_value,
                        cycle=cycle,
                        lead=int(lead),
                        output_path=staged_gfs_output_path(output_root, date_value, cycle, int(lead)),
                    )
                )
    return targets


def _remove_temp_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        LOGGER.warning("failed to remove temporary staged file %s", path)


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code is None or status_code in {408, 425, 429, 500, 502, 503, 504}


def _download_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    timeout: int,
    retries: int,
    backoff_seconds: int,
) -> dict[str, Any]:
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, timeout=timeout, stream=True) as response:
                response.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temp_path.replace(output_path)
            return {
                "status": "downloaded",
                "bytes_written": output_path.stat().st_size,
                "attempts": attempt,
                "source_url": url,
            }
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            _remove_temp_file(temp_path)
            if attempt >= retries or not _is_retryable_status(status_code):
                break
        except Exception as exc:  # pragma: no cover - defensive network path
            last_error = exc
            _remove_temp_file(temp_path)
            if attempt >= retries:
                break
        time.sleep(backoff_seconds * attempt)

    assert last_error is not None
    raise last_error


def _stage_output_candidates(output_path: Path) -> list[Path]:
    return [output_path, output_path.with_suffix(".nc")]


def _existing_stage_path(output_path: Path) -> Path | None:
    for candidate in _stage_output_candidates(output_path):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _open_meteo_grid(settings: AppSettings) -> tuple[list[float], list[float]]:
    lat_min = float(settings.get("grid.lat_min"))
    lat_max = float(settings.get("grid.lat_max"))
    lon_min = float(settings.get("grid.lon_min"))
    lon_max = float(settings.get("grid.lon_max"))
    step = float(settings.get("ingest.open_meteo_recent.grid_step_degrees", OPEN_METEO_GRID_STEP_DEGREES))
    latitudes = np.arange(lat_min, lat_max + 0.01, step, dtype=float).round(6).tolist()
    longitudes = np.arange(lon_min, lon_max + 0.01, step, dtype=float).round(6).tolist()
    return latitudes, longitudes


def _open_meteo_pairs(settings: AppSettings) -> list[tuple[float, float]]:
    latitudes, longitudes = _open_meteo_grid(settings)
    return [(float(lat), float(lon)) for lat in latitudes for lon in longitudes]


def _wind_components(speed_kmh: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed_ms = speed_kmh / 3.6
    radians = np.deg2rad(direction_deg)
    return -speed_ms * np.sin(radians), -speed_ms * np.cos(radians)


def _open_meteo_scalar(hourly: dict[str, Any], key: str, index: int) -> float:
    values = hourly.get(key, [])
    if index >= len(values):
        return float("nan")
    value = values[index]
    if value is None:
        return float("nan")
    return float(value)


def _open_meteo_params(valid_date: str, chunk_pairs: list[tuple[float, float]]) -> list[tuple[str, Any]]:
    params: list[tuple[str, Any]] = [
        ("start_date", valid_date),
        ("end_date", valid_date),
        ("models", "gfs_seamless"),
        ("timezone", "GMT"),
    ]
    for lat, lon in chunk_pairs:
        params.append(("latitude", lat))
        params.append(("longitude", lon))
    for variable in OPEN_METEO_HOURLY_VARIABLES:
        params.append(("hourly", variable))
    return params


def _normalize_open_meteo_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("unexpected Open-Meteo response payload")


def _stage_open_meteo_target(
    session: requests.Session,
    target: StageTarget,
    settings: AppSettings,
    timeout: int,
    retries: int,
    backoff_seconds: int,
) -> dict[str, Any]:
    valid_time = pd.Timestamp(cycle_datetime(target.date, target.cycle) + pd.Timedelta(hours=int(target.lead)), tz=UTC)
    if valid_time < OPEN_METEO_GFS_AVAILABLE_SINCE:
        raise ValueError(f"Open-Meteo recent historical GFS is unavailable before {OPEN_METEO_GFS_AVAILABLE_SINCE.date().isoformat()}")

    latitudes, longitudes = _open_meteo_grid(settings)
    points = _open_meteo_pairs(settings)
    time_key = valid_time.strftime("%Y-%m-%dT%H:00")
    target_output = target.output_path.with_suffix(".nc")
    location_records: dict[tuple[float, float], dict[str, float]] = {}
    source_url = OPEN_METEO_RECENT_ENDPOINT
    chunk_size = int(settings.get("ingest.open_meteo_recent.chunk_size", OPEN_METEO_CHUNK_SIZE))
    pause_seconds = float(settings.get("ingest.open_meteo_recent.pause_seconds", 0.2))

    for chunk_start in range(0, len(points), chunk_size):
        chunk_pairs = points[chunk_start : chunk_start + chunk_size]
        params = _open_meteo_params(valid_time.date().isoformat(), chunk_pairs)
        for attempt in range(1, retries + 1):
            try:
                response = session.get(source_url, params=params, timeout=timeout)
                response.raise_for_status()
                payload = _normalize_open_meteo_payload(response.json())
                if len(payload) != len(chunk_pairs):
                    raise ValueError(f"Open-Meteo payload count {len(payload)} does not match requested point count {len(chunk_pairs)}")
                for (requested_lat, requested_lon), location in zip(chunk_pairs, payload, strict=False):
                    hourly = location.get("hourly", {}) or {}
                    times = list(hourly.get("time", []))
                    if time_key not in times:
                        raise ValueError(f"Open-Meteo payload missing requested time {time_key}")
                    index = times.index(time_key)
                    u10, v10 = _wind_components(
                        np.array(_open_meteo_scalar(hourly, "wind_speed_10m", index), dtype=float),
                        np.array(_open_meteo_scalar(hourly, "wind_direction_10m", index), dtype=float),
                    )
                    u850, v850 = _wind_components(
                        np.array(_open_meteo_scalar(hourly, "wind_speed_850hPa", index), dtype=float),
                        np.array(_open_meteo_scalar(hourly, "wind_direction_850hPa", index), dtype=float),
                    )
                    u500, v500 = _wind_components(
                        np.array(_open_meteo_scalar(hourly, "wind_speed_500hPa", index), dtype=float),
                        np.array(_open_meteo_scalar(hourly, "wind_direction_500hPa", index), dtype=float),
                    )
                    location_records[(round(float(requested_lat), 6), round(float(requested_lon), 6))] = {
                        "t2m": _open_meteo_scalar(hourly, "temperature_2m", index) + 273.15,
                        "td2m": _open_meteo_scalar(hourly, "dew_point_2m", index) + 273.15,
                        "mslp": _open_meteo_scalar(hourly, "pressure_msl", index) * 100.0,
                        "u10": float(u10),
                        "v10": float(v10),
                        "cape": _open_meteo_scalar(hourly, "cape", index),
                        "cin": _open_meteo_scalar(hourly, "convective_inhibition", index),
                        "u850": float(u850),
                        "v850": float(v850),
                        "t700": _open_meteo_scalar(hourly, "temperature_700hPa", index) + 273.15,
                        "z500": _open_meteo_scalar(hourly, "geopotential_height_500hPa", index),
                        "u500": float(u500),
                        "v500": float(v500),
                        "pwat": _open_meteo_scalar(hourly, "total_column_integrated_water_vapour", index),
                        "t500": _open_meteo_scalar(hourly, "temperature_500hPa", index) + 273.15,
                    }
                if pause_seconds > 0:
                    time.sleep(pause_seconds)
                break
            except Exception as exc:  # pragma: no cover - live API path
                if attempt >= retries:
                    raise
                time.sleep(backoff_seconds * attempt)

    if len(location_records) != len(points):
        missing = len(points) - len(location_records)
        raise ValueError(f"Open-Meteo grid response missing {missing} grid points")

    data_vars: dict[str, tuple[tuple[str, str, str], np.ndarray]] = {}
    for variable in ["t2m", "td2m", "mslp", "u10", "v10", "cape", "cin", "u850", "v850", "t700", "z500", "u500", "v500", "pwat", "t500"]:
        values = np.empty((1, len(latitudes), len(longitudes)), dtype=np.float32)
        for lat_index, lat in enumerate(latitudes):
            for lon_index, lon in enumerate(longitudes):
                values[0, lat_index, lon_index] = np.float32(location_records[(round(lat, 6), round(lon, 6))][variable])
        data_vars[variable] = (("time", "lat", "lon"), values)

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": [valid_time.tz_convert(None)],
            "lat": np.array(latitudes, dtype=np.float32),
            "lon": np.array(longitudes, dtype=np.float32),
        },
    )
    target_output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(target_output)
    return {
        "status": "downloaded",
        "bytes_written": target_output.stat().st_size,
        "attempts": 1,
        "source_url": source_url,
        "output_path": str(target_output),
    }


def stage_historical_gfs(
    start: str,
    end: str,
    cycles: list[str],
    leads: list[int],
    output_root: Path | str,
    paths: DataPaths,
    *,
    settings: AppSettings | None = None,
    url_templates: list[str] | None = None,
    source_strategy: str = "auto",
    recent_window_days: int = DEFAULT_RECENT_WINDOW_DAYS,
    force: bool = False,
    timeout: int = 90,
    retries: int = 3,
    backoff_seconds: int = 3,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    settings = settings or AppSettings(raw={})
    stage_root = Path(output_root).resolve()
    targets = iter_stage_targets(start, end, cycles, leads, stage_root)
    source_names = ["custom_url_templates"] if url_templates else resolve_source_names(start, source_strategy=source_strategy, recent_window_days=recent_window_days)
    report: dict[str, Any] = {
        "source_strategy": "custom_url_templates" if url_templates else source_strategy,
        "source_names": source_names,
        "recent_window_days": int(recent_window_days),
        "start": start,
        "end": end,
        "cycles": [str(value) for value in cycles],
        "leads": [int(value) for value in leads],
        "output_root": str(stage_root),
        "url_templates": list(url_templates or []),
        "force": bool(force),
        "target_count": len(targets),
        "attempted_downloads": 0,
        "successful_downloads": 0,
        "skipped_existing_files": 0,
        "failed_downloads": 0,
        "successful_downloads_by_source": {},
        "failed_downloads_by_source": {},
        "results": [],
    }

    client = session or requests.Session()
    for target in targets:
        existing_stage = _existing_stage_path(target.output_path)
        if existing_stage is not None and not force:
            report["skipped_existing_files"] += 1
            report["results"].append(
                {
                    "date": target.date,
                    "cycle": target.cycle,
                    "lead_hour": target.lead,
                    "status": "skipped_existing",
                    "output_path": str(existing_stage),
                    "bytes_written": existing_stage.stat().st_size,
                }
            )
            continue

        report["attempted_downloads"] += 1
        attempts = build_source_attempts(
            target.date,
            target.cycle,
            target.lead,
            stage_root,
            url_templates=url_templates,
            source_strategy=source_strategy,
            recent_window_days=recent_window_days,
        )
        failure_messages: list[str] = []
        attempt_log: list[dict[str, Any]] = []
        for attempt in attempts:
            source_name = str(attempt["source_name"])
            url = str(attempt["url"])
            try:
                if source_name == "open_meteo_recent":
                    download = _stage_open_meteo_target(
                        client,
                        target,
                        settings,
                        timeout=timeout,
                        retries=retries,
                        backoff_seconds=backoff_seconds,
                    )
                else:
                    download = _download_file(
                        client,
                        url,
                        target.output_path,
                        timeout=timeout,
                        retries=retries,
                        backoff_seconds=backoff_seconds,
                    )
                result = {
                    "date": target.date,
                    "cycle": target.cycle,
                    "lead_hour": target.lead,
                    "status": "downloaded",
                    "source_name": source_name,
                    "output_path": str(download.get("output_path", target.output_path)),
                    "bytes_written": int(download["bytes_written"]),
                    "source_url": download["source_url"],
                    "attempts": int(download["attempts"]),
                    "source_attempts": attempt_log + [{"source_name": source_name, "url": url, "status": "downloaded"}],
                }
                report["successful_downloads"] += 1
                report["successful_downloads_by_source"][source_name] = report["successful_downloads_by_source"].get(source_name, 0) + 1
                report["results"].append(result)
                break
            except Exception as exc:
                message = f"{url}: {exc}"
                failure_messages.append(message)
                attempt_log.append({"source_name": source_name, "url": url, "status": "failed", "error": str(exc)})
                report["failed_downloads_by_source"][source_name] = report["failed_downloads_by_source"].get(source_name, 0) + 1
        else:
            report["failed_downloads"] += 1
            report["results"].append(
                {
                    "date": target.date,
                    "cycle": target.cycle,
                    "lead_hour": target.lead,
                    "status": "failed",
                    "output_path": str(target.output_path),
                    "source_attempts": attempt_log,
                    "attempted_urls": [attempt["url"] for attempt in attempts],
                    "error": " | ".join(failure_messages),
                }
            )

    report_path = _report_path(paths, start, end, cycles)
    report["report_path"] = str(report_path)
    write_json_metadata(report_path, report)
    LOGGER.info(
        "staged historical GFS downloads strategy=%s attempted=%d successful=%d skipped=%d failed=%d report=%s",
        report["source_strategy"],
        report["attempted_downloads"],
        report["successful_downloads"],
        report["skipped_existing_files"],
        report["failed_downloads"],
        report_path,
    )
    return report


def staging_defaults_from_settings(settings: AppSettings) -> dict[str, Any]:
    return {
        "leads": [int(value) for value in settings.get("ingest.leads", []) or []],
        "timeout": int(settings.get("ingest.timeout_seconds", 90)),
        "retries": int(settings.get("ingest.retries", 3)),
        "backoff_seconds": int(settings.get("ingest.backoff_seconds", 3)),
    }
