"""NOMADS and pluggable forecast ingest."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path
import time
from typing import Any, Protocol
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import xarray as xr

from severewx.config import AppSettings
from severewx.ingest.normalize import normalize_dataset
from severewx.ingest.stage_gfs import stage_historical_gfs
from severewx.ingest.storage import (
    load_json_metadata,
    raw_grib_metadata_path,
    raw_grib_path,
    save_forecast_dataset,
    write_json_metadata,
)
from severewx.utils.dates import cycle_datetime, iter_dates
from severewx.utils.logging import configure_logging
from severewx.utils.paths import DataPaths, build_paths


LOGGER = configure_logging()
SYNTHETIC_SOURCES = {"synthetic", "synthetic_fallback"}
REMOTE_STAGED_SOURCES = {"aws_recent", "open_meteo_recent", "staged_gfs_auto"}
REALTIME_SOURCES = {"nomads", "local_file", "local_staged_gfs", *REMOTE_STAGED_SOURCES}

STANDARD_FIELDS = [
    "t2m",
    "td2m",
    "mslp",
    "u10",
    "v10",
    "cape",
    "cin",
    "u850",
    "v850",
    "t700",
    "z500",
    "u500",
    "v500",
    "pwat",
    "t500",
]

FIELD_FILTERS: dict[str, dict[str, Any]] = {
    "t2m": {"shortName": "2t"},
    "td2m": {"shortName": "2d"},
    "mslp": {"shortName": "prmsl"},
    "u10": {"shortName": "10u"},
    "v10": {"shortName": "10v"},
    "cape": {"shortName": "cape", "typeOfLevel": "surface"},
    "cin": {"shortName": "cin", "typeOfLevel": "surface"},
    "u850": {"shortName": "u", "typeOfLevel": "isobaricInhPa", "level": 850},
    "v850": {"shortName": "v", "typeOfLevel": "isobaricInhPa", "level": 850},
    "t700": {"shortName": "t", "typeOfLevel": "isobaricInhPa", "level": 700},
    "z500": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
    "u500": {"shortName": "u", "typeOfLevel": "isobaricInhPa", "level": 500},
    "v500": {"shortName": "v", "typeOfLevel": "isobaricInhPa", "level": 500},
    "pwat": {"shortName": "pwat"},
    "t500": {"shortName": "t", "typeOfLevel": "isobaricInhPa", "level": 500},
}


class ForecastSource(Protocol):
    def fetch_cycle(self, date: str, cycle: str, settings: AppSettings, paths: DataPaths) -> tuple[xr.Dataset, dict[str, Any]]:
        """Fetch and normalize a forecast cycle."""


def _summary_path(paths: DataPaths, date: str, cycle: str) -> Path:
    return paths.interim / f"ingest_summary_{date}_{cycle}.json"


def _write_summary(paths: DataPaths, date: str, cycle: str, summary: dict[str, Any]) -> Path:
    path = _summary_path(paths, date, cycle)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _log_ingest_summary(summary: dict[str, Any]) -> None:
    LOGGER.info(
        "ingest summary source=%s mode=%s fields=%s valid_times=%d missing_leads=%s degraded=%s failures=%s",
        summary.get("source"),
        summary.get("source_mode"),
        ",".join(summary.get("available_fields", [])),
        len(summary.get("valid_times", [])),
        summary.get("missing_requested_leads", []),
        summary.get("fallbacks_used", {}),
        summary.get("failed_leads", []),
    )


def _source_mode(source_name: str) -> str:
    return "synthetic" if source_name in SYNTHETIC_SOURCES else "real"


def _configured_sources(settings: AppSettings) -> list[str]:
    primary = str(settings.get("ingest.source", "nomads")).lower()
    failovers = [str(value).lower() for value in settings.get("ingest.failover_sources", []) or []]
    ordered: list[str] = []
    for source_name in [primary, *failovers]:
        if source_name and source_name not in ordered:
            ordered.append(source_name)
    return ordered or ["nomads"]


def _source_origin(source_name: str) -> str:
    normalized = str(source_name).lower()
    if normalized in SYNTHETIC_SOURCES:
        return "synthetic"
    if normalized in {"local_file", "local_staged_gfs"}:
        return "local"
    return "remote"


def _remote_stage_strategy(source_name: str) -> str:
    if source_name == "staged_gfs_auto":
        return "auto"
    if source_name in {"aws_recent", "open_meteo_recent"}:
        return source_name
    raise ValueError(f"unsupported remote staged GFS source: {source_name}")


def _processing_scope(settings: AppSettings) -> dict[str, Any]:
    return {
        "required_fields": [str(value) for value in settings.get("ingest.required_fields", []) or []],
        "optional_fields": [str(value) for value in settings.get("ingest.optional_fields", []) or []],
        "requested_leads": [int(value) for value in settings.get("ingest.leads", []) or []],
        "domain": {
            "lat_min": float(settings.get("grid.lat_min")),
            "lat_max": float(settings.get("grid.lat_max")),
            "lon_min": float(settings.get("grid.lon_min")),
            "lon_max": float(settings.get("grid.lon_max")),
        },
    }


def _subset_to_processing_scope(dataset: xr.Dataset, settings: AppSettings) -> xr.Dataset:
    if "lat" in dataset.coords:
        dataset = dataset.sel(lat=slice(float(settings.get("grid.lat_min")), float(settings.get("grid.lat_max"))))
    if "lon" in dataset.coords:
        dataset = dataset.sel(lon=slice(float(settings.get("grid.lon_min")), float(settings.get("grid.lon_max"))))
    return dataset


@dataclass(slots=True)
class SyntheticForecastSource:
    """Synthetic fallback used for tests and local dev."""

    def fetch_cycle(self, date: str, cycle: str, settings: AppSettings, paths: DataPaths) -> tuple[xr.Dataset, dict[str, Any]]:
        init_time = cycle_datetime(date, cycle)
        leads = settings.get("ingest.leads", [])
        lat = np.arange(
            settings.get("grid.lat_min"),
            settings.get("grid.lat_max") + settings.get("grid.lat_step"),
            settings.get("grid.lat_step"),
            dtype=np.float32,
        )
        lon = np.arange(
            settings.get("grid.lon_min"),
            settings.get("grid.lon_max") + settings.get("grid.lon_step"),
            settings.get("grid.lon_step"),
            dtype=np.float32,
        )
        lon2d, lat2d = np.meshgrid(lon, lat)
        times: list[pd.Timestamp] = []
        fields: dict[str, list[np.ndarray]] = {name: [] for name in STANDARD_FIELDS}
        for lead in leads:
            valid_time = pd.Timestamp(init_time + timedelta(hours=int(lead)))
            times.append(valid_time)
            day_factor = 1.0 + lead / 96.0
            trough = np.sin(np.deg2rad((lon2d + 95.0) * 2.1)) + np.cos(np.deg2rad((lat2d - 35.0) * 2.5))
            moisture_axis = np.exp(-((lon2d + 92.0) ** 2) / 220.0) * np.exp(-((lat2d - 33.0) ** 2) / 120.0)
            jet = np.exp(-((lat2d - 37.0) ** 2) / 45.0)
            fields["t2m"].append((292.0 + 8.0 * np.cos(np.deg2rad(lat2d)) + 2.5 * trough).astype(np.float32))
            fields["td2m"].append((286.0 + 5.5 * moisture_axis + 1.5 * np.sin(np.deg2rad(lon2d))).astype(np.float32))
            fields["mslp"].append((101000.0 - 900.0 * trough / day_factor).astype(np.float32))
            fields["u10"].append((8.0 + 10.0 * jet + 3.0 * np.sin(np.deg2rad(lon2d))).astype(np.float32))
            fields["v10"].append((2.0 + 4.0 * trough).astype(np.float32))
            fields["cape"].append((250.0 + 2200.0 * moisture_axis * day_factor).astype(np.float32))
            fields["cin"].append((-20.0 - 140.0 * (1.0 - moisture_axis)).astype(np.float32))
            fields["u850"].append((14.0 + 18.0 * jet + 3.0 * trough).astype(np.float32))
            fields["v850"].append((4.0 + 10.0 * moisture_axis).astype(np.float32))
            fields["t700"].append((278.0 - 6.0 * trough).astype(np.float32))
            fields["z500"].append((5700.0 + 120.0 * trough).astype(np.float32))
            fields["u500"].append((20.0 + 20.0 * jet).astype(np.float32))
            fields["v500"].append((5.0 + 6.0 * np.sin(np.deg2rad(lat2d * 3.0))).astype(np.float32))
            fields["pwat"].append((20.0 + 22.0 * moisture_axis).astype(np.float32))
            fields["t500"].append((258.0 - 5.0 * trough).astype(np.float32))

        dataset = xr.Dataset(
            data_vars={name: (("time", "lat", "lon"), np.stack(values, axis=0)) for name, values in fields.items()},
            coords={"time": pd.to_datetime(times), "lat": lat, "lon": lon},
        )
        summary = {
            "source": "synthetic",
            "source_mode": "synthetic",
            "available_fields": sorted(dataset.data_vars),
            "valid_times": [pd.Timestamp(value).isoformat() for value in dataset["time"].values],
            "missing_requested_leads": [],
            "fallbacks_used": {},
            "failed_leads": [],
            "cache_overview": {"cache_hits": 0, "cache_misses": 0, "cache_unusable": 0},
        }
        dataset.attrs["source"] = "synthetic"
        return dataset, summary


def _validate_provider_result(dataset: xr.Dataset, summary: dict[str, Any], settings: AppSettings) -> tuple[xr.Dataset, dict[str, Any]]:
    dataset = _subset_to_processing_scope(dataset, settings)
    required_fields = list(settings.get("ingest.required_fields", []))
    missing_required = sorted(field for field in required_fields if field not in dataset.data_vars)
    coord_issues: list[str] = []
    for coord in ("time", "lat", "lon"):
        if coord not in dataset.coords:
            coord_issues.append(f"missing coordinate {coord}")
    if coord_issues:
        raise ValueError(f"provider result has invalid coordinates: {coord_issues}")
    if missing_required:
        raise ValueError(f"provider result is missing required fields after normalization: {missing_required}")
    summary["validated_required_fields"] = required_fields
    summary["validation_status"] = "ok"
    summary["processing_scope"] = _processing_scope(settings)
    return dataset, summary


@dataclass(slots=True)
class NomadsForecastSource:
    """Minimal but more robust GFS ingest from NOMADS filter endpoints."""

    session: requests.Session | None = None

    def _build_url(self, date: str, cycle: str, lead: int, settings: AppSettings) -> str:
        file_name = f"gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}"
        params = {
            "file": file_name,
            "dir": f"/gfs.{date.replace('-', '')}/{cycle}/atmos",
            "subregion": "",
            "leftlon": settings.get("grid.lon_min"),
            "rightlon": settings.get("grid.lon_max"),
            "toplat": settings.get("grid.lat_max"),
            "bottomlat": settings.get("grid.lat_min"),
            "var_TMP": "on",
            "var_DPT": "on",
            "var_PRMSL": "on",
            "var_UGRD": "on",
            "var_VGRD": "on",
            "var_CAPE": "on",
            "var_CIN": "on",
            "var_HGT": "on",
            "var_PWAT": "on",
            "lev_2_m_above_ground": "on",
            "lev_10_m_above_ground": "on",
            "lev_surface": "on",
            "lev_mean_sea_level": "on",
            "lev_850_mb": "on",
            "lev_700_mb": "on",
            "lev_500_mb": "on",
            "lev_entire_atmosphere_single_layer": "on",
        }
        return f"{settings.get('ingest.nomads_base_url')}?{urlencode(params)}"

    @staticmethod
    def _cache_metadata(date: str, cycle: str, lead: int, url: str = "", status: str = "unknown", **extra: Any) -> dict[str, Any]:
        payload = {
            "date": date,
            "cycle": cycle,
            "lead_hour": int(lead),
            "provider": "nomads",
            "status": status,
            "source_url": url,
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        }
        payload.update(extra)
        return payload

    @staticmethod
    def _cache_is_usable(cache_path: Path, metadata: dict[str, Any]) -> bool:
        if not cache_path.exists() or cache_path.stat().st_size == 0:
            return False
        status = str(metadata.get("status", "unknown"))
        return status in {"complete", "cache_hit_complete", "untracked_cache"}

    def _download_grib(self, target_path: Path, url: str, timeout: int, retries: int, backoff_seconds: int) -> tuple[Path, int]:
        session = self.session or requests.Session()
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = session.get(url, timeout=timeout)
                response.raise_for_status()
                target_path.write_bytes(response.content)
                return target_path, attempt
            except Exception as exc:
                last_error = exc
                if attempt == retries:
                    break
                LOGGER.warning("download failed attempt=%d/%d url=%s error=%s", attempt, retries, url, exc)
                time.sleep(backoff_seconds * attempt)
        assert last_error is not None
        raise last_error

    def _fetch_lead_grib(
        self,
        date: str,
        cycle: str,
        lead: int,
        settings: AppSettings,
        paths: DataPaths,
    ) -> tuple[Path, dict[str, Any]]:
        timeout = int(settings.get("ingest.timeout_seconds", 90))
        retries = int(settings.get("ingest.retries", 3))
        backoff_seconds = int(settings.get("ingest.backoff_seconds", 3))
        cache_path = raw_grib_path(paths, date, cycle, lead)
        metadata_path = raw_grib_metadata_path(paths, date, cycle, lead)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = self._build_url(date, cycle, int(lead), settings)
        existing_metadata = load_json_metadata(metadata_path)
        if self._cache_is_usable(cache_path, existing_metadata):
            return cache_path, {
                "lead_hour": int(lead),
                "cache_status": "hit_complete",
                "download_attempts": 0,
                "raw_cache_path": str(cache_path),
                "raw_cache_metadata_path": str(metadata_path),
                "source_url": url,
            }
        if cache_path.exists() and cache_path.stat().st_size > 0 and not existing_metadata:
            write_json_metadata(
                metadata_path,
                self._cache_metadata(date, cycle, lead, url=url, status="untracked_cache", file_size_bytes=cache_path.stat().st_size),
            )
            return cache_path, {
                "lead_hour": int(lead),
                "cache_status": "hit_untracked",
                "download_attempts": 0,
                "raw_cache_path": str(cache_path),
                "raw_cache_metadata_path": str(metadata_path),
                "source_url": url,
            }
        write_json_metadata(
            metadata_path,
            self._cache_metadata(
                date,
                cycle,
                lead,
                url=url,
                status="fetching",
                prior_status=existing_metadata.get("status", "missing"),
            ),
        )
        LOGGER.info("downloading %s", url)
        try:
            cache_path, attempts = self._download_grib(cache_path, url, timeout, retries, backoff_seconds)
        except Exception as exc:
            write_json_metadata(
                metadata_path,
                self._cache_metadata(
                    date,
                    cycle,
                    lead,
                    url=url,
                    status="failed",
                    download_attempts=retries,
                    error=str(exc),
                ),
            )
            raise
        write_json_metadata(
            metadata_path,
            self._cache_metadata(
                date,
                cycle,
                lead,
                url=url,
                status="complete",
                download_attempts=attempts,
                file_size_bytes=cache_path.stat().st_size,
            ),
        )
        return cache_path, {
            "lead_hour": int(lead),
            "cache_status": "miss_downloaded",
            "download_attempts": attempts,
            "raw_cache_path": str(cache_path),
            "raw_cache_metadata_path": str(metadata_path),
            "source_url": url,
        }

    @staticmethod
    def _open_grib_field(grib_path: Path, field_name: str) -> xr.DataArray | None:
        filters = FIELD_FILTERS[field_name]
        try:
            opened = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={"filter_by_keys": filters, "indexpath": ""},
            )
        except Exception:
            return None
        if not opened.data_vars:
            return None
        first_name = next(iter(opened.data_vars))
        data = opened[first_name]
        if "latitude" in data.coords:
            data = data.rename({"latitude": "lat"})
        if "longitude" in data.coords:
            data = data.rename({"longitude": "lon"})
        return data

    @classmethod
    def _open_grib(cls, grib_path: Path, valid_time: pd.Timestamp, settings: AppSettings) -> tuple[xr.Dataset, dict[str, Any]]:
        try:
            import cfgrib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("cfgrib is required for NOMADS GRIB ingest") from exc

        assembled = xr.Dataset()
        for field_name in STANDARD_FIELDS:
            field = cls._open_grib_field(grib_path, field_name)
            if field is not None:
                assembled[field_name] = field.astype(np.float32)
        if not assembled.data_vars:
            fallback = xr.open_dataset(grib_path, engine="cfgrib")
            normalized, summary = normalize_dataset(
                fallback,
                valid_time=valid_time,
                required_fields=list(settings.get("ingest.required_fields", [])),
                requested_leads=list(settings.get("ingest.leads", [])),
            )
            return normalized, summary
        normalized, summary = normalize_dataset(
            assembled,
            valid_time=valid_time,
            required_fields=list(settings.get("ingest.required_fields", [])),
            requested_leads=list(settings.get("ingest.leads", [])),
        )
        return normalized, summary

    def fetch_cycle(self, date: str, cycle: str, settings: AppSettings, paths: DataPaths) -> tuple[xr.Dataset, dict[str, Any]]:
        init_time = cycle_datetime(date, cycle)
        allow_partial_cycle = bool(settings.get("ingest.allow_partial_cycle", False))
        datasets: list[xr.Dataset] = []
        lead_summaries: list[dict[str, Any]] = []
        failed_leads: list[dict[str, Any]] = []
        cache_hit_count = 0
        cache_miss_count = 0
        cache_unusable_count = 0
        for lead in settings.get("ingest.leads", []):
            valid_time = pd.Timestamp(init_time + timedelta(hours=int(lead)))
            try:
                cache_path, cache_summary = self._fetch_lead_grib(date, cycle, int(lead), settings, paths)
                dataset, summary = self._open_grib(cache_path, valid_time, settings)
                summary.update(cache_summary)
                summary["requested_valid_time"] = valid_time.isoformat()
                summary["provider"] = "nomads"
                datasets.append(dataset)
                lead_summaries.append({"lead_hour": int(lead), **summary})
                if summary["cache_status"].startswith("hit"):
                    cache_hit_count += 1
                else:
                    cache_miss_count += 1
            except Exception as exc:
                metadata_path = raw_grib_metadata_path(paths, date, cycle, int(lead))
                existing_metadata = load_json_metadata(metadata_path)
                if existing_metadata:
                    existing_metadata["status"] = "unusable"
                    existing_metadata["error"] = str(exc)
                    write_json_metadata(metadata_path, existing_metadata)
                    cache_unusable_count += 1
                failed_leads.append({"lead_hour": int(lead), "error": str(exc)})
                if not allow_partial_cycle:
                    raise
                LOGGER.warning("skipping failed lead=%s cycle=%s date=%s error=%s", lead, cycle, date, exc)

        if not datasets:
            raise RuntimeError(f"no forecast leads available for {date} {cycle}Z")

        combined = xr.concat(datasets, dim="time").sortby("time")
        requested_leads = [int(lead) for lead in settings.get("ingest.leads", [])]
        seen_leads = [int(summary["lead_hour"]) for summary in lead_summaries]
        cycle_summary = {
            "source": "nomads",
            "source_mode": "real",
            "available_fields": sorted(combined.data_vars),
            "valid_times": [pd.Timestamp(value).isoformat() for value in combined["time"].values],
            "missing_requested_leads": sorted(set(requested_leads) - set(seen_leads)),
            "fallbacks_used": {
                key: value
                for summary in lead_summaries
                for key, value in summary.get("fallbacks_used", {}).items()
            },
            "lead_summaries": lead_summaries,
            "failed_leads": failed_leads,
            "cache_overview": {
                "cache_hits": cache_hit_count,
                "cache_misses": cache_miss_count,
                "cache_unusable": cache_unusable_count,
            },
        }
        combined.attrs["source"] = "nomads"
        combined.attrs["diagnostic_summary"] = json.dumps(cycle_summary)
        return _validate_provider_result(combined, cycle_summary, settings)


def _open_local_forecast_file(
    source_path: Path,
    valid_time: pd.Timestamp,
    settings: AppSettings,
) -> tuple[xr.Dataset, dict[str, Any]]:
    if source_path.suffix.lower() == ".nc":
        dataset = xr.load_dataset(source_path)
        return normalize_dataset(
            dataset,
            valid_time=valid_time,
            required_fields=list(settings.get("ingest.required_fields", [])),
            requested_leads=list(settings.get("ingest.leads", [])),
        )
    return NomadsForecastSource._open_grib(source_path, valid_time, settings)


@dataclass(slots=True)
class LocalFileForecastSource:
    """Load local historical forecast files through the same normalization path."""

    def _file_path(self, date: str, cycle: str, settings: AppSettings) -> Path:
        pattern = str(settings.get("ingest.local_file_pattern", "")).strip()
        if not pattern:
            raise ValueError("ingest.local_file_pattern is required for local_file source")
        return Path(pattern.format(date=date, cycle=cycle))

    def fetch_cycle(self, date: str, cycle: str, settings: AppSettings, paths: DataPaths) -> tuple[xr.Dataset, dict[str, Any]]:
        source_path = self._file_path(date, cycle, settings)
        if not source_path.exists():
            raise FileNotFoundError(f"local forecast file not found: {source_path}")
        normalized, summary = _open_local_forecast_file(source_path, pd.Timestamp(cycle_datetime(date, cycle)), settings)
        summary["source_path"] = str(source_path)
        summary["source"] = "local_file"
        summary["source_mode"] = "real"
        summary["source_origin"] = "local"
        summary.setdefault("failed_leads", [])
        summary.setdefault("cache_overview", {"cache_hits": 1, "cache_misses": 0, "cache_unusable": 0})
        normalized.attrs["source"] = "local_file"
        return _validate_provider_result(normalized, summary, settings)


@dataclass(slots=True)
class LocalStagedGFSForecastSource:
    """Load per-lead locally staged Day 1-4 GFS files through the standard normalization path."""

    def _file_patterns(self, settings: AppSettings) -> list[str]:
        configured = settings.get("ingest.local_staged_gfs.file_patterns", []) or []
        patterns = [str(value).strip() for value in configured if str(value).strip()]
        defaults = [
            "{root}/data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.grib2",
            "{root}/data/raw/staged_gfs/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.nc",
            "{root}/data/raw/staged_gfs/{year}/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.grib2",
            "{root}/data/raw/staged_gfs/{year}/{date}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.nc",
            "{root}/data/raw/staged_gfs/{date_nodash}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.grib2",
            "{root}/data/raw/staged_gfs/{date_nodash}/{cycle}/gfs.t{cycle}z.pgrb2.0p25.f{lead:03d}.nc",
        ]
        ordered: list[str] = []
        for pattern in [*patterns, *defaults]:
            if pattern and pattern not in ordered:
                ordered.append(pattern)
        return ordered

    @staticmethod
    def _format_tokens(date: str, cycle: str, lead: int, settings: AppSettings) -> dict[str, Any]:
        timestamp = pd.Timestamp(date)
        return {
            "root": str(settings.root),
            "date": date,
            "date_nodash": date.replace("-", ""),
            "year": f"{timestamp.year:04d}",
            "month": f"{timestamp.month:02d}",
            "day": f"{timestamp.day:02d}",
            "cycle": cycle,
            "lead": int(lead),
            "lead_padded": f"{int(lead):03d}",
        }

    def _candidate_paths(self, date: str, cycle: str, lead: int, settings: AppSettings) -> list[Path]:
        tokens = self._format_tokens(date, cycle, lead, settings)
        return [Path(pattern.format(**tokens)) for pattern in self._file_patterns(settings)]

    def _resolve_lead_path(self, date: str, cycle: str, lead: int, settings: AppSettings) -> tuple[Path | None, list[str]]:
        candidates = self._candidate_paths(date, cycle, lead, settings)
        checked = [str(path) for path in candidates]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                return candidate, checked
        return None, checked

    def validate_cycle(self, date: str, cycle: str, settings: AppSettings) -> dict[str, Any]:
        requested_leads = [int(lead) for lead in settings.get("ingest.leads", [])]
        usable: list[int] = []
        missing: list[int] = []
        unusable: list[int] = []
        source_files: list[dict[str, Any]] = []
        for lead in requested_leads:
            resolved, checked = self._resolve_lead_path(date, cycle, lead, settings)
            if resolved is not None:
                usable.append(lead)
                source_files.append(
                    {
                        "lead_hour": lead,
                        "path": str(resolved),
                        "checked_paths": checked,
                        "file_size_bytes": resolved.stat().st_size,
                        "usable": True,
                    }
                )
                continue
            missing.append(lead)
            source_files.append(
                {
                    "lead_hour": lead,
                    "path": "",
                    "checked_paths": checked,
                    "file_size_bytes": 0,
                    "usable": False,
                }
            )
        if len(usable) == len(requested_leads) and requested_leads:
            status = "complete"
        elif usable and missing:
            status = "partial"
        elif usable:
            status = "degraded"
        else:
            status = "unusable"
            unusable = requested_leads
        return {
            "date": date,
            "cycle": cycle,
            "requested_leads": requested_leads,
            "available_leads": usable,
            "missing_leads": missing,
            "unusable_leads": unusable,
            "usable_file_count": len(usable),
            "source_files": source_files,
            "status": status,
        }

    def fetch_cycle(self, date: str, cycle: str, settings: AppSettings, paths: DataPaths) -> tuple[xr.Dataset, dict[str, Any]]:
        init_time = cycle_datetime(date, cycle)
        allow_partial_cycle = bool(settings.get("ingest.allow_partial_cycle", False))
        inventory = self.validate_cycle(date, cycle, settings)
        if inventory["status"] == "unusable":
            raise FileNotFoundError(f"no staged local GFS files found for {date} {cycle}Z")

        datasets: list[xr.Dataset] = []
        lead_summaries: list[dict[str, Any]] = []
        failed_leads: list[dict[str, Any]] = []
        for lead in inventory["available_leads"]:
            valid_time = pd.Timestamp(init_time + timedelta(hours=int(lead)))
            source_path, checked = self._resolve_lead_path(date, cycle, int(lead), settings)
            if source_path is None:
                failed_leads.append({"lead_hour": int(lead), "error": "staged file became unavailable", "checked_paths": checked})
                if not allow_partial_cycle:
                    raise FileNotFoundError(f"staged file missing for lead {lead}")
                continue
            try:
                dataset, summary = _open_local_forecast_file(source_path, valid_time, settings)
                summary.update(
                    {
                        "lead_hour": int(lead),
                        "source_path": str(source_path),
                        "checked_paths": checked,
                        "provider": "local_staged_gfs",
                        "cache_status": "staged_local",
                    }
                )
                datasets.append(dataset)
                lead_summaries.append(summary)
            except Exception as exc:
                failed_leads.append({"lead_hour": int(lead), "error": str(exc), "source_path": str(source_path), "checked_paths": checked})
                if not allow_partial_cycle:
                    raise
                LOGGER.warning("skipping unusable staged lead=%s cycle=%s date=%s error=%s", lead, cycle, date, exc)

        if not datasets:
            raise RuntimeError(f"no staged local forecast leads usable for {date} {cycle}Z")

        combined = xr.concat(datasets, dim="time").sortby("time")
        requested_leads = [int(lead) for lead in settings.get("ingest.leads", [])]
        seen_leads = [int(summary["lead_hour"]) for summary in lead_summaries]
        inventory_status = dict(inventory)
        inventory_status["available_leads"] = seen_leads
        inventory_status["missing_leads"] = sorted(set(requested_leads) - set(seen_leads))
        inventory_status["failed_leads"] = failed_leads
        if failed_leads and seen_leads:
            inventory_status["status"] = "degraded"
        elif not seen_leads:
            inventory_status["status"] = "unusable"
        cycle_summary = {
            "source": "local_staged_gfs",
            "source_mode": "real",
            "source_origin": "local",
            "available_fields": sorted(combined.data_vars),
            "valid_times": [pd.Timestamp(value).isoformat() for value in combined["time"].values],
            "available_requested_leads": seen_leads,
            "missing_requested_leads": sorted(set(requested_leads) - set(seen_leads)),
            "fallbacks_used": {
                key: value
                for summary in lead_summaries
                for key, value in summary.get("fallbacks_used", {}).items()
            },
            "lead_summaries": lead_summaries,
            "failed_leads": failed_leads,
            "cache_overview": {"cache_hits": len(lead_summaries), "cache_misses": 0, "cache_unusable": len(failed_leads)},
            "staged_validation": inventory_status,
        }
        combined.attrs["source"] = "local_staged_gfs"
        combined.attrs["diagnostic_summary"] = json.dumps(cycle_summary)
        return _validate_provider_result(combined, cycle_summary, settings)


@dataclass(slots=True)
class RemoteStagedGFSForecastSource:
    """Stage a remote GFS cycle, then load it through the local staged-GFS reader."""

    source_name: str

    def fetch_cycle(self, date: str, cycle: str, settings: AppSettings, paths: DataPaths) -> tuple[xr.Dataset, dict[str, Any]]:
        strategy = _remote_stage_strategy(self.source_name)
        report = stage_historical_gfs(
            start=date,
            end=date,
            cycles=[cycle],
            leads=[int(value) for value in settings.get("ingest.leads", []) or []],
            output_root=paths.raw / "staged_gfs",
            paths=paths,
            settings=settings,
            source_strategy=strategy,
            timeout=int(settings.get("ingest.timeout_seconds", 90)),
            retries=int(settings.get("ingest.retries", 3)),
            backoff_seconds=int(settings.get("ingest.backoff_seconds", 3)),
        )
        if int(report.get("successful_downloads", 0) or 0) <= 0 and int(report.get("skipped_existing_files", 0) or 0) <= 0:
            raise RuntimeError(f"remote staged GFS source failed to stage any usable leads: {report.get('report_path', '')}")
        dataset, summary = LocalStagedGFSForecastSource().fetch_cycle(date, cycle, settings, paths)
        summary["source"] = self.source_name
        summary["source_mode"] = "real"
        summary["source_origin"] = "remote"
        summary["remote_stage_strategy"] = strategy
        summary["remote_stage_report_path"] = str(report.get("report_path", ""))
        summary["remote_stage_successful_downloads"] = int(report.get("successful_downloads", 0) or 0)
        summary["remote_stage_skipped_existing_files"] = int(report.get("skipped_existing_files", 0) or 0)
        summary["remote_stage_failed_downloads"] = int(report.get("failed_downloads", 0) or 0)
        summary["remote_stage_successful_downloads_by_source"] = dict(report.get("successful_downloads_by_source", {}) or {})
        dataset.attrs["source"] = self.source_name
        dataset.attrs["diagnostic_summary"] = json.dumps(summary)
        return _validate_provider_result(dataset, summary, settings)


def validate_local_staged_gfs_inventory(
    start: str,
    end: str,
    cycles: list[str],
    settings: AppSettings,
) -> dict[str, Any]:
    source = LocalStagedGFSForecastSource()
    results: list[dict[str, Any]] = []
    status_counts = {"complete": 0, "partial": 0, "degraded": 0, "unusable": 0}
    detected_dates: set[str] = set()
    detected_cycles: set[str] = set()
    detected_leads: set[int] = set()
    missing_leads: set[int] = set()

    for valid_date in iter_dates(start, end):
        date_value = valid_date.isoformat()
        for cycle in cycles:
            inventory = source.validate_cycle(date_value, cycle, settings)
            inventory["usable_paths"] = [entry["path"] for entry in inventory["source_files"] if entry["usable"]]
            results.append(inventory)
            status = str(inventory["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            if inventory["available_leads"]:
                detected_dates.add(date_value)
                detected_cycles.add(cycle)
                detected_leads.update(int(value) for value in inventory["available_leads"])
            missing_leads.update(int(value) for value in inventory["missing_leads"])

    return {
        "source": "local_staged_gfs",
        "start": start,
        "end": end,
        "cycles": cycles,
        "detected_dates": sorted(detected_dates),
        "detected_cycles": sorted(detected_cycles),
        "detected_lead_hours": sorted(detected_leads),
        "missing_lead_hours": sorted(missing_leads),
        "status_counts": status_counts,
        "cycle_results": results,
    }


def get_forecast_source(settings: AppSettings) -> ForecastSource:
    source = _configured_sources(settings)[0]
    if source == "local_staged_gfs":
        return LocalStagedGFSForecastSource()
    if source == "local_file":
        return LocalFileForecastSource()
    if source == "nomads":
        return NomadsForecastSource()
    if source in REMOTE_STAGED_SOURCES:
        return RemoteStagedGFSForecastSource(source)
    return SyntheticForecastSource()


def get_forecast_sources(settings: AppSettings) -> list[ForecastSource]:
    sources: list[ForecastSource] = []
    for source_name in _configured_sources(settings):
        if source_name == "local_staged_gfs":
            sources.append(LocalStagedGFSForecastSource())
        elif source_name == "local_file":
            sources.append(LocalFileForecastSource())
        elif source_name == "nomads":
            sources.append(NomadsForecastSource())
        elif source_name in REMOTE_STAGED_SOURCES:
            sources.append(RemoteStagedGFSForecastSource(source_name))
        else:
            sources.append(SyntheticForecastSource())
    return sources or [NomadsForecastSource()]


def ingest_forecast_cycle(date: str, cycle: str, settings: AppSettings | None = None) -> Path:
    if settings is None:
        from severewx.config import load_settings

        settings = load_settings()
    paths = build_paths(settings)
    provider_failures: list[dict[str, str]] = []
    dataset: xr.Dataset | None = None
    summary: dict[str, Any] | None = None
    for source_name, source in zip(_configured_sources(settings), get_forecast_sources(settings), strict=False):
        try:
            dataset, summary = source.fetch_cycle(date, cycle, settings, paths)
            summary["attempted_sources"] = [failure["source"] for failure in provider_failures] + [source_name]
            summary["provider_failures"] = provider_failures
            summary["source_origin"] = _source_origin(source_name)
            break
        except Exception as exc:
            provider_failures.append({"source": source_name, "error": str(exc)})
            LOGGER.warning("forecast provider failed source=%s date=%s cycle=%s error=%s", source_name, date, cycle, exc)
    if dataset is None or summary is None:
        if not settings.get("ingest.allow_synthetic_fallback", True):
            raise RuntimeError(f"forecast ingest failed for all configured sources: {provider_failures}")
        LOGGER.error("forecast ingest failed for all configured real sources; falling back to synthetic source")
        dataset, summary = SyntheticForecastSource().fetch_cycle(date, cycle, settings, paths)
        summary["source"] = "synthetic_fallback"
        summary["source_mode"] = "synthetic"
        summary["attempted_sources"] = [failure["source"] for failure in provider_failures] + ["synthetic_fallback"]
        summary["provider_failures"] = provider_failures
        summary["real_ingest_available"] = False
        summary["degraded_reason"] = "all configured real ingest providers failed"
        summary["source_origin"] = "synthetic"
    else:
        summary["real_ingest_available"] = _source_mode(str(summary.get("source", ""))) == "real"
    summary_path = _write_summary(paths, date, cycle, summary)
    _log_ingest_summary(summary | {"summary_path": str(summary_path)})
    LOGGER.info("ingest diagnostics written to %s", summary_path)
    return save_forecast_dataset(dataset, paths, date, cycle)
