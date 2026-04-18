"""Historical forecast-feature archive building."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import json
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from severewx.archive.coverage import write_archive_coverage_summary
from severewx.archive.feature_cache import build_cached_feature_archive_for_cycle
from severewx.config import AppSettings
from severewx.features.composites import build_feature_dataset
from severewx.ingest.nomads import _configured_sources, ingest_forecast_cycle
from severewx.ingest.storage import (
    forecast_path,
    historical_feature_path,
    historical_metadata_path,
    load_forecast_dataset,
    load_json_metadata,
    write_json_metadata,
)
from severewx.models.dataset import archived_cycle_identifiers, build_gridpoint_table
from severewx.utils.dates import iter_dates
from severewx.utils.logging import configure_logging
from severewx.utils.paths import DataPaths, build_paths


LOGGER = configure_logging()


@dataclass(slots=True)
class HistoricalArchiveResult:
    date: str
    cycle: str
    status: str
    feature_path: str
    metadata_path: str
    message: str = ""


REAL_ARCHIVE_STATUSES = {"local_real", "remote_real", "partial_real", "built_real", "built_partial_real"}


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _has_real_source_configured(settings: AppSettings) -> bool:
    return any(source in {"nomads", "local_file", "local_staged_gfs"} for source in _configured_sources(settings))


def _reuse_existing_archive_chunk(
    feature_path: Path,
    metadata_path: Path,
    settings: AppSettings,
) -> tuple[bool, str]:
    if not feature_path.exists() or not metadata_path.exists():
        return False, ""
    metadata = load_json_metadata(metadata_path)
    archive_status = str(metadata.get("archive_status", "unknown"))
    if archive_status in REAL_ARCHIVE_STATUSES:
        return True, archive_status
    if archive_status == "synthetic_degraded" and not _has_real_source_configured(settings):
        return True, archive_status
    return False, archive_status


def load_label_artifacts(paths: DataPaths) -> tuple[xr.Dataset | None, pd.DataFrame | None]:
    label_file = _latest_matching_file(paths.labels, "labels_*.nc")
    outbreak_file = _latest_matching_file(paths.labels, "outbreaks_*.parquet")
    label_cube = xr.load_dataset(label_file) if label_file else None
    outbreak_table = pd.read_parquet(outbreak_file) if outbreak_file else None
    return label_cube, outbreak_table


def _find_prior_run(date: str, cycle: str, paths: DataPaths) -> xr.Dataset | None:
    cycles = ["00", "06", "12", "18"]
    if cycle not in cycles:
        return None
    index = cycles.index(cycle)
    if index > 0:
        candidate = forecast_path(paths, date, cycles[index - 1])
        if candidate.exists():
            return xr.load_dataset(candidate)
    current_date = pd.Timestamp(date)
    previous_date = (current_date - timedelta(days=1)).date().isoformat()
    previous_cycle = cycles[-1]
    candidate = forecast_path(paths, previous_date, previous_cycle)
    if candidate.exists():
        return xr.load_dataset(candidate)
    return None


def _should_refresh_processed_forecast(date: str, cycle: str, settings: AppSettings, paths: DataPaths, force: bool) -> bool:
    if force:
        return True
    processed_path = forecast_path(paths, date, cycle)
    if not processed_path.exists():
        return True
    if not _has_real_source_configured(settings):
        return False
    ingest_summary_file = paths.interim / f"ingest_summary_{date}_{cycle}.json"
    if not ingest_summary_file.exists():
        return True
    ingest_summary = load_json_metadata(ingest_summary_file)
    return str(ingest_summary.get("source", "unknown")) in {"synthetic", "synthetic_fallback"}


def historical_cycle_metadata(
    frame: pd.DataFrame,
    forecast: xr.Dataset,
    init_date: str,
    cycle: str,
    ingest_summary: dict[str, Any] | None,
    archive_status: str,
) -> dict[str, Any]:
    label_columns = [column for column in ["tornado", "hail", "wind", "any"] if column in frame.columns]
    metadata: dict[str, Any] = {
        "init_date": init_date,
        "cycle": cycle,
        "forecast_source": str(forecast.attrs.get("source", "unknown")),
        "forecast_source_origin": str((ingest_summary or {}).get("source_origin", "unknown")),
        "available_fields": sorted(forecast.data_vars),
        "valid_times": [pd.Timestamp(value).isoformat() for value in pd.to_datetime(forecast["time"].values)],
        "lead_days": sorted(int(value) for value in frame["lead_day"].unique()) if "lead_day" in frame.columns else [],
        "row_count": int(len(frame)),
        "label_linkage_status": "linked" if any(int(frame[column].sum()) > 0 for column in label_columns) else "no_positive_labels",
        "outbreak_label_status": "linked" if "category" in frame.columns and frame["category"].ne("unknown").any() else "unknown_or_missing",
        "region_counts": frame["region"].value_counts().to_dict() if "region" in frame.columns else {},
        "region_lead_coverage": frame.groupby(["lead_day", "region"], as_index=False).size().rename(columns={"size": "row_count"}).to_dict(orient="records") if {"lead_day", "region"}.issubset(frame.columns) else [],
        "label_positive_counts": {column: int(frame[column].sum()) for column in label_columns},
        "outbreak_outcomes": frame["category"].value_counts().to_dict() if "category" in frame.columns else {},
        "outbreak_by_valid_date": frame.groupby("date")["category"].agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown").to_dict() if {"date", "category"}.issubset(frame.columns) else {},
        "forecast_source_counts": frame["forecast_source_kind"].value_counts().to_dict() if "forecast_source_kind" in frame.columns else {},
        "ingest_summary": ingest_summary or {},
        "processing_scope": (ingest_summary or {}).get("processing_scope", {}),
        "staged_validation": (ingest_summary or {}).get("staged_validation", {}),
        "degraded_features": ((ingest_summary or {}).get("fallbacks_used", {}) if ingest_summary else {}),
        "missing_or_failed_leads": ((ingest_summary or {}).get("failed_leads", []) if ingest_summary else []),
        "real_data": str(forecast.attrs.get("source", "unknown")) not in {"synthetic", "synthetic_fallback"},
        "archive_status": archive_status,
    }
    return metadata


def build_historical_archive_for_cycle(
    date: str,
    cycle: str,
    settings: AppSettings,
    paths: DataPaths,
    label_cube: xr.Dataset | None = None,
    outbreak_table: pd.DataFrame | None = None,
    force: bool = False,
) -> HistoricalArchiveResult:
    feature_path = historical_feature_path(paths, date, cycle)
    metadata_path = historical_metadata_path(paths, date, cycle)
    if not force:
        reusable, existing_status = _reuse_existing_archive_chunk(feature_path, metadata_path, settings)
        if reusable:
            return HistoricalArchiveResult(
                date=date,
                cycle=cycle,
                status="skipped",
                feature_path=str(feature_path),
                metadata_path=str(metadata_path),
                message=f"existing archive chunk reused ({existing_status})",
            )

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if _should_refresh_processed_forecast(date, cycle, settings, paths, force):
        ingest_forecast_cycle(date, cycle, settings=settings)
    forecast = load_forecast_dataset(paths, date, cycle)
    prior_run = _find_prior_run(date, cycle, paths)
    analog_reference_path = paths.models / "analog_reference.parquet"
    features = build_feature_dataset(
        forecast,
        settings=settings,
        prior_run=prior_run,
        analog_archive_path=analog_reference_path if analog_reference_path.exists() else None,
    )
    frame = build_gridpoint_table(
        features,
        label_cube=label_cube,
        outbreak_table=outbreak_table,
        init_date=date,
        init_cycle=cycle,
        source_kind=str(forecast.attrs.get("source", "historical_archive")),
    )
    frame["archive_source"] = "historical_feature_archive"

    ingest_summary_file = paths.interim / f"ingest_summary_{date}_{cycle}.json"
    ingest_summary = json.loads(ingest_summary_file.read_text(encoding="utf-8")) if ingest_summary_file.exists() else {}
    is_real = str(forecast.attrs.get("source", "unknown")) not in {"synthetic", "synthetic_fallback"}
    if is_real:
        source_origin = str(ingest_summary.get("source_origin", "remote"))
        if ingest_summary.get("failed_leads"):
            archive_status = "partial_real"
        elif source_origin == "local":
            archive_status = "local_real"
        else:
            archive_status = "remote_real"
        archive_data_kind = archive_status
        message = f"archive chunk created from {archive_status}"
    else:
        archive_status = "synthetic_degraded"
        archive_data_kind = "synthetic_degraded"
        message = "archive chunk created from fallback/synthetic source"
    frame["archive_chunk_status"] = archive_status
    frame["archive_data_kind"] = archive_data_kind
    frame.to_parquet(feature_path, index=False)

    metadata = historical_cycle_metadata(frame, forecast, date, cycle, ingest_summary, archive_status=archive_status)
    write_json_metadata(metadata_path, metadata)
    return HistoricalArchiveResult(date=date, cycle=cycle, status=archive_status, feature_path=str(feature_path), metadata_path=str(metadata_path), message=message)


def build_historical_archive(
    start: str,
    end: str,
    cycles: list[str],
    settings: AppSettings | None = None,
    force: bool = False,
) -> list[HistoricalArchiveResult]:
    settings = settings or AppSettings(raw={})
    if not settings.raw:
        from severewx.config import load_settings

        settings = load_settings()
    paths = build_paths(settings)
    label_cube, outbreak_table = load_label_artifacts(paths)
    results: list[HistoricalArchiveResult] = []
    all_dates = iter_dates(start, end)
    total = len(all_dates) * len(cycles)
    completed = 0
    for valid_date in all_dates:
        date_value = valid_date.isoformat()
        for cycle in cycles:
            completed += 1
            try:
                result = build_historical_archive_for_cycle(
                    date_value,
                    cycle,
                    settings=settings,
                    paths=paths,
                    label_cube=label_cube,
                    outbreak_table=outbreak_table,
                    force=force,
                )
            except Exception as exc:
                failure_metadata = {
                    "init_date": date_value,
                    "cycle": cycle,
                    "archive_status": "failed",
                    "real_data": False,
                    "error": str(exc),
                    "failed_at": pd.Timestamp.utcnow().isoformat(),
                }
                write_json_metadata(historical_metadata_path(paths, date_value, cycle), failure_metadata)
                result = HistoricalArchiveResult(
                    date=date_value,
                    cycle=cycle,
                    status="failed",
                    feature_path=str(historical_feature_path(paths, date_value, cycle)),
                    metadata_path=str(historical_metadata_path(paths, date_value, cycle)),
                    message=str(exc),
                )
            LOGGER.info("historical archive progress=%d/%d %s %s status=%s message=%s", completed, total, date_value, cycle, result.status, result.message)
            if result.status in {"local_real", "remote_real", "partial_real", "synthetic_degraded", "skipped"}:
                build_cached_feature_archive_for_cycle(date_value, cycle, settings, paths, force=force)
            results.append(result)
    write_archive_coverage_summary(paths, settings=settings)
    summary = {
        "start": start,
        "end": end,
        "cycles": cycles,
        "built_total": sum(result.status in {"local_real", "remote_real", "partial_real", "synthetic_degraded"} for result in results),
        "local_real": sum(result.status == "local_real" for result in results),
        "remote_real": sum(result.status == "remote_real" for result in results),
        "partial_real": sum(result.status == "partial_real" for result in results),
        "synthetic_degraded": sum(result.status == "synthetic_degraded" for result in results),
        "skipped": sum(result.status == "skipped" for result in results),
        "failed": sum(result.status == "failed" for result in results),
        "existing_archives": len(archived_cycle_identifiers(paths)),
        "results": [asdict(result) for result in results],
    }
    (paths.archive_metadata / "historical_archive_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results
