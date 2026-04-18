"""Cached training-feature archive derived from historical archive chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from severewx.config import AppSettings
from severewx.ingest.storage import historical_feature_path, historical_metadata_path, load_json_metadata, staged_raw_archive_dir, write_json_metadata
from severewx.utils.dates import iter_dates
from severewx.utils.logging import configure_logging
from severewx.utils.paths import DataPaths, build_paths


LOGGER = configure_logging()
HAZARD_TARGET_COLUMNS = ("tornado", "hail", "wind", "any")

FEATURE_ARCHIVE_METADATA_COLUMNS = [
    "date",
    "lead_day",
    "init_date",
    "init_cycle",
    "lat",
    "lon",
    "region",
    "forecast_source_kind",
    "archive_source",
    "archive_chunk_status",
    "archive_data_kind",
]

FEATURE_ARCHIVE_OUTBREAK_COLUMNS = [
    "tornado_outbreak",
    "hail_outbreak",
    "wind_outbreak",
    "any_outbreak",
    "significant_tornado_support",
    "spatial_coverage",
    "category",
]


@dataclass(slots=True)
class CachedFeatureArchiveResult:
    date: str
    cycle: str
    status: str
    feature_cache_path: str
    metadata_path: str
    message: str = ""


RAW_LIFECYCLE_SAFE_MODES = {"keep_raw", "move_raw_to_archive", "delete_raw_after_verified_cache"}


def cached_feature_archive_path(paths: DataPaths, date: str, cycle: str) -> Path:
    year = date[:4]
    return paths.processed / "cached_feature_archive" / year / date / cycle / "training_features.parquet"


def cached_feature_archive_metadata_path(paths: DataPaths, date: str, cycle: str) -> Path:
    year = date[:4]
    return paths.archive_metadata / year / date / f"{cycle}_feature_cache.json"


def discover_cached_feature_archives(paths: DataPaths) -> list[Path]:
    return sorted((paths.processed / "cached_feature_archive").glob("*/*/*/training_features.parquet"))


def _feature_columns(settings: AppSettings) -> list[str]:
    hazard_columns = [str(value) for value in settings.get("models.hazard_columns", []) or []]
    outbreak_columns = [str(value) for value in settings.get("models.outbreak_columns", []) or []]
    ordered: list[str] = []
    for column in [*FEATURE_ARCHIVE_METADATA_COLUMNS, *hazard_columns, *outbreak_columns, *HAZARD_TARGET_COLUMNS, *FEATURE_ARCHIVE_OUTBREAK_COLUMNS]:
        if column and column not in ordered:
            ordered.append(column)
    return ordered


def _raw_lifecycle_mode(settings: AppSettings) -> str:
    mode = str(settings.get("archive.raw_file_handling.mode", "keep_raw"))
    return mode if mode in RAW_LIFECYCLE_SAFE_MODES else "keep_raw"


def _staged_source_files(source_metadata: dict[str, Any]) -> list[Path]:
    staged_validation = source_metadata.get("staged_validation", {}) or {}
    files: list[Path] = []
    for entry in staged_validation.get("source_files", []) or []:
        if entry.get("usable") and entry.get("path"):
            files.append(Path(str(entry["path"])))
    return files


def _raw_lifecycle_is_verified(source_metadata: dict[str, Any], cache_metadata: dict[str, Any]) -> tuple[bool, str]:
    if str(source_metadata.get("forecast_source", "unknown")) != "local_staged_gfs":
        return False, "raw lifecycle only applies to local_staged_gfs source files"
    staged_validation = source_metadata.get("staged_validation", {}) or {}
    if str(staged_validation.get("status", "unknown")) != "complete":
        return False, "source validation did not complete successfully"
    if str(source_metadata.get("archive_status", "unknown")) != "local_real":
        return False, "historical archive chunk is not a verified local_real build"
    if str(cache_metadata.get("cache_status", "unknown")) != "cached_feature_archive_ready":
        return False, "cached feature archive metadata is not verified ready"
    if int(cache_metadata.get("row_count", 0)) <= 0:
        return False, "cached feature archive has no rows"
    return True, ""


def _apply_raw_lifecycle(
    date: str,
    cycle: str,
    settings: AppSettings,
    paths: DataPaths,
    source_metadata: dict[str, Any],
    cache_metadata: dict[str, Any],
) -> dict[str, Any]:
    mode = _raw_lifecycle_mode(settings)
    staged_files = _staged_source_files(source_metadata)
    existing_files = [path for path in staged_files if path.exists()]
    eligible, reason = _raw_lifecycle_is_verified(source_metadata, cache_metadata)
    summary = {
        "mode": mode,
        "raw_files_processed": len(existing_files),
        "raw_files_retained": len(existing_files),
        "raw_files_moved": 0,
        "raw_files_deleted": 0,
        "action": "retained",
        "eligible": eligible,
        "reason": reason or "retained by policy",
        "archive_destination": "",
        "processed_files": [str(path) for path in existing_files],
    }
    if mode == "keep_raw" or not existing_files:
        return summary
    if not eligible:
        return summary
    if mode == "move_raw_to_archive":
        destination_root = staged_raw_archive_dir(
            paths,
            date,
            cycle,
            archive_root=str(settings.get("archive.raw_file_handling.archive_root", "")).strip() or None,
        )
        destination_root.mkdir(parents=True, exist_ok=True)
        moved = 0
        for source_path in existing_files:
            destination = destination_root / source_path.name
            if destination.exists():
                destination.unlink()
            shutil.move(str(source_path), str(destination))
            moved += 1
        summary.update(
            {
                "raw_files_retained": 0,
                "raw_files_moved": moved,
                "action": "moved",
                "reason": "raw files moved after verified cache build",
                "archive_destination": str(destination_root),
            }
        )
        return summary
    if mode == "delete_raw_after_verified_cache":
        deleted = 0
        for source_path in existing_files:
            source_path.unlink(missing_ok=True)
            deleted += 1
        summary.update(
            {
                "raw_files_retained": 0,
                "raw_files_deleted": deleted,
                "action": "deleted",
                "reason": "raw files deleted after verified cache build",
            }
        )
    return summary


def _feature_cache_metadata(
    frame: pd.DataFrame,
    source_metadata: dict[str, Any],
    date: str,
    cycle: str,
    raw_lifecycle: dict[str, Any],
) -> dict[str, Any]:
    hazard_positive_counts = {
        column: int(frame[column].sum())
        for column in HAZARD_TARGET_COLUMNS
        if column in frame.columns
    }
    outbreak_counts = frame["category"].value_counts().to_dict() if "category" in frame.columns else {}
    forecast_source_counts = frame["forecast_source_kind"].value_counts().to_dict() if "forecast_source_kind" in frame.columns else {}
    return {
        "init_date": date,
        "cycle": cycle,
        "source_archive_path": str(source_metadata.get("feature_path", "")),
        "source_archive_status": str(source_metadata.get("archive_status", "unknown")),
        "forecast_source": str(source_metadata.get("forecast_source", "unknown")),
        "forecast_source_origin": str(source_metadata.get("forecast_source_origin", "unknown")),
        "lead_days": sorted(int(value) for value in frame["lead_day"].unique()) if "lead_day" in frame.columns else [],
        "row_count": int(len(frame)),
        "feature_count": int(len(frame.columns)),
        "column_names": list(frame.columns),
        "processing_scope": source_metadata.get("ingest_summary", {}).get("processing_scope", {}),
        "label_linkage_status": str(source_metadata.get("label_linkage_status", "unknown")),
        "outbreak_label_status": str(source_metadata.get("outbreak_label_status", "unknown")),
        "region_counts": frame["region"].value_counts().to_dict() if "region" in frame.columns else {},
        "hazard_positive_counts": hazard_positive_counts,
        "outbreak_counts": outbreak_counts,
        "forecast_source_counts": forecast_source_counts,
        "real_data": bool(source_metadata.get("real_data", False)),
        "cache_status": "cached_feature_archive_ready",
        "raw_lifecycle": raw_lifecycle,
    }


def _cache_refresh_reason(
    source_path: Path,
    source_metadata_path: Path,
    cache_path: Path,
    source_metadata: dict[str, Any],
    cache_metadata: dict[str, Any],
) -> str:
    if not cache_path.exists():
        return "cache parquet missing"
    if source_path.stat().st_mtime > cache_path.stat().st_mtime:
        return "source archive parquet is newer than cached feature parquet"
    if source_metadata_path.exists() and source_metadata_path.stat().st_mtime > cache_path.stat().st_mtime:
        return "source archive metadata is newer than cached feature parquet"

    provenance_pairs = [
        ("archive_status", "source_archive_status"),
        ("forecast_source", "forecast_source"),
        ("forecast_source_origin", "forecast_source_origin"),
        ("real_data", "real_data"),
    ]
    for source_key, cache_key in provenance_pairs:
        if source_metadata.get(source_key) != cache_metadata.get(cache_key):
            return f"source provenance changed: {source_key}"
    return ""


def build_cached_feature_archive_for_cycle(
    date: str,
    cycle: str,
    settings: AppSettings,
    paths: DataPaths,
    force: bool = False,
) -> CachedFeatureArchiveResult:
    source_path = historical_feature_path(paths, date, cycle)
    source_metadata_path = historical_metadata_path(paths, date, cycle)
    cache_path = cached_feature_archive_path(paths, date, cycle)
    metadata_path = cached_feature_archive_metadata_path(paths, date, cycle)
    if not source_path.exists() or not source_metadata_path.exists():
        return CachedFeatureArchiveResult(
            date=date,
            cycle=cycle,
            status="missing_source_archive",
            feature_cache_path=str(cache_path),
            metadata_path=str(metadata_path),
            message="historical archive chunk not available",
        )
    source_metadata = load_json_metadata(source_metadata_path)
    if cache_path.exists() and metadata_path.exists() and not force:
        cache_metadata = load_json_metadata(metadata_path)
        refresh_reason = _cache_refresh_reason(source_path, source_metadata_path, cache_path, source_metadata, cache_metadata)
        if not refresh_reason:
            raw_lifecycle = _apply_raw_lifecycle(date, cycle, settings, paths, source_metadata, cache_metadata)
            cache_metadata["raw_lifecycle"] = raw_lifecycle
            write_json_metadata(metadata_path, cache_metadata)
            return CachedFeatureArchiveResult(
                date=date,
                cycle=cycle,
                status="skipped",
                feature_cache_path=str(cache_path),
                metadata_path=str(metadata_path),
                message="cached feature archive reused",
            )
        raw_lifecycle = _apply_raw_lifecycle(date, cycle, settings, paths, source_metadata, cache_metadata)
        cache_metadata["raw_lifecycle"] = raw_lifecycle
        write_json_metadata(metadata_path, cache_metadata)

    source_frame = pd.read_parquet(source_path)
    selected_columns = [column for column in _feature_columns(settings) if column in source_frame.columns]
    cached_frame = source_frame[selected_columns].copy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached_frame.to_parquet(cache_path, index=False)

    source_metadata["feature_path"] = str(source_path)
    metadata = _feature_cache_metadata(cached_frame, source_metadata, date, cycle, raw_lifecycle={})
    write_json_metadata(metadata_path, metadata)
    raw_lifecycle = _apply_raw_lifecycle(date, cycle, settings, paths, source_metadata, metadata)
    metadata["raw_lifecycle"] = raw_lifecycle
    write_json_metadata(metadata_path, metadata)
    return CachedFeatureArchiveResult(
        date=date,
        cycle=cycle,
        status="built",
        feature_cache_path=str(cache_path),
        metadata_path=str(metadata_path),
        message="cached feature archive created",
    )


def build_cached_feature_archive(
    start: str,
    end: str,
    cycles: list[str],
    settings: AppSettings | None = None,
    force: bool = False,
) -> list[CachedFeatureArchiveResult]:
    settings = settings or AppSettings(raw={})
    if not settings.raw:
        from severewx.config import load_settings

        settings = load_settings()
    paths = build_paths(settings)
    results: list[CachedFeatureArchiveResult] = []
    total = len(iter_dates(start, end)) * len(cycles)
    completed = 0
    for valid_date in iter_dates(start, end):
        date_value = valid_date.isoformat()
        for cycle in cycles:
            completed += 1
            result = build_cached_feature_archive_for_cycle(date_value, cycle, settings, paths, force=force)
            LOGGER.info(
                "cached feature archive progress=%d/%d %s %s status=%s message=%s",
                completed,
                total,
                date_value,
                cycle,
                result.status,
                result.message,
            )
            results.append(result)
    summary = {
        "start": start,
        "end": end,
        "cycles": cycles,
        "built": sum(result.status == "built" for result in results),
        "skipped": sum(result.status == "skipped" for result in results),
        "missing_source_archive": sum(result.status == "missing_source_archive" for result in results),
        "results": [asdict(result) for result in results],
    }
    (paths.archive_metadata / "cached_feature_archive_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results
