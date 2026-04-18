"""Archive coverage accounting and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from severewx.config import AppSettings
from severewx.archive.feature_cache import discover_cached_feature_archives
from severewx.models.dataset import (
    HAZARD_TARGETS,
    _real_archive_mask,
    _synthetic_mask,
    discover_feature_archives,
    evaluate_training_guardrails,
)
from severewx.utils.paths import DataPaths


def _reporting_status(status: str) -> str:
    if status == "built_real":
        return "remote_real"
    if status == "built_partial_real":
        return "partial_real"
    return status


def load_archive_metadata(paths: DataPaths) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for metadata_file in sorted(paths.archive_metadata.glob("*/*/*.json")):
        if metadata_file.name in {"historical_archive_summary.json", "archive_coverage_summary.json"}:
            continue
        payloads.append(json.loads(metadata_file.read_text(encoding="utf-8")))
    return payloads


def load_cached_feature_archive_metadata(paths: DataPaths) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for metadata_file in sorted(paths.archive_metadata.glob("*/*/*_feature_cache.json")):
        payloads.append(json.loads(metadata_file.read_text(encoding="utf-8")))
    return payloads


def load_archive_frame(paths: DataPaths) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for archive_file in discover_feature_archives(paths):
        tables.append(pd.read_parquet(archive_file))
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def load_cached_feature_archive_frame(paths: DataPaths) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for archive_file in discover_cached_feature_archives(paths):
        tables.append(pd.read_parquet(archive_file))
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _coverage_rows(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if group_column not in frame.columns:
        return rows
    normalized = frame.copy()
    if group_column == "archive_chunk_status":
        normalized[group_column] = normalized[group_column].astype(str).map(_reporting_status)
    for group_value, subset in normalized.groupby(group_column):
        real_rows = int(_real_archive_mask(subset).sum())
        synthetic_rows = int(_synthetic_mask(subset).sum())
        total_rows = int(len(subset))
        rows.append(
            {
                group_column: int(group_value) if group_column == "lead_day" else group_value,
                "row_count": total_rows,
                "real_row_count": real_rows,
                "synthetic_row_count": synthetic_rows,
                "real_fraction": float(real_rows / max(total_rows, 1)),
            }
        )
    return rows


def archive_coverage_summary(
    paths: DataPaths,
    training_data_summary: dict[str, Any] | None = None,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    metadata_payloads = load_archive_metadata(paths)
    cached_metadata_payloads = load_cached_feature_archive_metadata(paths)
    frame = load_archive_frame(paths)
    cached_frame = load_cached_feature_archive_frame(paths)
    archive_status_counts: dict[str, int] = {}
    for payload in metadata_payloads:
        status = _reporting_status(str(payload.get("archive_status", "unknown")))
        archive_status_counts[status] = archive_status_counts.get(status, 0) + 1

    summary: dict[str, Any] = {
        "archive_chunk_count": len(metadata_payloads),
        "real_chunk_count": int(sum(bool(payload.get("real_data")) for payload in metadata_payloads)),
        "synthetic_or_fallback_chunk_count": int(sum(not bool(payload.get("real_data")) for payload in metadata_payloads)),
        "archive_status_counts": archive_status_counts,
        "archive_status_reporting_counts": {
            "local_real": archive_status_counts.get("local_real", 0),
            "remote_real": archive_status_counts.get("remote_real", 0),
            "partial_real": archive_status_counts.get("partial_real", 0),
            "synthetic_degraded": archive_status_counts.get("synthetic_degraded", 0),
            "failed": archive_status_counts.get("failed", 0),
        },
        "by_date_cycle": metadata_payloads,
        "coverage_by_lead_day": [],
        "coverage_by_region": [],
        "coverage_by_outbreak_class": [],
        "coverage_by_hazard": [],
        "coverage_by_archive_status": [],
        "hazard_label_availability": {},
        "real_archive_rows": 0,
        "synthetic_rows": 0,
        "real_fraction_overall": 0.0,
        "training_real_row_fraction": 0.0,
        "training_synthetic_row_fraction": 0.0,
        "cached_feature_archive_chunk_count": int(len(discover_cached_feature_archives(paths))),
        "cached_feature_archive_row_count": 0,
        "cached_feature_archive_real_row_count": 0,
        "cached_feature_archive_synthetic_row_count": 0,
        "cached_feature_archive_real_fraction": 0.0,
        "cached_feature_archive_by_lead_day": [],
        "cached_feature_archive_by_region": [],
        "cached_feature_archive_by_hazard": [],
        "raw_lifecycle_mode_counts": {},
        "raw_files_processed": 0,
        "raw_files_retained": 0,
        "raw_files_moved": 0,
        "raw_files_deleted": 0,
    }
    raw_mode_counts: dict[str, int] = {}
    for payload in cached_metadata_payloads:
        raw_lifecycle = payload.get("raw_lifecycle", {}) or {}
        mode = str(raw_lifecycle.get("mode", "keep_raw"))
        raw_mode_counts[mode] = raw_mode_counts.get(mode, 0) + 1
        summary["raw_files_processed"] += int(raw_lifecycle.get("raw_files_processed", 0))
        summary["raw_files_retained"] += int(raw_lifecycle.get("raw_files_retained", 0))
        summary["raw_files_moved"] += int(raw_lifecycle.get("raw_files_moved", 0))
        summary["raw_files_deleted"] += int(raw_lifecycle.get("raw_files_deleted", 0))
    summary["raw_lifecycle_mode_counts"] = raw_mode_counts
    if not cached_frame.empty:
        cached_real_mask = _real_archive_mask(cached_frame)
        cached_synthetic_mask = _synthetic_mask(cached_frame)
        summary["cached_feature_archive_row_count"] = int(len(cached_frame))
        summary["cached_feature_archive_real_row_count"] = int(cached_real_mask.sum())
        summary["cached_feature_archive_synthetic_row_count"] = int(cached_synthetic_mask.sum())
        summary["cached_feature_archive_real_fraction"] = float(summary["cached_feature_archive_real_row_count"] / max(len(cached_frame), 1))
        summary["cached_feature_archive_by_lead_day"] = _coverage_rows(cached_frame, "lead_day")
        summary["cached_feature_archive_by_region"] = _coverage_rows(cached_frame, "region")
        cached_hazard_rows: list[dict[str, Any]] = []
        for hazard in HAZARD_TARGETS:
            if hazard in cached_frame.columns:
                positive = cached_frame[hazard].astype(float) > 0
                cached_hazard_rows.append(
                    {
                        "hazard": hazard,
                        "positive_count": int(positive.sum()),
                        "real_positive_count": int((positive & cached_real_mask).sum()),
                        "synthetic_positive_count": int((positive & cached_synthetic_mask).sum()),
                    }
                )
        summary["cached_feature_archive_by_hazard"] = cached_hazard_rows
    if frame.empty:
        return summary

    real_mask = _real_archive_mask(frame)
    synthetic_mask = _synthetic_mask(frame)
    summary["real_archive_rows"] = int(real_mask.sum())
    summary["synthetic_rows"] = int(synthetic_mask.sum())
    summary["real_fraction_overall"] = float(summary["real_archive_rows"] / max(len(frame), 1))
    summary["coverage_by_lead_day"] = _coverage_rows(frame, "lead_day")
    summary["coverage_by_region"] = _coverage_rows(frame, "region")
    summary["coverage_by_archive_status"] = _coverage_rows(frame, "archive_chunk_status")
    if "category" in frame.columns:
        summary["coverage_by_outbreak_class"] = _coverage_rows(frame, "category")
    hazard_rows: list[dict[str, Any]] = []
    for hazard in HAZARD_TARGETS:
        if hazard in frame.columns:
            positive = frame[hazard].astype(float) > 0
            hazard_rows.append(
                {
                    "hazard": hazard,
                    "positive_count": int(positive.sum()),
                    "real_positive_count": int((positive & real_mask).sum()),
                    "synthetic_positive_count": int((positive & synthetic_mask).sum()),
                    "real_positive_fraction": float((positive & real_mask).sum() / max(int(positive.sum()), 1)),
                }
            )
            summary["hazard_label_availability"][hazard] = int(frame[hazard].notna().sum())
    summary["coverage_by_hazard"] = hazard_rows
    if settings is not None:
        summary["archive_guardrails"] = evaluate_training_guardrails(frame, settings)

    if training_data_summary:
        real_rows = float(training_data_summary.get("real_archive_rows", 0))
        synthetic_rows = float(training_data_summary.get("synthetic_rows", 0))
        total = max(real_rows + synthetic_rows, 1.0)
        summary["training_real_row_fraction"] = real_rows / total
        summary["training_synthetic_row_fraction"] = synthetic_rows / total
        summary["training_data_summary"] = training_data_summary
    return summary


def write_archive_coverage_summary(
    paths: DataPaths,
    training_data_summary: dict[str, Any] | None = None,
    settings: AppSettings | None = None,
) -> tuple[Path, Path]:
    summary = archive_coverage_summary(paths, training_data_summary=training_data_summary, settings=settings)
    json_path = paths.archive_metadata / "archive_coverage_summary.json"
    csv_path = paths.archive_metadata / "archive_coverage_by_lead_day.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary.get("coverage_by_lead_day", [])).to_csv(csv_path, index=False)
    return json_path, csv_path
