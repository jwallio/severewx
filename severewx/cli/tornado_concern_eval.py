"""Evaluate a prototype hierarchical tornado-concern diagnostic from saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from severewx.config import load_settings
from severewx.utils.paths import build_paths

SCORE_VARIANTS = ["baseline", "tornado_emphasis", "capped_broad", "learned_gated", "hybrid", "lead_time_calibrated", "v1_triage"]
COMPONENT_VARIANTS = [
    "baseline",
    "core_tornado_gated",
    "overlap_emphasis",
    "scp_core_hybrid",
    "hybrid",
    "tornado_hail_separation_v1",
    "tornado_signal_continuity_v1",
]
CHECKPOINT_COMPONENT_VARIANTS = ["baseline", "tornado_hail_separation_v1"]
SOURCE_VARIANTS = ["baseline", "compact_overlap", "tornado_purity_gate", "contamination_penalty", "hybrid", "failure_targeted_contamination"]
CORE_VARIANTS = ["baseline", "compact_core", "percentile_core", "aligned_core", "hybrid"]
RAW_CORE_VARIANTS = [
    "baseline",
    "masked_core_only",
    "masked_core_stronger",
    "masked_core_calibrated",
    "masked_core_two_stage",
    "masked_core_lower_floor",
    "masked_core_steeper",
    "footprint_discriminating",
]
BROADER_VALIDATION_RAW_CORE_VARIANTS = [
    "baseline",
    "masked_core_two_stage",
    "masked_core_lower_floor",
    "masked_core_steeper",
]
BASELINE_VS_MASKED_CORE_STEEPER_VARIANTS = [
    "baseline",
    "masked_core_steeper",
]
PRIORITY_TIERS = ["sig_tor", "outbreak", "all_tornado", "all"]
_PRIORITY_ORDER = {"sig_tor": 0, "outbreak": 1, "all_tornado": 2}


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    matches = list(directory.glob(pattern))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _latest_prediction_for_date(outputs_dir: Path, date: str) -> Path | None:
    return _latest_matching_file(outputs_dir, f"forecast_products_{date}_*.nc")


def _latest_forecast_metadata_for_date(outputs_dir: Path, date: str) -> Path | None:
    candidate = outputs_dir / f"forecast_metadata_{date}_00.json"
    if candidate.exists():
        return candidate
    return _latest_matching_file(outputs_dir, f"forecast_metadata_{date}_*.json")


def _verification_for_date(verification_dir: Path, date: str) -> Path | None:
    candidate = verification_dir / f"{date}_verification.json"
    if candidate.exists():
        return candidate
    return _latest_matching_file(verification_dir, f"{date}*_verification*.json")


def _load_dates(args: argparse.Namespace) -> list[str]:
    dates = list(args.dates or [])
    if args.dates_file:
        dates.extend(
            [
                line.strip()
                for line in Path(args.dates_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        )
    deduped = list(dict.fromkeys(dates))
    if not deduped:
        raise ValueError("at least one init date is required")
    return deduped


def write_init_dates_file(path: Path, dates: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(dates) + ("\n" if dates else ""), encoding="utf-8")


def _normalize_date_string(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid date: {value}")
    return str(parsed.strftime("%Y-%m-%d"))


def _date_in_range(date_str: str, start: str | None = None, end: str | None = None) -> bool:
    normalized = _normalize_date_string(date_str)
    if normalized is None:
        return False
    start_norm = _normalize_date_string(start)
    end_norm = _normalize_date_string(end)
    if start_norm is not None and normalized < start_norm:
        return False
    if end_norm is not None and normalized > end_norm:
        return False
    return True


def _real_ingest_from_interim_summary(interim_dir: Path, init_date: str) -> bool:
    summary_path = interim_dir / f"ingest_summary_{init_date}_00.json"
    if not summary_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    source = str(payload.get("source") or "unknown")
    source_mode = str(payload.get("source_mode") or "unknown")
    real_ingest_available = bool(payload.get("real_ingest_available", False))
    return source not in {"synthetic", "synthetic_fallback", "synthetic_degraded"} and source_mode == "real" and real_ingest_available


def _load_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _forecast_metadata_for_date(outputs_dir: Path, date: str) -> dict[str, Any]:
    path = _latest_forecast_metadata_for_date(outputs_dir, date)
    payload = _load_json_if_exists(path)
    payload["_path"] = str(path) if path is not None else ""
    return payload


def _archive_metadata_for_date(outputs_dir: Path, date: str) -> dict[str, Any]:
    root = outputs_dir.parent / "processed" / "archive_metadata"
    year = date[:4]
    candidate = root / year / date / "00.json"
    payload = _load_json_if_exists(candidate if candidate.exists() else None)
    payload["_path"] = str(candidate) if candidate.exists() else ""
    return payload


def _staged_input_inventory_for_date(outputs_dir: Path, date: str) -> dict[str, Any]:
    raw_root = outputs_dir.parent / "raw" / "staged_gfs" / date / "00"
    files = sorted(path for path in raw_root.glob("*") if path.is_file()) if raw_root.exists() else []
    return {
        "staged_input_dir_present": raw_root.exists(),
        "staged_input_file_count": len(files),
        "staged_input_dir": str(raw_root),
        "staged_input_sample_files": [path.name for path in files[:5]],
    }


def _tornado_relevant_dates_from_labels(labels_dir: Path) -> set[str]:
    dates: set[str] = set()
    for path in sorted(labels_dir.glob("outbreaks_*.parquet")):
        try:
            frame = pd.read_parquet(path, columns=["date", "tornado_outbreak", "significant_tornado_support"])
        except Exception:
            continue
        relevant = frame.loc[
            frame["tornado_outbreak"].fillna(0).astype(int).gt(0)
            | frame["significant_tornado_support"].fillna(0).astype(int).gt(0),
            "date",
        ]
        dates.update(pd.to_datetime(relevant, errors="coerce").dropna().dt.strftime("%Y-%m-%d").tolist())
    return dates


def _tornado_relevant_dates_from_spc_reports(labels_dir: Path) -> set[str]:
    dates: set[str] = set()
    for path in sorted(labels_dir.glob("spc_reports_*.parquet")):
        try:
            frame = pd.read_parquet(path, columns=["date", "hazard"])
        except Exception:
            continue
        tornado_rows = frame.loc[frame["hazard"].astype(str).str.lower().eq("tornado"), "date"]
        dates.update(pd.to_datetime(tornado_rows, errors="coerce").dropna().dt.strftime("%Y-%m-%d").tolist())
    return dates


def _outbreak_signal_by_date(labels_dir: Path | None, start: str | None = None, end: str | None = None) -> dict[str, dict[str, int]]:
    rows: list[pd.DataFrame] = []
    if labels_dir is None:
        return {}
    for path in sorted(labels_dir.glob("outbreaks_*.parquet")):
        try:
            frame = pd.read_parquet(path, columns=["date", "tornado_outbreak", "significant_tornado_support"])
        except Exception:
            continue
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["date"])
        if start is not None:
            frame = frame.loc[frame["date"] >= start]
        if end is not None:
            frame = frame.loc[frame["date"] <= end]
        if frame.empty:
            continue
        grouped = (
            frame.groupby("date", as_index=False)[["tornado_outbreak", "significant_tornado_support"]]
            .max()
            .fillna(0)
        )
        grouped = grouped.loc[
            grouped["tornado_outbreak"].fillna(0).astype(int).gt(0)
            | grouped["significant_tornado_support"].fillna(0).astype(int).gt(0)
        ]
        if grouped.empty:
            continue
        rows.append(grouped)
    if not rows:
        return {}
    combined = pd.concat(rows, ignore_index=True)
    combined = (
        combined.groupby("date", as_index=False)[["tornado_outbreak", "significant_tornado_support"]]
        .max()
        .fillna(0)
    )
    result: dict[str, dict[str, int]] = {}
    for row in combined.to_dict(orient="records"):
        result[str(row["date"])] = {
            "tornado_outbreak": int(row.get("tornado_outbreak", 0) or 0),
            "significant_tornado_support": int(row.get("significant_tornado_support", 0) or 0),
        }
    return result


def _spc_tornado_report_counts(
    labels_dir: Path | None,
    interim_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, int]:
    counts_by_date: dict[str, int] = {}
    sources: list[Path] = []
    if labels_dir is not None:
        sources.extend(sorted(labels_dir.glob("spc_reports_*.parquet")))
    if interim_dir is not None:
        sources.extend(sorted(interim_dir.glob("spc_reports_real*.parquet")))
    for path in sources:
        try:
            frame = pd.read_parquet(path, columns=["date", "hazard"])
        except Exception:
            continue
        frame = frame.loc[frame["hazard"].astype(str).str.lower().eq("tornado")].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["date"])
        if start is not None:
            frame = frame.loc[frame["date"] >= start]
        if end is not None:
            frame = frame.loc[frame["date"] <= end]
        if frame.empty:
            continue
        grouped = frame.groupby("date").size()
        for date, count in grouped.items():
            counts_by_date[str(date)] = max(counts_by_date.get(str(date), 0), int(count))
    return counts_by_date


def _verification_date_metadata(
    verification_dir: Path,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, dict[str, bool]]:
    metadata: dict[str, dict[str, bool]] = {}
    for verification_path in sorted(verification_dir.glob("*_verification.json")):
        try:
            verification_payload = json.loads(verification_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        init_date = str((verification_payload.get("run_summary", {}) or {}).get("init_date") or verification_path.name.split("_verification", 1)[0])
        init_date = _normalize_date_string(init_date)
        if init_date is None or not _date_in_range(init_date, start=start, end=end):
            continue
        _, is_real_ingest = _ingest_source_info(verification_payload)
        per_day = verification_payload.get("per_day", []) or []
        categories = [str(row.get("observed_category", "")) for row in per_day]
        has_tornado_relevant_day = any(
            "tornado_outbreak" in category or "significant_tornado_outbreak" in category
            for category in categories
        )
        metadata[init_date] = {
            "is_real_ingest": bool(is_real_ingest),
            "has_tornado_relevant_day": bool(has_tornado_relevant_day),
        }
    return metadata


def _interim_real_ingest_flags(interim_dir: Path | None, start: str | None = None, end: str | None = None) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    if interim_dir is None:
        return flags
    for summary_path in sorted(interim_dir.glob("ingest_summary_*_00.json")):
        parts = summary_path.stem.split("_")
        if len(parts) < 4:
            continue
        init_date = _normalize_date_string(parts[2])
        if init_date is None or not _date_in_range(init_date, start=start, end=end):
            continue
        flags[init_date] = _real_ingest_from_interim_summary(interim_dir, init_date)
    return flags


def _assign_priority_tier(
    significant_tornado_support: int,
    tornado_outbreak: int,
    has_spc_tornado_reports: bool,
    has_other_tornado_evidence: bool,
) -> str:
    # Priority precedence is deterministic:
    # 1. Any significant tornado-support evidence => sig_tor
    # 2. Otherwise any tornado-outbreak evidence => outbreak
    # 3. Otherwise any remaining tornado-relevant discovery evidence => all_tornado
    if significant_tornado_support > 0:
        return "sig_tor"
    if tornado_outbreak > 0:
        return "outbreak"
    if has_spc_tornado_reports or has_other_tornado_evidence:
        return "all_tornado"
    return "all_tornado"


def _priority_tier_included(priority_tier: str, requested_tier: str) -> bool:
    if requested_tier == "all":
        return True
    return priority_tier == requested_tier


def _classify_generation_failure(stage: str, message: str) -> str:
    lowered = (message or "").lower()
    if any(token in lowered for token in ["not supported", "unsupported", "outside archive", "date not supported", "archive window"]):
        return "date_not_supported_by_archive"
    if any(token in lowered for token in ["missing label", "labels_", "outbreaks_", "observed", "verification input", "downstream"]):
        return "downstream_eval_inputs_missing" if stage == "verification" else "missing_local_input_data"
    if any(token in lowered for token in ["missing", "not found", "no such file", "no files", "usable_file_count", "staged", "raw", "input data"]):
        return "missing_local_input_data"
    return "verification_failed" if stage == "verification" else "forecast_generation_failed"


def _derive_real_ingest_failure_detail(row: dict[str, Any]) -> str:
    if bool(row.get("real_ingest_confirmed", False)):
        return ""
    if bool(row.get("verification_metadata_present", False)) and not bool(row.get("verification_real_ingest", False)):
        return "verification_metadata_not_real"
    if bool(row.get("interim_ingest_summary_present", False)) and not bool(row.get("interim_ingest_real_confirmed", False)):
        source = str(row.get("interim_source", "") or "").lower()
        source_mode = str(row.get("interim_source_mode", "") or "").lower()
        real_flag = bool(row.get("interim_real_ingest_available_flag", False))
        if source in {"synthetic", "synthetic_fallback", "synthetic_degraded"}:
            return "interim_summary_synthetic_source"
        if source_mode and source_mode != "real":
            return "interim_summary_nonreal_mode"
        if not real_flag:
            return "interim_summary_real_flag_false"
        return "interim_summary_not_real"
    if bool(row.get("forecast_metadata_present", False)):
        source = str(row.get("forecast_metadata_source", "") or "").lower()
        source_mode = str(row.get("forecast_metadata_source_mode", "") or "").lower()
        real_flag = bool(row.get("forecast_metadata_real_ingest_available", False))
        if source in {"synthetic", "synthetic_fallback", "synthetic_degraded"}:
            return "forecast_metadata_synthetic_source"
        if source_mode and source_mode != "real":
            return "forecast_metadata_nonreal_mode"
        if not real_flag:
            return "forecast_metadata_real_flag_false"
        return "forecast_metadata_not_real"
    if bool(row.get("archive_metadata_present", False)):
        archive_status = str(row.get("archive_status", "") or "").lower()
        archive_source = str(row.get("archive_forecast_source", "") or "").lower()
        archive_real = bool(row.get("archive_real_data", False))
        if archive_status in {"synthetic_degraded", "failed"}:
            return f"archive_status_{archive_status}"
        if archive_source in {"synthetic", "synthetic_fallback", "synthetic_degraded"}:
            return "archive_metadata_synthetic_source"
        if not archive_real:
            return "archive_metadata_real_flag_false"
        return "archive_metadata_not_real"
    if bool(row.get("staged_input_dir_present", False)) or int(row.get("staged_input_file_count", 0) or 0) > 0:
        return "staged_inputs_present_without_real_provenance"
    return "no_local_real_ingest_evidence"


def _derive_failure_reason(row: dict[str, Any]) -> str:
    if bool(row.get("final_ready", False)):
        return ""
    if not bool(row.get("real_ingest_confirmed", False)):
        return "not_real_ingest_confirmed"
    if not bool(row.get("forecast_artifacts_present", False)):
        if bool(row.get("forecast_attempted", False)):
            return _classify_generation_failure("forecast", str(row.get("failure_detail", "")))
        return "missing_forecast_artifacts"
    if not bool(row.get("verification_artifacts_present", False)):
        if bool(row.get("verification_attempted", False)):
            return _classify_generation_failure("verification", str(row.get("failure_detail", "")))
        return "missing_verification_artifacts"
    return "downstream_eval_inputs_missing"


def build_tornado_concern_candidate_rows(
    outputs_dir: Path,
    verification_dir: Path,
    labels_dir: Path | None = None,
    interim_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    priority_tier: str = "all",
) -> pd.DataFrame:
    if priority_tier not in PRIORITY_TIERS:
        raise ValueError(f"unsupported priority tier: {priority_tier}")
    start_norm = _normalize_date_string(start)
    end_norm = _normalize_date_string(end)
    verification_meta = _verification_date_metadata(verification_dir, start=start_norm, end=end_norm)
    outbreak_signal = _outbreak_signal_by_date(labels_dir, start=start_norm, end=end_norm)
    spc_tornado_counts = _spc_tornado_report_counts(labels_dir, interim_dir=interim_dir, start=start_norm, end=end_norm)
    interim_real_flags = _interim_real_ingest_flags(interim_dir, start=start_norm, end=end_norm)
    verification_tornado_dates = {
        date for date, metadata in verification_meta.items() if bool(metadata.get("has_tornado_relevant_day", False))
    }
    candidate_dates = sorted(verification_tornado_dates | set(outbreak_signal) | set(spc_tornado_counts))

    rows: list[dict[str, Any]] = []
    for date in candidate_dates:
        verification_row = verification_meta.get(date, {})
        outbreak_row = outbreak_signal.get(date, {})
        verification_path = _verification_for_date(verification_dir, date)
        forecast_path = _latest_prediction_for_date(outputs_dir, date)
        forecast_metadata = _forecast_metadata_for_date(outputs_dir, date)
        archive_metadata = _archive_metadata_for_date(outputs_dir, date)
        staged_inventory = _staged_input_inventory_for_date(outputs_dir, date)
        tornado_report_count = int(spc_tornado_counts.get(date, 0) or 0)
        has_spc_tornado_reports = tornado_report_count > 0
        tornado_outbreak = int(outbreak_row.get("tornado_outbreak", 0) or 0)
        significant_tornado_support = int(outbreak_row.get("significant_tornado_support", 0) or 0)
        candidate_source_verification_metadata = bool(verification_row.get("has_tornado_relevant_day", False))
        candidate_source_outbreak_parquet = bool(date in outbreak_signal)
        candidate_source_spc_reports = bool(has_spc_tornado_reports)
        verification_metadata_present = date in verification_meta or verification_path is not None
        verification_real_ingest = bool(verification_row.get("is_real_ingest", False))
        interim_ingest_real_confirmed = bool(interim_real_flags.get(date, False))
        interim_summary_path = interim_dir / f"ingest_summary_{date}_00.json" if interim_dir is not None else None
        interim_payload = _load_json_if_exists(interim_summary_path)
        interim_ingest_summary_present = bool(interim_payload)
        real_ingest_confirmed = bool(verification_real_ingest or interim_ingest_real_confirmed)
        has_other_tornado_evidence = bool(candidate_source_verification_metadata or candidate_source_outbreak_parquet)
        assigned_tier = _assign_priority_tier(
            significant_tornado_support=significant_tornado_support,
            tornado_outbreak=tornado_outbreak,
            has_spc_tornado_reports=has_spc_tornado_reports,
            has_other_tornado_evidence=has_other_tornado_evidence,
        )
        if not _priority_tier_included(assigned_tier, priority_tier):
            continue
        forecast_present = forecast_path is not None
        verification_present = verification_path is not None
        fully_evaluable_initial = bool(real_ingest_confirmed and forecast_present and verification_present)
        forecast_ingest_summary = (forecast_metadata.get("ingest_summary", {}) or {}) if forecast_metadata else {}
        row = {
            "date": date,
            "init_date": date,
            "priority_tier": assigned_tier,
            "has_spc_tornado_reports": bool(has_spc_tornado_reports),
            "tornado_report_count": tornado_report_count,
            "tornado_outbreak": tornado_outbreak,
            "significant_tornado_support": significant_tornado_support,
            "candidate_source_verification_metadata": candidate_source_verification_metadata,
            "candidate_source_outbreak_parquet": candidate_source_outbreak_parquet,
            "candidate_source_spc_reports": candidate_source_spc_reports,
            "verification_metadata_present": verification_metadata_present,
            "verification_real_ingest": verification_real_ingest,
            "interim_ingest_summary_present": interim_ingest_summary_present,
            "interim_source": str(interim_payload.get("source", "")) if interim_payload else "",
            "interim_source_mode": str(interim_payload.get("source_mode", "")) if interim_payload else "",
            "interim_real_ingest_available_flag": bool(interim_payload.get("real_ingest_available", False)) if interim_payload else False,
            "interim_ingest_real_confirmed": interim_ingest_real_confirmed,
            "real_ingest_confirmed": real_ingest_confirmed,
            "forecast_metadata_present": bool(forecast_metadata and forecast_metadata.get("_path")),
            "forecast_metadata_source": str(forecast_ingest_summary.get("source", "")),
            "forecast_metadata_source_mode": str(forecast_ingest_summary.get("source_mode", "")),
            "forecast_metadata_real_ingest_available": bool(forecast_ingest_summary.get("real_ingest_available", False)),
            "forecast_metadata_path": str(forecast_metadata.get("_path", "")),
            "archive_metadata_present": bool(archive_metadata and archive_metadata.get("_path")),
            "archive_status": str(archive_metadata.get("archive_status", "")),
            "archive_forecast_source": str(archive_metadata.get("forecast_source", "")),
            "archive_forecast_source_origin": str(archive_metadata.get("forecast_source_origin", "")),
            "archive_real_data": bool(archive_metadata.get("real_data", False)),
            "archive_metadata_path": str(archive_metadata.get("_path", "")),
            "forecast_artifacts_present": bool(forecast_present),
            "verification_artifacts_present": bool(verification_present),
            "fully_evaluable_initial": fully_evaluable_initial,
            "final_ready": fully_evaluable_initial,
            "forecast_attempted": False,
            "verification_attempted": False,
            "forecast_generated": False,
            "verification_generated": False,
            "failure_detail": "",
            "expected_forecast_artifact_pattern": str(outputs_dir / f"forecast_products_{date}_*.nc"),
            "expected_verification_artifact_path": str(verification_dir / f"{date}_verification.json"),
            "forecast_generation_command": f"python -m severewx.cli.run_forecast --date {date} --cycle 00",
            "verification_generation_command": f"python -m severewx.cli.verify_day --date {date}",
        }
        row.update(staged_inventory)
        row["real_ingest_failure_detail"] = _derive_real_ingest_failure_detail(row)
        row["failure_reason"] = _derive_failure_reason(row)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "init_date",
                "priority_tier",
                "has_spc_tornado_reports",
                "tornado_report_count",
                "tornado_outbreak",
                "significant_tornado_support",
                "candidate_source_verification_metadata",
                "candidate_source_outbreak_parquet",
                "candidate_source_spc_reports",
                "verification_real_ingest",
                "verification_metadata_present",
                "interim_ingest_summary_present",
                "interim_source",
                "interim_source_mode",
                "interim_real_ingest_available_flag",
                "interim_ingest_real_confirmed",
                "real_ingest_confirmed",
                "forecast_metadata_present",
                "forecast_metadata_source",
                "forecast_metadata_source_mode",
                "forecast_metadata_real_ingest_available",
                "forecast_metadata_path",
                "archive_metadata_present",
                "archive_status",
                "archive_forecast_source",
                "archive_forecast_source_origin",
                "archive_real_data",
                "archive_metadata_path",
                "staged_input_dir_present",
                "staged_input_file_count",
                "staged_input_dir",
                "forecast_artifacts_present",
                "verification_artifacts_present",
                "fully_evaluable_initial",
                "final_ready",
                "forecast_attempted",
                "verification_attempted",
                "forecast_generated",
                "verification_generated",
                "failure_detail",
                "failure_reason",
                "real_ingest_failure_detail",
                "expected_forecast_artifact_pattern",
                "expected_verification_artifact_path",
                "forecast_generation_command",
                "verification_generation_command",
            ]
        )
    frame["priority_sort_key"] = frame["priority_tier"].map(_PRIORITY_ORDER).fillna(99).astype(int)
    return frame.sort_values(["priority_sort_key", "date"], kind="mergesort").reset_index(drop=True)


def select_real_case_init_dates(
    outputs_dir: Path,
    verification_dir: Path,
    labels_dir: Path | None = None,
    interim_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    priority_tier: str = "all",
) -> list[str]:
    frame = build_tornado_concern_candidate_rows(
        outputs_dir,
        verification_dir,
        labels_dir=labels_dir,
        interim_dir=interim_dir,
        start=start,
        end=end,
        priority_tier=priority_tier,
    )
    if frame.empty:
        return []
    selected = sorted(frame.loc[frame["real_ingest_confirmed"] & frame["forecast_artifacts_present"], "date"].astype(str).tolist())
    return list(dict.fromkeys(selected))


def audit_real_case_init_dates(
    outputs_dir: Path,
    verification_dir: Path,
    labels_dir: Path | None = None,
    interim_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    priority_tier: str = "all",
    sample_size: int = 5,
) -> dict[str, Any]:
    start_norm = _normalize_date_string(start)
    end_norm = _normalize_date_string(end)
    verification_meta = _verification_date_metadata(verification_dir, start=start_norm, end=end_norm)
    verification_dates = set(verification_meta)
    outbreak_dates = set(_outbreak_signal_by_date(labels_dir, start=start_norm, end=end_norm)) if labels_dir is not None else set()
    spc_tornado_dates = set(_spc_tornado_report_counts(labels_dir, interim_dir=interim_dir, start=start_norm, end=end_norm))
    frame = build_tornado_concern_candidate_rows(
        outputs_dir,
        verification_dir,
        labels_dir=labels_dir,
        interim_dir=interim_dir,
        start=start_norm,
        end=end_norm,
        priority_tier=priority_tier,
    )
    missing_forecast = sorted(frame.loc[~frame["forecast_artifacts_present"], "date"].astype(str).tolist())
    not_real_ingest = sorted(frame.loc[~frame["real_ingest_confirmed"], "date"].astype(str).tolist())
    final_selected = select_real_case_init_dates(
        outputs_dir,
        verification_dir,
        labels_dir=labels_dir,
        interim_dir=interim_dir,
        start=start_norm,
        end=end_norm,
        priority_tier=priority_tier,
    )
    return {
        "candidate_dates_from_verification_jsons": len(verification_dates),
        "candidate_dates_from_outbreaks_parquet": len(outbreak_dates),
        "candidate_dates_from_spc_reports_parquet": len(spc_tornado_dates),
        "dates_with_local_forecast_artifact_present": int(frame["forecast_artifacts_present"].sum()) if not frame.empty else 0,
        "dates_with_real_ingest_confirmed": int(frame["real_ingest_confirmed"].sum()) if not frame.empty else 0,
        "dates_with_tornado_relevant_label_or_report_evidence": len(frame),
        "final_selected_dates": len(final_selected),
        "missing_forecast_examples": missing_forecast[:sample_size],
        "not_real_ingest_examples": not_real_ingest[:sample_size],
        "missing_tornado_evidence_examples": [],
        "final_selected_examples": final_selected[:sample_size],
    }


def missing_real_case_forecast_artifacts(
    outputs_dir: Path,
    verification_dir: Path,
    labels_dir: Path | None = None,
    interim_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    priority_tier: str = "all",
) -> list[dict[str, str]]:
    frame = build_tornado_concern_candidate_rows(
        outputs_dir,
        verification_dir,
        labels_dir=labels_dir,
        interim_dir=interim_dir,
        start=start,
        end=end,
        priority_tier=priority_tier,
    )
    rows: list[dict[str, str]] = []
    for row in frame.to_dict(orient="records"):
        if not bool(row.get("real_ingest_confirmed", False)):
            continue
        if bool(row.get("forecast_artifacts_present", False)):
            continue
        rows.append(
            {
                "init_date": str(row["date"]),
                "expected_artifact_pattern": str(row["expected_forecast_artifact_pattern"]),
                "generation_command": str(row["forecast_generation_command"]),
            }
        )
    return rows


def discover_real_tornado_relevant_candidate_dates(
    verification_dir: Path,
    labels_dir: Path | None = None,
    interim_dir: Path | None = None,
    outputs_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    priority_tier: str = "all",
) -> list[str]:
    candidate_rows = build_tornado_concern_candidate_rows(
        outputs_dir or verification_dir.parent,
        verification_dir,
        labels_dir=labels_dir,
        interim_dir=interim_dir,
        start=start,
        end=end,
        priority_tier=priority_tier,
    )
    if candidate_rows.empty:
        return []
    return candidate_rows.loc[candidate_rows["real_ingest_confirmed"], "date"].astype(str).tolist()


def real_case_artifact_readiness(
    outputs_dir: Path,
    verification_dir: Path,
    labels_dir: Path | None = None,
    interim_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    priority_tier: str = "all",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = build_tornado_concern_candidate_rows(
        outputs_dir,
        verification_dir,
        labels_dir=labels_dir,
        interim_dir=interim_dir,
        start=start,
        end=end,
        priority_tier=priority_tier,
    )
    for row in frame.to_dict(orient="records"):
        rows.append(
            {
                "init_date": str(row["date"]),
                "forecast_present": bool(row["forecast_artifacts_present"]),
                "verification_present": bool(row["verification_artifacts_present"]),
                "fully_evaluable": bool(row["final_ready"]),
                "blocker": str(row["failure_reason"]),
                "expected_forecast_artifact_pattern": str(row["expected_forecast_artifact_pattern"]),
                "expected_verification_artifact_path": str(row["expected_verification_artifact_path"]),
                "forecast_generation_command": str(row["forecast_generation_command"]),
                "verification_generation_command": str(row["verification_generation_command"]),
            }
        )
    return rows


def summarize_real_case_backfill(rows: list[dict[str, Any]], sample_size: int = 5) -> dict[str, Any]:
    failed_rows = [row for row in rows if not row.get("fully_evaluable", False)]
    forecast_present = [
        bool(row.get("forecast_present", row.get("forecast_present_initial", False) or row.get("forecast_generated", False)))
        for row in rows
    ]
    verification_present = [
        bool(row.get("verification_present", row.get("verification_present_initial", False) or row.get("verification_generated", False)))
        for row in rows
    ]
    return {
        "candidate_dates_total": len(rows),
        "forecast_artifacts_already_present": sum(1 for row in rows if row.get("forecast_present_initial", row.get("forecast_present", False))),
        "forecast_artifacts_generated": sum(1 for row in rows if row.get("forecast_generated", False)),
        "forecast_artifacts_failed": sum(
            1 for row, is_present in zip(rows, forecast_present, strict=False) if row.get("forecast_attempted", False) and not is_present
        ),
        "verification_artifacts_already_present": sum(1 for row in rows if row.get("verification_present_initial", row.get("verification_present", False))),
        "verification_artifacts_generated": sum(1 for row in rows if row.get("verification_generated", False)),
        "verification_artifacts_failed": sum(
            1 for row, is_present in zip(rows, verification_present, strict=False) if row.get("verification_attempted", False) and not is_present
        ),
        "fully_evaluable_dates_total": sum(1 for row in rows if row.get("fully_evaluable", False)),
        "newly_added_ready_dates": sum(
            1
            for row in rows
            if row.get("fully_evaluable", False) and not row.get("fully_evaluable_initial", row.get("fully_evaluable", False))
        ),
        "failed_dates_total": len(failed_rows),
        "failed_dates_examples": [f"{row['init_date']}:{row.get('blocker','')}" for row in failed_rows[:sample_size]],
        "failed_dates_examples_with_reason": [f"{row['init_date']}:{row.get('blocker','')}" for row in failed_rows[:sample_size]],
    }


def summarize_tornado_concern_coverage(rows: pd.DataFrame, sample_size: int = 5) -> dict[str, Any]:
    if rows.empty:
        return {
            "candidate_dates_total": 0,
            "sig_tor_candidates_total": 0,
            "outbreak_candidates_total": 0,
            "all_tornado_candidates_total": 0,
            "real_ingest_confirmed_total": 0,
            "forecast_artifacts_already_present": 0,
            "forecast_artifacts_generated": 0,
            "forecast_artifacts_failed": 0,
            "verification_artifacts_already_present": 0,
            "verification_artifacts_generated": 0,
            "verification_artifacts_failed": 0,
            "fully_evaluable_dates_total": 0,
            "failed_dates_total": 0,
            "newly_added_ready_dates": 0,
            "failed_dates_examples_with_reason": [],
        }
    failed_rows = rows.loc[~rows["final_ready"].fillna(False)].copy()
    return {
        "candidate_dates_total": len(rows),
        "sig_tor_candidates_total": int((rows["priority_tier"] == "sig_tor").sum()),
        "outbreak_candidates_total": int((rows["priority_tier"] == "outbreak").sum()),
        "all_tornado_candidates_total": int((rows["priority_tier"] == "all_tornado").sum()),
        "real_ingest_confirmed_total": int(rows["real_ingest_confirmed"].fillna(False).sum()),
        "forecast_artifacts_already_present": int(rows["forecast_artifacts_present_initial"].fillna(rows["forecast_artifacts_present"]).sum())
        if "forecast_artifacts_present_initial" in rows.columns
        else int(rows["forecast_artifacts_present"].fillna(False).sum()),
        "forecast_artifacts_generated": int(rows["forecast_generated"].fillna(False).sum()) if "forecast_generated" in rows.columns else 0,
        "forecast_artifacts_failed": int(
            ((rows["forecast_attempted"].fillna(False)) & (~rows["forecast_artifacts_present"].fillna(False))).sum()
        )
        if "forecast_attempted" in rows.columns
        else 0,
        "verification_artifacts_already_present": int(rows["verification_artifacts_present_initial"].fillna(rows["verification_artifacts_present"]).sum())
        if "verification_artifacts_present_initial" in rows.columns
        else int(rows["verification_artifacts_present"].fillna(False).sum()),
        "verification_artifacts_generated": int(rows["verification_generated"].fillna(False).sum()) if "verification_generated" in rows.columns else 0,
        "verification_artifacts_failed": int(
            ((rows["verification_attempted"].fillna(False)) & (~rows["verification_artifacts_present"].fillna(False))).sum()
        )
        if "verification_attempted" in rows.columns
        else 0,
        "fully_evaluable_dates_total": int(rows["final_ready"].fillna(False).sum()),
        "failed_dates_total": len(failed_rows),
        "newly_added_ready_dates": int(
            ((rows["final_ready"].fillna(False)) & (~rows["fully_evaluable_initial"].fillna(False))).sum()
        )
        if "fully_evaluable_initial" in rows.columns
        else 0,
        "failed_dates_examples_with_reason": [
            f"{row['date']}:{row['failure_reason']}" for row in failed_rows.head(sample_size).to_dict(orient="records")
        ],
    }


def select_tornado_concern_recovery_candidates(
    rows: pd.DataFrame,
    *,
    priority_tier: str = "all",
    failure_detail: str = "no_local_real_ingest_evidence",
    max_dates: int | None = None,
) -> pd.DataFrame:
    if priority_tier not in PRIORITY_TIERS:
        raise ValueError(f"unsupported priority tier: {priority_tier}")
    selected = rows.copy()
    if selected.empty:
        return selected
    selected = selected.loc[selected["failure_reason"].astype(str).eq("not_real_ingest_confirmed")].copy()
    if failure_detail != "all":
        selected = selected.loc[selected["real_ingest_failure_detail"].astype(str).eq(failure_detail)].copy()
    if priority_tier != "all":
        selected = selected.loc[selected["priority_tier"].astype(str).eq(priority_tier)].copy()
    if "priority_sort_key" not in selected.columns:
        selected["priority_sort_key"] = selected["priority_tier"].map(_PRIORITY_ORDER).fillna(99).astype(int)
    selected = selected.sort_values(["priority_sort_key", "date"], kind="mergesort").reset_index(drop=True)
    if max_dates is not None:
        selected = selected.head(int(max_dates)).copy()
    return selected


def derive_tornado_concern_recovery_status(
    initial_row: dict[str, Any],
    final_row: dict[str, Any],
    *,
    staging_attempted: bool = False,
    staging_succeeded: bool = False,
    forecast_attempted: bool = False,
    forecast_succeeded: bool = False,
    verification_attempted: bool = False,
    verification_succeeded: bool = False,
    dry_run: bool = False,
) -> tuple[str, str]:
    if dry_run:
        return "dry_run", "no mutation performed"
    if bool(final_row.get("final_ready", False)):
        return "recovered_ready", "forecast, provenance, and verification are now ready"
    if bool(final_row.get("real_ingest_confirmed", False)):
        return "recovered_real_ingest_only", "real-ingest evidence recovered but readiness remains incomplete"
    if staging_attempted and not staging_succeeded:
        detail = str(final_row.get("real_ingest_failure_detail", "") or initial_row.get("real_ingest_failure_detail", ""))
        if detail in {"date_not_supported_by_archive", "no_supported_historical_source"}:
            return "no_supported_historical_source", detail
        return "staging_failed", detail or "staging did not create usable staged inputs"
    if forecast_attempted and not forecast_succeeded:
        return "forecast_failed", str(final_row.get("failure_detail", "") or initial_row.get("failure_detail", "") or final_row.get("failure_reason", "forecast_failed"))
    if verification_attempted and not verification_succeeded:
        return "verification_failed", str(final_row.get("failure_detail", "") or initial_row.get("failure_detail", "") or final_row.get("failure_reason", "verification_failed"))
    if str(final_row.get("real_ingest_failure_detail", "")) == "no_local_real_ingest_evidence":
        return "still_blocked_no_real_evidence", "no local real-ingest provenance was recovered"
    return "still_blocked_no_real_evidence", str(final_row.get("real_ingest_failure_detail", "") or final_row.get("failure_reason", "still blocked"))


def write_tornado_concern_candidate_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.copy()
    if "priority_sort_key" in ordered.columns:
        ordered = ordered.drop(columns=["priority_sort_key"])
    ordered.to_csv(path, index=False)


def write_tornado_concern_coverage_markdown(
    path: Path,
    frame: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    priority_tier: str = "all",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_tornado_concern_coverage(frame)
    tier_breakdown = (
        frame.groupby("priority_tier", as_index=False)
        .agg(
            union_candidates=("date", "count"),
            real_ingest_confirmed=("real_ingest_confirmed", "sum"),
            forecast_ready=("forecast_artifacts_present", "sum"),
            verification_ready=("verification_artifacts_present", "sum"),
            fully_evaluable=("final_ready", "sum"),
        )
        .sort_values(["priority_tier"], kind="mergesort")
        if not frame.empty
        else pd.DataFrame(columns=["priority_tier", "union_candidates", "real_ingest_confirmed", "forecast_ready", "verification_ready", "fully_evaluable"])
    )
    failure_breakdown = (
        frame.assign(failure_reason=frame["failure_reason"].replace("", "ready"))
        .groupby("failure_reason", as_index=False)
        .agg(count=("date", "count"))
        .sort_values(["count", "failure_reason"], ascending=[False, True], kind="mergesort")
        if not frame.empty
        else pd.DataFrame(columns=["failure_reason", "count"])
    )
    real_ingest_failure_detail_breakdown = (
        frame.loc[frame["failure_reason"].eq("not_real_ingest_confirmed")]
        .assign(real_ingest_failure_detail=frame["real_ingest_failure_detail"].replace("", "unspecified"))
        .groupby("real_ingest_failure_detail", as_index=False)
        .agg(count=("date", "count"))
        .sort_values(["count", "real_ingest_failure_detail"], ascending=[False, True], kind="mergesort")
        if not frame.empty
        else pd.DataFrame(columns=["real_ingest_failure_detail", "count"])
    )
    top_blocked = (
        frame.loc[~frame["final_ready"].fillna(False), ["date", "priority_tier", "failure_reason"]]
        .sort_values(["date"], kind="mergesort")
        .head(10)
        if not frame.empty
        else pd.DataFrame(columns=["date", "priority_tier", "failure_reason"])
    )
    counts_table = pd.DataFrame(
        [
            {"metric": "union_candidates", "value": summary["candidate_dates_total"]},
            {"metric": "real_ingest_confirmed", "value": summary["real_ingest_confirmed_total"]},
            {"metric": "forecast_ready", "value": int(frame["forecast_artifacts_present"].fillna(False).sum()) if not frame.empty else 0},
            {"metric": "verification_ready", "value": int(frame["verification_artifacts_present"].fillna(False).sum()) if not frame.empty else 0},
            {"metric": "fully_evaluable", "value": summary["fully_evaluable_dates_total"]},
        ]
    )
    lines = [
        "# Tornado Concern Coverage Summary",
        "",
        f"- Date range: {start or 'all'} to {end or 'all'}",
        f"- Priority tier filter: {priority_tier}",
        "",
        "## Counts",
        "",
        _markdown_table(counts_table, ["metric", "value"]),
        "",
        "## Breakdown By Priority Tier",
        "",
        _markdown_table(tier_breakdown, ["priority_tier", "union_candidates", "real_ingest_confirmed", "forecast_ready", "verification_ready", "fully_evaluable"]),
        "",
        "## Breakdown By Failure Reason",
        "",
        _markdown_table(failure_breakdown, ["failure_reason", "count"]),
        "",
        "## Real-Ingest Failure Detail Breakdown",
        "",
        _markdown_table(real_ingest_failure_detail_breakdown, ["real_ingest_failure_detail", "count"]),
        "",
        "## Top Blocked Dates",
        "",
        _markdown_table(top_blocked, ["date", "priority_tier", "failure_reason"]),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _topk_mean(values: np.ndarray, k: int = 250) -> float:
    flat = np.asarray(values, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0
    k = min(k, flat.size)
    partition = np.partition(flat, flat.size - k)[-k:]
    return float(partition.mean())


def _raw_tornado_concern_components(daily: xr.Dataset) -> dict[str, float]:
    support_gate = np.clip((daily["sig_tor_support"] - 1.0) / 1.0, 0.0, 1.0)
    overlap_gate = np.clip((daily["tornado_favored_overlap"] - 1.20) / 1.30, 0.0, 1.0)
    scp_gate = np.clip((daily["scp_proxy"] - 0.30) / 0.55, 0.0, 1.0) if "scp_proxy" in daily else overlap_gate
    hail_gate = (
        np.clip((daily["hail_favored_overlap"] - 1.60) / 1.10, 0.0, 1.0)
        if "hail_favored_overlap" in daily
        else xr.zeros_like(overlap_gate)
    )
    wind_gate = (
        np.clip((daily["wind_favored_overlap"] - 1.40) / 1.00, 0.0, 1.0)
        if "wind_favored_overlap" in daily
        else xr.zeros_like(overlap_gate)
    )
    outbreak_gate = np.clip((daily["outbreak_risk"] - 0.14) / 0.05, 0.0, 1.0)
    any_gate = np.clip((daily["any_prob"] - 0.03) / 0.08, 0.0, 1.0) if "any_prob" in daily else xr.zeros_like(outbreak_gate)
    context_gate = np.maximum(outbreak_gate, 0.6 * outbreak_gate + 0.4 * any_gate)
    tornado_mode_gate = overlap_gate * np.power(np.clip(scp_gate, 0.0, 1.0), 1.35)
    hail_excess = np.clip(hail_gate - (0.60 * overlap_gate + 0.75 * scp_gate), 0.0, 1.0)
    mode_balance = tornado_mode_gate / np.clip(tornado_mode_gate + 1.10 * hail_excess + 0.35 * wind_gate, 1.0e-6, None)
    support_mode_gate = tornado_mode_gate * (0.45 + 0.55 * support_gate) * (0.35 + 0.65 * mode_balance)
    core = support_mode_gate * (0.45 + 0.55 * context_gate)
    support_top = _topk_mean(support_gate.values, k=250)
    overlap_top = _topk_mean(overlap_gate.values, k=250)
    scp_top = _topk_mean(scp_gate.values, k=250)
    context_top = _topk_mean(context_gate.values, k=250)
    mode_balance_top = _topk_mean(mode_balance.values, k=250)
    joint_top = _topk_mean(support_mode_gate.values, k=250)
    joint_area = float(((support_gate >= 0.15) & (tornado_mode_gate >= 0.15) & (context_gate >= 0.20)).mean().item())
    core_top = _topk_mean(core.values, k=250)
    hail_like_penalty = hail_excess * (0.35 + 0.65 * context_gate)
    penalty_top = _topk_mean(hail_like_penalty.values, k=250)
    base_score = (
        core_top
        * (1.0 + 2.4 * joint_top)
        * (1.0 + 1.6 * overlap_top)
        * (1.0 + 1.2 * scp_top)
        * (1.0 + 1.4 * context_top)
        * (0.55 + 1.10 * mode_balance_top)
        * (1.0 + 6.0 * np.sqrt(max(joint_area, 0.0)))
        / (1.0 + 4.0 * penalty_top)
    )
    normalized_synoptic_support = _topk_mean(daily["synoptic_support"].values, k=250) / 800.0 if "synoptic_support" in daily else 0.0
    tornado_signal = scp_top * core_top
    broad_signal = context_top + 0.5 * normalized_synoptic_support
    eps = 1.0e-6
    discriminator = tornado_signal / (tornado_signal + broad_signal + eps)
    return {
        "raw_base_score_before_variant": float(base_score),
        "base_score": float(base_score),
        "tornado_signal": float(tornado_signal),
        "broad_signal": float(broad_signal),
        "normalized_synoptic_support": float(normalized_synoptic_support),
        "discriminator": float(discriminator),
        "discriminator_multiplier": float(0.3 + 0.7 * discriminator),
        "support_top": float(support_top),
        "context_top": float(context_top),
        "joint_area": float(joint_area),
        "core_top": float(core_top),
        "scp_top": float(scp_top),
        "penalty_top": float(penalty_top),
        "sig_tor_support_max": float(daily["sig_tor_support"].max().item()),
        "outbreak_risk_max": float(daily["outbreak_risk"].max().item()),
        "tornado_overlap_max": float(daily["tornado_favored_overlap"].max().item()),
        "hail_overlap_max": float(daily["hail_favored_overlap"].max().item()) if "hail_favored_overlap" in daily else float("nan"),
        "wind_overlap_max": float(daily["wind_favored_overlap"].max().item()) if "wind_favored_overlap" in daily else float("nan"),
        "raw_tornado_prob_max": float(daily["tornado_prob"].max().item()) if "tornado_prob" in daily else float("nan"),
        "raw_hail_prob_max": float(daily["hail_prob"].max().item()) if "hail_prob" in daily else float("nan"),
        "raw_wind_prob_max": float(daily["wind_prob"].max().item()) if "wind_prob" in daily else float("nan"),
        "raw_any_prob_max": float(daily["any_prob"].max().item()) if "any_prob" in daily else float("nan"),
    }


def _raw_core_terms_from_components(components: dict[str, Any], variant: str = "baseline") -> dict[str, float]:
    if variant not in RAW_CORE_VARIANTS:
        raise ValueError(f"unsupported raw core variant: {variant}")

    core_top = max(float(components.get("core_top", 0.0) or 0.0), 0.0)
    context_top = max(float(components.get("context_top", 0.0) or 0.0), 0.0)
    scp_top = max(float(components.get("scp_top", 0.0) or 0.0), 0.0)
    support_top = max(float(components.get("support_top", 0.0) or 0.0), 0.0)
    sig_tor_support_max = max(float(components.get("sig_tor_support_max", 0.0) or 0.0), 0.0)
    tornado_overlap_max = max(float(components.get("tornado_overlap_max", 0.0) or 0.0), 0.0)
    joint_area = max(float(components.get("joint_area", 0.0) or 0.0), 0.0)
    penalty_top = max(float(components.get("penalty_top", 0.0) or 0.0), 0.0)
    normalized_synoptic_support = max(float(components.get("normalized_synoptic_support", 0.0) or 0.0), 0.0)
    base_score = float(components.get("raw_base_score_before_variant", components.get("base_score", float("nan"))) or float("nan"))

    raw_core_mask_factor = 1.0
    if variant == "masked_core_only":
        scp_signal = float(np.clip((scp_top - 0.18) / 0.34, 0.0, 1.0))
        overlap_signal = float(np.clip((tornado_overlap_max - 1.30) / 1.30, 0.0, 1.0))
        sig_support_signal = float(np.clip((sig_tor_support_max - 1.10) / 0.95, 0.0, 1.0))
        support_signal = float(np.clip((support_top - 0.03) / 0.12, 0.0, 1.0))
        compactness_guard = float(np.clip(1.0 - joint_area / 0.0035, 0.0, 1.0))
        support_mask = (
            0.35 * scp_signal
            + 0.25 * overlap_signal
            + 0.20 * sig_support_signal
            + 0.10 * support_signal
            + 0.10 * compactness_guard
        )
        raw_core_mask_factor = float(np.clip(0.30 + 0.70 * support_mask, 0.30, 1.00))
    elif variant == "masked_core_stronger":
        scp_signal = float(np.clip((scp_top - 0.18) / 0.34, 0.0, 1.0))
        overlap_signal = float(np.clip((tornado_overlap_max - 1.30) / 1.30, 0.0, 1.0))
        sig_support_signal = float(np.clip((sig_tor_support_max - 1.10) / 0.95, 0.0, 1.0))
        support_signal = float(np.clip((support_top - 0.03) / 0.12, 0.0, 1.0))
        compactness_guard = float(np.clip(1.0 - joint_area / 0.0030, 0.0, 1.0))
        support_mask = (
            0.40 * scp_signal
            + 0.28 * overlap_signal
            + 0.18 * sig_support_signal
            + 0.08 * support_signal
            + 0.06 * compactness_guard
        )
        raw_core_mask_factor = float(np.clip(0.22 + 0.78 * support_mask, 0.22, 1.00))
    elif variant == "masked_core_calibrated":
        scp_signal = float(np.clip((scp_top - 0.18) / 0.34, 0.0, 1.0))
        overlap_signal = float(np.clip((tornado_overlap_max - 1.30) / 1.30, 0.0, 1.0))
        sig_support_signal = float(np.clip((sig_tor_support_max - 1.10) / 0.95, 0.0, 1.0))
        support_signal = float(np.clip((support_top - 0.03) / 0.12, 0.0, 1.0))
        compactness_guard = float(np.clip(1.0 - joint_area / 0.0032, 0.0, 1.0))
        support_mask = (
            0.45 * scp_signal
            + 0.30 * sig_support_signal
            + 0.13 * overlap_signal
            + 0.07 * support_signal
            + 0.05 * compactness_guard
        )
        raw_core_mask_factor = float(np.clip(0.22 + 0.78 * support_mask, 0.22, 1.00))
    elif variant == "masked_core_two_stage":
        scp_signal = float(np.clip((scp_top - 0.18) / 0.34, 0.0, 1.0))
        overlap_signal = float(np.clip((tornado_overlap_max - 1.30) / 1.30, 0.0, 1.0))
        sig_support_signal = float(np.clip((sig_tor_support_max - 1.10) / 0.95, 0.0, 1.0))
        support_signal = float(np.clip((support_top - 0.03) / 0.12, 0.0, 1.0))
        compactness_guard = float(np.clip(1.0 - joint_area / 0.0032, 0.0, 1.0))
        stage1_mask = (
            0.45 * scp_signal
            + 0.30 * sig_support_signal
            + 0.13 * overlap_signal
            + 0.07 * support_signal
            + 0.05 * compactness_guard
        )
        stage2_mask = float(np.clip(stage1_mask**1.25, 0.0, 1.0))
        raw_core_mask_factor = float(np.clip(0.22 + 0.78 * stage2_mask, 0.22, 1.00))
    elif variant == "masked_core_lower_floor":
        scp_signal = float(np.clip((scp_top - 0.18) / 0.34, 0.0, 1.0))
        overlap_signal = float(np.clip((tornado_overlap_max - 1.30) / 1.30, 0.0, 1.0))
        sig_support_signal = float(np.clip((sig_tor_support_max - 1.10) / 0.95, 0.0, 1.0))
        support_signal = float(np.clip((support_top - 0.03) / 0.12, 0.0, 1.0))
        compactness_guard = float(np.clip(1.0 - joint_area / 0.0032, 0.0, 1.0))
        stage1_mask = (
            0.45 * scp_signal
            + 0.30 * sig_support_signal
            + 0.13 * overlap_signal
            + 0.07 * support_signal
            + 0.05 * compactness_guard
        )
        stage2_mask = float(np.clip(stage1_mask**1.25, 0.0, 1.0))
        raw_core_mask_factor = float(np.clip(0.18 + 0.82 * stage2_mask, 0.18, 1.00))
    elif variant == "masked_core_steeper":
        scp_signal = float(np.clip((scp_top - 0.18) / 0.34, 0.0, 1.0))
        overlap_signal = float(np.clip((tornado_overlap_max - 1.30) / 1.30, 0.0, 1.0))
        sig_support_signal = float(np.clip((sig_tor_support_max - 1.10) / 0.95, 0.0, 1.0))
        support_signal = float(np.clip((support_top - 0.03) / 0.12, 0.0, 1.0))
        compactness_guard = float(np.clip(1.0 - joint_area / 0.0032, 0.0, 1.0))
        stage1_mask = (
            0.45 * scp_signal
            + 0.30 * sig_support_signal
            + 0.13 * overlap_signal
            + 0.07 * support_signal
            + 0.05 * compactness_guard
        )
        stage2_mask = float(np.clip(stage1_mask**1.35, 0.0, 1.0))
        raw_core_mask_factor = float(np.clip(0.18 + 0.82 * stage2_mask, 0.18, 1.00))
    elif variant == "footprint_discriminating":
        scp_signal = float(np.clip((scp_top - 0.18) / 0.34, 0.0, 1.0))
        overlap_signal = float(np.clip((tornado_overlap_max - 1.30) / 1.30, 0.0, 1.0))
        sig_support_signal = float(np.clip((sig_tor_support_max - 1.10) / 0.95, 0.0, 1.0))
        high_end_signal = float(np.clip(0.34 * scp_signal + 0.33 * overlap_signal + 0.33 * sig_support_signal, 0.0, 1.0))
        broad_footprint_signal = float(np.clip((joint_area - 0.0035) / 0.0100, 0.0, 1.0))
        weak_compactness_signal = float(np.clip(1.0 - core_top / (context_top + 1.0e-6), 0.0, 1.0))
        footprint_penalty = broad_footprint_signal * high_end_signal * (0.65 + 0.35 * weak_compactness_signal)
        raw_core_mask_factor = float(np.clip(1.0 - 0.38 * footprint_penalty, 0.62, 1.00))

    raw_core_final_value = core_top * raw_core_mask_factor
    raw_core_to_context_ratio = raw_core_final_value / (context_top + 1.0e-6)
    tornado_signal = scp_top * raw_core_final_value
    broad_signal = context_top + 0.5 * normalized_synoptic_support
    discriminator = tornado_signal / (tornado_signal + broad_signal + 1.0e-6)
    if not np.isfinite(base_score):
        base_score = (
            raw_core_final_value
            * (1.0 + 2.4 * joint_area)
            * (1.0 + 1.2 * scp_top)
            * (1.0 + 1.4 * context_top)
            / (1.0 + 4.0 * penalty_top)
        )
    adjusted_base_score = base_score * raw_core_mask_factor
    return {
        "raw_core_variant_name": variant,
        "raw_core_mask_factor": float(raw_core_mask_factor),
        "raw_core_final_value": float(raw_core_final_value),
        "raw_core_to_context_ratio": float(raw_core_to_context_ratio),
        "core_top": float(raw_core_final_value),
        "tornado_signal": float(tornado_signal),
        "broad_signal": float(broad_signal),
        "discriminator": float(discriminator),
        "discriminator_multiplier": float(0.3 + 0.7 * discriminator),
        "raw_base_score_before_variant": float(adjusted_base_score),
        "base_score": float(adjusted_base_score),
    }


def _apply_raw_core_variant_to_frame(frame: pd.DataFrame, variant: str = "baseline") -> pd.DataFrame:
    adjusted = frame.copy()
    rows = []
    for row in adjusted.to_dict(orient="records"):
        rows.append(_raw_core_terms_from_components(row, variant=variant))
    raw_frame = pd.DataFrame(rows, index=adjusted.index)
    for column in raw_frame.columns:
        adjusted[column] = raw_frame[column]
    return adjusted


def _core_terms_from_components(components: dict[str, Any], variant: str = "baseline") -> dict[str, float]:
    if variant not in CORE_VARIANTS:
        raise ValueError(f"unsupported core variant: {variant}")

    support_top = max(float(components.get("support_top", 0.0) or 0.0), 0.0)
    context_top = max(float(components.get("context_top", 0.0) or 0.0), 0.0)
    joint_area = max(float(components.get("joint_area", 0.0) or 0.0), 0.0)
    core_top = max(float(components.get("effective_core_top", components.get("core_top", 0.0)) or 0.0), 0.0)
    scp_top = max(float(components.get("scp_top", 0.0) or 0.0), 0.0)
    sig_tor_support_max = max(float(components.get("sig_tor_support_max", 0.0) or 0.0), 0.0)
    tornado_overlap_max = max(float(components.get("tornado_overlap_max", 0.0) or 0.0), 0.0)
    normalized_synoptic_support = max(float(components.get("normalized_synoptic_support", 0.0) or 0.0), 0.0)
    penalty_top = max(float(components.get("penalty_top", 0.0) or 0.0), 0.0)
    raw_base = float(components.get("raw_base_score_before_variant", components.get("base_score", float("nan"))) or float("nan"))

    core_signal = float(np.clip((core_top - 0.002) / 0.030, 0.0, 1.0))
    support_signal = float(np.clip((support_top - 0.03) / 0.12, 0.0, 1.0))
    scp_signal = float(np.clip((scp_top - 0.18) / 0.34, 0.0, 1.0))
    overlap_signal = float(np.clip((tornado_overlap_max - 1.30) / 1.30, 0.0, 1.0))
    sig_support_signal = float(np.clip((sig_tor_support_max - 1.10) / 0.95, 0.0, 1.0))
    context_signal = float(np.clip((context_top - 0.30) / 0.55, 0.0, 1.0))
    joint_signal = float(np.clip(joint_area / 0.0035, 0.0, 1.0))
    synoptic_signal = float(np.clip((normalized_synoptic_support - 0.55) / 0.35, 0.0, 1.0))
    raw_ratio = core_top / (context_top + 1.0e-6)
    raw_ratio_norm = float(np.clip(raw_ratio / 0.020, 0.0, 1.0))

    core_support_interaction = float(np.sqrt(max(core_signal * support_signal, 0.0)))
    core_overlap_interaction = float(np.sqrt(max(core_signal * overlap_signal, 0.0)))
    core_scp_interaction = float(np.sqrt(max(core_signal * scp_signal, 0.0)))
    core_alignment_proxy = float(
        np.clip(0.40 * core_scp_interaction + 0.30 * core_overlap_interaction + 0.20 * sig_support_signal + 0.10 * support_signal, 0.0, 1.0)
    )
    core_compactness_proxy = float(
        np.clip(0.45 * raw_ratio_norm + 0.30 * overlap_signal + 0.25 * max(1.0 - joint_signal, 0.0), 0.0, 1.0)
    )
    core_percentile_proxy = float(
        np.clip(0.45 * core_signal + 0.30 * core_support_interaction + 0.25 * min(core_overlap_interaction, core_scp_interaction + 0.10), 0.0, 1.0)
    )
    core_diffuseness_proxy = float(
        np.clip(0.45 * context_signal + 0.30 * joint_signal + 0.15 * synoptic_signal + 0.10 * max(1.0 - raw_ratio_norm, 0.0), 0.0, 1.0)
    )
    core_purity_proxy = float(np.clip(0.50 * core_alignment_proxy + 0.35 * core_compactness_proxy + 0.15 * core_percentile_proxy, 0.0, 1.0))
    spike_excess = max(core_signal - core_percentile_proxy, 0.0)

    core_multiplier = 1.0
    core_penalty_term = 1.0
    effective_core_top = core_top

    if variant == "compact_core":
        core_multiplier = float(np.clip(0.74 + 0.42 * core_compactness_proxy - 0.22 * core_diffuseness_proxy, 0.60, 1.12))
        effective_core_top = core_top * core_multiplier
    elif variant == "percentile_core":
        percentile_factor = 0.76 + 0.36 * core_percentile_proxy - 0.24 * spike_excess
        core_multiplier = float(np.clip(percentile_factor, 0.60, 1.10))
        effective_core_top = core_top * core_multiplier
    elif variant == "aligned_core":
        alignment_factor = 0.80 + 0.42 * core_alignment_proxy - 0.08 * max(context_signal - core_alignment_proxy, 0.0)
        core_multiplier = float(np.clip(alignment_factor, 0.68, 1.18))
        effective_core_top = core_top * core_multiplier
    elif variant == "hybrid":
        core_multiplier = float(
            np.clip(
                0.70
                + 0.20 * core_compactness_proxy
                + 0.16 * core_percentile_proxy
                + 0.18 * core_alignment_proxy
                - 0.18 * core_diffuseness_proxy
                - 0.08 * spike_excess,
                0.58,
                1.18,
            )
        )
        contamination = max(core_diffuseness_proxy - 0.70 * core_purity_proxy, 0.0)
        core_penalty_term = float(np.clip(1.0 - 0.20 * contamination, 0.78, 1.0))
        effective_core_top = core_top * core_multiplier * core_penalty_term

    effective_core_to_context_ratio = effective_core_top / (context_top + 1.0e-6)
    adjusted_tornado_signal = scp_top * effective_core_top
    adjusted_broad_signal = context_top + 0.5 * normalized_synoptic_support
    adjusted_discriminator = adjusted_tornado_signal / (adjusted_tornado_signal + adjusted_broad_signal + 1.0e-6)
    if not np.isfinite(raw_base):
        raw_base = (
            effective_core_top
            * (1.0 + 2.4 * joint_area)
            * (1.0 + 1.6 * min(tornado_overlap_max / 3.0, 1.0))
            * (1.0 + 1.2 * scp_top)
            * (1.0 + 1.4 * context_top)
            / (1.0 + 4.0 * penalty_top)
        )
    adjusted_base = raw_base * core_multiplier * core_penalty_term

    return {
        "core_variant_name": variant,
        "effective_core_top": float(effective_core_top),
        "effective_core_to_context_ratio": float(effective_core_to_context_ratio),
        "core_compactness_proxy": float(core_compactness_proxy),
        "core_percentile_proxy": float(core_percentile_proxy),
        "core_alignment_proxy": float(core_alignment_proxy),
        "core_diffuseness_proxy": float(core_diffuseness_proxy),
        "core_purity_proxy": float(core_purity_proxy),
        "core_support_interaction": float(core_support_interaction),
        "core_overlap_interaction": float(core_overlap_interaction),
        "core_scp_interaction": float(core_scp_interaction),
        "core_penalty_term": float(core_penalty_term),
        "tornado_signal": float(adjusted_tornado_signal),
        "broad_signal": float(adjusted_broad_signal),
        "discriminator": float(adjusted_discriminator),
        "discriminator_multiplier": float(0.3 + 0.7 * adjusted_discriminator),
        "raw_base_score_before_variant": float(adjusted_base),
        "base_score": float(adjusted_base),
    }


def _apply_core_variant_to_frame(frame: pd.DataFrame, variant: str = "baseline") -> pd.DataFrame:
    adjusted = frame.copy()
    core_rows = []
    for row in adjusted.to_dict(orient="records"):
        core_rows.append(_core_terms_from_components(row, variant=variant))
    core_frame = pd.DataFrame(core_rows, index=adjusted.index)
    for column in core_frame.columns:
        adjusted[column] = core_frame[column]
    return adjusted


def _source_terms_from_components(components: dict[str, Any], variant: str = "baseline") -> dict[str, float]:
    if variant not in SOURCE_VARIANTS:
        raise ValueError(f"unsupported source variant: {variant}")

    support_top = max(float(components.get("support_top", 0.0) or 0.0), 0.0)
    context_top = max(float(components.get("context_top", 0.0) or 0.0), 0.0)
    joint_area = max(float(components.get("joint_area", 0.0) or 0.0), 0.0)
    core_top = max(float(components.get("effective_core_top", components.get("core_top", 0.0)) or 0.0), 0.0)
    effective_core_top = max(float(components.get("effective_core_top", core_top) or core_top), 0.0)
    scp_top = max(float(components.get("scp_top", 0.0) or 0.0), 0.0)
    penalty_top = max(float(components.get("penalty_top", 0.0) or 0.0), 0.0)
    sig_tor_support_max = max(float(components.get("sig_tor_support_max", 0.0) or 0.0), 0.0)
    outbreak_risk_max = max(float(components.get("outbreak_risk_max", 0.0) or 0.0), 0.0)
    tornado_overlap_max = max(float(components.get("tornado_overlap_max", 0.0) or 0.0), 0.0)
    hail_overlap_raw = float(components.get("hail_overlap_max", 0.0) or 0.0)
    hail_overlap_max = max(hail_overlap_raw, 0.0) if np.isfinite(hail_overlap_raw) else 0.0
    normalized_synoptic_support = max(float(components.get("normalized_synoptic_support", 0.0) or 0.0), 0.0)
    base_score = float(components.get("raw_base_score_before_variant", components.get("base_score", float("nan"))) or float("nan"))

    scp_support_ratio = scp_top / (support_top + 1.0e-6)
    overlap_to_context_ratio = tornado_overlap_max / (1.0 + 2.0 * context_top + 1.0e-6)
    core_to_context_ratio = core_top / (context_top + 1.0e-6)
    effective_core_to_context_ratio = effective_core_top / (context_top + 1.0e-6)
    scp_support_ratio_norm = float(np.clip(scp_support_ratio / 2.0, 0.0, 1.0))
    overlap_to_context_norm = float(np.clip(overlap_to_context_ratio / 1.1, 0.0, 1.0))
    core_to_context_norm = float(np.clip(effective_core_to_context_ratio / 0.020, 0.0, 1.0))
    sig_support_signal = float(np.clip((sig_tor_support_max - 1.15) / 0.90, 0.0, 1.0))
    context_signal = float(np.clip((context_top - 0.35) / 0.45, 0.0, 1.0))
    synoptic_signal = float(np.clip((normalized_synoptic_support - 0.55) / 0.35, 0.0, 1.0))
    joint_penalty = float(np.clip(joint_area / 0.004, 0.0, 1.0))

    overlap_compactness_proxy = float(np.clip(0.60 * overlap_to_context_norm + 0.40 * (1.0 - joint_penalty), 0.0, 1.0))
    overlap_purity_proxy = float(np.clip(0.45 * scp_support_ratio_norm + 0.30 * overlap_to_context_norm + 0.25 * sig_support_signal, 0.0, 1.0))
    core_tornado_alignment_proxy = float(np.clip(0.45 * scp_support_ratio_norm + 0.30 * core_to_context_norm + 0.25 * sig_support_signal, 0.0, 1.0))
    support_tornado_alignment_proxy = float(np.clip(0.40 * sig_support_signal + 0.35 * scp_support_ratio_norm + 0.25 * overlap_purity_proxy, 0.0, 1.0))
    broad_contamination_proxy = float(
        np.clip(
            0.50 * context_signal + 0.35 * synoptic_signal + 0.15 * max(1.0 - overlap_compactness_proxy, 0.0)
            - 0.30 * overlap_purity_proxy
            - 0.20 * core_tornado_alignment_proxy,
            0.0,
            1.0,
        )
    )

    source_core_term = 1.0
    source_overlap_term = 1.0
    source_support_term = 1.0
    source_penalty_term = 1.0

    if variant == "compact_overlap":
        source_overlap_term = float(np.clip(0.78 + 0.32 * overlap_compactness_proxy * overlap_to_context_norm, 0.78, 1.10))
    elif variant == "tornado_purity_gate":
        purity_gate = 0.65 + 0.35 * (0.40 * overlap_purity_proxy + 0.35 * core_tornado_alignment_proxy + 0.25 * support_tornado_alignment_proxy)
        source_core_term = float(np.clip(purity_gate, 0.65, 1.00))
        source_overlap_term = float(np.clip(0.80 + 0.20 * overlap_purity_proxy, 0.80, 1.00))
        source_support_term = float(np.clip(0.80 + 0.20 * support_tornado_alignment_proxy, 0.80, 1.00))
    elif variant == "contamination_penalty":
        contamination = max(broad_contamination_proxy - 0.40 * core_tornado_alignment_proxy, 0.0)
        source_penalty_term = float(np.clip(1.0 - 0.30 * contamination, 0.72, 1.0))
    elif variant == "hybrid":
        source_overlap_term = float(np.clip(0.82 + 0.20 * overlap_compactness_proxy * overlap_to_context_norm, 0.82, 1.02))
        purity_gate = 0.72 + 0.28 * (0.40 * overlap_purity_proxy + 0.35 * core_tornado_alignment_proxy + 0.25 * support_tornado_alignment_proxy)
        source_core_term = float(np.clip(purity_gate, 0.72, 1.00))
        source_support_term = float(np.clip(0.86 + 0.14 * support_tornado_alignment_proxy, 0.86, 1.00))
        contamination = max(broad_contamination_proxy - 0.45 * core_tornado_alignment_proxy, 0.0)
        source_penalty_term = float(np.clip(1.0 - 0.20 * contamination, 0.80, 1.0))
    elif variant == "failure_targeted_contamination":
        tornado_structure = float(
            np.clip(
                0.40 * core_tornado_alignment_proxy + 0.35 * overlap_purity_proxy + 0.25 * support_tornado_alignment_proxy,
                0.0,
                1.0,
            )
        )
        weak_structure = max(1.0 - tornado_structure, 0.0)
        diffuse_contamination = broad_contamination_proxy * (0.65 * weak_structure + 0.35 * joint_penalty)
        source_core_term = float(np.clip(0.74 + 0.26 * tornado_structure, 0.74, 1.00))
        source_overlap_term = float(np.clip(0.82 + 0.20 * overlap_purity_proxy * overlap_to_context_norm, 0.82, 1.02))
        source_support_term = float(np.clip(0.84 + 0.16 * support_tornado_alignment_proxy, 0.84, 1.00))
        source_penalty_term = float(np.clip(1.0 - 0.55 * diffuse_contamination, 0.58, 1.00))

    adjusted_core_top = effective_core_top * source_core_term * source_penalty_term
    adjusted_support_top = support_top * source_support_term * source_penalty_term
    adjusted_overlap_max = tornado_overlap_max * source_overlap_term * source_penalty_term
    adjusted_sig_tor_support_max = sig_tor_support_max * source_support_term
    adjusted_joint_area = joint_area * (0.92 + 0.08 * overlap_compactness_proxy) * source_penalty_term
    adjusted_tornado_signal = scp_top * adjusted_core_top
    adjusted_broad_signal = context_top + 0.5 * normalized_synoptic_support
    adjusted_discriminator = adjusted_tornado_signal / (adjusted_tornado_signal + adjusted_broad_signal + 1.0e-6)
    if not np.isfinite(base_score):
        base_score = (
            effective_core_top
            * (1.0 + 2.4 * joint_area)
            * (1.0 + 1.6 * min(tornado_overlap_max / 3.0, 1.0))
            * (1.0 + 1.2 * scp_top)
            * (1.0 + 1.4 * context_top)
            / (1.0 + 4.0 * penalty_top)
        )
    adjusted_base_score = base_score * source_core_term * source_overlap_term * source_support_term * source_penalty_term

    return {
        "source_variant_name": variant,
        "score_variant_name": str(components.get("score_variant_name", "")),
        "overlap_compactness_proxy": overlap_compactness_proxy,
        "overlap_purity_proxy": overlap_purity_proxy,
        "core_tornado_alignment_proxy": core_tornado_alignment_proxy,
        "support_tornado_alignment_proxy": support_tornado_alignment_proxy,
        "broad_contamination_proxy": broad_contamination_proxy,
        "scp_support_ratio": float(scp_support_ratio),
        "overlap_to_context_ratio": float(overlap_to_context_ratio),
        "core_to_context_ratio": float(core_to_context_ratio),
        "effective_core_to_context_ratio": float(adjusted_core_top / (context_top + 1.0e-6)),
        "source_core_term": float(source_core_term),
        "source_overlap_term": float(source_overlap_term),
        "source_support_term": float(source_support_term),
        "source_penalty_term": float(source_penalty_term),
        "support_top": float(adjusted_support_top),
        "joint_area": float(adjusted_joint_area),
        "effective_core_top": float(adjusted_core_top),
        "sig_tor_support_max": float(adjusted_sig_tor_support_max),
        "tornado_overlap_max": float(adjusted_overlap_max),
        "tornado_signal": float(adjusted_tornado_signal),
        "broad_signal": float(adjusted_broad_signal),
        "discriminator": float(adjusted_discriminator),
        "discriminator_multiplier": float(0.3 + 0.7 * adjusted_discriminator),
        "raw_base_score_before_variant": float(adjusted_base_score),
        "base_score": float(adjusted_base_score),
        "core_minus_context": float(adjusted_core_top - context_top),
        "scp_x_core": float(scp_top * adjusted_core_top),
        "overlap_x_core": float(adjusted_overlap_max * adjusted_core_top),
        "support_x_overlap": float(adjusted_support_top * adjusted_overlap_max),
    }


def _apply_source_variant_to_frame(frame: pd.DataFrame, variant: str = "baseline") -> pd.DataFrame:
    adjusted = frame.copy()
    source_rows = []
    for row in adjusted.to_dict(orient="records"):
        source_rows.append(_source_terms_from_components(row, variant=variant))
    source_frame = pd.DataFrame(source_rows, index=adjusted.index)
    for column in source_frame.columns:
        adjusted[column] = source_frame[column]
    return adjusted


def _component_terms_from_components(components: dict[str, Any], variant: str = "baseline") -> dict[str, float]:
    if variant not in COMPONENT_VARIANTS:
        raise ValueError(f"unsupported component variant: {variant}")

    support_top = max(float(components.get("support_top", 0.0) or 0.0), 0.0)
    context_top = max(float(components.get("context_top", 0.0) or 0.0), 0.0)
    joint_area = max(float(components.get("joint_area", 0.0) or 0.0), 0.0)
    core_top = max(float(components.get("core_top", 0.0) or 0.0), 0.0)
    scp_top = max(float(components.get("scp_top", 0.0) or 0.0), 0.0)
    penalty_top = max(float(components.get("penalty_top", 0.0) or 0.0), 0.0)
    sig_tor_support_max = max(float(components.get("sig_tor_support_max", 0.0) or 0.0), 0.0)
    outbreak_risk_max = max(float(components.get("outbreak_risk_max", 0.0) or 0.0), 0.0)
    tornado_overlap_max = max(float(components.get("tornado_overlap_max", 0.0) or 0.0), 0.0)
    hail_overlap_raw = float(components.get("hail_overlap_max", 0.0) or 0.0)
    hail_overlap_max = max(hail_overlap_raw, 0.0) if np.isfinite(hail_overlap_raw) else 0.0
    normalized_synoptic_support = max(float(components.get("normalized_synoptic_support", 0.0) or 0.0), 0.0)

    core_signal = float(np.clip((core_top - 0.002) / 0.035, 0.0, 1.0))
    scp_signal = float(np.clip((scp_top - 0.18) / 0.32, 0.0, 1.0))
    overlap_signal = float(np.clip((tornado_overlap_max - 1.40) / 1.20, 0.0, 1.0))
    sig_support_signal = float(np.clip((sig_tor_support_max - 1.15) / 0.90, 0.0, 1.0))
    context_signal = float(np.clip((context_top - 0.35) / 0.45, 0.0, 1.0))
    synoptic_signal = float(np.clip((normalized_synoptic_support - 0.55) / 0.35, 0.0, 1.0))

    component_core_term = 1.0
    component_overlap_term = 1.0
    component_scp_term = 1.0
    component_penalty_term = 1.0
    component_hail_dominance = 0.0
    component_tornado_agreement = 0.0

    if variant == "baseline":
        adjusted_core_top = core_top
        adjusted_joint_area = joint_area
        adjusted_tornado_signal = float(components.get("tornado_signal", scp_top * core_top) or 0.0)
        adjusted_broad_signal = float(components.get("broad_signal", context_top + 0.5 * normalized_synoptic_support) or 0.0)
        adjusted_discriminator = float(components.get("discriminator", 0.0) or 0.0)
        adjusted_base_score = float(components.get("raw_base_score_before_variant", components.get("base_score", float("nan"))) or float("nan"))
        if not np.isfinite(adjusted_base_score):
            adjusted_base_score = (
                adjusted_core_top
                * (1.0 + 2.4 * adjusted_joint_area)
                * (1.0 + 1.6 * min(tornado_overlap_max / 3.0, 1.0))
                * (1.0 + 1.2 * scp_top)
                * (1.0 + 1.4 * context_top)
                / (1.0 + 4.0 * penalty_top)
            )
    else:
        if variant == "core_tornado_gated":
            tornado_gate = 0.45 + 0.55 * (0.45 * scp_signal + 0.35 * overlap_signal + 0.20 * sig_support_signal)
            component_core_term = float(np.clip(tornado_gate, 0.45, 1.00))
        elif variant == "overlap_emphasis":
            overlap_boost = 0.95 + 0.55 * overlap_signal * (0.35 + 0.65 * min(core_signal, scp_signal + 0.15))
            component_overlap_term = float(np.clip(overlap_boost, 0.95, 1.45))
        elif variant == "scp_core_hybrid":
            scp_core_boost = 0.80 + 0.70 * np.sqrt(max(core_signal * scp_signal, 0.0))
            component_scp_term = float(np.clip(scp_core_boost, 0.80, 1.50))
        elif variant == "hybrid":
            tornado_gate = 0.62 + 0.38 * (0.45 * scp_signal + 0.35 * overlap_signal + 0.20 * sig_support_signal)
            overlap_boost = 0.98 + 0.28 * overlap_signal * (0.35 + 0.65 * min(core_signal, scp_signal + 0.15))
            scp_core_boost = 0.90 + 0.35 * np.sqrt(max(core_signal * scp_signal, 0.0))
            generic_penalty = 1.0 - 0.15 * max(context_signal + 0.5 * synoptic_signal - (0.75 * scp_signal + 0.55 * overlap_signal), 0.0)
            component_core_term = float(np.clip(tornado_gate, 0.72, 1.00))
            component_overlap_term = float(np.clip(overlap_boost, 0.98, 1.25))
            component_scp_term = float(np.clip(scp_core_boost, 0.90, 1.25))
            component_penalty_term = float(np.clip(generic_penalty, 0.82, 1.0))
        elif variant == "tornado_hail_separation_v1":
            tornado_agreement = float(np.clip(0.40 * scp_signal + 0.35 * overlap_signal + 0.25 * sig_support_signal, 0.0, 1.0))
            hail_overlap_signal = float(np.clip((hail_overlap_max - 1.80) / 1.20, 0.0, 1.0))
            hail_penalty_signal = float(np.clip((penalty_top - 0.04) / 0.12, 0.0, 1.0))
            hail_dominance = float(np.clip(max(hail_overlap_signal, hail_penalty_signal) - 0.45 * tornado_agreement, 0.0, 1.0))
            weak_tornado_specific = float(np.clip(1.0 - tornado_agreement, 0.0, 1.0))
            preserved_tornado_core = float(np.sqrt(max(tornado_agreement * max(core_signal, scp_signal), 0.0)))
            component_hail_dominance = hail_dominance
            component_tornado_agreement = tornado_agreement
            component_core_term = float(np.clip(0.82 + 0.22 * preserved_tornado_core, 0.82, 1.08))
            component_overlap_term = float(np.clip(0.96 + 0.16 * overlap_signal * preserved_tornado_core, 0.96, 1.12))
            component_scp_term = float(np.clip(0.94 + 0.18 * scp_signal * preserved_tornado_core, 0.94, 1.12))
            component_penalty_term = float(np.clip(1.0 - 0.62 * hail_dominance * weak_tornado_specific, 0.45, 1.0))
        elif variant == "tornado_signal_continuity_v1":
            tornado_agreement = float(np.clip(0.36 * scp_signal + 0.34 * overlap_signal + 0.30 * sig_support_signal, 0.0, 1.0))
            aligned_support = float(np.clip((scp_signal * overlap_signal * sig_support_signal) ** (1.0 / 3.0), 0.0, 1.0))
            fragmented_but_supported = float(np.clip((aligned_support - core_signal) / 0.55, 0.0, 1.0))
            generic_context_excess = max(context_signal + 0.45 * synoptic_signal - (scp_signal + overlap_signal + sig_support_signal) / 3.0, 0.0)
            continuity_support = aligned_support * (0.45 + 0.55 * fragmented_but_supported) * (1.0 - 0.65 * generic_context_excess)
            component_tornado_agreement = tornado_agreement
            component_core_term = float(np.clip(1.0 + 0.10 * continuity_support, 1.0, 1.10))
            component_overlap_term = float(np.clip(1.0 + 0.08 * continuity_support * overlap_signal, 1.0, 1.08))
            component_scp_term = float(np.clip(1.0 + 0.08 * continuity_support * scp_signal, 1.0, 1.08))
            component_penalty_term = 1.0

        adjusted_core_top = core_top * component_core_term * component_overlap_term * component_scp_term
        adjusted_joint_area = joint_area * (0.92 + 0.08 * overlap_signal * max(core_signal, 0.25))
        adjusted_tornado_signal = scp_top * adjusted_core_top
        adjusted_broad_signal = context_top + 0.5 * normalized_synoptic_support
        adjusted_discriminator = adjusted_tornado_signal / (adjusted_tornado_signal + adjusted_broad_signal + 1.0e-6)
        adjusted_base_score = (
            adjusted_core_top
            * (1.0 + 2.4 * adjusted_joint_area)
            * (1.0 + 1.6 * min(tornado_overlap_max / 3.0, 1.0))
            * (1.0 + 1.2 * scp_top)
            * (1.0 + 1.4 * context_top)
            / (1.0 + 4.0 * penalty_top)
            * component_penalty_term
        )

    return {
        "component_variant_name": variant,
        "component_core_term": float(component_core_term),
        "component_overlap_term": float(component_overlap_term),
        "component_scp_term": float(component_scp_term),
        "component_penalty_term": float(component_penalty_term),
        "component_hail_dominance": float(component_hail_dominance),
        "component_tornado_agreement": float(component_tornado_agreement),
        "effective_core_top": float(adjusted_core_top),
        "joint_area": float(adjusted_joint_area),
        "tornado_signal": float(adjusted_tornado_signal),
        "broad_signal": float(adjusted_broad_signal),
        "discriminator": float(adjusted_discriminator),
        "discriminator_multiplier": float(0.3 + 0.7 * adjusted_discriminator),
        "raw_base_score_before_variant": float(adjusted_base_score),
        "base_score": float(adjusted_base_score),
        "core_minus_context": float(adjusted_core_top - context_top),
        "scp_x_core": float(scp_top * adjusted_core_top),
        "overlap_x_core": float(tornado_overlap_max * adjusted_core_top),
        "support_x_overlap": float(support_top * tornado_overlap_max),
    }


def _apply_component_variant_to_frame(frame: pd.DataFrame, variant: str = "baseline") -> pd.DataFrame:
    adjusted = frame.copy()
    component_rows = []
    for row in adjusted.to_dict(orient="records"):
        component_rows.append(_component_terms_from_components(row, variant=variant))
    component_frame = pd.DataFrame(component_rows, index=adjusted.index)
    for column in component_frame.columns:
        adjusted[column] = component_frame[column]
    return adjusted


def _variant_terms_from_components(components: dict[str, Any], variant: str = "baseline") -> dict[str, float]:
    if variant not in SCORE_VARIANTS:
        raise ValueError(f"unsupported score variant: {variant}")

    raw_base = float(components.get("raw_base_score_before_variant", components.get("base_score", 0.0)) or 0.0)
    tornado_signal = max(float(components.get("tornado_signal", 0.0) or 0.0), 0.0)
    broad_signal = max(float(components.get("broad_signal", 0.0) or 0.0), 0.0)
    discriminator = float(components.get("discriminator", 0.0) or 0.0)
    learned_prob = float(components.get("learned_tornado_concern_prob", float("nan")))
    learned_support = 0.0 if not np.isfinite(learned_prob) else float(np.clip((learned_prob - 0.25) / 0.50, 0.0, 1.0))
    tornado_activation = float(np.clip((tornado_signal - 0.0025) / 0.0100, 0.0, 1.0))
    broad_compression = min(broad_signal, 0.18) + 0.35 * max(broad_signal - 0.18, 0.0)
    capped_discriminator = tornado_signal / (tornado_signal + broad_compression + 1.0e-6)
    learned_alignment = float(np.clip((discriminator - 0.03) / 0.20, 0.0, 1.0))
    lead_days = 0.0
    try:
        init_date = pd.Timestamp(str(components.get("init_date", "")))
        valid_date = pd.Timestamp(str(components.get("valid_date", "")))
        if pd.notna(init_date) and pd.notna(valid_date):
            lead_days = max(float((valid_date.date() - init_date.date()).days), 0.0)
    except (TypeError, ValueError):
        lead_days = 0.0

    variant_base_score = raw_base
    variant_tornado_term = 1.0
    variant_broad_term = 0.3 + 0.7 * discriminator
    variant_learned_term = 1.0
    variant_penalty_term = 1.0

    if variant == "tornado_emphasis":
        variant_tornado_term = 1.0 + 0.40 * tornado_activation
    elif variant == "capped_broad":
        variant_broad_term = 0.3 + 0.7 * capped_discriminator
    elif variant == "learned_gated":
        variant_learned_term = 1.0 + 0.25 * learned_support * learned_alignment
        variant_penalty_term = float(np.clip(1.0 - 0.30 * learned_support * (1.0 - learned_alignment), 0.70, 1.0))
    elif variant == "hybrid":
        variant_tornado_term = 1.0 + 0.25 * tornado_activation
        variant_broad_term = 0.3 + 0.7 * capped_discriminator
        variant_learned_term = 1.0 + 0.18 * learned_support * learned_alignment
        mismatch = float(np.clip((broad_signal - tornado_signal) / (broad_signal + 1.0e-6), 0.0, 1.0))
        variant_penalty_term = float(np.clip(1.0 - 0.18 * mismatch * (1.0 - learned_alignment), 0.78, 1.0))
    elif variant == "lead_time_calibrated":
        # Opt-in challenger: later lead days are less reliable and often produced broad severe false-top days.
        variant_penalty_term = float(np.exp(-0.50 * lead_days))
    elif variant == "v1_triage":
        # V1 review ranker: keep the product fields unchanged, but strongly demote later-lead
        # broad-severe maxima so early tornado-focused days are easier to review first.
        variant_penalty_term = float(np.exp(-2.50 * lead_days))

    final_score = raw_base * variant_tornado_term * variant_broad_term * variant_learned_term * variant_penalty_term
    final_score = float(np.nan_to_num(final_score, nan=0.0, posinf=0.0, neginf=0.0))
    return {
        "score_variant_name": variant,
        "variant_name": variant,
        "variant_base_score": float(raw_base),
        "variant_tornado_term": float(variant_tornado_term),
        "variant_broad_term": float(variant_broad_term),
        "variant_learned_term": float(variant_learned_term),
        "variant_penalty_term": float(variant_penalty_term),
        "variant_final_score_before_preference": final_score,
        "tornado_concern_score": final_score,
    }


def _apply_score_variant_to_frame(frame: pd.DataFrame, variant: str = "baseline") -> pd.DataFrame:
    adjusted = frame.copy()
    variant_rows = []
    for row in adjusted.to_dict(orient="records"):
        variant_rows.append(_variant_terms_from_components(row, variant=variant))
    variant_frame = pd.DataFrame(variant_rows, index=adjusted.index)
    for column in variant_frame.columns:
        adjusted[column] = variant_frame[column]
    return adjusted


def _daily_tornado_concern_components(
    daily: xr.Dataset,
    learned_prob: float = float("nan"),
    raw_core_variant: str = "baseline",
    score_variant: str = "baseline",
    component_variant: str = "baseline",
    source_variant: str = "baseline",
    core_variant: str = "baseline",
    init_date: str | None = None,
    valid_date: str | None = None,
) -> dict[str, float]:
    components = _raw_tornado_concern_components(daily)
    components["learned_tornado_concern_prob"] = float(learned_prob)
    if init_date is not None:
        components["init_date"] = init_date
    if valid_date is not None:
        components["valid_date"] = valid_date
    components.update(_raw_core_terms_from_components(components, variant=raw_core_variant))
    components.update(_core_terms_from_components(components, variant=core_variant))
    components.update(_source_terms_from_components(components, variant=source_variant))
    components.update(_component_terms_from_components(components, variant=component_variant))
    components.update(_variant_terms_from_components(components, variant=score_variant))
    return components


def _ingest_source_info(verification_payload: dict[str, Any]) -> tuple[str, bool]:
    run_summary = verification_payload.get("run_summary", {}) or {}
    evaluation_metadata = verification_payload.get("evaluation_metadata", {}) or {}
    ingest_summary = evaluation_metadata.get("ingest_summary", {}) or {}
    source = str(ingest_summary.get("source") or run_summary.get("evaluation_source") or "unknown")
    is_real = source not in {"synthetic", "synthetic_fallback", "synthetic_degraded"}
    return source, is_real


def evaluate_tornado_concern_for_artifacts(
    prediction_path: Path,
    verification_path: Path,
    init_date: str | None = None,
    raw_core_variant: str = "baseline",
    score_variant: str = "baseline",
    component_variant: str = "baseline",
    source_variant: str = "baseline",
    core_variant: str = "baseline",
) -> pd.DataFrame:
    prediction = xr.load_dataset(prediction_path)
    verification_payload = json.loads(verification_path.read_text(encoding="utf-8"))
    ingest_source, is_real_ingest = _ingest_source_info(verification_payload)
    observed_by_date = {str(row.get("valid_date")): row for row in verification_payload.get("per_day", [])}
    times = pd.to_datetime(prediction["time"].values)
    date_strings = pd.Series([timestamp.date().isoformat() for timestamp in times])
    rows: list[dict[str, Any]] = []
    for valid_date in sorted(date_strings.unique()):
        mask = np.asarray(date_strings) == valid_date
        daily = prediction.isel(time=mask).max("time")
        observed = observed_by_date.get(valid_date, {})
        rows.append(
            {
                "init_date": init_date or str(verification_payload.get("run_summary", {}).get("init_date", "")),
                "valid_date": valid_date,
                "observed_category": observed.get("observed_category", "unknown"),
                "observed_tornado_outbreak": int(observed.get("observed_tornado_outbreak", 0) or 0),
                "observed_significant_tornado_support": int(observed.get("observed_significant_tornado_support", 0) or 0),
                "ingest_source": ingest_source,
                "is_real_ingest": bool(is_real_ingest),
                **_daily_tornado_concern_components(
                    daily,
                    learned_prob=float(daily["tornado_concern_prob"].max().item()) if "tornado_concern_prob" in daily else float("nan"),
                    raw_core_variant=raw_core_variant,
                    score_variant=score_variant,
                    component_variant=component_variant,
                    source_variant=source_variant,
                    core_variant=core_variant,
                    init_date=init_date or str(verification_payload.get("run_summary", {}).get("init_date", "")),
                    valid_date=valid_date,
                ),
            }
        )
    return pd.DataFrame(rows)


def apply_tornado_preference_mode(frame: pd.DataFrame, mode: str = "off") -> tuple[pd.DataFrame, pd.DataFrame]:
    adjusted = frame.copy()
    adjusted["ranking_tornado_concern_score"] = adjusted["tornado_concern_score"].astype(float)
    adjusted["preference_adjustment"] = 0.0
    adjusted["preference_applied"] = False
    changes: list[dict[str, Any]] = []
    if mode == "off":
        return adjusted, pd.DataFrame(columns=["init_date", "prior_top_valid_date", "new_top_valid_date", "preference_delta"])
    if mode not in {"conservative", "compact_tornado", "earliest_close"}:
        raise ValueError(f"unsupported tornado preference mode: {mode}")

    preference_columns: dict[str, float] = {
        "core_top": 0.0,
        "scp_top": 0.0,
        "tornado_signal": 0.0,
        "discriminator": 0.0,
        "core_compactness_proxy": 0.0,
        "core_purity_proxy": 0.0,
        "core_tornado_alignment_proxy": 0.0,
        "overlap_purity_proxy": 0.0,
        "broad_contamination_proxy": 0.0,
        "effective_core_to_context_ratio": 0.0,
    }
    for column, default in preference_columns.items():
        if column not in adjusted.columns:
            adjusted[column] = default

    for init_date, group in adjusted.groupby("init_date", sort=False):
        ranked = group.sort_values("ranking_tornado_concern_score", ascending=False)
        top = ranked.iloc[0]
        top_score = float(top["ranking_tornado_concern_score"])
        if top_score <= 0.0:
            continue

        close_margin = max(0.02, 0.35 * top_score)
        if mode == "earliest_close":
            top_valid_date = pd.to_datetime(str(top.get("valid_date", "")), errors="coerce")
            if pd.isna(top_valid_date):
                continue
            early_close_margin = max(0.02, 0.15 * top_score)
            earlier_rows = ranked.loc[
                (pd.to_datetime(ranked["valid_date"].astype(str), errors="coerce") < top_valid_date)
                & (ranked["ranking_tornado_concern_score"].astype(float) >= top_score - early_close_margin)
            ].copy()
            if earlier_rows.empty:
                continue
            candidate = earlier_rows.sort_values(["valid_date", "ranking_tornado_concern_score"], ascending=[True, False]).iloc[0]
            candidate_score = float(candidate["ranking_tornado_concern_score"])
            preference_delta = max(0.000001, top_score - candidate_score + 0.000001)
            adjusted.loc[candidate.name, "ranking_tornado_concern_score"] = candidate_score + preference_delta
            adjusted.loc[candidate.name, "preference_adjustment"] = preference_delta
            adjusted.loc[candidate.name, "preference_applied"] = True
            changes.append(
                {
                    "init_date": init_date,
                    "prior_top_valid_date": str(top["valid_date"]),
                    "new_top_valid_date": str(candidate["valid_date"]),
                    "preference_delta": float(preference_delta),
                }
            )
            continue

        if mode == "compact_tornado":
            compact_close_margin = max(0.08, 0.60 * top_score)
            broad_top = float(top.get("broad_contamination_proxy", 0.0) or 0.0) >= 0.35
            weak_top_alignment = float(top.get("core_tornado_alignment_proxy", 0.0) or 0.0) <= 0.70
            top_tornado_signal = max(float(top.get("tornado_signal", 0.0) or 0.0), 0.0)
            close_rows = ranked.loc[ranked["ranking_tornado_concern_score"] >= top_score - compact_close_margin].copy()
            candidates = close_rows.loc[
                (close_rows.index != top.name)
                & (close_rows["tornado_signal"].astype(float) >= max(0.0015, top_tornado_signal * 1.25))
                & (close_rows["discriminator"].astype(float) >= float(top["discriminator"]) + 0.025)
                & (close_rows["core_compactness_proxy"].astype(float) >= float(top["core_compactness_proxy"]) + 0.08)
                & (close_rows["core_tornado_alignment_proxy"].astype(float) >= float(top["core_tornado_alignment_proxy"]) + 0.08)
                & (close_rows["overlap_purity_proxy"].astype(float) >= float(top["overlap_purity_proxy"]) - 0.02)
                & (close_rows["broad_contamination_proxy"].astype(float) <= max(float(top["broad_contamination_proxy"]) - 0.06, 0.45))
            ].copy()
            if candidates.empty or not (broad_top or weak_top_alignment):
                continue

            candidate = candidates.sort_values(
                [
                    "core_tornado_alignment_proxy",
                    "core_purity_proxy",
                    "discriminator",
                    "tornado_signal",
                    "ranking_tornado_concern_score",
                ],
                ascending=False,
            ).iloc[0]
            candidate_score = float(candidate["ranking_tornado_concern_score"])
            preference_delta = min(
                compact_close_margin,
                max(0.0035, 0.06 * top_score, top_score - candidate_score + 0.001),
            )
            new_score = candidate_score + preference_delta
            if new_score <= top_score:
                continue
            adjusted.loc[candidate.name, "ranking_tornado_concern_score"] = new_score
            adjusted.loc[candidate.name, "preference_adjustment"] = preference_delta
            adjusted.loc[candidate.name, "preference_applied"] = True
            changes.append(
                {
                    "init_date": init_date,
                    "prior_top_valid_date": str(top["valid_date"]),
                    "new_top_valid_date": str(candidate["valid_date"]),
                    "preference_delta": float(preference_delta),
                }
            )
            continue

        close_rows = ranked.loc[ranked["ranking_tornado_concern_score"] >= top_score - close_margin].copy()
        candidates = close_rows.loc[
            (close_rows.index != top.name)
            & (close_rows["core_top"].astype(float) >= max(0.002, float(top["core_top"]) * 1.20))
            & (close_rows["scp_top"].astype(float) >= float(top["scp_top"]) + 0.05)
            & (close_rows["tornado_signal"].astype(float) >= float(top["tornado_signal"]) * 1.15)
            & (close_rows["discriminator"].astype(float) >= float(top["discriminator"]) + 0.03)
        ].copy()
        if candidates.empty:
            continue

        candidate = candidates.sort_values(
            ["discriminator", "tornado_signal", "core_top", "ranking_tornado_concern_score"],
            ascending=False,
        ).iloc[0]
        candidate_score = float(candidate["ranking_tornado_concern_score"])
        preference_delta = min(
            close_margin,
            max(0.0025, 0.08 * top_score, top_score - candidate_score + 0.001),
        )
        new_score = candidate_score + preference_delta
        if new_score <= top_score:
            continue
        adjusted.loc[candidate.name, "ranking_tornado_concern_score"] = new_score
        adjusted.loc[candidate.name, "preference_adjustment"] = preference_delta
        adjusted.loc[candidate.name, "preference_applied"] = True
        changes.append(
            {
                "init_date": init_date,
                "prior_top_valid_date": str(top["valid_date"]),
                "new_top_valid_date": str(candidate["valid_date"]),
                "preference_delta": float(preference_delta),
            }
        )
    return adjusted, pd.DataFrame(changes)


def _format_table(frame: pd.DataFrame) -> str:
    display_columns = [
        "init_date",
        "valid_date",
        "observed_category",
        "observed_tornado_outbreak",
        "observed_significant_tornado_support",
        "is_real_ingest",
        "tornado_concern_score",
        "learned_tornado_concern_prob",
        "support_top",
        "context_top",
        "joint_area",
        "core_top",
        "scp_top",
        "penalty_top",
        "sig_tor_support_max",
        "outbreak_risk_max",
        "tornado_overlap_max",
    ]
    formatted = frame.copy()
    for column in display_columns[4:]:
        if column in formatted:
            formatted[column] = formatted[column].map(lambda value: f"{float(value):.4f}")
    return formatted[display_columns].to_string(index=False)


def build_ranked_case_table(frame: pd.DataFrame, score_column: str = "ranking_tornado_concern_score") -> pd.DataFrame:
    ranked = frame.copy()
    for column, default in {
        "component_hail_dominance": 0.0,
        "component_tornado_agreement": 0.0,
        "hail_overlap_max": 0.0,
        "wind_overlap_max": 0.0,
        "raw_hail_prob_max": 0.0,
        "raw_wind_prob_max": 0.0,
    }.items():
        if column not in ranked.columns:
            ranked[column] = default
    ranked = ranked.sort_values(["init_date", score_column], ascending=[True, False]).copy()
    ranked["day_rank_within_init"] = ranked.groupby("init_date").cumcount() + 1
    top_scores = ranked.groupby("init_date")[score_column].transform("max")
    ranked["final_score_minus_window_top"] = ranked[score_column].astype(float) - top_scores.astype(float)
    summary = summarize_case_windows(frame, score_column=score_column)
    merged = ranked.merge(
        summary[
            [
                "init_date",
                "window_max_rank",
                "top_valid_date",
                "top_observed_category",
                "top_tornado_concern_score",
                "top_learned_tornado_concern_prob",
                "window_has_tornado_or_sigtor_day",
                "window_has_hail_outbreak_day",
                "window_has_outbreak_day",
                "hail_outranks_tornado_failure",
                "non_outbreak_outranks_outbreak_failure",
                "top_day_category_mismatch",
                "best_tornado_valid_date",
                "best_tornado_score",
                "top_minus_best_tornado_score",
                "top_minus_best_tornado_tornado_signal",
                "top_minus_best_tornado_broad_signal",
                "top_minus_best_tornado_learned_term",
                "top_minus_best_tornado_core_top",
                "top_minus_best_tornado_effective_core_top",
                "top_minus_best_tornado_scp_top",
                "top_minus_best_tornado_joint_area",
                "top_minus_best_tornado_tornado_overlap_max",
                "top_minus_best_tornado_sig_tor_support_max",
                "top_minus_best_tornado_context_top",
                "top_minus_best_tornado_effective_core_to_context_ratio",
                "top_minus_best_tornado_core_compactness_proxy",
                "top_minus_best_tornado_core_percentile_proxy",
                "top_minus_best_tornado_core_alignment_proxy",
                "top_minus_best_tornado_core_diffuseness_proxy",
                "top_minus_best_tornado_core_purity_proxy",
                "top_minus_best_tornado_normalized_synoptic_support",
                "top_minus_best_tornado_overlap_compactness_proxy",
                "top_minus_best_tornado_overlap_purity_proxy",
                "top_minus_best_tornado_core_tornado_alignment_proxy",
                "top_minus_best_tornado_support_tornado_alignment_proxy",
                "top_minus_best_tornado_broad_contamination_proxy",
                "top_minus_best_tornado_scp_support_ratio",
                "top_minus_best_tornado_overlap_to_context_ratio",
                "top_minus_best_tornado_core_to_context_ratio",
                "best_sig_tor_valid_date",
                "best_sig_tor_score",
                "top_minus_best_sig_tor_score",
                "failure_driver",
                "source_root_cause",
                "core_root_cause",
            ]
        ],
        on="init_date",
        how="left",
    )
    return merged[
        [
            "init_date",
            "valid_date",
            "day_rank_within_init",
            "window_max_rank",
            "observed_category",
            "observed_tornado_outbreak",
            "observed_significant_tornado_support",
            "is_real_ingest",
            "raw_core_variant_name",
            "raw_core_mask_factor",
            "raw_core_final_value",
            "raw_core_to_context_ratio",
            "core_variant_name",
            "source_variant_name",
            "component_variant_name",
            "score_variant_name",
            "effective_core_top",
            "effective_core_to_context_ratio",
            "core_compactness_proxy",
            "core_percentile_proxy",
            "core_alignment_proxy",
            "core_diffuseness_proxy",
            "core_purity_proxy",
            "core_support_interaction",
            "core_overlap_interaction",
            "core_scp_interaction",
            "core_penalty_term",
            "source_core_term",
            "source_overlap_term",
            "source_support_term",
            "source_penalty_term",
            "component_core_term",
            "component_overlap_term",
            "component_scp_term",
            "component_penalty_term",
            "component_hail_dominance",
            "component_tornado_agreement",
            "tornado_concern_score",
            "ranking_tornado_concern_score",
            "raw_base_score_before_variant",
            "variant_name",
            "variant_base_score",
            "variant_tornado_term",
            "variant_broad_term",
            "variant_learned_term",
            "variant_penalty_term",
            "variant_final_score_before_preference",
            "preference_adjustment",
            "preference_applied",
            "base_score",
            "tornado_signal",
            "broad_signal",
            "discriminator",
            "discriminator_multiplier",
            "normalized_synoptic_support",
            "support_top",
            "context_top",
            "joint_area",
            "core_top",
            "scp_top",
            "penalty_top",
            "sig_tor_support_max",
            "outbreak_risk_max",
            "tornado_overlap_max",
            "hail_overlap_max",
            "wind_overlap_max",
            "core_minus_context",
            "scp_x_core",
            "overlap_x_core",
            "support_x_overlap",
            "overlap_compactness_proxy",
            "overlap_purity_proxy",
            "core_tornado_alignment_proxy",
            "support_tornado_alignment_proxy",
            "broad_contamination_proxy",
            "scp_support_ratio",
            "overlap_to_context_ratio",
            "core_to_context_ratio",
            "raw_tornado_prob_max",
            "raw_hail_prob_max",
            "raw_wind_prob_max",
            "learned_tornado_concern_prob",
            "final_score_minus_window_top",
            "top_valid_date",
            "top_observed_category",
            "top_tornado_concern_score",
            "top_learned_tornado_concern_prob",
            "window_has_tornado_or_sigtor_day",
            "window_has_hail_outbreak_day",
            "window_has_outbreak_day",
            "hail_outranks_tornado_failure",
            "non_outbreak_outranks_outbreak_failure",
            "top_day_category_mismatch",
            "best_tornado_valid_date",
            "best_tornado_score",
            "top_minus_best_tornado_score",
            "top_minus_best_tornado_tornado_signal",
            "top_minus_best_tornado_broad_signal",
            "top_minus_best_tornado_learned_term",
            "top_minus_best_tornado_core_top",
            "top_minus_best_tornado_effective_core_top",
            "top_minus_best_tornado_scp_top",
            "top_minus_best_tornado_joint_area",
            "top_minus_best_tornado_tornado_overlap_max",
            "top_minus_best_tornado_sig_tor_support_max",
            "top_minus_best_tornado_context_top",
            "top_minus_best_tornado_effective_core_to_context_ratio",
            "top_minus_best_tornado_core_compactness_proxy",
            "top_minus_best_tornado_core_percentile_proxy",
            "top_minus_best_tornado_core_alignment_proxy",
            "top_minus_best_tornado_core_diffuseness_proxy",
            "top_minus_best_tornado_core_purity_proxy",
            "top_minus_best_tornado_normalized_synoptic_support",
            "top_minus_best_tornado_overlap_compactness_proxy",
            "top_minus_best_tornado_overlap_purity_proxy",
            "top_minus_best_tornado_core_tornado_alignment_proxy",
            "top_minus_best_tornado_support_tornado_alignment_proxy",
            "top_minus_best_tornado_broad_contamination_proxy",
            "top_minus_best_tornado_scp_support_ratio",
            "top_minus_best_tornado_overlap_to_context_ratio",
            "top_minus_best_tornado_core_to_context_ratio",
            "best_sig_tor_valid_date",
            "best_sig_tor_score",
            "top_minus_best_sig_tor_score",
            "failure_driver",
            "source_root_cause",
            "core_root_cause",
        ]
    ].reset_index(drop=True)


def summarize_case_windows(frame: pd.DataFrame, score_column: str = "ranking_tornado_concern_score") -> pd.DataFrame:
    ranked = frame.copy()
    defaults: dict[str, Any] = {
        "observed_category": "unknown",
        "observed_tornado_outbreak": 0,
        "observed_significant_tornado_support": 0,
        "learned_tornado_concern_prob": float("nan"),
        "ingest_source": "unknown",
        "is_real_ingest": False,
        "tornado_signal": 0.0,
        "broad_signal": 0.0,
        "variant_learned_term": 1.0,
        "core_top": 0.0,
        "raw_core_variant_name": "baseline",
        "raw_core_mask_factor": 1.0,
        "raw_core_final_value": 0.0,
        "raw_core_to_context_ratio": 0.0,
        "effective_core_top": 0.0,
        "scp_top": 0.0,
        "joint_area": 0.0,
        "tornado_overlap_max": 0.0,
        "sig_tor_support_max": 0.0,
        "context_top": 0.0,
        "effective_core_to_context_ratio": 0.0,
        "core_compactness_proxy": 0.0,
        "core_percentile_proxy": 0.0,
        "core_alignment_proxy": 0.0,
        "core_diffuseness_proxy": 0.0,
        "core_purity_proxy": 0.0,
        "core_support_interaction": 0.0,
        "core_overlap_interaction": 0.0,
        "core_scp_interaction": 0.0,
        "core_penalty_term": 1.0,
        "normalized_synoptic_support": 0.0,
        "overlap_compactness_proxy": 0.0,
        "overlap_purity_proxy": 0.0,
        "core_tornado_alignment_proxy": 0.0,
        "support_tornado_alignment_proxy": 0.0,
        "broad_contamination_proxy": 0.0,
        "scp_support_ratio": 0.0,
        "overlap_to_context_ratio": 0.0,
        "core_to_context_ratio": 0.0,
        "core_variant_name": "baseline",
        "source_variant_name": "baseline",
        "component_hail_dominance": 0.0,
        "component_tornado_agreement": 0.0,
        "hail_overlap_max": 0.0,
        "wind_overlap_max": 0.0,
        "raw_hail_prob_max": 0.0,
        "raw_wind_prob_max": 0.0,
    }
    for column, default in defaults.items():
        if column not in ranked.columns:
            ranked[column] = default
    if score_column not in ranked.columns:
        ranked[score_column] = ranked.get("tornado_concern_score", 0.0)
    ranked = ranked.sort_values(["init_date", score_column], ascending=[True, False]).copy()
    summary = (
        ranked.groupby("init_date", as_index=False)
        .first()[
            [
                "init_date",
                "valid_date",
                "observed_category",
                "observed_tornado_outbreak",
                "observed_significant_tornado_support",
                score_column,
                "learned_tornado_concern_prob",
                "ingest_source",
                "is_real_ingest",
                "tornado_signal",
                "broad_signal",
                "variant_learned_term",
                "core_top",
                "effective_core_top",
                "scp_top",
                "joint_area",
                "tornado_overlap_max",
                "sig_tor_support_max",
                "context_top",
                "effective_core_to_context_ratio",
                "core_compactness_proxy",
                "core_percentile_proxy",
                "core_alignment_proxy",
                "core_diffuseness_proxy",
                "core_purity_proxy",
                "core_support_interaction",
                "core_overlap_interaction",
                "core_scp_interaction",
                "core_penalty_term",
                "normalized_synoptic_support",
                "overlap_compactness_proxy",
                "overlap_purity_proxy",
                "core_tornado_alignment_proxy",
                "support_tornado_alignment_proxy",
                "broad_contamination_proxy",
                "scp_support_ratio",
                "overlap_to_context_ratio",
                "core_to_context_ratio",
                "core_variant_name",
                "source_variant_name",
            ]
        ]
        .rename(
            columns={
                "valid_date": "top_valid_date",
                "observed_category": "top_observed_category",
                "observed_tornado_outbreak": "top_observed_tornado_outbreak",
                "observed_significant_tornado_support": "top_observed_significant_tornado_support",
                score_column: "top_tornado_concern_score",
                "learned_tornado_concern_prob": "top_learned_tornado_concern_prob",
                "tornado_signal": "top_tornado_signal",
                "broad_signal": "top_broad_signal",
                "variant_learned_term": "top_variant_learned_term",
                "core_top": "top_core_top",
                "effective_core_top": "top_effective_core_top",
                "scp_top": "top_scp_top",
                "joint_area": "top_joint_area",
                "tornado_overlap_max": "top_tornado_overlap_max",
                "sig_tor_support_max": "top_sig_tor_support_max",
                "context_top": "top_context_top",
                "effective_core_to_context_ratio": "top_effective_core_to_context_ratio",
                "core_compactness_proxy": "top_core_compactness_proxy",
                "core_percentile_proxy": "top_core_percentile_proxy",
                "core_alignment_proxy": "top_core_alignment_proxy",
                "core_diffuseness_proxy": "top_core_diffuseness_proxy",
                "core_purity_proxy": "top_core_purity_proxy",
                "core_support_interaction": "top_core_support_interaction",
                "core_overlap_interaction": "top_core_overlap_interaction",
                "core_scp_interaction": "top_core_scp_interaction",
                "core_penalty_term": "top_core_penalty_term",
                "normalized_synoptic_support": "top_normalized_synoptic_support",
                "overlap_compactness_proxy": "top_overlap_compactness_proxy",
                "overlap_purity_proxy": "top_overlap_purity_proxy",
                "core_tornado_alignment_proxy": "top_core_tornado_alignment_proxy",
                "support_tornado_alignment_proxy": "top_support_tornado_alignment_proxy",
                "broad_contamination_proxy": "top_broad_contamination_proxy",
                "scp_support_ratio": "top_scp_support_ratio",
                "overlap_to_context_ratio": "top_overlap_to_context_ratio",
                "core_to_context_ratio": "top_core_to_context_ratio",
                "core_variant_name": "top_core_variant_name",
                "source_variant_name": "top_source_variant_name",
            }
        )
    )
    grouped = ranked.groupby("init_date")
    summary["window_has_tornado_or_sigtor_day"] = summary["init_date"].map(
        lambda init_date: bool(
            grouped.get_group(init_date)["observed_tornado_outbreak"].fillna(0).astype(int).gt(0).any()
            or grouped.get_group(init_date)["observed_significant_tornado_support"].fillna(0).astype(int).gt(0).any()
        )
    )
    summary["window_has_hail_outbreak_day"] = summary["init_date"].map(
        lambda init_date: bool(
            grouped.get_group(init_date)["observed_category"].astype(str).str.contains("hail_outbreak", case=False, na=False).any()
        )
    )
    summary["window_has_outbreak_day"] = summary["init_date"].map(
        lambda init_date: bool(
            grouped.get_group(init_date)["observed_category"].astype(str).str.contains("outbreak", case=False, na=False).any()
        )
    )
    summary["hail_outranks_tornado_failure"] = (
        summary["window_has_tornado_or_sigtor_day"]
        & summary["window_has_hail_outbreak_day"]
        & summary["top_observed_category"].astype(str).str.contains("hail_outbreak", case=False, na=False)
    )
    summary["non_outbreak_outranks_outbreak_failure"] = (
        summary["window_has_outbreak_day"]
        & summary["top_observed_category"].astype(str).str.contains("non_outbreak", case=False, na=False)
    )
    summary["top_day_category_mismatch"] = (
        summary["window_has_tornado_or_sigtor_day"]
        & ~summary["top_observed_category"].astype(str).str.contains("tornado_outbreak|significant_tornado_outbreak", case=False, na=False)
    )
    best_tornado_rows: list[dict[str, Any]] = []
    best_sig_rows: list[dict[str, Any]] = []
    for init_date, group in grouped:
        tornado_group = group.loc[group["observed_tornado_outbreak"].fillna(0).astype(int).gt(0)].sort_values(score_column, ascending=False)
        sig_group = group.loc[group["observed_significant_tornado_support"].fillna(0).astype(int).gt(0)].sort_values(score_column, ascending=False)
        best_tornado_rows.append(
            {
                "init_date": init_date,
                "best_tornado_valid_date": str(tornado_group.iloc[0]["valid_date"]) if not tornado_group.empty else "",
                "best_tornado_score": float(tornado_group.iloc[0][score_column]) if not tornado_group.empty else float("nan"),
                "best_tornado_tornado_signal": float(tornado_group.iloc[0]["tornado_signal"]) if not tornado_group.empty else float("nan"),
                "best_tornado_broad_signal": float(tornado_group.iloc[0]["broad_signal"]) if not tornado_group.empty else float("nan"),
                "best_tornado_learned_term": float(tornado_group.iloc[0]["variant_learned_term"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_top": float(tornado_group.iloc[0]["core_top"]) if not tornado_group.empty else float("nan"),
                "best_tornado_effective_core_top": float(tornado_group.iloc[0]["effective_core_top"]) if not tornado_group.empty else float("nan"),
                "best_tornado_scp_top": float(tornado_group.iloc[0]["scp_top"]) if not tornado_group.empty else float("nan"),
                "best_tornado_joint_area": float(tornado_group.iloc[0]["joint_area"]) if not tornado_group.empty else float("nan"),
                "best_tornado_tornado_overlap_max": float(tornado_group.iloc[0]["tornado_overlap_max"]) if not tornado_group.empty else float("nan"),
                "best_tornado_sig_tor_support_max": float(tornado_group.iloc[0]["sig_tor_support_max"]) if not tornado_group.empty else float("nan"),
                "best_tornado_context_top": float(tornado_group.iloc[0]["context_top"]) if not tornado_group.empty else float("nan"),
                "best_tornado_effective_core_to_context_ratio": float(tornado_group.iloc[0]["effective_core_to_context_ratio"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_compactness_proxy": float(tornado_group.iloc[0]["core_compactness_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_percentile_proxy": float(tornado_group.iloc[0]["core_percentile_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_alignment_proxy": float(tornado_group.iloc[0]["core_alignment_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_diffuseness_proxy": float(tornado_group.iloc[0]["core_diffuseness_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_purity_proxy": float(tornado_group.iloc[0]["core_purity_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_support_interaction": float(tornado_group.iloc[0]["core_support_interaction"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_overlap_interaction": float(tornado_group.iloc[0]["core_overlap_interaction"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_scp_interaction": float(tornado_group.iloc[0]["core_scp_interaction"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_penalty_term": float(tornado_group.iloc[0]["core_penalty_term"]) if not tornado_group.empty else float("nan"),
                "best_tornado_normalized_synoptic_support": float(tornado_group.iloc[0]["normalized_synoptic_support"]) if not tornado_group.empty else float("nan"),
                "best_tornado_overlap_compactness_proxy": float(tornado_group.iloc[0]["overlap_compactness_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_overlap_purity_proxy": float(tornado_group.iloc[0]["overlap_purity_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_tornado_alignment_proxy": float(tornado_group.iloc[0]["core_tornado_alignment_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_support_tornado_alignment_proxy": float(tornado_group.iloc[0]["support_tornado_alignment_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_broad_contamination_proxy": float(tornado_group.iloc[0]["broad_contamination_proxy"]) if not tornado_group.empty else float("nan"),
                "best_tornado_scp_support_ratio": float(tornado_group.iloc[0]["scp_support_ratio"]) if not tornado_group.empty else float("nan"),
                "best_tornado_overlap_to_context_ratio": float(tornado_group.iloc[0]["overlap_to_context_ratio"]) if not tornado_group.empty else float("nan"),
                "best_tornado_core_to_context_ratio": float(tornado_group.iloc[0]["core_to_context_ratio"]) if not tornado_group.empty else float("nan"),
            }
        )
        best_sig_rows.append(
            {
                "init_date": init_date,
                "best_sig_tor_valid_date": str(sig_group.iloc[0]["valid_date"]) if not sig_group.empty else "",
                "best_sig_tor_score": float(sig_group.iloc[0][score_column]) if not sig_group.empty else float("nan"),
            }
        )
    summary = summary.merge(pd.DataFrame(best_tornado_rows), on="init_date", how="left")
    summary = summary.merge(pd.DataFrame(best_sig_rows), on="init_date", how="left")
    summary["top_minus_best_tornado_score"] = summary["top_tornado_concern_score"].astype(float) - summary["best_tornado_score"].astype(float)
    summary["top_minus_best_sig_tor_score"] = summary["top_tornado_concern_score"].astype(float) - summary["best_sig_tor_score"].astype(float)
    summary["top_minus_best_tornado_tornado_signal"] = summary["top_tornado_signal"].astype(float) - summary["best_tornado_tornado_signal"].astype(float)
    summary["top_minus_best_tornado_broad_signal"] = summary["top_broad_signal"].astype(float) - summary["best_tornado_broad_signal"].astype(float)
    summary["top_minus_best_tornado_learned_term"] = summary["top_variant_learned_term"].astype(float) - summary["best_tornado_learned_term"].astype(float)
    summary["top_minus_best_tornado_core_top"] = summary["top_core_top"].astype(float) - summary["best_tornado_core_top"].astype(float)
    summary["top_minus_best_tornado_effective_core_top"] = summary["top_effective_core_top"].astype(float) - summary["best_tornado_effective_core_top"].astype(float)
    summary["top_minus_best_tornado_scp_top"] = summary["top_scp_top"].astype(float) - summary["best_tornado_scp_top"].astype(float)
    summary["top_minus_best_tornado_joint_area"] = summary["top_joint_area"].astype(float) - summary["best_tornado_joint_area"].astype(float)
    summary["top_minus_best_tornado_tornado_overlap_max"] = summary["top_tornado_overlap_max"].astype(float) - summary["best_tornado_tornado_overlap_max"].astype(float)
    summary["top_minus_best_tornado_sig_tor_support_max"] = summary["top_sig_tor_support_max"].astype(float) - summary["best_tornado_sig_tor_support_max"].astype(float)
    summary["top_minus_best_tornado_context_top"] = summary["top_context_top"].astype(float) - summary["best_tornado_context_top"].astype(float)
    summary["top_minus_best_tornado_effective_core_to_context_ratio"] = summary["top_effective_core_to_context_ratio"].astype(float) - summary["best_tornado_effective_core_to_context_ratio"].astype(float)
    summary["top_minus_best_tornado_core_compactness_proxy"] = summary["top_core_compactness_proxy"].astype(float) - summary["best_tornado_core_compactness_proxy"].astype(float)
    summary["top_minus_best_tornado_core_percentile_proxy"] = summary["top_core_percentile_proxy"].astype(float) - summary["best_tornado_core_percentile_proxy"].astype(float)
    summary["top_minus_best_tornado_core_alignment_proxy"] = summary["top_core_alignment_proxy"].astype(float) - summary["best_tornado_core_alignment_proxy"].astype(float)
    summary["top_minus_best_tornado_core_diffuseness_proxy"] = summary["top_core_diffuseness_proxy"].astype(float) - summary["best_tornado_core_diffuseness_proxy"].astype(float)
    summary["top_minus_best_tornado_core_purity_proxy"] = summary["top_core_purity_proxy"].astype(float) - summary["best_tornado_core_purity_proxy"].astype(float)
    summary["top_minus_best_tornado_normalized_synoptic_support"] = summary["top_normalized_synoptic_support"].astype(float) - summary["best_tornado_normalized_synoptic_support"].astype(float)
    summary["top_minus_best_tornado_overlap_compactness_proxy"] = summary["top_overlap_compactness_proxy"].astype(float) - summary["best_tornado_overlap_compactness_proxy"].astype(float)
    summary["top_minus_best_tornado_overlap_purity_proxy"] = summary["top_overlap_purity_proxy"].astype(float) - summary["best_tornado_overlap_purity_proxy"].astype(float)
    summary["top_minus_best_tornado_core_tornado_alignment_proxy"] = summary["top_core_tornado_alignment_proxy"].astype(float) - summary["best_tornado_core_tornado_alignment_proxy"].astype(float)
    summary["top_minus_best_tornado_support_tornado_alignment_proxy"] = summary["top_support_tornado_alignment_proxy"].astype(float) - summary["best_tornado_support_tornado_alignment_proxy"].astype(float)
    summary["top_minus_best_tornado_broad_contamination_proxy"] = summary["top_broad_contamination_proxy"].astype(float) - summary["best_tornado_broad_contamination_proxy"].astype(float)
    summary["top_minus_best_tornado_scp_support_ratio"] = summary["top_scp_support_ratio"].astype(float) - summary["best_tornado_scp_support_ratio"].astype(float)
    summary["top_minus_best_tornado_overlap_to_context_ratio"] = summary["top_overlap_to_context_ratio"].astype(float) - summary["best_tornado_overlap_to_context_ratio"].astype(float)
    summary["top_minus_best_tornado_core_to_context_ratio"] = summary["top_core_to_context_ratio"].astype(float) - summary["best_tornado_core_to_context_ratio"].astype(float)

    def _failure_driver(row: pd.Series) -> str:
        # Ordered to isolate the dominant component failure in flagged real windows.
        top_tornado_signal = float(row.get("top_tornado_signal", float("nan")))
        best_tornado_signal = float(row.get("best_tornado_tornado_signal", float("nan")))
        if np.isfinite(top_tornado_signal) and np.isfinite(best_tornado_signal) and top_tornado_signal < 0.01 and best_tornado_signal < 0.01:
            return "tornado_signal_near_zero_for_both"
        if float(row.get("top_minus_best_tornado_context_top", 0.0)) >= 0.10 and float(row.get("top_minus_best_tornado_core_top", 0.0)) >= -0.005:
            return "context_swamps_core"
        if float(row.get("top_minus_best_tornado_core_top", 0.0)) >= 0.01 and float(row.get("top_minus_best_tornado_tornado_overlap_max", 0.0)) <= 0.0:
            return "generic_core_beats_tornado_core"
        if float(row.get("top_minus_best_tornado_tornado_overlap_max", 0.0)) <= -0.20:
            return "overlap_too_weak"
        if float(row.get("top_minus_best_tornado_scp_top", 0.0)) <= -0.08:
            return "scp_not_separating"
        top_minus_best = row.get("top_minus_best_sig_tor_score")
        if pd.isna(top_minus_best):
            top_minus_best = row.get("top_minus_best_tornado_score")
        if pd.notna(top_minus_best) and float(top_minus_best) <= 0.02:
            return "near_tie_small_margin"
        learned_delta = row.get("top_minus_best_tornado_learned_term")
        if pd.notna(learned_delta) and float(learned_delta) >= 0.08:
            return "learned_term_dominates"
        if bool(row.get("hail_outranks_tornado_failure")):
            return "broad_context_dominates"
        if bool(row.get("non_outbreak_outranks_outbreak_failure")):
            return "weak_tornado_core_signal"
        return "none"

    summary["failure_driver"] = summary.apply(_failure_driver, axis=1)

    def _source_root_cause(row: pd.Series) -> str:
        # Source-field rules are ordered from strongest structural contamination signal to weakest.
        top_tornado_signal = float(row.get("top_tornado_signal", float("nan")))
        best_tornado_signal = float(row.get("best_tornado_tornado_signal", float("nan")))
        if np.isfinite(top_tornado_signal) and np.isfinite(best_tornado_signal) and top_tornado_signal < 0.01 and best_tornado_signal < 0.01:
            return "upstream_tornado_signal_absent"
        if float(row.get("top_minus_best_tornado_broad_contamination_proxy", 0.0)) >= 0.10 and float(row.get("top_minus_best_tornado_overlap_purity_proxy", 0.0)) <= 0.0:
            return "generic_overlap_contamination"
        if float(row.get("top_minus_best_tornado_joint_area", 0.0)) >= 0.0001 and float(row.get("top_minus_best_tornado_overlap_compactness_proxy", 0.0)) <= -0.05:
            return "broad_area_beats_compact_signal"
        if float(row.get("top_minus_best_tornado_core_tornado_alignment_proxy", 0.0)) <= -0.05:
            return "poor_tornado_purity"
        if float(row.get("top_minus_best_tornado_scp_support_ratio", 0.0)) <= -0.20 and float(row.get("top_minus_best_tornado_support_tornado_alignment_proxy", 0.0)) <= 0.0:
            return "scp_underleveraged_relative_to_support"
        return "none"

    summary["source_root_cause"] = summary.apply(_source_root_cause, axis=1)

    def _core_root_cause(row: pd.Series) -> str:
        # Core-specific rules isolate whether the bad ranking is driven by diffuse, spikey, or poorly aligned core construction.
        top_tornado_signal = float(row.get("top_tornado_signal", float("nan")))
        best_tornado_signal = float(row.get("best_tornado_tornado_signal", float("nan")))
        if np.isfinite(top_tornado_signal) and np.isfinite(best_tornado_signal) and top_tornado_signal < 0.01 and best_tornado_signal < 0.01:
            return "upstream_core_absent"
        if float(row.get("top_minus_best_tornado_core_diffuseness_proxy", 0.0)) >= 0.08 and float(row.get("top_minus_best_tornado_core_compactness_proxy", 0.0)) <= 0.0:
            return "diffuse_core_beats_compact_tornado_core"
        if float(row.get("top_minus_best_tornado_core_top", 0.0)) >= 0.008 and float(row.get("top_minus_best_tornado_core_percentile_proxy", 0.0)) <= -0.05:
            return "max_spike_without_purity"
        if float(row.get("top_minus_best_tornado_core_alignment_proxy", 0.0)) <= -0.05 and float(row.get("top_minus_best_tornado_sig_tor_support_max", 0.0)) <= 0.0:
            return "core_not_aligned_with_tornado_support"
        if float(row.get("top_minus_best_tornado_effective_core_to_context_ratio", 0.0)) >= 0.004 and float(row.get("top_minus_best_tornado_context_top", 0.0)) >= 0.08:
            return "context_dilution_of_tornado_core"
        return "none"

    summary["core_root_cause"] = summary.apply(_core_root_cause, axis=1)
    summary["window_max_rank"] = summary["top_tornado_concern_score"].rank(method="dense", ascending=False).astype(int)
    return summary.sort_values(["window_max_rank", "init_date"]).reset_index(drop=True)


def summarize_failure_patterns(window_summary: pd.DataFrame) -> pd.DataFrame:
    real_windows = window_summary.loc[window_summary["is_real_ingest"].fillna(False)].copy()
    rows = [
        {
            "pattern": "hail_outranks_tornado_failure",
            "count_real_windows": int(real_windows["hail_outranks_tornado_failure"].fillna(False).sum()),
        },
        {
            "pattern": "non_outbreak_outranks_outbreak_failure",
            "count_real_windows": int(real_windows["non_outbreak_outranks_outbreak_failure"].fillna(False).sum()),
        },
        {
            "pattern": "top_day_category_mismatch",
            "count_real_windows": int(real_windows["top_day_category_mismatch"].fillna(False).sum()),
        },
    ]
    return pd.DataFrame(rows)


def summarize_score_variants(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in SCORE_VARIANTS:
        variant_frame = _apply_score_variant_to_frame(frame, variant=variant)
        variant_frame["ranking_tornado_concern_score"] = variant_frame["tornado_concern_score"].astype(float)
        summary = summarize_case_windows(variant_frame)
        real_summary = summary.loc[summary["is_real_ingest"].fillna(False)].copy()
        top_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "top_valid_date"]
        top_20240426 = summary.loc[summary["init_date"] == "2024-04-26", "top_valid_date"]
        rows.append(
            {
                "variant": variant,
                "hail_outranks_tornado_failure_real": int(real_summary["hail_outranks_tornado_failure"].fillna(False).sum()),
                "non_outbreak_outranks_outbreak_failure_real": int(real_summary["non_outbreak_outranks_outbreak_failure"].fillna(False).sum()),
                "top_day_category_mismatch_real": int(real_summary["top_day_category_mismatch"].fillna(False).sum()),
                "top_day_for_2024_03_14_window": str(top_20240314.iloc[0]) if not top_20240314.empty else "",
                "top_day_for_2024_04_26_window": str(top_20240426.iloc[0]) if not top_20240426.empty else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_component_variants_from_artifacts(
    artifact_pairs: list[tuple[str, Path, Path]],
    raw_core_variant: str = "baseline",
    score_variant: str = "baseline",
    source_variant: str = "baseline",
    core_variant: str = "baseline",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component_variant in COMPONENT_VARIANTS:
        tables: list[pd.DataFrame] = []
        for init_date, prediction_path, verification_path in artifact_pairs:
            tables.append(
                evaluate_tornado_concern_for_artifacts(
                    prediction_path,
                    verification_path,
                    init_date=init_date,
                    raw_core_variant=raw_core_variant,
                    score_variant=score_variant,
                    component_variant=component_variant,
                    source_variant=source_variant,
                    core_variant=core_variant,
                )
            )
        frame = pd.concat(tables, ignore_index=True)
        frame["ranking_tornado_concern_score"] = frame["tornado_concern_score"].astype(float)
        summary = summarize_case_windows(frame)
        real_summary = summary.loc[summary["is_real_ingest"].fillna(False)].copy()
        top_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "top_valid_date"]
        top_20240426 = summary.loc[summary["init_date"] == "2024-04-26", "top_valid_date"]
        margin_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "top_minus_best_tornado_score"]
        rows.append(
            {
                "raw_core_variant": raw_core_variant,
                "core_variant": core_variant,
                "component_variant": component_variant,
                "score_variant": score_variant,
                "hail_outranks_tornado_failure_real": int(real_summary["hail_outranks_tornado_failure"].fillna(False).sum()),
                "non_outbreak_outranks_outbreak_failure_real": int(real_summary["non_outbreak_outranks_outbreak_failure"].fillna(False).sum()),
                "top_day_category_mismatch_real": int(real_summary["top_day_category_mismatch"].fillna(False).sum()),
                "top_day_for_2024_03_14_window": str(top_20240314.iloc[0]) if not top_20240314.empty else "",
                "top_day_for_2024_04_26_window": str(top_20240426.iloc[0]) if not top_20240426.empty else "",
                "margin_top_minus_best_tornado_for_2024_03_14": float(margin_20240314.iloc[0]) if not margin_20240314.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_source_variants_from_artifacts(
    artifact_pairs: list[tuple[str, Path, Path]],
    raw_core_variant: str = "baseline",
    component_variant: str = "baseline",
    score_variant: str = "baseline",
    core_variant: str = "baseline",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_variant in SOURCE_VARIANTS:
        tables: list[pd.DataFrame] = []
        for init_date, prediction_path, verification_path in artifact_pairs:
            tables.append(
                evaluate_tornado_concern_for_artifacts(
                    prediction_path,
                    verification_path,
                    init_date=init_date,
                    raw_core_variant=raw_core_variant,
                    score_variant=score_variant,
                    component_variant=component_variant,
                    source_variant=source_variant,
                    core_variant=core_variant,
                )
            )
        frame = pd.concat(tables, ignore_index=True)
        frame["ranking_tornado_concern_score"] = frame["tornado_concern_score"].astype(float)
        summary = summarize_case_windows(frame)
        real_summary = summary.loc[summary["is_real_ingest"].fillna(False)].copy()
        top_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "top_valid_date"]
        top_20240426 = summary.loc[summary["init_date"] == "2024-04-26", "top_valid_date"]
        margin_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "top_minus_best_tornado_score"]
        source_root_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "source_root_cause"]
        rows.append(
            {
                "raw_core_variant": raw_core_variant,
                "core_variant": core_variant,
                "source_variant": source_variant,
                "component_variant": component_variant,
                "score_variant": score_variant,
                "hail_outranks_tornado_failure_real": int(real_summary["hail_outranks_tornado_failure"].fillna(False).sum()),
                "non_outbreak_outranks_outbreak_failure_real": int(real_summary["non_outbreak_outranks_outbreak_failure"].fillna(False).sum()),
                "top_day_category_mismatch_real": int(real_summary["top_day_category_mismatch"].fillna(False).sum()),
                "top_day_for_2024_03_14_window": str(top_20240314.iloc[0]) if not top_20240314.empty else "",
                "top_day_for_2024_04_26_window": str(top_20240426.iloc[0]) if not top_20240426.empty else "",
                "margin_top_minus_best_tornado_for_2024_03_14": float(margin_20240314.iloc[0]) if not margin_20240314.empty else float("nan"),
                "source_root_cause_for_2024_03_14": str(source_root_20240314.iloc[0]) if not source_root_20240314.empty else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_core_variants_from_artifacts(
    artifact_pairs: list[tuple[str, Path, Path]],
    raw_core_variant: str = "baseline",
    source_variant: str = "baseline",
    component_variant: str = "baseline",
    score_variant: str = "baseline",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for core_variant in CORE_VARIANTS:
        tables: list[pd.DataFrame] = []
        for init_date, prediction_path, verification_path in artifact_pairs:
            tables.append(
                evaluate_tornado_concern_for_artifacts(
                    prediction_path,
                    verification_path,
                    init_date=init_date,
                    raw_core_variant=raw_core_variant,
                    score_variant=score_variant,
                    component_variant=component_variant,
                    source_variant=source_variant,
                    core_variant=core_variant,
                )
            )
        frame = pd.concat(tables, ignore_index=True)
        frame["ranking_tornado_concern_score"] = frame["tornado_concern_score"].astype(float)
        summary = summarize_case_windows(frame)
        real_summary = summary.loc[summary["is_real_ingest"].fillna(False)].copy()
        top_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "top_valid_date"]
        top_20240426 = summary.loc[summary["init_date"] == "2024-04-26", "top_valid_date"]
        margin_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "top_minus_best_tornado_score"]
        root_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "core_root_cause"]
        rows.append(
            {
                "core_variant": core_variant,
                "source_variant": source_variant,
                "component_variant": component_variant,
                "score_variant": score_variant,
                "hail_outranks_tornado_failure_real": int(real_summary["hail_outranks_tornado_failure"].fillna(False).sum()),
                "non_outbreak_outranks_outbreak_failure_real": int(real_summary["non_outbreak_outranks_outbreak_failure"].fillna(False).sum()),
                "top_day_category_mismatch_real": int(real_summary["top_day_category_mismatch"].fillna(False).sum()),
                "top_day_for_2024_03_14_window": str(top_20240314.iloc[0]) if not top_20240314.empty else "",
                "top_day_for_2024_04_26_window": str(top_20240426.iloc[0]) if not top_20240426.empty else "",
                "margin_top_minus_best_tornado_for_2024_03_14": float(margin_20240314.iloc[0]) if not margin_20240314.empty else float("nan"),
                "core_root_cause_for_2024_03_14": str(root_20240314.iloc[0]) if not root_20240314.empty else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_raw_core_variants_from_artifacts(
    artifact_pairs: list[tuple[str, Path, Path]],
    core_variant: str = "baseline",
    source_variant: str = "baseline",
    component_variant: str = "baseline",
    score_variant: str = "baseline",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_core_variant in RAW_CORE_VARIANTS:
        tables: list[pd.DataFrame] = []
        for init_date, prediction_path, verification_path in artifact_pairs:
            tables.append(
                evaluate_tornado_concern_for_artifacts(
                    prediction_path,
                    verification_path,
                    init_date=init_date,
                    raw_core_variant=raw_core_variant,
                    core_variant=core_variant,
                    source_variant=source_variant,
                    component_variant=component_variant,
                    score_variant=score_variant,
                )
            )
        frame = pd.concat(tables, ignore_index=True)
        frame["ranking_tornado_concern_score"] = frame["tornado_concern_score"].astype(float)
        summary = summarize_case_windows(frame)
        real_summary = summary.loc[summary["is_real_ingest"].fillna(False)].copy()
        top_20240314 = summary.loc[summary["init_date"] == "2024-03-14", "top_valid_date"]
        top_20240426 = summary.loc[summary["init_date"] == "2024-04-26", "top_valid_date"]
        rows.append(
            {
                "raw_core_variant": raw_core_variant,
                "core_variant": core_variant,
                "source_variant": source_variant,
                "component_variant": component_variant,
                "score_variant": score_variant,
                "top_day_for_2024_03_14_window": str(top_20240314.iloc[0]) if not top_20240314.empty else "",
                "top_day_for_2024_04_26_window": str(top_20240426.iloc[0]) if not top_20240426.empty else "",
                "hail_outranks_tornado_failure_real": int(real_summary["hail_outranks_tornado_failure"].fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_broader_validation_checkpoint_from_artifacts(
    artifact_pairs: list[tuple[str, Path, Path]],
    core_variant: str = "baseline",
    source_variant: str = "baseline",
    component_variant: str = "baseline",
    score_variant: str = "baseline",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_core_variant in BROADER_VALIDATION_RAW_CORE_VARIANTS:
        tables: list[pd.DataFrame] = []
        for init_date, prediction_path, verification_path in artifact_pairs:
            tables.append(
                evaluate_tornado_concern_for_artifacts(
                    prediction_path,
                    verification_path,
                    init_date=init_date,
                    raw_core_variant=raw_core_variant,
                    core_variant=core_variant,
                    source_variant=source_variant,
                    component_variant=component_variant,
                    score_variant=score_variant,
                )
            )
        frame = pd.concat(tables, ignore_index=True)
        frame["ranking_tornado_concern_score"] = frame["tornado_concern_score"].astype(float)
        summary = summarize_case_windows(frame)
        real_summary = summary.loc[summary["is_real_ingest"].fillna(False)].copy()
        tornado_windows = real_summary.loc[real_summary["best_tornado_valid_date"].astype(str).ne("")]
        rows.append(
            {
                "raw_core_variant": raw_core_variant,
                "real_ingest_windows": int(len(real_summary)),
                "hail_outranks_tornado_failure_real": int(real_summary["hail_outranks_tornado_failure"].fillna(False).sum()),
                "non_outbreak_outranks_outbreak_failure_real": int(real_summary["non_outbreak_outranks_outbreak_failure"].fillna(False).sum()),
                "top_day_category_mismatch_real": int(real_summary["top_day_category_mismatch"].fillna(False).sum()),
                "tornado_day_ranked_first_real": int(real_summary["top_observed_tornado_outbreak"].fillna(0).astype(int).sum()),
                "sig_tor_day_ranked_first_real": int(real_summary["top_observed_significant_tornado_support"].fillna(0).astype(int).sum()),
                "mean_top_minus_best_tornado_margin_real": float(tornado_windows["top_minus_best_tornado_score"].astype(float).mean()) if not tornado_windows.empty else float("nan"),
                "median_top_minus_best_tornado_margin_real": float(tornado_windows["top_minus_best_tornado_score"].astype(float).median()) if not tornado_windows.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_baseline_vs_masked_core_steeper_from_artifacts(
    artifact_pairs: list[tuple[str, Path, Path]],
    core_variant: str = "baseline",
    source_variant: str = "baseline",
    component_variant: str = "baseline",
    score_variant: str = "baseline",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_core_variant in BASELINE_VS_MASKED_CORE_STEEPER_VARIANTS:
        tables: list[pd.DataFrame] = []
        for init_date, prediction_path, verification_path in artifact_pairs:
            tables.append(
                evaluate_tornado_concern_for_artifacts(
                    prediction_path,
                    verification_path,
                    init_date=init_date,
                    raw_core_variant=raw_core_variant,
                    core_variant=core_variant,
                    source_variant=source_variant,
                    component_variant=component_variant,
                    score_variant=score_variant,
                )
            )
        frame = pd.concat(tables, ignore_index=True)
        frame["ranking_tornado_concern_score"] = frame["tornado_concern_score"].astype(float)
        summary = summarize_case_windows(frame)
        real_summary = summary.loc[summary["is_real_ingest"].fillna(False)].copy()
        tornado_windows = real_summary.loc[real_summary["best_tornado_valid_date"].astype(str).ne("")]
        rows.append(
            {
                "raw_core_variant": raw_core_variant,
                "real_ingest_windows": int(len(real_summary)),
                "hail_outranks_tornado_failure_real": int(real_summary["hail_outranks_tornado_failure"].fillna(False).sum()),
                "non_outbreak_outranks_outbreak_failure_real": int(real_summary["non_outbreak_outranks_outbreak_failure"].fillna(False).sum()),
                "top_day_category_mismatch_real": int(real_summary["top_day_category_mismatch"].fillna(False).sum()),
                "tornado_day_ranked_first_real": int(real_summary["top_observed_tornado_outbreak"].fillna(0).astype(int).sum()),
                "sig_tor_day_ranked_first_real": int(real_summary["top_observed_significant_tornado_support"].fillna(0).astype(int).sum()),
                "mean_top_minus_best_tornado_margin_real": float(tornado_windows["top_minus_best_tornado_score"].astype(float).mean()) if not tornado_windows.empty else float("nan"),
                "median_top_minus_best_tornado_margin_real": float(tornado_windows["top_minus_best_tornado_score"].astype(float).median()) if not tornado_windows.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_component_checkpoint_from_artifacts(
    artifact_pairs: list[tuple[str, Path, Path]],
    *,
    raw_core_variant: str = "baseline",
    core_variant: str = "baseline",
    source_variant: str = "baseline",
    score_variant: str = "baseline",
    requested_case_count: int | None = None,
    skipped_missing_count: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component_variant in CHECKPOINT_COMPONENT_VARIANTS:
        tables: list[pd.DataFrame] = []
        for init_date, prediction_path, verification_path in artifact_pairs:
            tables.append(
                evaluate_tornado_concern_for_artifacts(
                    prediction_path,
                    verification_path,
                    init_date=init_date,
                    raw_core_variant=raw_core_variant,
                    core_variant=core_variant,
                    source_variant=source_variant,
                    component_variant=component_variant,
                    score_variant=score_variant,
                )
            )
        if tables:
            frame = pd.concat(tables, ignore_index=True)
            frame["ranking_tornado_concern_score"] = frame["tornado_concern_score"].astype(float)
            summary = summarize_case_windows(frame)
        else:
            summary = pd.DataFrame()
        real_summary = summary.loc[summary["is_real_ingest"].fillna(False)].copy() if not summary.empty else pd.DataFrame()
        tornado_windows = real_summary.loc[real_summary["best_tornado_valid_date"].astype(str).ne("")] if not real_summary.empty else pd.DataFrame()
        rows.append(
            {
                "component_variant": component_variant,
                "raw_core_variant": raw_core_variant,
                "core_variant": core_variant,
                "source_variant": source_variant,
                "score_variant": score_variant,
                "requested_case_count": int(requested_case_count if requested_case_count is not None else len(artifact_pairs)),
                "ready_case_count": int(len(real_summary)),
                "artifact_pair_count": int(len(artifact_pairs)),
                "skipped_missing_count": int(skipped_missing_count),
                "hail_outranks_tornado_failure_real": int(real_summary["hail_outranks_tornado_failure"].fillna(False).sum()) if not real_summary.empty else 0,
                "non_outbreak_outranks_outbreak_failure_real": int(real_summary["non_outbreak_outranks_outbreak_failure"].fillna(False).sum()) if not real_summary.empty else 0,
                "top_day_category_mismatch_real": int(real_summary["top_day_category_mismatch"].fillna(False).sum()) if not real_summary.empty else 0,
                "tornado_day_ranked_first_real": int(real_summary["top_observed_tornado_outbreak"].fillna(0).astype(int).sum()) if not real_summary.empty else 0,
                "sig_tor_day_ranked_first_real": int(real_summary["top_observed_significant_tornado_support"].fillna(0).astype(int).sum()) if not real_summary.empty else 0,
                "mean_top_minus_best_tornado_margin_real": float(tornado_windows["top_minus_best_tornado_score"].astype(float).mean()) if not tornado_windows.empty else float("nan"),
                "median_top_minus_best_tornado_margin_real": float(tornado_windows["top_minus_best_tornado_score"].astype(float).median()) if not tornado_windows.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_component_checkpoint_markdown(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tornado Concern Component Checkpoint",
        "",
        "- Comparison: baseline vs tornado_hail_separation_v1",
        "- Acceptance: challenger is useful only if it reduces hail/top-day failures without increasing non_outbreak_outranks_outbreak_failure_real.",
        "",
        "## Side-by-side Metrics",
        "",
        _markdown_table(
            frame,
            [
                "component_variant",
                "ready_case_count",
                "skipped_missing_count",
                "hail_outranks_tornado_failure_real",
                "non_outbreak_outranks_outbreak_failure_real",
                "top_day_category_mismatch_real",
                "tornado_day_ranked_first_real",
                "sig_tor_day_ranked_first_real",
                "mean_top_minus_best_tornado_margin_real",
                "median_top_minus_best_tornado_margin_real",
            ],
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_window_summary(frame: pd.DataFrame) -> str:
    display_columns = [
        "window_max_rank",
        "init_date",
        "top_valid_date",
        "top_observed_category",
        "top_observed_tornado_outbreak",
        "top_observed_significant_tornado_support",
        "is_real_ingest",
        "hail_outranks_tornado_failure",
        "non_outbreak_outranks_outbreak_failure",
        "top_day_category_mismatch",
        "failure_driver",
        "top_tornado_concern_score",
        "top_learned_tornado_concern_prob",
    ]
    formatted = frame.copy()
    formatted["top_tornado_concern_score"] = formatted["top_tornado_concern_score"].map(lambda value: f"{float(value):.4f}")
    formatted["top_learned_tornado_concern_prob"] = formatted["top_learned_tornado_concern_prob"].map(
        lambda value: "nan" if pd.isna(value) else f"{float(value):.4f}"
    )
    return formatted[display_columns].to_string(index=False)


def _format_failure_summary(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False)


def _write_ranked_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty or any(column not in frame.columns for column in columns):
        return "_none_"
    subset = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in record) + " |"
        for record in subset.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _window_failure_comparison_markdown(ranked_frame: pd.DataFrame, window_summary: pd.DataFrame) -> str:
    sections: list[str] = []
    flagged_windows = window_summary.loc[
        window_summary["is_real_ingest"].fillna(False)
        & window_summary[
            ["hail_outranks_tornado_failure", "non_outbreak_outranks_outbreak_failure", "top_day_category_mismatch"]
        ].fillna(False).any(axis=1)
    ].copy()
    for row in flagged_windows.itertuples(index=False):
        window_rows = ranked_frame.loc[ranked_frame["init_date"] == row.init_date].copy()
        top_row = window_rows.sort_values("ranking_tornado_concern_score", ascending=False).iloc[[0]].assign(comparison_role="top_ranked_day")
        tornado_rows = window_rows.loc[window_rows["observed_tornado_outbreak"].fillna(0).astype(int).gt(0)]
        sig_rows = window_rows.loc[window_rows["observed_significant_tornado_support"].fillna(0).astype(int).gt(0)]
        compare_frames = [top_row]
        if not tornado_rows.empty:
            compare_frames.append(tornado_rows.sort_values("ranking_tornado_concern_score", ascending=False).iloc[[0]].assign(comparison_role="best_tornado_day"))
        if not sig_rows.empty:
            compare_frames.append(sig_rows.sort_values("ranking_tornado_concern_score", ascending=False).iloc[[0]].assign(comparison_role="best_sig_tor_day"))
        compare = pd.concat(compare_frames, ignore_index=True).drop_duplicates(subset=["valid_date", "comparison_role"])
        top_valid_date = str(row.top_valid_date)
        best_tornado_valid_date = str(row.best_tornado_valid_date) if getattr(row, "best_tornado_valid_date", "") else ""
        best_sig_valid_date = str(row.best_sig_tor_valid_date) if getattr(row, "best_sig_tor_valid_date", "") else ""
        sections.extend(
            [
                f"### {row.init_date}",
                "",
                f"- top_ranked_day: {top_valid_date}",
                f"- best_tornado_day: {best_tornado_valid_date}",
                f"- best_sig_tor_day: {best_sig_valid_date}",
                f"- failure_driver: {row.failure_driver}",
                f"- top_minus_best_tornado_score: {'' if pd.isna(row.top_minus_best_tornado_score) else f'{float(row.top_minus_best_tornado_score):.4f}'}",
                f"- top_minus_best_sig_tor_score: {'' if pd.isna(row.top_minus_best_sig_tor_score) else f'{float(row.top_minus_best_sig_tor_score):.4f}'}",
                f"- top_minus_best_tornado_tornado_signal: {'' if pd.isna(row.top_minus_best_tornado_tornado_signal) else f'{float(row.top_minus_best_tornado_tornado_signal):.4f}'}",
                f"- top_minus_best_tornado_broad_signal: {'' if pd.isna(row.top_minus_best_tornado_broad_signal) else f'{float(row.top_minus_best_tornado_broad_signal):.4f}'}",
                f"- top_minus_best_tornado_learned_term: {'' if pd.isna(row.top_minus_best_tornado_learned_term) else f'{float(row.top_minus_best_tornado_learned_term):.4f}'}",
                f"- top_minus_best_tornado_core_top: {'' if pd.isna(row.top_minus_best_tornado_core_top) else f'{float(row.top_minus_best_tornado_core_top):.4f}'}",
                f"- top_minus_best_tornado_scp_top: {'' if pd.isna(row.top_minus_best_tornado_scp_top) else f'{float(row.top_minus_best_tornado_scp_top):.4f}'}",
                f"- top_minus_best_tornado_joint_area: {'' if pd.isna(row.top_minus_best_tornado_joint_area) else f'{float(row.top_minus_best_tornado_joint_area):.4f}'}",
                f"- top_minus_best_tornado_tornado_overlap_max: {'' if pd.isna(row.top_minus_best_tornado_tornado_overlap_max) else f'{float(row.top_minus_best_tornado_tornado_overlap_max):.4f}'}",
                f"- top_minus_best_tornado_sig_tor_support_max: {'' if pd.isna(row.top_minus_best_tornado_sig_tor_support_max) else f'{float(row.top_minus_best_tornado_sig_tor_support_max):.4f}'}",
                f"- top_minus_best_tornado_context_top: {'' if pd.isna(row.top_minus_best_tornado_context_top) else f'{float(row.top_minus_best_tornado_context_top):.4f}'}",
                f"- top_minus_best_tornado_normalized_synoptic_support: {'' if pd.isna(row.top_minus_best_tornado_normalized_synoptic_support) else f'{float(row.top_minus_best_tornado_normalized_synoptic_support):.4f}'}",
                "",
                "#### Top vs Tornado Day",
                "",
                _markdown_table(
                    compare,
                    [
                        "comparison_role",
                        "valid_date",
                        "observed_category",
                        "tornado_concern_score",
                        "ranking_tornado_concern_score",
                        "support_top",
                        "normalized_synoptic_support",
                        "context_top",
                        "joint_area",
                        "core_top",
                        "scp_top",
                        "penalty_top",
                        "sig_tor_support_max",
                        "outbreak_risk_max",
                        "tornado_overlap_max",
                        "tornado_signal",
                        "broad_signal",
                        "discriminator",
                        "raw_tornado_prob_max",
                        "learned_tornado_concern_prob",
                        "core_minus_context",
                        "scp_x_core",
                        "overlap_x_core",
                        "support_x_overlap",
                        "raw_base_score_before_variant",
                        "variant_base_score",
                        "variant_tornado_term",
                        "variant_broad_term",
                        "variant_learned_term",
                        "variant_penalty_term",
                        "variant_final_score_before_preference",
                    ],
                ),
                "",
            ]
        )
    return "\n".join(sections)


def _window_source_field_audit_markdown(ranked_frame: pd.DataFrame, window_summary: pd.DataFrame) -> str:
    sections: list[str] = []
    flagged_windows = window_summary.loc[
        window_summary["is_real_ingest"].fillna(False)
        & window_summary[
            ["hail_outranks_tornado_failure", "non_outbreak_outranks_outbreak_failure", "top_day_category_mismatch"]
        ].fillna(False).any(axis=1)
    ].copy()
    for row in flagged_windows.itertuples(index=False):
        window_rows = ranked_frame.loc[ranked_frame["init_date"] == row.init_date].copy()
        compare_frames = [window_rows.sort_values("ranking_tornado_concern_score", ascending=False).iloc[[0]].assign(comparison_role="top_ranked_day")]
        tornado_rows = window_rows.loc[window_rows["observed_tornado_outbreak"].fillna(0).astype(int).gt(0)]
        sig_rows = window_rows.loc[window_rows["observed_significant_tornado_support"].fillna(0).astype(int).gt(0)]
        if not tornado_rows.empty:
            compare_frames.append(tornado_rows.sort_values("ranking_tornado_concern_score", ascending=False).iloc[[0]].assign(comparison_role="best_tornado_day"))
        if not sig_rows.empty:
            compare_frames.append(sig_rows.sort_values("ranking_tornado_concern_score", ascending=False).iloc[[0]].assign(comparison_role="best_sig_tor_day"))
        compare = pd.concat(compare_frames, ignore_index=True).drop_duplicates(subset=["valid_date", "comparison_role"])
        sections.extend(
            [
                f"### {row.init_date}",
                "",
                f"- top_ranked_day: {row.top_valid_date}",
                f"- best_tornado_day: {row.best_tornado_valid_date}",
                f"- best_sig_tor_day: {row.best_sig_tor_valid_date}",
                f"- source_root_cause: {row.source_root_cause}",
                f"- top_minus_best_tornado_core_top: {float(row.top_minus_best_tornado_core_top):.4f}" if pd.notna(row.top_minus_best_tornado_core_top) else "- top_minus_best_tornado_core_top: ",
                f"- top_minus_best_tornado_joint_area: {float(row.top_minus_best_tornado_joint_area):.4f}" if pd.notna(row.top_minus_best_tornado_joint_area) else "- top_minus_best_tornado_joint_area: ",
                f"- top_minus_best_tornado_tornado_overlap_max: {float(row.top_minus_best_tornado_tornado_overlap_max):.4f}" if pd.notna(row.top_minus_best_tornado_tornado_overlap_max) else "- top_minus_best_tornado_tornado_overlap_max: ",
                f"- top_minus_best_tornado_sig_tor_support_max: {float(row.top_minus_best_tornado_sig_tor_support_max):.4f}" if pd.notna(row.top_minus_best_tornado_sig_tor_support_max) else "- top_minus_best_tornado_sig_tor_support_max: ",
                f"- top_minus_best_tornado_overlap_compactness_proxy: {float(row.top_minus_best_tornado_overlap_compactness_proxy):.4f}" if pd.notna(row.top_minus_best_tornado_overlap_compactness_proxy) else "- top_minus_best_tornado_overlap_compactness_proxy: ",
                f"- top_minus_best_tornado_overlap_purity_proxy: {float(row.top_minus_best_tornado_overlap_purity_proxy):.4f}" if pd.notna(row.top_minus_best_tornado_overlap_purity_proxy) else "- top_minus_best_tornado_overlap_purity_proxy: ",
                f"- top_minus_best_tornado_core_tornado_alignment_proxy: {float(row.top_minus_best_tornado_core_tornado_alignment_proxy):.4f}" if pd.notna(row.top_minus_best_tornado_core_tornado_alignment_proxy) else "- top_minus_best_tornado_core_tornado_alignment_proxy: ",
                f"- top_minus_best_tornado_broad_contamination_proxy: {float(row.top_minus_best_tornado_broad_contamination_proxy):.4f}" if pd.notna(row.top_minus_best_tornado_broad_contamination_proxy) else "- top_minus_best_tornado_broad_contamination_proxy: ",
                "",
                _markdown_table(
                    compare,
                    [
                        "comparison_role",
                        "valid_date",
                        "observed_category",
                        "core_top",
                        "joint_area",
                        "tornado_overlap_max",
                        "sig_tor_support_max",
                        "outbreak_risk_max",
                        "scp_top",
                        "context_top",
                        "normalized_synoptic_support",
                        "support_top",
                        "overlap_compactness_proxy",
                        "overlap_purity_proxy",
                        "core_tornado_alignment_proxy",
                        "support_tornado_alignment_proxy",
                        "broad_contamination_proxy",
                        "scp_support_ratio",
                        "overlap_to_context_ratio",
                        "core_to_context_ratio",
                        "ranking_tornado_concern_score",
                    ],
                ),
                "",
            ]
        )
    return "\n".join(sections)


def _window_core_construction_audit_markdown(ranked_frame: pd.DataFrame, window_summary: pd.DataFrame) -> str:
    sections: list[str] = []
    flagged_windows = window_summary.loc[
        window_summary["is_real_ingest"].fillna(False)
        & window_summary[
            ["hail_outranks_tornado_failure", "non_outbreak_outranks_outbreak_failure", "top_day_category_mismatch"]
        ].fillna(False).any(axis=1)
    ].copy()
    for row in flagged_windows.itertuples(index=False):
        window_rows = ranked_frame.loc[ranked_frame["init_date"] == row.init_date].copy()
        compare_frames = [window_rows.sort_values("ranking_tornado_concern_score", ascending=False).iloc[[0]].assign(comparison_role="top_ranked_day")]
        tornado_rows = window_rows.loc[window_rows["observed_tornado_outbreak"].fillna(0).astype(int).gt(0)]
        sig_rows = window_rows.loc[window_rows["observed_significant_tornado_support"].fillna(0).astype(int).gt(0)]
        if not tornado_rows.empty:
            compare_frames.append(tornado_rows.sort_values("ranking_tornado_concern_score", ascending=False).iloc[[0]].assign(comparison_role="best_tornado_day"))
        if not sig_rows.empty:
            compare_frames.append(sig_rows.sort_values("ranking_tornado_concern_score", ascending=False).iloc[[0]].assign(comparison_role="best_sig_tor_day"))
        compare = pd.concat(compare_frames, ignore_index=True).drop_duplicates(subset=["valid_date", "comparison_role"])
        sections.extend(
            [
                f"### {row.init_date}",
                "",
                f"- top_ranked_day: {row.top_valid_date}",
                f"- best_tornado_day: {row.best_tornado_valid_date}",
                f"- best_sig_tor_day: {row.best_sig_tor_valid_date}",
                f"- core_root_cause: {row.core_root_cause}",
                f"- top_minus_best_tornado_core_top: {float(row.top_minus_best_tornado_core_top):.4f}" if pd.notna(row.top_minus_best_tornado_core_top) else "- top_minus_best_tornado_core_top: ",
                f"- top_minus_best_tornado_effective_core_top: {float(row.top_minus_best_tornado_effective_core_top):.4f}" if pd.notna(row.top_minus_best_tornado_effective_core_top) else "- top_minus_best_tornado_effective_core_top: ",
                f"- top_minus_best_tornado_core_to_context_ratio: {float(row.top_minus_best_tornado_core_to_context_ratio):.4f}" if pd.notna(row.top_minus_best_tornado_core_to_context_ratio) else "- top_minus_best_tornado_core_to_context_ratio: ",
                f"- top_minus_best_tornado_effective_core_to_context_ratio: {float(row.top_minus_best_tornado_effective_core_to_context_ratio):.4f}" if pd.notna(row.top_minus_best_tornado_effective_core_to_context_ratio) else "- top_minus_best_tornado_effective_core_to_context_ratio: ",
                f"- top_minus_best_tornado_core_compactness_proxy: {float(row.top_minus_best_tornado_core_compactness_proxy):.4f}" if pd.notna(row.top_minus_best_tornado_core_compactness_proxy) else "- top_minus_best_tornado_core_compactness_proxy: ",
                f"- top_minus_best_tornado_core_alignment_proxy: {float(row.top_minus_best_tornado_core_alignment_proxy):.4f}" if pd.notna(row.top_minus_best_tornado_core_alignment_proxy) else "- top_minus_best_tornado_core_alignment_proxy: ",
                f"- top_minus_best_tornado_core_diffuseness_proxy: {float(row.top_minus_best_tornado_core_diffuseness_proxy):.4f}" if pd.notna(row.top_minus_best_tornado_core_diffuseness_proxy) else "- top_minus_best_tornado_core_diffuseness_proxy: ",
                f"- top_minus_best_tornado_core_purity_proxy: {float(row.top_minus_best_tornado_core_purity_proxy):.4f}" if pd.notna(row.top_minus_best_tornado_core_purity_proxy) else "- top_minus_best_tornado_core_purity_proxy: ",
                "",
                _markdown_table(
                    compare,
                    [
                        "comparison_role",
                        "valid_date",
                        "observed_category",
                        "core_top",
                        "effective_core_top",
                        "context_top",
                        "core_to_context_ratio",
                        "effective_core_to_context_ratio",
                        "core_compactness_proxy",
                        "core_percentile_proxy",
                        "core_alignment_proxy",
                        "core_diffuseness_proxy",
                        "core_purity_proxy",
                        "scp_top",
                        "sig_tor_support_max",
                        "tornado_overlap_max",
                        "overlap_purity_proxy",
                        "scp_support_ratio",
                        "ranking_tornado_concern_score",
                    ],
                ),
                "",
            ]
        )
    return "\n".join(sections)


def _write_markdown_summary(
    path: Path,
    ranked_frame: pd.DataFrame,
    window_summary: pd.DataFrame,
    failure_summary: pd.DataFrame,
    raw_core_variant: str,
    raw_core_variant_summary: pd.DataFrame,
    broader_validation_checkpoint: pd.DataFrame,
    baseline_vs_masked_core_steeper_validation: pd.DataFrame,
    core_variant: str,
    core_variant_summary: pd.DataFrame,
    source_variant: str,
    source_variant_summary: pd.DataFrame,
    component_variant: str,
    component_variant_summary: pd.DataFrame,
    score_variant: str,
    variant_summary: pd.DataFrame,
    preference_mode: str,
    preference_changes: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    real_rows = int(ranked_frame["is_real_ingest"].fillna(False).sum()) if "is_real_ingest" in ranked_frame.columns else 0
    flagged_windows = window_summary.loc[
        window_summary[
            ["hail_outranks_tornado_failure", "non_outbreak_outranks_outbreak_failure", "top_day_category_mismatch"]
        ].fillna(False).any(axis=1)
    ].copy()
    lines = [
        "# Tornado Concern Evaluation",
        "",
        f"- Cases evaluated: {ranked_frame['init_date'].nunique()}",
        f"- Daily rows evaluated: {len(ranked_frame)}",
        f"- Real-ingest daily rows: {real_rows}",
        f"- Raw core variant: {raw_core_variant}",
        f"- Core variant: {core_variant}",
        f"- Source variant: {source_variant}",
        f"- Component variant: {component_variant}",
        f"- Score variant: {score_variant}",
        f"- Tornado preference mode: {preference_mode}",
        "",
        "## Window Summary",
        "",
        _markdown_table(
            window_summary,
            [
                "window_max_rank",
                "init_date",
                "top_valid_date",
                "top_observed_category",
                "top_observed_tornado_outbreak",
                "top_observed_significant_tornado_support",
                "is_real_ingest",
                "hail_outranks_tornado_failure",
                "non_outbreak_outranks_outbreak_failure",
                "top_day_category_mismatch",
                "failure_driver",
                "top_tornado_concern_score",
                "top_learned_tornado_concern_prob",
            ],
        ),
        "",
        "## Failure Pattern Counts",
        "",
        _markdown_table(failure_summary, ["pattern", "count_real_windows"]),
        "",
        "## Raw Core Variant Summary",
        "",
        _markdown_table(
            raw_core_variant_summary,
            [
                "raw_core_variant",
                "core_variant",
                "source_variant",
                "component_variant",
                "score_variant",
                "top_day_for_2024_03_14_window",
                "top_day_for_2024_04_26_window",
                "hail_outranks_tornado_failure_real",
            ],
        ),
        "",
        "## Broader Validation Checkpoint",
        "",
        _markdown_table(
            broader_validation_checkpoint,
            [
                "raw_core_variant",
                "real_ingest_windows",
                "hail_outranks_tornado_failure_real",
                "non_outbreak_outranks_outbreak_failure_real",
                "top_day_category_mismatch_real",
                "tornado_day_ranked_first_real",
                "sig_tor_day_ranked_first_real",
                "mean_top_minus_best_tornado_margin_real",
                "median_top_minus_best_tornado_margin_real",
            ],
        ),
        "",
        "## Baseline vs Masked Core Steeper Validation",
        "",
        _markdown_table(
            baseline_vs_masked_core_steeper_validation,
            [
                "raw_core_variant",
                "real_ingest_windows",
                "hail_outranks_tornado_failure_real",
                "non_outbreak_outranks_outbreak_failure_real",
                "top_day_category_mismatch_real",
                "tornado_day_ranked_first_real",
                "sig_tor_day_ranked_first_real",
                "mean_top_minus_best_tornado_margin_real",
                "median_top_minus_best_tornado_margin_real",
            ],
        ),
        "",
        "## Source Variant Summary",
        "",
        _markdown_table(
            source_variant_summary,
            [
                "raw_core_variant",
                "core_variant",
                "source_variant",
                "component_variant",
                "score_variant",
                "hail_outranks_tornado_failure_real",
                "non_outbreak_outranks_outbreak_failure_real",
                "top_day_category_mismatch_real",
                "top_day_for_2024_03_14_window",
                "top_day_for_2024_04_26_window",
                "margin_top_minus_best_tornado_for_2024_03_14",
                "source_root_cause_for_2024_03_14",
            ],
        ),
        "",
        "## Component Variant Summary",
        "",
        _markdown_table(
            component_variant_summary,
            [
                "raw_core_variant",
                "core_variant",
                "component_variant",
                "score_variant",
                "hail_outranks_tornado_failure_real",
                "non_outbreak_outranks_outbreak_failure_real",
                "top_day_category_mismatch_real",
                "top_day_for_2024_03_14_window",
                "top_day_for_2024_04_26_window",
                "margin_top_minus_best_tornado_for_2024_03_14",
            ],
        ),
        "",
        "## Core Variant Summary",
        "",
        _markdown_table(
            core_variant_summary,
            [
                "core_variant",
                "source_variant",
                "component_variant",
                "score_variant",
                "hail_outranks_tornado_failure_real",
                "non_outbreak_outranks_outbreak_failure_real",
                "top_day_category_mismatch_real",
                "top_day_for_2024_03_14_window",
                "top_day_for_2024_04_26_window",
                "margin_top_minus_best_tornado_for_2024_03_14",
                "core_root_cause_for_2024_03_14",
            ],
        ),
        "",
        "## Score Variant Summary",
        "",
        _markdown_table(
            variant_summary,
            [
                "variant",
                "hail_outranks_tornado_failure_real",
                "non_outbreak_outranks_outbreak_failure_real",
                "top_day_category_mismatch_real",
                "top_day_for_2024_03_14_window",
                "top_day_for_2024_04_26_window",
            ],
        ),
        "",
        "## Ranked Daily Rows",
        "",
        _markdown_table(
            ranked_frame,
            [
                "init_date",
                "valid_date",
                "day_rank_within_init",
                "window_max_rank",
                "observed_category",
                "observed_tornado_outbreak",
                "observed_significant_tornado_support",
                "is_real_ingest",
                "raw_core_variant_name",
                "raw_core_mask_factor",
                "raw_core_final_value",
                "raw_core_to_context_ratio",
                "core_variant_name",
                "source_variant_name",
                "component_variant_name",
                "score_variant_name",
                "effective_core_top",
                "effective_core_to_context_ratio",
                "core_compactness_proxy",
                "core_percentile_proxy",
                "core_alignment_proxy",
                "core_diffuseness_proxy",
                "core_purity_proxy",
                "core_support_interaction",
                "core_overlap_interaction",
                "core_scp_interaction",
                "core_penalty_term",
                "source_core_term",
                "source_overlap_term",
                "source_support_term",
                "source_penalty_term",
                "component_core_term",
                "component_overlap_term",
                "component_scp_term",
                "component_penalty_term",
                "component_hail_dominance",
                "component_tornado_agreement",
                "tornado_concern_score",
                "ranking_tornado_concern_score",
                "raw_tornado_prob_max",
                "raw_hail_prob_max",
                "raw_wind_prob_max",
                "learned_tornado_concern_prob",
                "support_top",
                "normalized_synoptic_support",
                "context_top",
                "joint_area",
                "core_top",
                "scp_top",
                "penalty_top",
                "sig_tor_support_max",
                "outbreak_risk_max",
                "tornado_overlap_max",
                "hail_overlap_max",
                "wind_overlap_max",
                "core_minus_context",
                "scp_x_core",
                "overlap_x_core",
                "support_x_overlap",
                "overlap_compactness_proxy",
                "overlap_purity_proxy",
                "core_tornado_alignment_proxy",
                "support_tornado_alignment_proxy",
                "broad_contamination_proxy",
                "scp_support_ratio",
                "overlap_to_context_ratio",
                "core_to_context_ratio",
                "raw_base_score_before_variant",
                "variant_name",
                "variant_base_score",
                "variant_tornado_term",
                "variant_broad_term",
                "variant_learned_term",
                "variant_penalty_term",
                "variant_final_score_before_preference",
                "tornado_signal",
                "broad_signal",
                "discriminator",
                "discriminator_multiplier",
                "final_score_minus_window_top",
            ],
        ),
        "",
        "## Conservative Tornado Preference Diagnostics",
        "",
        f"Mode: {preference_mode}",
        "",
    ]
    if preference_mode != "off":
        if preference_changes.empty:
            lines.extend(["No windows changed top day.", ""])
        else:
            lines.extend(
                [
                    _markdown_table(preference_changes, ["init_date", "prior_top_valid_date", "new_top_valid_date", "preference_delta"]),
                    "",
                ]
            )
    if not flagged_windows.empty:
        lines.extend(
            [
                "## Flagged Real-Window Failures",
                "",
                _markdown_table(
                    flagged_windows,
                    [
                        "init_date",
                        "top_valid_date",
                        "top_observed_category",
                        "is_real_ingest",
                        "hail_outranks_tornado_failure",
                        "non_outbreak_outranks_outbreak_failure",
                        "top_day_category_mismatch",
                        "failure_driver",
                    ],
                ),
                "",
                _window_failure_comparison_markdown(ranked_frame, window_summary),
                "",
                "## Bad-Window Component Audit",
                "",
                _window_failure_comparison_markdown(ranked_frame, window_summary),
                "",
                "## Source Field Audit For Flagged Real Windows",
                "",
                _window_source_field_audit_markdown(ranked_frame, window_summary),
                "",
                "## Core Construction Audit For Flagged Real Windows",
                "",
                _window_core_construction_audit_markdown(ranked_frame, window_summary),
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a prototype tornado-concern diagnostic from saved forecast artifacts")
    parser.add_argument("--dates", nargs="*", help="Forecast init dates, e.g. 2024-04-26 2024-05-06")
    parser.add_argument("--dates-file", help="Optional text file with one init date per line")
    parser.add_argument("--skip-missing", action="store_true", help="Warn and continue when an init date is missing forecast or verification artifacts")
    parser.add_argument("--output-csv", help="Optional path for a ranked CSV report")
    parser.add_argument("--output-md", help="Optional path for a concise Markdown summary")
    parser.add_argument("--component-checkpoint-output-csv", help="Optional side-by-side baseline vs tornado_hail_separation_v1 component checkpoint CSV")
    parser.add_argument("--component-checkpoint-output-md", help="Optional side-by-side baseline vs tornado_hail_separation_v1 component checkpoint Markdown")
    parser.add_argument("--component-checkpoint-only", action="store_true", help="Write only the component checkpoint comparison and skip the full ranked eval report")
    parser.add_argument("--include-variant-summaries", action="store_true", help="Include slower all-variant comparison sections in the Markdown report")
    parser.add_argument("--raw-core-variant", choices=RAW_CORE_VARIANTS, default="baseline")
    parser.add_argument("--core-variant", choices=CORE_VARIANTS, default="baseline")
    parser.add_argument("--source-variant", choices=SOURCE_VARIANTS, default="baseline")
    parser.add_argument("--component-variant", choices=COMPONENT_VARIANTS, default="baseline")
    parser.add_argument("--score-variant", choices=SCORE_VARIANTS, default="baseline")
    parser.add_argument("--tornado-preference-mode", choices=["off", "conservative", "compact_tornado", "earliest_close"], default="off")
    args = parser.parse_args()

    settings = load_settings()
    paths = build_paths(settings)

    artifact_pairs: list[tuple[str, Path, Path]] = []
    tables: list[pd.DataFrame] = []
    requested_dates = _load_dates(args)
    skipped_missing_count = 0
    for date in requested_dates:
        prediction_path = _latest_prediction_for_date(paths.outputs, date)
        verification_path = _verification_for_date(paths.verification, date)
        if prediction_path is None or verification_path is None:
            if args.skip_missing:
                skipped_missing_count += 1
                print(f"warning: skipping {date} because forecast or verification artifacts are missing")
                continue
            raise FileNotFoundError(f"missing forecast or verification artifacts for {date}")
        artifact_pairs.append((date, prediction_path, verification_path))
        if not args.component_checkpoint_only:
            frame = evaluate_tornado_concern_for_artifacts(
                prediction_path,
                verification_path,
                init_date=date,
                raw_core_variant=args.raw_core_variant,
                score_variant=args.score_variant,
                component_variant=args.component_variant,
                source_variant=args.source_variant,
                core_variant=args.core_variant,
            )
            tables.append(frame)

    if args.component_checkpoint_only:
        if not artifact_pairs:
            raise FileNotFoundError("no matching forecast and verification artifacts found for the requested init dates")
        checkpoint_summary = summarize_component_checkpoint_from_artifacts(
            artifact_pairs,
            raw_core_variant=args.raw_core_variant,
            core_variant=args.core_variant,
            source_variant=args.source_variant,
            score_variant=args.score_variant,
            requested_case_count=len(requested_dates),
            skipped_missing_count=skipped_missing_count,
        )
        if args.component_checkpoint_output_csv:
            checkpoint_csv = Path(args.component_checkpoint_output_csv)
            checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_summary.to_csv(checkpoint_csv, index=False)
        if args.component_checkpoint_output_md:
            write_component_checkpoint_markdown(Path(args.component_checkpoint_output_md), checkpoint_summary)
        print(f"component_checkpoint_rows={len(checkpoint_summary)}")
        return

    if not tables:
        raise FileNotFoundError("no matching forecast and verification artifacts found for the requested init dates")

    output = pd.concat(tables, ignore_index=True)
    adjusted_output, preference_changes = apply_tornado_preference_mode(output, mode=args.tornado_preference_mode)
    window_summary = summarize_case_windows(adjusted_output)
    failure_summary = summarize_failure_patterns(window_summary)
    include_variant_summaries = bool(args.include_variant_summaries) or len(artifact_pairs) <= 10
    if include_variant_summaries:
        source_variant_summary = summarize_source_variants_from_artifacts(
            artifact_pairs,
            raw_core_variant=args.raw_core_variant,
            core_variant=args.core_variant,
            component_variant=args.component_variant,
            score_variant=args.score_variant,
        )
        component_variant_summary = summarize_component_variants_from_artifacts(
            artifact_pairs,
            raw_core_variant=args.raw_core_variant,
            core_variant=args.core_variant,
            score_variant=args.score_variant,
            source_variant=args.source_variant,
        )
        raw_core_variant_summary = summarize_raw_core_variants_from_artifacts(
            artifact_pairs,
            core_variant=args.core_variant,
            source_variant=args.source_variant,
            component_variant=args.component_variant,
            score_variant=args.score_variant,
        )
        broader_validation_checkpoint = summarize_broader_validation_checkpoint_from_artifacts(
            artifact_pairs,
            core_variant=args.core_variant,
            source_variant=args.source_variant,
            component_variant=args.component_variant,
            score_variant=args.score_variant,
        )
        baseline_vs_masked_core_steeper_validation = summarize_baseline_vs_masked_core_steeper_from_artifacts(
            artifact_pairs,
            core_variant=args.core_variant,
            source_variant=args.source_variant,
            component_variant=args.component_variant,
            score_variant=args.score_variant,
        )
        core_variant_summary = summarize_core_variants_from_artifacts(
            artifact_pairs,
            raw_core_variant=args.raw_core_variant,
            source_variant=args.source_variant,
            component_variant=args.component_variant,
            score_variant=args.score_variant,
        )
    else:
        source_variant_summary = pd.DataFrame()
        component_variant_summary = pd.DataFrame()
        raw_core_variant_summary = pd.DataFrame()
        broader_validation_checkpoint = pd.DataFrame()
        baseline_vs_masked_core_steeper_validation = pd.DataFrame()
        core_variant_summary = pd.DataFrame()
    variant_summary = summarize_score_variants(output)
    ranked_output = build_ranked_case_table(adjusted_output)
    print(_format_table(output))
    print()
    print(_format_window_summary(window_summary))
    print()
    print(_format_failure_summary(failure_summary))

    if args.output_csv:
        _write_ranked_csv(Path(args.output_csv), ranked_output)
    if args.output_md:
        _write_markdown_summary(
            Path(args.output_md),
            ranked_output,
            window_summary,
            failure_summary,
            args.raw_core_variant,
            raw_core_variant_summary,
            broader_validation_checkpoint,
            baseline_vs_masked_core_steeper_validation,
            args.core_variant,
            core_variant_summary,
            args.source_variant,
            source_variant_summary,
            args.component_variant,
            component_variant_summary,
            args.score_variant,
            variant_summary,
            args.tornado_preference_mode,
            preference_changes,
        )
    if args.component_checkpoint_output_csv or args.component_checkpoint_output_md:
        checkpoint_summary = summarize_component_checkpoint_from_artifacts(
            artifact_pairs,
            raw_core_variant=args.raw_core_variant,
            core_variant=args.core_variant,
            source_variant=args.source_variant,
            score_variant=args.score_variant,
            requested_case_count=len(requested_dates),
            skipped_missing_count=skipped_missing_count,
        )
        if args.component_checkpoint_output_csv:
            checkpoint_csv = Path(args.component_checkpoint_output_csv)
            checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_summary.to_csv(checkpoint_csv, index=False)
        if args.component_checkpoint_output_md:
            write_component_checkpoint_markdown(Path(args.component_checkpoint_output_md), checkpoint_summary)


if __name__ == "__main__":
    main()
