"""Training-table and calibration artifacts for backtest runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from severewx.backtest.manifest import BacktestCase, BacktestManifest, load_backtest_manifest
from severewx.verify.metrics import (
    SPC_TORNADO_THRESHOLDS,
    brier_score,
    frequency_bias,
    reliability_bins,
    safe_roc_auc,
    threshold_summaries,
)


CALIBRATION_BINS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, 1.0)
SEGMENT_KEYS = ("fold", "lead_day", "region", "season", "regime", "source_set", "source_availability_tier")
CORE_SOURCES = {"hrrr_recent", "rap_recent", "nam_recent", "aws_recent"}
ECMWF_SOURCE = "ecmwf_recent"
OPENMETEO_SOURCE = "open_meteo_recent"


@dataclass(frozen=True, slots=True)
class TrainingTableResult:
    table_csv: Path
    table_json: Path
    summary_json: Path
    summary_csv: Path
    calibration_json: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _case_index(manifest: BacktestManifest) -> dict[str, BacktestCase]:
    return {case.case_id: case for case in manifest.cases}


def _source_flags(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source = str(row.get("source", ""))
        if not source or source == "forecast_consensus":
            continue
        grouped.setdefault(str(row.get("case_id", "")), []).append(row)
    flags: dict[str, dict[str, Any]] = {}
    for case_id, case_rows in grouped.items():
        available = sorted(str(row["source"]) for row in case_rows if row.get("status") == "available")
        unavailable = sorted(str(row["source"]) for row in case_rows if row.get("status") != "available")
        flags[case_id] = {
            "source_set": ",".join(available),
            "source_availability_tier": source_availability_tier(available),
            "source_count": len(available),
            "unavailable_source_count": len(unavailable),
            "unavailable_sources": ",".join(unavailable),
            "synthetic_source_count": sum(1 for row in case_rows if str(row.get("source_mode", "")).lower() == "synthetic"),
        }
    return flags


def source_availability_tier(available_sources: list[str]) -> str:
    available = set(available_sources)
    if CORE_SOURCES | {ECMWF_SOURCE, OPENMETEO_SOURCE} <= available:
        return "recent_full_stack_plus_openmeteo"
    if CORE_SOURCES | {ECMWF_SOURCE} <= available:
        return "recent_full_stack"
    if CORE_SOURCES <= available:
        return "core_no_ecmwf"
    if available == {"hrrr_recent"}:
        return "hrrr_only"
    if available:
        return "partial_real_sources"
    return "archive_blocked"


def _per_day_by_valid_date(verification_path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(verification_path)
    per_day = payload.get("per_day", [])
    if not isinstance(per_day, list):
        return {}
    return {str(row.get("valid_date", "")): row for row in per_day if isinstance(row, dict)}


def _product_metadata(path_value: object) -> dict[str, Any]:
    path = Path(str(path_value)) if path_value else Path()
    return _read_json(path) if path.exists() else {}


def _training_rows_for_run(manifest: BacktestManifest, run_dir: Path, *, include_unverified: bool) -> list[dict[str, Any]]:
    payload_path = run_dir / "source_availability.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"missing source availability payload: {payload_path}")
    payload = _read_json(payload_path)
    cases = _case_index(manifest)
    flags_by_case = _source_flags(payload.get("rows", []))
    scores_by_case = {str(row.get("case_id")): row for row in payload.get("scores", []) if isinstance(row, dict)}
    run_sources = payload.get("run_options", {}).get("sources", [])
    configured_source_set = ",".join(str(source) for source in run_sources) if isinstance(run_sources, list) else str(run_sources)

    rows: list[dict[str, Any]] = []
    for product in payload.get("products", []):
        if not isinstance(product, dict):
            continue
        case_id = str(product.get("case_id", ""))
        case = cases.get(case_id)
        if case is None:
            continue
        score = scores_by_case.get(case_id, {})
        verification_path = Path(str(score.get("verification_path", ""))) if score.get("verification_path") else None
        per_day = _per_day_by_valid_date(verification_path) if verification_path else {}
        valid_date = str(product.get("valid_date", ""))
        observed = per_day.get(valid_date, {})
        verified = bool(observed)
        if not include_unverified and not verified:
            continue
        metadata = _product_metadata(product.get("metadata_path"))
        source_flags = flags_by_case.get(case_id, {})
        raw_prob = observed.get("forecast_tornado_prob", product.get("max_tornado_concern_prob"))
        row = {
            "manifest_id": manifest.manifest_id,
            "run_dir": str(run_dir),
            "case_id": case.case_id,
            "date": case.date,
            "cycle": case.cycle,
            "valid_date": valid_date,
            "lead_day": int(product.get("day", observed.get("lead_day", 0)) or 0),
            "fold": case.fold,
            "region": case.region,
            "season": case.season,
            "regime": case.regime,
            "tags": ",".join(case.tags),
            "expected_signal": case.expected_signal,
            "configured_source_set": configured_source_set,
            "source_set": source_flags.get("source_set", ""),
            "source_availability_tier": source_flags.get("source_availability_tier", "archive_blocked"),
            "source_count": int(source_flags.get("source_count", 0) or 0),
            "unavailable_source_count": int(source_flags.get("unavailable_source_count", 0) or 0),
            "unavailable_sources": source_flags.get("unavailable_sources", ""),
            "synthetic_source_count": int(source_flags.get("synthetic_source_count", 0) or 0),
            "verified": verified,
            "verification_status": score.get("verification_status", "unavailable"),
            "raw_tornado_probability": float(raw_prob) if raw_prob is not None else np.nan,
            "product_max_tornado_probability": float(product.get("max_tornado_concern_prob") or 0.0),
            "forecast_confidence": float(observed.get("forecast_confidence", np.nan)) if observed else np.nan,
            "forecast_signal_quality": float(observed.get("forecast_signal_quality", np.nan)) if observed else np.nan,
            "observed_tornado_outbreak": int(observed.get("observed_tornado_outbreak", 0) or 0),
            "observed_any_outbreak": int(observed.get("observed_any_outbreak", 0) or 0),
            "observed_tornado_report_count": int(observed.get("observed_tornado_report_count", 0) or 0),
            "observed_category": str(observed.get("observed_category", "")),
            "observed_report_source_is_real": bool(observed.get("observed_report_source_is_real", False)),
            "tornado_brier": float(observed.get("tornado_brier", np.nan)) if observed else np.nan,
            "publication_status": product.get("publication_status", ""),
            "public_ready": bool(product.get("public_ready", False)),
            "readiness_status": metadata.get("readiness_status", metadata.get("publication_status", "")),
            "calibration_version": metadata.get("calibration_version", ""),
            "backtest_version": metadata.get("backtest_version", ""),
        }
        rows.append(row)
    return rows


def build_training_frame(manifest_path: Path, run_dirs: list[Path], *, include_unverified: bool = False) -> pd.DataFrame:
    manifest = load_backtest_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows.extend(_training_rows_for_run(manifest, run_dir, include_unverified=include_unverified))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["date", "cycle", "case_id", "lead_day", "source_set"]).reset_index(drop=True)


def _metric_block(frame: pd.DataFrame, *, probability_column: str = "raw_tornado_probability") -> dict[str, Any]:
    if frame.empty:
        return {"row_count": 0, "status": "empty"}
    y_true = frame["observed_tornado_outbreak"].astype(float).to_numpy()
    y_prob = frame[probability_column].astype(float).to_numpy()
    block = {
        "row_count": int(len(frame)),
        "event_count": int(np.nansum(y_true)),
        "brier": brier_score(y_true, y_prob),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "reliability": reliability_bins(y_true, y_prob, n_bins=10),
        "thresholds": threshold_summaries(y_true, y_prob),
    }
    block["frequency_bias_15pct"] = frequency_bias(y_true, y_prob, threshold=0.15)
    return block


def skill_summary(frame: pd.DataFrame) -> dict[str, Any]:
    has_calibrated = "calibrated_tornado_probability" in frame
    summary: dict[str, Any] = {
        "row_count": int(len(frame)),
        "verified_row_count": int(frame["verified"].sum()) if "verified" in frame else 0,
        "overall": _metric_block(frame),
        "overall_calibrated": _metric_block(frame, probability_column="calibrated_tornado_probability") if has_calibrated else {},
        "segments": {},
        "segments_calibrated": {},
    }
    if frame.empty:
        return summary
    for key in SEGMENT_KEYS:
        groups: dict[str, Any] = {}
        calibrated_groups: dict[str, Any] = {}
        for value, subset in frame.groupby(key, dropna=False):
            groups[str(value)] = _metric_block(subset)
            if has_calibrated:
                calibrated_groups[str(value)] = _metric_block(subset, probability_column="calibrated_tornado_probability")
        summary["segments"][key] = groups
        if has_calibrated:
            summary["segments_calibrated"][key] = calibrated_groups
    return summary


def _probability_bin(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    for left, right in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:]):
        if left <= value < right or (right == 1.0 and left <= value <= right):
            return f"{left:.2f}-{right:.2f}"
    return "missing"


def fit_apply_calibration(frame: pd.DataFrame, *, min_group_rows: int = 5, prior: float = 1.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return frame.copy(), {"status": "empty", "rules": []}
    calibrated = frame.copy()
    calibrated["probability_bin"] = calibrated["raw_tornado_probability"].map(_probability_bin)
    tune = calibrated[(calibrated["fold"] == "tune") & calibrated["verified"]].copy()
    if tune.empty:
        calibrated["calibrated_tornado_probability"] = calibrated["raw_tornado_probability"]
        return calibrated, {"status": "insufficient_tune_rows", "rules": []}

    rules: list[dict[str, Any]] = []
    for keys, group_frame in [
        (("lead_day", "region", "probability_bin"), tune),
        (("lead_day", "probability_bin"), tune),
        (("probability_bin",), tune),
    ]:
        for values, subset in group_frame.groupby(list(keys), dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            if len(subset) < min_group_rows:
                continue
            observed = float(subset["observed_tornado_outbreak"].sum())
            count = int(len(subset))
            calibrated_prob = (observed + prior) / (count + 2.0 * prior)
            rules.append(
                {
                    "keys": list(keys),
                    "values": [str(value) for value in values],
                    "row_count": count,
                    "observed_events": observed,
                    "calibrated_probability": calibrated_prob,
                }
            )
    calibrated["calibrated_tornado_probability"] = calibrated.apply(lambda row: _apply_calibration_rule(row, rules), axis=1)
    status = "candidate_fit" if rules else "insufficient_tune_bin_rows"
    return calibrated, {"status": status, "fit_fold": "tune", "min_group_rows": min_group_rows, "rules": rules}


def _apply_calibration_rule(row: pd.Series, rules: list[dict[str, Any]]) -> float:
    for rule in rules:
        values = [str(row.get(key, "")) for key in rule["keys"]]
        if values == rule["values"]:
            return float(rule["calibrated_probability"])
    return float(row.get("raw_tornado_probability", np.nan))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_training_artifacts(
    *,
    manifest_path: Path,
    run_dirs: list[Path],
    output_dir: Path,
    include_unverified: bool = False,
) -> TrainingTableResult:
    frame = build_training_frame(manifest_path, run_dirs, include_unverified=include_unverified)
    calibrated, calibration = fit_apply_calibration(frame)
    summary = skill_summary(calibrated)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_csv = output_dir / "training_table.csv"
    table_json = output_dir / "training_table.json"
    summary_json = output_dir / "skill_summary.json"
    summary_csv = output_dir / "skill_summary.csv"
    calibration_json = output_dir / "calibration_candidate.json"

    calibrated.to_csv(table_csv, index=False)
    table_json.write_text(json.dumps(_json_safe(calibrated.to_dict(orient="records")), indent=2, allow_nan=False), encoding="utf-8")
    summary_json.write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    calibration_json.write_text(json.dumps(_json_safe(calibration), indent=2, allow_nan=False), encoding="utf-8")
    _write_skill_summary_csv(summary_csv, summary)
    return TrainingTableResult(table_csv, table_json, summary_json, summary_csv, calibration_json)


def _write_skill_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    overall = summary.get("overall", {})
    rows.append(
        {
            "metric_set": "raw",
            "segment": "overall",
            "value": "all",
            "row_count": overall.get("row_count", 0),
            "event_count": overall.get("event_count", 0),
            "brier": overall.get("brier", ""),
            "roc_auc": overall.get("roc_auc", ""),
            "frequency_bias_15pct": overall.get("frequency_bias_15pct", ""),
        }
    )
    calibrated_overall = summary.get("overall_calibrated", {})
    if calibrated_overall:
        rows.append(
            {
                "metric_set": "calibrated",
                "segment": "overall",
                "value": "all",
                "row_count": calibrated_overall.get("row_count", 0),
                "event_count": calibrated_overall.get("event_count", 0),
                "brier": calibrated_overall.get("brier", ""),
                "roc_auc": calibrated_overall.get("roc_auc", ""),
                "frequency_bias_15pct": calibrated_overall.get("frequency_bias_15pct", ""),
            }
        )
    for segment, groups in summary.get("segments", {}).items():
        for value, payload in groups.items():
            rows.append(
                {
                    "metric_set": "raw",
                    "segment": segment,
                    "value": value,
                    "row_count": payload.get("row_count", 0),
                    "event_count": payload.get("event_count", 0),
                    "brier": payload.get("brier", ""),
                    "roc_auc": payload.get("roc_auc", ""),
                    "frequency_bias_15pct": payload.get("frequency_bias_15pct", ""),
                }
            )
    for segment, groups in summary.get("segments_calibrated", {}).items():
        for value, payload in groups.items():
            rows.append(
                {
                    "metric_set": "calibrated",
                    "segment": segment,
                    "value": value,
                    "row_count": payload.get("row_count", 0),
                    "event_count": payload.get("event_count", 0),
                    "brier": payload.get("brier", ""),
                    "roc_auc": payload.get("roc_auc", ""),
                    "frequency_bias_15pct": payload.get("frequency_bias_15pct", ""),
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric_set", "segment", "value", "row_count", "event_count", "brier", "roc_auc", "frequency_bias_15pct"])
        writer.writeheader()
        writer.writerows(rows)
