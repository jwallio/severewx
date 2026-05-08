"""Prepare deterministic consensus backtest cases and source availability ledgers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
import signal
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from severewx.backtest.manifest import BacktestCase, BacktestManifest, load_backtest_manifest, manifest_to_jsonable
from severewx.cli.build_tornado_concern_product import build_product_bundle
from severewx.cli.run_forecast_consensus import _parse_sources, _read_json, _run_source_product
from severewx.config import load_settings
from severewx.models.forecast_consensus import (
    CONSENSUS_FIELD,
    CONSENSUS_INPUT_FIELD,
    ConsensusSource,
    build_forecast_consensus,
    consensus_metadata_path,
    consensus_product_path,
    source_metadata_path,
    source_product_path,
)
from severewx.verify.daily import verify_daily_probabilities
from severewx.verify.metrics import SPC_TORNADO_THRESHOLDS, reliability_bins, threshold_summaries
from severewx.utils.paths import build_paths


DEFAULT_MANIFEST = Path("backtests/tornado_environment_consensus_v1.json")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_lead_hours(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    leads = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not leads:
        return None
    if any(lead < 0 for lead in leads):
        raise ValueError("lead hours must be non-negative")
    return sorted(dict.fromkeys(leads))


def _case_output_dir(base_output_dir: Path, manifest: BacktestManifest, case: BacktestCase) -> Path:
    return base_output_dir / manifest.manifest_id / f"{case.date}_{case.cycle}z"


def _source_set_key(sources: list[str]) -> str:
    return "__".join(sorted(str(source).replace("/", "_").replace("\\", "_") for source in sources)) or "no_sources"


def _case_artifact_paths(paths, base_output_dir: Path, manifest: BacktestManifest, case: BacktestCase, sources: list[str]):
    case_dir = _case_output_dir(base_output_dir, manifest, case)
    artifact_dir = case_dir / "consensus" / _source_set_key(sources)
    verification_dir = case_dir / "verification" / _source_set_key(sources)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)
    return replace(paths, outputs=artifact_dir, verification=verification_dir)


def _source_row(
    *,
    case: BacktestCase,
    source: str,
    status: str,
    product_path: Path,
    metadata_path: Path,
    reason: str = "",
) -> dict[str, Any]:
    metadata = _read_json(metadata_path)
    ingest = metadata.get("ingest_summary", {}) if isinstance(metadata.get("ingest_summary", {}), dict) else {}
    return {
        "case_id": case.case_id,
        "date": case.date,
        "cycle": case.cycle,
        "source": source,
        "status": status,
        "reason": reason,
        "source_mode": ingest.get("source_mode", ""),
        "source_model": ingest.get("source_model", ""),
        "available_fields": ",".join(ingest.get("available_fields", []) or []),
        "missing_requested_leads": ",".join(str(lead) for lead in ingest.get("missing_requested_leads", []) or []),
        "product_path": str(product_path) if product_path.exists() else "",
        "metadata_path": str(metadata_path) if metadata_path.exists() else "",
    }


def _write_outputs(
    output_dir: Path,
    manifest: BacktestManifest,
    rows: list[dict[str, Any]],
    *,
    dry_run: bool,
    selected_cases: list[BacktestCase],
    run_options: dict[str, Any],
    products: list[dict[str, Any]] | None = None,
    scores: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_copy = output_dir / "manifest.json"
    manifest_copy.write_text(json.dumps(_json_safe(manifest_to_jsonable(manifest)), indent=2, allow_nan=False), encoding="utf-8")
    json_path = output_dir / "source_availability.json"
    csv_path = output_dir / "source_availability.csv"
    payload = {
        "manifest_id": manifest.manifest_id,
        "target_product": manifest.target_product,
        "dry_run": dry_run,
        "case_count": len(manifest.cases),
        "selected_case_count": len(selected_cases),
        "selected_cases": [case.case_id for case in selected_cases],
        "run_options": run_options,
        "rows": rows,
        "case_summary": _case_summary(rows),
        "products": products or [],
        "scores": scores or [],
    }
    summary = _manifest_summary(manifest, rows, products or [], scores or [], dry_run=dry_run, selected_cases=selected_cases, run_options=run_options)
    summary_json_path = output_dir / "backtest_summary.json"
    summary_csv_path = output_dir / "backtest_summary.csv"
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    summary_json_path.write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    fieldnames = [
        "case_id",
        "date",
        "cycle",
        "source",
        "status",
        "reason",
        "source_mode",
        "source_model",
        "available_fields",
        "missing_requested_leads",
        "product_path",
        "metadata_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_summary_csv(summary_csv_path, summary)
    return manifest_copy, json_path, csv_path, summary_json_path, summary_csv_path


def _checkpoint_outputs(
    output_dir: Path,
    manifest: BacktestManifest,
    selected_cases: list[BacktestCase],
    rows: list[dict[str, Any]],
    products: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    dry_run: bool,
    run_options: dict[str, Any],
) -> None:
    _write_outputs(
        output_dir,
        manifest,
        rows,
        dry_run=dry_run,
        selected_cases=selected_cases,
        run_options=run_options,
        products=products,
        scores=scores,
    )


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = ["section", "name", "value"]
    rows: list[dict[str, Any]] = []
    for key in ["manifest_id", "case_count", "dry_run", "readiness_status"]:
        rows.append({"section": "summary", "name": key, "value": summary.get(key, "")})
    for fold, count in summary.get("fold_counts", {}).items():
        rows.append({"section": "fold_counts", "name": fold, "value": count})
    for tag, count in summary.get("tag_counts", {}).items():
        rows.append({"section": "tag_counts", "name": tag, "value": count})
    for source, payload in summary.get("source_ablation", {}).items():
        rows.append({"section": "source_ablation", "name": source, "value": json.dumps(payload, sort_keys=True)})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    matches = list(directory.glob(pattern))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _select_cases(
    manifest: BacktestManifest,
    *,
    limit: int | None,
    batch_size: int | None,
    batch_number: int | None,
) -> list[BacktestCase]:
    cases = list(manifest.cases)
    if batch_size is not None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        effective_batch_number = batch_number or 1
        if effective_batch_number <= 0:
            raise ValueError("batch_number must be positive")
        start = (effective_batch_number - 1) * batch_size
        cases = cases[start : start + batch_size]
    if limit is not None:
        cases = cases[:limit]
    return cases


def _existing_source_artifact(paths, case: BacktestCase, source: str) -> ConsensusSource | None:
    product_path = source_product_path(paths.outputs, case.date, case.cycle, source)
    metadata_path = source_metadata_path(paths.outputs, case.date, case.cycle, source)
    if product_path.exists() and metadata_path.exists():
        metadata = _read_json(metadata_path)
        ingest = metadata.get("ingest_summary", {}) if isinstance(metadata.get("ingest_summary", {}), dict) else {}
        if str(ingest.get("source_mode", "real")).lower() != "synthetic":
            return ConsensusSource(source, product_path, metadata_path)
    return None


def _existing_consensus(paths, case: BacktestCase, expected_sources: list[str] | None = None) -> tuple[Path | None, Path | None]:
    product_path = consensus_product_path(paths.outputs, case.date, case.cycle)
    metadata_path = consensus_metadata_path(paths.outputs, case.date, case.cycle)
    if product_path.exists() and metadata_path.exists():
        if expected_sources is not None:
            metadata = _read_json(metadata_path)
            included_sources = sorted(str(source) for source in metadata.get("included_sources", []) or [])
            if included_sources != sorted(str(source) for source in expected_sources):
                return None, None
        return product_path, metadata_path
    return None, None


def _existing_case_products(paths, manifest: BacktestManifest, case: BacktestCase, output_dir: Path, sources: list[str]) -> list[dict[str, Any]]:
    outdir = _case_output_dir(output_dir, manifest, case) / "products" / _source_set_key(sources)
    rows: list[dict[str, Any]] = []
    for day in manifest.target_product.get("lead_days", [1, 2, 3]):
        valid_date = (pd.Timestamp(case.date) + pd.Timedelta(days=int(day) - 1)).date().isoformat()
        matches = sorted(outdir.glob(f"tornado_concern_init_{case.date}_{case.cycle}z_valid_{valid_date}.json"))
        if not matches:
            return []
        metadata = _read_json(matches[-1])
        if not metadata:
            return []
        rows.append(
            {
                "case_id": case.case_id,
                "date": case.date,
                "cycle": case.cycle,
                "day": int(day),
                "valid_date": valid_date,
                "metadata_path": str(matches[-1]),
                "image_path": metadata.get("main_image_path", metadata.get("image_path", "")),
                "publication_status": metadata.get("publication_status", ""),
                "public_ready": bool(metadata.get("public_ready", False)),
                "max_tornado_concern_prob": metadata.get("max_tornado_concern_prob"),
                "resume_status": "reused_existing",
                "product_status": "available",
                "reason": "",
            }
        )
    return rows


def _existing_verification(paths, case: BacktestCase) -> Path | None:
    path = paths.verification / "backtests" / f"{case.case_id}_verification.json"
    return path if path.exists() else None


class _SourceTimeout(Exception):
    pass


def _run_source_product_with_timeout(date: str, cycle: str, source: str, settings, *, skip_render: bool, timeout_seconds: int | None) -> ConsensusSource | None:
    if timeout_seconds is None or timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return _run_source_product(date, cycle, source, settings, skip_render=skip_render)

    def _raise_timeout(signum, frame):
        raise _SourceTimeout(f"source timed out after {timeout_seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout_seconds)
    try:
        return _run_source_product(date, cycle, source, settings, skip_render=skip_render)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_case_consensus(paths, case: BacktestCase, artifacts: list[ConsensusSource]) -> tuple[Path | None, Path | None, str]:
    if not artifacts:
        return None, None, "no_available_sources"
    output_path, metadata_path = build_forecast_consensus(
        date=case.date,
        cycle=case.cycle,
        sources=artifacts,
        output_path=consensus_product_path(paths.outputs, case.date, case.cycle),
        metadata_path=consensus_metadata_path(paths.outputs, case.date, case.cycle),
        field_name=CONSENSUS_INPUT_FIELD,
        unavailable_sources=[],
    )
    return output_path, metadata_path, ""


def _build_case_products(paths, settings, manifest: BacktestManifest, case: BacktestCase, output_dir: Path, sources: list[str]) -> list[dict[str, Any]]:
    product_target = manifest.target_product
    product_rows: list[dict[str, Any]] = []
    outdir = _case_output_dir(output_dir, manifest, case) / "products" / _source_set_key(sources)
    for day in product_target.get("lead_days", [1, 2, 3]):
        valid_date = (pd.Timestamp(case.date) + pd.Timedelta(days=int(day) - 1)).date().isoformat()
        try:
            metadata = build_product_bundle(
                date=case.date,
                cycle=case.cycle,
                valid_date=valid_date,
                field_name=str(product_target["field"]),
                outdir=outdir,
                map_style=str(product_target.get("map_style", "outlook")),
                map_domain=str(product_target.get("map_domain", "conus")),
                display_preset=str(product_target.get("display_preset", "auto")),
                artifact_source=str(product_target.get("artifact_source", "consensus")),
                require_production_basemap=True,
                overwrite=True,
                settings=settings,
                paths=paths,
            )
        except Exception as exc:
            product_rows.append(
                {
                    "case_id": case.case_id,
                    "date": case.date,
                    "cycle": case.cycle,
                    "day": int(day),
                    "valid_date": valid_date,
                    "metadata_path": "",
                    "image_path": "",
                    "publication_status": "failed_guardrails",
                    "public_ready": False,
                    "max_tornado_concern_prob": None,
                    "product_status": "failed",
                    "reason": str(exc),
                }
            )
            continue
        product_rows.append(
            {
                "case_id": case.case_id,
                "date": case.date,
                "cycle": case.cycle,
                "day": int(day),
                "valid_date": valid_date,
                "metadata_path": metadata.get("metadata_path", ""),
                "image_path": metadata.get("main_image_path", ""),
                "publication_status": metadata.get("publication_status", ""),
                "public_ready": bool(metadata.get("public_ready", False)),
                "max_tornado_concern_prob": metadata.get("max_tornado_concern_prob"),
                "product_status": "available",
                "reason": "",
            }
        )
    return product_rows


def _verify_case(paths, case: BacktestCase) -> tuple[Path | None, str]:
    consensus_path = consensus_product_path(paths.outputs, case.date, case.cycle)
    label_path = _latest_matching_file(paths.labels, "labels_*.nc")
    if not consensus_path.exists():
        return None, "missing_consensus"
    if label_path is None:
        return None, "missing_labels"
    metadata_path = consensus_metadata_path(paths.outputs, case.date, case.cycle)
    outbreak_path = _latest_matching_file(paths.labels, "outbreaks_*.parquet")
    training_summary_path = paths.models / "training_data_summary.json"
    with xr.load_dataset(consensus_path) as consensus_ds:
        if CONSENSUS_FIELD not in consensus_ds:
            return None, "missing_consensus_field"
        prediction_ds = xr.Dataset(
            {
                "tornado_prob": consensus_ds[CONSENSUS_FIELD],
                "hail_prob": xr.zeros_like(consensus_ds[CONSENSUS_FIELD]),
                "wind_prob": xr.zeros_like(consensus_ds[CONSENSUS_FIELD]),
                "any_prob": consensus_ds[CONSENSUS_FIELD],
                "outbreak_risk": consensus_ds[CONSENSUS_FIELD],
                "sig_tor_support": consensus_ds[CONSENSUS_FIELD],
                "confidence_score": consensus_ds.get("consensus_confidence_modifier", xr.ones_like(consensus_ds[CONSENSUS_FIELD])),
                "signal_quality_score": consensus_ds.get("model_agreement_count", xr.zeros_like(consensus_ds[CONSENSUS_FIELD])),
                "bust_risk_score": xr.zeros_like(consensus_ds[CONSENSUS_FIELD]),
            },
            coords={"time": consensus_ds["time"], "lat": consensus_ds["lat"], "lon": consensus_ds["lon"]},
        )
        label_ds = xr.load_dataset(label_path)
        outbreak_table = pd.read_parquet(outbreak_path) if outbreak_path else None
        evaluation_metadata = _read_json(metadata_path)
        training_summary = _read_json(training_summary_path)
        output_path = paths.verification / "backtests" / f"{case.case_id}_verification.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        verify_daily_probabilities(
            prediction_ds,
            label_ds,
            output_path,
            target_date=case.date,
            outbreak_table=outbreak_table,
            training_data_summary=training_summary,
            evaluation_metadata=evaluation_metadata,
        )
        label_ds.close()
    return output_path, ""


def _score_case(case: BacktestCase, products: list[dict[str, Any]], verification_path: Path | None, reason: str) -> dict[str, Any]:
    case_products = [product for product in products if product["case_id"] == case.case_id]
    max_risk = max((float(product.get("max_tornado_concern_prob") or 0.0) for product in case_products), default=0.0)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "date": case.date,
        "cycle": case.cycle,
        "expected_signal": case.expected_signal,
        "verification_path": str(verification_path) if verification_path else "",
        "verification_status": "available" if verification_path else "unavailable",
        "verification_reason": reason,
        "max_day1_3_tornado_concern": max_risk,
        "public_ready_days": sum(1 for product in case_products if product.get("public_ready")),
    }
    if verification_path and verification_path.exists():
        payload = _read_json(verification_path)
        per_day = payload.get("per_day", []) if isinstance(payload.get("per_day", []), list) else []
        row["observed_tornado_outbreak_any"] = any(bool(day.get("observed_tornado_outbreak", 0)) for day in per_day)
        row["observed_any_outbreak_any"] = any(bool(day.get("observed_any_outbreak", 0)) for day in per_day)
        row["mean_tornado_brier"] = sum(float(day.get("tornado_brier", 0.0) or 0.0) for day in per_day) / len(per_day) if per_day else 0.0
    return row


def _case_values(manifest: BacktestManifest, key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in manifest.cases:
        value = str(getattr(case, key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _tag_counts(manifest: BacktestManifest) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in manifest.cases:
        for tag in case.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def _source_ablation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sources = sorted({str(row["source"]) for row in rows if str(row["source"]) != "forecast_consensus"})
    result: dict[str, dict[str, Any]] = {}
    for source in sources:
        source_rows = [row for row in rows if row["source"] == source]
        available = sum(1 for row in source_rows if row["status"] == "available")
        result[source] = {
            "available_cases": available,
            "unavailable_cases": sum(1 for row in source_rows if row["status"] == "unavailable"),
            "planned_cases": sum(1 for row in source_rows if row["status"] == "planned"),
            "availability_fraction": available / len(source_rows) if source_rows else 0.0,
            "report_type": "availability_ablation",
        }
    return result


def _hydration_status_rows(paths, manifest: BacktestManifest, cases: list[BacktestCase], sources: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for source in sources:
            product_path = source_product_path(paths.outputs, case.date, case.cycle, source)
            metadata_path = source_metadata_path(paths.outputs, case.date, case.cycle, source)
            metadata = _read_json(metadata_path)
            ingest = metadata.get("ingest_summary", {}) if isinstance(metadata.get("ingest_summary", {}), dict) else {}
            product_exists = product_path.exists() and product_path.stat().st_size > 0
            metadata_exists = bool(metadata)
            source_mode = str(ingest.get("source_mode", ""))
            status = "available" if product_exists and metadata_exists and source_mode.lower() != "synthetic" else "missing"
            if product_exists and metadata_exists and source_mode.lower() == "synthetic":
                status = "synthetic_blocked"
            elif product_exists != metadata_exists:
                status = "partial"
            rows.append(
                {
                    "case_id": case.case_id,
                    "date": case.date,
                    "cycle": case.cycle,
                    "source": source,
                    "status": status,
                    "product_exists": product_exists,
                    "metadata_exists": metadata_exists,
                    "source_mode": source_mode,
                    "source_model": str(ingest.get("source_model", "")),
                    "product_path": str(product_path),
                    "metadata_path": str(metadata_path),
                }
            )
    return rows


def write_hydration_status(
    *,
    manifest_path: Path,
    output_dir: Path,
    sources_value: str = "manifest",
    limit: int | None = None,
    batch_size: int | None = None,
    batch_number: int | None = None,
) -> tuple[Path, Path]:
    manifest = load_backtest_manifest(manifest_path)
    settings = load_settings()
    paths = build_paths(settings)
    sources = _resolve_manifest_sources(manifest, sources_value)
    cases = _select_cases(manifest, limit=limit, batch_size=batch_size, batch_number=batch_number)
    rows = _hydration_status_rows(paths, manifest, cases, sources)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hydration_status.json"
    csv_path = output_dir / "hydration_status.csv"
    payload = {
        "manifest_id": manifest.manifest_id,
        "selected_case_count": len(cases),
        "selected_cases": [case.case_id for case in cases],
        "sources": sources,
        "available": sum(1 for row in rows if row["status"] == "available"),
        "missing": sum(1 for row in rows if row["status"] == "missing"),
        "partial": sum(1 for row in rows if row["status"] == "partial"),
        "synthetic_blocked": sum(1 for row in rows if row["status"] == "synthetic_blocked"),
        "rows": rows,
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    fieldnames = [
        "case_id",
        "date",
        "cycle",
        "source",
        "status",
        "product_exists",
        "metadata_exists",
        "source_mode",
        "source_model",
        "product_path",
        "metadata_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _readiness_status(rows: list[dict[str, Any]], products: list[dict[str, Any]], scores: list[dict[str, Any]], dry_run: bool) -> str:
    if dry_run:
        return "planned_only"
    if any(str(row.get("source_mode", "")).lower() == "synthetic" for row in rows):
        return "failed_guardrails"
    if products and any(not bool(product.get("public_ready")) for product in products):
        return "internal_review_only"
    if scores and any(str(score.get("verification_status")) != "available" for score in scores):
        return "internal_review_only"
    return "public_ready_candidate" if products or scores else "source_prep_only"


def _score_metric_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {
            "spc_probability_bins": list(SPC_TORNADO_THRESHOLDS),
            "thresholds": [],
            "reliability": [],
            "calibration_status": "insufficient_scores",
        }
    y_prob = pd.Series([float(score.get("max_day1_3_tornado_concern") or 0.0) for score in scores], dtype=float).to_numpy()
    y_true = pd.Series([bool(score.get("observed_tornado_outbreak_any", False)) for score in scores], dtype=float).to_numpy()
    return {
        "spc_probability_bins": list(SPC_TORNADO_THRESHOLDS),
        "thresholds": threshold_summaries(y_true, y_prob),
        "reliability": reliability_bins(y_true, y_prob, n_bins=10),
        "calibration_status": "candidate_only" if len(scores) < 100 else "ready_for_fit",
        "case_count": len(scores),
    }


def _candidate_lead_weight_report(manifest: BacktestManifest, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_stats = _source_ablation(rows)
    available_sources = {
        source: stats["availability_fraction"]
        for source, stats in source_stats.items()
        if float(stats["availability_fraction"]) > 0.0
    }
    total = sum(float(value) for value in available_sources.values())
    normalized = {source: value / total for source, value in available_sources.items()} if total else {}
    return {
        "method": "availability_normalized_seed",
        "status": "candidate_only",
        "note": "Replace with metric-optimized weights after locked v2 cases are hydrated and scored.",
        "default_sources": manifest.target_product.get("default_sources", []),
        "candidate_weights": normalized,
    }


def _source_comparison_summary(manifest: BacktestManifest, rows: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    groups = manifest.target_product.get("source_comparison_groups", {})
    if not isinstance(groups, dict) or not groups:
        return {}
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_case.setdefault(str(row["case_id"]), []).append(row)
    scores_by_case = {str(score["case_id"]): score for score in scores if "case_id" in score}
    summary: dict[str, Any] = {}
    for group_name, raw_sources in sorted(groups.items()):
        sources = [str(source) for source in raw_sources] if isinstance(raw_sources, list) else []
        case_rows: list[dict[str, Any]] = []
        for case_id, source_rows in rows_by_case.items():
            available = sorted({str(row["source"]) for row in source_rows if row["status"] == "available"})
            missing = [source for source in sources if source not in available]
            score = scores_by_case.get(case_id, {})
            case_rows.append(
                {
                    "case_id": case_id,
                    "available_sources": [source for source in sources if source in available],
                    "missing_sources": missing,
                    "all_sources_available": not missing,
                    "observed_tornado_outbreak_any": score.get("observed_tornado_outbreak_any"),
                }
            )
        complete = sum(1 for row in case_rows if row["all_sources_available"])
        summary[str(group_name)] = {
            "sources": sources,
            "case_count": len(case_rows),
            "complete_case_count": complete,
            "complete_fraction": complete / len(case_rows) if case_rows else 0.0,
            "score_status": "availability_only",
            "score_note": "Run this group as the selected --sources set to produce a group-specific consensus score.",
            "cases": case_rows,
        }
    return summary


def _manifest_summary(
    manifest: BacktestManifest,
    rows: list[dict[str, Any]],
    products: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    *,
    dry_run: bool,
    selected_cases: list[BacktestCase],
    run_options: dict[str, Any],
) -> dict[str, Any]:
    return {
        "manifest_id": manifest.manifest_id,
        "description": manifest.description,
        "dry_run": dry_run,
        "case_count": len(manifest.cases),
        "selected_case_count": len(selected_cases),
        "selected_cases": [case.case_id for case in selected_cases],
        "run_options": run_options,
        "target_product": manifest.target_product,
        "fold_counts": _case_values(manifest, "fold"),
        "region_counts": _case_values(manifest, "region"),
        "season_counts": _case_values(manifest, "season"),
        "regime_counts": _case_values(manifest, "regime"),
        "tag_counts": _tag_counts(manifest),
        "source_ablation": _source_ablation(rows),
        "source_comparison": _source_comparison_summary(manifest, rows, scores),
        "learned_weight_report": _candidate_lead_weight_report(manifest, rows),
        "metric_summary": _score_metric_summary(scores),
        "spc_comparison": {
            "status": "not_configured",
            "day1_day2_target": "SPC tornado probability within 25 miles of a point",
            "day3_target": "experimental Severewx tornado guidance compared with SPC Day 3 categorical/total severe context",
        },
        "readiness_status": _readiness_status(rows, products, scores, dry_run),
    }


def _case_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case_id"]), str(row["date"]), str(row["cycle"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (case_id, date, cycle), case_rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        available = [str(row["source"]) for row in case_rows if row["status"] == "available"]
        unavailable = [
            {"source": str(row["source"]), "reason": str(row["reason"])}
            for row in case_rows
            if row["status"] != "available"
        ]
        summary.append(
            {
                "case_id": case_id,
                "date": date,
                "cycle": cycle,
                "available_sources": available,
                "unavailable_sources": unavailable,
                "ready_for_scoring": len(available) >= 1,
            }
        )
    return summary


def _resolve_manifest_sources(manifest: BacktestManifest, sources_value: str) -> list[str]:
    value = str(sources_value).strip()
    normalized = value.lower()
    if normalized == "manifest":
        return [str(source).lower() for source in manifest.target_product["default_sources"]]
    if normalized.startswith("group:"):
        group_name = value.split(":", 1)[1].strip()
        groups = manifest.target_product.get("source_comparison_groups", {})
        if not isinstance(groups, dict) or group_name not in groups:
            available = ", ".join(sorted(str(key) for key in groups)) if isinstance(groups, dict) else ""
            raise ValueError(f"unknown source comparison group: {group_name}; available groups: {available}")
        return [str(source).lower() for source in groups[group_name]]
    return _parse_sources(value)


def prepare_backtest_cases(
    *,
    manifest_path: Path,
    output_dir: Path,
    sources_value: str = "manifest",
    dry_run: bool = False,
    limit: int | None = None,
    batch_size: int | None = None,
    batch_number: int | None = None,
    resume: bool = False,
    coverage_only: bool = False,
    source_timeout_seconds: int | None = None,
    lead_hours: list[int] | None = None,
    skip_render: bool = True,
    build_consensus: bool = False,
    build_products: bool = False,
    verify: bool = False,
    score: bool = False,
) -> tuple[BacktestManifest, list[dict[str, Any]], tuple[Path, Path, Path, Path, Path]]:
    manifest = load_backtest_manifest(manifest_path)
    settings = load_settings()
    if lead_hours is not None:
        settings.raw.setdefault("ingest", {})["leads"] = [int(value) for value in lead_hours]
    paths = build_paths(settings)
    if coverage_only:
        build_consensus = False
        build_products = False
        verify = False
        score = False
    sources = _resolve_manifest_sources(manifest, sources_value)
    cases = _select_cases(manifest, limit=limit, batch_size=batch_size, batch_number=batch_number)
    run_options = {
        "sources": sources,
        "limit": limit,
        "batch_size": batch_size,
        "batch_number": batch_number,
        "resume": resume,
        "coverage_only": coverage_only,
        "source_timeout_seconds": source_timeout_seconds,
        "lead_hours": [int(value) for value in lead_hours] if lead_hours is not None else None,
        "build_consensus": build_consensus,
        "build_products": build_products,
        "verify": verify,
        "score": score,
    }
    rows: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for case in cases:
        artifacts: list[ConsensusSource] = []
        for source in sources:
            product_path = source_product_path(paths.outputs, case.date, case.cycle, source)
            metadata_path = source_metadata_path(paths.outputs, case.date, case.cycle, source)
            if dry_run:
                rows.append(
                    _source_row(
                        case=case,
                        source=source,
                        status="planned",
                        reason="dry_run",
                        product_path=product_path,
                        metadata_path=metadata_path,
                    )
                )
                _checkpoint_outputs(
                    output_dir,
                    manifest,
                    cases,
                    rows,
                    products,
                    scores,
                    dry_run=dry_run,
                    run_options=run_options,
                )
                continue
            artifact = _existing_source_artifact(paths, case, source) if resume else None
            if artifact is not None:
                artifacts.append(artifact)
                rows.append(
                    _source_row(
                        case=case,
                        source=source,
                        status="available",
                        reason="reused_existing",
                        product_path=product_path,
                        metadata_path=metadata_path,
                    )
                )
                _checkpoint_outputs(
                    output_dir,
                    manifest,
                    cases,
                    rows,
                    products,
                    scores,
                    dry_run=dry_run,
                    run_options=run_options,
                )
                continue
            artifact = None
            source_error = ""
            try:
                artifact = _run_source_product_with_timeout(
                    case.date,
                    case.cycle,
                    source,
                    settings,
                    skip_render=skip_render,
                    timeout_seconds=source_timeout_seconds,
                )
            except Exception as exc:
                source_error = str(exc)
            if artifact is not None:
                artifacts.append(artifact)
            status = "available" if artifact is not None else "unavailable"
            reason = "" if artifact is not None else (source_error or "source_run_failed_or_unavailable")
            rows.append(
                _source_row(
                    case=case,
                    source=source,
                    status=status,
                    reason=reason,
                    product_path=product_path,
                    metadata_path=metadata_path,
                )
            )
            _checkpoint_outputs(
                output_dir,
                manifest,
                cases,
                rows,
                products,
                scores,
                dry_run=dry_run,
                run_options=run_options,
            )
        if dry_run:
            continue
        expected_consensus_sources = [artifact.name for artifact in artifacts]
        artifact_paths = _case_artifact_paths(paths, output_dir, manifest, case, expected_consensus_sources)
        consensus_path = consensus_product_path(artifact_paths.outputs, case.date, case.cycle)
        if build_consensus or build_products or verify or score:
            existing_consensus_path, _ = _existing_consensus(artifact_paths, case, expected_consensus_sources) if resume else (None, None)
            if existing_consensus_path is not None:
                consensus_path = existing_consensus_path
                consensus_reason = "reused_existing"
            else:
                consensus_path, _, consensus_reason = _build_case_consensus(artifact_paths, case, artifacts)
            rows.append(
                {
                    "case_id": case.case_id,
                    "date": case.date,
                    "cycle": case.cycle,
                    "source": "forecast_consensus",
                    "status": "available" if consensus_path else "unavailable",
                    "reason": consensus_reason,
                    "source_mode": "real" if consensus_path else "",
                    "source_model": "consensus",
                    "available_fields": CONSENSUS_FIELD if consensus_path else "",
                    "missing_requested_leads": "",
                    "product_path": str(consensus_path) if consensus_path else "",
                    "metadata_path": str(consensus_metadata_path(artifact_paths.outputs, case.date, case.cycle)) if consensus_path else "",
                }
            )
        if build_products and consensus_path:
            existing_products = _existing_case_products(paths, manifest, case, output_dir, expected_consensus_sources) if resume else []
            products.extend(existing_products or _build_case_products(artifact_paths, settings, manifest, case, output_dir, expected_consensus_sources))
        verification_path = None
        verification_reason = ""
        if verify or score:
            existing_verification = _existing_verification(artifact_paths, case) if resume else None
            if existing_verification is not None:
                verification_path, verification_reason = existing_verification, "reused_existing"
            else:
                verification_path, verification_reason = _verify_case(artifact_paths, case)
        if score:
            scores.append(_score_case(case, products, verification_path, verification_reason))
        _checkpoint_outputs(
            output_dir,
            manifest,
            cases,
            rows,
            products,
            scores,
            dry_run=dry_run,
            run_options=run_options,
        )
    outputs = _write_outputs(output_dir, manifest, rows, dry_run=dry_run, selected_cases=cases, run_options=run_options, products=products, scores=scores)
    return manifest, rows, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed backtest cases and source availability ledgers")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default="data/outputs/verification/backtests")
    parser.add_argument("--sources", default="manifest", help="manifest, auto, or comma-delimited source list")
    parser.add_argument("--limit", type=int, help="Limit number of manifest cases")
    parser.add_argument("--batch-size", type=int, help="Number of manifest cases to run in this batch")
    parser.add_argument("--batch-number", type=int, help="1-based batch number to run with --batch-size")
    parser.add_argument("--resume", action="store_true", help="Reuse existing source, consensus, product, and verification artifacts when present")
    parser.add_argument("--coverage-only", action="store_true", help="Hydrate or reuse source artifacts and write availability ledgers without consensus, products, verification, or scoring")
    parser.add_argument("--source-timeout-seconds", type=int, help="Best-effort per-source timeout on platforms that support SIGALRM")
    parser.add_argument("--lead-hours", help="Comma-delimited ingest lead hours for this run, for example 0,24,48")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write the ledger without ingesting sources")
    parser.add_argument("--status-only", action="store_true", help="Write existing source artifact hydration status without ingesting or scoring")
    parser.add_argument("--render", action="store_true", help="Render per-source maps while preparing source products")
    parser.add_argument("--build-consensus", action="store_true", help="Build consensus artifact after source prep")
    parser.add_argument("--build-products", action="store_true", help="Build Day 1-3 CONUS consensus products")
    parser.add_argument("--verify", action="store_true", help="Attach verification labels when local labels exist")
    parser.add_argument("--score", action="store_true", help="Write first-pass score rows")
    args = parser.parse_args()

    if bool(args.status_only):
        json_path, csv_path = write_hydration_status(
            manifest_path=Path(args.manifest),
            output_dir=Path(args.output_dir),
            sources_value=str(args.sources),
            limit=args.limit,
            batch_size=args.batch_size,
            batch_number=args.batch_number,
        )
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"manifest_id={payload['manifest_id']}")
        print(f"selected_case_count={payload['selected_case_count']}")
        print(f"sources={len(payload['sources'])}")
        print(f"available={payload['available']}")
        print(f"missing={payload['missing']}")
        print(f"partial={payload['partial']}")
        print(f"synthetic_blocked={payload['synthetic_blocked']}")
        print(f"hydration_json={json_path}")
        print(f"hydration_csv={csv_path}")
        return

    manifest, rows, outputs = prepare_backtest_cases(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        sources_value=str(args.sources),
        dry_run=bool(args.dry_run),
        limit=args.limit,
        batch_size=args.batch_size,
        batch_number=args.batch_number,
        resume=bool(args.resume),
        coverage_only=bool(args.coverage_only),
        source_timeout_seconds=args.source_timeout_seconds,
        lead_hours=_parse_lead_hours(args.lead_hours),
        skip_render=not bool(args.render),
        build_consensus=bool(args.build_consensus),
        build_products=bool(args.build_products),
        verify=bool(args.verify),
        score=bool(args.score),
    )
    available = sum(1 for row in rows if row["status"] == "available")
    planned = sum(1 for row in rows if row["status"] == "planned")
    unavailable = sum(1 for row in rows if row["status"] == "unavailable")
    print(f"manifest_id={manifest.manifest_id}")
    print(f"cases={len({row['case_id'] for row in rows})}")
    print(f"sources={len({row['source'] for row in rows})}")
    print(f"available={available}")
    print(f"planned={planned}")
    print(f"unavailable={unavailable}")
    print(f"manifest_copy={outputs[0]}")
    print(f"availability_json={outputs[1]}")
    print(f"availability_csv={outputs[2]}")
    print(f"summary_json={outputs[3]}")
    print(f"summary_csv={outputs[4]}")


if __name__ == "__main__":
    main()
