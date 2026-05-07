"""Prepare deterministic consensus backtest cases and source availability ledgers."""

from __future__ import annotations

import argparse
import csv
import json
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
from severewx.utils.paths import build_paths


DEFAULT_MANIFEST = Path("backtests/tornado_environment_consensus_v1.json")


def _case_output_dir(base_output_dir: Path, manifest: BacktestManifest, case: BacktestCase) -> Path:
    return base_output_dir / manifest.manifest_id / f"{case.date}_{case.cycle}z"


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
    products: list[dict[str, Any]] | None = None,
    scores: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_copy = output_dir / "manifest.json"
    manifest_copy.write_text(json.dumps(manifest_to_jsonable(manifest), indent=2), encoding="utf-8")
    json_path = output_dir / "source_availability.json"
    csv_path = output_dir / "source_availability.csv"
    payload = {
        "manifest_id": manifest.manifest_id,
        "target_product": manifest.target_product,
        "dry_run": dry_run,
        "case_count": len(manifest.cases),
        "rows": rows,
        "case_summary": _case_summary(rows),
        "products": products or [],
        "scores": scores or [],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    return manifest_copy, json_path, csv_path


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    matches = list(directory.glob(pattern))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


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


def _build_case_products(paths, settings, manifest: BacktestManifest, case: BacktestCase, output_dir: Path) -> list[dict[str, Any]]:
    product_target = manifest.target_product
    product_rows: list[dict[str, Any]] = []
    outdir = _case_output_dir(output_dir, manifest, case) / "products"
    for day in product_target.get("lead_days", [1, 2, 3]):
        valid_date = (pd.Timestamp(case.date) + pd.Timedelta(days=int(day) - 1)).date().isoformat()
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


def prepare_backtest_cases(
    *,
    manifest_path: Path,
    output_dir: Path,
    sources_value: str = "manifest",
    dry_run: bool = False,
    limit: int | None = None,
    skip_render: bool = True,
    build_consensus: bool = False,
    build_products: bool = False,
    verify: bool = False,
    score: bool = False,
) -> tuple[BacktestManifest, list[dict[str, Any]], tuple[Path, Path, Path]]:
    manifest = load_backtest_manifest(manifest_path)
    settings = load_settings()
    paths = build_paths(settings)
    sources = (
        [str(source).lower() for source in manifest.target_product["default_sources"]]
        if sources_value == "manifest"
        else _parse_sources(sources_value)
    )
    cases = list(manifest.cases[:limit] if limit is not None else manifest.cases)
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
                continue
            artifact = _run_source_product(case.date, case.cycle, source, settings, skip_render=skip_render)
            if artifact is not None:
                artifacts.append(artifact)
            status = "available" if artifact is not None else "unavailable"
            reason = "" if artifact is not None else "source_run_failed_or_unavailable"
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
        if dry_run:
            continue
        consensus_path = consensus_product_path(paths.outputs, case.date, case.cycle)
        if build_consensus or build_products or verify or score:
            consensus_path, _, consensus_reason = _build_case_consensus(paths, case, artifacts)
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
                    "metadata_path": str(consensus_metadata_path(paths.outputs, case.date, case.cycle)) if consensus_path else "",
                }
            )
        if build_products and consensus_path:
            products.extend(_build_case_products(paths, settings, manifest, case, output_dir))
        verification_path = None
        verification_reason = ""
        if verify or score:
            verification_path, verification_reason = _verify_case(paths, case)
        if score:
            scores.append(_score_case(case, products, verification_path, verification_reason))
    outputs = _write_outputs(output_dir, manifest, rows, dry_run=dry_run, products=products, scores=scores)
    return manifest, rows, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed backtest cases and source availability ledgers")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default="data/outputs/verification/backtests")
    parser.add_argument("--sources", default="manifest", help="manifest, auto, or comma-delimited source list")
    parser.add_argument("--limit", type=int, help="Limit number of manifest cases")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write the ledger without ingesting sources")
    parser.add_argument("--render", action="store_true", help="Render per-source maps while preparing source products")
    parser.add_argument("--build-consensus", action="store_true", help="Build consensus artifact after source prep")
    parser.add_argument("--build-products", action="store_true", help="Build Day 1-3 CONUS consensus products")
    parser.add_argument("--verify", action="store_true", help="Attach verification labels when local labels exist")
    parser.add_argument("--score", action="store_true", help="Write first-pass score rows")
    args = parser.parse_args()

    manifest, rows, outputs = prepare_backtest_cases(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        sources_value=str(args.sources),
        dry_run=bool(args.dry_run),
        limit=args.limit,
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


if __name__ == "__main__":
    main()
