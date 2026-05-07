"""Prepare deterministic consensus backtest cases and source availability ledgers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from severewx.backtest.manifest import BacktestCase, BacktestManifest, load_backtest_manifest, manifest_to_jsonable
from severewx.cli.run_forecast_consensus import _parse_sources, _read_json, _run_source_product
from severewx.config import load_settings
from severewx.models.forecast_consensus import source_metadata_path, source_product_path
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


def _write_outputs(output_dir: Path, manifest: BacktestManifest, rows: list[dict[str, Any]], *, dry_run: bool) -> tuple[Path, Path, Path]:
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
    for case in cases:
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
    outputs = _write_outputs(output_dir, manifest, rows, dry_run=dry_run)
    return manifest, rows, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed backtest cases and source availability ledgers")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default="data/outputs/verification/backtests")
    parser.add_argument("--sources", default="manifest", help="manifest, auto, or comma-delimited source list")
    parser.add_argument("--limit", type=int, help="Limit number of manifest cases")
    parser.add_argument("--dry-run", action="store_true", help="Validate and write the ledger without ingesting sources")
    parser.add_argument("--render", action="store_true", help="Render per-source maps while preparing source products")
    args = parser.parse_args()

    manifest, rows, outputs = prepare_backtest_cases(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        sources_value=str(args.sources),
        dry_run=bool(args.dry_run),
        limit=args.limit,
        skip_render=not bool(args.render),
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
