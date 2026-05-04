"""Summarize generated tornado-concern product readiness from metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


STATUS_COLUMNS = [
    "date",
    "cycle",
    "valid_period_label",
    "variant",
    "field_name",
    "map_style",
    "map_domain",
    "artifact_source_requested",
    "production_basemap_required",
    "render_basemap_mode",
    "render_basemap_warning",
    "publication_status",
    "public_ready",
    "display_visible_signal",
    "failure_reasons",
    "source_count",
    "included_sources",
    "excluded_source_count",
    "max_signal_agreement_count",
    "max_public_display_tornado_concern_prob",
    "public_display_grid_cells_ge_02pct",
    "display_largest_object_cells_ge_02pct",
    "metadata_path",
    "image_path",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _metadata_files(products_dir: Path, *, recursive: bool = False) -> list[Path]:
    pattern = "**/*.json" if recursive else "*.json"
    paths: list[Path] = []
    for path in sorted(products_dir.glob(pattern)):
        if path.name.endswith("manifest.json"):
            continue
        payload = _read_json(path)
        if payload and {"date", "field_name", "publication_status"}.issubset(payload):
            paths.append(path)
    return paths


def _source_fields(payload: dict[str, Any]) -> dict[str, Any]:
    consensus = payload.get("forecast_consensus_summary", {})
    ingest = payload.get("forecast_ingest_summary", {})
    consensus = consensus if isinstance(consensus, dict) else {}
    ingest = ingest if isinstance(ingest, dict) else {}
    included = consensus.get("included_sources") or ingest.get("included_sources") or []
    if not included and ingest.get("source"):
        included = [str(ingest.get("source"))]
    excluded = consensus.get("excluded_sources") or ingest.get("excluded_sources") or []
    audit = payload.get("consensus_audit", {})
    audit = audit if isinstance(audit, dict) else {}
    return {
        "source_count": len(included),
        "included_sources": ",".join(str(source) for source in included),
        "excluded_source_count": len(excluded),
        "max_signal_agreement_count": int(audit.get("max_signal_agreement_count", 0) or 0),
    }


def _status_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": str(payload.get("date", "")),
        "cycle": str(payload.get("cycle", "")),
        "valid_period_label": str(payload.get("valid_period_label", payload.get("valid_date", ""))),
        "variant": str(payload.get("variant", "")),
        "field_name": str(payload.get("field_name", "")),
        "map_style": str(payload.get("map_style", "")),
        "map_domain": str(payload.get("map_domain", "")),
        "artifact_source_requested": str(payload.get("artifact_source_requested", "")),
        "production_basemap_required": bool(payload.get("production_basemap_required", False)),
        "render_basemap_mode": str(payload.get("render_basemap_mode", "")),
        "render_basemap_warning": str(payload.get("render_basemap_warning", "") or "none"),
        "publication_status": str(payload.get("publication_status", "")),
        "public_ready": bool(payload.get("public_ready", False)),
        "display_visible_signal": bool(payload.get("display_visible_signal", True)),
        "failure_reasons": str(payload.get("failure_reasons", "") or "none"),
        "max_public_display_tornado_concern_prob": float(payload.get("max_public_display_tornado_concern_prob", 0.0) or 0.0),
        "public_display_grid_cells_ge_02pct": int(payload.get("public_display_grid_cells_ge_02pct", 0) or 0),
        "display_largest_object_cells_ge_02pct": int(payload.get("display_largest_object_cells_ge_02pct", 0) or 0),
        "metadata_path": str(path),
        "image_path": str(payload.get("main_image_path", "")),
        **_source_fields(payload),
    }


def collect_product_status(metadata_paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in metadata_paths:
        payload = _read_json(path)
        if payload is None:
            continue
        rows.append(_status_row(path, payload))
    if not rows:
        return pd.DataFrame(columns=STATUS_COLUMNS)
    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=STATUS_COLUMNS)
    return frame.sort_values(["date", "cycle", "valid_period_label", "variant"], kind="mergesort").reset_index(drop=True)


def product_status_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "# Tornado Concern Product Status",
        "",
        "| date | valid_period | variant | domain | render | publication | visible | sources | max_display | failure_reasons |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {date} | {valid_period_label} | {variant} | {map_domain} | {render_basemap_mode} | {publication_status} | {display_visible_signal} | {source_count} | {max_display:.4f} | {failure_reasons} |".format(
                date=row.get("date", ""),
                valid_period_label=row.get("valid_period_label", ""),
                variant=row.get("variant", ""),
                map_domain=row.get("map_domain", ""),
                render_basemap_mode=row.get("render_basemap_mode", ""),
                publication_status=row.get("publication_status", ""),
                display_visible_signal=str(bool(row.get("display_visible_signal", False))),
                source_count=int(row.get("source_count", 0) or 0),
                max_display=float(row.get("max_public_display_tornado_concern_prob", 0.0) or 0.0),
                failure_reasons=row.get("failure_reasons", "none"),
            )
        )
    return "\n".join(lines) + "\n"


def blocked_status_count(frame: pd.DataFrame) -> int:
    if frame.empty or "publication_status" not in frame:
        return 0
    return int(frame["publication_status"].astype(str).ne("public_candidate").sum())


def fallback_render_count(frame: pd.DataFrame) -> int:
    if frame.empty or "render_basemap_mode" not in frame:
        return 0
    return int(frame["render_basemap_mode"].astype(str).eq("matplotlib_fallback").sum())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", nargs="+", help="One or more product metadata JSON files")
    parser.add_argument("--products-dir", help="Directory containing product metadata JSON files")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --products-dir")
    parser.add_argument("--output-csv", help="Optional CSV status table path")
    parser.add_argument("--output-md", help="Optional Markdown status report path")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Exit nonzero when any product is not public_candidate")
    parser.add_argument("--fail-on-fallback-render", action="store_true", help="Exit nonzero when any product used the Matplotlib fallback basemap")
    args = parser.parse_args(argv)

    if bool(args.metadata) == bool(args.products_dir):
        parser.error("provide exactly one of --metadata or --products-dir")
    metadata_paths = [Path(path) for path in args.metadata] if args.metadata else _metadata_files(Path(args.products_dir), recursive=bool(args.recursive))
    frame = collect_product_status(metadata_paths)
    markdown = product_status_markdown(frame)
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_csv, index=False)
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    if args.fail_on_blocked and blocked_status_count(frame):
        raise SystemExit(2)
    if args.fail_on_fallback_render and fallback_render_count(frame):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
