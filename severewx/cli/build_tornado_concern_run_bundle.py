"""Build a direct/consensus tornado-concern map bundle for one valid window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from severewx.cli.build_tornado_concern_product import DEFAULT_CONSENSUS_PRODUCT_FIELD, DEFAULT_PRODUCT_FIELD, build_product_bundle
from severewx.cli.tornado_concern_product_status import collect_product_status, product_status_markdown
from severewx.config import load_settings
from severewx.render.layout import cartopy_available, conus_template


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metadata_summary(metadata: dict[str, Any]) -> list[str]:
    diagnostics = metadata.get("ingredient_diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    normalized = diagnostics.get("ingredient_normalized_at_peak", {})
    peak_values = diagnostics.get("ingredient_values_at_peak", {})
    return [
        f"- publication_status: `{metadata.get('publication_status', '')}`",
        f"- public_ready: `{metadata.get('public_ready', False)}`",
        f"- render_basemap_mode: `{metadata.get('render_basemap_mode', '')}`",
        f"- max_public_display: `{_safe_float(metadata.get('max_public_display_tornado_concern_prob')):.4f}`",
        f"- display_cells_ge_02pct: `{int(metadata.get('public_display_grid_cells_ge_02pct', 0) or 0)}`",
        f"- display_cells_ge_10pct: `{int(metadata.get('public_display_grid_cells_ge_10pct', 0) or 0)}`",
        f"- largest_display_object_ge_02pct: `{int(metadata.get('display_largest_object_cells_ge_02pct', 0) or 0)}`",
        f"- top_valid_date: `{metadata.get('top_valid_date', '')}`",
        f"- peak_location: `{diagnostics.get('peak_lat', '')}, {diagnostics.get('peak_lon', '')}`",
        f"- limiting_ingredient_at_peak: `{diagnostics.get('limiting_ingredient_at_peak', '')}`",
        f"- normalized_peak_ingredients: `{json.dumps(normalized, sort_keys=True)}`",
        f"- peak_ingredient_values: `{json.dumps(peak_values, sort_keys=True)}`",
    ]


def _consensus_summary(metadata: dict[str, Any]) -> list[str]:
    forecast = metadata.get("forecast_consensus_summary", {})
    forecast = forecast if isinstance(forecast, dict) else {}
    audit = metadata.get("consensus_audit", {})
    audit = audit if isinstance(audit, dict) else {}
    excluded = forecast.get("excluded_sources", []) or []
    excluded_lines = []
    for item in excluded:
        if isinstance(item, dict):
            source = str(item.get("source", ""))
            reason = str(item.get("reason", ""))
            excluded_lines.append(f"  - `{source}`: `{reason}`")
        else:
            excluded_lines.append(f"  - `{item}`")
    time_weight_lines = []
    for item in forecast.get("time_weights", []) or []:
        if not isinstance(item, dict):
            continue
        valid_time = item.get("valid_time", "")
        lead_hour = item.get("lead_hour", "")
        contributing = ",".join(str(source) for source in item.get("contributing_sources", []) or [])
        applied = json.dumps(item.get("applied_weights", {}) or {}, sort_keys=True)
        time_weight_lines.append(f"  - `{valid_time}` lead `{lead_hour}` contributing `{contributing}` weights `{applied}`")
    lines = [
        f"- publication_status: `{metadata.get('publication_status', '')}`",
        f"- failure_reasons: `{metadata.get('failure_reasons', '') or 'none'}`",
        f"- included_sources: `{','.join(forecast.get('included_sources', []) or [])}`",
        f"- excluded_source_count: `{len(forecast.get('excluded_sources', []) or [])}`",
        f"- max_signal_agreement_count: `{audit.get('max_signal_agreement_count', '')}`",
    ]
    if excluded_lines:
        lines.extend(["- excluded_sources:"] + excluded_lines)
    if time_weight_lines:
        lines.extend(["- consensus_time_weights:"] + time_weight_lines)
    return lines


def _recommended_actions(status_frame: pd.DataFrame, *, consensus: dict[str, Any] | None) -> list[str]:
    actions: list[str] = []
    if not status_frame.empty and status_frame["render_basemap_mode"].astype(str).eq("matplotlib_fallback").any():
        actions.append("Install/enable Cartopy in the production runtime, then rerun the bundle without `--allow-fallback-publication`.")
    if consensus is None:
        actions.append("Consensus product did not generate; inspect forecast_consensus artifacts for the init cycle.")
    else:
        audit = consensus.get("consensus_audit", {})
        audit = audit if isinstance(audit, dict) else {}
        forecast = consensus.get("forecast_consensus_summary", {})
        forecast = forecast if isinstance(forecast, dict) else {}
        if int(audit.get("max_signal_agreement_count", 0) or 0) < 2:
            excluded = forecast.get("excluded_sources", []) or []
            excluded_names = [
                str(item.get("source", ""))
                for item in excluded
                if isinstance(item, dict) and item.get("source")
            ]
            source_text = ", ".join(excluded_names) if excluded_names else "excluded consensus sources"
            actions.append(f"Recover or rerun missing consensus sources ({source_text}) before treating consensus as operational.")
    if not actions:
        actions.append("Review the direct regional map against SPC/meteorology before publication.")
    return actions


def _environment_payload(*, require_production_basemap: bool) -> dict[str, Any]:
    settings = load_settings()
    template = conus_template(settings)
    return {
        "cartopy_available": cartopy_available(),
        "render_cartopy_requested": bool(settings.get("render.cartopy", True)),
        "render_basemap_mode": "cartopy" if template.use_cartopy else "matplotlib_fallback",
        "render_projection_name": template.projection_name,
        "production_basemap_required": require_production_basemap,
    }


def _manifest_payload(
    *,
    date: str,
    cycle: str,
    valid_start: str,
    valid_end: str,
    outdir: Path,
    products: dict[str, dict[str, Any]],
    status_csv: Path,
    status_md: Path,
    run_summary_md: Path,
    environment_json: Path,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "date": date,
        "cycle": cycle,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "outdir": str(outdir.resolve()),
        "status_csv": str(status_csv.resolve()),
        "status_md": str(status_md.resolve()),
        "run_summary_md": str(run_summary_md.resolve()),
        "environment_json": str(environment_json.resolve()),
        "products": {
            name: {
                "publication_status": payload.get("publication_status", ""),
                "public_ready": payload.get("public_ready", False),
                "image_path": payload.get("main_image_path", ""),
                "metadata_path": payload.get("metadata_path", ""),
                "summary_path": payload.get("summary_path", ""),
            }
            for name, payload in products.items()
        },
        "errors": errors,
    }


def _write_bundle_summary(
    path: Path,
    *,
    status_frame: pd.DataFrame,
    direct_regional: dict[str, Any],
    consensus: dict[str, Any] | None,
    environment: dict[str, Any],
    errors: list[str],
) -> None:
    lines = [
        "# Tornado Concern Run Bundle",
        "",
        "## Environment",
        "",
        f"- cartopy_available: `{environment.get('cartopy_available', False)}`",
        f"- render_cartopy_requested: `{environment.get('render_cartopy_requested', False)}`",
        f"- render_basemap_mode: `{environment.get('render_basemap_mode', '')}`",
        f"- production_basemap_required: `{environment.get('production_basemap_required', False)}`",
        "",
        "## Product Status",
        "",
        product_status_markdown(status_frame).strip(),
        "",
        "## Direct Regional Quality",
        "",
        *_metadata_summary(direct_regional),
        "",
        "## Consensus Source Availability",
        "",
    ]
    if consensus is None:
        lines.append("- consensus_status: `not_generated`")
    else:
        lines.extend(_consensus_summary(consensus))
    lines.extend(["", "## Recommended Next Actions", ""])
    lines.extend(f"- {action}" for action in _recommended_actions(status_frame, consensus=consensus))
    if errors:
        lines.extend(["", "## Generation Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_run_bundle(
    *,
    date: str,
    cycle: str,
    valid_start: str,
    valid_end: str,
    outdir: Path,
    require_production_basemap: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    products: dict[str, dict[str, Any]] = {}
    specs = [
        (
            "direct_regional",
            {
                "field_name": DEFAULT_PRODUCT_FIELD,
                "artifact_source": "prediction",
                "map_domain": "regional",
            },
        ),
        (
            "direct_conus",
            {
                "field_name": DEFAULT_PRODUCT_FIELD,
                "artifact_source": "prediction",
                "map_domain": "conus",
            },
        ),
        (
            "consensus_conus",
            {
                "field_name": DEFAULT_CONSENSUS_PRODUCT_FIELD,
                "artifact_source": "consensus",
                "map_domain": "conus",
            },
        ),
    ]
    for name, kwargs in specs:
        try:
            products[name] = build_product_bundle(
                date=date,
                cycle=cycle,
                valid_start=valid_start,
                valid_end=valid_end,
                outdir=outdir / name,
                map_style="outlook",
                require_production_basemap=require_production_basemap,
                overwrite=overwrite,
                **kwargs,
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    metadata_paths = [Path(payload["metadata_path"]) for payload in products.values() if payload.get("metadata_path")]
    status_frame = collect_product_status(metadata_paths)
    status_csv = outdir / "status.csv"
    status_md = outdir / "status.md"
    run_summary_md = outdir / "run_summary.md"
    environment_json = outdir / "environment.json"
    manifest_json = outdir / "manifest.json"
    environment = _environment_payload(require_production_basemap=require_production_basemap)
    status_frame.to_csv(status_csv, index=False)
    status_md.write_text(product_status_markdown(status_frame), encoding="utf-8")
    environment_json.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    if "direct_regional" in products:
        _write_bundle_summary(
            run_summary_md,
            status_frame=status_frame,
            direct_regional=products["direct_regional"],
            consensus=products.get("consensus_conus"),
            environment=environment,
            errors=errors,
        )
    else:
        run_summary_md.write_text("# Tornado Concern Run Bundle\n\nNo direct regional product was generated.\n", encoding="utf-8")
    manifest = _manifest_payload(
        date=date,
        cycle=cycle,
        valid_start=valid_start,
        valid_end=valid_end,
        outdir=outdir,
        products=products,
        status_csv=status_csv,
        status_md=status_md,
        run_summary_md=run_summary_md,
        environment_json=environment_json,
        errors=errors,
    )
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "outdir": str(outdir.resolve()),
        "status_csv": str(status_csv.resolve()),
        "status_md": str(status_md.resolve()),
        "run_summary_md": str(run_summary_md.resolve()),
        "manifest_json": str(manifest_json.resolve()),
        "environment_json": str(environment_json.resolve()),
        "products": products,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Forecast init date YYYY-MM-DD")
    parser.add_argument("--cycle", required=True, help="Forecast init cycle, e.g. 00")
    parser.add_argument("--valid-start", required=True, help="Custom valid-window start time, e.g. 2026-05-05T12:00Z")
    parser.add_argument("--valid-end", required=True, help="Custom valid-window end time, e.g. 2026-05-06T12:00Z")
    parser.add_argument("--outdir", required=True, help="Bundle output directory")
    parser.add_argument(
        "--allow-fallback-publication",
        action="store_true",
        help="Do not downgrade fallback Matplotlib renders to needs_render_review.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = build_run_bundle(
        date=args.date,
        cycle=args.cycle,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        outdir=Path(args.outdir),
        require_production_basemap=not bool(args.allow_fallback_publication),
        overwrite=bool(args.overwrite),
    )
    print(f"outdir={result['outdir']}")
    print(f"status_md={result['status_md']}")
    print(f"run_summary_md={result['run_summary_md']}")
    for error in result["errors"]:
        print(f"error={error}")


if __name__ == "__main__":
    main()
