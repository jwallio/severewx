"""Static archive generation."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

from severewx.utils.paths import DataPaths


def _escape(value: object) -> str:
    return html.escape(str(value))


def _table_from_mapping(mapping: dict[str, object], include_keys: list[str] | None = None) -> str:
    keys = include_keys or list(mapping.keys())
    rows = ["<table>"]
    for key in keys:
        if key in mapping:
            rows.append(f"<tr><th>{_escape(key)}</th><td>{_escape(mapping[key])}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def _copy_publish_asset(paths: DataPaths, source: str | Path) -> str | None:
    source_path = Path(source)
    if not source_path.exists():
        return None
    try:
        relative = source_path.resolve().relative_to(paths.root.resolve())
    except ValueError:
        relative = Path(source_path.name)
    destination = paths.archive / "assets" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination.relative_to(paths.archive).as_posix()


def _metadata_index(paths: DataPaths) -> dict[str, dict[str, Any]]:
    metadata_files = sorted(paths.outputs.glob("forecast_metadata_*.json"))
    index: dict[str, dict[str, Any]] = {}
    for metadata_file in metadata_files:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        index[str(payload["init_date"])] = payload
    return index


def _verification_index(paths: DataPaths) -> dict[str, dict[str, Any]]:
    verification_files = sorted(paths.verification.glob("*_verification.json"))
    index: dict[str, dict[str, Any]] = {}
    for verification_file in verification_files:
        payload = json.loads(verification_file.read_text(encoding="utf-8"))
        index[str(payload["run_summary"]["init_date"])] = payload
    return index


def _append_forecast_runs(rows: list[str], paths: DataPaths) -> None:
    map_files = sorted(paths.maps.glob("*.png"))
    verification_index = _verification_index(paths)
    metadata_index = _metadata_index(paths)
    by_date: dict[str, list[Path]] = {}
    for map_file in map_files:
        key = map_file.name.split("_")[0]
        by_date.setdefault(key, []).append(map_file)

    rows.append("<section><h2>Forecast Runs</h2>")
    rows.append("<p class='section-copy'>Daily forecast maps, verification summaries, and case-review boards from the repo outputs.</p>")
    for date, files in sorted(by_date.items(), reverse=True):
        rows.append(f"<article class='run-card'><div class='run-header'><h3>{_escape(date)}</h3></div>")
        verification = verification_index.get(date)
        metadata = metadata_index.get(date)
        if verification:
            summary = verification["run_summary"]
            rows.append("<div class='info-grid'>")
            rows.append("<div><h4>Verification Summary</h4>")
            rows.append(
                _table_from_mapping(
                    summary,
                    include_keys=[
                        "init_date",
                        "n_valid_days",
                        "training_data_source",
                        "evaluation_source",
                        "any_outbreak_hit_rate",
                        "any_false_outbreak_alarms",
                        "tornado_outbreak_hit_rate",
                        "tornado_false_outbreak_alarms",
                        "significant_tornado_outbreak_hit_rate",
                        "significant_tornado_false_outbreak_alarms",
                    ],
                )
            )
            rows.append("</div>")
            if verification.get("training_data_summary"):
                training = verification["training_data_summary"]
                rows.append("<div><h4>Training Coverage</h4>")
                rows.append(
                    _table_from_mapping(
                        {
                            "source_preference": training.get("source_preference"),
                            "training_quality_tier": training.get("training_quality_tier"),
                            "guardrails_passed": training.get("guardrails_passed"),
                            "degraded_mode_reason": training.get("degraded_mode_reason"),
                            "row_count": training.get("row_count"),
                            "date_count": training.get("date_count"),
                            "real_archive_rows": training.get("real_archive_rows"),
                            "synthetic_rows": training.get("synthetic_rows"),
                            "real_fraction_overall": training.get("real_fraction_overall"),
                        }
                    )
                )
                rows.append("</div>")
            rows.append("</div>")
            if verification.get("tornado_outbreak_case_review"):
                rows.append("<h4>Tornado Case Review</h4><table>")
                rows.append("<tr><th>valid_date</th><th>lead_day</th><th>forecast_outbreak_prob</th><th>forecast_tornado_prob</th><th>observed_category</th></tr>")
                for case in verification["tornado_outbreak_case_review"][:5]:
                    rows.append(
                        "<tr>"
                        f"<td>{_escape(case.get('valid_date'))}</td>"
                        f"<td>{_escape(case.get('lead_day'))}</td>"
                        f"<td>{_escape(case.get('forecast_outbreak_prob'))}</td>"
                        f"<td>{_escape(case.get('forecast_tornado_prob'))}</td>"
                        f"<td>{_escape(case.get('observed_category'))}</td>"
                        "</tr>"
                    )
                rows.append("</table>")
            if verification.get("review_graphics"):
                rows.append("<h4>Case Review Boards</h4><div class='thumb-grid'>")
                for review_file in verification["review_graphics"]:
                    asset = _copy_publish_asset(paths, review_file)
                    if not asset:
                        continue
                    name = Path(review_file).name
                    rows.append(f"<figure><a href='{asset}'><img src='{asset}' alt='{_escape(name)}'></a><figcaption>{_escape(name)}</figcaption></figure>")
                rows.append("</div>")
        if metadata:
            rows.append("<h4>Diagnostics</h4>")
            ingest = metadata.get("ingest_summary", {})
            rows.append(
                _table_from_mapping(
                    {
                        "source": ingest.get("source"),
                        "source_mode": ingest.get("source_mode"),
                        "fields": ", ".join(ingest.get("available_fields", [])),
                        "missing_requested_leads": ingest.get("missing_requested_leads", []),
                        "fallbacks_used": ingest.get("fallbacks_used", {}),
                        "cache_overview": ingest.get("cache_overview", {}),
                        "provider_failures": ingest.get("provider_failures", []),
                    }
                )
            )
            lead_rows = metadata.get("lead_day_summary", [])
            if lead_rows:
                rows.append("<table><tr><th>valid_date</th><th>lead_day</th><th>outbreak</th><th>confidence</th><th>signal_quality</th><th>bust_risk</th></tr>")
                for lead_row in lead_rows[:8]:
                    rows.append(
                        "<tr>"
                        f"<td>{_escape(lead_row.get('date'))}</td>"
                        f"<td>{_escape(lead_row.get('lead_day'))}</td>"
                        f"<td>{_escape(lead_row.get('max_outbreak_risk'))}</td>"
                        f"<td>{_escape(lead_row.get('mean_confidence'))}</td>"
                        f"<td>{_escape(lead_row.get('mean_signal_quality'))}</td>"
                        f"<td>{_escape(lead_row.get('mean_bust_risk'))}</td>"
                        "</tr>"
                    )
                rows.append("</table>")
        rows.append("<h4>Maps</h4><div class='thumb-grid'>")
        for file_path in files:
            asset = _copy_publish_asset(paths, file_path)
            if not asset:
                continue
            rows.append(f"<figure><a href='{asset}'><img src='{asset}' alt='{_escape(file_path.name)}'></a><figcaption>{_escape(file_path.name)}</figcaption></figure>")
        rows.append("</div></article>")
    rows.append("</section>")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _run_bundle_cards(paths: DataPaths) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for manifest_path in sorted(paths.outputs.rglob("manifest.json"), reverse=True):
        if paths.archive in manifest_path.parents:
            continue
        manifest = _read_json(manifest_path)
        if not manifest:
            continue
        products = manifest.get("products", {})
        if not isinstance(products, dict):
            products = {}
        status_md = manifest_path.parent / "status.md"
        run_summary_md = manifest_path.parent / "run_summary.md"
        environment_json = manifest_path.parent / "environment.json"
        cards.append(
            {
                "name": manifest_path.parent.name,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "products": products,
                "status_md": status_md if status_md.exists() else None,
                "run_summary_md": run_summary_md if run_summary_md.exists() else None,
                "environment_json": environment_json if environment_json.exists() else None,
            }
        )
    return cards


def _append_run_bundles(rows: list[str], paths: DataPaths) -> None:
    bundle_cards = _run_bundle_cards(paths)
    rows.append("<section><h2>Tornado-Concern Run Bundles</h2>")
    rows.append("<p class='section-copy'>Manual bundle runs with direct regional, direct CONUS, and consensus CONUS products plus readiness diagnostics.</p>")
    if not bundle_cards:
        rows.append("<p>No run bundles found under data/outputs.</p></section>")
        return
    for card in bundle_cards:
        manifest = card["manifest"]
        products = card["products"]
        rows.append("<article class='run-card'>")
        rows.append(
            "<div class='run-header'>"
            f"<h3>{_escape(card['name'])}</h3>"
            f"<div class='run-meta'>init <strong>{_escape(manifest.get('date', ''))}</strong> cycle <strong>{_escape(manifest.get('cycle', ''))}Z</strong></div>"
            f"<div class='run-meta'>valid <strong>{_escape(manifest.get('valid_start', ''))}</strong> to <strong>{_escape(manifest.get('valid_end', ''))}</strong></div>"
            "</div>"
        )
        rows.append("<div class='link-row'>")
        for key in ["status_md", "run_summary_md", "environment_json"]:
            path = card.get(key)
            if not path:
                continue
            asset = _copy_publish_asset(paths, path)
            if asset:
                rows.append(f"<a href='{asset}'>{_escape(Path(path).name)}</a>")
        manifest_asset = _copy_publish_asset(paths, card["manifest_path"])
        if manifest_asset:
            rows.append(f"<a href='{manifest_asset}'>manifest.json</a>")
        rows.append("</div>")
        rows.append("<div class='product-grid'>")
        for product_name, payload in products.items():
            if not isinstance(payload, dict):
                continue
            image_asset = _copy_publish_asset(paths, payload.get("image_path", ""))
            metadata_asset = _copy_publish_asset(paths, payload.get("metadata_path", ""))
            summary_asset = _copy_publish_asset(paths, payload.get("summary_path", ""))
            rows.append("<section class='product-card'>")
            rows.append(
                f"<h4>{_escape(product_name)}</h4>"
                f"<p class='product-status'>publication <strong>{_escape(payload.get('publication_status', ''))}</strong> | public_ready <strong>{_escape(payload.get('public_ready', ''))}</strong></p>"
            )
            if image_asset:
                rows.append(f"<a href='{image_asset}'><img src='{image_asset}' alt='{_escape(product_name)}'></a>")
            rows.append("<div class='link-row'>")
            if metadata_asset:
                rows.append(f"<a href='{metadata_asset}'>metadata</a>")
            if summary_asset:
                rows.append(f"<a href='{summary_asset}'>summary</a>")
            rows.append("</div>")
            rows.append("</section>")
        rows.append("</div></article>")
    rows.append("</section>")


def build_archive_site(paths: DataPaths) -> Path:
    if paths.archive.exists():
        shutil.rmtree(paths.archive)
    paths.archive.mkdir(parents=True, exist_ok=True)
    (paths.archive / ".nojekyll").write_text("", encoding="utf-8")

    rows: list[str] = [
        "<html><head><title>severewx runs</title><style>"
        "body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f6f8;color:#18212b;line-height:1.45;}"
        "main{max-width:1280px;margin:0 auto;padding:24px;}"
        "header{padding:24px 0 8px 0;border-bottom:1px solid #d8e0e6;margin-bottom:24px;}"
        "h1,h2,h3,h4{margin:0 0 10px 0;}"
        "p{margin:0 0 12px 0;}"
        "section{margin-bottom:32px;}"
        ".section-copy{color:#4a5b6d;max-width:900px;}"
        ".run-card{background:#fff;border:1px solid #d8e0e6;border-radius:8px;padding:18px;margin:0 0 20px 0;}"
        ".run-header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between;margin-bottom:12px;}"
        ".run-meta{color:#4a5b6d;font-size:14px;}"
        ".info-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;}"
        ".thumb-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;}"
        ".product-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;}"
        ".product-card{border:1px solid #d8e0e6;border-radius:8px;padding:12px;background:#fbfcfd;}"
        ".product-status{font-size:14px;color:#4a5b6d;}"
        ".link-row{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 12px 0;}"
        "a{color:#0b63c8;text-decoration:none;}a:hover{text-decoration:underline;}"
        "table{border-collapse:collapse;width:100%;margin:8px 0 16px 0;font-size:14px;}"
        "td,th{border:1px solid #d8e0e6;padding:6px 8px;vertical-align:top;text-align:left;}"
        "img{display:block;width:100%;height:auto;border:1px solid #d8e0e6;border-radius:6px;background:#fff;}"
        "figure{margin:0;}figcaption{font-size:13px;color:#4a5b6d;margin-top:6px;word-break:break-word;}"
        "</style></head><body><main>",
        "<header><h1>severewx Runs</h1><p class='section-copy'>Static run browser for forecast outputs, tornado-concern bundles, diagnostics, and review graphics published from the repository data tree.</p></header>",
    ]
    _append_run_bundles(rows, paths)
    _append_forecast_runs(rows, paths)
    rows.append("</main></body></html>")
    output = paths.archive / "index.html"
    output.write_text("".join(rows), encoding="utf-8")
    return output
