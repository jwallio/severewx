"""Static archive generation."""

from __future__ import annotations

import json
from pathlib import Path

from severewx.utils.paths import DataPaths


def _table_from_mapping(mapping: dict[str, object], include_keys: list[str] | None = None) -> str:
    keys = include_keys or list(mapping.keys())
    rows = ["<table>"]
    for key in keys:
        if key in mapping:
            rows.append(f"<tr><th>{key}</th><td>{mapping[key]}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def build_archive_site(paths: DataPaths) -> Path:
    map_files = sorted(paths.maps.glob("*.png"))
    verification_files = sorted(paths.verification.glob("*_verification.json"))
    metadata_files = sorted(paths.outputs.glob("forecast_metadata_*.json"))

    verification_index: dict[str, dict[str, object]] = {}
    for verification_file in verification_files:
        payload = json.loads(verification_file.read_text(encoding="utf-8"))
        verification_index[payload["run_summary"]["init_date"]] = payload

    metadata_index: dict[str, dict[str, object]] = {}
    for metadata_file in metadata_files:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata_index[payload["init_date"]] = payload

    rows: list[str] = [
        "<html><head><title>severewx archive</title><style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;line-height:1.35;}img{max-width:380px;margin:8px;border:1px solid #bbb;}section{margin-bottom:40px;}table{border-collapse:collapse;margin:8px 0 16px 0;}td,th{border:1px solid #ccc;padding:6px 10px;vertical-align:top;}h1,h2,h3{margin-bottom:8px;} .grid{display:flex;flex-wrap:wrap;} .block{margin-right:24px;}</style></head><body>",
        "<h1>severewx archive</h1>",
    ]
    by_date: dict[str, list[Path]] = {}
    for map_file in map_files:
        key = map_file.name.split("_")[0]
        by_date.setdefault(key, []).append(map_file)

    for date, files in sorted(by_date.items()):
        rows.append(f"<section><h2>{date}</h2>")
        verification = verification_index.get(date)
        metadata = metadata_index.get(date)
        if verification:
            summary = verification["run_summary"]
            rows.append("<div class='grid'>")
            rows.append("<div class='block'><h3>Verification Summary</h3>")
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
                rows.append("<div class='block'><h3>Training Coverage</h3>")
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
            if verification.get("tornado_outbreak_case_review"):
                rows.append("<div class='block'><h3>Tornado Case Review</h3><table>")
                rows.append("<tr><th>valid_date</th><th>lead_day</th><th>forecast_outbreak_prob</th><th>forecast_tornado_prob</th><th>observed_category</th></tr>")
                for case in verification["tornado_outbreak_case_review"][:5]:
                    rows.append(
                        "<tr>"
                        f"<td>{case.get('valid_date')}</td>"
                        f"<td>{case.get('lead_day')}</td>"
                        f"<td>{case.get('forecast_outbreak_prob')}</td>"
                        f"<td>{case.get('forecast_tornado_prob')}</td>"
                        f"<td>{case.get('observed_category')}</td>"
                        "</tr>"
                    )
                rows.append("</table></div>")
            rows.append("</div>")
            if verification.get("review_graphics"):
                rows.append("<h3>Case Review Boards</h3><div class='grid'>")
                for review_file in verification["review_graphics"]:
                    review_path = Path(review_file)
                    if not review_path.exists():
                        continue
                    relative = review_path.relative_to(paths.archive.parent).as_posix()
                    rows.append(f"<div><a href='../{relative}'><img src='../{relative}' alt='{review_path.name}'></a><div>{review_path.name}</div></div>")
                rows.append("</div>")
        if metadata:
            rows.append("<h3>Diagnostics</h3>")
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
                        f"<td>{lead_row.get('date')}</td>"
                        f"<td>{lead_row.get('lead_day')}</td>"
                        f"<td>{lead_row.get('max_outbreak_risk')}</td>"
                        f"<td>{lead_row.get('mean_confidence')}</td>"
                        f"<td>{lead_row.get('mean_signal_quality')}</td>"
                        f"<td>{lead_row.get('mean_bust_risk')}</td>"
                        "</tr>"
                    )
                rows.append("</table>")
        rows.append("<h3>Maps</h3><div class='grid'>")
        for file_path in files:
            relative = file_path.relative_to(paths.archive.parent).as_posix()
            rows.append(f"<div><a href='../{relative}'><img src='../{relative}' alt='{file_path.name}'></a><div>{file_path.name}</div></div>")
        rows.append("</div></section>")
    rows.append("</body></html>")
    output = paths.archive / "index.html"
    output.write_text("".join(rows), encoding="utf-8")
    return output
