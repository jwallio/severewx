import json

from severewx.archive.site import build_archive_site
from severewx.config import load_settings
from severewx.utils.paths import build_paths


def test_archive_site_builds_pages_ready_run_index(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)

    forecast_map = paths.maps / "2026-05-03_day1.png"
    forecast_map.write_bytes(b"forecast-map")
    review_graphic = paths.verification / "2026-05-03_00_day1_2026-05-03_case_review.png"
    review_graphic.write_bytes(b"review-graphic")
    (paths.verification / "2026-05-03_verification.json").write_text(
        json.dumps(
            {
                "run_summary": {
                    "init_date": "2026-05-03",
                    "n_valid_days": 4,
                    "training_data_source": "archive",
                    "evaluation_source": "reports",
                    "any_outbreak_hit_rate": 0.5,
                    "any_false_outbreak_alarms": 1,
                    "tornado_outbreak_hit_rate": 0.25,
                    "tornado_false_outbreak_alarms": 2,
                    "significant_tornado_outbreak_hit_rate": 0.0,
                    "significant_tornado_false_outbreak_alarms": 0,
                },
                "training_data_summary": {"source_preference": "archive", "guardrails_passed": True},
                "review_graphics": [str(review_graphic)],
            }
        ),
        encoding="utf-8",
    )
    (paths.outputs / "forecast_metadata_2026-05-03_00.json").write_text(
        json.dumps(
            {
                "init_date": "2026-05-03",
                "ingest_summary": {"source": "nomads", "source_mode": "real", "available_fields": ["cape"]},
                "lead_day_summary": [
                    {
                        "date": "2026-05-03",
                        "lead_day": 1,
                        "max_outbreak_risk": 0.4,
                        "mean_confidence": 0.7,
                        "mean_signal_quality": 0.6,
                        "mean_bust_risk": 0.2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle_dir = paths.outputs / "github_actions" / "run_bundle_2026-05-05_12z_to_2026-05-06_12z"
    direct_dir = bundle_dir / "direct_regional"
    direct_dir.mkdir(parents=True, exist_ok=True)
    direct_image = direct_dir / "direct.png"
    direct_image.write_bytes(b"direct-image")
    direct_metadata = direct_dir / "direct.json"
    direct_metadata.write_text(json.dumps({"publication_status": "public_candidate"}), encoding="utf-8")
    direct_summary = direct_dir / "direct.md"
    direct_summary.write_text("# direct", encoding="utf-8")
    (bundle_dir / "status.md").write_text("# status", encoding="utf-8")
    (bundle_dir / "run_summary.md").write_text("# run summary", encoding="utf-8")
    (bundle_dir / "environment.json").write_text(json.dumps({"cartopy_available": True}), encoding="utf-8")
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "date": "2026-05-03",
                "cycle": "00",
                "valid_start": "2026-05-05T12:00Z",
                "valid_end": "2026-05-06T12:00Z",
                "products": {
                    "direct_regional": {
                        "publication_status": "public_candidate",
                        "public_ready": True,
                        "image_path": str(direct_image),
                        "metadata_path": str(direct_metadata),
                        "summary_path": str(direct_summary),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output = build_archive_site(paths)
    html_text = output.read_text(encoding="utf-8")

    assert "severewx Runs" in html_text
    assert "Tornado-Concern Run Bundles" in html_text
    assert "run_bundle_2026-05-05_12z_to_2026-05-06_12z" in html_text
    assert "Forecast Runs" in html_text
    assert (paths.archive / ".nojekyll").exists()
    assert (paths.archive / "assets" / "data" / "outputs" / "maps" / forecast_map.name).exists()
    assert (paths.archive / "assets" / "data" / "outputs" / "github_actions" / bundle_dir.name / "direct_regional" / direct_image.name).exists()
