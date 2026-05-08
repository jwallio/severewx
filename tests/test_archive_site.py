import json
from datetime import date, timedelta

from severewx.archive.site import build_archive_site
from severewx.config import load_settings
from severewx.utils.paths import build_paths


def _write_product(paths, init_date: str, cycle: str, day: int) -> tuple[object, object]:
    valid_date = date.fromisoformat(init_date) + timedelta(days=day - 1)
    product_dir = paths.outputs / "github_actions" / f"tornado_concern_product_{init_date}_{cycle}z"
    product_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tornado_concern_init_{init_date}_{cycle}z_valid_{valid_date.isoformat()}"
    product_image = product_dir / f"{stem}.png"
    product_image.write_bytes(f"product-image-{init_date}-{day}".encode("ascii"))
    product_summary = product_dir / f"{stem}.md"
    product_summary.write_text(f"# Day {day}", encoding="utf-8")
    product_metadata = product_dir / f"{stem}.json"
    product_metadata.write_text(
        json.dumps(
            {
                "title": f"Risk Outlook {init_date}",
                "date": init_date,
                "cycle": cycle,
                "valid_date": valid_date.isoformat(),
                "valid_period_label": f"{valid_date.isoformat()} 00-24 UTC",
                "map_style": "outlook",
                "map_domain": "conus",
                "artifact_source_requested": "consensus",
                "render_basemap_mode": "cartopy",
                "render_basemap_warning": "",
                "forecast_consensus_summary": {
                    "included_sources": ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent"],
                    "excluded_sources": [],
                },
                "publication_status": "needs_render_review",
                "public_ready": day != 2,
                "backtest_version": "tornado_environment_consensus_v2",
                "calibration_version": "regional_lead_day_v1",
                "readiness_status": "internal_review_only",
                "generation_timestamp": f"{init_date}T0{day}:00:00Z",
                "main_image_path": str(product_image),
                "summary_path": str(product_summary),
                "metadata_path": str(product_metadata),
            }
        ),
        encoding="utf-8",
    )
    return product_dir, product_image


def test_archive_site_builds_latest_run_day_dropdown_without_old_maps(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)

    paths.maps.mkdir(parents=True, exist_ok=True)
    old_forecast_map = paths.maps / "2026-05-03_00_day1_2026-05-03_any_severe.png"
    old_forecast_map.write_bytes(b"old-map")

    for day in (1, 2, 3):
        _write_product(paths, "2026-05-03", "00", day)
    latest_dir = None
    latest_image = None
    for day in (1, 2, 3):
        latest_dir, latest_image = _write_product(paths, "2026-05-05", "12", day)

    (paths.outputs / "forecast_consensus_metadata_2026-05-05_12.json").write_text(
        json.dumps(
            {
                "init_date": "2026-05-05",
                "ingest_summary": {"source": "forecast_consensus", "source_mode": "real", "available_fields": ["cape"]},
                "included_sources": ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent"],
                "source_models": {
                    "hrrr_recent": "HRRR",
                    "rap_recent": "RAP",
                    "nam_recent": "NAM",
                    "aws_recent": "GFS",
                },
                "lead_day_summary": [
                    {
                        "date": "2026-05-05",
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

    output = build_archive_site(paths)
    html_text = output.read_text(encoding="utf-8")

    assert "severewx Runs" in html_text
    assert "Latest Forecast Run" in html_text
    assert "2026-05-05 12Z Risk Outlook" in html_text
    assert "Day 1 Risk" in html_text
    assert "Day 2 Risk" in html_text
    assert "Day 3 Risk" in html_text
    assert "window.SEVEREWX_RUN" in html_text
    assert "hrrr_recent, rap_recent, nam_recent, aws_recent" in html_text
    assert "tornado_environment_consensus_v2" in html_text
    assert "regional_lead_day_v1" in html_text
    assert "internal_review_only" in html_text
    assert "\"artifactSource\": \"consensus\"" in html_text
    assert "\"mapDomain\": \"conus\"" in html_text
    assert "Forecast diagnostics" in html_text
    assert "2026-05-03 00Z Risk Outlook" not in html_text
    assert "Forecast Runs" not in html_text
    assert old_forecast_map.name not in html_text
    assert latest_dir is not None
    assert latest_image is not None
    assert (paths.archive / ".nojekyll").exists()
    assert (paths.archive / "assets" / "data" / "outputs" / "github_actions" / latest_dir.name / latest_image.name).exists()
    assert not (paths.archive / "assets" / "data" / "outputs" / "maps" / old_forecast_map.name).exists()


def test_archive_site_skips_latest_run_when_images_are_missing(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)

    for day in (1, 2, 3):
        _write_product(paths, "2026-05-05", "12", day)
    missing_dir, missing_image = _write_product(paths, "2026-05-06", "00", 1)
    missing_image.unlink()

    html_text = build_archive_site(paths).read_text(encoding="utf-8")

    assert missing_dir is not None
    assert "2026-05-05 12Z Risk Outlook" in html_text
    assert "2026-05-06 00Z Risk Outlook" not in html_text
    assert missing_image.name not in html_text


def test_archive_site_ignores_verification_products_for_public_viewer(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)

    for day in (1, 2, 3):
        _write_product(paths, "2026-05-05", "12", day)
    verification_dir = paths.verification / "pilot_products"
    verification_dir.mkdir(parents=True, exist_ok=True)
    for day in (1, 2, 3):
        product_dir, product_image = _write_product(paths, "2026-05-06", "00", day)
        target_image = verification_dir / product_image.name
        target_metadata = verification_dir / product_image.with_suffix(".json").name
        target_summary = verification_dir / product_image.with_suffix(".md").name
        target_image.write_bytes(product_image.read_bytes())
        payload = json.loads(product_image.with_suffix(".json").read_text(encoding="utf-8"))
        payload["main_image_path"] = str(target_image)
        payload["metadata_path"] = str(target_metadata)
        payload["summary_path"] = str(target_summary)
        target_metadata.write_text(json.dumps(payload), encoding="utf-8")
        target_summary.write_text("# verification", encoding="utf-8")
        for path in [product_image, product_image.with_suffix(".json"), product_image.with_suffix(".md")]:
            path.unlink()
        product_dir.rmdir()

    html_text = build_archive_site(paths).read_text(encoding="utf-8")

    assert "2026-05-05 12Z Risk Outlook" in html_text
    assert "2026-05-06 00Z Risk Outlook" not in html_text


def test_archive_site_warns_when_product_uses_fallback_basemap(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    _write_product(paths, "2026-05-05", "12", 1)
    _write_product(paths, "2026-05-05", "12", 2)
    _write_product(paths, "2026-05-05", "12", 3)
    metadata_file = next(paths.outputs.rglob("tornado_concern_init_2026-05-05_12z_valid_2026-05-06.json"))
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    payload["render_basemap_mode"] = "matplotlib_fallback"
    payload["render_basemap_warning"] = "cartopy_unavailable_or_disabled"
    metadata_file.write_text(json.dumps(payload), encoding="utf-8")

    html_text = build_archive_site(paths).read_text(encoding="utf-8")

    assert "Production CONUS basemap was not available for Day 2 Risk" in html_text


def test_archive_site_warns_when_latest_run_ingest_is_not_real(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    for day in (1, 2, 3):
        _write_product(paths, "2026-05-05", "12", day)
    (paths.outputs / "forecast_metadata_2026-05-05_12.json").write_text(
        json.dumps(
            {
                "init_date": "2026-05-05",
                "ingest_summary": {
                    "source": "synthetic_fallback",
                    "source_mode": "synthetic",
                    "real_ingest_available": False,
                    "available_fields": ["cape"],
                },
            }
        ),
        encoding="utf-8",
    )

    html_text = build_archive_site(paths).read_text(encoding="utf-8")

    assert "Non-real forecast ingest detected" in html_text
    assert "source=synthetic_fallback" in html_text
