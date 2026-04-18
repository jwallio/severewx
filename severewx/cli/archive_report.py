"""CLI for archive coverage reporting."""

from __future__ import annotations

import argparse
import json

from severewx.archive.coverage import write_archive_coverage_summary
from severewx.config import load_settings
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize historical archive coverage")
    parser.parse_args()
    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)
    training_summary_file = paths.models / "training_data_summary.json"
    training_data_summary = json.loads(training_summary_file.read_text(encoding="utf-8")) if training_summary_file.exists() else None
    json_path, csv_path = write_archive_coverage_summary(paths, training_data_summary=training_data_summary, settings=settings)
    coverage = json.loads(json_path.read_text(encoding="utf-8"))
    logger.info("archive coverage summary written to %s", json_path)
    logger.info("archive lead-day coverage csv written to %s", csv_path)
    logger.info(
        "archive coverage real_fraction=%.3f cached_feature_rows=%s cached_feature_real_fraction=%.3f raw_modes=%s raw_retained=%s raw_moved=%s raw_deleted=%s quality_tier=%s statuses=%s",
        coverage.get("real_fraction_overall", 0.0),
        coverage.get("cached_feature_archive_row_count", 0),
        coverage.get("cached_feature_archive_real_fraction", 0.0),
        coverage.get("raw_lifecycle_mode_counts", {}),
        coverage.get("raw_files_retained", 0),
        coverage.get("raw_files_moved", 0),
        coverage.get("raw_files_deleted", 0),
        ((coverage.get("archive_guardrails") or {}).get("training_quality_tier", "unknown")),
        coverage.get("archive_status_reporting_counts", coverage.get("archive_status_counts", {})),
    )
    failures = (coverage.get("archive_guardrails") or {}).get("failures", [])
    if failures:
        logger.warning("archive guardrail failures: %s", failures)


if __name__ == "__main__":
    main()
