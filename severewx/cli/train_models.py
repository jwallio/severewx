"""CLI for model training."""

from __future__ import annotations

import argparse
import json

import pandas as pd
import xarray as xr

from severewx.archive.coverage import write_archive_coverage_summary
from severewx.config import load_settings
from severewx.features.analogs import save_analog_reference
from severewx.models.dataset import load_training_table
from severewx.models.outbreak import save_outbreak_model
from severewx.models.train import train_and_save_hazard_model
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def _latest_matching_file(directory, pattern: str):
    matches = list(directory.glob(pattern))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train severe-weather models")
    parser.add_argument("--hazards", nargs="+", required=True)
    parser.add_argument("--allow-degraded", action="store_true", help="Allow degraded archive-backed training when real coverage guardrails fail")
    args = parser.parse_args()
    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)

    label_file = _latest_matching_file(paths.labels, "labels_*.nc")
    outbreak_file = _latest_matching_file(paths.labels, "outbreaks_*.parquet")
    label_cube = xr.load_dataset(label_file) if label_file else None
    outbreak_table = pd.read_parquet(outbreak_file) if outbreak_file else None
    training_bundle = load_training_table(
        paths,
        settings,
        label_cube=label_cube,
        outbreak_table=outbreak_table,
        analog_archive_path=None,
        requested_hazards=args.hazards,
        allow_degraded=args.allow_degraded,
    )
    training_frame = training_bundle.frame
    if (
        settings.get("models.prefer_feature_archive", True)
        and not training_bundle.metadata.get("training_allowed_without_override", True)
        and not args.allow_degraded
    ):
        failure_reasons = training_bundle.metadata.get("guardrail_failures", [])
        if not failure_reasons:
            degraded_reason = str(training_bundle.metadata.get("degraded_mode_reason", "")).strip()
            source_preference = str(training_bundle.metadata.get("source_preference", "")).strip()
            if degraded_reason:
                failure_reasons = [degraded_reason]
            elif source_preference:
                failure_reasons = [f"training source {source_preference} is not eligible for strict archive-preferred training"]
            else:
                failure_reasons = ["training metadata did not satisfy strict archive-preferred training requirements"]
        raise RuntimeError(
            "archive-preferred training failed: "
            + "; ".join(failure_reasons)
            + " (rerun with --allow-degraded to override)"
        )
    training_summary_path = paths.models / "training_data_summary.json"
    training_summary_path.write_text(json.dumps(training_bundle.metadata, indent=2), encoding="utf-8")
    logger.info("training data source preference=%s rows=%d", training_bundle.metadata.get("source_preference"), len(training_frame))
    logger.info(
        "training quality tier=%s source=%s cache_used=%s real_rows=%s synthetic_rows=%s guardrails_passed=%s degraded_reason=%s",
        training_bundle.metadata.get("training_quality_tier"),
        training_bundle.metadata.get("source_preference"),
        training_bundle.metadata.get("training_cache_used"),
        training_bundle.metadata.get("real_archive_rows"),
        training_bundle.metadata.get("synthetic_rows"),
        training_bundle.metadata.get("guardrails_passed"),
        training_bundle.metadata.get("degraded_mode_reason"),
    )
    logger.info("saved training data summary to %s", training_summary_path)
    coverage_json, coverage_csv = write_archive_coverage_summary(paths, training_data_summary=training_bundle.metadata, settings=settings)
    logger.info("updated archive coverage summary %s", coverage_json)
    logger.info("updated archive lead-day coverage csv %s", coverage_csv)
    analog_reference_path = paths.models / "analog_reference.parquet"
    save_analog_reference(training_frame, analog_reference_path, settings)
    logger.info("saved analog reference to %s", analog_reference_path)

    for hazard in args.hazards:
        if hazard == "outbreak":
            artifact = save_outbreak_model(training_frame, settings, paths)
            logger.info("trained outbreak model with backend=%s", artifact["backend"])
            continue
        artifact = train_and_save_hazard_model(hazard, training_frame, settings, paths)
        logger.info("trained %s model with brier=%.4f roc_auc=%s", hazard, artifact["metrics"]["brier_score"], artifact["metrics"]["roc_auc"])


if __name__ == "__main__":
    main()
