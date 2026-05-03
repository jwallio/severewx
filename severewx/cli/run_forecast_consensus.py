"""CLI for multi-source forecast products and tornado-environment consensus."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import xarray as xr

from severewx.config import AppSettings, load_settings
from severewx.ingest.nomads import ingest_forecast_cycle
from severewx.ingest.storage import forecast_path, load_forecast_dataset
from severewx.models.forecast_consensus import (
    AUTO_CONSENSUS_SOURCES,
    CONSENSUS_INPUT_FIELD,
    ConsensusSource,
    build_forecast_consensus,
    consensus_metadata_path,
    consensus_product_path,
    source_metadata_path,
    source_product_path,
)
from severewx.models.predict import predict_hazard_frame, prediction_frame_to_dataset
from severewx.render.maps import render_probability_maps
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def _parse_sources(value: str) -> list[str]:
    if value.strip().lower() == "auto":
        return list(AUTO_CONSENSUS_SOURCES)
    return [source.strip().lower() for source in value.split(",") if source.strip()]


def _settings_for_source(settings: AppSettings, source: str) -> AppSettings:
    raw = copy.deepcopy(settings.raw)
    raw.setdefault("ingest", {})["source"] = source
    raw.setdefault("ingest", {})["failover_sources"] = []
    raw.setdefault("ingest", {})["allow_synthetic_fallback"] = False
    return AppSettings(raw=raw)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _source_available(metadata: dict[str, Any]) -> bool:
    ingest = metadata.get("ingest_summary", {}) if isinstance(metadata.get("ingest_summary", {}), dict) else {}
    return str(ingest.get("source_mode", "real")) != "synthetic"


def _run_source_product(date: str, cycle: str, source: str, settings: AppSettings, *, skip_render: bool) -> ConsensusSource | None:
    logger = configure_logging()
    source_settings = _settings_for_source(settings, source)
    paths = build_paths(source_settings)
    try:
        ingest_forecast_cycle(date, cycle, settings=source_settings)
        forecast = load_forecast_dataset(paths, date, cycle)
        analog_reference_path = paths.models / "analog_reference.parquet"
        prediction_frame = predict_hazard_frame(
            forecast,
            source_settings,
            paths,
            prior_run=None,
            analog_archive_path=analog_reference_path if analog_reference_path.exists() else None,
        )
        prediction_ds = prediction_frame_to_dataset(prediction_frame)
        output_path = source_product_path(paths.outputs, date, cycle, source)
        prediction_ds.to_netcdf(output_path)
        if not skip_render:
            render_probability_maps(prediction_ds, date, cycle, source_settings, paths)
        ingest_summary_path = paths.interim / f"ingest_summary_{date}_{cycle}.json"
        metadata = {
            "init_date": date,
            "cycle": cycle,
            "source": source,
            "ingest_summary": _read_json(ingest_summary_path),
            "source_forecast_path": str(forecast_path(paths, date, cycle)),
            "lead_day_summary": prediction_frame.groupby(["date", "lead_day"], as_index=False).agg(
                max_tornado_prob=("tornado_prob", "max"),
                max_hail_prob=("hail_prob", "max"),
                max_wind_prob=("wind_prob", "max"),
                max_any_prob=("any_prob", "max"),
                max_outbreak_risk=("outbreak_risk", "max"),
                mean_confidence=("confidence_score", "mean"),
                mean_signal_quality=("signal_quality_score", "mean"),
                mean_bust_risk=("bust_risk_score", "mean"),
                analog_outbreak_support=("analog_outbreak_support", "mean"),
            ).to_dict(orient="records"),
        }
        metadata_path = source_metadata_path(paths.outputs, date, cycle, source)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        forecast.close()
        prediction_ds.close()
        if not _source_available(metadata):
            logger.warning("excluding synthetic source product from consensus source=%s", source)
            return None
        logger.info("saved source forecast products source=%s path=%s", source, output_path)
        return ConsensusSource(source, output_path, metadata_path)
    except Exception as exc:
        logger.warning("source forecast failed source=%s date=%s cycle=%s error=%s", source, date, cycle, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-source severe-weather forecast consensus")
    parser.add_argument("--date", required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--sources", default="auto", help="auto or comma-delimited source list")
    parser.add_argument("--skip-render", action="store_true", help="Skip per-source map rendering")
    args = parser.parse_args()

    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)
    sources = _parse_sources(args.sources)
    source_artifacts: list[ConsensusSource] = []
    unavailable_sources: list[dict[str, str]] = []
    for source in sources:
        artifact = _run_source_product(args.date, args.cycle, source, settings, skip_render=bool(args.skip_render))
        if artifact is None:
            unavailable_sources.append({"source": source, "reason": "source_run_failed_or_unavailable"})
            continue
        source_artifacts.append(artifact)
    if not source_artifacts:
        raise RuntimeError(f"no source forecast products were available for consensus: {sources}")
    output_path, metadata_path = build_forecast_consensus(
        date=args.date,
        cycle=args.cycle,
        sources=source_artifacts,
        output_path=consensus_product_path(paths.outputs, args.date, args.cycle),
        metadata_path=consensus_metadata_path(paths.outputs, args.date, args.cycle),
        field_name=CONSENSUS_INPUT_FIELD,
        unavailable_sources=unavailable_sources,
    )
    logger.info("saved forecast consensus to %s metadata=%s", output_path, metadata_path)


if __name__ == "__main__":
    main()
