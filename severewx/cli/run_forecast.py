"""CLI for Day 1-4 forecast generation."""

from __future__ import annotations

import argparse
import json

import xarray as xr

from severewx.config import load_settings
from severewx.ingest.nomads import ingest_forecast_cycle
from severewx.ingest.storage import forecast_path, load_forecast_dataset
from severewx.models.predict import predict_hazard_frame, prediction_frame_to_dataset
from severewx.render.maps import render_probability_maps
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def _find_prior_run(date: str, cycle: str, paths) -> xr.Dataset | None:
    cycles = ["00", "06", "12", "18"]
    try:
        index = cycles.index(cycle)
    except ValueError:
        return None
    if index == 0:
        return None
    prior_path = forecast_path(paths, date, cycles[index - 1])
    return xr.load_dataset(prior_path) if prior_path.exists() else None


def _metadata_path(paths, date: str, cycle: str):
    return paths.outputs / f"forecast_metadata_{date}_{cycle}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run severe-weather forecast")
    parser.add_argument("--date", required=True)
    parser.add_argument("--cycle", required=True)
    args = parser.parse_args()
    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)
    if not forecast_path(paths, args.date, args.cycle).exists():
        ingest_forecast_cycle(args.date, args.cycle, settings=settings)
    forecast = load_forecast_dataset(paths, args.date, args.cycle)
    prior_run = _find_prior_run(args.date, args.cycle, paths)
    analog_reference_path = paths.models / "analog_reference.parquet"
    prediction_frame = predict_hazard_frame(
        forecast,
        settings,
        paths,
        prior_run=prior_run,
        analog_archive_path=analog_reference_path if analog_reference_path.exists() else None,
    )
    prediction_ds = prediction_frame_to_dataset(prediction_frame)
    output_path = paths.outputs / f"forecast_products_{args.date}_{args.cycle}.nc"
    prediction_ds.to_netcdf(output_path)
    rendered = render_probability_maps(prediction_ds, args.date, args.cycle, settings, paths)
    ingest_summary_path = paths.interim / f"ingest_summary_{args.date}_{args.cycle}.json"
    metadata = {
        "init_date": args.date,
        "cycle": args.cycle,
        "ingest_summary": json.loads(ingest_summary_path.read_text(encoding="utf-8")) if ingest_summary_path.exists() else {},
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
    _metadata_path(paths, args.date, args.cycle).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("saved forecast products to %s", output_path)
    board_count = sum(path.name.endswith("_daily_board.png") for path in rendered)
    logger.info("rendered %d graphics (%d daily boards)", len(rendered), board_count)


if __name__ == "__main__":
    main()
