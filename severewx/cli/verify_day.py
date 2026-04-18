"""CLI for daily verification."""

from __future__ import annotations

import argparse
import json

import xarray as xr
import pandas as pd

from severewx.config import load_settings
from severewx.render.review import render_case_review_boards
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths
from severewx.verify.daily import verify_daily_probabilities


def _latest_prediction_for_date(paths, date: str):
    matches = sorted(paths.outputs.glob(f"forecast_products_{date}_*.nc"))
    return matches[-1] if matches else None


def _latest_label_cube(paths):
    matches = sorted(paths.labels.glob("labels_*.nc"))
    return matches[-1] if matches else None


def _latest_outbreak_table(paths):
    matches = sorted(paths.labels.glob("outbreaks_*.parquet"))
    return matches[-1] if matches else None


def _latest_metadata_for_date(paths, date: str):
    matches = sorted(paths.outputs.glob(f"forecast_metadata_{date}_*.json"))
    return matches[-1] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a completed day")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)
    prediction_file = _latest_prediction_for_date(paths, args.date)
    label_file = _latest_label_cube(paths)
    outbreak_file = _latest_outbreak_table(paths)
    metadata_file = _latest_metadata_for_date(paths, args.date)
    training_summary_file = paths.models / "training_data_summary.json"
    if prediction_file is None or label_file is None:
        raise FileNotFoundError("verification requires a forecast_products file and a labels file")
    prediction_ds = xr.load_dataset(prediction_file)
    label_ds = xr.load_dataset(label_file)
    outbreak_table = pd.read_parquet(outbreak_file) if outbreak_file else None
    evaluation_metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file and metadata_file.exists() else {}
    training_data_summary = json.loads(training_summary_file.read_text(encoding="utf-8")) if training_summary_file.exists() else {}
    output = paths.verification / f"{args.date}_verification.json"
    verify_daily_probabilities(
        prediction_ds,
        label_ds,
        output,
        target_date=args.date,
        outbreak_table=outbreak_table,
        training_data_summary=training_data_summary,
        evaluation_metadata=evaluation_metadata,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    cycle = str(evaluation_metadata.get("cycle") or prediction_file.stem.split("_")[-1])
    review_outputs = render_case_review_boards(
        prediction_ds,
        label_ds,
        payload,
        args.date,
        cycle,
        settings,
        paths,
    )
    payload["review_graphics"] = [str(path) for path in review_outputs]
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("saved verification to %s", output)
    logger.info("rendered %d case-review boards", len(review_outputs))


if __name__ == "__main__":
    main()
