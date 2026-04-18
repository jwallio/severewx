"""CLI for building a daily tornado event catalog from tor.json."""

from __future__ import annotations

import argparse

from severewx.config import load_settings
from severewx.labels.tornado_catalog import (
    TornadoCatalogThresholds,
    build_daily_tornado_summary,
    build_tornado_candidate_dates,
    load_tornado_features,
    thresholds_dict,
)
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a daily tornado event catalog from tor.json")
    parser.add_argument("--input", default="data/tor.json")
    parser.add_argument("--significant-tornado-outbreak-count", type=int, default=3)
    parser.add_argument("--significant-tornado-outbreak-tornado-count", type=int, default=10)
    parser.add_argument("--tornado-outbreak-count", type=int, default=8)
    parser.add_argument("--tornado-outbreak-significant-count", type=int, default=2)
    parser.add_argument("--moderate-tornado-day-count", type=int, default=4)
    parser.add_argument("--moderate-tornado-day-significant-count", type=int, default=1)
    args = parser.parse_args()

    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)
    thresholds = TornadoCatalogThresholds(
        significant_tornado_outbreak_count=args.significant_tornado_outbreak_count,
        significant_tornado_outbreak_tornado_count=args.significant_tornado_outbreak_tornado_count,
        tornado_outbreak_count=args.tornado_outbreak_count,
        tornado_outbreak_significant_count=args.tornado_outbreak_significant_count,
        moderate_tornado_day_count=args.moderate_tornado_day_count,
        moderate_tornado_day_significant_count=args.moderate_tornado_day_significant_count,
    )

    features = load_tornado_features(args.input)
    summary = build_daily_tornado_summary(features, thresholds)
    candidates = build_tornado_candidate_dates(summary)

    summary_path = paths.interim / "tornado_daily_summary.csv"
    candidate_path = paths.interim / "tornado_candidate_dates.csv"
    summary.to_csv(summary_path, index=False)
    candidates.to_csv(candidate_path, index=False)

    logger.info(
        "tornado catalog written summary_rows=%d candidate_rows=%d summary=%s candidates=%s thresholds=%s",
        len(summary),
        len(candidates),
        summary_path,
        candidate_path,
        thresholds_dict(thresholds),
    )


if __name__ == "__main__":
    main()
