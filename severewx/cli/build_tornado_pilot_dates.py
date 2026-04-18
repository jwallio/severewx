"""CLI for building a balanced pilot historical GFS staging list from tornado catalogs."""

from __future__ import annotations

import argparse

from severewx.config import load_settings
from severewx.labels.tornado_catalog import (
    DEFAULT_PILOT_CLASS_TARGETS,
    build_tornado_pilot_dates,
    load_tornado_summary_csv,
)
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced pilot tornado staging-date list")
    parser.add_argument("--summary", default="data/interim/tornado_daily_summary.csv")
    parser.add_argument("--candidates", default="data/interim/tornado_candidate_dates.csv")
    parser.add_argument("--significant-count", type=int, default=DEFAULT_PILOT_CLASS_TARGETS["significant_tornado_outbreak"])
    parser.add_argument("--outbreak-count", type=int, default=DEFAULT_PILOT_CLASS_TARGETS["tornado_outbreak"])
    parser.add_argument("--moderate-count", type=int, default=DEFAULT_PILOT_CLASS_TARGETS["moderate_tornado_day"])
    parser.add_argument("--low-count", type=int, default=DEFAULT_PILOT_CLASS_TARGETS["low_tornado_day"])
    parser.add_argument("--max-per-decade", type=int)
    parser.add_argument("--min-spacing-days", type=int, default=0)
    args = parser.parse_args()

    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)
    summary = load_tornado_summary_csv(args.summary)
    candidates = load_tornado_summary_csv(args.candidates)
    pilot = build_tornado_pilot_dates(
        summary,
        candidates,
        class_targets={
            "significant_tornado_outbreak": int(args.significant_count),
            "tornado_outbreak": int(args.outbreak_count),
            "moderate_tornado_day": int(args.moderate_count),
            "low_tornado_day": int(args.low_count),
        },
        max_per_decade=args.max_per_decade,
        min_spacing_days=int(args.min_spacing_days),
    )

    output_path = paths.interim / "tornado_pilot_dates.csv"
    pilot.to_csv(output_path, index=False)
    logger.info(
        "tornado pilot list written rows=%d class_counts=%s output=%s",
        len(pilot),
        pilot["outbreak_class"].value_counts().to_dict() if not pilot.empty else {},
        output_path,
    )


if __name__ == "__main__":
    main()
