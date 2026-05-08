"""CLI for staging NOAA forecast files into local per-source layouts."""

from __future__ import annotations

import argparse

from severewx.config import load_settings
from severewx.ingest.stage_gfs import NOAA_GRIB_SOURCE_SPECS, OPEN_METEO_SOURCE_SPECS, stage_historical_gfs, staging_defaults_from_settings
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and stage NOAA forecast GRIB2 files for ingest/backfill")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end")
    parser.add_argument("--cycles", nargs="+", default=["00", "12"])
    parser.add_argument("--leads", nargs="+", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--source-strategy", choices=["auto", "open_meteo_recent", "openmeteo_recent", *sorted(OPEN_METEO_SOURCE_SPECS), *sorted(NOAA_GRIB_SOURCE_SPECS)], default="auto")
    parser.add_argument("--recent-window-days", type=int, default=45)
    parser.add_argument("--url-template", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--retries", type=int)
    parser.add_argument("--backoff-seconds", type=int)
    args = parser.parse_args()

    logger = configure_logging()
    settings = load_settings()
    defaults = staging_defaults_from_settings(settings)
    paths = build_paths(settings)
    if args.output_root:
        output_root = args.output_root
    elif args.source_strategy in {"auto", "ncei_historical", "aws_recent", "open_meteo_recent", "openmeteo_recent"}:
        output_root = paths.raw / "staged_gfs"
    elif args.source_strategy in OPEN_METEO_SOURCE_SPECS:
        output_root = paths.raw / "staged_forecasts" / args.source_strategy
    else:
        output_root = paths.raw / "staged_forecasts" / args.source_strategy
    end = args.end or args.start
    leads = args.leads or defaults["leads"]
    report = stage_historical_gfs(
        args.start,
        end,
        [str(value) for value in args.cycles],
        [int(value) for value in leads],
        output_root,
        paths,
        settings=settings,
        url_templates=list(args.url_template) or None,
        source_strategy=args.source_strategy,
        recent_window_days=int(args.recent_window_days),
        force=bool(args.force),
        timeout=int(args.timeout or defaults["timeout"]),
        retries=int(args.retries or defaults["retries"]),
        backoff_seconds=int(args.backoff_seconds or defaults["backoff_seconds"]),
    )

    logger.info(
        "forecast stage complete source=%s attempted=%d successful=%d skipped=%d failed=%d",
        report["source_strategy"],
        report["attempted_downloads"],
        report["successful_downloads"],
        report["skipped_existing_files"],
        report["failed_downloads"],
    )
    logger.info(
        "forecast stage by source successful=%s failed=%s",
        report.get("successful_downloads_by_source", {}),
        report.get("failed_downloads_by_source", {}),
    )
    if report["failed_downloads"]:
        logger.warning("forecast staging failures were reported; inspect %s", report["report_path"])


if __name__ == "__main__":
    main()
