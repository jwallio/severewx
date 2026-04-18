"""CLI for SPC labels and outbreak labels."""

from __future__ import annotations

import argparse

from severewx.config import load_settings
from severewx.labels.grid import build_label_cube
from severewx.labels.outbreaks import build_outbreak_table
from severewx.labels.spc_reports import load_spc_reports
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gridded severe labels")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--reports-file")
    args = parser.parse_args()
    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)
    report_table = load_spc_reports(args.reports_file, args.start, args.end)
    reports = report_table.frame
    label_cube = build_label_cube(reports, args.start, args.end, settings)
    outbreak_table = build_outbreak_table(reports, args.start, args.end, settings)
    label_path = paths.labels / f"labels_{args.start}_{args.end}.nc"
    outbreak_path = paths.labels / f"outbreaks_{args.start}_{args.end}.parquet"
    report_path = paths.labels / f"spc_reports_{args.start}_{args.end}.parquet"
    label_cube.to_netcdf(label_path)
    outbreak_table.to_parquet(outbreak_path, index=False)
    reports.to_parquet(report_path, index=False)
    logger.info("saved labels to %s", label_path)
    logger.info("saved outbreak labels to %s", outbreak_path)
    logger.info("SPC report source=%s real=%s", report_table.source_name, report_table.is_real_source)


if __name__ == "__main__":
    main()
