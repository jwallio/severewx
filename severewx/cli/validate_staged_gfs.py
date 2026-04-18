"""CLI for validating staged local GFS files before archive build."""

from __future__ import annotations

import argparse
import json

from severewx.config import load_settings
from severewx.ingest.nomads import validate_local_staged_gfs_inventory
from severewx.utils.logging import configure_logging
from severewx.utils.paths import build_paths


def _report_path(paths, start: str, end: str, cycles: list[str]):
    cycle_part = "-".join(cycles)
    return paths.interim / f"staged_gfs_validation_{start}_{end}_{cycle_part}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate staged local GFS files before historical archive build")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end")
    parser.add_argument("--cycles", nargs="+", default=["00", "12"])
    args = parser.parse_args()

    logger = configure_logging()
    settings = load_settings()
    paths = build_paths(settings)
    end = args.end or args.start
    report = validate_local_staged_gfs_inventory(args.start, end, args.cycles, settings)
    report_path = _report_path(paths, args.start, end, args.cycles)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("staged GFS validation written to %s", report_path)
    logger.info(
        "staged GFS validation detected_dates=%s detected_cycles=%s detected_leads=%s statuses=%s",
        report.get("detected_dates", []),
        report.get("detected_cycles", []),
        report.get("detected_lead_hours", []),
        report.get("status_counts", {}),
    )
    for cycle_result in report.get("cycle_results", []):
        logger.info(
            "staged cycle date=%s cycle=%s status=%s usable_leads=%s missing_leads=%s usable_files=%d",
            cycle_result.get("date"),
            cycle_result.get("cycle"),
            cycle_result.get("status"),
            cycle_result.get("available_leads", []),
            cycle_result.get("missing_leads", []),
            len(cycle_result.get("usable_paths", [])),
        )


if __name__ == "__main__":
    main()
