from __future__ import annotations

import argparse
from pathlib import Path

from severewx.cli.tornado_concern_eval import (
    PRIORITY_TIERS,
    build_tornado_concern_candidate_rows,
    summarize_tornado_concern_coverage,
    write_tornado_concern_candidate_csv,
    write_tornado_concern_coverage_markdown,
)
from severewx.config import load_settings
from severewx.utils.paths import build_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tornado-concern historical coverage and readiness from local artifacts")
    parser.add_argument("--start", help="Inclusive start date filter YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive end date filter YYYY-MM-DD")
    parser.add_argument("--priority-tier", choices=PRIORITY_TIERS, default="all", help="Candidate priority tier filter")
    parser.add_argument("--output-csv", help="Optional path for a per-date readiness audit CSV")
    parser.add_argument("--output-md", help="Optional path for a markdown readiness summary")
    args = parser.parse_args()

    settings = load_settings()
    paths = build_paths(settings)
    frame = build_tornado_concern_candidate_rows(
        paths.outputs,
        paths.verification,
        labels_dir=paths.labels,
        interim_dir=paths.interim,
        start=args.start,
        end=args.end,
        priority_tier=args.priority_tier,
    )
    if args.output_csv:
        write_tornado_concern_candidate_csv(Path(args.output_csv), frame.sort_values(["date"], kind="mergesort"))
    if args.output_md:
        write_tornado_concern_coverage_markdown(
            Path(args.output_md),
            frame.sort_values(["date"], kind="mergesort"),
            start=args.start,
            end=args.end,
            priority_tier=args.priority_tier,
        )

    summary = summarize_tornado_concern_coverage(frame)
    for key in [
        "candidate_dates_total",
        "sig_tor_candidates_total",
        "outbreak_candidates_total",
        "all_tornado_candidates_total",
        "real_ingest_confirmed_total",
        "forecast_artifacts_already_present",
        "verification_artifacts_already_present",
        "fully_evaluable_dates_total",
        "failed_dates_total",
    ]:
        print(f"{key}={summary[key]}")
    print(f"failed_dates_examples_with_reason={','.join(summary['failed_dates_examples_with_reason'])}")


if __name__ == "__main__":
    main()
