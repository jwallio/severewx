from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from severewx.cli.tornado_concern_eval import (
    audit_real_case_init_dates,
    build_tornado_concern_candidate_rows,
    discover_real_tornado_relevant_candidate_dates,
    missing_real_case_forecast_artifacts,
    real_case_artifact_readiness,
    select_real_case_init_dates,
    summarize_real_case_backfill,
    summarize_tornado_concern_coverage,
    PRIORITY_TIERS,
    write_tornado_concern_candidate_csv,
    write_tornado_concern_coverage_markdown,
    write_init_dates_file,
)
from severewx.config import load_settings
from severewx.utils.paths import build_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local tornado-concern candidate and ready dates from existing artifacts")
    parser.add_argument("--output", required=True, help="Path to write one init date per line")
    parser.add_argument("--ready-output", help="Optional path to write only fully evaluable init dates after backfill")
    parser.add_argument("--ranked-candidates-csv", help="Optional path for a ranked candidate-universe CSV")
    parser.add_argument("--audit-csv", help="Optional path for a per-date readiness audit CSV")
    parser.add_argument("--start", help="Inclusive start date filter YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive end date filter YYYY-MM-DD")
    parser.add_argument("--priority-tier", choices=PRIORITY_TIERS, default="all", help="Candidate priority tier filter")
    parser.add_argument("--audit", action="store_true", help="Print compact stage-by-stage audit counts")
    parser.add_argument("--audit-missing-artifacts", action="store_true", help="Print missing forecast artifacts for real tornado-relevant init dates")
    parser.add_argument("--backfill", action="store_true", help="Backfill missing forecast/verification artifacts using existing CLI workflows")
    args = parser.parse_args()

    settings = load_settings()
    paths = build_paths(settings)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_rows = build_tornado_concern_candidate_rows(
        paths.outputs,
        paths.verification,
        labels_dir=paths.labels,
        interim_dir=paths.interim,
        start=args.start,
        end=args.end,
        priority_tier=args.priority_tier,
    )
    if args.backfill:
        candidate_dates = candidate_rows.loc[candidate_rows["real_ingest_confirmed"], "date"].astype(str).tolist() if not candidate_rows.empty else []
        write_init_dates_file(output_path, candidate_dates)
        working_rows = candidate_rows.copy()
        working_rows["forecast_artifacts_present_initial"] = working_rows["forecast_artifacts_present"]
        working_rows["verification_artifacts_present_initial"] = working_rows["verification_artifacts_present"]
        for index, row in working_rows.iterrows():
            if not bool(row["real_ingest_confirmed"]):
                continue
            if not bool(row["forecast_artifacts_present"]):
                working_rows.at[index, "forecast_attempted"] = True
                command = ["python", "-m", "severewx.cli.run_forecast", "--date", row["date"], "--cycle", "00"]
                result = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
                refreshed = build_tornado_concern_candidate_rows(
                    paths.outputs,
                    paths.verification,
                    labels_dir=paths.labels,
                    interim_dir=paths.interim,
                    start=row["date"],
                    end=row["date"],
                    priority_tier="all",
                )
                refreshed_present = bool(not refreshed.empty and bool(refreshed.iloc[0]["forecast_artifacts_present"]))
                working_rows.at[index, "forecast_artifacts_present"] = refreshed_present
                working_rows.at[index, "forecast_generated"] = bool(result.returncode == 0 and refreshed_present)
                if not refreshed_present:
                    working_rows.at[index, "failure_detail"] = result.stderr.strip() or result.stdout.strip() or str(result.returncode)
                    continue
            if not bool(row["verification_artifacts_present"]):
                working_rows.at[index, "verification_attempted"] = True
                command = ["python", "-m", "severewx.cli.verify_day", "--date", row["date"]]
                result = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
                refreshed = build_tornado_concern_candidate_rows(
                    paths.outputs,
                    paths.verification,
                    labels_dir=paths.labels,
                    interim_dir=paths.interim,
                    start=row["date"],
                    end=row["date"],
                    priority_tier="all",
                )
                refreshed_forecast = bool(not refreshed.empty and bool(refreshed.iloc[0]["forecast_artifacts_present"]))
                refreshed_verification = bool(not refreshed.empty and bool(refreshed.iloc[0]["verification_artifacts_present"]))
                working_rows.at[index, "forecast_artifacts_present"] = refreshed_forecast
                working_rows.at[index, "verification_artifacts_present"] = refreshed_verification
                working_rows.at[index, "verification_generated"] = bool(result.returncode == 0 and refreshed_verification)
                if not refreshed_verification:
                    working_rows.at[index, "failure_detail"] = result.stderr.strip() or result.stdout.strip() or str(result.returncode)
                    continue
        working_rows["final_ready"] = (
            working_rows["real_ingest_confirmed"].fillna(False)
            & working_rows["forecast_artifacts_present"].fillna(False)
            & working_rows["verification_artifacts_present"].fillna(False)
        )
        for index, row in working_rows.iterrows():
            failure_reason = row["failure_reason"]
            refreshed_row = dict(row)
            refreshed_row["final_ready"] = bool(row["final_ready"])
            refreshed_row["forecast_artifacts_present"] = bool(row["forecast_artifacts_present"])
            refreshed_row["verification_artifacts_present"] = bool(row["verification_artifacts_present"])
            from severewx.cli.tornado_concern_eval import _derive_failure_reason  # local import to keep script thin
            failure_reason = _derive_failure_reason(refreshed_row)
            working_rows.at[index, "failure_reason"] = failure_reason
        ready_dates = working_rows.loc[working_rows["final_ready"], "date"].astype(str).tolist()
        ready_output_path = Path(args.ready_output) if args.ready_output else output_path
        write_init_dates_file(ready_output_path, ready_dates)
        print(f"wrote {len(candidate_dates)} candidate init dates to {output_path}")
        print(f"wrote {len(ready_dates)} ready init dates to {ready_output_path}")
        if args.ranked_candidates_csv:
            write_tornado_concern_candidate_csv(Path(args.ranked_candidates_csv), working_rows)
        if args.audit_csv:
            write_tornado_concern_candidate_csv(Path(args.audit_csv), working_rows.sort_values(["date"], kind="mergesort"))
        summary = summarize_tornado_concern_coverage(working_rows)
        for key in [
            "candidate_dates_total",
            "sig_tor_candidates_total",
            "outbreak_candidates_total",
            "all_tornado_candidates_total",
            "real_ingest_confirmed_total",
            "forecast_artifacts_already_present",
            "forecast_artifacts_generated",
            "forecast_artifacts_failed",
            "verification_artifacts_already_present",
            "verification_artifacts_generated",
            "verification_artifacts_failed",
            "fully_evaluable_dates_total",
            "newly_added_ready_dates",
            "failed_dates_total",
        ]:
            print(f"{key}={summary[key]}")
        print(f"failed_dates_examples_with_reason={','.join(summary['failed_dates_examples_with_reason'])}")
    else:
        dates = select_real_case_init_dates(
            paths.outputs,
            paths.verification,
            labels_dir=paths.labels,
            interim_dir=paths.interim,
            start=args.start,
            end=args.end,
            priority_tier=args.priority_tier,
        )
        write_init_dates_file(output_path, dates)
        print(f"wrote {len(dates)} init dates to {output_path}")
        if args.ranked_candidates_csv:
            write_tornado_concern_candidate_csv(Path(args.ranked_candidates_csv), candidate_rows)
        if args.audit_csv:
            write_tornado_concern_candidate_csv(Path(args.audit_csv), candidate_rows.sort_values(["date"], kind="mergesort"))
    if args.audit:
        audit = audit_real_case_init_dates(
            paths.outputs,
            paths.verification,
            labels_dir=paths.labels,
            interim_dir=paths.interim,
            start=args.start,
            end=args.end,
            priority_tier=args.priority_tier,
        )
        for key in [
            "candidate_dates_from_verification_jsons",
            "candidate_dates_from_outbreaks_parquet",
            "candidate_dates_from_spc_reports_parquet",
            "dates_with_local_forecast_artifact_present",
            "dates_with_real_ingest_confirmed",
            "dates_with_tornado_relevant_label_or_report_evidence",
            "final_selected_dates",
        ]:
            print(f"{key}={audit[key]}")
        for key in [
            "missing_forecast_examples",
            "not_real_ingest_examples",
            "missing_tornado_evidence_examples",
            "final_selected_examples",
        ]:
            print(f"{key}={','.join(audit[key])}")
    if args.audit_missing_artifacts:
        for row in missing_real_case_forecast_artifacts(
            paths.outputs,
            paths.verification,
            labels_dir=paths.labels,
            interim_dir=paths.interim,
            start=args.start,
            end=args.end,
            priority_tier=args.priority_tier,
        ):
            print(
                "missing_artifact"
                f" init_date={row['init_date']}"
                f" expected_artifact_pattern={row['expected_artifact_pattern']}"
                f" generation_command={row['generation_command']}"
            )


if __name__ == "__main__":
    main()
