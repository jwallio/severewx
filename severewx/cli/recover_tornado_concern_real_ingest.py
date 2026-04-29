from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from severewx.cli.tornado_concern_eval import (
    PRIORITY_TIERS,
    _PRIORITY_ORDER,
    _classify_generation_failure,
    build_tornado_concern_candidate_rows,
    derive_tornado_concern_recovery_status,
    select_tornado_concern_recovery_candidates,
)
from severewx.config import load_settings
from severewx.utils.paths import build_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
RECOVERY_STATUSES = [
    "recovered_ready",
    "recovered_real_ingest_only",
    "still_blocked_no_real_evidence",
    "staging_failed",
    "forecast_failed",
    "verification_failed",
    "no_supported_historical_source",
    "dry_run",
]


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_none_"
    subset = frame.loc[:, columns].copy()
    subset = subset.fillna("")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in record) + " |" for record in subset.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def _refresh_candidate_row(paths: Any, date: str) -> dict[str, Any]:
    frame = build_tornado_concern_candidate_rows(
        paths.outputs,
        paths.verification,
        labels_dir=paths.labels,
        interim_dir=paths.interim,
        start=date,
        end=date,
        priority_tier="all",
    )
    if frame.empty:
        return {
            "date": date,
            "priority_tier": "all_tornado",
            "real_ingest_confirmed": False,
            "forecast_artifacts_present": False,
            "verification_artifacts_present": False,
            "final_ready": False,
            "failure_reason": "missing_local_input_data",
            "real_ingest_failure_detail": "missing_local_input_data",
            "staged_input_dir_present": False,
            "staged_input_file_count": 0,
            "forecast_metadata_present": False,
        }
    return frame.iloc[0].to_dict()


def _safe_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)


def _staging_report_path(paths: Any, date: str) -> Path:
    return Path(paths.interim) / f"staged_gfs_download_{date}_{date}_00.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _derive_staging_detail(paths: Any, date: str, result: subprocess.CompletedProcess[str] | None) -> str:
    report = _load_json(_staging_report_path(paths, date))
    if report:
        if int(report.get("successful_downloads", 0) or 0) > 0:
            return ""
        failed_rows = [item for item in report.get("results", []) if str(item.get("status")) == "failed"]
        messages = " | ".join(str(item.get("error", "")) for item in failed_rows if item.get("error"))
        if messages:
            classified = _classify_generation_failure("forecast", messages)
            if classified == "date_not_supported_by_archive":
                return "no_supported_historical_source"
            if "unsupported" in messages.lower():
                return "no_supported_historical_source"
            return classified
    if result is None:
        return ""
    combined = " | ".join(part for part in [result.stderr.strip(), result.stdout.strip()] if part)
    if not combined:
        return ""
    classified = _classify_generation_failure("forecast", combined)
    if classified == "date_not_supported_by_archive":
        return "no_supported_historical_source"
    return classified


def _count_by(frame: pd.DataFrame, column: str, allowed: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        values = allowed or []
        return pd.DataFrame([{column: value, "count": 0} for value in values], columns=[column, "count"])
    grouped = (
        frame.assign(**{column: frame[column].replace("", "ready").fillna("ready")})
        .groupby(column, as_index=False)
        .agg(count=("date", "count"))
        .sort_values(["count", column], ascending=[False, True], kind="mergesort")
    )
    if not allowed:
        return grouped
    lookup = dict(zip(grouped[column], grouped["count"], strict=False))
    return pd.DataFrame([{column: value, "count": int(lookup.get(value, 0))} for value in allowed])


def write_recovery_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_recovery_markdown(
    path: Path,
    frame: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
    priority_tier: str,
    failure_detail: str,
    max_dates: int | None,
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = _count_by(frame, "recovery_status", RECOVERY_STATUSES)
    tier_counts = _count_by(frame, "priority_tier", PRIORITY_TIERS[:-1])
    success_examples = (
        frame.loc[frame["recovery_status"].isin(["recovered_ready", "recovered_real_ingest_only"]), ["date", "priority_tier", "recovery_status"]]
        .sort_values(["date"], kind="mergesort")
        .head(10)
    )
    failure_examples = (
        frame.loc[~frame["recovery_status"].isin(["recovered_ready", "recovered_real_ingest_only", "dry_run"]), ["date", "priority_tier", "recovery_status", "recovery_notes"]]
        .sort_values(["date"], kind="mergesort")
        .head(10)
    )
    lines = [
        "# Tornado Concern Real-Ingest Recovery Summary",
        "",
        f"- Date range: {start or 'all'} to {end or 'all'}",
        f"- Priority tier filter: {priority_tier}",
        f"- Failure detail filter: {failure_detail}",
        f"- Max dates: {max_dates if max_dates is not None else 'all'}",
        f"- Dry run: {dry_run}",
        "",
        "## Recovery Status Counts",
        "",
        _markdown_table(status_counts, ["recovery_status", "count"]),
        "",
        "## Priority Tier Counts",
        "",
        _markdown_table(tier_counts, ["priority_tier", "count"]),
        "",
        "## Success Examples",
        "",
        _markdown_table(success_examples, ["date", "priority_tier", "recovery_status"]),
        "",
        "## Failure Examples",
        "",
        _markdown_table(failure_examples, ["date", "priority_tier", "recovery_status", "recovery_notes"]),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _recover_date(paths: Any, initial_row: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    date = str(initial_row["date"])
    working_row = dict(initial_row)
    staged_initially_present = _safe_bool(initial_row.get("staged_input_dir_present")) or int(initial_row.get("staged_input_file_count", 0) or 0) > 0
    forecast_initially_present = _safe_bool(initial_row.get("forecast_artifacts_present"))
    verification_initially_present = _safe_bool(initial_row.get("verification_artifacts_present"))
    forecast_metadata_initially_present = _safe_bool(initial_row.get("forecast_metadata_present"))

    staging_attempted = False
    staged_succeeded = staged_initially_present
    forecast_attempted = False
    forecast_succeeded = forecast_initially_present and forecast_metadata_initially_present
    verification_attempted = False
    verification_succeeded = verification_initially_present
    notes: list[str] = []
    staging_result: subprocess.CompletedProcess[str] | None = None
    forecast_result: subprocess.CompletedProcess[str] | None = None
    verification_result: subprocess.CompletedProcess[str] | None = None

    if not dry_run and not staged_initially_present:
        staging_attempted = True
        staging_result = _run_cli(["python", "-m", "severewx.cli.stage_historical_gfs", "--start", date, "--cycles", "00"])
        refreshed = _refresh_candidate_row(paths, date)
        working_row.update(refreshed)
        staged_succeeded = _safe_bool(refreshed.get("staged_input_dir_present")) or int(refreshed.get("staged_input_file_count", 0) or 0) > 0
        if not staged_succeeded:
            detail = _derive_staging_detail(paths, date, staging_result)
            if detail:
                notes.append(detail)

    can_attempt_forecast = staged_succeeded or staged_initially_present
    if not dry_run and can_attempt_forecast and (not forecast_initially_present or not forecast_metadata_initially_present):
        forecast_attempted = True
        forecast_result = _run_cli(["python", "-m", "severewx.cli.run_forecast", "--date", date, "--cycle", "00"])
        refreshed = _refresh_candidate_row(paths, date)
        working_row.update(refreshed)
        forecast_succeeded = _safe_bool(refreshed.get("forecast_artifacts_present")) and _safe_bool(refreshed.get("forecast_metadata_present"))
        if not forecast_succeeded and forecast_result is not None:
            detail = _classify_generation_failure("forecast", (forecast_result.stderr or "") + " " + (forecast_result.stdout or ""))
            notes.append(detail)

    can_attempt_verification = _safe_bool(working_row.get("forecast_artifacts_present"))
    if not dry_run and can_attempt_verification and not verification_initially_present:
        verification_attempted = True
        verification_result = _run_cli(["python", "-m", "severewx.cli.verify_day", "--date", date])
        refreshed = _refresh_candidate_row(paths, date)
        working_row.update(refreshed)
        verification_succeeded = _safe_bool(refreshed.get("verification_artifacts_present"))
        if not verification_succeeded and verification_result is not None:
            detail = _classify_generation_failure("verification", (verification_result.stderr or "") + " " + (verification_result.stdout or ""))
            notes.append(detail)

    final_row = _refresh_candidate_row(paths, date) if not dry_run else dict(initial_row)
    status, status_note = derive_tornado_concern_recovery_status(
        initial_row,
        final_row,
        staging_attempted=staging_attempted,
        staging_succeeded=staged_succeeded,
        forecast_attempted=forecast_attempted,
        forecast_succeeded=forecast_succeeded,
        verification_attempted=verification_attempted,
        verification_succeeded=verification_succeeded,
        dry_run=dry_run,
    )
    if status_note:
        notes.append(status_note)
    if staging_attempted and not staged_succeeded and not notes:
        notes.append("staging did not create local staged inputs")

    return {
        "date": date,
        "priority_tier": str(initial_row.get("priority_tier", "")),
        "initial_failure_reason": str(initial_row.get("failure_reason", "")),
        "initial_real_ingest_failure_detail": str(initial_row.get("real_ingest_failure_detail", "")),
        "staged_inputs_initially_present": staged_initially_present,
        "staging_attempted": staging_attempted,
        "staging_succeeded": staged_succeeded,
        "forecast_initially_present": forecast_initially_present,
        "forecast_attempted": forecast_attempted,
        "forecast_succeeded": forecast_succeeded,
        "verification_initially_present": verification_initially_present,
        "verification_attempted": verification_attempted,
        "verification_succeeded": verification_succeeded,
        "final_real_ingest_confirmed": _safe_bool(final_row.get("real_ingest_confirmed")),
        "final_fully_evaluable": _safe_bool(final_row.get("final_ready")),
        "final_failure_reason": str(final_row.get("failure_reason", "")),
        "final_real_ingest_failure_detail": str(final_row.get("real_ingest_failure_detail", "")),
        "recovery_status": status,
        "recovery_notes": "; ".join(dict.fromkeys(note for note in notes if note)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover missing local historical real-ingest provenance for tornado-concern candidates")
    parser.add_argument("--start", help="Inclusive start date filter YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive end date filter YYYY-MM-DD")
    parser.add_argument("--priority-tier", choices=PRIORITY_TIERS, default="all", help="Candidate priority tier filter")
    parser.add_argument("--failure-detail", default="no_local_real_ingest_evidence", help="Real-ingest failure detail filter or 'all'")
    parser.add_argument("--max-dates", type=int, help="Limit candidate count after deterministic sorting")
    parser.add_argument("--dry-run", action="store_true", help="Select recovery targets and write reports without mutating artifacts")
    parser.add_argument("--output-csv", help="Optional path for a per-date recovery audit CSV")
    parser.add_argument("--output-md", help="Optional path for a markdown recovery summary")
    args = parser.parse_args()

    settings = load_settings()
    paths = build_paths(settings)
    all_rows = build_tornado_concern_candidate_rows(
        paths.outputs,
        paths.verification,
        labels_dir=paths.labels,
        interim_dir=paths.interim,
        start=args.start,
        end=args.end,
        priority_tier="all",
    )
    selected = select_tornado_concern_recovery_candidates(
        all_rows,
        priority_tier=args.priority_tier,
        failure_detail=args.failure_detail,
        max_dates=args.max_dates,
    )

    recovery_rows = [_recover_date(paths, row, dry_run=bool(args.dry_run)) for row in selected.to_dict(orient="records")]
    recovery_frame = pd.DataFrame(recovery_rows) if recovery_rows else pd.DataFrame(
        columns=[
            "date",
            "priority_tier",
            "initial_failure_reason",
            "initial_real_ingest_failure_detail",
            "staged_inputs_initially_present",
            "staging_attempted",
            "staging_succeeded",
            "forecast_initially_present",
            "forecast_attempted",
            "forecast_succeeded",
            "verification_initially_present",
            "verification_attempted",
            "verification_succeeded",
            "final_real_ingest_confirmed",
            "final_fully_evaluable",
            "final_failure_reason",
            "final_real_ingest_failure_detail",
            "recovery_status",
            "recovery_notes",
        ]
    )
    if not recovery_frame.empty:
        recovery_frame["priority_sort_key"] = recovery_frame["priority_tier"].map(_PRIORITY_ORDER).fillna(99).astype(int)
        recovery_frame = recovery_frame.sort_values(["priority_sort_key", "date"], kind="mergesort").drop(columns=["priority_sort_key"])

    if args.output_csv:
        write_recovery_csv(Path(args.output_csv), recovery_frame)
    if args.output_md:
        write_recovery_markdown(
            Path(args.output_md),
            recovery_frame,
            start=args.start,
            end=args.end,
            priority_tier=args.priority_tier,
            failure_detail=args.failure_detail,
            max_dates=args.max_dates,
            dry_run=bool(args.dry_run),
        )

    print(f"attempted_dates_total={len(recovery_frame)}")
    for status in RECOVERY_STATUSES:
        count = int(recovery_frame["recovery_status"].eq(status).sum()) if not recovery_frame.empty else 0
        if count:
            print(f"{status}={count}")
    for tier in PRIORITY_TIERS[:-1]:
        count = int(recovery_frame["priority_tier"].eq(tier).sum()) if not recovery_frame.empty else 0
        if count:
            print(f"{tier}_attempted={count}")
    failed = recovery_frame.loc[
        ~recovery_frame["recovery_status"].isin(["recovered_ready", "recovered_real_ingest_only", "dry_run"]),
        ["date", "recovery_status", "recovery_notes"],
    ]
    if not failed.empty:
        sample = [f"{row.date}:{row.recovery_status}:{row.recovery_notes}" for row in failed.head(5).itertuples(index=False)]
        print(f"failed_examples={','.join(sample)}")


if __name__ == "__main__":
    main()
