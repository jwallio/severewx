from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from severewx.cli import recover_tornado_concern_real_ingest as recovery_cli
from severewx.cli.tornado_concern_eval import PRIORITY_TIERS, build_tornado_concern_candidate_rows, select_tornado_concern_recovery_candidates
from severewx.config import load_settings
from severewx.utils.paths import build_paths


WAVE_CHOICES = ["sig_tor", "outbreak", "mixed", "hard_negative"]
MIXED_WAVE_TIERS = ["sig_tor", "outbreak", "all_tornado"]
HARD_NEGATIVE_PROXY_TIERS = ["outbreak", "all_tornado"]


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_none_"
    subset = frame.loc[:, columns].fillna("").copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in record) + " |" for record in subset.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def _read_dates_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _select_dates_file_rows(rows: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    if rows.empty or not dates:
        return pd.DataFrame(columns=rows.columns)
    order = {date: index for index, date in enumerate(dict.fromkeys(dates))}
    selected = rows.loc[rows["date"].astype(str).isin(order)].copy()
    if selected.empty:
        return selected
    selected["_requested_order"] = selected["date"].astype(str).map(order).fillna(999999).astype(int)
    return selected.sort_values(["_requested_order", "date"], kind="mergesort").drop(columns=["_requested_order"]).drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)


def select_wave_candidates(rows: pd.DataFrame, *, wave: str, max_dates: int | None = None) -> pd.DataFrame:
    if wave not in WAVE_CHOICES:
        raise ValueError(f"unsupported wave: {wave}")
    if wave in {"sig_tor", "outbreak"}:
        return select_tornado_concern_recovery_candidates(
            rows,
            priority_tier=wave,
            failure_detail="no_local_real_ingest_evidence",
            max_dates=max_dates,
        )
    if wave == "hard_negative":
        failure_reason = rows["failure_reason"] if "failure_reason" in rows.columns else pd.Series("", index=rows.index)
        failure_detail = rows["real_ingest_failure_detail"] if "real_ingest_failure_detail" in rows.columns else pd.Series("", index=rows.index)
        priority_tier = rows["priority_tier"] if "priority_tier" in rows.columns else pd.Series("", index=rows.index)
        eligible = rows[
            failure_reason.astype(str).eq("not_real_ingest_confirmed")
            & failure_detail.astype(str).eq("no_local_real_ingest_evidence")
            & priority_tier.astype(str).isin(HARD_NEGATIVE_PROXY_TIERS)
        ].copy()
        if eligible.empty:
            return eligible
        sort_columns = [column for column in ["priority_sort_key", "priority_tier", "date"] if column in eligible.columns]
        eligible = eligible.sort_values(sort_columns, kind="mergesort").drop_duplicates(subset=["date"], keep="first")
        if max_dates is not None:
            eligible = eligible.head(int(max_dates))
        return eligible.reset_index(drop=True)

    tier_frames = [
        select_tornado_concern_recovery_candidates(
            rows,
            priority_tier=tier,
            failure_detail="no_local_real_ingest_evidence",
            max_dates=None,
        )
        for tier in MIXED_WAVE_TIERS
    ]
    indices = [0] * len(tier_frames)
    ordered_rows: list[dict[str, Any]] = []
    seen_dates: set[str] = set()
    while True:
        progressed = False
        for tier_index, frame in enumerate(tier_frames):
            if indices[tier_index] >= len(frame):
                continue
            candidate = frame.iloc[indices[tier_index]].to_dict()
            indices[tier_index] += 1
            progressed = True
            date = str(candidate.get("date", ""))
            if not date or date in seen_dates:
                continue
            seen_dates.add(date)
            ordered_rows.append(candidate)
            if max_dates is not None and len(ordered_rows) >= int(max_dates):
                return pd.DataFrame(ordered_rows)
        if not progressed:
            break
    return pd.DataFrame(ordered_rows)


def plan_recovery_batches(selected: pd.DataFrame, *, batch_size: int) -> list[pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if selected.empty:
        return []
    batches: list[pd.DataFrame] = []
    for start_index in range(0, len(selected), batch_size):
        batch = selected.iloc[start_index : start_index + batch_size].copy().reset_index(drop=True)
        batches.append(batch)
    return batches


def _status_count(frame: pd.DataFrame, status: str) -> int:
    if frame.empty or "recovery_status" not in frame.columns:
        return 0
    return int(frame["recovery_status"].astype(str).eq(status).sum())


def _batch_summary_row(batch_number: int, batch_frame: pd.DataFrame, results: pd.DataFrame) -> dict[str, Any]:
    return {
        "batch_id": f"batch_{batch_number:02d}",
        "planned_dates": int(len(batch_frame)),
        "attempted_dates_total": int(len(results)),
        "recovered_ready": _status_count(results, "recovered_ready"),
        "recovered_real_ingest_only": _status_count(results, "recovered_real_ingest_only"),
        "still_blocked_no_real_evidence": _status_count(results, "still_blocked_no_real_evidence"),
        "staging_failed": _status_count(results, "staging_failed"),
        "forecast_failed": _status_count(results, "forecast_failed"),
        "verification_failed": _status_count(results, "verification_failed"),
        "no_supported_historical_source": _status_count(results, "no_supported_historical_source"),
        "dry_run": _status_count(results, "dry_run"),
        "dates": ",".join(batch_frame["date"].astype(str).tolist()) if not batch_frame.empty else "",
    }


def write_wave_summary_markdown(
    path: Path,
    *,
    wave: str,
    start: str | None,
    end: str | None,
    batch_size: int,
    dry_run: bool,
    selected: pd.DataFrame,
    summary_frame: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = selected.loc[:, ["date", "priority_tier"]].head(12) if not selected.empty else pd.DataFrame(columns=["date", "priority_tier"])
    markdown = "\n".join(
        [
            "# Tornado Concern Recovery Wave",
            "",
            f"- Wave: {wave}",
            f"- Date range: {start or 'all'} to {end or 'all'}",
            f"- Batch size: {batch_size}",
            f"- Dry run: {dry_run}",
            f"- Planned dates: {len(selected)}",
            f"- Batch count: {len(summary_frame)}",
            "",
            "## Planned Batch Summary",
            "",
            _markdown_table(
                summary_frame,
                [
                    "batch_id",
                    "planned_dates",
                    "attempted_dates_total",
                    "recovered_ready",
                    "recovered_real_ingest_only",
                    "still_blocked_no_real_evidence",
                    "dry_run",
                ],
            ),
            "",
            "## Planned Date Preview",
            "",
            _markdown_table(preview, ["date", "priority_tier"]),
        ]
    )
    path.write_text(markdown + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic batched tornado-concern real-ingest recovery wave")
    parser.add_argument("--start", help="Inclusive start date filter YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive end date filter YYYY-MM-DD")
    parser.add_argument("--wave", required=True, choices=WAVE_CHOICES, help="Recovery wave type")
    parser.add_argument("--max-dates", type=int, help="Optional max dates to include after deterministic selection")
    parser.add_argument("--dates-file", help="Optional reviewed date list to recover instead of selecting by wave proxy")
    parser.add_argument("--batch-size", type=int, default=5, help="Dates per batch")
    parser.add_argument("--output-dir", required=True, help="Output directory for batch audits and wave summary")
    parser.add_argument("--dry-run", action="store_true", help="Plan the batches and write reports without mutating artifacts")
    args = parser.parse_args(argv)

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
    if args.dates_file:
        selected = _select_dates_file_rows(all_rows, _read_dates_file(Path(args.dates_file)))
        if args.max_dates is not None:
            selected = selected.head(int(args.max_dates))
    else:
        selected = select_wave_candidates(all_rows, wave=args.wave, max_dates=args.max_dates)
    batches = plan_recovery_batches(selected, batch_size=int(args.batch_size)) if not selected.empty else []

    output_dir = Path(args.output_dir)
    batches_dir = output_dir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    combined_rows: list[pd.DataFrame] = []
    for batch_number, batch in enumerate(batches, start=1):
        recovery_rows = [recovery_cli._recover_date(paths, row, dry_run=bool(args.dry_run)) for row in batch.to_dict(orient="records")]
        results = pd.DataFrame(recovery_rows)
        if not results.empty:
            results.insert(0, "batch_id", f"batch_{batch_number:02d}")
        combined_rows.append(results)
        batch_csv = batches_dir / f"batch_{batch_number:02d}_recovery.csv"
        batch_md = batches_dir / f"batch_{batch_number:02d}_recovery.md"
        recovery_cli.write_recovery_csv(batch_csv, results)
        recovery_cli.write_recovery_markdown(
            batch_md,
            results,
            start=args.start,
            end=args.end,
            priority_tier=args.wave if args.wave in PRIORITY_TIERS else "all",
            failure_detail="no_local_real_ingest_evidence",
            max_dates=len(batch),
            dry_run=bool(args.dry_run),
        )
        summary_rows.append(_batch_summary_row(batch_number, batch, results))

    summary_frame = pd.DataFrame(summary_rows)
    wave_summary_csv = output_dir / "recovery_wave_summary.csv"
    wave_summary_md = output_dir / "recovery_wave_summary.md"
    summary_frame.to_csv(wave_summary_csv, index=False)
    write_wave_summary_markdown(
        wave_summary_md,
        wave=args.wave,
        start=args.start,
        end=args.end,
        batch_size=int(args.batch_size),
        dry_run=bool(args.dry_run),
        selected=selected,
        summary_frame=summary_frame,
    )
    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined.to_csv(output_dir / "recovery_wave_cases.csv", index=False)

    print(f"planned_dates_total={len(selected)}")
    print(f"batches_total={len(batches)}")
    print(f"wave_summary_csv={wave_summary_csv}")
    print(f"wave_summary_md={wave_summary_md}")


if __name__ == "__main__":
    main()
