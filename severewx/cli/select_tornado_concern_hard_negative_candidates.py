"""Select severe-active, tornado-weak hard-negative recovery candidates from local evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LABELS_DIR = Path("data/labels")
DEFAULT_OUTPUTS_DIR = Path("data/outputs")
DEFAULT_VERIFICATION_DIR = Path("data/outputs/verification")
DEFAULT_INTERIM_DIR = Path("data/interim")
DEFAULT_READY_DATES_FILE = Path("data/outputs/verification/tornado_concern_ready_dates.txt")
DEFAULT_MIN_HAIL_WIND_REPORTS = 25


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _normalize_date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(parsed) else str(parsed.strftime("%Y-%m-%d"))


def _date_in_range(date: str, *, start: str | None, end: str | None) -> bool:
    if start and date < _normalize_date(start):
        return False
    if end and date > _normalize_date(end):
        return False
    return True


def _read_dates(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _latest_matching_file(directory: Path, pattern: str) -> Path | None:
    matches = list(directory.glob(pattern)) if directory.exists() else []
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _has_forecast_artifact(outputs_dir: Path, date: str) -> bool:
    return _latest_matching_file(outputs_dir, f"forecast_products_{date}_*.nc") is not None


def _has_verification_artifact(verification_dir: Path, date: str) -> bool:
    return (verification_dir / f"{date}_verification.json").exists() or _latest_matching_file(verification_dir, f"{date}*_verification*.json") is not None


def _real_ingest_confirmed(interim_dir: Path, date: str) -> bool:
    path = interim_dir / f"ingest_summary_{date}_00.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    source = str(payload.get("source") or "").lower()
    source_mode = str(payload.get("source_mode") or "").lower()
    return source not in {"synthetic", "synthetic_fallback", "synthetic_degraded"} and source_mode == "real" and bool(payload.get("real_ingest_available", False))


def _read_spc_report_counts(labels_dir: Path, *, start: str | None, end: str | None) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(labels_dir.glob("spc_reports_*.parquet"))
    unavailable: list[str] = []
    if not paths:
        return pd.DataFrame(), [str(labels_dir / "spc_reports_*.parquet")]
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            unavailable.append(str(path))
            continue
        required = {"date", "hazard"}
        if not required.issubset(frame.columns):
            unavailable.append(str(path))
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["date"])
        frame = frame.loc[frame["date"].map(lambda value: _date_in_range(str(value), start=start, end=end))]
        if frame.empty:
            continue
        frame["hazard"] = frame["hazard"].astype(str).str.lower()
        frame["significant"] = frame["significant"].fillna(0).astype(float) if "significant" in frame.columns else 0.0
        frames.append(frame[["date", "hazard", "significant"]])
    if not frames:
        return pd.DataFrame(), unavailable
    combined = pd.concat(frames, ignore_index=True)
    counts = combined.assign(report_count=1).pivot_table(index="date", columns="hazard", values="report_count", aggfunc="sum", fill_value=0)
    for column in ["hail", "wind", "tornado"]:
        if column not in counts.columns:
            counts[column] = 0
    significant_tornado = (
        combined.loc[combined["hazard"].eq("tornado") & combined["significant"].astype(float).gt(0)]
        .groupby("date")
        .size()
        .rename("significant_tornado_reports")
    )
    result = counts.reset_index().rename(
        columns={
            "hail": "hail_reports",
            "wind": "wind_reports",
            "tornado": "tornado_reports",
        }
    )
    result = result.merge(significant_tornado.reset_index(), on="date", how="left")
    result["significant_tornado_reports"] = result["significant_tornado_reports"].fillna(0).astype(int)
    result["hail_reports"] = result["hail_reports"].fillna(0).astype(int)
    result["wind_reports"] = result["wind_reports"].fillna(0).astype(int)
    result["tornado_reports"] = result["tornado_reports"].fillna(0).astype(int)
    result["total_severe_reports"] = result["hail_reports"] + result["wind_reports"] + result["tornado_reports"]
    result["source"] = "spc_reports"
    return result, unavailable


def _read_outbreak_flags(labels_dir: Path, *, start: str | None, end: str | None) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(labels_dir.glob("outbreaks_*.parquet"))
    unavailable: list[str] = []
    if not paths:
        return pd.DataFrame(columns=["date", "outbreak_tornado_positive", "outbreak_sig_tor_positive"]), [str(labels_dir / "outbreaks_*.parquet")]
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            unavailable.append(str(path))
            continue
        if "date" not in frame.columns:
            unavailable.append(str(path))
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["date"])
        frame = frame.loc[frame["date"].map(lambda value: _date_in_range(str(value), start=start, end=end))]
        if frame.empty:
            continue
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "outbreak_tornado_positive", "outbreak_sig_tor_positive"]), unavailable
    combined = pd.concat(frames, ignore_index=True)
    for column in ["tornado_outbreak", "significant_tornado_support", "significant_tornado_count"]:
        if column not in combined.columns:
            combined[column] = 0
    grouped = combined.groupby("date", as_index=False)[["tornado_outbreak", "significant_tornado_support", "significant_tornado_count"]].max()
    grouped["outbreak_tornado_positive"] = grouped["tornado_outbreak"].fillna(0).astype(float).gt(0)
    grouped["outbreak_sig_tor_positive"] = (
        grouped["significant_tornado_support"].fillna(0).astype(float).gt(0)
        | grouped["significant_tornado_count"].fillna(0).astype(float).gt(0)
    )
    return grouped[["date", "outbreak_tornado_positive", "outbreak_sig_tor_positive"]], unavailable


def _with_artifact_flags(frame: pd.DataFrame, *, outputs_dir: Path, verification_dir: Path, interim_dir: Path, ready_dates: set[str]) -> pd.DataFrame:
    working = frame.copy()
    working["already_ready"] = working["date"].astype(str).isin(ready_dates)
    working["has_forecast_artifact"] = working["date"].astype(str).map(lambda date: _has_forecast_artifact(outputs_dir, date))
    working["has_verification_artifact"] = working["date"].astype(str).map(lambda date: _has_verification_artifact(verification_dir, date))
    working["real_ingest_confirmed"] = working["date"].astype(str).map(lambda date: _real_ingest_confirmed(interim_dir, date))
    return working


def build_local_hard_negative_pool(
    *,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    outputs_dir: Path = DEFAULT_OUTPUTS_DIR,
    verification_dir: Path = DEFAULT_VERIFICATION_DIR,
    interim_dir: Path = DEFAULT_INTERIM_DIR,
    ready_dates_file: Path = DEFAULT_READY_DATES_FILE,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report_counts, report_unavailable = _read_spc_report_counts(labels_dir, start=start, end=end)
    outbreak_flags, outbreak_unavailable = _read_outbreak_flags(labels_dir, start=start, end=end)
    unavailable = report_unavailable + outbreak_unavailable
    if report_counts.empty:
        return pd.DataFrame(), {"unavailable_sources": unavailable}
    ready_dates = _read_dates(ready_dates_file)
    pool = report_counts.merge(outbreak_flags, on="date", how="left")
    pool["outbreak_tornado_positive"] = pool["outbreak_tornado_positive"].fillna(False).map(_boolish)
    pool["outbreak_sig_tor_positive"] = pool["outbreak_sig_tor_positive"].fillna(False).map(_boolish)
    pool["excluded_outbreak_positive"] = pool["outbreak_tornado_positive"] | pool["outbreak_sig_tor_positive"] | pool["significant_tornado_reports"].fillna(0).astype(int).gt(0)
    pool = _with_artifact_flags(pool, outputs_dir=outputs_dir, verification_dir=verification_dir, interim_dir=interim_dir, ready_dates=ready_dates)
    return pool, {"unavailable_sources": unavailable}


def select_local_hard_negative_candidates(
    pool: pd.DataFrame,
    *,
    target_count: int = 50,
    max_tornado_reports: int = 0,
    min_hail_wind_reports: int = DEFAULT_MIN_HAIL_WIND_REPORTS,
    exclude_ready: bool = True,
    exclude_outbreak_positive: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if pool.empty:
        return pd.DataFrame(), {"excluded_outbreak_positive_count": 0, "zero_tornado_candidates": 0, "one_tornado_fallback_candidates": 0}
    working = pool.copy()
    working["hail_wind_reports"] = working["hail_reports"].fillna(0).astype(int) + working["wind_reports"].fillna(0).astype(int)
    severe_mask = working["hail_wind_reports"].ge(int(min_hail_wind_reports))
    eligible = working.loc[severe_mask].copy()
    excluded_outbreak_positive_count = int(eligible["excluded_outbreak_positive"].fillna(False).map(_boolish).sum())
    if exclude_outbreak_positive:
        eligible = eligible.loc[~eligible["excluded_outbreak_positive"].fillna(False).map(_boolish)].copy()
    if exclude_ready:
        eligible = eligible.loc[~eligible["already_ready"].fillna(False).map(_boolish)].copy()

    primary = eligible.loc[eligible["tornado_reports"].fillna(0).astype(int).le(int(max_tornado_reports))].copy()
    primary["selection_reason"] = "zero_tornado_hail_wind_heavy" if int(max_tornado_reports) == 0 else "tornado_weak_hail_wind_heavy"
    selected = primary
    fallback = pd.DataFrame(columns=eligible.columns)
    if int(max_tornado_reports) == 0 and len(selected) < int(target_count):
        fallback = eligible.loc[
            eligible["tornado_reports"].fillna(0).astype(int).gt(0)
            & eligible["tornado_reports"].fillna(0).astype(int).le(1)
            & eligible["significant_tornado_reports"].fillna(0).astype(int).eq(0)
        ].copy()
        fallback["selection_reason"] = "one_tornado_report_fallback_hail_wind_heavy"
        selected = pd.concat([selected, fallback], ignore_index=True)
    if selected.empty:
        return selected, {
            "excluded_outbreak_positive_count": excluded_outbreak_positive_count,
            "zero_tornado_candidates": 0,
            "one_tornado_fallback_candidates": 0,
        }
    selected["recovery_priority"] = (
        selected["hail_wind_reports"].astype(float)
        - 50.0 * selected["tornado_reports"].astype(float)
        + 10.0 * selected["real_ingest_confirmed"].map(_boolish).astype(int)
        + 5.0 * selected["has_forecast_artifact"].map(_boolish).astype(int)
        + 5.0 * selected["has_verification_artifact"].map(_boolish).astype(int)
    )
    selected = selected.sort_values(
        ["tornado_reports", "recovery_priority", "date"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates(subset=["date"], keep="first")
    selected = selected.head(int(target_count)).reset_index(drop=True)
    return selected, {
        "excluded_outbreak_positive_count": excluded_outbreak_positive_count,
        "zero_tornado_candidates": int(selected["tornado_reports"].fillna(0).astype(int).eq(0).sum()),
        "one_tornado_fallback_candidates": int(selected["tornado_reports"].fillna(0).astype(int).eq(1).sum()),
    }


def _top_rows(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "day_rank_within_init" not in working.columns:
        score_column = "ranking_tornado_concern_score" if "ranking_tornado_concern_score" in working.columns else "tornado_concern_score"
        working = working.sort_values(["init_date", score_column], ascending=[True, False], kind="mergesort").copy()
        working["day_rank_within_init"] = working.groupby("init_date").cumcount() + 1
    return working.loc[working["day_rank_within_init"].astype(int).eq(1)].copy()


def select_eval_failure_fallback_candidates(frame: pd.DataFrame, *, max_dates: int = 50) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in _top_rows(frame).iterrows():
        category = str(row.get("top_observed_category", row.get("observed_category", ""))).lower()
        top_is_tornado_verified = int(row.get("top_observed_tornado_outbreak", row.get("observed_tornado_outbreak", 0)) or 0) > 0
        top_is_sig_tor_verified = int(row.get("top_observed_significant_tornado_support", row.get("observed_significant_tornado_support", 0)) or 0) > 0
        hail_wind_context = "hail" in category or "wind" in category or _float(row, "top_context_top", _float(row, "context_top")) >= 0.75
        if not hail_wind_context or top_is_tornado_verified or top_is_sig_tor_verified:
            continue
        rows.append(
            {
                "date": str(row.get("init_date", "")),
                "source": "eval_failure_fallback",
                "hail_reports": 0,
                "wind_reports": 0,
                "tornado_reports": 0,
                "significant_tornado_reports": 0,
                "total_severe_reports": 0,
                "already_ready": True,
                "has_forecast_artifact": True,
                "has_verification_artifact": True,
                "real_ingest_confirmed": bool(row.get("is_real_ingest", False)),
                "excluded_outbreak_positive": False,
                "selection_reason": "eval_failure_hail_wind_context_fallback",
                "recovery_priority": _float(row, "top_minus_best_tornado_score", 0.0),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["recovery_priority", "date"], ascending=[False, True], kind="mergesort").head(int(max_dates)).reset_index(drop=True)


def write_candidate_dates(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = frame["date"].astype(str).tolist() if not frame.empty and "date" in frame.columns else []
    path.write_text("\n".join(dates) + ("\n" if dates else ""), encoding="utf-8")


def write_markdown(path: Path, frame: pd.DataFrame, *, output_dates: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unavailable = metadata.get("unavailable_sources", []) or []
    lines = [
        "# Tornado Concern Hard-negative Recovery Candidates",
        "",
        f"- total_candidates_found: {len(frame)}",
        f"- zero_tornado_report_candidates: {metadata.get('zero_tornado_candidates', 0)}",
        f"- one_tornado_report_fallback_candidates: {metadata.get('one_tornado_fallback_candidates', 0)}",
        f"- excluded_outbreak_sig_tor_or_tornado_positive: {metadata.get('excluded_outbreak_positive_count', 0)}",
        f"- candidate_dates_file: `{output_dates}`",
        "",
        "## Unavailable Sources",
        "",
    ]
    lines.extend([f"- {source}" for source in unavailable] if unavailable else ["- none"])
    lines.extend(
        [
            "",
            "## Recovery Command",
            "",
            f"`python -m severewx.cli.run_tornado_concern_recovery_wave --wave hard_negative --dates-file {output_dates.as_posix()} --batch-size 10 --output-dir data/outputs/verification/hard_negative_recovery --dry-run`",
            "",
            "Remove `--dry-run` only after reviewing the candidate dates.",
            "",
            "## Candidates",
            "",
            "| date | source | hail | wind | tornado | sigtor | total | ready | real_ingest | reason | priority |",
            "|---|---|---:|---:|---:|---:|---:|---|---|---|---:|",
        ]
    )
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {date} | {source} | {hail} | {wind} | {tor} | {sig} | {total} | {ready} | {real} | {reason} | {priority:.1f} |".format(
                date=row.get("date", ""),
                source=row.get("source", ""),
                hail=int(row.get("hail_reports", 0) or 0),
                wind=int(row.get("wind_reports", 0) or 0),
                tor=int(row.get("tornado_reports", 0) or 0),
                sig=int(row.get("significant_tornado_reports", 0) or 0),
                total=int(row.get("total_severe_reports", 0) or 0),
                ready=row.get("already_ready", False),
                real=row.get("real_ingest_confirmed", False),
                reason=row.get("selection_reason", ""),
                priority=float(row.get("recovery_priority", 0.0) or 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _output_columns() -> list[str]:
    return [
        "date",
        "source",
        "hail_reports",
        "wind_reports",
        "tornado_reports",
        "significant_tornado_reports",
        "total_severe_reports",
        "already_ready",
        "has_forecast_artifact",
        "has_verification_artifact",
        "real_ingest_confirmed",
        "excluded_outbreak_positive",
        "selection_reason",
        "recovery_priority",
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", help="Optional ranked eval CSV for --use-eval-fallback")
    parser.add_argument("--use-eval-fallback", action="store_true", help="Use the legacy eval-failure selector instead of local report evidence")
    parser.add_argument("--output", default="data/outputs/verification/hard_negative_recovery_candidates.txt")
    parser.add_argument("--report-csv")
    parser.add_argument("--report-md")
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--max-dates", type=int, help="Backward-compatible alias for --target-count")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--exclude-ready", dest="exclude_ready", action="store_true", default=True)
    parser.add_argument("--include-ready", dest="exclude_ready", action="store_false")
    parser.add_argument("--max-tornado-reports", type=int, default=0)
    parser.add_argument("--min-hail-wind-reports", type=int, default=DEFAULT_MIN_HAIL_WIND_REPORTS)
    parser.add_argument("--exclude-outbreak-positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR))
    parser.add_argument("--outputs-dir", default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--verification-dir", default=str(DEFAULT_VERIFICATION_DIR))
    parser.add_argument("--interim-dir", default=str(DEFAULT_INTERIM_DIR))
    parser.add_argument("--ready-dates-file", default=str(DEFAULT_READY_DATES_FILE))
    args = parser.parse_args(argv)

    target_count = int(args.max_dates if args.max_dates is not None else args.target_count)
    metadata: dict[str, Any] = {}
    if args.use_eval_fallback:
        if not args.eval_csv:
            parser.error("--use-eval-fallback requires --eval-csv")
        candidates = select_eval_failure_fallback_candidates(pd.read_csv(args.eval_csv), max_dates=target_count)
        metadata = {
            "unavailable_sources": [],
            "excluded_outbreak_positive_count": 0,
            "zero_tornado_candidates": int(len(candidates)),
            "one_tornado_fallback_candidates": 0,
        }
    else:
        pool, metadata = build_local_hard_negative_pool(
            labels_dir=Path(args.labels_dir),
            outputs_dir=Path(args.outputs_dir),
            verification_dir=Path(args.verification_dir),
            interim_dir=Path(args.interim_dir),
            ready_dates_file=Path(args.ready_dates_file),
            start=args.start,
            end=args.end,
        )
        candidates, counts = select_local_hard_negative_candidates(
            pool,
            target_count=target_count,
            max_tornado_reports=int(args.max_tornado_reports),
            min_hail_wind_reports=int(args.min_hail_wind_reports),
            exclude_ready=bool(args.exclude_ready),
            exclude_outbreak_positive=bool(args.exclude_outbreak_positive),
        )
        metadata.update(counts)
    output_columns = _output_columns()
    if candidates.empty:
        candidates = pd.DataFrame(columns=output_columns)
    else:
        for column in output_columns:
            if column not in candidates.columns:
                candidates[column] = False if column.startswith(("already", "has_", "real_", "excluded_")) else 0
        candidates = candidates.loc[:, output_columns]

    output = Path(args.output)
    write_candidate_dates(output, candidates)
    if args.report_csv:
        report_csv = Path(args.report_csv)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(report_csv, index=False)
    if args.report_md:
        write_markdown(Path(args.report_md), candidates, output_dates=output, metadata=metadata)
    print(f"candidate_dates={len(candidates)}")
    print(f"zero_tornado_report_candidates={metadata.get('zero_tornado_candidates', 0)}")
    print(f"one_tornado_report_fallback_candidates={metadata.get('one_tornado_fallback_candidates', 0)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
