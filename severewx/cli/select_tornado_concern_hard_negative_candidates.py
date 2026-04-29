"""Select reviewed hard-negative recovery candidates from existing eval data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _top_rows(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "day_rank_within_init" not in working.columns:
        score_column = "ranking_tornado_concern_score" if "ranking_tornado_concern_score" in working.columns else "tornado_concern_score"
        working = working.sort_values(["init_date", score_column], ascending=[True, False], kind="mergesort").copy()
        working["day_rank_within_init"] = working.groupby("init_date").cumcount() + 1
    return working.loc[working["day_rank_within_init"].astype(int).eq(1)].copy()


def select_hard_negative_candidates(frame: pd.DataFrame, *, max_dates: int = 50) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in _top_rows(frame).iterrows():
        category = str(row.get("top_observed_category", row.get("observed_category", ""))).lower()
        top_is_tornado_verified = int(row.get("top_observed_tornado_outbreak", row.get("observed_tornado_outbreak", 0)) or 0) > 0
        top_is_sig_tor_verified = int(row.get("top_observed_significant_tornado_support", row.get("observed_significant_tornado_support", 0)) or 0) > 0
        hail_wind_context = (
            "hail" in category
            or "wind" in category
            or _float(row, "hail_overlap_max") >= 1.80
            or _float(row, "wind_overlap_max") >= 1.80
            or _float(row, "top_context_top", _float(row, "context_top")) >= 0.75
        )
        weak_tornado_verification = (
            not top_is_tornado_verified
            and not top_is_sig_tor_verified
            and (
                _boolish(row.get("hail_outranks_tornado_failure", False))
                or _boolish(row.get("top_day_category_mismatch", False))
                or "non_outbreak" in category
            )
        )
        if not hail_wind_context or not weak_tornado_verification:
            continue
        best_tornado_signal = _float(row, "best_tornado_tornado_signal", _float(row, "tornado_signal"))
        broad_pressure = max(
            _float(row, "top_minus_best_tornado_context_top"),
            _float(row, "top_minus_best_tornado_broad_contamination_proxy"),
            _float(row, "hail_overlap_max") - _float(row, "best_tornado_hail_support", _float(row, "best_tornado_hail_overlap_max")),
            _float(row, "wind_overlap_max") - _float(row, "best_tornado_wind_support", _float(row, "best_tornado_wind_overlap_max")),
        )
        rows.append(
            {
                "date": str(row.get("init_date", "")),
                "top_valid_date": str(row.get("top_valid_date", row.get("valid_date", ""))),
                "best_tornado_valid_date": str(row.get("best_tornado_valid_date", "")),
                "top_observed_category": str(row.get("top_observed_category", row.get("observed_category", ""))),
                "top_minus_best_tornado_score": _float(row, "top_minus_best_tornado_score", float("nan")),
                "best_tornado_signal": best_tornado_signal,
                "broad_pressure": broad_pressure,
                "selection_reason": "hail_wind_heavy_weak_tornado_verification",
            }
        )
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).drop_duplicates(subset=["date"], keep="first")
    result = result.sort_values(
        ["broad_pressure", "top_minus_best_tornado_score", "date"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return result.head(int(max_dates)).reset_index(drop=True)


def write_candidate_dates(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = frame["date"].astype(str).tolist() if not frame.empty and "date" in frame.columns else []
    path.write_text("\n".join(dates) + ("\n" if dates else ""), encoding="utf-8")


def write_markdown(path: Path, frame: pd.DataFrame, *, output_dates: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tornado Concern Hard-negative Recovery Candidates",
        "",
        f"- candidate_count: {len(frame)}",
        f"- candidate_dates_file: `{output_dates}`",
        "",
        "## Review Command",
        "",
        f"`python -m severewx.cli.run_tornado_concern_recovery_wave --wave hard_negative --dates-file {output_dates.as_posix()} --batch-size 10 --output-dir data/outputs/verification/hard_negative_recovery --dry-run`",
        "",
        "Remove `--dry-run` only after reviewing the candidate dates.",
        "",
        "## Candidates",
        "",
        "| date | top_valid_date | best_tornado_valid_date | top_category | margin | broad_pressure |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {date} | {top} | {best} | {category} | {margin:.4f} | {pressure:.4f} |".format(
                date=row.get("date", ""),
                top=row.get("top_valid_date", ""),
                best=row.get("best_tornado_valid_date", ""),
                category=row.get("top_observed_category", ""),
                margin=float(row.get("top_minus_best_tornado_score", 0.0) or 0.0),
                pressure=float(row.get("broad_pressure", 0.0) or 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True, help="Ranked tornado-concern eval CSV")
    parser.add_argument("--output", default="data/outputs/verification/hard_negative_recovery_candidates.txt")
    parser.add_argument("--report-csv")
    parser.add_argument("--report-md")
    parser.add_argument("--max-dates", type=int, default=50)
    args = parser.parse_args(argv)

    candidates = select_hard_negative_candidates(pd.read_csv(args.eval_csv), max_dates=int(args.max_dates))
    output = Path(args.output)
    write_candidate_dates(output, candidates)
    if args.report_csv:
        report_csv = Path(args.report_csv)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(report_csv, index=False)
    if args.report_md:
        write_markdown(Path(args.report_md), candidates, output_dates=output)
    print(f"candidate_dates={len(candidates)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
