"""Compare baseline and component-challenger tornado-concern ranked eval CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


FAILURE_COLUMNS = [
    "hail_outranks_tornado_failure",
    "top_day_category_mismatch",
    "non_outbreak_outranks_outbreak_failure",
]


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _top_windows(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    working = frame.copy()
    if "day_rank_within_init" not in working.columns:
        score_column = "ranking_tornado_concern_score" if "ranking_tornado_concern_score" in working.columns else "tornado_concern_score"
        working = working.sort_values(["init_date", score_column], ascending=[True, False], kind="mergesort").copy()
        working["day_rank_within_init"] = working.groupby("init_date").cumcount() + 1
    top = working.loc[working["day_rank_within_init"].astype(int).eq(1)].copy()
    keep = [
        "init_date",
        "top_valid_date",
        "top_observed_category",
        "top_observed_tornado_outbreak",
        "top_observed_significant_tornado_support",
        "best_tornado_valid_date",
        "top_minus_best_tornado_score",
        "ranking_tornado_concern_score",
        *FAILURE_COLUMNS,
    ]
    for column in keep:
        if column not in top.columns:
            top[column] = False if column in FAILURE_COLUMNS else ""
    return top.loc[:, keep].rename(columns={column: f"{prefix}_{column}" for column in keep if column != "init_date"})


def build_delta_report(baseline: pd.DataFrame, challenger: pd.DataFrame) -> pd.DataFrame:
    base = _top_windows(baseline, "baseline")
    chal = _top_windows(challenger, "challenger")
    merged = base.merge(chal, on="init_date", how="outer").fillna("")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        baseline_failures = {column for column in FAILURE_COLUMNS if _boolish(row.get(f"baseline_{column}", False))}
        challenger_failures = {column for column in FAILURE_COLUMNS if _boolish(row.get(f"challenger_{column}", False))}
        fixed = sorted(baseline_failures - challenger_failures)
        introduced = sorted(challenger_failures - baseline_failures)
        remaining = sorted(baseline_failures & challenger_failures)
        baseline_margin = pd.to_numeric(row.get("baseline_top_minus_best_tornado_score", float("nan")), errors="coerce")
        challenger_margin = pd.to_numeric(row.get("challenger_top_minus_best_tornado_score", float("nan")), errors="coerce")
        margin_delta = float(baseline_margin) - float(challenger_margin) if pd.notna(baseline_margin) and pd.notna(challenger_margin) else float("nan")
        top_changed = str(row.get("baseline_top_valid_date", "")) != str(row.get("challenger_top_valid_date", ""))
        category_changed = str(row.get("baseline_top_observed_category", "")) != str(row.get("challenger_top_observed_category", ""))
        improved = bool(fixed) or (margin_delta > 0.0 and not introduced)
        worsened = bool(introduced) or (margin_delta < 0.0 and not fixed)
        margin_improved_without_rank_category = margin_delta > 0.0 and not top_changed and not category_changed and not fixed
        rows.append(
            {
                "init_date": row.get("init_date", ""),
                "baseline_top_valid_date": row.get("baseline_top_valid_date", ""),
                "challenger_top_valid_date": row.get("challenger_top_valid_date", ""),
                "baseline_top_observed_category": row.get("baseline_top_observed_category", ""),
                "challenger_top_observed_category": row.get("challenger_top_observed_category", ""),
                "baseline_margin": baseline_margin,
                "challenger_margin": challenger_margin,
                "margin_delta_baseline_minus_challenger": margin_delta,
                "top_day_changed": top_changed,
                "top_category_changed": category_changed,
                "fixed_failures": ";".join(fixed),
                "introduced_failures": ";".join(introduced),
                "remaining_failures": ";".join(remaining),
                "hail_over_tornado_fixed": "hail_outranks_tornado_failure" in fixed,
                "hail_over_tornado_remaining": "hail_outranks_tornado_failure" in remaining,
                "top_day_mismatch_fixed": "top_day_category_mismatch" in fixed,
                "top_day_mismatch_remaining": "top_day_category_mismatch" in remaining,
                "non_outbreak_over_outbreak_remained_or_worsened": (
                    "non_outbreak_outranks_outbreak_failure" in remaining
                    or "non_outbreak_outranks_outbreak_failure" in introduced
                ),
                "margin_improved_but_rank_category_unchanged": margin_improved_without_rank_category,
                "improved_by_challenger": improved,
                "worsened_by_challenger": worsened,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["worsened_by_challenger", "improved_by_challenger", "init_date"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)


def checkpoint_acceptance(frame: pd.DataFrame) -> bool:
    hail_fixed = int(frame["hail_over_tornado_fixed"].fillna(False).sum()) if "hail_over_tornado_fixed" in frame.columns else 0
    top_fixed = int(frame["top_day_mismatch_fixed"].fillna(False).sum()) if "top_day_mismatch_fixed" in frame.columns else 0
    non_outbreak_worse = int(
        frame["introduced_failures"].fillna("").astype(str).str.contains("non_outbreak_outranks_outbreak_failure", regex=False).sum()
    ) if "introduced_failures" in frame.columns else 0
    return bool((hail_fixed > 0 or top_fixed > 0) and non_outbreak_worse == 0)


def write_markdown(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    acceptance = checkpoint_acceptance(frame)
    lines = [
        "# Tornado Concern Component Delta",
        "",
        f"- dates_compared: {len(frame)}",
        f"- dates_improved_by_challenger: {int(frame['improved_by_challenger'].fillna(False).sum()) if not frame.empty else 0}",
        f"- dates_worsened_by_challenger: {int(frame['worsened_by_challenger'].fillna(False).sum()) if not frame.empty else 0}",
        f"- hail_over_tornado_failures_fixed: {int(frame['hail_over_tornado_fixed'].fillna(False).sum()) if not frame.empty else 0}",
        f"- hail_over_tornado_failures_remaining: {int(frame['hail_over_tornado_remaining'].fillna(False).sum()) if not frame.empty else 0}",
        f"- top_day_mismatches_fixed: {int(frame['top_day_mismatch_fixed'].fillna(False).sum()) if not frame.empty else 0}",
        f"- top_day_mismatches_remaining: {int(frame['top_day_mismatch_remaining'].fillna(False).sum()) if not frame.empty else 0}",
        f"- non_outbreak_over_outbreak_remained_or_worsened: {int(frame['non_outbreak_over_outbreak_remained_or_worsened'].fillna(False).sum()) if not frame.empty else 0}",
        f"- checkpoint_acceptance_satisfied: {acceptance}",
        "",
        "## Changed Cases",
        "",
        "| init_date | baseline_top | challenger_top | fixed | introduced | remaining | margin_delta |",
        "|---|---|---|---|---|---|---:|",
    ]
    changed = frame.loc[
        frame[["fixed_failures", "introduced_failures"]].fillna("").astype(str).ne("").any(axis=1)
        | frame["top_day_changed"].fillna(False)
        | frame["margin_improved_but_rank_category_unchanged"].fillna(False)
    ].copy() if not frame.empty else pd.DataFrame()
    for row in changed.to_dict(orient="records"):
        lines.append(
            "| {date} | {base} | {chal} | {fixed} | {introduced} | {remaining} | {delta:.4f} |".format(
                date=row.get("init_date", ""),
                base=row.get("baseline_top_valid_date", ""),
                chal=row.get("challenger_top_valid_date", ""),
                fixed=row.get("fixed_failures", ""),
                introduced=row.get("introduced_failures", ""),
                remaining=row.get("remaining_failures", ""),
                delta=float(row.get("margin_delta_baseline_minus_challenger", 0.0) or 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--challenger-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    frame = build_delta_report(pd.read_csv(args.baseline_csv), pd.read_csv(args.challenger_csv))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    write_markdown(Path(args.output_md), frame)
    print(f"dates_compared={len(frame)}")
    print(f"checkpoint_acceptance_satisfied={checkpoint_acceptance(frame)}")


if __name__ == "__main__":
    main()
