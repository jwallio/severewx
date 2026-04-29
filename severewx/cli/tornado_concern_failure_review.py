from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from severewx.cli.tornado_concern_eval import _latest_prediction_for_date, _verification_for_date
from severewx.config import load_settings
from severewx.utils.paths import build_paths


def _load_ready_dates(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_none_"
    subset = frame.loc[:, columns].fillna("").copy()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in record) + " |" for record in subset.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def _top_window_rows(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "day_rank_within_init" in working.columns:
        working["day_rank_within_init"] = pd.to_numeric(working["day_rank_within_init"], errors="coerce").fillna(999).astype(int)
        working = working.sort_values(["init_date", "day_rank_within_init", "ranking_tornado_concern_score"], ascending=[True, True, False], kind="mergesort")
        return working.groupby("init_date", as_index=False).first()
    working = working.sort_values(["init_date", "ranking_tornado_concern_score"], ascending=[True, False], kind="mergesort")
    return working.groupby("init_date", as_index=False).first()


def _coerce_bool(series: pd.Series, default: bool = False) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    truthy = {"true", "1", "yes", "y"}
    falsy = {"false", "0", "no", "n", ""}
    def _convert(value: Any) -> bool:
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        return bool(value)
    return series.map(_convert)


def _discover_artifact_paths(init_date: str) -> tuple[str, str]:
    settings = load_settings()
    paths = build_paths(settings)
    prediction = _latest_prediction_for_date(paths.outputs, init_date)
    verification = _verification_for_date(paths.verification, init_date)
    return (
        str(prediction.resolve()) if prediction is not None else "",
        str(verification.resolve()) if verification is not None else "",
    )


def select_failure_review_cases(
    frame: pd.DataFrame,
    *,
    ready_dates: set[str] | None = None,
    max_cases: int = 20,
    include_hail_outranks: bool = False,
    include_category_mismatches: bool = False,
    include_missed_tornado_first: bool = False,
) -> pd.DataFrame:
    working = _top_window_rows(frame)
    if ready_dates:
        working = working.loc[working["init_date"].astype(str).isin(ready_dates)].copy()
    if working.empty:
        return working

    if not any([include_hail_outranks, include_category_mismatches, include_missed_tornado_first]):
        include_hail_outranks = True
        include_category_mismatches = True
        include_missed_tornado_first = True

    working["hail_outranks_tornado_failure"] = _coerce_bool(working.get("hail_outranks_tornado_failure", pd.Series(dtype=object)))
    working["top_day_category_mismatch"] = _coerce_bool(working.get("top_day_category_mismatch", pd.Series(dtype=object)))
    best_tornado_valid_date = working.get("best_tornado_valid_date", pd.Series(dtype=object)).fillna("").astype(str)
    top_valid_date = working.get("top_valid_date", pd.Series(dtype=object)).fillna("").astype(str)
    working["missed_tornado_first"] = best_tornado_valid_date.ne("") & best_tornado_valid_date.ne(top_valid_date)

    selected_flags: list[pd.Series] = []
    if include_hail_outranks:
        selected_flags.append(working["hail_outranks_tornado_failure"])
    if include_category_mismatches:
        selected_flags.append(working["top_day_category_mismatch"])
    if include_missed_tornado_first:
        selected_flags.append(working["missed_tornado_first"])
    selector = selected_flags[0].copy()
    for extra in selected_flags[1:]:
        selector = selector | extra
    selected = working.loc[selector].copy()
    if selected.empty:
        return selected

    def _failure_types(row: pd.Series) -> list[str]:
        labels: list[str] = []
        if bool(row.get("hail_outranks_tornado_failure", False)):
            labels.append("hail_outranks")
        if bool(row.get("top_day_category_mismatch", False)):
            labels.append("category_mismatch")
        if bool(row.get("missed_tornado_first", False)):
            labels.append("missed_tornado_first")
        return labels

    selected["failure_types"] = selected.apply(_failure_types, axis=1)
    selected["primary_failure_type"] = selected["failure_types"].map(lambda values: values[0] if values else "other")
    priority_order = {"hail_outranks": 0, "category_mismatch": 1, "missed_tornado_first": 2, "other": 9}
    selected["priority_rank"] = selected["primary_failure_type"].map(priority_order).fillna(9).astype(int)
    selected["top_minus_best_tornado_score"] = pd.to_numeric(selected.get("top_minus_best_tornado_score", 0.0), errors="coerce")
    selected["ranking_tornado_concern_score"] = pd.to_numeric(selected.get("ranking_tornado_concern_score", selected.get("top_tornado_concern_score", 0.0)), errors="coerce")
    selected = selected.sort_values(
        ["priority_rank", "top_minus_best_tornado_score", "ranking_tornado_concern_score", "init_date"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).head(int(max_cases)).copy()
    selected["failure_types"] = selected["failure_types"].map(lambda values: ";".join(values))
    artifacts = selected["init_date"].astype(str).map(_discover_artifact_paths)
    selected["forecast_artifact_path"] = artifacts.map(lambda value: value[0])
    selected["verification_artifact_path"] = artifacts.map(lambda value: value[1])
    selected["top_ranked_observed_category"] = selected.get("top_observed_category", selected.get("observed_category", "unknown")).fillna("unknown")
    return selected.reset_index(drop=True)


def build_failure_review_packet(
    *,
    eval_csv: Path,
    ready_dates_file: Path | None,
    outdir: Path,
    max_cases: int,
    include_hail_outranks: bool,
    include_category_mismatches: bool,
    include_missed_tornado_first: bool,
    overwrite: bool,
) -> tuple[Path, Path]:
    summary_path = outdir / "tornado_concern_failure_review.md"
    csv_path = outdir / "tornado_concern_failure_review_cases.csv"
    if not overwrite:
        conflicts = [path for path in [summary_path, csv_path] if path.exists()]
        if conflicts:
            joined = ", ".join(str(path) for path in conflicts)
            raise FileExistsError(f"refusing to overwrite existing failure-review outputs: {joined}")

    frame = pd.read_csv(eval_csv)
    ready_dates = _load_ready_dates(ready_dates_file)
    selected = select_failure_review_cases(
        frame,
        ready_dates=ready_dates,
        max_cases=max_cases,
        include_hail_outranks=include_hail_outranks,
        include_category_mismatches=include_category_mismatches,
        include_missed_tornado_first=include_missed_tornado_first,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(csv_path, index=False)

    summary_counts = (
        selected.groupby("primary_failure_type", as_index=False)
        .agg(case_count=("init_date", "count"))
        .sort_values(["case_count", "primary_failure_type"], ascending=[False, True], kind="mergesort")
    ) if not selected.empty else pd.DataFrame(columns=["primary_failure_type", "case_count"])
    markdown = "\n".join(
        [
            "# Tornado Concern Failure Review",
            "",
            f"- Eval CSV: `{eval_csv}`",
            f"- Ready dates filter count: {len(ready_dates) if ready_dates else 'all rows from eval CSV'}",
            f"- Cases selected: {len(selected)}",
            f"- Max cases requested: {max_cases}",
            "",
            "## Failure Type Counts",
            "",
            _markdown_table(summary_counts, ["primary_failure_type", "case_count"]),
            "",
            "## Selected Cases",
            "",
            _markdown_table(
                selected,
                [
                    "init_date",
                    "primary_failure_type",
                    "failure_types",
                    "top_valid_date",
                    "top_ranked_observed_category",
                    "best_tornado_valid_date",
                    "top_minus_best_tornado_score",
                    "failure_driver",
                    "source_root_cause",
                    "core_root_cause",
                ],
            ),
        ]
    )
    summary_path.write_text(markdown + "\n", encoding="utf-8")
    return summary_path, csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a compact failure-review packet from tornado-concern eval outputs")
    parser.add_argument("--eval-csv", required=True, help="Ranked tornado_concern_eval CSV to review")
    parser.add_argument("--ready-dates-file", help="Optional ready-dates file to filter the review packet")
    parser.add_argument("--outdir", required=True, help="Output directory for the review packet")
    parser.add_argument("--max-cases", type=int, default=20, help="Maximum number of cases to include")
    parser.add_argument("--include-hail-outranks", action="store_true", help="Include hail-outranks-tornado failures")
    parser.add_argument("--include-category-mismatches", action="store_true", help="Include top-day category mismatches")
    parser.add_argument("--include-missed-tornado-first", action="store_true", help="Include tornado windows where the best tornado day was not ranked first")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing packet outputs in the outdir")
    args = parser.parse_args(argv)

    summary_path, csv_path = build_failure_review_packet(
        eval_csv=Path(args.eval_csv),
        ready_dates_file=Path(args.ready_dates_file) if args.ready_dates_file else None,
        outdir=Path(args.outdir),
        max_cases=int(args.max_cases),
        include_hail_outranks=bool(args.include_hail_outranks),
        include_category_mismatches=bool(args.include_category_mismatches),
        include_missed_tornado_first=bool(args.include_missed_tornado_first),
        overwrite=bool(args.overwrite),
    )
    print(f"summary_path={summary_path}")
    print(f"cases_csv={csv_path}")


if __name__ == "__main__":
    main()
