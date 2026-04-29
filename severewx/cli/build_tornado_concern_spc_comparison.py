"""Build a lightweight case-level SPC comparison scaffold for tornado concern."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


SPC_COLUMNS = ["date", "spc_day1_category", "spc_tornado_prob", "spc_source_url", "spc_notes"]


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _read_optional_csv(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _base_dates_from_eval(eval_frame: pd.DataFrame) -> pd.DataFrame:
    if eval_frame.empty:
        return pd.DataFrame(columns=["date"])
    date_col = "init_date" if "init_date" in eval_frame.columns else "date"
    if date_col not in eval_frame.columns:
        raise ValueError("eval CSV must include init_date or date")
    top_rows = eval_frame.copy()
    if "day_rank_within_init" in top_rows.columns:
        top_rows = top_rows[top_rows["day_rank_within_init"].astype(str).eq("1") | top_rows["day_rank_within_init"].eq(1)]
    records: list[dict[str, Any]] = []
    for row in top_rows.to_dict(orient="records"):
        records.append(
            {
                "date": _safe_text(row.get(date_col)),
                "model_top_valid_date": _safe_text(row.get("top_valid_date", row.get("valid_date", ""))),
                "model_top_observed_category": _safe_text(row.get("top_observed_category", row.get("observed_category", ""))),
                "model_top_score": row.get("ranking_tornado_concern_score", row.get("tornado_concern_score", "")),
                "hail_outranks_tornado_failure": row.get("hail_outranks_tornado_failure", ""),
                "top_day_category_mismatch": row.get("top_day_category_mismatch", ""),
                "tornado_day_ranked_first": row.get("tornado_day_ranked_first", ""),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["date"])
    return frame.drop_duplicates(subset=["date"], keep="first")


def _merge_public_candidates(base: pd.DataFrame, public_candidates: pd.DataFrame) -> pd.DataFrame:
    if base.empty or public_candidates.empty:
        return base
    columns = [
        column
        for column in [
            "date",
            "public_ready",
            "failure_reasons",
            "max_tornado_concern_prob",
            "observed_tornado_relevant",
            "observed_significant_tornado_support",
            "observed_outbreak_relevant",
            "observed_categories",
        ]
        if column in public_candidates.columns
    ]
    if "date" not in columns:
        return base
    return base.merge(public_candidates.loc[:, columns], on="date", how="left")


def _normalize_spc(spc_frame: pd.DataFrame) -> pd.DataFrame:
    if spc_frame.empty:
        return pd.DataFrame(columns=SPC_COLUMNS)
    if "date" not in spc_frame.columns:
        raise ValueError("SPC CSV must include date")
    normalized = spc_frame.copy()
    if "spc_category" in normalized.columns and "spc_day1_category" not in normalized.columns:
        normalized["spc_day1_category"] = normalized["spc_category"]
    for column in SPC_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    return normalized.loc[:, SPC_COLUMNS]


def build_spc_comparison_frame(*, eval_csv: Path, public_candidates_csv: Path | None, spc_csv: Path | None) -> pd.DataFrame:
    eval_frame = pd.read_csv(eval_csv)
    public_candidates = _read_optional_csv(public_candidates_csv)
    spc_frame = _normalize_spc(_read_optional_csv(spc_csv))
    base = _base_dates_from_eval(eval_frame)
    base = _merge_public_candidates(base, public_candidates)
    if not spc_frame.empty:
        base = base.merge(spc_frame, on="date", how="left")
    else:
        for column in SPC_COLUMNS:
            if column != "date":
                base[column] = ""
    base["spc_label_status"] = base["spc_day1_category"].map(lambda value: "labeled" if _safe_text(value).strip() else "needs_spc_label")
    preferred = [
        "date",
        "spc_label_status",
        "spc_day1_category",
        "spc_tornado_prob",
        "spc_source_url",
        "model_top_valid_date",
        "model_top_observed_category",
        "model_top_score",
        "public_ready",
        "failure_reasons",
        "observed_categories",
        "spc_notes",
    ]
    columns = [column for column in preferred if column in base.columns] + [column for column in base.columns if column not in preferred]
    return base.loc[:, columns].sort_values("date", kind="mergesort").reset_index(drop=True)


def _write_markdown(path: Path, frame: pd.DataFrame) -> None:
    labeled = int(frame["spc_label_status"].eq("labeled").sum()) if "spc_label_status" in frame.columns else 0
    lines = [
        "# Tornado Concern SPC Comparison Scaffold",
        "",
        "Case-level benchmark scaffold only. This is intentionally separate from training inputs.",
        "",
        f"- cases: {len(frame)}",
        f"- spc_labeled: {labeled}",
        f"- needs_spc_label: {len(frame) - labeled}",
        "",
        "## Next Manual Fields",
        "",
        "Fill spc_day1_category, spc_tornado_prob, spc_source_url, and spc_notes from archived SPC outlook metadata.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--public-candidates-csv")
    parser.add_argument("--spc-csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    frame = build_spc_comparison_frame(
        eval_csv=Path(args.eval_csv),
        public_candidates_csv=Path(args.public_candidates_csv) if args.public_candidates_csv else None,
        spc_csv=Path(args.spc_csv) if args.spc_csv else None,
    )
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    _write_markdown(output_md, frame)
    print(f"cases={len(frame)}")
    print(f"needs_spc_label={int(frame['spc_label_status'].eq('needs_spc_label').sum()) if not frame.empty else 0}")


if __name__ == "__main__":
    main()
