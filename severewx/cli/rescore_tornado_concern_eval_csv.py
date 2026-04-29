"""Rescore an existing tornado-concern eval CSV with variant settings."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from severewx.cli.tornado_concern_eval import (
    COMPONENT_VARIANTS,
    CORE_VARIANTS,
    RAW_CORE_VARIANTS,
    SCORE_VARIANTS,
    SOURCE_VARIANTS,
    _apply_component_variant_to_frame,
    _apply_core_variant_to_frame,
    _apply_raw_core_variant_to_frame,
    _apply_score_variant_to_frame,
    _apply_source_variant_to_frame,
    _write_ranked_csv,
    build_ranked_case_table,
    summarize_case_windows,
    summarize_failure_patterns,
)

_DERIVED_RANK_COLUMNS = {
    "day_rank_within_init",
    "window_max_rank",
    "final_score_minus_window_top",
    "top_valid_date",
    "top_observed_category",
    "top_tornado_concern_score",
    "top_learned_tornado_concern_prob",
    "window_has_tornado_or_sigtor_day",
    "window_has_hail_outbreak_day",
    "window_has_outbreak_day",
    "hail_outranks_tornado_failure",
    "non_outbreak_outranks_outbreak_failure",
    "top_day_category_mismatch",
    "best_tornado_valid_date",
    "best_tornado_score",
    "top_minus_best_tornado_score",
    "top_minus_best_tornado_tornado_signal",
    "top_minus_best_tornado_broad_signal",
    "top_minus_best_tornado_learned_term",
    "top_minus_best_tornado_core_top",
    "top_minus_best_tornado_effective_core_top",
    "top_minus_best_tornado_scp_top",
    "top_minus_best_tornado_joint_area",
    "top_minus_best_tornado_tornado_overlap_max",
    "top_minus_best_tornado_sig_tor_support_max",
    "top_minus_best_tornado_context_top",
    "top_minus_best_tornado_effective_core_to_context_ratio",
    "top_minus_best_tornado_core_compactness_proxy",
    "top_minus_best_tornado_core_percentile_proxy",
    "top_minus_best_tornado_core_alignment_proxy",
    "top_minus_best_tornado_core_diffuseness_proxy",
    "top_minus_best_tornado_core_purity_proxy",
    "top_minus_best_tornado_normalized_synoptic_support",
    "top_minus_best_tornado_overlap_compactness_proxy",
    "top_minus_best_tornado_overlap_purity_proxy",
    "top_minus_best_tornado_core_tornado_alignment_proxy",
    "top_minus_best_tornado_support_tornado_alignment_proxy",
    "top_minus_best_tornado_broad_contamination_proxy",
    "top_minus_best_tornado_scp_support_ratio",
    "top_minus_best_tornado_overlap_to_context_ratio",
    "top_minus_best_tornado_core_to_context_ratio",
    "best_sig_tor_valid_date",
    "best_sig_tor_score",
    "top_minus_best_sig_tor_score",
    "failure_driver",
    "source_root_cause",
    "core_root_cause",
}


def _metric_rows(window_summary: pd.DataFrame) -> list[tuple[str, object]]:
    real_summary = window_summary.loc[window_summary["is_real_ingest"].fillna(False)].copy()
    tornado_windows = real_summary.loc[real_summary["best_tornado_valid_date"].astype(str).ne("")]
    return [
        ("real_ingest_windows", int(len(real_summary))),
        ("hail_outranks_tornado_failure_real", int(real_summary["hail_outranks_tornado_failure"].fillna(False).sum())),
        ("non_outbreak_outranks_outbreak_failure_real", int(real_summary["non_outbreak_outranks_outbreak_failure"].fillna(False).sum())),
        ("top_day_category_mismatch_real", int(real_summary["top_day_category_mismatch"].fillna(False).sum())),
        ("tornado_day_ranked_first_real", int(real_summary["top_observed_tornado_outbreak"].fillna(0).astype(int).sum())),
        (
            "sig_tor_day_ranked_first_real",
            int(real_summary["top_observed_significant_tornado_support"].fillna(0).astype(int).sum()),
        ),
        ("mean_top_minus_best_tornado_margin_real", float(tornado_windows["top_minus_best_tornado_score"].mean())),
        ("median_top_minus_best_tornado_margin_real", float(tornado_windows["top_minus_best_tornado_score"].median())),
    ]


def _write_markdown(
    path: Path,
    *,
    input_csv: Path,
    raw_core_variant: str,
    core_variant: str,
    source_variant: str,
    component_variant: str,
    score_variant: str,
    window_summary: pd.DataFrame,
    failure_summary: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tornado Concern Rescore",
        "",
        f"- input_csv: `{input_csv}`",
        f"- raw_core_variant: `{raw_core_variant}`",
        f"- core_variant: `{core_variant}`",
        f"- source_variant: `{source_variant}`",
        f"- component_variant: `{component_variant}`",
        f"- score_variant: `{score_variant}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for metric, value in _metric_rows(window_summary):
        if isinstance(value, float):
            lines.append(f"| {metric} | {value:.6f} |")
        else:
            lines.append(f"| {metric} | {value} |")
    lines.extend(["", "## Failure Patterns", "", "| pattern | count_real_windows |", "|---|---:|"])
    for row in failure_summary.to_dict(orient="records"):
        lines.append(f"| {row['pattern']} | {row['count_real_windows']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rescore_eval_csv(
    input_csv: Path,
    *,
    raw_core_variant: str = "baseline",
    core_variant: str = "baseline",
    source_variant: str = "baseline",
    component_variant: str = "baseline",
    score_variant: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(input_csv)
    frame = frame.drop(columns=[column for column in _DERIVED_RANK_COLUMNS if column in frame.columns])
    for column, default in {
        "preference_adjustment": 0.0,
        "preference_applied": False,
        "raw_tornado_prob_max": 0.0,
    }.items():
        if column not in frame.columns:
            frame[column] = default
    rescored = _apply_raw_core_variant_to_frame(frame, raw_core_variant)
    rescored = _apply_core_variant_to_frame(rescored, core_variant)
    rescored = _apply_source_variant_to_frame(rescored, source_variant)
    rescored = _apply_component_variant_to_frame(rescored, component_variant)
    rescored = _apply_score_variant_to_frame(rescored, score_variant)
    rescored["ranking_tornado_concern_score"] = rescored["tornado_concern_score"].astype(float)
    ranked = build_ranked_case_table(rescored)
    window_summary = summarize_case_windows(rescored)
    failure_summary = summarize_failure_patterns(window_summary)
    return ranked, window_summary, failure_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--raw-core-variant", choices=RAW_CORE_VARIANTS, default="baseline")
    parser.add_argument("--core-variant", choices=CORE_VARIANTS, default="baseline")
    parser.add_argument("--source-variant", choices=SOURCE_VARIANTS, default="baseline")
    parser.add_argument("--component-variant", choices=COMPONENT_VARIANTS, default="baseline")
    parser.add_argument("--score-variant", choices=SCORE_VARIANTS, default="baseline")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    ranked, window_summary, failure_summary = rescore_eval_csv(
        input_csv,
        raw_core_variant=args.raw_core_variant,
        core_variant=args.core_variant,
        source_variant=args.source_variant,
        component_variant=args.component_variant,
        score_variant=args.score_variant,
    )
    _write_ranked_csv(Path(args.output_csv), ranked)
    _write_markdown(
        Path(args.output_md),
        input_csv=input_csv,
        raw_core_variant=args.raw_core_variant,
        core_variant=args.core_variant,
        source_variant=args.source_variant,
        component_variant=args.component_variant,
        score_variant=args.score_variant,
        window_summary=window_summary,
        failure_summary=failure_summary,
    )
    for metric, value in _metric_rows(window_summary):
        print(f"{metric}={value}")


if __name__ == "__main__":
    main()
