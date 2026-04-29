"""Build focused diagnostics for tornado-concern top-day failures."""

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


def _failure_bucket(top_row: pd.Series, best_tornado_row: pd.Series | None) -> str:
    if _boolish(top_row.get("hail_outranks_tornado_failure", False)):
        return "hail_over_tornado"
    if _boolish(top_row.get("non_outbreak_outranks_outbreak_failure", False)):
        return "non_outbreak_over_outbreak"
    if str(top_row.get("failure_driver", "")).lower() == "tornado_signal_near_zero_for_both":
        return "tornado_signal_near_zero"
    best_signal = _float(best_tornado_row, "tornado_signal") if best_tornado_row is not None else 0.0
    if _float(top_row, "tornado_signal") < 0.01 and best_signal < 0.01:
        return "tornado_signal_near_zero"
    return "tornado_signal_near_zero"


def build_failure_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    if "day_rank_within_init" not in working.columns:
        working["day_rank_within_init"] = working.groupby("init_date").cumcount() + 1
    failures = working.loc[
        working["day_rank_within_init"].astype(int).eq(1)
        & working.get("top_day_category_mismatch", pd.Series(False, index=working.index)).fillna(False).map(_boolish)
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, top_row in failures.iterrows():
        init_date = str(top_row.get("init_date", ""))
        group = working.loc[working["init_date"].astype(str).eq(init_date)].copy()
        best_tornado_valid_date = str(top_row.get("best_tornado_valid_date", "") or "")
        best_matches = group.loc[group["valid_date"].astype(str).eq(best_tornado_valid_date)] if best_tornado_valid_date else pd.DataFrame()
        best_row = best_matches.iloc[0] if not best_matches.empty else None
        top_hail_support = _float(top_row, "hail_overlap_max", _float(top_row, "penalty_top"))
        best_hail_support = _float(best_row, "hail_overlap_max", _float(best_row, "penalty_top")) if best_row is not None else float("nan")
        top_wind_support = _float(top_row, "wind_overlap_max", _float(top_row, "raw_wind_prob_max"))
        best_wind_support = _float(best_row, "wind_overlap_max", _float(best_row, "raw_wind_prob_max")) if best_row is not None else float("nan")
        rows.append(
            {
                "init_date": init_date,
                "failure_bucket": _failure_bucket(top_row, best_row),
                "failure_driver": str(top_row.get("failure_driver", "")),
                "top_valid_date": str(top_row.get("top_valid_date", top_row.get("valid_date", ""))),
                "best_tornado_valid_date": best_tornado_valid_date,
                "top_observed_category": str(top_row.get("top_observed_category", top_row.get("observed_category", ""))),
                "top_score": _float(top_row, "ranking_tornado_concern_score", _float(top_row, "tornado_concern_score")),
                "best_tornado_score": _float(top_row, "best_tornado_score", float("nan")),
                "top_minus_best_tornado_score": _float(top_row, "top_minus_best_tornado_score", float("nan")),
                "top_tornado_support": _float(top_row, "tornado_overlap_max"),
                "best_tornado_support": _float(best_row, "tornado_overlap_max") if best_row is not None else float("nan"),
                "top_sig_tor_support": _float(top_row, "sig_tor_support_max"),
                "best_sig_tor_support": _float(best_row, "sig_tor_support_max") if best_row is not None else float("nan"),
                "top_hail_support": top_hail_support,
                "best_tornado_hail_support": best_hail_support,
                "top_wind_support": top_wind_support,
                "best_tornado_wind_support": best_wind_support,
                "top_learned_tornado_prob": _float(top_row, "learned_tornado_concern_prob", float("nan")),
                "best_tornado_learned_prob": _float(best_row, "learned_tornado_concern_prob", float("nan")) if best_row is not None else float("nan"),
                "top_raw_core_score": _float(top_row, "raw_base_score_before_variant", _float(top_row, "base_score")),
                "best_tornado_raw_core_score": _float(best_row, "raw_base_score_before_variant", _float(best_row, "base_score")) if best_row is not None else float("nan"),
                "top_core_top": _float(top_row, "core_top"),
                "best_tornado_core_top": _float(best_row, "core_top") if best_row is not None else float("nan"),
                "top_scp_top": _float(top_row, "scp_top"),
                "best_tornado_scp_top": _float(best_row, "scp_top") if best_row is not None else float("nan"),
                "top_context_top": _float(top_row, "context_top"),
                "best_tornado_context_top": _float(best_row, "context_top") if best_row is not None else float("nan"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["failure_bucket", "init_date"], kind="mergesort").reset_index(drop=True)


WEAK_SIGNAL_BUCKETS = [
    "missing_or_weak_sig_tor_support",
    "weak_scp_proxy",
    "weak_tornado_favored_overlap",
    "learned_probability_overpowering_ingredients",
    "broad_hail_or_wind_context_dominates",
    "weak_signal_both_top_and_best_tornado_day",
]


def _has_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in frame.columns for column in columns)


def _missing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


def _has_any_column_set(frame: pd.DataFrame, alternatives: list[list[str]]) -> bool:
    return any(all(column in frame.columns for column in columns) for columns in alternatives)


def _missing_alternative_note(frame: pd.DataFrame, alternatives: list[list[str]]) -> list[str]:
    if _has_any_column_set(frame, alternatives):
        return []
    return sorted(set(column for columns in alternatives for column in columns))


def _derived_best_value(row: pd.Series, *, direct: str, top: str, delta: str, default: float = 0.0) -> float:
    if direct in row.index:
        return _float(row, direct, default)
    if top in row.index and delta in row.index:
        return _float(row, top, default) - _float(row, delta, 0.0)
    return default


def build_weak_signal_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if frame.empty:
        return pd.DataFrame(), {}
    working = frame.copy()
    if "day_rank_within_init" not in working.columns:
        score_column = "ranking_tornado_concern_score" if "ranking_tornado_concern_score" in working.columns else "tornado_concern_score"
        working = working.sort_values(["init_date", score_column], ascending=[True, False], kind="mergesort").copy()
        working["day_rank_within_init"] = working.groupby("init_date").cumcount() + 1

    required_by_bucket = {
        "missing_or_weak_sig_tor_support": [["best_tornado_sig_tor_support_max"], ["sig_tor_support_max", "top_minus_best_tornado_sig_tor_support_max"]],
        "weak_scp_proxy": [["best_tornado_scp_top"], ["scp_top", "top_minus_best_tornado_scp_top"]],
        "weak_tornado_favored_overlap": [["best_tornado_tornado_overlap_max"], ["tornado_overlap_max", "top_minus_best_tornado_tornado_overlap_max"]],
        "learned_probability_overpowering_ingredients": [["top_minus_best_tornado_learned_term"]],
        "broad_hail_or_wind_context_dominates": [["top_minus_best_tornado_context_top"]],
        "weak_signal_both_top_and_best_tornado_day": [["top_tornado_signal", "best_tornado_tornado_signal"], ["tornado_signal", "top_minus_best_tornado_tornado_signal"]],
    }
    notes = {bucket: _missing_alternative_note(working, alternatives) for bucket, alternatives in required_by_bucket.items()}
    failure_mask = working["day_rank_within_init"].astype(int).eq(1)
    failure_columns = [col for col in ["top_day_category_mismatch", "hail_outranks_tornado_failure", "non_outbreak_outranks_outbreak_failure"] if col in working.columns]
    if failure_columns:
        failure_flags = pd.DataFrame(
            {column: working[column].fillna(False).map(_boolish) for column in failure_columns},
            index=working.index,
        )
        failure_mask &= failure_flags.any(axis=1)
    failures = working.loc[failure_mask].copy()

    rows: list[dict[str, Any]] = []
    for _, row in failures.iterrows():
        buckets: list[str] = []
        unavailable: list[str] = []
        if notes["missing_or_weak_sig_tor_support"]:
            unavailable.extend(notes["missing_or_weak_sig_tor_support"])
        else:
            best_sig_tor = _derived_best_value(
                row,
                direct="best_tornado_sig_tor_support_max",
                top="sig_tor_support_max",
                delta="top_minus_best_tornado_sig_tor_support_max",
            )
        if not notes["missing_or_weak_sig_tor_support"] and (best_sig_tor < 1.15 or _float(row, "top_minus_best_tornado_sig_tor_support_max") <= -0.15):
            buckets.append("missing_or_weak_sig_tor_support")

        if notes["weak_scp_proxy"]:
            unavailable.extend(notes["weak_scp_proxy"])
        else:
            best_scp = _derived_best_value(row, direct="best_tornado_scp_top", top="scp_top", delta="top_minus_best_tornado_scp_top")
        if not notes["weak_scp_proxy"] and (best_scp < 0.18 or _float(row, "top_minus_best_tornado_scp_top") <= -0.08):
            buckets.append("weak_scp_proxy")

        if notes["weak_tornado_favored_overlap"]:
            unavailable.extend(notes["weak_tornado_favored_overlap"])
        else:
            best_overlap = _derived_best_value(
                row,
                direct="best_tornado_tornado_overlap_max",
                top="tornado_overlap_max",
                delta="top_minus_best_tornado_tornado_overlap_max",
            )
        if not notes["weak_tornado_favored_overlap"] and (best_overlap < 1.40 or _float(row, "top_minus_best_tornado_tornado_overlap_max") <= -0.20):
            buckets.append("weak_tornado_favored_overlap")

        if notes["learned_probability_overpowering_ingredients"]:
            unavailable.extend(notes["learned_probability_overpowering_ingredients"])
        elif _float(row, "top_minus_best_tornado_learned_term") >= 0.08:
            buckets.append("learned_probability_overpowering_ingredients")

        if notes["broad_hail_or_wind_context_dominates"]:
            unavailable.extend(notes["broad_hail_or_wind_context_dominates"])
        else:
            top_hail_advantage = _float(row, "hail_overlap_max") - _float(row, "best_tornado_hail_support", _float(row, "best_tornado_hail_overlap_max"))
            top_wind_advantage = _float(row, "wind_overlap_max") - _float(row, "best_tornado_wind_support", _float(row, "best_tornado_wind_overlap_max"))
            if (
                _boolish(row.get("hail_outranks_tornado_failure", False))
                or _float(row, "top_minus_best_tornado_context_top") >= 0.10
                or _float(row, "top_minus_best_tornado_broad_contamination_proxy") >= 0.10
                or max(top_hail_advantage, top_wind_advantage) >= 0.25
            ):
                buckets.append("broad_hail_or_wind_context_dominates")

        if notes["weak_signal_both_top_and_best_tornado_day"]:
            unavailable.extend(notes["weak_signal_both_top_and_best_tornado_day"])
        else:
            top_signal = _float(row, "top_tornado_signal", _float(row, "tornado_signal"))
            best_signal = _derived_best_value(
                row,
                direct="best_tornado_tornado_signal",
                top="tornado_signal",
                delta="top_minus_best_tornado_tornado_signal",
            )
            if str(row.get("failure_driver", "")).lower() == "tornado_signal_near_zero_for_both" or (top_signal < 0.01 and best_signal < 0.01):
                buckets.append("weak_signal_both_top_and_best_tornado_day")

        buckets = list(dict.fromkeys(buckets))
        unavailable = sorted(set(unavailable))
        rows.append(
            {
                "init_date": str(row.get("init_date", "")),
                "top_valid_date": str(row.get("top_valid_date", row.get("valid_date", ""))),
                "best_tornado_valid_date": str(row.get("best_tornado_valid_date", "")),
                "top_observed_category": str(row.get("top_observed_category", row.get("observed_category", ""))),
                "top_minus_best_tornado_score": _float(row, "top_minus_best_tornado_score", float("nan")),
                "failure_driver": str(row.get("failure_driver", "")),
                "weak_signal_buckets": ";".join(buckets),
                "not_available": ";".join(unavailable),
                **{bucket: bucket in buckets for bucket in WEAK_SIGNAL_BUCKETS},
            }
        )
    if not rows:
        return pd.DataFrame(), notes
    result = pd.DataFrame(rows)
    return result.sort_values(["top_minus_best_tornado_score", "init_date"], ascending=[False, True], kind="mergesort").reset_index(drop=True), notes


def _write_weak_signal_markdown(path: Path, diagnostics: pd.DataFrame, notes: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tornado Concern Weak Signal Diagnostics",
        "",
        f"- relevant_failures: {len(diagnostics)}",
        "",
        "## Buckets",
        "",
    ]
    if diagnostics.empty:
        lines.append("No relevant top-day failure rows found.")
    else:
        for bucket in WEAK_SIGNAL_BUCKETS:
            lines.append(f"- {bucket}: {int(diagnostics[bucket].fillna(False).sum())}")
    unavailable = {bucket: columns for bucket, columns in notes.items() if columns}
    if unavailable:
        lines.extend(["", "## Not Available", ""])
        for bucket, columns in unavailable.items():
            lines.append(f"- {bucket}: missing {','.join(columns)}")
    lines.extend(
        [
            "",
            "## Ranked Failures",
            "",
            "| init_date | top_valid_date | best_tornado_valid_date | margin | buckets | not_available |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in diagnostics.to_dict(orient="records"):
        lines.append(
            "| {init_date} | {top_valid_date} | {best_tornado_valid_date} | {margin:.4f} | {buckets} | {missing} |".format(
                init_date=row.get("init_date", ""),
                top_valid_date=row.get("top_valid_date", ""),
                best_tornado_valid_date=row.get("best_tornado_valid_date", ""),
                margin=float(row.get("top_minus_best_tornado_score", 0.0) or 0.0),
                buckets=row.get("weak_signal_buckets", ""),
                missing=row.get("not_available", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, diagnostics: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tornado Concern Failure Diagnostics",
        "",
        f"- mismatch_windows: {len(diagnostics)}",
        "",
        "## Buckets",
        "",
    ]
    if diagnostics.empty:
        lines.append("No top-day mismatch windows found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    counts = diagnostics["failure_bucket"].value_counts().sort_index()
    for bucket, count in counts.items():
        lines.append(f"- {bucket}: {int(count)}")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| init_date | bucket | top_valid_date | best_tornado_valid_date | top_category | margin | driver |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for row in diagnostics.to_dict(orient="records"):
        lines.append(
            "| {init_date} | {bucket} | {top_valid} | {best_tor} | {cat} | {margin:.4f} | {driver} |".format(
                init_date=row.get("init_date", ""),
                bucket=row.get("failure_bucket", ""),
                top_valid=row.get("top_valid_date", ""),
                best_tor=row.get("best_tornado_valid_date", ""),
                cat=row.get("top_observed_category", ""),
                margin=float(row.get("top_minus_best_tornado_score", 0.0) or 0.0),
                driver=row.get("failure_driver", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--weak-signal-output-csv")
    parser.add_argument("--weak-signal-output-md")
    args = parser.parse_args(argv)

    eval_frame = pd.read_csv(args.eval_csv)
    diagnostics = build_failure_diagnostics(eval_frame)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_csv, index=False)
    _write_markdown(Path(args.output_md), diagnostics)
    if args.weak_signal_output_csv or args.weak_signal_output_md:
        weak_diagnostics, notes = build_weak_signal_diagnostics(eval_frame)
        if args.weak_signal_output_csv:
            weak_csv = Path(args.weak_signal_output_csv)
            weak_csv.parent.mkdir(parents=True, exist_ok=True)
            weak_diagnostics.to_csv(weak_csv, index=False)
        if args.weak_signal_output_md:
            _write_weak_signal_markdown(Path(args.weak_signal_output_md), weak_diagnostics, notes)
    print(f"mismatch_windows={len(diagnostics)}")
    if not diagnostics.empty:
        for bucket, count in diagnostics["failure_bucket"].value_counts().sort_index().items():
            print(f"{bucket}={int(count)}")


if __name__ == "__main__":
    main()
