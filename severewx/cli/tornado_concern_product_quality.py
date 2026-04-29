"""Score and organize generated tornado-concern product bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd


METRIC_FIELDS = (
    "public_display_grid_cells_ge_02pct",
    "public_display_grid_cells_ge_05pct",
    "public_display_grid_cells_ge_10pct",
    "public_display_grid_cells_ge_20pct",
    "public_display_grid_cells_ge_35pct",
    "display_object_count_ge_02pct",
    "display_largest_object_cells_ge_02pct",
    "contour_label_count",
)


def _load_product_metadata(products_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(products_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row = {
            "date": str(payload.get("date", "")),
            "valid_date": str(payload.get("valid_date", "")),
            "metadata_path": str(path),
            "image_path": str(payload.get("main_image_path", "")),
            "summary_path": str(payload.get("summary_path", "")),
            "map_domain": str(payload.get("map_domain", "")),
            "display_preset_effective": str(payload.get("display_preset_effective", "")),
            "max_tornado_concern_prob": float(payload.get("max_tornado_concern_prob", 0.0) or 0.0),
            "map_extent": ",".join(str(value) for value in payload.get("map_extent", [])),
        }
        for field in METRIC_FIELDS:
            row[field] = int(payload.get(field, 0) or 0)
        rows.append(row)
    return pd.DataFrame(rows)


def _quality_issues(row: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    ge02 = int(row.get("public_display_grid_cells_ge_02pct", 0) or 0)
    ge10 = int(row.get("public_display_grid_cells_ge_10pct", 0) or 0)
    ge35 = int(row.get("public_display_grid_cells_ge_35pct", 0) or 0)
    largest = int(row.get("display_largest_object_cells_ge_02pct", 0) or 0)
    labels = int(row.get("contour_label_count", 0) or 0)
    max_value = float(row.get("max_tornado_concern_prob", 0.0) or 0.0)
    if ge02 < 160 or max_value < 0.10:
        issues.append("weak_or_sparse_display")
    if ge10 == 0:
        issues.append("no_10pct_contour")
    if ge02 > 1600 or largest > 1300:
        issues.append("broad_low_end_footprint")
    if labels > 18:
        issues.append("label_clutter")
    if max_value >= 0.45 and ge35 < 5:
        issues.append("tiny_high_end_area")
    return issues


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    issues = _quality_issues(row)
    ge10 = int(row.get("public_display_grid_cells_ge_10pct", 0) or 0)
    ge35 = int(row.get("public_display_grid_cells_ge_35pct", 0) or 0)
    labels = int(row.get("contour_label_count", 0) or 0)
    score = 100.0
    score += min(20.0, ge10 / 20.0)
    score += min(10.0, ge35 / 10.0)
    score -= 18.0 * len(issues)
    score -= max(0, labels - 12) * 1.5
    return {
        **row,
        "quality_score": round(score, 3),
        "quality_issues": ";".join(issues),
    }


def _queue_for_issues(issues: str) -> str:
    issue_set = {issue for issue in str(issues or "").split(";") if issue}
    if "broad_low_end_footprint" in issue_set:
        return "broad_low_end_footprint"
    if "weak_or_sparse_display" in issue_set or "no_10pct_contour" in issue_set:
        return "weak_or_sparse_display"
    if "tiny_high_end_area" in issue_set:
        return "tiny_high_end_area"
    if issue_set:
        return "other_map_tuning"
    return ""


def _suggested_action(queue: str) -> str:
    if queue == "broad_low_end_footprint":
        return "tighten display-only low-end footprint filtering, then inspect environment construction if persistent"
    if queue == "weak_or_sparse_display":
        return "inspect support-envelope continuity and weak-case display treatment"
    if queue == "tiny_high_end_area":
        return "inspect high-end support continuity and label placement around compact cores"
    if queue:
        return "manual product review"
    return ""


def score_products(products_dir: Path) -> pd.DataFrame:
    frame = _load_product_metadata(products_dir)
    if frame.empty:
        return frame
    rows = [_score_row(row) for row in frame.to_dict(orient="records")]
    return pd.DataFrame(rows).sort_values(["quality_score", "date"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def build_failure_queue(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "quality_issues" not in frame:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        issues = str(row.get("quality_issues", "") or "")
        queue = _queue_for_issues(issues)
        if not queue:
            continue
        rows.append(
            {
                "date": row.get("date", ""),
                "valid_date": row.get("valid_date", ""),
                "queue": queue,
                "quality_score": row.get("quality_score", 0.0),
                "quality_issues": issues,
                "display_preset_effective": row.get("display_preset_effective", ""),
                "max_tornado_concern_prob": row.get("max_tornado_concern_prob", 0.0),
                "public_display_grid_cells_ge_02pct": row.get("public_display_grid_cells_ge_02pct", 0),
                "public_display_grid_cells_ge_10pct": row.get("public_display_grid_cells_ge_10pct", 0),
                "public_display_grid_cells_ge_35pct": row.get("public_display_grid_cells_ge_35pct", 0),
                "display_largest_object_cells_ge_02pct": row.get("display_largest_object_cells_ge_02pct", 0),
                "contour_label_count": row.get("contour_label_count", 0),
                "image_path": row.get("image_path", ""),
                "metadata_path": row.get("metadata_path", ""),
                "suggested_action": _suggested_action(queue),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["queue", "quality_score", "date"], ascending=[True, True, True], kind="mergesort").reset_index(drop=True)


def _copy_product_files(frame: pd.DataFrame, output_dir: Path, *, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in frame.head(limit).iterrows():
        for field in ("image_path", "summary_path", "metadata_path"):
            source = Path(str(row.get(field, "")))
            if source.exists():
                shutil.copy2(source, output_dir / source.name)


def _write_markdown(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tornado Concern Product Quality",
        "",
        f"- products_scored: {len(frame)}",
        f"- products_with_issues: {int(frame['quality_issues'].astype(str).ne('').sum()) if not frame.empty else 0}",
        "",
        "| rank | date | score | preset | max | ge02 | ge10 | ge35 | labels | issues |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(frame.to_dict(orient="records"), start=1):
        lines.append(
            "| {rank} | {date} | {score:.1f} | {preset} | {maxv:.2f} | {ge02} | {ge10} | {ge35} | {labels} | {issues} |".format(
                rank=rank,
                date=row.get("date", ""),
                score=float(row.get("quality_score", 0.0) or 0.0),
                preset=row.get("display_preset_effective", ""),
                maxv=float(row.get("max_tornado_concern_prob", 0.0) or 0.0),
                ge02=int(row.get("public_display_grid_cells_ge_02pct", 0) or 0),
                ge10=int(row.get("public_display_grid_cells_ge_10pct", 0) or 0),
                ge35=int(row.get("public_display_grid_cells_ge_35pct", 0) or 0),
                labels=int(row.get("contour_label_count", 0) or 0),
                issues=row.get("quality_issues", "") or "none",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison(baseline_dir: Path, candidate_frame: pd.DataFrame, output_path: Path) -> None:
    baseline = score_products(baseline_dir)
    if baseline.empty or candidate_frame.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    keep = ["date", "quality_score", *METRIC_FIELDS, "map_extent"]
    merged = baseline[keep].merge(candidate_frame[keep], on="date", suffixes=("_before", "_after"), how="inner")
    for field in ("quality_score", *METRIC_FIELDS):
        merged[f"{field}_delta"] = merged[f"{field}_after"].astype(float) - merged[f"{field}_before"].astype(float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)


def _write_failure_queue_markdown(path: Path, queue: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Tornado Concern Hybrid Product Failure Queue",
        "",
        f"- queued_cases: {len(queue)}",
    ]
    if queue.empty:
        lines.extend(["", "No product-tuning cases were queued."])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    counts = queue["queue"].value_counts().sort_index()
    lines.extend(["", "## Queue Counts", ""])
    for name, count in counts.items():
        lines.append(f"- {name}: {int(count)}")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| date | queue | score | max | ge02 | ge10 | ge35 | largest | labels | issues | suggested_action |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in queue.to_dict(orient="records"):
        lines.append(
            "| {date} | {queue} | {score:.1f} | {maxv:.2f} | {ge02} | {ge10} | {ge35} | {largest} | {labels} | {issues} | {action} |".format(
                date=row.get("date", ""),
                queue=row.get("queue", ""),
                score=float(row.get("quality_score", 0.0) or 0.0),
                maxv=float(row.get("max_tornado_concern_prob", 0.0) or 0.0),
                ge02=int(row.get("public_display_grid_cells_ge_02pct", 0) or 0),
                ge10=int(row.get("public_display_grid_cells_ge_10pct", 0) or 0),
                ge35=int(row.get("public_display_grid_cells_ge_35pct", 0) or 0),
                largest=int(row.get("display_largest_object_cells_ge_02pct", 0) or 0),
                labels=int(row.get("contour_label_count", 0) or 0),
                issues=row.get("quality_issues", ""),
                action=row.get("suggested_action", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--best-dir")
    parser.add_argument("--needs-dir")
    parser.add_argument("--best-count", type=int, default=10)
    parser.add_argument("--needs-count", type=int, default=10)
    parser.add_argument("--baseline-dir")
    parser.add_argument("--comparison-csv")
    parser.add_argument("--failure-queue-csv")
    parser.add_argument("--failure-queue-md")
    args = parser.parse_args(argv)

    frame = score_products(Path(args.products_dir))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    _write_markdown(Path(args.output_md), frame)
    if args.best_dir:
        _copy_product_files(frame, Path(args.best_dir), limit=int(args.best_count))
    if args.needs_dir:
        needs = frame.sort_values(["quality_score", "date"], ascending=[True, True], kind="mergesort")
        _copy_product_files(needs, Path(args.needs_dir), limit=int(args.needs_count))
    if args.baseline_dir and args.comparison_csv:
        _write_comparison(Path(args.baseline_dir), frame, Path(args.comparison_csv))
    if args.failure_queue_csv:
        queue = build_failure_queue(frame)
        output_queue_csv = Path(args.failure_queue_csv)
        output_queue_csv.parent.mkdir(parents=True, exist_ok=True)
        queue.to_csv(output_queue_csv, index=False)
        if args.failure_queue_md:
            _write_failure_queue_markdown(Path(args.failure_queue_md), queue)
    print(f"products_scored={len(frame)}")
    print(f"products_with_issues={int(frame['quality_issues'].astype(str).ne('').sum()) if not frame.empty else 0}")


if __name__ == "__main__":
    main()
