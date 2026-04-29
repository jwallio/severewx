"""Prepare review and showcase artifacts for tornado-concern public prototypes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


REVIEW_STATUS = "needs_human_review"
BLOCKED_CATEGORIES = ("guardrail_heavy", "no_public_signal", "overbroad_footprint", "other")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _product_paths(products_dir: Path, date: str, cycle: str = "00") -> dict[str, str]:
    new_matches = sorted(products_dir.glob(f"tornado_concern_init_{date}_{cycle}z_valid_*.png"))
    if new_matches:
        image_path = new_matches[0]
        stem = image_path.with_suffix("").name
        return {
            "image_path": str(image_path),
            "summary_path": str(products_dir / f"{stem}.md"),
            "metadata_path": str(products_dir / f"{stem}.json"),
        }
    stem = f"tornado_concern_{date}_{cycle}z"
    return {
        "image_path": str(products_dir / f"{stem}.png"),
        "summary_path": str(products_dir / f"{stem}.md"),
        "metadata_path": str(products_dir / f"{stem}.json"),
    }


def _blocked_category(failure_reasons: str) -> str:
    reasons = failure_reasons.lower()
    if "guardrail_heavy_map" in reasons:
        return "guardrail_heavy"
    if "no_public_signal" in reasons:
        return "no_public_signal"
    if "overbroad_low_risk_footprint" in reasons or "large_contiguous_low_risk_area" in reasons:
        return "overbroad_footprint"
    return "other"


def _blocked_action(category: str) -> str:
    if category == "guardrail_heavy":
        return "review upstream guardrail dependence and missing native tornado signal"
    if category == "no_public_signal":
        return "inspect whether this is a true weak-signal case or missing forecast support"
    if category == "overbroad_footprint":
        return "inspect raw/core footprint construction and low-risk area spread"
    return "manual review"


def _read_public_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"public candidates CSV is missing required date column: {path}")
    if "public_ready" not in frame.columns:
        frame["public_ready"] = False
    if "failure_reasons" not in frame.columns:
        frame["failure_reasons"] = ""
    return frame


def build_visual_review_frame(candidates: pd.DataFrame, *, products_dir: Path, review_count: int | None, showcase_count: int) -> pd.DataFrame:
    ready = candidates[candidates["public_ready"].map(_truthy)].copy()
    if review_count is not None:
        ready = ready.head(int(review_count)).copy()
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ready.to_dict(orient="records"), start=1):
        date = _safe_text(row.get("date"))
        paths = _product_paths(products_dir, date)
        rows.append(
            {
                "rank": rank,
                "date": date,
                "visual_review_status": REVIEW_STATUS,
                "visual_notes": "",
                "recommended_for_showcase": rank <= int(showcase_count),
                "public_ready": bool(_truthy(row.get("public_ready", False))),
                "audit_status": _safe_text(row.get("status", row.get("audit_status", ""))),
                "failure_reasons": _safe_text(row.get("failure_reasons", "")),
                "observed_tornado_relevant": bool(_truthy(row.get("observed_tornado_relevant", False))),
                "observed_significant_tornado_support": bool(_truthy(row.get("observed_significant_tornado_support", False))),
                "observed_outbreak_relevant": bool(_truthy(row.get("observed_outbreak_relevant", False))),
                "observed_categories": _safe_text(row.get("observed_categories", "")),
                "max_tornado_concern_prob": row.get("max_tornado_concern_prob", ""),
                "low_risk_cells": row.get("low_risk_cells", ""),
                "high_risk_cells": row.get("high_risk_cells", ""),
                "guardrail_cells": row.get("guardrail_cells", ""),
                **paths,
                "product_files_exist": all(Path(path).exists() for path in paths.values()),
            }
        )
    return pd.DataFrame(rows)


def build_blocked_work_queue(candidates: pd.DataFrame) -> pd.DataFrame:
    blocked = candidates[~candidates["public_ready"].map(_truthy)].copy()
    rows: list[dict[str, Any]] = []
    for row in blocked.to_dict(orient="records"):
        reasons = _safe_text(row.get("failure_reasons", ""))
        category = _blocked_category(reasons)
        rows.append(
            {
                "date": _safe_text(row.get("date")),
                "queue": category,
                "science_priority": 1 if category == "guardrail_heavy" else 2 if category in {"no_public_signal", "overbroad_footprint"} else 3,
                "failure_reasons": reasons,
                "suggested_next_action": _blocked_action(category),
                "max_tornado_concern_prob": row.get("max_tornado_concern_prob", ""),
                "low_risk_cells": row.get("low_risk_cells", ""),
                "largest_low_risk_component_cells": row.get("largest_low_risk_component_cells", ""),
                "guardrail_cells": row.get("guardrail_cells", ""),
                "observed_categories": _safe_text(row.get("observed_categories", "")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["date", "queue", "science_priority", "failure_reasons", "suggested_next_action"])
    return frame.sort_values(["science_priority", "queue", "date"], kind="mergesort").reset_index(drop=True)


def _copy_showcase_files(showcase: pd.DataFrame, showcase_dir: Path) -> pd.DataFrame:
    showcase_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row in showcase.to_dict(orient="records"):
        output_row = dict(row)
        for column in ["image_path", "summary_path", "metadata_path"]:
            source = Path(str(row.get(column, "")))
            if source.exists():
                target = showcase_dir / source.name
                shutil.copy2(source, target)
                output_row[f"showcase_{column}"] = str(target)
            else:
                output_row[f"showcase_{column}"] = ""
        rows.append(output_row)
    return pd.DataFrame(rows)


def _write_markdown(path: Path, *, review: pd.DataFrame, blocked: pd.DataFrame, showcase: pd.DataFrame) -> None:
    queue_counts = blocked["queue"].value_counts().to_dict() if not blocked.empty and "queue" in blocked.columns else {}
    lines = [
        "# Tornado Concern Public Prototype Review",
        "",
        "Prototype outputs only. These are not official warning guidance.",
        "",
        f"- review_candidates: {len(review)}",
        f"- showcase_candidates: {len(showcase)}",
        f"- blocked_cases: {len(blocked)}",
        *[f"- blocked_{category}: {int(queue_counts.get(category, 0))}" for category in BLOCKED_CATEGORIES],
        "",
        "## Review Workflow",
        "",
        "Mark each visual_review_status as approve, needs_adjustment, or reject after manual map review.",
        "Only approve maps that are both audit-passing and visually credible.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_gallery(path: Path, review: pd.DataFrame) -> None:
    lines = [
        "# Tornado Concern Visual Review Gallery",
        "",
        "Prototype outputs only. Use this gallery to approve, adjust, or reject candidate maps quickly.",
        "",
    ]
    if review.empty:
        lines.append("_No public-ready candidates were available for visual review._")
    for row in review.to_dict(orient="records"):
        image_path = Path(str(row.get("image_path", "")))
        date = _safe_text(row.get("date"))
        lines.extend(
            [
                f"## Rank {row.get('rank', '')}: {date}",
                "",
                f"- visual_review_status: {_safe_text(row.get('visual_review_status'))}",
                f"- observed_categories: {_safe_text(row.get('observed_categories'))}",
                f"- max_tornado_concern_prob: {_safe_text(row.get('max_tornado_concern_prob'))}",
                f"- audit_status: {_safe_text(row.get('audit_status'))}",
                "",
            ]
        )
        if image_path.exists():
            lines.extend([f"![Tornado concern map for {date}]({image_path.resolve().as_posix()})", ""])
        else:
            lines.extend([f"_Missing image artifact: {image_path}_", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_public_prototype(
    *,
    public_candidates_csv: Path,
    products_dir: Path,
    output_dir: Path,
    review_count: int | None,
    showcase_count: int,
    overwrite: bool,
) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty; pass --overwrite to replace generated artifacts: {output_dir}")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _read_public_candidates(public_candidates_csv)
    review = build_visual_review_frame(candidates, products_dir=products_dir, review_count=review_count, showcase_count=showcase_count)
    blocked = build_blocked_work_queue(candidates)
    showcase = review[review["recommended_for_showcase"].map(_truthy)].copy() if not review.empty else review.copy()
    showcase_manifest = _copy_showcase_files(showcase, output_dir / "showcase") if not showcase.empty else showcase.copy()

    review_csv = output_dir / "visual_review.csv"
    blocked_csv = output_dir / "blocked_work_queue.csv"
    showcase_csv = output_dir / "showcase_manifest.csv"
    gallery_md = output_dir / "visual_review_gallery.md"
    readme = output_dir / "README.md"
    review.to_csv(review_csv, index=False)
    blocked.to_csv(blocked_csv, index=False)
    showcase_manifest.to_csv(showcase_csv, index=False)
    _write_review_gallery(gallery_md, review)
    _write_markdown(readme, review=review, blocked=blocked, showcase=showcase_manifest)
    return {
        "visual_review_csv": review_csv,
        "visual_review_gallery": gallery_md,
        "blocked_work_queue_csv": blocked_csv,
        "showcase_manifest_csv": showcase_csv,
        "readme": readme,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-candidates-csv", required=True)
    parser.add_argument("--products-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--review-count", type=int, default=None, help="Limit visual review rows; default reviews all public-ready candidates")
    parser.add_argument("--showcase-count", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    outputs = prepare_public_prototype(
        public_candidates_csv=Path(args.public_candidates_csv),
        products_dir=Path(args.products_dir),
        output_dir=Path(args.output_dir),
        review_count=args.review_count,
        showcase_count=int(args.showcase_count),
        overwrite=bool(args.overwrite),
    )
    for key, value in outputs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
