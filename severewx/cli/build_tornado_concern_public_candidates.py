"""Build a ranked public-candidate list for tornado-concern products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from severewx.cli.build_tornado_concern_product import DEFAULT_PRODUCT_FIELD, _derive_product_field_dataset
from severewx.cli.tornado_concern_eval import _latest_prediction_for_date, _verification_for_date
from severewx.cli.tornado_concern_product_audit import audit_tornado_concern_product
from severewx.config import load_settings
from severewx.utils.paths import build_paths


def _read_dates(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.strip().startswith("#")]


def _top_valid_date(prediction_path: Path | None, *, field_name: str = "tornado_concern_prob") -> str:
    if prediction_path is None or not prediction_path.exists():
        return ""
    with xr.open_dataset(prediction_path) as dataset:
        dataset = _derive_product_field_dataset(dataset, field_name)
        if field_name not in dataset or "time" not in dataset[field_name].dims:
            return ""
        field = dataset[field_name]
        series = field.max(dim=[dim for dim in field.dims if dim != "time"], skipna=True)
        values = series.values
        if values.size == 0:
            return ""
        index = int(values.argmax())
        return pd.to_datetime(dataset["time"].values[index]).date().isoformat()


def _verification_relevance(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "verification_path": "",
            "observed_tornado_relevant": False,
            "observed_significant_tornado_support": False,
            "observed_outbreak_relevant": False,
            "observed_categories": "",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    per_day = payload.get("per_day", []) if isinstance(payload, dict) else []
    categories = [str(row.get("observed_category", "")) for row in per_day if isinstance(row, dict)]
    tornado_relevant = any(int(row.get("observed_tornado_outbreak", 0) or 0) > 0 for row in per_day if isinstance(row, dict))
    sigtor = any(int(row.get("observed_significant_tornado_support", 0) or 0) > 0 for row in per_day if isinstance(row, dict))
    outbreak = any("outbreak" in category for category in categories)
    return {
        "verification_path": str(path),
        "observed_tornado_relevant": bool(tornado_relevant),
        "observed_significant_tornado_support": bool(sigtor),
        "observed_outbreak_relevant": bool(outbreak),
        "observed_categories": ";".join(list(dict.fromkeys(category for category in categories if category))),
    }


def build_public_candidate_rows(
    dates: list[str],
    paths: Any,
    *,
    valid_date_mode: str = "artifact",
    field_name: str = "tornado_concern_prob",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date in dates:
        prediction_path = _latest_prediction_for_date(paths.outputs, date)
        if valid_date_mode == "best":
            selected_valid_date = _top_valid_date(prediction_path, field_name=field_name)
        elif valid_date_mode == "init":
            selected_valid_date = date
        else:
            selected_valid_date = ""
        verification_path = _verification_for_date(paths.verification, date)
        audit = audit_tornado_concern_product(date, prediction_path, field_name=field_name, valid_date=selected_valid_date or None)
        relevance = _verification_relevance(verification_path)
        row = {**audit, **relevance}
        row["selected_valid_date"] = selected_valid_date
        row["valid_date_mode"] = valid_date_mode
        row["public_candidate_rank_score"] = (
            1000 * int(bool(row.get("public_ready", False)))
            + 120 * int(bool(row.get("observed_significant_tornado_support", False)))
            + 80 * int(bool(row.get("observed_tornado_relevant", False)))
            + 20 * int(bool(row.get("observed_outbreak_relevant", False)))
            + float(row.get("high_risk_cells", 0) or 0)
            + 0.01 * float(row.get("max_tornado_concern_prob", 0.0) or 0.0)
            - 0.05 * float(row.get("low_risk_cells", 0) or 0)
            - 0.10 * float(row.get("guardrail_cells", 0) or 0)
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["public_ready", "public_candidate_rank_score", "date"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _write_markdown(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = int(frame["public_ready"].fillna(False).sum()) if "public_ready" in frame else 0
    lines = [
        "# Tornado Concern Public Candidates",
        "",
        f"- dates_audited: {len(frame)}",
        f"- public_candidates: {ready}",
        f"- internal_review_only: {len(frame) - ready}",
        "",
        "## Ranked Candidates",
        "",
        "| rank | date | public_ready | reasons | max | low_cells | high_cells | tornado_relevant | sigtor |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(frame.to_dict(orient="records"), start=1):
        lines.append(
            "| {rank} | {date} | {ready} | {reasons} | {max_prob:.4f} | {low} | {high} | {tor} | {sig} |".format(
                rank=rank,
                date=row.get("date", ""),
                ready=row.get("public_ready", False),
                reasons=row.get("failure_reasons", "") or "none",
                max_prob=float(row.get("max_tornado_concern_prob", float("nan"))),
                low=int(row.get("low_risk_cells", 0) or 0),
                high=int(row.get("high_risk_cells", 0) or 0),
                tor=row.get("observed_tornado_relevant", False),
                sig=row.get("observed_significant_tornado_support", False),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates-file", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--valid-date-mode", choices=["artifact", "init", "best"], default="artifact")
    parser.add_argument("--field", default=DEFAULT_PRODUCT_FIELD)
    args = parser.parse_args(argv)

    settings = load_settings()
    paths = build_paths(settings)
    frame = build_public_candidate_rows(
        _read_dates(Path(args.dates_file)),
        paths,
        valid_date_mode=args.valid_date_mode,
        field_name=args.field,
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    _write_markdown(Path(args.output_md), frame)
    print(f"dates_audited={len(frame)}")
    print(f"public_candidates={int(frame['public_ready'].fillna(False).sum()) if not frame.empty else 0}")


if __name__ == "__main__":
    main()
