"""Audit tornado-concern forecast artifacts for public-map readiness."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from severewx.cli.tornado_concern_eval import _latest_prediction_for_date
from severewx.config import load_settings
from severewx.models.tornado_concern import tornado_environment_outlook_grid, tornado_environment_outlook_hybrid_grid, tornado_environment_outlook_v2_grid
from severewx.utils.paths import build_paths

OUTLOOK_DISPLAY_MIN_OBJECT_CELLS = 24
OUTLOOK_FIELDS = {
    "tornado_environment_outlook",
    "tornado_environment_outlook_v2",
    "tornado_environment_outlook_hybrid",
}


def _load_dates(args: argparse.Namespace) -> list[str]:
    dates = list(args.dates or [])
    if args.dates_file:
        dates.extend(
            [
                line.strip()
                for line in Path(args.dates_file).read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        )
    deduped = list(dict.fromkeys(dates))
    if not deduped:
        raise ValueError("provide at least one date or --dates-file")
    return deduped


def _largest_component(mask: np.ndarray) -> int:
    active = np.asarray(mask, dtype=bool)
    if active.ndim != 2 or not active.any():
        return 0
    visited = np.zeros(active.shape, dtype=bool)
    largest = 0
    rows, cols = active.shape
    for row in range(rows):
        for col in range(cols):
            if not active[row, col] or visited[row, col]:
                continue
            count = 0
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                r, c = queue.popleft()
                count += 1
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and active[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            largest = max(largest, count)
    return largest


def _bbox_fill_ratio(mask: np.ndarray) -> float:
    active = np.asarray(mask, dtype=bool)
    if active.ndim != 2 or not active.any():
        return 0.0
    points = np.argwhere(active)
    row_min, col_min = points.min(axis=0)
    row_max, col_max = points.max(axis=0)
    area = int((row_max - row_min + 1) * (col_max - col_min + 1))
    return float(active.sum() / max(area, 1))


def _filter_small_outlook_objects(values: np.ndarray, *, min_cells: int = OUTLOOK_DISPLAY_MIN_OBJECT_CELLS) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if clean.ndim != 2 or min_cells <= 1:
        return clean
    active = clean >= 0.02
    visited = np.zeros(active.shape, dtype=bool)
    filtered = clean.copy()
    rows, cols = active.shape
    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not active[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            component: list[tuple[int, int]] = []
            visited[start_row, start_col] = True
            while stack:
                row, col = stack.pop()
                component.append((row, col))
                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        if row_offset == 0 and col_offset == 0:
                            continue
                        next_row = row + row_offset
                        next_col = col + col_offset
                        if (
                            0 <= next_row < rows
                            and 0 <= next_col < cols
                            and not visited[next_row, next_col]
                            and active[next_row, next_col]
                        ):
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))
            if len(component) < min_cells:
                for row, col in component:
                    filtered[row, col] = 0.0
    return filtered


def _select_valid_date_array(dataset: xr.Dataset, field_name: str, valid_date: str | None) -> xr.DataArray:
    field = dataset[field_name]
    if valid_date is None or "time" not in field.dims:
        return field.max("time") if "time" in field.dims else field
    dates = pd.to_datetime(dataset["time"].values).date
    target_date = pd.Timestamp(valid_date).date()
    mask = np.asarray([date == target_date for date in dates], dtype=bool)
    if not bool(mask.any()):
        available = ", ".join(list(dict.fromkeys(date.isoformat() for date in dates)))
        raise ValueError(f"forecast artifact has no valid times for {valid_date}; available valid dates: {available}")
    selected = field.isel(time=mask)
    return selected.max("time") if "time" in selected.dims else selected


def _dataset_values(dataset: xr.Dataset, name: str) -> np.ndarray | None:
    return np.asarray(dataset[name].values, dtype=float) if name in dataset else None


def _derive_audit_field(dataset: xr.Dataset, field_name: str) -> xr.Dataset:
    if field_name in dataset or field_name not in OUTLOOK_FIELDS:
        return dataset
    grid_funcs = {
        "tornado_environment_outlook": tornado_environment_outlook_grid,
        "tornado_environment_outlook_v2": tornado_environment_outlook_v2_grid,
        "tornado_environment_outlook_hybrid": tornado_environment_outlook_hybrid_grid,
    }
    grid_func = grid_funcs.get(field_name, tornado_environment_outlook_grid)
    template_name = next(
        (name for name in ("sig_tor_support", "tornado_favored_overlap", "scp_proxy") if name in dataset),
        None,
    )
    if template_name is None:
        return dataset
    field = dataset[template_name]
    if "time" not in field.dims:
        outlook = grid_func(
            sig_tor_support=_dataset_values(dataset, "sig_tor_support"),
            tornado_favored_overlap=_dataset_values(dataset, "tornado_favored_overlap"),
            scp_proxy=_dataset_values(dataset, "scp_proxy"),
            low_lcl_support=_dataset_values(dataset, "low_lcl_support"),
            synoptic_support=_dataset_values(dataset, "synoptic_support"),
            outbreak_risk=_dataset_values(dataset, "outbreak_risk"),
            cin=_dataset_values(dataset, "cin"),
        )
        return dataset.assign({field_name: (field.dims, outlook)})
    outlooks = []
    for index in range(dataset.sizes["time"]):
        subset = dataset.isel(time=index)
        outlooks.append(
            grid_func(
                sig_tor_support=_dataset_values(subset, "sig_tor_support"),
                tornado_favored_overlap=_dataset_values(subset, "tornado_favored_overlap"),
                scp_proxy=_dataset_values(subset, "scp_proxy"),
                low_lcl_support=_dataset_values(subset, "low_lcl_support"),
                synoptic_support=_dataset_values(subset, "synoptic_support"),
                outbreak_risk=_dataset_values(subset, "outbreak_risk"),
                cin=_dataset_values(subset, "cin"),
            )
        )
    return dataset.assign({field_name: (field.dims, np.stack(outlooks, axis=0))})


def audit_tornado_concern_product(
    date: str,
    prediction_path: Path | None,
    *,
    field_name: str = "tornado_concern_prob",
    valid_date: str | None = None,
    low_threshold: float = 0.05,
    medium_threshold: float = 0.10,
    high_threshold: float = 0.35,
    max_low_risk_cells: int = 35,
    max_low_risk_fraction: float = 0.010,
    max_component_cells: int = 30,
    max_guardrail_fraction: float = 0.50,
    min_signal: float = 0.05,
) -> dict[str, Any]:
    if prediction_path is None or not prediction_path.exists():
        return {
            "date": date,
            "artifact_path": "",
            "status": "missing_artifact",
            "public_ready": False,
            "failure_reasons": "missing_artifact",
        }

    dataset = _derive_audit_field(xr.load_dataset(prediction_path), field_name)
    if field_name not in dataset:
        return {
            "date": date,
            "artifact_path": str(prediction_path),
            "status": "missing_field",
            "public_ready": False,
            "failure_reasons": f"missing_{field_name}",
        }

    if field_name in OUTLOOK_FIELDS:
        low_threshold = min(low_threshold, 0.02)
        max_low_risk_cells = max(max_low_risk_cells, 2500)
        max_low_risk_fraction = max(max_low_risk_fraction, 0.25)
        max_component_cells = max(max_component_cells, 2400)

    aggregate = _select_valid_date_array(dataset, field_name, valid_date)
    values = np.asarray(aggregate.values, dtype=float)
    if field_name in OUTLOOK_FIELDS:
        values = _filter_small_outlook_objects(values)
    finite = np.isfinite(values)
    total_cells = int(finite.sum())
    low_mask = finite & (values >= low_threshold)
    medium_mask = finite & (values >= medium_threshold)
    high_mask = finite & (values >= high_threshold)
    low_cells = int(low_mask.sum())
    medium_cells = int(medium_mask.sum())
    high_cells = int(high_mask.sum())
    largest_component_cells = _largest_component(low_mask)
    bbox_fill_ratio = _bbox_fill_ratio(low_mask)
    guardrail_cells = 0
    if field_name not in OUTLOOK_FIELDS and "tornado_concern_guardrail_applied" in dataset:
        guardrail = dataset["tornado_concern_guardrail_applied"]
        guardrail_aggregate = guardrail.max("time") if "time" in guardrail.dims else guardrail
        guardrail_cells = int(np.asarray(guardrail_aggregate.values, dtype=bool).sum())
    max_prob = float(np.nanmax(values)) if finite.any() else float("nan")
    mean_prob = float(np.nanmean(values)) if finite.any() else float("nan")
    low_fraction = float(low_cells / max(total_cells, 1))
    guardrail_fraction = float(guardrail_cells / max(low_cells + guardrail_cells, 1))

    failures: list[str] = []
    if max_prob < min_signal:
        failures.append("no_public_signal")
    if low_cells > max_low_risk_cells or (total_cells >= 1000 and low_fraction > max_low_risk_fraction):
        failures.append("overbroad_low_risk_footprint")
    if largest_component_cells > max_component_cells:
        failures.append("large_contiguous_low_risk_area")
    if guardrail_fraction > max_guardrail_fraction:
        failures.append("guardrail_heavy_map")

    return {
        "date": date,
        "valid_date": valid_date or "",
        "field_name": field_name,
        "artifact_path": str(prediction_path),
        "status": "ok" if not failures else "flagged",
        "public_ready": not failures,
        "failure_reasons": ";".join(failures),
        "max_tornado_concern_prob": max_prob,
        "mean_tornado_concern_prob": mean_prob,
        "low_threshold": low_threshold,
        "medium_threshold": medium_threshold,
        "high_threshold": high_threshold,
        "low_risk_cells": low_cells,
        "medium_risk_cells": medium_cells,
        "high_risk_cells": high_cells,
        "total_cells": total_cells,
        "low_risk_fraction": low_fraction,
        "largest_low_risk_component_cells": largest_component_cells,
        "low_risk_bbox_fill_ratio": bbox_fill_ratio,
        "guardrail_cells": guardrail_cells,
        "guardrail_fraction": guardrail_fraction,
    }


def _write_markdown(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    ready = int(rows["public_ready"].fillna(False).sum()) if "public_ready" in rows else 0
    flagged = total - ready
    lines = [
        "# Tornado Concern Product Audit",
        "",
        f"- dates_audited: {total}",
        f"- public_ready: {ready}",
        f"- flagged_or_missing: {flagged}",
        "",
        "## Flagged Dates",
        "",
        "| date | status | reasons | max | low_cells | largest_component | guardrail_cells |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    flagged_rows = rows.loc[~rows["public_ready"].fillna(False)].copy()
    for row in flagged_rows.to_dict(orient="records"):
        lines.append(
            "| {date} | {status} | {reasons} | {max_prob:.4f} | {low_cells} | {largest} | {guardrail} |".format(
                date=row.get("date", ""),
                status=row.get("status", ""),
                reasons=row.get("failure_reasons", ""),
                max_prob=float(row.get("max_tornado_concern_prob", float("nan"))),
                low_cells=int(row.get("low_risk_cells", 0) or 0),
                largest=int(row.get("largest_low_risk_component_cells", 0) or 0),
                guardrail=int(row.get("guardrail_cells", 0) or 0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--dates-file")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--low-threshold", type=float, default=0.05)
    parser.add_argument("--valid-date", help="Optional exact 24-hour valid date to audit")
    parser.add_argument("--field-name", default="tornado_concern_prob")
    parser.add_argument("--medium-threshold", type=float, default=0.10)
    parser.add_argument("--high-threshold", type=float, default=0.35)
    parser.add_argument("--max-low-risk-cells", type=int, default=35)
    parser.add_argument("--max-low-risk-fraction", type=float, default=0.010)
    parser.add_argument("--max-component-cells", type=int, default=30)
    parser.add_argument("--max-guardrail-fraction", type=float, default=0.50)
    parser.add_argument("--min-signal", type=float, default=0.05)
    args = parser.parse_args(argv)

    settings = load_settings()
    paths = build_paths(settings)
    rows = [
        audit_tornado_concern_product(
            date,
            _latest_prediction_for_date(paths.outputs, date),
            field_name=args.field_name,
            valid_date=args.valid_date,
            low_threshold=args.low_threshold,
            medium_threshold=args.medium_threshold,
            high_threshold=args.high_threshold,
            max_low_risk_cells=args.max_low_risk_cells,
            max_low_risk_fraction=args.max_low_risk_fraction,
            max_component_cells=args.max_component_cells,
            max_guardrail_fraction=args.max_guardrail_fraction,
            min_signal=args.min_signal,
        )
        for date in _load_dates(args)
    ]
    frame = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    _write_markdown(Path(args.output_md), frame)
    print(f"dates_audited={len(frame)}")
    print(f"public_ready={int(frame['public_ready'].fillna(False).sum())}")
    print(f"flagged_or_missing={int((~frame['public_ready'].fillna(False)).sum())}")


if __name__ == "__main__":
    main()
