"""Build ingredient diagnostics for tornado environment outlook fields."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from severewx.cli.build_tornado_concern_product import (
    OUTLOOK_INGREDIENT_FIELDS,
    _derive_product_field_dataset,
    _display_object_metrics,
    _prediction_artifact_for_cycle,
    _read_dates_file,
    _select_valid_date_dataset,
)
from severewx.config import load_settings
from severewx.utils.paths import build_paths


THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.30, 0.45)


def _aggregate_values(dataset: xr.Dataset, field_name: str) -> np.ndarray:
    if field_name not in dataset:
        return np.zeros((0, 0), dtype=float)
    field = dataset[field_name]
    aggregate = field.max("time") if "time" in field.dims else field
    return np.nan_to_num(np.asarray(aggregate.values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def _field_metrics(values: np.ndarray, *, prefix: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        f"{prefix}_max": float(np.nanmax(values)) if values.size else 0.0,
        f"{prefix}_mean": float(np.nanmean(values)) if values.size else 0.0,
        **{f"{prefix}_{key}": value for key, value in _display_object_metrics(values).items()},
    }
    for threshold in THRESHOLDS:
        metrics[f"{prefix}_ge{int(threshold * 100):02d}"] = int(np.count_nonzero(values >= threshold)) if values.size else 0
    return metrics


def _ingredient_metrics(dataset: xr.Dataset, field_values: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    peak_row = peak_col = None
    if field_values.size and not np.isnan(field_values).all():
        peak_row, peak_col = np.unravel_index(int(np.nanargmax(field_values)), field_values.shape)
    for ingredient in OUTLOOK_INGREDIENT_FIELDS:
        values = _aggregate_values(dataset, ingredient)
        if values.size == 0:
            continue
        metrics[f"{ingredient}_max"] = float(np.nanmax(values))
        metrics[f"{ingredient}_mean"] = float(np.nanmean(values))
        if peak_row is not None and peak_col is not None and peak_row < values.shape[0] and peak_col < values.shape[1]:
            metrics[f"{ingredient}_at_primary_peak"] = float(values[peak_row, peak_col])
    limiting_candidates = {
        "sig_tor_support": metrics.get("sig_tor_support_at_primary_peak", 0.0) / 1.25,
        "tornado_favored_overlap": metrics.get("tornado_favored_overlap_at_primary_peak", 0.0) / 2.50,
        "scp_proxy": metrics.get("scp_proxy_at_primary_peak", 0.0) / 0.75,
    }
    if limiting_candidates:
        clipped = {key: float(np.clip(value, 0.0, 1.0)) for key, value in limiting_candidates.items()}
        metrics["limiting_ingredient_at_primary_peak"] = min(clipped, key=clipped.get)
    return metrics


def build_field_diagnostics(
    *,
    dates: list[str],
    cycle: str,
    output_dir: Path,
    field: str = "tornado_environment_outlook",
    challenger_field: str = "tornado_environment_outlook_hybrid",
    overwrite: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    paths = build_paths(settings)
    rows: list[dict[str, Any]] = []
    for date in dates:
        prediction_path = _prediction_artifact_for_cycle(paths.outputs, date, cycle)
        if prediction_path is None:
            rows.append({"date": date, "cycle": cycle, "status": "missing_forecast_artifact"})
            continue
        with xr.open_dataset(prediction_path) as source:
            dataset = _derive_product_field_dataset(source, field)
            dataset = _derive_product_field_dataset(dataset, challenger_field)
            try:
                valid_dataset = _select_valid_date_dataset(dataset, date)
            except ValueError:
                valid_dataset = dataset
            primary_values = _aggregate_values(valid_dataset, field)
            challenger_values = _aggregate_values(valid_dataset, challenger_field)
            row: dict[str, Any] = {
                "date": date,
                "cycle": cycle,
                "status": "ok",
                "artifact_path": str(prediction_path),
                "field": field,
                "challenger_field": challenger_field,
            }
            row.update(_field_metrics(primary_values, prefix="primary"))
            row.update(_field_metrics(challenger_values, prefix="challenger"))
            for threshold in THRESHOLDS:
                key = f"ge{int(threshold * 100):02d}"
                row[f"delta_{key}"] = int(row.get(f"challenger_{key}", 0)) - int(row.get(f"primary_{key}", 0))
            row["delta_largest_object_ge02"] = int(row.get("challenger_display_largest_object_cells_ge_02pct", 0)) - int(
                row.get("primary_display_largest_object_cells_ge_02pct", 0)
            )
            row.update(_ingredient_metrics(valid_dataset, primary_values))
            rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        "# Tornado Environment Outlook Field Diagnostics",
        "",
        f"- dates_checked: {len(frame)}",
        f"- successful: {int(frame.get('status', pd.Series(dtype=str)).eq('ok').sum()) if not frame.empty else 0}",
        "",
        "| date | status | primary_ge02 | challenger_ge02 | delta_ge02 | primary_ge10 | challenger_ge10 | delta_ge10 | largest_delta | limiting_peak_ingredient |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            "| {date} | {status} | {p02} | {c02} | {d02} | {p10} | {c10} | {d10} | {largest} | {limiting} |".format(
                date=row.get("date", ""),
                status=row.get("status", ""),
                p02=int(row.get("primary_ge02", 0) or 0),
                c02=int(row.get("challenger_ge02", 0) or 0),
                d02=int(row.get("delta_ge02", 0) or 0),
                p10=int(row.get("primary_ge10", 0) or 0),
                c10=int(row.get("challenger_ge10", 0) or 0),
                d10=int(row.get("delta_ge10", 0) or 0),
                largest=int(row.get("delta_largest_object_ge02", 0) or 0),
                limiting=row.get("limiting_ingredient_at_primary_peak", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates-file", required=True)
    parser.add_argument("--cycle", default="00")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--field", default="tornado_environment_outlook")
    parser.add_argument("--challenger-field", default="tornado_environment_outlook_hybrid")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    dates = _read_dates_file(Path(args.dates_file))
    output_dir = Path(args.outdir)
    csv_path = output_dir / "tornado_environment_outlook_field_diagnostics.csv"
    md_path = output_dir / "tornado_environment_outlook_field_diagnostics.md"
    if not args.overwrite and (csv_path.exists() or md_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing diagnostics in {output_dir}")
    frame = build_field_diagnostics(
        dates=dates,
        cycle=str(args.cycle).zfill(2),
        output_dir=output_dir,
        field=args.field,
        challenger_field=args.challenger_field,
        overwrite=bool(args.overwrite),
    )
    frame.to_csv(csv_path, index=False)
    _write_markdown(md_path, frame)
    print(f"dates_checked={len(frame)}")
    print(f"successful={int(frame['status'].eq('ok').sum()) if not frame.empty and 'status' in frame else 0}")


if __name__ == "__main__":
    main()
