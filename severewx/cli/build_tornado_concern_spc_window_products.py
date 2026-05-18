"""Build SPC-window-aligned Day 1-3 tornado-concern products for one live run."""

from __future__ import annotations

import argparse
import json
import re
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr

from severewx.cli.build_tornado_concern_product import DEFAULT_CONSENSUS_PRODUCT_FIELD, build_product_bundle
from severewx.cli.diagnose_tornado_concern_spc_window_signal import _time_index, _variant_arrays_for_window
from severewx.cli.tornado_concern_product_status import collect_product_status, product_status_markdown
from severewx.models.forecast_consensus import AUTO_CONSENSUS_SOURCES, CONSENSUS_FIELD, consensus_metadata_path, consensus_product_path

DEFAULT_SPC_DAY1_OUTLOOK_URL = "https://www.spc.noaa.gov/products/outlook/day1otlk.txt"
SPC_VALID_RE = re.compile(r"Valid\s+(\d{2})(\d{4})Z\s*-\s*(\d{2})(\d{4})Z", re.IGNORECASE)


def _format_z(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%MZ")


def _resolve_spc_day(timestamp_day: int, hhmm: str, *, init_date: str) -> pd.Timestamp:
    init = pd.Timestamp(init_date)
    candidate = pd.Timestamp(year=init.year, month=init.month, day=timestamp_day, hour=int(hhmm[:2]), minute=int(hhmm[2:]))
    if candidate < init - pd.Timedelta(days=15):
        candidate = candidate + pd.DateOffset(months=1)
    elif candidate > init + pd.Timedelta(days=15):
        candidate = candidate - pd.DateOffset(months=1)
    return pd.Timestamp(candidate)


def parse_spc_day1_valid_window(text: str, *, init_date: str) -> tuple[str, str]:
    match = SPC_VALID_RE.search(text)
    if not match:
        raise ValueError("SPC Day 1 outlook text does not contain a valid line like 'Valid 181300Z - 191200Z'")
    start_day, start_hhmm, end_day, end_hhmm = match.groups()
    start = _resolve_spc_day(int(start_day), start_hhmm, init_date=init_date)
    end = _resolve_spc_day(int(end_day), end_hhmm, init_date=init_date)
    if end <= start:
        end = end + pd.DateOffset(months=1)
    return _format_z(start), _format_z(end)


def fetch_spc_day1_valid_window(*, init_date: str, url: str = DEFAULT_SPC_DAY1_OUTLOOK_URL, timeout_seconds: int = 20) -> tuple[str, str]:
    with urlopen(url, timeout=timeout_seconds) as response:
        text = response.read().decode("utf-8", errors="replace")
    return parse_spc_day1_valid_window(text, init_date=init_date)


def default_spc_day1_valid_window(init_date: str) -> tuple[str, str]:
    start = pd.Timestamp(init_date) + pd.Timedelta(hours=13)
    end = pd.Timestamp(init_date) + pd.Timedelta(days=1, hours=12)
    return _format_z(start), _format_z(end)


def spc_window_specs(init_date: str, *, day1_valid_start: str, day1_valid_end: str | None = None) -> list[dict[str, str]]:
    """Return public Day 1-3 windows using SPC-style convective-day timing."""
    day1_start = pd.Timestamp(day1_valid_start)
    if day1_start.tzinfo is not None:
        day1_start = day1_start.tz_convert("UTC").tz_localize(None)
    init_day = pd.Timestamp(init_date)
    day1_end = pd.Timestamp(day1_valid_end) if day1_valid_end else init_day + pd.Timedelta(days=1, hours=12)
    if day1_end.tzinfo is not None:
        day1_end = day1_end.tz_convert("UTC").tz_localize(None)
    day2_start = init_day + pd.Timedelta(days=1, hours=12)
    day3_start = init_day + pd.Timedelta(days=2, hours=12)
    return [
        {
            "day": "day1",
            "valid_start": _format_z(day1_start),
            "valid_end": _format_z(day1_end),
        },
        {
            "day": "day2",
            "valid_start": _format_z(day2_start),
            "valid_end": _format_z(day2_start + timedelta(days=1)),
        },
        {
            "day": "day3",
            "valid_start": _format_z(day3_start),
            "valid_end": _format_z(day3_start + timedelta(days=1)),
        },
    ]


def build_spc_window_products(
    *,
    date: str,
    cycle: str,
    day1_valid_start: str,
    day1_valid_end: str | None = None,
    outdir: Path,
    require_production_basemap: bool = True,
    diagnostic_consensus_variant: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    products: list[dict[str, Any]] = []
    specs = spc_window_specs(date, day1_valid_start=day1_valid_start, day1_valid_end=day1_valid_end)
    for spec in specs:
        artifact_path = None
        artifact_metadata_path = None
        if diagnostic_consensus_variant:
            artifact_path, artifact_metadata_path = _build_window_variant_artifact(
                date=date,
                cycle=cycle,
                valid_start=spec["valid_start"],
                valid_end=spec["valid_end"],
                outdir=outdir,
                variant=diagnostic_consensus_variant,
                overwrite=overwrite,
            )
        metadata = build_product_bundle(
            date=date,
            cycle=cycle,
            valid_start=spec["valid_start"],
            valid_end=spec["valid_end"],
            field_name=DEFAULT_CONSENSUS_PRODUCT_FIELD,
            artifact_source="consensus",
            map_style="outlook",
            map_domain="conus",
            outdir=outdir,
            artifact_path=artifact_path,
            artifact_metadata_path=artifact_metadata_path,
            require_production_basemap=require_production_basemap,
            overwrite=overwrite,
        )
        metadata["spc_day"] = spec["day"]
        products.append(metadata)
    metadata_paths = [Path(item["metadata_path"]) for item in products if item.get("metadata_path")]
    status_frame = collect_product_status(metadata_paths)
    status_csv = outdir / "product_status.csv"
    status_md = outdir / "product_status.md"
    summary_json = outdir / "spc_window_products_summary.json"
    status_frame.to_csv(status_csv, index=False)
    status_md.write_text(product_status_markdown(status_frame), encoding="utf-8")
    public_ready_count = int(status_frame["publication_status"].astype(str).eq("public_candidate").sum()) if not status_frame.empty else 0
    blocked_count = int(status_frame["publication_status"].astype(str).ne("public_candidate").sum()) if not status_frame.empty else 0
    summary = {
        "date": date,
        "cycle": str(cycle).zfill(2),
        "product_valid_period": "spc_convective_windows",
        "day1_valid_start": day1_valid_start,
        "day1_valid_end": day1_valid_end or specs[0]["valid_end"],
        "diagnostic_consensus_variant": diagnostic_consensus_variant or "",
        "required_public_product_count": 3,
        "public_ready_product_count": public_ready_count,
        "blocked_product_count": blocked_count,
        "status_csv": str(status_csv.resolve()),
        "status_md": str(status_md.resolve()),
        "products": [
            {
                "spc_day": spec["day"],
                "valid_start": product.get("valid_start", spec["valid_start"]),
                "valid_end": product.get("valid_end", spec["valid_end"]),
                "publication_status": product.get("publication_status", ""),
                "public_ready": bool(product.get("public_ready", False)),
                "failure_reasons": product.get("failure_reasons", ""),
                "metadata_path": product.get("metadata_path", ""),
                "main_image_path": product.get("main_image_path", ""),
            }
            for spec, product in zip(specs, products)
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_json.resolve())
    return summary


def _build_window_variant_artifact(
    *,
    date: str,
    cycle: str,
    valid_start: str,
    valid_end: str,
    outdir: Path,
    variant: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Write a one-time-step diagnostic consensus artifact for an SPC window."""
    if variant not in {
        "variant_temporal_primary_core_expanded_corridor",
        "variant_temporal_primary_core_clean_corridor",
        "variant_temporal_primary_core_source_guarded_corridor",
    }:
        raise ValueError(f"unsupported SPC-window diagnostic consensus variant: {variant}")
    base_path = consensus_product_path(Path("data/outputs"), date, str(cycle).zfill(2))
    if not base_path.exists():
        raise FileNotFoundError(f"missing base consensus artifact: {base_path}")
    variant_dir = outdir / "diagnostic_consensus_variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(valid_start).tz_localize(None)
    end_ts = pd.Timestamp(valid_end).tz_localize(None)
    slug = f"{variant}_{start_ts.strftime('%Y%m%dT%HZ')}_to_{end_ts.strftime('%Y%m%dT%HZ')}"
    output_path = variant_dir / f"forecast_consensus_{date}_{str(cycle).zfill(2)}z_{slug}.nc"
    metadata_path = variant_dir / f"forecast_consensus_metadata_{date}_{str(cycle).zfill(2)}z_{slug}.json"
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise FileExistsError(f"diagnostic consensus variant artifact already exists: {output_path}")
    with xr.open_dataset(base_path) as reference:
        indices = _time_index(reference, valid_start, valid_end)
        if not indices:
            raise ValueError(f"base consensus has no valid times in window {valid_start} to {valid_end}")
        variants = _variant_arrays_for_window(
            outputs_dir=Path("data/outputs"),
            date=date,
            cycle=str(cycle).zfill(2),
            sources=list(AUTO_CONSENSUS_SOURCES),
            reference=reference,
            time_indices=indices,
        )
        if variant not in variants:
            raise ValueError(f"diagnostic variant was not generated for window: {variant}")
        field = np.asarray(variants[variant], dtype=np.float32)
        agreement = np.where(field >= 0.02, 2.0, 0.0).astype(np.float32)
        confidence = np.where(field >= 0.02, 0.75, 0.5).astype(np.float32)
        dataset = xr.Dataset(
            {
                CONSENSUS_FIELD: (("time", "lat", "lon"), field[np.newaxis, :, :]),
                "model_agreement_count": (("time", "lat", "lon"), agreement[np.newaxis, :, :]),
                "consensus_confidence_modifier": (("time", "lat", "lon"), confidence[np.newaxis, :, :]),
            },
            coords={"time": [np.datetime64(start_ts.to_datetime64())], "lat": reference["lat"].values, "lon": reference["lon"].values},
            attrs={
                "source": "spc_window_diagnostic_consensus_variant",
                "consensus_variant": variant,
                "base_consensus_path": str(base_path),
                "valid_start": valid_start,
                "valid_end": valid_end,
            },
        )
    dataset.to_netcdf(output_path)
    dataset.close()
    base_metadata = {}
    metadata_source = consensus_metadata_path(Path("data/outputs"), date, str(cycle).zfill(2))
    if metadata_source.exists():
        try:
            base_metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            base_metadata = {}
    metadata = dict(base_metadata) if isinstance(base_metadata, dict) else {}
    metadata.update(
        {
            "source": "spc_window_diagnostic_consensus_variant",
            "consensus_variant": variant,
            "base_consensus_path": str(base_path),
            "consensus_field": CONSENSUS_FIELD,
            "included_sources": sorted(AUTO_CONSENSUS_SOURCES),
            "valid_start": valid_start,
            "valid_end": valid_end,
        }
    )
    ingest = metadata.get("ingest_summary", {}) if isinstance(metadata.get("ingest_summary", {}), dict) else {}
    ingest.update({"source": "spc_window_diagnostic_consensus_variant", "source_mode": "real", "real_ingest_available": True})
    metadata["ingest_summary"] = ingest
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path, metadata_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Forecast init date YYYY-MM-DD")
    parser.add_argument("--cycle", required=True, help="Forecast init cycle, e.g. 12")
    parser.add_argument(
        "--day1-valid-start",
        default="auto",
        help="SPC Day 1 window start, e.g. 2026-05-15T16:30Z. Use 'auto' to fetch the current SPC Day 1 valid line.",
    )
    parser.add_argument("--day1-valid-end", help="Optional SPC Day 1 window end. Auto mode parses this from the SPC Day 1 valid line.")
    parser.add_argument("--spc-day1-url", default=DEFAULT_SPC_DAY1_OUTLOOK_URL, help="SPC Day 1 outlook text URL for auto valid-window parsing")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--diagnostic-consensus-variant",
        choices=[
            "variant_temporal_primary_core_expanded_corridor",
            "variant_temporal_primary_core_clean_corridor",
            "variant_temporal_primary_core_source_guarded_corridor",
        ],
        help="Build SPC-window products from a diagnostic temporal consensus variant instead of the default consensus.",
    )
    parser.add_argument("--allow-fallback-publication", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    day1_valid_start = args.day1_valid_start
    day1_valid_end = args.day1_valid_end
    if str(day1_valid_start).strip().lower() == "auto":
        try:
            day1_valid_start, day1_valid_end = fetch_spc_day1_valid_window(init_date=args.date, url=args.spc_day1_url)
        except Exception as exc:
            day1_valid_start, day1_valid_end = default_spc_day1_valid_window(args.date)
            print(f"warning=failed_to_fetch_spc_day1_valid_window; using_default={day1_valid_start}_to_{day1_valid_end}; error={exc}")

    result = build_spc_window_products(
        date=args.date,
        cycle=args.cycle,
        day1_valid_start=day1_valid_start,
        day1_valid_end=day1_valid_end,
        outdir=Path(args.outdir),
        require_production_basemap=not bool(args.allow_fallback_publication),
        diagnostic_consensus_variant=args.diagnostic_consensus_variant,
        overwrite=bool(args.overwrite),
    )
    print(f"summary_json={result['summary_json']}")
    print(f"public_ready_product_count={result['public_ready_product_count']}")
    print(f"blocked_product_count={result['blocked_product_count']}")


if __name__ == "__main__":
    main()
