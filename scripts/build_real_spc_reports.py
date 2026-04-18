"""Build a real SPC severe-reports parquet compatible with severewx.cli.build_labels."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from severewx.labels.spc_reports import normalize_spc_reports


BASE_URL = "https://www.spc.noaa.gov/climo/reports"
HAZARD_FILES = {
    "tornado": "torn",
    "hail": "hail",
    "wind": "wind",
}
TORNADO_SCALE_PATTERN = re.compile(r"\b(?:E?F)([0-5])\b", re.IGNORECASE)


def _iter_dates(start: str, end: str) -> list[date]:
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    current = start_date
    dates: list[date] = []
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def _fetch_daily_csv(valid_date: date, hazard: str, timeout: int) -> pd.DataFrame:
    suffix = HAZARD_FILES[hazard]
    stamp = valid_date.strftime("%y%m%d")
    url = f"{BASE_URL}/{stamp}_rpts_raw_{suffix}.csv"
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        return pd.DataFrame()
    response.raise_for_status()
    lines = response.text.splitlines()
    if len(lines) < 2:
        return pd.DataFrame()
    header = [part.strip() for part in lines[1].split(",")]
    rows: list[list[str]] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", len(header) - 1)]
        if len(parts) != len(header):
            continue
        rows.append(parts)
    return pd.DataFrame(rows, columns=header)


def _parse_f_scale(value: object) -> float:
    text = str(value).strip().upper()
    digits = "".join(ch for ch in text if ch.isdigit())
    return float(digits) if digits else 0.0


def _parse_tornado_scale(scale_value: object, comments_value: object) -> float:
    scale = _parse_f_scale(scale_value)
    if scale > 0.0:
        return scale
    match = TORNADO_SCALE_PATTERN.search(str(comments_value))
    return float(match.group(1)) if match else 0.0


def _normalize_daily_frame(frame: pd.DataFrame, valid_date: date, hazard: str) -> pd.DataFrame:
    renamed = frame.rename(columns={"LAT": "lat", "LON": "lon", "Time": "time"})
    normalized = pd.DataFrame(
        {
            "date": valid_date.isoformat(),
            "time": renamed["time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4),
            "lat": pd.to_numeric(renamed["lat"], errors="coerce"),
            "lon": pd.to_numeric(renamed["lon"], errors="coerce"),
            "hazard": hazard,
            "magnitude": 0.0,
            "significant": 0,
            "source_name": "spc_climo_reports",
            "is_real_source": True,
        }
    )
    if hazard == "tornado":
        scale = renamed.get("F_Scale", renamed.get("EF_Scale", pd.Series([""] * len(renamed), index=renamed.index)))
        comments = renamed.get("Comments", renamed.get("Remarks", pd.Series([""] * len(renamed), index=renamed.index)))
        normalized["magnitude"] = [
            _parse_tornado_scale(scale_value, comments_value)
            for scale_value, comments_value in zip(scale, comments, strict=False)
        ]
        normalized["significant"] = (normalized["magnitude"] >= 2.0).astype(int)
    elif hazard == "hail":
        size = renamed.get("Size(1/100in.)", pd.Series([0] * len(renamed), index=renamed.index))
        normalized["magnitude"] = pd.to_numeric(size, errors="coerce").fillna(0.0) / 100.0
    elif hazard == "wind":
        speed = renamed.get("Speed(MPH)", pd.Series([0] * len(renamed), index=renamed.index))
        normalized["magnitude"] = pd.to_numeric(speed, errors="coerce").fillna(0.0) * 0.868976
    cleaned = normalized.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return cleaned


def build_real_spc_reports(start: str, end: str, *, timeout: int = 30) -> pd.DataFrame:
    ingest_timestamp = pd.Timestamp.utcnow().isoformat()
    frames: list[pd.DataFrame] = []
    for valid_date in _iter_dates(start, end):
        for hazard in HAZARD_FILES:
            daily_frame = _fetch_daily_csv(valid_date, hazard, timeout)
            if daily_frame.empty:
                continue
            frames.append(_normalize_daily_frame(daily_frame, valid_date, hazard))
    if not frames:
        raise RuntimeError(f"no SPC reports fetched for {start} through {end}")
    combined = pd.concat(frames, ignore_index=True)
    return normalize_spc_reports(
        combined,
        source_name="spc_climo_reports",
        is_real_source=True,
        ingest_timestamp=ingest_timestamp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real SPC severe-reports parquet")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    frame = build_real_spc_reports(args.start, args.end, timeout=args.timeout)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
