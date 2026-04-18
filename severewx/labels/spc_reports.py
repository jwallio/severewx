"""SPC severe report loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from severewx.utils.dates import parse_ymd


EXPECTED_COLUMNS = {"date", "time", "lat", "lon", "hazard", "magnitude"}


@dataclass(slots=True)
class ReportTable:
    frame: pd.DataFrame
    source_name: str
    is_real_source: bool
    ingest_timestamp: str


def _synthesize_reports(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(parse_ymd(start), parse_ymd(end), freq="D")
    rows: list[dict[str, object]] = []
    for index, valid_date in enumerate(dates):
        active = index % 5 in {1, 2}
        n_reports = 18 if active else 3
        for report_index in range(n_reports):
            hazard = ["tornado", "hail", "wind"][report_index % 3]
            rows.append(
                {
                    "date": valid_date.date().isoformat(),
                    "time": f"{18 + (report_index % 6):02d}00",
                    "lat": 31.0 + 0.5 * report_index,
                    "lon": -97.0 + 0.7 * report_index,
                    "hazard": hazard,
                    "magnitude": 2.5 if hazard == "hail" else 70.0 if hazard == "wind" else 2.0,
                    "significant": int(hazard == "tornado" and report_index % 4 == 0),
                }
            )
    return pd.DataFrame(rows)


def normalize_spc_reports(
    frame: pd.DataFrame,
    *,
    source_name: str = "unknown",
    is_real_source: bool = False,
    ingest_timestamp: str | None = None,
) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            "slat": "lat",
            "slon": "lon",
            "event_type": "hazard",
            "mag": "magnitude",
            "valid_date": "date",
        }
    )
    missing = EXPECTED_COLUMNS.difference(renamed.columns)
    if missing:
        raise ValueError(f"missing SPC report columns: {sorted(missing)}")
    normalized = renamed.copy()
    normalized["hazard"] = normalized["hazard"].astype(str).str.lower().replace(
        {"tor": "tornado", "wnd": "wind", "wind damage": "wind"}
    )
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.date.astype(str)
    normalized["time"] = normalized["time"].astype(str).str.zfill(4)
    normalized["lat"] = normalized["lat"].astype(float)
    normalized["lon"] = normalized["lon"].astype(float)
    normalized["magnitude"] = normalized["magnitude"].fillna(0).astype(float)
    if "significant" not in normalized.columns:
        normalized["significant"] = 0
    normalized["significant"] = normalized["significant"].fillna(0).astype(int)
    resolved_ingest_timestamp = ingest_timestamp or pd.Timestamp.utcnow().isoformat()
    if "source_name" not in normalized.columns:
        normalized["source_name"] = source_name
    normalized["source_name"] = normalized["source_name"].fillna(source_name).astype(str)
    if "is_real_source" not in normalized.columns:
        normalized["is_real_source"] = bool(is_real_source)
    normalized["is_real_source"] = normalized["is_real_source"].fillna(bool(is_real_source)).astype(bool)
    if "ingest_timestamp" not in normalized.columns:
        normalized["ingest_timestamp"] = resolved_ingest_timestamp
    normalized["ingest_timestamp"] = normalized["ingest_timestamp"].fillna(resolved_ingest_timestamp).astype(str)
    normalized["report_id"] = np.arange(len(normalized))
    return normalized[
        [
            "report_id",
            "date",
            "time",
            "lat",
            "lon",
            "hazard",
            "magnitude",
            "significant",
            "source_name",
            "is_real_source",
            "ingest_timestamp",
        ]
    ]


def load_spc_reports(source_path: str | Path | None, start: str, end: str) -> ReportTable:
    ingest_timestamp = pd.Timestamp.utcnow().isoformat()
    if source_path is None:
        frame = normalize_spc_reports(
            _synthesize_reports(start, end),
            source_name="synthetic_fallback",
            is_real_source=False,
            ingest_timestamp=ingest_timestamp,
        )
        return ReportTable(frame=frame, source_name="synthetic_fallback", is_real_source=False, ingest_timestamp=ingest_timestamp)
    path = Path(source_path)
    if not path.exists():
        frame = normalize_spc_reports(
            _synthesize_reports(start, end),
            source_name="synthetic_fallback",
            is_real_source=False,
            ingest_timestamp=ingest_timestamp,
        )
        return ReportTable(frame=frame, source_name="synthetic_fallback", is_real_source=False, ingest_timestamp=ingest_timestamp)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"unsupported report file type: {path.suffix}")
    source_name = str(frame["source_name"].iloc[0]) if "source_name" in frame.columns and not frame.empty else f"file:{path.name}"
    is_real_source = bool(frame["is_real_source"].all()) if "is_real_source" in frame.columns and not frame.empty else True
    frame = normalize_spc_reports(
        frame,
        source_name=source_name,
        is_real_source=is_real_source,
        ingest_timestamp=ingest_timestamp,
    )
    mask = (frame["date"] >= start) & (frame["date"] <= end)
    filtered = frame.loc[mask].reset_index(drop=True)
    return ReportTable(
        frame=filtered,
        source_name=source_name,
        is_real_source=is_real_source,
        ingest_timestamp=str(filtered["ingest_timestamp"].iloc[0]) if not filtered.empty else ingest_timestamp,
    )
