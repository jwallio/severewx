"""Outbreak label logic.

The thresholds here are intentionally configurable because outbreak definitions vary by
forecaster and verification target. V2 uses a broad, operationally pragmatic scheme:

- `tornado_outbreak`: many tornado reports, many significant tornado reports, or a concentrated
  tornado cluster in one broad region
- `significant_tornado_support`: fewer total tornadoes can still qualify if the day supports
  multiple significant tornado reports or a concentrated regional significant-tornado signal
- `hail_outbreak`: large hail concentration, especially if significant hail is elevated
- `wind_mcs_outbreak`: broad wind report coverage with a regional cluster large enough to imply
  organized severe wind or MCS behavior
- `active_non_outbreak_severe_day`: meaningful severe coverage that does not meet outbreak rules
- `null_day`: no meaningful severe signal
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from severewx.calibration.regional import assign_region
from severewx.config import AppSettings


@dataclass(slots=True)
class OutbreakThresholds:
    tornado_reports: int
    significant_tornado_reports: int
    hail_reports: int
    wind_reports: int
    any_reports: int
    regional_cluster_reports: int
    active_severe_reports: int
    tornado_region_reports: int
    hail_region_reports: int
    wind_region_reports: int
    significant_hail_reports: int
    significant_wind_reports: int


def outbreak_thresholds(settings: AppSettings) -> OutbreakThresholds:
    return OutbreakThresholds(**settings.get("labels.outbreak", {}))


def _significant_report_flags(reports: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    frame = reports.copy()
    frame["region"] = frame.apply(lambda row: assign_region(float(row["lat"]), float(row["lon"])), axis=1) if not frame.empty else []
    frame["sig_tornado"] = ((frame["hazard"] == "tornado") & (frame["significant"] > 0)).astype(int)
    frame["sig_hail"] = (
        (frame["hazard"] == "hail") & (frame["magnitude"].astype(float) >= float(settings.get("labels.hail_significant_inches", 2.0)))
    ).astype(int)
    frame["sig_wind"] = (
        (frame["hazard"] == "wind") & (frame["magnitude"].astype(float) >= float(settings.get("labels.wind_significant_kts", 65.0)))
    ).astype(int)
    return frame


def regional_cluster_count(reports: pd.DataFrame) -> int:
    if reports.empty:
        return 0
    binned_lat = np.floor(reports["lat"].to_numpy(dtype=float) / 2.5)
    binned_lon = np.floor(reports["lon"].to_numpy(dtype=float) / 2.5)
    _, counts = np.unique(np.column_stack([binned_lat, binned_lon]), axis=0, return_counts=True)
    return int(counts.max(initial=0))


def regional_hazard_counts(reports: pd.DataFrame) -> pd.DataFrame:
    if reports.empty:
        return pd.DataFrame(columns=["region", "tornado", "hail", "wind", "any", "sig_tornado", "sig_hail", "sig_wind"])
    grouped = reports.groupby("region", as_index=False).agg(
        tornado=("hazard", lambda values: int((values == "tornado").sum())),
        hail=("hazard", lambda values: int((values == "hail").sum())),
        wind=("hazard", lambda values: int((values == "wind").sum())),
        any=("hazard", "size"),
        sig_tornado=("sig_tornado", "sum"),
        sig_hail=("sig_hail", "sum"),
        sig_wind=("sig_wind", "sum"),
    )
    return grouped.sort_values("any", ascending=False).reset_index(drop=True)


def classify_outbreak_day(reports: pd.DataFrame, settings: AppSettings) -> dict[str, int | float | str]:
    thresholds = outbreak_thresholds(settings)
    reports = _significant_report_flags(reports, settings)
    tornado_count = int((reports["hazard"] == "tornado").sum())
    hail_count = int((reports["hazard"] == "hail").sum())
    wind_count = int((reports["hazard"] == "wind").sum())
    any_count = int(len(reports))
    sig_tornado_count = int(reports.get("sig_tornado", pd.Series(dtype=int)).sum())
    sig_hail_count = int(reports.get("sig_hail", pd.Series(dtype=int)).sum())
    sig_wind_count = int(reports.get("sig_wind", pd.Series(dtype=int)).sum())

    regional = regional_hazard_counts(reports)
    dominant_region = str(regional.iloc[0]["region"]) if not regional.empty else "none"
    dominant_region_reports = int(regional.iloc[0]["any"]) if not regional.empty else 0
    max_tornado_region = int(regional["tornado"].max()) if not regional.empty else 0
    max_hail_region = int(regional["hail"].max()) if not regional.empty else 0
    max_wind_region = int(regional["wind"].max()) if not regional.empty else 0
    max_sig_tor_region = int(regional["sig_tornado"].max()) if not regional.empty else 0

    tornado_score = sum(
        [
            tornado_count >= thresholds.tornado_reports,
            sig_tornado_count >= thresholds.significant_tornado_reports,
            max_tornado_region >= thresholds.tornado_region_reports,
            max_sig_tor_region >= max(1, thresholds.significant_tornado_reports - 1),
        ]
    )
    hail_score = sum(
        [
            hail_count >= thresholds.hail_reports,
            sig_hail_count >= thresholds.significant_hail_reports,
            max_hail_region >= thresholds.hail_region_reports,
        ]
    )
    wind_score = sum(
        [
            wind_count >= thresholds.wind_reports,
            sig_wind_count >= thresholds.significant_wind_reports,
            max_wind_region >= thresholds.wind_region_reports,
        ]
    )
    region_cluster = regional_cluster_count(reports)

    tornado_outbreak = int(tornado_score >= 2)
    significant_tornado_support = int(
        sig_tornado_count >= thresholds.significant_tornado_reports
        or (tornado_count >= max(6, thresholds.tornado_reports // 2) and max_sig_tor_region >= 2)
    )
    hail_outbreak = int(hail_score >= 2)
    wind_mcs_outbreak = int(wind_score >= 2 or (wind_count >= thresholds.wind_reports and region_cluster >= thresholds.regional_cluster_reports))
    any_outbreak = int(
        any_count >= thresholds.any_reports
        or dominant_region_reports >= thresholds.regional_cluster_reports
        or tornado_outbreak
        or hail_outbreak
        or wind_mcs_outbreak
    )
    active_non_outbreak = int(any_count >= thresholds.active_severe_reports and not any_outbreak)

    category = "null_day"
    if any_count > 0:
        category = "non_outbreak_severe_day"
    if active_non_outbreak:
        category = "active_non_outbreak_severe_day"
    if wind_mcs_outbreak:
        category = "wind_mcs_outbreak_day"
    if hail_outbreak:
        category = "hail_outbreak_day"
    if significant_tornado_support:
        category = "significant_tornado_support_day"
    if tornado_outbreak:
        category = "tornado_outbreak_day"
    if tornado_outbreak and significant_tornado_support:
        category = "significant_tornado_outbreak_day"

    outbreak_score = float(
        1.3 * tornado_score
        + 0.9 * hail_score
        + 0.9 * wind_score
        + 0.6 * (region_cluster >= thresholds.regional_cluster_reports)
        + 0.4 * (any_count >= thresholds.any_reports)
    )

    return {
        "tornado_outbreak": tornado_outbreak,
        "hail_outbreak": hail_outbreak,
        "wind_outbreak": wind_mcs_outbreak,
        "wind_mcs_outbreak": wind_mcs_outbreak,
        "any_outbreak": any_outbreak,
        "significant_tornado_support": significant_tornado_support,
        "active_non_outbreak_severe": active_non_outbreak,
        "regional_cluster_count": region_cluster,
        "dominant_region": dominant_region,
        "dominant_region_reports": dominant_region_reports,
        "report_count": any_count,
        "tornado_report_count": tornado_count,
        "hail_report_count": hail_count,
        "wind_report_count": wind_count,
        "significant_tornado_count": sig_tornado_count,
        "significant_hail_count": sig_hail_count,
        "significant_wind_count": sig_wind_count,
        "outbreak_score": outbreak_score,
        "category": category,
    }


def build_outbreak_table(reports: pd.DataFrame, start: str, end: str, settings: AppSettings) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    report_source_name = (
        str(reports["source_name"].iloc[0]) if "source_name" in reports.columns and not reports.empty else "unknown"
    )
    report_source_is_real = bool(reports["is_real_source"].all()) if "is_real_source" in reports.columns and not reports.empty else False
    report_ingest_timestamp = (
        str(reports["ingest_timestamp"].iloc[0]) if "ingest_timestamp" in reports.columns and not reports.empty else ""
    )
    rows: list[dict[str, int | float | str]] = []
    for valid_date in dates:
        valid = valid_date.date().isoformat()
        subset = reports.loc[reports["date"] == valid]
        row = {"date": valid}
        row.update(classify_outbreak_day(subset, settings))
        row["report_source_name"] = report_source_name
        row["report_source_is_real"] = report_source_is_real
        row["report_ingest_timestamp"] = report_ingest_timestamp
        rows.append(row)
    return pd.DataFrame(rows)
