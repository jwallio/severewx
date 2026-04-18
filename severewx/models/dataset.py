"""Training and inference dataset assembly."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from severewx.archive.feature_cache import discover_cached_feature_archives
from severewx.calibration.regional import assign_regions
from severewx.config import AppSettings
from severewx.features.base import flatten_feature_dataset
from severewx.features.composites import build_feature_dataset
from severewx.ingest.storage import historical_feature_path, historical_metadata_path
from severewx.utils.paths import DataPaths


HAZARD_TARGETS = ("tornado", "hail", "wind", "any")
STRICT_OUTBREAK_TARGETS = ("tornado_outbreak", "any_outbreak", "significant_tornado_support")
SYNTHETIC_SOURCE_KINDS = {"synthetic", "synthetic_fallback"}
OUTBREAK_DEFAULTS: dict[str, Any] = {
    "tornado_outbreak": 0,
    "hail_outbreak": 0,
    "wind_outbreak": 0,
    "wind_mcs_outbreak": 0,
    "any_outbreak": 0,
    "significant_tornado_support": 0,
    "active_non_outbreak_severe": 0,
    "regional_cluster_count": 0,
    "dominant_region": "unknown",
    "dominant_region_reports": 0,
    "report_count": 0,
    "tornado_report_count": 0,
    "hail_report_count": 0,
    "wind_report_count": 0,
    "significant_tornado_count": 0,
    "significant_hail_count": 0,
    "significant_wind_count": 0,
    "outbreak_score": 0.0,
    "category": "unknown",
}


def _archive_reporting_status(value: str) -> str:
    if value == "built_real":
        return "remote_real"
    if value == "built_partial_real":
        return "partial_real"
    return value


def _archive_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.get("archive_source", pd.Series("", index=frame.index)).eq("historical_feature_archive")


def _synthetic_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.get("forecast_source_kind", pd.Series("", index=frame.index)).isin(SYNTHETIC_SOURCE_KINDS)


def _real_archive_mask(frame: pd.DataFrame) -> pd.Series:
    return _archive_mask(frame) & ~_synthetic_mask(frame)


def _guardrail_thresholds(settings: AppSettings) -> dict[str, Any]:
    thresholds = settings.get("models.training_guardrails", {}) or {}
    return {
        "minimum_real_rows_overall": int(thresholds.get("minimum_real_rows_overall", 0)),
        "minimum_real_rows_by_hazard": {str(key): int(value) for key, value in (thresholds.get("minimum_real_rows_by_hazard", {}) or {}).items()},
        "minimum_real_rows_by_lead_day": {int(key): int(value) for key, value in (thresholds.get("minimum_real_rows_by_lead_day", {}) or {}).items()},
        "maximum_synthetic_fraction": float(thresholds.get("maximum_synthetic_fraction", 1.0)),
        "real_heavy_fraction": float(thresholds.get("real_heavy_fraction", 0.75)),
        "mixed_fraction": float(thresholds.get("mixed_fraction", 0.35)),
    }


def training_quality_tier(real_fraction: float, settings: AppSettings) -> str:
    thresholds = _guardrail_thresholds(settings)
    if real_fraction >= thresholds["real_heavy_fraction"]:
        return "real-heavy"
    if real_fraction >= thresholds["mixed_fraction"]:
        return "mixed"
    return "synthetic-heavy"


def evaluate_training_guardrails(
    frame: pd.DataFrame,
    settings: AppSettings,
    requested_hazards: list[str] | None = None,
) -> dict[str, Any]:
    requested_hazards = requested_hazards or [*HAZARD_TARGETS, "outbreak"]
    thresholds = _guardrail_thresholds(settings)
    real_mask = _real_archive_mask(frame)
    synthetic_mask = _synthetic_mask(frame)
    real_rows = int(real_mask.sum())
    synthetic_rows = int(synthetic_mask.sum())
    total_rows = int(len(frame))
    real_fraction = float(real_rows / max(total_rows, 1))
    synthetic_fraction = float(synthetic_rows / max(total_rows, 1))

    hazard_real_rows: dict[str, int] = {}
    hazard_real_fractions: dict[str, float] = {}
    for hazard, minimum in thresholds["minimum_real_rows_by_hazard"].items():
        if hazard not in requested_hazards or hazard not in frame.columns:
            continue
        positive_mask = frame[hazard].astype(float) > 0
        real_positive_rows = int((positive_mask & real_mask).sum())
        total_positive_rows = int(positive_mask.sum())
        hazard_real_rows[hazard] = real_positive_rows
        hazard_real_fractions[hazard] = float(real_positive_rows / max(total_positive_rows, 1))

    lead_day_real_rows: dict[int, int] = {}
    lead_day_real_fractions: dict[int, float] = {}
    if "lead_day" in frame.columns:
        for lead_day, group in frame.groupby("lead_day"):
            lead_day_int = int(lead_day)
            real_count = int(_real_archive_mask(group).sum())
            lead_day_real_rows[lead_day_int] = real_count
            lead_day_real_fractions[lead_day_int] = float(real_count / max(len(group), 1))

    failures: list[str] = []
    if real_rows < thresholds["minimum_real_rows_overall"]:
        failures.append(
            f"overall real archive rows {real_rows} below minimum {thresholds['minimum_real_rows_overall']}"
        )
    if synthetic_fraction > thresholds["maximum_synthetic_fraction"]:
        failures.append(
            f"synthetic fraction {synthetic_fraction:.3f} exceeds maximum {thresholds['maximum_synthetic_fraction']:.3f}"
        )
    for hazard, minimum in thresholds["minimum_real_rows_by_hazard"].items():
        if hazard not in requested_hazards or hazard not in frame.columns:
            continue
        if hazard_real_rows.get(hazard, 0) < minimum:
            failures.append(f"{hazard} real positive rows {hazard_real_rows.get(hazard, 0)} below minimum {minimum}")
    for lead_day, minimum in thresholds["minimum_real_rows_by_lead_day"].items():
        if lead_day_real_rows.get(lead_day, 0) < minimum:
            failures.append(f"lead day {lead_day} real rows {lead_day_real_rows.get(lead_day, 0)} below minimum {minimum}")

    quality_tier = training_quality_tier(real_fraction, settings)
    return {
        "requested_hazards": requested_hazards,
        "thresholds": thresholds,
        "passed": not failures,
        "failures": failures,
        "real_archive_rows": real_rows,
        "synthetic_rows": synthetic_rows,
        "real_fraction_overall": real_fraction,
        "synthetic_fraction_overall": synthetic_fraction,
        "real_rows_by_hazard": hazard_real_rows,
        "real_fraction_by_hazard": hazard_real_fractions,
        "real_rows_by_lead_day": {str(key): value for key, value in lead_day_real_rows.items()},
        "real_fraction_by_lead_day": {str(key): value for key, value in lead_day_real_fractions.items()},
        "training_quality_tier": quality_tier,
    }


@dataclass(slots=True)
class TrainingDataBundle:
    frame: pd.DataFrame
    metadata: dict[str, Any]


def _date_lookup(label_cube: xr.Dataset, date_value: str) -> xr.Dataset | None:
    if "date" not in label_cube.coords:
        return None
    try:
        return label_cube.sel(date=np.datetime64(date_value))
    except Exception:
        return None


def build_gridpoint_table(
    feature_dataset: xr.Dataset,
    label_cube: xr.Dataset | None = None,
    outbreak_table: pd.DataFrame | None = None,
    init_date: str | None = None,
    init_cycle: str | None = None,
    source_kind: str | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    outbreak_lookup = outbreak_table.set_index("date") if outbreak_table is not None and not outbreak_table.empty else None

    for valid_time in pd.to_datetime(feature_dataset["time"].values):
        feature_slice = feature_dataset.sel(time=valid_time)
        frame = flatten_feature_dataset(feature_slice)
        valid_date = valid_time.date().isoformat()
        frame["date"] = valid_date
        frame["lead_day"] = int(((valid_time - pd.to_datetime(feature_dataset["time"].values[0])).total_seconds() // 3600) // 24) + 1
        frame["init_date"] = init_date or pd.to_datetime(feature_dataset["time"].values[0]).date().isoformat()
        frame["init_cycle"] = init_cycle or pd.to_datetime(feature_dataset["time"].values[0]).strftime("%H")
        frame["forecast_source_kind"] = source_kind or str(feature_dataset.attrs.get("source", "unknown"))
        if label_cube is not None:
            label_slice = _date_lookup(label_cube, valid_date)
            for hazard in HAZARD_TARGETS:
                if label_slice is not None and hazard in label_slice:
                    label_frame = label_slice[hazard].to_dataframe(name=hazard).reset_index()[["lat", "lon", hazard]]
                    frame = frame.merge(label_frame, on=["lat", "lon"], how="left")
                else:
                    frame[hazard] = 0
        if outbreak_lookup is not None and valid_date in outbreak_lookup.index:
            outbreak_values = outbreak_lookup.loc[valid_date]
            if isinstance(outbreak_values, pd.DataFrame):
                outbreak_values = outbreak_values.iloc[0]
            for column, value in outbreak_values.to_dict().items():
                frame[column] = value
        else:
            for column, value in OUTBREAK_DEFAULTS.items():
                frame[column] = value
        frame["region"] = assign_regions(frame)
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    numeric_columns = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_columns] = combined[numeric_columns].fillna(0)
    return combined


def discover_feature_archives(paths: DataPaths) -> list[Path]:
    return sorted(paths.feature_archive.glob("*/*/*/features.parquet"))


def discover_training_forecasts(paths: DataPaths) -> list[Path]:
    return sorted(paths.processed.glob("forecast_*.nc"))


def archived_cycle_identifiers(paths: DataPaths) -> set[tuple[str, str]]:
    identifiers: set[tuple[str, str]] = set()
    for file_path in discover_feature_archives(paths):
        try:
            cycle = file_path.parent.name
            date = file_path.parent.parent.name
            identifiers.add((date, cycle))
        except Exception:
            continue
    return identifiers


def summarize_training_frame(
    frame: pd.DataFrame,
    source_preference: str,
    settings: AppSettings,
    requested_hazards: list[str] | None = None,
    degraded_reason: str = "",
) -> dict[str, Any]:
    real_mask = _real_archive_mask(frame)
    synthetic_mask = _synthetic_mask(frame)
    augmented = bool(real_mask.any() and synthetic_mask.any())
    guardrails = evaluate_training_guardrails(frame, settings, requested_hazards=requested_hazards)
    summary: dict[str, Any] = {
        "source_preference": source_preference,
        "row_count": int(len(frame)),
        "date_count": int(frame["date"].nunique()) if "date" in frame.columns else 0,
        "init_cycle_count": int(frame[["init_date", "init_cycle"]].drop_duplicates().shape[0]) if {"init_date", "init_cycle"}.issubset(frame.columns) else 0,
        "forecast_source_counts": frame.get("forecast_source_kind", pd.Series(dtype=str)).value_counts().to_dict(),
        "coverage_by_lead_day": [],
        "coverage_by_region": [],
        "coverage_by_hazard_and_lead_day": [],
        "hazard_positive_counts": {},
        "archive_row_counts_by_status": frame.get("archive_chunk_status", pd.Series(dtype=str)).astype(str).map(_archive_reporting_status).value_counts().to_dict(),
        "real_row_counts_by_source_kind": frame.loc[real_mask, "forecast_source_kind"].value_counts().to_dict() if "forecast_source_kind" in frame.columns else {},
        "real_archive_rows": int(real_mask.sum()),
        "synthetic_rows": int(synthetic_mask.sum()),
        "archive_preferred_mode": bool(
            source_preference.startswith("historical_feature_archive")
            or source_preference.startswith("cached_feature_archive")
        ),
        "archive_preferred_satisfied": bool(source_preference in {"cached_feature_archive", "historical_feature_archive"}),
        "archive_preferred_partially_satisfied": augmented,
        "training_cache_used": bool(source_preference.startswith("cached_feature_archive")),
        "guardrails_passed": bool(guardrails["passed"]),
        "guardrail_failures": guardrails["failures"],
        "guardrail_thresholds": guardrails["thresholds"],
        "training_allowed_without_override": bool(
            (not settings.get("models.prefer_feature_archive", True))
            or (source_preference in {"cached_feature_archive", "historical_feature_archive"} and guardrails["passed"])
        ),
        "real_fraction_overall": guardrails["real_fraction_overall"],
        "synthetic_fraction_overall": guardrails["synthetic_fraction_overall"],
        "real_rows_by_hazard": guardrails["real_rows_by_hazard"],
        "real_fraction_by_hazard": guardrails["real_fraction_by_hazard"],
        "real_rows_by_lead_day": guardrails["real_rows_by_lead_day"],
        "real_fraction_by_lead_day": guardrails["real_fraction_by_lead_day"],
        "training_quality_tier": guardrails["training_quality_tier"],
        "requested_hazards": guardrails["requested_hazards"],
        "degraded_mode_reason": degraded_reason,
        "final_training_mode": source_preference,
    }
    for hazard in HAZARD_TARGETS:
        if hazard in frame.columns:
            summary["hazard_positive_counts"][hazard] = int(frame[hazard].sum())
            if "lead_day" in frame.columns:
                hazard_lead = frame.groupby("lead_day", as_index=False)[hazard].sum()
                summary["coverage_by_hazard_and_lead_day"].extend(
                    hazard_lead.rename(columns={hazard: "positive_count"}).assign(hazard=hazard).to_dict(orient="records")
                )
    if {"lead_day", "forecast_source_kind"}.issubset(frame.columns):
        lead = frame.groupby(["lead_day", "forecast_source_kind"], as_index=False).size()
        summary["coverage_by_lead_day"] = lead.to_dict(orient="records")
    if {"region", "forecast_source_kind"}.issubset(frame.columns):
        region = frame.groupby(["region", "forecast_source_kind"], as_index=False).size()
        summary["coverage_by_region"] = region.to_dict(orient="records")
    return summary


def _load_archived_training_frame(paths: DataPaths) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for archive_file in discover_feature_archives(paths):
        frame = pd.read_parquet(archive_file)
        if "archive_source" not in frame.columns:
            frame["archive_source"] = "historical_feature_archive"
        tables.append(frame)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _load_cached_feature_training_frame(paths: DataPaths) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for archive_file in discover_cached_feature_archives(paths):
        frame = pd.read_parquet(archive_file)
        if "archive_source" not in frame.columns:
            frame["archive_source"] = "historical_feature_archive"
        tables.append(frame)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _required_training_targets(requested_hazards: list[str]) -> list[str]:
    required = [hazard for hazard in HAZARD_TARGETS if hazard in requested_hazards]
    if "outbreak" in requested_hazards:
        required.extend(list(STRICT_OUTBREAK_TARGETS))
    return required


def _strict_outbreak_training_issue(
    outbreak_table: pd.DataFrame | None,
    requested_hazards: list[str],
) -> str:
    if "outbreak" not in requested_hazards:
        return ""
    if outbreak_table is None or outbreak_table.empty:
        return "outbreak/full strict training requires outbreak labels built from real SPC reports"
    if "report_source_is_real" not in outbreak_table.columns:
        return "outbreak label table missing SPC report provenance column report_source_is_real"
    if not bool(outbreak_table["report_source_is_real"].fillna(False).all()):
        return "outbreak label table was built from synthetic/fallback SPC reports"
    missing_targets = [column for column in STRICT_OUTBREAK_TARGETS if column not in outbreak_table.columns]
    if missing_targets:
        return f"outbreak label table missing strict outbreak targets: {missing_targets}"
    if all(float(outbreak_table[column].sum()) == 0.0 for column in STRICT_OUTBREAK_TARGETS):
        return "outbreak label table has zero positives across strict outbreak targets"
    return ""


def _required_target_issue(frame: pd.DataFrame, required_targets: list[str]) -> str:
    missing_targets = [column for column in required_targets if column not in frame.columns]
    if missing_targets:
        return f"missing required hazard or outbreak columns: {missing_targets}"
    zero_positive_targets = [column for column in required_targets if float(frame[column].sum()) == 0.0]
    if zero_positive_targets:
        return f"zero-positive required hazard or outbreak columns: {zero_positive_targets}"
    return ""


def _overlay_outbreak_labels(frame: pd.DataFrame, outbreak_table: pd.DataFrame | None) -> pd.DataFrame:
    if outbreak_table is None or outbreak_table.empty or "date" not in frame.columns or "date" not in outbreak_table.columns:
        return frame
    overlay_columns = [column for column in outbreak_table.columns if column != "date"]
    if not overlay_columns:
        return frame
    updated = frame.drop(columns=[column for column in overlay_columns if column in frame.columns], errors="ignore")
    return updated.merge(outbreak_table[["date", *overlay_columns]], on="date", how="left")


def load_training_table(
    paths: DataPaths,
    settings: AppSettings,
    label_cube: xr.Dataset | None = None,
    outbreak_table: pd.DataFrame | None = None,
    analog_archive_path: Path | None = None,
    requested_hazards: list[str] | None = None,
    allow_degraded: bool = False,
) -> TrainingDataBundle:
    requested_hazards = requested_hazards or [*HAZARD_TARGETS, "outbreak"]
    required_targets = _required_training_targets(requested_hazards)
    strict_outbreak_issue = _strict_outbreak_training_issue(outbreak_table, requested_hazards)
    if settings.get("models.prefer_feature_archive", True):
        archived = _load_cached_feature_training_frame(paths)
        source_name = "cached_feature_archive"
        if archived.empty:
            archived = _load_archived_training_frame(paths)
            source_name = "historical_feature_archive"
        if not archived.empty:
            archived = archived.replace([np.inf, -np.inf], np.nan)
            archived_numeric = archived.select_dtypes(include=[np.number]).columns
            archived[archived_numeric] = archived[archived_numeric].fillna(0)
            if "outbreak" in requested_hazards:
                archived = _overlay_outbreak_labels(archived, outbreak_table)
            target_issue = _required_target_issue(archived, required_targets)
            if strict_outbreak_issue or target_issue:
                degraded_reason = strict_outbreak_issue or target_issue
                if allow_degraded:
                    synthetic = synthesize_training_table(settings)
                    archived = pd.concat([archived, synthetic], ignore_index=True)
                    return TrainingDataBundle(
                        frame=archived,
                        metadata=summarize_training_frame(
                            archived,
                            f"{source_name}_degraded_augmented",
                            settings,
                            requested_hazards=requested_hazards,
                            degraded_reason=degraded_reason,
                        ),
                    )
                return TrainingDataBundle(
                    frame=archived,
                    metadata=summarize_training_frame(
                        archived,
                        f"{source_name}_degraded",
                        settings,
                        requested_hazards=requested_hazards,
                        degraded_reason=degraded_reason,
                    ),
                )
            summary = summarize_training_frame(
                archived,
                source_name,
                settings,
                requested_hazards=requested_hazards,
            )
            if summary["guardrails_passed"]:
                return TrainingDataBundle(frame=archived, metadata=summary)
            if allow_degraded:
                synthetic = synthesize_training_table(settings)
                augmented = pd.concat([archived, synthetic], ignore_index=True)
                degraded_reason = "; ".join(summary["guardrail_failures"]) or "archive guardrails not satisfied"
                return TrainingDataBundle(
                    frame=augmented,
                    metadata=summarize_training_frame(
                        augmented,
                        f"{source_name}_degraded_augmented",
                        settings,
                        requested_hazards=requested_hazards,
                        degraded_reason=degraded_reason,
                    ),
                )
            degraded_reason = "; ".join(summary["guardrail_failures"]) or "archive guardrails not satisfied"
            return TrainingDataBundle(
                frame=archived,
                metadata=summarize_training_frame(
                    archived,
                    f"{source_name}_degraded",
                    settings,
                    requested_hazards=requested_hazards,
                    degraded_reason=degraded_reason,
                ),
            )

    tables: list[pd.DataFrame] = []
    for forecast_file in discover_training_forecasts(paths):
        forecast = xr.load_dataset(forecast_file)
        init_tokens = forecast_file.stem.split("_")
        init_date = init_tokens[1] if len(init_tokens) > 2 else pd.to_datetime(forecast["time"].values[0]).date().isoformat()
        init_cycle = init_tokens[2] if len(init_tokens) > 2 else pd.to_datetime(forecast["time"].values[0]).strftime("%H")
        features = build_feature_dataset(forecast, settings=settings, analog_archive_path=analog_archive_path)
        tables.append(
            build_gridpoint_table(
                features,
                label_cube=label_cube,
                outbreak_table=outbreak_table,
                init_date=init_date,
                init_cycle=init_cycle,
                source_kind=str(forecast.attrs.get("source", "processed_forecast_cache")),
            )
        )
    if not tables:
        synthetic = synthesize_training_table(settings)
        return TrainingDataBundle(
            frame=synthetic,
            metadata=summarize_training_frame(
                synthetic,
                "synthetic_fallback",
                settings,
                requested_hazards=requested_hazards,
                degraded_reason="no historical feature archive or processed forecast cache available",
            ),
        )

    combined = pd.concat(tables, ignore_index=True)
    target_issue = strict_outbreak_issue or _required_target_issue(combined, required_targets)
    if target_issue.startswith("missing required hazard or outbreak columns"):
        synthetic = synthesize_training_table(settings)
        return TrainingDataBundle(
            frame=synthetic,
            metadata=summarize_training_frame(
                synthetic,
                "synthetic_fallback",
                settings,
                requested_hazards=requested_hazards,
                degraded_reason=target_issue,
            ),
        )
    if target_issue:
        synthetic = synthesize_training_table(settings)
        combined = pd.concat([combined, synthetic], ignore_index=True)
        return TrainingDataBundle(
            frame=combined,
            metadata=summarize_training_frame(
                combined,
                "processed_forecast_cache_degraded_augmented",
                settings,
                requested_hazards=requested_hazards,
                degraded_reason=target_issue,
            ),
        )
    return TrainingDataBundle(
        frame=combined,
        metadata=summarize_training_frame(
            combined,
            "processed_forecast_cache",
            settings,
            requested_hazards=requested_hazards,
        ),
    )


def synthesize_training_table(settings: AppSettings, n_dates: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(int(settings.get("models.random_state", 42)))
    dates = pd.date_range("2024-04-01", periods=n_dates, freq="D")
    rows: list[dict[str, float | int | str]] = []
    regimes = [
        "tornado_outbreak",
        "hail_outbreak",
        "wind_outbreak",
        "significant_tornado_support",
        "active_severe",
        "null",
    ]
    for index, valid_date in enumerate(dates):
        regime = regimes[index % len(regimes)]
        for lat in np.arange(30.0, 45.0, 2.5):
            for lon in np.arange(-103.0, -85.0, 2.5):
                for lead_day in range(1, 5):
                    regime_core = np.exp(-(((lat - 36.0) ** 2) / 18.0 + ((lon + 94.0) ** 2) / 36.0))
                    lead_decay = max(0.45, 1.0 - 0.18 * (lead_day - 1))
                    if regime == "tornado_outbreak":
                        cape = float(rng.normal(2100, 220) * (0.85 + 0.35 * regime_core))
                        shear = float(rng.normal(34, 4) * (0.9 + 0.25 * regime_core))
                        ll_shear = float(rng.normal(18, 3) * (0.9 + 0.25 * regime_core))
                        td2m = float(rng.normal(291, 2))
                        lapse = float(rng.normal(6.8, 0.3))
                        forcing = float(rng.normal(62, 7))
                    elif regime == "hail_outbreak":
                        cape = float(rng.normal(2600, 280) * (0.85 + 0.30 * regime_core))
                        shear = float(rng.normal(26, 4))
                        ll_shear = float(rng.normal(11, 2.5))
                        td2m = float(rng.normal(287, 2.5))
                        lapse = float(rng.normal(7.7, 0.35))
                        forcing = float(rng.normal(40, 6))
                    elif regime == "wind_outbreak":
                        cape = float(rng.normal(1700, 260))
                        shear = float(rng.normal(31, 4))
                        ll_shear = float(rng.normal(10, 2.5))
                        td2m = float(rng.normal(288, 2.5))
                        lapse = float(rng.normal(6.6, 0.3))
                        forcing = float(rng.normal(70, 7))
                    elif regime == "significant_tornado_support":
                        cape = float(rng.normal(1700, 220) * (0.9 + 0.25 * regime_core))
                        shear = float(rng.normal(30, 3.5) * (0.9 + 0.15 * regime_core))
                        ll_shear = float(rng.normal(16, 2.5) * (0.9 + 0.15 * regime_core))
                        td2m = float(rng.normal(290, 2))
                        lapse = float(rng.normal(6.7, 0.25))
                        forcing = float(rng.normal(52, 7))
                    elif regime == "active_severe":
                        cape = float(rng.normal(1300, 220))
                        shear = float(rng.normal(21, 3.5))
                        ll_shear = float(rng.normal(8, 2.0))
                        td2m = float(rng.normal(286, 2.5))
                        lapse = float(rng.normal(6.4, 0.25))
                        forcing = float(rng.normal(35, 6))
                    else:
                        cape = float(rng.normal(400, 150))
                        shear = float(rng.normal(14, 3))
                        ll_shear = float(rng.normal(5, 1.5))
                        td2m = float(rng.normal(281, 3))
                        lapse = float(rng.normal(5.9, 0.25))
                        forcing = float(rng.normal(20, 5))

                    cape *= lead_decay
                    shear *= 0.9 + 0.12 * lead_decay
                    ll_shear *= 0.9 + 0.12 * lead_decay
                    analog_similarity = float(np.clip(rng.normal(0.72 if "outbreak" in regime or "support" in regime else 0.32, 0.12), 0, 1))
                    low_lcl_support = float(np.clip(rng.normal(0.95 if regime in {"tornado_outbreak", "significant_tornado_support"} else 0.55, 0.12), 0, 1.2))
                    moisture_quality = float(np.clip((td2m - 287.0) / 6.0, 0, 1.5))
                    helicity = float(np.clip(ll_shear * shear * (1.2 + regime_core), 10, 450))
                    scp_proxy = float(np.clip((cape / 1800.0) * (shear / 20.0) * (helicity / 180.0), 0, 8))
                    stp_proxy = float(np.clip((cape / 1800.0) * (shear / 20.0) * (ll_shear / 12.0) * low_lcl_support * moisture_quality, 0, 8))
                    cape_shear_overlap = float(np.clip((cape / 1800.0) * (shear / 20.0), 0, 5))
                    tornado_overlap = float(np.clip(cape_shear_overlap * (ll_shear / 12.0) * low_lcl_support * moisture_quality, 0, 6))
                    hail_overlap = float(np.clip((cape / 2200.0) * (lapse / 7.0) * (shear / 22.0), 0, 6))
                    wind_overlap = float(np.clip((shear / 22.0) * (forcing / 65.0) * ((td2m - 280.0) / 10.0), 0, 6))
                    sig_support = float(np.clip(0.5 * stp_proxy + 0.25 * tornado_overlap + 0.15 * (forcing / 70.0) + 0.1 * (ll_shear / 12.0), 0, 2.5))
                    tornado = int(regime in {"tornado_outbreak", "significant_tornado_support"} and tornado_overlap > 0.9 and ll_shear > 11 and rng.random() > 0.28)
                    hail = int((regime == "hail_outbreak" and hail_overlap > 0.85 and rng.random() > 0.22) or (cape > 1000 and lapse > 6.8 and rng.random() > 0.6))
                    wind = int((regime == "wind_outbreak" and wind_overlap > 0.8 and rng.random() > 0.22) or (forcing > 48 and shear > 22 and rng.random() > 0.62))
                    any_severe = int(max(tornado, hail, wind))
                    any_outbreak = int(regime in {"tornado_outbreak", "hail_outbreak", "wind_outbreak"} and any_severe)
                    sig_tornado_support = int(regime in {"tornado_outbreak", "significant_tornado_support"} and sig_support > 0.7)
                    rows.append(
                        {
                            "date": valid_date.date().isoformat(),
                            "lead_day": lead_day,
                            "init_date": valid_date.date().isoformat(),
                            "init_cycle": "00",
                            "forecast_source_kind": "synthetic",
                            "lat": lat,
                            "lon": lon,
                            "region": assign_region_for_point(lat, lon),
                            "cape": cape,
                            "cin": float(rng.normal(-45 if regime in {"tornado_outbreak", "significant_tornado_support"} else -70, 25)),
                            "t2m": float(rng.normal(296, 4)),
                            "td2m": td2m,
                            "mslp": float(rng.normal(100700 if regime in {"tornado_outbreak", "wind_outbreak"} else 101200, 220)),
                            "wind10m_speed": float(rng.normal(12 if regime != "null" else 7, 2)),
                            "shear_0_6km": shear,
                            "ll_shear_proxy": ll_shear,
                            "llj_speed": float(rng.normal(22 if regime in {"tornado_outbreak", "significant_tornado_support"} else 15, 3)),
                            "helicity_proxy": helicity,
                            "pwat": float(rng.normal(34 if regime in {"tornado_outbreak", "wind_outbreak"} else 24, 4)),
                            "moisture_transport": float(rng.normal(450 if regime in {"tornado_outbreak", "wind_outbreak"} else 220, 70)),
                            "moisture_quality": moisture_quality,
                            "low_lcl_support": low_lcl_support,
                            "lapse_rate_700_500": lapse,
                            "forcing_proxy": forcing,
                            "forcing_instability_overlap": float(np.clip((forcing / 90.0) * (cape / 1800.0), 0, 2)),
                            "scp_proxy": scp_proxy,
                            "stp_proxy": stp_proxy,
                            "cape_shear_overlap": cape_shear_overlap,
                            "tornado_favored_overlap": tornado_overlap,
                            "hail_favored_overlap": hail_overlap,
                            "wind_favored_overlap": wind_overlap,
                            "analog_similarity": analog_similarity,
                            "analog_tornado_similarity": analog_similarity if regime in {"tornado_outbreak", "significant_tornado_support"} else analog_similarity * 0.45,
                            "analog_hail_similarity": analog_similarity if regime == "hail_outbreak" else analog_similarity * 0.45,
                            "analog_wind_similarity": analog_similarity if regime == "wind_outbreak" else analog_similarity * 0.45,
                            "analog_tornado_rate": analog_similarity * (0.9 if regime in {"tornado_outbreak", "significant_tornado_support"} else 0.25),
                            "analog_hail_rate": analog_similarity * (0.85 if regime == "hail_outbreak" else 0.25),
                            "analog_wind_rate": analog_similarity * (0.85 if regime == "wind_outbreak" else 0.25),
                            "analog_outbreak_support": analog_similarity * (0.85 if any_outbreak else 0.25),
                            "analog_sigtor_support": analog_similarity * (0.9 if sig_tornado_support else 0.15),
                            "analog_mode_coherence": float(np.clip(analog_similarity - 0.25 if any_outbreak else 0.2, 0, 1)),
                            "prior_run_delta": float(rng.normal(120 if any_outbreak else 210, 35)),
                            "neighborhood_mean_any": float(rng.normal(0.78 if any_outbreak else 0.28, 0.1)),
                            "synoptic_support": float(rng.normal(82 if regime in {"tornado_outbreak", "wind_outbreak"} else 38, 10)),
                            "sig_tor_support": sig_support,
                            "spatial_coverage": float(rng.normal(0.68 if any_outbreak else 0.18, 0.12)),
                            "tornado_corridor_index": float(rng.normal(0.8 if regime in {"tornado_outbreak", "significant_tornado_support"} else 0.22, 0.12)),
                            "outbreak_corridor_index": float(rng.normal(0.82 if any_outbreak else 0.2, 0.12)),
                            "tornado": tornado,
                            "hail": hail,
                            "wind": wind,
                            "any": any_severe,
                            "tornado_outbreak": int(regime == "tornado_outbreak"),
                            "hail_outbreak": int(regime == "hail_outbreak"),
                            "wind_outbreak": int(regime == "wind_outbreak"),
                            "any_outbreak": any_outbreak,
                            "significant_tornado_support": sig_tornado_support,
                            "category": regime,
                        }
                    )
    frame = pd.DataFrame(rows)
    return frame


def assign_region_for_point(lat: float, lon: float) -> str:
    frame = pd.DataFrame({"lat": [lat], "lon": [lon]})
    return str(assign_regions(frame).iloc[0])
