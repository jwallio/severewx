"""Daily tornado event catalog utilities built from tor.json feature records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class TornadoCatalogThresholds:
    significant_tornado_outbreak_count: int = 3
    significant_tornado_outbreak_tornado_count: int = 10
    tornado_outbreak_count: int = 8
    tornado_outbreak_significant_count: int = 2
    moderate_tornado_day_count: int = 4
    moderate_tornado_day_significant_count: int = 1


OUTBREAK_CLASS_RANK = {
    "significant_tornado_outbreak": 4,
    "tornado_outbreak": 3,
    "moderate_tornado_day": 2,
    "low_tornado_day": 1,
}

DEFAULT_PILOT_CLASS_TARGETS = {
    "significant_tornado_outbreak": 5,
    "tornado_outbreak": 5,
    "moderate_tornado_day": 5,
    "low_tornado_day": 5,
}


def load_tornado_features(json_path: Path | str) -> list[dict[str, Any]]:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("tor.json must be a list of feature-like records")
    return [item for item in payload if isinstance(item, dict)]


def _feature_rows(features: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        properties = feature.get("properties", {}) or {}
        feature_id = properties.get("id")
        start_time = properties.get("datetime-start")
        if not feature_id or not start_time:
            continue
        states = properties.get("states")
        if isinstance(states, list):
            normalized_states = sorted(str(value).strip() for value in states if str(value).strip())
        elif states is None:
            normalized_states = []
        else:
            normalized_states = [str(states).strip()] if str(states).strip() else []
        raw_f_scale = properties.get("f-scale")
        if isinstance(raw_f_scale, list):
            numeric_f_scale = [int(value) for value in raw_f_scale if value is not None]
            f_scale = max(numeric_f_scale) if numeric_f_scale else pd.NA
        elif raw_f_scale is None:
            f_scale = pd.NA
        else:
            f_scale = int(raw_f_scale)
        rows.append(
            {
                "id": str(feature_id),
                "date": pd.Timestamp(start_time).tz_convert("UTC").date().isoformat()
                if pd.Timestamp(start_time).tzinfo is not None
                else pd.Timestamp(start_time, tz="UTC").date().isoformat(),
                "f_scale": f_scale,
                "states": normalized_states,
                "path_length_mi": float(properties["path-length"]) if properties.get("path-length") is not None else pd.NA,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["id", "date", "f_scale", "states", "path_length_mi"])
    frame = pd.DataFrame(rows)
    frame["f_scale"] = pd.to_numeric(frame["f_scale"], errors="coerce").astype("Int64")
    frame["path_length_mi"] = pd.to_numeric(frame["path_length_mi"], errors="coerce")
    return frame


def classify_tornado_day(
    tornado_count: int,
    significant_tornado_count: int,
    thresholds: TornadoCatalogThresholds,
) -> str:
    if (
        significant_tornado_count >= thresholds.significant_tornado_outbreak_count
        or (
            tornado_count >= thresholds.significant_tornado_outbreak_tornado_count
            and significant_tornado_count >= thresholds.tornado_outbreak_significant_count
        )
    ):
        return "significant_tornado_outbreak"
    if (
        tornado_count >= thresholds.tornado_outbreak_count
        or significant_tornado_count >= thresholds.tornado_outbreak_significant_count
    ):
        return "tornado_outbreak"
    if (
        tornado_count >= thresholds.moderate_tornado_day_count
        or significant_tornado_count >= thresholds.moderate_tornado_day_significant_count
    ):
        return "moderate_tornado_day"
    return "low_tornado_day"


def build_daily_tornado_summary(
    features: list[dict[str, Any]],
    thresholds: TornadoCatalogThresholds | None = None,
) -> pd.DataFrame:
    thresholds = thresholds or TornadoCatalogThresholds()
    frame = _feature_rows(features)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "tornado_count",
                "significant_tornado_count",
                "max_f_scale",
                "states_affected",
                "states_affected_count",
                "total_path_length_mi",
                "outbreak_class",
                "outbreak_rank",
            ]
        )

    def _state_union(series: pd.Series) -> str:
        values = sorted({state for states in series for state in states})
        return ";".join(values)

    grouped = (
        frame.drop_duplicates(subset=["date", "id"])
        .groupby("date", as_index=False)
        .agg(
            tornado_count=("id", "nunique"),
            significant_tornado_count=("f_scale", lambda values: int((values.fillna(-1) >= 2).sum())),
            max_f_scale=("f_scale", lambda values: int(values.dropna().max()) if not values.dropna().empty else pd.NA),
            states_affected=("states", _state_union),
            total_path_length_mi=("path_length_mi", lambda values: float(values.dropna().sum()) if not values.dropna().empty else 0.0),
        )
    )
    grouped["states_affected_count"] = grouped["states_affected"].map(lambda value: 0 if not value else len(str(value).split(";")))
    grouped["outbreak_class"] = grouped.apply(
        lambda row: classify_tornado_day(
            int(row["tornado_count"]),
            int(row["significant_tornado_count"]),
            thresholds,
        ),
        axis=1,
    )
    grouped["outbreak_rank"] = grouped["outbreak_class"].map(OUTBREAK_CLASS_RANK).astype(int)
    return grouped.sort_values("date").reset_index(drop=True)


def build_tornado_candidate_dates(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    candidates = summary.loc[summary["outbreak_rank"] >= OUTBREAK_CLASS_RANK["moderate_tornado_day"]].copy()
    if candidates.empty:
        candidates = summary.copy()
    return candidates.sort_values(
        ["outbreak_rank", "significant_tornado_count", "tornado_count", "max_f_scale", "total_path_length_mi", "date"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def thresholds_dict(thresholds: TornadoCatalogThresholds) -> dict[str, Any]:
    return asdict(thresholds)


def load_tornado_summary_csv(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    return frame


def _rank_tornado_days(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["date"] = pd.to_datetime(ranked["date"])
    ranked["decade"] = (ranked["date"].dt.year // 10) * 10
    return ranked.sort_values(
        ["significant_tornado_count", "tornado_count", "max_f_scale", "total_path_length_mi", "states_affected_count", "date"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)


def build_tornado_pilot_dates(
    summary: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    class_targets: dict[str, int] | None = None,
    *,
    max_per_decade: int | None = None,
    min_spacing_days: int = 0,
) -> pd.DataFrame:
    class_targets = class_targets or dict(DEFAULT_PILOT_CLASS_TARGETS)
    output_columns = [
        "date",
        "outbreak_class",
        "tornado_count",
        "significant_tornado_count",
        "max_f_scale",
        "total_path_length_mi",
        "states_affected_count",
    ]
    if summary.empty:
        return pd.DataFrame(columns=output_columns)

    summary_ranked = _rank_tornado_days(summary)
    candidate_ranked = _rank_tornado_days(candidates) if candidates is not None and not candidates.empty else summary_ranked
    selected_rows: list[pd.Series] = []

    for outbreak_class, target_count in class_targets.items():
        source = summary_ranked if outbreak_class == "low_tornado_day" else candidate_ranked
        subset = source.loc[source["outbreak_class"] == outbreak_class].copy()
        if subset.empty or target_count <= 0:
            continue
        picked_dates: list[pd.Timestamp] = []
        decade_counts: dict[int, int] = {}
        for _, row in subset.iterrows():
            date_value = pd.Timestamp(row["date"])
            decade = int(row["decade"])
            if max_per_decade is not None and decade_counts.get(decade, 0) >= max_per_decade:
                continue
            if min_spacing_days > 0 and any(abs((date_value - prior).days) < min_spacing_days for prior in picked_dates):
                continue
            selected_rows.append(row)
            picked_dates.append(date_value)
            decade_counts[decade] = decade_counts.get(decade, 0) + 1
            if len(picked_dates) >= int(target_count):
                break

    if not selected_rows:
        return pd.DataFrame(columns=output_columns)

    pilot = pd.DataFrame(selected_rows).copy()
    pilot["date"] = pd.to_datetime(pilot["date"]).dt.date.astype(str)
    pilot["outbreak_rank"] = pilot["outbreak_class"].map(OUTBREAK_CLASS_RANK).astype(int)
    pilot = pilot.sort_values(
        ["outbreak_rank", "significant_tornado_count", "tornado_count", "max_f_scale", "total_path_length_mi", "date"],
        ascending=[False, False, False, False, False, False],
    )
    return pilot[output_columns].reset_index(drop=True)
