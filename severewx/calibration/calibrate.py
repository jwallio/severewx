"""Post-hoc calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from severewx.calibration.regional import assign_regions
from severewx.config import AppSettings


@dataclass(slots=True)
class GroupCalibrator:
    method: str
    groups: dict[tuple[str, int], Any]
    global_model: Any
    use_region: bool
    use_lead_day: bool

    def apply(self, frame: pd.DataFrame, score_column: str) -> np.ndarray:
        working = frame.copy()
        working["region"] = assign_regions(working)
        result = np.zeros(len(working), dtype=float)
        for position, (_, row) in enumerate(working.iterrows()):
            key = self.group_key(str(row["region"]), int(row["lead_day"]))
            model = self.groups.get(key, self.global_model)
            score = float(row[score_column])
            if self.method == "isotonic":
                result[position] = float(model.predict([score])[0])
            else:
                result[position] = float(model.predict_proba([[score]])[0, 1])
        return np.clip(result, 0.0, 1.0)

    def group_key(self, region: str, lead_day: int) -> tuple[str, int]:
        region_key = region if self.use_region else "all"
        lead_key = lead_day if self.use_lead_day else -1
        return region_key, lead_key


def _fit_model(method: str, x: np.ndarray, y: np.ndarray) -> Any:
    if method == "platt":
        model = LogisticRegression()
        model.fit(x.reshape(-1, 1), y)
        return model
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x, y)
    return model


def fit_group_calibrator(frame: pd.DataFrame, score_column: str, target_column: str, settings: AppSettings) -> GroupCalibrator:
    method = str(settings.get("calibration.method", "isotonic")).lower()
    working = frame.copy()
    working["region"] = assign_regions(working)
    use_region = bool(settings.get("calibration.by_region", True))
    use_lead_day = bool(settings.get("calibration.by_lead_day", True))
    min_group_samples = int(settings.get("calibration.min_group_samples", 25))
    groups: dict[tuple[str, int], Any] = {}
    global_model = _fit_model(method, working[score_column].to_numpy(dtype=float), working[target_column].to_numpy(dtype=int))
    group_columns = []
    if use_region:
        group_columns.append("region")
    if use_lead_day:
        group_columns.append("lead_day")
    if group_columns:
        for keys, subset in working.groupby(group_columns):
            if subset[target_column].nunique() < 2 or len(subset) < min_group_samples:
                continue
            if not isinstance(keys, tuple):
                keys = (keys,)
            region = str(keys[group_columns.index("region")]) if "region" in group_columns else "all"
            lead_day = int(keys[group_columns.index("lead_day")]) if "lead_day" in group_columns else -1
            groups[(region, lead_day)] = _fit_model(
                method,
                subset[score_column].to_numpy(dtype=float),
                subset[target_column].to_numpy(dtype=int),
            )
    return GroupCalibrator(method=method, groups=groups, global_model=global_model, use_region=use_region, use_lead_day=use_lead_day)
