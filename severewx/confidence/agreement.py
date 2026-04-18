"""Agreement metrics across hazard outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_agreement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    hazard_matrix = frame[["tornado_prob", "hail_prob", "wind_prob"]].to_numpy(dtype=float)
    spread = hazard_matrix.std(axis=1)
    sorted_hazards = np.sort(hazard_matrix, axis=1)
    dominant_margin = sorted_hazards[:, -1] - sorted_hazards[:, -2]
    any_alignment = 1.0 - np.abs(frame["any_prob"].to_numpy(dtype=float) - hazard_matrix.max(axis=1))
    consensus = 1.0 - np.clip(spread / 0.28, 0.0, 1.0)
    frame["dominant_hazard_margin"] = np.clip(dominant_margin, 0.0, 1.0)
    frame["agreement_score"] = np.clip(0.45 * consensus + 0.30 * any_alignment + 0.25 * frame["dominant_hazard_margin"], 0.0, 1.0)
    frame["disagreement_score"] = 1.0 - frame["agreement_score"]
    return frame
