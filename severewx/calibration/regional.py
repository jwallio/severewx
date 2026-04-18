"""Broad regional groupings for calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_region(lat: float, lon: float) -> str:
    if lon < -110:
        return "west"
    if lon < -100 and lat < 40:
        return "south_plains"
    if lon < -84 and lat < 38:
        return "southeast"
    if lon < -84:
        return "east"
    return "midwest"


def assign_regions(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(lambda row: assign_region(float(row["lat"]), float(row["lon"])), axis=1)
