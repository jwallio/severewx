from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import xarray as xr

from severewx.cli.tornado_concern_eval import _latest_forecast_metadata_for_date, _latest_prediction_for_date
from severewx.cli.tornado_concern_product_audit import audit_tornado_concern_product
from severewx.config import load_settings
from severewx.models.forecast_consensus import CONSENSUS_FIELD, consensus_metadata_path, consensus_product_path
from severewx.models.tornado_concern import (
    envelope_tornado_concern_grid,
    tornado_environment_outlook_grid,
    tornado_environment_outlook_hybrid_grid,
    tornado_environment_outlook_v2_grid,
)
from severewx.render.layout import SUBTITLE_COLOR, TEXT_COLOR, conus_template, create_map_figure, map_draw_kwargs, style_map_axes
from severewx.utils.paths import build_paths


PRODUCT_BINS = (0.00, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)
PRODUCT_COLORS = ("#fff1ea", "#ffd0bf", "#fcae91", "#fb7b5f", "#ef4538", "#d11a22", "#a50f15", "#7f0009", "#520005")
OUTLOOK_BINS = (0.00, 0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, 1.00)
OUTLOOK_COLORS = ("#ffffff00", "#77be7a", "#c7a994", "#ffea7a", "#ff7078", "#bf83d8", "#94008b", "#354d86")
CONTOUR_DISPLAY_INTERPOLATION_FACTOR = 5
CONTOUR_DISPLAY_MIN_THRESHOLD = 0.02
PUBLIC_DISPLAY_NEIGHBORHOOD_RADIUS = 2
PUBLIC_DISPLAY_ENVELOPE_GAIN = 3.5
PUBLIC_DISPLAY_PEAK_SUPPORT_CAP_GAIN = 2.5
PUBLIC_DISPLAY_SUPPORT_FRACTION_FLOOR = 0.15
PUBLIC_DISPLAY_FULL_SUPPORT_FRACTION = 0.35
PUBLIC_DISPLAY_LABEL_LEVELS = (0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)
OUTLOOK_LABEL_LEVELS_BY_PRESET = {
    "weak": (0.05, 0.10, 0.15, 0.30, 0.45, 0.60),
    "standard": (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60),
    "broad": (0.10, 0.15, 0.30, 0.45, 0.60),
}
PUBLIC_DISPLAY_MIN_VISIBLE_CELLS_GE_02 = 4
PUBLIC_DISPLAY_MIN_VISIBLE_CELLS_GE_05 = 1
OUTLOOK_DISPLAY_MIN_OBJECT_CELLS = 24
OUTLOOK_DISPLAY_PRESETS = {"auto", "weak", "standard", "broad"}
DEFAULT_PRODUCT_FIELD = "tornado_environment_outlook_hybrid"
DEFAULT_CONSENSUS_PRODUCT_FIELD = CONSENSUS_FIELD
DEFAULT_PRODUCT_MAP_STYLE = "outlook"
DEFAULT_PRODUCT_MAP_DOMAIN = "regional"
OUTLOOK_PRESET_OBJECT_CELLS = {
    "weak": ((0.02, 8), (0.05, 4), (0.10, 3), (0.15, 2), (0.30, 2), (0.45, 2)),
    "standard": ((0.02, 24), (0.05, 10), (0.10, 4), (0.15, 1), (0.30, 1), (0.45, 1)),
    "broad": ((0.02, 120), (0.05, 55), (0.10, 12), (0.15, 4), (0.30, 1), (0.45, 1)),
}
OUTLOOK_LABEL_MIN_CELLS = {
    "weak": {0.02: 16, 0.05: 8, 0.10: 3, 0.15: 1, 0.30: 1, 0.45: 1, 0.60: 1},
    "standard": {0.02: 45, 0.05: 25, 0.10: 10, 0.15: 4, 0.30: 1, 0.45: 1, 0.60: 1},
    "broad": {0.02: 180, 0.05: 110, 0.10: 45, 0.15: 18, 0.30: 4, 0.45: 2, 0.60: 1},
}
OUTLOOK_INGREDIENT_FIELDS = (
    "sig_tor_support",
    "tornado_favored_overlap",
    "scp_proxy",
    "low_lcl_support",
    "synoptic_support",
    "outbreak_risk",
    "cin",
)
REGIONAL_MIN_WIDTH_DEGREES = 14.0
REGIONAL_MIN_HEIGHT_DEGREES = 9.0
REGIONAL_PADDING_DEGREES = 2.0
CONUS_EXTENT = (-125.0, -66.5, 24.0, 50.0)


def _normalize_cycle(value: str) -> str:
    return str(value).zfill(2)


def _read_dates_file(path: Path) -> list[str]:
    dates: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            dates.append(value)
    return dates


def _markdown_table(rows: list[tuple[str, Any]]) -> str:
    header = "| field | value |"
    divider = "| --- | --- |"
    body = [f"| {field} | {value} |" for field, value in rows]
    return "\n".join([header, divider, *body])


def _interp_axis(values: np.ndarray, source_axis: np.ndarray, target_axis: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(values, axis, 0)
    interpolated = np.empty((len(target_axis), *moved.shape[1:]), dtype=float)
    for index in np.ndindex(moved.shape[1:]):
        interpolated[(slice(None), *index)] = np.interp(target_axis, source_axis, moved[(slice(None), *index)])
    return np.moveaxis(interpolated, 0, axis)


def _upsample_grid_for_display(
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    values: np.ndarray,
    factor: int = CONTOUR_DISPLAY_INTERPOLATION_FACTOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if factor <= 1 or values.ndim != 2 or len(lat_values) < 2 or len(lon_values) < 2:
        return lon_values, lat_values, values
    target_lat = np.linspace(float(lat_values[0]), float(lat_values[-1]), (len(lat_values) - 1) * factor + 1)
    target_lon = np.linspace(float(lon_values[0]), float(lon_values[-1]), (len(lon_values) - 1) * factor + 1)
    source_lat = np.asarray(lat_values, dtype=float)
    source_lon = np.asarray(lon_values, dtype=float)
    working_values = np.asarray(values, dtype=float)
    if source_lat[0] > source_lat[-1]:
        source_lat = source_lat[::-1]
        working_values = working_values[::-1, :]
    if source_lon[0] > source_lon[-1]:
        source_lon = source_lon[::-1]
        working_values = working_values[:, ::-1]
    interp_lat = _interp_axis(working_values, source_lat, np.sort(target_lat), axis=0)
    interp_values = _interp_axis(interp_lat, source_lon, np.sort(target_lon), axis=1)
    if target_lat[0] > target_lat[-1]:
        interp_values = interp_values[::-1, :]
    if target_lon[0] > target_lon[-1]:
        interp_values = interp_values[:, ::-1]
    return target_lon, target_lat, interp_values


def _neighborhood_mean(values: np.ndarray, *, radius: int) -> np.ndarray:
    if radius <= 0 or values.ndim != 2:
        return np.asarray(values, dtype=float)
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    padded = np.pad(clean, radius, mode="edge")
    output = np.zeros_like(clean, dtype=float)
    weights = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=float)
    sigma = max(radius / 1.6, 0.75)
    for row_offset in range(2 * radius + 1):
        for col_offset in range(2 * radius + 1):
            dy = row_offset - radius
            dx = col_offset - radius
            distance = float(np.hypot(dy, dx))
            if distance <= radius + 0.01:
                weights[row_offset, col_offset] = float(np.exp(-(distance**2) / (2.0 * sigma**2)))
    weights /= float(weights.sum()) if weights.sum() else 1.0
    for row_offset in range(2 * radius + 1):
        for col_offset in range(2 * radius + 1):
            output += (
                padded[row_offset : row_offset + clean.shape[0], col_offset : col_offset + clean.shape[1]]
                * weights[row_offset, col_offset]
            )
    return output


def _public_display_concern_field(values: np.ndarray) -> np.ndarray:
    """Build a map-only field that damps isolated bullseyes but keeps supported risk envelopes."""
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if clean.ndim != 2 or clean.size == 0:
        return clean
    local_mean = _neighborhood_mean(clean, radius=PUBLIC_DISPLAY_NEIGHBORHOOD_RADIUS)
    support_fraction = _neighborhood_mean((clean >= CONTOUR_DISPLAY_MIN_THRESHOLD).astype(float), radius=PUBLIC_DISPLAY_NEIGHBORHOOD_RADIUS)
    support_ramp = np.clip(
        (support_fraction - PUBLIC_DISPLAY_SUPPORT_FRACTION_FLOOR)
        / max(PUBLIC_DISPLAY_FULL_SUPPORT_FRACTION - PUBLIC_DISPLAY_SUPPORT_FRACTION_FLOOR, 0.001),
        0.0,
        1.0,
    )
    envelope = local_mean * PUBLIC_DISPLAY_ENVELOPE_GAIN * support_ramp
    support_gain = 0.75 + (PUBLIC_DISPLAY_PEAK_SUPPORT_CAP_GAIN - 0.75) * support_ramp
    supported_peak_cap = local_mean * support_gain
    supported_raw = np.minimum(clean, supported_peak_cap)
    raw_max = float(np.nanmax(clean)) if clean.size else 0.0
    return np.clip(np.maximum(envelope, supported_raw), 0.0, min(1.0, raw_max))


def _filter_small_outlook_objects(values: np.ndarray, *, min_cells: int = OUTLOOK_DISPLAY_MIN_OBJECT_CELLS) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if clean.ndim != 2 or min_cells <= 1:
        return clean
    active = clean >= CONTOUR_DISPLAY_MIN_THRESHOLD
    visited = np.zeros(active.shape, dtype=bool)
    filtered = clean.copy()
    rows, cols = active.shape
    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not active[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            component: list[tuple[int, int]] = []
            visited[start_row, start_col] = True
            while stack:
                row, col = stack.pop()
                component.append((row, col))
                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        if row_offset == 0 and col_offset == 0:
                            continue
                        next_row = row + row_offset
                        next_col = col + col_offset
                        if (
                            0 <= next_row < rows
                            and 0 <= next_col < cols
                            and not visited[next_row, next_col]
                            and active[next_row, next_col]
                        ):
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))
            if len(component) < min_cells:
                for row, col in component:
                    filtered[row, col] = 0.0
    return filtered


def _component_sizes(mask: np.ndarray) -> list[int]:
    active = np.asarray(mask, dtype=bool)
    if active.ndim != 2 or not active.any():
        return []
    visited = np.zeros(active.shape, dtype=bool)
    rows, cols = active.shape
    sizes: list[int] = []
    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not active[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            size = 0
            while stack:
                row, col = stack.pop()
                size += 1
                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        if row_offset == 0 and col_offset == 0:
                            continue
                        next_row = row + row_offset
                        next_col = col + col_offset
                        if (
                            0 <= next_row < rows
                            and 0 <= next_col < cols
                            and active[next_row, next_col]
                            and not visited[next_row, next_col]
                        ):
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))
            sizes.append(size)
    return sizes


def _largest_component_size(mask: np.ndarray) -> int:
    sizes = _component_sizes(mask)
    return max(sizes) if sizes else 0


def _expand_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    active = np.asarray(mask, dtype=bool)
    if active.ndim != 2 or radius <= 0 or not active.any():
        return active
    expanded = np.zeros_like(active, dtype=bool)
    rows, cols = np.where(active)
    offsets = [
        (row_offset, col_offset)
        for row_offset in range(-radius, radius + 1)
        for col_offset in range(-radius, radius + 1)
        if float(np.hypot(row_offset, col_offset)) <= radius + 0.01
    ]
    row_count, col_count = active.shape
    for row, col in zip(rows, cols, strict=False):
        for row_offset, col_offset in offsets:
            next_row = int(row) + row_offset
            next_col = int(col) + col_offset
            if 0 <= next_row < row_count and 0 <= next_col < col_count:
                expanded[next_row, next_col] = True
    return expanded


def _resolve_display_preset(values: np.ndarray, requested: str = "auto") -> str:
    if requested not in OUTLOOK_DISPLAY_PRESETS:
        raise ValueError(f"unsupported display preset: {requested}")
    if requested != "auto":
        return requested
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if clean.size == 0:
        return "weak"
    ge02 = int(np.count_nonzero(clean >= 0.02))
    ge10 = int(np.count_nonzero(clean >= 0.10))
    max_value = float(np.nanmax(clean)) if clean.size else 0.0
    largest_ge02 = _largest_component_size(clean >= 0.02)
    if ge02 >= 1400 or largest_ge02 >= 1300:
        return "broad"
    if max_value < 0.10 or ge10 < 100 or ge02 < 160:
        return "weak"
    return "standard"


def _tier_filter_outlook_objects(values: np.ndarray, *, preset: str) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if clean.ndim != 2:
        return clean
    retained = np.zeros_like(clean, dtype=float)
    for threshold, min_cells in sorted(OUTLOOK_PRESET_OBJECT_CELLS[preset], reverse=True):
        filtered = _filter_small_outlook_objects(np.where(clean >= threshold, clean, 0.0), min_cells=min_cells)
        retained = np.maximum(retained, filtered)
    if preset == "broad":
        if np.any(retained >= 0.30):
            keep_enhanced = _expand_mask(retained >= 0.30, radius=4) | (retained >= 0.30)
            retained = np.where((retained >= 0.15) & (retained < 0.30) & ~keep_enhanced, 0.0, retained)
        if np.any(retained >= 0.15):
            keep_high = _expand_mask(retained >= 0.15, radius=5) | (retained >= 0.15)
            retained = np.where((retained >= 0.10) & (retained < 0.15) & ~keep_high, 0.0, retained)
        high_support = retained >= 0.10
        mid_support = retained >= 0.05
        keep_mid = _expand_mask(high_support, radius=4) | (retained >= 0.15)
        keep_low = _expand_mask(mid_support & keep_mid, radius=4) | (retained >= 0.10)
        retained = np.where((retained >= 0.05) & (retained < 0.10) & ~keep_mid, 0.0, retained)
        retained = np.where((retained >= 0.02) & (retained < 0.05) & ~keep_low, 0.0, retained)
    return retained


def _public_display_outlook_field(values: np.ndarray, *, display_preset: str = "auto") -> np.ndarray:
    """Map-facing outlook field: preserve source grid, but suppress isolated public-display specks."""
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    preset = _resolve_display_preset(clean, display_preset)
    return _tier_filter_outlook_objects(np.where(clean >= CONTOUR_DISPLAY_MIN_THRESHOLD, clean, 0.0), preset=preset)


def _display_object_metrics(values: np.ndarray) -> dict[str, int]:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ge02_sizes = _component_sizes(clean >= 0.02)
    return {
        "display_object_count_ge_02pct": len(ge02_sizes),
        "display_largest_object_cells_ge_02pct": max(ge02_sizes) if ge02_sizes else 0,
    }


def _visible_label_levels(values: np.ndarray, *, display_preset: str = "standard") -> tuple[float, ...]:
    finite_max = float(np.nanmax(values)) if np.isfinite(values).any() else 0.0
    levels: list[float] = []
    minimums = OUTLOOK_LABEL_MIN_CELLS.get(display_preset, OUTLOOK_LABEL_MIN_CELLS["standard"])
    candidate_levels = OUTLOOK_LABEL_LEVELS_BY_PRESET.get(display_preset, PUBLIC_DISPLAY_LABEL_LEVELS)
    for level in candidate_levels:
        if finite_max < level:
            continue
        if int(np.count_nonzero(np.asarray(values, dtype=float) >= level)) >= int(minimums.get(level, 1)):
            levels.append(level)
    high_levels = [level for level in candidate_levels if level >= 0.10 and finite_max >= level]
    if high_levels and not any(level >= 0.10 for level in levels):
        levels.append(max(high_levels))
    return tuple(sorted(set(levels)))


def _display_extent(
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    values: np.ndarray,
    *,
    map_domain: str,
) -> tuple[float, float, float, float] | None:
    if map_domain != "regional":
        return None
    clean = np.asarray(values, dtype=float)
    active = np.isfinite(clean) & (clean >= CONTOUR_DISPLAY_MIN_THRESHOLD)
    if clean.ndim != 2 or not active.any():
        return None
    rows, cols = np.where(active)
    lat = np.asarray(lat_values, dtype=float)
    lon = np.asarray(lon_values, dtype=float)
    lon_min = float(np.nanmin(lon[cols])) - REGIONAL_PADDING_DEGREES
    lon_max = float(np.nanmax(lon[cols])) + REGIONAL_PADDING_DEGREES
    lat_min = float(np.nanmin(lat[rows])) - REGIONAL_PADDING_DEGREES
    lat_max = float(np.nanmax(lat[rows])) + REGIONAL_PADDING_DEGREES
    if lon_max - lon_min < REGIONAL_MIN_WIDTH_DEGREES:
        center = 0.5 * (lon_min + lon_max)
        lon_min = center - 0.5 * REGIONAL_MIN_WIDTH_DEGREES
        lon_max = center + 0.5 * REGIONAL_MIN_WIDTH_DEGREES
    if lat_max - lat_min < REGIONAL_MIN_HEIGHT_DEGREES:
        center = 0.5 * (lat_min + lat_max)
        lat_min = center - 0.5 * REGIONAL_MIN_HEIGHT_DEGREES
        lat_max = center + 0.5 * REGIONAL_MIN_HEIGHT_DEGREES
    conus_lon_min, conus_lon_max, conus_lat_min, conus_lat_max = CONUS_EXTENT
    return (
        max(conus_lon_min, lon_min),
        min(conus_lon_max, lon_max),
        max(conus_lat_min, lat_min),
        min(conus_lat_max, lat_max),
    )


def _set_map_extent(ax: Any, extent: tuple[float, float, float, float]) -> None:
    lon_min, lon_max, lat_min, lat_max = extent
    try:
        import cartopy.crs as ccrs  # type: ignore

        if hasattr(ax, "set_extent"):
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            return
    except Exception:
        pass
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)


def _annotate_no_visible_signal(ax: Any, *, map_style: str) -> None:
    if map_style not in {"contours", "outlook"}:
        return
    label = "No visible 2% outlook signal" if map_style == "outlook" else "No visible 2% tornado concern signal"
    ax.text(
        0.5,
        0.50,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4d5560",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f6f8fa", "edgecolor": "#d4dbe2", "alpha": 0.92},
        zorder=6,
    )


def _prediction_artifact_for_cycle(outputs_dir: Path, date: str, cycle: str) -> Path | None:
    exact = outputs_dir / f"forecast_products_{date}_{cycle}.nc"
    if exact.exists():
        return exact
    return _latest_prediction_for_date(outputs_dir, date)


def _consensus_artifact_for_cycle(outputs_dir: Path, date: str, cycle: str) -> Path | None:
    exact = consensus_product_path(outputs_dir, date, cycle)
    return exact if exact.exists() else None


def _select_prediction_artifact(outputs_dir: Path, date: str, cycle: str, field_name: str, *, artifact_source: str = "auto") -> tuple[Path | None, str]:
    if artifact_source == "prediction":
        if field_name == DEFAULT_CONSENSUS_PRODUCT_FIELD:
            raise ValueError("--artifact-source prediction cannot be combined with the consensus field")
        return _prediction_artifact_for_cycle(outputs_dir, date, cycle), field_name
    if artifact_source == "consensus":
        return _consensus_artifact_for_cycle(outputs_dir, date, cycle), DEFAULT_CONSENSUS_PRODUCT_FIELD if field_name == DEFAULT_PRODUCT_FIELD else field_name
    if artifact_source != "auto":
        raise ValueError(f"unsupported artifact source: {artifact_source}")
    if field_name == DEFAULT_PRODUCT_FIELD:
        consensus = _consensus_artifact_for_cycle(outputs_dir, date, cycle)
        if consensus is not None:
            return consensus, DEFAULT_CONSENSUS_PRODUCT_FIELD
    if field_name == DEFAULT_CONSENSUS_PRODUCT_FIELD:
        return _consensus_artifact_for_cycle(outputs_dir, date, cycle), field_name
    return _prediction_artifact_for_cycle(outputs_dir, date, cycle), field_name


def _forecast_metadata_for_cycle(outputs_dir: Path, date: str, cycle: str) -> Path | None:
    exact = outputs_dir / f"forecast_metadata_{date}_{cycle}.json"
    if exact.exists():
        return exact
    return _latest_forecast_metadata_for_date(outputs_dir, date)


def _metadata_for_prediction_artifact(outputs_dir: Path, date: str, cycle: str, prediction_path: Path) -> Path | None:
    if prediction_path.name == consensus_product_path(outputs_dir, date, cycle).name:
        exact = consensus_metadata_path(outputs_dir, date, cycle)
        return exact if exact.exists() else None
    return _forecast_metadata_for_cycle(outputs_dir, date, cycle)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _valid_date_strings(dataset: xr.Dataset) -> list[str]:
    times = pd.to_datetime(dataset["time"].values)
    return [timestamp.date().isoformat() for timestamp in times]


def _valid_time_strings(dataset: xr.Dataset) -> list[str]:
    if "time" not in dataset.coords:
        return []
    times = pd.to_datetime(dataset["time"].values)
    return [timestamp.strftime("%Y-%m-%dT%H:%M:%SZ") for timestamp in times]


def _select_valid_date_dataset(dataset: xr.Dataset, valid_date: str) -> xr.Dataset:
    if "time" not in dataset.coords:
        return dataset
    dates = pd.to_datetime(dataset["time"].values).date
    target_date = pd.Timestamp(valid_date).date()
    mask = np.asarray([date == target_date for date in dates], dtype=bool)
    if not bool(mask.any()):
        available = ", ".join(list(dict.fromkeys(date.isoformat() for date in dates)))
        raise ValueError(f"forecast artifact has no valid times for {valid_date}; available valid dates: {available}")
    return dataset.isel(time=mask)


def _select_valid_time_window_dataset(dataset: xr.Dataset, valid_start: str, valid_end: str) -> xr.Dataset:
    if "time" not in dataset.coords:
        raise ValueError("forecast artifact has no time coordinate for a custom valid-time window")
    times = pd.to_datetime(dataset["time"].values)
    start = pd.Timestamp(valid_start)
    end = pd.Timestamp(valid_end)
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"invalid valid-time window: {valid_start} to {valid_end}")
    if end <= start:
        raise ValueError(f"--valid-end must be after --valid-start: {valid_start} to {valid_end}")
    mask = np.asarray((times >= start) & (times < end), dtype=bool)
    if not bool(mask.any()):
        available = ", ".join(timestamp.strftime("%Y-%m-%dT%H:%M:%SZ") for timestamp in times)
        raise ValueError(f"forecast artifact has no valid times in {valid_start} to {valid_end}; available valid times: {available}")
    return dataset.isel(time=mask)


def _top_valid_date(dataset: xr.Dataset, field_name: str) -> str:
    if field_name not in dataset:
        return ""
    field = dataset[field_name]
    if "time" not in field.dims:
        dates = _valid_date_strings(dataset)
        return dates[0] if dates else ""
    series = field.max(dim=[dim for dim in field.dims if dim != "time"], skipna=True)
    values = np.asarray(series.values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        return ""
    index = int(np.nanargmax(values))
    return _valid_date_strings(dataset)[index]


def _best_valid_date(dataset: xr.Dataset, field_name: str) -> str:
    if field_name not in dataset:
        raise KeyError(f"missing required forecast field: {field_name}")
    if "time" not in dataset[field_name].dims:
        dates = _valid_date_strings(dataset)
        return dates[0] if dates else ""
    return _top_valid_date(dataset, field_name)


def _derive_environment_envelope_dataset(dataset: xr.Dataset) -> xr.Dataset:
    if "tornado_concern_environment_envelope" in dataset:
        return dataset
    if "tornado_concern_prob" not in dataset:
        return dataset
    field = dataset["tornado_concern_prob"]
    if "time" not in field.dims:
        envelope = envelope_tornado_concern_grid(
            np.asarray(field.values, dtype=float),
            sig_tor_support=np.asarray(dataset["sig_tor_support"].values, dtype=float) if "sig_tor_support" in dataset else None,
            tornado_favored_overlap=np.asarray(dataset["tornado_favored_overlap"].values, dtype=float) if "tornado_favored_overlap" in dataset else None,
            scp_proxy=np.asarray(dataset["scp_proxy"].values, dtype=float) if "scp_proxy" in dataset else None,
            outbreak_risk=np.asarray(dataset["outbreak_risk"].values, dtype=float) if "outbreak_risk" in dataset else None,
        )
        return dataset.assign(tornado_concern_environment_envelope=(field.dims, envelope))
    envelopes = []
    for index in range(dataset.sizes["time"]):
        subset = dataset.isel(time=index)
        envelopes.append(
            envelope_tornado_concern_grid(
                np.asarray(subset["tornado_concern_prob"].values, dtype=float),
                sig_tor_support=np.asarray(subset["sig_tor_support"].values, dtype=float) if "sig_tor_support" in subset else None,
                tornado_favored_overlap=np.asarray(subset["tornado_favored_overlap"].values, dtype=float) if "tornado_favored_overlap" in subset else None,
                scp_proxy=np.asarray(subset["scp_proxy"].values, dtype=float) if "scp_proxy" in subset else None,
                outbreak_risk=np.asarray(subset["outbreak_risk"].values, dtype=float) if "outbreak_risk" in subset else None,
            )
        )
    envelope_values = np.stack(envelopes, axis=0)
    return dataset.assign(tornado_concern_environment_envelope=(field.dims, envelope_values))


def _dataset_values(dataset: xr.Dataset, name: str) -> np.ndarray | None:
    return np.asarray(dataset[name].values, dtype=float) if name in dataset else None


def _derive_environment_outlook_dataset(dataset: xr.Dataset, *, field_name: str = "tornado_environment_outlook") -> xr.Dataset:
    if field_name in dataset:
        return dataset
    grid_funcs = {
        "tornado_environment_outlook": tornado_environment_outlook_grid,
        "tornado_environment_outlook_v2": tornado_environment_outlook_v2_grid,
        "tornado_environment_outlook_hybrid": tornado_environment_outlook_hybrid_grid,
    }
    grid_func = grid_funcs.get(field_name, tornado_environment_outlook_grid)
    template_name = next(
        (name for name in ("sig_tor_support", "tornado_favored_overlap", "scp_proxy") if name in dataset),
        None,
    )
    if template_name is None:
        return dataset
    field = dataset[template_name]
    if "time" not in field.dims:
        outlook = grid_func(
            sig_tor_support=_dataset_values(dataset, "sig_tor_support"),
            tornado_favored_overlap=_dataset_values(dataset, "tornado_favored_overlap"),
            scp_proxy=_dataset_values(dataset, "scp_proxy"),
            low_lcl_support=_dataset_values(dataset, "low_lcl_support"),
            synoptic_support=_dataset_values(dataset, "synoptic_support"),
            outbreak_risk=_dataset_values(dataset, "outbreak_risk"),
            cin=_dataset_values(dataset, "cin"),
        )
        return dataset.assign({field_name: (field.dims, outlook)})
    outlooks = []
    for index in range(dataset.sizes["time"]):
        subset = dataset.isel(time=index)
        outlooks.append(
            grid_func(
                sig_tor_support=_dataset_values(subset, "sig_tor_support"),
                tornado_favored_overlap=_dataset_values(subset, "tornado_favored_overlap"),
                scp_proxy=_dataset_values(subset, "scp_proxy"),
                low_lcl_support=_dataset_values(subset, "low_lcl_support"),
                synoptic_support=_dataset_values(subset, "synoptic_support"),
                outbreak_risk=_dataset_values(subset, "outbreak_risk"),
                cin=_dataset_values(subset, "cin"),
            )
        )
    outlook_values = np.stack(outlooks, axis=0)
    return dataset.assign({field_name: (field.dims, outlook_values)})


def _derive_product_field_dataset(dataset: xr.Dataset, field_name: str) -> xr.Dataset:
    if field_name == "tornado_concern_environment_envelope":
        return _derive_environment_envelope_dataset(dataset)
    if field_name in {"tornado_environment_outlook", "tornado_environment_outlook_v2", "tornado_environment_outlook_hybrid"}:
        return _derive_environment_outlook_dataset(dataset, field_name=field_name)
    return dataset


def _product_variant(field_name: str) -> str:
    if field_name == "tornado_concern_environment_envelope":
        return "environment_envelope"
    if field_name == "tornado_environment_outlook":
        return "environment_outlook"
    if field_name == "tornado_environment_outlook_v2":
        return "environment_outlook_v2"
    if field_name == "tornado_environment_outlook_hybrid":
        return "environment_outlook_hybrid"
    if field_name == DEFAULT_CONSENSUS_PRODUCT_FIELD:
        return "environment_outlook_hybrid_consensus"
    return "baseline"


def _product_stats(dataset: xr.Dataset, field_name: str) -> dict[str, Any]:
    if field_name not in dataset:
        raise KeyError(f"missing required forecast field: {field_name}")
    field = dataset[field_name]
    values = np.asarray(field.values, dtype=float)
    valid_dates = _valid_date_strings(dataset)
    valid_times = _valid_time_strings(dataset)
    grid_cell_count = int(np.prod([dataset.sizes.get(dim, 0) for dim in ("lat", "lon")])) if {"lat", "lon"}.issubset(dataset.sizes) else 0
    summary: dict[str, Any] = {
        "valid_date_start": valid_dates[0] if valid_dates else "",
        "valid_date_end": valid_dates[-1] if valid_dates else "",
        "valid_time_start": valid_times[0] if valid_times else "",
        "valid_time_end": valid_times[-1] if valid_times else "",
        "valid_times": valid_times,
        "valid_day_count": len(list(dict.fromkeys(valid_dates))),
        "grid_shape": {"lat": int(dataset.sizes.get("lat", 0)), "lon": int(dataset.sizes.get("lon", 0))},
        "grid_cell_count": grid_cell_count,
        "max_tornado_concern_prob": float(np.nanmax(values)) if values.size else float("nan"),
        "mean_tornado_concern_prob": float(np.nanmean(values)) if values.size else float("nan"),
        "top_valid_date": _top_valid_date(dataset, field_name),
    }
    for threshold in (0.10, 0.20, 0.35):
        mask = np.asarray(dataset[field_name].max("time").values, dtype=float) >= threshold if "time" in dataset[field_name].dims else values >= threshold
        covered = int(np.count_nonzero(mask)) if mask.size else 0
        summary[f"grid_cells_ge_{int(threshold * 100):02d}pct"] = covered
        summary[f"grid_fraction_ge_{int(threshold * 100):02d}pct"] = float(covered / grid_cell_count) if grid_cell_count else 0.0
    if "outbreak_risk" in dataset:
        summary["max_outbreak_risk"] = float(np.nanmax(np.asarray(dataset["outbreak_risk"].values, dtype=float)))
    if "sig_tor_support" in dataset:
        summary["max_sig_tor_support"] = float(np.nanmax(np.asarray(dataset["sig_tor_support"].values, dtype=float)))
    return summary


def _aggregate_grid(dataset: xr.Dataset, field_name: str) -> np.ndarray | None:
    if field_name not in dataset:
        return None
    field = dataset[field_name]
    aggregate = field.max("time") if "time" in field.dims else field
    return np.asarray(aggregate.values, dtype=float)


def _ingredient_diagnostics(dataset: xr.Dataset, field_name: str) -> dict[str, Any]:
    field_values = _aggregate_grid(dataset, field_name)
    if field_values is None or field_values.size == 0 or np.isnan(field_values).all():
        return {}
    peak_row, peak_col = np.unravel_index(int(np.nanargmax(field_values)), field_values.shape)
    diagnostics: dict[str, Any] = {
        "peak_lat": float(np.asarray(dataset["lat"].values, dtype=float)[peak_row]) if "lat" in dataset else float("nan"),
        "peak_lon": float(np.asarray(dataset["lon"].values, dtype=float)[peak_col]) if "lon" in dataset else float("nan"),
        "ingredient_maxima": {},
        "ingredient_values_at_peak": {},
    }
    for ingredient in OUTLOOK_INGREDIENT_FIELDS:
        values = _aggregate_grid(dataset, ingredient)
        if values is None or values.size == 0:
            continue
        diagnostics["ingredient_maxima"][ingredient] = float(np.nanmax(values))
        diagnostics["ingredient_values_at_peak"][ingredient] = float(values[peak_row, peak_col])
    normalized = {
        "sig_tor_support": float(diagnostics["ingredient_values_at_peak"].get("sig_tor_support", 0.0) or 0.0) / 1.25,
        "tornado_favored_overlap": float(diagnostics["ingredient_values_at_peak"].get("tornado_favored_overlap", 0.0) or 0.0) / 2.50,
        "scp_proxy": float(diagnostics["ingredient_values_at_peak"].get("scp_proxy", 0.0) or 0.0) / 0.75,
    }
    normalized = {key: float(np.clip(value, 0.0, 1.0)) for key, value in normalized.items()}
    diagnostics["ingredient_normalized_at_peak"] = normalized
    diagnostics["limiting_ingredient_at_peak"] = min(normalized, key=normalized.get) if normalized else ""
    return diagnostics


def _public_display_stats(
    dataset: xr.Dataset,
    field_name: str,
    *,
    map_style: str = "contours",
    display_preset: str = "auto",
) -> dict[str, Any]:
    if field_name not in dataset:
        return {}
    field = dataset[field_name]
    aggregate = field.max("time") if "time" in field.dims else field
    source_values = np.asarray(aggregate.values, dtype=float)
    resolved_preset = _resolve_display_preset(source_values, display_preset) if map_style == "outlook" else "standard"
    display_values = (
        _public_display_outlook_field(source_values, display_preset=resolved_preset)
        if map_style == "outlook"
        else _public_display_concern_field(source_values)
    )
    stats: dict[str, Any] = {
        "max_public_display_tornado_concern_prob": float(np.nanmax(display_values)) if display_values.size else float("nan"),
        "mean_public_display_tornado_concern_prob": float(np.nanmean(display_values)) if display_values.size else float("nan"),
        "display_preset_effective": resolved_preset,
        **_display_object_metrics(display_values),
    }
    for threshold in (0.01, 0.02, 0.05, 0.10, 0.20, 0.35):
        stats[f"public_display_grid_cells_ge_{int(threshold * 100):02d}pct"] = int(np.count_nonzero(display_values >= threshold)) if display_values.size else 0
    return stats


def _display_readiness_audit(display_stats: dict[str, Any], *, map_style: str) -> dict[str, Any]:
    if map_style not in {"contours", "outlook"}:
        return {"status": "skipped", "public_ready": True, "failure_reasons": ""}
    ge2 = int(display_stats.get("public_display_grid_cells_ge_02pct", 0) or 0)
    ge5 = int(display_stats.get("public_display_grid_cells_ge_05pct", 0) or 0)
    display_max = float(display_stats.get("max_public_display_tornado_concern_prob", 0.0) or 0.0)
    failures: list[str] = []
    if display_max < CONTOUR_DISPLAY_MIN_THRESHOLD:
        failures.append("display_no_visible_signal")
    if ge2 < PUBLIC_DISPLAY_MIN_VISIBLE_CELLS_GE_02:
        failures.append("display_too_sparse_ge_02pct")
    if map_style == "contours" and ge5 < PUBLIC_DISPLAY_MIN_VISIBLE_CELLS_GE_05:
        failures.append("display_no_5pct_area")
    return {
        "status": "ok" if not failures else "flagged",
        "public_ready": not failures,
        "failure_reasons": ";".join(failures),
        "min_visible_cells_ge_02pct": PUBLIC_DISPLAY_MIN_VISIBLE_CELLS_GE_02,
        "min_visible_cells_ge_05pct": PUBLIC_DISPLAY_MIN_VISIBLE_CELLS_GE_05,
    }


def _consensus_readiness_audit(dataset: xr.Dataset, field_name: str) -> dict[str, Any]:
    if field_name != DEFAULT_CONSENSUS_PRODUCT_FIELD:
        return {"status": "skipped", "public_ready": True, "failure_reasons": ""}
    failures: list[str] = []
    if "model_agreement_count" not in dataset:
        failures.append("consensus_missing_agreement_count")
        max_signal_agreement = 0
    else:
        field_values = _aggregate_grid(dataset, field_name)
        agreement_values = _aggregate_grid(dataset, "model_agreement_count")
        if field_values is None or agreement_values is None:
            max_signal_agreement = 0
        else:
            signal_mask = np.asarray(field_values, dtype=float) >= CONTOUR_DISPLAY_MIN_THRESHOLD
            max_signal_agreement = int(np.nanmax(np.where(signal_mask, agreement_values, 0.0))) if signal_mask.any() else 0
        if max_signal_agreement < 2:
            failures.append("consensus_less_than_two_supporting_sources")
    confidence_values = _aggregate_grid(dataset, "consensus_confidence_modifier")
    min_confidence_modifier = float(np.nanmin(confidence_values)) if confidence_values is not None and confidence_values.size else float("nan")
    mean_confidence_modifier = float(np.nanmean(confidence_values)) if confidence_values is not None and confidence_values.size else float("nan")
    return {
        "status": "ok" if not failures else "flagged",
        "public_ready": not failures,
        "failure_reasons": ";".join(failures),
        "max_signal_agreement_count": max_signal_agreement,
        "min_confidence_modifier": min_confidence_modifier,
        "mean_confidence_modifier": mean_confidence_modifier,
    }


CORE_PUBLIC_SOURCES = {"hrrr_recent", "rap_recent", "nam_recent", "aws_recent"}
PUBLIC_CANDIDATE_SOURCE = "ecmwf_recent"


def _source_readiness_audit(forecast_metadata: dict[str, Any], field_name: str) -> dict[str, Any]:
    if field_name != DEFAULT_CONSENSUS_PRODUCT_FIELD:
        return {"status": "skipped", "public_ready": True, "failure_reasons": ""}
    ingest = forecast_metadata.get("ingest_summary", {}) if isinstance(forecast_metadata.get("ingest_summary", {}), dict) else {}
    included = {str(source) for source in forecast_metadata.get("included_sources", []) if str(source)}
    excluded = forecast_metadata.get("excluded_sources", [])
    failures: list[str] = []
    if str(ingest.get("source_mode", "real")).lower() == "synthetic":
        failures.append("source_synthetic_ingest")
    if not bool(ingest.get("real_ingest_available", True)):
        failures.append("source_real_ingest_unavailable")
    missing_core = sorted(CORE_PUBLIC_SOURCES - included)
    if missing_core:
        failures.append("source_missing_core:" + ",".join(missing_core))
    if PUBLIC_CANDIDATE_SOURCE not in included:
        failures.append(f"source_missing_candidate:{PUBLIC_CANDIDATE_SOURCE}")
    return {
        "status": "ok" if not failures else "flagged",
        "public_ready": not failures,
        "failure_reasons": ";".join(failures),
        "required_core_sources": sorted(CORE_PUBLIC_SOURCES),
        "candidate_source": PUBLIC_CANDIDATE_SOURCE,
        "included_sources": sorted(included),
        "excluded_sources": excluded if isinstance(excluded, list) else [],
    }


def _render_readiness_audit(render_metadata: dict[str, Any], *, require_production_basemap: bool) -> dict[str, Any]:
    if not require_production_basemap:
        return {"status": "skipped", "public_ready": True, "failure_reasons": ""}
    if bool(render_metadata.get("render_uses_cartopy", False)):
        return {"status": "ok", "public_ready": True, "failure_reasons": ""}
    return {
        "status": "flagged",
        "public_ready": False,
        "failure_reasons": "render_basemap_fallback",
    }


def _format_valid_time(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %HZ")


def _format_valid_period_label(valid_date: str, valid_start: str | None = None, valid_end: str | None = None) -> str:
    if valid_start and valid_end:
        return f"{_format_valid_time(valid_start)} to {_format_valid_time(valid_end)}"
    return f"{valid_date} 00-24 UTC"


def _valid_period_slug(valid_date: str, valid_start: str | None = None, valid_end: str | None = None) -> str:
    if not (valid_start and valid_end):
        return valid_date
    start = pd.Timestamp(valid_start).strftime("%Y-%m-%d_%Hz").lower()
    end = pd.Timestamp(valid_end).strftime("%Y-%m-%d_%Hz").lower()
    return f"{start}_to_{end}"


def _default_title(valid_label: str) -> str:
    return f"Tornado Concern Outlook | Valid {valid_label}"


def _default_summary_text(stats: dict[str, Any], *, field_name: str = "tornado_concern_prob", valid_label: str | None = None) -> str:
    start = str(stats.get("valid_date_start", ""))
    label = valid_label or f"{start} 00-24 UTC"
    peak = float(stats.get("max_tornado_concern_prob", float("nan")))
    peak_date = str(stats.get("top_valid_date", "")) or "unknown"
    if field_name in {"tornado_environment_outlook", "tornado_environment_outlook_v2", "tornado_environment_outlook_hybrid", DEFAULT_CONSENSUS_PRODUCT_FIELD}:
        return (
            "This 24-hour prototype shows an ingredient-only tornado environment outlook from saved forecast fields. "
            f"It avoids the learned tornado-concern gridpoint field and highlights coherent overlap of significant-tornado support, "
            f"tornado-favored overlap, and SCP-style support for {label}."
        )
    if np.isnan(peak):
        return f"Baseline tornado-concern prototype using the saved forecast artifact. Valid {label}."
    return (
        "This 24-hour map shows the maximum baseline tornado-concern signal from the saved forecast artifact "
        f"for {label}. The strongest concern in this product peaks on {peak_date} "
        f"with a max raw gridpoint value of {peak:.3f}. The public map uses a neighborhood-supported display field "
        "so isolated gridpoint spikes are damped and the lightest red shading begins at 2%."
    )


def _ensure_writable(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    conflicts = [path for path in paths if path.exists()]
    if conflicts:
        joined = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(f"refusing to overwrite existing product outputs: {joined}")


def _render_product_map(
    dataset: xr.Dataset,
    *,
    field_name: str,
    date: str,
    cycle: str,
    valid_date: str,
    valid_period_label: str,
    title: str,
    summary_text: str,
    image_path: Path,
    map_style: str = "contours",
    map_domain: str = "conus",
    display_preset: str = "auto",
    settings: Any | None = None,
) -> dict[str, Any]:
    if field_name not in dataset:
        raise KeyError(f"forecast artifact is missing {field_name}")
    aggregate = dataset[field_name].max("time") if "time" in dataset[field_name].dims else dataset[field_name]
    lat_values = dataset["lat"].values
    lon_values = dataset["lon"].values
    settings = settings or load_settings()
    render_template = conus_template(settings)
    fig, ax = create_map_figure(settings)
    values = np.asarray(aggregate.values, dtype=float)
    draw_kwargs = map_draw_kwargs(ax)
    label_count = 0
    extent_values = values
    display_visible_signal = True
    resolved_preset = _resolve_display_preset(values, display_preset) if map_style == "outlook" else "standard"
    if map_style == "outlook":
        public_display_values = _public_display_outlook_field(values, display_preset=resolved_preset)
        extent_values = public_display_values
        display_lon, display_lat, display_values = _upsample_grid_for_display(lon_values, lat_values, public_display_values)
        display_values = np.where(display_values >= 0.02, display_values, np.nan)
        display_visible_signal = bool(np.isfinite(display_values).any())
        cmap = ListedColormap(list(OUTLOOK_COLORS), name="tornado_concern_outlook")
        norm = BoundaryNorm(OUTLOOK_BINS, len(OUTLOOK_COLORS), clip=True)
        mesh = ax.contourf(
            display_lon,
            display_lat,
            display_values,
            levels=OUTLOOK_BINS,
            cmap=cmap,
            norm=norm,
            alpha=0.82,
            **draw_kwargs,
        )
        label_levels = _visible_label_levels(display_values, display_preset=resolved_preset)
        if label_levels:
            ax.contour(
                display_lon,
                display_lat,
                display_values,
                levels=label_levels,
                colors="#461010",
                linewidths=0.75,
                alpha=0.75,
                **draw_kwargs,
            )
    elif map_style == "contours":
        public_display_values = _public_display_concern_field(values)
        extent_values = public_display_values
        display_lon, display_lat, display_values = _upsample_grid_for_display(lon_values, lat_values, public_display_values)
        display_levels = tuple(value for value in PRODUCT_BINS if value >= CONTOUR_DISPLAY_MIN_THRESHOLD)
        display_values = np.where(display_values >= CONTOUR_DISPLAY_MIN_THRESHOLD, display_values, np.nan)
        display_visible_signal = bool(np.isfinite(display_values).any())
        contour_colors = PRODUCT_COLORS[1:]
        cmap = ListedColormap(list(contour_colors), name="tornado_concern_product_contours")
        norm = BoundaryNorm(display_levels, len(contour_colors), clip=True)
        mesh = ax.contourf(
            display_lon,
            display_lat,
            display_values,
            levels=display_levels,
            cmap=cmap,
            norm=norm,
            extend="neither",
            **draw_kwargs,
        )
        label_levels = _visible_label_levels(display_values)
        if label_levels:
            ax.contour(
                display_lon,
                display_lat,
                display_values,
                levels=label_levels,
                colors="#7f1d1d",
                linewidths=0.65,
                alpha=0.75,
                **draw_kwargs,
            )
    else:
        cmap = ListedColormap(list(PRODUCT_COLORS), name="tornado_concern_product")
        norm = BoundaryNorm(PRODUCT_BINS, len(PRODUCT_COLORS), clip=True)
        mesh = ax.pcolormesh(
            lon_values,
            lat_values,
            values,
            shading="auto",
            cmap=cmap,
            norm=norm,
            **draw_kwargs,
        )
    style_map_axes(ax, lon_values, lat_values, settings=settings)
    regional_extent = _display_extent(lon_values, lat_values, extent_values, map_domain=map_domain)
    if regional_extent is not None:
        _set_map_extent(ax, regional_extent)
    if not display_visible_signal:
        _annotate_no_visible_signal(ax, map_style=map_style)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=16)
    ax.text(
        0.0,
        1.01,
        f"24-hour valid period {valid_period_label} | Forecast initialized {date} {cycle}Z"
        + (" | Regional zoom" if regional_extent is not None else ""),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=SUBTITLE_COLOR,
    )
    colorbar_ticks = [value for value in (OUTLOOK_BINS if map_style == "outlook" else PRODUCT_BINS) if map_style != "contours" or value >= CONTOUR_DISPLAY_MIN_THRESHOLD]
    colorbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        fraction=0.08,
        pad=0.07,
        ticks=colorbar_ticks,
    )
    colorbar.ax.set_xticklabels([f"{int(value * 100)}%" for value in colorbar_ticks], fontsize=8)
    colorbar_label = (
        "Ingredient-only tornado environment outlook"
        if field_name in {"tornado_environment_outlook", "tornado_environment_outlook_v2", "tornado_environment_outlook_hybrid", DEFAULT_CONSENSUS_PRODUCT_FIELD}
        else "Baseline tornado concern"
    )
    colorbar.set_label(colorbar_label, fontsize=9)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.outline.set_linewidth(0.7)
    fig.text(
        0.04,
        0.02,
        "Outlook contours use display interpolation; source grid values are preserved in metadata."
        if map_style == "outlook"
        else "Public display uses neighborhood-supported smoothing; source grid values are preserved in metadata.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#4d5560",
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return {
        "contour_label_count": int(label_count),
        "map_extent": list(regional_extent) if regional_extent is not None else list(CONUS_EXTENT),
        "display_preset_effective": resolved_preset,
        "display_visible_signal": display_visible_signal,
        "render_projection_name": render_template.projection_name,
        "render_uses_cartopy": render_template.use_cartopy,
        "render_basemap_mode": "cartopy" if render_template.use_cartopy else "matplotlib_fallback",
        "render_basemap_warning": "" if render_template.use_cartopy else "cartopy_unavailable_or_disabled",
    }


def _manifest_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": metadata["date"],
        "init_date": metadata["init_date"],
        "cycle": metadata["cycle"],
        "valid_date": metadata["valid_date"],
        "product_valid_period": metadata["product_valid_period"],
        "variant": metadata["variant"],
        "title": metadata["title"],
        "image_path": metadata["main_image_path"],
        "summary_path": metadata["summary_path"],
        "metadata_path": metadata["metadata_path"],
        "generated_at": metadata["generation_timestamp"],
        "top_valid_date": metadata.get("top_valid_date", ""),
        "max_tornado_concern_prob": metadata.get("max_tornado_concern_prob"),
        "publication_status": metadata.get("publication_status", ""),
        "public_ready": metadata.get("public_ready", False),
        "audit_status": metadata.get("audit_status", ""),
        "failure_reasons": metadata.get("failure_reasons", ""),
    }


def _update_archive_manifest(path: Path, metadata: dict[str, Any]) -> None:
    entry = _manifest_entry(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if not frame.empty:
            frame = frame.loc[
                ~(
                    frame["date"].astype(str).eq(entry["date"])
                    & frame["cycle"].astype(str).eq(entry["cycle"])
                    & frame.get("valid_date", pd.Series("", index=frame.index)).astype(str).eq(entry["valid_date"])
                    & frame["variant"].astype(str).eq(entry["variant"])
                )
            ].copy()
        frame = pd.concat([frame, pd.DataFrame([entry])], ignore_index=True)
        frame = frame.sort_values(["date", "cycle", "variant"], kind="mergesort")
        frame.to_csv(path, index=False)
        return

    existing = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                existing = [item for item in payload if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            existing = []
    filtered = [
        item
        for item in existing
        if not (
            str(item.get("date", "")) == entry["date"]
            and str(item.get("cycle", "")) == entry["cycle"]
            and str(item.get("valid_date", "")) == entry["valid_date"]
            and str(item.get("variant", "")) == entry["variant"]
        )
    ]
    filtered.append(entry)
    filtered = sorted(filtered, key=lambda item: (str(item.get("date", "")), str(item.get("cycle", "")), str(item.get("variant", ""))))
    path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")


def build_product_bundle(
    *,
    date: str,
    cycle: str,
    valid_date: str | None = None,
    valid_start: str | None = None,
    valid_end: str | None = None,
    field_name: str = DEFAULT_PRODUCT_FIELD,
    outdir: Path,
    title: str | None = None,
    summary_text: str | None = None,
    archive_manifest: Path | None = None,
    map_style: str = "contours",
    map_domain: str = "conus",
    display_preset: str = "auto",
    artifact_source: str = "auto",
    require_production_basemap: bool = False,
    audit_public_readiness: bool = True,
    overwrite: bool = False,
    settings: Any | None = None,
    paths: Any | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    paths = paths or build_paths(settings)
    cycle = _normalize_cycle(cycle)
    prediction_path, field_name = _select_prediction_artifact(paths.outputs, date, cycle, field_name, artifact_source=artifact_source)
    if prediction_path is None:
        raise FileNotFoundError(f"missing tornado-concern forecast artifact for {date} {cycle}Z")
    metadata_path = _metadata_for_prediction_artifact(paths.outputs, date, cycle, prediction_path)
    forecast_metadata = _load_json(metadata_path)
    with xr.open_dataset(prediction_path) as dataset:
        dataset = _derive_product_field_dataset(dataset, field_name)
        effective_valid_date = valid_date or (pd.Timestamp(valid_start).date().isoformat() if valid_start else date)
        if valid_start or valid_end:
            if not (valid_start and valid_end):
                raise ValueError("--valid-start and --valid-end must be provided together")
            valid_dataset = _select_valid_time_window_dataset(dataset, valid_start, valid_end)
        else:
            valid_dataset = _select_valid_date_dataset(dataset, effective_valid_date)
        stats = _product_stats(valid_dataset, field_name)
        display_stats = _public_display_stats(valid_dataset, field_name, map_style=map_style, display_preset=display_preset)
        ingredient_diagnostics = _ingredient_diagnostics(valid_dataset, field_name)
        valid_period_label = _format_valid_period_label(effective_valid_date, valid_start, valid_end)
        effective_title = title or _default_title(valid_period_label)
        effective_summary = summary_text or _default_summary_text(stats, field_name=field_name, valid_label=valid_period_label)
        if map_style not in {"pixels", "contours", "outlook"}:
            raise ValueError(f"unsupported map style: {map_style}")
        if map_domain not in {"conus", "regional"}:
            raise ValueError(f"unsupported map domain: {map_domain}")
        if display_preset not in OUTLOOK_DISPLAY_PRESETS:
            raise ValueError(f"unsupported display preset: {display_preset}")

        stem = f"tornado_concern_init_{date}_{cycle}z_valid_{_valid_period_slug(effective_valid_date, valid_start, valid_end)}"
        image_path = outdir / f"{stem}.png"
        summary_path = outdir / f"{stem}.md"
        product_metadata_path = outdir / f"{stem}.json"
        _ensure_writable([image_path, summary_path, product_metadata_path], overwrite)
        outdir.mkdir(parents=True, exist_ok=True)

        render_metadata = _render_product_map(
            valid_dataset,
            field_name=field_name,
            date=date,
            cycle=cycle,
            valid_date=effective_valid_date,
            valid_period_label=valid_period_label,
            title=effective_title,
            summary_text=effective_summary,
            image_path=image_path,
            map_style=map_style,
            map_domain=map_domain,
            display_preset=display_preset,
            settings=settings,
        )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if audit_public_readiness and field_name in {
        "tornado_concern_environment_envelope",
        "tornado_environment_outlook",
        "tornado_environment_outlook_v2",
        "tornado_environment_outlook_hybrid",
        DEFAULT_CONSENSUS_PRODUCT_FIELD,
    }:
        artifact_audit_result = {
            "date": date,
            "valid_date": effective_valid_date,
            "field_name": field_name,
            "status": "ok",
            "public_ready": True,
            "failure_reasons": "",
            "max_tornado_concern_prob": stats.get("max_tornado_concern_prob"),
            "mean_tornado_concern_prob": stats.get("mean_tornado_concern_prob"),
        }
    else:
        artifact_audit_result = (
            audit_tornado_concern_product(date, prediction_path, field_name=field_name, valid_date=effective_valid_date)
            if audit_public_readiness
            else {"status": "skipped", "public_ready": False, "failure_reasons": "audit_skipped"}
        )
    display_audit_result = _display_readiness_audit(display_stats, map_style=map_style)
    consensus_audit_result = _consensus_readiness_audit(valid_dataset, field_name)
    source_audit_result = _source_readiness_audit(forecast_metadata, field_name)
    render_audit_result = _render_readiness_audit(render_metadata, require_production_basemap=require_production_basemap)
    combined_failure_reasons = ";".join(
        reason
        for reason in [
            str(artifact_audit_result.get("failure_reasons", "") or ""),
            str(display_audit_result.get("failure_reasons", "") or ""),
            str(consensus_audit_result.get("failure_reasons", "") or ""),
            str(source_audit_result.get("failure_reasons", "") or ""),
            str(render_audit_result.get("failure_reasons", "") or ""),
        ]
        if reason
    )
    combined_public_ready = (
        bool(artifact_audit_result.get("public_ready", False))
        and bool(display_audit_result.get("public_ready", False))
        and bool(consensus_audit_result.get("public_ready", True))
        and bool(source_audit_result.get("public_ready", True))
        and bool(render_audit_result.get("public_ready", True))
    )
    publication_status = "public_candidate" if combined_public_ready else "internal_review_only"
    non_render_public_ready = (
        bool(artifact_audit_result.get("public_ready", False))
        and bool(display_audit_result.get("public_ready", False))
        and bool(consensus_audit_result.get("public_ready", True))
        and bool(source_audit_result.get("public_ready", True))
    )
    if not combined_public_ready and non_render_public_ready and render_audit_result.get("failure_reasons") == "render_basemap_fallback":
        publication_status = "needs_render_review"
    product_metadata: dict[str, Any] = {
        "date": date,
        "init_date": date,
        "cycle": cycle,
        "valid_date": effective_valid_date,
        "valid_start": valid_start or "",
        "valid_end": valid_end or "",
        "valid_period_label": valid_period_label,
        "product_valid_period": "custom_valid_time_window" if valid_start and valid_end else "24h_utc_date",
        "variant": _product_variant(field_name),
        "field_name": field_name,
        "title": effective_title,
        "summary_text": effective_summary,
        "map_style": map_style,
        "map_domain": map_domain,
        "color_ramp": "spc_probability_bins" if map_style == "outlook" else "red_gradient",
        "display_gradient": "outlook_probability_bins" if map_style == "outlook" else "threshold_banded",
        "display_transform": "outlook_envelope_field" if map_style == "outlook" else "neighborhood_supported_public_field" if map_style == "contours" else "raw_grid_values",
        "display_preset_requested": display_preset,
        "display_preset_effective": render_metadata.get("display_preset_effective", display_stats.get("display_preset_effective", "standard")),
        "artifact_source_requested": artifact_source,
        "production_basemap_required": require_production_basemap,
        "display_neighborhood_radius_cells": PUBLIC_DISPLAY_NEIGHBORHOOD_RADIUS if map_style == "contours" else 0,
        "display_envelope_gain": PUBLIC_DISPLAY_ENVELOPE_GAIN if map_style == "contours" else 1.0,
        "display_peak_support_cap_gain": PUBLIC_DISPLAY_PEAK_SUPPORT_CAP_GAIN if map_style == "contours" else 1.0,
        "display_signal_floor": "2pct_outlook_area" if map_style == "outlook" else "2pct_concern_envelope" if map_style == "contours" else "raw_pixel_bins",
        "display_interpolation": "bilinear" if map_style in {"contours", "outlook"} else "native_grid",
        "display_interpolation_factor": CONTOUR_DISPLAY_INTERPOLATION_FACTOR if map_style in {"contours", "outlook"} else 1,
        "display_min_threshold": 0.02 if map_style == "outlook" else CONTOUR_DISPLAY_MIN_THRESHOLD if map_style == "contours" else 0.0,
        "display_min_object_cells": OUTLOOK_DISPLAY_MIN_OBJECT_CELLS if map_style == "outlook" else 0,
        "generation_timestamp": generated_at,
        "main_image_path": str(image_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "metadata_path": str(product_metadata_path.resolve()),
        "source_forecast_artifact_path": str(prediction_path.resolve()),
        "source_forecast_metadata_path": str(metadata_path.resolve()) if metadata_path is not None else "",
        "public_ready": combined_public_ready,
        "audit_status": "ok" if combined_public_ready else "flagged",
        "failure_reasons": combined_failure_reasons,
        "publication_status": publication_status,
        "product_audit": artifact_audit_result,
        "display_audit": display_audit_result,
        "consensus_audit": consensus_audit_result,
        "source_audit": source_audit_result,
        "render_audit": render_audit_result,
        **render_metadata,
        "ingredient_diagnostics": ingredient_diagnostics,
        "reference_design_notes": [
            "CSU-MLP-style ingredient overlap supports the public tornado-environment outlook.",
            "TORP/TorNet-style object filtering suppresses isolated display specks without altering source grid values.",
        ]
        if field_name in {"tornado_environment_outlook", "tornado_environment_outlook_v2", "tornado_environment_outlook_hybrid", DEFAULT_CONSENSUS_PRODUCT_FIELD}
        else [],
        **stats,
        **display_stats,
    }
    if forecast_metadata:
        product_metadata["forecast_run_summary"] = forecast_metadata.get("run_summary", {})
        product_metadata["forecast_ingest_summary"] = forecast_metadata.get("ingest_summary", {})
        product_metadata["forecast_consensus_summary"] = {
            key: forecast_metadata.get(key)
            for key in [
                "included_sources",
                "excluded_sources",
                "source_models",
                "reference_source",
                "primary_model_code_map",
                "time_weights",
            ]
            if key in forecast_metadata
        }

    summary_rows = [
        ("title", effective_title),
        ("date", date),
        ("cycle", f"{cycle}Z"),
        ("valid_date", effective_valid_date),
        ("valid_start", product_metadata["valid_start"]),
        ("valid_end", product_metadata["valid_end"]),
        ("product_valid_period", product_metadata["product_valid_period"]),
        ("variant", product_metadata["variant"]),
        ("field_name", field_name),
        ("artifact_source_requested", product_metadata["artifact_source_requested"]),
        ("production_basemap_required", str(product_metadata["production_basemap_required"])),
        ("map_style", map_style),
        ("map_domain", map_domain),
        ("display_preset", product_metadata["display_preset_effective"]),
        ("color_ramp", product_metadata["color_ramp"]),
        ("display_gradient", product_metadata["display_gradient"]),
        ("display_transform", product_metadata["display_transform"]),
        ("display_signal_floor", product_metadata["display_signal_floor"]),
        ("display_interpolation", product_metadata["display_interpolation"]),
        ("display_min_threshold", f"{float(product_metadata['display_min_threshold']):.2f}"),
        ("display_visible_signal", str(product_metadata.get("display_visible_signal", True))),
        ("render_basemap_mode", product_metadata.get("render_basemap_mode", "")),
        ("render_basemap_warning", product_metadata.get("render_basemap_warning", "") or "none"),
        ("publication_status", product_metadata["publication_status"]),
        ("public_ready", str(product_metadata["public_ready"])),
        ("audit_status", product_metadata["audit_status"]),
        ("failure_reasons", product_metadata["failure_reasons"] or "none"),
        ("consensus_sources", ",".join(product_metadata.get("forecast_consensus_summary", {}).get("included_sources", []) or [])),
        ("consensus_max_signal_agreement", product_metadata.get("consensus_audit", {}).get("max_signal_agreement_count", "")),
        ("valid_window", product_metadata["valid_period_label"]),
        ("selected_valid_times", ",".join(str(value) for value in stats.get("valid_times", []))),
        ("top_valid_date", stats.get("top_valid_date", "")),
        ("peak_location", f"{ingredient_diagnostics.get('peak_lat', '')}, {ingredient_diagnostics.get('peak_lon', '')}" if ingredient_diagnostics else ""),
        ("limiting_ingredient_at_peak", ingredient_diagnostics.get("limiting_ingredient_at_peak", "") if ingredient_diagnostics else ""),
        ("map_extent", ",".join(str(value) for value in product_metadata.get("map_extent", []))),
        ("contour_label_count", product_metadata.get("contour_label_count", 0)),
        ("max_raw_tornado_concern_prob", f"{float(stats.get('max_tornado_concern_prob', float('nan'))):.4f}"),
        ("max_public_display_tornado_concern_prob", f"{float(display_stats.get('max_public_display_tornado_concern_prob', float('nan'))):.4f}"),
        ("image_path", image_path.name),
        ("metadata_path", product_metadata_path.name),
        ("source_forecast_artifact", str(prediction_path)),
    ]
    summary_markdown = "\n".join(
        [
            f"# {effective_title}",
            "",
            effective_summary,
            "",
            "## Product Facts",
            "",
            _markdown_table(summary_rows),
            "",
            "## Artifacts",
            "",
            f"- Map image: `{image_path.name}`",
            f"- Metadata: `{product_metadata_path.name}`",
            f"- Source forecast artifact: `{prediction_path}`",
            f"- Source forecast metadata: `{metadata_path}`" if metadata_path is not None else "- Source forecast metadata: _not available_",
        ]
    )
    summary_path.write_text(summary_markdown + "\n", encoding="utf-8")
    product_metadata_path.write_text(json.dumps(product_metadata, indent=2), encoding="utf-8")

    if archive_manifest is not None:
        _update_archive_manifest(archive_manifest, product_metadata)

    return product_metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a public-facing tornado-concern product bundle from saved baseline artifacts")
    parser.add_argument("--date", help="Forecast init date YYYY-MM-DD")
    parser.add_argument("--dates-file", help="Optional newline-delimited list of forecast init dates to build in one process")
    parser.add_argument("--cycle", required=True, help="Forecast init cycle, e.g. 00")
    parser.add_argument("--valid-date", help="24-hour valid date YYYY-MM-DD. Defaults to each init date.")
    parser.add_argument("--valid-start", help="Custom valid-window start time, e.g. 2026-05-05T12:00Z")
    parser.add_argument("--valid-end", help="Custom valid-window end time, e.g. 2026-05-06T12:00Z. End is exclusive.")
    parser.add_argument(
        "--field",
        choices=[
            "tornado_concern_prob",
            "tornado_concern_environment_envelope",
            "tornado_environment_outlook",
            "tornado_environment_outlook_v2",
            "tornado_environment_outlook_hybrid",
            DEFAULT_CONSENSUS_PRODUCT_FIELD,
        ],
        default=DEFAULT_PRODUCT_FIELD,
        help="Forecast field to render. Baseline keeps tornado_concern_prob; outlook fields are ingredient-only prototypes.",
    )
    parser.add_argument(
        "--valid-date-mode",
        choices=["init", "best"],
        default="init",
        help="Select valid date automatically when --valid-date is not provided. 'best' uses the strongest 24-hour valid day.",
    )
    parser.add_argument("--outdir", required=True, help="Output directory for the product bundle")
    parser.add_argument("--title", help="Optional custom product title")
    parser.add_argument("--summary-text", help="Optional custom summary override")
    parser.add_argument("--archive-manifest", help="Optional JSON or CSV manifest to append/update")
    parser.add_argument(
        "--map-style",
        choices=["pixels", "contours", "outlook"],
        default=DEFAULT_PRODUCT_MAP_STYLE,
        help="Display style for the map. 'outlook' uses filled probability tiers for the envelope challenger.",
    )
    parser.add_argument(
        "--map-domain",
        choices=["conus", "regional"],
        default=DEFAULT_PRODUCT_MAP_DOMAIN,
        help="Map extent. 'regional' auto-zooms around the displayed outlook footprint.",
    )
    parser.add_argument(
        "--display-preset",
        choices=sorted(OUTLOOK_DISPLAY_PRESETS),
        default="auto",
        help="Outlook display filtering preset. 'auto' selects weak, standard, or broad from the product footprint.",
    )
    parser.add_argument(
        "--artifact-source",
        choices=["auto", "prediction", "consensus"],
        default="auto",
        help="Forecast artifact source. 'auto' preserves consensus preference for the default hybrid field; 'prediction' forces forecast_products.",
    )
    parser.add_argument(
        "--require-production-basemap",
        action="store_true",
        help="Mark products rendered without Cartopy as needs_render_review/internal-only.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing product files in the target outdir")
    parser.add_argument("--skip-audit", action="store_true", help="Skip public-readiness audit metadata")
    args = parser.parse_args(argv)

    if bool(args.date) == bool(args.dates_file):
        parser.error("provide exactly one of --date or --dates-file")
    if bool(args.valid_start) != bool(args.valid_end):
        parser.error("--valid-start and --valid-end must be provided together")
    if (args.valid_start or args.valid_end) and args.valid_date:
        parser.error("--valid-date cannot be combined with --valid-start/--valid-end")
    dates = _read_dates_file(Path(args.dates_file)) if args.dates_file else [str(args.date)]
    if not dates:
        parser.error("--dates-file did not contain any dates")

    settings = load_settings()
    paths = build_paths(settings)
    for date in dates:
        selected_valid_date = args.valid_date
        if selected_valid_date is None and args.valid_date_mode == "best":
            prediction_path = _prediction_artifact_for_cycle(paths.outputs, date, _normalize_cycle(args.cycle))
            if prediction_path is None:
                raise FileNotFoundError(f"missing tornado-concern forecast artifact for {date} {_normalize_cycle(args.cycle)}Z")
            with xr.open_dataset(prediction_path) as dataset:
                dataset = _derive_product_field_dataset(dataset, args.field)
                selected_valid_date = _best_valid_date(dataset, args.field)
        metadata = build_product_bundle(
            date=date,
            cycle=args.cycle,
            valid_date=None if args.valid_start else selected_valid_date or date,
            valid_start=args.valid_start,
            valid_end=args.valid_end,
            field_name=args.field,
            outdir=Path(args.outdir),
            title=args.title if len(dates) == 1 else None,
            summary_text=args.summary_text,
            archive_manifest=Path(args.archive_manifest) if args.archive_manifest else None,
            map_style=args.map_style,
            map_domain=args.map_domain,
            display_preset=args.display_preset,
            artifact_source=args.artifact_source,
            require_production_basemap=bool(args.require_production_basemap),
            audit_public_readiness=not bool(args.skip_audit),
            overwrite=bool(args.overwrite),
            settings=settings,
            paths=paths,
        )
        print(f"date={metadata['date']}")
        print(f"main_image_path={metadata['main_image_path']}")
        print(f"summary_path={metadata['summary_path']}")
        print(f"metadata_path={metadata['metadata_path']}")
        print(f"valid_date={metadata.get('valid_date', '')}")
        print(f"top_valid_date={metadata.get('top_valid_date', '')}")
        print(f"max_tornado_concern_prob={metadata.get('max_tornado_concern_prob', float('nan'))}")
        print(f"publication_status={metadata.get('publication_status', '')}")
        print(f"public_ready={metadata.get('public_ready', False)}")


if __name__ == "__main__":
    main()
