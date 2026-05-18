"""Diagnose source signal and underdispersion for SPC-window tornado outlooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from severewx.models.forecast_consensus import (
    AUTO_CONSENSUS_SOURCES,
    CONSENSUS_FIELD,
    CONSENSUS_INPUT_FIELD,
    consensus_product_path,
    consensus_weights_for_lead,
    source_product_path,
)
from severewx.utils.dates import cycle_datetime


def _parse_windows(values: list[str]) -> list[tuple[str, str, str]]:
    windows: list[tuple[str, str, str]] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"window must be name,start,end: {value}")
        windows.append((parts[0], parts[1], parts[2]))
    return windows


def _time_index(dataset: xr.Dataset, start: str, end: str) -> list[int]:
    times = pd.to_datetime(dataset["time"].values)
    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None)
    return [index for index, value in enumerate(times) if start_ts <= pd.Timestamp(value).tz_localize(None) < end_ts]


def _lead_hour_from_init(init_time: pd.Timestamp, valid_time: pd.Timestamp) -> int:
    return int(round((valid_time - init_time).total_seconds() / 3600.0))


def _safe_stats(values: np.ndarray, *, lat: np.ndarray, lon: np.ndarray) -> dict[str, Any]:
    clean = np.asarray(values, dtype=float)
    finite = np.isfinite(clean)
    if clean.size == 0 or not finite.any():
        return {
            "max_probability": None,
            "cells_ge_02": 0,
            "cells_ge_05": 0,
            "peak_lat": None,
            "peak_lon": None,
            "footprint_lat_min": None,
            "footprint_lat_max": None,
            "footprint_lon_min": None,
            "footprint_lon_max": None,
            "footprint_centroid_lat": None,
            "footprint_centroid_lon": None,
            "object_count_ge_02": 0,
            "largest_object_cells_ge_02": 0,
            "detached_object_count_ge_02": 0,
            "detached_cells_ge_02": 0,
            "detached_cell_fraction_ge_02": None,
        }
    max_value = float(np.nanmax(clean))
    peak = np.unravel_index(int(np.nanargmax(clean)), clean.shape)
    footprint = clean >= 0.02
    rows, cols = np.where(footprint)
    weights = clean[footprint]
    return {
        "max_probability": max_value,
        "cells_ge_02": int(np.nansum(clean >= 0.02)),
        "cells_ge_05": int(np.nansum(clean >= 0.05)),
        "peak_lat": float(lat[peak[0]]) if lat.size else None,
        "peak_lon": float(lon[peak[1]]) if lon.size else None,
        "footprint_lat_min": float(lat[rows].min()) if rows.size else None,
        "footprint_lat_max": float(lat[rows].max()) if rows.size else None,
        "footprint_lon_min": float(lon[cols].min()) if cols.size else None,
        "footprint_lon_max": float(lon[cols].max()) if cols.size else None,
        "footprint_centroid_lat": float(np.average(lat[rows], weights=weights)) if rows.size and float(np.sum(weights)) > 0.0 else None,
        "footprint_centroid_lon": float(np.average(lon[cols], weights=weights)) if cols.size and float(np.sum(weights)) > 0.0 else None,
        **_object_stats(footprint),
    }


def _object_stats(mask: np.ndarray) -> dict[str, Any]:
    active = np.asarray(mask, dtype=bool)
    if active.ndim != 2 or not active.any():
        return {
            "object_count_ge_02": 0,
            "largest_object_cells_ge_02": 0,
            "detached_object_count_ge_02": 0,
            "detached_cells_ge_02": 0,
            "detached_cell_fraction_ge_02": None,
        }
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
    largest = max(sizes, default=0)
    total = int(np.count_nonzero(active))
    detached_cells = max(total - largest, 0)
    return {
        "object_count_ge_02": len(sizes),
        "largest_object_cells_ge_02": largest,
        "detached_object_count_ge_02": max(len(sizes) - 1, 0),
        "detached_cells_ge_02": detached_cells,
        "detached_cell_fraction_ge_02": detached_cells / total if total else None,
    }


def _window_max(dataset: xr.Dataset, field: str, indices: list[int]) -> np.ndarray | None:
    if field not in dataset or not indices:
        return None
    return np.asarray(dataset[field].isel(time=indices).max(dim="time", skipna=True).values, dtype=float)


def _nanmax_stack(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    finite = np.isfinite(stack)
    return np.max(np.where(finite, stack, -np.inf), axis=0).astype(float)


def _neighbor_max(values: np.ndarray, radius: int = 1) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    radius = max(int(radius), 0)
    if radius == 0:
        return clean
    padded = np.pad(clean, radius, mode="edge")
    output = np.full_like(clean, -np.inf, dtype=float)
    for row_offset in range(2 * radius + 1):
        for col_offset in range(2 * radius + 1):
            output = np.maximum(output, padded[row_offset : row_offset + clean.shape[0], col_offset : col_offset + clean.shape[1]])
    return output


def _object_corridor_envelope(values: np.ndarray, agreement: np.ndarray, *, min_cells: int = 12) -> np.ndarray:
    """Keep p80 corridor components that have at least some two-source support."""
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    support = np.asarray(agreement, dtype=float)
    active = clean >= 0.02
    output = np.zeros_like(clean, dtype=float)
    visited = np.zeros(active.shape, dtype=bool)
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
                continue
            component_rows = np.asarray([row for row, _ in component], dtype=int)
            component_cols = np.asarray([col for _, col in component], dtype=int)
            if not np.any(_neighbor_max(support, radius=2)[component_rows, component_cols] >= 2):
                continue
            output[component_rows, component_cols] = clean[component_rows, component_cols]
    return output


def _detached_noise_suppressed_envelope(
    values: np.ndarray,
    agreement: np.ndarray,
    *,
    min_cells: int = 12,
    min_detached_cells: int = 80,
    detached_keep_probability: float = 0.15,
    detached_keep_agreement: int = 3,
) -> np.ndarray:
    """Keep the main supported corridor and only retain unusually strong detached objects.

    This is a diagnostic science variant, not a display smoothing step. It leaves
    probabilities unchanged inside retained objects and zeros detached weak objects
    that are common in broad low-end ingredient fields.
    """
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    support = np.asarray(agreement, dtype=float)
    active = clean >= 0.02
    output = np.zeros_like(clean, dtype=float)
    visited = np.zeros(active.shape, dtype=bool)
    rows, cols = active.shape
    components: list[dict[str, Any]] = []
    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not active[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            component: list[tuple[int, int]] = []
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
                continue
            component_rows = np.asarray([row for row, _ in component], dtype=int)
            component_cols = np.asarray([col for _, col in component], dtype=int)
            if not np.any(_neighbor_max(support, radius=2)[component_rows, component_cols] >= 2):
                continue
            component_values = clean[component_rows, component_cols]
            component_support = support[component_rows, component_cols]
            components.append(
                {
                    "rows": component_rows,
                    "cols": component_cols,
                    "size": int(len(component)),
                    "max_probability": float(np.nanmax(component_values)) if component_values.size else 0.0,
                    "max_agreement": float(np.nanmax(component_support)) if component_support.size else 0.0,
                }
            )
    if not components:
        return output
    main_index = max(range(len(components)), key=lambda index: (components[index]["size"], components[index]["max_probability"]))
    for index, component in enumerate(components):
        keep = index == main_index
        if not keep:
            keep = (
                component["size"] >= min_detached_cells
                and component["max_probability"] >= detached_keep_probability
                and component["max_agreement"] >= detached_keep_agreement
            )
        if keep:
            output[component["rows"], component["cols"]] = clean[component["rows"], component["cols"]]
    return output


def _primary_core_noise_suppressed_envelope(
    values: np.ndarray,
    agreement: np.ndarray,
    *,
    core_threshold: float = 0.10,
    keep_radius_cells: int = 12,
) -> np.ndarray:
    """Retain the primary high-probability corridor and remove bridge-connected weak noise."""
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    support = np.asarray(agreement, dtype=float)
    core = (clean >= core_threshold) & (_neighbor_max(support, radius=2) >= 2)
    if not core.any():
        return _detached_noise_suppressed_envelope(clean, support)
    visited = np.zeros(core.shape, dtype=bool)
    rows, cols = core.shape
    components: list[dict[str, Any]] = []
    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not core[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            component: list[tuple[int, int]] = []
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
                            and core[next_row, next_col]
                            and not visited[next_row, next_col]
                        ):
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))
            component_rows = np.asarray([row for row, _ in component], dtype=int)
            component_cols = np.asarray([col for _, col in component], dtype=int)
            component_values = clean[component_rows, component_cols]
            components.append(
                {
                    "rows": component_rows,
                    "cols": component_cols,
                    "size": int(len(component)),
                    "max_probability": float(np.nanmax(component_values)) if component_values.size else 0.0,
                }
            )
    primary = max(components, key=lambda component: (component["size"], component["max_probability"]))
    primary_mask = np.zeros_like(core, dtype=bool)
    primary_mask[primary["rows"], primary["cols"]] = True
    keep_mask = (_neighbor_max(primary_mask.astype(float), radius=keep_radius_cells) > 0) & (clean >= 0.02)
    output = np.zeros_like(clean, dtype=float)
    output[keep_mask] = clean[keep_mask]
    return output


def _supported_components(values: np.ndarray, agreement: np.ndarray, *, min_cells: int = 12) -> list[dict[str, Any]]:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    support = np.asarray(agreement, dtype=float)
    active = clean >= 0.02
    visited = np.zeros(active.shape, dtype=bool)
    rows, cols = active.shape
    components: list[dict[str, Any]] = []
    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not active[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            component: list[tuple[int, int]] = []
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
                            and active[next_row, next_col]
                            and not visited[next_row, next_col]
                        ):
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))
            if len(component) < min_cells:
                continue
            component_rows = np.asarray([row for row, _ in component], dtype=int)
            component_cols = np.asarray([col for _, col in component], dtype=int)
            if not np.any(_neighbor_max(support, radius=2)[component_rows, component_cols] >= 2):
                continue
            component_values = clean[component_rows, component_cols]
            weights = np.clip(component_values, 0.02, None)
            components.append(
                {
                    "rows": component_rows,
                    "cols": component_cols,
                    "size": int(len(component)),
                    "max_probability": float(np.nanmax(component_values)) if component_values.size else 0.0,
                    "centroid_row": float(np.average(component_rows, weights=weights)) if weights.size else float(np.mean(component_rows)),
                    "centroid_col": float(np.average(component_cols, weights=weights)) if weights.size else float(np.mean(component_cols)),
                }
            )
    return components


def _temporal_dominant_corridor_envelope(
    values_by_time: list[np.ndarray],
    agreement_by_time: list[np.ndarray],
    *,
    max_centroid_distance_cells: float = 20.0,
) -> np.ndarray:
    """Aggregate only the dominant temporal corridor through a 24-hour window."""
    if not values_by_time:
        return np.asarray([], dtype=float)
    clean_values = [np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) for values in values_by_time]
    component_rows: list[dict[str, Any]] = []
    for time_index, (values, agreement) in enumerate(zip(clean_values, agreement_by_time, strict=False)):
        for component in _supported_components(values, agreement):
            component_rows.append({**component, "time_index": time_index})
    if not component_rows:
        return _nanmax_stack(clean_values)
    anchor = max(component_rows, key=lambda component: (component["max_probability"], component["size"]))
    output = np.zeros_like(clean_values[0], dtype=float)
    for component in component_rows:
        distance = float(np.hypot(component["centroid_row"] - anchor["centroid_row"], component["centroid_col"] - anchor["centroid_col"]))
        keep = component is anchor or distance <= max_centroid_distance_cells
        if keep:
            values = clean_values[int(component["time_index"])]
            output[component["rows"], component["cols"]] = np.maximum(output[component["rows"], component["cols"]], values[component["rows"], component["cols"]])
    return output


def _temporal_primary_core_corridor_envelope(values_by_time: list[np.ndarray], agreement_by_time: list[np.ndarray]) -> np.ndarray:
    """Temporal corridor selection after per-time primary-core noise suppression."""
    primary_values = [
        _primary_core_noise_suppressed_envelope(values, agreement)
        for values, agreement in zip(values_by_time, agreement_by_time, strict=False)
    ]
    return _temporal_dominant_corridor_envelope(primary_values, agreement_by_time)


def _temporal_primary_core_expanded_corridor_envelope(
    values_by_time: list[np.ndarray],
    agreement_by_time: list[np.ndarray],
    *,
    expansion_radius_cells: int = 12,
) -> np.ndarray:
    """Recover nearby supported corridor area around the dominant temporal core.

    This keeps the temporal-primary core as the anchor, then admits nearby
    two-source-supported 2%+ values from the source p80 envelope. It is intended
    to recover adjacent corridor breadth without accepting distant temporal
    episodes that happen to occur in the same 24-hour SPC window.
    """
    if not values_by_time:
        return np.asarray([], dtype=float)
    primary = _temporal_primary_core_corridor_envelope(values_by_time, agreement_by_time)
    anchor = np.nan_to_num(np.asarray(primary, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) >= 0.02
    if not anchor.any():
        return primary
    corridor_mask = _neighbor_max(anchor.astype(float), radius=expansion_radius_cells) > 0
    output = np.zeros_like(primary, dtype=float)
    for values, agreement in zip(values_by_time, agreement_by_time, strict=False):
        clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        support = _neighbor_max(np.asarray(agreement, dtype=float), radius=1) >= 2
        keep = corridor_mask & support & (clean >= 0.02)
        output[keep] = np.maximum(output[keep], clean[keep])
    return output


def _remove_small_detached_objects(
    values: np.ndarray,
    *,
    min_detached_cells: int = 80,
    min_detached_max_probability: float = 0.15,
) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    active = clean >= 0.02
    visited = np.zeros(active.shape, dtype=bool)
    rows, cols = active.shape
    components: list[dict[str, Any]] = []
    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not active[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            component: list[tuple[int, int]] = []
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
                            and active[next_row, next_col]
                            and not visited[next_row, next_col]
                        ):
                            visited[next_row, next_col] = True
                            stack.append((next_row, next_col))
            component_rows = np.asarray([row for row, _ in component], dtype=int)
            component_cols = np.asarray([col for _, col in component], dtype=int)
            component_values = clean[component_rows, component_cols]
            components.append(
                {
                    "rows": component_rows,
                    "cols": component_cols,
                    "size": int(len(component)),
                    "max_probability": float(np.nanmax(component_values)) if component_values.size else 0.0,
                }
            )
    if not components:
        return clean
    main = max(components, key=lambda component: (component["size"], component["max_probability"]))
    output = np.zeros_like(clean, dtype=float)
    for component in components:
        keep = component is main or (
            component["size"] >= min_detached_cells and component["max_probability"] >= min_detached_max_probability
        )
        if keep:
            output[component["rows"], component["cols"]] = clean[component["rows"], component["cols"]]
    return output


def _temporal_primary_core_clean_corridor_envelope(
    values_by_time: list[np.ndarray],
    agreement_by_time: list[np.ndarray],
) -> np.ndarray:
    """Expanded temporal-core corridor with small detached artifacts removed."""
    expanded = _temporal_primary_core_expanded_corridor_envelope(values_by_time, agreement_by_time)
    return _remove_small_detached_objects(expanded)


def _cap_high_end_by_temporal_support(
    values: np.ndarray,
    values_by_time: list[np.ndarray],
    *,
    cap_probability: float = 0.15,
    high_end_threshold: float = 0.30,
    min_high_end_times: int = 2,
    neighborhood_radius_cells: int = 2,
) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if not values_by_time:
        return np.minimum(clean, cap_probability)
    high_end_count = np.zeros_like(clean, dtype=float)
    for values_at_time in values_by_time:
        high_end = np.nan_to_num(np.asarray(values_at_time, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) >= high_end_threshold
        high_end_count += (_neighbor_max(high_end.astype(float), radius=neighborhood_radius_cells) > 0).astype(float)
    high_end_allowed = high_end_count >= min_high_end_times
    return np.where((clean > cap_probability) & ~high_end_allowed, cap_probability, clean)


def _cap_extreme_by_source_support(
    values: np.ndarray,
    high_agreement_by_time: list[np.ndarray],
    *,
    cap_probability: float = 0.30,
    extreme_threshold: float = 0.30,
    min_high_source_count: int = 3,
    neighborhood_radius_cells: int = 1,
) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if not high_agreement_by_time:
        return np.minimum(clean, cap_probability)
    supported = np.zeros_like(clean, dtype=bool)
    for high_agreement in high_agreement_by_time:
        high_count = np.nan_to_num(np.asarray(high_agreement, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        supported |= _neighbor_max(high_count, radius=neighborhood_radius_cells) >= min_high_source_count
    return np.where((clean > extreme_threshold) & ~supported, cap_probability, clean)


def _temporal_primary_core_guarded_corridor_envelope(
    values_by_time: list[np.ndarray],
    agreement_by_time: list[np.ndarray],
) -> np.ndarray:
    """Clean temporal-core corridor with high-end probabilities requiring temporal support."""
    clean = _temporal_primary_core_clean_corridor_envelope(values_by_time, agreement_by_time)
    return _cap_high_end_by_temporal_support(clean, values_by_time)


def _temporal_primary_core_source_guarded_corridor_envelope(
    values_by_time: list[np.ndarray],
    agreement_by_time: list[np.ndarray],
    high_agreement_by_time: list[np.ndarray],
) -> np.ndarray:
    """Clean corridor that caps extreme probabilities unless three sources support high-end risk."""
    clean = _temporal_primary_core_clean_corridor_envelope(values_by_time, agreement_by_time)
    return _cap_extreme_by_source_support(clean, high_agreement_by_time)


def _aligned_source_arrays(
    *,
    outputs_dir: Path,
    date: str,
    cycle: str,
    sources: list[str],
    reference: xr.Dataset,
    valid_time: pd.Timestamp,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for source in sources:
        path = source_product_path(outputs_dir, date, cycle, source)
        if not path.exists():
            continue
        with xr.open_dataset(path) as dataset:
            if CONSENSUS_INPUT_FIELD not in dataset or "time" not in dataset.coords:
                continue
            times = [pd.Timestamp(value).tz_localize(None) for value in dataset["time"].values]
            if valid_time not in times:
                continue
            selected = dataset[CONSENSUS_INPUT_FIELD].isel(time=times.index(valid_time))
            aligned = selected.interp(lat=reference["lat"], lon=reference["lon"], kwargs={"fill_value": np.nan})
            values = np.asarray(aligned.values, dtype=float)
            if np.isfinite(values).any():
                arrays[source] = values
    return arrays


def _variant_arrays_for_window(
    *,
    outputs_dir: Path,
    date: str,
    cycle: str,
    sources: list[str],
    reference: xr.Dataset,
    time_indices: list[int],
) -> dict[str, np.ndarray]:
    init_time = pd.Timestamp(cycle_datetime(date, cycle)).tz_localize(None)
    per_time: dict[str, list[np.ndarray]] = {
        "variant_weighted_mean": [],
        "variant_source_max": [],
        "variant_source_p80": [],
        "variant_agreement_gated_envelope": [],
        "variant_agreement_neighborhood_envelope": [],
        "variant_agreement_corridor_envelope": [],
        "variant_object_corridor_envelope": [],
        "variant_detached_noise_suppressed": [],
        "variant_primary_core_noise_suppressed": [],
    }
    source_p80_by_time: list[np.ndarray] = []
    agreement_by_time: list[np.ndarray] = []
    high_agreement_by_time: list[np.ndarray] = []
    valid_times = [pd.Timestamp(value).tz_localize(None) for value in reference["time"].values]
    shape = (reference.sizes["lat"], reference.sizes["lon"])
    for index in time_indices:
        valid_time = valid_times[index]
        arrays = _aligned_source_arrays(outputs_dir=outputs_dir, date=date, cycle=cycle, sources=sources, reference=reference, valid_time=valid_time)
        if not arrays:
            for values in per_time.values():
                values.append(np.full(shape, np.nan, dtype=float))
            continue
        lead_hour = _lead_hour_from_init(init_time, valid_time)
        configured_weights = consensus_weights_for_lead(lead_hour)
        weights_by_source = {source: float(configured_weights.get(source, 0.0)) for source in arrays}
        if not any(weight > 0.0 for weight in weights_by_source.values()):
            weights_by_source = {source: 1.0 for source in arrays}
        total = sum(weights_by_source.values()) or 1.0
        source_order = list(arrays)
        stack = np.stack([arrays[source] for source in source_order], axis=0)
        weights = np.asarray([weights_by_source[source] / total for source in source_order], dtype=float).reshape((-1, 1, 1))
        valid_mask = np.isfinite(stack)
        weighted_sum = np.nansum(np.where(valid_mask, stack * weights, 0.0), axis=0)
        available_weight = np.sum(np.where(valid_mask, weights, 0.0), axis=0)
        weighted_mean = np.divide(weighted_sum, available_weight, out=np.full_like(weighted_sum, np.nan), where=available_weight > 0)
        source_max = np.nanmax(stack, axis=0)
        source_p80 = np.nanpercentile(stack, 80, axis=0)
        agreement = np.sum((stack >= 0.02) & valid_mask, axis=0)
        high_agreement = np.sum((stack >= 0.15) & valid_mask, axis=0)
        # Diagnostic only: preserve the two-source agreement rule while testing whether an upper envelope would restore supported risk.
        envelope = np.where(agreement >= 2, np.maximum(weighted_mean, source_p80), 0.0)
        neighborhood_envelope = np.where((_neighbor_max(agreement) >= 2) & (source_p80 >= 0.02), np.maximum(weighted_mean, source_p80), 0.0)
        corridor_envelope = np.where((_neighbor_max(agreement, radius=3) >= 2) & (source_p80 >= 0.02), np.maximum(weighted_mean, source_p80), 0.0)
        object_corridor_envelope = _object_corridor_envelope(source_p80, agreement)
        detached_noise_suppressed = _detached_noise_suppressed_envelope(source_p80, agreement)
        primary_core_noise_suppressed = _primary_core_noise_suppressed_envelope(source_p80, agreement)
        per_time["variant_weighted_mean"].append(weighted_mean)
        per_time["variant_source_max"].append(source_max)
        per_time["variant_source_p80"].append(source_p80)
        per_time["variant_agreement_gated_envelope"].append(envelope)
        per_time["variant_agreement_neighborhood_envelope"].append(neighborhood_envelope)
        per_time["variant_agreement_corridor_envelope"].append(corridor_envelope)
        per_time["variant_object_corridor_envelope"].append(object_corridor_envelope)
        per_time["variant_detached_noise_suppressed"].append(detached_noise_suppressed)
        per_time["variant_primary_core_noise_suppressed"].append(primary_core_noise_suppressed)
        source_p80_by_time.append(source_p80)
        agreement_by_time.append(agreement)
        high_agreement_by_time.append(high_agreement)
    variants: dict[str, np.ndarray] = {}
    for name, arrays in per_time.items():
        if arrays:
            variants[name] = _nanmax_stack(arrays)
    if source_p80_by_time:
        variants["variant_temporal_dominant_corridor"] = _temporal_dominant_corridor_envelope(source_p80_by_time, agreement_by_time)
        variants["variant_temporal_primary_core_corridor"] = _temporal_primary_core_corridor_envelope(source_p80_by_time, agreement_by_time)
        variants["variant_temporal_primary_core_expanded_corridor"] = _temporal_primary_core_expanded_corridor_envelope(
            source_p80_by_time,
            agreement_by_time,
        )
        variants["variant_temporal_primary_core_clean_corridor"] = _temporal_primary_core_clean_corridor_envelope(
            source_p80_by_time,
            agreement_by_time,
        )
        variants["variant_temporal_primary_core_guarded_corridor"] = _temporal_primary_core_guarded_corridor_envelope(
            source_p80_by_time,
            agreement_by_time,
        )
        variants["variant_temporal_primary_core_source_guarded_corridor"] = _temporal_primary_core_source_guarded_corridor_envelope(
            source_p80_by_time,
            agreement_by_time,
            high_agreement_by_time,
        )
    return variants


def _aligned_source_window_max(
    *,
    outputs_dir: Path,
    date: str,
    cycle: str,
    source: str,
    reference: xr.Dataset,
    time_indices: list[int],
) -> tuple[np.ndarray | None, int]:
    valid_times = [pd.Timestamp(value).tz_localize(None) for value in reference["time"].values]
    arrays: list[np.ndarray] = []
    for index in time_indices:
        aligned = _aligned_source_arrays(
            outputs_dir=outputs_dir,
            date=date,
            cycle=cycle,
            sources=[source],
            reference=reference,
            valid_time=valid_times[index],
        ).get(source)
        if aligned is not None:
            arrays.append(aligned)
    if not arrays:
        return None, 0
    return _nanmax_stack(arrays), len(arrays)


def _classify_window(rows: list[dict[str, Any]]) -> str:
    consensus = next((row for row in rows if row["series"] == "consensus"), None)
    sources = [row for row in rows if row["series"].startswith("source:")]
    if not consensus or not sources:
        return "insufficient_artifacts"
    source_cells = max(int(row.get("cells_ge_02", 0) or 0) for row in sources)
    source_max = max(float(row.get("max_probability") or 0.0) for row in sources)
    source_count_ge_02 = sum(int(row.get("cells_ge_02", 0) or 0) > 0 for row in sources)
    consensus_cells = int(consensus.get("cells_ge_02", 0) or 0)
    if source_max < 0.02:
        return "all_sources_weak"
    if source_count_ge_02 >= 2 and consensus_cells < max(4, int(source_cells * 0.25)):
        return "likely_consensus_underdispersion"
    if source_count_ge_02 == 1:
        return "single_source_signal"
    return "consensus_signal_present"


def build_signal_diagnostic(
    *,
    date: str,
    cycle: str,
    outputs_dir: Path,
    windows: list[tuple[str, str, str]],
    sources: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    consensus_path = consensus_product_path(outputs_dir, date, cycle)
    if not consensus_path.exists():
        raise FileNotFoundError(f"missing consensus artifact: {consensus_path}")
    rows: list[dict[str, Any]] = []
    classifications: dict[str, str] = {}
    source_coverage: dict[str, Any] = {}
    with xr.open_dataset(consensus_path) as consensus_ds:
        lat = np.asarray(consensus_ds["lat"].values, dtype=float)
        lon = np.asarray(consensus_ds["lon"].values, dtype=float)
        for window_name, valid_start, valid_end in windows:
            indices = _time_index(consensus_ds, valid_start, valid_end)
            window_rows: list[dict[str, Any]] = []
            coverage_rows: dict[str, dict[str, Any]] = {}
            consensus_values = _window_max(consensus_ds, CONSENSUS_FIELD, indices)
            if consensus_values is not None:
                row = {
                    "window": window_name,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "series": "consensus",
                    "available_time_count": len(indices),
                    **_safe_stats(consensus_values, lat=lat, lon=lon),
                }
                rows.append(row)
                window_rows.append(row)
            for source in sources:
                path = source_product_path(outputs_dir, date, cycle, source)
                if not path.exists():
                    row = {
                        "window": window_name,
                        "valid_start": valid_start,
                        "valid_end": valid_end,
                        "series": f"source:{source}",
                        "available_time_count": 0,
                        "missing_reason": "missing_source_product",
                        **_safe_stats(np.asarray([]), lat=lat, lon=lon),
                    }
                    rows.append(row)
                    window_rows.append(row)
                    coverage_rows[source] = {
                        "available_time_count": 0,
                        "expected_time_count": len(indices),
                        "coverage_fraction": 0.0,
                        "missing_reason": "missing_source_product",
                    }
                    continue
                values, available_count = _aligned_source_window_max(
                    outputs_dir=outputs_dir,
                    date=date,
                    cycle=cycle,
                    source=source,
                    reference=consensus_ds,
                    time_indices=indices,
                )
                row = {
                    "window": window_name,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "series": f"source:{source}",
                    "available_time_count": available_count,
                    "missing_reason": "" if values is not None else "no_valid_times_or_field",
                    **_safe_stats(values if values is not None else np.asarray([]), lat=lat, lon=lon),
                }
                rows.append(row)
                window_rows.append(row)
                coverage_rows[source] = {
                    "available_time_count": available_count,
                    "expected_time_count": len(indices),
                    "coverage_fraction": available_count / len(indices) if indices else 0.0,
                    "missing_reason": "" if values is not None else "no_valid_times_or_field",
                }
            for variant, values in _variant_arrays_for_window(
                outputs_dir=outputs_dir,
                date=date,
                cycle=cycle,
                sources=sources,
                reference=consensus_ds,
                time_indices=indices,
            ).items():
                rows.append(
                    {
                        "window": window_name,
                        "valid_start": valid_start,
                        "valid_end": valid_end,
                        "series": variant,
                        "available_time_count": len(indices),
                        **_safe_stats(values, lat=lat, lon=lon),
                    }
                )
            classifications[window_name] = _classify_window(window_rows)
            source_coverage[window_name] = {
                "expected_time_count": len(indices),
                "sources": coverage_rows,
                "full_window_sources": sorted(
                    source for source, row in coverage_rows.items() if len(indices) > 0 and row["available_time_count"] == len(indices)
                ),
                "partial_window_sources": sorted(
                    source for source, row in coverage_rows.items() if 0 < row["available_time_count"] < len(indices)
                ),
                "missing_window_sources": sorted(source for source, row in coverage_rows.items() if row["available_time_count"] == 0),
            }
    frame = pd.DataFrame(rows)
    summary = {
        "date": date,
        "cycle": str(cycle).zfill(2),
        "sources": sources,
        "window_classifications": classifications,
        "window_source_coverage": source_coverage,
        "likely_underdispersed_windows": [name for name, status in classifications.items() if status == "likely_consensus_underdispersion"],
    }
    return frame, summary


def _write_markdown(path: Path, frame: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# SPC-Window Tornado Signal Diagnostic",
        "",
        f"- date: `{summary['date']}`",
        f"- cycle: `{summary['cycle']}Z`",
        f"- sources: `{','.join(summary['sources'])}`",
        f"- likely_underdispersed_windows: `{','.join(summary['likely_underdispersed_windows']) or 'none'}`",
        "",
        "## Window Classifications",
        "",
    ]
    for name, status in summary["window_classifications"].items():
        lines.append(f"- `{name}`: `{status}`")
    lines.extend(["", "## Effective Source Coverage", ""])
    for name, coverage in summary.get("window_source_coverage", {}).items():
        full = ",".join(coverage.get("full_window_sources", [])) or "none"
        partial = ",".join(coverage.get("partial_window_sources", [])) or "none"
        missing = ",".join(coverage.get("missing_window_sources", [])) or "none"
        lines.append(
            f"- `{name}`: expected_times `{coverage.get('expected_time_count', 0)}`, "
            f"full `{full}`, partial `{partial}`, missing `{missing}`"
        )
    lines.extend(["", "## Signal Summary", ""])
    display = frame.copy()
    for column in ["max_probability", "peak_lat", "peak_lon"]:
        if column in display:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    columns = list(display.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in display.fillna("").to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--outputs-dir", default="data/outputs")
    parser.add_argument("--source", action="append", dest="sources", help="Source to include. Defaults to candidate source set.")
    parser.add_argument("--window", action="append", required=True, help="Window as name,start,end")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    frame, summary = build_signal_diagnostic(
        date=args.date,
        cycle=str(args.cycle).zfill(2),
        outputs_dir=Path(args.outputs_dir),
        windows=_parse_windows(args.window),
        sources=args.sources or list(AUTO_CONSENSUS_SOURCES),
    )
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(output_md, frame, summary)
    print(f"output_csv={output_csv.resolve()}")
    print(f"output_json={output_json.resolve()}")
    print(f"output_md={output_md.resolve()}")
    print(f"likely_underdispersed_windows={','.join(summary['likely_underdispersed_windows']) or 'none'}")


if __name__ == "__main__":
    main()
