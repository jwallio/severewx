"""Second-stage tornado concern model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from severewx.calibration.calibrate import fit_group_calibrator
from severewx.config import AppSettings
from severewx.models.model_io import load_model_artifact, save_model_artifact
from severewx.models.outbreak import apply_outbreak_model
from severewx.models.train import _make_backend, _split_train_valid
from severewx.verify.metrics import brier_score, safe_roc_auc


TORNADO_CONCERN_FEATURES = [
    "sig_tor_support",
    "tornado_favored_overlap",
    "hail_favored_overlap",
    "wind_favored_overlap",
    "tornado_hail_mode_delta",
    "tornado_wind_mode_delta",
    "competing_mode_overlap",
    "tornado_mode_advantage",
    "outbreak_risk",
    "scp_proxy",
    "any_prob",
    "wind_prob",
    "hail_prob",
    "competing_hazard_prob",
    "tornado_prob_advantage",
    "tornado_any_fraction",
    "low_lcl_support",
    "synoptic_support",
    "qg_support_proxy",
    "analog_outbreak_support",
    "analog_sigtor_support",
    "tornado_scp_interaction",
    "sigtor_scp_interaction",
    "tornado_support_interaction",
]
TORNADO_CONCERN_MODEL_COLUMNS = ["lead_day", *TORNADO_CONCERN_FEATURES]
COHERENT_FIELD_RADIUS_CELLS = 4
COHERENT_FIELD_SIGMA_CELLS = 1.8
COHERENT_FIELD_MIN_SEED_PROB = 0.02
ENVELOPE_FIELD_MIN_SUPPORT = 0.10


@dataclass(slots=True)
class TrainedTornadoConcernModel:
    artifact: dict[str, Any]


@dataclass(slots=True)
class IdentityCalibrator:
    def apply(self, frame: pd.DataFrame, score_column: str) -> np.ndarray:
        return frame[score_column].to_numpy(dtype=float)


def coherent_tornado_concern_grid(
    raw_values: np.ndarray,
    *,
    sig_tor_support: np.ndarray | None = None,
    tornado_favored_overlap: np.ndarray | None = None,
    scp_proxy: np.ndarray | None = None,
    outbreak_risk: np.ndarray | None = None,
    radius: int = COHERENT_FIELD_RADIUS_CELLS,
    sigma: float = COHERENT_FIELD_SIGMA_CELLS,
    min_seed_prob: float = COHERENT_FIELD_MIN_SEED_PROB,
) -> np.ndarray:
    """Spread calibrated concern peaks only into nearby tornado-supportive grid cells."""
    raw = np.nan_to_num(np.asarray(raw_values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if raw.ndim != 2 or raw.size == 0 or radius <= 0:
        return np.clip(raw, 0.0, 1.0)

    support_parts: list[np.ndarray] = []
    if sig_tor_support is not None:
        support_parts.append(np.clip(np.nan_to_num(np.asarray(sig_tor_support, dtype=float), nan=0.0) / 0.75, 0.0, 1.0))
    if tornado_favored_overlap is not None:
        support_parts.append(np.clip(np.nan_to_num(np.asarray(tornado_favored_overlap, dtype=float), nan=0.0) / 1.25, 0.0, 1.0))
    if scp_proxy is not None:
        support_parts.append(np.clip(np.nan_to_num(np.asarray(scp_proxy, dtype=float), nan=0.0) / 0.35, 0.0, 1.0))
    if outbreak_risk is not None:
        support_parts.append(np.clip(np.nan_to_num(np.asarray(outbreak_risk, dtype=float), nan=0.0) / 0.30, 0.0, 1.0))
    support = np.maximum.reduce(support_parts) if support_parts else np.ones_like(raw, dtype=float)
    support = np.where(np.isfinite(support), support, 0.0)
    support = np.clip(support, 0.0, 1.0)

    seed = np.where(raw >= min_seed_prob, raw, 0.0)
    if not seed.any():
        return np.clip(raw, 0.0, 1.0)

    spread = np.zeros_like(raw, dtype=float)
    padded = np.pad(seed, radius, mode="constant", constant_values=0.0)
    for row_offset in range(2 * radius + 1):
        for col_offset in range(2 * radius + 1):
            dy = row_offset - radius
            dx = col_offset - radius
            distance = float(np.hypot(dy, dx))
            if distance > radius:
                continue
            weight = float(np.exp(-(distance**2) / (2.0 * sigma**2)))
            shifted = padded[row_offset : row_offset + raw.shape[0], col_offset : col_offset + raw.shape[1]]
            spread = np.maximum(spread, shifted * weight)

    coherent = np.maximum(raw, spread * np.sqrt(support))
    return np.clip(coherent, 0.0, min(1.0, float(np.nanmax(raw)) if raw.size else 1.0))


def envelope_tornado_concern_grid(
    raw_values: np.ndarray,
    *,
    sig_tor_support: np.ndarray | None = None,
    tornado_favored_overlap: np.ndarray | None = None,
    scp_proxy: np.ndarray | None = None,
    outbreak_risk: np.ndarray | None = None,
) -> np.ndarray:
    """Prototype outlook-style field: broad support footprint with nested raw/coherent maxima."""
    raw = np.nan_to_num(np.asarray(raw_values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    coherent = coherent_tornado_concern_grid(
        raw,
        sig_tor_support=sig_tor_support,
        tornado_favored_overlap=tornado_favored_overlap,
        scp_proxy=scp_proxy,
        outbreak_risk=outbreak_risk,
    )
    support_parts: list[np.ndarray] = []
    sig_norm = None
    overlap_norm = None
    scp_norm = None
    outbreak_norm = None
    if sig_tor_support is not None:
        sig_norm = np.clip(np.nan_to_num(np.asarray(sig_tor_support, dtype=float), nan=0.0) / 1.25, 0.0, 1.0)
        support_parts.append(sig_norm)
    if tornado_favored_overlap is not None:
        overlap_norm = np.clip(np.nan_to_num(np.asarray(tornado_favored_overlap, dtype=float), nan=0.0) / 2.50, 0.0, 1.0)
        support_parts.append(overlap_norm)
    if scp_proxy is not None:
        scp_norm = np.clip(np.nan_to_num(np.asarray(scp_proxy, dtype=float), nan=0.0) / 0.75, 0.0, 1.0)
        support_parts.append(scp_norm)
    if outbreak_risk is not None:
        outbreak_norm = np.clip(np.nan_to_num(np.asarray(outbreak_risk, dtype=float), nan=0.0) / 0.40, 0.0, 1.0)
        support_parts.append(outbreak_norm)
    if not support_parts:
        return coherent
    max_support = np.maximum.reduce(support_parts)
    mean_support = np.mean(np.stack(support_parts, axis=0), axis=0)
    support = 0.55 * max_support + 0.45 * mean_support
    if sig_norm is not None and overlap_norm is not None:
        structural_gate = np.sqrt(np.clip(sig_norm * overlap_norm, 0.0, 1.0))
    elif overlap_norm is not None and scp_norm is not None:
        structural_gate = np.sqrt(np.clip(overlap_norm * scp_norm, 0.0, 1.0))
    else:
        structural_gate = max_support
    support = support * np.clip(0.35 + 0.65 * structural_gate, 0.0, 1.0)
    support = np.clip(np.nan_to_num(support, nan=0.0), 0.0, 1.0)
    envelope = np.zeros_like(raw, dtype=float)
    envelope = np.where(support >= 0.38, 0.02, envelope)
    envelope = np.where(support >= 0.55, 0.05, envelope)
    envelope = np.where((support >= 0.68) & (coherent >= 0.05), 0.10, envelope)
    envelope = np.where((support >= 0.78) & (coherent >= 0.10), 0.15, envelope)
    envelope = np.where((support >= 0.90) & (coherent >= 0.30), 0.30, envelope)
    return np.clip(np.maximum(envelope, coherent), 0.0, 1.0)


def tornado_environment_outlook_grid(
    *,
    sig_tor_support: np.ndarray | None = None,
    tornado_favored_overlap: np.ndarray | None = None,
    scp_proxy: np.ndarray | None = None,
    low_lcl_support: np.ndarray | None = None,
    synoptic_support: np.ndarray | None = None,
    outbreak_risk: np.ndarray | None = None,
    cin: np.ndarray | None = None,
) -> np.ndarray:
    """Ingredient-only public outlook field that does not use learned tornado-concern spikes."""
    required = [sig_tor_support, tornado_favored_overlap, scp_proxy]
    if any(value is None for value in required):
        template = next((value for value in required if value is not None), None)
        return np.zeros_like(np.asarray(template, dtype=float)) if template is not None else np.zeros((0, 0), dtype=float)

    sig = np.clip(np.nan_to_num(np.asarray(sig_tor_support, dtype=float), nan=0.0) / 1.25, 0.0, 1.0)
    overlap = np.clip(np.nan_to_num(np.asarray(tornado_favored_overlap, dtype=float), nan=0.0) / 2.50, 0.0, 1.0)
    scp = np.clip(np.nan_to_num(np.asarray(scp_proxy, dtype=float), nan=0.0) / 0.75, 0.0, 1.0)
    low_lcl = (
        np.clip(np.nan_to_num(np.asarray(low_lcl_support, dtype=float), nan=0.0) / 0.90, 0.0, 1.0)
        if low_lcl_support is not None
        else np.ones_like(sig)
    )
    synoptic = (
        np.clip(np.nan_to_num(np.asarray(synoptic_support, dtype=float), nan=0.0) / 120.0, 0.0, 1.0)
        if synoptic_support is not None
        else np.ones_like(sig)
    )
    outbreak = (
        np.clip(np.nan_to_num(np.asarray(outbreak_risk, dtype=float), nan=0.0) / 0.35, 0.0, 1.0)
        if outbreak_risk is not None
        else np.zeros_like(sig)
    )
    if cin is not None:
        cin_values = np.nan_to_num(np.asarray(cin, dtype=float), nan=0.0)
        cap_factor = np.clip(1.0 - np.maximum(np.abs(cin_values) - 75.0, 0.0) / 175.0, 0.25, 1.0)
    else:
        cap_factor = np.ones_like(sig)

    core = np.sqrt(np.clip(sig * overlap * np.maximum(scp, 0.05), 0.0, 1.0))
    environment = core * (0.45 + 0.25 * low_lcl + 0.20 * synoptic + 0.10 * outbreak) * cap_factor
    environment = np.clip(environment, 0.0, 1.0)

    outlook = np.zeros_like(environment, dtype=float)
    outlook = np.where(environment >= 0.22, 0.02, outlook)
    outlook = np.where(environment >= 0.34, 0.05, outlook)
    outlook = np.where(environment >= 0.48, 0.10, outlook)
    outlook = np.where(environment >= 0.62, 0.15, outlook)
    outlook = np.where(environment >= 0.78, 0.30, outlook)
    outlook = np.where(environment >= 0.90, 0.45, outlook)
    return outlook


def _expand_bool_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
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


def tornado_environment_outlook_v2_grid(
    *,
    sig_tor_support: np.ndarray | None = None,
    tornado_favored_overlap: np.ndarray | None = None,
    scp_proxy: np.ndarray | None = None,
    low_lcl_support: np.ndarray | None = None,
    synoptic_support: np.ndarray | None = None,
    outbreak_risk: np.ndarray | None = None,
    cin: np.ndarray | None = None,
) -> np.ndarray:
    """Compact ingredient-only outlook challenger that reduces broad low-end blankets.

    This intentionally stays separate from the baseline product field. It requires
    stronger agreement between tornado-specific ingredients and only keeps lower
    tiers near stronger cores when a stronger core exists.
    """
    required = [sig_tor_support, tornado_favored_overlap, scp_proxy]
    if any(value is None for value in required):
        template = next((value for value in required if value is not None), None)
        return np.zeros_like(np.asarray(template, dtype=float)) if template is not None else np.zeros((0, 0), dtype=float)

    sig = np.clip(np.nan_to_num(np.asarray(sig_tor_support, dtype=float), nan=0.0) / 1.35, 0.0, 1.0)
    overlap = np.clip(np.nan_to_num(np.asarray(tornado_favored_overlap, dtype=float), nan=0.0) / 2.75, 0.0, 1.0)
    scp = np.clip(np.nan_to_num(np.asarray(scp_proxy, dtype=float), nan=0.0) / 0.85, 0.0, 1.0)
    low_lcl = (
        np.clip(np.nan_to_num(np.asarray(low_lcl_support, dtype=float), nan=0.0) / 0.95, 0.0, 1.0)
        if low_lcl_support is not None
        else np.ones_like(sig)
    )
    synoptic = (
        np.clip(np.nan_to_num(np.asarray(synoptic_support, dtype=float), nan=0.0) / 140.0, 0.0, 1.0)
        if synoptic_support is not None
        else np.ones_like(sig)
    )
    outbreak = (
        np.clip(np.nan_to_num(np.asarray(outbreak_risk, dtype=float), nan=0.0) / 0.40, 0.0, 1.0)
        if outbreak_risk is not None
        else np.zeros_like(sig)
    )
    if cin is not None:
        cin_values = np.nan_to_num(np.asarray(cin, dtype=float), nan=0.0)
        cap_factor = np.clip(1.0 - np.maximum(np.abs(cin_values) - 60.0, 0.0) / 150.0, 0.20, 1.0)
    else:
        cap_factor = np.ones_like(sig)

    agreement = np.cbrt(np.clip(sig * overlap * np.maximum(scp, 0.02), 0.0, 1.0))
    structure = np.sqrt(np.clip(sig * overlap, 0.0, 1.0))
    background = 0.60 + 0.18 * low_lcl + 0.12 * synoptic + 0.10 * outbreak
    environment = np.clip(agreement * structure * background * cap_factor, 0.0, 1.0)

    outlook = np.zeros_like(environment, dtype=float)
    outlook = np.where((environment >= 0.16) & (agreement >= 0.22), 0.02, outlook)
    outlook = np.where((environment >= 0.24) & (agreement >= 0.30), 0.05, outlook)
    outlook = np.where((environment >= 0.34) & (agreement >= 0.42), 0.10, outlook)
    outlook = np.where((environment >= 0.48) & (agreement >= 0.55), 0.15, outlook)
    outlook = np.where((environment >= 0.78) & (agreement >= 0.78), 0.30, outlook)
    outlook = np.where((environment >= 0.86) & (agreement >= 0.86), 0.45, outlook)

    if np.any(outlook >= 0.30):
        keep_15 = _expand_bool_mask(outlook >= 0.30, radius=3) | (outlook >= 0.30)
        outlook = np.where((outlook >= 0.15) & (outlook < 0.30) & ~keep_15, 0.0, outlook)
    if np.any(outlook >= 0.15):
        keep_10 = _expand_bool_mask(outlook >= 0.15, radius=4) | (outlook >= 0.15)
        outlook = np.where((outlook >= 0.10) & (outlook < 0.15) & ~keep_10, 0.0, outlook)
    if np.any(outlook >= 0.10):
        keep_05 = _expand_bool_mask(outlook >= 0.10, radius=4) | (outlook >= 0.10)
        keep_02 = _expand_bool_mask(outlook >= 0.05, radius=4) | (outlook >= 0.05)
        outlook = np.where((outlook >= 0.05) & (outlook < 0.10) & ~keep_05, 0.0, outlook)
        outlook = np.where((outlook >= 0.02) & (outlook < 0.05) & ~keep_02, 0.0, outlook)
    return outlook


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


def _is_broad_outlook_field(values: np.ndarray) -> bool:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if clean.ndim != 2 or clean.size == 0:
        return False
    ge02 = int(np.count_nonzero(clean >= 0.02))
    ge10 = int(np.count_nonzero(clean >= 0.10))
    largest_ge02 = max(_component_sizes(clean >= 0.02), default=0)
    grid_cells = int(clean.size)
    ge02_fraction = ge02 / max(grid_cells, 1)
    largest_fraction = largest_ge02 / max(grid_cells, 1)
    return (
        ge02 >= 1450
        or largest_ge02 >= 1300
        or (ge02 >= 1200 and ge10 >= 850)
        or (ge02_fraction >= 0.45 and ge10 >= 0.35 * grid_cells)
        or largest_fraction >= 0.40
    )


def tornado_environment_outlook_hybrid_grid(
    *,
    sig_tor_support: np.ndarray | None = None,
    tornado_favored_overlap: np.ndarray | None = None,
    scp_proxy: np.ndarray | None = None,
    low_lcl_support: np.ndarray | None = None,
    synoptic_support: np.ndarray | None = None,
    outbreak_risk: np.ndarray | None = None,
    cin: np.ndarray | None = None,
) -> np.ndarray:
    """Use the compact challenger only when the original outlook is objectively broad."""
    baseline = tornado_environment_outlook_grid(
        sig_tor_support=sig_tor_support,
        tornado_favored_overlap=tornado_favored_overlap,
        scp_proxy=scp_proxy,
        low_lcl_support=low_lcl_support,
        synoptic_support=synoptic_support,
        outbreak_risk=outbreak_risk,
        cin=cin,
    )
    if not _is_broad_outlook_field(baseline):
        return baseline
    return tornado_environment_outlook_v2_grid(
        sig_tor_support=sig_tor_support,
        tornado_favored_overlap=tornado_favored_overlap,
        scp_proxy=scp_proxy,
        low_lcl_support=low_lcl_support,
        synoptic_support=synoptic_support,
        outbreak_risk=outbreak_risk,
        cin=cin,
    )


def _ensure_base_probability_columns(frame: pd.DataFrame, settings: AppSettings, paths: Any) -> pd.DataFrame:
    working = frame.copy()
    for hazard in ("tornado", "hail", "wind", "any"):
        probability_column = f"{hazard}_prob"
        if probability_column in working.columns:
            continue
        artifact = load_model_artifact(paths, hazard)
        columns = [column for column in artifact["columns"] if column in working.columns]
        raw = artifact["model"].predict_proba(working[columns])[:, 1]
        calibrated = artifact["calibrator"].apply(working.assign(raw_pred=raw), "raw_pred")
        working[probability_column] = np.clip(calibrated, 0.0, 1.0)
    if "outbreak_risk" not in working.columns:
        outbreak_artifact = load_model_artifact(paths, "outbreak")
        working = apply_outbreak_model(outbreak_artifact, working)
    return working


def _candidate_pixel_mask(frame: pd.DataFrame, settings: AppSettings) -> pd.Series:
    config = settings.get("models.tornado_concern", {}) or {}
    masks: list[pd.Series] = []
    threshold_specs = [
        ("sig_tor_support", "candidate_sig_tor_support", 0.10),
        ("tornado_favored_overlap", "candidate_tornado_overlap", 0.15),
        ("scp_proxy", "candidate_scp_proxy", 0.10),
        ("low_lcl_support", "candidate_low_lcl_support", 0.75),
        ("synoptic_support", "candidate_synoptic_support", 60.0),
    ]
    for column, key, default in threshold_specs:
        if column not in frame.columns:
            continue
        threshold = float(config.get(key, default))
        masks.append(frame[column].fillna(0.0).astype(float).ge(threshold))
    if not masks:
        return pd.Series(True, index=frame.index)
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined = combined | mask
    return combined


def _significant_tornado_dates(outbreak_table: pd.DataFrame | None) -> set[str]:
    if outbreak_table is None or outbreak_table.empty or "date" not in outbreak_table.columns:
        return set()
    if "significant_tornado_support" in outbreak_table.columns:
        mask = outbreak_table["significant_tornado_support"].fillna(0).astype(float).gt(0)
    else:
        mask = pd.Series(False, index=outbreak_table.index)
    return set(outbreak_table.loc[mask, "date"].astype(str))


def _hard_negative_mask(frame: pd.DataFrame, settings: AppSettings) -> pd.Series:
    masks: list[pd.Series] = []
    for label in ("hail", "wind", "any"):
        if label in frame.columns:
            masks.append(frame[label].fillna(0).astype(float).gt(0))
    hail_overlap_threshold = float(settings.get("models.tornado_concern.hard_negative_hail_overlap", 0.35))
    if "hail_favored_overlap" in frame.columns:
        masks.append(frame["hail_favored_overlap"].fillna(0).astype(float).ge(hail_overlap_threshold))
    wind_overlap_threshold = float(settings.get("models.tornado_concern.hard_negative_wind_overlap", 0.35))
    if "wind_favored_overlap" in frame.columns:
        masks.append(frame["wind_favored_overlap"].fillna(0).astype(float).ge(wind_overlap_threshold))
    if not masks:
        return pd.Series(False, index=frame.index)
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined = combined | mask
    return combined


def _augment_concern_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    for column in ("tornado_favored_overlap", "hail_favored_overlap", "wind_favored_overlap"):
        if column not in working.columns:
            working[column] = 0.0
    for column in ("tornado_prob", "hail_prob", "wind_prob", "any_prob", "sig_tor_support", "scp_proxy"):
        if column not in working.columns:
            working[column] = 0.0
    working["tornado_hail_mode_delta"] = (
        working["tornado_favored_overlap"].fillna(0.0).astype(float) - working["hail_favored_overlap"].fillna(0.0).astype(float)
    )
    working["tornado_wind_mode_delta"] = (
        working["tornado_favored_overlap"].fillna(0.0).astype(float) - working["wind_favored_overlap"].fillna(0.0).astype(float)
    )
    working["competing_mode_overlap"] = np.maximum(
        working["hail_favored_overlap"].fillna(0.0).astype(float),
        working["wind_favored_overlap"].fillna(0.0).astype(float),
    )
    working["tornado_mode_advantage"] = (
        working["tornado_favored_overlap"].fillna(0.0).astype(float) - working["competing_mode_overlap"].fillna(0.0).astype(float)
    )
    working["competing_hazard_prob"] = np.maximum(
        working["hail_prob"].fillna(0.0).astype(float),
        working["wind_prob"].fillna(0.0).astype(float),
    )
    working["tornado_prob_advantage"] = working["tornado_prob"].fillna(0.0).astype(float) - working["competing_hazard_prob"].fillna(0.0).astype(float)
    working["tornado_any_fraction"] = working["tornado_prob"].fillna(0.0).astype(float) / np.maximum(
        working["any_prob"].fillna(0.0).astype(float),
        0.001,
    )
    working["tornado_scp_interaction"] = (
        working["tornado_favored_overlap"].fillna(0.0).astype(float) * working["scp_proxy"].fillna(0.0).astype(float)
    )
    working["sigtor_scp_interaction"] = working["sig_tor_support"].fillna(0.0).astype(float) * working["scp_proxy"].fillna(0.0).astype(float)
    working["tornado_support_interaction"] = (
        working["tornado_favored_overlap"].fillna(0.0).astype(float) * working["sig_tor_support"].fillna(0.0).astype(float)
    )
    return working


def _apply_public_readiness_guardrail(frame: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    config = settings.get("models.tornado_concern.public_readiness_guardrail", {}) or {}
    if not bool(config.get("enabled", True)):
        return frame
    if "tornado_concern_prob" not in frame.columns:
        return frame
    working = frame.copy()
    defaults = {
        "tornado_support_interaction": 0.0,
        "sigtor_scp_interaction": 0.0,
        "tornado_corridor_index": 0.0,
        "tornado_favored_overlap": 0.0,
        "scp_proxy": 0.0,
        "analog_outbreak_support": 0.0,
    }
    for column, default in defaults.items():
        if column not in working.columns:
            working[column] = default
    display_threshold = float(config.get("display_threshold", 0.05))
    max_allowed_prob = float(config.get("max_allowed_prob", 0.049))
    interaction_threshold = float(config.get("min_interaction", 0.003))
    corridor_threshold = float(config.get("min_tornado_corridor_index", 0.01))
    overlap_threshold = float(config.get("min_tornado_overlap", 0.08))
    scp_threshold = float(config.get("min_scp_proxy", 0.02))
    analog_outbreak_threshold = float(config.get("min_analog_outbreak_support", 0.60))
    weak_interaction = (
        working["tornado_support_interaction"].fillna(0.0).astype(float).lt(interaction_threshold)
        & working["sigtor_scp_interaction"].fillna(0.0).astype(float).lt(interaction_threshold)
    )
    weak_structure = (
        working["tornado_corridor_index"].fillna(0.0).astype(float).lt(corridor_threshold)
        & working["tornado_favored_overlap"].fillna(0.0).astype(float).lt(overlap_threshold)
        & working["scp_proxy"].fillna(0.0).astype(float).lt(scp_threshold)
        & working["analog_outbreak_support"].fillna(0.0).astype(float).lt(analog_outbreak_threshold)
    )
    cap_mask = working["tornado_concern_prob"].fillna(0.0).astype(float).ge(display_threshold) & weak_interaction & weak_structure
    if cap_mask.any():
        original = working["tornado_concern_prob"].astype(float)
        working.loc[cap_mask, "tornado_concern_prob"] = np.minimum(original.loc[cap_mask], max_allowed_prob)
        working["tornado_concern_guardrail_applied"] = cap_mask
    else:
        working["tornado_concern_guardrail_applied"] = False
    return working


def tornado_concern_training_frame(
    gridpoint_frame: pd.DataFrame,
    settings: AppSettings,
    paths: Any,
    outbreak_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    working = gridpoint_frame.copy()
    if "tornado" not in working.columns:
        raise ValueError("tornado concern training requires the spatial tornado label column")
    working["tornado_concern_target"] = working["tornado"].fillna(0).astype(float).gt(0).astype(int)
    candidate_mask = _candidate_pixel_mask(working, settings)
    positive_mask = working["tornado_concern_target"].eq(1)
    working = working.loc[positive_mask | candidate_mask].copy()
    positive_mask = working["tornado_concern_target"].eq(1)
    negative_mask = ~positive_mask
    max_negative_rows = int(settings.get("models.tornado_concern.max_negative_rows", 250000))
    negative_rows = working.loc[negative_mask]
    if len(negative_rows) > max_negative_rows:
        hard_negative_rows = negative_rows.loc[_hard_negative_mask(negative_rows, settings)]
        other_negative_rows = negative_rows.drop(index=hard_negative_rows.index, errors="ignore")
        hard_negative_fraction = float(settings.get("models.tornado_concern.hard_negative_fraction", 0.70))
        hard_negative_target = min(len(hard_negative_rows), max(1, int(round(max_negative_rows * hard_negative_fraction))))
        if len(hard_negative_rows) > hard_negative_target:
            hard_negative_rows = hard_negative_rows.sample(
                n=hard_negative_target,
                random_state=int(settings.get("models.random_state", 42)),
            )
        remaining = max_negative_rows - len(hard_negative_rows)
        if remaining > 0 and len(other_negative_rows) > remaining:
            other_negative_rows = other_negative_rows.sample(
                n=remaining,
                random_state=int(settings.get("models.random_state", 42)),
            )
        negative_rows = pd.concat([hard_negative_rows, other_negative_rows], ignore_index=False)
    combined = pd.concat([working.loc[positive_mask], negative_rows], ignore_index=True)
    combined = _ensure_base_probability_columns(combined, settings, paths)
    combined = _augment_concern_features(combined)
    for column in TORNADO_CONCERN_FEATURES:
        if column not in combined.columns:
            combined[column] = 0.0
    combined["tornado_concern_sample_weight"] = 1.0
    significant_dates = _significant_tornado_dates(outbreak_table)
    if not significant_dates and "significant_tornado_support" in combined.columns:
        significant_dates = set(
            combined.loc[combined["significant_tornado_support"].fillna(0).astype(float).gt(0), "date"].astype(str)
        )
    if significant_dates:
        positive_weight = float(settings.get("models.tornado_concern.significant_positive_weight", 2.0))
        enhanced_positive_mask = combined["tornado_concern_target"].eq(1) & combined["date"].astype(str).isin(significant_dates)
        combined.loc[enhanced_positive_mask, "tornado_concern_sample_weight"] = positive_weight
    hard_negative_weight = float(settings.get("models.tornado_concern.hard_negative_weight", 1.5))
    if hard_negative_weight != 1.0:
        hard_negative_mask = combined["tornado_concern_target"].eq(0) & _hard_negative_mask(combined, settings)
        combined.loc[hard_negative_mask, "tornado_concern_sample_weight"] = hard_negative_weight
    return combined.sort_values(["date", "lead_day", "lat", "lon"]).reset_index(drop=True)


def train_tornado_concern_model(
    gridpoint_frame: pd.DataFrame,
    settings: AppSettings,
    paths: Any,
    outbreak_table: pd.DataFrame | None = None,
) -> TrainedTornadoConcernModel:
    frame = tornado_concern_training_frame(gridpoint_frame, settings, paths, outbreak_table=outbreak_table)
    columns = [column for column in TORNADO_CONCERN_MODEL_COLUMNS if column in frame.columns]
    train_frame, valid_frame = _split_train_valid(frame, float(settings.get("models.validation_fraction", 0.2)))
    model, backend = _make_backend(settings)
    fit_kwargs: dict[str, Any] = {}
    if "tornado_concern_sample_weight" in train_frame.columns:
        fit_kwargs["sample_weight"] = train_frame["tornado_concern_sample_weight"].to_numpy(dtype=float)
    model.fit(train_frame[columns], train_frame["tornado_concern_target"], **fit_kwargs)
    raw_valid = model.predict_proba(valid_frame[columns])[:, 1]
    if valid_frame["tornado_concern_target"].nunique() < 2:
        calibrator: Any = IdentityCalibrator()
    else:
        calibrator = fit_group_calibrator(valid_frame.assign(raw_pred=raw_valid), "raw_pred", "tornado_concern_target", settings)
    calibrated = calibrator.apply(valid_frame.assign(raw_pred=raw_valid), "raw_pred")
    artifact = {
        "hazard": "tornado_concern",
        "backend": backend,
        "model": model,
        "columns": columns,
        "metrics": {
            "brier_score": brier_score(valid_frame["tornado_concern_target"].to_numpy(), calibrated),
            "roc_auc": safe_roc_auc(valid_frame["tornado_concern_target"].to_numpy(), calibrated),
        },
        "calibrator": calibrator,
        "target_name": "tornado_concern_target",
        "sample_weighting": {
            "significant_positive_weight": float(settings.get("models.tornado_concern.significant_positive_weight", 2.0)),
            "hard_negative_weight": float(settings.get("models.tornado_concern.hard_negative_weight", 1.5)),
        },
    }
    return TrainedTornadoConcernModel(artifact=artifact)


def save_tornado_concern_model(
    gridpoint_frame: pd.DataFrame,
    settings: AppSettings,
    paths: Any,
    outbreak_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    trained = train_tornado_concern_model(gridpoint_frame, settings, paths, outbreak_table=outbreak_table)
    save_model_artifact(trained.artifact, paths, "tornado_concern")
    return trained.artifact


def apply_tornado_concern_model(frame: pd.DataFrame, paths: Any, settings: AppSettings | None = None) -> pd.DataFrame:
    settings = settings or AppSettings(raw={})
    try:
        artifact = load_model_artifact(paths, "tornado_concern")
    except FileNotFoundError:
        return frame
    working = frame.copy()
    for column in artifact.get("columns", []):
        if column not in working.columns:
            working[column] = 0.0
    working = _augment_concern_features(working)
    raw = artifact["model"].predict_proba(working[artifact["columns"]])[:, 1]
    calibrated = artifact["calibrator"].apply(working.assign(raw_pred=raw), "raw_pred")
    working["tornado_concern_prob"] = np.clip(calibrated, 0.0, 1.0)
    working = _apply_public_readiness_guardrail(working, settings)
    working["tornado_concern_environment_envelope"] = working["tornado_concern_prob"]
    working["tornado_environment_outlook"] = 0.0
    working["tornado_environment_outlook_v2"] = 0.0
    working["tornado_environment_outlook_hybrid"] = 0.0
    group_columns = [column for column in ["time"] if column in working.columns]
    if group_columns:
        for _, index in working.groupby(group_columns, sort=False).groups.items():
            subset = working.loc[index]
            rows = len(subset["lat"].unique())
            cols = len(subset["lon"].unique())
            envelope = envelope_tornado_concern_grid(
                subset["tornado_concern_prob"].to_numpy(dtype=float).reshape(rows, cols),
                sig_tor_support=subset["sig_tor_support"].to_numpy(dtype=float).reshape(rows, cols) if "sig_tor_support" in subset else None,
                tornado_favored_overlap=subset["tornado_favored_overlap"].to_numpy(dtype=float).reshape(rows, cols) if "tornado_favored_overlap" in subset else None,
                scp_proxy=subset["scp_proxy"].to_numpy(dtype=float).reshape(rows, cols) if "scp_proxy" in subset else None,
                outbreak_risk=subset["outbreak_risk"].to_numpy(dtype=float).reshape(rows, cols) if "outbreak_risk" in subset else None,
            ).ravel()
            outlook = tornado_environment_outlook_grid(
                sig_tor_support=subset["sig_tor_support"].to_numpy(dtype=float).reshape(rows, cols) if "sig_tor_support" in subset else None,
                tornado_favored_overlap=subset["tornado_favored_overlap"].to_numpy(dtype=float).reshape(rows, cols) if "tornado_favored_overlap" in subset else None,
                scp_proxy=subset["scp_proxy"].to_numpy(dtype=float).reshape(rows, cols) if "scp_proxy" in subset else None,
                low_lcl_support=subset["low_lcl_support"].to_numpy(dtype=float).reshape(rows, cols) if "low_lcl_support" in subset else None,
                synoptic_support=subset["synoptic_support"].to_numpy(dtype=float).reshape(rows, cols) if "synoptic_support" in subset else None,
                outbreak_risk=subset["outbreak_risk"].to_numpy(dtype=float).reshape(rows, cols) if "outbreak_risk" in subset else None,
                cin=subset["cin"].to_numpy(dtype=float).reshape(rows, cols) if "cin" in subset else None,
            ).ravel()
            outlook_v2 = tornado_environment_outlook_v2_grid(
                sig_tor_support=subset["sig_tor_support"].to_numpy(dtype=float).reshape(rows, cols) if "sig_tor_support" in subset else None,
                tornado_favored_overlap=subset["tornado_favored_overlap"].to_numpy(dtype=float).reshape(rows, cols) if "tornado_favored_overlap" in subset else None,
                scp_proxy=subset["scp_proxy"].to_numpy(dtype=float).reshape(rows, cols) if "scp_proxy" in subset else None,
                low_lcl_support=subset["low_lcl_support"].to_numpy(dtype=float).reshape(rows, cols) if "low_lcl_support" in subset else None,
                synoptic_support=subset["synoptic_support"].to_numpy(dtype=float).reshape(rows, cols) if "synoptic_support" in subset else None,
                outbreak_risk=subset["outbreak_risk"].to_numpy(dtype=float).reshape(rows, cols) if "outbreak_risk" in subset else None,
                cin=subset["cin"].to_numpy(dtype=float).reshape(rows, cols) if "cin" in subset else None,
            ).ravel()
            outlook_hybrid = tornado_environment_outlook_hybrid_grid(
                sig_tor_support=subset["sig_tor_support"].to_numpy(dtype=float).reshape(rows, cols) if "sig_tor_support" in subset else None,
                tornado_favored_overlap=subset["tornado_favored_overlap"].to_numpy(dtype=float).reshape(rows, cols) if "tornado_favored_overlap" in subset else None,
                scp_proxy=subset["scp_proxy"].to_numpy(dtype=float).reshape(rows, cols) if "scp_proxy" in subset else None,
                low_lcl_support=subset["low_lcl_support"].to_numpy(dtype=float).reshape(rows, cols) if "low_lcl_support" in subset else None,
                synoptic_support=subset["synoptic_support"].to_numpy(dtype=float).reshape(rows, cols) if "synoptic_support" in subset else None,
                outbreak_risk=subset["outbreak_risk"].to_numpy(dtype=float).reshape(rows, cols) if "outbreak_risk" in subset else None,
                cin=subset["cin"].to_numpy(dtype=float).reshape(rows, cols) if "cin" in subset else None,
            ).ravel()
            working.loc[index, "tornado_concern_environment_envelope"] = envelope
            working.loc[index, "tornado_environment_outlook"] = outlook
            working.loc[index, "tornado_environment_outlook_v2"] = outlook_v2
            working.loc[index, "tornado_environment_outlook_hybrid"] = outlook_hybrid
    return working
