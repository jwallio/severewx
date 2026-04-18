"""Light analog and historical context features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from severewx.config import AppSettings


def synoptic_vector_columns(settings: AppSettings) -> list[str]:
    configured = list(settings.get("features.analog_feature_columns", []))
    extras = [
        "synoptic_support",
        "sig_tor_support",
        "outbreak_corridor_index",
        "tornado_corridor_index",
        "moisture_transport",
    ]
    return list(dict.fromkeys([*configured, *extras]))


def summarize_fields(dataset: xr.Dataset, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | pd.Timestamp]] = []
    for valid_time in pd.to_datetime(dataset["time"].values):
        subset = dataset.sel(time=valid_time)
        row: dict[str, float | pd.Timestamp] = {"time": valid_time}
        for column in columns:
            if column in subset:
                row[column] = float(subset[column].mean().item())
                row[f"max_{column}"] = float(subset[column].max().item())
        rows.append(row)
    return pd.DataFrame(rows)


def build_analog_reference(frame: pd.DataFrame, settings: AppSettings) -> pd.DataFrame:
    columns = synoptic_vector_columns(settings)
    aggregations: dict[str, str] = {}
    for column in columns:
        if column in frame.columns:
            aggregations[column] = "mean"
            aggregations[f"max_{column}"] = "max"
    if not aggregations:
        return pd.DataFrame()
    grouped = frame.groupby(["date", "lead_day"], as_index=False).agg(
        **{
            column if not column.startswith("max_") else column: (column.replace("max_", ""), method)
            for column, method in aggregations.items()
        },
        tornado_outbreak=("tornado_outbreak", "max"),
        hail_outbreak=("hail_outbreak", "max"),
        wind_outbreak=("wind_outbreak", "max"),
        any_outbreak=("any_outbreak", "max"),
        significant_tornado_support=("significant_tornado_support", "max"),
    )
    mode_columns = ["tornado_outbreak", "hail_outbreak", "wind_outbreak"]
    mode_frame = grouped[mode_columns]
    all_na_mask = mode_frame.isna().all(axis=1)
    grouped["dominant_mode"] = (
        mode_frame.fillna(float("-inf"))
        .idxmax(axis=1)
        .str.replace("_outbreak", "", regex=False)
        .mask(all_na_mask, "none")
    )
    return grouped


def save_analog_reference(frame: pd.DataFrame, path: Path, settings: AppSettings) -> Path:
    reference = build_analog_reference(frame, settings)
    reference.to_parquet(path, index=False)
    return path


def load_analog_archive(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported analog archive: {path}")


def _similarity_score(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std[std == 0] = 1.0
    normalized_target = (target - mean) / std
    normalized_reference = (reference - mean) / std
    return np.linalg.norm(normalized_reference - normalized_target, axis=1)


def _subset_similarity(archive: pd.DataFrame, subset_mask: pd.Series, columns: list[str], target: np.ndarray, top_k: int) -> tuple[float, float]:
    subset_mask = pd.Series(subset_mask, index=archive.index, dtype=bool)
    sample = archive.loc[subset_mask, columns]
    if sample.empty:
        return 0.0, 0.0
    distances = _similarity_score(sample.to_numpy(dtype=float), target)
    top_indices = np.argsort(distances)[:top_k]
    top_sample = archive.loc[sample.index[top_indices]]
    similarity = 1.0 / (1.0 + float(distances[top_indices].mean()))
    support = float(top_sample.get("any_outbreak", pd.Series([0.0])).mean())
    return similarity, support


def compute_analog_features(dataset: xr.Dataset, archive: pd.DataFrame, settings: AppSettings) -> xr.Dataset:
    columns = synoptic_vector_columns(settings)
    summary = summarize_fields(dataset, columns)
    top_k = int(settings.get("features.analog_top_k", 15))

    metrics: dict[str, list[float]] = {
        "analog_similarity": [],
        "analog_tornado_similarity": [],
        "analog_hail_similarity": [],
        "analog_wind_similarity": [],
        "analog_tornado_rate": [],
        "analog_hail_rate": [],
        "analog_wind_rate": [],
        "analog_any_rate": [],
        "analog_outbreak_support": [],
        "analog_sigtor_support": [],
        "analog_mode_coherence": [],
    }
    metadata_rows: list[dict[str, Any]] = []

    for _, row in summary.iterrows():
        available = [column for column in summary.columns if column != "time" and column in archive.columns]
        if archive.empty or not available:
            for key in metrics:
                metrics[key].append(0.0)
            metadata_rows.append({"time": pd.Timestamp(row["time"]), "dominant_mode": "none"})
            continue
        target = row[available].to_numpy(dtype=float)
        archive_values = archive[available].to_numpy(dtype=float)
        distances = _similarity_score(archive_values, target)
        top_indices = np.argsort(distances)[:top_k]
        top_sample = archive.iloc[top_indices]
        overall_similarity = 1.0 / (1.0 + float(distances[top_indices].mean()))
        tornado_mask = archive["tornado_outbreak"].gt(0) | archive["significant_tornado_support"].gt(0) if {"tornado_outbreak", "significant_tornado_support"}.issubset(archive.columns) else pd.Series(False, index=archive.index)
        hail_mask = archive["hail_outbreak"].gt(0) if "hail_outbreak" in archive.columns else pd.Series(False, index=archive.index)
        wind_mask = archive["wind_outbreak"].gt(0) if "wind_outbreak" in archive.columns else pd.Series(False, index=archive.index)
        tornado_similarity, _ = _subset_similarity(
            archive,
            tornado_mask,
            available,
            target,
            top_k=min(top_k, max(3, top_k // 2)),
        )
        hail_similarity, _ = _subset_similarity(archive, hail_mask, available, target, top_k=min(top_k, max(3, top_k // 2)))
        wind_similarity, _ = _subset_similarity(archive, wind_mask, available, target, top_k=min(top_k, max(3, top_k // 2)))

        tornado_rate = float(top_sample.get("tornado_outbreak", pd.Series([0.0])).mean())
        hail_rate = float(top_sample.get("hail_outbreak", pd.Series([0.0])).mean())
        wind_rate = float(top_sample.get("wind_outbreak", pd.Series([0.0])).mean())
        any_rate = float(top_sample.get("any_outbreak", pd.Series([0.0])).mean())
        sigtor_rate = float(top_sample.get("significant_tornado_support", pd.Series([0.0])).mean())
        mode_values = np.array([tornado_similarity, hail_similarity, wind_similarity], dtype=float)
        dominant_mode = ["tornado", "hail", "wind"][int(mode_values.argmax())] if np.any(mode_values > 0) else "none"
        mode_coherence = float(mode_values.max() - np.median(mode_values))

        metrics["analog_similarity"].append(overall_similarity)
        metrics["analog_tornado_similarity"].append(tornado_similarity)
        metrics["analog_hail_similarity"].append(hail_similarity)
        metrics["analog_wind_similarity"].append(wind_similarity)
        metrics["analog_tornado_rate"].append(tornado_rate)
        metrics["analog_hail_rate"].append(hail_rate)
        metrics["analog_wind_rate"].append(wind_rate)
        metrics["analog_any_rate"].append(any_rate)
        metrics["analog_outbreak_support"].append(any_rate)
        metrics["analog_sigtor_support"].append(sigtor_rate)
        metrics["analog_mode_coherence"].append(mode_coherence)
        metadata_rows.append(
            {
                "time": pd.Timestamp(row["time"]),
                "dominant_mode": dominant_mode,
                "top_analog_dates": ",".join(str(value) for value in top_sample["date"].head(3).tolist()) if "date" in top_sample else "",
            }
        )

    analog_ds = xr.Dataset(
        {key: ("time", np.asarray(values, dtype=np.float32)) for key, values in metrics.items()},
        coords={"time": dataset["time"]},
    )
    dataset = dataset.merge(analog_ds.broadcast_like(dataset[["cape"]]))
    dataset.attrs["analog_metadata"] = pd.DataFrame(metadata_rows).to_json(orient="records", date_format="iso")
    return dataset
