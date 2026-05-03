"""Lead-time weighted multi-source forecast consensus helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from severewx.utils.dates import cycle_datetime


AUTO_CONSENSUS_SOURCES = ["hrrr_recent", "rap_recent", "nam_recent", "aws_recent"]
CONSENSUS_INPUT_FIELD = "tornado_environment_outlook_hybrid"
CONSENSUS_FIELD = "tornado_environment_outlook_hybrid_consensus"
DISPLAY_AGREEMENT_THRESHOLD = 0.02
SPREAD_CONFIDENCE_SCALE = 0.20


@dataclass(frozen=True, slots=True)
class ConsensusSource:
    name: str
    product_path: Path
    metadata_path: Path | None = None


def source_product_path(outputs_dir: Path, date: str, cycle: str, source: str) -> Path:
    return outputs_dir / f"forecast_products_{date}_{cycle}_{source}.nc"


def source_metadata_path(outputs_dir: Path, date: str, cycle: str, source: str) -> Path:
    return outputs_dir / f"forecast_metadata_{date}_{cycle}_{source}.json"


def consensus_product_path(outputs_dir: Path, date: str, cycle: str) -> Path:
    return outputs_dir / f"forecast_consensus_{date}_{cycle}.nc"


def consensus_metadata_path(outputs_dir: Path, date: str, cycle: str) -> Path:
    return outputs_dir / f"forecast_consensus_metadata_{date}_{cycle}.json"


def consensus_weights_for_lead(lead_hour: int | float) -> dict[str, float]:
    lead = float(lead_hour)
    if lead <= 18:
        return {"hrrr_recent": 0.45, "rap_recent": 0.25, "nam_recent": 0.15, "aws_recent": 0.15}
    if lead <= 48:
        return {"hrrr_recent": 0.30, "nam_recent": 0.30, "aws_recent": 0.25, "gefs_mean_recent": 0.15}
    if lead <= 60:
        return {"aws_recent": 0.40, "gefs_mean_recent": 0.35, "nam_recent": 0.25}
    if lead <= 84:
        return {"aws_recent": 0.45, "gefs_mean_recent": 0.40, "nam_recent": 0.15}
    if lead <= 96:
        return {"gefs_mean_recent": 0.55, "aws_recent": 0.45}
    return {"gefs_mean_recent": 0.65, "aws_recent": 0.35}


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _is_real_source(metadata: dict[str, Any]) -> bool:
    ingest = metadata.get("ingest_summary", {}) if isinstance(metadata.get("ingest_summary", {}), dict) else {}
    return str(ingest.get("source_mode", "real")) != "synthetic"


def _source_model(metadata: dict[str, Any], source: str) -> str:
    ingest = metadata.get("ingest_summary", {}) if isinstance(metadata.get("ingest_summary", {}), dict) else {}
    return str(ingest.get("source_model") or ingest.get("source") or source)


def _timestamps(dataset: xr.Dataset) -> list[pd.Timestamp]:
    if "time" not in dataset.coords:
        return []
    return [pd.Timestamp(value).tz_localize(None) for value in dataset["time"].values]


def _lead_hour(init_time: pd.Timestamp, valid_time: pd.Timestamp) -> int:
    return int(round((valid_time - init_time).total_seconds() / 3600.0))


def _align_slice(source_dataset: xr.Dataset, field_name: str, valid_time: pd.Timestamp, reference: xr.Dataset) -> np.ndarray | None:
    source_times = _timestamps(source_dataset)
    if valid_time not in source_times or field_name not in source_dataset:
        return None
    selected = source_dataset[field_name].isel(time=source_times.index(valid_time))
    aligned = selected.interp(lat=reference["lat"], lon=reference["lon"], kwargs={"fill_value": np.nan})
    return np.asarray(aligned.values, dtype=float)


def _normalize_weights(weights: dict[str, float], present_sources: list[str]) -> dict[str, float]:
    present = {source: float(weights.get(source, 0.0)) for source in present_sources if float(weights.get(source, 0.0)) > 0.0}
    if not present:
        present = {source: 1.0 for source in present_sources}
    total = sum(present.values())
    return {source: value / total for source, value in present.items()} if total else {}


def build_forecast_consensus(
    *,
    date: str,
    cycle: str,
    sources: list[ConsensusSource],
    output_path: Path,
    metadata_path: Path,
    field_name: str = CONSENSUS_INPUT_FIELD,
    unavailable_sources: list[dict[str, str]] | None = None,
) -> tuple[Path, Path]:
    loaded: dict[str, xr.Dataset] = {}
    source_metadata: dict[str, dict[str, Any]] = {}
    excluded_sources: list[dict[str, str]] = list(unavailable_sources or [])
    for source in sources:
        metadata = _load_json(source.metadata_path)
        source_metadata[source.name] = metadata
        if not source.product_path.exists():
            excluded_sources.append({"source": source.name, "reason": "missing_product"})
            continue
        if not _is_real_source(metadata):
            excluded_sources.append({"source": source.name, "reason": "synthetic_source"})
            continue
        dataset = xr.load_dataset(source.product_path)
        if field_name not in dataset:
            dataset.close()
            excluded_sources.append({"source": source.name, "reason": f"missing_field:{field_name}"})
            continue
        loaded[source.name] = dataset

    if not loaded:
        raise RuntimeError("no real source product datasets available for consensus")

    reference_name = "aws_recent" if "aws_recent" in loaded else next(iter(loaded))
    reference = loaded[reference_name]
    init_time = pd.Timestamp(cycle_datetime(date, cycle))
    valid_times = _timestamps(reference)
    if not valid_times:
        raise RuntimeError(f"reference source {reference_name} has no valid times")

    consensus_values: list[np.ndarray] = []
    mean_values: list[np.ndarray] = []
    max_values: list[np.ndarray] = []
    spread_values: list[np.ndarray] = []
    agreement_values: list[np.ndarray] = []
    primary_values: list[np.ndarray] = []
    confidence_values: list[np.ndarray] = []
    time_metadata: list[dict[str, Any]] = []
    source_codes = {source: index + 1 for index, source in enumerate(sorted(loaded))}

    for valid_time in valid_times:
        lead_hour = _lead_hour(init_time, valid_time)
        configured_weights = consensus_weights_for_lead(lead_hour)
        arrays: dict[str, np.ndarray] = {}
        for source_name, dataset in loaded.items():
            aligned = _align_slice(dataset, field_name, valid_time, reference)
            if aligned is not None and not np.isnan(aligned).all():
                arrays[source_name] = aligned
        normalized_weights = _normalize_weights(configured_weights, list(arrays))
        if not normalized_weights:
            shape = (reference.sizes["lat"], reference.sizes["lon"])
            consensus = np.full(shape, np.nan, dtype=np.float32)
            mean = consensus.copy()
            maximum = consensus.copy()
            spread = consensus.copy()
            agreement = np.zeros(shape, dtype=np.float32)
            primary = np.zeros(shape, dtype=np.int16)
        else:
            stack = np.stack([arrays[source] for source in normalized_weights], axis=0)
            source_order = list(normalized_weights)
            weights = np.asarray([normalized_weights[source] for source in source_order], dtype=float).reshape((-1, 1, 1))
            valid_mask = np.isfinite(stack)
            weighted_sum = np.nansum(np.where(valid_mask, stack * weights, 0.0), axis=0)
            available_weight = np.sum(np.where(valid_mask, weights, 0.0), axis=0)
            consensus = np.divide(weighted_sum, available_weight, out=np.full_like(weighted_sum, np.nan), where=available_weight > 0)
            mean = np.nanmean(stack, axis=0)
            maximum = np.nanmax(stack, axis=0)
            minimum = np.nanmin(stack, axis=0)
            spread = maximum - minimum
            agreement = np.sum((stack >= DISPLAY_AGREEMENT_THRESHOLD) & valid_mask, axis=0).astype(np.float32)
            consensus = np.where(agreement >= 2, consensus, 0.0)
            weighted_fields = np.where(valid_mask, stack * weights, -np.inf)
            primary_index = np.argmax(weighted_fields, axis=0)
            primary = np.zeros_like(primary_index, dtype=np.int16)
            for index, source_name in enumerate(source_order):
                primary[primary_index == index] = source_codes[source_name]
            primary[~np.isfinite(consensus)] = 0
        confidence = 1.0 - np.clip(spread / SPREAD_CONFIDENCE_SCALE, 0.0, 1.0)
        confidence = np.where(agreement < 2, confidence * 0.5, confidence)

        consensus_values.append(consensus.astype(np.float32))
        mean_values.append(mean.astype(np.float32))
        max_values.append(maximum.astype(np.float32))
        spread_values.append(spread.astype(np.float32))
        agreement_values.append(agreement.astype(np.float32))
        primary_values.append(primary.astype(np.int16))
        confidence_values.append(confidence.astype(np.float32))
        time_metadata.append(
            {
                "valid_time": valid_time.isoformat(),
                "lead_hour": lead_hour,
                "configured_weights": configured_weights,
                "applied_weights": normalized_weights,
                "contributing_sources": list(normalized_weights),
            }
        )

    dims = ("time", "lat", "lon")
    consensus_ds = xr.Dataset(
        {
            CONSENSUS_FIELD: (dims, np.stack(consensus_values, axis=0)),
            "model_mean": (dims, np.stack(mean_values, axis=0)),
            "model_max": (dims, np.stack(max_values, axis=0)),
            "model_spread": (dims, np.stack(spread_values, axis=0)),
            "model_agreement_count": (dims, np.stack(agreement_values, axis=0)),
            "primary_model": (dims, np.stack(primary_values, axis=0)),
            "consensus_confidence_modifier": (dims, np.stack(confidence_values, axis=0)),
        },
        coords={
            "time": reference["time"].values,
            "lat": reference["lat"].values,
            "lon": reference["lon"].values,
        },
        attrs={
            "source": "forecast_consensus",
            "consensus_input_field": field_name,
            "primary_model_code_map": json.dumps(source_codes, sort_keys=True),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consensus_ds.to_netcdf(output_path)
    consensus_ds.close()

    metadata = {
        "init_date": date,
        "cycle": cycle,
        "source": "forecast_consensus",
        "source_mode": "real",
        "consensus_input_field": field_name,
        "consensus_field": CONSENSUS_FIELD,
        "reference_source": reference_name,
        "included_sources": sorted(loaded),
        "excluded_sources": excluded_sources,
        "source_models": {source: _source_model(source_metadata.get(source, {}), source) for source in sorted(loaded)},
        "primary_model_code_map": source_codes,
        "time_weights": time_metadata,
        "source_artifacts": {
            source.name: {
                "product_path": str(source.product_path),
                "metadata_path": str(source.metadata_path) if source.metadata_path is not None else "",
            }
            for source in sources
        },
        "ingest_summary": {
            "source": "forecast_consensus",
            "source_mode": "real",
            "source_origin": "derived",
            "included_sources": sorted(loaded),
            "excluded_sources": excluded_sources,
            "real_ingest_available": True,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for dataset in loaded.values():
        dataset.close()
    return output_path, metadata_path
