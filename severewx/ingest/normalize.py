"""Normalization and validation for operational forecast fields."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


STANDARD_NAME_MAP = {
    "t2m": ["t2m", "2t", "t", "TMP_2maboveground", "TMP_2_m_above_ground"],
    "td2m": ["td2m", "2d", "d2m", "DPT_2maboveground", "DPT_2_m_above_ground"],
    "mslp": ["mslp", "prmsl", "PRMSL_meansealevel", "PRMSL_mean_sea_level"],
    "u10": ["u10", "10u", "UGRD_10maboveground", "UGRD_10_m_above_ground"],
    "v10": ["v10", "10v", "VGRD_10maboveground", "VGRD_10_m_above_ground"],
    "cape": ["cape", "CAPE_surface"],
    "cin": ["cin", "CIN_surface"],
    "u850": ["u850", "UGRD_850mb"],
    "v850": ["v850", "VGRD_850mb"],
    "t700": ["t700", "TMP_700mb"],
    "z500": ["z500", "HGT_500mb"],
    "u500": ["u500", "UGRD_500mb"],
    "v500": ["v500", "VGRD_500mb"],
    "pwat": ["pwat", "PWAT_entireatmosphereconsideredasasinglelayer"],
    "t500": ["t500", "TMP_500mb"],
}


def _standardize_coords(dataset: xr.Dataset) -> xr.Dataset:
    coord_map = {}
    if "latitude" in dataset.coords:
        coord_map["latitude"] = "lat"
    if "longitude" in dataset.coords:
        coord_map["longitude"] = "lon"
    if "valid_time" in dataset.coords and "time" not in dataset.coords:
        coord_map["valid_time"] = "time"
    if coord_map:
        dataset = dataset.rename(coord_map)

    if "valid_time" in dataset.coords and "time" in dataset.coords:
        valid_time = pd.to_datetime(np.atleast_1d(dataset["valid_time"].values)).tz_localize(None)
        dataset = dataset.drop_vars("valid_time").assign_coords(time=valid_time[0] if len(valid_time) == 1 else valid_time)

    if "lon" in dataset.coords:
        lon = xr.where(dataset["lon"] > 180, dataset["lon"] - 360, dataset["lon"])
        dataset = dataset.assign_coords(lon=lon).sortby("lon")
    if "lat" in dataset.coords:
        dataset = dataset.sortby("lat")
    return dataset


def _pick_var(dataset: xr.Dataset, candidates: Iterable[str]) -> xr.DataArray | None:
    for name in candidates:
        if name in dataset:
            return dataset[name]
    return None


def _strip_auxiliary_coords(data: xr.DataArray) -> xr.DataArray:
    keep = {"time", "lat", "lon", "latitude", "longitude", "valid_time"}
    drop_names = [name for name in data.coords if name not in keep]
    if drop_names:
        data = data.reset_coords(names=drop_names, drop=True)
    return data


def _ensure_time_coord(dataset: xr.Dataset, valid_time: pd.Timestamp | None) -> xr.Dataset:
    if "time" not in dataset.coords:
        if valid_time is None:
            raise ValueError("valid_time is required when the input dataset has no time coordinate")
        dataset = dataset.expand_dims(time=[pd.Timestamp(valid_time)])
    elif "time" not in dataset.dims:
        time_values = pd.to_datetime(np.atleast_1d(dataset["time"].values)).tz_localize(None)
        dataset = dataset.drop_vars("time").expand_dims(time=time_values)
    elif dataset.sizes.get("time", 0) == 0:
        raise ValueError("input dataset has an empty time dimension")
    else:
        dataset = dataset.assign_coords(time=pd.to_datetime(dataset["time"].values).tz_localize(None))
    return dataset


def _apply_missing_field_fallbacks(dataset: xr.Dataset) -> tuple[xr.Dataset, dict[str, str]]:
    fallbacks: dict[str, str] = {}
    if "td2m" not in dataset and "t2m" in dataset:
        if "pwat" in dataset:
            td2m = xr.where(dataset["t2m"] < 260.0, dataset["t2m"] - 8.0, 273.15 + dataset["pwat"] * 0.42)
            dataset["td2m"] = xr.where(td2m > dataset["t2m"], dataset["t2m"], td2m).astype(np.float32)
            fallbacks["td2m"] = "derived from t2m and pwat proxy"
        else:
            dataset["td2m"] = (dataset["t2m"] - 12.0).astype(np.float32)
            fallbacks["td2m"] = "derived from fixed low-level moisture proxy"
    if "cin" not in dataset and "cape" in dataset:
        dataset["cin"] = (-0.05 * dataset["cape"]).astype(np.float32)
        fallbacks["cin"] = "derived from CAPE penalty proxy"
    if "pwat" not in dataset and "td2m" in dataset:
        dataset["pwat"] = ((dataset["td2m"] - 265.0) * 1.8).clip(min=0.0, max=60.0).astype(np.float32)
        fallbacks["pwat"] = "derived from low-level moisture proxy"
    if "t500" not in dataset and "t700" in dataset:
        dataset["t500"] = (dataset["t700"] - 18.0).astype(np.float32)
        fallbacks["t500"] = "derived from 700-500 mb lapse assumption"
    return dataset, fallbacks


def _validate_coord_consistency(dataset: xr.Dataset) -> list[str]:
    issues: list[str] = []
    for coord in ("time", "lat", "lon"):
        if coord not in dataset.coords:
            issues.append(f"missing coordinate {coord}")
    if issues:
        return issues
    if dataset["lat"].ndim != 1 or dataset["lon"].ndim != 1:
        issues.append("lat/lon must be one-dimensional")
    if not np.all(np.diff(dataset["lat"].values) >= 0):
        issues.append("latitude coordinate is not monotonic increasing")
    if not np.all(np.diff(dataset["lon"].values) >= 0):
        issues.append("longitude coordinate is not monotonic increasing")
    if pd.Index(dataset["time"].values).duplicated().any():
        issues.append("time coordinate contains duplicates")
    for name, data in dataset.data_vars.items():
        expected_dims = tuple(dim for dim in ("time", "lat", "lon") if dim in data.dims)
        if data.dims != expected_dims:
            issues.append(f"{name} has inconsistent dims {data.dims}")
    return issues


def dataset_diagnostics(dataset: xr.Dataset, requested_leads: list[int] | None = None) -> dict[str, Any]:
    valid_times = [pd.Timestamp(value).isoformat() for value in pd.to_datetime(dataset["time"].values)]
    available_fields = sorted(dataset.data_vars)
    missing_leads: list[int] = []
    if requested_leads:
        init_time = pd.Timestamp(dataset["time"].values[0])
        seen_leads = {
            int(round((pd.Timestamp(value) - init_time).total_seconds() / 3600.0))
            for value in pd.to_datetime(dataset["time"].values)
        }
        missing_leads = sorted(set(int(lead) for lead in requested_leads) - seen_leads)
    diagnostics = {
        "available_fields": available_fields,
        "valid_times": valid_times,
        "grid_shape": {
            "time": int(dataset.sizes.get("time", 0)),
            "lat": int(dataset.sizes.get("lat", 0)),
            "lon": int(dataset.sizes.get("lon", 0)),
        },
        "missing_requested_leads": missing_leads,
        "coord_issues": _validate_coord_consistency(dataset),
    }
    return diagnostics


def normalize_dataset(
    dataset: xr.Dataset,
    valid_time: pd.Timestamp | None = None,
    required_fields: list[str] | None = None,
    requested_leads: list[int] | None = None,
) -> tuple[xr.Dataset, dict[str, Any]]:
    dataset = _standardize_coords(dataset)
    normalized = xr.Dataset(coords={key: dataset.coords[key] for key in dataset.coords if key in {"time", "lat", "lon"}})
    for standard_name, candidates in STANDARD_NAME_MAP.items():
        value = _pick_var(dataset, candidates)
        if value is not None:
            normalized[standard_name] = _strip_auxiliary_coords(value).astype(np.float32)
    extra_coords = [name for name in normalized.coords if name not in {"time", "lat", "lon"}]
    if extra_coords:
        normalized = normalized.reset_coords(names=extra_coords, drop=True)

    normalized = _ensure_time_coord(normalized, valid_time)
    normalized, fallbacks = _apply_missing_field_fallbacks(normalized)
    diagnostics = dataset_diagnostics(normalized, requested_leads=requested_leads)
    diagnostics["fallbacks_used"] = fallbacks
    required = required_fields or []
    missing_required = sorted(field for field in required if field not in normalized)
    diagnostics["missing_required_fields"] = missing_required
    if diagnostics["coord_issues"]:
        raise ValueError(f"normalized dataset coordinate issues: {diagnostics['coord_issues']}")
    if missing_required:
        raise ValueError(f"normalized dataset is missing required fields: {missing_required}")
    normalized.attrs["diagnostic_summary"] = str(diagnostics)
    return normalized, diagnostics
