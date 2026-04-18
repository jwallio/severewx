"""Neighborhood label utilities."""

from __future__ import annotations

import numpy as np


EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def neighborhood_hits(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    report_lat: np.ndarray,
    report_lon: np.ndarray,
    radius_km: float,
) -> np.ndarray:
    if report_lat.size == 0:
        return np.zeros(grid_lat.shape, dtype=np.int8)
    distances = haversine_km(
        grid_lat[..., None],
        grid_lon[..., None],
        report_lat[None, None, :],
        report_lon[None, None, :],
    )
    return (distances <= radius_km).any(axis=-1).astype(np.int8)
