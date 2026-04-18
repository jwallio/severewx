import numpy as np
import pandas as pd
import xarray as xr

from severewx.config import load_settings
from severewx.ingest.nomads import FIELD_FILTERS, SyntheticForecastSource
from severewx.ingest.normalize import normalize_dataset
from severewx.ingest.storage import raw_grib_metadata_path
from severewx.utils.paths import build_paths


def test_normalize_dataset_derives_optional_fields_and_diagnostics() -> None:
    settings = load_settings()
    raw = xr.Dataset(
        {
            "t2m": (("lat", "lon"), np.full((2, 2), 295.0)),
            "mslp": (("lat", "lon"), np.full((2, 2), 100900.0)),
            "u10": (("lat", "lon"), np.full((2, 2), 10.0)),
            "v10": (("lat", "lon"), np.full((2, 2), 2.0)),
            "cape": (("lat", "lon"), np.full((2, 2), 1500.0)),
            "u850": (("lat", "lon"), np.full((2, 2), 20.0)),
            "v850": (("lat", "lon"), np.full((2, 2), 6.0)),
            "t700": (("lat", "lon"), np.full((2, 2), 275.0)),
            "z500": (("lat", "lon"), np.full((2, 2), 5700.0)),
            "u500": (("lat", "lon"), np.full((2, 2), 30.0)),
            "v500": (("lat", "lon"), np.full((2, 2), 10.0)),
        },
        coords={"lat": [35.0, 36.0], "lon": [-98.0, -97.0]},
    )
    normalized, diagnostics = normalize_dataset(
        raw,
        valid_time=pd.Timestamp("2026-04-09T00:00:00"),
        required_fields=list(settings.get("ingest.required_fields", [])),
        requested_leads=[0, 6],
    )
    assert "td2m" in normalized
    assert "cin" in normalized
    assert diagnostics["missing_required_fields"] == []
    assert "td2m" in diagnostics["fallbacks_used"]


def test_normalize_dataset_handles_scalar_time_and_valid_time_without_conflict() -> None:
    settings = load_settings()
    raw = xr.Dataset(
        {
            "TMP_2_m_above_ground": (("latitude", "longitude"), np.full((2, 2), 295.0)),
            "PRMSL_mean_sea_level": (("latitude", "longitude"), np.full((2, 2), 100900.0)),
            "UGRD_10_m_above_ground": (("latitude", "longitude"), np.full((2, 2), 10.0)),
            "VGRD_10_m_above_ground": (("latitude", "longitude"), np.full((2, 2), 2.0)),
            "CAPE_surface": (("latitude", "longitude"), np.full((2, 2), 1500.0)),
            "UGRD_850mb": (("latitude", "longitude"), np.full((2, 2), 20.0)),
            "VGRD_850mb": (("latitude", "longitude"), np.full((2, 2), 6.0)),
            "TMP_700mb": (("latitude", "longitude"), np.full((2, 2), 275.0)),
            "HGT_500mb": (("latitude", "longitude"), np.full((2, 2), 5700.0)),
            "UGRD_500mb": (("latitude", "longitude"), np.full((2, 2), 30.0)),
            "VGRD_500mb": (("latitude", "longitude"), np.full((2, 2), 10.0)),
        },
        coords={
            "latitude": [35.0, 36.0],
            "longitude": [262.0, 263.0],
            "time": np.datetime64("2024-03-14T00:00:00"),
            "valid_time": np.datetime64("2024-03-14T06:00:00"),
        },
    )
    normalized, diagnostics = normalize_dataset(
        raw,
        required_fields=list(settings.get("ingest.required_fields", [])),
        requested_leads=[0],
    )
    assert normalized.dims["time"] == 1
    assert "valid_time" not in normalized.coords
    assert pd.Timestamp(normalized["time"].values[0]) == pd.Timestamp("2024-03-14T06:00:00")
    assert normalized["lon"].values.tolist() == [-98.0, -97.0]
    assert diagnostics["coord_issues"] == []


def test_normalize_dataset_strips_auxiliary_scalar_coords_from_fields() -> None:
    settings = load_settings()
    raw = xr.Dataset(
        {
            "TMP_2_m_above_ground": xr.DataArray(
                np.full((2, 2), 295.0),
                dims=("latitude", "longitude"),
                coords={
                    "latitude": [35.0, 36.0],
                    "longitude": [262.0, 263.0],
                    "step": np.timedelta64(6, "h"),
                },
            ),
            "PRMSL_mean_sea_level": xr.DataArray(
                np.full((2, 2), 100900.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "UGRD_10_m_above_ground": xr.DataArray(
                np.full((2, 2), 10.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "VGRD_10_m_above_ground": xr.DataArray(
                np.full((2, 2), 2.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "CAPE_surface": xr.DataArray(
                np.full((2, 2), 1500.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "UGRD_850mb": xr.DataArray(
                np.full((2, 2), 20.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "VGRD_850mb": xr.DataArray(
                np.full((2, 2), 6.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "TMP_700mb": xr.DataArray(
                np.full((2, 2), 275.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "HGT_500mb": xr.DataArray(
                np.full((2, 2), 5700.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "UGRD_500mb": xr.DataArray(
                np.full((2, 2), 30.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
            "VGRD_500mb": xr.DataArray(
                np.full((2, 2), 10.0),
                dims=("latitude", "longitude"),
                coords={"latitude": [35.0, 36.0], "longitude": [262.0, 263.0], "step": np.timedelta64(6, "h")},
            ),
        },
        coords={"time": np.datetime64("2024-04-26T06:00:00")},
    )
    normalized, diagnostics = normalize_dataset(
        raw,
        required_fields=list(settings.get("ingest.required_fields", [])),
        requested_leads=[0],
    )
    assert "step" not in normalized.coords
    assert diagnostics["coord_issues"] == []


def test_synthetic_ingest_summary_reports_cache_and_mode(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    dataset, summary = SyntheticForecastSource().fetch_cycle("2026-04-09", "00", settings, paths)
    assert dataset.attrs["source"] == "synthetic"
    assert summary["source_mode"] == "synthetic"
    assert summary["cache_overview"]["cache_hits"] == 0
    assert not raw_grib_metadata_path(paths, "2026-04-09", "00", 0).exists()


def test_grib_filters_disambiguate_surface_cape_and_cin() -> None:
    assert FIELD_FILTERS["cape"]["typeOfLevel"] == "surface"
    assert FIELD_FILTERS["cin"]["typeOfLevel"] == "surface"
