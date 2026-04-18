import numpy as np
import pandas as pd
import xarray as xr

from severewx.config import load_settings
from severewx.features.composites import build_feature_dataset


def test_feature_dataset_preserves_core_shape() -> None:
    settings = load_settings()
    dataset = xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), np.full((1, 2, 2), 295.0)),
            "td2m": (("time", "lat", "lon"), np.full((1, 2, 2), 289.0)),
            "mslp": (("time", "lat", "lon"), np.full((1, 2, 2), 100800.0)),
            "u10": (("time", "lat", "lon"), np.full((1, 2, 2), 10.0)),
            "v10": (("time", "lat", "lon"), np.full((1, 2, 2), 2.0)),
            "cape": (("time", "lat", "lon"), np.full((1, 2, 2), 1500.0)),
            "cin": (("time", "lat", "lon"), np.full((1, 2, 2), -50.0)),
            "u850": (("time", "lat", "lon"), np.full((1, 2, 2), 20.0)),
            "v850": (("time", "lat", "lon"), np.full((1, 2, 2), 6.0)),
            "t700": (("time", "lat", "lon"), np.full((1, 2, 2), 275.0)),
            "z500": (("time", "lat", "lon"), np.full((1, 2, 2), 5700.0)),
            "u500": (("time", "lat", "lon"), np.full((1, 2, 2), 30.0)),
            "v500": (("time", "lat", "lon"), np.full((1, 2, 2), 10.0)),
            "pwat": (("time", "lat", "lon"), np.full((1, 2, 2), 30.0)),
            "t500": (("time", "lat", "lon"), np.full((1, 2, 2), 255.0)),
        },
        coords={"time": pd.to_datetime(["2026-04-09T00:00:00"]), "lat": [35.0, 36.0], "lon": [-98.0, -97.0]},
    )
    features = build_feature_dataset(dataset, settings)
    assert features["shear_0_6km"].shape == (1, 2, 2)
    assert features["sig_tor_support"].shape == (1, 2, 2)
    assert features["analog_similarity"].shape == (1, 2, 2)
