import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from severewx.config import load_settings
from severewx.render.layout import board_output_path, cartopy_available, conus_template, create_map_figure, map_output_path, style_map_axes
from severewx.render.maps import render_probability_maps
from severewx.utils.paths import build_paths


def _prediction_dataset(include_wind: bool = True) -> xr.Dataset:
    times = pd.to_datetime(["2026-04-10T00:00:00", "2026-04-10T06:00:00"])
    data_vars = {
        "any_prob": (("time", "lat", "lon"), np.array([[[0.2, 0.5], [0.3, 0.6]], [[0.4, 0.7], [0.5, 0.8]]])),
        "tornado_prob": (("time", "lat", "lon"), np.array([[[0.1, 0.2], [0.15, 0.25]], [[0.2, 0.3], [0.22, 0.35]]])),
        "hail_prob": (("time", "lat", "lon"), np.array([[[0.3, 0.4], [0.35, 0.45]], [[0.25, 0.5], [0.4, 0.55]]])),
        "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.05, 0.1], [0.08, 0.12]], [[0.09, 0.15], [0.1, 0.18]]])),
        "confidence_score": (("time", "lat", "lon"), np.array([[[0.6, 0.7], [0.62, 0.74]], [[0.58, 0.69], [0.61, 0.72]]])),
        "bust_risk_score": (("time", "lat", "lon"), np.array([[[0.2, 0.3], [0.22, 0.34]], [[0.25, 0.28], [0.24, 0.31]]])),
    }
    if include_wind:
        data_vars["wind_prob"] = (("time", "lat", "lon"), np.array([[[0.25, 0.3], [0.28, 0.35]], [[0.2, 0.4], [0.26, 0.42]]]))
    return xr.Dataset(data_vars, coords={"time": times, "lat": [34.0, 35.0], "lon": [-98.0, -97.0]})


def test_render_output_paths_are_deterministic(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    paths = build_paths(settings)
    assert map_output_path(paths, "2026-04-09", "00", "2026-04-10", 2, "tornado").name == "2026-04-09_00_day2_2026-04-10_tornado.png"
    assert board_output_path(paths, "2026-04-09", "00", "2026-04-10", 2).name == "2026-04-09_00_day2_2026-04-10_daily_board.png"


def test_render_probability_maps_writes_daily_suite_and_board_with_missing_panel(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["render"]["cartopy"] = False
    paths = build_paths(settings)
    dataset = _prediction_dataset(include_wind=False)
    outputs = render_probability_maps(dataset, "2026-04-09", "00", settings, paths)
    assert len(outputs) == 8
    expected_names = {
        "2026-04-09_00_day1_2026-04-10_any_severe.png",
        "2026-04-09_00_day1_2026-04-10_tornado.png",
        "2026-04-09_00_day1_2026-04-10_hail.png",
        "2026-04-09_00_day1_2026-04-10_wind.png",
        "2026-04-09_00_day1_2026-04-10_outbreak_risk.png",
        "2026-04-09_00_day1_2026-04-10_confidence.png",
        "2026-04-09_00_day1_2026-04-10_bust_risk.png",
        "2026-04-09_00_day1_2026-04-10_daily_board.png",
    }
    assert {path.name for path in outputs} == expected_names
    for path in outputs:
        assert path.exists()
        assert path.stat().st_size > 0


def test_conus_template_uses_fixed_extent_and_fallback_mode_when_disabled() -> None:
    settings = load_settings()
    settings.raw["render"]["cartopy"] = False
    template = conus_template(settings)
    assert template.extent == (-125.0, -66.5, 24.0, 50.0)
    assert template.use_cartopy is False
    assert template.projection_name == "matplotlib"


def test_conus_template_prefers_cartopy_when_available() -> None:
    settings = load_settings()
    template = conus_template(settings)
    assert template.extent == (-125.0, -66.5, 24.0, 50.0)
    assert template.use_cartopy is cartopy_available()


def test_style_map_axes_applies_conus_extent_in_fallback_mode() -> None:
    settings = load_settings()
    settings.raw["render"]["cartopy"] = False
    fig, ax = create_map_figure(settings)
    style_map_axes(ax, np.array([-98.0, -97.0]), np.array([34.0, 35.0]), settings=settings)
    assert tuple(round(value, 1) for value in ax.get_xlim()) == (-125.0, -66.5)
    assert tuple(round(value, 1) for value in ax.get_ylim()) == (24.0, 50.0)
    plt.close(fig)
