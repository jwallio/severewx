import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from severewx.config import load_settings
from severewx.render.layout import create_board_figure, review_output_path
from severewx.render.review import (
    REVIEW_GAP,
    REVIEW_LEFT_WIDTH,
    REVIEW_OUTER_PAD_X,
    REVIEW_OUTER_PAD_Y,
    SUMMARY_CARD_LAYOUT,
    _review_zone_bounds,
    _summary_specs,
    render_case_review_boards,
)
from severewx.utils.paths import build_paths


matplotlib.use("Agg")


def _prediction_dataset(include_tornado: bool = True) -> xr.Dataset:
    data_vars = {
        "any_prob": (("time", "lat", "lon"), np.array([[[0.7]], [[0.5]]])),
        "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.6]], [[0.3]]])),
        "confidence_score": (("time", "lat", "lon"), np.array([[[0.65]], [[0.55]]])),
        "bust_risk_score": (("time", "lat", "lon"), np.array([[[0.3]], [[0.45]]])),
    }
    if include_tornado:
        data_vars["tornado_prob"] = (("time", "lat", "lon"), np.array([[[0.4]], [[0.2]]]))
    return xr.Dataset(
        data_vars,
        coords={"time": pd.to_datetime(["2026-04-09T00:00:00", "2026-04-09T06:00:00"]), "lat": [35.0], "lon": [-97.0]},
    )


def _label_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "tornado": (("date", "lat", "lon"), np.array([[[1]]])),
            "hail": (("date", "lat", "lon"), np.array([[[0]]])),
            "wind": (("date", "lat", "lon"), np.array([[[0]]])),
            "any": (("date", "lat", "lon"), np.array([[[1]]])),
        },
        coords={"date": np.array([np.datetime64("2026-04-09")]), "lat": [35.0], "lon": [-97.0]},
    )


def _mismatched_label_dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "tornado": (("date", "lat", "lon"), np.array([[[1]]], dtype=float)),
            "hail": (("date", "lat", "lon"), np.array([[[0]]], dtype=float)),
            "wind": (("date", "lat", "lon"), np.array([[[0]]], dtype=float)),
            "any": (("date", "lat", "lon"), np.array([[[1]]], dtype=float)),
        },
        coords={"date": np.array([np.datetime64("2026-04-09")]), "lat": [34.5], "lon": [-97.5]},
    )


def _verification_payload() -> dict[str, object]:
    return {
        "run_summary": {
            "training_quality_tier": "mixed",
            "training_data_source": "historical_feature_archive",
            "evaluation_source": "nomads",
        },
        "per_day": [
            {
                "valid_date": "2026-04-09",
                "lead_day": 1,
                "observed_category": "tornado_outbreak_day",
                "forecast_tornado_prob": 0.4,
                "forecast_any_prob": 0.7,
                "forecast_outbreak_prob": 0.6,
                "forecast_confidence": 0.65,
                "forecast_bust_risk": 0.3,
                "any_brier": 0.12,
                "tornado_brier": 0.08,
            }
        ],
    }


def test_review_output_path_is_deterministic(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["render"]["cartopy"] = False
    paths = build_paths(settings)
    assert review_output_path(paths, "2026-04-09", "00", "2026-04-10", 2).name == "2026-04-09_00_day2_2026-04-10_case_review.png"


def test_render_case_review_boards_writes_board(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["render"]["cartopy"] = False
    paths = build_paths(settings)
    outputs = render_case_review_boards(
        _prediction_dataset(),
        _label_dataset(),
        _verification_payload(),
        "2026-04-09",
        "00",
        settings,
        paths,
    )
    assert len(outputs) == 1
    assert outputs[0].exists()
    assert outputs[0].stat().st_size > 0


def test_render_case_review_boards_handles_missing_panels(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["render"]["cartopy"] = False
    paths = build_paths(settings)
    outputs = render_case_review_boards(
        _prediction_dataset(include_tornado=False),
        None,
        _verification_payload(),
        "2026-04-09",
        "00",
        settings,
        paths,
    )
    assert len(outputs) == 1
    assert outputs[0].exists()
    assert outputs[0].stat().st_size > 0


def test_render_case_review_boards_keeps_distinct_observed_and_summary_zone(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["render"]["cartopy"] = False
    paths = build_paths(settings)
    outputs = render_case_review_boards(
        _prediction_dataset(include_tornado=False),
        None,
        _verification_payload(),
        "2026-04-09",
        "00",
        settings,
        paths,
    )
    assert outputs[0].name.endswith("_case_review.png")
    assert outputs[0].stat().st_size > 0


def test_render_case_review_boards_aligns_mismatched_label_grid(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["render"]["cartopy"] = False
    paths = build_paths(settings)
    prediction = xr.Dataset(
        {
            "tornado_prob": (("time", "lat", "lon"), np.array([[[0.4, 0.2], [0.1, 0.0]]])),
            "any_prob": (("time", "lat", "lon"), np.array([[[0.7, 0.6], [0.5, 0.4]]])),
            "outbreak_risk": (("time", "lat", "lon"), np.array([[[0.6, 0.3], [0.2, 0.1]]])),
            "confidence_score": (("time", "lat", "lon"), np.array([[[0.65, 0.6], [0.55, 0.5]]])),
            "bust_risk_score": (("time", "lat", "lon"), np.array([[[0.3, 0.35], [0.4, 0.45]]])),
        },
        coords={"time": pd.to_datetime(["2026-04-09T00:00:00"]), "lat": [34.0, 36.0], "lon": [-99.0, -97.0]},
    )
    outputs = render_case_review_boards(
        prediction,
        _mismatched_label_dataset(),
        _verification_payload(),
        "2026-04-09",
        "00",
        settings,
        paths,
    )
    assert len(outputs) == 1
    assert outputs[0].exists()
    assert outputs[0].stat().st_size > 0


def test_review_zone_bounds_do_not_overlap(tmp_path) -> None:
    settings = load_settings()
    settings.raw["paths"]["root"] = str(tmp_path)
    settings.raw["render"]["cartopy"] = False
    fig, axes = create_board_figure(settings, nrows=3, ncols=2, figsize=(14.5, 15.5))
    observed_bounds, summary_bounds = _review_zone_bounds(axes[5])
    observed_right = observed_bounds[0] + observed_bounds[2]
    summary_left = summary_bounds[0]
    assert observed_right < summary_left
    assert observed_bounds[1] == summary_bounds[1]
    assert observed_bounds[3] == summary_bounds[3]
    anchor = axes[5].get_position()
    usable_width = anchor.width * (1 - 2 * REVIEW_OUTER_PAD_X - REVIEW_GAP)
    expected_left_width = usable_width * REVIEW_LEFT_WIDTH
    expected_summary_width = usable_width - expected_left_width
    assert round(observed_bounds[2], 6) == round(expected_left_width, 6)
    assert round(summary_bounds[2], 6) == round(expected_summary_width, 6)
    assert round(observed_bounds[0], 6) == round(anchor.x0 + anchor.width * REVIEW_OUTER_PAD_X, 6)
    assert round(observed_bounds[1], 6) == round(anchor.y0 + anchor.height * REVIEW_OUTER_PAD_Y, 6)
    plt.close(fig)


def test_summary_specs_use_fixed_short_labels() -> None:
    settings = load_settings()
    badges, cards = _summary_specs(_verification_payload()["per_day"][0], _verification_payload(), settings)
    assert len(badges) == 3
    assert [badge[2] for badge in badges][1:] == ["Conf Moderate", "Train Mixed"]
    assert list(cards) == ["Outcome", "Forecast", "Assessment", "Verification"]
    assert [label for label, _, _ in cards["Outcome"]] == ["Valid", "Lead", "Obs"]
    assert [label for label, _, _ in cards["Forecast"]] == ["Tor Max", "Outbreak Max", "Any Max"]
    assert [label for label, _, _ in cards["Assessment"]] == ["Conf", "Bust", "Train"]
    assert [label for label, _, _ in cards["Verification"]] == ["Tor Brier", "Any Brier", "Eval"]


def test_summary_card_bounds_do_not_overlap() -> None:
    rects = [bounds for _, bounds in SUMMARY_CARD_LAYOUT]
    for index, (x0, y0, width, height) in enumerate(rects):
        for other_x0, other_y0, other_width, other_height in rects[index + 1 :]:
            horizontal_overlap = (x0 < other_x0 + other_width) and (other_x0 < x0 + width)
            vertical_overlap = (y0 < other_y0 + other_height) and (other_y0 < y0 + height)
            assert not (horizontal_overlap and vertical_overlap)
