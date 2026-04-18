"""Verification case-review board rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import xarray as xr

from severewx.config import AppSettings
from severewx.render.colors import (
    badge_palette,
    observed_outcome_colormap,
    observed_outcome_labels,
    observed_outcome_norm,
    product_style,
)
from severewx.render.layout import (
    CARD_EDGE_COLOR,
    CARD_FACE_COLOR,
    CARD_INSET_FACE,
    SUBTITLE_COLOR,
    TEXT_COLOR,
    cartopy_available,
    create_board_figure,
    map_draw_kwargs,
    panel_subtitle,
    panel_title,
    review_output_path,
    style_map_axes,
)
from severewx.render.maps import _daily_groups, _draw_panel, _placeholder_panel
from severewx.utils.paths import DataPaths


REVIEW_FORECAST_PRODUCTS = (
    "tornado",
    "outbreak_risk",
    "any_severe",
    "confidence",
)

REVIEW_OUTER_PAD_X = 0.02
REVIEW_OUTER_PAD_Y = 0.06
REVIEW_GAP = 0.045
REVIEW_LEFT_WIDTH = 0.35
SUMMARY_CARD_LAYOUT = (
    ("Outcome", (0.05, 0.42, 0.41, 0.18)),
    ("Forecast", (0.52, 0.42, 0.43, 0.18)),
    ("Assessment", (0.05, 0.18, 0.41, 0.18)),
    ("Verification", (0.52, 0.18, 0.43, 0.18)),
)


def _daily_prediction_views(dataset: xr.Dataset) -> list[tuple[str, int, xr.Dataset]]:
    return _daily_groups(dataset)


def _aggregate_for_review(day_dataset: xr.Dataset, style_key: str) -> xr.DataArray | None:
    style = product_style(style_key)
    field = day_dataset.get(style.dataset_var)
    if field is None:
        return None
    if "time" not in field.dims:
        return field
    if style.aggregate == "mean":
        return field.mean("time")
    return field.max("time")


def _label_slice(label_dataset: xr.Dataset | None, valid_date: str, fallback_like: xr.DataArray) -> xr.Dataset | None:
    if label_dataset is None or "date" not in label_dataset.coords:
        return None
    if np.datetime64(valid_date) not in label_dataset["date"].values:
        return None
    return label_dataset.sel(date=np.datetime64(valid_date))


def _observed_outcome_panel(label_slice: xr.Dataset | None, template_field: xr.DataArray) -> xr.DataArray | None:
    if label_slice is None:
        return None
    observed = xr.zeros_like(template_field).astype("float32")
    if "wind" in label_slice:
        observed = xr.where(label_slice["wind"] > 0, 1.0, observed)
    if "hail" in label_slice:
        observed = xr.where(label_slice["hail"] > 0, 2.0, observed)
    if "tornado" in label_slice:
        observed = xr.where(label_slice["tornado"] > 0, 3.0, observed)
    if "any" in label_slice:
        observed = xr.where((label_slice["any"] > 0) & (observed == 0), 1.0, observed)
    return observed


def _tier(value: float, moderate: float, high: float) -> str:
    if np.isnan(value):
        return "unknown"
    if value >= high:
        return "high"
    if value >= moderate:
        return "moderate"
    return "low"


def _score_value(value: Any, default: str = "N/A", fmt: str = "{:.2f}") -> str:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(numeric):
        return default
    return fmt.format(numeric)


def _truncate_text(value: Any, max_chars: int) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _badge(ax, x: float, y: float, label: str, kind: str, tier: str, large: bool = False) -> None:
    face, text = badge_palette(kind, tier)
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.1 if large else 6.8,
        fontweight="bold",
        color=text,
        bbox={
            "boxstyle": "round,pad=0.24,rounding_size=0.10",
            "facecolor": face,
            "edgecolor": face,
        },
        clip_on=True,
    )


def _outcome_badge_label(value: str) -> str:
    normalized = value.replace("_", " ").strip().title()
    replacements = {
        "Tornado Outbreak Day": "Tornado Outbreak",
        "Significant Tornado Outbreak Day": "Sig Tor Outbreak",
        "Wind Mcs Outbreak Day": "Wind/MCS Outbreak",
        "Hail Outbreak Day": "Hail Outbreak",
        "Null Day": "Null Day",
        "Unknown": "Unknown",
    }
    return replacements.get(normalized, normalized[:20].rstrip())


def _training_badge_label(value: str) -> str:
    labels = {
        "real-heavy": "Train Real",
        "mixed": "Train Mixed",
        "synthetic-heavy": "Train Synth",
        "unknown": "Train Unknown",
    }
    return labels.get(value, f"Train {value.replace('-', ' ').title()}")


def _observed_placeholder(ax) -> None:
    _style_review_subpanel(ax, "Observed Outcome")
    placeholder = FancyBboxPatch(
        (0.10, 0.37),
        0.80,
        0.22,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=ax.transAxes,
        linewidth=0.9,
        edgecolor=CARD_EDGE_COLOR,
        facecolor=CARD_INSET_FACE,
    )
    ax.add_patch(placeholder)
    ax.text(
        0.5,
        0.49,
        "Observed labels\nunavailable",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=TEXT_COLOR,
        clip_on=True,
    )
    ax.text(
        0.5,
        0.32,
        "Fallback evaluation view",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.3,
        color=SUBTITLE_COLOR,
        clip_on=True,
    )


def _style_review_subpanel(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_axis_off()
    ax.set_facecolor(CARD_FACE_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(CARD_EDGE_COLOR)
        spine.set_linewidth(0.95)
    ax.patch.set_edgecolor(CARD_EDGE_COLOR)
    ax.patch.set_linewidth(0.95)
    ax.text(
        0.5,
        0.95,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.7,
        fontweight="bold",
        color=TEXT_COLOR,
        clip_on=True,
    )
    if subtitle:
        ax.text(
            0.5,
            0.89,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.9,
            color=SUBTITLE_COLOR,
            clip_on=True,
        )
    divider_y = 0.855 if subtitle else 0.89
    ax.plot([0.05, 0.95], [divider_y, divider_y], transform=ax.transAxes, color="#dde3ea", linewidth=0.8, clip_on=False)


def _review_zone_bounds(anchor_ax) -> tuple[list[float], list[float]]:
    bounds = anchor_ax.get_position()
    x0, y0, width, height = bounds.x0, bounds.y0, bounds.width, bounds.height
    outer_x = width * REVIEW_OUTER_PAD_X
    outer_y = height * REVIEW_OUTER_PAD_Y
    gap = width * REVIEW_GAP
    usable_width = width - 2 * outer_x - gap
    observed_width = usable_width * REVIEW_LEFT_WIDTH
    summary_width = usable_width - observed_width
    card_height = height - 2 * outer_y
    observed_bounds = [x0 + outer_x, y0 + outer_y, observed_width, card_height]
    summary_bounds = [observed_bounds[0] + observed_width + gap, y0 + outer_y, summary_width, card_height]
    return observed_bounds, summary_bounds


def _review_zone_axes(fig, anchor_ax, use_geo_projection: bool, projection) -> tuple[Any, Any]:
    observed_bounds, summary_bounds = _review_zone_bounds(anchor_ax)
    observed_ax = fig.add_axes(observed_bounds)
    summary_ax = fig.add_axes(summary_bounds)
    for axis in (observed_ax, summary_ax):
        axis.set_facecolor(CARD_FACE_COLOR)
    return observed_ax, summary_ax


def _draw_observed_panel(
    fig,
    card_ax,
    lat_values,
    lon_values,
    observed_field: xr.DataArray | None,
    init_date: str,
    cycle: str,
    valid_date: str,
    lead_day: int,
    settings: AppSettings,
    projection,
) -> None:
    if observed_field is None:
        _observed_placeholder(card_ax)
        return

    _style_review_subpanel(card_ax, "Observed Outcome")
    bbox = card_ax.get_position()
    inset_bounds = [bbox.x0 + 0.09 * bbox.width, bbox.y0 + 0.23 * bbox.height, 0.82 * bbox.width, 0.42 * bbox.height]
    add_axes_kwargs: dict[str, Any] = {}
    if projection is not None:
        add_axes_kwargs["projection"] = projection
    map_ax = fig.add_axes(inset_bounds, **add_axes_kwargs)
    map_ax.set_facecolor("white")
    mesh = map_ax.pcolormesh(
        lon_values,
        lat_values,
        np.asarray(observed_field.values, dtype=float),
        shading="auto",
        cmap=observed_outcome_colormap(),
        norm=observed_outcome_norm(),
        **map_draw_kwargs(map_ax),
    )
    style_map_axes(map_ax, lon_values, lat_values, settings=settings)
    colorbar = fig.colorbar(mesh, ax=map_ax, orientation="horizontal", fraction=0.08, pad=0.11, ticks=[0.25, 1.0, 2.0, 3.0])
    colorbar.ax.set_xticklabels(observed_outcome_labels(), fontsize=5.8)
    colorbar.set_label("Observed footprint", fontsize=6.2)
    colorbar.outline.set_linewidth(0.7)


def _review_template_field(day_dataset: xr.Dataset) -> xr.DataArray | None:
    for product_key in ("tornado", "outbreak_risk", "any_severe", "confidence", "bust_risk"):
        field = _aggregate_for_review(day_dataset, product_key)
        if field is not None:
            return field
    return None


def _score_row(ax, x_label: float, x_value: float, y: float, label: str, value: str, emphasis: bool = False) -> None:
    ax.text(x_label, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color=SUBTITLE_COLOR, clip_on=True)
    ax.text(
        x_value,
        y,
        value,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=7.3 if emphasis else 7.0,
        color=TEXT_COLOR,
        fontweight="bold" if emphasis else "normal",
        clip_on=True,
    )


def _summary_specs(day_row: dict[str, Any], verification_payload: dict[str, Any], settings: AppSettings) -> tuple[list[tuple[float, float, str, str, str]], dict[str, list[tuple[str, str, bool]]]]:
    run_summary = verification_payload.get("run_summary", {})
    confidence_thresholds = settings.get("confidence.thresholds", {}) or {}
    bust_thresholds = settings.get("bustrisk.thresholds", {}) or {}
    confidence_tier = _tier(
        float(day_row.get("forecast_confidence", np.nan)),
        float(confidence_thresholds.get("moderate", 0.4)),
        float(confidence_thresholds.get("high", 0.7)),
    )
    bust_tier = _tier(
        float(day_row.get("forecast_bust_risk", np.nan)),
        float(bust_thresholds.get("moderate", 0.4)),
        float(bust_thresholds.get("high", 0.7)),
    )
    training_tier = str(run_summary.get("training_quality_tier", "unknown"))
    observed_outcome = str(day_row.get("observed_category", "unknown"))
    observed_summary = {
        "Tornado Outbreak": "Tor Outbreak",
        "Significant Tornado Outbreak Day": "Sig Tor",
        "Sig Tor Outbreak": "Sig Tor",
        "Wind/MCS Outbreak": "Wind/MCS",
        "Hail Outbreak": "Hail Outbreak",
        "Null Day": "Null",
        "Unknown": "Unknown",
    }.get(_outcome_badge_label(observed_outcome), _truncate_text(_outcome_badge_label(observed_outcome), 11))
    badges = [
        (0.05, 0.77, observed_summary, "outcome", observed_outcome.replace("_", "-")),
        (0.05, 0.70, f"Conf {confidence_tier.title()}", "confidence", confidence_tier),
        (0.56, 0.70, _training_badge_label(training_tier), "training", training_tier),
    ]
    cards = {
        "Outcome": [
            ("Valid", _truncate_text(day_row.get("valid_date", "unknown"), 10), True),
            ("Lead", _truncate_text(day_row.get("lead_day", "unknown"), 6), False),
            ("Obs", observed_summary, False),
        ],
        "Forecast": [
            ("Tor Max", _score_value(day_row.get("forecast_tornado_prob"), fmt="{:.2f}"), True),
            ("Outbreak Max", _score_value(day_row.get("forecast_outbreak_prob"), fmt="{:.2f}"), True),
            ("Any Max", _score_value(day_row.get("forecast_any_prob"), fmt="{:.2f}"), False),
        ],
        "Assessment": [
            ("Conf", _truncate_text(f"{_score_value(day_row.get('forecast_confidence'), fmt='{:.2f}')} {confidence_tier.title()}", 11), False),
            ("Bust", _truncate_text(f"{_score_value(day_row.get('forecast_bust_risk'), fmt='{:.2f}')} {bust_tier.title()}", 11), False),
            ("Train", _truncate_text(training_tier.replace("-", " ").title(), 11), False),
        ],
        "Verification": [
            ("Tor Brier", _score_value(day_row.get("tornado_brier"), fmt="{:.3f}"), False),
            ("Any Brier", _score_value(day_row.get("any_brier"), fmt="{:.3f}"), False),
            ("Eval", _truncate_text(run_summary.get("evaluation_source", "unknown"), 11), False),
        ],
    }
    return badges, cards


def _section_box(ax, x0: float, y0: float, width: float, height: float, title: str, rows: list[tuple[str, str, bool]]) -> None:
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        linewidth=0.85,
        edgecolor="#d8dee6",
        facecolor=CARD_INSET_FACE,
    )
    ax.add_patch(patch)
    ax.text(x0 + 0.025, y0 + height - 0.032, title.upper(), transform=ax.transAxes, ha="left", va="center", fontsize=5.8, fontweight="bold", color="#5f6872", clip_on=True)
    row_y = y0 + height - 0.074
    row_step = 0.038
    for label, value, emphasis in rows:
        _score_row(ax, x0 + 0.025, x0 + width - 0.025, row_y, label, _truncate_text(value, 11), emphasis=emphasis)
        row_y -= row_step


def _draw_summary_panel(ax, day_row: dict[str, Any] | None, verification_payload: dict[str, Any], settings: AppSettings) -> None:
    day_row = day_row or {}
    _style_review_subpanel(ax, "Case Review Summary")
    badges, cards = _summary_specs(day_row, verification_payload, settings)
    for x, y, label, kind, tier in badges:
        _badge(ax, x, y, _truncate_text(label, 13), kind, tier, large=False)
    for title, bounds in SUMMARY_CARD_LAYOUT:
        x0, y0, width, height = bounds
        _section_box(ax, x0, y0, width, height, title, cards[title])


def render_case_review_boards(
    prediction_dataset: xr.Dataset,
    label_dataset: xr.Dataset | None,
    verification_payload: dict[str, Any],
    init_date: str,
    cycle: str,
    settings: AppSettings,
    paths: DataPaths,
) -> list[Path]:
    """Render one compact case-review board per valid day."""

    outputs: list[Path] = []
    lat_values = prediction_dataset["lat"].values
    lon_values = prediction_dataset["lon"].values
    per_day_lookup = {
        (str(row.get("valid_date")), int(row.get("lead_day", 0))): row
        for row in verification_payload.get("per_day", [])
    }

    for valid_date, lead_day, day_dataset in _daily_prediction_views(prediction_dataset):
        fig, axes_list = create_board_figure(settings, nrows=3, ncols=2, figsize=(14.5, 15.5))

        for axis, product_key in zip(axes_list[:4], REVIEW_FORECAST_PRODUCTS, strict=False):
            style = product_style(product_key)
            field = _aggregate_for_review(day_dataset, product_key)
            _draw_panel(
                fig,
                axis,
                lat_values,
                lon_values,
                field,
                style,
                panel_title(style, valid_date, lead_day),
                panel_subtitle(init_date, cycle, valid_date),
                compact=True,
                settings=settings,
                emphasis=product_key in {"tornado", "outbreak_risk"},
            )

        style = product_style("bust_risk")
        bust_field = _aggregate_for_review(day_dataset, "bust_risk")
        _draw_panel(
            fig,
            axes_list[4],
            lat_values,
            lon_values,
            bust_field,
            style,
            panel_title(style, valid_date, lead_day),
            panel_subtitle(init_date, cycle, valid_date),
            compact=True,
            settings=settings,
            emphasis=False,
        )

        template_field = _review_template_field(day_dataset)
        observed_field = (
            _observed_outcome_panel(_label_slice(label_dataset, valid_date, template_field), template_field)
            if template_field is not None
            else None
        )

        use_geo_projection = cartopy_available() and axes_list[0].__class__.__module__.startswith("cartopy")
        projection = axes_list[0].projection if use_geo_projection else None
        observed_card_ax, summary_ax = _review_zone_axes(fig, axes_list[5], use_geo_projection, projection)
        axes_list[5].set_visible(False)

        _draw_observed_panel(
            fig,
            observed_card_ax,
            lat_values,
            lon_values,
            observed_field,
            init_date,
            cycle,
            valid_date,
            lead_day,
            settings,
            projection,
        )

        day_row = per_day_lookup.get((valid_date, lead_day))
        _draw_summary_panel(summary_ax, day_row, verification_payload, settings)

        fig.suptitle(
            f"severewx Case Review | Init {init_date} {cycle}Z | Valid {valid_date} | Day {lead_day}",
            x=0.05,
            y=0.975,
            ha="left",
            fontsize=16,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        fig.text(
            0.05,
            0.952,
            "Tornado and outbreak panels are emphasized; supporting panels provide confidence, bust context, and observed outcome review.",
            ha="left",
            va="top",
            fontsize=9.3,
            color=SUBTITLE_COLOR,
        )
        output_path = review_output_path(paths, init_date, cycle, valid_date, lead_day)
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        outputs.append(output_path)
    return outputs
