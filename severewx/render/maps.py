"""Daily public map rendering and multi-panel board assembly."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from severewx.config import AppSettings
from severewx.render.colors import ProductStyle, colormap_for_product, norm_for_product, ordered_product_styles
from severewx.render.layout import (
    TEXT_COLOR,
    apply_panel_heading,
    board_output_path,
    board_title,
    create_board_figure,
    create_map_figure,
    emphasize_panel_frame,
    map_draw_kwargs,
    map_output_path,
    panel_subtitle,
    panel_title,
    style_map_axes,
)
from severewx.render.legends import add_product_colorbar
from severewx.utils.paths import DataPaths


def _daily_groups(dataset: xr.Dataset) -> list[tuple[str, int, xr.Dataset]]:
    times = pd.to_datetime(dataset["time"].values)
    valid_dates = [timestamp.date().isoformat() for timestamp in times]
    ordered_dates = list(dict.fromkeys(valid_dates))
    groups: list[tuple[str, int, xr.Dataset]] = []
    for lead_day, valid_date in enumerate(ordered_dates, start=1):
        mask = np.asarray(valid_dates) == valid_date
        groups.append((valid_date, lead_day, dataset.isel(time=mask)))
    return groups


def _aggregate_product(day_dataset: xr.Dataset, style: ProductStyle) -> xr.DataArray | None:
    if style.dataset_var not in day_dataset:
        return None
    field = day_dataset[style.dataset_var]
    if "time" not in field.dims:
        return field
    if style.aggregate == "mean":
        return field.mean("time")
    return field.max("time")


def _placeholder_panel(ax, title: str, subtitle: str, message: str) -> None:
    ax.set_axis_off()
    ax.set_facecolor("#f7f7f5")
    ax.text(0.04, 0.92, title, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.text(0.04, 0.84, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#4d5560")
    ax.text(
        0.5,
        0.48,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#5b6470",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#eef1f4", "edgecolor": "#d4dbe2"},
    )


def _draw_panel(
    fig,
    ax,
    lat_values,
    lon_values,
    field: xr.DataArray | None,
    style: ProductStyle,
    title: str,
    subtitle: str,
    compact: bool,
    settings: AppSettings,
    emphasis: bool = False,
) -> None:
    if field is None:
        _placeholder_panel(ax, title, subtitle, "Product unavailable")
        return
    values = np.asarray(field.values, dtype=float)
    if values.size == 0 or np.isnan(values).all():
        _placeholder_panel(ax, title, subtitle, "No valid data")
        return
    mesh = ax.pcolormesh(
        lon_values,
        lat_values,
        values,
        shading="auto",
        cmap=colormap_for_product(style.key),
        norm=norm_for_product(style.key),
        **map_draw_kwargs(ax),
    )
    style_map_axes(ax, lon_values, lat_values, settings=settings)
    apply_panel_heading(ax, title, subtitle)
    if emphasis:
        emphasize_panel_frame(ax)
    add_product_colorbar(fig, ax, mesh, style, compact=compact)


def _render_single_map(
    lat_values,
    lon_values,
    field: xr.DataArray | None,
    style: ProductStyle,
    date: str,
    cycle: str,
    valid_date: str,
    lead_day: int,
    settings: AppSettings,
    output_path: Path,
) -> Path:
    fig, ax = create_map_figure(settings)
    _draw_panel(
        fig,
        ax,
        lat_values,
        lon_values,
        field,
        style,
        panel_title(style, valid_date, lead_day),
        panel_subtitle(date, cycle, valid_date),
        compact=False,
        settings=settings,
    )
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def _render_daily_board(
    lat_values,
    lon_values,
    day_dataset: xr.Dataset,
    date: str,
    cycle: str,
    valid_date: str,
    lead_day: int,
    settings: AppSettings,
    output_path: Path,
) -> Path:
    fig, axes = create_board_figure(settings)
    styles = ordered_product_styles()
    for axis, style in zip(axes, styles, strict=False):
        field = _aggregate_product(day_dataset, style)
        _draw_panel(
            fig,
            axis,
            lat_values,
            lon_values,
            field,
            style,
            panel_title(style, valid_date, lead_day),
            panel_subtitle(date, cycle, valid_date),
            compact=True,
            settings=settings,
        )
    for axis in axes[len(styles) :]:
        axis.set_axis_off()
    fig.suptitle(board_title(date, cycle, valid_date, lead_day), x=0.05, y=0.975, ha="left", fontsize=16, fontweight="bold", color=TEXT_COLOR)
    fig.text(
        0.05,
        0.952,
        "Daily hazard maxima are used for severe and outbreak products; confidence and bust risk use daily means.",
        ha="left",
        va="top",
        fontsize=9,
        color="#4d5560",
    )
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def render_probability_maps(dataset: xr.Dataset, date: str, cycle: str, settings: AppSettings, paths: DataPaths) -> list[Path]:
    """Render the seven daily products and one daily board for each valid day."""

    outputs: list[Path] = []
    lat_values = dataset["lat"].values
    lon_values = dataset["lon"].values
    for valid_date, lead_day, day_dataset in _daily_groups(dataset):
        for style in ordered_product_styles():
            field = _aggregate_product(day_dataset, style)
            outputs.append(
                _render_single_map(
                    lat_values,
                    lon_values,
                    field,
                    style,
                    date,
                    cycle,
                    valid_date,
                    lead_day,
                    settings,
                    map_output_path(paths, date, cycle, valid_date, lead_day, style.key),
                )
            )
        outputs.append(
            _render_daily_board(
                lat_values,
                lon_values,
                day_dataset,
                date,
                cycle,
                valid_date,
                lead_day,
                settings,
                board_output_path(paths, date, cycle, valid_date, lead_day),
            )
        )
    return outputs
