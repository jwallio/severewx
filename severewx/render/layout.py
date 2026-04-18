"""Figure layout helpers for public forecast graphics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from severewx.config import AppSettings
from severewx.render.colors import ProductStyle
from severewx.utils.paths import DataPaths


try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover - exercised through helper tests instead
    ccrs = None
    cfeature = None


MAP_FACE_COLOR = "#f7f7f5"
TEXT_COLOR = "#202224"
SUBTITLE_COLOR = "#434b56"
GRID_COLOR = "#d9dee4"
GEOGRAPHY_COLOR = "#8a9198"
COASTLINE_COLOR = "#6b727a"
CONUS_EXTENT = (-125.0, -66.5, 24.0, 50.0)
CARD_FACE_COLOR = "#f6f8fa"
CARD_EDGE_COLOR = "#d7dde4"
CARD_INSET_FACE = "#eef2f6"


@dataclass(frozen=True, slots=True)
class MapTemplate:
    """Reusable fixed-domain map configuration."""

    extent: tuple[float, float, float, float]
    use_cartopy: bool
    projection_name: str
    show_gridlines: bool


def cartopy_available() -> bool:
    return bool(ccrs is not None and cfeature is not None)


def conus_template(settings: AppSettings) -> MapTemplate:
    requested_cartopy = bool(settings.get("render.cartopy", True))
    use_cartopy = requested_cartopy and cartopy_available()
    return MapTemplate(
        extent=tuple(float(value) for value in settings.get("render.conus_extent", list(CONUS_EXTENT))),
        use_cartopy=use_cartopy,
        projection_name="PlateCarree" if use_cartopy else "matplotlib",
        show_gridlines=bool(settings.get("render.show_gridlines", False)),
    )


def _projection_kwargs(settings: AppSettings) -> dict[str, Any]:
    template = conus_template(settings)
    if template.use_cartopy:
        return {"subplot_kw": {"projection": ccrs.PlateCarree()}}
    return {}


def _create_panel_grid(settings: AppSettings, nrows: int, ncols: int, figsize: tuple[float, float], margins: dict[str, float]):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=int(settings.get("render.dpi", 130)),
        **_projection_kwargs(settings),
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(**margins)
    axes_array = np.atleast_1d(axes).ravel().tolist()
    for axis in axes_array:
        axis.set_facecolor(MAP_FACE_COLOR)
    return fig, axes_array


def create_map_figure(settings: AppSettings):
    width, height = settings.get("render.figure_size", [11, 7])
    fig, axes = _create_panel_grid(
        settings,
        1,
        1,
        (float(width), float(height)),
        {"left": 0.04, "right": 0.985, "top": 0.89, "bottom": 0.08},
    )
    return fig, axes[0]


def create_board_figure(settings: AppSettings, nrows: int = 4, ncols: int = 2, figsize: tuple[float, float] = (15.5, 17.0)):
    return _create_panel_grid(
        settings,
        nrows,
        ncols,
        figsize,
        {"left": 0.03, "right": 0.985, "top": 0.93, "bottom": 0.04, "hspace": 0.26, "wspace": 0.08},
    )


def _is_geoaxes(ax) -> bool:
    return cartopy_available() and ax.__class__.__module__.startswith("cartopy")


def map_draw_kwargs(ax) -> dict[str, Any]:
    if _is_geoaxes(ax):
        return {"transform": ccrs.PlateCarree()}
    return {}


def style_map_axes(ax, lon_values, lat_values, settings: AppSettings | None = None) -> None:
    settings = settings or AppSettings(raw={})
    template = conus_template(settings)
    lon_min, lon_max, lat_min, lat_max = template.extent
    if _is_geoaxes(ax):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.35, edgecolor=GEOGRAPHY_COLOR, facecolor="none", alpha=0.75, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.45, edgecolor=COASTLINE_COLOR, alpha=0.9, zorder=3)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor=COASTLINE_COLOR, alpha=0.95, zorder=3)
        outline = ax.spines.get("geo")
        if outline is not None:
            outline.set_edgecolor("#d5d9de")
            outline.set_linewidth(0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        if template.show_gridlines:
            gridliner = ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=False,
                linewidth=0.25,
                color=GRID_COLOR,
                alpha=0.25,
                linestyle="--",
            )
            gridliner.xlines = True
            gridliner.ylines = True
        return

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if template.show_gridlines:
        ax.grid(color=GRID_COLOR, linewidth=0.3, linestyle="--", alpha=0.2)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#d5d9de")
        spine.set_linewidth(0.8)


def panel_title(style: ProductStyle, valid_date: str, lead_day: int) -> str:
    return f"{style.display_name} | Day {lead_day}"


def panel_subtitle(init_date: str, cycle: str, valid_date: str) -> str:
    return f"Init {init_date} {cycle}Z | Valid {valid_date}"


def board_title(date: str, cycle: str, valid_date: str, lead_day: int) -> str:
    return f"severewx Daily Board | Init {date} {cycle}Z | Valid {valid_date} | Day {lead_day}"


def apply_panel_heading(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12.2, color=TEXT_COLOR, pad=16)
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.8, color=SUBTITLE_COLOR)


def emphasize_panel_frame(ax) -> None:
    outline = ax.spines.get("geo")
    if outline is not None:
        outline.set_edgecolor("#808891")
        outline.set_linewidth(1.15)
        return
    for spine in ax.spines.values():
        spine.set_color("#7a848f")
        spine.set_linewidth(1.15)


def style_info_panel(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_axis_off()
    ax.set_facecolor(CARD_FACE_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(CARD_EDGE_COLOR)
        spine.set_linewidth(0.9)
    ax.patch.set_edgecolor(CARD_EDGE_COLOR)
    ax.patch.set_linewidth(1.0)
    ax.text(0.05, 0.95, title, transform=ax.transAxes, ha="left", va="top", fontsize=12.4, fontweight="bold", color=TEXT_COLOR)
    if subtitle:
        ax.text(0.05, 0.89, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=8.9, color=SUBTITLE_COLOR)


def map_output_path(paths: DataPaths, date: str, cycle: str, valid_date: str, lead_day: int, product_key: str) -> Path:
    return paths.maps / f"{date}_{cycle}_day{lead_day}_{valid_date}_{product_key}.png"


def board_output_path(paths: DataPaths, date: str, cycle: str, valid_date: str, lead_day: int) -> Path:
    return paths.maps / f"{date}_{cycle}_day{lead_day}_{valid_date}_daily_board.png"


def review_output_path(paths: DataPaths, date: str, cycle: str, valid_date: str, lead_day: int) -> Path:
    return paths.verification / f"{date}_{cycle}_day{lead_day}_{valid_date}_case_review.png"
