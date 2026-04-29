"""Generate side-by-side tornado-concern map style prototypes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
import xarray as xr

from severewx.cli.build_tornado_concern_product import (
    _best_valid_date,
    _prediction_artifact_for_cycle,
    _read_dates_file,
    _select_valid_date_dataset,
    _upsample_grid_for_display,
)
from severewx.config import load_settings
from severewx.render.layout import SUBTITLE_COLOR, TEXT_COLOR, create_board_figure, map_draw_kwargs, style_map_axes
from severewx.utils.paths import build_paths


STYLE_NAMES = ("raw_pixels", "interpolated_heatmap", "heatmap_with_contours")
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "tornado_concern_restart_red",
    ["#fff7f3", "#fde0dd", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15", "#67000d"],
)
CONTOUR_LEVELS = (0.05, 0.10, 0.15, 0.30, 0.45, 0.60)


def _normalize_cycle(value: str) -> str:
    return str(value).zfill(2)


def _valid_date_strings(dataset: xr.Dataset) -> list[str]:
    if "time" not in dataset.coords:
        return []
    return [timestamp.date().isoformat() for timestamp in pd.to_datetime(dataset["time"].values)]


def _select_valid_date(dataset: xr.Dataset, mode: str, explicit_valid_date: str | None) -> str:
    if explicit_valid_date:
        return explicit_valid_date
    if mode == "best":
        return _best_valid_date(dataset, "tornado_concern_prob")
    dates = _valid_date_strings(dataset)
    return dates[0] if dates else ""


def _field_for_product(dataset: xr.Dataset, valid_date: str) -> xr.DataArray:
    selected = _select_valid_date_dataset(dataset, valid_date) if valid_date else dataset
    field = selected["tornado_concern_prob"]
    return field.max("time") if "time" in field.dims else field


def _style_panel(
    ax: Any,
    *,
    style_name: str,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    values: np.ndarray,
    settings: Any,
) -> Any:
    draw_kwargs = map_draw_kwargs(ax)
    norm = Normalize(vmin=0.0, vmax=0.60, clip=True)
    masked_values = np.ma.masked_where(~np.isfinite(values) | (values <= 0.0), values)
    if style_name == "raw_pixels":
        artist = ax.pcolormesh(lon_values, lat_values, masked_values, shading="auto", cmap=HEATMAP_CMAP, norm=norm, alpha=0.88, **draw_kwargs)
    else:
        display_lon, display_lat, display_values = _upsample_grid_for_display(lon_values, lat_values, values, factor=6)
        display_masked = np.ma.masked_where(~np.isfinite(display_values) | (display_values <= 0.0), display_values)
        artist = ax.pcolormesh(
            display_lon,
            display_lat,
            display_masked,
            shading="auto",
            cmap=HEATMAP_CMAP,
            norm=norm,
            alpha=0.72,
            **draw_kwargs,
        )
        if style_name == "heatmap_with_contours":
            levels = [level for level in CONTOUR_LEVELS if np.nanmax(display_values) >= level]
            if levels:
                contours = ax.contour(
                    display_lon,
                    display_lat,
                    display_values,
                    levels=levels,
                    colors="#8b1010",
                    linewidths=0.85,
                    alpha=0.85,
                    **draw_kwargs,
                )
                labels = ax.clabel(
                    contours,
                    contours.levels,
                    inline=True,
                    inline_spacing=4,
                    fontsize=7.5,
                    fmt={level: f"{int(round(level * 100))}%" for level in contours.levels},
                    colors="#4d0707",
                )
                for label in labels:
                    label.set_path_effects([path_effects.withStroke(linewidth=2.4, foreground="white")])
    style_map_axes(ax, lon_values, lat_values, settings=settings)
    return artist


def _write_summary_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Tornado Concern Map Style Restart",
        "",
        "Prototype comparison boards only. These outputs do not change scoring or operational products.",
        "",
        "| init_date | valid_date | board | raw_max | cells_ge_2pct | cells_ge_5pct |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {init_date} | {valid_date} | {board_path} | {raw_max:.4f} | {cells_ge_02pct} | {cells_ge_05pct} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_style_board(
    *,
    date: str,
    cycle: str,
    valid_date: str | None,
    valid_date_mode: str,
    outdir: Path,
    settings: Any,
    paths: Any,
) -> dict[str, Any]:
    prediction_path = _prediction_artifact_for_cycle(paths.outputs, date, cycle)
    if prediction_path is None:
        raise FileNotFoundError(f"missing tornado-concern forecast artifact for {date} {cycle}Z")
    with xr.open_dataset(prediction_path) as dataset:
        selected_valid_date = _select_valid_date(dataset, valid_date_mode, valid_date)
        field = _field_for_product(dataset, selected_valid_date)
        lon_values = np.asarray(dataset["lon"].values, dtype=float)
        lat_values = np.asarray(dataset["lat"].values, dtype=float)
        values = np.asarray(field.values, dtype=float)

    fig, axes = create_board_figure(settings, nrows=1, ncols=3, figsize=(17.5, 5.8))
    fig.suptitle(
        f"Tornado Concern Map Style Prototype | Init {date} {cycle}Z | Valid {selected_valid_date}",
        x=0.03,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    artist = None
    for ax, style_name in zip(axes, STYLE_NAMES, strict=True):
        artist = _style_panel(
            ax,
            style_name=style_name,
            lon_values=lon_values,
            lat_values=lat_values,
            values=values,
            settings=settings,
        )
        title = style_name.replace("_", " ").title()
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=TEXT_COLOR, pad=13)
        ax.text(
            0.0,
            1.01,
            "Raw field shown honestly; style only changes display rendering",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color=SUBTITLE_COLOR,
        )
    if artist is not None:
        colorbar = fig.colorbar(artist, ax=axes, orientation="horizontal", fraction=0.075, pad=0.08)
        ticks = [0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60]
        colorbar.set_ticks(ticks)
        colorbar.ax.set_xticklabels([f"{int(value * 100)}%" for value in ticks], fontsize=8)
        colorbar.set_label("Raw baseline tornado concern probability", fontsize=9)
    outdir.mkdir(parents=True, exist_ok=True)
    board_path = outdir / f"tornado_concern_style_restart_init_{date}_{cycle}z_valid_{selected_valid_date}.png"
    fig.savefig(board_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    finite = np.isfinite(values)
    return {
        "init_date": date,
        "cycle": cycle,
        "valid_date": selected_valid_date,
        "valid_date_mode": valid_date_mode,
        "source_forecast_artifact": str(prediction_path.resolve()),
        "board_path": str(board_path.resolve()),
        "raw_max": float(np.nanmax(values)) if finite.any() else float("nan"),
        "raw_mean": float(np.nanmean(values)) if finite.any() else float("nan"),
        "cells_ge_02pct": int(np.count_nonzero(values >= 0.02)) if finite.any() else 0,
        "cells_ge_05pct": int(np.count_nonzero(values >= 0.05)) if finite.any() else 0,
        "cells_ge_10pct": int(np.count_nonzero(values >= 0.10)) if finite.any() else 0,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Forecast init date YYYY-MM-DD")
    parser.add_argument("--dates-file", help="Optional newline-delimited init dates")
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--valid-date")
    parser.add_argument("--valid-date-mode", choices=["init", "best"], default="best")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if bool(args.date) == bool(args.dates_file):
        parser.error("provide exactly one of --date or --dates-file")
    dates = _read_dates_file(Path(args.dates_file)) if args.dates_file else [str(args.date)]
    output_dir = Path(args.outdir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; pass --overwrite: {output_dir}")

    settings = load_settings()
    paths = build_paths(settings)
    cycle = _normalize_cycle(args.cycle)
    rows = [
        build_style_board(
            date=date,
            cycle=cycle,
            valid_date=args.valid_date if len(dates) == 1 else None,
            valid_date_mode=args.valid_date_mode,
            outdir=output_dir,
            settings=settings,
            paths=paths,
        )
        for date in dates
    ]
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "map_style_restart_manifest.csv"
    md_path = output_dir / "README.md"
    frame.to_csv(csv_path, index=False)
    _write_summary_markdown(md_path, rows)
    print(f"boards_total={len(frame)}")
    print(f"manifest_csv={csv_path}")
    print(f"summary_md={md_path}")


if __name__ == "__main__":
    main()
