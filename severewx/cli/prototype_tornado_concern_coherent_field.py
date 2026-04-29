"""Prototype spatially coherent tornado-concern fields from saved artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap, Normalize
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
from severewx.models.tornado_concern import coherent_tornado_concern_grid, envelope_tornado_concern_grid
from severewx.render.layout import SUBTITLE_COLOR, TEXT_COLOR, create_board_figure, map_draw_kwargs, style_map_axes
from severewx.utils.paths import build_paths


HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "tornado_concern_coherent_red",
    ["#fff7f3", "#fde0dd", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15", "#67000d"],
)
CONTOUR_LEVELS = (0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60)
OUTLOOK_LEVELS = (0.00, 0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, 1.00)
OUTLOOK_COLORS = ("#ffffff00", "#77be7a", "#c7a994", "#ffea7a", "#ff7078", "#bf83d8", "#94008b", "#354d86")


def _normalize_cycle(value: str) -> str:
    return str(value).zfill(2)


def _valid_date_strings(dataset: xr.Dataset) -> list[str]:
    return [timestamp.date().isoformat() for timestamp in pd.to_datetime(dataset["time"].values)] if "time" in dataset.coords else []


def _select_valid_date(dataset: xr.Dataset, mode: str, explicit_valid_date: str | None) -> str:
    if explicit_valid_date:
        return explicit_valid_date
    if mode == "best":
        return _best_valid_date(dataset, "tornado_concern_prob")
    dates = _valid_date_strings(dataset)
    return dates[0] if dates else ""


def _field_values(dataset: xr.Dataset, field_name: str, valid_date: str) -> np.ndarray | None:
    if field_name not in dataset:
        return None
    selected = _select_valid_date_dataset(dataset, valid_date) if valid_date else dataset
    field = selected[field_name]
    field = field.max("time") if "time" in field.dims else field
    return np.asarray(field.values, dtype=float)


def _draw_heatmap_panel(
    ax: Any,
    *,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    values: np.ndarray,
    settings: Any,
    title: str,
    subtitle: str,
    contours: bool,
) -> Any:
    draw_kwargs = map_draw_kwargs(ax)
    norm = Normalize(vmin=0.0, vmax=0.60, clip=True)
    display_lon, display_lat, display_values = _upsample_grid_for_display(lon_values, lat_values, values, factor=6)
    masked = np.ma.masked_where(~np.isfinite(display_values) | (display_values <= 0.0), display_values)
    artist = ax.pcolormesh(display_lon, display_lat, masked, shading="auto", cmap=HEATMAP_CMAP, norm=norm, alpha=0.76, **draw_kwargs)
    if contours and np.isfinite(display_values).any():
        levels = [level for level in CONTOUR_LEVELS if float(np.nanmax(display_values)) >= level]
        if levels:
            contour_set = ax.contour(display_lon, display_lat, display_values, levels=levels, colors="#8b1010", linewidths=0.85, alpha=0.85, **draw_kwargs)
            labels = ax.clabel(
                contour_set,
                contour_set.levels,
                inline=True,
                inline_spacing=4,
                fontsize=7.4,
                fmt={level: f"{int(round(level * 100))}%" for level in contour_set.levels},
                colors="#4d0707",
            )
            for label in labels:
                label.set_path_effects([path_effects.withStroke(linewidth=2.3, foreground="white")])
    style_map_axes(ax, lon_values, lat_values, settings=settings)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=TEXT_COLOR, pad=13)
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.1, color=SUBTITLE_COLOR)
    return artist


def _draw_outlook_panel(
    ax: Any,
    *,
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    values: np.ndarray,
    settings: Any,
    title: str,
    subtitle: str,
) -> Any:
    draw_kwargs = map_draw_kwargs(ax)
    display_lon, display_lat, display_values = _upsample_grid_for_display(lon_values, lat_values, values, factor=6)
    masked = np.ma.masked_where(~np.isfinite(display_values) | (display_values < 0.02), display_values)
    cmap = ListedColormap(list(OUTLOOK_COLORS), name="tornado_concern_outlook_style")
    norm = BoundaryNorm(OUTLOOK_LEVELS, len(OUTLOOK_COLORS), clip=True)
    artist = ax.contourf(display_lon, display_lat, masked, levels=OUTLOOK_LEVELS, cmap=cmap, norm=norm, alpha=0.82, **draw_kwargs)
    levels = [level for level in CONTOUR_LEVELS if np.isfinite(display_values).any() and float(np.nanmax(display_values)) >= level]
    if levels:
        contour_set = ax.contour(display_lon, display_lat, display_values, levels=levels, colors="#461010", linewidths=0.75, alpha=0.75, **draw_kwargs)
        labels = ax.clabel(
            contour_set,
            contour_set.levels,
            inline=True,
            inline_spacing=4,
            fontsize=7.4,
            fmt={level: f"{int(round(level * 100))}%" for level in contour_set.levels},
            colors="#2d0505",
        )
        for label in labels:
            label.set_path_effects([path_effects.withStroke(linewidth=2.3, foreground="white")])
    style_map_axes(ax, lon_values, lat_values, settings=settings)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=TEXT_COLOR, pad=13)
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.1, color=SUBTITLE_COLOR)
    return artist


def build_coherent_board(
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
        raw = _field_values(dataset, "tornado_concern_prob", selected_valid_date)
        if raw is None:
            raise KeyError("forecast artifact is missing tornado_concern_prob")
        support = _field_values(dataset, "sig_tor_support", selected_valid_date)
        overlap = _field_values(dataset, "tornado_favored_overlap", selected_valid_date)
        scp = _field_values(dataset, "scp_proxy", selected_valid_date)
        outbreak = _field_values(dataset, "outbreak_risk", selected_valid_date)
        coherent = coherent_tornado_concern_grid(
            raw,
            sig_tor_support=support,
            tornado_favored_overlap=overlap,
            scp_proxy=scp,
            outbreak_risk=outbreak,
        )
        envelope = envelope_tornado_concern_grid(
            raw,
            sig_tor_support=support,
            tornado_favored_overlap=overlap,
            scp_proxy=scp,
            outbreak_risk=outbreak,
        )
        support_display = np.maximum.reduce([arr for arr in [support, overlap, scp, outbreak] if arr is not None])
        support_display = np.clip(np.nan_to_num(support_display, nan=0.0), 0.0, 1.0)
        lon_values = np.asarray(dataset["lon"].values, dtype=float)
        lat_values = np.asarray(dataset["lat"].values, dtype=float)

    fig, axes = create_board_figure(settings, nrows=1, ncols=4, figsize=(20.0, 5.7))
    fig.suptitle(
        f"Tornado Concern Coherent-Field Prototype | Init {date} {cycle}Z | Valid {selected_valid_date}",
        x=0.03,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    artist = _draw_heatmap_panel(
        axes[0],
        lon_values=lon_values,
        lat_values=lat_values,
        values=raw,
        settings=settings,
        title="Raw Concern",
        subtitle="Current saved tornado_concern_prob",
        contours=False,
    )
    _draw_heatmap_panel(
        axes[1],
        lon_values=lon_values,
        lat_values=lat_values,
        values=support_display,
        settings=settings,
        title="Support Envelope",
        subtitle="Tornado-support features used as spread constraint",
        contours=False,
    )
    _draw_heatmap_panel(
        axes[2],
        lon_values=lon_values,
        lat_values=lat_values,
        values=coherent,
        settings=settings,
        title="Coherent Field",
        subtitle="Raw seeds spread only into nearby support",
        contours=False,
    )
    _draw_outlook_panel(
        axes[3],
        lon_values=lon_values,
        lat_values=lat_values,
        values=envelope,
        settings=settings,
        title="Outlook-Style Envelope",
        subtitle="Broad support footprint with nested probability tiers",
    )
    colorbar = fig.colorbar(artist, ax=axes, orientation="horizontal", fraction=0.075, pad=0.08)
    ticks = [0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60]
    colorbar.set_ticks(ticks)
    colorbar.ax.set_xticklabels([f"{int(value * 100)}%" for value in ticks], fontsize=8)
    colorbar.set_label("Tornado concern probability / normalized support", fontsize=9)
    outdir.mkdir(parents=True, exist_ok=True)
    board_path = outdir / f"tornado_concern_coherent_init_{date}_{cycle}z_valid_{selected_valid_date}.png"
    fig.savefig(board_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return {
        "init_date": date,
        "cycle": cycle,
        "valid_date": selected_valid_date,
        "source_forecast_artifact": str(prediction_path.resolve()),
        "board_path": str(board_path.resolve()),
        "raw_max": float(np.nanmax(raw)),
        "coherent_max": float(np.nanmax(coherent)),
        "envelope_max": float(np.nanmax(envelope)),
        "raw_cells_ge_02pct": int(np.count_nonzero(raw >= 0.02)),
        "coherent_cells_ge_02pct": int(np.count_nonzero(coherent >= 0.02)),
        "envelope_cells_ge_02pct": int(np.count_nonzero(envelope >= 0.02)),
        "raw_cells_ge_05pct": int(np.count_nonzero(raw >= 0.05)),
        "coherent_cells_ge_05pct": int(np.count_nonzero(coherent >= 0.05)),
        "envelope_cells_ge_05pct": int(np.count_nonzero(envelope >= 0.05)),
        "raw_cells_ge_10pct": int(np.count_nonzero(raw >= 0.10)),
        "coherent_cells_ge_10pct": int(np.count_nonzero(coherent >= 0.10)),
        "envelope_cells_ge_10pct": int(np.count_nonzero(envelope >= 0.10)),
    }


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Tornado Concern Coherent Field Prototype",
        "",
        "Prototype only. This does not replace the baseline model or saved forecast artifacts.",
        "",
        "| init_date | valid_date | raw_ge_2pct | coherent_ge_2pct | envelope_ge_2pct | raw_ge_5pct | envelope_ge_5pct | board |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {init_date} | {valid_date} | {raw_cells_ge_02pct} | {coherent_cells_ge_02pct} | {envelope_cells_ge_02pct} | {raw_cells_ge_05pct} | {envelope_cells_ge_05pct} | {board_path} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--dates-file")
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--valid-date")
    parser.add_argument("--valid-date-mode", choices=["init", "best"], default="best")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.date) == bool(args.dates_file):
        parser.error("provide exactly one of --date or --dates-file")
    dates = _read_dates_file(Path(args.dates_file)) if args.dates_file else [str(args.date)]
    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; pass --overwrite: {outdir}")
    settings = load_settings()
    paths = build_paths(settings)
    cycle = _normalize_cycle(args.cycle)
    rows = [
        build_coherent_board(
            date=date,
            cycle=cycle,
            valid_date=args.valid_date if len(dates) == 1 else None,
            valid_date_mode=args.valid_date_mode,
            outdir=outdir,
            settings=settings,
            paths=paths,
        )
        for date in dates
    ]
    frame = pd.DataFrame(rows)
    csv_path = outdir / "coherent_field_manifest.csv"
    md_path = outdir / "README.md"
    frame.to_csv(csv_path, index=False)
    _write_summary(md_path, rows)
    print(f"boards_total={len(frame)}")
    print(f"manifest_csv={csv_path}")
    print(f"summary_md={md_path}")


if __name__ == "__main__":
    main()
