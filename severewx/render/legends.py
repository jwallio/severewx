"""Legend helpers for public forecast graphics."""

from __future__ import annotations

from matplotlib.figure import Figure

from severewx.render.colors import ProductStyle, legend_tick_labels, legend_tick_values


def add_product_colorbar(fig: Figure, ax, mappable, style: ProductStyle, compact: bool = False) -> None:
    """Attach a restrained horizontal legend to a single panel."""

    fraction = 0.055 if compact else 0.08
    pad = 0.05 if compact else 0.07
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="horizontal",
        fraction=fraction,
        pad=pad,
        ticks=legend_tick_values(style.key),
    )
    tick_values = legend_tick_labels(style.key)
    if compact and len(tick_values) > 5:
        display_labels = [label if index in {0, 2, 4, len(tick_values) - 1} else "" for index, label in enumerate(tick_values)]
    else:
        display_labels = tick_values
    colorbar.ax.set_xticklabels(display_labels, fontsize=7 if compact else 8)
    colorbar.set_label(style.legend_label, fontsize=8 if compact else 9)
    colorbar.outline.set_linewidth(0.7)
