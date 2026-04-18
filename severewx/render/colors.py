"""Render product styling and color ramps."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.colors import BoundaryNorm, ListedColormap


@dataclass(frozen=True, slots=True)
class ProductStyle:
    """Declarative style settings for a rendered forecast product."""

    key: str
    dataset_var: str
    display_name: str
    legend_label: str
    aggregate: str
    bins: tuple[float, ...]
    colors: tuple[str, ...]


PRODUCT_STYLES: tuple[ProductStyle, ...] = (
    ProductStyle(
        key="any_severe",
        dataset_var="any_prob",
        display_name="Any Severe",
        legend_label="Probability",
        aggregate="max",
        bins=(0.00, 0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, 1.00),
        colors=("#f4f7fb", "#d9e7f5", "#b8d2ea", "#86b4d6", "#5f95bf", "#3f6f9f", "#274f78", "#16324f"),
    ),
    ProductStyle(
        key="tornado",
        dataset_var="tornado_prob",
        display_name="Tornado",
        legend_label="Probability",
        aggregate="max",
        bins=(0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.25, 0.40, 1.00),
        colors=("#fff1f1", "#ffcccc", "#ff9f9f", "#ef6f6f", "#dc4a4a", "#bc3030", "#8f1b1b", "#5c0f0f"),
    ),
    ProductStyle(
        key="hail",
        dataset_var="hail_prob",
        display_name="Hail",
        legend_label="Probability",
        aggregate="max",
        bins=(0.00, 0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, 1.00),
        colors=("#f5fbf5", "#daf0da", "#b8dfbb", "#8dcb93", "#5eaf68", "#3e8b48", "#2d6a35", "#1a4720"),
    ),
    ProductStyle(
        key="wind",
        dataset_var="wind_prob",
        display_name="Wind",
        legend_label="Probability",
        aggregate="max",
        bins=(0.00, 0.02, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60, 1.00),
        colors=("#fff9f1", "#fee9c7", "#fdd49a", "#f9b46d", "#ec8f4d", "#cf6636", "#a24527", "#6e2a16"),
    ),
    ProductStyle(
        key="outbreak_risk",
        dataset_var="outbreak_risk",
        display_name="Outbreak Risk",
        legend_label="Risk",
        aggregate="max",
        bins=(0.00, 0.03, 0.08, 0.15, 0.25, 0.40, 0.55, 0.70, 1.00),
        colors=("#fefee8", "#f0f4c8", "#dce8a3", "#bfd67d", "#99bd57", "#719439", "#4a6823", "#2b3f14"),
    ),
    ProductStyle(
        key="confidence",
        dataset_var="confidence_score",
        display_name="Confidence",
        legend_label="Score",
        aggregate="mean",
        bins=(0.00, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00),
        colors=("#f7f8f9", "#e4e7eb", "#cad0d7", "#a6afb9", "#7d8894", "#57626e", "#2f3944"),
    ),
    ProductStyle(
        key="bust_risk",
        dataset_var="bust_risk_score",
        display_name="Bust Risk",
        legend_label="Risk",
        aggregate="mean",
        bins=(0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00),
        colors=("#f6f9fc", "#deebf5", "#bfd6e8", "#97b9d6", "#6f95be", "#4b719d", "#2d496f"),
    ),
)


def product_style(product_key: str) -> ProductStyle:
    for style in PRODUCT_STYLES:
        if style.key == product_key or style.dataset_var == product_key:
            return style
    raise KeyError(f"unknown render product {product_key}")


def ordered_product_styles() -> tuple[ProductStyle, ...]:
    return PRODUCT_STYLES


def colormap_for_product(name: str) -> ListedColormap:
    style = product_style(name)
    return ListedColormap(list(style.colors), name=style.key)


def norm_for_product(name: str) -> BoundaryNorm:
    style = product_style(name)
    return BoundaryNorm(style.bins, len(style.colors), clip=True)


def legend_tick_values(name: str) -> list[float]:
    style = product_style(name)
    return list(style.bins)


def legend_tick_labels(name: str) -> list[str]:
    values = legend_tick_values(name)
    labels: list[str] = []
    for value in values:
        labels.append(f"{int(round(value * 100.0))}%")
    return labels


def observed_outcome_colormap() -> ListedColormap:
    return ListedColormap(
        ["#f2f4f7", "#e3c36d", "#64a86b", "#c84d4d"],
        name="observed_outcomes",
    )


def observed_outcome_norm() -> BoundaryNorm:
    return BoundaryNorm((0.0, 0.5, 1.5, 2.5, 3.5), 4, clip=True)


def observed_outcome_labels() -> list[str]:
    return ["None", "Wind", "Hail", "Tornado"]


def badge_palette(kind: str, tier: str) -> tuple[str, str]:
    normalized = tier.lower().replace("_", "-")
    palette: dict[str, dict[str, tuple[str, str]]] = {
        "outcome": {
            "tornado-outbreak-day": ("#6f1111", "#ffffff"),
            "significant-tornado-outbreak-day": ("#541010", "#ffffff"),
            "hail-outbreak-day": ("#2d6a35", "#ffffff"),
            "wind-mcs-outbreak-day": ("#6e2a16", "#ffffff"),
            "null-day": ("#e9edf2", "#49515a"),
            "unknown": ("#eef1f4", "#5b6470"),
        },
        "confidence": {
            "high": ("#31414f", "#ffffff"),
            "moderate": ("#8e98a4", "#ffffff"),
            "low": ("#dbe1e7", "#49515a"),
            "unknown": ("#eef1f4", "#5b6470"),
        },
        "bust": {
            "high": ("#375d87", "#ffffff"),
            "moderate": ("#88a8c7", "#17314a"),
            "low": ("#dce8f2", "#31414f"),
            "unknown": ("#eef1f4", "#5b6470"),
        },
        "training": {
            "real-heavy": ("#365b2f", "#ffffff"),
            "mixed": ("#a47d2c", "#ffffff"),
            "synthetic-heavy": ("#8a4a34", "#ffffff"),
            "unknown": ("#eef1f4", "#5b6470"),
        },
    }
    group = palette.get(kind, {})
    return group.get(normalized, group.get("unknown", ("#eef1f4", "#5b6470")))
