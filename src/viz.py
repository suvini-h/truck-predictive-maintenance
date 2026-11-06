"""Shared chart style and helpers for the analysis notebooks.

Palette: the validated default categorical slots in fixed order (1 blue, 2 orange, 3 aqua;
the first three validate all-pairs on the light surface) for identity, a one-hue blue ramp
for magnitude. Aqua sits below 3:1 contrast on the light surface, so charts that use it are
direct-labelled or come with a table. Chrome (grid, axes) is kept recessive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GREY_DARK, GREY_LIGHT, SURFACE = "#6f6e69", "#b5b3ab", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Colour follows the entity: the same split / group keeps its colour in every chart.
SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": AQUA}
GROUP_COLORS = {"healthy": BLUE, "repaired": ORANGE}

PCT = mpl.ticker.PercentFormatter(1.0, decimals=0)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (10, 4.2), "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.labelcolor": INK2, "axes.labelsize": 9.5, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "ytick.left": False, "lines.linewidth": 2, "lines.markersize": 4,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    })


def blue_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)


def label_end(ax, x, y, text, color=INK2, dy=0) -> None:
    """Direct label at the end of a line (text in ink, not the series colour)."""
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                fontsize=9, color=color, va="center")


def suptitle(fig, title: str, subtitle: str | None = None) -> None:
    """Figure title + subtitle at a fixed physical spacing, whatever the figure height."""
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", y=1 + 0.12 / h)
    if subtitle:
        fig.text(0.01, 1 - 0.09 / h, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)


def save(fig, name: str, figs_dir) -> Path:
    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    path = figs_dir / f"{name}.png"
    fig.savefig(path)
    return path
