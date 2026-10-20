"""Shared visual system for every static figure.

Palette: a validated categorical set (adjacent-pair CVD dE >= 9 on the light
surface), a single-hue blue sequential ramp and a blue <-> red diverging ramp
with a neutral grey midpoint. Three slots sit below 3:1 contrast on the
surface, so every chart that uses them is direct-labelled or has a legend.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ..config import FIGURE_DIR

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

SEGMENT_COLORS = {
    "Champions": CATEGORICAL[0],
    "Loyal": CATEGORICAL[1],
    "New / Promising": CATEGORICAL[2],
    "At Risk": CATEGORICAL[3],
    "Hibernating": CATEGORICAL[4],
}

BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
RED_RAMP = ["#fbe3e2", "#f3b5b4", "#ea8584", "#e34948", "#b8302f"]

SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#1c5cab", "#6da7ec", "#cde2fb", NEUTRAL, "#f3b5b4", "#e34948", "#b8302f"]
)


def apply_theme() -> None:
    # Inter is used when installed; otherwise matplotlib's bundled DejaVu Sans. Silence the fallback chatter.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update(
        {
            "font.family": ["Inter", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.facecolor": SURFACE,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.facecolor": SURFACE,
            "figure.dpi": 110,
            "figure.subplot.top": 0.97,
            "savefig.dpi": 200,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "text.color": INK,
            "lines.linewidth": 2,
            "hatch.color": SURFACE,
            "hatch.linewidth": 1.0,
            "lines.solid_capstyle": "round",
        }
    )


def headline(fig, title: str, subtitle: str | None = None) -> None:
    """Figure-level title block (a claim, then the evidence line).

    Stored on the figure and placed by :func:`save` above the tight bounding box
    of everything else, so it never collides with panel titles.
    """
    fig._headline = (title, subtitle)


def _place_headline(fig) -> None:
    from matplotlib.transforms import offset_copy

    title, subtitle = fig._headline
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    x0, top = bbox.x0 / w, bbox.y1 / h
    base = offset_copy(fig.transFigure, fig=fig, y=10, units="points")
    if subtitle:
        fig.text(x0, top, subtitle, ha="left", va="bottom", fontsize=9.5, color=INK_2, transform=base)
    lift = offset_copy(fig.transFigure, fig=fig, y=(16 + 14 * (subtitle.count("\n") + 1)) if subtitle else 10, units="points")
    fig.text(x0, top, title, ha="left", va="bottom", fontsize=14, fontweight="bold", color=INK, transform=lift)


def source_note(fig, text: str = "Source: UCI Online Retail (Chen, 2015), Dec 2010 - Dec 2011. Customers with a CustomerID only.") -> None:
    fig._source = text


def _place_source(fig) -> None:
    from matplotlib.transforms import offset_copy

    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    fig.text(bbox.x0 / w, bbox.y0 / h, fig._source, ha="left", va="top", fontsize=7.5, color=MUTED,
             transform=offset_copy(fig.transFigure, fig=fig, y=-8, units="points"))


def save(fig, name: str, directory: Path = FIGURE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    if getattr(fig, "_source", None):
        _place_source(fig)
    if getattr(fig, "_headline", None):
        _place_headline(fig)
    fig.savefig(path)
    plt.close(fig)
    return path
