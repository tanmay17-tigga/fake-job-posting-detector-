"""Shared chart colors and style so every figure looks the same."""

import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt

REAL_COLOR = "#2a78d6"   # blue  = real job ads
FAKE_COLOR = "#eb6834"   # orange = fake job ads
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
        }
    )
