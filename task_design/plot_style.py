"""Shared dark matplotlib theme for all task_design figures."""

import matplotlib.pyplot as plt

BACKGROUND = "#121212"
FOREGROUND = "#e6e6e6"
ACCENT = "turquoise"  # task-type separators
DIVIDER = "#ff6b6b"  # interaction-order separators


def apply_dark_theme():
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "axes.edgecolor": FOREGROUND,
            "axes.labelcolor": FOREGROUND,
            "text.color": FOREGROUND,
            "xtick.color": FOREGROUND,
            "ytick.color": FOREGROUND,
        }
    )
