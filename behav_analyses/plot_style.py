"""Shared dark matplotlib theme for behavioral analysis figures (same as task_design)."""

import matplotlib.pyplot as plt

BACKGROUND = "#121212"
FOREGROUND = "#e6e6e6"
CORRECT = "#4cc9f0"
INCORRECT = "#ff5a5f"
ACCENT = "#ffa630"


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
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
