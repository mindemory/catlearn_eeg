"""Plotting: the dark theme and the pieces the figures share.

The theme is behav_analyses/plot_style.py, loaded under its own module name: task_design/ has
a different plot_style.py, which kernel_model/kernel_modes.py imports, and loading both as
"plot_style" would make one shadow the other.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from helpers.stats import mean_sem
from params import EXCLUDED_COLOR, MISSING_COLOR, VERSION_COLORS, VERSIONS

_path = Path(__file__).resolve().parents[2] / "behav_analyses" / "plot_style.py"
_spec = importlib.util.spec_from_file_location("behav_plot_style", _path)
_theme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_theme)
apply_dark_theme = _theme.apply_dark_theme


def accuracy_cmap():
    """0-1 accuracy, 0.5 white (chance); missing cells grey."""
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(MISSING_COLOR)
    return cmap


def difference_cmap():
    """Version A - B: orange where A is higher, purple where B is; missing cells grey."""
    cmap = plt.get_cmap("PuOr_r").copy()
    cmap.set_bad(MISSING_COLOR)
    return cmap


def heatmap(ax, mat, cmap, lim, labels, fontsize=8):
    """Rows x trials heatmap (trial 1 at the left), NaN cells in the colormap's 'bad' colour."""
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto",
                   interpolation="nearest", extent=(0.5, mat.shape[1] + 0.5, mat.shape[0] - 0.5, -0.5))
    ax.set_yticks(range(len(labels)), labels, fontsize=fontsize)
    return im


def band(ax, rows, color, label, lw=1.8):
    """Mean over participants (rows) per trial, with a +- 1 SEM band."""
    m, s = mean_sem(rows)
    x = np.arange(1, len(m) + 1)
    ax.plot(x, m, color=color, lw=lw, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, lw=0)


def participant_lines(ax, df, x, y, order, size=4):
    """Each participant's values as dots on the categories, joined by a faint dashed line
    (red for participants excluded from the averages: df['excluded'])."""
    pos = {g: i for i, g in enumerate(order)}
    for _, d in df.groupby("participant"):
        col = EXCLUDED_COLOR if d["excluded"].iloc[0] else "0.75"
        d = d.set_index(x).reindex(order)
        ax.plot([pos[g] for g in order], d[y], ls="--", lw=0.7, color=col, alpha=0.6, marker="o", ms=size,
                markerfacecolor=col, markeredgewidth=0, zorder=3)


def version_box(ax, df, col, title, p, ylim=None):
    """Box plots per version with each participant as a dot; the title gives the A vs B p
    (red when p < .05)."""
    sns.boxplot(data=df, x="version", y=col, order=VERSIONS, ax=ax, hue="version", palette=VERSION_COLORS, width=0.5,
                fliersize=0, boxprops={"alpha": 0.4}, linewidth=1.1, legend=False)
    sns.stripplot(data=df, x="version", y=col, order=VERSIONS, ax=ax, hue="version", palette=VERSION_COLORS, size=4.5,
                  jitter=0.13, legend=False)
    ax.set_title(f"{title}\np = {p:.2f}", fontsize=9, **({"color": EXCLUDED_COLOR} if p < 0.05 else {}))
    ax.set_xlabel("")
    ax.set_ylabel("")
    if ylim:
        ax.set_ylim(ylim)


def save(fig, path, dpi=140, tight_rect=None):
    """Save and close a figure (tight_layout first if tight_rect is given)."""
    if tight_rect is not None:
        fig.tight_layout(rect=tight_rect)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
