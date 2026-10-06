"""Draw the stimulus configuration of one example task for each task type.

Stimuli are drawn at their positions in feature space (a grid for 2 dimensions, a
cube for 3, two linked cubes for 4) and colored by category. Neighboring stimuli
(one level apart on one dimension) are connected, each half of an edge taking the
color of its end. Types and their numbering match plot_task_kernels.py; the
nonzero kernel loadings of each example are listed under its panel.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from plot_style import FOREGROUND, apply_dark_theme
from plot_task_kernels import DESIGNS, effect_name, group_by_type, save, task_loadings, to_roman

CATEGORY_COLORS = {-1: "#ff5a5f", 1: "#4cc9f0"}
DIM_COLORS = ["#ffa630", "#c77dff", "#2ec4b6", "#b5e48c"]
# Screen direction of each feature axis: A up, B toward the viewer, C right, D a second cube
DIM_VECTORS = {
    2: [(0, 1), (1, 0)],
    3: [(0, 1), (-0.5, -0.6), (1, 0)],
    4: [(0, 1), (-0.5, -0.6), (1, 0), (2.1, 0.9)],
}


def stimulus_positions(n_dims, n_levels):
    levels = np.array(np.unravel_index(np.arange(n_levels**n_dims), (n_levels,) * n_dims)).T
    vectors = np.array(DIM_VECTORS[n_dims])
    return levels, levels / (n_levels - 1) @ vectors


def draw_task(ax, labels, levels, positions, n_levels):
    n_stim = len(labels)
    for i in range(n_stim):
        for j in range(i + 1, n_stim):
            diff = np.abs(levels[i] - levels[j])
            if diff.sum() != 1:
                continue
            last_dim = diff.argmax() == 3  # thinner links between the two cubes
            mid = (positions[i] + positions[j]) / 2
            for end, stim in ((positions[i], i), (positions[j], j)):
                ax.plot(*zip(end, mid), color=CATEGORY_COLORS[labels[stim]], lw=1.2 if last_dim else 2.5,
                        alpha=0.5 if last_dim else 0.9, zorder=1, solid_capstyle="butt")
    size = {2: 380, 3: 180, 4: 110}[n_levels] if levels.shape[1] < 4 else 150
    ax.scatter(*positions.T, s=size, c=[CATEGORY_COLORS[l] for l in labels],
               edgecolors="white", linewidths=1.2, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.12)


def draw_axis_key(ax, n_dims):
    for d, (dx, dy) in enumerate(DIM_VECTORS[n_dims]):
        scale = 0.45 / np.hypot(dx, dy)
        ax.annotate("", xy=(dx * scale, dy * scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=DIM_COLORS[d], lw=2.5))
        ax.text(dx * scale * 1.35, dy * scale * 1.35, "ABCD"[d], color=DIM_COLORS[d],
                fontsize=16, fontweight="bold", ha="center", va="center")
    ax.scatter([0], [0], s=30, c=FOREGROUND, zorder=3)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def loading_text(effects, row, per_line=4):
    items = [f"{effect_name(e)} {v:.2f}" for e, v in zip(effects, row) if v > 1e-6]
    return "\n".join("  ".join(items[i : i + per_line]) for i in range(0, len(items), per_line))


def plot_configs(design, out_dir):
    dims = [int(x) for x in design.split("x")]
    n_dims, n_levels = len(dims), dims[0]

    effects, loadings, labels = task_loadings(n_dims, n_levels)
    loadings, type_sizes, order = group_by_type(effects, loadings, n_dims)
    labels = labels[order]
    first = np.cumsum([0] + type_sizes[:-1])  # one example task per type
    levels, positions = stimulus_positions(n_dims, n_levels)

    n_panels = len(first) + 1  # plus the axis key
    n_cols = min(n_panels, 6 if n_dims == 4 else 5)
    n_rows = int(np.ceil(n_panels / n_cols))
    panel_w = 3.6 if n_dims == 4 else 2.8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, 3.3 * n_rows + 0.8), squeeze=False)
    axes = axes.ravel()

    draw_axis_key(axes[0], n_dims)
    for ax, t, i in zip(axes[1:], range(len(first)), first):
        draw_task(ax, labels[i], levels, positions, n_levels)
        ax.set_title(f"type {to_roman(t + 1)}", fontsize=12)
        ax.text(0.5, -0.02, loading_text(effects, loadings[i]), transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="#aaaaaa")
    for ax in axes[n_panels:]:
        ax.axis("off")

    handles = [Line2D([], [], ls="", marker="o", ms=10, mfc=CATEGORY_COLORS[c], mec="white", label=name)
               for c, name in ((-1, "category 0"), (1, "category 1"))]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig_h = fig.get_figheight()
    fig.suptitle(f"Stimulus configuration of each task type ({design})", fontsize=15,
                 y=1 - 0.15 / fig_h, va="top")
    fig.tight_layout(rect=(0, 0.5 / fig_h, 1, 1 - 0.6 / fig_h))  # room for legend and title
    save(fig, out_dir / "configurations", f"configs_{design}", tight=False)
    print(f"{design}: {len(first)} types")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--designs", nargs="+", default=DESIGNS, help="e.g. 2x2 3x3 2x2x2")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/mrugank/Documents/data/catlearn_eeg/task_design/kernels"),
    )
    args = parser.parse_args()
    apply_dark_theme()

    for design in args.designs:
        plot_configs(design, args.out_dir)


if __name__ == "__main__":
    main()
