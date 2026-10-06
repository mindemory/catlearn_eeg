"""Visualize candidate spatial configurations for the 2x2 task.

The screen is divided into a 7x7 grid with fixation in the center cell. On each
trial the two stimuli (dimension A and dimension B) appear in two grid cells; each
configuration below fixes which cells those are. Cells are (row, col) with (0, 0)
at the top left and fixation at (3, 3).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plot_style import apply_dark_theme

GRID = 7
FIXATION = (GRID // 2, GRID // 2)
DATA_DIR = Path("/Users/mrugank/Documents/data/catlearn_eeg/task_design")
DIM_COLORS = {"A": "#ffa630", "B": "#c77dff"}

# name -> (cell of A, cell of B)
CONFIGS = {
    "horizontal, near": ((3, 2), (3, 4)),
    "horizontal, mid": ((3, 1), (3, 5)),
    "horizontal, periphery": ((3, 0), (3, 6)),
    "vertical, near": ((2, 3), (4, 3)),
    "vertical, mid": ((1, 3), (5, 3)),
    "vertical, periphery": ((0, 3), (6, 3)),
    "upper quadrants, near": ((2, 2), (2, 4)),
    "upper quadrants, mid": ((1, 1), (1, 5)),
    "upper quadrants, periphery": ((0, 0), (0, 6)),
    "lower quadrants, near": ((4, 2), (4, 4)),
    "lower quadrants, mid": ((5, 1), (5, 5)),
    "lower quadrants, periphery": ((6, 0), (6, 6)),
    "diagonal ↘, near": ((2, 2), (4, 4)),
    "diagonal ↘, mid": ((1, 1), (5, 5)),
    "diagonal ↘, periphery": ((0, 0), (6, 6)),
    "diagonal ↗, near": ((4, 2), (2, 4)),
    "diagonal ↗, mid": ((5, 1), (1, 5)),
    "diagonal ↗, periphery": ((6, 0), (0, 6)),
}


def eccentricity(cell):
    return np.hypot(cell[0] - FIXATION[0], cell[1] - FIXATION[1])


def draw_config(ax, name, cells, patches):
    # Mid-gray screen with faint grid lines
    ax.add_patch(Rectangle((-0.5, -0.5), GRID, GRID, color="0.5", zorder=0))
    for k in range(GRID + 1):
        ax.plot([k - 0.5] * 2, [-0.5, GRID - 0.5], color="0.42", lw=0.6, zorder=1)
        ax.plot([-0.5, GRID - 0.5], [k - 0.5] * 2, color="0.42", lw=0.6, zorder=1)

    # Fixation cross
    fr, fc = FIXATION
    ax.plot([fc - 0.2, fc + 0.2], [fr, fr], color="black", lw=2, zorder=3)
    ax.plot([fc, fc], [fr - 0.2, fr + 0.2], color="black", lw=2, zorder=3)

    for (label, (r, c)), patch in zip(zip("AB", cells), patches):
        inset = 0.05
        ax.imshow(patch, cmap="gray", vmin=0, vmax=1, zorder=2,
                  extent=(c - 0.5 + inset, c + 0.5 - inset, r + 0.5 - inset, r - 0.5 + inset))
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=DIM_COLORS[label], lw=2.5, zorder=4))
        ax.text(c - 0.5, r - 0.5, label, color="black", fontsize=8, fontweight="bold", ha="left", va="top",
                zorder=5, bbox=dict(boxstyle="square,pad=0.15", fc=DIM_COLORS[label], ec="none"))

    (r1, c1), (r2, c2) = cells
    ecc = ", ".join(f"{eccentricity(cell):.1f}" for cell in cells)
    separation = np.hypot(r1 - r2, c1 - c2)
    ax.set_title(f"{name}\neccentricity {ecc} | separation {separation:.1f}", fontsize=9)
    ax.set_xlim(-0.5, GRID - 0.5)
    ax.set_ylim(GRID - 0.5, -0.5)  # row 0 at the top
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stimuli", nargs=2, default=["noisepatch_01", "noisepatch_02"],
                        help="stimulus images shown as A and B")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "spatial" / "spatial_configs_2x2.png")
    args = parser.parse_args()
    apply_dark_theme()

    patches = [plt.imread(DATA_DIR / "stimuli" / "png" / f"{name}.png")[..., 0] for name in args.stimuli]

    # One column per group, rows go near -> mid -> periphery
    n_rows = 3
    n_cols = int(np.ceil(len(CONFIGS) / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.9 * n_rows), squeeze=False)
    for k, (name, cells) in enumerate(CONFIGS.items()):
        draw_config(axes[k % n_rows, k // n_rows], name, cells, patches)
    for k in range(len(CONFIGS), n_rows * n_cols):
        axes[k % n_rows, k // n_rows].axis("off")

    fig.suptitle("Spatial configurations for the 2x2 task (7x7 grid, units = grid cells)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {len(CONFIGS)} configurations to {args.out}")


if __name__ == "__main__":
    main()
