"""Pairwise perceptual-similarity estimates for the alien and fractal stimulus sets.

Images are RGBA; the alpha channel defines the object. Each image is cropped to its
object, padded to a square and resized so that pairs can be compared. Three
image-computable measures are reported, each scaled to [0, 1] (1 = identical):

- color: histogram intersection of the object's colors in CIELAB space
  (perceptually uniform, so equal distances look roughly equally different).
  Ignores where colors are.
- shape: intersection-over-union of the two silhouettes. Ignores color.
- pixel: 1 - mean per-pixel CIELAB color difference (Delta E), with images
  composited on mid-gray, scaled by the largest difference within the set.
  Sensitive to both color and layout.

Pairs are ranked by the mean of the measures used for that set. These are low-level proxies; human
similarity ratings (or deep-network features) would capture higher-level resemblance.
"""

import argparse
import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

from plot_style import apply_dark_theme

DATA_DIR = Path("/Users/mrugank/Documents/data/catlearn_eeg/task_design")
ALL_MEASURES = ("color", "shape", "pixel")
# name -> (folder, filename pattern, measures to use)
SETS = {
    # alienNN only: black_alien, yellow_alien (a sprite sheet) and spaceship are not in the pool
    "aliens": ("stimuli_aliens", r"alien\d+\.png", ALL_MEASURES),
    "fractals": ("stimuli_fractals", r"\d+\.png", ALL_MEASURES),
    # Noise patches are opaque grayscale squares: every silhouette is identical, so shape is
    # skipped, and "color" reduces to comparing luminance distributions
    "noise": ("stimuli/png", r"noisepatch_\d+\.png", ("color", "pixel")),
}
SIZE = 128
GRAY = 0.5
LAB_BINS = (6, 8, 8)
LAB_RANGE = ((0, 100), (-100, 100), (-100, 100))


def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.stem)]


def load_image(path):
    """Crop to the object, pad to a square, resize. Returns RGB in [0, 1] and alpha in [0, 1]."""
    im = Image.open(path).convert("RGBA")
    im = im.crop(im.getchannel("A").getbbox())
    side = max(im.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    arr = np.asarray(square.resize((SIZE, SIZE), Image.LANCZOS), float) / 255
    return arr[..., :3], arr[..., 3]


def rgb_to_lab(rgb):
    """sRGB in [0, 1] -> CIELAB (D65)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805],
                          [0.2126, 0.7152, 0.0722],
                          [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def similarity_matrices(images):
    rgbs, alphas = zip(*images)
    labs = [rgb_to_lab(rgb) for rgb in rgbs]

    # Color histograms of object pixels, weighted by alpha
    hists = []
    for lab, alpha in zip(labs, alphas):
        h, _ = np.histogramdd(lab.reshape(-1, 3), bins=LAB_BINS, range=LAB_RANGE, weights=alpha.ravel())
        hists.append(h.ravel() / h.sum())
    hists = np.array(hists)
    color = np.minimum(hists[:, None], hists[None]).sum(-1)

    masks = np.array([a > 0.5 for a in alphas]).reshape(len(images), -1)
    inter = masks.astype(float) @ masks.T.astype(float)
    area = masks.sum(1)
    shape = inter / (area[:, None] + area[None] - inter)

    # Composite on gray, then mean Delta E between every pair
    comp = np.array([rgb_to_lab(rgb * a[..., None] + GRAY * (1 - a[..., None])) for rgb, a in zip(rgbs, alphas)])
    comp = comp.reshape(len(images), -1, 3)
    delta_e = np.array([[np.linalg.norm(comp[i] - comp[j], axis=-1).mean() for j in range(len(images))]
                        for i in range(len(images))])
    pixel = 1 - delta_e / delta_e.max()

    return {"color": color, "shape": shape, "pixel": pixel}


def cluster_order(sim):
    dist = squareform(1 - sim, checks=False)
    dist = np.clip(dist, 0, None)
    return leaves_list(optimal_leaf_ordering(linkage(dist, method="average"), dist))


def plot_matrices(name, labels, mats, out_path):
    combined = np.mean(list(mats.values()), axis=0)
    order = cluster_order(combined)
    panels = {**mats, f"mean of {' + '.join(mats)}": combined}
    n = len(labels)
    fontsize = 7 if n <= 25 else 4

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 6))
    for ax, (title, m) in zip(axes, panels.items()):
        off = m[~np.eye(n, dtype=bool)]
        im = ax.imshow(m[np.ix_(order, order)], cmap="magma", vmin=off.min(), vmax=off.max(),
                       interpolation="none")
        ax.set_xticks(range(n), labels[order], rotation=90, fontsize=fontsize)
        ax.set_yticks(range(n), labels[order], fontsize=fontsize)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{name}: pairwise similarity (ordered by clustering on the mean; "
                 f"color scale spans off-diagonal range)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_pairs(name, labels, images, pairs, measures, out_path, n_pairs):
    top = pairs.head(n_pairs)
    n_cols = 4
    n_rows = int(np.ceil(len(top) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(2.2 * n_cols * 2, 3.1 * n_rows), squeeze=False)
    for k, (_, row) in enumerate(top.iterrows()):
        r, c = divmod(k, n_cols)
        for side, stim in enumerate((row["stim_1"], row["stim_2"])):
            rgb, a = images[list(labels).index(stim)]
            ax = axes[r, 2 * c + side]
            ax.imshow(rgb * a[..., None] + GRAY * (1 - a[..., None]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(stim, fontsize=8)
        scores = " ".join(f"{m} {row[m]:.2f}" for m in measures)
        axes[r, 2 * c].set_title(f"#{k + 1}  mean {row['mean']:.2f}\n{scores}", fontsize=8, loc="left")
    for ax in axes.ravel()[2 * len(top):]:
        ax.axis("off")
    fig.suptitle(f"{name}: {len(top)} most similar pairs (mean of {', '.join(measures)})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    parser.add_argument("--n-pairs", type=int, default=12, help="most similar pairs to show")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "stimulus_similarity")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_dark_theme()

    for name in args.sets:
        folder, pattern, measures = SETS[name]
        files = sorted((f for f in (DATA_DIR / folder).glob("*.png") if re.fullmatch(pattern, f.name)),
                       key=natural_key)
        labels = np.array([f.stem for f in files])
        images = [load_image(f) for f in files]
        mats = {k: m for k, m in similarity_matrices(images).items() if k in measures}

        pairs = pd.DataFrame(
            [{"stim_1": labels[i], "stim_2": labels[j], **{k: m[i, j] for k, m in mats.items()}}
             for i, j in itertools.combinations(range(len(files)), 2)]
        )
        pairs["mean"] = pairs[list(mats)].mean(axis=1)
        pairs = pairs.sort_values("mean", ascending=False, ignore_index=True)
        pairs.round(4).to_csv(args.out_dir / f"{name}_pairs.csv", index=False)

        plot_matrices(name, labels, mats, args.out_dir / f"{name}_similarity.png")
        plot_top_pairs(name, labels, images, pairs, list(mats), args.out_dir / f"{name}_top_pairs.png",
                       args.n_pairs)

        corr = pairs[list(mats)].corr(method="spearman").round(2)
        print(f"== {name}: {len(files)} stimuli, {len(pairs)} pairs")
        print("Spearman correlation between measures:\n" + corr.to_string())
        print("Most similar pairs:\n" + pairs.head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
