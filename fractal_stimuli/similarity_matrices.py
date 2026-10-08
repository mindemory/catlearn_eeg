"""Colour, shape and perceptual similarity between all candidate fractals, as sorted heatmaps.

Three measures, each a similarity in [0, 1] (1 = identical) between every pair of the pool's
candidates (fractal_pool_vivid by default):
  colour      histogram intersection of the fractals' colours in CIELAB (6 x 8 x 8 bins over
              L*, a*, b*), counting only the fractal's own pixels: same colours in the same
              proportions = 1, wherever they are
  shape       intersection over union of the two silhouettes (the filled outlines, colour
              ignored), at 128 x 128 px
  perceptual  1 - DreamSim distance (embed_dreamsim.py), fractals on the task's grey

Each heatmap is sorted by average-linkage clustering of that measure, so groups of mutually
similar fractals show up as bright squares on the diagonal. A second row puts all three in
the perceptual order, to show how far colour and shape line up with what DreamSim sees.

Writes to <pool>/similarity/:
  similarity_heatmaps.png    the six heatmaps
  clusters_<measure>.png     example fractals from the largest clusters of each measure
  similarity_<measure>.npy   the matrices (float32, row k-1 = candidate k)
  summary.txt                Spearman correlations between the measures across all pairs

  ~/miniforge3/envs/kernelbehav/bin/python fractal_stimuli/similarity_matrices.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from skimage import color

POOL = Path.home() / "Documents" / "data" / "catlearn_eeg" / "fractal_pool_vivid"
SIZE = 128
LAB_BINS = (6, 8, 8)
LAB_RANGE = ((0, 100), (-100, 100), (-100, 100))
MEASURES = ("colour", "shape", "perceptual")


def load(pool):
    files = sorted((pool / "candidates").glob("*.png"), key=lambda p: int(p.stem))
    assert [int(p.stem) for p in files] == list(range(1, len(files) + 1)), "candidates must be 1.png ... N.png"
    rgba = np.stack([np.asarray(Image.open(f).convert("RGBA").resize((SIZE, SIZE), Image.LANCZOS), np.float32) / 255
                     for f in files])
    return rgba[..., :3], rgba[..., 3]


def colour_similarity(rgb, alpha):
    lab = color.rgb2lab(rgb.reshape(-1, SIZE, 3)).reshape(len(rgb), -1, 3)
    hists = np.stack([np.histogramdd(l, bins=LAB_BINS, range=LAB_RANGE, weights=a.ravel())[0].ravel()
                      for l, a in zip(lab, alpha)])
    hists /= hists.sum(1, keepdims=True)
    sim = np.empty((len(hists), len(hists)), np.float32)
    for i in range(len(hists)):                       # intersection = sum of bin-wise minima
        sim[i] = np.minimum(hists[i], hists).sum(1)
    return sim


def shape_similarity(alpha):
    masks = (alpha > 0.5).reshape(len(alpha), -1).astype(np.float32)
    inter = masks @ masks.T
    area = masks.sum(1)
    return (inter / (area[:, None] + area[None] - inter)).astype(np.float32)


def cluster_order(sim):
    dist = np.clip(1 - sim, 0, None).astype(np.float64)
    np.fill_diagonal(dist, 0)
    link = linkage(squareform(dist, checks=False), method="average")
    return leaves_list(link), link


def heat(ax, sim, order, title):
    off = sim[~np.eye(len(sim), dtype=bool)]
    lo, hi = np.percentile(off, [1, 99])
    im = ax.imshow(sim[np.ix_(order, order)], cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def cluster_sheet(pool, link, sim, name, path, k=8, per=10, tile=96):
    """The k largest of 3k clusters, `per` random members each, one row per cluster"""
    labels = fcluster(link, 3 * k, criterion="maxclust")
    ids, counts = np.unique(labels, return_counts=True)
    rng = np.random.default_rng(0)
    rows = []
    for c in ids[np.argsort(-counts)][:k]:
        members = np.flatnonzero(labels == c)
        within = sim[np.ix_(members, members)][np.triu_indices(len(members), 1)]
        pick = rng.choice(members, min(per, len(members)), replace=False)
        strip = Image.new("RGB", (per * tile, tile), (128, 128, 128))
        for j, m in enumerate(pick):
            im = Image.open(pool / "candidates" / f"{m + 1}.png").convert("RGBA").resize((tile, tile), Image.LANCZOS)
            bg = Image.new("RGBA", (tile, tile), (128, 128, 128, 255))
            bg.alpha_composite(im)
            strip.paste(bg.convert("RGB"), (j * tile, 0))
        rows.append((strip, len(members), within.mean() if len(within) else np.nan))
    fig, axes = plt.subplots(len(rows), 1, figsize=(per * 1.15, 1.35 * len(rows)))
    for ax, (strip, n, w) in zip(axes, rows):
        ax.imshow(strip)
        ax.axis("off")
        ax.set_title(f"cluster of {n} fractals (mean {name} similarity within it {w:.2f})", fontsize=9, loc="left")
    fig.suptitle(f"{name}: the {len(rows)} largest clusters ({per} random members each)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, default=POOL)
    args = parser.parse_args()
    out = args.pool / "similarity"
    out.mkdir(exist_ok=True)
    plt.style.use("dark_background")
    plt.rcParams.update({"figure.facecolor": "#121212", "axes.facecolor": "#121212", "savefig.facecolor": "#121212"})

    rgb, alpha = load(args.pool)
    sims = {
        "colour": colour_similarity(rgb, alpha),
        "shape": shape_similarity(alpha),
        "perceptual": (1 - np.load(args.pool / "dreamsim_distances.npy")).astype(np.float32),
    }
    for name, s in sims.items():
        np.fill_diagonal(s, 1)
        np.save(out / f"similarity_{name}.npy", s)
    orders, links = {}, {}
    for name, s in sims.items():
        orders[name], links[name] = cluster_order(s)

    n = len(rgb)
    fig, axes = plt.subplots(2, 3, figsize=(19, 12.5))
    for j, name in enumerate(MEASURES):
        heat(axes[0, j], sims[name], orders[name], f"{name}, sorted by its own clustering")
        heat(axes[1, j], sims[name], orders["perceptual"], f"{name}, in the perceptual (DreamSim) order")
    fig.suptitle(f"Similarity between all {n} candidate fractals ({args.pool.name}); colour scale: 1st-99th "
                 "percentile of the pairs", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "similarity_heatmaps.png", dpi=120)
    plt.close(fig)
    for name in MEASURES:
        cluster_sheet(args.pool, links[name], sims[name], name, out / f"clusters_{name}.png")

    iu = np.triu_indices(n, 1)
    lines = [f"{n} candidates, {len(iu[0])} pairs", "similarity across pairs: median (5th-95th percentile)"]
    for name in MEASURES:
        v = sims[name][iu]
        lines.append(f"  {name:<10} {np.median(v):.3f} ({np.percentile(v, 5):.3f}-{np.percentile(v, 95):.3f})")
    lines.append("Spearman correlation between measures across all pairs:")
    for a, b in (("colour", "shape"), ("colour", "perceptual"), ("shape", "perceptual")):
        lines.append(f"  {a} vs {b}: {spearmanr(sims[a][iu], sims[b][iu])[0]:.2f}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
