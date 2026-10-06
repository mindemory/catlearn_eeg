"""Generate 1/f^alpha noise patches (alpha = 1.5 by default), chosen from a larger pool to be mutually dissimilar."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from noise_patches import circular_aperture, make_noise_patch
from plot_style import apply_dark_theme


def save_patch(patch, out_dir, name):
    """Save a single patch as a pixel-exact PNG and as an SVG (embedded raster)."""
    plt.imsave(out_dir / "png" / f"{name}.png", patch, cmap="gray", vmin=0, vmax=1)

    size = patch.shape[0]
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1, interpolation="none")
    fig.savefig(out_dir / "svg" / f"{name}.svg")
    plt.close(fig)


def plot_similarity(patches, path):
    """Pairwise cosine similarity of pixel intensities.

    Intensities are taken relative to the mid-gray background, so the shared
    background does not inflate similarity.
    """
    vectors = np.array([p.ravel() - 0.5 for p in patches])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors @ vectors.T
    off_diag = similarity[~np.eye(len(patches), dtype=bool)]

    n = len(patches)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="none")
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(similarity[i, j]) > 0.5 else "black")
    ax.set_xticks(range(n), range(1, n + 1), fontsize=8)
    ax.set_yticks(range(n), range(1, n + 1), fontsize=8)
    ax.set_xlabel("stimulus")
    ax.set_ylabel("stimulus")
    ax.set_title(f"stimulus similarity (pixel intensity)\n"
                 f"off-diagonal: mean {off_diag.mean():.2f}, range [{off_diag.min():.2f}, {off_diag.max():.2f}]")
    fig.colorbar(im, ax=ax, label="cosine similarity", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved similarity matrix to {path}")


def contrast_vectors(patches):
    """Unit-norm contrast (intensity - background gray) vectors; their dot products are cosine similarities."""
    vectors = np.array([p.ravel() - 0.5 for p in patches])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def select_dissimilar(similarity, n, rng, n_starts=20, max_swaps=500):
    """Pick n items whose largest pairwise |similarity| is as small as possible.

    Greedy farthest-point selection from several random starts, then swaps that
    replace a member of the currently most similar pair while that lowers the
    (max, mean) of the pairwise |similarity|.
    """
    dist = np.abs(similarity)
    np.fill_diagonal(dist, 0)
    pool = len(dist)

    def score(sel):
        off = dist[np.ix_(sel, sel)][np.triu_indices(len(sel), 1)]
        return off.max(), off.mean()

    best = None
    for start in rng.choice(pool, size=min(n_starts, pool), replace=False):
        sel = [start]
        worst = dist[start].copy()
        worst[start] = np.inf
        while len(sel) < n:
            k = int(np.argmin(worst))
            sel.append(k)
            worst = np.maximum(worst, dist[k])
            worst[sel] = np.inf
        if best is None or score(sel) < score(best):
            best = sel

    sel = list(best)
    for _ in range(max_swaps):
        sub = dist[np.ix_(sel, sel)]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        current, improved = score(sel), False
        for member in (i, j):
            others = [s for idx, s in enumerate(sel) if idx != member]
            worst = dist[:, others].max(1)
            worst[sel] = np.inf
            candidate = sel.copy()
            candidate[member] = int(np.argmin(worst))
            if score(candidate) < current:
                sel, improved = candidate, True
                break
        if not improved:
            break
    return sel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=2000,
                        help="candidates generated; the n most mutually dissimilar are kept (= n-patches to skip)")
    parser.add_argument("--size", type=int, default=256, help="patch size in pixels")
    parser.add_argument(
        "--alpha", type=float, default=1.5, help="amplitude-spectrum slope (0.5 = pink, 1 = brown noise)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/mrugank/Documents/data/catlearn_eeg/task_design/stimuli"),
    )
    args = parser.parse_args()
    for ext in ("png", "svg"):
        (args.out_dir / ext).mkdir(parents=True, exist_ok=True)
    apply_dark_theme()

    rng = np.random.default_rng(args.seed)
    mask = circular_aperture(args.size)
    # Mid-gray background (0.5) with noise contrast inside the aperture
    pool = np.array([
        0.5 + 0.5 * make_noise_patch(args.size, rng, args.alpha) * mask
        for _ in range(max(args.pool_size, args.n_patches))
    ], dtype=np.float32)
    vectors = contrast_vectors(pool)
    similarity = vectors @ vectors.T
    if len(pool) > args.n_patches:
        chosen = select_dissimilar(similarity, args.n_patches, rng)
    else:
        chosen = list(range(args.n_patches))
    patches = list(pool[chosen])

    def abs_offdiag(idx):
        sub = np.abs(similarity[np.ix_(idx, idx)])
        return sub[np.triu_indices(len(idx), 1)]

    random_pick = abs_offdiag(list(range(args.n_patches)))
    kept = abs_offdiag(chosen)
    print(f"|cosine similarity| between patches: random {args.n_patches} -> max {random_pick.max():.2f}, "
          f"mean {random_pick.mean():.2f}; selected from pool of {len(pool)} -> max {kept.max():.2f}, "
          f"mean {kept.mean():.2f}")

    for i, patch in enumerate(patches):
        save_patch(patch, args.out_dir, f"noisepatch_{i + 1:02d}")

    # Overview grid of all patches
    n_cols = min(5, args.n_patches)
    n_rows = int(np.ceil(args.n_patches / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < len(patches):
            ax.imshow(patches[i], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{i + 1}")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(args.out_dir / ext / f"overview.{ext}", dpi=150)
    print(f"Saved {args.n_patches} patches (png + svg) and overview to {args.out_dir}")

    plot_similarity(patches, args.out_dir / "similarity.png")


if __name__ == "__main__":
    main()
