"""Plot kernel (ANOVA / Walsh) loadings for every binary category task on a factorial stimulus set.

Stimuli are all combinations of D features with K levels each (e.g. 2x2x2). A task
assigns each stimulus to one of two categories (balanced, or off by one when the
number of stimuli is odd). The label vector is decomposed into orthogonal effect
subspaces: constant, main effects (A, B, ...), and interactions (AB, ..., ABC...).
The loading on an effect is the norm of the label vector's projection onto that
subspace, normalized so the squared loadings of a task sum to 1. For K = 2 every
effect is one-dimensional and this is the Walsh/Fourier decomposition of the task.

Tasks with identical loadings (e.g. related by relabeling feature levels or swapping
categories) are shown once. Columns are grouped into types: loading patterns that
are identical up to permuting the stimulus dimensions.

A second figure shows the pairwise cosine similarity between task loadings, with
tasks ordered by hierarchical clustering.
"""

import argparse
import itertools
import string
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist

from plot_style import ACCENT, DIVIDER, apply_dark_theme

DESIGNS = ["2x2", "3x3", "4x4", "2x2x2", "2x2x2x2"]


def contrast_basis(k):
    """Orthonormal K x (K-1) basis orthogonal to the constant vector."""
    q, _ = np.linalg.qr(np.column_stack([np.ones(k), np.eye(k)[:, : k - 1]]))
    return q[:, 1:]


def effect_bases(n_dims, n_levels):
    """Return effects (tuples of dim indices, ordered by interaction order) and their bases."""
    const = np.ones((n_levels, 1)) / np.sqrt(n_levels)
    contrast = contrast_basis(n_levels)
    effects = [s for order in range(n_dims + 1) for s in itertools.combinations(range(n_dims), order)]
    bases = []
    for effect in effects:
        basis = np.ones((1, 1))
        for d in range(n_dims):
            basis = np.kron(basis, contrast if d in effect else const)
        bases.append(basis)
    return effects, bases


def effect_name(effect):
    return "cst" if not effect else "".join(string.ascii_uppercase[d] for d in effect)


def task_loadings(n_dims, n_levels):
    """Loadings (n_tasks x n_effects) of all balanced tasks, deduplicated.

    Also returns one example label vector (+1 / -1 per stimulus) for each loading row.
    """
    n_stim = n_levels**n_dims
    effects, bases = effect_bases(n_dims, n_levels)

    positives = np.array(list(itertools.combinations(range(n_stim), n_stim // 2)))
    labels = -np.ones((len(positives), n_stim))
    np.put_along_axis(labels, positives, 1, axis=1)
    unit = labels / np.sqrt(n_stim)  # unit norm, so squared loadings sum to 1

    loadings = np.column_stack([np.linalg.norm(unit @ b, axis=1) for b in bases])
    loadings, first = np.unique(np.round(loadings, 6), axis=0, return_index=True)
    return effects, loadings, labels[first]


def group_by_type(effects, loadings, n_dims):
    """Order tasks by type (equivalence under dimension permutation).

    Returns the reordered loadings, the number of tasks in each type, and the ordering.
    """
    index = {e: i for i, e in enumerate(effects)}
    perms = [
        [index[tuple(sorted(p[d] for d in e))] for e in effects]
        for p in itertools.permutations(range(n_dims))
    ]
    orders = np.array([len(e) for e in effects])

    def canonical(row):
        return max(tuple(row[perm]) for perm in perms)

    def type_key(row):
        # Simpler types first: fewer relevant dimensions, then lower mean interaction
        # order, then lower highest order. For 2x2x2 this gives Shepard types I-VI.
        nonzero = row > 1e-6
        n_relevant = len({d for e, nz in zip(effects, nonzero) if nz for d in e})
        mean_order = round(np.dot(row**2, orders), 6)
        return (n_relevant, mean_order, orders[nonzero].max(), canonical(row))

    order = sorted(range(len(loadings)), key=lambda i: (type_key(loadings[i]), tuple(-loadings[i])))
    type_ids, sizes = [], []
    for i in order:
        key = canonical(loadings[i])
        if not type_ids or type_ids[-1] != key:
            type_ids.append(key)
            sizes.append(0)
        sizes[-1] += 1
    return loadings[order], sizes, np.array(order)


def to_roman(n):
    numerals = [(50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, symbol in numerals:
        while n >= value:
            out += symbol
            n -= value
    return out


def save(fig, out_dir, name, tight=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def plot_loadings(design, effects, loadings, type_sizes, out_dir, max_tasks):
    """Heatmap of kernel loadings (kernel vector x task), tasks grouped by type."""
    n_all, n_types = len(loadings), len(type_sizes)
    title = f"kernel loading for each task\n{design}: {n_all} tasks, {n_types} types"
    per_type = n_all > max_tasks
    if per_type:
        # Too many columns to read: keep the first task of each type
        loadings = loadings[np.cumsum([0] + type_sizes[:-1])]
        title += " (one example per type)"
    n_tasks, n_effects = loadings.shape

    fig_w = min(max(4, 0.35 * n_tasks + 2), 60)
    fig_h = max(3, 0.4 * n_effects + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(loadings.T, cmap="inferno", vmin=0, vmax=1, aspect="auto", interpolation="none")

    ax.set_yticks(range(n_effects), [effect_name(e) for e in effects])
    ax.set_ylabel("kernel vector")
    if per_type:
        ax.set_xlabel("task type")
        ax.set_xticks(range(n_tasks), range(1, n_tasks + 1), fontsize=8)
    else:
        ax.set_xlabel("task")
        ax.set_xticks(range(n_tasks))
    ax.set_title(title)

    # Separate interaction orders
    orders = [len(e) for e in effects]
    for i in range(1, n_effects):
        if orders[i] != orders[i - 1]:
            ax.axhline(i - 0.5, color=DIVIDER, ls="--", lw=1)

    # Separate and label task types
    if not per_type:
        edges = np.cumsum([0] + type_sizes)
        for edge in edges[1:-1]:
            ax.axvline(edge - 0.5, color=ACCENT, ls="--", lw=2)
        for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
            ax.text((start + stop - 1) / 2, 1.02, to_roman(i + 1), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=ACCENT, fontsize=10)
        ax.set_title(ax.get_title(), pad=22)

    fig.colorbar(im, ax=ax, label="loading", fraction=0.03 if n_tasks > 20 else 0.08)
    save(fig, out_dir / "loadings", f"loadings_{design}")


def plot_similarity(design, loadings, type_sizes, out_dir, max_tasks):
    """Task x task cosine similarity of kernel loadings, ordered by hierarchical clustering."""
    n_tasks = len(loadings)
    type_labels = np.repeat([to_roman(i + 1) for i in range(len(type_sizes))], type_sizes)

    if n_tasks > 2:
        dist = pdist(loadings, metric="cosine")
        order = leaves_list(optimal_leaf_ordering(linkage(dist, method="average"), dist))
    else:
        order = np.arange(n_tasks)
    unit = loadings[order] / np.linalg.norm(loadings[order], axis=1, keepdims=True)
    similarity = unit @ unit.T

    size = min(max(4, 0.25 * n_tasks + 2), 12)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(similarity, cmap="plasma", vmin=0, vmax=1, interpolation="none")
    if n_tasks <= max_tasks:
        ticks = range(n_tasks)
        ax.set_xticks(ticks, type_labels[order], rotation=90, fontsize=8)
        ax.set_yticks(ticks, type_labels[order], fontsize=8)
        ax.set_xlabel("task (type)")
        ax.set_ylabel("task (type)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("task")
        ax.set_ylabel("task")
    ax.set_title(f"task similarity (kernel loadings)\n{design}: {n_tasks} tasks, clustered")
    fig.colorbar(im, ax=ax, label="cosine similarity", fraction=0.046, pad=0.04)
    save(fig, out_dir / "similarity", f"similarity_{design}")


def plot_design(design, out_dir, max_tasks):
    dims = [int(x) for x in design.split("x")]
    n_dims, n_levels = len(dims), dims[0]
    assert all(d == n_levels for d in dims), "only equal levels per dimension supported"

    effects, loadings, _ = task_loadings(n_dims, n_levels)
    loadings, type_sizes, _ = group_by_type(effects, loadings, n_dims)
    plot_loadings(design, effects, loadings, type_sizes, out_dir, max_tasks)
    plot_similarity(design, loadings, type_sizes, out_dir, max_tasks)
    print(f"{design}: {len(loadings)} tasks in {len(type_sizes)} types")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--designs", nargs="+", default=DESIGNS, help="e.g. 2x2 3x3 2x2x2")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/Users/mrugank/Documents/data/catlearn_eeg/task_design/kernels"),
    )
    parser.add_argument(
        "--max-tasks", type=int, default=60, help="above this, plot one example task per type"
    )
    args = parser.parse_args()
    apply_dark_theme()

    for design in args.designs:
        plot_design(design, args.out_dir, args.max_tasks)


if __name__ == "__main__":
    main()
