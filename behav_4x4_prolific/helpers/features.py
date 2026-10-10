"""Stimulus features: how similar a block's fractals are, and how well that similarity lines up
with the block's rule, per task mode.

Each participant's test blocks use their own fractal groups, with fractals assigned to levels
at random, so the similarity structure varies across participants independently of the rule.

  similarity(feature)            candidate x candidate similarity in [0, 1] from
                                 fractal_stimuli/similarity_matrices.py (params.FRACTAL_SIMILARITY_DIR):
                                 colour (CIELAB histogram overlap), shape (silhouette overlap),
                                 perceptual (1 - DreamSim distance; flat within the task's groups
                                 by design, so a check rather than a predictor)
  block_fractals(b)              the pool candidates showing levels 1-4 of each side of a block
  side_similarity(S, cands)      4 x 4 similarity among one side's levels
  pair_kernel(SA, SB, how)       16 x 16 similarity between pairs (pair = 4 a + b):
                                 'product' SA (x) SB, two pairs alike when both symbols are;
                                 'sum' (SA (x) 1 + 1 (x) SB) / 2, alike when either is (no AB part)
  mode_alignment(K, ystar)       kernel-target alignment of the centred kernel with each mode's
                                 component of the rule: <Kc, y_e y_e'> / (||Kc|| ||y_e y_e'||), in
                                 [-1, 1]. High: pairs that look alike share that component's sign,
                                 so similarity helps; negative: similar pairs need opposite answers.
                                 'all': the same for the whole rule.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from helpers.measures import PROJ
from params import FRACTAL_GROUPS_CSV, FRACTAL_SIMILARITY_DIR, MODES

EPS = 1e-9


@lru_cache(maxsize=None)
def candidates():
    """Task fractal file number -> candidate number in the fractal pool."""
    g = pd.read_csv(FRACTAL_GROUPS_CSV)
    return dict(zip(g["file"].astype(int), g["candidate"].astype(int)))


@lru_cache(maxsize=None)
def similarity(feature):
    """Candidate x candidate similarity matrix (row k-1 = candidate k)."""
    return np.load(FRACTAL_SIMILARITY_DIR / f"similarity_{feature}.npy")


def block_fractals(b):
    """(left, right): the pool candidates showing levels 0-3 of each side of a block."""
    c = candidates()
    left = b.groupby("level_a")["fractal_a"].first().reindex(range(4)).astype(int).map(c).to_numpy()
    right = b.groupby("level_b")["fractal_b"].first().reindex(range(4)).astype(int).map(c).to_numpy()
    return left, right


def side_similarity(S, cands):
    return S[np.ix_(cands - 1, cands - 1)]


def pair_kernel(SA, SB, how="product"):
    if how == "product":
        return np.kron(SA, SB)
    if how == "sum":
        return (np.kron(SA, np.ones((4, 4))) + np.kron(np.ones((4, 4)), SB)) / 2
    raise ValueError(how)


def mode_alignment(K, ystar):
    """{mode: kernel-target alignment} (NaN for a mode the rule doesn't use), and 'all': the
    alignment with the whole rule."""
    H = np.eye(len(K)) - 1 / len(K)
    Kc = H @ K @ H
    norm_k = np.linalg.norm(Kc)
    out = {}
    for m, y in [(m, PROJ[m] @ ystar) for m in MODES] + [("all", H @ ystar)]:
        yy = y @ y
        out[m] = (y @ Kc @ y) / (norm_k * yy) if yy > EPS and norm_k > EPS else np.nan
    return out


def block_features(b, features, how="product"):
    """Per feature: mean similarity within each side and the mode alignments of one block."""
    left, right = block_fractals(b)
    ystar = 2.0 * b.groupby("compound")["category"].first().reindex(range(16)).to_numpy() - 1
    off = ~np.eye(4, dtype=bool)
    out = {}
    for f in features:
        S = similarity(f)
        SA, SB = side_similarity(S, left), side_similarity(S, right)
        out[f] = {"within_left": SA[off].mean(), "within_right": SB[off].mean(),
                  **{f"align_{m}": v for m, v in mode_alignment(pair_kernel(SA, SB, how), ystar).items()}}
    return out
