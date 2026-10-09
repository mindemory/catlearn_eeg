"""Measures computed from trials: per block, over moving windows, per task mode, per level and
pair.

Blocks
  test_blocks(r)        a participant's test blocks, numbered 1.. in order
  block_name, rule_type 'Test block 1 (VI)', 'VI'
Scores (accuracy; F1 with category 1, the F key, as the positive class, late answers counting
as not F; RT of the answered trials)
  scores(b)             over a set of trials
  moving(b, window)     per trial, over a centred window
  early_late(r, edge)   over the first and the last `edge` trials of every test block
  checks(b, fast)       P(F), late, fast answers, P(same key as the previous answer)
  associations(b)       P(F) per level and pair against the rule's share of F
Modes (kernel_model/kernel_modes.py, Design('4x4')): the participant's choice function over
the 16 pairs, y = 2 P(F | pair) - 1 (late trials left out, unseen pairs 0), and the rule's
y* = +-1, both projected onto the modes A (left symbol), B (right symbol) and AB. Per mode, over
the pairs where the rule's component isn't 0: accuracy = sign agreement (0 counts as half), F1
with the rule's positive side as positive. NaN for a mode the rule doesn't use.
  mode_scores(y, ystar), mode_heat(b, window)
Levels and pairs: rows a1-a4 (left symbol), b1-b4 (right symbol), the 16 pairs a1b1 ... a4b4,
labelled with the rule's share of F (levels) or the category (pairs).
  level_rows(b), level_heat(b, window)
Versions A and B: the B rules of VI and X are the A rules with left and right swapped.
  to_a_terms(mat)       a version-B level/pair matrix in version A's terms
  contrast_data(r, ...) per test type: moving accuracy and per-mode accuracy
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernel_model"))
from kernel_modes import Design  # noqa: E402

from params import MODES  # noqa: E402

N_LEVELS = 8                                     # a1-a4, b1-b4 (then the 16 pairs)
DESIGN = Design("4x4")
PROJ = {m: DESIGN.projectors[DESIGN.mode_names.index(m)] for m in MODES}
EPS = 1e-9


# ---------------------------------------------------------------- blocks
def rule_type(rule):
    """'x4_VI_A' -> 'VI'."""
    return rule.split("_")[1]


def block_name(k, rule):
    """'Test block 1 (VI)' from the test block's number and its rule, e.g. 'x4_VI_A'."""
    return f"Test block {k} ({rule_type(rule)})"


def test_blocks(r):
    """The test blocks of one participant, numbered 1.. in order: [(number, rule, trials)]."""
    t = r[r["phase"] == "test"].sort_values(["block", "trial_in_block"])
    return [(k + 1, b["rule"].iloc[0], b) for k, (_, b) in enumerate(t.groupby("block"))]


# ---------------------------------------------------------------- scores
def scores(b):
    """Accuracy, F1 (F positive) and mean RT of a set of trials."""
    f, pos = b["choice"].eq(1), b["category"].eq(1)
    tp, fp, fn = int((f & pos).sum()), int((f & ~pos).sum()), int((~f & pos).sum())
    return {"accuracy": b["correct"].mean(), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
            "rt": b["rt"].mean()}


def moving(b, window):
    """Accuracy, F1 and RT per trial of a block, over a centred window of `window` trials."""
    roll = lambda s: s.rolling(window, center=True, min_periods=1)   # noqa: E731
    f = b["choice"].eq(1).astype(float)
    pos = b["category"].eq(1).astype(float)
    tp, fp, fn = roll(f * pos).sum(), roll(f * (1 - pos)).sum(), roll((1 - f) * pos).sum()
    denom = 2 * tp + fp + fn
    return pd.DataFrame({"trial": b["trial_in_block"].to_numpy(),
                         "accuracy": roll(b["correct"].astype(float)).mean().to_numpy(),
                         "f1": (2 * tp / denom.where(denom > 0)).to_numpy(),
                         "rt": roll(b["rt"]).mean().to_numpy()})


def curves(r, window):
    """Moving curves of every test block of one participant: one row per trial."""
    out = [moving(b, window).assign(test_block=k, rule=rule) for k, rule, b in test_blocks(r)]
    return pd.concat(out, ignore_index=True)[["test_block", "rule", "trial", "accuracy", "f1", "rt"]]


def early_late(r, edge):
    """Scores over the first and the last `edge` trials of every test block."""
    rows = []
    for k, rule, b in test_blocks(r):
        for part, seg in (("early", b.head(edge)), ("late", b.tail(edge))):
            rows.append({"test_block": k, "rule": rule, "part": part, **scores(seg)})
    return pd.DataFrame(rows)


def checks(b, fast):
    """Bias and sanity checks of a set of trials."""
    a = b[~b["timeout"]]
    keys = a["choice"].to_numpy()
    return {"p_f": a["choice"].eq(1).mean(), "late": b["timeout"].mean(), "fast": (a["rt"] < fast).mean(),
            "p_repeat": np.mean(keys[1:] == keys[:-1]) if len(keys) > 1 else np.nan}


def associations(b):
    """P(F) per level and pair of one test block, with the rule's F share."""
    a = b[~b["timeout"]]
    cat = b.groupby("compound")["category"].first()
    rows = []
    for side, col in (("a", "level_a"), ("b", "level_b")):
        for lvl in range(4):
            comps = [4 * lvl + j for j in range(4)] if side == "a" else [4 * i + lvl for i in range(4)]
            rows.append({"item": f"{side}{lvl + 1}", "kind": "level", "p_f": a.loc[a[col] == lvl, "choice"].eq(1).mean(),
                         "rule_f_share": cat[comps].mean()})
    for c in range(16):
        rows.append({"item": f"a{c // 4 + 1}b{c % 4 + 1}", "kind": "pair",
                     "p_f": a.loc[a["compound"] == c, "choice"].eq(1).mean(), "rule_f_share": float(cat[c])})
    return rows


# ---------------------------------------------------------------- modes
def mode_scores(y, ystar):
    """Accuracy and F1 per mode of a choice function y against the target y* (16 pairs)."""
    out = {}
    for m in MODES:
        c, t = PROJ[m] @ y, PROJ[m] @ ystar
        use = np.abs(t) > EPS
        if not use.any():
            out[m] = {"accuracy": np.nan, "f1": np.nan}
            continue
        c, t = c[use], t[use]
        agree = np.where(np.abs(c) < EPS, 0.5, (np.sign(c) == np.sign(t)).astype(float))
        tp = np.sum((c > EPS) & (t > 0))
        fp = np.sum((c > EPS) & (t < 0))
        fn = np.sum((c <= EPS) & (t > 0))
        out[m] = {"accuracy": agree.mean(), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan}
    return out


def mode_heat(b, window):
    """{'accuracy' / 'f1': array (modes, trials)} over centred windows of `window` trials."""
    ystar = 2.0 * b.groupby("compound")["category"].first().reindex(range(16)).to_numpy() - 1
    n = len(b)
    heat = {k: np.full((len(MODES), n), np.nan) for k in ("accuracy", "f1")}
    half = window // 2
    for i in range(n):
        w = b.iloc[max(0, i - half):min(n, i - half + window)]
        w = w[~w["timeout"]]
        p = w.groupby("compound")["choice"].mean().reindex(range(16))
        y = np.nan_to_num(2 * p.to_numpy() - 1)
        s = mode_scores(y, ystar)
        for r, m in enumerate(MODES):
            for k in heat:
                heat[k][r, i] = s[m][k]
    return heat


# ---------------------------------------------------------------- levels and pairs
def level_rows(b):
    """(label, mask) for every level and pair of a block."""
    cat = b.groupby("compound")["category"].first()
    out = []
    for i in range(4):
        out.append((f"a{i + 1} ({int(cat[[4 * i + j for j in range(4)]].sum())}/4 F)", b["level_a"].eq(i)))
    for j in range(4):
        out.append((f"b{j + 1} ({int(cat[[4 * i + j for i in range(4)]].sum())}/4 F)", b["level_b"].eq(j)))
    for i in range(4):
        for j in range(4):
            c = 4 * i + j
            out.append((f"a{i + 1}b{j + 1} ({'F' if cat[c] == 1 else 'J'})", b["compound"].eq(c)))
    return out


def level_heat(b, window):
    """{'accuracy': (24, trials), 'f1': (8, trials)} and the row labels. F1 only for levels (a pair
    is always F or always J), NaN for a level with no F pair."""
    roll = lambda s: s.rolling(window, center=True, min_periods=1).sum().to_numpy()   # noqa: E731
    f = b["choice"].eq(1).astype(float)
    pos = b["category"].eq(1).astype(float)
    ok = b["correct"].astype(float)
    labels, acc, f1 = [], [], []
    for k, (label, mask) in enumerate(level_rows(b)):
        m = mask.astype(float)
        n = roll(m)
        with np.errstate(all="ignore"):
            acc.append(np.where(n > 0, roll(ok * m) / n, np.nan))
            if k < N_LEVELS:
                tp, fp, fn = roll(f * pos * m), roll(f * (1 - pos) * m), roll((1 - f) * pos * m)
                d = 2 * tp + fp + fn
                has_f = (pos * m).any()                  # a level with no F pair has no positive class
                f1.append(np.where((d > 0) & has_f, 2 * tp / d, np.nan))
        labels.append(label)
    return {"accuracy": np.array(acc), "f1": np.array(f1)}, labels


# ---------------------------------------------------------------- versions A and B
# version B -> version A's terms for the swapped blocks (VI, X): B's row index -> A's row index
SWAP = ([N_LEVELS // 2 + k for k in range(4)] + [k for k in range(4)]
        + [N_LEVELS + 4 * j + i for i in range(4) for j in range(4)])


def to_a_terms(mat):
    """Rows of a version-B level/pair matrix (VI, X) rearranged into version A's terms."""
    out = np.empty_like(mat)
    out[SWAP] = mat
    return out


def contrast_data(r, window, mode_window):
    """{rule_type: {'acc': moving accuracy, 'A' / 'B' / 'AB': mode accuracy over trials}}
    (NaN for a mode the block's rule doesn't use)."""
    out = {}
    for _, rule, b in test_blocks(r):
        heat = mode_heat(b, mode_window)["accuracy"]
        d = {"acc": moving(b, window)["accuracy"].to_numpy()}
        d.update({m: heat[MODES.index(m)] for m in MODES})
        out[rule_type(rule)] = d
    return out
