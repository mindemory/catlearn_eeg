"""Learning curves in the test blocks: accuracy, F1 and RT over trials, and early vs late.

For every participant, each test block's trials in order, smoothed with a centred moving
window of --window trials (default 16, one pass through the 16 pairs):
  accuracy   proportion correct (late answers count as wrong)
  F1         2 TP / (2 TP + FP + FN), category 1 (the F key) as the positive class; late
             answers count as not F
  RT         mean RT of the answered trials in the window (ms; bottom row)
Then, per version (A, B), the mean over participants trial by trial (± 1 SEM bands; only
participants with complete test blocks, and not excluded: load_data.exclusions, key bias),
and early vs late: the same three measures over
the first and the last --edge trials of each test block (default 48, three passes).

Test blocks are numbered 1-3 (the session's blocks 3-5, after the two practice blocks).

Outputs, in ~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/analysis/:
  <participant>/learning_curves.png (top: accuracy, F1; bottom: RT), learning_curves.csv
  average_version_<A|B>/ the same, plus early_late.png and early_late.csv (one row per
      participant, test block and part; the figure: seaborn box plots over participants,
      dashed lines = each participant's early -> late),
      participants.txt

  ~/miniforge3/envs/kernelbehav/bin/python B01_learning_curves.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import OUT_DIR, load_trials, save_exclusions  # noqa: E402
from style import apply_dark_theme  # noqa: E402

COLORS = {"accuracy": "#ffa62b", "f1": "#c77dff", "rt": "#4cc9f0"}   # mango, violet, sky blue (dark background)
LABELS = {"accuracy": "Accuracy", "f1": "F1", "rt": "RT (ms)"}


def block_name(k, rule):
    """'Test block 1 (VI)' from the test block's number and its rule, e.g. 'x4_VI_A'."""
    return f"Test block {k} ({rule.split('_')[1]})"


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


def test_blocks(r):
    """The test blocks of one participant, numbered 1.. in order: [(number, rule, trials)]."""
    t = r[r["phase"] == "test"].sort_values(["block", "trial_in_block"])
    return [(k + 1, b["rule"].iloc[0], b) for k, (_, b) in enumerate(t.groupby("block"))]


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


# ---------------------------------------------------------------- figures
def plot_curves(c, title, window, path, sem=None):
    """One column per test block: accuracy and F1 (top, 0-1), RT (bottom, one range for all
    blocks); ± 1 SEM bands if given."""
    blocks = list(c["test_block"].unique())
    fig, axes = plt.subplots(2, len(blocks), figsize=(4.6 * len(blocks), 5.6), sharex="col", sharey="row",
                             squeeze=False, gridspec_kw={"height_ratios": [1.3, 1]})
    rows = ((0, ("accuracy", "f1")), (1, ("rt",)))
    for k, blk in enumerate(blocks):
        b = c[c["test_block"] == blk]
        x = b["trial"]
        for row, measures in rows:
            ax = axes[row, k]
            for m in measures:
                ax.plot(x, b[m], color=COLORS[m], lw=1.8 if m != "f1" else 1.4, label=LABELS[m])
                if sem is not None:
                    s = sem[sem["test_block"] == blk][m].to_numpy()
                    ax.fill_between(x, b[m] - s, b[m] + s, color=COLORS[m], alpha=0.15, lw=0)
        axes[0, k].axhline(0.5, color="0.45", lw=0.7, ls=":")
        axes[0, k].set_title(block_name(blk, b["rule"].iloc[0]), fontsize=10)
        axes[1, k].set_xlabel("Trial")
        axes[1, k].set_xlim(1, x.max())
    # accuracy and F1 on their full scale; RT on one range covering every block (and band)
    axes[0, 0].set_ylim(0, 1)
    lo = c["rt"] - (sem["rt"].to_numpy() if sem is not None else 0)
    hi = c["rt"] + (sem["rt"].to_numpy() if sem is not None else 0)
    pad = 0.05 * (np.nanmax(hi) - np.nanmin(lo))
    axes[1, 0].set_ylim(np.nanmin(lo) - pad, np.nanmax(hi) + pad)
    axes[0, 0].set_ylabel("Accuracy / F1")
    axes[1, 0].set_ylabel("RT (ms)")
    axes[0, -1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_early_late(el, title, path):
    """Accuracy, F1 and RT early vs late in each test block: seaborn box plots over
    participants, with each participant's early -> late as a dashed line."""
    measures = ["accuracy", "f1", "rt"]
    el = el.copy()
    el["group"] = [f"{block_name(k, r).replace('Test block ', 'B')}\n{p}" for k, r, p in
                   zip(el["test_block"], el["rule"], el["part"])]
    order = list(dict.fromkeys(el.sort_values(["test_block", "part"])["group"]))
    fig, axes = plt.subplots(1, len(measures), figsize=(4.2 * len(measures), 3.6), squeeze=False)
    for ax, m in zip(axes[0], measures):
        sns.boxplot(data=el, x="group", y=m, order=order, ax=ax, color=COLORS[m], width=0.55, fliersize=0,
                    boxprops={"alpha": 0.45}, linecolor=COLORS[m], linewidth=1.2)
        pos = {g: i for i, g in enumerate(order)}
        for (_, blk), d in el.groupby(["participant", "test_block"]):
            d = d.set_index("part")
            g0, g1 = d.loc["early", "group"], d.loc["late", "group"]
            ax.plot([pos[g0], pos[g1]], [d.loc["early", m], d.loc["late", m]], ls="--", lw=0.8, color="0.65",
                    marker="o", ms=3, zorder=3)
        if m != "rt":
            ax.axhline(0.5, color="0.45", lw=0.7, ls=":")
        ax.set_xlabel("")
        ax.set_ylabel(LABELS[m])
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=int, default=16, help="trials in the centred moving window")
    parser.add_argument("--edge", type=int, default=48, help="trials in the early and the late part of a block")
    parser.add_argument("--no-sync", action="store_true", help="don't copy new files from the Drive folder")
    args = parser.parse_args()
    apply_dark_theme()

    trials = load_trials(sync=not args.no_sync)
    excluded = save_exclusions(trials)
    per = []
    for pid, r in trials.groupby("participant", sort=False):
        if r[r["phase"] == "test"].empty:
            continue
        c = curves(r, args.window)
        out = OUT_DIR / pid
        out.mkdir(parents=True, exist_ok=True)
        c.to_csv(out / "learning_curves.csv", index=False)
        version = r["version"].iloc[0]
        note = " - excluded (key bias)" if pid in excluded else ""
        plot_curves(c, f"Participant {pid[:8]} (version {version}){note}", args.window, out / "learning_curves.png")
        print(f"{pid} (version {version}): {len(c)} test trials -> {out}")
        per.append((pid, version, c, early_late(r, args.edge)))

    for version in sorted({v for _, v, _, _ in per}):
        group = [(pid, c, el) for pid, v, c, el in per if v == version and pid not in excluded]
        full = max(len(c) for _, c, _ in group)
        group = [g for g in group if len(g[1]) == full]          # complete test blocks only
        if len(group) < 2:
            continue
        out = OUT_DIR / f"average_version_{version}"
        out.mkdir(parents=True, exist_ok=True)
        keys = ["test_block", "rule", "trial"]
        g = pd.concat([c for _, c, _ in group]).groupby(keys, sort=True)[["accuracy", "f1", "rt"]]
        mean, sem = g.mean().reset_index(), g.sem().reset_index()
        mean.merge(sem, on=keys, suffixes=("", "_sem")).assign(n=len(group)).to_csv(out / "learning_curves.csv",
                                                                                    index=False)
        plot_curves(mean, f"Mean of {len(group)} participants (version {version})", args.window,
                    out / "learning_curves.png", sem=sem)
        el = pd.concat([e.assign(participant=pid) for pid, _, e in group], ignore_index=True)
        el.to_csv(out / "early_late.csv", index=False)
        plot_early_late(el, f"Early vs late {args.edge} trials, {len(group)} participants (version {version})",
                        out / "early_late.png")
        (out / "participants.txt").write_text("\n".join(pid for pid, _, _ in group) + "\n")
        print(f"average of {len(group)} (version {version}) -> {out}")


if __name__ == "__main__":
    main()
