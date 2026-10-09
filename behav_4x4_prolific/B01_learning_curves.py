"""Learning curves in the test blocks: accuracy, F1 and RT over trials, and early vs late.

For every participant, each test block's trials in order, smoothed with a centred moving
window of --window trials (default params.WINDOW = 16, one pass through the 16 pairs):
  accuracy   proportion correct (late answers count as wrong)
  F1         2 TP / (2 TP + FP + FN), category 1 (the F key) as the positive class; late
             answers count as not F
  RT         mean RT of the answered trials in the window (ms; bottom row)
Then, per version (A, B), the mean over participants trial by trial (+- 1 SEM bands; kept
participants with complete test blocks), and early vs late: the same three measures over the
first and the last --edge trials of each test block (default 48, three passes).

Test blocks are numbered 1-3 (the session's blocks 3-5, after the two practice blocks).

Outputs, in analysis/:
  <participant>/learning_curves.png (top: accuracy, F1; bottom: RT), learning_curves.csv
  average_version_<A|B>/ the same, plus early_late.png and early_late.csv (one row per
      participant, test block and part; the figure: seaborn box plots over participants,
      dashed lines = each participant's early -> late), participants.txt

  ~/miniforge3/envs/kernelbehav/bin/python B01_learning_curves.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers.measures import block_name, curves, early_late
from helpers.plots import apply_dark_theme, save
from helpers.study import Study, parser
from params import EDGE, MEASURE_COLORS, MEASURE_LABELS, VERSIONS, WINDOW


def plot_curves(c, title, path, sem=None):
    """One column per test block: accuracy and F1 (top, 0-1), RT (bottom, one range for all
    blocks); +- 1 SEM bands if given."""
    blocks = list(c["test_block"].unique())
    fig, axes = plt.subplots(2, len(blocks), figsize=(4.6 * len(blocks), 5.6), sharex="col", sharey="row",
                             squeeze=False, gridspec_kw={"height_ratios": [1.3, 1]})
    for k, blk in enumerate(blocks):
        b = c[c["test_block"] == blk]
        x = b["trial"]
        for row, measures in ((0, ("accuracy", "f1")), (1, ("rt",))):
            ax = axes[row, k]
            for m in measures:
                ax.plot(x, b[m], color=MEASURE_COLORS[m], lw=1.8 if m != "f1" else 1.4, label=MEASURE_LABELS[m])
                if sem is not None:
                    s = sem[sem["test_block"] == blk][m].to_numpy()
                    ax.fill_between(x, b[m] - s, b[m] + s, color=MEASURE_COLORS[m], alpha=0.15, lw=0)
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
    save(fig, path, tight_rect=(0, 0, 1, 0.96))


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
        sns.boxplot(data=el, x="group", y=m, order=order, ax=ax, color=MEASURE_COLORS[m], width=0.55, fliersize=0,
                    boxprops={"alpha": 0.45}, linecolor=MEASURE_COLORS[m], linewidth=1.2)
        pos = {g: i for i, g in enumerate(order)}
        for (_, blk), d in el.groupby(["participant", "test_block"]):
            d = d.set_index("part")
            g0, g1 = d.loc["early", "group"], d.loc["late", "group"]
            ax.plot([pos[g0], pos[g1]], [d.loc["early", m], d.loc["late", m]], ls="--", lw=0.8, color="0.65",
                    marker="o", ms=3, zorder=3)
        if m != "rt":
            ax.axhline(0.5, color="0.45", lw=0.7, ls=":")
        ax.set_xlabel("")
        ax.set_ylabel(MEASURE_LABELS[m])
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(title, fontsize=11)
    save(fig, path, tight_rect=(0, 0, 1, 0.94))


def main():
    p = parser(__doc__)
    p.add_argument("--window", type=int, default=WINDOW, help="trials in the centred moving window")
    p.add_argument("--edge", type=int, default=EDGE, help="trials in the early and the late part of a block")
    args = p.parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    excluded = study.save_exclusions()

    per = {}
    for pid, version, r in study.participants():
        c = curves(r, args.window)
        out = study.out(pid)
        c.to_csv(out / "learning_curves.csv", index=False)
        note = " - excluded" if pid in excluded else ""
        plot_curves(c, f"Participant {pid[:8]} (version {version}){note}", out / "learning_curves.png")
        print(f"{pid} (version {version}): {len(c)} test trials -> {out}")
        per[pid] = (c, early_late(r, args.edge))

    groups = study.groups()
    for version in VERSIONS:
        group = groups[version]
        if len(group) < 2:
            continue
        out = study.out(f"average_version_{version}")
        keys = ["test_block", "rule", "trial"]
        g = pd.concat([per[p][0] for p in group]).groupby(keys, sort=True)[["accuracy", "f1", "rt"]]
        mean, sem = g.mean().reset_index(), g.sem().reset_index()
        mean.merge(sem, on=keys, suffixes=("", "_sem")).assign(n=len(group)).to_csv(out / "learning_curves.csv",
                                                                                    index=False)
        plot_curves(mean, f"Mean of {len(group)} participants (version {version})", out / "learning_curves.png",
                    sem=sem)
        el = pd.concat([per[p][1].assign(participant=p) for p in group], ignore_index=True)
        el.to_csv(out / "early_late.csv", index=False)
        plot_early_late(el, f"Early vs late {args.edge} trials, {len(group)} participants (version {version})",
                        out / "early_late.png")
        (out / "participants.txt").write_text("\n".join(group) + "\n")
        print(f"average of {len(group)} (version {version}) -> {out}")


if __name__ == "__main__":
    main()
