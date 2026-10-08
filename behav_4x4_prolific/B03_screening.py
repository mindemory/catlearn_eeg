"""Screening metrics over participants (box plots), per version: response bias, sanity
checks, and how each symbol and pair is associated with the F key.

Per participant and test block (and over all three, "All"), from all trials of the block:
  P(F)        share of answered trials with F; each rule has 8 F and 8 J pairs, so a
              participant without a key bias is near 0.5
  late        share of trials without an answer within 4 s
  fast        share of answered trials with RT < --fast ms (anticipating or mashing keys)
  P(repeat)   share of answered trials with the same key as the previous answered trial
              (0.5 without a strategy; high = perseverating, low = alternating)
and the key associations: P(F) per level (a1-a4: left symbol, b1-b4: right symbol) and
per pair (a1b1 ... a4b4), next to the rule's share of F pairs for that level (or the pair's
category), drawn as a white diamond.

In screening.png each participant's values are dots joined by a faint dashed line, red for
participants excluded from the averages (load_data.exclusions: P(F) over the test trials
outside 0.25-0.75, the red dotted lines); associations.png shows the kept participants only.

Outputs, in ~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/analysis/average_version_<A|B>/sanity_tests/:
  screening.png, screening.csv          bias and sanity checks
  associations.png, associations.csv    P(F) per level and pair, per test block

  ~/miniforge3/envs/kernelbehav/bin/python B03_screening.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from B01_learning_curves import block_name, test_blocks  # noqa: E402
from load_data import BIAS_LIMITS, OUT_DIR, load_trials, save_exclusions  # noqa: E402
from style import apply_dark_theme  # noqa: E402

BOX = "#4cc9f0"          # sky blue, as RT in B01
ASSOC = "#ffa62b"        # mango, as accuracy in B01
CHECKS = {"p_f": ("P(F)", 0.5), "late": ("Late (no answer)", 0.0), "fast": ("Fast (RT < {fast} ms)", 0.0),
          "p_repeat": ("P(same key as previous)", 0.5)}


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


EXCLUDED = "#ff5a5f"


def participant_lines(ax, df, x, y, order, size=4):
    """Each participant's values as dots on the categories, joined by a faint dashed line
    (red for participants excluded from the averages)."""
    pos = {g: i for i, g in enumerate(order)}
    for _, d in df.groupby("participant"):
        col = EXCLUDED if d["excluded"].iloc[0] else "0.75"
        d = d.set_index(x).reindex(order)
        ax.plot([pos[g] for g in order], d[y], ls="--", lw=0.7, color=col, alpha=0.6, marker="o", ms=size,
                markerfacecolor=col, markeredgewidth=0, zorder=3)


def plot_checks(df, title, fast, path):
    order = list(dict.fromkeys(df["block_label"]))
    fig, axes = plt.subplots(1, len(CHECKS), figsize=(3.8 * len(CHECKS), 3.6), squeeze=False)
    for ax, (key, (label, ref)) in zip(axes[0], CHECKS.items()):
        sns.boxplot(data=df, x="block_label", y=key, order=order, ax=ax, color=BOX, width=0.55, fliersize=0,
                    boxprops={"alpha": 0.45}, linecolor=BOX, linewidth=1.2)
        participant_lines(ax, df, "block_label", key, order)
        ax.axhline(ref, color="0.5", lw=0.8, ls=":")
        if key == "p_f":
            for lim in BIAS_LIMITS:
                ax.axhline(lim, color=EXCLUDED, lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(label.format(fast=fast), fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        if key in ("p_f", "p_repeat"):
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, max(0.1, df[key].max() * 1.2))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_associations(df, title, path):
    blocks = sorted(df["test_block"].unique())
    fig, axes = plt.subplots(len(blocks), 2, figsize=(15, 3.1 * len(blocks)), squeeze=False,
                             gridspec_kw={"width_ratios": [8, 16]})
    for r, blk in enumerate(blocks):
        d = df[df["test_block"] == blk]
        for c, kind in enumerate(("level", "pair")):
            ax = axes[r, c]
            dk = d[d["kind"] == kind]
            order = list(dict.fromkeys(dk["item"]))
            sns.boxplot(data=dk, x="item", y="p_f", order=order, ax=ax, color=ASSOC, width=0.6, fliersize=0,
                        boxprops={"alpha": 0.45}, linecolor=ASSOC, linewidth=1.1)
            sns.stripplot(data=dk, x="item", y="p_f", order=order, ax=ax, color="0.8", size=3, jitter=0.12)
            target = dk.groupby("item")["rule_f_share"].first().reindex(order)
            ax.scatter(range(len(order)), target, marker="D", s=28, color="white", edgecolor="black", zorder=5,
                       label="rule: share of F")
            if kind == "level":
                ax.axvline(3.5, color="0.4", lw=0.8)
            ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("")
            ax.set_ylabel("P(F)" if c == 0 else "")
            ax.tick_params(axis="x", labelsize=8, rotation=0 if kind == "level" else 45)
            ax.set_title(f"{block_name(blk, d['rule'].iloc[0])}: {'levels' if kind == 'level' else 'pairs'}",
                         fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", type=float, default=250, help="RT (ms) below which an answer counts as fast")
    parser.add_argument("--no-sync", action="store_true", help="don't copy new files from the Drive folder")
    args = parser.parse_args()
    apply_dark_theme()

    trials = load_trials(sync=not args.no_sync)
    excluded = save_exclusions(trials)
    chk, assoc = [], []
    for pid, r in trials.groupby("participant", sort=False):
        blocks = test_blocks(r)
        if not blocks:
            continue
        version = r["version"].iloc[0]
        for k, rule, b in blocks:
            chk.append({"participant": pid, "version": version, "test_block": k, "rule": rule,
                        "block_label": block_name(k, rule).replace("Test block ", "B"), **checks(b, args.fast)})
            assoc += [{"participant": pid, "version": version, "test_block": k, "rule": rule, **a}
                      for a in associations(b)]
        allb = pd.concat([b for _, _, b in blocks])
        chk.append({"participant": pid, "version": version, "test_block": 0, "rule": "", "block_label": "All",
                    **checks(allb, args.fast)})
    chk, assoc = pd.DataFrame(chk), pd.DataFrame(assoc)
    chk["excluded"] = chk["participant"].isin(excluded)
    assoc["excluded"] = assoc["participant"].isin(excluded)

    for version, cv in chk.groupby("version"):
        n = cv["participant"].nunique()
        out = OUT_DIR / f"average_version_{version}" / "sanity_tests"
        out.mkdir(parents=True, exist_ok=True)
        cv = cv.sort_values(["test_block"], key=lambda s: s.replace(0, 99))
        cv.to_csv(out / "screening.csv", index=False)
        n_ex = int(cv.groupby("participant")["excluded"].first().sum())
        plot_checks(cv, f"Screening, {n} participants (version {version}; red: excluded, {n_ex})", args.fast,
                    out / "screening.png")
        av = assoc[(assoc["version"] == version) & ~assoc["excluded"]]     # associations: kept participants
        av.to_csv(out / "associations.csv", index=False)
        plot_associations(av, f"P(F) per level and pair, {av['participant'].nunique()} participants (version {version})",
                          out / "associations.png")
        print(f"version {version}: {n} participants -> {out}/screening.png, associations.png")


if __name__ == "__main__":
    main()
