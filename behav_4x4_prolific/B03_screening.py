"""Screening metrics over participants (box plots), per version: response bias, sanity
checks, and how each symbol and pair is associated with the F key.

Per participant and test block (and over all three, "All"), from all trials of the block:
  P(F)        share of answered trials with F; each rule has 8 F and 8 J pairs, so a
              participant without a key bias is near 0.5
  late        share of trials without an answer within 4 s
  fast        share of answered trials with RT < --fast ms (anticipating or mashing keys)
  P(repeat)   share of answered trials with the same key as the previous answered trial
              (0.5 without a strategy; high = perseverating, low = alternating)
and the key associations: P(F) per level (a1-a4: left symbol, b1-b4: right symbol) and per
pair (a1b1 ... a4b4), next to the rule's share of F pairs for that level (or the pair's
category), drawn as a white diamond.

In screening.png each participant's values are dots joined by a faint dashed line, red for
excluded participants (helpers.data.exclusions; the red dotted lines are the key-bias limits,
params.BIAS_LIMITS); associations.png shows the kept participants only.

Outputs, in analysis/average_version_<A|B>/sanity_tests/:
  screening.png, screening.csv          bias and sanity checks
  associations.png, associations.csv    P(F) per level and pair, per test block

  ~/miniforge3/envs/kernelbehav/bin/python B03_screening.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from helpers.measures import associations, block_name, checks
from helpers.plots import apply_dark_theme, participant_lines, save
from helpers.study import Study, parser
from params import ASSOC_COLOR, BIAS_LIMITS, BOX_COLOR, EXCLUDED_COLOR, FAST_RT_MS

CHECKS = {"p_f": ("P(F)", 0.5), "late": ("Late (no answer)", 0.0), "fast": ("Fast (RT < {fast} ms)", 0.0),
          "p_repeat": ("P(same key as previous)", 0.5)}


def plot_checks(df, title, fast, path):
    order = list(dict.fromkeys(df["block_label"]))
    fig, axes = plt.subplots(1, len(CHECKS), figsize=(3.8 * len(CHECKS), 3.6), squeeze=False)
    for ax, (key, (label, ref)) in zip(axes[0], CHECKS.items()):
        sns.boxplot(data=df, x="block_label", y=key, order=order, ax=ax, color=BOX_COLOR, width=0.55, fliersize=0,
                    boxprops={"alpha": 0.45}, linecolor=BOX_COLOR, linewidth=1.2)
        participant_lines(ax, df, "block_label", key, order)
        ax.axhline(ref, color="0.5", lw=0.8, ls=":")
        if key == "p_f":
            for lim in BIAS_LIMITS:
                ax.axhline(lim, color=EXCLUDED_COLOR, lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(label.format(fast=fast), fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        if key in ("p_f", "p_repeat"):
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, max(0.1, df[key].max() * 1.2))
    fig.suptitle(title, fontsize=11)
    save(fig, path, tight_rect=(0, 0, 1, 0.93))


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
            sns.boxplot(data=dk, x="item", y="p_f", order=order, ax=ax, color=ASSOC_COLOR, width=0.6, fliersize=0,
                        boxprops={"alpha": 0.45}, linecolor=ASSOC_COLOR, linewidth=1.1)
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
    save(fig, path, tight_rect=(0, 0, 1, 0.96))


def main():
    p = parser(__doc__)
    p.add_argument("--fast", type=float, default=FAST_RT_MS, help="RT (ms) below which an answer counts as fast")
    args = p.parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    excluded = study.save_exclusions()

    chk, assoc = [], []
    for pid, version, r in study.participants():
        blocks = study.test_blocks(pid)
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
        out = study.out(f"average_version_{version}", "sanity_tests")
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
