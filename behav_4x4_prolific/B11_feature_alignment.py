"""Do the fractals' colour (and shape) similarities shape which parts of the rule are learned?

Each participant's test blocks use their own fractals (their own groups, levels assigned at
random), so how the fractals resemble each other, and whether look-alikes happen to share a
category, varies across participants independently of the rule: a natural experiment.

Per participant and test block (helpers.features):
  pair similarity    a 16 x 16 kernel over the block's pairs from its actual fractals: two
                     pairs are alike when both their left and their right symbols are
                     (params.PAIR_KERNEL = 'product'; --kernel sum: when either is)
  alignment          for each mode of the rule (A = left symbol, B = right symbol, AB), the
                     kernel-target alignment of the centred kernel with that mode's component
                     of the rule (-1..1): high when look-alike pairs share the component's sign
                     (similarity should help), negative when they need opposite answers
  features           colour (CIELAB histogram overlap), shape (silhouette overlap), and
                     perceptual (1 - DreamSim distance: equated within the task's groups by
                     design, so it barely varies and serves as a check)
Learning (helpers.measures.contrast_data, as in B07): per-mode accuracy over the block
(--mode-window trials), and block accuracy for the whole rule.

Per test type (VI, X, II; versions A and B pooled: everyone is scored against their own rule
and their own fractals), the Spearman correlation of alignment with the matching mode's
accuracy across participants:
  VI, X   'main' = the mode the rule's main effect sits on (A for version A, B for version B),
          and AB
  II      A, B and AB (the same rule for everyone)
  all     the whole rule's alignment vs block accuracy
raw, and partial on ability (the participant's mean accuracy in the other two test blocks),
with permutation p values (uncorrected). Kept participants with complete test blocks.

Outputs, in analysis/features/:
  alignment.csv        per participant, block, feature and mode: alignment, within-side
                       similarity, mode accuracy (whole block, second half)
  tests.csv            the correlations
  <feature>.png        alignment vs learning, per test type and mode
  spread.png           how much each feature's alignment varies across participants

  ~/miniforge3/envs/kernelbehav/bin/python B11_feature_alignment.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from helpers.features import block_features
from helpers.measures import contrast_data, rule_type
from helpers.plots import apply_dark_theme, save
from helpers.stats import nanmean, perm_corr
from helpers.study import Study, parser
from params import (CONTRAST_WINDOW, FEATURES, N_PERM_SANITY, PAIR_KERNEL, TEST_TYPES, VERSION_COLORS, VERSIONS,
                    WINDOW)

TARGETS = {"VI": ["main", "AB", "all"], "X": ["main", "AB", "all"], "II": ["A", "B", "AB", "all"]}
ROWS = ["main / A", "B", "AB", "all"]                     # figure rows


def target_mode(target, version):
    """The rule's mode a target refers to for a participant of `version`."""
    return ("A" if version == "A" else "B") if target == "main" else target


def collect(study, groups, how, mode_window):
    rows = []
    for v in VERSIONS:
        for pid in groups[v]:
            acc = contrast_data(study.trials_of(pid), WINDOW, mode_window)
            block_acc = {typ: nanmean(d["acc"]) for typ, d in acc.items()}
            for _, rule, b in study.test_blocks(pid):
                typ = rule_type(rule)
                feats = block_features(b, FEATURES, how)
                others = np.mean([a for t, a in block_acc.items() if t != typ])
                for target in TARGETS[typ]:
                    m = target_mode(target, v)
                    if target == "all":
                        y = acc[typ]["acc"]
                    else:
                        y = acc[typ][m]
                    for f, fv in feats.items():
                        rows.append({"participant": pid, "version": v, "test_type": typ, "target": target,
                                     "mode": m, "feature": f, "within_left": fv["within_left"],
                                     "within_right": fv["within_right"],
                                     "alignment": fv["align_all"] if target == "all" else fv[f"align_{m}"],
                                     "learning": nanmean(y), "learning_second_half": nanmean(y[len(y) // 2:]),
                                     "ability_other_blocks": others})
    return pd.DataFrame(rows)


def run_tests(df):
    tests = []
    for (typ, target, f), s in df.groupby(["test_type", "target", "feature"], sort=False):
        for outcome in ("learning", "learning_second_half"):
            r, p, n = perm_corr(s["alignment"], s[outcome], n=N_PERM_SANITY)
            rp, pp, _ = perm_corr(s["alignment"], s[outcome], covariate=s["ability_other_blocks"], n=N_PERM_SANITY)
            tests.append({"test_type": typ, "target": target, "feature": f, "outcome": outcome, "n": n,
                          "rho": r, "p": p, "rho_partial_ability": rp, "p_partial": pp,
                          "alignment_mean": s["alignment"].mean(), "alignment_sd": s["alignment"].std()})
    return pd.DataFrame(tests)


def plot_feature(df, tests, feature, tag, path):
    fig, axes = plt.subplots(len(ROWS), len(TEST_TYPES), figsize=(13, 12.5), squeeze=False)
    for c, typ in enumerate(TEST_TYPES):
        for r, row in enumerate(ROWS):
            ax = axes[r, c]
            target = {"main / A": "main" if typ != "II" else "A"}.get(row, row)
            s = df[(df["test_type"] == typ) & (df["target"] == target) & (df["feature"] == feature)]
            if s.empty:
                ax.axis("off")
                continue
            for v in VERSIONS:
                sv = s[s["version"] == v]
                ax.scatter(sv["alignment"], sv["learning"], s=28, color=VERSION_COLORS[v], label=f"version {v}",
                           zorder=3)
            if s["alignment"].std() > 0:
                k, b0 = np.polyfit(s["alignment"], s["learning"], 1)
                xs = np.linspace(s["alignment"].min(), s["alignment"].max(), 20)
                ax.plot(xs, k * xs + b0, color="0.75", lw=1, ls="--")
            t = tests[(tests["test_type"] == typ) & (tests["target"] == target) & (tests["feature"] == feature)
                      & (tests["outcome"] == "learning")].iloc[0]
            what = {"main": "main mode (A in A, B in B)", "all": "whole rule"}.get(target, f"mode {target}")
            ax.set_title(f"{typ}, {what}\nrho = {t['rho']:.2f} (p = {t['p']:.2f}); "
                         f"given ability: {t['rho_partial_ability']:.2f} (p = {t['p_partial']:.2f})", fontsize=8.5,
                         **({"color": "#ff5a5f"} if t["p"] < 0.05 else {}))
            ax.axhline(0.5, color="0.45", lw=0.7, ls=":")
            ax.set_xlabel(f"{feature} alignment" if r == len(ROWS) - 1 or typ != "II" and row == "all" else "",
                          fontsize=8)
            ax.set_ylabel("block accuracy" if target == "all" else "mode accuracy", fontsize=8)
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{feature.capitalize()} similarity aligned with the rule vs learning ({tag}; p uncorrected)",
                 fontsize=11)
    save(fig, path, dpi=120, tight_rect=(0, 0, 1, 0.96))


def plot_spread(df, path):
    d = df[df["target"] != "all"].copy()
    d["mode"] = d["target"].where(d["target"] != "main", "main")
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(4.2 * len(FEATURES), 3.6), sharey=False)
    for ax, f in zip(axes, FEATURES):
        sns.boxplot(data=d[d["feature"] == f], x="test_type", y="alignment", hue="mode", order=TEST_TYPES, ax=ax,
                    fliersize=2, linewidth=1)
        ax.set_title(f, fontsize=10)
        ax.set_xlabel("")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("How much the alignment varies across participants, per feature and mode", fontsize=11)
    save(fig, path, dpi=120, tight_rect=(0, 0, 1, 0.92))


def main():
    p = parser(__doc__)
    p.add_argument("--kernel", choices=["product", "sum"], default=PAIR_KERNEL, help="how pair similarity is built")
    p.add_argument("--mode-window", type=int, default=CONTRAST_WINDOW, help="trials in the mode-accuracy window")
    args = p.parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    study.save_exclusions()
    groups = study.groups()
    tag = f"{study.tag(groups)}; {args.kernel} kernel"
    out = study.out("features" if args.kernel == PAIR_KERNEL else f"features_{args.kernel}")

    df = collect(study, groups, args.kernel, args.mode_window)
    df.to_csv(out / "alignment.csv", index=False)
    tests = run_tests(df)
    tests.to_csv(out / "tests.csv", index=False)
    for f in FEATURES:
        plot_feature(df, tests, f, tag, out / f"{f}.png")
    plot_spread(df, out / "spread.png")

    show = tests[tests["outcome"] == "learning"].drop(columns="outcome")
    print(show.round(3).to_string(index=False))
    print(f"-> {out}/alignment.csv, tests.csv, {', '.join(f + '.png' for f in FEATURES)}, spread.png")


if __name__ == "__main__":
    main()
