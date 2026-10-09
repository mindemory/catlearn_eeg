"""Version A vs version B: is learning the same with left and right swapped (VI, X), and does
the earlier blocks' relevant side carry over into type II?

The B rules of VI and X are exactly the A rules with left and right swapped (x4_VI_B is
x4_VI_A transposed, x4_X_B is x4_X_A transposed), and type II is the same rule for both and
unchanged by the swap. So, per test block:
  VI, X   version A's rule has modes A + AB, version B's has B + AB. Without a left/right
          asymmetry, version A's mode A should look like version B's mode B, and AB like AB.
  II      both versions get the same rule, with equal A and B components. A difference can
          only come from the earlier blocks: if what was relevant before is learned faster,
          version A is ahead on mode A (left) and version B on mode B (right).

Measures (helpers.measures.contrast_data): accuracy over a centred window of --window trials,
and per-mode accuracy (sign agreement of the participant's choice function with the rule after
projecting both onto the modes of kernel_model/kernel_modes.py) over a wider window
(--mode-window, default params.CONTRAST_WINDOW = 32: about two trials per pair). Modes are
shown as they are (A = left symbol, B = right symbol), never pooled across versions.

Statistics, with few participants per version: version differences of per-participant means
(whole block and second half) by permutation of the version labels. In II, the side index
(mode A minus mode B, per participant) is compared between versions; the prediction is
positive for A, negative for B. Kept participants with complete test blocks.

Outputs, in analysis/a_vs_b/:
  curves.png          accuracy per test block, A vs B (mean +- SEM)
  mode_curves.png     per-mode accuracy over trials, A and B overlaid, each mode on its own
  modes.png           the same as heatmaps: version A, version B (0-1), and A - B on its own
                      diverging scale
  ii_side.png         type II: mode A and mode B per version, and each participant's side index
  summary.csv         per participant, block and measure: whole-block and second-half means
  tests.csv           the permutation tests

  ~/miniforge3/envs/kernelbehav/bin/python B07_a_vs_b.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.measures import contrast_data
from helpers.plots import accuracy_cmap, apply_dark_theme, band, difference_cmap, heatmap, save
from helpers.stats import RNG, mean_sem, nanmean, perm_diff
from helpers.study import Study, parser
from params import CONTRAST_WINDOW, DIFF_LIM, MODES, N_PERM, TEST_TYPES, VERSION_COLORS, VERSIONS, WINDOW

MAIN_MODE = "main mode (A's A vs B's B)"
SIDE_INDEX = "side index (A - B)"


def summarize(data):
    """Per participant, block and measure: whole-block and second-half means."""
    rows = []
    for pid, (v, d) in data.items():
        for typ, dd in d.items():
            for key, vals in dd.items():
                if np.all(np.isnan(vals)):
                    continue
                rows.append({"participant": pid, "version": v, "test_block": typ, "measure": key,
                             "mean": nanmean(vals), "second_half": nanmean(vals[len(vals) // 2:])})
        side = d["II"]["A"] - d["II"]["B"]
        rows.append({"participant": pid, "version": v, "test_block": "II", "measure": SIDE_INDEX,
                     "mean": np.nanmean(side), "second_half": np.nanmean(side[len(side) // 2:])})
    return pd.DataFrame(rows)


def run_tests(summary):
    def values(v, typ, measure, part):
        s = summary[(summary["version"] == v) & (summary["test_block"] == typ) & (summary["measure"] == measure)]
        return s[part].to_numpy()

    comparisons = [(typ, "acc", "acc") for typ in TEST_TYPES]
    comparisons += [(typ, MAIN_MODE, None) for typ in ("VI", "X")]
    comparisons += [(typ, "AB", "AB") for typ in TEST_TYPES]
    comparisons += [("II", m, m) for m in ("A", "B", SIDE_INDEX)]
    tests = []
    for typ, label, measure in comparisons:
        for part in ("mean", "second_half"):
            a, b = values("A", typ, measure or "A", part), values("B", typ, measure or "B", part)
            diff, p = perm_diff(a, b, n=N_PERM)
            tests.append({"test_block": typ, "measure": label, "part": part, "A": a.mean(), "B": b.mean(),
                          "A - B": diff, "p": p})
    return pd.DataFrame(tests), values


def main():
    p = parser(__doc__)
    p.add_argument("--window", type=int, default=WINDOW, help="trials in the accuracy window")
    p.add_argument("--mode-window", type=int, default=CONTRAST_WINDOW, help="trials in the mode-accuracy window")
    args = p.parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    study.save_exclusions()
    groups = study.groups()
    data = {pid: (v, contrast_data(study.trials_of(pid), args.window, args.mode_window))
            for v in VERSIONS for pid in groups[v]}
    n = {v: len(groups[v]) for v in VERSIONS}
    tag = study.tag(groups)
    out = study.out("a_vs_b")

    def stack(v, typ, key):
        return np.array([data[p][1][typ][key] for p in groups[v]])

    summary = summarize(data)
    summary.to_csv(out / "summary.csv", index=False)
    tests, values = run_tests(summary)
    tests.to_csv(out / "tests.csv", index=False)

    def p_of(typ, label, part="mean"):
        t = tests[(tests["test_block"] == typ) & (tests["measure"] == label) & (tests["part"] == part)]
        return t["p"].iloc[0]

    # ------------------------------------------------ accuracy curves
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, typ in zip(axes, TEST_TYPES):
        for v in VERSIONS:
            band(ax, stack(v, typ, "acc"), VERSION_COLORS[v], f"version {v} (n={n[v]})")
        ax.set_title(f"{typ}   (A vs B: p = {p_of(typ, 'acc'):.2f})", fontsize=10)
        ax.axhline(0.5, color="0.45", lw=0.7, ls=":")
        ax.set_xlabel("Trial")
        ax.set_ylim(0.3, 1)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Accuracy, version A vs B ({tag}; {args.window}-trial window)", fontsize=11)
    save(fig, out / "curves.png", tight_rect=(0, 0, 1, 0.93))

    # ------------------------------------------------ mode curves: one panel per block and mode
    fig, axes = plt.subplots(3, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    for c, typ in enumerate(TEST_TYPES):
        for r, mode in enumerate(MODES):
            ax = axes[r, c]
            for v in VERSIONS:
                rows_ = stack(v, typ, mode)
                if not np.all(np.isnan(rows_)):          # skip a version whose rule doesn't use this mode
                    band(ax, rows_, VERSION_COLORS[v], f"version {v}")
            ax.axhline(0.5, color="0.45", lw=0.7, ls=":")
            if typ == "II" or mode == "AB":
                ax.set_title(f"{typ}, mode {mode}   (A vs B: p = {p_of(typ, mode):.2f})", fontsize=9)
            else:
                only = "version A only" if mode == "A" else "version B only"
                ax.set_title(f"{typ}, mode {mode} ({only}; A's A vs B's B: p = {p_of(typ, MAIN_MODE):.2f})",
                             fontsize=9)
            if c == 0:
                ax.set_ylabel(f"mode {mode}" + {"A": " (left)", "B": " (right)", "AB": ""}[mode])
            if r == 2:
                ax.set_xlabel("Trial")
    axes[0, 0].set_ylim(0.2, 1.02)
    axes[0, 2].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Mode accuracy over trials, version A vs B ({tag}; {args.mode_window}-trial window, "
                 "mean +- SEM)", fontsize=11)
    save(fig, out / "mode_curves.png", tight_rect=(0, 0, 1, 0.95))

    # ------------------------------------------------ mode heatmaps: A, B (0-1), A - B (own scale)
    fig, axes = plt.subplots(3, 3, figsize=(15, 6.8), constrained_layout=True)
    for c, typ in enumerate(TEST_TYPES):
        mats = {v: np.stack([mean_sem(stack(v, typ, m))[0] for m in MODES]) for v in VERSIONS}
        if typ == "II":
            diff, dlabels = mats["A"] - mats["B"], ["A", "B", "AB"]
        else:                                    # each version's own main mode, and AB
            diff = np.stack([mats["A"][0] - mats["B"][1], mats["A"][2] - mats["B"][2]])
            dlabels = ["A's A - B's B", "AB"]
        for r, (what, mat, labels, cmap, lim) in enumerate((
                (f"version A (n={n['A']})", mats["A"], MODES, accuracy_cmap(), (0, 1)),
                (f"version B (n={n['B']})", mats["B"], MODES, accuracy_cmap(), (0, 1)),
                ("A - B", diff, dlabels, difference_cmap(), (-DIFF_LIM, DIFF_LIM)))):
            ax = axes[r, c]
            im = heatmap(ax, mat, cmap, lim, labels)
            if r == 0:
                ax.set_title(typ, fontsize=10)
            if c == 0:
                ax.set_ylabel(what)
            if r == 2:
                ax.set_xlabel("Trial")
            if c == 2 and r == 1:
                fig.colorbar(im, ax=axes[:2, :], fraction=0.02, pad=0.01, label="mode accuracy (0.5 = chance)")
            if c == 2 and r == 2:
                fig.colorbar(im, ax=axes[2, :], fraction=0.02, pad=0.01, label="A - B (orange: A higher)")
    fig.suptitle(f"Mode accuracy, version A vs B ({tag}; {args.mode_window}-trial window; grey: mode not in "
                 "the rule)", fontsize=11)
    save(fig, out / "modes.png", dpi=130)

    # ------------------------------------------------ II: mode A and mode B per version
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1, 1, 0.75]})
    for ax, mode in zip(axes[:2], ("A", "B")):
        for v in VERSIONS:
            band(ax, stack(v, "II", mode), VERSION_COLORS[v], f"version {v} (n={n[v]})")
        ax.set_title(f"II, mode {mode} ({'left' if mode == 'A' else 'right'} symbol)   "
                     f"A vs B: p = {p_of('II', mode):.2f}", fontsize=10)
        ax.axhline(0.5, color="0.45", lw=0.7, ls=":")
        ax.set_ylim(0.2, 1.02)
        ax.set_xlabel("Trial")
    axes[0].set_ylabel("Mode accuracy in II")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[2]
    for j, v in enumerate(VERSIONS):
        vals = values(v, "II", SIDE_INDEX, "mean")
        ax.scatter(j + RNG.uniform(-0.13, 0.13, len(vals)), vals, s=24, color=VERSION_COLORS[v], zorder=3)
        ax.hlines(vals.mean(), j - 0.25, j + 0.25, color="white", lw=2)
    ax.axhline(0, color="0.45", lw=0.7, ls=":")
    ax.set_xticks([0, 1], ["version A", "version B"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("mode A - mode B in II")
    ax.set_title(f"side index (A vs B: p = {p_of('II', SIDE_INDEX):.3f})", fontsize=10)
    fig.suptitle(f"Type II: is each version better on the side that mattered in VI and X? ({tag})", fontsize=11)
    save(fig, out / "ii_side.png", tight_rect=(0, 0, 1, 0.92))

    print(tests.round(3).to_string(index=False))
    print(f"-> {out}/curves.png, mode_curves.png, modes.png, ii_side.png, summary.csv, tests.csv")


if __name__ == "__main__":
    main()
