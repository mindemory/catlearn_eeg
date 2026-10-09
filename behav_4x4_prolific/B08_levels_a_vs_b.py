"""Version A vs version B per symbol level and pair (B04's rows), as heatmaps and 4 x 4 grids.

Rows as in B04: the levels a1-a4 (left symbol) and b1-b4 (right symbol), then the 16 pairs.
Levels are positions in each block's design (which fractal plays a1 is random), so they line
up across participants of a version. Between versions:
  VI, X   version B's rule is version A's with left and right swapped, so B's b_k plays the
          role of A's a_k (and B's a_k of A's b_k), and B's pair a_i b_j that of A's a_j b_i.
          Version B is shown in version A's terms (helpers.measures.to_a_terms), so every row
          compares like with like; rows are labelled with version A's rule.
  II      the same rule for both, and symmetric itself: rows compare directly, as they are.
          Any A - B pattern here is history: e.g. version A ahead on the a levels.

Each value is the share correct among that row's trials in a centred window of --window
trials (default params.CONTRAST_WINDOW = 32: ~8 trials per level, ~2 per pair), averaged over
the kept participants with complete test blocks.

Outputs, in analysis/a_vs_b/:
  levels.png        per block: version A, version B (in A's terms for VI and X), A - B
  pair_grids.png    per block: 4 x 4 accuracy of each pair over the second half of the block
                    (F pairs marked), version A, version B, A - B
  levels.csv        per participant, block and row: whole-block and second-half accuracy
  levels_tests.csv  A vs B permutation tests per row (p uncorrected, and BH-adjusted q)

  ~/miniforge3/envs/kernelbehav/bin/python B08_levels_a_vs_b.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.measures import N_LEVELS, level_heat, rule_type, to_a_terms
from helpers.plots import accuracy_cmap, apply_dark_theme, difference_cmap, heatmap, save
from helpers.stats import bh, nanmean, perm_diff
from helpers.study import Study, parser
from params import CONTRAST_WINDOW, DIFF_LIM, N_PERM_ROWS, TEST_TYPES, VERSIONS


def row_tests(table):
    tests = []
    for (typ, k, lab), s in table.groupby(["test_block", "row_index", "row"], sort=False):
        for part in ("mean", "second_half"):
            a, b = s.loc[s["version"] == "A", part].dropna(), s.loc[s["version"] == "B", part].dropna()
            diff, p = perm_diff(a, b, n=N_PERM_ROWS)
            tests.append({"test_block": typ, "row_index": k, "row": lab, "part": part, "A": a.mean(), "B": b.mean(),
                          "A - B": diff, "p": p})
    tests = pd.DataFrame(tests)
    tests["q_bh"] = np.nan
    for _, idx in tests.groupby(["test_block", "part"]).groups.items():
        tests.loc[idx, "q_bh"] = bh(tests.loc[idx, "p"])
    return tests


def main():
    p = parser(__doc__)
    p.add_argument("--window", type=int, default=CONTRAST_WINDOW, help="trials in the centred moving window")
    args = p.parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    study.save_exclusions()
    groups = study.groups()

    data, labels = {}, {}
    for v in VERSIONS:
        for pid in groups[v]:
            d = {}
            for _, rule, b in study.test_blocks(pid):
                typ = rule_type(rule)
                heat, labs = level_heat(b, args.window)
                acc = heat["accuracy"]
                if v == "B" and typ != "II":
                    acc = to_a_terms(acc)
                elif v == "A":
                    labels.setdefault(typ, labs)     # rows labelled with version A's rule
                d[typ] = acc
            data[pid] = (v, d)
    n = {v: len(groups[v]) for v in VERSIONS}
    tag = study.tag(groups)
    out = study.out("a_vs_b")

    def mean(v, typ):
        return nanmean(np.stack([data[p][1][typ] for p in groups[v]]), 0)

    table = pd.DataFrame([{"participant": pid, "version": v, "test_block": typ, "row": lab, "row_index": k,
                           "mean": nanmean(acc[k]), "second_half": nanmean(acc[k, acc.shape[1] // 2:])}
                          for pid, (v, d) in data.items() for typ, acc in d.items()
                          for k, lab in enumerate(labels[typ])])
    tests = row_tests(table)
    table.to_csv(out / "levels.csv", index=False)
    tests.to_csv(out / "levels_tests.csv", index=False)

    # ------------------------------------------------ level and pair heatmaps
    fig, axes = plt.subplots(3, 3, figsize=(16, 21), constrained_layout=True)
    for c, typ in enumerate(TEST_TYPES):
        ma, mb = mean("A", typ), mean("B", typ)
        sig = tests[(tests["test_block"] == typ) & (tests["part"] == "mean")].set_index("row_index")["p"]
        swapped = typ != "II"
        for r, (what, mat, cmap, lim) in enumerate((
                (f"version A (n={n['A']})", ma, accuracy_cmap(), (0, 1)),
                (f"version B (n={n['B']})" + (", sides swapped" if swapped else ""), mb, accuracy_cmap(), (0, 1)),
                ("A - B", ma - mb, difference_cmap(), (-DIFF_LIM, DIFF_LIM)))):
            ax = axes[r, c]
            labs = labels[typ]
            if r == 2:                       # mark rows with p < .05 (uncorrected) in the difference
                labs = [f"{lab}  *" if sig.get(k, 1) < 0.05 else lab for k, lab in enumerate(labs)]
            im = heatmap(ax, mat, cmap, lim, labs, fontsize=7)
            for y in (3.5, 7.5):
                ax.axhline(y, color="0.9", lw=0.8)
            if r == 0:
                ax.set_title(f"{typ}" + ("  (rows: version A's rule; B shown with sides swapped)" if swapped
                                         else "  (same rule for both)"), fontsize=10)
            ax.set_ylabel(what)
            if r == 2:
                ax.set_xlabel("Trial")
            if c == 2 and r == 1:
                fig.colorbar(im, ax=axes[:2, :], fraction=0.015, pad=0.01, label="accuracy (0.5 = chance)")
            if c == 2 and r == 2:
                fig.colorbar(im, ax=axes[2, :], fraction=0.015, pad=0.01, label="A - B (orange: A higher)")
    fig.suptitle(f"Accuracy per level and pair, version A vs B ({tag}; {args.window}-trial window; "
                 "* p < .05 uncorrected, whole block)", fontsize=12)
    save(fig, out / "levels.png", dpi=110)

    # ------------------------------------------------ 4 x 4 pair grids, second half
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for c, typ in enumerate(TEST_TYPES):
        sub = tests[(tests["test_block"] == typ) & (tests["part"] == "second_half") & (tests["row_index"] >= N_LEVELS)]
        sub = sub.sort_values("row_index")
        grids = {v: sub[v].to_numpy().reshape(4, 4) for v in VERSIONS}
        cats = [lab.endswith("(F)") for lab in sub["row"]]
        for r, (what, mat, cmap, lim) in enumerate((
                (f"version A (n={n['A']})", grids["A"], accuracy_cmap(), (0, 1)),
                (f"version B (n={n['B']})" + (", sides swapped" if typ != "II" else ""), grids["B"], accuracy_cmap(),
                 (0, 1)),
                ("A - B", grids["A"] - grids["B"], difference_cmap(), (-DIFF_LIM, DIFF_LIM)))):
            ax = axes[r, c]
            im = ax.imshow(mat, cmap=cmap, vmin=lim[0], vmax=lim[1])
            centre, dark = (0, 0.2) if r == 2 else (0.5, 0.3)
            for k, (val, is_f) in enumerate(zip(mat.ravel(), cats)):
                i, j = divmod(k, 4)
                ax.text(j, i, f"{'F' if is_f else 'J'}\n{val:+.2f}" if r == 2 else f"{'F' if is_f else 'J'}\n{val:.2f}",
                        ha="center", va="center", fontsize=8,     # white on dark cells, black on pale ones
                        color="white" if abs(val - centre) > dark else "black")
            ax.set_xticks(range(4), [f"b{j + 1}" for j in range(4)])
            ax.set_yticks(range(4), [f"a{i + 1}" for i in range(4)])
            if r == 0:
                ax.set_title(typ, fontsize=11)
            if c == 0:
                ax.set_ylabel(what)
            if c == 2 and r == 1:
                fig.colorbar(im, ax=axes[:2, :], fraction=0.02, pad=0.01, label="accuracy, second half")
            if c == 2 and r == 2:
                fig.colorbar(im, ax=axes[2, :], fraction=0.02, pad=0.01, label="A - B")
    fig.suptitle(f"Accuracy per pair in the second half of each block, version A vs B ({tag})", fontsize=12)
    save(fig, out / "pair_grids.png", dpi=120)

    hits = tests[(tests["part"] == "mean") & (tests["p"] < 0.05)]
    print(hits[["test_block", "row", "A", "B", "A - B", "p", "q_bh"]].round(3).to_string(index=False)
          if len(hits) else "no row with p < .05")
    print(f"-> {out}/levels.png, pair_grids.png, levels.csv, levels_tests.csv")


if __name__ == "__main__":
    main()
