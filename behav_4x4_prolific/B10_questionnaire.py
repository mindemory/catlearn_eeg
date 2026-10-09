"""The post-task questionnaire: ratings, strategies and free text, by version, and how the
answers line up with what each participant actually did.

Pages (catlearn_4x4_prolific/src/questionnaire.js): difficulty overall and per test block
(1 very easy - 7 very hard); per test block the strategy (left symbol / right symbol / each
symbol separately / specific pairs / guessed) and the hardest block; strategy and rule in
their own words; NASA-TLX workload (1-7); focus and bonus motivation (1-7); distraction,
notes, handedness, video-game hours; Need for Cognition (6 items, 1-5; items 3 and 4
reverse-scored; the score is the mean); comments. Test blocks 1-3 are VI, X and II for both
versions.

Against the data (per test block): accuracy, and the mode accuracies of B07 (mode A = left
symbol, mode B = right symbol, AB = the pair interaction; NaN where the block's rule has no
such component). E.g. participants who report "specific pairs" should have the higher AB
accuracy, and difficulty ratings should fall with accuracy (Spearman correlations).

Everyone who saw the questionnaire (it was added after the first participants of version A);
excluded participants are included and marked.

Outputs, in analysis/questionnaire/:
  answers.csv          one row per participant: every answer, NfC score, minutes on the
                       questionnaire, version, exclusion, and the per-block measures
  answers.md           readable tables and every free-text answer
  ratings.png          the rating scales by version
  strategies.png       reported strategy per block and the hardest block, by version
  strategy_vs_data.png per block, accuracy and AB accuracy by reported strategy

  ~/miniforge3/envs/kernelbehav/bin/python B10_questionnaire.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from helpers.data import session_files
from helpers.measures import contrast_data
from helpers.plots import apply_dark_theme, save, version_box
from helpers.report import md_table
from helpers.sessions import questionnaire
from helpers.stats import nanmean, perm_diff
from helpers.study import Study, parser
from params import CONTRAST_WINDOW, N_PERM_SANITY, STRATEGY_COLORS, VERSION_COLORS, VERSIONS, WINDOW

BLOCKS = {1: "VI", 2: "X", 3: "II"}
STRATEGIES = ["left", "right", "separate", "pairs", "guess"]
STRATEGY_LABELS = {"left": "left symbol", "right": "right symbol", "separate": "each separately",
                   "pairs": "specific pairs", "guess": "guessed"}
RATINGS = {"q_difficulty_overall": "Difficulty overall", "q_difficulty_block1": "Difficulty VI",
           "q_difficulty_block2": "Difficulty X", "q_difficulty_block3": "Difficulty II",
           "q_tlx_mental": "Mental demand", "q_tlx_physical": "Physical demand", "q_tlx_temporal": "Time pressure",
           "q_tlx_performance": "Success", "q_tlx_effort": "Effort", "q_tlx_frustration": "Frustration",
           "q_focus": "Focus", "q_motivation": "Bonus motivation", "nfc": "Need for Cognition (1-5)"}
TEXT = {"q_strategy": "How they decided", "q_rule": "Rule found", "q_comments": "Comments"}


def answers_with_data(study):
    """One row per participant with a questionnaire: the answers and the per-block measures."""
    df = pd.DataFrame([a for a in (questionnaire(f) for f in session_files()) if a])
    df["excluded"] = df["participant"].isin(study.excluded)
    for typ in BLOCKS.values():
        for col in ("accuracy", "mode_A", "mode_B", "mode_AB"):
            df[f"{typ}_{col}"] = np.nan
    for i, pid in enumerate(df["participant"]):
        if study.test_trials(pid) == 0:
            continue
        for typ, dd in contrast_data(study.trials_of(pid), WINDOW, CONTRAST_WINDOW).items():
            df.loc[i, f"{typ}_accuracy"] = nanmean(dd["acc"])
            for m in ("A", "B", "AB"):
                df.loc[i, f"{typ}_mode_{m}"] = nanmean(dd[m])
    for c in RATINGS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["version", "participant"]).reset_index(drop=True)


def plot_strategies(df, tag, path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    for ax, (k, typ) in zip(axes[:3], BLOCKS.items()):
        counts = df.groupby(["version", f"q_strategy_block{k}"]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=STRATEGIES, fill_value=0).reindex(VERSIONS, fill_value=0)
        bottom = np.zeros(2)
        for s in STRATEGIES:
            ax.bar([0, 1], counts[s], bottom=bottom, color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s], width=0.6)
            bottom += counts[s].to_numpy()
        ax.set_xticks([0, 1], ["version A\n(left matters)" if typ != "II" else "version A",
                               "version B\n(right matters)" if typ != "II" else "version B"])
        ax.set_title(f"{typ}: how did you decide?", fontsize=10)
        ax.set_ylabel("participants")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=3)
    hard = df.groupby(["version", "q_hardest_block"]).size().unstack(fill_value=0)
    order = [c for c in ["The first", "The second", "The third", "All about the same"] if c in hard.columns]
    hard = hard.reindex(columns=order, fill_value=0).reindex(VERSIONS, fill_value=0)
    x = np.arange(len(order))
    for j, v in enumerate(VERSIONS):
        axes[3].bar(x + (j - 0.5) * 0.38, hard.loc[v], width=0.38, color=VERSION_COLORS[v], label=f"version {v}")
    names = {"The first": "VI (1st)", "The second": "X (2nd)", "The third": "II (3rd)", "All about the same": "same"}
    axes[3].set_xticks(x, [names[o] for o in order])
    axes[3].set_title("Which block was hardest?", fontsize=10)
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Reported strategies ({tag})", fontsize=11)
    save(fig, path, dpi=130, tight_rect=(0, 0, 1, 0.92))


def plot_strategy_vs_data(df, tag, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey="row")
    for c, (k, typ) in enumerate(BLOCKS.items()):
        for r, (col, label) in enumerate(((f"{typ}_accuracy", "accuracy"), (f"{typ}_mode_AB", "AB mode accuracy"))):
            ax = axes[r, c]
            s = df.dropna(subset=[col])
            present = [st for st in STRATEGIES if (s[f"q_strategy_block{k}"] == st).any()]
            for j, st in enumerate(present):
                sub = s[s[f"q_strategy_block{k}"] == st]
                for v in VERSIONS:
                    vals = sub.loc[sub["version"] == v, col]
                    jitter = np.random.default_rng(j).uniform(-0.05, 0.05, len(vals))
                    ax.scatter(j + (0.12 if v == "B" else -0.12) + jitter, vals, color=VERSION_COLORS[v], s=30, zorder=3)
                ax.hlines(sub[col].mean(), j - 0.3, j + 0.3, color="white", lw=1.6)
            ax.set_xticks(range(len(present)), [STRATEGY_LABELS[st] for st in present], fontsize=8, rotation=15)
            ax.axhline(0.5, color="0.45", lw=0.7, ls=":")
            if r == 0:
                ax.set_title(f"{typ}: reported strategy vs data", fontsize=10)
            if c == 0:
                ax.set_ylabel(label)
    for v in VERSIONS:
        axes[0, 0].scatter([], [], color=VERSION_COLORS[v], label=f"version {v}")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Do the reported strategies match the data? ({tag}; white line: mean)", fontsize=11)
    save(fig, path, dpi=130, tight_rect=(0, 0, 1, 0.94))


def correlations(df):
    corr = []
    for k, typ in BLOCKS.items():
        s = df[[f"q_difficulty_block{k}", f"{typ}_accuracy"]].dropna()
        rho, p = spearmanr(s.iloc[:, 0], s.iloc[:, 1])
        corr.append({"pair": f"difficulty {typ} vs accuracy {typ}", "rho": rho, "p": p, "n": len(s)})
    acc_all = df[[f"{t}_accuracy" for t in BLOCKS.values()]].mean(axis=1)
    for c, label in (("q_tlx_performance", "felt success"), ("q_focus", "focus"), ("q_motivation", "bonus motivation"),
                     ("nfc", "Need for Cognition")):
        s = pd.DataFrame({"x": df[c], "y": acc_all}).dropna()
        rho, p = spearmanr(s["x"], s["y"])
        corr.append({"pair": f"{label} vs mean test accuracy", "rho": rho, "p": p, "n": len(s)})
    return pd.DataFrame(corr)


def write_md(df, tests, corr, tag, path):
    show = df.assign(id=df["participant"].str[:8])
    lines = [f"# Questionnaire ({tag})", ""]
    lines += ["## Ratings by version (means; p: permutation, uncorrected)", "", md_table(tests.round(3)), ""]
    lines += ["## Ratings vs data (Spearman)", "", md_table(corr.round(3)), ""]
    strat = show[["id", "version", "excluded"] + [f"q_strategy_block{k}" for k in BLOCKS]
                 + [f"{t}_accuracy" for t in BLOCKS.values()] + [f"{t}_mode_AB" for t in BLOCKS.values()]
                 + ["q_hardest_block"]].round(2)
    strat.columns = ["id", "version", "excluded", "strategy VI", "strategy X", "strategy II", "acc VI", "acc X",
                     "acc II", "AB VI", "AB X", "AB II", "hardest"]
    lines += ["## Strategies and the data per participant", "", md_table(strat), ""]
    other = show[["id", "version", "q_distracted", "q_notes", "q_handedness", "q_gaming", "questionnaire_min"]].round(1)
    lines += ["## Other answers", "", md_table(other), "", "## In their own words", ""]
    for _, r in show.iterrows():
        lines.append(f"**{r['id']}** (version {r['version']}{', excluded' if r['excluded'] else ''}; "
                     f"accuracy VI / X / II: {r['VI_accuracy']:.2f} / {r['X_accuracy']:.2f} / {r['II_accuracy']:.2f})")
        lines += [f"- *{label}:* {r[c].strip()}" for c, label in TEXT.items()
                  if isinstance(r.get(c), str) and r[c].strip()]
        lines.append("")
    path.write_text("\n".join(lines))
    return strat


def main():
    args = parser(__doc__).parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    df = answers_with_data(study)
    n = df["version"].value_counts().to_dict()
    tag = f"A: {n.get('A', 0)}, B: {n.get('B', 0)}"
    out = study.out("questionnaire")
    df.to_csv(out / "answers.csv", index=False)

    tests = []
    for c, label in RATINGS.items():
        a, b = df.loc[df["version"] == "A", c].dropna(), df.loc[df["version"] == "B", c].dropna()
        diff, p = perm_diff(a, b, n=N_PERM_SANITY)
        tests.append({"rating": label, "A": a.mean(), "B": b.mean(), "A - B": diff, "p": p})
    tests = pd.DataFrame(tests)

    cols = 5
    fig, axes = plt.subplots(int(np.ceil(len(RATINGS) / cols)), cols, figsize=(2.6 * cols, 3.1 * np.ceil(len(RATINGS) / cols)))
    for ax, (c, label) in zip(axes.ravel(), RATINGS.items()):
        version_box(ax, df, c, label, tests.loc[tests["rating"] == label, "p"].iloc[0],
                    ylim=(0.5, 5.5) if c == "nfc" else (0.5, 7.5))
    for ax in axes.ravel()[len(RATINGS):]:
        ax.axis("off")
    fig.suptitle(f"Questionnaire ratings by version ({tag}; 1-7 unless stated; p: A vs B permutation, "
                 "uncorrected)", fontsize=11)
    save(fig, out / "ratings.png", dpi=130, tight_rect=(0, 0, 1, 0.94))
    plot_strategies(df, tag, out / "strategies.png")
    plot_strategy_vs_data(df, tag, out / "strategy_vs_data.png")
    corr = correlations(df)
    strat = write_md(df, tests, corr, tag, out / "answers.md")

    print(tests.round(3).to_string(index=False))
    print()
    print(corr.round(3).to_string(index=False))
    print()
    print(strat.to_string(index=False))
    print(f"-> {out}/answers.csv, answers.md, ratings.png, strategies.png, strategy_vs_data.png")


if __name__ == "__main__":
    main()
