"""Version A vs version B on the sanity measures: do the two versions differ in anything other
than what the rules predict?

Per participant and test block (VI, X, II; from all of the block's trials):
  accuracy, F1 (F positive), median RT of the answered trials,
  P(F), late answers, fast answers (RT < params.FAST_RT_MS), P(same key as the previous answer)
  (helpers.measures.scores, checks). In VI and X the two versions have mirror-image rules, so
  none of these should differ; in II they have the same rule.
Per participant, once:
  session minutes (total, setup, the two practice blocks, the breaks), practice trials and
  accuracy (type I, XOR), viewing distance, quiz attempts, window switches
  (helpers.sessions.session_info).

Each measure: box plots per version (dots = participants) and the version difference by
permutation of the version labels (p uncorrected; many measures, so read single p < .05
with care). Plus RT and F1 over trials, A vs B (centred --window trials, mean +- SEM).
Kept participants with complete test blocks.

Outputs, in analysis/a_vs_b/:
  sanity_blocks.png     the per-block measures
  sanity_session.png    the per-participant session measures
  sanity_curves.png     RT and F1 over trials
  sanity.csv            every value, and sanity_tests.csv the tests

  ~/miniforge3/envs/kernelbehav/bin/python B09_sanity_a_vs_b.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helpers.data import session_files
from helpers.measures import checks, moving, rule_type, scores
from helpers.plots import apply_dark_theme, band, save, version_box
from helpers.sessions import session_info
from helpers.stats import perm_diff
from helpers.study import Study, parser
from params import FAST_RT_MS, N_PERM_SANITY, TEST_TYPES, VERSION_COLORS, VERSIONS, WINDOW

BLOCK_MEASURES = {"accuracy": "Accuracy", "f1": "F1", "rt_median": "Median RT (ms)", "p_f": "P(F)",
                  "late": "Late (no answer)", "fast": f"Fast (RT < {FAST_RT_MS} ms)", "p_repeat": "P(same key as previous)"}
SESSION_MEASURES = {"total_min": "Session (min)", "setup_min": "Setup (min)", "practice1_min": "Practice I (min)",
                    "practice2_min": "Practice XOR (min)", "break_min": "Breaks (min)",
                    "practice1_trials": "Practice I trials", "practice2_trials": "Practice XOR trials",
                    "practice1_acc": "Practice I accuracy", "practice2_acc": "Practice XOR accuracy",
                    "view_dist_cm": "Viewing distance (cm)", "quiz_attempts": "Quiz attempts",
                    "blur_count": "Window switches"}


def main():
    p = parser(__doc__)
    p.add_argument("--window", type=int, default=WINDOW, help="trials in the centred moving window")
    p.add_argument("--fast", type=float, default=FAST_RT_MS, help="RT (ms) below which an answer counts as fast")
    args = p.parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    study.save_exclusions()
    groups = study.groups()
    kept = {pid: v for v in VERSIONS for pid in groups[v]}

    block_rows, curves = [], {}
    for pid, version in kept.items():
        for _, rule, b in study.test_blocks(pid):
            typ = rule_type(rule)
            s = scores(b)
            block_rows.append({"participant": pid, "version": version, "test_block": typ, "accuracy": s["accuracy"],
                               "f1": s["f1"], "rt_median": b["rt"].median(), **checks(b, args.fast)})
            m = moving(b, args.window)
            curves[(pid, typ)] = (version, m["rt"].to_numpy(), m["f1"].to_numpy())
    blocks_df = pd.DataFrame(block_rows)

    session_rows = []
    for f in session_files():
        df = pd.read_csv(f, low_memory=False)
        pid = str(df["prolific_pid"].dropna().iloc[0]) if df["prolific_pid"].notna().any() else ""
        if pid not in kept or not df["part"].eq("final").any():
            continue
        info = session_info(df)
        info["break_min"] = sum(v for k, v in info.items() if k.endswith("_break_min") and pd.notna(v))
        session_rows.append({"participant": pid, "version": info["version"],
                             **{k: info.get(k, np.nan) for k in SESSION_MEASURES}})
    session_df = pd.DataFrame(session_rows)
    n = {v: len(groups[v]) for v in VERSIONS}
    tag = study.tag(groups)

    tests = []
    for typ in TEST_TYPES:
        s = blocks_df[blocks_df["test_block"] == typ]
        for col in BLOCK_MEASURES:
            a, b = s.loc[s["version"] == "A", col].dropna(), s.loc[s["version"] == "B", col].dropna()
            diff, p_ = perm_diff(a, b, n=N_PERM_SANITY)
            tests.append({"measure": col, "test_block": typ, "A": a.mean(), "B": b.mean(), "A - B": diff, "p": p_})
    for col in SESSION_MEASURES:
        a = session_df.loc[session_df["version"] == "A", col].dropna().astype(float)
        b = session_df.loc[session_df["version"] == "B", col].dropna().astype(float)
        diff, p_ = perm_diff(a, b, n=N_PERM_SANITY)
        tests.append({"measure": col, "test_block": "session", "A": a.mean(), "B": b.mean(), "A - B": diff, "p": p_})
    tests = pd.DataFrame(tests)
    out = study.out("a_vs_b")
    blocks_df.merge(session_df, on=["participant", "version"], how="left").to_csv(out / "sanity.csv", index=False)
    tests.to_csv(out / "sanity_tests.csv", index=False)

    def p_of(col, typ):
        return tests[(tests["measure"] == col) & (tests["test_block"] == typ)]["p"].iloc[0]

    # ------------------------------------------------ per-block measures
    fig, axes = plt.subplots(len(TEST_TYPES), len(BLOCK_MEASURES), figsize=(2.3 * len(BLOCK_MEASURES), 8.6))
    for r, typ in enumerate(TEST_TYPES):
        s = blocks_df[blocks_df["test_block"] == typ]
        for c, (col, title) in enumerate(BLOCK_MEASURES.items()):
            version_box(axes[r, c], s, col, f"{typ}: {title}", p_of(col, typ))
    fig.suptitle(f"Sanity measures per test block, version A vs B ({tag}; red title: p < .05, uncorrected)",
                 fontsize=11)
    save(fig, out / "sanity_blocks.png", dpi=130, tight_rect=(0, 0, 1, 0.96))

    # ------------------------------------------------ session measures
    cols = 6
    fig, axes = plt.subplots(int(np.ceil(len(SESSION_MEASURES) / cols)), cols,
                             figsize=(2.4 * cols, 3.1 * np.ceil(len(SESSION_MEASURES) / cols)))
    for ax, (col, title) in zip(axes.ravel(), SESSION_MEASURES.items()):
        version_box(ax, session_df.astype({col: float}), col, title, p_of(col, "session"))
    for ax in axes.ravel()[len(SESSION_MEASURES):]:
        ax.axis("off")
    fig.suptitle(f"Session measures, version A vs B ({tag}; red title: p < .05, uncorrected)", fontsize=11)
    save(fig, out / "sanity_session.png", dpi=130, tight_rect=(0, 0, 1, 0.94))

    # ------------------------------------------------ RT and F1 over trials
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), sharex=True)
    for c, typ in enumerate(TEST_TYPES):
        for v in VERSIONS:
            rt = np.array([cv[1] for (_, t), cv in curves.items() if t == typ and cv[0] == v])
            f1 = np.array([cv[2] for (_, t), cv in curves.items() if t == typ and cv[0] == v])
            band(axes[0, c], rt, VERSION_COLORS[v], f"version {v} (n={n[v]})")
            band(axes[1, c], f1, VERSION_COLORS[v], f"version {v}")
        axes[0, c].set_title(f"{typ}: RT (median RT A vs B: p = {p_of('rt_median', typ):.2f})", fontsize=10)
        axes[1, c].set_title(f"{typ}: F1 (A vs B: p = {p_of('f1', typ):.2f})", fontsize=10)
        axes[1, c].axhline(0.5, color="0.45", lw=0.7, ls=":")
        axes[1, c].set_ylim(0.2, 1)
        axes[1, c].set_xlabel("Trial")
    axes[0, 0].set_ylabel("RT (ms)")
    axes[1, 0].set_ylabel("F1")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"RT and F1 over trials, version A vs B ({tag}; {args.window}-trial window, mean +- SEM)",
                 fontsize=11)
    save(fig, out / "sanity_curves.png", dpi=130, tight_rect=(0, 0, 1, 0.94))

    print(tests.round(3).to_string(index=False))
    print(f"-> {out}/sanity_blocks.png, sanity_session.png, sanity_curves.png, sanity.csv, sanity_tests.csv")


if __name__ == "__main__":
    main()
