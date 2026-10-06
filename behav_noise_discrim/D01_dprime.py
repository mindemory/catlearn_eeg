"""Sensitivity (d') and bias (c) at every alpha, per participant and for the group.

Per participant and alpha, from trials with an answer and a good exposure (timing_ok):
  H = P(said different | different pair),  F = P(said different | same pair), each with the
      log-linear correction (k + 0.5) / (n + 1), so perfect rates give finite z-scores
  d_yn  = z(H) - z(F)              yes/no d': the usual index, no model of the task
  c     = -(z(H) + z(F)) / 2       bias: > 0 = tends to say "same", < 0 = "different"
  d_sd  d' under the differencing model of same-different (Macmillan & Creelman, 2005):
        the observer says "different" when |x_left - x_right| > k, so
        F = 2 Phi(-k / sqrt 2),  H = Phi((d - k) / sqrt 2) + Phi((-d - k) / sqrt 2).
        It is never below 0, so near chance it is biased upward (with 20 + 20 trials, a
        true d' of ~0.1 reads ~0.6 on average); use d_yn to judge whether an alpha is at chance
  pc, rt_median, late_rate
  d_lo, d_hi  95% interval of d_yn (parametric bootstrap of the hit and false-alarm counts);
        with 10 + 10 trials per alpha a single alpha's d' is roughly +-0.55, so the intervals
        are wide; pool neighbouring alphas for precise statements
Nothing assumes how d' depends on alpha; the curve is plotted as measured.

Exclusion flags (decide before data collection; nothing is dropped here):
  at_chance    not above chance over the upper half of the alphas (alpha >= 2.22, 200
               trials): one-sided binomial test, p >= .05. No alpha is easy at 100 ms (the
               first full pilot reached d' ~1, 71% correct, above alpha 1.33), so this checks
               that the participant did the task at all, without assuming any level is easy
  quiz_failed  comprehension check never passed
  bad_timing   > 10% of trials with an unknown or wrong exposure

Outputs (OUT_DIR = ~/Documents/data/catlearn_eeg/noise_discrim_prolific/analysis/):
  dprime_by_alpha.csv        participant x alpha
  participants.csv           sessions table with the exclusion flags
  dprime_group.png           group mean +- SEM of d_yn, d_sd, c and pc, participants as thin lines
  <participant>/dprime.png   one participant: hit and false-alarm rates, d', c, RT
  <participant>/curves.png   one participant: accuracy (all, same, different) and d' by alpha
  <participant>/learning.png one participant: accuracy by block, and d' by alpha in the first vs
                             second half of the blocks (practice trials are never included)

  ~/miniforge3/envs/kernelbehav/bin/python D01_dprime.py      (copies new files from Drive first)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binomtest, norm

from load_data import DATA_DIR, DRIVE_DIR, MIN_VERSION, OUT_DIR, load_all, sync_from_drive
from style import ACCENT, CORRECT, FOREGROUND, INCORRECT, apply_dark_theme

UPPER_ALPHA = 2.2   # the upper half of the alphas: 2.22 to 4.0
CHANCE_P = 0.05
BAD_TIMING_MAX = 0.10


def d_differencing(h, f):
    """d' of the differencing model from hit and false-alarm rates (0 if H <= F)"""
    if h <= f:
        return 0.0
    k = -np.sqrt(2) * norm.ppf(f / 2)
    hit = lambda d: norm.cdf((d - k) / np.sqrt(2)) + norm.cdf((-d - k) / np.sqrt(2)) - h  # noqa: E731
    return brentq(hit, 0, 20) if hit(20) > 0 else 20.0


def wilson(k, n, z=1.96):
    """95% Wilson interval of a proportion"""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    return centre - half, centre + half


def d_interval(sig, noise, n_boot=2000, seed=0):
    """95% interval of the yes/no d' (log-linear rates), resampling hits and false alarms"""
    rng = np.random.default_rng(seed)
    hs = (rng.binomial(len(sig), sig["said_different"].mean(), n_boot) + 0.5) / (len(sig) + 1)
    fs = (rng.binomial(len(noise), noise["said_different"].mean(), n_boot) + 0.5) / (len(noise) + 1)
    return tuple(np.percentile(norm.ppf(hs) - norm.ppf(fs), [2.5, 97.5]))


def by_alpha(trials):
    rows = []
    for (pid, level, alpha), g in trials.groupby(["participant", "alpha_level", "alpha"]):
        use = g[g["timing_ok"] & ~g["timeout"]]
        sig, noise = use[use["signal"]], use[~use["signal"]]
        h = (sig["said_different"].sum() + 0.5) / (len(sig) + 1)
        f = (noise["said_different"].sum() + 0.5) / (len(noise) + 1)
        zh, zf = norm.ppf(h), norm.ppf(f)
        d_lo, d_hi = d_interval(sig, noise) if len(sig) and len(noise) else (np.nan, np.nan)
        h_lo, h_hi = wilson(sig["said_different"].sum(), len(sig))
        f_lo, f_hi = wilson(noise["said_different"].sum(), len(noise))
        n_corr = int(use["correct"].sum())
        pc_lo, pc_hi = wilson(n_corr, len(use))
        rows.append({"participant": pid, "alpha_level": level, "alpha": alpha, "n_different": len(sig),
                     "n_same": len(noise), "H": h, "F": f, "H_lo": h_lo, "H_hi": h_hi,
                     "F_lo": f_lo, "F_hi": f_hi, "d_yn": zh - zf, "d_lo": d_lo, "d_hi": d_hi, "c": -(zh + zf) / 2,
                     "d_sd": d_differencing(h, f), "pc": use["correct"].mean(),
                     "pc_lo": pc_lo, "pc_hi": pc_hi, "pc_same": noise["correct"].mean(),
                     "pc_different": sig["correct"].mean(),
                     "rt_median": use["rt"].median(), "late_rate": g["timeout"].mean(),
                     "n_timing_bad": int((~g["timing_ok"]).sum())})
    return pd.DataFrame(rows)


def flag(sessions, trials):
    s = sessions.copy()
    upper = trials[(trials["alpha"] >= UPPER_ALPHA) & ~trials["timeout"]].groupby("participant")["correct"]
    k, n = upper.sum(), upper.count()
    s["pc_upper"] = s["participant"].map(k / n)
    s["p_upper"] = s["participant"].map({pid: binomtest(int(k[pid]), int(n[pid]), 0.5, alternative="greater").pvalue
                                          for pid in k.index})
    s["at_chance"] = s["p_upper"] >= CHANCE_P
    s["quiz_failed"] = s["comprehension_passed"].eq(False)
    s["bad_timing"] = s["timing_bad_rate"] > BAD_TIMING_MAX
    s["exclude"] = s[["at_chance", "quiz_failed", "bad_timing"]].any(axis=1)
    return s


def plot_participant(t, out_path, pid):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    a = t["alpha"]
    axes[0].errorbar(a, t["H"], yerr=[t["H"] - t["H_lo"], t["H_hi"] - t["H"]], fmt="o-", capsize=2,
                     color=CORRECT, label="hit: said different | different")
    axes[0].errorbar(a + 0.04, t["F"], yerr=[t["F"] - t["F_lo"], t["F_hi"] - t["F"]], fmt="o-", capsize=2,
                     color=INCORRECT, label="false alarm: said different | same")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("rate")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[1].errorbar(a, t["d_yn"], yerr=[t["d_yn"] - t["d_lo"], t["d_hi"] - t["d_yn"]], fmt="o-", capsize=2,
                     color=ACCENT, label="yes/no d' (95% interval)")
    axes[1].plot(a, t["d_sd"], "s--", color=FOREGROUND, alpha=0.7, label="differencing d'")
    axes[1].axhline(0, color=FOREGROUND, lw=0.6, alpha=0.4)
    axes[1].set_ylabel("d'")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(a, t["c"], "o-", color=FOREGROUND)
    axes[2].axhline(0, color=FOREGROUND, lw=0.6, alpha=0.4)
    axes[2].set_ylabel("c  (> 0: says 'same')")
    axes[3].plot(a, t["rt_median"], "o-", color=FOREGROUND)
    axes[3].set_ylabel("median RT (ms)")
    for ax in axes:
        ax.set_xlabel("alpha")
    n_s, n_d = int(t["n_same"].median()), int(t["n_different"].median())
    fig.suptitle(f"{pid}: same / different by alpha ({n_s} same + {n_d} different trials per alpha)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_curves(t, out_path, pid):
    """Accuracy and d' by alpha, with 95% intervals"""
    a = t["alpha"]
    n_s, n_d = int(t["n_same"].median()), int(t["n_different"].median())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(a, t["pc_different"], "o--", color=CORRECT, alpha=0.75, ms=4, label="different pairs (hit rate)")
    ax.plot(a, t["pc_same"], "o--", color=INCORRECT, alpha=0.75, ms=4, label="same pairs (1 - false-alarm rate)")
    ax.errorbar(a, t["pc"], yerr=[t["pc"] - t["pc_lo"], t["pc_hi"] - t["pc"]], fmt="o-", color=ACCENT, lw=2.2,
                capsize=3, label="all trials (95% interval)")
    ax.axhline(0.5, color=FOREGROUND, lw=0.7, ls=":", alpha=0.6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("proportion correct")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    ax.errorbar(a, t["d_yn"], yerr=[t["d_yn"] - t["d_lo"], t["d_hi"] - t["d_yn"]], fmt="o-", color=ACCENT, lw=2.2,
                capsize=3, label="yes/no d' (95% interval)")
    ax.plot(a, t["d_sd"], "s--", color=FOREGROUND, alpha=0.6, ms=4, label="differencing-model d'")
    ax.axhline(0, color=FOREGROUND, lw=0.7, ls=":", alpha=0.6)
    ax.set_ylabel("d'")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        ax.set_xlabel("alpha (spectral slope)")
        ax.set_xticks(a, [f"{v:.2f}" for v in a], fontsize=8)
    fig.suptitle(f"{pid}: accuracy and sensitivity by alpha ({n_s} same + {n_d} different trials per alpha)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_learning(trials, out_path, pid):
    """Does performance change over the session? Accuracy by block, and d' by alpha in the
    first vs the second half of the blocks"""
    blocks = sorted(trials["block"].unique())
    half = blocks[len(blocks) // 2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    use = trials[~trials["timeout"]]
    k, n = use.groupby("block")["correct"].sum(), use.groupby("block")["correct"].count()
    ci = np.array([wilson(k[b], n[b]) for b in k.index])
    p = k / n
    ax.errorbar(k.index, p, yerr=[p - ci[:, 0], ci[:, 1] - p], fmt="o-", color=ACCENT, capsize=3, lw=2)
    ax.axhline(0.5, color=FOREGROUND, lw=0.7, ls=":", alpha=0.6)
    ax.axvline(half - 0.5, color=FOREGROUND, lw=0.7, ls="--", alpha=0.4)
    ax.set_xticks(blocks)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("block")
    ax.set_ylabel("proportion correct (all alphas)")
    ax = axes[1]
    for label, sub, color, dx in ((f"{blocks[0]}-{half - 1}", trials[trials["block"] < half], FOREGROUND, -0.04),
                                  (f"{half}-{blocks[-1]}", trials[trials["block"] >= half], ACCENT, 0.04)):
        t = by_alpha(sub).sort_values("alpha")
        ax.errorbar(t["alpha"] + dx, t["d_yn"], yerr=[t["d_yn"] - t["d_lo"], t["d_hi"] - t["d_yn"]], fmt="o-",
                    color=color, capsize=2, label=f"blocks {label} ({int(t['n_same'].median())} + "
                                                  f"{int(t['n_different'].median())} trials per alpha)")
    ax.axhline(0, color=FOREGROUND, lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("alpha (spectral slope)")
    ax.set_ylabel("yes/no d' (95% interval)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{pid}: change over the session")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_group(table, sessions, out_path):
    keep = sessions.loc[~sessions["exclude"], "participant"]
    t = table[table["participant"].isin(keep)]
    n = t["participant"].nunique()
    panels = [("d_yn", "yes/no d'"), ("d_sd", "differencing d'"), ("c", "c  (> 0: says 'same')"), ("pc", "proportion correct")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, (col, label) in zip(axes, panels):
        for _, g in t.groupby("participant"):
            ax.plot(g["alpha"], g[col], color=FOREGROUND, lw=0.6, alpha=0.25)
        m = t.groupby("alpha")[col].agg(["mean", "sem"]).reset_index()
        ax.errorbar(m["alpha"], m["mean"], yerr=m["sem"], color=ACCENT, marker="o", capsize=3, lw=2)
        ax.axhline(0.5 if col == "pc" else 0, color=FOREGROUND, lw=0.6, alpha=0.4)
        ax.set_xlabel("alpha")
        ax.set_ylabel(label)
    fig.suptitle(f"Same / different by alpha: mean +- SEM of {n} participants (thin lines: each participant)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default: {OUT_DIR}")
    parser.add_argument("--include-debug", action="store_true", help="also analyse debug runs (?debug=1)")
    parser.add_argument("--drive-dir", type=Path, default=DRIVE_DIR, help="DataPipe's Google Drive folder")
    parser.add_argument("--no-sync", action="store_true", help="skip copying new files from the Drive folder")
    parser.add_argument("--include-partial", action="store_true", help="also analyse unfinished sessions")
    parser.add_argument("--min-version", default=MIN_VERSION, help="oldest task version to analyse (default %(default)s)")
    args = parser.parse_args()
    out = args.out_dir or args.data_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    apply_dark_theme()

    if not args.no_sync:
        copied = sync_from_drive(args.drive_dir, args.data_dir)
        print(f"copied {len(copied)} new or changed file(s) from {args.drive_dir}" + (f": {', '.join(copied)}" if copied else ""))
    trials, sessions = load_all(args.data_dir, include_debug=args.include_debug,
                                include_partial=args.include_partial, min_version=args.min_version)
    table = by_alpha(trials)
    sessions = flag(sessions, trials)
    table.round(4).to_csv(out / "dprime_by_alpha.csv", index=False)
    sessions.to_csv(out / "participants.csv", index=False)
    for pid, t in table.groupby("participant"):
        (out / str(pid)).mkdir(exist_ok=True)
        plot_participant(t.sort_values("alpha"), out / str(pid) / "dprime.png", pid)
        plot_curves(t.sort_values("alpha"), out / str(pid) / "curves.png", pid)
        plot_learning(trials[trials["participant"] == pid], out / str(pid) / "learning.png", pid)
    plot_group(table, sessions, out / "dprime_group.png")

    print(f"{sessions['participant'].nunique()} participants, {len(trials)} trials -> {out}")
    print(sessions[["participant", "n_trials", "finished", "pcorrect", "pc_upper", "p_upper", "late_rate", "timing_bad_rate",
                    "calibration_source", "exclude"]].round(3).to_string(index=False))
    print("\nGroup (included participants), by alpha:")
    keep = table["participant"].isin(sessions.loc[~sessions["exclude"], "participant"])
    print(table[keep].groupby("alpha")[["d_yn", "d_sd", "c", "pc", "rt_median"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
