"""Learning curves of the slot-machine task: performance and RT over rounds, block by block.

For every participant, one column per block:
  top     running accuracy, hit rate, miss rate, false-alarm rate and F1 over the last
          --window rounds ("win" pairs are the signal, YES the positive answer; see
          load_data.py), chance at 0.5; for practice blocks the round where the criterion
          was met is marked
  bottom  RT of every round (correct / wrong), late rounds at the deadline, running median

Outputs, in ~/Documents/data/catlearn_eeg/catlearn_4x4_prolific/analysis/:
  <participant>/learning_curves.png
  <participant>/block_summary.csv    per block: n, accuracy, hit / miss / FA rates, F1, late, RT
  block_summary_all.csv               the same for every participant
  sessions.csv                        one row per participant (design, calibration, bonus...)

  ~/miniforge3/envs/kernelbehav/bin/python B01_learning_curves.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_data import DATA_DIR, OUT_DIR, load_all, sdt_rates
from style import ACCENT, CORRECT, FOREGROUND, INCORRECT, apply_dark_theme

RATE_STYLES = {
    "accuracy": dict(color=FOREGROUND, lw=2.0, label="accuracy"),
    "hit_rate": dict(color="#2ecc40", lw=1.4, label="hit rate"),
    "miss_rate": dict(color=ACCENT, lw=1.2, ls="--", label="miss rate"),
    "fa_rate": dict(color=INCORRECT, lw=1.4, label="false-alarm rate"),
    "f1": dict(color=CORRECT, lw=1.4, ls=":", label="F1"),
}
RESPONSE_DEADLINE_MS = 4000


def rule_label(rule):
    """'x4_XII_A' -> '4x4 type XII (A)', 'x2_XOR' -> '2x2 XOR'."""
    size, *rest = rule.split("_")
    size = f"{size[1]}x{size[1]}"
    if rest[0] == "XOR":
        return f"{size} XOR"
    version = f" ({rest[1]})" if len(rest) > 1 else ""
    return f"{size} type {rest[0]}{version}"


def running_rates(b, window):
    """Running accuracy, hit / miss / FA rates and F1 over the last `window` rounds of a block."""
    roll = lambda col: b[col].astype(float).rolling(window, min_periods=1).sum()   # noqa: E731
    n = pd.Series(1.0, index=b.index).rolling(window, min_periods=1).sum()
    hits, misses, fas = roll("hit"), roll("miss"), roll("fa")
    n_win = b["category"].eq(1).astype(float).rolling(window, min_periods=1).sum()
    n_lose = b["category"].eq(0).astype(float).rolling(window, min_periods=1).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({
            "accuracy": roll("correct") / n,
            "hit_rate": hits / n_win.replace(0, np.nan),
            "miss_rate": misses / n_win.replace(0, np.nan),
            "fa_rate": fas / n_lose.replace(0, np.nan),
            "f1": 2 * hits / (2 * hits + misses + fas).replace(0, np.nan),
        })


def criterion_round(b, window=10, min_correct=8):
    """Round at which the practice criterion was met (None if never)."""
    ok = b["correct"].astype(int).rolling(window).sum() >= min_correct
    return int(b.loc[ok, "trial_in_block"].iloc[0]) if ok.any() else None


def plot_participant(r, session, window, out_dir):
    blocks = sorted(r["block"].unique())
    fig, axes = plt.subplots(2, len(blocks), figsize=(3.6 * len(blocks) + 1, 6.4), sharey="row",
                             gridspec_kw={"height_ratios": [1.3, 1]}, squeeze=False)
    for j, blk in enumerate(blocks):
        b = r[r["block"] == blk].sort_values("trial_in_block").reset_index(drop=True)
        x = b["trial_in_block"].to_numpy()
        ax, ax_rt = axes[0, j], axes[1, j]

        rates = running_rates(b, window)
        for col, style in RATE_STYLES.items():
            ax.plot(x, rates[col], **style)
        ax.axhline(0.5, color="0.4", lw=0.8)
        ax.set_ylim(-0.03, 1.03)
        if b["phase"].iloc[0] == "practice":
            c = criterion_round(b)
            if c is not None:
                ax.axvline(c, color=ACCENT, lw=1, ls=":")
                ax.text(c, 1.02, " criterion", color=ACCENT, fontsize=7, va="bottom")
        s = sdt_rates(b)
        ax.set_title(f"block {blk}: {rule_label(b['rule'].iloc[0])}\n"
                     f"acc {s['accuracy']:.2f}  hit {s['hit_rate']:.2f}  FA {s['fa_rate']:.2f}  F1 {s['f1']:.2f}",
                     fontsize=9)

        answered = b[~b["timeout"]]
        ax_rt.scatter(answered["trial_in_block"], answered["rt"], s=14, zorder=2,
                      c=np.where(answered["correct"], CORRECT, INCORRECT))
        late = b[b["timeout"]]
        ax_rt.scatter(late["trial_in_block"], np.full(len(late), RESPONSE_DEADLINE_MS), marker="x", s=20,
                      color=INCORRECT, zorder=2)
        rt = b["rt"].where(~b["timeout"])
        ax_rt.plot(x, rt.rolling(window, min_periods=1).median(), color=FOREGROUND, lw=1.5)
        ax_rt.axhline(RESPONSE_DEADLINE_MS, color="0.35", lw=0.8, ls="--")
        ax_rt.set_ylim(0, RESPONSE_DEADLINE_MS * 1.05)
        ax_rt.set_xlabel("round in block")
    axes[0, 0].set_ylabel(f"rate (running, {window} rounds)")
    axes[1, 0].set_ylabel("RT (ms)")
    axes[0, -1].legend(frameon=False, fontsize=7, loc="lower right")
    axes[1, -1].scatter([], [], c=CORRECT, s=14, label="correct")
    axes[1, -1].scatter([], [], c=INCORRECT, s=14, label="wrong")
    axes[1, -1].scatter([], [], c=INCORRECT, marker="x", s=20, label="late")
    axes[1, -1].plot([], [], color=FOREGROUND, label="running median")
    axes[1, -1].legend(frameon=False, fontsize=7, loc="upper right")
    if isinstance(session.get("version"), str):            # F/J study: fixed sequence
        title = f"{session['participant']}: {session['sequence']}, version {session['version']}"
    else:
        title = (f"{session['participant']}: test type {session['test_type']} ({session['test_first']} first), "
                 f"YES = {str(session['key_yes']).upper()}")
    if session["debug"]:
        title += "   [DEBUG RUN: shortened blocks]"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)


def block_summary(r):
    rows = []
    for (pid, blk), b in r.groupby(["participant", "block"]):
        rows.append({"participant": pid, "block": blk, "phase": b["phase"].iloc[0], "rule": b["rule"].iloc[0],
                     "rule_type": b["rule_type"].iloc[0], "size": b["size"].iloc[0], **sdt_rates(b)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-sync", action="store_true", help="skip copying new files from the DataPipe Drive folder")
    parser.add_argument("--window", type=int, default=8, help="running window, rounds")
    args = parser.parse_args()
    apply_dark_theme()

    rounds, sessions = load_all(args.data_dir, sync=not args.no_sync)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = block_summary(rounds)
    summary.to_csv(OUT_DIR / "block_summary_all.csv", index=False)
    sessions.to_csv(OUT_DIR / "sessions.csv", index=False)
    for _, session in sessions.iterrows():
        pid = session["participant"]
        out = OUT_DIR / pid
        plot_participant(rounds[rounds["participant"] == pid], session, args.window, out)
        summary[summary["participant"] == pid].to_csv(out / "block_summary.csv", index=False)
        design = (f"{session['sequence']} ({session['version']})" if isinstance(session.get("version"), str)
                  else f"test type {session['test_type']}")
        print(f"{pid}{' (debug run)' if session['debug'] else ''}: {design}, "
              f"{session['n_rounds']} rounds")
        cols = ["block", "rule", "n", "accuracy", "hit_rate", "miss_rate", "fa_rate", "f1", "late_rate", "rt_median"]
        print(summary.loc[summary["participant"] == pid, cols].round(2).to_string(index=False))
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
