"""Learning curves (accuracy and reaction time over trials) for one subject in the 2x2 task.

Each panel shows single trials plus a running average, and binned means with error
bars (accuracy: binomial SE; RT: SE of the mean over correct and incorrect trials).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from load_data import load_subject  # noqa: E402
from plot_style import ACCENT, CORRECT, INCORRECT, apply_dark_theme  # noqa: E402

OUT_DIR = Path("/Users/mrugank/Documents/data/catlearn_eeg/behav_analyses/task_2by2")


def binned(df, column, bin_size):
    bins = (df["trial_overall"] - 1) // bin_size
    grouped = df.groupby(bins)[column]
    centers = df.groupby(bins)["trial_overall"].mean()
    return centers, grouped.mean(), grouped.sem()


def plot_learning_curves(df, window, bin_size, out_path):
    subject, task, config = df["subject"].iloc[0], df["task"].iloc[0], df["config"].iloc[0]
    trials = df["trial_overall"]
    fig, (ax_acc, ax_rt) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Accuracy: single trials as ticks at 0/1, running mean, binned mean
    acc = df["correct"].astype(float)
    ax_acc.scatter(trials, acc, c=np.where(df["correct"], CORRECT, INCORRECT), s=14, zorder=2)
    running = acc.rolling(window, min_periods=1).mean()
    ax_acc.plot(trials, running, color="white", lw=1.5, label=f"running mean ({window} trials)")
    x, m, _ = binned(df, "correct", bin_size)
    se = np.sqrt(m * (1 - m) / bin_size)
    ax_acc.errorbar(x, m, yerr=se, color=ACCENT, marker="o", capsize=3, lw=2, label=f"{bin_size}-trial bins")
    ax_acc.axhline(0.5, color="0.5", ls="--", lw=1, label="chance")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title(f"sub{subject} | task {task} | {config} | overall accuracy {acc.mean():.0%}")
    ax_acc.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)

    # Reaction time: single trials colored by outcome, running median, binned mean
    ax_rt.scatter(trials, df["rt"], c=np.where(df["correct"], CORRECT, INCORRECT), s=14, zorder=2)
    ax_rt.plot(trials, df["rt"].rolling(window, min_periods=1).median(), color="white", lw=1.5,
               label=f"running median ({window} trials)")
    x, m, se = binned(df, "rt", bin_size)
    ax_rt.errorbar(x, m, yerr=se, color=ACCENT, marker="o", capsize=3, lw=2, label=f"{bin_size}-trial bins")
    ax_rt.scatter([], [], c=CORRECT, s=14, label="correct")
    ax_rt.scatter([], [], c=INCORRECT, s=14, label="incorrect")
    ax_rt.set_ylabel("reaction time (s)")
    ax_rt.set_xlabel("trial")
    ax_rt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)

    # Block boundaries, if there is more than one block
    for start in df.loc[df["trial"] == 1, "trial_overall"].iloc[1:]:
        for ax in (ax_acc, ax_rt):
            ax.axvline(start - 0.5, color="0.4", lw=1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", default="01")
    parser.add_argument("--window", type=int, default=8, help="running-average window (trials)")
    parser.add_argument("--bin-size", type=int, default=10, help="trials per bin")
    args = parser.parse_args()
    apply_dark_theme()

    df = load_subject(args.subject)
    out_path = OUT_DIR / f"sub{args.subject}" / f"learning_curves_sub{args.subject}.png"
    plot_learning_curves(df, args.window, args.bin_size, out_path)

    n_out = df["timed_out"].sum()
    print(f"sub{args.subject}: {len(df)} trials in {df['block'].nunique()} block(s), "
          f"accuracy {df['correct'].mean():.0%}, median RT {df['rt'].median():.3f} s, {n_out} timeouts")
    print(df.groupby((df["trial_overall"] - 1) // args.bin_size)
          .agg(accuracy=("correct", "mean"), median_rt=("rt", "median")).round(3).to_string())
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
