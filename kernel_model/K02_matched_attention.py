"""Same learning, different attention: two kernel learners with matched first-block curves.

The 2x2 task with new stimuli every block (rule sequences as in K01). Two learners:

  focused  adaptive kernel: weights drift toward what has been learned, so attention
           narrows onto the rule's location(s)
  spread   fixed kernel (attention stays on A and B alike), with its learning rate chosen
           so its block-1 accuracy curve on rule A matches the focused learner's

Their first A block looks the same behaviourally; they differ in attention and in what
happens when the rule repeats or switches. Attention proxy: kernel weight on each
location's features, attn_A = w_A + w_AB, attn_B = w_B + w_AB, and the index
(attn_A - attn_B) / (attn_A + attn_B) is what the 12 / 15 Hz SSVEP contrast should track.

Outputs go to ~/Documents/data/catlearn_eeg/kernel_model/2x2/matched_attention/:
  matched_attention.png   accuracy and attention index per sequence, both learners
  predictions.json        run-averaged curves per sequence and learner, and all parameters
                          (read by the slide builders)
  block_summary.csv       mean accuracy in the first / last 10 trials of every block

  ~/miniforge3/envs/kernelbehav/bin/python K02_matched_attention.py
"""

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from kernel_modes import DATA_ROOT, Design, simulate_blocks
from plot_style import FOREGROUND, apply_dark_theme  # from task_design, via kernel_modes' sys.path

SEQUENCES = [["A", "A", "AB"], ["A", "B", "AB"], ["AB", "AB", "A"]]
RULE_COLORS = {"A": "#ffa630", "B": "#c77dff", "AB": "#2ec4b6"}
FOCUSED_COLOR = "#ffa630"


def match_lr(design, w0, focused_kw, common, grid):
    """Fixed-kernel learning rate whose block-1 rule-A curve best matches the focused learner."""
    task = [design.task_index("A")]
    target = simulate_blocks(design, task, w0, seed=1, **focused_kw, **common)["p_correct"].mean(0)
    errs = []
    for lr in grid:
        curve = simulate_blocks(design, task, w0, lr=lr, seed=1, **common)["p_correct"].mean(0)
        errs.append(np.abs(curve - target).mean())
    best = int(np.argmin(errs))
    return float(grid[best]), float(errs[best])


def attention_index(w):
    attn_a, attn_b = w[:, 1] + w[:, 3], w[:, 2] + w[:, 3]
    return (attn_a - attn_b) / (attn_a + attn_b)


def plot(results, n_trials_block, out_dir):
    fig, axes = plt.subplots(len(SEQUENCES), 2, figsize=(11, 2.6 * len(SEQUENCES)), sharex=True, squeeze=False)
    styles = {"focused": dict(color=FOCUSED_COLOR, lw=2), "spread": dict(color=FOREGROUND, lw=1.5, ls="--")}
    for r, seq in enumerate(SEQUENCES):
        key = "-".join(seq)
        for learner, st in styles.items():
            res = results[key][learner]
            axes[r, 0].plot(res["acc"], label=learner, **st)
            axes[r, 1].plot(res["attn_index"], **st)
        for ax in axes[r]:
            for b in range(1, len(seq)):
                ax.axvline(b * n_trials_block - 0.5, color="0.35", ls=":", lw=0.8)
            for b, rule in enumerate(seq):
                ax.text((b + 0.5) * n_trials_block, 1.02, rule, transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", color=RULE_COLORS[rule], fontweight="bold")
        axes[r, 0].axhline(0.5, color="0.4", lw=0.7)
        axes[r, 0].set_ylim(0.45, 1.0)
        axes[r, 0].set_ylabel(f"{key}\np(correct)")
        axes[r, 1].axhline(0, color="0.4", lw=0.7)
        axes[r, 1].set_ylim(-1, 1)
        axes[r, 1].set_ylabel("attention index\n(+1 = A, -1 = B)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("trial")
    fig.suptitle("focused (adaptive kernel) vs spread (fixed kernel, learning rate matched in block 1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "matched_attention.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-reps", type=int, default=10, help="passes through the 4 compounds per block (10 -> 40 trials)")
    parser.add_argument("--n-runs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1, help="focused learner's learning rate")
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--order-decay", type=float, default=0.5)
    parser.add_argument("--adapt-rate", type=float, default=0.05)
    parser.add_argument("--relax-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    apply_dark_theme()

    design = Design("2x2")
    w0 = design.prior_weights(args.order_decay)
    out_dir = DATA_ROOT / "kernel_model" / "2x2" / "matched_attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_trials_block = design.n_stim * args.n_reps
    common = dict(n_reps=args.n_reps, beta=args.beta, n_runs=args.n_runs)
    focused_kw = dict(lr=args.lr, adapt_rate=args.adapt_rate, relax_rate=args.relax_rate)
    spread_lr, err = match_lr(design, w0, focused_kw, common, np.round(np.arange(0.10, 0.40, 0.01), 2))
    learners = {"focused": focused_kw, "spread": dict(lr=spread_lr, adapt_rate=0.0, relax_rate=0.0)}
    print(f"spread learner lr {spread_lr:.2f} matches the focused block-1 A curve (mean |diff| {err:.3f})")

    results, rows = {}, []
    for seq in SEQUENCES:
        key = "-".join(seq)
        results[key] = {}
        for learner, kw in learners.items():
            r = simulate_blocks(design, [design.task_index(t) for t in seq], w0, seed=args.seed, **kw, **common)
            w = r["weights"].mean(0)
            acc = r["p_correct"].mean(0)
            results[key][learner] = {"acc": acc.round(4).tolist(), "attn_index": attention_index(w).round(4).tolist(),
                                     "weights": w.round(4).tolist()}
            blocks = acc.reshape(len(seq), n_trials_block)
            for b, rule in enumerate(seq):
                rows.append({"sequence": key, "learner": learner, "block": b + 1, "rule": rule,
                             "acc_first10": round(blocks[b, :10].mean(), 4), "acc_last10": round(blocks[b, -10:].mean(), 4),
                             "attn_index_end": round(float(attention_index(w)[(b + 1) * n_trials_block - 1]), 4)})
            print(f"{key:8s} {learner:8s} first/last 10: "
                  + "  ".join(f"B{b + 1} {blocks[b, :10].mean():.2f}/{blocks[b, -10:].mean():.2f}" for b in range(len(seq))))

    plot(results, n_trials_block, out_dir)
    out = {"design": "2x2", "trials_per_block": n_trials_block, "mode_names": design.mode_names,
           "prior_weights": w0.round(4).tolist(), "params": vars(args),
           "learners": learners, "spread_lr_match_error": err, "seqs": results}
    (out_dir / "predictions.json").write_text(json.dumps(out))
    with open(out_dir / "block_summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
