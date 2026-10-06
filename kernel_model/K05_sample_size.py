"""What can the pilot (10 per rule type) and the main study (30 per type) detect?

Simulated participants run the 4x4 slot design (K04_slot4x4.py: 2x2 practice to criterion,
then 3 test blocks A, A, B of 128 trials; B, B, A is the mirror image). Unlike K04 each
participant has their own learning rate, choice consistency and (adaptive learner) speed
of attention narrowing, and every answer is sampled (right / wrong), as in real data.

For each rule type a pool of participants is simulated once per learner (fixed / adaptive
kernel); experiments of n per type are then drawn from the pools many times, and each
effect is tested as it would be in the data:
  switch cost       block 3 (B version) below block 2 (A version), within participants,
                    all types pooled; and within type X alone
  learning to learn block 2 above block 1 (same rule, new fractals), all types pooled
  type differences  block-1 accuracy differs across the 7 types (one-way ANOVA)
Power = share of simulated experiments with p < .05 (one-sided for the directional
effects). Under the fixed kernel the switch cost and learning to learn are zero, so their
"power" there is the false-positive rate.

Outputs go to ~/Documents/data/catlearn_eeg/kernel_model/4x4/slot_design/:
  sample_size.png        power vs participants per type, per effect and learner
  sample_size.csv        the same numbers, plus mean effect sizes
  participant_pool.npz   simulated block accuracies of every pool participant

  ~/miniforge3/envs/kernelbehav/bin/python K05_sample_size.py
"""

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from K04_slot4x4 import TEST_TYPES, run_block, test_tasks, transfer
from kernel_modes import DATA_ROOT, Design
from plot_style import FOREGROUND, apply_dark_theme  # from task_design, via kernel_modes' sys.path


def simulate_pool(d2, d4, typ, adaptive, n, args, rng):
    """Block accuracies (n, 3) of n simulated participants of one rule type."""
    lr = np.exp(rng.normal(np.log(args.lr), args.sd_log_lr, n))
    beta = np.exp(rng.normal(np.log(args.beta), args.sd_log_beta, n))
    adapt = np.exp(rng.normal(np.log(args.adapt_rate), args.sd_log_adapt, n)) if adaptive else np.zeros(n)
    relax = np.full(n, args.relax_rate if adaptive else 0.0)
    w0_2, w0_4 = d2.prior_weights(args.order_decay), d4.prior_weights(args.order_decay)
    kw = dict(lr=lr, beta=beta, adapt=adapt, relax=relax)
    w = np.tile(w0_2, (n, 1))
    for task in (d2.task_index("A"), d2.task_index("AB")):
        w, _, _ = run_block(d2, d2.task_labels[task], w, rng, args.practice_reps, w0=w0_2,
                            criterion=tuple(args.criterion), **kw)
    w = transfer(w, d2, d4)
    a, b = test_tasks(d4)[typ]
    acc = []
    for task in (a, a, b):
        w, p, _ = run_block(d4, d4.task_labels[task], w, rng, args.test_reps, w0=w0_4, **kw)
        acc.append((rng.random(p.shape) < p).mean(1))     # sampled answers, as in real data
    return np.stack(acc, axis=1)


def power(pools, n, n_exp, rng):
    """Share of simulated experiments (n per type) in which each effect is significant."""
    hits = {"switch cost (all types)": 0, "switch cost (type X)": 0,
            "learning to learn (all types)": 0, "type differences (block 1)": 0}
    for _ in range(n_exp):
        sample = {t: pools[t][rng.choice(len(pools[t]), n, replace=False)] for t in TEST_TYPES}
        allp = np.concatenate(list(sample.values()))
        switch = allp[:, 2] - allp[:, 1]
        gain = allp[:, 1] - allp[:, 0]
        hits["switch cost (all types)"] += stats.ttest_1samp(switch, 0, alternative="less").pvalue < 0.05
        x = sample["X"]
        hits["switch cost (type X)"] += stats.ttest_1samp(x[:, 2] - x[:, 1], 0, alternative="less").pvalue < 0.05
        hits["learning to learn (all types)"] += stats.ttest_1samp(gain, 0, alternative="greater").pvalue < 0.05
        hits["type differences (block 1)"] += stats.f_oneway(*[s[:, 0] for s in sample.values()]).pvalue < 0.05
    return {k: v / n_exp for k, v in hits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=int, default=400, help="simulated participants per type and learner")
    parser.add_argument("--n-per-type", type=int, nargs="+", default=[5, 10, 15, 20, 30, 40])
    parser.add_argument("--n-exp", type=int, default=2000, help="simulated experiments per sample size")
    parser.add_argument("--test-reps", type=int, default=8)
    parser.add_argument("--practice-reps", type=int, default=20)
    parser.add_argument("--criterion", type=int, nargs=2, default=[10, 8])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--adapt-rate", type=float, default=0.05)
    parser.add_argument("--relax-rate", type=float, default=0.005)
    parser.add_argument("--order-decay", type=float, default=0.5)
    parser.add_argument("--sd-log-lr", type=float, default=0.4, help="individual differences (log scale)")
    parser.add_argument("--sd-log-beta", type=float, default=0.3)
    parser.add_argument("--sd-log-adapt", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    apply_dark_theme()

    d2, d4 = Design("2x2"), Design("4x4")
    out_dir = DATA_ROOT / "kernel_model" / "4x4" / "slot_design"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pools, rows, saved = {}, [], {}
    for learner in ("fixed", "adaptive"):
        pools[learner] = {t: simulate_pool(d2, d4, t, learner == "adaptive", args.pool, args, rng) for t in TEST_TYPES}
        allp = np.concatenate(list(pools[learner].values()))
        sw, gn = allp[:, 2] - allp[:, 1], allp[:, 1] - allp[:, 0]
        print(f"{learner}: switch cost mean {sw.mean():+.3f} (sd {sw.std():.3f}, d = {sw.mean() / sw.std():+.2f}); "
              f"learning to learn {gn.mean():+.3f} (sd {gn.std():.3f}); block-1 accuracy by type: "
              + " ".join(f"{t} {pools[learner][t][:, 0].mean():.2f}" for t in TEST_TYPES))
        for t in TEST_TYPES:
            saved[f"{learner}/{t}"] = pools[learner][t]
        for n in args.n_per_type:
            pw = power(pools[learner], n, args.n_exp, rng)
            for effect, value in pw.items():
                rows.append({"learner": learner, "n_per_type": n, "n_total": 7 * n, "effect": effect,
                             "power": round(value, 3)})
            print(f"  n = {n:2d} per type ({7 * n:3d} total): "
                  + "  ".join(f"{k}: {v:.2f}" for k, v in pw.items()))

    effects = list(dict.fromkeys(r["effect"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, learner in zip(axes, ("fixed", "adaptive")):
        for effect, col in zip(effects, ("#ffa630", "#c77dff", "#4cc9f0", "#2ec4b6")):
            xs = [r["n_per_type"] for r in rows if r["learner"] == learner and r["effect"] == effect]
            ys = [r["power"] for r in rows if r["learner"] == learner and r["effect"] == effect]
            ax.plot(xs, ys, "o-", color=col, label=effect)
        for n, lab in ((10, "pilot"), (30, "main")):
            ax.axvline(n, color="0.45", ls=":", lw=1)
            ax.text(n, 1.02, lab, ha="center", va="bottom", color=FOREGROUND, fontsize=9)
        ax.axhline(0.8, color="0.4", lw=0.7)
        ax.axhline(0.05, color="0.3", lw=0.7, ls="--")
        ax.set_title(f"{learner} kernel", fontsize=10, pad=16)
        ax.set_xlabel("participants per rule type (7 types)")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("power (p < .05)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Kernel-model participants: what each sample size detects "
                 "(fixed kernel: switch cost and learning to learn are false positives)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_size.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "sample_size.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    np.savez(out_dir / "participant_pool.npz", params=json.dumps(vars(args)), test_types=TEST_TYPES, **saved)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
