"""Simulate kernel learners through a 3-block design and plot learning in feature modes.

Blocks 1 and 2 use a rule of the same type; block 3 switches type. Each scenario is run
twice with the same starting bias:

  fixed kernel     the weights never change (lazy / pure-NTK learner). With new stimuli
                   each block, blocks 1 and 2 are then identical by construction.
  adaptive kernel  after each trial the weights drift toward the mode make-up of what has
                   been learned, with a slow pull back to the prior. Experience with one
                   rule type can then speed up the next block of that type and slow down a
                   switch to a type that needs a down-weighted mode.

Parameters are placeholders until fitted to data; everything is exposed on the command line.

Outputs go to ~/Documents/data/catlearn_eeg/kernel_model/<design>/:
  blocks_<scenario>.png   accuracy, per-mode learning progress and kernel weights over trials
  summary_blocks.png      block-mean accuracy per scenario, fixed vs adaptive
  simulation_means.npz    run-averaged curves and all parameters, for later analysis

Run with the kernelbehav env:
  ~/miniforge3/envs/kernelbehav/bin/python K01_simulate_blocks.py --design 2x2
"""

import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np

from kernel_modes import DATA_ROOT, Design, simulate_blocks
from plot_style import ACCENT, FOREGROUND, apply_dark_theme  # from task_design, via kernel_modes' sys.path

MODE_COLORS = {"cst": "#9e9e9e", "A": "#ffa630", "B": "#c77dff", "AB": "#2ec4b6"}  # A/B as in the design figures


def mode_color(name, i):
    return MODE_COLORS.get(name, plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10])


def default_scenarios(design):
    """Scenario name -> task names per block."""
    if design.name == "2x2":
        return {
            "I-I-II_same_dim": ["A", "A", "AB"],
            "I-I-II_dim_shift": ["A", "B", "AB"],
            "II-II-I": ["AB", "AB", "A"],
        }
    # Other designs: the simplest and the most complex type, each practised then switched
    first = design.task_names[int(np.argmax(design.task_types == 0))]
    last = design.task_names[int(np.argmax(design.task_types == design.task_types.max()))]
    print(f"note: generic scenarios for {design.name}; edit default_scenarios() to choose specific tasks")
    return {"I-I-last": [first, first, last], "last-last-I": [last, last, first]}


def block_means(p_correct, block):
    """Block-mean accuracy per run: (runs, blocks)."""
    return np.stack([p_correct[:, block == b].mean(axis=1) for b in np.unique(block)], axis=1)


def plot_scenario(design, name, tasks, results, n_trials_block, out_dir):
    fixed, adapt = results["fixed"], results["adaptive"]
    T = fixed["p_correct"].shape[1]
    trials = np.arange(1, T + 1)
    n_blocks = len(tasks)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)

    # Accuracy
    ax = axes[0]
    for res, ls, label in ((fixed, "--", "fixed kernel"), (adapt, "-", "adaptive kernel")):
        m = res["p_correct"].mean(axis=0)
        sem = res["p_correct"].std(axis=0) / np.sqrt(res["p_correct"].shape[0])
        ax.plot(trials, m, ls=ls, color=FOREGROUND, lw=1.5, label=label)
        ax.fill_between(trials, m - sem, m + sem, color=FOREGROUND, alpha=0.15, lw=0)
    ax.axhline(0.5, color="0.4", lw=0.8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("p(correct)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    # Learning progress in each mode the current rule uses
    ax = axes[1]
    for e, mname in enumerate(design.mode_names):
        c = mode_color(mname, e)
        for res, ls in ((fixed, "--"), (adapt, "-")):
            prog = res["progress"][:, :, e]
            if np.all(np.isnan(prog)):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN before a mode is used
                m = np.nanmean(prog, axis=0)
            ax.plot(trials, m, ls=ls, color=c, lw=1.8, label=mname if ls == "-" else None)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("fraction of target\nlearned, per mode")
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=design.n_modes, title="mode (solid adaptive, dashed fixed)",
              title_fontsize=8)

    # Kernel weights
    ax = axes[2]
    for e, mname in enumerate(design.mode_names):
        c = mode_color(mname, e)
        ax.plot(trials, fixed["weights"][:, :, e].mean(axis=0), ls="--", color=c, lw=1.2)
        ax.plot(trials, adapt["weights"][:, :, e].mean(axis=0), ls="-", color=c, lw=1.8, label=mname)
    ax.set_ylim(0, None)
    ax.set_ylabel("kernel weight")
    ax.set_xlabel("trial")
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=design.n_modes)

    for ax in axes:
        for b in range(1, n_blocks):
            ax.axvline(b * n_trials_block + 0.5, color=ACCENT, ls="--", lw=1)
    for b, t in enumerate(tasks):
        axes[0].text((b + 0.5) / n_blocks, 1.02, f"block {b + 1}: {design.task_label(design.task_index(t))}",
                     transform=axes[0].transAxes, ha="center", va="bottom", fontsize=9, color=ACCENT)
    fig.suptitle(f"{design.name}: {name}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / f"blocks_{name}.png", dpi=150)
    plt.close(fig)


def plot_summary(design, scenarios, all_results, out_dir):
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.6 * len(scenarios), 3.6), sharey=True, squeeze=False)
    for ax, (name, tasks) in zip(axes[0], scenarios.items()):
        res = all_results[name]
        x = np.arange(1, len(tasks) + 1)
        for key, ls, marker in (("fixed", "--", "o"), ("adaptive", "-", "s")):
            bm = block_means(res[key]["p_correct"], res[key]["block"])
            m, sem = bm.mean(axis=0), bm.std(axis=0) / np.sqrt(bm.shape[0])
            ax.errorbar(x, m, yerr=sem, ls=ls, marker=marker, color=FOREGROUND, capsize=3, label=f"{key} kernel")
        ax.set_xticks(x, [f"{b}\n{t}" for b, t in zip(x, tasks)])
        ax.set_title(name, fontsize=10)
        ax.axhline(0.5, color="0.4", lw=0.8)
    axes[0][0].set_ylabel("block-mean p(correct)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"{design.name}: block means (x = block, rule)")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_blocks.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", default="2x2")
    parser.add_argument("--n-reps", type=int, default=10, help="passes through all stimuli per block (2x2: 10 -> 40 trials)")
    parser.add_argument("--n-runs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=3.0, help="choice sigmoid slope")
    parser.add_argument("--forget", type=float, default=0.0)
    parser.add_argument("--order-decay", type=float, default=0.5, help="prior eigenvalue ratio per interaction order")
    parser.add_argument("--adapt-rate", type=float, default=0.02, help="adaptive kernel: drift toward learned modes per trial")
    parser.add_argument("--relax-rate", type=float, default=0.002, help="adaptive kernel: pull back to the prior per trial")
    parser.add_argument("--same-stimuli", action="store_true", help="reuse stimuli across blocks (no reset of learned values)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    apply_dark_theme()

    design = Design(args.design)
    w0 = design.prior_weights(args.order_decay)
    scenarios = default_scenarios(design)
    out_dir = DATA_ROOT / "kernel_model" / design.name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_trials_block = design.n_stim * args.n_reps

    common = dict(n_reps=args.n_reps, lr=args.lr, forget=args.forget, beta=args.beta,
                  reset_between_blocks=not args.same_stimuli, n_runs=args.n_runs)
    print(f"{design.name}: modes {design.mode_names}, prior weights {np.round(w0, 3).tolist()}, "
          f"{n_trials_block} trials/block, {'same' if args.same_stimuli else 'new'} stimuli each block")

    all_results, means = {}, {}
    for s_i, (name, task_names) in enumerate(scenarios.items()):
        tasks = [design.task_index(t) for t in task_names]
        # Same seed for both learners, so they see identical trial orders
        seed = args.seed + 1000 * s_i
        res = {
            "fixed": simulate_blocks(design, tasks, w0, adapt_rate=0.0, relax_rate=0.0, seed=seed, **common),
            "adaptive": simulate_blocks(design, tasks, w0, adapt_rate=args.adapt_rate, relax_rate=args.relax_rate,
                                        seed=seed, **common),
        }
        all_results[name] = res
        plot_scenario(design, name, task_names, res, n_trials_block, out_dir)

        print(f"\n{name}: blocks {' -> '.join(task_names)}")
        for key in ("fixed", "adaptive"):
            bm = block_means(res[key]["p_correct"], res[key]["block"]).mean(axis=0)
            print(f"  {key:8s} block-mean p(correct): " + "  ".join(f"B{b + 1} {v:.3f}" for b, v in enumerate(bm)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # modes the rule never uses stay NaN
                means[f"{name}/{key}/p_correct"] = res[key]["p_correct"].mean(axis=0)
                means[f"{name}/{key}/progress"] = np.nanmean(res[key]["progress"], axis=0)
            means[f"{name}/{key}/weights"] = res[key]["weights"].mean(axis=0)
        w_adapt = res["adaptive"]["weights"].mean(axis=0)
        ends = [n_trials_block * (b + 1) - 1 for b in range(len(tasks))]
        print("  adaptive weights at end of each block (" + ", ".join(design.mode_names) + "): "
              + " | ".join(np.array2string(w_adapt[t], precision=2, separator=",") for t in ends))

    plot_summary(design, scenarios, all_results, out_dir)
    np.savez(out_dir / "simulation_means.npz", params=json.dumps(vars(args)), mode_names=design.mode_names,
             prior_weights=w0, scenarios=json.dumps(scenarios), **means)
    print(f"\nSaved figures and simulation_means.npz to {out_dir}")


if __name__ == "__main__":
    main()
