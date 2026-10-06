"""Order effects: how does learning one rule type first change learning the next?

For every pair of tasks (x, y) the learner does x -> x -> y -> y (new stimuli every
block) and we compare its accuracy over the two y blocks with a learner that meets y
first (y -> y). Task pairs are averaged within types (e.g. all type I -> type II pairs:
same and different dimensions), giving a type-by-type transfer matrix.

A fixed kernel shows no transfer by construction (new stimuli, values reset), so this
uses the adaptive kernel only; its weights carry over and are the only thing that can
transfer. Parameters are placeholders until fitted to data.

Outputs go to ~/Documents/data/catlearn_eeg/kernel_model/<design>/order_transfer/:
  transfer_types.png   type x type matrix: accuracy change (points) vs learning y first
  transfer.json        type matrix, fresh accuracy per type, parameters (read by the slides)
  transfer_tasks.csv   every task pair: accuracy of y after x, y first, and the change

  ~/miniforge3/envs/kernelbehav/bin/python K03_order_transfer.py --design 2x2x2
"""

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from kernel_modes import DATA_ROOT, Design, simulate_blocks, to_roman
from plot_style import apply_dark_theme  # from task_design, via kernel_modes' sys.path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", default="2x2x2")
    parser.add_argument("--n-reps", type=int, default=8, help="passes through all stimuli per block (2x2x2: 8 -> 64 trials)")
    parser.add_argument("--n-runs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--order-decay", type=float, default=0.5)
    parser.add_argument("--adapt-rate", type=float, default=0.05)
    parser.add_argument("--relax-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    apply_dark_theme()

    design = Design(args.design)
    w0 = design.prior_weights(args.order_decay)
    out_dir = DATA_ROOT / "kernel_model" / design.name / "order_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(n_reps=args.n_reps, lr=args.lr, beta=args.beta, adapt_rate=args.adapt_rate,
              relax_rate=args.relax_rate, n_runs=args.n_runs)
    types = design.task_types
    n_tasks, n_types = len(types), int(types.max()) + 1
    type_names = [to_roman(t + 1) for t in range(n_types)]

    fresh = np.array([simulate_blocks(design, [y, y], w0, seed=args.seed + 100 + y, **kw)["p_correct"].mean()
                      for y in range(n_tasks)])
    after = np.zeros((n_tasks, n_tasks))
    for x in range(n_tasks):
        for y in range(n_tasks):
            r = simulate_blocks(design, [x, x, y, y], w0, seed=args.seed + 1000 + n_tasks * x + y, **kw)
            after[x, y] = r["p_correct"][:, r["block"] >= 2].mean()
    delta = after - fresh[None, :]
    T = np.array([[delta[np.ix_(types == X, types == Y)].mean() for Y in range(n_types)] for X in range(n_types)])
    fresh_type = np.array([fresh[types == Y].mean() for Y in range(n_types)])

    np.set_printoptions(precision=1, suppress=True)
    print(f"{design.name}: {n_tasks} tasks, {n_types} types, {design.n_stim * args.n_reps} trials/block")
    print("accuracy when learned first, by type:", np.round(fresh_type, 3))
    print("transfer (rows: first type, cols: second type), accuracy points:")
    print(100 * T)

    # figure
    fig, ax = plt.subplots(figsize=(1.0 * n_types + 2.5, 0.9 * n_types + 1.5))
    lim = np.abs(100 * T).max()
    im = ax.imshow(100 * T, cmap="coolwarm", vmin=-lim, vmax=lim)
    for i in range(n_types):
        for j in range(n_types):
            ax.text(j, i, f"{100 * T[i, j]:+.0f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_xticks(range(n_types), type_names)
    ax.set_yticks(range(n_types), type_names)
    ax.set_xlabel("type learned second (blocks 3-4)")
    ax.set_ylabel("type learned first (blocks 1-2)")
    fig.colorbar(im, ax=ax, label="accuracy change vs learning it first (points)")
    ax.set_title(f"{design.name}: order effects in the adaptive kernel", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_types.png", dpi=150)
    plt.close(fig)

    (out_dir / "transfer.json").write_text(json.dumps({
        "design": design.name, "types": type_names, "fresh": fresh_type.tolist(), "transfer": T.tolist(),
        "trials_per_block": design.n_stim * args.n_reps, "prior_weights": w0.tolist(), "params": vars(args)}))
    with open(out_dir / "transfer_tasks.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["first_task", "first_type", "second_task", "second_type", "acc_after_first", "acc_learned_first",
                     "change"])
        for x in range(n_tasks):
            for y in range(n_tasks):
                wr.writerow([design.task_names[x], type_names[types[x]], design.task_names[y], type_names[types[y]],
                             round(after[x, y], 4), round(fresh[y], 4), round(delta[x, y], 4)])
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
