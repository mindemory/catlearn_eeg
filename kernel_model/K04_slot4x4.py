"""Kernel-model expectations for the 4x4 slot-machine study (catlearn_4x4_prolific).

The study, as simulated here:
  practice  two 2x2 machines: type I (A; B is symmetric), then XOR. Each block ends once
            8 of the last 10 sampled answers are correct (at most 80 trials).
  test      one 4x4 type per participant (I, VI, X, XII, XIV, XV, XVI), 3 blocks of 128
            trials: the A version twice, then the B version (B, B, A is the mirror image
            and gives the same predictions with a symmetric prior). XVI has no A/B version.
  New fractals every block: learned values restart at 0; only the kernel carries over.

Learners (kernel_modes.py):
  fixed     the kernel never changes (lazy learner; attention spread over A, B, AB)
  adaptive  the kernel drifts toward the modes of what has been learned (rich learner;
            attention narrows onto the rule)
each with and without the practice blocks. The kernel's eigenvalue per mode is carried from
the 2x2 to the 4x4 design (mode weight w_e = eigenvalue * dim / n, renormalised).

Outputs go to ~/Documents/data/catlearn_eeg/kernel_model/4x4/slot_design/:
  learning_curves.png   accuracy over the 3 test blocks, one line per type, per learner
  block_summary.png     mean accuracy per test block and type, per learner
  practice.png          trials needed in the two practice blocks (with-practice learners)
  summary.csv           per type x learner x block: mean, first-32 and last-32 accuracy,
                        trials to 80% (rolling 16)
  curves.npz            run-averaged accuracy curves and kernel weights, all parameters

  ~/miniforge3/envs/kernelbehav/bin/python K04_slot4x4.py
"""

import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from kernel_modes import DATA_ROOT, Design, to_roman
from plot_style import FOREGROUND, apply_dark_theme  # from task_design, via kernel_modes' sys.path

TEST_TYPES = ["I", "VI", "X", "XII", "XIV", "XV", "XVI"]
TYPE_COLORS = plt.cm.plasma(np.linspace(0.05, 0.9, len(TEST_TYPES)))


def run_block(design, ystar, w, rng, n_reps, lr, beta, adapt, relax, w0, criterion=None):
    """One block for every run (rows of w). New stimuli: values start at 0.

    criterion = (window, min_correct): a run stops once that many of its last `window`
    sampled answers were correct (its later trials are NaN). lr, beta, adapt and relax are
    scalars or one value per run (individual differences). Returns the kernel weights at
    the end, p(correct) per trial (runs, trials) and trials used per run.
    """
    R, n = w.shape[0], design.n_stim
    lr, beta, adapt, relax = (np.broadcast_to(np.asarray(v, float), (R,)) for v in (lr, beta, adapt, relax))
    P = design.projectors
    eig_scale = n / design.mode_dims
    T = n * n_reps
    seqs = np.stack([np.concatenate([rng.permutation(n) for _ in range(n_reps)]) for _ in range(R)])
    y = np.zeros((R, n))
    p_correct = np.full((R, T), np.nan)
    active = np.ones(R, bool)
    recent = np.zeros((R, criterion[0] if criterion else 1), bool)
    used = np.full(R, T)
    runs = np.arange(R)
    for k in range(T):
        if not active.any():
            break
        s = seqs[:, k]
        ys = y[runs, s]
        p = 1.0 / (1.0 + np.exp(-beta * ys * ystar[s]))
        p_correct[active, k] = p[active]
        col = np.einsum("re,enr->rn", w * eig_scale, P[:, :, s])
        err = ystar[s] - ys
        y[active] += lr[active, None] * err[active, None] * col[active]
        if (adapt > 0).any():
            shares = design.mode_shares(y)
            ok = active & ~np.isnan(shares[:, 0])
            a, r = adapt[ok, None], relax[ok, None]
            w[ok] = (1 - a - r) * w[ok] + a * shares[ok] + r * w0
        if criterion:
            window, need = criterion
            correct = rng.random(R) < p                     # the sampled answer
            recent = np.roll(recent, -1, axis=1)
            recent[:, -1] = correct
            done = active & (k + 1 >= window) & (recent.sum(1) >= need)
            used[done] = k + 1
            active &= ~done
    return w, p_correct, used


def transfer(w, src, dst):
    """Carry kernel eigenvalues per mode from design src to design dst (same mode names)."""
    eig = w * src.n_stim / src.mode_dims
    w_new = eig * dst.mode_dims
    return w_new / w_new.sum(1, keepdims=True)


def test_tasks(d4):
    """{type name: (A-version task, B-version task)} for the 4x4 design."""
    names = [to_roman(t + 1) for t in d4.task_types]
    out = {}
    for typ in TEST_TYPES:
        idx = [i for i, n in enumerate(names) if n == typ]
        a = max(idx, key=lambda i: d4.task_loadings[i, 1])   # loading on mode A
        b = max(idx, key=lambda i: d4.task_loadings[i, 2])   # loading on mode B
        out[typ] = (a, b)
    return out


def trials_to(acc_curve, level=0.8, w=16):
    c = np.convolve(acc_curve, np.ones(w) / w, mode="valid")
    hit = np.where(c >= level)[0]
    return int(hit[0] + w) if len(hit) else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-runs", type=int, default=500)
    parser.add_argument("--test-reps", type=int, default=8, help="passes through the 16 pairs per test block")
    parser.add_argument("--practice-reps", type=int, default=20, help="maximum passes through the 4 pairs")
    parser.add_argument("--criterion", type=int, nargs=2, default=[10, 8], help="window, correct needed")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--order-decay", type=float, default=0.5)
    parser.add_argument("--adapt-rate", type=float, default=0.05)
    parser.add_argument("--relax-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    apply_dark_theme()

    d2, d4 = Design("2x2"), Design("4x4")
    assert d2.mode_names == d4.mode_names
    w0_2, w0_4 = d2.prior_weights(args.order_decay), d4.prior_weights(args.order_decay)
    tasks = test_tasks(d4)
    out_dir = DATA_ROOT / "kernel_model" / "4x4" / "slot_design"
    out_dir.mkdir(parents=True, exist_ok=True)
    R = args.n_runs
    learners = {
        "fixed": dict(adapt=0.0, relax=0.0, practice=True),
        "adaptive": dict(adapt=args.adapt_rate, relax=args.relax_rate, practice=True),
        "fixed, no practice": dict(adapt=0.0, relax=0.0, practice=False),
        "adaptive, no practice": dict(adapt=args.adapt_rate, relax=args.relax_rate, practice=False),
    }
    T = 16 * args.test_reps

    curves, weights, rows, practice_used = {}, {}, [], {}
    for li, (lname, L) in enumerate(learners.items()):
        for ti, typ in enumerate(TEST_TYPES):
            rng = np.random.default_rng(args.seed + 1000 * li + ti)
            kw = dict(lr=args.lr, beta=args.beta, adapt=L["adapt"], relax=L["relax"])
            w = np.tile(w0_2, (R, 1))
            if L["practice"]:
                used = []
                for task in (d2.task_index("A"), d2.task_index("AB")):
                    w, _, u = run_block(d2, d2.task_labels[task], w, rng, args.practice_reps, w0=w0_2,
                                        criterion=tuple(args.criterion), **kw)
                    used.append(u)
                practice_used[(lname, typ)] = np.array(used)       # (2 blocks, runs)
            w = transfer(w, d2, d4)
            a, b = tasks[typ]
            acc, wt = [], []
            for blk, task in enumerate((a, a, b)):
                w_start = w.mean(0)
                w, p, _ = run_block(d4, d4.task_labels[task], w, rng, args.test_reps, w0=w0_4, **kw)
                acc.append(p.mean(0))
                wt.append(w_start)
                rows.append({"learner": lname, "type": typ, "block": blk + 1,
                             "rule": d4.task_names[task] + (" (A)" if task == a else " (B)"),
                             "acc_mean": round(float(p.mean()), 4), "acc_first32": round(float(p[:, :32].mean()), 4),
                             "acc_last32": round(float(p[:, -32:].mean()), 4),
                             "trials_to_80": trials_to(p.mean(0)),
                             **{f"w_start_{m}": round(float(v), 4) for m, v in zip(d4.mode_names, w_start)}})
            curves[(lname, typ)] = np.concatenate(acc)
            weights[(lname, typ)] = np.array(wt)
        print(f"{lname:22s} " + "  ".join(
            f"{typ}: " + "/".join(f"{r['acc_mean']:.2f}" for r in rows if r['learner'] == lname and r['type'] == typ)
            for typ in TEST_TYPES))

    # ---- figures
    names = list(learners)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, sharey=True)
    x = np.arange(3 * T) + 1
    for ax, lname in zip(axes.flat, names):
        for ti, typ in enumerate(TEST_TYPES):
            c = curves[(lname, typ)]
            sm = np.convolve(np.pad(c, 8, mode="edge"), np.ones(16) / 16, mode="valid")[: len(c)]
            ax.plot(x, sm, color=TYPE_COLORS[ti], lw=1.4, label=typ)
        for b in (1, 2):
            ax.axvline(b * T + 0.5, color="0.4", ls="--", lw=0.8)
        for b, lab in enumerate(("A", "A", "B")):
            ax.text((b + 0.5) * T, 1.01, lab, ha="center", va="bottom", color=FOREGROUND, fontsize=9)
        ax.axhline(0.5, color="0.4", lw=0.7)
        ax.set_title(lname, fontsize=10, pad=14)
        ax.set_ylim(0.45, 1.0)
    axes[0, 0].legend(title="type", fontsize=8, title_fontsize=8, frameon=False, ncol=2, loc="lower right")
    for ax in axes[:, 0]:
        ax.set_ylabel("p(correct)")
    for ax in axes[-1]:
        ax.set_xlabel(f"test trial ({T} per block, new fractals each block)")
    fig.suptitle("Kernel model: 4x4 test blocks (A, A, B) by rule type", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.6), sharey=True)
    xs = np.arange(len(TEST_TYPES))
    for ax, lname in zip(axes, names):
        for b, (off, col) in enumerate(zip((-0.25, 0, 0.25), ("#ffa630", "#ffd166", "#c77dff"))):
            vals = [next(r["acc_mean"] for r in rows if r["learner"] == lname and r["type"] == t and r["block"] == b + 1)
                    for t in TEST_TYPES]
            ax.bar(xs + off, vals, 0.25, color=col, label=f"block {b + 1} ({'AAB'[b]})")
        ax.set_xticks(xs, TEST_TYPES)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(lname, fontsize=10)
        ax.axhline(0.5, color="0.4", lw=0.7)
    axes[0].set_ylabel("block-mean p(correct)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "block_summary.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for li, lname in enumerate(n for n in names if learners[n]["practice"]):
        u = np.concatenate([practice_used[(lname, t)] for t in TEST_TYPES], axis=1)
        for b, lab in enumerate(("type I", "XOR")):
            ax.hist(u[b], bins=np.arange(10, 85, 4), histtype="step", lw=1.5,
                    ls="-" if b == 0 else "--", color=("#ffa630", "#4cc9f0")[li], label=f"{lname}: {lab}")
    ax.set_xlabel("practice trials until 8 of the last 10 correct (max 80)")
    ax.set_ylabel("simulated participants")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "practice.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    np.savez(out_dir / "curves.npz", params=json.dumps(vars(args)), test_types=TEST_TYPES, learners=names,
             mode_names=d4.mode_names,
             **{f"{l}/{t}/acc": c for (l, t), c in curves.items()},
             **{f"{l}/{t}/w_start": w for (l, t), w in weights.items()},
             **{f"{l}/{t}/practice_trials": u for (l, t), u in practice_used.items()})
    for lname in (n for n in names if learners[n]["practice"]):
        u = np.concatenate([practice_used[(lname, t)] for t in TEST_TYPES], axis=1)
        print(f"practice ({lname}): median trials type I {np.median(u[0]):.0f}, XOR {np.median(u[1]):.0f}; "
              f"hit the 80-trial cap: {(u[0] >= 80).mean():.0%} / {(u[1] >= 80).mean():.0%}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
