"""Parameter recovery for the kernel learner, model 1 (helpers/kernel.py): can this design
pin down the model's parameters, and which ones?

Model 1: per test block, mode weights w = (cst, A, B, AB) on the simplex; a learning rate lr and
a choice sensitivity beta shared across blocks (11 free parameters per participant).

Procedure:
  1. Synthetic participants with known parameters ("truths"): per test block, weights from a
     Dirichlet(1, 2, 2, 2) (any mix, the constant mode a little rarer); lr log-uniform in
     --lr-range; beta log-uniform in --beta-range. The ranges are set so the simulated block
     accuracy covers what real participants reach (checked in accuracy_check.png).
  2. Each synthetic participant runs a real participant's exact test session: the same version
     (rules VI -> X -> II), the same order of pairs, the same late answers. Only the choices
     are simulated, from the model.
  3. Each synthetic participant is fit by maximum likelihood exactly as real data will be
     (--starts random starts, L-BFGS-B with exact gradients from JAX).
  4. Fitted vs true, per parameter: correlation, bias, RMSE; and how estimation errors go
     together (trade-offs such as lr vs beta). Also the quantity the version comparison rests
     on: log(w_A / w_B) per block. As a check on the optimiser, the fitted likelihood should
     never be worse than the likelihood at the true parameters.

Outputs, in analysis/kernel_recovery/:
  recovery.csv          truth and fit per synthetic participant
  summary.csv           per parameter: Pearson r, Spearman rho, bias, RMSE
  weights.png           fitted vs true weights, per block and mode
  scalars.png           lr, beta and log(w_A / w_B) per block
  error_correlations.png  correlations of the estimation errors across parameters
  accuracy_check.png    simulated vs real block accuracy

  ~/miniforge3/envs/kernelbehav/bin/python B12_kernel_recovery.py
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from helpers.kernel import KERNEL_MODES, Sequences, fit, nll, pack, simulate
from helpers.plots import apply_dark_theme, save
from helpers.study import Study, parser
from params import SEED, TEST_TYPES, VERSION_COLORS, VERSIONS


def draw_truth(rng, lr_range, beta_range):
    W = rng.dirichlet([1, 2, 2, 2], size=len(TEST_TYPES))
    lr = float(np.exp(rng.uniform(*np.log(lr_range))))
    beta = float(np.exp(rng.uniform(*np.log(beta_range))))
    return W, lr, beta


def flat(W, lr, beta, prefix):
    out = {f"{prefix}_{typ}_{m}": W[b, e] for b, typ in enumerate(TEST_TYPES) for e, m in enumerate(KERNEL_MODES)}
    out.update({f"{prefix}_lr": lr, f"{prefix}_beta": beta})
    out.update({f"{prefix}_{typ}_logAB": np.log(W[b, 1] / W[b, 2]) for b, typ in enumerate(TEST_TYPES)})
    return out


def block_accuracy(seq, choice):
    ok = seq.answered > 0
    correct = ((2 * choice - 1) == seq.target) & ok
    return [correct[seq.block == b].sum() / max(1, (seq.block == b).sum()) for b in range(len(TEST_TYPES))]


def main():
    p = parser(__doc__)
    p.add_argument("--n", type=int, default=150, help="synthetic participants")
    p.add_argument("--starts", type=int, default=6, help="random starts per fit")
    p.add_argument("--lr-range", type=float, nargs=2, default=(0.01, 0.2))
    p.add_argument("--beta-range", type=float, nargs=2, default=(0.5, 5.0))
    args = p.parse_args()
    apply_dark_theme()
    study = Study.from_args(args)
    study.save_exclusions()
    groups = study.groups()
    real = [(pid, v, Sequences.from_trials(study.trials_of(pid))) for v in VERSIONS for pid in groups[v]]
    rng = np.random.default_rng(SEED)
    out = study.out("kernel_recovery")

    rows, t0 = [], time.time()
    for i in range(args.n):
        pid, version, seq = real[i % len(real)]
        W, lr, beta = draw_truth(rng, args.lr_range, args.beta_range)
        sim = seq.with_choices(simulate(W, lr, beta, seq, rng))
        Wf, lrf, betaf, nll_fit, _ = fit(sim, rng, n_starts=args.starts)
        rows.append({"synthetic": i, "template": pid, "version": version, "nll_fit": nll_fit,
                     "nll_truth": nll(pack(W, lr, beta), sim),
                     **{f"acc_{typ}": a for typ, a in zip(TEST_TYPES, block_accuracy(sim, sim.choice))},
                     **flat(W, lr, beta, "true"), **flat(Wf, lrf, betaf, "fit")})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{args.n} fitted ({time.time() - t0:.0f} s)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "recovery.csv", index=False)

    names = [c[len("true_"):] for c in df.columns if c.startswith("true_")]
    summary = []
    for n in names:
        t, f = df[f"true_{n}"], df[f"fit_{n}"]
        summary.append({"parameter": n, "pearson_r": np.corrcoef(t, f)[0, 1], "spearman_rho": spearmanr(t, f)[0],
                        "bias": (f - t).mean(), "rmse": np.sqrt(((f - t) ** 2).mean()),
                        "true_sd": t.std()})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)

    # ------------------------------------------------ weights
    fig, axes = plt.subplots(len(TEST_TYPES), len(KERNEL_MODES), figsize=(13, 9.5))
    for r, typ in enumerate(TEST_TYPES):
        for c, m in enumerate(KERNEL_MODES):
            ax = axes[r, c]
            for v in VERSIONS:
                d = df[df["version"] == v]
                ax.scatter(d[f"true_{typ}_{m}"], d[f"fit_{typ}_{m}"], s=10, alpha=0.7, color=VERSION_COLORS[v],
                           label=f"version {v} sequence")
            ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
            s = summary.set_index("parameter").loc[f"{typ}_{m}"]
            ax.set_title(f"{typ}, w_{m}: r = {s['pearson_r']:.2f}, RMSE {s['rmse']:.2f}", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if r == len(TEST_TYPES) - 1:
                ax.set_xlabel("true", fontsize=8)
            if c == 0:
                ax.set_ylabel("fitted", fontsize=8)
    axes[0, -1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Recovery of the mode weights ({args.n} synthetic participants on real sessions)", fontsize=11)
    save(fig, out / "weights.png", dpi=120, tight_rect=(0, 0, 1, 0.95))

    # ------------------------------------------------ scalars and A:B ratios
    scal = [("lr", "learning rate", True), ("beta", "choice sensitivity beta", True)]
    scal += [(f"{typ}_logAB", f"{typ}: log(w_A / w_B)", False) for typ in TEST_TYPES]
    fig, axes = plt.subplots(1, len(scal), figsize=(3.4 * len(scal), 3.6))
    for ax, (n, label, logscale) in zip(axes, scal):
        t, f = df[f"true_{n}"], df[f"fit_{n}"]
        ax.scatter(t, f, s=10, alpha=0.7, color="#c77dff")
        lo, hi = min(t.min(), f.min()), max(t.max(), f.max())
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.8, ls="--")
        if logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        s = summary.set_index("parameter").loc[n]
        ax.set_title(f"{label}\nr = {s['pearson_r']:.2f}, rho = {s['spearman_rho']:.2f}", fontsize=9)
        ax.set_xlabel("true", fontsize=8)
        ax.set_ylabel("fitted", fontsize=8)
    fig.suptitle("Recovery of the learning rate, the choice sensitivity and the A:B balance", fontsize=11)
    save(fig, out / "scalars.png", dpi=120, tight_rect=(0, 0, 1, 0.9))

    # ------------------------------------------------ trade-offs: correlations of the errors
    core = [f"{typ}_{m}" for typ in TEST_TYPES for m in KERNEL_MODES] + ["lr", "beta"]
    err = pd.DataFrame({n: df[f"fit_{n}"] - df[f"true_{n}"] for n in core})
    err["lr"] = np.log(df["fit_lr"]) - np.log(df["true_lr"])
    err["beta"] = np.log(df["fit_beta"]) - np.log(df["true_beta"])
    corr = err.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(core)), core, rotation=90, fontsize=7)
    ax.set_yticks(range(len(core)), core, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="correlation of errors (fitted - true; lr, beta in log)")
    ax.set_title("Which estimation errors go together (trade-offs)", fontsize=10)
    save(fig, out / "error_correlations.png", dpi=120, tight_rect=(0, 0, 1, 1))

    # ------------------------------------------------ realism: simulated vs real accuracy
    real_acc = [a for _, _, seq in real for a in block_accuracy(seq, seq.choice)]
    sim_acc = df[[f"acc_{typ}" for typ in TEST_TYPES]].to_numpy().ravel()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0.3, 1, 29)
    ax.hist(real_acc, bins=bins, alpha=0.6, density=True, color="#ffa62b", label=f"real ({len(real)} participants)")
    ax.hist(sim_acc, bins=bins, alpha=0.6, density=True, color="#4cc9f0", label=f"simulated ({args.n})")
    ax.set_xlabel("block accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Do the synthetic learners look like real participants?", fontsize=10)
    save(fig, out / "accuracy_check.png", dpi=120, tight_rect=(0, 0, 1, 1))

    worse = (df["nll_fit"] > df["nll_truth"] + 1e-3).mean()
    print(summary.round(3).to_string(index=False))
    print(f"fits worse than the true parameters (optimiser misses): {worse:.0%}")
    print(f"block accuracy: real {np.mean(real_acc):.2f} [{np.min(real_acc):.2f}-{np.max(real_acc):.2f}], "
          f"simulated {np.mean(sim_acc):.2f} [{np.min(sim_acc):.2f}-{np.max(sim_acc):.2f}]")
    print(f"-> {out}/recovery.csv, summary.csv, weights.png, scalars.png, error_correlations.png, accuracy_check.png")


if __name__ == "__main__":
    main()
