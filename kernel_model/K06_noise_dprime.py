"""Predictions for the 2x2x2 noise study: how alpha sets d' before learning, how well the
rule is learned, and how much learning should sharpen each position (A, B, C).

A simple observer model. Every number here is a prediction under stated assumptions, not data.

1. d' before learning. The two patches of a pair differ by their DreamSim distance D
   (task_design/noise_similarity.py; for the study's alphas the pair distance the stimuli
   were selected at). d'0 is assumed proportional to D:
       d'0 = d'_fractal * D / 0.256       (0.256 = the fractal pairs' distance)
   d'_fractal, the d' of a fractal pair, is unknown. The pilot measures it. So the model is
   run over a grid of d'0, and the study's alphas are placed on that grid for several
   assumed d'_fractal (--fractal-dprimes). An observer who has to tell which of a pair's
   patches is shown, with an unbiased criterion, is right with p_id = Phi(d'0 / 2).
2. Learning the rule. The adaptive kernel learner (kernel_modes.py, as in K04) runs the
   catlearn_2x2x2_prolific session:
   - practice: two 2x2 machines with fractals (type I on A or B, half the runs each, then
     XOR), each to 8 of the last 10;
   - transfer: modes the 2x2 machines have (cst, A, B, AB) keep their trained eigenvalue
     relative to the prior; C, AC, BC and ABC stay at the 2x2x2 prior;
   - test: two machines of type III (A+B+AC+BC), 152 rounds each, new patches in machine 2.
   On every test round each position's patch is misread (taken for the other patch of its
   pair) with probability 1 - p_id. The learner updates the combination it perceived,
   with the true outcome.
3. Learning-driven gain in d' (acquired distinctiveness), per position f:
       gain_f = G * learned_f * (1 - r),   r = 2 p_id - 1
   - learned_f: the share of the rule's f-related part (modes containing f) learned by the
     end of machine 2. This is the floor: patches too alike to tell apart leave the rule
     unlearned, and nothing drives sharpening.
   - (1 - r): room left to improve. This is the ceiling: patches that are already easy
     have little to gain.
   G (--gain) only sets the scale.
   C has no linear mode in this rule (it matters only through AC and BC) and was never
   trained in practice, so it is learned later: its gain peaks at a larger d'0.

Outputs go to ~/Documents/data/catlearn_eeg/kernel_model/2x2x2/noise_dprime/:
  baseline.png  d'0 relative to fractal pairs, by alpha (the study's alphas marked)
  learning.png  accuracy and share reaching 22 of 24 in machine 2, by d'0
  gain.png      d' gain for A / B vs C, by d'0
  (learning and gain: below the curve, where the study's alphas fall for each d'_fractal)
  grid.csv      model outputs per d'0
  study.csv     the same, interpolated, per study alpha and assumed d'_fractal

  ~/miniforge3/envs/kernelbehav/bin/python K06_noise_dprime.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from K04_slot4x4 import run_block
from kernel_modes import DATA_ROOT, Design
from plot_style import FOREGROUND, apply_dark_theme  # from task_design, via kernel_modes' sys.path

REPO = Path(__file__).resolve().parents[1]
SIMILARITY = DATA_ROOT / "task_design" / "noise_similarity" / "summary.csv"
NOISE_DIR = REPO / "catlearn_2x2x2_prolific" / "stimuli" / "noise"
FRACTAL_D = 0.256
STUDY_ALPHAS = ["1.50", "1.75", "2.00", "2.25"]
STUDY_COLORS = ["#4cc9f0", "#2ec4b6", "#ffa630", "#c77dff"]
AB_COL, C_COL = "#4cc9f0", "#ffa630"


def dreamsim_medians():
    with open(SIMILARITY) as f:
        rows = [r for r in csv.DictReader(f) if r["measure"] == "dreamsim"]
    return np.array([float(r["alpha"]) for r in rows]), np.array([float(r["median"]) for r in rows])


def study_distances():
    """Pair distance each study alpha's stimuli were selected at."""
    return np.array([json.loads((NOISE_DIR / f"a{a}" / "groups.json").read_text())["target_distance"]
                     for a in STUDY_ALPHAS])


def transfer_2x2(w, d2, d3):
    """Kernel weights after 2x2 practice -> 2x2x2: shared modes keep their trained
    eigenvalue relative to the prior, the others stay at the 2x2x2 prior."""
    p2, p3 = d2.prior_weights(), d3.prior_weights()
    ratio = np.ones((w.shape[0], d3.n_modes))
    for i, name in enumerate(d2.mode_names):
        ratio[:, d3.mode_names.index(name)] = w[:, i] / p2[i]
    w_new = p3 * ratio
    return w_new / w_new.sum(1, keepdims=True)


def test_block(d3, ystar, w, w0, rng, n_trials, p_id, args):
    """One test machine with misread patches. Returns weights, values y, expected accuracy
    per round and the round at which 22 of the last 24 sampled answers were first correct."""
    R, n = w.shape[0], d3.n_stim
    P, eig_scale = d3.projectors, n / d3.mode_dims
    place = 1 << np.arange(d3.n_dims)[::-1]                               # A slowest
    bits = (np.arange(n)[:, None] // place) & 1                          # (n, 3)
    seqs = np.stack([np.concatenate([rng.permutation(n) for _ in range(n_trials // n)]) for _ in range(R)])
    y = np.zeros((R, n))
    acc = np.zeros((R, n_trials))
    recent = np.zeros((R, args.test_criterion[0]), bool)
    reached = np.full(R, np.nan)
    runs = np.arange(R)
    for k in range(n_trials):
        s = seqs[:, k]
        flip = rng.random((R, d3.n_dims)) > p_id                          # misread positions
        seen = ((bits[s] ^ flip) * place).sum(1)
        p = 1.0 / (1.0 + np.exp(-args.beta * y[runs, seen] * ystar[s]))
        acc[:, k] = p
        col = np.einsum("re,enr->rn", w * eig_scale, P[:, :, seen])
        y += args.lr * (ystar[s] - y[runs, seen])[:, None] * col
        if args.adapt_rate > 0:
            shares = d3.mode_shares(y)
            ok = ~np.isnan(shares[:, 0])
            w[ok] = (1 - args.adapt_rate - args.relax_rate) * w[ok] + args.adapt_rate * shares[ok] \
                + args.relax_rate * w0
        recent = np.roll(recent, -1, axis=1)
        recent[:, -1] = rng.random(R) < p
        newly = np.isnan(reached) & (k + 1 >= recent.shape[1]) & (recent.sum(1) >= args.test_criterion[1])
        reached[newly] = k + 1
    return w, y, acc, reached


def learned_by_position(d3, y, ystar):
    """Share of the rule's part involving each position (modes containing it) that y has learned."""
    out = []
    for f in "ABC":
        idx = [i for i, m in enumerate(d3.mode_names) if f in m and m != "cst"]
        target = sum(d3.projectors[i] @ ystar for i in idx)
        out.append(np.clip((y @ target) / (target @ target), 0, 1))
    return np.stack(out, 1)                                                 # (runs, 3)


def study_strip(ax, d0_by_scale, scales):
    """Rows of markers: where the study alphas fall on the d'0 axis for each assumed d'_fractal."""
    for row, s in enumerate(scales):
        for d, c, a in zip(d0_by_scale[s], STUDY_COLORS, STUDY_ALPHAS):
            ax.plot(d, row, "o", ms=8, color=c, label=f"α {a}" if row == 0 else None)
    ax.set_yticks(range(len(scales)), [f"fractal d′ = {s:g}" for s in scales], fontsize=8)
    ax.set_ylim(len(scales) - 0.4, -0.6)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.55))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=400)
    parser.add_argument("--fractal-dprimes", type=float, nargs="+", default=[3, 6, 9],
                        help="assumed d' of a fractal pair (unknown until the pilot)")
    parser.add_argument("--dprime-grid", type=float, nargs=3, default=[0.2, 6.0, 0.2], help="start stop step")
    parser.add_argument("--gain", type=float, default=1.0, help="scale of the learning-driven d' gain")
    parser.add_argument("--test-trials", type=int, default=152)
    parser.add_argument("--practice-reps", type=int, default=20)
    parser.add_argument("--criterion", type=int, nargs=2, default=[10, 8])
    parser.add_argument("--test-criterion", type=int, nargs=2, default=[24, 22])
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--adapt-rate", type=float, default=0.05)
    parser.add_argument("--relax-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    apply_dark_theme()

    d2, d3 = Design("2x2"), Design("2x2x2")
    out_dir = DATA_ROOT / "kernel_model" / "2x2x2" / "noise_dprime"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # practice once (fractals), shared by every d'0
    w0_2, w0_3 = d2.prior_weights(), d3.prior_weights()
    kw = dict(lr=args.lr, beta=args.beta, adapt=args.adapt_rate, relax=args.relax_rate)
    halves = []
    for first in ("A", "B"):                                 # type I on A or B, half the runs each
        w = np.tile(w0_2, (args.runs // 2, 1))
        for task in (d2.task_index(first), d2.task_index("AB")):
            w, _, _ = run_block(d2, d2.task_labels[task], w, rng, args.practice_reps, w0=w0_2,
                                criterion=tuple(args.criterion), **kw)
        halves.append(w)
    w_test = transfer_2x2(np.concatenate(halves), d2, d3)
    ystar = d3.task_labels[d3.task_index("A+B+AC+BC")]

    lo, hi, step = args.dprime_grid
    grid = np.round(np.arange(lo, hi + step / 2, step), 3)
    rows = []
    for d0 in grid:
        p_id = norm.cdf(d0 / 2)
        w_a = w_test.copy()
        for _ in range(2):                                   # two machines, new patches
            w_a, y, acc, hit = test_block(d3, ystar, w_a, w0_3, rng, args.test_trials, p_id, args)
        learned = learned_by_position(d3, y, ystar).mean(0)
        gain = args.gain * learned * (2 - 2 * p_id)          # (1 - r), r = 2 p_id - 1
        rows.append({"dprime_before": d0, "p_identify": round(p_id, 3), "acc_machine2": round(acc.mean(), 3),
                     "reach_criterion_machine2": round(np.mean(~np.isnan(hit)), 3),
                     "learned_A": round(learned[0], 3), "learned_B": round(learned[1], 3),
                     "learned_C": round(learned[2], 3), "gain_AB": round((gain[0] + gain[1]) / 2, 3),
                     "gain_C": round(gain[2], 3)})
    col = lambda k: np.array([row[k] for row in rows])

    # where the study alphas fall, per assumed fractal d'
    dist = study_distances()
    d0_by_scale = {s: s * dist / FRACTAL_D for s in args.fractal_dprimes}
    study_rows = []
    for s in args.fractal_dprimes:
        for a, d, d0 in zip(STUDY_ALPHAS, dist, d0_by_scale[s]):
            study_rows.append({"alpha": a, "dreamsim": round(d, 4), "fractal_dprime": s, "dprime_before": round(d0, 2),
                               **{k: round(float(np.interp(d0, grid, col(k))), 3)
                                  for k in ("acc_machine2", "reach_criterion_machine2", "gain_AB", "gain_C")}})
            print("  ".join(f"{k} {v}" for k, v in study_rows[-1].items()))
    peak_ab, peak_c = grid[col("gain_AB").argmax()], grid[col("gain_C").argmax()]
    print(f"gain peaks at d'0 = {peak_ab:g} (A, B) and {peak_c:g} (C)")

    # ---- d' before learning, relative to fractal pairs (no scale assumption)
    alphas, med = dreamsim_medians()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(alphas, med / FRACTAL_D, "o-", color=FOREGROUND, lw=2, ms=4)
    for a, d, c in zip(STUDY_ALPHAS, dist, STUDY_COLORS):
        ax.plot(float(a), d / FRACTAL_D, "o", ms=11, color=c, zorder=3)
        ax.text(float(a), d / FRACTAL_D + 0.07, f"{d / FRACTAL_D:.0%}", ha="center", color=c, fontsize=9)
    ax.axhline(1, color="#ffa630", ls=":", lw=1.2)
    ax.text(0.5, 0.97, "fractal pairs", color="#ffa630", va="top", fontsize=9)
    ax.set_xlabel("α (spectral slope)")
    ax.set_ylabel("d′ before learning\n(share of a fractal pair's)")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    fig.savefig(out_dir / "baseline.png", dpi=200)
    plt.close(fig)

    def with_strip(draw, ylabel, name):
        fig, (ax, sx) = plt.subplots(2, 1, figsize=(6, 4.4), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
        draw(ax)
        ax.set_ylabel(ylabel)
        study_strip(sx, d0_by_scale, args.fractal_dprimes)
        sx.set_xlabel("d′ before learning")
        sx.set_xlim(0, hi)
        fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.25)
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)

    def draw_learning(ax):
        ax.plot(grid, col("acc_machine2"), color=FOREGROUND, lw=2, label="mean accuracy, machine 2")
        ax.plot(grid, col("reach_criterion_machine2"), color="#2ec4b6", lw=2, label="share reaching 22 of 24")
        ax.axhline(0.75, color="0.4", lw=0.8, ls="--")
        ax.text(0.1, 0.77, "one feature alone: 75%", color="0.6", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=9, loc="center right")

    def draw_gain(ax):
        ax.plot(grid, col("gain_AB"), color=AB_COL, lw=2, label="A and B (bottom)")
        ax.plot(grid, col("gain_C"), color=C_COL, lw=2, label="C (top)")
        ax.set_ylim(0, max(col("gain_AB").max(), col("gain_C").max()) * 1.2)
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    with_strip(draw_learning, "predicted", "learning.png")
    with_strip(draw_gain, "predicted d′ gain\n(trained − untrained)", "gain.png")

    for name, data in (("grid.csv", rows), ("study.csv", study_rows)):
        with open(out_dir / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
