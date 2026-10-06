"""Timed RAM (real task timing): rule-naive pretraining, then the 3-block design at 1000 trials per block.

Same logic as R03_pretrain_blocks.py, but with ram_timed's trial structure: 1 s fixation,
then up to 4 s of stimulus during which the agent responds whenever it chooses (RT in
250-ms steps), correct/incorrect feedback, next trial. Learning stays in the weights
("timing only"): one update per trial in the experiment.

  pretraining  'perception', rule-naive: each mini-block has 2 fresh generated patches
               (never the real 20); every trial shows ONE of them in a random grid cell,
               all 49 cells equally often (central cell included), and the agent reports
               which one -- with the same trial timing, so it also learns when to respond.
               Saved and reused whenever the settings match.
  experiment   3 blocks x --trials trials, new real patches every block, rule the same for
               blocks 1-2 and different in block 3. Every scenario starts from the same
               pretrained agents with a fresh optimizer. --test-batch sets how many trials
               share one weight update (1 = trial by trial).

With --no-pretrain the agents start from random weights and learn everything -- seeing,
looking, when to respond, the rule -- within the experiment itself. That needs far more
trials (use e.g. --trials 64000 --test-batch 64), so learning curves are read in relative
terms (block 2 vs block 1, early vs late) rather than trial-for-trial against people.

Outputs in ~/Documents/data/catlearn_eeg/ram_model/2x2/timed_config<NN>_<sensor>[_scratch]<tag>/:
  pretrained_agents.npz, pretraining.png
  blocks_<scenario>.png   accuracy, RT and timeouts, where gaze aims (and where it already
                          is at stimulus onset), stimulus-phase fixations per block
  summary_blocks.png      accuracy and RT by third of each block
  experiment_logs.npz     per-trial metrics per agent

  ~/miniforge3/envs/kernelbehav/bin/python R04_timed_blocks.py --config 1 --sensor graded
"""

import argparse
import json
import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from env_2by2 import (DATA_ROOT, GRID, N_POOL, Design, block_labels, load_patch, make_patch_bank,
                      render_single_everywhere, render_tiles, spatial_config, tile_size)
from plot_style import ACCENT, FOREGROUND, apply_dark_theme
from R01_train_blocks import LOC_COLORS, SENSORS
from R03_pretrain_blocks import SCENARIOS, load_agents, save_agents, smooth
from ram_timed import DEFAULTS, STEP_MS, init_params, make_train_step

METRICS = ("acc", "correct", "timeout", "rt_ms", "travel", "aim_A", "aim_B", "onset_A", "onset_B")
AGENT_KEYS = ("config", "sensor", "periph_noise", "sigma", "saccade_cost", "time_cost", "n_fix", "n_stim", "cell_px")
PRETRAIN_KEYS = AGENT_KEYS + ("agents", "pre_blocks", "pre_updates", "pre_batch", "pre_lr", "bank", "bank_seed", "seed")


def make_cfg(args):
    s = SENSORS[args.sensor]
    return {**DEFAULTS, "scales": s["scales"], "full_view": s["full_view"],
            "periph_noise": args.periph_noise if args.sensor == "noisy" else 0.0,
            "sigma": args.sigma, "saccade_cost": args.saccade_cost, "time_cost": args.time_cost,
            "n_fix": args.n_fix, "n_stim": args.n_stim}


def n_channels(args):
    s = SENSORS[args.sensor]
    return len(s["scales"]) + s["full_view"]


def pretrain(args, cfg):
    size = tile_size(args.cell_px)
    bank = make_patch_bank(args.bank, size, seed=args.bank_seed)
    rng = np.random.default_rng(args.seed)
    optimizer = optax.adam(args.pre_lr)
    params = jax.vmap(partial(init_params, n_channels=n_channels(args)))(jax.random.split(jax.random.PRNGKey(args.seed), args.agents))
    opt_state = jax.vmap(optimizer.init)(params)
    step = jax.jit(jax.vmap(make_train_step(optimizer, args.pre_batch, cfg), in_axes=(0, 0, 0, 0, 0, None)))
    _, ab_locs = render_tiles(args.config, bank[:4], args.cell_px)       # only used for gaze metrics
    ab_locs = jnp.asarray(ab_locs)
    key = jax.random.PRNGKey(args.seed + 1)
    curve = {m: np.zeros((args.pre_blocks, args.agents)) for m in ("acc", "rt_ms", "timeout")}
    t0 = time.time()
    for mb in range(args.pre_blocks):
        picks = np.stack([rng.choice(len(bank), 2, replace=False) for _ in range(args.agents)])
        rendered = [render_single_everywhere(bank[p], args.cell_px) for p in picks]
        images = jnp.asarray(np.stack([r[0] for r in rendered]))
        labels = jnp.asarray(np.stack([r[1] for r in rendered]))
        tail = {m: [] for m in curve}
        for u in range(args.pre_updates):
            key, sub = jax.random.split(key)
            params, opt_state, m = step(params, opt_state, jax.random.split(sub, args.agents), images, labels, ab_locs)
            if u >= args.pre_updates - 5:
                for name in curve:
                    tail[name].append(np.asarray(m[name]))
        for name in curve:
            curve[name][mb] = np.mean(tail[name], axis=0)
        if (mb + 1) % max(1, args.pre_blocks // 10) == 0:
            lo = max(0, mb - 9)
            print(f"  pretraining mini-block {mb + 1}/{args.pre_blocks} ({time.time() - t0:.0f}s): end-of-mini-block "
                  f"acc {curve['acc'][lo:mb + 1].mean():.2f}, RT {curve['rt_ms'][lo:mb + 1].mean():.0f} ms, "
                  f"timeouts {curve['timeout'][lo:mb + 1].mean():.2f}", flush=True)
    return params, curve


def plot_pretraining(curve, args, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    x = np.arange(1, args.pre_blocks + 1)
    for ax, name, label in ((axes[0], "acc", "accuracy (timeouts = errors)"), (axes[1], "rt_ms", "RT (ms)")):
        v = curve[name]
        m, sem = v.mean(1), v.std(1) / np.sqrt(v.shape[1])
        ax.plot(x, m, color=FOREGROUND)
        ax.fill_between(x, m - sem, m + sem, color=FOREGROUND, alpha=0.2, lw=0)
        ax.set_ylabel(label, fontsize=9)
    axes[0].axhline(0.5, color="0.4", lw=0.8)
    axes[0].set_ylim(0.4, 1.02)
    axes[1].set_xlabel(f"pretraining mini-block ({args.pre_updates} updates x {args.pre_batch} trials; "
                       "2 new patches, one per trial in a random cell of all 49)", fontsize=8)
    fig.suptitle(f"rule-naive perceptual pretraining (timed), {args.agents} agents, sensor '{args.sensor}'", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "pretraining.png", dpi=150)
    plt.close(fig)


def run_scenario(params0, design, task_names, args, cfg, seed):
    n_blocks = len(task_names)
    rng = np.random.default_rng(seed)
    size = tile_size(args.cell_px)
    real = np.stack([load_patch(k, size) for k in range(1, N_POOL + 1)])
    patches = np.stack([rng.choice(N_POOL, 4 * n_blocks, replace=False) for _ in range(args.agents)])
    images = jnp.asarray(np.stack([[render_tiles(args.config, real[patches[a, 4 * b:4 * b + 4]], args.cell_px)[0]
                                    for b in range(n_blocks)] for a in range(args.agents)]))
    ab_locs = jnp.asarray(render_tiles(args.config, real[:4], args.cell_px)[1])

    optimizer = optax.adam(args.test_lr)
    params, opt_state = params0, jax.vmap(optimizer.init)(params0)
    step = jax.jit(jax.vmap(make_train_step(optimizer, args.test_batch, cfg), in_axes=(0, 0, 0, 0, None, None)))
    key = jax.random.PRNGKey(seed)
    U = n_updates(args)                                  # logged once per update (batch means)
    logs = {m: np.zeros((args.agents, n_blocks * U)) for m in METRICS}
    end_locs = []
    tail_updates = max(1, args.heatmap_trials // args.test_batch)
    for b, task in enumerate(task_names):
        labels = jnp.asarray(block_labels(design, task))
        tail_locs, tail_mask = [], []
        for u in range(U):
            key, sub = jax.random.split(key)
            params, opt_state, m = step(params, opt_state, jax.random.split(sub, args.agents), images[:, b], labels, ab_locs)
            for name in METRICS:
                logs[name][:, b * U + u] = np.asarray(m[name])
            if u >= U - tail_updates:
                tail_locs.append(np.asarray(m["locs"]))
                tail_mask.append(np.asarray(m["loc_mask"]))
        end_locs.append((np.concatenate(tail_locs, axis=1), np.concatenate(tail_mask, axis=1)))
    if args.test_batch == 1:
        logs["rt_ms"] = np.where(logs["timeout"] > 0.5, np.nan, logs["rt_ms"])  # RT only for responded trials
    return logs, end_locs, patches + 1


def n_updates(args):
    return args.trials // args.test_batch


def smooth_nan(x, w):
    """Moving average that ignores NaNs (for RT, undefined on timeouts)."""
    valid = np.isfinite(x).astype(float)
    num = smooth(np.nan_to_num(x), w)
    den = smooth(valid, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def thirds(x, n_blocks, T):
    """(agents, blocks, 3): nan-mean in the first, middle and last third of each block."""
    v = x.reshape(x.shape[0], n_blocks, T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmean(c, axis=-1) for c in np.array_split(v, 3, axis=-1)], axis=-1)


def band(ax, x, v, color, label=None, ls="-"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(v, 0)
        sem = np.nanstd(v, 0) / np.sqrt(np.sum(np.isfinite(v), 0).clip(1))
    ax.plot(x, m, color=color, lw=1.6, ls=ls, label=label)
    ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.2, lw=0)


def plot_scenario(design, name, task_names, logs, end_locs, args, out_dir):
    n_blocks, T, U = len(task_names), args.trials, n_updates(args)
    x = (np.arange(n_blocks * U) + 1) * args.test_batch
    w = args.smooth
    fig = plt.figure(figsize=(3.4 * n_blocks + 1, 10.5))
    gs = fig.add_gridspec(4, n_blocks, height_ratios=[1, 0.8, 1, 1.15])
    ax_acc = fig.add_subplot(gs[0, :])
    ax_rt = fig.add_subplot(gs[1, :], sharex=ax_acc)
    ax_aim = fig.add_subplot(gs[2, :], sharex=ax_acc)

    band(ax_acc, x, smooth(logs["acc"], w), FOREGROUND)
    ax_acc.axhline(0.5, color="0.4", lw=0.8)
    ax_acc.set_ylim(0.4, 1.02)
    ax_acc.set_ylabel("p(correct)\n(timeouts = errors)", fontsize=9)

    band(ax_rt, x, smooth_nan(logs["rt_ms"], w), FOREGROUND, label="RT, responded trials")
    ax_rt.set_ylabel("RT (ms)", fontsize=9)
    ax_rt.set_ylim(0, STEP_MS * args.n_stim)
    ax_to = ax_rt.twinx()
    ax_to.plot(x, smooth(logs["timeout"], w).mean(0), color="#ff5a5f", lw=1.2, ls=":", label="timeouts (right axis)")
    ax_to.set_ylim(0, 1)
    ax_to.set_ylabel("timeout rate", color="#ff5a5f", fontsize=8)
    ax_to.tick_params(axis="y", colors="#ff5a5f", labelsize=8)
    handles = ax_rt.get_legend_handles_labels()[0] + ax_to.get_legend_handles_labels()[0]
    ax_rt.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=8, loc="upper right")

    for loc in ("A", "B"):
        band(ax_aim, x, smooth(logs[f"aim_{loc}"], w), LOC_COLORS[loc], label=f"stimulus-phase fixations aimed at {loc}")
        band(ax_aim, x, smooth(logs[f"onset_{loc}"], w), LOC_COLORS[loc], label=f"gaze already at {loc} at onset", ls="--")
    ax_aim.set_ylim(-0.02, 1.02)
    ax_aim.set_ylabel("fraction", fontsize=9)
    ax_aim.set_xlabel("trial" + (f" (weights updated every {args.test_batch} trials; curves are per-update means)"
                                 if args.test_batch > 1 else ""))
    ax_aim.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    for ax in (ax_acc, ax_rt, ax_aim):
        for b in range(1, n_blocks):
            ax.axvline(b * T + 0.5 * args.test_batch, color=ACCENT, ls="--", lw=1)
    for b, t in enumerate(task_names):
        ax_acc.text((b + 0.5) / n_blocks, 1.02, f"block {b + 1}: {design.task_label(design.task_index(t))}",
                    transform=ax_acc.transAxes, ha="center", va="bottom", fontsize=9, color=ACCENT)

    _, cells = spatial_config(args.config)
    for b in range(n_blocks):
        ax = fig.add_subplot(gs[3, b])
        locs, mask = end_locs[b]
        pts, wts = locs.reshape(-1, 2), mask.reshape(-1)
        ax.hist2d(pts[:, 0], pts[:, 1], bins=GRID * 4, range=[[-1, 1], [-1, 1]], weights=wts, cmap="magma")
        for k in range(1, GRID):
            ax.axhline(-1 + 2 * k / GRID, color="0.35", lw=0.5)
            ax.axvline(-1 + 2 * k / GRID, color="0.35", lw=0.5)
        for loc, (row, col) in zip("AB", cells):
            x0, y0 = -1 + 2 * col / GRID, -1 + 2 * row / GRID
            ax.add_patch(plt.Rectangle((x0, y0), 2 / GRID, 2 / GRID, fill=False, ec=LOC_COLORS[loc], lw=2))
            ax.text(x0 + 0.02, y0 + 0.02, loc, color=LOC_COLORS[loc], fontsize=9, fontweight="bold", va="top")
        ax.set_xlim(-1, 1)
        ax.set_ylim(1, -1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"stimulus-phase fixations,\nlast {args.heatmap_trials} trials of block {b + 1}", fontsize=9)

    config_name, _ = spatial_config(args.config)
    fig.suptitle(f"timed RAM {'from scratch' if args.no_pretrain else 'after pretraining'} (1 s fixation, <= {STEP_MS * args.n_stim / 1000:g} s response window), {name} | config {args.config} "
                 f"({config_name}), sensor '{args.sensor}', saccade cost {args.saccade_cost}, time cost {args.time_cost} | "
                 f"{args.agents} agents, {T} trials/block", y=0.997, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"blocks_{name}.png", dpi=150)
    plt.close(fig)


def plot_summary(all_logs, args, out_dir):
    names = list(all_logs)
    fig, axes = plt.subplots(2, len(names), figsize=(3.8 * len(names), 6.4), sharey="row", squeeze=False)
    for j, name in enumerate(names):
        tasks = SCENARIOS[name]
        for i, (metric, label) in enumerate((("acc", "p(correct)"), ("rt_ms", "RT (ms)"))):
            ax = axes[i, j]
            th = thirds(all_logs[name][metric], len(tasks), n_updates(args))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m, sem = np.nanmean(th, 0), np.nanstd(th, 0) / np.sqrt(th.shape[0])
            for b in range(len(tasks)):
                ax.errorbar(b + np.array([0.2, 0.5, 0.8]), m[b], yerr=sem[b], color=FOREGROUND, marker="o", ms=4, capsize=2)
            ax.set_xticks(np.arange(len(tasks)) + 0.5, [f"block {b + 1}\n{t}" for b, t in enumerate(tasks)])
            for b in range(1, len(tasks)):
                ax.axvline(b, color=ACCENT, ls="--", lw=1)
            if metric == "acc":
                ax.axhline(0.5, color="0.4", lw=0.8)
                ax.set_title(name, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{label}\nfirst / middle / last third", fontsize=9)
    start = "from scratch" if args.no_pretrain else "after pretraining"
    fig.suptitle(f"timed RAM {start}, {args.trials} trials per block (points: thirds of each block)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_blocks.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=int, default=1)
    parser.add_argument("--sensor", choices=list(SENSORS), default="graded")
    parser.add_argument("--periph-noise", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=DEFAULTS["sigma"])
    parser.add_argument("--saccade-cost", type=float, default=DEFAULTS["saccade_cost"])
    parser.add_argument("--time-cost", type=float, default=DEFAULTS["time_cost"])
    parser.add_argument("--n-fix", type=int, default=DEFAULTS["n_fix"], help="fixation steps (250 ms each)")
    parser.add_argument("--n-stim", type=int, default=DEFAULTS["n_stim"], help="response-window steps (250 ms each)")
    parser.add_argument("--cell-px", type=int, default=16)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--pre-blocks", type=int, default=100)
    parser.add_argument("--pre-updates", type=int, default=100)
    parser.add_argument("--pre-batch", type=int, default=64)
    parser.add_argument("--pre-lr", type=float, default=1e-3)
    parser.add_argument("--bank", type=int, default=4000)
    parser.add_argument("--bank-seed", type=int, default=1)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--no-pretrain", action="store_true", help="start the experiment from random weights")
    parser.add_argument("--trials", type=int, default=1000, help="trials per block")
    parser.add_argument("--test-batch", type=int, default=1, help="trials per weight update in the experiment")
    parser.add_argument("--test-lr", type=float, default=1e-3)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--heatmap-trials", type=int, default=100)
    parser.add_argument("--smooth", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-tag", default="")
    args = parser.parse_args()
    apply_dark_theme()

    design = Design("2x2")
    cfg = make_cfg(args)
    scratch = "_scratch" if args.no_pretrain else ""
    out_dir = DATA_ROOT / "ram_model" / "2x2" / f"timed_config{args.config:02d}_{args.sensor}{scratch}{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    agents_path = out_dir / "pretrained_agents.npz"
    pre_meta = {k: getattr(args, k) for k in PRETRAIN_KEYS}
    if args.no_pretrain:
        # Random starting weights, one set per agent; every scenario starts from the same ones
        params0 = jax.vmap(partial(init_params, n_channels=n_channels(args)))(
            jax.random.split(jax.random.PRNGKey(args.seed), args.agents))
        print(f"no pretraining: {args.agents} agents from random weights")
    elif agents_path.exists() and not args.retrain:
        params0, meta = load_agents(agents_path)
        if {k: meta.get(k) for k in PRETRAIN_KEYS} != pre_meta:
            print("saved pretrained agents used different settings; pretraining again")
            params0 = None
        else:
            print(f"loaded pretrained agents from {agents_path}")
    else:
        params0 = None
    if params0 is None:
        print(f"pretraining {args.agents} agents: {args.pre_blocks} mini-blocks x {args.pre_updates} updates x {args.pre_batch} trials")
        params0, curve = pretrain(args, cfg)
        save_agents(agents_path, params0, {**pre_meta, "pretrain_curve": {k: v.tolist() for k, v in curve.items()}})
        plot_pretraining(curve, args, out_dir)

    all_logs, saved = {}, {}
    for s_i, name in enumerate(args.scenarios):
        tasks = SCENARIOS[name]
        t0 = time.time()
        logs, end_locs, patches = run_scenario(params0, design, tasks, args, cfg, seed=args.seed + 100 + s_i)
        all_logs[name] = logs
        plot_scenario(design, name, tasks, logs, end_locs, args, out_dir)
        U = n_updates(args)
        acc = thirds(logs["acc"], len(tasks), U).mean(0)
        rt = np.nanmean(thirds(logs["rt_ms"], len(tasks), U), 0)
        to = thirds(logs["timeout"], len(tasks), U).mean(0)
        aim = [thirds(logs[f"aim_{l}"], len(tasks), U).mean(0)[:, 2] for l in "AB"]
        print(f"{name:8s} ({time.time() - t0:4.0f}s) acc first->last third: "
              + "  ".join(f"B{b + 1} {acc[b, 0]:.2f}->{acc[b, 2]:.2f}" for b in range(len(tasks)))
              + " | RT ms: " + "  ".join(f"B{b + 1} {rt[b, 0]:.0f}->{rt[b, 2]:.0f}" for b in range(len(tasks)))
              + " | timeouts last third: " + " ".join(f"{to[b, 2]:.2f}" for b in range(len(tasks)))
              + " | aims A/B last third: " + " ".join(f"{a:.2f}/{c:.2f}" for a, c in zip(*aim)), flush=True)
        for metric, arr in logs.items():
            saved[f"{name}/{metric}"] = arr
        saved[f"{name}/patches"] = patches
    plot_summary(all_logs, args, out_dir)
    np.savez(out_dir / "experiment_logs.npz", params=json.dumps(vars(args)), scenarios=json.dumps(SCENARIOS), **saved)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
