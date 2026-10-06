"""Pretrain RAM agents, then run the 3-block design at 1000 trials per block (free gaze).

A RAM trained from scratch needs tens of thousands of trials to learn to see and look, so
it learns nothing in a human-sized budget. Here each agent first gets a "visual
upbringing", then does the experiment. Pretraining always uses freshly generated noise
patches (same recipe as the real stimuli, task_design/generate_stimuli.py, never the
real 20). Two kinds (--pretrain-task):

  perception   (default, rule-naive) each mini-block has 2 fresh patches; every trial
               shows ONE of them in a random grid cell -- all 49 cells equally often,
               the central fixation cell included -- and the agent reports which one.
               It learns to see patches, to look for them anywhere (fovea and
               periphery), and to map identity to a response from feedback, but never
               meets two-patch compounds, the config's two locations, the A/B/XOR
               rules or rule switches.

  rules        the experiment's own task with fresh patches and a random rule (A, B or
               AB) each mini-block: an 'experienced participant' that has already
               practised every rule type on this layout.

Pretrained agents are saved and reused whenever the settings match. Then:

  experiment   from the pretrained weights: 3 blocks x --trials trials, new real patches
               every block (4 of the 20 per block), ONE UPDATE PER TRIAL -- the same
               trial-by-trial regime as the kernel model. Rule stays the same for blocks
               1-2 and changes in block 3. Every scenario starts from the same pretrained
               agents with a fresh optimizer.

Scenarios (2x2): A-A-B (switch relevant dimension), A-A-AB (switch to XOR),
AB-AB-A (switch from XOR).

Outputs in ~/Documents/data/catlearn_eeg/ram_model/2x2/pretrained-<task>_config<NN>_<sensor>/:
  pretrained_agents.npz     stacked weights of all agents + pretraining settings
  pretraining.png           accuracy at the end of each mini-block over pretraining
  blocks_<scenario>.png     per-trial accuracy, where gaze aims, end-of-block fixations
  summary_blocks.png        block means, and early vs late thirds of each block
  experiment_logs.npz       all per-trial metrics per agent

  ~/miniforge3/envs/kernelbehav/bin/python R03_pretrain_blocks.py --config 2 --sensor full
"""

import argparse
import json
import time
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
from ram import init_params, make_train_step

SCENARIOS = {
    "A-A-B": ["A", "A", "B"],
    "A-A-AB": ["A", "A", "AB"],
    "AB-AB-A": ["AB", "AB", "A"],
}
METRICS = ("p_correct", "reward", "saccade_dist", "aim_A", "aim_B", "first_aim_A", "first_aim_B")
PRETRAIN_KEYS = ("pretrain_task", "config", "sensor", "periph_noise", "saccade_cost", "agents", "glimpses", "sigma", "cell_px",
                 "pre_blocks", "pre_updates", "pre_batch", "pre_lr", "bank", "bank_seed", "seed")


def sensor_kwargs(args):
    """Sensor settings plus the saccade cost -- everything about the agent passed to make_train_step."""
    s = SENSORS[args.sensor]
    return {"scales": s["scales"], "full_view": s["full_view"],
            "periph_noise": args.periph_noise if args.sensor == "noisy" else 0.0,
            "saccade_cost": args.saccade_cost}


def n_channels(args):
    s = SENSORS[args.sensor]
    return len(s["scales"]) + s["full_view"]


# ---------------------------------------------------------------- saving / loading

def save_agents(path, params, meta):
    flat = {"/".join(str(getattr(k, "key", k)) for k in p): np.asarray(v)
            for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(path, meta=json.dumps(meta), **flat)


def load_agents(path):
    z = np.load(path)
    params = {}
    for name in z.files:
        if name != "meta":
            layer, leaf = name.split("/")
            params.setdefault(layer, {})[leaf] = jnp.asarray(z[name])
    return params, json.loads(str(z["meta"]))


# ---------------------------------------------------------------- pretraining

def pretrain(design, args):
    size = tile_size(args.cell_px)
    bank = make_patch_bank(args.bank, size, seed=args.bank_seed)
    rng = np.random.default_rng(args.seed)
    rule_labels = np.stack([block_labels(design, r) for r in ("A", "B", "AB")])

    optimizer = optax.adam(args.pre_lr)
    params = jax.vmap(partial(init_params, n_channels=n_channels(args)))(jax.random.split(jax.random.PRNGKey(args.seed), args.agents))
    opt_state = jax.vmap(optimizer.init)(params)
    step = jax.jit(jax.vmap(make_train_step(optimizer, args.pre_batch, args.glimpses, True, args.sigma, **sensor_kwargs(args)),
                            in_axes=(0, 0, 0, 0, 0, None)))
    key = jax.random.PRNGKey(args.seed + 1)
    curve = np.zeros((args.pre_blocks, args.agents))
    t0 = time.time()
    _, ab_locs = render_tiles(args.config, bank[:4], args.cell_px)   # only used for gaze metrics
    ab_locs = jnp.asarray(ab_locs)
    for mb in range(args.pre_blocks):
        if args.pretrain_task == "perception":
            # 2 fresh patches per agent, each shown alone in every one of the 49 cells
            picks = np.stack([rng.choice(len(bank), 2, replace=False) for _ in range(args.agents)])
            rendered = [render_single_everywhere(bank[p], args.cell_px) for p in picks]
            images = jnp.asarray(np.stack([r[0] for r in rendered]))
            labels = jnp.asarray(np.stack([r[1] for r in rendered]))
        else:
            # the experiment's task: 4 fresh patches and a random rule per agent
            picks = np.stack([rng.choice(len(bank), 4, replace=False) for _ in range(args.agents)])
            images = jnp.asarray(np.stack([render_tiles(args.config, bank[p], args.cell_px)[0] for p in picks]))
            labels = jnp.asarray(rule_labels[rng.integers(0, 3, args.agents)])
        acc = []
        for u in range(args.pre_updates):
            key, sub = jax.random.split(key)
            params, opt_state, m = step(params, opt_state, jax.random.split(sub, args.agents), images, labels, ab_locs)
            if u >= args.pre_updates - 5:
                acc.append(np.asarray(m["p_correct"]))
        curve[mb] = np.mean(acc, axis=0)
        if (mb + 1) % max(1, args.pre_blocks // 10) == 0:
            print(f"  pretraining mini-block {mb + 1}/{args.pre_blocks} ({time.time() - t0:.0f}s): "
                  f"end-of-mini-block p(correct) {curve[max(0, mb - 9):mb + 1].mean():.2f}", flush=True)
    return params, curve


def plot_pretraining(curve, args, out_dir):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(1, len(curve) + 1)
    m, sem = curve.mean(1), curve.std(1) / np.sqrt(curve.shape[1])
    ax.plot(x, m, color=FOREGROUND)
    ax.fill_between(x, m - sem, m + sem, color=FOREGROUND, alpha=0.2, lw=0)
    ax.axhline(0.5, color="0.4", lw=0.8)
    ax.set_ylim(0.4, 1.02)
    what = ("2 new patches, one per trial in a random cell of all 49" if args.pretrain_task == "perception"
            else "4 new patches + random A/B/AB rule")
    ax.set_xlabel(f"pretraining mini-block ({args.pre_updates} updates x {args.pre_batch} trials; {what})", fontsize=8)
    ax.set_ylabel("p(correct) at\nend of mini-block")
    ax.set_title(f"'{args.pretrain_task}' pretraining, {args.agents} agents, sensor '{args.sensor}'", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "pretraining.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- experiment

def run_scenario(params0, design, task_names, args, seed):
    n_blocks = len(task_names)
    rng = np.random.default_rng(seed)
    size = tile_size(args.cell_px)
    real = np.stack([load_patch(k, size) for k in range(1, N_POOL + 1)])
    patches = np.stack([rng.choice(N_POOL, 4 * n_blocks, replace=False) for _ in range(args.agents)])  # 0-based
    images = np.stack([[render_tiles(args.config, real[patches[a, 4 * b:4 * b + 4]], args.cell_px)[0]
                        for b in range(n_blocks)] for a in range(args.agents)])       # (agents, blocks, 4, H, W)
    _, ab_locs = render_tiles(args.config, real[:4], args.cell_px)
    images, ab_locs = jnp.asarray(images), jnp.asarray(ab_locs)

    optimizer = optax.adam(args.test_lr)
    params, opt_state = params0, jax.vmap(optimizer.init)(params0)
    step = jax.jit(jax.vmap(make_train_step(optimizer, args.test_batch, args.glimpses, True, args.sigma, **sensor_kwargs(args)),
                            in_axes=(0, 0, 0, 0, None, None)))
    key = jax.random.PRNGKey(seed)
    n_updates = args.trials // args.test_batch
    logs = {m: np.zeros((args.agents, n_blocks * n_updates)) for m in METRICS}
    end_locs = []
    for b, task in enumerate(task_names):
        labels = jnp.asarray(block_labels(design, task))
        tail = []
        for u in range(n_updates):
            key, sub = jax.random.split(key)
            params, opt_state, m = step(params, opt_state, jax.random.split(sub, args.agents), images[:, b], labels, ab_locs)
            for name in METRICS:
                logs[name][:, b * n_updates + u] = np.asarray(m[name])
            if u >= n_updates - args.heatmap_trials // args.test_batch:
                tail.append(np.asarray(m["locs"]))
        end_locs.append(np.concatenate(tail, axis=1))                             # (agents, trials, glimpses-1, 2)
    return logs, end_locs, patches + 1


def smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.stack([np.convolve(np.pad(r, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid") for r in x])


def thirds(logs, metric, n_blocks, n_per_block):
    """(agents, blocks, 3): mean in the first, middle and last third of each block."""
    v = logs[metric].reshape(logs[metric].shape[0], n_blocks, n_per_block)
    return np.stack([c.mean(-1) for c in np.array_split(v, 3, axis=-1)], axis=-1)


def plot_scenario(design, name, task_names, logs, end_locs, args, out_dir):
    n_blocks = len(task_names)
    n_per = args.trials // args.test_batch
    x = (np.arange(n_blocks * n_per) + 1) * args.test_batch
    fig = plt.figure(figsize=(3.4 * n_blocks + 1, 8.5))
    gs = fig.add_gridspec(3, n_blocks, height_ratios=[1, 1, 1.15])
    ax_acc = fig.add_subplot(gs[0, :])
    ax_aim = fig.add_subplot(gs[1, :], sharex=ax_acc)

    v = smooth(logs["p_correct"], args.smooth)
    m, sem = v.mean(0), v.std(0) / np.sqrt(v.shape[0])
    ax_acc.plot(x, m, color=FOREGROUND, lw=1.5)
    ax_acc.fill_between(x, m - sem, m + sem, color=FOREGROUND, alpha=0.2, lw=0)
    ax_acc.axhline(0.5, color="0.4", lw=0.8)
    ax_acc.set_ylim(0.4, 1.02)
    ax_acc.set_ylabel("p(correct)")

    for loc in ("A", "B"):
        v = smooth(logs[f"aim_{loc}"], args.smooth)
        m, sem = v.mean(0), v.std(0) / np.sqrt(v.shape[0])
        ax_aim.plot(x, m, color=LOC_COLORS[loc], lw=1.8, label=f"aims at location {loc}")
        ax_aim.fill_between(x, m - sem, m + sem, color=LOC_COLORS[loc], alpha=0.2, lw=0)
    ax_aim.set_ylim(-0.02, 1.02)
    ax_aim.set_ylabel("fraction of chosen\nfixations (policy mean)")
    ax_aim.set_xlabel("trial")
    ax_sac = ax_aim.twinx()
    v = smooth(logs["saccade_dist"], args.smooth).mean(0)
    ax_sac.plot(x, v, color="0.7", ls=":", lw=1.2, label="fixation travel per trial (right axis)")
    ax_sac.set_ylim(0, None)
    ax_sac.set_ylabel("fixation travel\n(display units)", color="0.7", fontsize=8)
    ax_sac.tick_params(axis="y", colors="0.7", labelsize=8)
    handles = ax_aim.get_legend_handles_labels()[0] + ax_sac.get_legend_handles_labels()[0]
    ax_aim.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=9, loc="upper left")
    for ax in (ax_acc, ax_aim):
        for b in range(1, n_blocks):
            ax.axvline(b * args.trials + 0.5, color=ACCENT, ls="--", lw=1)
    for b, t in enumerate(task_names):
        ax_acc.text((b + 0.5) / n_blocks, 1.02, f"block {b + 1}: {design.task_label(design.task_index(t))}",
                    transform=ax_acc.transAxes, ha="center", va="bottom", fontsize=9, color=ACCENT)

    _, cells = spatial_config(args.config)
    for b in range(n_blocks):
        ax = fig.add_subplot(gs[2, b])
        pts = end_locs[b].reshape(-1, 2)
        ax.hist2d(pts[:, 0], pts[:, 1], bins=GRID * 4, range=[[-1, 1], [-1, 1]], cmap="magma")
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
        ax.set_title(f"fixations, last {args.heatmap_trials} trials of block {b + 1}", fontsize=9)

    config_name, _ = spatial_config(args.config)
    fig.suptitle(f"RAM after '{args.pretrain_task}' pretraining, {name} | config {args.config} ({config_name}), sensor '{args.sensor}', "
                 f"saccade cost {args.saccade_cost} | "
                 f"{args.agents} agents, {args.trials} trials/block, 1 update per {args.test_batch} trial(s)",
                 y=0.995, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"blocks_{name}.png", dpi=150)
    plt.close(fig)


def plot_summary(all_logs, args, out_dir):
    names = list(all_logs)
    fig, axes = plt.subplots(1, len(names), figsize=(3.8 * len(names), 3.8), sharey=True, squeeze=False)
    n_per = args.trials // args.test_batch
    for ax, name in zip(axes[0], names):
        tasks = SCENARIOS[name]
        th = thirds(all_logs[name], "p_correct", len(tasks), n_per)               # (agents, blocks, 3)
        m, sem = th.mean(0), th.std(0) / np.sqrt(th.shape[0])
        for b in range(len(tasks)):
            xs = b + np.array([0.2, 0.5, 0.8])
            ax.errorbar(xs, m[b], yerr=sem[b], color=FOREGROUND, marker="o", ms=4, capsize=2)
        ax.set_xticks(np.arange(len(tasks)) + 0.5, [f"block {b + 1}\n{t}" for b, t in enumerate(tasks)])
        for b in range(1, len(tasks)):
            ax.axvline(b, color=ACCENT, ls="--", lw=1)
        ax.axhline(0.5, color="0.4", lw=0.8)
        ax.set_title(name, fontsize=10)
    axes[0][0].set_ylabel("p(correct), first / middle / last\nthird of each block")
    fig.suptitle(f"pretrained RAM, {args.trials} trials per block (points: thirds of each block)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_blocks.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=int, default=2)
    parser.add_argument("--sensor", choices=list(SENSORS), default="full")
    parser.add_argument("--periph-noise", type=float, default=0.3)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--glimpses", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.3, help="fixation noise SD, [-1, 1] display units")
    parser.add_argument("--saccade-cost", type=float, default=0.2,
                        help="gaze-reward penalty per unit of fixation travel, in pretraining and experiment (0 = none)")
    parser.add_argument("--cell-px", type=int, default=16)
    # pretraining
    parser.add_argument("--pretrain-task", choices=["perception", "rules"], default="perception")
    parser.add_argument("--pre-blocks", type=int, default=150, help="pretraining mini-blocks")
    parser.add_argument("--pre-updates", type=int, default=40, help="updates per pretraining mini-block")
    parser.add_argument("--pre-batch", type=int, default=64)
    parser.add_argument("--pre-lr", type=float, default=1e-3)
    parser.add_argument("--bank", type=int, default=4000, help="number of generated pretraining patches")
    parser.add_argument("--bank-seed", type=int, default=1)
    parser.add_argument("--retrain", action="store_true", help="ignore saved pretrained agents")
    # experiment
    parser.add_argument("--trials", type=int, default=1000, help="trials per block")
    parser.add_argument("--test-batch", type=int, default=1, help="trials per update in the experiment (1 = trial by trial)")
    parser.add_argument("--test-lr", type=float, default=1e-3)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--heatmap-trials", type=int, default=100)
    parser.add_argument("--smooth", type=int, default=50, help="moving average over trials, for plots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-tag", default="", help="suffix for the output folder")
    args = parser.parse_args()
    apply_dark_theme()

    design = Design("2x2")
    out_dir = DATA_ROOT / "ram_model" / "2x2" / f"pretrained-{args.pretrain_task}_config{args.config:02d}_{args.sensor}{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pretrain, or reuse agents pretrained with identical settings
    agents_path = out_dir / "pretrained_agents.npz"
    pre_meta = {k: getattr(args, k) for k in PRETRAIN_KEYS}
    params0 = None
    if agents_path.exists() and not args.retrain:
        params0, meta = load_agents(agents_path)
        if {k: meta.get(k) for k in PRETRAIN_KEYS} != pre_meta:
            print("saved pretrained agents used different settings; pretraining again")
            params0 = None
        else:
            print(f"loaded pretrained agents from {agents_path}")
    if params0 is None:
        print(f"pretraining {args.agents} agents: {args.pre_blocks} mini-blocks x {args.pre_updates} updates x {args.pre_batch} trials")
        params0, curve = pretrain(design, args)
        save_agents(agents_path, params0, {**pre_meta, "pretrain_curve": curve.tolist()})
        plot_pretraining(curve, args, out_dir)

    # Experiment
    all_logs, saved = {}, {}
    n_per = args.trials // args.test_batch
    for s_i, name in enumerate(args.scenarios):
        tasks = SCENARIOS[name]
        t0 = time.time()
        logs, end_locs, patches = run_scenario(params0, design, tasks, args, seed=args.seed + 100 + s_i)
        all_logs[name] = logs
        plot_scenario(design, name, tasks, logs, end_locs, args, out_dir)
        th = thirds(logs, "p_correct", len(tasks), n_per).mean(0)
        aim_a = thirds(logs, "aim_A", len(tasks), n_per).mean(0)[:, 2]
        aim_b = thirds(logs, "aim_B", len(tasks), n_per).mean(0)[:, 2]
        print(f"{name:8s} ({time.time() - t0:4.0f}s) p(correct) first/last third: "
              + "  ".join(f"B{b + 1} {th[b, 0]:.2f}->{th[b, 2]:.2f}" for b in range(len(tasks)))
              + " | aims A/B, last third: " + "  ".join(f"B{b + 1} {a:.2f}/{c:.2f}" for b, (a, c) in enumerate(zip(aim_a, aim_b)))
              + " | travel/trial, last third: " + "  ".join(f"B{b + 1} {v:.2f}" for b, v in
                                                          enumerate(thirds(logs, "saccade_dist", len(tasks), n_per).mean(0)[:, 2])),
              flush=True)
        for metric, arr in logs.items():
            saved[f"{name}/{metric}"] = arr
        saved[f"{name}/patches"] = patches
    plot_summary(all_logs, args, out_dir)
    np.savez(out_dir / "experiment_logs.npz", params=json.dumps(vars(args)), scenarios=json.dumps(SCENARIOS), **saved)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
