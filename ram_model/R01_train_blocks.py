"""Train Recurrent Attention Model agents through the 3-block design, free gaze vs fixate.

Same scenarios as the kernel model (kernel_model/K01_simulate_blocks.py): blocks 1-2 use
a rule of one type, block 3 switches type. Each agent is one simulated subject with its
own randomly drawn noise patches, new patches every block (4 per block from the pool of
20), trained continuously across blocks -- weights and optimizer state carry over, so
anything learned about *where to look* can transfer or interfere.

Two conditions per scenario, with identical agents, patches and random seeds:
  free    the agent chooses its fixations (glimpse 0 is always central)
  fixate  every glimpse is central: only covert, peripheral information is available

Sensor presets (see env_2by2.glimpse and ram.episode):
  full    fovea + 2x + 4x crops + whole-display view: the periphery alone solves the task
  noisy   as full, plus fresh noise on every non-foveal channel (--periph-noise): covert
          information is available but unreliable, so looking pays off
  graded  fovea + 2x + 4x crops, no whole-display view: acuity falls off with distance
          from fixation and nothing beyond ~2 cells is seen, so where you look matters
  weak    fovea + 2x crop only: from the center the patches are invisible

Outputs go to ~/Documents/data/catlearn_eeg/ram_model/2x2/config<NN>_<sensor>/:
  ram_<scenario>.png   accuracy (free vs fixate), where the policy aims (A vs B), and
                       fixation heatmaps at the end of each block
  ram_logs.npz         all logged metrics per agent, plus parameters

Run with the kernelbehav env:
  ~/miniforge3/envs/kernelbehav/bin/python R01_train_blocks.py --config 2 --sensor noisy
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

from env_2by2 import DATA_ROOT, GRID, N_POOL, Design, block_labels, render_block, spatial_config
from K01_simulate_blocks import default_scenarios  # kernel_model is on sys.path via env_2by2
from plot_style import ACCENT, FOREGROUND, apply_dark_theme
from ram import init_params, make_train_step

SENSORS = {
    "full": {"scales": (16, 32, 64), "full_view": True},
    "noisy": {"scales": (16, 32, 64), "full_view": True},
    "graded": {"scales": (16, 32, 64), "full_view": False},
    "weak": {"scales": (16, 32), "full_view": False},
}
LOC_COLORS = {"A": "#ffa630", "B": "#c77dff"}  # as in the design figures
SCALAR_METRICS = ("p_correct", "reward", "gaze_A", "gaze_B", "aim_A", "aim_B", "first_aim_A", "first_aim_B")


def train_scenario(design, task_names, free_gaze, args, seed):
    """Train args.agents agents through the blocks. Returns logs and end-of-block fixations."""
    n_blocks = len(task_names)
    sensor = SENSORS[args.sensor]
    periph_noise = args.periph_noise if args.sensor == "noisy" else 0.0
    n_channels = len(sensor["scales"]) + sensor["full_view"]

    # Each agent: its own 4 new patches per block, drawn without replacement from the pool
    rng = np.random.default_rng(seed)
    patches = np.stack([rng.choice(np.arange(1, N_POOL + 1), 4 * n_blocks, replace=False) for _ in range(args.agents)])
    images, ab_locs = [], None
    for a in range(args.agents):
        per_block = []
        for b in range(n_blocks):
            imgs, ab_locs = render_block(args.config, patches[a, 4 * b:4 * b + 4], cell_px=args.cell_px)
            per_block.append(imgs)
        images.append(per_block)
    images = jnp.asarray(np.array(images))                               # (agents, blocks, 4, H, W)
    labels = jnp.asarray(np.stack([block_labels(design, t) for t in task_names]))  # (blocks, 4)
    ab_locs = jnp.asarray(ab_locs)

    optimizer = optax.adam(args.lr)
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)
    params = jax.vmap(partial(init_params, n_channels=n_channels, glimpse_px=16))(jax.random.split(k_init, args.agents))
    opt_state = jax.vmap(optimizer.init)(params)
    step = jax.jit(jax.vmap(
        make_train_step(optimizer, args.batch, args.glimpses, free_gaze, args.sigma,
                        scales=sensor["scales"], full_view=sensor["full_view"], periph_noise=periph_noise),
        in_axes=(0, 0, 0, 0, None, None)))

    logs = {m: [] for m in SCALAR_METRICS}
    log_block, end_locs = [], []
    for b in range(n_blocks):
        for u in range(args.updates):
            key, sub = jax.random.split(key)
            params, opt_state, m = step(params, opt_state, jax.random.split(sub, args.agents), images[:, b], labels[b], ab_locs)
            if u % args.log_every == 0 or u == args.updates - 1:
                for name in SCALAR_METRICS:
                    logs[name].append(np.asarray(m[name]))
                log_block.append(b)
        end_locs.append(np.asarray(m["locs"]))                            # (agents, batch, glimpses-1, 2)
    logs = {k: np.stack(v, axis=1) for k, v in logs.items()}              # (agents, n_logged)
    return logs, np.array(log_block), end_locs, patches


def smooth(x, w):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.stack([np.convolve(np.pad(r, (w // 2, w - 1 - w // 2), mode="edge"), kernel, mode="valid") for r in x])


def block_end_mean(logs, log_block, metric, frac=0.1):
    """Per-agent mean of a metric over the last `frac` of each block's logs: (agents, blocks)."""
    out = []
    for b in np.unique(log_block):
        idx = np.where(log_block == b)[0]
        out.append(logs[metric][:, idx[-max(1, int(len(idx) * frac)):]].mean(axis=1))
    return np.stack(out, axis=1)


def plot_scenario(design, name, task_names, results, args, out_dir):
    n_blocks = len(task_names)
    free_logs, log_block, free_end, _ = results["free"]
    fix_logs = results["fixate"][0]
    x = np.arange(len(log_block)) * args.log_every
    boundaries = [x[np.argmax(log_block == b)] for b in range(1, n_blocks)]
    w = args.smooth

    fig = plt.figure(figsize=(3.4 * n_blocks + 1, 8.5))
    gs = fig.add_gridspec(3, n_blocks, height_ratios=[1, 1, 1.15])
    ax_acc = fig.add_subplot(gs[0, :])
    ax_aim = fig.add_subplot(gs[1, :], sharex=ax_acc)

    for logs, ls, label in ((free_logs, "-", "free gaze"), (fix_logs, "--", "fixate")):
        v = smooth(logs["p_correct"], w)
        m, sem = v.mean(0), v.std(0) / np.sqrt(v.shape[0])
        ax_acc.plot(x, m, ls=ls, color=FOREGROUND, lw=1.5, label=label)
        ax_acc.fill_between(x, m - sem, m + sem, color=FOREGROUND, alpha=0.15, lw=0)
    ax_acc.axhline(0.5, color="0.4", lw=0.8)
    ax_acc.set_ylim(0.4, 1.02)
    ax_acc.set_ylabel("p(correct)")
    ax_acc.legend(frameon=False, fontsize=9, loc="lower right")

    for loc in ("A", "B"):
        v = smooth(free_logs[f"aim_{loc}"], w)
        m, sem = v.mean(0), v.std(0) / np.sqrt(v.shape[0])
        ax_aim.plot(x, m, color=LOC_COLORS[loc], lw=1.8, label=f"aims at location {loc}")
        ax_aim.fill_between(x, m - sem, m + sem, color=LOC_COLORS[loc], alpha=0.2, lw=0)
    ax_aim.set_ylim(-0.02, 1.02)
    ax_aim.set_ylabel("fraction of chosen\nfixations (policy mean)")
    ax_aim.set_xlabel("training updates")
    ax_aim.legend(frameon=False, fontsize=9, loc="upper left")

    for ax in (ax_acc, ax_aim):
        for bd in boundaries:
            ax.axvline(bd, color=ACCENT, ls="--", lw=1)
    for b, t in enumerate(task_names):
        ax_acc.text((b + 0.5) / n_blocks, 1.02, f"block {b + 1}: {design.task_label(design.task_index(t))}",
                    transform=ax_acc.transAxes, ha="center", va="bottom", fontsize=9, color=ACCENT)

    # Where sampled fixations landed at the end of each block (free gaze, all agents)
    _, cells = spatial_config(args.config)
    for b in range(n_blocks):
        ax = fig.add_subplot(gs[2, b])
        pts = free_end[b].reshape(-1, 2)
        ax.hist2d(pts[:, 0], pts[:, 1], bins=GRID * 4, range=[[-1, 1], [-1, 1]], cmap="magma")
        for k in range(1, GRID):
            ax.axhline(-1 + 2 * k / GRID, color="0.35", lw=0.5)
            ax.axvline(-1 + 2 * k / GRID, color="0.35", lw=0.5)
        for loc, (row, col) in zip("AB", cells):
            x0, y0 = -1 + 2 * col / GRID, -1 + 2 * row / GRID
            ax.add_patch(plt.Rectangle((x0, y0), 2 / GRID, 2 / GRID, fill=False, ec=LOC_COLORS[loc], lw=2))
            ax.text(x0 + 0.02, y0 + 0.02, loc, color=LOC_COLORS[loc], fontsize=9, fontweight="bold", va="top")
        ax.set_xlim(-1, 1)
        ax.set_ylim(1, -1)  # y downward, like the screen
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"fixations, end of block {b + 1}", fontsize=9)

    config_name, _ = spatial_config(args.config)
    fig.suptitle(f"RAM, {design.name} {name} | config {args.config} ({config_name}), sensor '{args.sensor}'"
                 + (f" (noise {args.periph_noise})" if args.sensor == "noisy" else "")
                 + f", {args.agents} agents", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"ram_{name}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=int, default=2, help="spatial configuration 1-18 (2 = horizontal, mid)")
    parser.add_argument("--sensor", choices=list(SENSORS), default="noisy")
    parser.add_argument("--periph-noise", type=float, default=0.3, help="noise SD on non-foveal channels (sensor 'noisy')")
    parser.add_argument("--scenarios", nargs="+", default=None, help="subset of scenario names (default: all for 2x2)")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--updates", type=int, default=3000, help="training updates per block")
    parser.add_argument("--batch", type=int, default=64, help="trials per update")
    parser.add_argument("--glimpses", type=int, default=4, help="glimpses per trial, the first at central fixation")
    parser.add_argument("--sigma", type=float, default=0.3, help="fixation noise SD, in [-1, 1] display units")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cell-px", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smooth", type=int, default=10, help="moving-average window over logged points, for plots")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    apply_dark_theme()

    design = Design("2x2")
    scenarios = default_scenarios(design)
    if args.scenarios:
        scenarios = {k: v for k, v in scenarios.items() if k in args.scenarios}
    out_dir = DATA_ROOT / "ram_model" / design.name / f"config{args.config:02d}_{args.sensor}"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    for s_i, (name, task_names) in enumerate(scenarios.items()):
        seed = args.seed + 1000 * s_i
        results = {}
        for cond, free in (("free", True), ("fixate", False)):
            t0 = time.time()
            results[cond] = train_scenario(design, task_names, free, args, seed)
            logs, log_block = results[cond][0], results[cond][1]
            acc = block_end_mean(logs, log_block, "p_correct").mean(0)
            line = f"{name:18s} {cond:6s} ({time.time() - t0:5.0f}s) end-of-block p(correct): " + \
                   "  ".join(f"B{b + 1} {v:.2f}" for b, v in enumerate(acc))
            if free:
                aim_a = block_end_mean(logs, log_block, "aim_A").mean(0)
                aim_b = block_end_mean(logs, log_block, "aim_B").mean(0)
                line += " | aims A/B: " + "  ".join(f"B{b + 1} {a:.2f}/{bb:.2f}" for b, (a, bb) in enumerate(zip(aim_a, aim_b)))
            print(line, flush=True)
            for metric, arr in logs.items():
                saved[f"{name}/{cond}/{metric}"] = arr
            saved[f"{name}/{cond}/log_block"] = log_block
            saved[f"{name}/{cond}/patches"] = results[cond][3]
        plot_scenario(design, name, task_names, results, args, out_dir)

    np.savez(out_dir / "ram_logs.npz", params=json.dumps(vars(args)), scenarios=json.dumps(scenarios), **saved)
    print(f"Saved figures and ram_logs.npz to {out_dir}")


if __name__ == "__main__":
    main()
