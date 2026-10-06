"""Analysis of R05 runs across spatial configs: performance by task, gaze, kernel modes.

Reads the per-config folders written by R05_gpu_blocks.py (see DATA_FORMAT.md in each)
and makes one set of figures comparing configs:

  performance.png          accuracy and RT over the 3 blocks, per scenario, one line per config
  performance_agents.png   accuracy of every agent (learning is often all-or-none per agent)
  task_summary.png         per learning episode (rule x history): final accuracy, trials to
                           criterion, final RT -- every agent a dot
  gaze_heatmaps.png        stimulus-phase fixations at the end of each block
  gaze_timecourse.png      where gaze aims along the A-B axis, step by step within a trial
                           (fixation period -> stimulus), at the end of each block
  gaze_learning.png        gaze position at stimulus onset over training, A-B axis
  behavior_modes.png       the agents' choice function decomposed into kernel modes
                           (how strongly responses follow A, B and the A x B interaction)
  ntk_modes.png            the network's empirical NTK decomposed into modes, at the start
                           and after each block (its learning bias for the next block)
  ntk_vs_learning.png      does the NTK's weight on a block's rule predict how fast that
                           block is learned?

Kernel modes (kernel_model/kernel_modes.py): over the 4 compounds, the constant, A, B and
AB (XOR) directions. For behavior, y(s) = 2 P(choice 1 | s) - 1 and a mode's strength is
|P_e y| / 2, which is 1 for a perfectly performed task of that mode. For the NTK, a
mode's share is tr(P_e K) / tr(K), measured on held-out fractals (see ram_mlx.empirical_ntk
and R06_ntk_modes.py); the "stimulus-dependent" split is that share among A, B, AB.

  ~/miniforge3/envs/kernelbehav/bin/python R07_analyze_configs.py --configs 1 4 7 10 13 16
"""

import argparse
import csv
import json
import warnings

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from scipy.stats import spearmanr

import ram_mlx as R
from env_2by2 import DATA_ROOT, GRID, Design, load_stimulus_set, render_tiles, spatial_config, tile_size
from plot_style import ACCENT, FOREGROUND, apply_dark_theme
from R01_train_blocks import LOC_COLORS, SENSORS
from R03_pretrain_blocks import smooth
from R04_timed_blocks import smooth_nan

MODE_COLORS = {"cst": "0.6", "A": LOC_COLORS["A"], "B": LOC_COLORS["B"], "AB": "#4cc9f0"}
CONFIG_COLORS = ["#ff6b6b", "#ffd166", "#06d6a0", "#4cc9f0", "#c77dff", "#f78c6b", "#a0c4ff", "#caffbf"]
# Learning episodes: (label, [(scenario, block), ...]) -- same rule and history pooled
EPISODES = [
    ("A\nfirst block", [("A-A-B", 0), ("A-A-AB", 0)]),
    ("A\n2nd block", [("A-A-B", 1), ("A-A-AB", 1)]),
    ("B after A\n(shift)", [("A-A-B", 2)]),
    ("AB after A", [("A-A-AB", 2)]),
    ("AB\nfirst block", [("AB-AB-A", 0)]),
    ("AB\n2nd block", [("AB-AB-A", 1)]),
    ("A after AB", [("AB-AB-A", 2)]),
]
CRITERION = 0.8


# ---------------------------------------------------------------- loading

def load_config(config, tag):
    d = DATA_ROOT / "ram_model" / "2x2" / f"timed_config{config:02d}{tag}"
    meta = json.loads((d / "meta.json").read_text())
    logs = np.load(d / "experiment_logs.npz")
    scenarios = json.loads(str(logs["scenarios"]))
    run = {"dir": d, "meta": meta, "scenarios": scenarios, "ab_locs": logs["ab_locs"], "logs": {}, "trials": {},
           "weights": {}, "patches": {}}
    for scen in scenarios:
        run["logs"][scen] = {k.split("/", 1)[1]: logs[k] for k in logs.files if k.startswith(scen + "/")}
        run["patches"][scen] = logs[f"{scen}/patches"]
        z = np.load(d / f"trials_{scen}.npz")
        run["trials"][scen] = {k: z[k] for k in z.files}
        run["weights"][scen] = np.load(d / f"weights_{scen}.npz")
    return run


def block_params(wz, b):
    """Weights at the end of block b from a weights_<scen>.npz, as a nested numpy dict."""
    p = {}
    for name in wz.files:
        blk, layer, leaf = name.split("/")
        if blk == f"block{b}":
            p.setdefault(layer, {})[leaf] = wz[name]
    return p


# ---------------------------------------------------------------- measures

def ab_axis(locs, ab):
    """Position along the A-B axis: A center = -1, B center = +1 (near configs are symmetric
    about fixation, so fixation = 0)."""
    mid, half = (ab[0] + ab[1]) / 2, (ab[1] - ab[0]) / 2
    return ((locs - mid) @ half) / (half @ half)


def trials_to_criterion(acc, U, batch, w=20):
    """Trials before accuracy reaches CRITERION: the first update from which the mean of the
    next w updates is >= CRITERION (0 = already there from the block's start, e.g. by
    transfer); NaN if never."""
    c = np.pad(np.cumsum(acc, axis=1), [(0, 0), (1, 0)])
    ahead = (c[:, w:] - c[:, :-w]) / w                                   # mean of updates u..u+w-1
    hit = ahead >= CRITERION
    first = np.where(hit.any(1), hit.argmax(1), -1)
    return np.where(first >= 0, first * batch, np.nan)


def choice_modes(tr, design, window=5):
    """Behavioral mode strengths |P_e y| / 2 per agent, in windows of `window` logged updates.

    Returns (agents, n_windows, modes) and the window centers (logged-update index)."""
    stim, choice = tr["stim"].astype(int), tr["choice"].astype(int)
    A, L, _ = stim.shape
    n_win = L // window
    out = np.full((A, n_win, design.n_modes), np.nan)
    for w in range(n_win):
        sl = slice(w * window, (w + 1) * window)
        s, c = stim[:, sl].reshape(A, -1), choice[:, sl].reshape(A, -1)
        y = np.full((A, 4), np.nan)
        for k in range(4):
            m = (s == k) & (c >= 0)
            y[:, k] = 2 * np.where(m.sum(1) > 0, ((c == 1) & m).sum(1) / np.maximum(m.sum(1), 1), 0.5) - 1
        out[:, w] = np.sqrt(np.einsum("an,emn,am->ae", y, design.projectors, y).clip(0)) / 2
    return out, np.arange(n_win) * window + window / 2


def ntk_shares(runs, configs, design, pool, k_stims, n_draws, seed):
    """Empirical-NTK mode shares on held-out fractals, per config / scenario / state / agent.

    States: 0 = initial weights, b + 1 = after block b. Returns two dicts {k: array
    (configs, scen, 4 states, agents, modes)} averaged over n_draws held-out stimulus sets:
    the shares tr(P_e K) / tr(K), and the absolute powers tr(P_e K).
    """
    scen_names = list(runs[configs[0]]["scenarios"])
    meta = runs[configs[0]]["meta"]
    s = SENSORS[meta["params"]["sensor"]]
    cfg = {**meta["cfg"], "scales": tuple(s["scales"]), "full_view": s["full_view"]}
    K_ag = meta["params"]["agents"]
    n_ch = R.n_sensor_channels(cfg, 3 if meta["params"]["color"] else 1)
    init = R.to_numpy(R.init_params(K_ag, n_channels=n_ch, seed=meta["params"]["seed"]))
    # orthonormal basis of every mode (rows), and which mode each row belongs to
    basis, owner = [], []
    for e, P in enumerate(design.projectors):
        lam, vec = np.linalg.eigh(P)
        for j in np.where(lam > 0.5)[0]:
            basis.append(vec[:, j])
            owner.append(e)
    basis = np.array(basis)
    onehot = np.eye(design.n_modes)[owner]                                # (rows, modes)
    rng = np.random.default_rng(seed)
    out = {k: np.zeros((len(configs), len(scen_names), 4, K_ag, design.n_modes)) for k in k_stims}
    out_power = {k: np.zeros_like(v) for k, v in out.items()}
    for ci, config in enumerate(configs):
        run = runs[config]
        # one GPU batch: every (scenario, state, agent, draw)
        sets, images, index = [], [], []
        for si, scen in enumerate(scen_names):
            states = [init] + [block_params(run["weights"][scen], b) for b in range(3)]
            for st, p in enumerate(states):
                for a in range(K_ag):
                    used = set(run["patches"][scen][a] - 1)
                    free = [i for i in range(len(pool)) if i not in used]
                    for dr in range(n_draws):
                        held = rng.choice(free, 4, replace=False)
                        images.append(render_tiles(config, pool[held])[0])
                        sets.append({layer: {leaf: v[a] for leaf, v in d.items()} for layer, d in p.items()})
                        index.append((si, st, a))
        params = {layer: {leaf: mx.array(np.stack([q[layer][leaf] for q in sets])) for leaf in sets[0][layer]}
                  for layer in sets[0]}
        images = mx.array(np.stack(images))
        for k in k_stims:
            power = R.ntk_directions(params, images, cfg, k, basis) @ onehot   # (N, modes) = tr(P_e K)
            shares = power / power.sum(1, keepdims=True)
            for (si, st, a), sh, pw in zip(index, shares, power):
                out[k][ci, si, st, a] += sh / n_draws
                out_power[k][ci, si, st, a] += pw / n_draws
        print(f"  NTK config {config} done", flush=True)
    return out, out_power


# ---------------------------------------------------------------- figures

def rule_titles(ax, scen_rules, design):
    for b, t in enumerate(scen_rules):
        ax.text((b + 0.5) / len(scen_rules), 1.01, f"block {b + 1}: {t}", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8, color=ACCENT)


def block_lines(ax, n_blocks, per_block):
    for b in range(1, n_blocks):
        ax.axvline(b * per_block, color=ACCENT, ls="--", lw=0.8)


def band(ax, x, v, color, label=None, lw=1.5, alpha=0.18):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(v, 0)
        sem = np.nanstd(v, 0) / np.sqrt(np.sum(np.isfinite(v), 0).clip(1))
    ax.plot(x, m, color=color, lw=lw, label=label)
    ax.fill_between(x, m - sem, m + sem, color=color, alpha=alpha, lw=0)


def config_label(config):
    return f"{config}: {spatial_config(config)[0]}"


def plot_performance(runs, configs, out_dir, U, batch, w=50):
    scen_names = list(runs[configs[0]]["scenarios"])
    x = (np.arange(3 * U) + 1) * batch / 1000
    fig, axes = plt.subplots(len(scen_names), 2, figsize=(14, 3.1 * len(scen_names)), squeeze=False)
    for si, scen in enumerate(scen_names):
        ax_a, ax_r = axes[si]
        for ci, config in enumerate(configs):
            lg = runs[config]["logs"][scen]
            band(ax_a, x, smooth(lg["acc"], w), CONFIG_COLORS[ci], config_label(config), lw=1.2)
            band(ax_r, x, smooth_nan(lg["rt_ms"], w), CONFIG_COLORS[ci], lw=1.2)
        ax_a.axhline(0.5, color="0.4", lw=0.7)
        ax_a.set_ylim(0.4, 1.02)
        ax_a.set_ylabel(f"{scen}\np(correct)")
        ax_r.set_ylabel("RT (ms), responded")
        ax_r.set_ylim(250, 1200)
        for ax in (ax_a, ax_r):
            block_lines(ax, 3, U * batch / 1000)
            rule_titles(ax, runs[configs[0]]["scenarios"][scen], None)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("trials (thousands)")
    fig.suptitle(f"performance by scenario (mean +- SEM over agents, smoothed over {w} updates)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "performance.png", dpi=150)
    plt.close(fig)


def plot_performance_agents(runs, configs, out_dir, U, batch, w=50):
    scen_names = list(runs[configs[0]]["scenarios"])
    x = (np.arange(3 * U) + 1) * batch / 1000
    fig, axes = plt.subplots(len(configs), len(scen_names), figsize=(4.6 * len(scen_names), 1.9 * len(configs)),
                             sharex=True, sharey=True, squeeze=False)
    for ci, config in enumerate(configs):
        for si, scen in enumerate(scen_names):
            ax = axes[ci, si]
            acc = smooth(runs[config]["logs"][scen]["acc"], w)
            for a, row in enumerate(acc):
                ax.plot(x, row, lw=0.9, color=plt.cm.tab10(a))
            ax.axhline(0.5, color="0.4", lw=0.6)
            block_lines(ax, 3, U * batch / 1000)
            ax.set_ylim(0.4, 1.02)
            if ci == 0:
                ax.set_title(scen, fontsize=10, pad=16)
                rule_titles(ax, runs[config]["scenarios"][scen], None)
            if si == 0:
                ax.set_ylabel(config_label(config).replace(": ", "\n"), fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("trials (thousands)")
    fig.suptitle("p(correct) of each agent (colors = agent; agent i starts from the same weights everywhere)",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_agents.png", dpi=150)
    plt.close(fig)


def episode_table(runs, configs, U, batch):
    rows = []
    for config in configs:
        run = runs[config]
        for scen, rules in run["scenarios"].items():
            lg = run["logs"][scen]
            for b in range(3):
                sl = slice(b * U, (b + 1) * U)
                ttc = trials_to_criterion(lg["acc"][:, sl], U, batch)
                tail = slice(b * U + 2 * U // 3, (b + 1) * U)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rt = np.nanmean(lg["rt_ms"][:, tail], 1)
                tr = run["trials"][scen]
                keep = tr["block"] == b
                keep[np.where(keep)[0][:-20]] = False                      # last 20 logged updates of the block
                onset = ab_axis(tr["onset_loc"][:, keep].astype(float), run["ab_locs"]).mean((1, 2))
                for a in range(len(ttc)):
                    rows.append({"config": config, "scenario": scen, "block": b + 1, "rule": rules[b], "agent": a,
                                 "final_acc": lg["acc"][a, tail].mean(), "trials_to_80": ttc[a], "final_rt_ms": rt[a],
                                 "final_timeout": lg["timeout"][a, tail].mean(), "onset_gaze_AB_axis": onset[a],
                                 "final_aim_A": lg["aim_A"][a, tail].mean(), "final_aim_B": lg["aim_B"][a, tail].mean()})
    return rows


def plot_task_summary(rows, configs, out_dir, U, batch):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    cap = U * batch
    rng = np.random.default_rng(0)
    for ei, (label, members) in enumerate(EPISODES):
        for ci, config in enumerate(configs):
            sel = [r for r in rows if r["config"] == config and (r["scenario"], r["block"] - 1) in members]
            jit = ei + (ci - (len(configs) - 1) / 2) * 0.11 + rng.uniform(-0.03, 0.03, len(sel))
            acc = np.array([r["final_acc"] for r in sel])
            ttc = np.array([r["trials_to_80"] for r in sel])
            rt = np.array([r["final_rt_ms"] for r in sel])
            kw = dict(s=14, color=CONFIG_COLORS[ci], alpha=0.85, lw=0,
                      label=config_label(config) if ei == 0 else None)
            axes[0].scatter(jit, acc, **kw)
            axes[1].scatter(jit, np.where(np.isfinite(ttc), ttc, cap * 1.25) / 1000, **kw)
            axes[2].scatter(jit, rt, **kw)
    axes[0].set_ylabel("p(correct), last third of block")
    axes[0].axhline(0.5, color="0.4", lw=0.7)
    axes[1].set_ylabel(f"trials to {CRITERION:.0%} (thousands)")
    axes[1].axhline(cap / 1000, color="0.4", lw=0.7, ls=":")
    axes[1].text(len(EPISODES) - 0.5, cap * 1.25 / 1000, "never", color="0.7", fontsize=8, va="center", ha="right")
    axes[1].set_yscale("symlog", linthresh=1, linscale=0.5)
    axes[1].set_ylim(-0.1, cap * 1.6 / 1000)
    axes[2].set_ylabel("RT (ms), last third of block")
    for ax in axes:
        ax.set_xticks(range(len(EPISODES)))
        ax.set_xticklabels([e[0] for e in EPISODES], fontsize=8)
        for sep in (1.5, 3.5, 5.5):
            ax.axvline(sep, color="0.3", lw=0.6)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("learning episodes (rule x history): every dot is one agent", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "task_summary.png", dpi=150)
    plt.close(fig)


def draw_cells(ax, config):
    _, cells = spatial_config(config)
    for k in range(1, GRID):
        ax.axhline(-1 + 2 * k / GRID, color="0.3", lw=0.4)
        ax.axvline(-1 + 2 * k / GRID, color="0.3", lw=0.4)
    for loc, (row, col) in zip("AB", cells):
        x0, y0 = -1 + 2 * col / GRID, -1 + 2 * row / GRID
        ax.add_patch(plt.Rectangle((x0, y0), 2 / GRID, 2 / GRID, fill=False, ec=LOC_COLORS[loc], lw=1.5))
        ax.text(x0 + 0.03, y0 + 0.03, loc, color=LOC_COLORS[loc], fontsize=7, fontweight="bold", va="top")
    ax.add_patch(plt.Rectangle((-1 / GRID, -1 / GRID), 2 / GRID, 2 / GRID, fill=False, ec="0.8", lw=0.8, ls=":"))
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, -1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_gaze_heatmaps(runs, configs, out_dir, n_fix, last=20):
    scen_names = list(runs[configs[0]]["scenarios"])
    cols = [(s, b) for s in scen_names for b in range(3)]
    fig, axes = plt.subplots(len(configs), len(cols), figsize=(1.75 * len(cols), 1.85 * len(configs)), squeeze=False)
    for ci, config in enumerate(configs):
        for j, (scen, b) in enumerate(cols):
            tr = runs[config]["trials"][scen]
            idx = np.where(tr["block"] == b)[0][-last:]
            locs = tr["locs"][:, idx, :, n_fix - 1:].astype(float)             # stimulus phase (onset onward)
            alive = tr["loc_alive"][:, idx, :, n_fix - 1:].astype(float)
            alive[..., 0] = 1                                                   # onset gaze always counts
            ax = axes[ci, j]
            ax.hist2d(locs[..., 0].ravel(), locs[..., 1].ravel(), bins=GRID * 5, range=[[-1, 1], [-1, 1]],
                      weights=alive.ravel(), cmap="magma")
            draw_cells(ax, config)
            if ci == 0:
                ax.set_title(f"{scen}\nblock {b + 1}: {runs[config]['scenarios'][scen][b]}", fontsize=8)
            if j == 0:
                ax.set_ylabel(config_label(config).replace(": ", "\n"), fontsize=7)
    fig.suptitle(f"stimulus-phase fixations (onset until response), last {last} logged updates of each block "
                 f"(5 agents x {last * 32} trials); dotted = fixation cell", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "gaze_heatmaps.png", dpi=150)
    plt.close(fig)


def plot_gaze_timecourse(runs, configs, out_dir, n_fix, step_ms, last=20, max_step=8):
    """Aimed gaze along the A-B axis vs. time from stimulus onset, alive trials only."""
    scen_names = list(runs[configs[0]]["scenarios"])
    fig, axes = plt.subplots(len(scen_names), 3, figsize=(13, 2.9 * len(scen_names)), sharex=True, sharey=True,
                             squeeze=False)
    # mean_locs index i = aim made at step i, used at step i + 1; step n_fix = onset
    n_idx = n_fix - 1 + max_step
    t_ms = (np.arange(n_idx) + 1 - n_fix) * step_ms
    for si, scen in enumerate(scen_names):
        for b in range(3):
            ax = axes[si, b]
            for ci, config in enumerate(configs):
                run = runs[config]
                tr = run["trials"][scen]
                idx = np.where(tr["block"] == b)[0][-last:]
                aim = ab_axis(tr["mean_locs"][:, idx, :, :n_idx].astype(float), run["ab_locs"])    # (A, L, B, n)
                alive = tr["loc_alive"][:, idx, :, :n_idx].astype(bool)
                alive[..., :n_fix] = True                                   # the onset gaze is always used
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    per_agent = np.nanmean(np.where(alive, aim, np.nan).reshape(aim.shape[0], -1, n_idx), 1)
                    # drop steps where hardly any trials are still running
                    frac = alive.reshape(aim.shape[0], -1, n_idx).mean(1)
                per_agent[frac < 0.02] = np.nan
                band(ax, t_ms, per_agent, CONFIG_COLORS[ci], config_label(config) if si == b == 0 else None, lw=1.2)
            ax.axhline(0, color="0.5", lw=0.6)
            ax.axvline(0, color="0.7", lw=0.8, ls=":")
            ax.text(0.01, 0.97, "toward B", transform=ax.transAxes, fontsize=7, color=LOC_COLORS["B"], va="top")
            ax.text(0.01, 0.03, "toward A", transform=ax.transAxes, fontsize=7, color=LOC_COLORS["A"], va="bottom")
            if si == 0:
                ax.set_title(f"block {b + 1}", fontsize=9)
            ax.text(0.99, 0.97, f"{scen}: {run['scenarios'][scen][b]}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=ACCENT)
            if b == 0:
                ax.set_ylabel("aimed gaze, A-B axis\n(A = -1, fixation 0, B = +1)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("ms from stimulus onset (< 0: blank fixation period)")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="lower right")
    axes[0, 0].set_ylim(-0.6, 0.6)
    fig.suptitle(f"where the policy aims, step by step (last {last} logged updates of each block; still-running "
                 "trials only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "gaze_timecourse.png", dpi=150)
    plt.close(fig)


def plot_gaze_learning(runs, configs, out_dir, U, batch, window=5):
    scen_names = list(runs[configs[0]]["scenarios"])
    fig, axes = plt.subplots(len(scen_names), 1, figsize=(12, 2.6 * len(scen_names)), sharex=True, squeeze=False)
    for si, scen in enumerate(scen_names):
        ax = axes[si, 0]
        for ci, config in enumerate(configs):
            run = runs[config]
            tr = run["trials"][scen]
            onset = ab_axis(tr["onset_loc"].astype(float), run["ab_locs"]).mean(2)           # (A, L)
            L = onset.shape[1] // window
            v = onset[:, :L * window].reshape(onset.shape[0], L, window).mean(2)
            logged = tr["block"] * U + tr["update"]
            x = logged[:L * window].reshape(L, window).mean(1) * batch / 1000
            band(ax, x, v, CONFIG_COLORS[ci], config_label(config) if si == 0 else None, lw=1.2)
        ax.axhline(0, color="0.5", lw=0.6)
        block_lines(ax, 3, U * batch / 1000)
        rule_titles(ax, runs[configs[0]]["scenarios"][scen], None)
        ax.set_ylabel(f"{scen}\ngaze at onset, A-B axis", fontsize=8)
        ax.set_ylim(-0.45, 0.45)
        ax.text(0.005, 0.95, "toward B", transform=ax.transAxes, fontsize=7, color=LOC_COLORS["B"], va="top")
        ax.text(0.005, 0.05, "toward A", transform=ax.transAxes, fontsize=7, color=LOC_COLORS["A"], va="bottom")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    axes[-1, 0].set_xlabel("trials (thousands)")
    fig.suptitle("anticipatory gaze: where the eye is when the stimuli appear (A = -1, fixation 0, B = +1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "gaze_learning.png", dpi=150)
    plt.close(fig)


def plot_behavior_modes(runs, configs, design, out_dir, U, batch, window=5):
    scen_names = list(runs[configs[0]]["scenarios"])
    fig, axes = plt.subplots(len(configs), len(scen_names), figsize=(4.6 * len(scen_names), 1.95 * len(configs)),
                             sharex=True, sharey=True, squeeze=False)
    for ci, config in enumerate(configs):
        for si, scen in enumerate(scen_names):
            ax = axes[ci, si]
            tr = runs[config]["trials"][scen]
            modes, centers = choice_modes(tr, design, window)
            logged = tr["block"] * U + tr["update"]
            x = np.interp(centers, np.arange(len(logged)), logged) * batch / 1000
            for e, name in enumerate(design.mode_names):
                if name == "cst":
                    continue
                band(ax, x, modes[:, :, e], MODE_COLORS[name], name if ci == si == 0 else None, lw=1.2)
            block_lines(ax, 3, U * batch / 1000)
            ax.set_ylim(0, 1.02)
            if ci == 0:
                ax.set_title(scen, fontsize=10, pad=16)
                rule_titles(ax, runs[config]["scenarios"][scen], None)
            if si == 0:
                ax.set_ylabel(config_label(config).replace(": ", "\n"), fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="center right", title="mode", title_fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("trials (thousands)")
    fig.suptitle("choice function in kernel modes: |P_e y| / 2 with y(s) = 2 P(choice 1 | s) - 1 (1 = responses fully "
                 f"follow that mode; windows of {window * batch} trials per agent, sampling floor ~0.1)", fontsize=9,
                 y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "behavior_modes.png", dpi=150)
    plt.close(fig)


def plot_ntk_modes(shares, power, runs, configs, design, out_dir, k):
    scen_names = list(runs[configs[0]]["scenarios"])
    stim_modes = [e for e, n in enumerate(design.mode_names) if n != "cst"]
    fig, axes = plt.subplots(3, len(scen_names), figsize=(5.2 * len(scen_names), 11), squeeze=False)
    width = 0.8 / len(configs)
    for si, scen in enumerate(scen_names):
        rules = runs[configs[0]]["scenarios"][scen]
        states = ["initial"] + [f"after B{b + 1}\n({rules[b]})" for b in range(3)]
        ax = axes[0, si]
        for ci, config in enumerate(configs):
            sh = shares[ci, si]                                              # (states, agents, modes)
            stim = sh[..., stim_modes] / sh[..., stim_modes].sum(-1, keepdims=True)
            m = stim.mean(1)                                                 # (states, 3)
            x = np.arange(4) + (ci - (len(configs) - 1) / 2) * width
            bottom = np.zeros(4)
            for j, e in enumerate(stim_modes):
                ax.bar(x, m[:, j], width * 0.92, bottom=bottom, color=MODE_COLORS[design.mode_names[e]],
                       label=design.mode_names[e] if ci == 0 and si == 0 else None, ec=CONFIG_COLORS[ci], lw=0.8)
                bottom += m[:, j]
        ax.set_xticks(range(4))
        ax.set_xticklabels(states, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{scen}", fontsize=10)
        if si == 0:
            ax.set_ylabel("share of the stimulus-dependent NTK\n(A + B + AB = 1)")
        ax2 = axes[1, si]
        for ci, config in enumerate(configs):
            sh = shares[ci, si]
            ab_ratio = sh[..., 3] / ((sh[..., 1] + sh[..., 2]) / 2)
            x = np.arange(4) + (ci - (len(configs) - 1) / 2) * width
            ax2.scatter(np.repeat(x[:, None], sh.shape[1], 1), ab_ratio, s=10, color=CONFIG_COLORS[ci],
                        label=config_label(config) if si == 0 else None)
            ax2.plot(x, np.exp(np.log(ab_ratio).mean(1)), color=CONFIG_COLORS[ci], lw=1)
        ax2.set_yscale("log")
        ax2.axhline(1, color="0.5", lw=0.6)
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(states, fontsize=8)
        if si == 0:
            ax2.set_ylabel("AB / mean(A, B)\n(interaction vs main-effect learning rate)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left", title="mode", title_fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=7, loc="upper left")
    for si, scen in enumerate(scen_names):
        rules = runs[configs[0]]["scenarios"][scen]
        ax3 = axes[2, si]
        for ci, config in enumerate(configs):
            pw = power[ci, si]                                                # (states, agents, modes)
            sens = pw[..., 1:].sum(-1) / pw[..., 0]
            x = np.arange(4) + (ci - (len(configs) - 1) / 2) * width
            ax3.scatter(np.repeat(x[:, None], pw.shape[1], 1), sens, s=10, color=CONFIG_COLORS[ci])
        ax3.set_yscale("log")
        ax3.axhline(1, color="0.5", lw=0.6)
        ax3.set_xticks(range(4))
        ax3.set_xticklabels(["initial"] + [f"after B{b + 1}\n({rules[b]})" for b in range(3)], fontsize=8)
        if si == 0:
            ax3.set_ylabel("stimulus sensitivity\n(A + B + AB) / constant NTK power")
    fig.suptitle(f"empirical NTK of the choice readout ({k * 250} ms after onset) on held-out fractals, by mode; "
                 "bars grouped by config (edge color), dots = agents", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"ntk_modes_k{k}.png", dpi=150)
    plt.close(fig)


def plot_ntk_vs_learning(shares, rows, runs, configs, design, out_dir, k, U, batch):
    """Relevant-mode share of the stimulus-dependent NTK at the START of a block vs. that
    block's trials to criterion."""
    scen_names = list(runs[configs[0]]["scenarios"])
    stim_modes = [e for e, n in enumerate(design.mode_names) if n != "cst"]
    ep_of = {m: i for i, (_, members) in enumerate(EPISODES) for m in members}
    cap = U * batch
    fig, ax = plt.subplots(figsize=(8.5, 6))
    xs, ys = [], []
    ep_colors = plt.cm.Set2(np.linspace(0, 1, len(EPISODES)))
    per_ep = {}
    for r in rows:
        ci, si = configs.index(r["config"]), scen_names.index(r["scenario"])
        sh = shares[ci, si, r["block"] - 1, r["agent"]]                      # state before this block
        rel = design.mode_names.index(r["rule"])
        x = sh[rel] / sh[stim_modes].sum()
        y = r["trials_to_80"] if np.isfinite(r["trials_to_80"]) else cap * 1.25
        ei = ep_of[(r["scenario"], r["block"] - 1)]
        per_ep.setdefault(ei, []).append((x, y))
        xs.append(x)
        ys.append(y)
    for ei, pts in sorted(per_ep.items()):
        px, py = np.array(pts).T
        r_e, p_e = spearmanr(px, py)
        ax.scatter(px, py / 1000, s=16, color=ep_colors[ei], alpha=0.85, lw=0,
                   label=f"{EPISODES[ei][0].replace(chr(10), ' ')}: rho {r_e:+.2f} (p {p_e:.1g})")
    rho, p = spearmanr(xs, ys)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1, linscale=0.5)
    ax.set_ylim(-0.1, cap * 1.6 / 1000)
    ax.axhline(cap / 1000, color="0.4", lw=0.6, ls=":")
    ax.set_xlabel("NTK share of the block's rule mode at block start\n(among A, B, AB; held-out fractals)")
    ax.set_ylabel(f"trials to {CRITERION:.0%} (thousands; top row = never)")
    ax.set_title(f"does the kernel predict learning speed?  Spearman rho = {rho:.2f} (p = {p:.1g}, n = {len(xs)})",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower left", title="within episode", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"ntk_vs_learning_k{k}.png", dpi=150)
    plt.close(fig)
    return rho, p


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=int, nargs="+", default=[1, 4, 7, 10, 13, 16])
    parser.add_argument("--tag", default="_graded_scratch_fractals-color", help="run folder suffix after config<NN>")
    parser.add_argument("--k-stim", type=int, nargs="+", default=[1, 3], help="NTK readout steps after onset")
    parser.add_argument("--ntk-draws", type=int, default=3, help="held-out stimulus sets per NTK")
    parser.add_argument("--skip-ntk", action="store_true")
    parser.add_argument("--out", default=None, help="output folder (default: analysis_<configs><tag>)")
    args = parser.parse_args()
    apply_dark_theme()

    design = Design("2x2")
    runs = {c: load_config(c, args.tag) for c in args.configs}
    meta = runs[args.configs[0]]["meta"]
    U, batch = meta["updates_per_block"], meta["trials_per_update"]
    n_fix, step_ms = meta["cfg"]["n_fix"], meta["step_ms"]
    name = "configs" + "-".join(f"{c:02d}" for c in args.configs)
    out_dir = DATA_ROOT / "ram_model" / "2x2" / (args.out or f"analysis_{name}{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"analysing configs {args.configs} -> {out_dir}", flush=True)

    plot_performance(runs, args.configs, out_dir, U, batch)
    plot_performance_agents(runs, args.configs, out_dir, U, batch)
    rows = episode_table(runs, args.configs, U, batch)
    plot_task_summary(rows, args.configs, out_dir, U, batch)
    plot_gaze_heatmaps(runs, args.configs, out_dir, n_fix)
    plot_gaze_timecourse(runs, args.configs, out_dir, n_fix, step_ms)
    plot_gaze_learning(runs, args.configs, out_dir, U, batch)
    plot_behavior_modes(runs, args.configs, design, out_dir, U, batch)
    print("behavior and gaze figures done", flush=True)

    if not args.skip_ntk:
        pool = load_stimulus_set(meta["params"]["stimuli"], tile_size(meta["params"]["cell_px"]),
                                 color=meta["params"]["color"])
        shares, power = ntk_shares(runs, args.configs, design, pool, args.k_stim, args.ntk_draws, seed=1)
        scen_names = list(runs[args.configs[0]]["scenarios"])
        ntk_rows = []
        for k in args.k_stim:
            plot_ntk_modes(shares[k], power[k], runs, args.configs, design, out_dir, k)
            rho, p = plot_ntk_vs_learning(shares[k], rows, runs, args.configs, design, out_dir, k, U, batch)
            print(f"  k={k}: NTK rule-mode share vs trials to criterion, Spearman rho {rho:.2f} (p {p:.1g})")
            for ci, c in enumerate(args.configs):
                for si, scen in enumerate(scen_names):
                    for st in range(4):
                        for a in range(shares[k].shape[3]):
                            ntk_rows.append({"k_stim": k, "config": c, "scenario": scen,
                                             "state": "initial" if st == 0 else f"after_block{st}", "agent": a,
                                             **{f"share_{n}": shares[k][ci, si, st, a, e]
                                                for e, n in enumerate(design.mode_names)},
                                             **{f"power_{n}": power[k][ci, si, st, a, e]
                                                for e, n in enumerate(design.mode_names)}})
        with open(out_dir / "ntk_modes.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(ntk_rows[0]))
            wr.writeheader()
            wr.writerows(ntk_rows)
        np.savez(out_dir / "ntk_shares.npz", configs=args.configs, scenarios=scen_names, modes=design.mode_names,
                 **{f"k{k}": v for k, v in shares.items()}, **{f"power_k{k}": v for k, v in power.items()})

    with open(out_dir / "episodes.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved figures and tables to {out_dir}")


if __name__ == "__main__":
    main()
