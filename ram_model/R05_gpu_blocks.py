"""Timed RAM on the GPU (MLX): 3-block design for many conditions in one batch, from scratch.

Every condition = (spatial config, scenario); each gets --agents independent networks,
and ALL conditions train side by side in one GPU batch (ram_mlx.py) -- the GPU's cost per
trial falls as the batch grows, so adding conditions is cheap. Agents start from random
weights (no pretraining) and learn everything within the experiment: seeing, looking,
when to respond, the rule. Agent i starts from the same random weights in every
condition, so conditions can be compared agent by agent.

Trial structure and learning: ram_timed.py (1 s fixation, up to 4 s response window at
250 ms per step, correct/incorrect feedback, saccade and time costs). Blocks: new stimuli
every block (4 per block from the pool); rule the same for blocks 1-2, different in
block 3 (SCENARIOS in R03_pretrain_blocks.py). Each block is --updates weight updates of
--batch trials. Stimulus pools: --stimuli noise (20 grayscale patches), aliens (19) or
fractals (72); --color gives color stimuli and a color sensor (every crop in R, G, B).

Everything needed for analysis is saved per config (see DATA_FORMAT.md written into each
output folder): per-update summaries for every update, every trial of every
--trial-log-every-th update (stimulus, answer, response, RT, gaze path), weights at the
end of each block, and figures.

Output folder per config:
  ~/Documents/data/catlearn_eeg/ram_model/2x2/timed_config<NN>_<sensor>_scratch_<stimuli>[-color]<tag>/

  ~/miniforge3/envs/kernelbehav/bin/python R05_gpu_blocks.py --configs 1 2 3 --stimuli fractals --color
"""

import argparse
import itertools
import json
import platform
import time
from datetime import datetime
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

import ram_mlx as R
from env_2by2 import DATA_ROOT, GRID, Design, block_labels, load_stimulus_set, render_tiles, spatial_config, tile_size
from plot_style import apply_dark_theme
from R01_train_blocks import SENSORS
from R03_pretrain_blocks import SCENARIOS
from R04_timed_blocks import METRICS, plot_scenario, plot_summary, thirds
from ram_timed import DEFAULTS, STEP_MS

TRIAL_FIELDS = ("stim", "label", "choice", "correct", "timeout", "rt_steps", "locs", "mean_locs", "loc_alive", "onset_loc")


def data_format_md(args, cfg, n_blocks, U):
    T = cfg["n_fix"] + cfg["n_stim"]
    n_log = len(range(0, U, args.trial_log_every))
    return f"""# Data format -- timed RAM, 3-block design (written by ram_model/R05_gpu_blocks.py)

One folder per spatial config. Inside, one set of files per scenario (`<scen>` in
{list(SCENARIOS)}: rules per block {SCENARIOS}).
Agents: {args.agents} per scenario; agent i starts from the same random weights in every
scenario and config (seed {args.seed}). All arrays have the agent as their first axis.

Design: {n_blocks} blocks x {U} weight updates x {args.batch} trials per update
({U * args.batch} trials per block). New stimuli every block. Stimuli: {args.stimuli}
({'color' if args.color else 'grayscale'}), sensor '{args.sensor}'. Trial timing:
{cfg['n_fix']} fixation steps + up to {cfg['n_stim']} stimulus steps, {STEP_MS} ms per step.
Saccade cost {cfg['saccade_cost']}, time cost {cfg['time_cost']}, fixation noise SD {cfg['sigma']}.

Compound / stimulus index `s` (0-3) = 2 * levelA + levelB: location A shows patch a1
(levelA 0) or a2 (levelA 1), location B shows b1 or b2. Category labels are 0/1.
Gaze coordinates are in [-1, 1] display units (x right, y DOWN; the display is 2 wide,
7 x 7 grid cells of 2/7 each; fixation cell = center, (0, 0)).

## experiment_logs.npz -- per-update summaries, EVERY update
- `<scen>/<metric>`: (agents, {n_blocks * U}) one value per update (mean over its
  {args.batch} trials); column u + b * {U} = update u of block b.
  metrics: acc (expected accuracy; timeouts = errors), correct (sampled response
  correct), timeout (rate), rt_ms (mean RT of responded trials, NaN if none),
  travel (fixation travel per trial), aim_A / aim_B (fraction of stimulus-phase
  fixations whose policy MEAN is within one cell of location A / B), onset_A / onset_B
  (fraction of trials with gaze already at A / B at stimulus onset).
- `<scen>/patches`: (agents, {4 * n_blocks}) stimulus numbers used (1-based file numbers of
  the {args.stimuli} set), 4 per block in order [a1, a2, b1, b2].
- `<scen>/labels`: ({n_blocks}, 4) category of each compound s in each block.
- `ab_locs`: (2, 2) gaze coordinates of the centers of location A and location B.
- `params`, `scenarios`, `cfg`: JSON strings with every setting.

## trials_<scen>.npz -- trial-level data, every trial of every {args.trial_log_every}th update
- `update`: ({n_blocks * n_log},) update index within the block of each logged update;
  `block`: same length, block index (0-based). Logged updates per block: {n_log}.
- per trial, shape (agents, {n_blocks * n_log}, {args.batch}, ...):
  - `stim` (int8, compound 0-3), `label` (int8), `choice` (int8: response 0/1, -1 =
    timeout), `correct` (int8), `timeout` (int8), `rt_steps` (int8: steps from onset to
    the response, 1 = first stimulus step; {cfg['n_stim']} on timeouts; RT ms = rt_steps * {STEP_MS})
  - `locs` (float16, ..., {T - 1}, 2): fixation actually taken at steps 1..{T - 1} (step 0
    is the central fixation). Steps 1..{cfg['n_fix'] - 1} fall in the fixation period, steps
    {cfg['n_fix']}..{T - 1} in the stimulus period (step {cfg['n_fix']} = stimulus onset).
  - `mean_locs` (float16, same shape): where the policy AIMED at each step (before noise).
  - `loc_alive` (uint8, ..., {T - 1}): 1 while the trial was still running at that step
    (0 after the response: those locs are frozen copies, ignore them).
  - `onset_loc` (float16, ..., 2): gaze position at stimulus onset.

## weights_<scen>.npz -- network weights at the end of each block
- `block<b>/<layer>/<w|b>`: (agents, ...) in ram_mlx layout (convs (O, kh, kw, I)).
  Initial weights: ram_mlx.init_params(agents, n_channels, seed={args.seed}).

## Figures
- `blocks_<scen>.png`: accuracy, RT and timeouts, gaze aims and onset gaze over trials,
  stimulus-phase fixation heatmaps at the end of each block.
- `summary_blocks.png`: accuracy and RT by thirds of each block, all scenarios.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=int, nargs="+", default=[1], help="spatial configs 1-18")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--agents", type=int, default=5, help="agents per condition")
    parser.add_argument("--sensor", choices=list(SENSORS), default="graded")
    parser.add_argument("--stimuli", choices=["noise", "aliens", "fractals"], default="noise")
    parser.add_argument("--color", action="store_true", help="color stimuli and a color sensor (aliens, fractals)")
    parser.add_argument("--periph-noise", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=DEFAULTS["sigma"])
    parser.add_argument("--saccade-cost", type=float, default=DEFAULTS["saccade_cost"])
    parser.add_argument("--time-cost", type=float, default=DEFAULTS["time_cost"])
    parser.add_argument("--n-fix", type=int, default=DEFAULTS["n_fix"])
    parser.add_argument("--n-stim", type=int, default=DEFAULTS["n_stim"])
    parser.add_argument("--updates", type=int, default=1000, help="weight updates per block")
    parser.add_argument("--batch", type=int, default=32, help="trials per update")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cell-px", type=int, default=16)
    parser.add_argument("--trial-log-every", type=int, default=10, help="save every trial of every k-th update")
    parser.add_argument("--heatmap-trials", type=int, default=100)
    parser.add_argument("--smooth", type=int, default=50, help="moving average over updates, for plots")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-tag", default="")
    args = parser.parse_args()
    apply_dark_theme()
    t_start = time.time()

    design = Design("2x2")
    s = SENSORS[args.sensor]
    cfg = {**DEFAULTS, "scales": s["scales"], "full_view": s["full_view"],
           "periph_noise": args.periph_noise if args.sensor == "noisy" else 0.0,
           "sigma": args.sigma, "saccade_cost": args.saccade_cost, "time_cost": args.time_cost,
           "n_fix": args.n_fix, "n_stim": args.n_stim}
    if args.color and args.stimuli == "noise":
        parser.error("--color needs --stimuli aliens or fractals (the noise patches are grayscale)")
    n_color = 3 if args.color else 1
    n_ch = R.n_sensor_channels(cfg, n_color)
    stim_tag = f"_{args.stimuli}{'-color' if args.color else ''}"
    T = cfg["n_fix"] + cfg["n_stim"]

    # ---- conditions and agents
    conditions = list(itertools.product(args.configs, args.scenarios))
    n_blocks = len(SCENARIOS[args.scenarios[0]])
    K = args.agents
    A = K * len(conditions)
    agent_cond = np.repeat(np.arange(len(conditions)), K)
    print(f"[{datetime.now():%H:%M:%S}] stimuli: {args.stimuli}{' (color)' if args.color else ''}, {n_ch} sensor channels; "
          f"configs {args.configs}; {len(conditions)} conditions x {K} agents = {A} networks in one GPU batch; "
          f"{n_blocks} blocks x {args.updates} updates x {args.batch} trials", flush=True)

    rng = np.random.default_rng(args.seed)
    pool = load_stimulus_set(args.stimuli, tile_size(args.cell_px), color=args.color)
    patches = np.stack([rng.choice(len(pool), 4 * n_blocks, replace=False) for _ in range(A)])   # 0-based
    images, labels, ab_locs = [], [], []
    for a in range(A):
        config, scen = conditions[agent_cond[a]]
        per_block = [render_tiles(config, pool[patches[a, 4 * b:4 * b + 4]], args.cell_px) for b in range(n_blocks)]
        images.append([pb[0] for pb in per_block])
        ab_locs.append(per_block[0][1])
        labels.append([block_labels(design, t) for t in SCENARIOS[scen]])
    images = np.array(images, dtype=np.float32)                       # (A, blocks, 4, H, H[, 3])
    labels = np.array(labels, dtype=np.int32)                         # (A, blocks, 4)
    ab_locs = np.array(ab_locs)                                       # (A, 2, 2)

    base = R.init_params(K, n_channels=n_ch, seed=args.seed)           # same start for agent i everywhere
    params = R.select_agents(base, mx.array(np.tile(np.arange(K), len(conditions))))
    trainer = R.Trainer(params, cfg, lr=args.lr)

    # ---- storage
    U = args.updates
    logged = list(range(0, U, args.trial_log_every))
    n_log = len(logged)
    logs = {m: np.zeros((A, n_blocks * U)) for m in METRICS}
    trials = {
        "stim": np.zeros((A, n_blocks * n_log, args.batch), np.int8),
        "label": np.zeros((A, n_blocks * n_log, args.batch), np.int8),
        "choice": np.zeros((A, n_blocks * n_log, args.batch), np.int8),
        "correct": np.zeros((A, n_blocks * n_log, args.batch), np.int8),
        "timeout": np.zeros((A, n_blocks * n_log, args.batch), np.int8),
        "rt_steps": np.zeros((A, n_blocks * n_log, args.batch), np.int8),
        "locs": np.zeros((A, n_blocks * n_log, args.batch, T - 1, 2), np.float16),
        "mean_locs": np.zeros((A, n_blocks * n_log, args.batch, T - 1, 2), np.float16),
        "loc_alive": np.zeros((A, n_blocks * n_log, args.batch, T - 1), np.uint8),
        "onset_loc": np.zeros((A, n_blocks * n_log, args.batch, 2), np.float16),
    }
    log_update = np.array(logged * n_blocks)
    log_block = np.repeat(np.arange(n_blocks), n_log)
    block_weights = []
    tail_updates = max(1, args.heatmap_trials // args.batch)
    end_locs = []

    # ---- train all conditions together
    t0 = time.time()
    for b in range(n_blocks):
        imgs_b = mx.array(images[:, b])
        labs_b = mx.array(labels[:, b])
        sensor_b = R.prepare_sensor(images[:, b], cfg["scales"])
        tail_locs, tail_mask = [], []
        li = b * n_log
        for u in range(U):
            res = trainer.step(imgs_b, labs_b, args.batch, rng, sensor=sensor_b)
            summ = R.summarize(res, ab_locs)
            for m in METRICS:
                logs[m][:, b * U + u] = summ[m]
            if u % args.trial_log_every == 0:
                trials["stim"][:, li] = res["img_idx"]
                trials["label"][:, li] = res["label"]
                trials["choice"][:, li] = res["choice"]
                trials["correct"][:, li] = res["correct"]
                trials["timeout"][:, li] = res["timeout"]
                trials["rt_steps"][:, li] = res["rt_steps"]
                trials["locs"][:, li] = res["locs"]
                trials["mean_locs"][:, li] = res["mean_locs"]
                trials["loc_alive"][:, li] = res["loc_alive"]
                trials["onset_loc"][:, li] = res["onset_loc"]
                li += 1
            if u >= U - tail_updates:
                tail_locs.append(res["locs"])
                tail_mask.append(res["loc_mask"])
            if (u + 1) % max(1, U // 10) == 0:
                el = time.time() - t0
                done = b * U + u + 1
                print(f"[{datetime.now():%H:%M:%S}] block {b + 1}/{n_blocks}, update {u + 1}/{U} ({el / 60:.1f} min, "
                      f"~{el / done * (n_blocks * U - done) / 60:.0f} min left): mean acc {np.mean(summ['acc']):.2f}, "
                      f"RT {np.nanmean(summ['rt_ms']):.0f} ms", flush=True)
        end_locs.append((np.concatenate(tail_locs, axis=1), np.concatenate(tail_mask, axis=1)))
        block_weights.append(R.to_numpy(trainer.params))
    train_min = (time.time() - t0) / 60
    print(f"[{datetime.now():%H:%M:%S}] training done in {train_min:.1f} min", flush=True)

    # ---- save per config
    meta = {"params": vars(args), "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "scenarios": SCENARIOS, "configs": args.configs, "n_blocks": n_blocks, "updates_per_block": U,
            "trials_per_update": args.batch, "step_ms": STEP_MS, "train_minutes": round(train_min, 2),
            "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "mlx": mx.__version__, "numpy": np.__version__,
            "python": platform.python_version()}
    for config in args.configs:
        out_dir = DATA_ROOT / "ram_model" / "2x2" / f"timed_config{config:02d}_{args.sensor}_scratch{stim_tag}{args.out_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "DATA_FORMAT.md").write_text(data_format_md(args, cfg, n_blocks, U))
        (out_dir / "meta.json").write_text(json.dumps({**meta, "config": config, "config_name": spatial_config(config)[0]}, indent=2))
        plot_args = SimpleNamespace(trials=U * args.batch, test_batch=args.batch, smooth=args.smooth, config=config,
                                    sensor=args.sensor, n_stim=args.n_stim, saccade_cost=args.saccade_cost,
                                    time_cost=args.time_cost, agents=K, heatmap_trials=args.heatmap_trials,
                                    no_pretrain=True)
        cfg_logs, saved = {}, {}
        for c_i, (c_config, scen) in enumerate(conditions):
            if c_config != config:
                continue
            idx = np.where(agent_cond == c_i)[0]
            sub = {m: logs[m][idx] for m in METRICS}
            cfg_logs[scen] = sub
            for m, arr in sub.items():
                saved[f"{scen}/{m}"] = arr
            saved[f"{scen}/patches"] = patches[idx] + 1
            saved[f"{scen}/labels"] = labels[idx[0]]
            np.savez(out_dir / f"trials_{scen}.npz", update=log_update, block=log_block,
                     **{k: v[idx] for k, v in trials.items()})
            np.savez(out_dir / f"weights_{scen}.npz",
                     **{f"block{b}/{layer}/{leaf}": v[idx] for b, bw in enumerate(block_weights)
                        for layer, d in bw.items() for leaf, v in d.items()})
            plot_scenario(design, scen, SCENARIOS[scen], sub, [(l[idx], mk[idx]) for l, mk in end_locs], plot_args, out_dir)
            acc = thirds(sub["acc"], n_blocks, U).mean(0)
            rt = np.nanmean(thirds(sub["rt_ms"], n_blocks, U), 0)
            print(f"config {config:2d} {scen:8s} acc first->last third: "
                  + "  ".join(f"B{b + 1} {acc[b, 0]:.2f}->{acc[b, 2]:.2f}" for b in range(n_blocks))
                  + " | RT ms: " + "  ".join(f"B{b + 1} {rt[b, 0]:.0f}->{rt[b, 2]:.0f}" for b in range(n_blocks)), flush=True)
        plot_summary(cfg_logs, plot_args, out_dir)
        np.savez(out_dir / "experiment_logs.npz", params=json.dumps(vars(args)), scenarios=json.dumps(SCENARIOS),
                 cfg=json.dumps(meta["cfg"]), ab_locs=ab_locs[np.where(agent_cond == conditions.index((config, args.scenarios[0])))[0][0]],
                 **saved)
        print(f"[{datetime.now():%H:%M:%S}] saved config {config} to {out_dir}", flush=True)
    print(f"[{datetime.now():%H:%M:%S}] all done in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
