"""Empirical NTK of the timed RAM over the 4 compounds, decomposed into kernel modes.

Puts the RAM in the same currency as the kernel model (kernel_model/): the network's
choice readout f(stimulus) changes under gradient descent according to its neural
tangent kernel K[s, s'] = <df(s)/dtheta, df(s')/dtheta>. Decomposing K into the
const / A / B / AB modes (kernel_modes.Design) gives the network's learning bias: a mode
with more weight is learned faster. If the AB (XOR) weight is tiny, XOR is learned
extremely slowly -- whatever the stimuli look like.

Readout f: the choice logit difference (category 1 - category 0) after the 1 s fixation
and --k-stim stimulus steps, with gaze following the policy's mean (no noise, no early
response), so f is deterministic. Kernels are measured on 4 patches the agent has NOT
seen (held out from its experiment), since each block brings new patches: this is the
kernel that governs learning the next block.

Agents: the random initial weights (as used by R05, --seed) and the final weights saved
by R05 for each scenario.

  ~/miniforge3/envs/kernelbehav/bin/python R06_ntk_modes.py --config 1
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

import ram_mlx as R
from env_2by2 import DATA_ROOT, N_ALIENS, N_POOL, Design, glimpse, load_alien, load_patch, render_tiles, tile_size
from R01_train_blocks import SENSORS
from ram_timed import DEFAULTS, _encode, _lstm


def mlx_agent_to_jax(p, a):
    """One agent's weights from ram_mlx layout (leading agent axis, NHWC) to ram_timed layout."""
    what = p["what"]["w"][a].reshape(8, 8, 32, -1).transpose(2, 0, 1, 3).reshape(2048, -1)   # NHWC -> NCHW flatten
    out = {"conv1": {"w": p["conv1"]["w"][a].transpose(0, 3, 1, 2), "b": p["conv1"]["b"][a]},
           "conv2": {"w": p["conv2"]["w"][a].transpose(0, 3, 1, 2), "b": p["conv2"]["b"][a]},
           "what": {"w": what, "b": p["what"]["b"][a]}}
    for k in ("where", "clock", "lstm", "loc", "choice", "baseline", "stop"):
        out[k] = {"w": p[k]["w"][a], "b": p[k]["b"][a]}
    return jax.tree_util.tree_map(jnp.asarray, out)


def readout(params, img, cfg, k_stim):
    """Deterministic choice readout: logit(cat 1) - logit(cat 0) after fixation + k_stim steps."""
    n_fix, n_stim = cfg["n_fix"], cfg["n_stim"]
    h = c = jnp.zeros(params["lstm"]["w"].shape[1] // 4)
    loc = jnp.zeros(2)
    blank = jnp.zeros_like(img)
    for t in range(n_fix + k_stim):
        on = t >= n_fix
        g = glimpse(img if on else blank, loc, scales=cfg["scales"], full_view=cfg["full_view"])
        clock = jnp.array([1.0 if on else 0.0, (t - n_fix) / n_stim if on else t / n_fix])
        h, c = _lstm(params, (h, c), _encode(params, g, loc, clock))
        loc = jax.lax.stop_gradient(jnp.tanh(h @ params["loc"]["w"] + params["loc"]["b"]))
    logits = h @ params["choice"]["w"] + params["choice"]["b"]
    return logits[1] - logits[0]


def ntk(params, images, cfg, k_stim):
    """4 x 4 empirical NTK of the readout over the 4 compound displays."""
    jac = jax.jacrev(lambda p: jnp.stack([readout(p, im, cfg, k_stim) for im in images]))(params)
    J = jnp.concatenate([j.reshape(len(images), -1) for j in jax.tree_util.tree_leaves(jac)], axis=1)
    return np.asarray(J @ J.T)


def mode_shares(K, design):
    """Share of the kernel's trace in each mode: tr(P_e K) / tr(K)."""
    tr = np.trace(K)
    return np.array([np.trace(P @ K) for P in design.projectors]) / tr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=int, default=1)
    parser.add_argument("--sensor", default="graded")
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="R05's seed (initial weights)")
    parser.add_argument("--k-stim", type=int, nargs="+", default=[1, 3], help="stimulus steps before the readout")
    parser.add_argument("--out-tag", default="")
    parser.add_argument("--init-only", action="store_true", help="only probe the initial weights (no run needed)")
    parser.add_argument("--stimuli", choices=["noise", "aliens"], default="noise", help="which set to probe with")
    parser.add_argument("--contrast", type=float, default=1.0, help="scale stimulus contrast about the grey background")
    args = parser.parse_args()

    design = Design("2x2")
    s = SENSORS[args.sensor]
    cfg = {**DEFAULTS, "scales": s["scales"], "full_view": s["full_view"]}
    run_dir = DATA_ROOT / "ram_model" / "2x2" / f"timed_config{args.config:02d}_{args.sensor}_scratch{args.out_tag}"
    size = tile_size()
    if args.stimuli == "aliens":
        real = np.stack([load_alien(k, size) for k in range(1, N_ALIENS + 1)])
    else:
        real = np.stack([load_patch(k, size) for k in range(1, N_POOL + 1)])
    real = real * args.contrast
    if args.init_only:
        logs, scenarios = None, {}
    else:
        logs = np.load(run_dir / "experiment_logs.npz")
        scenarios = json.loads(str(logs["scenarios"]))

    # Which weights to probe: initial, and final per scenario
    n_ch = len(s["scales"]) + int(s["full_view"])
    sets = {"initial (random)": (R.to_numpy(R.init_params(args.agents, n_channels=n_ch, seed=args.seed)), None)}
    for scen in scenarios:
        f = run_dir / f"final_agents_{scen}.npz"
        if f.exists():
            z = np.load(f)
            p = {}
            for name in z.files:
                layer, leaf = name.split("/")
                p.setdefault(layer, {})[leaf] = z[name]
            sets[f"after {scen}"] = (p, logs[f"{scen}/patches"])

    print(f"mode shares of the RAM's empirical NTK (config {args.config}, {args.agents} agents, held-out {args.stimuli}, "
          f"contrast x{args.contrast:g}); modes {design.mode_names}; stimulus RMS {np.sqrt((real ** 2).mean()):.3f}")
    rng = np.random.default_rng(1)
    for k in args.k_stim:
        print(f"\nreadout after {k} stimulus step(s) ({250 * k} ms):")
        for label, (p, used) in sets.items():
            shares = []
            for a in range(args.agents):
                seen = set() if used is None else set(np.asarray(used[a]).ravel() - 1)
                pool = [i for i in range(len(real)) if i not in seen]
                held = rng.choice(pool, 4, replace=False)
                images, _ = render_tiles(args.config, real[held])
                shares.append(mode_shares(ntk(mlx_agent_to_jax(p, a), jnp.asarray(images), cfg, k), design))
            shares = np.array(shares)
            m, sem = shares.mean(0), shares.std(0) / np.sqrt(len(shares))
            print(f"  {label:22s} " + "  ".join(f"{n} {mm:.4f}+-{ss:.4f}" for n, mm, ss in zip(design.mode_names, m, sem))
                  + f"  | stimulus-dependent (A+B+AB) {m[1:].sum():.2e} | AB / mean(A,B) = {m[3] / max(1e-12, (m[1] + m[2]) / 2):.4f}")


if __name__ == "__main__":
    main()
