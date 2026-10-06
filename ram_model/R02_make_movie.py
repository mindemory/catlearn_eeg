"""Movie of the RAM doing the 2x2 task: what it looks at and what it sees, before vs after training.

Trains one agent on one rule, then records trials with its initial (untrained) weights
and with its trained weights. Each trial is n_glimpses frames plus a feedback frame:

  left   the display, the current fixation (+), the path of earlier fixations, and the
         footprints of the sensor's crops around the fixation (fovea, 2x, 4x)
  right  exactly what the model receives on that glimpse, channel by channel
  bottom the model's choice probabilities and whether it was correct

By default the trained agent is shown with near-deterministic fixations (the policy's
mean, --test-sigma), to show where it aims; training used noisy fixations (--sigma).
Both accuracies are printed.

Outputs in ~/Documents/data/catlearn_eeg/ram_model/2x2/movies/, all named
ram_<rule>_config<NN>_<sensor>:
  .gif           the movie
  _training.png  learning curve of this agent: accuracy and where it aims (A vs B)
  _params.npz    the trained weights (reload with load_params), plus the run settings

  ~/miniforge3/envs/kernelbehav/bin/python R02_make_movie.py --rule A --sensor weak
"""

import argparse
import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.animation import FuncAnimation, PillowWriter

from env_2by2 import DATA_ROOT, GRID, Design, block_labels, render_block, spatial_config
from plot_style import ACCENT, FOREGROUND, apply_dark_theme
from R01_train_blocks import LOC_COLORS, SENSORS
from ram import episode, init_params, make_train_step

CHANNEL_NAMES = {16: "fovea (16 px, full res)", 32: "32 px crop, /2", 64: "64 px crop, /4"}


def save_params(path, params, **meta):
    """Save a params pytree as flat arrays keyed by their path, e.g. 'lstm/w'."""
    flat = {"/".join(str(getattr(k, "key", k)) for k in path_): np.asarray(v)
            for path_, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(path, meta=json.dumps(meta), **flat)


def load_params(path):
    """Inverse of save_params: returns (params, meta)."""
    z = np.load(path)
    params = {}
    for name in z.files:
        if name == "meta":
            continue
        layer, leaf = name.split("/")
        params.setdefault(layer, {})[leaf] = jnp.asarray(z[name])
    return params, json.loads(str(z["meta"]))


def record_trials(params, images, labels, key, n_trials, args, sigma):
    """Run n_trials single trials; returns a list of per-trial dicts of numpy arrays."""
    sensor = SENSORS[args.sensor]
    noise = args.periph_noise if args.sensor == "noisy" else 0.0
    run = jax.jit(lambda im, lab, k: episode(params, im, lab, k, args.glimpses, True, sigma,
                                             sensor["scales"], sensor["full_view"], noise)[1])
    trials = []
    for i in range(n_trials):
        key, k_s, k_ep = jax.random.split(key, 3)
        s = int(jax.random.randint(k_s, (), 0, 4))
        m = run(images[s], labels[s], k_ep)
        trials.append({"s": s, "label": int(labels[s]), "img": np.asarray(images[s]),
                       **{k: np.asarray(m[k]) for k in ("glimpses", "glimpse_locs", "probs", "choice", "reward")}})
    return trials


def accuracy(params, images, labels, key, args, sigma, n=512):
    sensor = SENSORS[args.sensor]
    noise = args.periph_noise if args.sensor == "noisy" else 0.0
    k_s, k_ep = jax.random.split(key)
    s = jax.random.randint(k_s, (n,), 0, 4)
    run = jax.vmap(lambda im, lab, k: episode(params, im, lab, k, args.glimpses, True, sigma,
                                              sensor["scales"], sensor["full_view"], noise)[1]["p_correct"])
    return float(run(images[s], labels[s], jax.random.split(k_ep, n)).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rule", default="A", help="A, B or AB")
    parser.add_argument("--config", type=int, default=2)
    parser.add_argument("--sensor", choices=list(SENSORS), default="weak")
    parser.add_argument("--periph-noise", type=float, default=0.3)
    parser.add_argument("--patches", type=int, nargs=4, default=[3, 7, 11, 19], help="pool indices a1 a2 b1 b2")
    parser.add_argument("--updates", type=int, default=4000)
    parser.add_argument("--glimpses", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.3, help="fixation noise during training")
    parser.add_argument("--test-sigma", type=float, default=0.02, help="fixation noise in the trained-agent movie")
    parser.add_argument("--trials-before", type=int, default=3)
    parser.add_argument("--trials-after", type=int, default=6)
    parser.add_argument("--fps", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    apply_dark_theme()

    design = Design("2x2")
    sensor = SENSORS[args.sensor]
    imgs, ab_locs = render_block(args.config, args.patches)
    images, labels = jnp.asarray(imgs), jnp.asarray(block_labels(design, args.rule))

    # Train one agent
    optimizer = optax.adam(1e-3)
    params0 = init_params(jax.random.PRNGKey(args.seed), n_channels=len(sensor["scales"]) + sensor["full_view"])
    params, opt_state = params0, optimizer.init(params0)
    step = jax.jit(make_train_step(optimizer, 64, args.glimpses, True, args.sigma, scales=sensor["scales"],
                                   full_view=sensor["full_view"],
                                   periph_noise=args.periph_noise if args.sensor == "noisy" else 0.0))
    key = jax.random.PRNGKey(args.seed + 1)
    curve = {"update": [], "p_correct": [], "aim_A": [], "aim_B": []}
    for u in range(args.updates):
        key, k = jax.random.split(key)
        params, opt_state, m = step(params, opt_state, k, images, labels, jnp.asarray(ab_locs))
        if u % 10 == 0:
            curve["update"].append(u)
            for name in ("p_correct", "aim_A", "aim_B"):
                curve[name].append(float(m[name]))

    key, k1, k2, k3, k4, k5 = jax.random.split(key, 6)
    acc_before = accuracy(params0, images, labels, k1, args, args.sigma)
    acc_train = accuracy(params, images, labels, k2, args, args.sigma)
    acc_test = accuracy(params, images, labels, k3, args, args.test_sigma)
    print(f"p(correct): untrained {acc_before:.2f} | trained, noisy fixations (sigma {args.sigma}) {acc_train:.2f} | "
          f"trained, policy-mean fixations (sigma {args.test_sigma}) {acc_test:.2f}")

    out_dir = DATA_ROOT / "ram_model" / "2x2" / "movies"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ram_{args.rule}_config{args.config:02d}_{args.sensor}"
    save_params(out_dir / f"{stem}_params.npz", params, **vars(args),
                acc_untrained=acc_before, acc_trained_noisy=acc_train, acc_trained_policy_mean=acc_test)

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    w = 10
    sm = lambda v: np.convolve(np.pad(v, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")
    upd = np.array(curve["update"])
    axes[0].plot(upd, sm(np.array(curve["p_correct"])), color=FOREGROUND)
    axes[0].axhline(0.5, color="0.4", lw=0.8)
    axes[0].set_ylim(0.4, 1.02)
    axes[0].set_ylabel("p(correct)")
    for loc_name in ("A", "B"):
        axes[1].plot(upd, sm(np.array(curve[f"aim_{loc_name}"])), color=LOC_COLORS[loc_name], label=f"aims at {loc_name}")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("fraction of chosen\nfixations (policy mean)")
    axes[1].set_xlabel("training updates (64 trials each)")
    axes[1].legend(frameon=False, fontsize=9)
    config_title, _ = spatial_config(args.config)
    fig.suptitle(f"RAM training, rule {args.rule} | config {args.config} ({config_title}) | sensor '{args.sensor}'", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_training.png", dpi=150)
    plt.close(fig)

    segments = [("before training", record_trials(params0, images, labels, k4, args.trials_before, args, args.sigma)),
                (f"after {args.updates} training updates", record_trials(params, images, labels, k5, args.trials_after, args,
                                                                          args.test_sigma))]
    frames = [(phase, trial, t) for phase, trials in segments for trial in trials for t in range(args.glimpses + 1)]

    # ---- figure
    n_ch = len(sensor["scales"]) + sensor["full_view"]
    ch_names = [CHANNEL_NAMES.get(s, f"{s} px") for s in sensor["scales"]] + (["whole display, /7"] if sensor["full_view"] else [])
    fig = plt.figure(figsize=(4.2 + 2.1 * n_ch, 5.2))
    gs = fig.add_gridspec(2, 1 + n_ch, width_ratios=[2.2] + [1] * n_ch, height_ratios=[1, 0.18])
    ax_disp = fig.add_subplot(gs[0, 0])
    ax_ch = [fig.add_subplot(gs[0, 1 + c]) for c in range(n_ch)]
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis("off")
    H = imgs.shape[-1]
    _, cells = spatial_config(args.config)
    config_name, _ = spatial_config(args.config)
    cat_names = {0: "category 0", 1: "category 1"}

    def to_px(loc):
        return (loc[0] + 1) / 2 * H - 0.5, (loc[1] + 1) / 2 * H - 0.5

    def draw(frame):
        phase, trial, t = frame
        glimpse_t = min(t, args.glimpses - 1)
        ax_disp.clear()
        ax_disp.imshow(trial["img"], cmap="gray", vmin=-0.5, vmax=0.5)
        for k in range(1, GRID):
            ax_disp.axhline(k * H / GRID - 0.5, color="0.3", lw=0.4)
            ax_disp.axvline(k * H / GRID - 0.5, color="0.3", lw=0.4)
        for loc_name, (row, col) in zip("AB", cells):
            cp = H / GRID
            ax_disp.add_patch(plt.Rectangle((col * cp - 0.5, row * cp - 0.5), cp, cp, fill=False,
                                            ec=LOC_COLORS[loc_name], lw=1.2, ls=":"))
            ax_disp.text(col * cp + 1, row * cp + 3, loc_name, color=LOC_COLORS[loc_name], fontsize=8, fontweight="bold")
        path = np.array([to_px(l) for l in trial["glimpse_locs"][:glimpse_t + 1]])
        ax_disp.plot(path[:, 0], path[:, 1], "-o", color=ACCENT, ms=3, lw=1, alpha=0.8)
        fx, fy = path[-1]
        for s, ls in zip(sensor["scales"], ["-", "--", ":"]):
            ax_disp.add_patch(plt.Rectangle((fx - s / 2, fy - s / 2), s, s, fill=False, ec="white", lw=1, ls=ls))
        ax_disp.plot(fx, fy, "+", color="red", ms=12, mew=2)
        ax_disp.set_xlim(-0.5, H - 0.5)
        ax_disp.set_ylim(H - 0.5, -0.5)
        ax_disp.set_xticks([])
        ax_disp.set_yticks([])
        la, lb = divmod(trial["s"], 2)
        ax_disp.set_title(f"{phase}\nglimpse {glimpse_t + 1}/{args.glimpses}  |  shown: a{la + 1} at A, b{lb + 1} at B",
                          fontsize=9)

        g = trial["glimpses"][glimpse_t]
        for c, ax in enumerate(ax_ch):
            ax.clear()
            ax.imshow(g[c], cmap="gray", vmin=-0.5, vmax=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(ch_names[c], fontsize=8)
        ax_ch[0].set_ylabel("what the model sees", fontsize=8)

        ax_txt.clear()
        ax_txt.axis("off")
        if t == args.glimpses:
            p = trial["probs"]
            ok = bool(trial["reward"])
            ax_txt.text(0.5, 0.5, f"model: p({cat_names[1]}) = {p[1]:.2f}  ->  chose {cat_names[int(trial['choice'])]}"
                        f"   |   correct answer: {cat_names[trial['label']]}   ->   {'CORRECT' if ok else 'INCORRECT'}",
                        ha="center", va="center", fontsize=11, color="#4cd964" if ok else "#ff5a5f",
                        transform=ax_txt.transAxes)
        else:
            ax_txt.text(0.5, 0.5, "sampling ...", ha="center", va="center", fontsize=10, color=FOREGROUND,
                        transform=ax_txt.transAxes)
        fig.suptitle(f"RAM on 2x2, rule {args.rule} | config {args.config} ({config_name}) | sensor '{args.sensor}'",
                     fontsize=10)
        return []

    out = out_dir / f"{stem}.gif"
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=90)
    plt.close(fig)
    print(f"Saved {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
