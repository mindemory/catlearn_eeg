"""Timed RAM on the Apple GPU (MLX): same model and learning rules as ram_timed.py.

Everything is batched over a leading AGENT axis: an "agent" is one independent network
(one simulated participant), and agents from different conditions (spatial configs,
scenarios, stimulus sets...) can share one GPU batch. Per-agent convolutions run as
grouped convolutions (groups = agents), per-agent dense layers as batched matmuls. Adam
updates every weight independently, so stacking agents in one tensor still trains them
completely independently (each agent's gradient depends only on its own loss).

Trial structure, learning rules, costs and defaults: see ram_timed.py (DEFAULTS, STEP_MS
are shared). One difference in mechanics: all random numbers an update needs (which
display each trial shows, fixation noise, respond/choice draws) are sampled outside the
compiled GPU function and passed in, so that function is deterministic -- it compiles
cleanly and can be tested against fixed inputs.

Shapes: A agents, B trials per agent per update, n displays per agent, H x H pixels,
G = glimpse size (16), C = sensor channels, T = n_fix + n_stim steps.
"""

from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from ram_timed import DEFAULTS, STEP_MS  # noqa: F401  (re-exported for callers)

HIDDEN = 128
GLIMPSE = 16


# ---------------------------------------------------------------- parameters

def init_params(n_agents, n_channels=3, seed=0, hidden=HIDDEN, stop_bias=-2.0):
    """Per-agent weights, same initial distributions as ram_timed.init_params.

    Conv weights are MLX layout (O, kh, kw, I) with a leading agent axis.
    """
    rng = np.random.default_rng(seed)
    A, C, Hd = n_agents, n_channels, hidden
    conv_out = 32 * (GLIMPSE // 2) ** 2

    def dense(n_in, n_out, scale=1.0):
        return {"w": (rng.standard_normal((A, n_in, n_out)) * scale / np.sqrt(n_in)).astype(np.float32),
                "b": np.zeros((A, n_out), np.float32)}

    p = {
        "conv1": {"w": (rng.standard_normal((A, 16, 3, 3, C)) / np.sqrt(C * 9)).astype(np.float32),
                  "b": np.zeros((A, 16), np.float32)},
        "conv2": {"w": (rng.standard_normal((A, 32, 3, 3, 16)) / np.sqrt(16 * 9)).astype(np.float32),
                  "b": np.zeros((A, 32), np.float32)},
        "what": dense(conv_out, Hd),
        "where": dense(2, Hd),
        "clock": dense(2, Hd),
        "lstm": dense(2 * Hd, 4 * Hd),
        "loc": dense(Hd, 2, 0.1),
        "choice": dense(Hd, 2, 0.1),
        "baseline": dense(Hd, 1, 0.1),
        "stop": dense(Hd, 1, 0.1),
    }
    p["stop"]["b"][:] = stop_bias
    return _to_mx(p)


def _to_mx(tree):
    return {k: _to_mx(v) if isinstance(v, dict) else mx.array(v) for k, v in tree.items()}


def to_numpy(tree):
    return {k: to_numpy(v) if isinstance(v, dict) else np.array(v) for k, v in tree.items()}


def select_agents(tree, idx):
    """Sub-tree for a subset of agents (e.g. to save one condition's agents)."""
    return {k: select_agents(v, idx) if isinstance(v, dict) else v[idx] for k, v in tree.items()}


# ---------------------------------------------------------------- sensor

def pad_images(images, scales):
    """(A, n, H, H) -> zero-padded by max(scales)//2 on each side, so crops can run off-screen."""
    pad = max(scales) // 2
    return mx.pad(images, [(0, 0), (0, 0), (pad, pad), (pad, pad)])


def color_channels(images):
    """Number of color channels: images are (A, n, H, H) luminance or (A, n, H, H, K) color."""
    return images.shape[4] if images.ndim == 5 else 1


def n_sensor_channels(cfg, n_color):
    """Glimpse channels: every crop (and the whole-display view, if any) in every color."""
    return (len(cfg["scales"]) + int(cfg["full_view"])) * n_color


def prepare_sensor(images, scales):
    """Precompute, once per set of displays, what the sensor samples from.

    Averaging an s x s crop down to 16 x 16 (factor f = s / 16) equals box-filtering the
    image with an f x f window and then taking every f-th pixel. So for each scale we store
    the padded image box-filtered at that scale (via an integral image), per color channel;
    a glimpse then gathers just 16 x 16 values per scale instead of s x s -- exact, cheaper.

    images: (A, n, H, H) or (A, n, H, H, K). Returns a list of (f, filtered) with
    filtered: (A, n, W, W, K) mx arrays (K = 1 for luminance).
    """
    images = np.asarray(images, dtype=np.float64)
    if images.ndim == 4:
        images = images[..., None]
    pad = max(scales) // 2
    padded = np.pad(images, [(0, 0), (0, 0), (pad, pad), (pad, pad), (0, 0)])
    integral = np.pad(padded.cumsum(axis=2).cumsum(axis=3), [(0, 0), (0, 0), (1, 0), (1, 0), (0, 0)])
    out = []
    for s in scales:
        f = s // GLIMPSE
        box = (integral[:, :, f:, f:] - integral[:, :, :-f, f:] - integral[:, :, f:, :-f]
               + integral[:, :, :-f, :-f]) / (f * f)
        out.append((f, mx.array(box.astype(np.float32))))
    return out


def glimpses(sensor, img_idx, loc, H, scales, full_view, images=None):
    """Glimpse stacks (A, B, G, G, C) for each agent's trials, matching env_2by2.glimpse.

    sensor: prepare_sensor(...) output; img_idx: (A, B) display shown on each trial;
    loc: (A, B, 2). Channels are ordered crop by crop (fovea first), color within crop, so
    for luminance C = number of crops, as in env_2by2.glimpse. Crop starts are clamped
    like jax.lax.dynamic_slice, so results match.
    """
    A, B = img_idx.shape
    pad = max(scales) // 2
    Hp = H + 2 * pad
    loc = mx.stop_gradient(loc)       # crop positions are indices: no gradient (as in ram_timed)
    cx = (loc[..., 0] + 1) / 2 * H
    cy = (loc[..., 1] + 1) / 2 * H
    K = sensor[0][1].shape[-1]
    base = mx.arange(A)[:, None] * sensor[0][1].shape[1] + img_idx       # (A, B) flat display index
    grid = mx.arange(GLIMPSE)
    kk = mx.arange(K)
    chans = []
    for s, (f, filt) in zip(scales, sensor):
        W = filt.shape[2]
        x0 = mx.clip(mx.round(cx - s / 2).astype(mx.int32) + pad, 0, Hp - s)
        y0 = mx.clip(mx.round(cy - s / 2).astype(mx.int32) + pad, 0, Hp - s)
        rows = y0[..., None, None] + f * grid[None, None, :, None]       # (A, B, G, 1)
        cols = x0[..., None, None] + f * grid[None, None, None, :]       # (A, B, 1, G)
        flat = ((base[..., None, None] * W + rows) * W + cols)[..., None] * K + kk   # (A, B, G, G, K)
        chans.append(mx.take(filt.reshape(-1), flat))
    if full_view:
        if images.ndim == 4:
            images = images[..., None]
        f = H // GLIMPSE
        shown = images[mx.arange(A)[:, None], img_idx]                    # (A, B, H, H, K)
        chans.append(shown.reshape(A, B, GLIMPSE, f, GLIMPSE, f, K).mean(axis=(3, 5)))
    return mx.concatenate(chans, axis=-1)


# ---------------------------------------------------------------- network pieces

def _encode(p, g, loc, clock):
    """g: (A, B, G, G, C) -> (A, B, hidden). Agents fold into conv channels (groups = A)."""
    A, B, _, _, C = g.shape
    x = g.transpose(1, 2, 3, 0, 4).reshape(B, GLIMPSE, GLIMPSE, A * C)
    w1 = p["conv1"]["w"].reshape(-1, 3, 3, C)
    x = mx.maximum(mx.conv2d(x, w1, stride=1, padding=1, groups=A) + p["conv1"]["b"].reshape(-1), 0)
    # conv2 is a stride-2 'SAME' conv as in JAX/XLA (pad 0 before, 1 after on 16x16), computed
    # as a stride-1 conv padded (0, 2) keeping every 2nd output: identical values, but MLX's
    # backward pass is ~5x faster through stride-1 convs than through strided ones
    w2 = p["conv2"]["w"].reshape(-1, 3, 3, 16)
    x = mx.pad(x, [(0, 0), (0, 2), (0, 2), (0, 0)])
    x = mx.conv2d(x, w2, stride=1, padding=0, groups=A)[:, ::2, ::2, :]
    x = mx.maximum(x + p["conv2"]["b"].reshape(-1), 0)
    half = GLIMPSE // 2
    x = x.reshape(B, half, half, A, 32).transpose(3, 0, 1, 2, 4).reshape(A, B, -1)
    what = x @ p["what"]["w"] + p["what"]["b"][:, None, :]
    where = loc @ p["where"]["w"] + p["where"]["b"][:, None, :]
    when = (clock @ p["clock"]["w"] + p["clock"]["b"])[:, None, :]      # clock: (2,) shared
    return mx.maximum(what + where + when, 0)


def _lstm(p, h, c, x):
    z = mx.concatenate([x, h], axis=-1) @ p["lstm"]["w"] + p["lstm"]["b"][:, None, :]
    i, f, g, o = mx.split(z, 4, axis=-1)
    c = mx.sigmoid(f + 1.0) * c + mx.sigmoid(i) * mx.tanh(g)
    return mx.sigmoid(o) * mx.tanh(c), c


def _head(p, name, h):
    return h @ p[name]["w"] + p[name]["b"][:, None, :]


def _log_sigmoid(z):
    return -mx.logaddexp(mx.zeros_like(z), -z)


# ---------------------------------------------------------------- one batch of timed trials

def batch_loss(p, sensor, images, labels, img_idx, noise_loc, u_stop, u_choice, noise_periph, cfg):
    """Loss (summed over agents, mean over trials) and per-trial outputs.

    sensor: prepare_sensor(images, scales); images: (A, n, H, H[, K]); labels: (A, n);
    img_idx: (A, B); noise_loc: (A, B, T-1, 2) standard normal; u_stop, u_choice:
    (A, B, n_stim) uniform; noise_periph: (A, B, T, G, G, C) standard normal (or None).
    """
    n_fix, n_stim = cfg["n_fix"], cfg["n_stim"]
    T = n_fix + n_stim
    A, B = img_idx.shape
    H = images.shape[2]
    Hd = p["lstm"]["w"].shape[-1] // 4
    n_color = color_channels(images)
    C = n_sensor_channels(cfg, n_color)
    lab = mx.take_along_axis(labels, img_idx, axis=1)                    # (A, B)

    h = mx.zeros((A, B, Hd))
    c = mx.zeros((A, B, Hd))
    loc = mx.zeros((A, B, 2))
    alive = mx.ones((A, B))
    # Per-step contributions, stacked and summed once at the end: running sums over 20
    # steps form elementwise chains that MLX's compiler would fuse into one oversized kernel
    resp_terms, correct_terms, plabel_terms, ce_terms, penalty_terms, travel_terms = [], [], [], [], [], []
    choice_terms = []                    # response made (1 = category 1), 0 until responded
    logp_terms, values, masks = [], [], []
    locs, mean_locs, loc_mask, loc_on = [], [], [], []
    onset_loc = loc
    blank = mx.zeros((A, B, GLIMPSE, GLIMPSE, C))

    for t in range(T):
        on = t >= n_fix
        if t == n_fix:
            onset_loc = loc
        g = glimpses(sensor, img_idx, loc, H, cfg["scales"], cfg["full_view"], images) if on else blank
        if noise_periph is not None:
            # noise on every channel but the fovea (the first n_color channels)
            keep = mx.array([0.0] * n_color + [1.0] * (C - n_color))
            g = g + cfg["periph_noise"] * noise_periph[:, :, t] * keep
        clock = mx.array([1.0 if on else 0.0, (t - n_fix) / n_stim if on else t / n_fix])
        h, c = _lstm(p, h, c, _encode(p, g, loc, clock))
        h_sg = mx.stop_gradient(h)
        values.append(_head(p, "baseline", h_sg)[..., 0])
        masks.append(alive)

        logp_t = mx.zeros((A, B))
        if on:
            k = t - n_fix
            z = _head(p, "stop", h_sg)[..., 0]
            stop = (u_stop[:, :, k] < mx.sigmoid(z)).astype(mx.float32)
            logp_t = logp_t + alive * (stop * _log_sigmoid(z) + (1 - stop) * _log_sigmoid(-z))

            logits = _head(p, "choice", h)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            choice = (u_choice[:, :, k] < mx.exp(log_probs[..., 1])).astype(mx.int32)
            lp_label = mx.take_along_axis(log_probs, lab[..., None], axis=-1)[..., 0]
            respond = alive * stop
            resp_terms.append(respond * (k + 1))
            correct_terms.append(respond * (choice == lab))
            choice_terms.append(respond * choice)
            plabel_terms.append(respond * mx.exp(lp_label))
            ce_terms.append(-respond * lp_label)
            alive = alive * (1 - stop)

        if t < T - 1:
            mean = mx.tanh(_head(p, "loc", h_sg))
            new_loc = mx.stop_gradient(mx.clip(mean + cfg["sigma"] * noise_loc[:, :, t], -1.0, 1.0))
            logp_t = logp_t + alive * mx.sum(-0.5 * ((new_loc - mean) / cfg["sigma"]) ** 2, axis=-1)
            penalty_terms.append(alive * cfg["saccade_cost"] * mx.sqrt(mx.sum((mean - loc) ** 2, axis=-1) + 1e-8))
            travel_terms.append(alive * mx.sqrt(mx.sum((new_loc - loc) ** 2, axis=-1) + 1e-12))
            locs.append(new_loc)
            mean_locs.append(mean)
            loc_mask.append(alive)
            loc_on.append(1.0 if t + 1 >= n_fix else 0.0)
            # Gaze location carries no gradient (sampled; the respond/wait decision is a
            # non-differentiable comparison) -- made explicit for MLX, implicit in JAX
            loc = mx.stop_gradient(alive[..., None] * new_loc + (1 - alive[..., None]) * loc)
        logp_terms.append(logp_t)

    def total(terms):
        return mx.sum(mx.stack(terms, axis=-1), axis=-1)

    timed_out = alive
    resp_steps = total(resp_terms) + timed_out * n_stim
    correct = total(correct_terms)
    choice_made = mx.where(timed_out > 0.5, -1.0, total(choice_terms))   # -1 = no response
    p_label = total(plabel_terms)
    ce = total(ce_terms)
    penalty = total(penalty_terms)
    travel = total(travel_terms)
    reward = correct - cfg["time_cost"] * resp_steps

    values = mx.stack(values, axis=-1)                                   # (A, B, T)
    masks = mx.stack(masks, axis=-1)
    advantage = mx.stop_gradient(reward[..., None] - values)
    loss_reinforce = -mx.sum(mx.stack(logp_terms, axis=-1) * advantage, axis=-1)
    loss_baseline = mx.sum(masks * (reward[..., None] - values) ** 2, axis=-1)
    per_trial = ce + loss_reinforce + loss_baseline + penalty
    loss = mx.sum(mx.mean(per_trial, axis=1))                             # independent agents

    out = {
        "acc": p_label, "correct": correct, "timeout": timed_out, "rt_steps": resp_steps,
        "choice": choice_made, "label": lab,
        "reward": reward, "travel": travel,
        "locs": mx.stack(locs, axis=2), "mean_locs": mx.stack(mean_locs, axis=2),   # (A, B, T-1, 2)
        "loc_mask": mx.stack(loc_mask, axis=2) * mx.array(loc_on),                   # stimulus phase, running
        "loc_alive": mx.stack(loc_mask, axis=2),                                     # trial still running (any phase)
        "onset_loc": onset_loc,
    }
    return loss, out


# ---------------------------------------------------------------- training

class Trainer:
    """Holds agents' weights and optimizer; one call = one update of every agent.

    trainer = Trainer(params, cfg, lr)
    out = trainer.step(images, labels, batch_size, rng)   # numpy dict of per-trial outputs
    """

    def __init__(self, params, cfg, lr=1e-3, compile=True):
        self.params = params
        self.cfg = dict(cfg)
        self.opt = optim.Adam(learning_rate=lr)
        self.opt.init(self.params)
        cfg_static = self.cfg

        def loss_fn(p, sensor, images, labels, img_idx, noise_loc, u_stop, u_choice, noise_periph):
            return batch_loss(p, sensor, images, labels, img_idx, noise_loc, u_stop, u_choice, noise_periph, cfg_static)

        vg = mx.value_and_grad(loss_fn)
        self._vg = mx.compile(vg) if compile else vg

    def step(self, images, labels, batch_size, rng, sensor=None):
        """images (A, n, H, H) or (A, n, H, H, K) mx array, labels (A, n) mx int array. Pass sensor =
        prepare_sensor(images, scales) when the same displays are used for many updates."""
        cfg = self.cfg
        A, n = labels.shape
        T = cfg["n_fix"] + cfg["n_stim"]
        C = n_sensor_channels(cfg, color_channels(images))
        if sensor is None:
            sensor = prepare_sensor(np.array(images), cfg["scales"])
        img_idx = mx.array(rng.integers(0, n, (A, batch_size)), dtype=mx.int32)
        noise_loc = mx.array(rng.standard_normal((A, batch_size, T - 1, 2)), dtype=mx.float32)
        u_stop = mx.array(rng.random((A, batch_size, cfg["n_stim"])), dtype=mx.float32)
        u_choice = mx.array(rng.random((A, batch_size, cfg["n_stim"])), dtype=mx.float32)
        noise_periph = (mx.array(rng.standard_normal((A, batch_size, T, GLIMPSE, GLIMPSE, C)), dtype=mx.float32)
                        if cfg["periph_noise"] > 0 else None)
        (loss, out), grads = self._vg(self.params, sensor, images, labels, img_idx, noise_loc, u_stop, u_choice,
                                      noise_periph)
        self.params = self.opt.apply_gradients(grads, self.params)
        mx.eval(self.params, self.opt.state, loss, out)
        res = {k: np.array(v) for k, v in out.items()}
        res["loss"] = float(loss)
        res["img_idx"] = np.array(img_idx)
        return res


def summarize(res, ab_locs):
    """Per-agent means for one update (numpy). ab_locs: (A, 2, 2) centers of A / B per agent."""
    def at_ab(locs):
        d = np.linalg.norm(locs[..., None, :] - ab_locs.reshape(ab_locs.shape[0], *([1] * (locs.ndim - 2)), 2, 2), axis=-1)
        near = d < 2.0 / 7.0
        return near[..., 0] & (d[..., 0] <= d[..., 1]), near[..., 1] & (d[..., 1] < d[..., 0])

    m = res["loc_mask"]                                                  # (A, B, T-1)
    aim_a, aim_b = at_ab(res["mean_locs"])
    on_a, on_b = at_ab(res["onset_loc"])
    responded = 1 - res["timeout"]
    n_resp = responded.sum(1)
    return {
        "acc": res["acc"].mean(1),
        "correct": res["correct"].mean(1),
        "timeout": res["timeout"].mean(1),
        "rt_ms": np.where(n_resp > 0, STEP_MS * (res["rt_steps"] * responded).sum(1) / np.maximum(n_resp, 1), np.nan),
        "travel": res["travel"].mean(1),
        "aim_A": (aim_a * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1),
        "aim_B": (aim_b * m).sum((1, 2)) / np.maximum(m.sum((1, 2)), 1),
        "onset_A": on_a.mean(1),
        "onset_B": on_b.mean(1),
    }


# ---------------------------------------------------------------- empirical NTK

def readout(p, sensor, images, cfg, k_stim):
    """Deterministic choice readout, (A, n): logit(cat 1) - logit(cat 0) for every display,
    after the fixation period and k_stim stimulus steps, gaze following the policy mean."""
    A, n = images.shape[:2]
    H = images.shape[2]
    n_fix, n_stim = cfg["n_fix"], cfg["n_stim"]
    C = n_sensor_channels(cfg, color_channels(images))
    Hd = p["lstm"]["w"].shape[-1] // 4
    img_idx = mx.broadcast_to(mx.arange(n)[None], (A, n)).astype(mx.int32)
    h = mx.zeros((A, n, Hd))
    c = mx.zeros((A, n, Hd))
    loc = mx.zeros((A, n, 2))
    blank = mx.zeros((A, n, GLIMPSE, GLIMPSE, C))
    for t in range(n_fix + k_stim):
        on = t >= n_fix
        g = glimpses(sensor, img_idx, loc, H, cfg["scales"], cfg["full_view"], images) if on else blank
        clock = mx.array([1.0 if on else 0.0, (t - n_fix) / n_stim if on else t / n_fix])
        h, c = _lstm(p, h, c, _encode(p, g, loc, clock))
        loc = mx.stop_gradient(mx.tanh(_head(p, "loc", h)))
    logits = _head(p, "choice", h)
    return logits[..., 1] - logits[..., 0]


def empirical_ntk(p, images, cfg, k_stim):
    """Per-agent empirical NTK of the readout over the displays: (A, n, n) numpy.

    K[a, s, s'] = <df_a(s)/dtheta_a, df_a(s')/dtheta_a>; agents are independent, so one
    gradient of the summed readout per display gives every agent's row at once.
    """
    sensor = prepare_sensor(np.array(images), cfg["scales"])
    A, n = images.shape[:2]
    rows = []
    for s in range(n):
        g = mx.grad(lambda pp: mx.sum(readout(pp, sensor, images, cfg, k_stim)[:, s]))(p)
        leaves = [v for d in g.values() for v in d.values()]
        rows.append(mx.concatenate([v.reshape(A, -1) for v in leaves], axis=1))
    J = mx.stack(rows, axis=1)                                            # (A, n, P)
    K = J @ J.transpose(0, 2, 1)
    mx.eval(K)
    return np.array(K)


def ntk_directions(p, images, cfg, k_stim, basis):
    """Per-agent NTK along given directions over the displays: (A, m) numpy with entry
    v_j^T K v_j for each row v_j of basis (m, n).

    Same quantity as empirical_ntk projected onto the basis, but computed from the
    gradient of the projected readout sum_s v_j[s] f(s), so it is a sum of squares: exact
    in float32 even when a direction carries 1e-6 of the kernel's trace (projecting K
    afterwards cancels large entries and loses such small modes to rounding).
    """
    sensor = prepare_sensor(np.array(images), cfg["scales"])
    basis = mx.array(np.asarray(basis, dtype=np.float32))
    out = []
    for v in basis:
        g = mx.grad(lambda pp: mx.sum(readout(pp, sensor, images, cfg, k_stim) * v))(p)
        leaves = [x for d in g.values() for x in d.values()]
        out.append(sum(mx.sum(x.reshape(x.shape[0], -1) ** 2, axis=1) for x in leaves))
    out = mx.stack(out, axis=1)
    mx.eval(out)
    return np.array(out)
