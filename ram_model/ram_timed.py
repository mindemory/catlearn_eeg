"""Recurrent Attention Model with the real task's trial timing (learning stays in the weights).

One trial, in steps of STEP_MS = 250 ms (about one human fixation), mirroring task_2by2:

    fixation   n_fix steps (1 s)    blank screen; gaze starts at the center and may move
    stimulus   up to n_stim steps   the display is on. At every step the agent decides
               (4 s)                whether to respond now and, if so, which category;
                                    otherwise it keeps looking and picks the next fixation.
                                    Responding ends the trial; RT = steps since onset.
                                    No response by the last step = "too slow" (incorrect).
    feedback                        correct / incorrect drives learning only: every trial
                                    starts from a fresh network state ("timing only"), so
                                    the feedback and ITI periods need no simulated steps.

A clock input tells the network whether the stimulus is on and how far into the phase it
is, the way a person notices stimulus onset even with the patches off to the side.

Learning, per trial:
  choice head   cross-entropy on the correct category at the response step (the same
                information 2-choice correct/incorrect feedback gives). Trains the shared
                CNN / LSTM.
  stop head     REINFORCE on respond-vs-wait, reward = 1 if correct, minus
                time_cost * (steps until the response). The time cost creates a
                speed-accuracy tradeoff; without it the agent would always wait until the
                deadline.
  location head REINFORCE with the same reward, plus a direct penalty
                saccade_cost * ||intended fixation - current fixation|| on every move the
                policy makes, so gaze stays put unless moving pays off. (The cost is on the
                intended move, not the noisy sample, so it penalises only what the policy
                controls.)
  baseline      predicts the reward at every step (variance reduction).
Stop, location and baseline heads read a stop-gradient copy of the LSTM state, as in ram.py.
"""

import jax
import jax.numpy as jnp
import optax

from env_2by2 import glimpse
from ram import HIDDEN, _at_ab, _conv, _dense, _lstm
from ram import init_params as _init_base

STEP_MS = 250

DEFAULTS = {
    "n_fix": 4,            # 1 s fixation
    "n_stim": 16,          # 4 s response window
    "sigma": 0.15,         # fixation noise SD, [-1, 1] display units
    "scales": (16, 32, 64),
    "full_view": False,    # the 'graded' sensor
    "periph_noise": 0.0,
    "saccade_cost": 0.2,   # per unit of intended fixation displacement
    "time_cost": 0.01,     # per 250-ms step until the response
}


def init_params(key, n_channels=3, glimpse_px=16, hidden=HIDDEN, stop_bias=-2.0):
    """ram.init_params plus a clock embedding and a stop (respond-now) head.

    stop_bias = -2 -> p(respond) ~ 0.12 per step at first: an unhurried start with
    ~13% timeouts, rather than instant guessing.
    """
    k_base, k_clock, k_stop = jax.random.split(key, 3)
    params = _init_base(k_base, n_channels, glimpse_px, hidden)
    params["clock"] = _dense(k_clock, 2, hidden)
    params["stop"] = _dense(k_stop, hidden, 1, scale=0.1)
    params["stop"]["b"] = jnp.array([stop_bias])
    return params


def _safe_norm(v, eps=1e-8):
    """||v|| with a finite gradient at v = 0 (plain norm gives 0/0 there, e.g. when the
    policy intends to stay exactly where it is on the first, all-blank fixation step)."""
    return jnp.sqrt(jnp.sum(v ** 2) + eps)


def _encode(params, g, loc, clock):
    h = _conv(g, params["conv1"], 1)
    h = _conv(h, params["conv2"], 2)
    what = h.reshape(-1) @ params["what"]["w"] + params["what"]["b"]
    where = loc @ params["where"]["w"] + params["where"]["b"]
    when = clock @ params["clock"]["w"] + params["clock"]["b"]
    return jax.nn.relu(what + where + when)


def episode(params, img, label, key, cfg):
    """One timed trial. cfg: dict like DEFAULTS (static). Returns (loss, metrics)."""
    n_fix, n_stim = cfg["n_fix"], cfg["n_stim"]
    T = n_fix + n_stim
    h = c = jnp.zeros(params["lstm"]["w"].shape[1] // 4)
    loc = jnp.zeros(2)
    blank = jnp.zeros_like(img)

    alive = jnp.ones(())                 # 1 until the agent responds
    resp_steps = jnp.zeros(())           # steps from onset to response
    correct = jnp.zeros(())
    p_label = jnp.zeros(())              # p(correct category) at the response step
    ce = jnp.zeros(())
    penalty = jnp.zeros(())              # saccade-cost penalty (direct, differentiable)
    travel = jnp.zeros(())               # actual (noisy) fixation travel, for reporting
    logp_terms, value_terms = [], []     # per step: (masked log-prob sum), (baseline, mask)
    locs, mean_locs, loc_mask, loc_on = [], [], [], []
    onset_loc = jnp.zeros(2)

    for t in range(T):
        on = t >= n_fix
        if t == n_fix:
            onset_loc = loc                                   # where gaze is at stimulus onset
        g = glimpse(img if on else blank, loc, scales=cfg["scales"], full_view=cfg["full_view"])
        if cfg["periph_noise"] > 0:
            key, k = jax.random.split(key)
            g = g + (cfg["periph_noise"] * jax.random.normal(k, g.shape)).at[0].set(0.0)
        clock = jnp.array([1.0 if on else 0.0, (t - n_fix) / n_stim if on else t / n_fix])
        h, c = _lstm(params, (h, c), _encode(params, g, loc, clock))
        h_sg = jax.lax.stop_gradient(h)
        value_terms.append(((h_sg @ params["baseline"]["w"] + params["baseline"]["b"])[0], alive))

        logp_t = jnp.zeros(())
        if on:
            z = (h_sg @ params["stop"]["w"] + params["stop"]["b"])[0]
            key, k_stop, k_choice = jax.random.split(key, 3)
            stop = (jax.random.uniform(k_stop) < jax.nn.sigmoid(z)).astype(jnp.float32)
            logp_t = logp_t + alive * (stop * jax.nn.log_sigmoid(z) + (1 - stop) * jax.nn.log_sigmoid(-z))

            logits = h @ params["choice"]["w"] + params["choice"]["b"]
            log_probs = jax.nn.log_softmax(logits)
            choice = jax.random.categorical(k_choice, logits)
            respond = alive * stop
            resp_steps = resp_steps + respond * (t - n_fix + 1)
            correct = correct + respond * (choice == label)
            p_label = p_label + respond * jnp.exp(log_probs[label])
            ce = ce - respond * log_probs[label]
            alive = alive * (1 - stop)

        if t < T - 1:
            # Next fixation (only matters, and only costs, if the trial goes on)
            mean = jnp.tanh(h_sg @ params["loc"]["w"] + params["loc"]["b"])
            key, k = jax.random.split(key)
            new_loc = jax.lax.stop_gradient(jnp.clip(mean + cfg["sigma"] * jax.random.normal(k, (2,)), -1.0, 1.0))
            logp_t = logp_t + alive * jnp.sum(-0.5 * ((new_loc - mean) / cfg["sigma"]) ** 2)
            penalty = penalty + alive * cfg["saccade_cost"] * _safe_norm(mean - loc)
            travel = travel + alive * jnp.linalg.norm(new_loc - loc)
            locs.append(new_loc)
            mean_locs.append(mean)
            loc_mask.append(alive)
            loc_on.append(1.0 if t + 1 >= n_fix else 0.0)
            loc = alive * new_loc + (1 - alive) * loc
        logp_terms.append(logp_t)

    timed_out = alive
    resp_steps = resp_steps + timed_out * n_stim
    reward = correct - cfg["time_cost"] * resp_steps

    baselines = jnp.stack([v for v, _ in value_terms])
    masks = jnp.stack([m for _, m in value_terms])
    advantage = jax.lax.stop_gradient(reward - baselines)
    loss_reinforce = -jnp.sum(jnp.stack(logp_terms) * advantage)
    loss_baseline = jnp.sum(masks * (reward - baselines) ** 2)
    loss = ce + loss_reinforce + loss_baseline + penalty

    metrics = {
        "acc": p_label,                          # expected accuracy; 0 on timeouts (they count as errors)
        "correct": correct,
        "timeout": timed_out,
        "rt_steps": resp_steps,
        "reward": reward,
        "travel": travel,
        "locs": jnp.stack(locs),                 # (T-1, 2): fixation for steps 1..T-1
        "mean_locs": jnp.stack(mean_locs),
        "loc_mask": jnp.stack(loc_mask),         # 1 while the trial was still running
        "loc_on": jnp.array(loc_on),             # 1 for fixations during the stimulus phase
        "onset_loc": onset_loc,
    }
    return loss, metrics


def make_train_step(optimizer, batch_size, cfg):
    """One update on a batch of timed trials drawn uniformly from a set of displays.

    Returned function: (params, opt_state, key, images (n,H,W), labels (n,), ab_locs (2,2))
    -> (params, opt_state, metrics). Wrap in jax.vmap to train many agents at once.
    """

    def batch_loss(params, key, images, labels):
        k_s, k_ep = jax.random.split(key)
        s = jax.random.randint(k_s, (batch_size,), 0, images.shape[0])
        losses, metrics = jax.vmap(lambda im, lab, k: episode(params, im, lab, k, cfg))(
            images[s], labels[s], jax.random.split(k_ep, batch_size))
        return losses.mean(), (metrics, s)

    def step(params, opt_state, key, images, labels, ab_locs):
        (loss, (m, s)), grads = jax.value_and_grad(batch_loss, has_aux=True)(params, key, images, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Gaze during the stimulus phase, over fixations made while the trial was running
        stim_mask = m["loc_mask"] * m["loc_on"]                            # (batch, T-1)
        aim_a, aim_b = _at_ab(m["mean_locs"], ab_locs)
        n_stim_fix = jnp.maximum(stim_mask.sum(), 1.0)
        onset_a, onset_b = _at_ab(m["onset_loc"], ab_locs)
        responded = 1 - m["timeout"]
        out = {
            "loss": loss,
            "acc": m["acc"].mean(),
            "correct": m["correct"].mean(),
            "timeout": m["timeout"].mean(),
            "rt_ms": STEP_MS * jnp.sum(m["rt_steps"] * responded) / jnp.maximum(responded.sum(), 1.0),
            "travel": m["travel"].mean(),
            "aim_A": jnp.sum(aim_a * stim_mask) / n_stim_fix,
            "aim_B": jnp.sum(aim_b * stim_mask) / n_stim_fix,
            "onset_A": onset_a.mean(),           # gaze already at A when the stimulus appears
            "onset_B": onset_b.mean(),
            "locs": m["locs"],
            "loc_mask": stim_mask,
            "stim": s,
        }
        return params, opt_state, out

    return step
