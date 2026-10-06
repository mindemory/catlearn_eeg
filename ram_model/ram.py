"""Recurrent Attention Model (Mnih et al., 2014) in plain JAX, for the 2x2 task.

One trial = one episode of n_glimpses glimpses:

    glimpse 0 at central fixation (as the human task starts)
    for each glimpse:  sensor -> small CNN (+ location embedding) -> LSTM
                       location head samples where to look next (free gaze)
    after the last glimpse: choice head picks the category (F / J)

Training per episode:
  choice head     cross-entropy on the correct category. With two choices and
                  correct/incorrect feedback this is the same information a human gets.
  location head   REINFORCE: log-prob of each chosen fixation, weighted by
                  (reward - baseline), where reward = 1 if the sampled choice was correct.
                  This is the temporal credit assignment: a fixation early in the trial
                  is credited with the outcome at the end.
  baseline head   predicts the reward from the hidden state (variance reduction).

Optional saccade cost: the reward that trains the location (and baseline) heads is
    reward - saccade_cost * (total distance the fixation travels in the trial)
in [-1, 1] display units (the display is 2 wide). Gaze then stays put unless moving pays
off, instead of drifting when where it looks doesn't matter. The choice head is unaffected.

The location and baseline heads see a stop-gradient copy of the LSTM state, so the
shared CNN/LSTM are trained by the choice loss only.

In fixate mode the location head is ignored and every glimpse is at the center, so only
what the periphery conveys (covertly) can drive the choice.

Parameters are a plain dict pytree, so everything vmaps over agents.
"""

import jax
import jax.numpy as jnp
import optax

from env_2by2 import glimpse

HIDDEN = 128


def _dense(key, n_in, n_out, scale=1.0):
    w = jax.random.normal(key, (n_in, n_out)) * scale / jnp.sqrt(n_in)
    return {"w": w, "b": jnp.zeros(n_out)}


def init_params(key, n_channels=4, glimpse_px=16, hidden=HIDDEN):
    ks = jax.random.split(key, 9)
    conv_out = 32 * (glimpse_px // 2) ** 2
    return {
        "conv1": {"w": jax.random.normal(ks[0], (16, n_channels, 3, 3)) / jnp.sqrt(n_channels * 9), "b": jnp.zeros(16)},
        "conv2": {"w": jax.random.normal(ks[1], (32, 16, 3, 3)) / jnp.sqrt(16 * 9), "b": jnp.zeros(32)},
        "what": _dense(ks[2], conv_out, hidden),
        "where": _dense(ks[3], 2, hidden),
        "lstm": _dense(ks[4], 2 * hidden, 4 * hidden),
        "loc": _dense(ks[5], hidden, 2, scale=0.1),       # small init: first saccades near center
        "choice": _dense(ks[6], hidden, 2, scale=0.1),
        "baseline": _dense(ks[7], hidden, 1, scale=0.1),
    }


def _conv(x, p, stride):
    y = jax.lax.conv_general_dilated(x[None], p["w"], (stride, stride), "SAME",
                                     dimension_numbers=("NCHW", "OIHW", "NCHW"))[0]
    return jax.nn.relu(y + p["b"][:, None, None])


def _encode(params, g, loc):
    """Glimpse network: 'what' (CNN on the glimpse) combined with 'where' (the location)."""
    h = _conv(g, params["conv1"], 1)
    h = _conv(h, params["conv2"], 2)
    what = h.reshape(-1) @ params["what"]["w"] + params["what"]["b"]
    where = loc @ params["where"]["w"] + params["where"]["b"]
    return jax.nn.relu(what + where)


def _lstm(params, state, x):
    h, c = state
    z = jnp.concatenate([x, h]) @ params["lstm"]["w"] + params["lstm"]["b"]
    i, f, g, o = jnp.split(z, 4)
    c = jax.nn.sigmoid(f + 1.0) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)
    return h, c


def episode(params, img, label, key, n_glimpses=4, free_gaze=True, sigma=0.15, scales=(16, 32, 64), full_view=True,
            periph_noise=0.0, saccade_cost=0.0):
    """Run one trial. Returns (loss, metrics).

    scales / full_view configure the sensor (see env_2by2.glimpse). periph_noise adds fresh
    Gaussian noise of that SD to every channel except the fovea on each glimpse, making
    peripheral (covert) information unreliable while foveal information stays clean.
    """
    h = c = jnp.zeros(params["lstm"]["w"].shape[1] // 4)
    loc = jnp.zeros(2)
    logps, baselines, locs, mean_locs, seen, seen_at = [], [], [], [], [], []
    for t in range(n_glimpses):
        g = glimpse(img, loc, scales=scales, full_view=full_view)
        if periph_noise > 0:
            key, k = jax.random.split(key)
            noise = periph_noise * jax.random.normal(k, g.shape)
            g = g + noise.at[0].set(0.0)
        seen.append(g)
        seen_at.append(loc)
        x = _encode(params, g, loc)
        h, c = _lstm(params, (h, c), x)
        if t < n_glimpses - 1:
            # Location and baseline heads read a detached copy of the state: REINFORCE's
            # high-variance gradient trains only those heads, and the shared CNN/LSTM learn
            # from the choice loss (as in standard RAM implementations).
            h_sg = jax.lax.stop_gradient(h)
            baselines.append((h_sg @ params["baseline"]["w"] + params["baseline"]["b"])[0])
            if free_gaze:
                key, k = jax.random.split(key)
                mean = jnp.tanh(h_sg @ params["loc"]["w"] + params["loc"]["b"])
                new_loc = jax.lax.stop_gradient(jnp.clip(mean + sigma * jax.random.normal(k, (2,)), -1.0, 1.0))
                logps.append(jnp.sum(-0.5 * ((new_loc - mean) / sigma) ** 2))
                loc = new_loc
            else:
                mean = jnp.zeros(2)
                logps.append(jnp.zeros(()))
            locs.append(loc)
            mean_locs.append(mean)

    logits = h @ params["choice"]["w"] + params["choice"]["b"]
    key, k = jax.random.split(key)
    choice = jax.random.categorical(k, logits)
    reward = (choice == label).astype(jnp.float32)

    # Total fixation travel: center -> first chosen fixation -> ... -> last
    path = jnp.concatenate([jnp.zeros((1, 2)), jnp.stack(locs)])
    saccade_dist = jnp.sum(jnp.linalg.norm(jnp.diff(path, axis=0), axis=-1))
    gaze_reward = reward - saccade_cost * saccade_dist

    logp_label = jax.nn.log_softmax(logits)[label]
    baselines = jnp.stack(baselines)
    advantage = jax.lax.stop_gradient(gaze_reward - baselines)
    loss_choice = -logp_label
    loss_reinforce = -jnp.sum(jnp.stack(logps) * advantage)
    loss_baseline = jnp.sum((gaze_reward - baselines) ** 2)
    loss = loss_choice + loss_reinforce + loss_baseline

    metrics = {"p_correct": jnp.exp(logp_label), "reward": reward, "saccade_dist": saccade_dist, "locs": jnp.stack(locs),
               "mean_locs": jnp.stack(mean_locs),
               # what the model saw: each glimpse stack and where it was taken (for movies)
               "glimpses": jnp.stack(seen), "glimpse_locs": jnp.stack(seen_at),
               "probs": jax.nn.softmax(logits), "choice": choice}
    return loss, metrics


def _at_ab(locs, ab_locs):
    """Boolean masks: fixation within one cell (2/7 in [-1, 1] units) of A / of B, whichever is nearer."""
    dist = jnp.linalg.norm(locs[..., None, :] - ab_locs, axis=-1)
    near = dist < (2.0 / 7.0)
    return near[..., 0] & (dist[..., 0] <= dist[..., 1]), near[..., 1] & (dist[..., 1] < dist[..., 0])


def make_train_step(optimizer, batch_size, n_glimpses, free_gaze, sigma, scales=(16, 32, 64), full_view=True,
                    periph_noise=0.0, saccade_cost=0.0):
    """One update on a batch of trials drawn uniformly from a set of displays (a block's 4
    compounds, or any other set, e.g. for pretraining).

    Returned function: (params, opt_state, key, images (n,H,W), labels (n,), ab_locs (2,2))
    -> (params, opt_state, metrics). Wrap in jax.vmap to train many agents at once.
    """

    def batch_loss(params, key, images, labels):
        k_s, k_ep = jax.random.split(key)
        s = jax.random.randint(k_s, (batch_size,), 0, images.shape[0])
        keys = jax.random.split(k_ep, batch_size)
        losses, metrics = jax.vmap(
            lambda im, lab, k: episode(params, im, lab, k, n_glimpses, free_gaze, sigma, scales, full_view, periph_noise,
                                      saccade_cost)
        )(images[s], labels[s], keys)
        return losses.mean(), (metrics, s)

    def step(params, opt_state, key, images, labels, ab_locs):
        (loss, (metrics, s)), grads = jax.value_and_grad(batch_loss, has_aux=True)(params, key, images, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        # Where the chosen fixations landed (noisy samples) and where the policy aimed (means)
        locs, mean_locs = metrics["locs"], metrics["mean_locs"]           # (batch, n_glimpses-1, 2)
        at_a, at_b = _at_ab(locs, ab_locs)
        aim_a, aim_b = _at_ab(mean_locs, ab_locs)
        out = {
            "loss": loss,
            "p_correct": metrics["p_correct"].mean(),
            "reward": metrics["reward"].mean(),
            "saccade_dist": metrics["saccade_dist"].mean(),
            "gaze_A": at_a.mean(),          # sampled fixations
            "gaze_B": at_b.mean(),
            "aim_A": aim_a.mean(),          # policy means
            "aim_B": aim_b.mean(),
            "first_aim_A": aim_a[:, 0].mean(),
            "first_aim_B": aim_b[:, 0].mean(),
            "locs": locs,
            "mean_locs": mean_locs,
            "stim": s,
        }
        return params, opt_state, out

    return step
