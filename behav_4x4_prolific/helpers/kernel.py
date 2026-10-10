"""The kernel learner (Peng, Ehrlich, Lee & Murray 2025; kernel_model/kernel_modes.py) as a
likelihood over a participant's test-block choices, its simulation, and maximum-likelihood fits.

Model 1 (fixed kernel per block). The learner keeps a value y over the 16 pairs, reset to 0 at
the start of each test block (new fractals). Its kernel is a weighted sum of the four mode
projectors of the 4x4 design,
    K_b = sum_e (16 w_be / d_e) P_e,     e in (cst, A, B, AB), d_e = 1, 3, 3, 9,
with weights w_b on the simplex (one set per test block b); K has ones on its diagonal.
On a trial with pair s:
    P(F) = sigmoid(beta * y_s)                               the choice
    y   <- y + lr * (y*_s - y_s) * K_b[:, s]                 after the feedback (y*_s = +1 F, -1 J)
Late answers (no choice) give no likelihood and no update: the feedback (red circles) doesn't
say which key was right. lr is the share of the error corrected on the pair itself (0-1); the
kernel spreads that correction to the other pairs, mode by mode.

Parameters (11): w for each of the 3 test blocks (3 free each: softmax of logits, cst's logit
fixed at 0), lr (logistic of a free value), beta (exp of a free value).

  Sequences.from_trials(test trials)   arrays of one participant's test blocks
  nll(theta, seq)                      negative log-likelihood (JAX; jit and grad below)
  simulate(params, seq, rng)           choices of a model learner on the same trial sequence
  fit(seq, n_starts, rng)              maximum likelihood, several starts (L-BFGS-B, exact gradient)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from helpers.measures import DESIGN, rule_type, test_blocks

jax.config.update("jax_enable_x64", True)

KERNEL_MODES = list(DESIGN.mode_names)              # ['cst', 'A', 'B', 'AB']
DIMS = np.asarray(DESIGN.mode_dims, float)
_P = jnp.asarray(DESIGN.projectors)
_DIMS = jnp.asarray(DIMS)
N_BLOCKS = 3
N_PARAMS = N_BLOCKS * (len(KERNEL_MODES) - 1) + 2


@dataclass
class Sequences:
    """One participant's test blocks as arrays over trials (blocks concatenated)."""
    pair: np.ndarray        # compound 0-15
    target: np.ndarray      # +1 F, -1 J
    choice: np.ndarray      # 1 F, 0 J (0 where late)
    answered: np.ndarray    # 1 answered, 0 late
    block: np.ndarray       # 0, 1, 2
    start: np.ndarray       # 1 on a block's first trial
    types: tuple            # rule type per block, e.g. ('VI', 'X', 'II')

    @classmethod
    def from_trials(cls, r):
        parts = test_blocks(r)
        pair, target, choice, answered, block = [], [], [], [], []
        for k, (_, _, b) in enumerate(parts):
            pair.append(b["compound"].to_numpy())
            target.append(2.0 * b["category"].to_numpy() - 1)
            choice.append(np.nan_to_num(b["choice"].to_numpy(float), nan=0.0))
            answered.append((~b["timeout"]).to_numpy().astype(float))
            block.append(np.full(len(b), k))
        block = np.concatenate(block)
        return cls(np.concatenate(pair).astype(int), np.concatenate(target), np.concatenate(choice),
                   np.concatenate(answered), block.astype(int), np.r_[1, np.diff(block) != 0].astype(float),
                   tuple(rule_type(rule) for _, rule, _ in parts))

    def with_choices(self, choice):
        return Sequences(self.pair, self.target, np.asarray(choice, float), self.answered, self.block, self.start,
                         self.types)

    def arrays(self):
        return tuple(jnp.asarray(a) for a in (self.pair, self.target, self.choice, self.answered, self.block,
                                             self.start))


# ---------------------------------------------------------------- parameters
def unpack(theta):
    """(W (blocks x modes), lr, beta) from the free parameter vector."""
    logits = jnp.concatenate([jnp.zeros((N_BLOCKS, 1)), jnp.reshape(theta[:N_BLOCKS * 3], (N_BLOCKS, 3))], 1)
    return jax.nn.softmax(logits, axis=1), jax.nn.sigmoid(theta[-2]), jnp.exp(theta[-1])


def pack(W, lr, beta):
    W = np.clip(np.asarray(W, float), 1e-6, None)
    logits = np.log(W[:, 1:]) - np.log(W[:, :1])
    return np.r_[logits.ravel(), np.log(lr / (1 - lr)), np.log(beta)]


def kernels(W):
    return jnp.einsum("be,enm->bnm", 16 * W / _DIMS, _P)


# ---------------------------------------------------------------- likelihood
def _nll(theta, pair, target, choice, answered, block, start):
    W, lr, beta = unpack(theta)
    Ks = kernels(W)

    def step(y, x):
        s, t, c, a, b, first = x
        y = jnp.where(first > 0, jnp.zeros_like(y), y)
        ys = y[s]
        logit = beta * ys
        ll = a * (c * jax.nn.log_sigmoid(logit) + (1 - c) * jax.nn.log_sigmoid(-logit))
        y = y + a * lr * (t - ys) * Ks[b][:, s]
        return y, ll

    _, lls = jax.lax.scan(step, jnp.zeros(16), (pair, target, choice, answered, block, start))
    return -jnp.sum(lls)


_value_and_grad = jax.jit(jax.value_and_grad(_nll))
_value = jax.jit(_nll)


def nll(theta, seq):
    return float(_value(jnp.asarray(theta), *seq.arrays()))


# ---------------------------------------------------------------- simulation
def simulate(W, lr, beta, seq, rng):
    """Choices (1 F / 0 J; 0 where late) of a model learner on the participant's own trial
    sequence, pairs and late answers."""
    Ks = np.asarray(kernels(jnp.asarray(W)))
    y = np.zeros(16)
    choice = np.zeros(len(seq.pair))
    for i, (s, t, a, b, first) in enumerate(zip(seq.pair, seq.target, seq.answered, seq.block, seq.start)):
        if first:
            y[:] = 0
        if not a:
            continue
        choice[i] = rng.random() < 1 / (1 + np.exp(-beta * y[s]))
        y += lr * (t - y[s]) * Ks[b][:, s]
    return choice


# ---------------------------------------------------------------- fitting
def fit(seq, rng, n_starts=6, theta0=None):
    """Maximum likelihood over several starts; returns (W, lr, beta, nll, theta)."""
    arrays = seq.arrays()

    def f(theta):
        v, g = _value_and_grad(jnp.asarray(theta), *arrays)
        return float(v), np.asarray(g, float)

    starts = [] if theta0 is None else [np.asarray(theta0, float)]
    while len(starts) < n_starts:
        W = rng.dirichlet(np.ones(len(KERNEL_MODES)), size=N_BLOCKS)
        starts.append(pack(W, rng.uniform(0.03, 0.4), np.exp(rng.uniform(np.log(0.8), np.log(8)))))
    bounds = [(-8, 8)] * (N_BLOCKS * 3) + [(-6, 4), (-3, 4)]
    best = None
    for x0 in starts:
        res = minimize(f, np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds]), jac=True, method="L-BFGS-B",
                       bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    W, lr, beta = unpack(jnp.asarray(best.x))
    return np.asarray(W), float(lr), float(beta), float(best.fun), best.x


def fit_group(seqs, rng, n_starts=4):
    """Maximum likelihood of one parameter set shared by several participants (the sum of their
    log-likelihoods); returns (W, lr, beta, nll, theta)."""
    arrays = [s.arrays() for s in seqs]

    def f(theta):
        th = jnp.asarray(theta)
        v, g = 0.0, np.zeros(len(theta))
        for a in arrays:
            vi, gi = _value_and_grad(th, *a)
            v += float(vi)
            g += np.asarray(gi, float)
        return v, g

    bounds = [(-8, 8)] * (N_BLOCKS * 3) + [(-6, 4), (-3, 4)]
    best = None
    for _ in range(n_starts):
        W = rng.dirichlet(np.ones(len(KERNEL_MODES)), size=N_BLOCKS)
        x0 = pack(W, rng.uniform(0.03, 0.4), np.exp(rng.uniform(np.log(0.8), np.log(8))))
        res = minimize(f, np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds]), jac=True, method="L-BFGS-B",
                       bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    W, lr, beta = unpack(jnp.asarray(best.x))
    return np.asarray(W), float(lr), float(beta), float(best.fun), best.x
