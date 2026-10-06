"""Kernel (feature-mode) learner for factorial category tasks with D dimensions x K levels.

A learner is described by a kernel K over the n = K**D stimuli, written as a weighted
sum of orthogonal feature modes (constant, main effects A, B, ..., interactions AB, ...):

    K = sum_e  lambda_e * P_e,        P_e = projector onto mode e (rank d_e)

With every diagonal entry of K fixed to 1, the mode weights w_e = lambda_e * d_e / n are
the share of the kernel's trace in each mode and sum to 1. They are the learner's
inductive bias: under the trialwise update

    y  <-  y + lr * (y*_s - y_s) * K[:, s]  -  forget * y

the component of the target in mode e is learned at a rate set by lambda_e (see
kernel_learning_shared_2410/fitting_method.pdf). This is also how a network trained by
gradient descent learns in the linearized regime, with K as its neural tangent kernel.

A fixed kernel is the lazy / pure-NTK learner. An adaptive kernel additionally drifts
toward the mode make-up of what it has learned, a simple stand-in for feature learning
(the rich regime) that lets experience with one rule change the bias carried into the
next block.

Mode bases, task enumeration and task types come from task_design/plot_task_kernels.py,
so task definitions live in one place.
"""

import sys
from pathlib import Path

import numpy as np

TASK_DESIGN_DIR = Path(__file__).resolve().parents[1] / "task_design"
sys.path.insert(0, str(TASK_DESIGN_DIR))
from plot_task_kernels import effect_bases, effect_name, group_by_type, task_loadings, to_roman  # noqa: E402

DATA_ROOT = Path.home() / "Documents" / "data" / "catlearn_eeg"


class Design:
    """Stimuli, feature modes and category tasks of one factorial design, e.g. '2x2'.

    Stimulus index s enumerates feature levels with the first dimension slowest, the
    same ordering effect_bases() and task_loadings() use.
    """

    def __init__(self, name):
        dims = [int(x) for x in name.split("x")]
        if len(set(dims)) != 1:
            raise ValueError(f"only equal levels per dimension are supported, got {name}")
        self.name = name
        self.n_dims, self.n_levels = len(dims), dims[0]
        self.n_stim = self.n_levels**self.n_dims

        effects, bases = effect_bases(self.n_dims, self.n_levels)
        self.effects = effects
        self.mode_names = [effect_name(e) for e in effects]
        self.mode_dims = np.array([b.shape[1] for b in bases])     # d_e, rank of each mode
        self.mode_orders = np.array([len(e) for e in effects])      # 0 const, 1 main, 2 pair...
        self.projectors = np.stack([b @ b.T for b in bases])        # (modes, n, n)

        # Every balanced task, deduplicated and grouped into types (as in the design figures)
        effects_t, loadings, labels = task_loadings(self.n_dims, self.n_levels)
        loadings, type_sizes, order = group_by_type(effects_t, loadings, self.n_dims)
        self.task_labels = labels[order]                            # (tasks, n), +1 / -1
        self.task_loadings = loadings                               # (tasks, modes)
        self.task_types = np.repeat(np.arange(len(type_sizes)), type_sizes)
        self.task_names = [self._describe(row) for row in loadings]

    def _describe(self, loading_row):
        relevant = [self.mode_names[i] for i in np.argsort(-loading_row) if loading_row[i] > 1e-6]
        return "+".join(relevant)

    @property
    def n_modes(self):
        return len(self.mode_names)

    def task_index(self, name):
        """Index of the task whose relevant modes are exactly `name`, e.g. 'A' or 'AB'."""
        hits = [i for i, t in enumerate(self.task_names) if t == name]
        if len(hits) != 1:
            raise KeyError(f"{self.name}: task '{name}' matches {len(hits)} tasks; "
                           f"available: {sorted(set(self.task_names))}")
        return hits[0]

    def task_label(self, i):
        return f"{self.task_names[i]} (type {to_roman(self.task_types[i] + 1)})"

    # ------------------------------------------------------------------ kernels

    def prior_weights(self, order_decay=0.5):
        """Default bias: each mode's eigenvalue falls by `order_decay` per interaction
        order (simpler functions are learned faster); weights then scale with mode rank."""
        eig = order_decay ** self.mode_orders.astype(float)
        w = eig * self.mode_dims
        return w / w.sum()

    def kernel(self, w):
        """K = sum_e (n w_e / d_e) P_e, which has ones on the diagonal when sum(w) = 1."""
        eig = self.n_stim * np.asarray(w) / self.mode_dims
        return np.tensordot(eig, self.projectors, axes=1)

    def mode_shares(self, y):
        """Share of ||y||^2 in each mode (sums to 1). For a task's +/-1 label vector this
        equals its squared kernel loadings; for a kernel's own function it is its bias."""
        y = np.atleast_2d(y)
        power = np.einsum("rn,emn,rm->re", y, self.projectors, y)
        total = power.sum(axis=1, keepdims=True)
        return np.divide(power, total, out=np.full_like(power, np.nan), where=total > 1e-12)


def make_trial_sequence(n_stim, n_reps, rng):
    """n_reps shuffled passes through all stimuli, as in a human block."""
    return np.concatenate([rng.permutation(n_stim) for _ in range(n_reps)])


def simulate_blocks(design, block_tasks, w0, n_reps=10, lr=0.1, forget=0.0, beta=3.0,
                    adapt_rate=0.0, relax_rate=0.0, reset_between_blocks=True,
                    n_runs=500, seed=0):
    """Simulate n_runs learners through a sequence of blocks.

    block_tasks          task index per block
    w0                   starting mode weights (also what the adaptive kernel relaxes to)
    lr, forget           update y += lr * err * K[:, s] - forget * y
    beta                 choice p(correct) = sigmoid(beta * y_s * y*_s)
    adapt_rate           per-trial drift of the weights toward mode_shares(y); 0 = fixed kernel
    relax_rate           per-trial pull of the weights back to w0 (keeps every mode learnable)
    reset_between_blocks new stimuli each block: learned values restart at 0, the kernel
                         carries over. False = same stimuli, values carry over too.

    Returns a dict of arrays, trials concatenated across blocks:
      p_correct   (runs, trials)         expected accuracy on the stimulus shown
      progress    (runs, trials, modes)  learned fraction of the target's component in
                                         each mode (1 = fully learned); NaN where the
                                         current task has no component in that mode
      weights     (runs, trials, modes)  kernel weights in use on each trial
      block       (trials,)              block index of each trial
    """
    rng = np.random.default_rng(seed)
    n, E = design.n_stim, design.n_modes
    P = design.projectors
    eig_scale = n / design.mode_dims
    w0 = np.asarray(w0, dtype=float)

    n_trials_block = n * n_reps
    T = n_trials_block * len(block_tasks)
    p_correct = np.zeros((n_runs, T))
    progress = np.full((n_runs, T, E), np.nan)
    weights = np.zeros((n_runs, T, E))
    block = np.repeat(np.arange(len(block_tasks)), n_trials_block)

    w = np.tile(w0, (n_runs, 1))
    y = np.zeros((n_runs, n))
    runs = np.arange(n_runs)
    t = 0
    for b, task in enumerate(block_tasks):
        ystar = design.task_labels[task]
        target_parts = P @ ystar                          # (modes, n): P_e y*
        target_power = np.einsum("en,en->e", target_parts, target_parts)
        has_target = target_power > 1e-12
        if b > 0 and reset_between_blocks:
            y[:] = 0.0
        seqs = np.stack([make_trial_sequence(n, n_reps, rng) for _ in runs])
        for k in range(n_trials_block):
            s = seqs[:, k]
            ys = y[runs, s]
            p_correct[:, t] = 1.0 / (1.0 + np.exp(-beta * ys * ystar[s]))
            progress[:, t, has_target] = (y @ target_parts[has_target].T) / target_power[has_target]
            weights[:, t] = w

            # Kernel column K[:, s] for each run's own weights
            col = np.einsum("re,enr->rn", w * eig_scale, P[:, :, s])
            err = ystar[s] - ys
            y += lr * err[:, None] * col - forget * y

            if adapt_rate > 0:
                shares = design.mode_shares(y)
                ok = ~np.isnan(shares[:, 0])              # skip until something is learned
                w[ok] = (1 - adapt_rate - relax_rate) * w[ok] + adapt_rate * shares[ok] + relax_rate * w0
            t += 1

    return {"p_correct": p_correct, "progress": progress, "weights": weights, "block": block}
