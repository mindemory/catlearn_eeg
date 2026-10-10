"""Statistics for small groups: permutation tests, multiple-comparison adjustment, mean +- SEM.

The permutation tests draw from one generator seeded with params.SEED, so a script gives the
same p values on every run (as long as it runs its tests in the same order).
"""

import warnings

import numpy as np

from params import N_PERM, SEED

RNG = np.random.default_rng(SEED)


def perm_diff(a, b, n=N_PERM):
    """Mean(a) - mean(b) and its two-sided permutation p (group labels shuffled)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    null = np.empty(n)
    for i in range(n):
        p = RNG.permutation(pooled)
        null[i] = p[:len(a)].mean() - p[len(a):].mean()
    return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n + 1)


def sign_flip(x, n=N_PERM):
    """Mean(x) and its two-sided sign-flip permutation p (paired differences)."""
    x = np.asarray(x, float)
    obs = x.mean()
    null = (RNG.choice([-1, 1], size=(n, len(x))) * x).mean(axis=1)
    return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n + 1)


def bh(p):
    """Benjamini-Hochberg adjusted q values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def nanmean(x, axis=None):
    """np.nanmean without the warning for all-NaN slices (they stay NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=axis)


def mean_sem(rows):
    """Mean and SEM over participants (rows), per column, ignoring NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(rows, 0)
        s = np.nanstd(rows, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(rows), 0))
    return m, s


def _ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def _residual(y, z):
    """y with a linear fit on z (and an intercept) removed."""
    Z = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta


def perm_corr(x, y, covariate=None, n=N_PERM):
    """Spearman correlation of x and y (partial on `covariate` if given: ranks residualised on
    its ranks) and its two-sided permutation p (y shuffled). NaN pairs are dropped."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~np.isnan(x) & ~np.isnan(y)
    z = None
    if covariate is not None:
        z = np.asarray(covariate, float)
        keep &= ~np.isnan(z)
    x, y = _ranks(x[keep]), _ranks(y[keep])
    if z is not None:
        z = _ranks(z[keep])
        x, y = _residual(x, z), _residual(y, z)
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan, int(len(x))
    r = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, RNG.permutation(y))[0, 1] for _ in range(n)])
    return r, (np.sum(np.abs(null) >= abs(r) - 1e-12) + 1) / (n + 1), int(len(x))
