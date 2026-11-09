"""Envelope models and multiple testing (paper Sec. III-F, calibration).

Counts: beta-binomial, fitted by the method of moments (binomial when the calibration
data show no overdispersion). The p-value of an observed count is a parametric
bootstrap that refits the model on every resample, so it accounts for the error of
the fit: resample N calibration counts and one new count from the fitted model, refit
on the N counts, and compare the new count's plug-in p-value with the observed one.

Means: Student-t prediction, (tau - mean) / (sd * sqrt(1 + 1/N)) with N - 1 degrees of
freedom, one-sided.

Holm's step-down procedure holds the family-wise rate of a run's tests.
"""
import numpy as np
from scipy import stats as ss


def fit_betabinom(counts, n):
    """Method-of-moments fit. Returns (a, b) or ('binom', p) without overdispersion."""
    x = np.asarray(counts, dtype=np.float64)
    m = x.mean() / n
    if m <= 0.0:
        return ("zero", 0.0)
    if m >= 1.0:
        return ("binom", 1.0)
    v = x.var(ddof=1) if len(x) > 1 else 0.0
    vb = n * m * (1.0 - m)
    if v <= vb:
        return ("binom", m)
    rho = (v / vb - 1.0) / (n - 1.0)
    rho = min(rho, 1.0 - 1e-9)
    s = (1.0 - rho) / rho
    return ("bb", (m * s, (1.0 - m) * s))


def _sf_ge(c, n, model):
    """P[X >= c] under a fitted model."""
    kind, par = model
    if kind == "zero":
        return 1.0 if c <= 0 else 0.0
    if kind == "binom":
        return float(ss.binom.sf(c - 1, n, par))
    a, b = par
    return float(ss.betabinom.sf(c - 1, n, a, b))


def _draw(model, n, size, rng):
    kind, par = model
    if kind == "zero":
        return np.zeros(size, dtype=np.int64)
    if kind == "binom":
        return rng.binomial(n, par, size=size)
    a, b = par
    return rng.binomial(n, rng.beta(a, b, size=size))


def count_pvalue(obs, calib, n, n_boot, rng):
    """Bootstrap p-value (with refit) that a healthy run gives a count >= obs."""
    calib = np.asarray(calib)
    N = len(calib)
    model = fit_betabinom(calib, n)
    p_obs = _sf_ge(obs, n, model)
    if model[0] == "zero":
        raise ValueError("all calibration counts are zero: the statistic is exact")
    sims = _draw(model, n, (n_boot, N + 1), rng)
    hits = 0
    for row in sims:
        m_b = fit_betabinom(row[:N], n)
        if _sf_ge(int(row[N]), n, m_b) <= p_obs:
            hits += 1
    return (hits + 1) / (n_boot + 1), p_obs, model


def mean_pvalue(obs, calib, lower=False, floor=0.0):
    """Student-t prediction p-value; upper tail unless lower=True.

    floor: the resolution of the statistic. The spread is max(sd, floor), so a
    calibration sample whose values coincide does not claim a spread finer than the
    statistic can resolve."""
    x = np.asarray(calib, dtype=np.float64)
    N = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    sd = max(sd, floor)
    if sd == 0.0:
        bad = obs < m if lower else obs > m
        return 0.0 if bad else 1.0
    t = (obs - m) / (sd * np.sqrt(1.0 + 1.0 / N))
    return float(ss.t.cdf(t, N - 1) if lower else ss.t.sf(t, N - 1))


def holm(pvalues, alpha_fw):
    """Holm step-down. pvalues: dict key -> p. Returns dict key -> level at which it was
    rejected (only rejected keys)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    for i, (key, p) in enumerate(items, start=1):
        level = alpha_fw / (m - i + 1)
        if p <= level:
            out[key] = level
        else:
            break
    return out
