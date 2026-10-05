"""Single-season occupancy models for unlearning audits.

Data: Y is an (n_facts, K) 0/1 matrix. Y[i, j] = 1 if probe type j detected fact i.
Each fact is a "site", each probe type is one "visit".

    z_i ~ Bernoulli(psi)                      fact i still stored
    y_ij | z_i = 1 ~ Bernoulli(p_ij)          probe j detects a stored fact
    y_ij | z_i = 0 ~ Bernoulli(f_j)           false positive (f_j fixed, measured on the retain-only model)

Models:
    "M0"  logit p_ij = a_j                    (probe effects only; MacKenzie et al. 2002)
    "Mh"  logit p_ij = a_j + sigma * e_i      (fact random effect, e_i ~ N(0, 1); Gauss-Hermite)
    "Mh2" logit p_ij = a_j + b * c_i          (two latent fact classes, c_i ~ Bernoulli(w))

Mh is the pre-registered primary model; M0 and Mh2 are sensitivity checks.

Estimand ("detectably stored"): a stored fact must be detectable above noise. Without this, a "stored" class whose
detection rates equal the false-positive rates is indistinguishable from "absent", and psi is not identified when
little is stored (found in simulation, 2026-10-05). Constraint: mean_j p_j - mean_j f_j >= MIN_EXCESS, where p_j is
the detection probability of the baseline stored class (median fact for Mh, lower class for Mh2).
Under heterogeneity, psi is only weakly identified (Link 2003), which is why all three are reported.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, log_expit

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)   # probabilists' Hermite: weight exp(-x^2/2)
GH_LOGW = np.log(GH_W / GH_W.sum())

MODELS = ("M0", "Mh", "Mh2")
MIN_EXCESS = 0.10
PENALTY = 1e4


def _n_params(model, K):
    return 1 + K + {"M0": 0, "Mh": 1, "Mh2": 2}[model]


def _loglik_stored(Y, a, model, extra):
    """log P(y_i | z_i = 1) for every fact, shape (n,)."""
    if model == "M0":
        eta = a[None, :]
        return (Y * log_expit(eta) + (1 - Y) * log_expit(-eta)).sum(1)
    if model == "Mh":
        sigma = np.exp(extra[0])
        eta = a[None, None, :] + sigma * GH_X[None, :, None]           # (1, Q, K)
        ll = (Y[:, None, :] * log_expit(eta) + (1 - Y[:, None, :]) * log_expit(-eta)).sum(2)  # (n, Q)
        return logsumexp(ll + GH_LOGW[None, :], axis=1)
    if model == "Mh2":
        b, w = extra[0], expit(extra[1])
        out = []
        for shift, lw in ((0.0, np.log1p(-w + 1e-12)), (b, np.log(w + 1e-12))):
            eta = a[None, :] + shift
            out.append((Y * log_expit(eta) + (1 - Y) * log_expit(-eta)).sum(1) + lw)
        return np.logaddexp(*out)
    raise ValueError(model)


def _loglik_absent(Y, f):
    f = np.clip(np.asarray(f, float), 1e-6, 1 - 1e-6)
    return (Y * np.log(f) + (1 - Y) * np.log1p(-f)).sum(1)


def negloglik(theta, Y, f, model, min_excess=MIN_EXCESS):
    K = Y.shape[1]
    lpsi, a, extra = theta[0], theta[1:1 + K], theta[1 + K:]
    l1 = log_expit(lpsi) + _loglik_stored(Y, a, model, extra)
    l0 = log_expit(-lpsi) + _loglik_absent(Y, f)
    gap = min_excess - (expit(a).mean() - np.mean(f))
    return -np.logaddexp(l1, l0).sum() + PENALTY * max(gap, 0.0) ** 2


def fit(Y, f=None, model="Mh", n_starts=6, seed=0, min_excess=MIN_EXCESS):
    """Maximum likelihood fit. Returns dict with psi, p (per probe, at e=0), sigma/extra, nll, aic."""
    Y = np.asarray(Y, float)
    n, K = Y.shape
    f = np.zeros(K) if f is None else np.asarray(f, float)
    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_starts):
        th = np.concatenate([[rng.normal(0, 1.5)], rng.normal(0, 1.5, K),
                             {"M0": [], "Mh": [rng.normal(0, 0.5)], "Mh2": [rng.normal(2, 1), rng.normal(0, 1)]}[model]])
        bounds = [(-12, 12)] + [(-12, 12)] * K + {"M0": [], "Mh": [(-4, 2.5)], "Mh2": [(0, 12), (-8, 8)]}[model]
        r = minimize(negloglik, th, args=(Y, f, model, min_excess), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    return {"model": model, "psi": float(expit(th[0])), "a": th[1:1 + K], "p": expit(th[1:1 + K]),
            "extra": th[1 + K:], "nll": float(best.fun), "aic": float(2 * best.fun + 2 * _n_params(model, K)),
            "theta": th, "converged": bool(best.success)}


def _boot_one(Y, f, model, idx, b):
    return fit(Y[idx], f, model, n_starts=2, seed=b)["psi"]


def bootstrap_ci(Y, f=None, model="Mh", B=300, alpha=0.05, seed=0, n_jobs=1):
    """Nonparametric bootstrap over facts. Returns (lo, hi, draws). n_jobs=-1 uses all CPU cores (joblib)."""
    Y = np.asarray(Y, float)
    rng = np.random.default_rng(seed)
    idxs = [rng.integers(0, len(Y), len(Y)) for _ in range(B)]
    if n_jobs == 1:
        draws = [_boot_one(Y, f, model, idx, b) for b, idx in enumerate(idxs)]
    else:
        from joblib import Parallel, delayed
        draws = Parallel(n_jobs=n_jobs)(delayed(_boot_one)(Y, f, model, idx, b) for b, idx in enumerate(idxs))
    draws = np.array(draws)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)), draws


def simulate(n, psi, a, f=None, sigma=0.0, seed=0):
    """Draw a detection matrix from the Mh model (sigma = 0 gives M0)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    K = len(a)
    f = np.zeros(K) if f is None else np.asarray(f, float)
    z = rng.random(n) < psi
    e = rng.normal(size=(n, 1))
    p = expit(a[None, :] + sigma * e)
    Y = np.where(z[:, None], rng.random((n, K)) < p, rng.random((n, K)) < f[None, :])
    return Y.astype(int), z


def _gof_one(fitres, n, f, sim_seed, b, K):
    Yb = _sim_from_fit(fitres, n, f, sim_seed)
    rb = fit(Yb, f, fitres["model"], n_starts=2, seed=b)
    exp = np.maximum(np.bincount(_sim_from_fit(rb, n * 20, f, 0).sum(1), minlength=K + 1) / 20.0, 0.5)
    obs = np.bincount(Yb.sum(1), minlength=K + 1)
    return float(((obs - exp) ** 2 / exp).sum())


def gof_count_test(Y, fitres, f=None, B=200, seed=0, n_jobs=1):
    """Parametric-bootstrap goodness of fit on the distribution of detection counts (0..K).

    Statistic: chi-square between observed and expected counts of facts detected by exactly k probes.
    Returns the bootstrap p-value (small = misfit)."""
    Y = np.asarray(Y, int)
    n, K = Y.shape
    f = np.zeros(K) if f is None else np.asarray(f, float)

    def expected_counts(res):
        sims = [_sim_from_fit(res, n * 20, f, s) for s in range(1)]
        h = np.bincount(sims[0].sum(1), minlength=K + 1) / 20.0
        return np.maximum(h, 0.5)

    def chi2(Ym, exp):
        obs = np.bincount(Ym.sum(1), minlength=K + 1)
        return float(((obs - exp) ** 2 / exp).sum())

    rng = np.random.default_rng(seed)
    exp = expected_counts(fitres)
    t_obs = chi2(Y, exp)
    seeds = [int(rng.integers(1 << 30)) for _ in range(B)]
    if n_jobs == 1:
        t_b = [_gof_one(fitres, n, f, sd, b, K) for b, sd in enumerate(seeds)]
    else:
        from joblib import Parallel, delayed
        t_b = Parallel(n_jobs=n_jobs)(delayed(_gof_one)(fitres, n, f, sd, b, K) for b, sd in enumerate(seeds))
    return float((np.sum(np.array(t_b) >= t_obs) + 1) / (B + 1))


def _sim_from_fit(res, n, f, seed):
    rng = np.random.default_rng(seed)
    K = len(res["a"])
    z = rng.random(n) < res["psi"]
    a = res["a"][None, :]
    if res["model"] == "M0":
        eta = np.repeat(a, n, 0)
    elif res["model"] == "Mh":
        eta = a + np.exp(res["extra"][0]) * rng.normal(size=(n, 1))
    else:
        c = rng.random((n, 1)) < expit(res["extra"][1])
        eta = a + res["extra"][0] * c
    Y = np.where(z[:, None], rng.random((n, K)) < expit(eta), rng.random((n, K)) < f[None, :])
    return Y.astype(int)


def naive_estimates(Y, f=None, single_col=0):
    """Estimators used in practice: one probe's detection rate, and 'detected by any probe'."""
    Y = np.asarray(Y, int)
    return {"single_probe": float(Y[:, single_col].mean()), "any_of_K": float(Y.max(1).mean())}
