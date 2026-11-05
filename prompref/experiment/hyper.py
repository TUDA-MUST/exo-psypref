"""PsyPref-hyper: learner A with a population hyperprior on log ell and lambda (our design notes, "H2. Population hyperprior for s and lambda").

**What is planned.** PsyPref-hyper's priors on log ell and lambda come from the other folds'
sessions in the same condition, leave-fold-out: an empirical-Bayes fit of a Normal on log ell and a
Beta on lambda, pooled over those users' answers and their particle posteriors. Only observable
answers are used. Everything else is A's.

**Why a first pass.** The stored A rows (e-main at lambda = 0.1, e-lapse at 0.3) hold each
session's answers and posterior means, but neither the pairs shown in z nor the particles, so the
marginal likelihood of a user's answers under another prior cannot be computed from them. A's
sessions on the 200 population users are therefore rerun with ``hyper_log`` (block e-hyper-1): the
logging draws no random numbers and only reads the final posterior, so every rerun must reproduce
its stored row in every stored field (``scripts/icorr_hyper_fit.py`` checks this and refuses to fit
otherwise). Each row then carries :data:`HYPER_PARTICLES` draws of (log ell, logit lambda) from the
final posterior (k = 40), taken by systematic resampling with the fixed offset 1/2.

**The fit.** A user's answers y_u were fitted under A's prior pi_0 (log ell ~ N(0, 0.5^2), lambda ~
Beta(1, 9)). Under another prior pi_eta their marginal likelihood is, exactly,

    p(y_u | eta) / p(y_u | pi_0) = E_{posterior_0}[ pi_eta(ell, lambda) / pi_0(ell, lambda) ],

which the particles estimate by a mean (z* integrates out: both priors share its prior). The
empirical-Bayes estimate maximises the sum of the logs over the training users (type-II maximum
likelihood) over eta = (mu, sigma, Beta mean m, Beta concentration kappa = a + b), by L-BFGS-B from
several starts. Bounds keep the importance sampling meaningful: sigma >= :data:`SD_MIN` and kappa <=
:data:`KAPPA_MAX` (a Beta SD of about 0.015 at m = 0.3); a fit at a bound is reported as such. The
users' importance effective sample sizes at the optimum are reported with each fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, log_expit, logit, logsumexp

from ..learning.smc import IdealPointPrior, SMCPosterior

#: Final-posterior draws logged per first-pass session.
HYPER_PARTICLES = 2000
#: Bounds of the fit (module docstring).
SD_MIN, SD_MAX = 0.02, 3.0
KAPPA_MIN, KAPPA_MAX = 2.0, 1000.0
MEAN_MIN, MEAN_MAX = 1e-3, 0.999
MU_MIN, MU_MAX = -3.0, 3.0
HYPER_DIR = "hyper"
HYPER_SCHEMA = "hyper-prior-v1"
#: Conditions of block e-hyper: the main condition and lambda = 0.3.
HYPER_LAPSES = (0.1, 0.3)


def lapse_label(lapse: float) -> str:
    return f"{float(lapse):.1f}"


def posterior_sample(posterior: SMCPosterior, m: int = HYPER_PARTICLES) -> dict:
    """``m`` equally weighted draws of (log ell, logit lambda) from the weighted particles, by
    systematic resampling with offset 1/2. Reads the posterior; draws no random numbers."""
    w = posterior.weights()
    positions = (0.5 + np.arange(int(m))) / int(m)
    idx = np.minimum(np.searchsorted(np.cumsum(w), positions), posterior.n - 1)
    d = posterior.d
    return dict(log_ell=[float(v) for v in posterior.x[idx, d]],
                logit_lam=[float(v) for v in posterior.x[idx, d + 1]])


def log_normal(x, mu: float, sd: float):
    x = np.asarray(x, dtype=float)
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2.0 * np.pi)


def log_beta_of_logit(t, a: float, b: float):
    """The Beta(a, b) log density of lambda = expit(t), in lambda (no Jacobian: it cancels in every
    ratio of two priors on the same coordinates)."""
    t = np.asarray(t, dtype=float)
    return (a - 1.0) * log_expit(t) + (b - 1.0) * log_expit(-t) - betaln(a, b)


def log_prior_ratio(log_ell, logit_lam, eta: dict, base: IdealPointPrior = IdealPointPrior()):
    """``log pi_eta - log pi_0`` at each draw."""
    return (log_normal(log_ell, eta["mu"], eta["sd"])
            + log_beta_of_logit(logit_lam, eta["lam_a"], eta["lam_b"])
            - log_normal(log_ell, base.log_ell0, base.ell_sd)
            - log_beta_of_logit(logit_lam, base.lam_a, base.lam_b))


def user_log_ratios(log_ell: np.ndarray, logit_lam: np.ndarray, eta: dict,
                    base: IdealPointPrior = IdealPointPrior()) -> np.ndarray:
    """``(U,)``: log p(y_u | eta) - log p(y_u | pi_0), each the log of a particle mean."""
    r = log_prior_ratio(log_ell, logit_lam, eta, base)
    return logsumexp(r, axis=1) - np.log(r.shape[1])


def importance_ess(log_ell, logit_lam, eta: dict, base: IdealPointPrior = IdealPointPrior()) -> np.ndarray:
    """``(U,)``: each user's effective sample size of the weights pi_eta / pi_0, as a share of
    the draws."""
    r = log_prior_ratio(log_ell, logit_lam, eta, base)
    w = np.exp(r - r.max(axis=1, keepdims=True))
    return (w.sum(axis=1) ** 2 / (w * w).sum(axis=1)) / r.shape[1]


def eta_of(theta) -> dict:
    """(mu, log sigma, logit m, log kappa) -> {mu, sd, lam_a, lam_b}."""
    mu, log_sd, logit_m, log_kappa = (float(v) for v in theta)
    m, kappa = float(expit(logit_m)), float(np.exp(log_kappa))
    return dict(mu=mu, sd=float(np.exp(log_sd)), lam_a=m * kappa, lam_b=(1.0 - m) * kappa)


def theta_of(eta: dict) -> np.ndarray:
    kappa = eta["lam_a"] + eta["lam_b"]
    return np.array([eta["mu"], np.log(eta["sd"]), logit(eta["lam_a"] / kappa), np.log(kappa)])


BOUNDS = [(MU_MIN, MU_MAX), (np.log(SD_MIN), np.log(SD_MAX)),
          (float(logit(MEAN_MIN)), float(logit(MEAN_MAX))), (np.log(KAPPA_MIN), np.log(KAPPA_MAX))]


@dataclass(frozen=True)
class HyperFit:
    mu: float
    sd: float
    lam_a: float
    lam_b: float
    log_marginal_ratio: float
    n_users: int
    at_bound: tuple
    ess_quantiles: tuple
    starts: tuple

    def prior_fields(self) -> dict:
        """The point fields of PsyPref-hyper (``Phase5Point.hyper_*``)."""
        return dict(hyper_log_ell0=self.mu, hyper_ell_sd=self.sd, hyper_lam_a=self.lam_a,
                    hyper_lam_b=self.lam_b)

    def to_dict(self) -> dict:
        kappa = self.lam_a + self.lam_b
        m = self.lam_a / kappa
        return dict(mu=self.mu, sd=self.sd, lam_a=self.lam_a, lam_b=self.lam_b, lam_mean=m,
                    lam_sd=float(np.sqrt(m * (1.0 - m) / (kappa + 1.0))), kappa=kappa,
                    log_marginal_ratio=self.log_marginal_ratio, n_users=self.n_users,
                    at_bound=list(self.at_bound), ess_quantiles=list(self.ess_quantiles),
                    starts=[list(s) for s in self.starts])


def _starts(log_ell: np.ndarray, logit_lam: np.ndarray, base: IdealPointPrior) -> list[dict]:
    """The prior A used, the pooled posterior's moments, and two spreads of the latter."""
    lam = expit(logit_lam)
    m = float(np.clip(lam.mean(), 0.01, 0.9))
    v = float(lam.var())
    kappa = float(np.clip(m * (1.0 - m) / max(v, 1e-6) - 1.0, KAPPA_MIN * 1.5, KAPPA_MAX / 1.5))
    mu = float(np.clip(log_ell.mean(), MU_MIN + 0.1, MU_MAX - 0.1))
    sd = float(np.clip(log_ell.std(), SD_MIN * 1.5, SD_MAX / 1.5))
    pooled = dict(mu=mu, sd=sd, lam_a=m * kappa, lam_b=(1.0 - m) * kappa)
    starts = [dict(mu=base.log_ell0, sd=base.ell_sd, lam_a=base.lam_a, lam_b=base.lam_b), pooled]
    for f in (0.5, 0.25):
        starts.append(dict(mu=mu, sd=max(sd * f, SD_MIN * 1.5), lam_a=m * min(kappa / f, 900.0),
                           lam_b=(1.0 - m) * min(kappa / f, 900.0)))
    return starts


def negative_log_marginal(theta, log_ell: np.ndarray, logit_lam: np.ndarray,
                          base: IdealPointPrior = IdealPointPrior()) -> tuple[float, np.ndarray]:
    """The fit's objective, minus the sum over users of :func:`user_log_ratios`, and its exact
    gradient in theta = (mu, log sigma, logit m, log kappa): each user's gradient is the
    posterior-weighted mean (weights pi_eta / pi_0, normalised per user) of the prior's score."""
    eta = eta_of(theta)
    mu, sd, a, b = eta["mu"], eta["sd"], eta["lam_a"], eta["lam_b"]
    r = log_prior_ratio(log_ell, logit_lam, eta, base)
    top = r.max(axis=1, keepdims=True)
    e = np.exp(r - top)
    total = e.sum(axis=1, keepdims=True)
    value = float(np.sum(np.log(total[:, 0]) + top[:, 0] - np.log(r.shape[1])))
    w = e / total
    z = (log_ell - mu) / sd
    d_mu = (w * z).sum(axis=1) / sd
    d_log_sd = (w * (z * z - 1.0)).sum(axis=1)
    d_a = (w * log_expit(logit_lam)).sum(axis=1) - digamma(a) + digamma(a + b)
    d_b = (w * log_expit(-logit_lam)).sum(axis=1) - digamma(b) + digamma(a + b)
    m, kappa = a / (a + b), a + b
    d_logit_m = kappa * m * (1.0 - m) * (d_a - d_b)
    d_log_kappa = a * d_a + b * d_b
    grad = np.array([d_mu.sum(), d_log_sd.sum(), d_logit_m.sum(), d_log_kappa.sum()])
    return -value, -grad


def fit_hyperprior(log_ell, logit_lam, base: IdealPointPrior = IdealPointPrior()) -> HyperFit:
    """The empirical-Bayes Normal on log ell and Beta on lambda from ``(U, M)`` draws of the users'
    posteriors under ``base`` (module docstring)."""
    log_ell = np.asarray(log_ell, dtype=float)
    logit_lam = np.asarray(logit_lam, dtype=float)
    if log_ell.ndim != 2 or log_ell.shape != logit_lam.shape or log_ell.shape[0] < 2:
        raise ValueError(f"expected two (users, draws) arrays, got {log_ell.shape} and "
                         f"{logit_lam.shape}")
    if not (np.all(np.isfinite(log_ell)) and np.all(np.isfinite(logit_lam))):
        raise ValueError("every draw must be finite")

    def objective(theta):
        return negative_log_marginal(theta, log_ell, logit_lam, base)

    best = None
    starts = _starts(log_ell, logit_lam, base)
    log_ell = np.ascontiguousarray(log_ell)
    logit_lam = np.ascontiguousarray(logit_lam)
    for start in starts:
        x0 = np.clip(theta_of(start), [lo for lo, _ in BOUNDS], [hi for _, hi in BOUNDS])
        res = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=BOUNDS,
                       options=dict(maxiter=500, ftol=1e-13, gtol=1e-9))
        if best is None or res.fun < best.fun:
            best = res
    eta = eta_of(best.x)
    at_bound = tuple(name for (lo, hi), v, name in zip(BOUNDS, best.x, ("mu", "sd", "lam_mean",
                                                                         "kappa"))
                     if abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6)
    ess = importance_ess(log_ell, logit_lam, eta, base)
    return HyperFit(mu=eta["mu"], sd=eta["sd"], lam_a=eta["lam_a"], lam_b=eta["lam_b"],
                    log_marginal_ratio=-float(best.fun), n_users=int(log_ell.shape[0]),
                    at_bound=at_bound,
                    ess_quantiles=tuple(float(q) for q in np.quantile(ess, [0.0, 0.1, 0.5])),
                    starts=tuple(tuple(float(v) for v in theta_of(s)) for s in starts))


def hyper_prior_path(run_dir) -> Path:
    return Path(run_dir) / HYPER_DIR / "hyper_prior.json"


@lru_cache(maxsize=4)
def _load(run_dir: str) -> dict:
    path = hyper_prior_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run block e-hyper-1, then "
                                f"scripts/icorr_hyper_fit.py --run {run_dir}")
    side = json.loads(path.read_text(encoding="utf-8"))
    if side.get("schema") != HYPER_SCHEMA:
        raise ValueError(f"{path} has schema {side.get('schema')!r}, expected {HYPER_SCHEMA!r}")
    return side


def hyper_fields(run_dir: str, lapse: float, fold: str) -> dict:
    """PsyPref-hyper's point fields for the held-out ``fold`` in the condition ``lapse``."""
    side = _load(str(run_dir))
    try:
        fit = side["fits"][lapse_label(lapse)][fold]
    except KeyError:
        raise KeyError(f"{hyper_prior_path(run_dir)} holds no fit for lapse {lapse} and held-out "
                       f"fold {fold}") from None
    if fold in fit["train_folds"]:
        raise ValueError(f"the fit for {fold} was trained on it")
    return dict(hyper_log_ell0=float(fit["mu"]), hyper_ell_sd=float(fit["sd"]),
                hyper_lam_a=float(fit["lam_a"]), hyper_lam_b=float(fit["lam_b"]))
