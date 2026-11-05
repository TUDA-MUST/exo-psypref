"""PsyPref-RT: learner A with a joint likelihood of choice and response time (our design notes, "RT. Response time, isolated").

**The model.** A's likelihood (``smc.per_duel_log_likelihood``) of an answer c is

    p_A(c | x) = lam/2 + (1 - lam) Phi(+-Delta / ell^2),   Delta = 1/2 (|z_B - z*|^2 - |z_A - z*|^2).

PsyPref-RT keeps it as the choice part and adds the response time r through the DDM of
:mod:`prompref.user.ddm`, at decision time tau = r - t0:

    p(c, r | x) = h(tau | 0) [ lam/2 g(tau) + (1 - lam) Phi(+-Delta / ell^2) T(v, tau) ],
    v = (Delta / ell^2) / (CHOICE_SCALE a),   T(v, tau) = cosh(v a / 2) exp(-v^2 tau / 2),

where h(tau | 0) is PyDDM's zero-drift first-passage density, T the drift's exact factor on the
density given the choice, and g(tau) = mean_j T(v_j, tau) the marginal factor of the lapse's
response time. The
drift is tied to A's own probit argument Delta / ell^2: one ell scales both the choice and the speed.

- **The choice part is A's exactly.** Since the integral of h(tau | 0) T(v, tau) over tau is 1 for
  every v, and so is that of h(tau | 0) g(tau), integrating r out gives p_A(c | x): the response
  time adds information, never a different choice model. A's choice model is misspecified for a DDM
  simulated user (probit against logistic, matched on decisiveness) in exactly the way it is for S5a's
  logistic simulated user, in both learners.
- **What PsyPref-RT is given:** the DDM's constants (a, t0, the choice scale) and the marginal, as
  the simulated simulated user has them. Only z*, ell and lambda are inferred, as in A. A real study would
  estimate a and t0; giving them is the upper bound of what response time can add.
- **Everything else is A's:** the prior, the particles and their generator, the moves, the EIG
  duels and probes (scored on the choice, whose predictive is A's), the trust region and the
  delivered curve. A missing response time (``nan``) makes that answer's term A's bit for bit, so
  PsyPref-RT given no response time is A (tested on a whole session).

``smc.py`` and ``psychometric.py`` are gated and untouched; this module subclasses them, as
:mod:`prompref.experiment.stress` does.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_expit, log_ndtr

from ..learning.psychometric import _STREAM_SALT, PsychometricLearner
from ..learning.smc import IdealPointPrior, SMCPosterior, log_likelihood, log_prior, per_duel_log_likelihood
from ..user.ddm import DDMParams, log_tilt, log_zero_drift_density


def rt_answer_terms(tau: float, params: DDMParams) -> tuple[float, float]:
    """``(log h(tau | 0), log g(tau) / h(tau | 0))`` of one answer: the parts that do not depend on
    the particle."""
    return log_zero_drift_density(tau, params.boundary), params.log_marginal_factor(tau)


def rt_per_duel_log_likelihood(x: np.ndarray, z_a: np.ndarray, z_b: np.ndarray, prefers_a, tau,
                               params: DDMParams, terms=None) -> np.ndarray:
    """PsyPref-RT's log-likelihood of each of ``n`` answers under each of ``N`` particles, shape
    ``(N, n)`` (module docstring). ``tau`` holds the decision times (response time minus t0); an
    answer whose ``tau`` is nan gets A's term, computed by A's own function. ``terms`` are the
    answers' :func:`rt_answer_terms`, recomputed when not given."""
    x = np.atleast_2d(x)
    z_a, z_b = np.atleast_2d(z_a), np.atleast_2d(z_b)
    prefers_a = np.atleast_1d(np.asarray(prefers_a, dtype=bool))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    finite = np.isfinite(tau)
    if not finite.any():
        return per_duel_log_likelihood(x, z_a, z_b, prefers_a)
    seen, unseen = np.flatnonzero(finite), np.flatnonzero(~finite)
    out = np.empty((x.shape[0], tau.size))
    if unseen.size:
        out[:, unseen] = per_duel_log_likelihood(x, z_a[unseen], z_b[unseen], prefers_a[unseen])
    if terms is None:
        terms = [rt_answer_terms(float(tau[i]), params) if np.isfinite(tau[i]) else (np.nan, np.nan)
                 for i in range(tau.size)]
    d = z_a.shape[1]
    z_star, log_ell, t = x[:, :d], x[:, d], x[:, d + 1]
    za, zb, pa = z_a[seen], z_b[seen], prefers_a[seen]
    delta = (z_star @ (za - zb).T
             - 0.5 * (np.sum(za * za, axis=1) - np.sum(zb * zb, axis=1))[None, :])
    arg = delta * np.exp(-2.0 * log_ell)[:, None]
    signed = np.where(pa[None, :], arg, -arg)
    log_h0 = np.array([terms[i][0] for i in seen], dtype=float)
    log_g = np.array([terms[i][1] for i in seen], dtype=float)
    lapse = (log_expit(t) - np.log(2.0))[:, None] + log_g[None, :]
    attentive = (log_expit(-t)[:, None] + log_ndtr(signed)
                 + log_tilt(params.drift_of_probit(arg), tau[seen][None, :], params.boundary))
    out[:, seen] = log_h0[None, :] + np.logaddexp(lapse, attentive)
    return out


def rt_log_likelihood(x, z_a, z_b, prefers_a, tau, params: DDMParams, terms=None) -> np.ndarray:
    """``(N,)``: the sum over answers of :func:`rt_per_duel_log_likelihood`."""
    return np.sum(rt_per_duel_log_likelihood(x, z_a, z_b, prefers_a, tau, params, terms), axis=1)


class RTPosterior(SMCPosterior):
    """A's SMC posterior (same particles, prior, generator and moves) with the joint likelihood.
    ``update`` takes the answer's decision time ``tau`` (nan: no response time, A's term)."""

    def __init__(self, d: int, rng: np.random.Generator, params: DDMParams,
                 n_particles: int = 4000, prior: IdealPointPrior = IdealPointPrior(),
                 n_moves: int | None = None, ess_fraction: float = 0.5) -> None:
        super().__init__(d, rng, n_particles=n_particles, prior=prior, n_moves=n_moves,
                         ess_fraction=ess_fraction)
        self.params = params
        self._tau: list[float] = []
        self._terms: list[tuple[float, float]] = []

    def log_target(self, x: np.ndarray) -> np.ndarray:
        lp = log_prior(x, self.d, self.prior)
        if self._prefers_a:
            if np.all(np.isnan(self._tau)):
                # No response time at all: A's own expression, bit for bit.
                return lp + log_likelihood(x, np.array(self._z_a), np.array(self._z_b),
                                           np.array(self._prefers_a))
            lp = lp + rt_log_likelihood(x, np.array(self._z_a), np.array(self._z_b),
                                        np.array(self._prefers_a), np.array(self._tau),
                                        self.params, self._terms)
        return lp

    def update(self, z_a: np.ndarray, z_b: np.ndarray, prefers_a: bool,
               tau: float = float("nan")) -> None:
        z_a, z_b = np.asarray(z_a, dtype=float), np.asarray(z_b, dtype=float)
        if z_a.shape != (self.d,) or z_b.shape != (self.d,):
            raise ValueError(f"expected z of shape ({self.d},), got {z_a.shape} and "
                             f"{z_b.shape}")
        tau = float(tau)
        if np.isfinite(tau) and tau <= 0.0:
            raise ValueError(f"a decision time must be positive, got {tau}")
        terms = rt_answer_terms(tau, self.params) if np.isfinite(tau) else (np.nan, np.nan)
        self._z_a.append(z_a)
        self._z_b.append(z_b)
        self._prefers_a.append(bool(prefers_a))
        self._tau.append(tau)
        self._terms.append(terms)
        if np.isfinite(tau):
            self.log_w += rt_log_likelihood(self.x, z_a[None], z_b[None], [bool(prefers_a)], [tau],
                                            self.params, [terms])
        else:
            self.log_w += log_likelihood(self.x, z_a[None], z_b[None], [bool(prefers_a)])
        if self.ess < self.ess_fraction * self.n:
            self._resample_move()


class RTLearner(PsychometricLearner):
    """PsyPref-RT: learner A (EIG duels) with :class:`RTPosterior`. The session binds the simulated user
    (:meth:`bind_response_times`); each update then reads the simulated user's ``last_response_time``, the
    response time of the answer it is given. Unbound, every answer carries no response time."""

    def __init__(self, design, seed: int = 0, *, params: DDMParams, n_particles: int = 4000,
                 prior: IdealPointPrior = IdealPointPrior(), **kwargs) -> None:
        if "duels" in kwargs:
            raise ValueError("PsyPref-RT is A with EIG duels")
        super().__init__(design, seed, duels="eig", n_particles=n_particles, prior=prior, **kwargs)
        self.params = params
        # The parent's posterior, rebuilt from an identical generator, with the joint likelihood.
        self.posterior = RTPosterior(self.space.d,
                                     np.random.default_rng([int(seed), _STREAM_SALT, 1]),
                                     params, n_particles=n_particles, prior=self.prior)
        self._rt_source = None
        self._tau_trace: list[float] = []

    def bind_response_times(self, source) -> None:
        """``source.last_response_time`` (s) is read at every update."""
        self._rt_source = source

    def _decision_time(self) -> float:
        if self._rt_source is None:
            return float("nan")
        rt = getattr(self._rt_source, "last_response_time", None)
        if rt is None:
            return float("nan")
        return float(rt) - self.params.t0

    def update(self, theta_a: np.ndarray, theta_b: np.ndarray, response: int) -> None:
        """A's ``update`` with the answer's decision time passed to the posterior."""
        tau = self._decision_time()
        z_a, z_b = self.space.to_z(theta_a), self.space.to_z(theta_b)
        predicted_p_a = float(self.posterior.weights() @ np.exp(
            log_likelihood(self.posterior.x, z_a[None], z_b[None], [True])))
        self._duel_trace.append([self._pending_multiple,
                                 float(np.linalg.norm(z_a - z_b)), predicted_p_a])
        self._pending_multiple = None
        if response == 2:
            self.n_no_preference += 1
        elif response in (0, 1):
            self.posterior.update(z_a, z_b, prefers_a=(response == 0), tau=tau)
            self._tau_trace.append(tau)
        else:
            raise ValueError(f"response must be 0, 1 or 2, got {response!r}")
        self._ell_trace.append(self.posterior.ell_mean())
        self._lambda_trace.append(self.posterior.lambda_mean())
        if self.prior.nu is not None:
            self._kappa_trace.append(self.posterior.kappa_mean())

    def posterior_summary(self) -> dict:
        summary = super().posterior_summary()
        if self._rt_source is not None:
            summary["extras"]["rt_n_used"] = int(np.sum(np.isfinite(self._tau_trace)))
        return summary
