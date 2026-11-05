"""ICORR reframe, stress tests (our design notes, "Week to Fri 30 Oct", part S, S1-S7
and PS1-PS3): PsyPref-M, the psychometric learner A
with a learned perceptual metric (S2).

**The model.** A's likelihood (``smc.per_duel_log_likelihood``) is

    P(A > B) = lam/2 + (1 - lam) Phi(Delta / ell^2),  Delta = 1/2 (|z_B - z*|^2 - |z_A - z*|^2),

with the identity metric in the whitened coordinates z. PsyPref-M replaces the identity by a metric
``M_c`` that weights the gait phases by

    w_c(phi) = exp(sum_j c_j f_j(phi)) / mean_phi exp(sum_j c_j f_j(phi)),

``f_j`` the cosine and sine of harmonics 1 and 2 (4 coefficients), with the prior
``c_j ~ N(0, 0.5^2)``. The coefficients are appended to A's particle state, which becomes
``(z*, log ell, logit lam, c_1, ..., c_4)``; everything else is A's.

**The metric is relative to A's own.** At ``w = 1`` the curve metric in z is
``G_1 = J^T J / N`` (``J`` the Jacobian of the unclipped curve in z, N = 101 phase points), which is
not the identity A uses: z whitens the population prior, not the curve RMS. The planned reduction
("with all coefficients at 0 it reduces exactly to A's likelihood") therefore fixes the metric as the
phase weighting's change *relative to* the uniform weighting, carried over to A's metric:

    M_c = G_1^{-1/2} G_w G_1^{-1/2} = I + Q^T diag(w_c - 1) Q / N,   Q = J G_1^{-1/2},

where ``G_w = J^T diag(w) J / N`` and ``Q^T Q / N = I``. The symmetric root is the one choice that
does not depend on how the z coordinates are ordered or rotated. Where A's metric is the curve
metric (``G_1 = I``), ``M_c`` is exactly the weighted curve metric ``G_w``. The mean-1 normalisation
of ``w_c`` keeps the overall scale with ``ell``, as the wearers' weights are normalised.

**Exact reduction.** ``Delta`` is computed as A's ``Delta`` plus a correction that is linear in
``w_c - 1``. At ``c = 0`` the weight is 1 exactly, the correction is 0.0 exactly, and every
likelihood value equals A's bit for bit.

**One likelihood.** The posterior update, the expected information gain that picks the duels and
the one-step predictive all use this likelihood, as A's all use A's. The acquisition criterion is
A's (the Gaussian-proxy EIG on z*, the same candidates, probes and trust region). ``smc.py`` and
``acquisition.py`` are gated and untouched; this module subclasses them, as
:mod:`prompref.experiment.review` does.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_expit, log_ndtr

from ..learning.acquisition import P_FLOOR, TRIM_QUANTILE, _log_det_cov, trim_mask
from ..learning.candidates import (candidate_duels, clip_into_region, coordinate_directions,
                                   step_region, truncate_duel)
from ..learning.psychometric import _STREAM_SALT, PROBE_LEVELS, PsychometricLearner
from ..learning.smc import BowlPrior, SMCPosterior, log_prior
from ..user.phase_weight import harmonic_basis

#: PsyPref-M's phase weighting: Fourier harmonics 1-2, 4 coefficients, prior SD 0.5.
METRIC_HARMONICS = (1, 2)
METRIC_PRIOR_SD = 0.5


def metric_features(harmonics=METRIC_HARMONICS) -> np.ndarray:
    """(101, 2 len(harmonics)) cos and sin of each harmonic on the 101-point phase grid, periodic
    (point 100 = point 0), interleaved as :func:`~prompref.user.phase_weight.harmonic_basis`."""
    distinct = harmonic_basis(harmonics)
    return np.vstack([distinct, distinct[:1]])


class PhaseMetric:
    """PsyPref-M's metric in z for one learner space (module docstring).

    ``jacobian`` is the (N, d) Jacobian of the unclipped curve in z. ``q`` is ``J G_1^{-1/2} / sqrt(N)``,
    so that ``q^T q = I`` and ``M_c = I + q^T diag(w_c - 1) q``."""

    def __init__(self, jacobian: np.ndarray, features: np.ndarray | None = None) -> None:
        jac = np.asarray(jacobian, dtype=float)
        if jac.ndim != 2 or jac.shape[0] < jac.shape[1]:
            raise ValueError(f"the Jacobian must be (points, d) with points >= d, got {jac.shape}")
        self.n_points, self.d = jac.shape
        self.jacobian = jac
        self.g1 = jac.T @ jac / self.n_points
        vals, vecs = np.linalg.eigh(self.g1)
        if vals.min() <= 1e-12 * vals.max():
            raise ValueError("the curve metric of the space is singular")
        self.g1_inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
        self.g1_sqrt = (vecs * np.sqrt(vals)) @ vecs.T
        self.q = jac @ self.g1_inv_sqrt / np.sqrt(self.n_points)
        self.features = metric_features() if features is None else np.asarray(features, float)
        if self.features.shape[0] != self.n_points:
            raise ValueError(f"features have {self.features.shape[0]} points, the curve "
                             f"{self.n_points}")
        self.n_coef = self.features.shape[1]

    @classmethod
    def from_space(cls, space) -> "PhaseMetric":
        """The metric of a ProMP learner space: ``J = Phi U diag(sqrt(eigenvalues))``, the basis
        matrix times the whitening's map from z to weights (the device clip is left out, as A's
        bowl leaves it out)."""
        basis = space.design.basis
        whitening = space.whitening
        if whitening.d != whitening.mu.size:
            raise ValueError("PsyPref-M needs a full-rank whitening (d = number of weights)")
        jac = np.asarray(basis.design, dtype=float) @ (whitening._u * whitening._scale)
        return cls(jac)

    def weights(self, coef: np.ndarray) -> np.ndarray:
        """(n, N) phase weights ``w_c``, mean 1 over the N points, for (n, n_coef) coefficients."""
        coef = np.atleast_2d(np.asarray(coef, dtype=float))
        raw = np.exp(coef @ self.features.T)
        return raw / np.mean(raw, axis=1, keepdims=True)

    def matrix(self, coef: np.ndarray) -> np.ndarray:
        """``M_c`` for one coefficient vector, (d, d)."""
        excess = self.weights(np.asarray(coef, dtype=float)[None, :])[0] - 1.0
        return np.eye(self.d) + (self.q.T * excess) @ self.q

    def weighted_g(self, coef: np.ndarray) -> np.ndarray:
        """``G_w = J^T diag(w_c) J / N``, the weighted curve metric in z, (d, d)."""
        w = self.weights(np.asarray(coef, dtype=float)[None, :])[0]
        return (self.jacobian.T * w) @ self.jacobian / self.n_points

    def correction(self, z_star: np.ndarray, coef: np.ndarray, z_a: np.ndarray,
                   z_b: np.ndarray) -> np.ndarray:
        """(n, C): ``1/2 [(z_B - z*)^T E (z_B - z*) - (z_A - z*)^T E (z_A - z*)]`` with
        ``E = q^T diag(w_c - 1) q``, per particle and duel. Exactly 0.0 where ``c = 0``."""
        excess = self.weights(coef) - 1.0                      # (n, N)
        s = z_star @ self.q.T                                   # (n, N)
        a = np.atleast_2d(z_a) @ self.q.T                       # (C, N)
        b = np.atleast_2d(z_b) @ self.q.T
        return 0.5 * (excess @ (b * b - a * a).T) - (excess * s) @ (b - a).T


def bowl_part(x: np.ndarray, d: int) -> np.ndarray:
    """A's own particle columns ``(z*, log ell, logit lam)`` as a C-contiguous (n, d + 2) array, laid
    out exactly as A's particles are, so every computation on them matches A's bit for bit."""
    return np.ascontiguousarray(np.atleast_2d(x)[:, :d + 2])


def metric_per_duel_log_likelihood(x: np.ndarray, z_a: np.ndarray, z_b: np.ndarray, prefers_a,
                                   metric: PhaseMetric) -> np.ndarray:
    """PsyPref-M's log-likelihood of each of ``n`` answers under each of ``N`` particles,
    shape ``(N, n)``. A's ``smc.per_duel_log_likelihood`` operation for operation, with
    ``Delta`` plus :meth:`PhaseMetric.correction`."""
    x = np.atleast_2d(x)
    z_a, z_b = np.atleast_2d(z_a), np.atleast_2d(z_b)
    prefers_a = np.atleast_1d(np.asarray(prefers_a, dtype=bool))
    d = z_a.shape[1]
    xa = bowl_part(x, d)
    z_star, log_ell, t = xa[:, :d], xa[:, d], xa[:, d + 1]
    delta = (z_star @ (z_a - z_b).T
             - 0.5 * (np.sum(z_a * z_a, axis=1) - np.sum(z_b * z_b, axis=1))[None, :])
    delta = delta + metric.correction(z_star, x[:, d + 2:], z_a, z_b)
    arg = delta * np.exp(-2.0 * log_ell)[:, None]
    signed = np.where(prefers_a[None, :], arg, -arg)
    log_half_lapse = (log_expit(t) - np.log(2.0))[:, None]
    log_attentive = log_expit(-t)[:, None]
    return np.logaddexp(log_half_lapse, log_attentive + log_ndtr(signed))


def metric_log_likelihood(x, z_a, z_b, prefers_a, metric: PhaseMetric) -> np.ndarray:
    """``(N,)``: the sum over answers of :func:`metric_per_duel_log_likelihood`."""
    return np.sum(metric_per_duel_log_likelihood(x, z_a, z_b, prefers_a, metric), axis=1)


def metric_expected_information_gain(x: np.ndarray, w: np.ndarray, d: int, z_a: np.ndarray,
                                     z_b: np.ndarray, metric: PhaseMetric, *,
                                     trim: float | None = None) -> np.ndarray:
    """``acquisition.expected_information_gain`` with PsyPref-M's answer probabilities: the
    Gaussian-proxy EIG on z* for each candidate pair, shape ``(C,)``."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    w = np.asarray(w, dtype=float)
    if trim is not None:
        keep = trim_mask(bowl_part(x, d), w, d, trim)
        x, w = x[keep], w[keep]
        w = w / w.sum()
    xa = bowl_part(x, d)
    zs = xa[:, :d]
    outer = (zs[:, :, None] * zs[:, None, :]).reshape(len(zs), d * d)
    probs = np.exp(metric_per_duel_log_likelihood(x, z_a, z_b, np.ones(len(np.atleast_2d(z_a)),
                                                                         dtype=bool), metric))
    weighted = w[:, None] * probs
    p = np.clip(weighted.sum(axis=0), P_FLOOR, 1.0 - P_FLOOR)
    h_prior = 0.5 * _log_det_cov(zs, outer, w[:, None])[0]
    h_a = 0.5 * _log_det_cov(zs, outer, weighted / p)
    h_b = 0.5 * _log_det_cov(zs, outer, (w[:, None] - weighted) / (1.0 - p))
    return h_prior - p * h_a - (1.0 - p) * h_b


class MetricPosterior(SMCPosterior):
    """A's SMC posterior with the metric coefficients appended to the particle state.

    The particles start as A's (the same generator and prior draws), then draw their coefficients
    from ``N(0, METRIC_PRIOR_SD^2)``. The prior adds ``-|c|^2 / (2 METRIC_PRIOR_SD^2)``; the moves
    act on all ``d + 2 + n_coef`` coordinates with A's proposal rule. ``free_metric=False`` pins
    every coefficient at 0, draws none, and moves only A's coordinates: a check that PsyPref-M is
    A in everything but the metric (its sessions then reproduce A's)."""

    def __init__(self, d: int, rng: np.random.Generator, metric: PhaseMetric,
                 n_particles: int = 4000, prior: BowlPrior = BowlPrior(),
                 n_moves: int | None = None, ess_fraction: float = 0.5,
                 free_metric: bool = True) -> None:
        super().__init__(d, rng, n_particles=n_particles, prior=prior, n_moves=n_moves,
                         ess_fraction=ess_fraction)
        if metric.d != self.d:
            raise ValueError(f"the metric is {metric.d}-dimensional, the posterior {self.d}")
        self.metric = metric
        self.free_metric = bool(free_metric)
        if self.free_metric:
            coef = METRIC_PRIOR_SD * self.rng.normal(size=(self.n, metric.n_coef))
        else:
            coef = np.zeros((self.n, metric.n_coef))
        self.x = np.column_stack([self.x, coef])

    @property
    def n_free(self) -> int:
        return self.x.shape[1] if self.free_metric else self.d + 2

    def coef(self) -> np.ndarray:
        return self.x[:, self.d + 2:]

    def log_target(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        lp = log_prior(bowl_part(x, self.d), self.d, self.prior)
        coef = x[:, self.d + 2:]
        lp = lp - 0.5 * np.sum(coef * coef, axis=1) / METRIC_PRIOR_SD**2
        if self._prefers_a:
            lp = lp + metric_log_likelihood(x, np.array(self._z_a), np.array(self._z_b),
                                            np.array(self._prefers_a), self.metric)
        return lp

    def update(self, z_a: np.ndarray, z_b: np.ndarray, prefers_a: bool) -> None:
        z_a, z_b = np.asarray(z_a, dtype=float), np.asarray(z_b, dtype=float)
        if z_a.shape != (self.d,) or z_b.shape != (self.d,):
            raise ValueError(f"expected z of shape ({self.d},), got {z_a.shape} and "
                             f"{z_b.shape}")
        self._z_a.append(z_a)
        self._z_b.append(z_b)
        self._prefers_a.append(bool(prefers_a))
        self.log_w += metric_log_likelihood(self.x, z_a[None], z_b[None], [bool(prefers_a)],
                                            self.metric)
        if self.ess < self.ess_fraction * self.n:
            self._resample_move()

    def _resample_move(self) -> None:
        w = self.weights()
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), self.n - 1)
        self.x = self.x[idx]
        self.log_w = np.zeros(self.n)
        self.n_resample += 1

        free = self.n_free
        cov = (np.cov(np.ascontiguousarray(self.x[:, :free]), rowvar=False) * (2.38**2 / free)
               + 1e-9 * np.eye(free))
        chol = np.linalg.cholesky(cov)
        current = self.log_target(self.x)
        for _ in range(self.n_moves):
            if free == self.x.shape[1]:
                proposal = self.x + self.rng.normal(size=self.x.shape) @ chol.T
            else:
                proposal = self.x.copy()
                proposal[:, :free] = (bowl_part(self.x, self.d)
                                      + self.rng.normal(size=(self.n, free)) @ chol.T)
            candidate = self.log_target(proposal)
            accept = np.log(self.rng.random(self.n)) < candidate - current
            self.x[accept] = proposal[accept]
            current[accept] = candidate[accept]
            self.acceptance.append(float(accept.mean()))

    def mean(self) -> np.ndarray:
        w = self.weights()
        return np.concatenate([w @ bowl_part(self.x, self.d), w @ self.coef()])

    def sd(self) -> np.ndarray:
        w = self.weights()
        parts = []
        for block in (bowl_part(self.x, self.d), np.ascontiguousarray(self.coef())):
            m = w @ block
            parts.append(np.sqrt(w @ (block - m) ** 2))
        return np.concatenate(parts)


class MetricLearner(PsychometricLearner):
    """PsyPref-M: learner A (EIG duels, probes, trust region, prior) with :class:`MetricPosterior`
    and its likelihood in the update, the acquisition and the one-step predictive."""

    def __init__(self, design, seed: int = 0, *, space, n_particles: int = 4000,
                 prior: BowlPrior = BowlPrior(), metric: PhaseMetric | None = None,
                 free_metric: bool = True, **kwargs) -> None:
        if "duels" in kwargs:
            raise ValueError("PsyPref-M is A with EIG duels")
        super().__init__(design, seed, duels="eig", space=space, n_particles=n_particles,
                         prior=prior, **kwargs)
        self.metric = PhaseMetric.from_space(self.space) if metric is None else metric
        # The parent's posterior, rebuilt from an identical generator, with the metric appended.
        self.posterior = MetricPosterior(self.space.d,
                                         np.random.default_rng([int(seed), _STREAM_SALT, 1]),
                                         self.metric, n_particles=n_particles, prior=self.prior,
                                         free_metric=free_metric)

    def _propose_eig(self) -> tuple[np.ndarray, np.ndarray]:
        """A's ``_propose_eig`` line for line, with :func:`metric_expected_information_gain`."""
        own, duel, post, rng = self.space, self.duel_space, self.posterior, self._duel_rng
        ell = post.ell_mean()
        z_hat = post.zstar_mean()
        z_worn = self._z_worn()
        region = step_region(z_worn, self.r_step)
        z_a = z_b = None
        if rng.random() < self.p_probe:
            if self.duel_directions == "free":
                u = rng.normal(size=own.d)
                u /= np.linalg.norm(u)
            else:
                dirs = coordinate_directions(own)
                u = dirs[int(rng.integers(len(dirs)))]
            sign = 1.0 if rng.random() < 0.5 else -1.0
            k = float(rng.choice(PROBE_LEVELS))
            c_own = duel.transfer(
                clip_into_region(own.transfer(z_hat - sign * k * ell * u, duel), duel, **region),
                own)
            z_a, z_b = truncate_duel(c_own, u, 0.5 * ell, own, duel, **region)
            if np.linalg.norm(z_a - z_b) < 1e-9:
                z_a = z_b = None
        if z_a is not None:
            kind, multiple = "probe", k
        else:
            cands = candidate_duels(bowl_part(post.x, own.d), post.weights(), own, duel, rng,
                                    ell_hat=ell, max_mult=self.max_mult,
                                    directions=self.duel_directions, **region)
            trim = TRIM_QUANTILE if self.prior.nu is not None else None
            best = int(np.argmax(metric_expected_information_gain(
                post.x, post.weights(), own.d, cands["z_a"], cands["z_b"], self.metric,
                trim=trim)))
            z_a, z_b = cands["z_a"][best], cands["z_b"][best]
            kind, multiple = "acq", float(cands["multiple"][best])
        self._pending_multiple = multiple
        self._duel_kind.append(kind)
        delta = 0.5 * (float(np.dot(z_b - z_hat, z_b - z_hat))
                       - float(np.dot(z_a - z_hat, z_a - z_hat)))
        self._duel_intensity.append(delta / (ell * ell))
        return own.to_theta(z_a), own.to_theta(z_b)

    def update(self, theta_a: np.ndarray, theta_b: np.ndarray, response: int) -> None:
        """A's ``update`` with PsyPref-M's one-step predictive."""
        z_a, z_b = self.space.to_z(theta_a), self.space.to_z(theta_b)
        predicted_p_a = float(self.posterior.weights() @ np.exp(
            metric_log_likelihood(self.posterior.x, z_a[None], z_b[None], [True], self.metric)))
        self._duel_trace.append([self._pending_multiple,
                                 float(np.linalg.norm(z_a - z_b)), predicted_p_a])
        self._pending_multiple = None
        if response == 2:
            self.n_no_preference += 1
        elif response in (0, 1):
            self.posterior.update(z_a, z_b, prefers_a=(response == 0))
        else:
            raise ValueError(f"response must be 0, 1 or 2, got {response!r}")
        self._ell_trace.append(self.posterior.ell_mean())
        self._lambda_trace.append(self.posterior.lambda_mean())
        if self.prior.nu is not None:
            self._kappa_trace.append(self.posterior.kappa_mean())

    def posterior_summary(self) -> dict:
        summary = super().posterior_summary()
        if self.posterior.free_metric:
            post = self.posterior
            w = post.weights()
            coef = np.ascontiguousarray(post.coef())
            mean = w @ coef
            summary["extras"].update(
                metric_coef_mean=[float(v) for v in mean],
                metric_coef_sd=[float(v) for v in np.sqrt(w @ (coef - mean) ** 2)],
                metric_weight_mean=[float(v) for v in w @ self.metric.weights(coef)])
        return summary
