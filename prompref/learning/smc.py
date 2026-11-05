"""Sequential Monte Carlo posterior for learner A's bowl model -- design-free.

Knows nothing about torque curves: only whitened points ``z`` and answers to "was A
preferred". That is deliberate. The Phase 3 gate checks this module against an
independently coded reference sampler on synthetic data, which is only possible if
nothing here depends on a design.

Model::

    P(A > B) = lam/2 + (1 - lam) * Phi(Delta / ell^2)
    Delta    = 1/2 (|z_B - z*|^2 - |z_A - z*|^2)

Particles live in the unconstrained coordinates ``x = (z*, log ell, logit lam)``.

Heavy tails: ``z* | kappa ~ N(0, kappa I)`` with
``kappa ~ InvGamma(nu/2, nu/2)``, so ``z*`` is marginally a multivariate t. The particles
never store kappa. The likelihood does not depend on it, so it is
integrated out exactly: the prior on ``z*`` is the t density, and every kappa readout is
Rao-Blackwellised through ``kappa | z* ~ InvGamma((nu + d)/2, (nu + |z*|^2)/2)``.

SMC rather than MCMC: the posterior is needed between every
comparison, and an answer only reweights the particles. Rerunning MCMC per answer is
what made the sibling project's learner cost a second per comparison.

**Log densities here omit normalising constants**, so they differ from ``scipy.stats``
densities by a constant. Harmless for sampling; a trap for any test that compares
absolute values rather than differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, expit, log_expit, log_ndtr, logit, polygamma

#: Rejuvenation moves per resampling. Under a heavy-tailed
#: prior the particle cloud's covariance is inflated by tail particles, so random-walk proposals
#: are too wide for the bulk (acceptance 0.09-0.18 against 0.23 under the Gaussian) and five
#: moves leave the posterior too narrow: 90 % widths 0.78-0.93 of the reference on the recovery
#: gate's t4 dataset. Ten moves pass every gate dataset.
GAUSSIAN_MOVES = 5
HEAVY_TAIL_MOVES = 10


@dataclass(frozen=True)
class BowlPrior:
    """``z* ~ N(0, scale I)``, or ``z* ~ t_nu(0, I)`` with ``nu``;
    ``log ell ~ N(log_ell0, ell_sd^2)``; ``lam ~ Beta(lam_a, lam_b)``.

    The defaults are Phase 4's prior, drawn with exactly Phase 4's random numbers.
    ``scale`` widens the Gaussian (the wide arm, ``scale=9``); a heavy-tailed prior has unit
    scale, so the two are never combined.
    """

    log_ell0: float = 0.0
    ell_sd: float = 0.5
    lam_a: float = 1.0
    lam_b: float = 9.0
    nu: float | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.nu is not None and not (np.isfinite(self.nu) and self.nu > 0.0):
            raise ValueError(f"nu must be positive and finite, or None; got {self.nu}")
        if not self.scale > 0.0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.nu is not None and self.scale != 1.0:
            raise ValueError("a heavy-tailed prior has unit scale; scale widens the Gaussian only")

    def sample_zstar(self, rng: np.random.Generator, n: int, d: int) -> np.ndarray:
        """``(n, d)`` prior draws of z*. With ``nu``, kappa first, then ``z* | kappa``.

        The Gaussian draws ``rng.normal(size=(n, d))`` and nothing else, so the default prior
        consumes Phase 4's random numbers exactly.
        """
        if self.nu is None:
            z = rng.normal(size=(n, d))
            return z if self.scale == 1.0 else np.sqrt(self.scale) * z
        kappa = 1.0 / rng.gamma(0.5 * self.nu, 2.0 / self.nu, size=n)
        return np.sqrt(kappa)[:, None] * rng.normal(size=(n, d))

    def sample(self, rng: np.random.Generator, n: int, d: int) -> np.ndarray:
        return np.column_stack([
            self.sample_zstar(rng, n, d),
            rng.normal(self.log_ell0, self.ell_sd, size=n),
            logit(rng.beta(self.lam_a, self.lam_b, size=n)),
        ])

    @property
    def lambda_sd(self) -> float:
        a, b = self.lam_a, self.lam_b
        return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0))))


def log_prior(x: np.ndarray, d: int, prior: BowlPrior = BowlPrior()) -> np.ndarray:
    """Unnormalised log prior of each particle, including the logit Jacobian.

    ``Beta(a, b)`` on ``lam = sigmoid(t)`` times ``d lam / d t = lam (1 - lam)`` is
    ``lam^a (1 - lam)^b``, which is where the two ``log_expit`` terms come from. The z* term
    is the Gaussian's ``-|z*|^2 / (2 scale)``, or with ``nu`` the multivariate t's
    ``-(nu + d)/2 log(1 + |z*|^2 / nu)``. At the defaults the arithmetic is Phase 4's
    operation for operation, so the tight arm's Metropolis decisions are unchanged.
    """
    x = np.atleast_2d(x)
    z_star, log_ell, t = x[:, :d], x[:, d], x[:, d + 1]
    squared = np.sum(z_star * z_star, axis=1)
    if prior.nu is None:
        z_term = -0.5 * squared / prior.scale
    else:
        z_term = -0.5 * (prior.nu + d) * np.log1p(squared / prior.nu)
    return (z_term
            - 0.5 * ((log_ell - prior.log_ell0) / prior.ell_sd) ** 2
            + prior.lam_a * log_expit(t) + prior.lam_b * log_expit(-t))


def per_duel_log_likelihood(x: np.ndarray, z_a: np.ndarray, z_b: np.ndarray,
                   prefers_a) -> np.ndarray:
    """Log-likelihood of each of ``n`` answers under each of ``N`` particles, shape ``(N, n)``.

    ``Delta`` is expanded to ``z*. (z_A - z_B) - (|z_A|^2 - |z_B|^2) / 2``, which is
    linear in ``z*`` -- the reason the posterior is well behaved.
    """
    x = np.atleast_2d(x)
    z_a, z_b = np.atleast_2d(z_a), np.atleast_2d(z_b)
    prefers_a = np.atleast_1d(np.asarray(prefers_a, dtype=bool))
    d = z_a.shape[1]
    z_star, log_ell, t = x[:, :d], x[:, d], x[:, d + 1]
    delta = (z_star @ (z_a - z_b).T
             - 0.5 * (np.sum(z_a * z_a, axis=1) - np.sum(z_b * z_b, axis=1))[None, :])
    arg = delta * np.exp(-2.0 * log_ell)[:, None]
    signed = np.where(prefers_a[None, :], arg, -arg)
    log_half_lapse = (log_expit(t) - np.log(2.0))[:, None]
    log_attentive = log_expit(-t)[:, None]
    return np.logaddexp(log_half_lapse, log_attentive + log_ndtr(signed))


def log_likelihood(x: np.ndarray, z_a: np.ndarray, z_b: np.ndarray,
                   prefers_a) -> np.ndarray:
    """Log-likelihood of ``n`` answers under each of ``N`` particles, shape ``(N,)``."""
    return np.sum(per_duel_log_likelihood(x, z_a, z_b, prefers_a), axis=1)


class SMCPosterior:
    """Particle approximation to the bowl model's posterior, updated one answer at a time.

    Each answer reweights. When the effective sample size falls below
    ``ess_fraction * n_particles`` the particles are systematically resampled and moved
    by ``n_moves`` random-walk Metropolis steps (5 under a Gaussian prior, 10 under a
    heavy-tailed one) targeting the full posterior, with a proposal covariance of
    ``2.38^2 / D`` times the particle covariance.
    """

    def __init__(self, d: int, rng: np.random.Generator, n_particles: int = 4000,
                 prior: BowlPrior = BowlPrior(), n_moves: int | None = None,
                 ess_fraction: float = 0.5) -> None:
        self.d = int(d)
        self.rng = rng
        self.n = int(n_particles)
        self.prior = prior
        if n_moves is None:
            n_moves = GAUSSIAN_MOVES if prior.nu is None else HEAVY_TAIL_MOVES
        self.n_moves = int(n_moves)
        self.ess_fraction = float(ess_fraction)
        self.x = prior.sample(rng, self.n, self.d)
        self.log_w = np.zeros(self.n)
        self._z_a: list[np.ndarray] = []
        self._z_b: list[np.ndarray] = []
        self._prefers_a: list[bool] = []
        self.n_resample = 0
        self.acceptance: list[float] = []

    @property
    def n_answers(self) -> int:
        return len(self._prefers_a)

    def weights(self) -> np.ndarray:
        w = np.exp(self.log_w - self.log_w.max())
        return w / w.sum()

    @property
    def ess(self) -> float:
        w = self.weights()
        return float(1.0 / np.sum(w * w))

    def update(self, z_a: np.ndarray, z_b: np.ndarray, prefers_a: bool) -> None:
        z_a, z_b = np.asarray(z_a, dtype=float), np.asarray(z_b, dtype=float)
        if z_a.shape != (self.d,) or z_b.shape != (self.d,):
            raise ValueError(f"expected z of shape ({self.d},), got {z_a.shape} and "
                             f"{z_b.shape}")
        self._z_a.append(z_a)
        self._z_b.append(z_b)
        self._prefers_a.append(bool(prefers_a))
        self.log_w += log_likelihood(self.x, z_a[None], z_b[None], [bool(prefers_a)])
        if self.ess < self.ess_fraction * self.n:
            self._resample_move()

    def log_target(self, x: np.ndarray) -> np.ndarray:
        """Unnormalised log posterior given every answer so far."""
        lp = log_prior(x, self.d, self.prior)
        if self._prefers_a:
            lp = lp + log_likelihood(x, np.array(self._z_a), np.array(self._z_b),
                                     np.array(self._prefers_a))
        return lp

    def _resample_move(self) -> None:
        w = self.weights()
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), self.n - 1)
        self.x = self.x[idx]
        self.log_w = np.zeros(self.n)
        self.n_resample += 1

        dim = self.x.shape[1]
        cov = np.cov(self.x, rowvar=False) * (2.38**2 / dim) + 1e-9 * np.eye(dim)
        chol = np.linalg.cholesky(cov)
        current = self.log_target(self.x)
        for _ in range(self.n_moves):
            proposal = self.x + self.rng.normal(size=self.x.shape) @ chol.T
            candidate = self.log_target(proposal)
            accept = np.log(self.rng.random(self.n)) < candidate - current
            self.x[accept] = proposal[accept]
            current[accept] = candidate[accept]
            self.acceptance.append(float(accept.mean()))

    def mean(self) -> np.ndarray:
        return self.weights() @ self.x

    def sd(self) -> np.ndarray:
        w = self.weights()
        m = w @ self.x
        return np.sqrt(w @ (self.x - m) ** 2)

    def interval(self, level: float = 0.9) -> np.ndarray:
        """Central weighted credible interval per coordinate, shape ``(d + 2, 2)``."""
        w = self.weights()
        lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
        out = np.empty((self.x.shape[1], 2))
        for j in range(self.x.shape[1]):
            order = np.argsort(self.x[:, j])
            out[j] = np.interp([lo, hi], np.cumsum(w[order]), self.x[order, j])
        return out

    def zstar_mean(self) -> np.ndarray:
        return self.mean()[: self.d]

    def ell_mean(self) -> float:
        return float(self.weights() @ np.exp(self.x[:, self.d]))

    def lambda_mean(self) -> float:
        return float(self.weights() @ expit(self.x[:, self.d + 1]))

    def lambda_sd(self) -> float:
        w = self.weights()
        lam = expit(self.x[:, self.d + 1])
        m = w @ lam
        return float(np.sqrt(w @ (lam - m) ** 2))

    def _kappa_posterior(self) -> tuple[float, np.ndarray]:
        """Shape and per-particle scale of ``kappa | z*``."""
        if self.prior.nu is None:
            raise ValueError("this posterior has a Gaussian prior; it has no kappa")
        zs = self.x[:, : self.d]
        return 0.5 * (self.prior.nu + self.d), 0.5 * (self.prior.nu + np.sum(zs * zs, axis=1))

    def kappa_mean(self) -> float:
        """``E[kappa | answers]``, Rao-Blackwellised: the weighted mean of ``b / (a - 1)``."""
        a, b = self._kappa_posterior()
        if a <= 1.0:
            raise ValueError(f"E[kappa | z*] needs nu + d > 2; got nu={self.prior.nu}, d={self.d}")
        return float(self.weights() @ (b / (a - 1.0)))

    def log_kappa_moments(self) -> tuple[float, float]:
        """Mean and SD of ``log kappa | answers``.

        Per particle ``E[log kappa | z*] = log b - digamma(a)`` and
        ``Var[log kappa | z*] = trigamma(a)``, combined by the law of total variance.
        """
        a, b = self._kappa_posterior()
        w = self.weights()
        conditional = np.log(b) - digamma(a)
        mean = float(w @ conditional)
        return mean, float(np.sqrt(polygamma(1, a) + w @ (conditional - mean) ** 2))

    def sample_log_kappa(self, rng: np.random.Generator) -> np.ndarray:
        """One draw of ``log kappa | z*`` per particle; weight them with :meth:`weights`."""
        a, b = self._kappa_posterior()
        return -np.log(rng.gamma(a, 1.0 / b))
