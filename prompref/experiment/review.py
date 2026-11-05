"""ICORR reframe, review response (our design notes, "Review response: comparator
fairness, ablations and a sanity check").

Two learner variants, each switched on by a :class:`~prompref.experiment.phase5.Phase5Point`
field that stays out of the key and the row at its default, so every earlier session is built
exactly as before:

- **G with a start at the prior's mean** (``g_start="generic"``). Arens' Algorithm S1 opens with a random
  duel between two candidates, so G's incumbent after the first answer is a random prior draw.
  Here candidate 0 is replaced by z = 0, the prior mean in the arm's normalized coordinates, which
  is where A, LineCoSpar and SortCMA start; the first duel is that candidate against one random
  other candidate. Everything after the first duel is :class:`GPUCBLearner` unchanged. The
  candidate set keeps its size (``n_candidates`` stays 2048, and with it the row's resume key);
  ``best`` before any answer is candidate 0, generic assistance. Gate on or off is G's own
  ``jnd_scale`` (1.0, the true JND, or ``None``: the UCB pair is asked as proposed).
  z = 0's device-clipped curve is not exactly generic assistance (the mean of device-clipped prior
  curves); each row records the RMS gap between the two (``g_generic_gap_nm``).
- **A without lapse** (``a_no_lapse=True``). A's likelihood with lambda fixed at 0, and no
  intensity probes (``p_probe = 0``); the simulated user still lapses at the point's ``lapse``. Every
  particle's logit-lapse coordinate is -inf, so ``smc.per_duel_log_likelihood`` -- the one
  likelihood the posterior update, the acquisition and the one-step predictive all call -- gives
  ``log Phi(signed)`` exactly (``log_expit(-inf) = -inf``, ``log_expit(inf) = -0.0``). That
  coordinate is held out of the prior and of the Metropolis moves, which act on (z*, log ell)
  only. ``smc.py`` and ``acquisition.py`` are gated (``prompref/validation/stamp.py``) and are not
  touched; this module subclasses them.
"""

from __future__ import annotations

import numpy as np

from ..learning.gp_ucb import GPUCBLearner
from ..learning.psychometric import _STREAM_SALT, PsychometricLearner
from ..learning.smc import IdealPointPrior, SMCPosterior, log_likelihood, log_prior
from ..learning.normalized import WhitenedLearner
from ..user.simulated_user import rms_distance
from .metrics import prior_mean_curve

#: The values of ``Phase5Point.g_start`` besides ``None`` (Arens' random first duel).
G_STARTS = ("generic",)
#: G's generic candidate: index 0 of its candidate set.
GENERIC_INDEX = 0


# == G with a start at the prior's mean ===================================================================

class GenericStartGPUCB(GPUCBLearner):
    """:class:`GPUCBLearner` whose candidate 0 is z = 0 and whose first duel is candidate 0
    against one random other candidate. The candidate draw itself is the parent's, unchanged."""

    def __init__(self, design, seed: int = 0, **kwargs) -> None:
        super().__init__(design, seed, **kwargs)
        generic = np.asarray(design.project(np.zeros(design.n_params)), dtype=float)
        self.candidates[GENERIC_INDEX] = generic
        self.curves[GENERIC_INDEX] = design.curve(generic)

    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pending is None and not self._pairs:
            # The first duel: the incumbent is generic assistance, its partner a random draw from
            # the other candidates (Algorithm S1's random first comparison, half of it). As in the
            # parent, it is shown as drawn even when it fails the gate, and logged as exhausted.
            partner = 1 + int(self._rng.integers(len(self.candidates) - 1))
            if not self._perceivable(GENERIC_INDEX, partner):
                self.n_gate_blocked += 1
                self.n_gate_exhausted += 1
            self._pending = (GENERIC_INDEX, partner)
            return self.candidates[GENERIC_INDEX].copy(), self.candidates[partner].copy()
        return super().propose()


def prior_mean_start_g(space, seed: int, *, prior, jnd, n_candidates: int, **tuned) -> WhitenedLearner:
    """G with a start at the prior's mean in the arm's search region, as ``WhitenedLearner.gp_ucb`` builds plain G."""
    inner = GenericStartGPUCB(space.as_design(), seed=seed, jnd=jnd, prior=prior,
                              n_candidates=int(n_candidates), **tuned)
    return WhitenedLearner(space, inner)


def prior_mean_start_fields(learner: WhitenedLearner, design, ideal_nm) -> dict:
    """Row fields of a start at the prior's mean G session: the RMS distance in Nm of G's generic candidate
    to the ideal, and its RMS gap to generic assistance (the curve ``d_prior`` is the distance
    of). Draws no session random number (``prior_mean_curve`` uses its own seed-0 generator)."""
    curve = np.asarray(design.curve(learner.space.to_theta(learner.inner.candidates[GENERIC_INDEX])),
                       dtype=float)
    return {"g_generic_distance": rms_distance(curve, ideal_nm),
            "g_generic_gap_nm": rms_distance(curve, prior_mean_curve(design))}


# == A without lapse ==========================================================================

class NoLapsePosterior(SMCPosterior):
    """A's SMC posterior with lambda = 0 exactly.

    The particles keep the layout (z*, log ell, logit lam), with the last column -inf. They are
    drawn from the same generator and prior as A's own posterior, so z* and log ell start as A's
    do. The lapse terms of the prior are left out (evaluated at logit 0 they are one constant over
    particles, which cancels in every Metropolis ratio), and the moves act on (z*, log ell) with
    the proposal scale 2.38^2 / (d + 1) of that dimension.
    """

    def __init__(self, d: int, rng: np.random.Generator, n_particles: int = 4000,
                 prior: IdealPointPrior = IdealPointPrior(), n_moves: int | None = None,
                 ess_fraction: float = 0.5) -> None:
        super().__init__(d, rng, n_particles=n_particles, prior=prior, n_moves=n_moves,
                         ess_fraction=ess_fraction)
        self.x[:, self.d + 1] = -np.inf

    @property
    def n_free(self) -> int:
        return self.d + 1

    def log_target(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        finite = x.copy()
        finite[:, self.d + 1] = 0.0
        lp = log_prior(finite, self.d, self.prior)
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

        free = self.n_free
        cov = np.cov(self.x[:, :free], rowvar=False) * (2.38**2 / free) + 1e-9 * np.eye(free)
        chol = np.linalg.cholesky(cov)
        current = self.log_target(self.x)
        for _ in range(self.n_moves):
            proposal = self.x.copy()
            proposal[:, :free] += self.rng.normal(size=(self.n, free)) @ chol.T
            candidate = self.log_target(proposal)
            accept = np.log(self.rng.random(self.n)) < candidate - current
            self.x[accept] = proposal[accept]
            current[accept] = candidate[accept]
            self.acceptance.append(float(accept.mean()))

    def mean(self) -> np.ndarray:
        return np.append(self.weights() @ self.x[:, :self.n_free], -np.inf)

    def sd(self) -> np.ndarray:
        w = self.weights()
        xs = self.x[:, :self.n_free]
        m = w @ xs
        return np.append(np.sqrt(w @ (xs - m) ** 2), 0.0)

    def interval(self, level: float = 0.9) -> np.ndarray:
        w = self.weights()
        lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
        out = np.full((self.x.shape[1], 2), -np.inf)
        for j in range(self.n_free):
            order = np.argsort(self.x[:, j])
            out[j] = np.interp([lo, hi], np.cumsum(w[order]), self.x[order, j])
        return out

    def lambda_mean(self) -> float:
        return 0.0

    def lambda_sd(self) -> float:
        return 0.0


class NoLapseLearner(PsychometricLearner):
    """Learner A (EIG duels) with lambda fixed at 0 in its model and no intensity probes."""

    def __init__(self, design, seed: int = 0, *, n_particles: int = 4000, **kwargs) -> None:
        for name in ("duels", "p_probe"):
            if name in kwargs:
                raise ValueError(f"A without lapse fixes {name}")
        super().__init__(design, seed, duels="eig", p_probe=0.0, n_particles=n_particles,
                         **kwargs)
        # The parent's posterior, rebuilt from an identical generator, with lambda pinned at 0.
        self.posterior = NoLapsePosterior(self.space.d,
                                          np.random.default_rng([int(seed), _STREAM_SALT, 1]),
                                          n_particles=n_particles, prior=self.prior)

    def posterior_summary(self) -> dict:
        summary = super().posterior_summary()
        # lambda was never learned: it is fixed, so it was not identified either.
        summary["extras"]["lambda_identified"] = False
        return summary
