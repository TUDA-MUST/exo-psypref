"""LineCoSpar -- comparator D.

Reimplemented from Tucker et al., IROS 2020 (arXiv 2003.06495) and checked line by line
against the authors' reference code, ``github.com/myracheng/linecospar``. That repository
carries no licence, so it is read as a reference and **not vendored**.

What LineCoSpar does, per iteration:

1. Draw a line through the current anchor in a uniformly random direction, discretised
   at a fixed step and extended both ways until it leaves the feasible set.
2. Fit a preference GP over every previously executed point plus the new line, by
   Laplace approximation with a sigmoid link.
3. Thompson-sample one utility function and execute its argmax.
4. Compare the executed point with the previously executed one -- ``num_samples = 1``,
   ``buffer_size = 1`` in the authors' exoskeleton drivers.
5. Anchor the next line at the posterior-mean maximiser.

Four things this module does that are easy to get wrong:

- **The anchor lags one answer.** The reference computes the next anchor from the same
  posterior it Thompson-samples from, *before* the new answer arrives. Reproduced.
- **Coordinates are the design's feasible box scaled to the unit cube.** The published
  lengthscale 0.15 and step 0.1 are unit-cube numbers from the synthetic experiments.
  Scaling keeps them meaningful across families whose parameters range from %GC to Nm.
- **Feasibility is the design's own**, not just the box: a point is on the line only if
  ``design.project`` leaves it unchanged. For every current family that coincides with
  the reference's box test -- showed Schaefer's box already implies its timing
  order -- but a family with a genuine non-box constraint would otherwise be walked
  straight through it.
- **Every proposal is a real comparison, including the first.** The reference spends its
  first Thompson sample filling an empty buffer. Our harness charges every ``propose``
  against the budget, so the random start point is the first buffer entry -- which is
  also what a simulated user walking the starting gait experiences.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit

from ..designs.base import Design

#: Salt separating this learner's stream from the observer's. The sweep hands both the
#: same ``seed``; unsalted, ``default_rng(seed)`` would give them identical draws.
_STREAM_SALT = 0x11EC05

#: The published kernel lengthscale and line step (Tucker et al. 2020, synthetic
#: experiments), in unit-cube coordinates. D_z's are tuned against these.
LENGTHSCALE = 0.15
STEP = 0.1


class LineCoSparLearner:
    """Random one-dimensional subspaces, a preference GP, and Thompson sampling."""

    def __init__(self, design: Design, seed: int = 0, *, lengthscale: float = LENGTHSCALE,
                 signal_variance: float = 1e-4, gp_noise_variance: float = 1e-5,
                 preference_noise: float = 0.005, step: float = STEP,
                 refresh_every: int = 1, max_direction_draws: int = 10_000,
                 start: np.ndarray | None = None) -> None:
        if refresh_every < 1:
            raise ValueError(f"refresh_every must be at least 1, got {refresh_every}")
        self.design = design
        self.lengthscale = float(lengthscale)
        self.signal_variance = float(signal_variance)
        self.gp_noise_variance = float(gp_noise_variance)
        self.preference_noise = float(preference_noise)
        self.step = float(step)
        self.refresh_every = int(refresh_every)
        self.max_direction_draws = int(max_direction_draws)

        self._low = np.asarray(design.low, dtype=float)
        self._span = np.asarray(design.high, dtype=float) - self._low
        self._rng = np.random.default_rng([int(seed), _STREAM_SALT])

        # The reference starts uniformly over its feasible set, and that stays the
        # default -- drawn first, so the default stream is unchanged. D_z passes the
        # prior mean instead.
        if start is None:
            start = design.project(self._to_theta(self._rng.uniform(size=self._low.size)))
        else:
            start = design.project(np.asarray(start, dtype=float))
        self._visited: list[np.ndarray] = [self._to_unit(start)]
        #: ``(buffer index, new index, response)``, indices into ``_visited``.
        self._data: list[tuple[int, int, int]] = []
        self._buffer = 0
        self._anchor = self._visited[0].copy()
        self._anchor_next: np.ndarray | None = None
        self._line_pts: np.ndarray | None = None
        self._pending: tuple[np.ndarray, int | None] | None = None
        self._best_cache: np.ndarray | None = None
        self._iteration = 0

        self.n_same_pair = 0
        self.n_empty_line = 0

    # -- coordinates -------------------------------------------------------------------

    def _to_theta(self, u: np.ndarray) -> np.ndarray:
        return self._low + np.asarray(u, dtype=float) * self._span

    def _to_unit(self, theta: np.ndarray) -> np.ndarray:
        safe = np.where(self._span > 0.0, self._span, 1.0)
        return (np.asarray(theta, dtype=float) - self._low) / safe

    def _feasible(self, u: np.ndarray) -> bool:
        if np.any(u < 0.0) or np.any(u > 1.0):
            return False
        theta = self._to_theta(u)
        return bool(np.allclose(self.design.project(theta), theta, rtol=0.0, atol=1e-9))

    # -- the subspace ------------------------------------------------------------------

    def _line(self, anchor: np.ndarray) -> np.ndarray:
        """Feasible points ``anchor ± k·step·direction``, k = 1, 2, ...; anchor excluded.

        Mirrors the reference's ``get_prior_randomdir``: the positive ray first, then the
        negative, each stopping at the first infeasible point, and a fresh direction if
        both rays are empty. Directions have no component along a zero-width dimension,
        which would otherwise let the line wander in unit coordinates without moving
        the curve.
        """
        n = anchor.size
        for _ in range(self.max_direction_draws):
            direction = self._rng.normal(size=n)
            direction[self._span <= 0.0] = 0.0
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                continue
            direction /= norm
            points = []
            for sign in (1.0, -1.0):
                k = 1
                while True:
                    p = anchor + sign * k * self.step * direction
                    if not self._feasible(p):
                        break
                    points.append(p)
                    k += 1
            if points:
                return np.array(points)
        self.n_empty_line += 1
        return np.empty((0, n))

    # -- the preference GP -------------------------------------------------------------

    def _kernel(self, pts: np.ndarray) -> np.ndarray:
        diff = (pts[:, None, :] - pts[None, :, :]) / self.lengthscale
        k = self.signal_variance * np.exp(-0.5 * np.sum(diff * diff, axis=-1))
        return k + self.gp_noise_variance * np.eye(len(pts))

    def _pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Winner and loser indices of every recorded answer."""
        if not self._data:
            empty = np.empty(0, dtype=int)
            return empty, empty
        arr = np.asarray(self._data, dtype=int)
        a, b, label = arr[:, 0], arr[:, 1], arr[:, 2]
        return np.where(label == 1, b, a), np.where(label == 1, a, b)

    @staticmethod
    def _objective(f, k_inv, win, lose, c):
        """Negative log posterior and its gradient: ``½fᵀK⁻¹f − Σ log σ((f_w − f_l)/c)``."""
        z = (f[win] - f[lose]) / c
        value = 0.5 * f @ k_inv @ f + float(np.sum(np.logaddexp(0.0, -z)))
        grad = k_inv @ f
        s = expit(-z) / c
        np.add.at(grad, win, -s)
        np.add.at(grad, lose, s)
        return value, grad

    @staticmethod
    def _hessian(f, k_inv, win, lose, c):
        z = (f[win] - f[lose]) / c
        h = expit(z) * expit(-z) / c**2
        hess = k_inv.copy()
        np.add.at(hess, (win, win), h)
        np.add.at(hess, (lose, lose), h)
        np.add.at(hess, (win, lose), -h)
        np.add.at(hess, (lose, win), -h)
        return hess

    def _laplace(self, pts: np.ndarray, with_covariance: bool = True):
        """Posterior mean over ``pts``, plus the eigenpairs of its precision if asked.

        ``pts`` must start with ``_visited`` in order, since the data index into it.
        """
        k = self._kernel(pts)
        k_inv = cho_solve(cho_factor(k), np.eye(len(pts)))
        k_inv = 0.5 * (k_inv + k_inv.T)
        win, lose = self._pairs()
        c = self.preference_noise
        res = minimize(self._objective, np.zeros(len(pts)), args=(k_inv, win, lose, c),
                       jac=True, method="L-BFGS-B")
        mean = res.x
        if not with_covariance:
            return mean, None, None
        evals, evecs = np.linalg.eigh(self._hessian(mean, k_inv, win, lose, c))
        return mean, np.clip(evals, 1e-12 * evals.max(), None), evecs

    # -- the learner protocol ----------------------------------------------------------

    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pending is not None:
            raise RuntimeError("update() the previous proposal before proposing again")
        if self._line_pts is None or self._iteration % self.refresh_every == 0:
            self._line_pts = self._line(self._anchor)
        visited = np.array(self._visited)
        pts = np.vstack([visited, self._line_pts]) if len(self._line_pts) else visited

        mean, evals, evecs = self._laplace(pts)
        # The next line's anchor comes from THIS posterior, before the answer -- the lag.
        self._anchor_next = pts[int(np.argmax(mean))].copy()

        sample = mean + evecs @ (self._rng.normal(size=len(pts)) / np.sqrt(evals))
        idx = int(np.argmax(sample))
        self._pending = (pts[idx].copy(), idx if idx < len(visited) else None)
        return self._to_theta(visited[self._buffer]), self._to_theta(pts[idx])

    def update(self, theta_a: np.ndarray, theta_b: np.ndarray, response: int) -> None:
        """A is the previously executed point, B the new one: 0 prefers A, 1 prefers B.

        A "no preference" answer adds no likelihood term, as in the reference, where a
        0.5 label is skipped. A comparison of a point with itself is also skipped -- but
        it has still spent a comparison, and is counted in ``n_same_pair``.
        """
        if self._pending is None:
            raise RuntimeError("call propose() before update()")
        point, idx = self._pending
        self._pending = None
        if idx is None:
            idx = self._index_of(point)
        if idx == self._buffer:
            self.n_same_pair += 1
        elif response in (0, 1):
            self._data.append((self._buffer, idx, int(response)))
        self._buffer = idx
        self._anchor = self._anchor_next
        self._iteration += 1
        self._best_cache = None

    def _index_of(self, point: np.ndarray) -> int:
        for i, v in enumerate(self._visited):
            if np.array_equal(v, point):
                return i
        self._visited.append(point)
        return len(self._visited) - 1

    def best(self) -> np.ndarray:
        """Posterior-mean maximiser over the points actually executed.

        Restricted to executed points: recommending a line point nobody has walked would
        be odd. It costs nothing in fidelity -- unexecuted points never enter the
        likelihood, so the MAP at the executed points is the same with or without them.
        Draws no random numbers, so calling it cannot perturb the proposal sequence.
        """
        if self._best_cache is None:
            visited = np.array(self._visited)
            mean, _, _ = self._laplace(visited, with_covariance=False)
            self._best_cache = visited[int(np.argmax(mean))]
        return self._to_theta(self._best_cache)

    def posterior_summary(self) -> dict:
        return {"extras": {
            "n_visited": len(self._visited),
            "n_same_pair": int(self.n_same_pair),
            "n_empty_line": int(self.n_empty_line),
            "n_answers": len(self._data),
        }}
