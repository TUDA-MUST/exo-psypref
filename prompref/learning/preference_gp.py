"""The preference GP behind comparator G, after Arens et al. 2025.

A zero-mean Gaussian process over a latent utility, a probit pairwise likelihood, and a
Laplace approximation of the posterior -- the model of Chu & Ghahramani (2005) as Arens
uses it through the `GPro` library, reimplemented here because that library is not
installed and no dependency may be added.

Two deliberate differences from ``linecospar.py``'s GP, which exists for comparator D:

- **The link is probit**, Phi((f_w - f_l) / (sqrt(2) sigma)), not logistic. That is what the
  paper specifies, and it is what makes the preference noise sigma a perceptual scale.
- **Predictions are made at arbitrary query points**, because UCB scores candidates that are
  not training points. LineCoSpar only ever needs the fitted mean at points it already has.

Hyperparameters are **fixed, never refitted**: the paper sets them once,
and refitting them would compare against a method the paper did not publish.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm

#: Arens et al.: output scale 10 with preference noise 1. Their ratio is what sets how much
#: a single comparison moves the posterior, so both travel together.
OUTPUT_SCALE = 10.0
PREFERENCE_NOISE = 1.0

#: Damped Newton (Rasmussen & Williams, Algorithm 3.1), not L-BFGS-B: the objective is
#: strictly convex, so Newton has a quadratically-converging exact Hessian available
#: (``_precision``) and reaches machine-precision stationarity in tens of iterations,
#: where L-BFGS-B's quasi-Newton approximation was seen stalling on its evaluation cap
#: well short of the true MAP. This is also what GPro does.
#:
#: Convergence is judged by the Newton decrement ``lambda_sq = grad @ step`` (Boyd &
#: Vandenberghe Sec. 9.5.1), not by the absolute gradient norm. The decrement is
#: affine-invariant and estimates twice the remaining suboptimality; an absolute
#: gradient tolerance is not reachable in general, because ``fit`` forms K^-1 explicitly
#: and its own numerical error (growing with ``cond(K)``, which grows with the number of
#: points) can exceed a tight absolute tolerance while the fit itself has long since
#: stopped moving.
NEWTON_DECREMENT_TOL = 1e-8
NEWTON_MAX_ITERS = 100
#: Backtracking halvings per Newton step. Small and fixed because the objective is convex:
#: near the optimum a full Newton step is accepted immediately.
#:
#: It is *not* only a guard on the early, large-gradient iterations, as this comment said
#: until the review corrected it. This branch's own measurements disproved that: 6.5 %
#: of fits exhaust all twenty halvings, and they do so at the floating-point floor, where no
#: halving of a certified descent direction can improve the objective any further. That is
#: the common case for exit 2 of ``_fit_map``, and it is exactly the path
#: ``FLOOR_DECREMENT_TOL`` exists to accept -- so raising this constant would not help those
#: fits, it would only make each of them slower.
BACKTRACK_MAX_HALVINGS = 20

#: The looser bound that judges the *exhausted-backtracking* exit alone, and is applied
#: relative to the objective: ``lambda_sq <= FLOOR_DECREMENT_TOL * max(1, |value|)``.
#:
#: When no halving of a descent direction -- taken through a Hessian that ``cho_factor``
#: has just certified positive definite -- improves a strictly convex objective, the fit
#: is sitting on its floating-point floor; there is nowhere left to go. That floor is not
#: a fixed number: it scales with ``cond(K)`` (1e9-1e10 at the sizes comparator G reaches,
#: because ``fit`` forms K^-1 explicitly) and with the magnitude of the objective. Measured
#: over 6000 fits at 40-300 points and 0.67-1.5 pairs per point, the decrement on this path
#: spans 1e-8..9e-6 -- three orders of magnitude, which is why no *absolute* tolerance is
#: safe here -- while the ratio below stays under 2e-7, and under 6e-8 at the pair densities
#: comparator G reaches. The constant sits at 1e-5 rather than 1e-6 because the asymmetry
#: is stark: one false raise kills a 13,400-session sweep, whereas a bound that is loose by
#: an order of magnitude only admits a point whose suboptimality the decrement still bounds.
#: 1e-5 leaves roughly 60x margin at the worst point ever measured and ~200x inside
#: comparator G's own envelope, and still rejects a broken fit -- ratio of order 1 -- by
#: five orders. The task report has the distribution behind those numbers.
#:
#: A relative bound is also the dimensionally right shape, since the decrement estimates
#: twice the remaining suboptimality *of the objective* and so belongs on the objective's
#: own scale. Checked directly at 100 points, where the decrement is 1.0e-6: an independent
#: ``trust-ncg`` run with this same exact Hessian and ``gtol=1e-12`` finds an objective
#: 5.1e-7 lower -- half the decrement, as the estimate predicts -- with the same argmax,
#: and its own warm and cold starts disagree by 3.2e-7, which is the floor restated. A
#: genuinely broken fit, whose decrement is of the same order as the objective itself,
#: still fails this bound by orders of magnitude.
FLOOR_DECREMENT_TOL = 1e-5


class PreferenceGP:
    """Latent utility from pairwise comparisons, by Laplace approximation."""

    def __init__(self, lengthscale: float, output_scale: float = OUTPUT_SCALE,
                 preference_noise: float = PREFERENCE_NOISE, jitter: float = 1e-8) -> None:
        if lengthscale <= 0.0:
            raise ValueError(f"lengthscale must be positive, got {lengthscale}")
        if output_scale <= 0.0:
            raise ValueError(f"output_scale must be positive, got {output_scale}")
        if preference_noise <= 0.0:
            raise ValueError(f"preference_noise must be positive, got {preference_noise}")
        self.lengthscale = float(lengthscale)
        self.output_scale = float(output_scale)
        self.preference_noise = float(preference_noise)
        self.jitter = float(jitter)
        self._points: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._k_inv: np.ndarray | None = None
        self._posterior_precision: np.ndarray | None = None

    # -- kernel ------------------------------------------------------------------------

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = (a[:, None, :] - b[None, :, :]) / self.lengthscale
        return self.output_scale * np.exp(-0.5 * np.sum(diff * diff, axis=-1))

    # -- likelihood --------------------------------------------------------------------

    def _terms(self, f: np.ndarray, win: np.ndarray, lose: np.ndarray):
        """z, the standard-normal pdf/cdf ratio, and the scale, shared by value and gradient."""
        scale = np.sqrt(2.0) * self.preference_noise
        z = (f[win] - f[lose]) / scale
        # log Phi(z) is computed by its stable form; the ratio phi/Phi is the gradient factor.
        log_cdf = norm.logcdf(z)
        ratio = np.exp(norm.logpdf(z) - log_cdf)
        return z, log_cdf, ratio, scale

    def _objective(self, f, k_inv, win, lose):
        _, log_cdf, ratio, scale = self._terms(f, win, lose)
        value = 0.5 * f @ k_inv @ f - float(np.sum(log_cdf))
        grad = k_inv @ f
        np.add.at(grad, win, -ratio / scale)
        np.add.at(grad, lose, ratio / scale)
        return value, grad

    def _precision(self, f, k_inv, win, lose):
        z, _, ratio, scale = self._terms(f, win, lose)
        # d/dz of -log Phi(z) is -ratio; its second derivative is ratio * (ratio + z).
        h = ratio * (ratio + z) / scale**2
        precision = k_inv.copy()
        np.add.at(precision, (win, win), h)
        np.add.at(precision, (lose, lose), h)
        np.add.at(precision, (win, lose), -h)
        np.add.at(precision, (lose, win), -h)
        return precision

    def _fit_map(self, k_inv: np.ndarray, win: np.ndarray, lose: np.ndarray) -> np.ndarray:
        """The Laplace MAP by damped Newton: exact Hessian, backtracking on the objective.

        The loop has three exits, and each gets the criterion that actually fits it:

        1. **Converged.** The Newton decrement meets ``NEWTON_DECREMENT_TOL`` before the
           step is taken -- free, since the step is already in hand. Almost every fit
           leaves here.
        2. **Backtracking exhausted.** No halving of the descent direction improves the
           objective, so the fit is on its floating-point floor (see
           ``FLOOR_DECREMENT_TOL``) and is accepted against the relative bound. Note that
           the tight in-loop test is *unusable* on this path: ``f`` and ``grad`` are
           unchanged here, so any recomputed decrement reproduces the in-loop one bitwise,
           and the path is only reached when that value is already above the tolerance --
           checking it again would raise on every exhaustion by construction.
        3. **Iteration cap.** ``NEWTON_MAX_ITERS`` accepted steps on a strictly convex
           problem without meeting the tight criterion is genuine non-convergence, and
           raises unconditionally after the loop.
        """
        f = np.zeros(k_inv.shape[0])
        value, grad = self._objective(f, k_inv, win, lose)
        for _ in range(NEWTON_MAX_ITERS):
            precision = self._precision(f, k_inv, win, lose)
            step = cho_solve(cho_factor(precision), grad)
            lambda_sq = float(grad @ step)
            # Both `<=` and `>` are False for a NaN, so a non-finite decrement would slip
            # through every comparison below and return a corrupt fit silently.
            if not np.isfinite(lambda_sq):
                raise RuntimeError(
                    "Laplace MAP diverged: the Newton decrement is not finite "
                    f"({lambda_sq}), max|grad| = {float(np.max(np.abs(grad)))}")
            if lambda_sq <= NEWTON_DECREMENT_TOL:
                return f
            step_size = 1.0
            stepped = False
            for _ in range(BACKTRACK_MAX_HALVINGS):
                candidate_f = f - step_size * step
                candidate_value, candidate_grad = self._objective(candidate_f, k_inv, win, lose)
                # Strictly less, not `<=`: equality means the objective cannot be decreased
                # along this direction in this arithmetic, which *is* the floor. Accepting
                # it as progress would let a finished fit take zero-progress steps until
                # the iteration cap, which raises unconditionally -- turning a converged
                # fit into a hard failure. Refusing it sends the fit to exit 2 instead,
                # where the floor criterion decides.
                if candidate_value < value:
                    f, value, grad = candidate_f, candidate_value, candidate_grad
                    stepped = True
                    break
                step_size *= 0.5
            if not stepped:
                # Exit 2. `f`, `value`, `grad` and `lambda_sq` all describe this same point,
                # so the floor bound is decided here, with no recomputation to go wrong.
                if lambda_sq <= FLOOR_DECREMENT_TOL * max(1.0, abs(value)):
                    return f
                raise RuntimeError(
                    "Laplace MAP stopped far from stationarity: backtracking was exhausted "
                    f"at Newton decrement = {lambda_sq} with objective value = {value}, "
                    f"above the floor bound {FLOOR_DECREMENT_TOL} * max(1, |value|) = "
                    f"{FLOOR_DECREMENT_TOL * max(1.0, abs(value))}; "
                    f"max|grad| = {float(np.max(np.abs(grad)))}")
        # Exit 3, and now the only way to reach this line: every one of the
        # NEWTON_MAX_ITERS iterations took an accepted step and none met the tight
        # criterion. The decrement is recomputed at the final `f`, because the last
        # accepted step moved it after the loop's last `lambda_sq` was formed.
        precision = self._precision(f, k_inv, win, lose)
        step = cho_solve(cho_factor(precision), grad)
        lambda_sq = float(grad @ step)
        raise RuntimeError(
            f"Laplace MAP did not converge: the iteration cap of {NEWTON_MAX_ITERS} Newton "
            f"iterations was reached with Newton decrement = {lambda_sq}, "
            f"max|grad| = {float(np.max(np.abs(grad)))}")

    # -- fitting -----------------------------------------------------------------------

    def fit(self, points: np.ndarray, pairs) -> None:
        """Fit the latent utility at ``points`` from ``(winner, loser)`` index pairs."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError(
                f"points must be a 2D array of shape (n_points, n_dims), got shape {points.shape}")
        pairs = list(pairs)
        for i, j in pairs:
            if not (0 <= int(i) < len(points) and 0 <= int(j) < len(points)):
                raise ValueError(f"pair index out of range for {len(points)} points: {(i, j)}")
        self._points = points
        k = self._kernel(points, points) + self.jitter * self.output_scale * np.eye(len(points))
        k_inv = cho_solve(cho_factor(k), np.eye(len(points)))
        self._k_inv = 0.5 * (k_inv + k_inv.T)
        if pairs:
            arr = np.asarray(pairs, dtype=int)
            win, lose = arr[:, 0], arr[:, 1]
            self._mean = self._fit_map(self._k_inv, win, lose)
            self._posterior_precision = self._precision(self._mean, self._k_inv, win, lose)
        else:
            self._mean = np.zeros(len(points))
            self._posterior_precision = self._k_inv.copy()

    def mean_at_points(self) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("fit() before reading the posterior")
        return self._mean.copy()

    # -- prediction --------------------------------------------------------------------

    def predict(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation of the latent utility at ``query``.

        The standard Laplace predictive: mean ``k* K^-1 f``, variance
        ``k** - k* (K^-1 - K^-1 P^-1 K^-1) k*``, with ``P`` the posterior precision.

        ``k**`` is taken as ``output_scale``, the kernel's true value at zero separation,
        while ``fit`` adds ``jitter * output_scale`` to ``K``'s diagonal for conditioning.
        A prediction *at a training point* is therefore inconsistent with the fitted
        diagonal by that jitter -- one part in 1e7, and in the direction of a slightly
        smaller variance. Left as it is deliberately: ``output_scale`` is the prior variance
        the model actually specifies, and the jitter is a numerical device, not a belief
        about the process. It is recorded here because the discrepancy is real and someone
        chasing a 1e-7 mismatch should find the answer rather than the arithmetic.
        """
        if self._points is None or self._mean is None:
            raise RuntimeError("fit() before predicting")
        query = np.asarray(query, dtype=float)
        n_dims = self._points.shape[1]
        if query.ndim != 2 or query.shape[1] != n_dims:
            raise ValueError(
                f"query must have shape (n_query, {n_dims}), got {query.shape}")
        k_star = self._kernel(query, self._points)
        alpha = self._k_inv @ self._mean
        mean = k_star @ alpha
        covariance_inv = self._k_inv - self._k_inv @ np.linalg.solve(
            self._posterior_precision, self._k_inv)
        var = self.output_scale - np.einsum("qi,ij,qj->q", k_star, covariance_inv, k_star)
        return mean, np.sqrt(np.clip(var, 0.0, None))
