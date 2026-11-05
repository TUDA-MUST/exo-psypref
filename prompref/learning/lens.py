"""The ProMP lens.

Learner A stays unchanged in the fold's ProMP space, where its bowl likelihood is a fair model of
curve-distance preferences.
The wearer only ever sees curves of an interpretable surface family: each duel end A proposes is
replaced by its best surface fit, A is told the ProMP projection of the curve actually shown,
and the delivered curve is the surface fit to A's optimum. No random draws.
"""

from __future__ import annotations

import time

import numpy as np

from ..population.curves import physics_clip
from ..priors.franks_prior import fit_curve


class SurfaceFitter:
    """Least-squares surface fits from three starts -- the surface's own initial guess, the
    previous fit, and the surface prior's mean -- keeping the best."""

    def __init__(self, surface, start_mean) -> None:
        self.surface = surface
        self.start_mean = np.asarray(start_mean, dtype=float)
        self.last: np.ndarray | None = None
        #: RMS (Nm) between the last fit's device-clipped curve and its target.
        self.last_rms: float | None = None
        self.n_fits = 0
        self.seconds = 0.0

    def fit(self, target) -> np.ndarray:
        started = time.perf_counter()
        target = np.asarray(target, dtype=float)
        starts = [self.surface.initial_guess(target), self.start_mean]
        if self.last is not None:
            starts.append(self.last)
        theta, rms = fit_curve(self.surface, target, starts)
        self.last, self.last_rms = theta, rms
        self.n_fits += 1
        self.seconds += time.perf_counter() - started
        return theta


class LensLearner:
    def __init__(self, inner, inner_space, surface, fitter: SurfaceFitter) -> None:
        self.inner = inner
        self.inner_space = inner_space
        self.surface = surface
        self.fitter = fitter
        self._basis = inner_space.design.inner.basis.design
        self._pinv = np.linalg.pinv(self._basis)
        self._residuals: list[float] = []
        # design §8 risk table: how far each shown curve is from the one A proposed (a poor fit or a
        # local minimum), and how often the ProMP box clips the projection A is told.
        self._proposed_shown: list[float] = []
        self._clipped = 0

    def _shown_curve(self, theta) -> np.ndarray:
        return physics_clip(self.surface.curve(theta))

    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        shown = []
        for w in self.inner.propose():
            shown.append(self.fitter.fit(self.inner_space.design.curve(w)))
            self._proposed_shown.append(float(self.fitter.last_rms))
        return shown[0], shown[1]

    def update(self, theta_a, theta_b, response: int) -> None:
        projected = []
        for theta in (theta_a, theta_b):
            shown = self._shown_curve(theta)
            raw = self._pinv @ shown
            w = self.inner_space.design.project(raw)
            self._clipped += int(np.any(w != raw))
            self._residuals.append(float(np.sqrt(np.mean((self._basis @ w - shown) ** 2))))
            projected.append(w)
        self.inner.update(projected[0], projected[1], response)

    def best(self) -> np.ndarray:
        return self.fitter.fit(self.inner_space.design.curve(self.inner.best()))

    def should_stop(self) -> bool:
        stop = getattr(self.inner, "should_stop", None)
        return bool(stop()) if stop is not None else False

    def posterior_summary(self) -> dict:
        summary = self.inner.posterior_summary()
        extras = dict(summary.get("extras", {}))
        gaps = self._proposed_shown
        extras.update(lens_shown_residual_median=float(np.median(self._residuals)) if self._residuals else None,
                      lens_n_fits=int(self.fitter.n_fits), lens_fit_s=float(self.fitter.seconds),
                      lens_proposed_shown_rms_median=float(np.median(gaps)) if gaps else None,
                      lens_proposed_shown_rms_p95=float(np.percentile(gaps, 95)) if gaps else None,
                      lens_projection_clipped=int(self._clipped))
        return {**summary, "extras": extras}
