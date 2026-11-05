"""Learner A's confidence, logged after every comparison.

After comparison k the session records, under A's own model and posterior:

- ``cal_p_win`` (P-hat_win(k)): the posterior-predictive probability that the identified
  assistance beats generic assistance in one validation duel. The weighted particle mean of
  ``lam_i / 2 + (1 - lam_i) Phi(Delta_i / ell_i^2)``, with
  ``Delta_i = 1/2 (|z_gen - z*_i|^2 - |z_hat_k - z*_i|^2)``. It is computed exactly as
  :meth:`~prompref.learning.psychometric.PsychometricLearner.update` computes its one-step
  predictive ``predicted_p_a``, with the identified assistance as A and generic assistance as B.
- ``cal_gain_nm`` (G-hat(k)): the weighted particle mean of
  ``RMS(tau_gen - tau*_i) - RMS(tau_k - tau*_i)`` in Nm, device-clipped curves. Its true
  counterpart is ``d_prior - distances[k - 1]``.
- ``cal_sigma_curve_nm`` (sigma_curve(k)): the weighted RMS, over particles, of the RMS distance
  in Nm between each particle's curve and the identified curve,
  ``sqrt(sum_i w_i RMS(tau*_i - tau_k)^2)``.

Definitions:
- ``z_hat_k`` is the identified assistance as delivered: ``space.to_z(learner.best)``, the
  path ``run_session`` scores (``tau_k = design.curve(learner.best)``).
- ``tau_gen`` is generic assistance, :func:`~prompref.experiment.metrics.prior_mean_curve`
  (the curve ``d_prior`` is the distance of). ``z_gen`` is its ProMP projection
  (``VonMisesBasis.project``, the least-squares weights the data prior is fitted with) in the
  posterior's normalized coordinates. It is not clipped to the trust search region.
- ``tau*_i`` is particle i's curve: ``space.design.curve(space.to_theta(z*_i))``, batched.

Nothing here draws a random number or changes the learner: the logging reads the posterior's
particles and weights. ``prior_mean_curve`` draws from its own generator (seed 0), the same draws
``d_prior`` already comes from, and touches no session stream. A session run with the logging
therefore reproduces the session without it, field for field.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..designs.device import DeviceDesign
from ..designs.promp import PrompS0
from ..learning.psychometric import PsychometricLearner
from ..learning.smc import log_likelihood
from ..learning.space import LearnerSpace
from ..population.curves import physics_clip
from .metrics import prior_mean_curve

#: The logged fields, in the order they are recorded.
FIELDS = ("cal_p_win", "cal_gain_nm", "cal_sigma_curve_nm")

#: Particle snapshots for the video logs: before the first comparison and after
#: these comparisons.
SNAPSHOT_KS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 40)


def check_space(space: LearnerSpace) -> None:
    """The logging needs a device-clipped, unstructured ProMP behind full-rank normalization."""
    design = space.design
    if not isinstance(design, DeviceDesign) or type(design.inner) is not PrompS0:
        raise ValueError("calibration logging needs a device-clipped PrompS0 design, got "
                         f"{type(design).__name__}")
    if space.d != design.n_params:
        raise ValueError(f"calibration logging needs full-rank whitening, got d={space.d} of "
                         f"{design.n_params}")


def generic_z(space: LearnerSpace, generic_curve: np.ndarray) -> np.ndarray:
    """Generic assistance in the space's normalized coordinates: the ProMP projection of its curve
    (least-squares basis weights, ``VonMisesBasis.project``), unclipped."""
    theta = space.design.basis.project(np.asarray(generic_curve, dtype=float))
    return space.to_z(theta)


def particle_curves(space: LearnerSpace, zstar: np.ndarray) -> np.ndarray:
    """``(N, n_phase)`` device-clipped curves of ``N`` points ``z*``, row i equal to
    ``space.design.curve(space.to_theta(zstar[i]))`` (batched; equal up to rounding)."""
    design = space.design
    theta = design.project(space.normalization.to_w(np.atleast_2d(np.asarray(zstar, dtype=float))))
    weights = design.to_weights(theta)
    return physics_clip(weights @ design.basis.design.T, design.limit)


def rms_rows(curves: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """RMS distance of each row of ``curves`` to ``curve``, as ``oracle.rms_distance`` takes it
    (the mean over every grid point)."""
    diff = np.asarray(curves, dtype=float) - np.asarray(curve, dtype=float)[None, :]
    return np.sqrt(np.mean(diff * diff, axis=1))


def predicted_win(x: np.ndarray, w: np.ndarray, z_hat: np.ndarray, z_gen: np.ndarray) -> float:
    """P-hat_win: the weighted mean over particles ``x`` (rows ``(z*, log ell, logit lam)``) of
    A's probability that ``z_hat`` is preferred over ``z_gen`` in one duel."""
    z_hat = np.asarray(z_hat, dtype=float)
    z_gen = np.asarray(z_gen, dtype=float)
    p = np.exp(log_likelihood(x, z_hat[None], z_gen[None], [True]))
    return float(np.asarray(w, dtype=float) @ p)


def predicted_gain(curves: np.ndarray, w: np.ndarray, generic: np.ndarray,
                   delivered: np.ndarray) -> float:
    """G-hat: the weighted mean over particle curves of RMS(generic - tau*_i) - RMS(delivered -
    tau*_i), in Nm."""
    return float(np.asarray(w, dtype=float) @ (rms_rows(curves, generic)
                                               - rms_rows(curves, delivered)))


def curve_spread(curves: np.ndarray, w: np.ndarray, delivered: np.ndarray) -> float:
    """sigma_curve: sqrt(sum_i w_i RMS(tau*_i - delivered)^2), in Nm."""
    d = rms_rows(curves, delivered)
    return float(np.sqrt(np.asarray(w, dtype=float) @ (d * d)))


class CalibrationMonitor:
    """Computes the three logged values once per comparison and hands them to ``run_session``
    through its ``scores`` hook (:meth:`scores`).

    ``run_session`` applies every score to the same ``delivered`` array after each comparison,
    so the values are computed on the first call that sees a new ``delivered`` and reused for the
    other two. ``snapshot``, if given, is called as ``snapshot(k, learner)`` at k = 0 (on
    construction, before the first comparison) and after each comparison k in ``snapshot_ks``;
    it must only read the learner.
    """

    def __init__(self, learner: PsychometricLearner, space: LearnerSpace,
                 generic_curve: np.ndarray | None = None, *,
                 snapshot: Callable | None = None, snapshot_ks=SNAPSHOT_KS) -> None:
        if not isinstance(learner, PsychometricLearner):
            raise ValueError(f"calibration logging reads A's posterior; got "
                             f"{type(learner).__name__}")
        if learner.space is not space:
            raise ValueError("the monitor's space must be the learner's own")
        check_space(space)
        self.learner, self.space = learner, space
        self.generic = np.asarray(prior_mean_curve(space.design) if generic_curve is None
                                  else generic_curve, dtype=float)
        self.z_gen = generic_z(space, self.generic)
        self.snapshot = snapshot
        self.snapshot_ks = tuple(int(k) for k in snapshot_ks)
        self.k = 0
        self._last = None
        self._values: dict[str, float] = {}
        if snapshot is not None and 0 in self.snapshot_ks:
            snapshot(0, learner)

    def values(self, delivered: np.ndarray) -> dict[str, float]:
        if delivered is not self._last:
            self._last = delivered
            self.k += 1
            post = self.learner.posterior
            w = post.weights()
            z_hat = self.space.to_z(self.learner.best())
            curves = particle_curves(self.space, post.x[:, : self.space.d])
            self._values = {
                "cal_p_win": predicted_win(post.x, w, z_hat, self.z_gen),
                "cal_gain_nm": predicted_gain(curves, w, self.generic, delivered),
                "cal_sigma_curve_nm": curve_spread(curves, w, delivered),
            }
            if self.snapshot is not None and self.k in self.snapshot_ks:
                self.snapshot(self.k, self.learner)
        return self._values

    def scores(self) -> dict[str, Callable]:
        """``run_session``'s ``scores``: one function of the delivered curve per field."""
        return {name: (lambda delivered, name=name: self.values(delivered)[name])
                for name in FIELDS}
