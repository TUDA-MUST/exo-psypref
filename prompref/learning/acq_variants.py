"""Three single changes to learner A's acquisition. Each is an opt-in option of
:class:`~prompref.learning.psychometric.PsychometricLearner`, off by default:

- **A-curve** (``acquisition="curve"``). EIG ranks duels by the expected reduction of
  ``log det Cov(z*)``, D-optimality in z. A-curve ranks the same candidates by the expected
  reduction of ``trace(M Cov(z*))``, A-optimality in curve space, where ``M`` is the curve
  metric in whitened z (:func:`curve_metric`): ``(z1 - z2)^T M (z1 - z2)`` is the mean squared
  difference of the two (undevice-clipped) torque curves. The probability floor, the ridge and
  the heavy-tail trim are EIG's (:mod:`prompref.learning.acquisition`).
- **A-width** (``widths="target"``). For each candidate centre and direction the duel's width
  is the one at which the posterior-predicted probability of the more likely answer is 0.80
  (:data:`TARGET_P`), found by a 1-D root find inside the search region
  (:func:`target_width_duel`). It replaces the fixed ladder {0.25, 0.5, 1, 2, 4} ell-hat, so
  there is one candidate per (centre, direction) instead of five.
- **A-sub** (``direction_set="curve_subspace"``). Candidate directions are only the posterior's
  principal axes that carry 95 % (:data:`SUBSPACE_SHARE`) of its curve-space variance
  (:func:`curve_subspace_axes`), plus the 3 random directions, in place of all d posterior axes.

The reading of A-width's rule: "the
probability of the more likely answer" is taken per posterior particle -- the wearer's
decisiveness the particle predicts, ``max(P_i, 1 - P_i)`` -- and averaged over the posterior,
``sum_i w_i max(P_i(A), 1 - P_i(A))``. The other reading, the more likely answer of the
posterior's mixture prediction ``max(pbar, 1 - pbar)``, leaves a duel centred on the posterior
mean without any width: by symmetry its mixture prediction stays near 0.5 at every width.

Nothing here edits the gated ``acquisition.py`` or ``smc.py``; this module only imports them.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, ndtr

from .acquisition import P_FLOOR, RIDGE, answer_probabilities, trim_mask
from .candidates import _exit, truncate_duel
from .space import LearnerSpace

#: A-width's target: the posterior-predicted probability of the more likely answer.
TARGET_P = 0.80
#: A-sub's share of the posterior's curve-space variance the kept axes must carry.
SUBSPACE_SHARE = 0.95
#: A-width: grid points on the asymmetrically truncated stretch [h_lo, h_hi] of the requested
#: half-width, where the prediction need not be monotone (see :func:`target_width_duel`).
WIDTH_GRID = 16

ACQUISITIONS = ("eig", "curve")
WIDTHS = ("ladder", "target")
DIRECTION_SETS = ("posterior", "curve_subspace")

#: The planned variants as learner options: one change each (section 9a).
VARIANTS = {"curve": dict(acquisition="curve"),
            "width": dict(widths="target"),
            "sub": dict(direction_set="curve_subspace")}


def curve_metric(space: LearnerSpace) -> np.ndarray:
    """``M = Lambda^1/2 U^T (B^T B / 101) U Lambda^1/2``, the curve metric in ``space``'s z.

    ``B`` is the ProMP basis (101 phase points x K), ``U`` and ``Lambda`` the whitening's kept
    eigenvectors and eigenvalues. No device clipping: the metric is that of the family's own
    linear curves. Only a :class:`~prompref.designs.promp.PrompS0` (behind a device or not) has
    one; its weights are its parameters, so its curve is linear in z.
    """
    from ..designs.device import DeviceDesign
    from ..designs.promp import PrompS0

    design = space.design.inner if isinstance(space.design, DeviceDesign) else space.design
    if type(design) is not PrompS0:
        raise ValueError(f"the curve metric needs a PrompS0 design (linear in its weights); got "
                         f"{type(design).__name__}")
    wh = space.whitening
    j = wh.eigenvectors[:, : wh.d] * np.sqrt(wh.eigenvalues[: wh.d])[None, :]   # U Lambda^1/2
    metric = j.T @ design.basis.gram @ j
    return 0.5 * (metric + metric.T)


def _trace_m_cov(zs: np.ndarray, quad: np.ndarray, metric: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
    """``trace(M Cov)`` of ``zs`` for each weight column, ``(C,)``, with EIG's ridge on Cov.

    ``Cov = E_w[z z^T] - mean mean^T`` exactly as :func:`acquisition._log_det_cov` builds it, so
    ``trace(M Cov) = E_w[z^T M z] - mean^T M mean``; the ridge adds ``RIDGE * trace(M)``."""
    mean = zs.T @ weights                                   # (d, C)
    return (quad @ weights - np.einsum("ic,ij,jc->c", mean, metric, mean)
            + RIDGE * float(np.trace(metric)))


def expected_curve_variance_reduction(x: np.ndarray, w: np.ndarray, d: int, z_a: np.ndarray,
                                      z_b: np.ndarray, metric: np.ndarray, *,
                                      trim: float | None = None) -> np.ndarray:
    """A-curve's score for each candidate pair, shape ``(C,)``: the expected reduction of
    ``trace(M Cov(z*))`` over the two answers, ``T_0 - p T_A - (1 - p) T_B``.

    The answer probability ``p`` is floored at ``P_FLOOR`` and the branch weights are formed as
    in :func:`~prompref.learning.acquisition.expected_information_gain`; ``trim`` likewise."""
    if trim is not None:
        keep = trim_mask(x, w, d, trim)
        x, w = np.asarray(x, dtype=float)[keep], np.asarray(w, dtype=float)[keep]
        w = w / w.sum()
    metric = np.asarray(metric, dtype=float)
    zs = np.asarray(x, dtype=float)[:, :d]
    w = np.asarray(w, dtype=float)
    quad = np.einsum("ni,ij,nj->n", zs, metric, zs)
    weighted = w[:, None] * answer_probabilities(x, z_a, z_b)
    p = np.clip(weighted.sum(axis=0), P_FLOOR, 1.0 - P_FLOOR)
    t_prior = _trace_m_cov(zs, quad, metric, w[:, None])[0]
    t_a = _trace_m_cov(zs, quad, metric, weighted / p)
    t_b = _trace_m_cov(zs, quad, metric, (w[:, None] - weighted) / (1.0 - p))
    return t_prior - p * t_a - (1.0 - p) * t_b


def _sym_power(metric: np.ndarray, power: float) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(metric, dtype=float))
    if vals.min() <= 0.0:
        raise ValueError(f"the curve metric must be positive definite; smallest eigenvalue "
                         f"{vals.min():.3e}")
    return (vecs * vals**power) @ vecs.T


def curve_subspace_axes(cov: np.ndarray, metric: np.ndarray,
                        share: float = SUBSPACE_SHARE) -> list[np.ndarray]:
    """A-sub's directions: the principal axes of the posterior's curve-space covariance
    ``M^1/2 Cov M^1/2`` that together carry ``share`` of its trace (the fewest, largest first),
    mapped back to z by ``M^-1/2`` and normalised to unit length in z."""
    if not 0.0 < share <= 1.0:
        raise ValueError(f"share must be in (0, 1], got {share}")
    root, inv_root = _sym_power(metric, 0.5), _sym_power(metric, -0.5)
    curve_cov = root @ np.asarray(cov, dtype=float) @ root
    vals, vecs = np.linalg.eigh(0.5 * (curve_cov + curve_cov.T))
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]
    if vals.sum() <= 0.0:
        raise ValueError("the posterior has no curve-space variance")
    n_keep = int(np.argmax(np.cumsum(vals) / vals.sum() >= share)) + 1
    dirs = inv_root @ vecs[:, :n_keep]
    return [dirs[:, i] / np.linalg.norm(dirs[:, i]) for i in range(n_keep)]


def predicted_decisiveness(x: np.ndarray, w: np.ndarray, z_a: np.ndarray,
                           z_b: np.ndarray) -> float:
    """The posterior-predicted probability of the more likely answer: ``sum_i w_i max(P_i(A), 1 - P_i(A))`` for one pair."""
    p = answer_probabilities(x, np.asarray(z_a)[None], np.asarray(z_b)[None])[:, 0]
    return float(np.asarray(w, dtype=float) @ np.maximum(p, 1.0 - p))


def target_width_duel(x: np.ndarray, w: np.ndarray, c_own: np.ndarray, u: np.ndarray,
                      own: LearnerSpace, duel: LearnerSpace, *, z_worn=None, r_step=None,
                      target: float = TARGET_P) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """A-width's duel on the line ``c_own +/- h u``: ``(z_a, z_b, capped)``, or ``None`` when the
    region leaves no width at all (as the ladder drops a zero-width candidate).

    The ends at a requested half-width ``h`` are :func:`~prompref.learning.candidates.truncate_duel`'s,
    so they stay inside the trust ball (and a step ball, if one is given). ``f(h)`` is
    :func:`predicted_decisiveness` of that pair; ``f(0) = 0.5`` exactly. With ``h_lo`` and
    ``h_hi`` the smaller and larger exit distances along ``+u`` and ``-u``:

    - on ``[0, h_lo]`` the duel is symmetric and ``f`` strictly increasing, so if
      ``f(h_lo) >= target`` the root there is unique (``brentq``);
    - otherwise ``f`` is scanned at :data:`WIDTH_GRID` points on ``(h_lo, h_hi]``, where one end
      is fixed and ``f`` need not be monotone, and ``brentq`` finds the first crossing;
    - if the target is not reached inside the region, the duel is the one at which ``f`` is
      largest among ``h_lo`` and the grid, and ``capped`` is True.
    """
    if r_step is not None and z_worn is None:
        raise ValueError("a step region needs z_worn, the duel-space point it is centred on")
    region = dict(z_worn=z_worn, r_step=r_step)
    c_duel = own.transfer(c_own, duel)
    v = own.transfer(u, duel) - own.transfer(np.zeros(own.d), duel)
    balls = [(np.zeros(duel.d), duel.radius)]
    if r_step is not None:
        balls.append((np.asarray(z_worn, dtype=float), float(r_step)))
    plus = min(_exit(c_duel, v, ball) for ball in balls)
    minus = min(_exit(c_duel, -v, ball) for ball in balls)
    h_lo, h_hi = min(plus, minus), max(plus, minus)
    if h_hi < 1e-9:
        return None

    def ends(h):
        return truncate_duel(c_own, u, h, own, duel, **region)

    # predicted_decisiveness of ends(h), in closed form along the line (the same likelihood,
    # tested to 1e-12): with ends c + a u and c - b u, a = min(h, plus), b = min(h, minus),
    # Delta_i = (a + b) (u. (z*_i - c) - (a - b) |u|^2 / 2).
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    d = own.d
    s = (x[:, :d] - np.asarray(c_own, dtype=float)) @ np.asarray(u, dtype=float)
    uu = float(np.dot(u, u))
    inv_ell2 = np.exp(-2.0 * x[:, d])
    lam = expit(x[:, d + 1])

    def f(h):
        a, b = min(h, plus), min(h, minus)
        delta = (a + b) * (s - 0.5 * (a - b) * uu)
        return float(w @ (0.5 * lam + (1.0 - lam) * ndtr(np.abs(delta) * inv_ell2))) - target

    if h_lo > 0.0:
        f_lo = f(h_lo)
        if f_lo >= 0.0:
            h = brentq(f, 0.0, h_lo) if f_lo > 0.0 else h_lo
            a, b = ends(h)
            return a, b, False
    else:
        f_lo = -np.inf
    grid = h_lo + (h_hi - h_lo) * np.arange(1, WIDTH_GRID + 1) / WIDTH_GRID
    values = [f(h) for h in grid] if h_hi - h_lo > 1e-9 else []
    previous = h_lo
    for h, value in zip(grid, values):
        if value >= 0.0:
            a, b = ends(brentq(f, previous, h) if value > 0.0 else h)
            return a, b, False
        previous = h
    best = h_lo if not values or f_lo >= max(values) else grid[int(np.argmax(values))]
    if best <= 0.0:
        best = h_hi
    a, b = ends(best)
    if np.linalg.norm(a - b) < 1e-9:
        return None
    return a, b, True
