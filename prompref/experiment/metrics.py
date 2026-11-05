"""What a run is scored on.

The primary question is about convergence *speed* as well as quality, so every
configuration reports a pair: ``N90`` for how fast, ``d_floor`` for how good it could
possibly get. Neither is reported without the other.

Delivered distance -- never regret -- is the cross-family score. Regret is measured
against the best member of the candidate family, so it penalises a family that fits
better; the sibling project measured exactly that effect.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from ..user.simulated_user import rms_distance


def design_floor(design, ideal_nm: np.ndarray, n_starts: int = 40,
                 seed: int = 0) -> float:
    """Best distance this family can reach, by multi-start local optimisation."""
    rng = np.random.default_rng(seed)
    bounds = list(zip(design.low, design.high))

    def objective(theta: np.ndarray) -> float:
        return rms_distance(design.curve(design.project(theta)), ideal_nm)

    best = np.inf
    for start in design.sample_prior(rng, n_starts):
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        best = min(best, float(result.fun))
    return best


def prior_mean_distance(design, ideal_nm: np.ndarray, n: int = 2000,
                        seed: int = 0) -> float:
    """``|| E[tau(theta)] - tau* ||_RMS`` -- the distance of the **mean curve**.

    Note what this is *not*: the distance of the curve at the mean parameter. The two
    coincide only for families linear in their parameters, and our design notes requires
    this one. Since ``d_prior`` is ``N90``'s denominator and the error's sign flips by
    family, using the curve at the mean parameter would bias precisely the two ablations
    built on cross-family contrast -- ``families`` and ``linearity``.

    Averaging is over prior draws with a fixed seed, so the estimate is reproducible;
    ``project`` is applied because the prior's support is the feasible set.
    """
    rng = np.random.default_rng(seed)
    theta = design.sample_prior(rng, int(n))
    curves = np.array([design.curve(design.project(t)) for t in theta])
    return rms_distance(curves.mean(axis=0), ideal_nm)


def prior_mean_curve(design, n: int = 2000, seed: int = 0) -> np.ndarray:
    """The **mean curve** whose RMS distance :func:`prior_mean_distance` returns: generic
    assistance. The same draws, projection and average, so
    ``rms_distance(prior_mean_curve(d), ideal) == prior_mean_distance(d, ideal)`` exactly.
    ``prior_mean_distance`` is left as it was, so no stored ``d_prior`` can move."""
    rng = np.random.default_rng(seed)
    theta = design.sample_prior(rng, int(n))
    curves = np.array([design.curve(design.project(t)) for t in theta])
    return curves.mean(axis=0)


def n90(distances: Sequence[float], d_prior: float, d_floor: float) -> int | None:
    """Comparisons needed to close 90 % of the achievable gap, or ``None`` if never.

    ``None`` is right-censoring and must be carried through the analysis rather than
    dropped: averaging only the runs that converged would flatter every method.
    """
    threshold = d_floor + 0.1 * (d_prior - d_floor)
    for i, d in enumerate(distances):
        if d <= threshold + 1e-12:
            return i + 1
    return None


#: Below this (Nm RMS) the best-profile distance is treated as degenerate and ``delivered_over_floor``
#: is not reported at all. A best-profile distance this small means the ideal is *exactly* reachable by
#: the family -- which is true of any stand-in ideal drawn from the family being tuned,
#: as ``sweep._ideal_for`` currently does.
DEGENERATE_FLOOR_NM = 1e-6


def summarise_run(distances: Sequence[float], d_prior: float,
                  d_floor: float) -> dict:
    """One run's scores, ready to be written as a JSONL row.

    ``delivered_over_floor`` is ``None``, not ``inf`` and not a large finite number,
    whenever ``d_floor`` is degenerate. With a best-profile distance at ~1e-9 the ratio is nothing but
    the optimiser's residual -- it came out around 1e8 -- and a number like that is worse
    than a missing one, because it looks like a measurement and can be averaged,
    plotted and compared. ``None`` cannot. The ratio becomes meaningful again once a
    shared, family-independent ideal exists (Plan 2); see ``sweep._ideal_for``.
    """
    distances = [float(d) for d in distances]
    reached = n90(distances, d_prior, d_floor)
    delivered = distances[-1] if distances else float(d_prior)
    degenerate = float(d_floor) < DEGENERATE_FLOOR_NM
    return {
        "delivered": delivered,
        "delivered_over_floor": None if degenerate else delivered / float(d_floor),
        "d_prior": float(d_prior),
        "d_floor": float(d_floor),
        "n90": reached,
        "censored": reached is None,
        # Deliberately NOT "n_comparisons": a sweep row is `dict(point.__dict__,
        # **record)`, so a record key of that name would overwrite the Point's own
        # field. They are equal today, but the moment an early-stopping rule lands they
        # diverge -- and such a row's key could never match its Point again, so the
        # point would re-run on every resume, forever. The Point's field stays
        # authoritative; this records how many comparisons actually ran.
        "n_comparisons_run": len(distances),
    }
