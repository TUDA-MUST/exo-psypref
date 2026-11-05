"""The Franks family's per-fold data prior and box.

Fitted to 2,000 curves drawn from the fold's fitted ProMP data prior -- which encodes that
fold's population -- because generating fresh population curves needs the Molinaro drive.
Each draw is fitted by multi-start least squares in the wide fit box; the prior is the mean and
covariance of the fits, and the learner box is the fits' 0.5-99.5th percentile range widened by
25 % on each side, inside the fit box, with both magnitudes capped at the ±12 Nm device limit.
``fit_stats`` records, per coordinate, the share of fits on each fit-box bound and which of the
three set each side of the box. Franks' own bounds are in their Supplementary Material, which
was not available.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..designs.franks import FIT_HIGH, FIT_LOW, PARAM_NAMES, FranksDesign
from ..population.curves import TORQUE_LIMIT_NM, physics_clip
from ..prior import Whitening


@dataclass
class FranksPrior:
    mu: np.ndarray
    sigma: np.ndarray
    low: np.ndarray
    high: np.ndarray
    fold: str = ""
    fit_stats: dict | None = None
    family: str = field(default="franks", init=False)

    def __post_init__(self) -> None:
        for name in ("mu", "sigma", "low", "high"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))

    def design(self) -> FranksDesign:
        return FranksDesign(self.low, self.high, prior=(self.mu, self.sigma))

    def whitening(self) -> Whitening:
        return Whitening(self.mu, self.sigma, self.mu.size)

    def space(self, radius: float):
        from ..learning.space import LearnerSpace

        return LearnerSpace(self.design(), self.whitening(), float(radius))

    def save(self, stem) -> None:
        payload = dict(mu=self.mu.tolist(), sigma=self.sigma.tolist(), low=self.low.tolist(),
                       high=self.high.tolist(), fold=self.fold, fit_stats=self.fit_stats)
        Path(f"{stem}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, stem) -> "FranksPrior":
        p = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
        return cls(p["mu"], p["sigma"], p["low"], p["high"], p.get("fold", ""), p.get("fit_stats"))


def fit_curve(design, target, starts) -> tuple[np.ndarray, float]:
    """The best least-squares fit of ``design``'s device-clipped curve to ``target`` from
    ``starts``, inside the design's box. Returns (theta, RMS in Nm)."""
    target = np.asarray(target, dtype=float)
    best = None
    failed = []
    for start in starts:
        try:
            res = least_squares(lambda th: physics_clip(design.curve(th)) - target,
                                design.project(np.asarray(start, dtype=float)),
                                bounds=(design.low, design.high), max_nfev=400)
        except ValueError as exc:
            # scipy's bounded trust-region solver can stop with "`x` is not within the trust
            # region." on a start; that start is skipped and the others decide (test P,
            # e-proj-pchip). A fit where no start fails is exactly what it always was.
            if "trust region" not in str(exc):
                raise
            failed.append(str(exc))
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise ValueError(f"every start of the surface fit failed: {failed}")
    return best.x, float(np.sqrt(2.0 * best.cost / target.size))


def _fit_one(target) -> tuple[np.ndarray, float]:
    design = FranksDesign()
    centre = 0.5 * (FIT_LOW + FIT_HIGH)
    return fit_curve(design, target, [design.initial_guess(target), centre])


def fit_franks_prior(promp_prior, fold: str, *, n_draws: int = 2000, seed=0, widen: float = 0.25,
                     workers: int = 1) -> FranksPrior:
    rng = np.random.default_rng(seed)
    promp = promp_prior.design()
    curves = [physics_clip(promp.curve(w)) for w in promp.sample_prior(rng, int(n_draws))]
    if workers > 1:
        with ProcessPoolExecutor(min(int(workers), 8)) as pool:
            fits = list(pool.map(_fit_one, curves, chunksize=16))
    else:
        fits = [_fit_one(c) for c in curves]
    thetas = np.array([t for t, _ in fits])
    rms = np.array([r for _, r in fits])
    low, high, sides = learner_box(thetas, widen)
    mu = thetas.mean(axis=0)
    sigma = np.cov(thetas, rowvar=False) + 1e-9 * np.eye(thetas.shape[1])
    stats = dict(n=int(len(rms)), median_rms=float(np.median(rms)),
                 share_above_0p5=float(np.mean(rms > 0.5)),
                 share_at_fit_low=dict(zip(PARAM_NAMES, map(float, share_at(thetas, FIT_LOW)))),
                 share_at_fit_high=dict(zip(PARAM_NAMES, map(float, share_at(thetas, FIT_HIGH)))),
                 quantiles={name: [float(v) for v in q] for name, q in
                            zip(PARAM_NAMES, np.percentile(thetas, QUANTILES, axis=0).T)},
                 box_sides=sides)
    return FranksPrior(mu, sigma, low, high, fold=fold, fit_stats=stats)


#: Tolerance, in internal coordinates, for "at a bound" (1e-4 in a log coordinate is 0.01 %).
AT_BOUND_TOL = 1e-4
#: Percentiles of the fits recorded per coordinate in ``fit_stats["quantiles"]``.
QUANTILES = (0.5, 5.0, 25.0, 50.0, 75.0, 95.0, 99.5)


def share_at(thetas, bound) -> np.ndarray:
    """Per coordinate, the share of ``thetas`` within AT_BOUND_TOL of ``bound``."""
    return np.mean(np.abs(np.asarray(thetas, dtype=float) - np.asarray(bound, dtype=float))
                   <= AT_BOUND_TOL, axis=0)


def learner_box(thetas, widen: float = 0.25) -> tuple[np.ndarray, np.ndarray, dict]:
    """our design notes's box: the fits' 0.5-99.5th percentile range widened by ``widen`` of itself on
    each side, clipped to the fit box, with both peak magnitudes capped at the device limit.
    Also returns, per side, what set it: "percentile", "fit box" or "device cap"."""
    lo, hi = np.percentile(np.asarray(thetas, dtype=float), [0.5, 99.5], axis=0)
    span = hi - lo
    want_low, want_high = lo - widen * span, hi + widen * span
    low = np.maximum(want_low, FIT_LOW)
    high = np.minimum(want_high, FIT_HIGH)
    cap = np.full(high.shape, np.inf)
    cap[6:8] = np.log(TORQUE_LIMIT_NM)
    sides = {}
    for i, name in enumerate(PARAM_NAMES):
        sides[f"{name} low"] = "fit box" if want_low[i] < FIT_LOW[i] else "percentile"
        sides[f"{name} high"] = ("device cap" if high[i] > cap[i] else
                                 "fit box" if want_high[i] > FIT_HIGH[i] else "percentile")
    return low, np.minimum(high, cap), sides
