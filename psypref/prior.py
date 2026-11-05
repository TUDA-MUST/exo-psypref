"""The prior over torque profiles: a Gaussian over the weights of a ProMP, from walking data or your own curves.

A curve is sampled at 101 points over the gait cycle (0-100 %), with flexion torque positive.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from prompref.basis import VonMisesBasis
from prompref.designs.device import DeviceDesign
from prompref.experiment import amp_free as AF
from prompref.learning.arms import ARMS, Arm
from prompref.learning.space import LearnerSpace
from prompref.prior import Normalization

DATA = Path(__file__).resolve().parent / "data"


def _loguniform_moments(lo: float, hi: float) -> tuple[float, float]:
    """The first two moments of a peak drawn log-uniformly from [lo, hi]."""
    if not 0.0 < lo < hi:
        raise ValueError(f"a peak range needs 0 < low < high, got ({lo}, {hi})")
    span = math.log(hi / lo)
    return (hi - lo) / span, (hi ** 2 - lo ** 2) / (2.0 * span)


def _with_floor(sigma: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    """Few or similar curves leave some directions without variance. They stay nearly fixed, but the prior stays usable."""
    vals, vecs = np.linalg.eigh(0.5 * (sigma + sigma.T))
    return (vecs * np.maximum(vals, floor * vals.max())) @ vecs.T


def _shape_moments(w_minus: np.ndarray, w_plus: np.ndarray) -> dict:
    return dict(m_minus=w_minus.mean(axis=0), m_plus=w_plus.mean(axis=0), s_mm=w_minus.T @ w_minus / len(w_minus),
                s_pp=w_plus.T @ w_plus / len(w_plus), s_mp=w_minus.T @ w_plus / len(w_minus))


def _weight_moments(m: dict, peak_range=None, level=None) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of ``a w_minus + b w_plus``, with both peaks at ``level`` or log-uniform in ``peak_range``."""
    if level is not None:
        a1, a2 = float(level), float(level) ** 2
    else:
        a1, a2 = _loguniform_moments(*peak_range)
    mu = a1 * (m["m_minus"] + m["m_plus"])
    second = a2 * (m["s_mm"] + m["s_pp"]) + a1 * a1 * (m["s_mp"] + m["s_mp"].T)
    return mu, second - np.outer(mu, mu)


class Prior:
    """What is known before the first comparison.

    Build one with :meth:`from_walking`, :meth:`from_curves` or :meth:`from_moments`. Give ``peak_range=(low, high)`` in Nm
    when both peaks may vary, and ``level`` when both stay fixed. ``width`` widens the prior and its search region. It
    costs little up to about 2, and much at 3 (see docs/choosing_a_prior.md).
    """

    def __init__(self, mu, sigma, *, peak_range=None, level=None, limit=None, width: float = 1.0, generic=None,
                 source: str = "moments") -> None:
        self.mu, self.sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
        if self.mu.ndim != 1 or self.sigma.shape != (self.mu.size, self.mu.size):
            raise ValueError(f"mu needs K values and sigma K x K, got {self.mu.shape} and {self.sigma.shape}")
        if peak_range is not None and level is not None:
            raise ValueError("give a peak range or a level, not both")
        if peak_range is None and level is None and limit is None:
            raise ValueError("give peak_range, level or limit: the device's torque limit must be known")
        if width <= 0.0:
            raise ValueError(f"width must be positive, got {width}")
        self.peak_range = None if peak_range is None else (float(peak_range[0]), float(peak_range[1]))
        self.level = None if level is None else float(level)
        self.limit = limit
        self.width = float(width)
        self.generic = None if generic is None else np.asarray(generic, dtype=float)
        self.source = source

    @property
    def n_weights(self) -> int:
        return self.mu.size

    def __repr__(self) -> str:
        where = (f"peak_range={self.peak_range}" if self.peak_range else
                 f"level={self.level}" if self.level is not None else f"limit={self.limit}")
        return f"Prior({self.source}, {self.n_weights} weights, {where}, width={self.width:g})"

    @classmethod
    def from_walking(cls, peak_range=None, level=None, *, width: float = 1.0, n_weights: int = 8,
                     leave_out: str | None = None) -> "Prior":
        """Hip moments of five able-bodied subjects at 1.1 m/s, delayed and shifted like self-tuned hip profiles.

        ``leave_out`` drops one subject (AB30-AB34), as the paper does for a simulated user of that subject.
        """
        if (peak_range is None) == (level is None):
            raise ValueError("give a peak range or a level")
        path = DATA / f"walking_moments_k{n_weights}.npz"
        if not path.exists():
            raise ValueError(f"walking priors exist for 8, 12, 16 and 24 weights, not {n_weights}")
        key = leave_out or "all"
        with np.load(path) as z:
            if f"{key}_m_minus" not in z.files:
                raise ValueError(f"leave_out must be one of AB30-AB34, got {leave_out!r}")
            m = {k: z[f"{key}_{k}"] for k in ("m_minus", "m_plus", "s_mm", "s_pp", "s_mp")}
            parts = z[f"{key}_generic_minus"], z[f"{key}_generic_plus"]
        mu, sigma = _weight_moments(m, peak_range, level)
        centre = float(level) if level is not None else math.sqrt(peak_range[0] * peak_range[1])
        return cls(mu, sigma, peak_range=peak_range, level=level, width=width, generic=AF.compose(centre, centre, *parts),
                   source="walking data")

    @classmethod
    def from_curves(cls, curves, peak_range=None, level=None, *, limit=None, width: float = 1.0,
                    n_weights: int = 8) -> "Prior":
        """Your own curves, one per row, at 101 points over the gait cycle.

        With ``peak_range`` or ``level``, only their shapes count: each curve needs an extension and a flexion part, and
        the peaks come from the range or the level. Otherwise the curves count as they are, including their torques, which
        also suits assistance in one direction only. Then ``limit`` gives the device's torque limit.
        """
        curves = np.asarray(curves, dtype=float)
        if curves.ndim != 2 or curves.shape[1] != 101:
            raise ValueError(f"curves must be n x 101, got {curves.shape}")
        if curves.shape[0] < 2 * n_weights:
            raise ValueError(f"{curves.shape[0]} curves cannot fix a covariance of {n_weights} weights; "
                             f"give at least {2 * n_weights}, or use fewer weights")
        basis = VonMisesBasis(n_weights)
        if peak_range is None and level is None:
            w = basis.project(curves.T).T
            return cls(w.mean(axis=0), _with_floor(np.cov(w, rowvar=False)), limit=limit, width=width,
                       generic=curves.mean(axis=0), source=f"{curves.shape[0]} curves")
        parts = [AF.unit_parts(c) for c in curves]
        s_minus, s_plus = np.array([p[0] for p in parts]), np.array([p[1] for p in parts])
        m = _shape_moments(basis.project(s_minus.T).T, basis.project(s_plus.T).T)
        mu, sigma = _weight_moments(m, peak_range, level)
        centre = float(level) if level is not None else math.sqrt(peak_range[0] * peak_range[1])
        m_minus, m_plus = s_minus.mean(axis=0), s_plus.mean(axis=0)
        generic = AF.compose(centre, centre, m_minus / -m_minus.min(), m_plus / m_plus.max())
        return cls(mu, _with_floor(sigma), peak_range=peak_range, level=level, limit=limit, width=width, generic=generic,
                   source=f"{curves.shape[0]} curve shapes")

    @classmethod
    def from_moments(cls, mu, sigma, peak_range=None, level=None, *, limit=None, width: float = 1.0, generic=None) -> "Prior":
        """A prior you already have: the mean and covariance of the ProMP weights."""
        return cls(mu, sigma, peak_range=peak_range, level=level, limit=limit, width=width, generic=generic)

    def widened(self, width: float) -> "Prior":
        """The same prior, ``width`` times wider, with a search region that grows along."""
        return Prior(self.mu, self.sigma, peak_range=self.peak_range, level=self.level, limit=self.limit, width=width,
                     generic=self.generic, source=self.source)

    def _arm(self) -> str:
        if self.width == 1.0:
            return "tight"
        name = f"width_{self.width:g}"
        ARMS.setdefault(name, Arm(name, scale=self.width ** 2))
        return name

    def _space(self) -> LearnerSpace:
        arm = self._arm()
        basis = None if self.n_weights == 8 else VonMisesBasis(self.n_weights)
        if self.level is not None:
            design = DeviceDesign(AF.FixedLevelDesign(self.level, basis=basis), limit=self.level + 1e-9)
            return LearnerSpace(design, Normalization(self.mu, self.sigma, self.mu.size), ARMS[arm].radius(self.mu.size))
        limit = self.peak_range[1] if self.peak_range is not None else float(self.limit)
        return AF.promp_space(self.mu, self.sigma, limit=limit, basis=basis, arm=arm)
