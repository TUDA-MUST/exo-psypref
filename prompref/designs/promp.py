"""The three structure levels of the ProMP representation.

S0 carries no structure at all: the prior does the work. S1 hard-codes the two bouts'
sign and phase windows by reparameterisation, which costs nothing at inference time. S2
hard-codes the two-bout shape and leaves only the within-bout detail free -- and with
zero shape weights it *is* the bump family, which is the Phase 1 gate.

The reason to encode structure in the parameterisation rather than the prior: a Gaussian
prior cannot say "there are exactly two bouts, extension first". It can only make
three-bout curves unlikely.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..basis import VonMisesBasis
from .base import BoxDesign
from .bump import LOW as BUMP_LOW
from .bump import HIGH as BUMP_HIGH
from .bump import BumpDesign

#: Phase windows, %GC, for the sign-constrained level.
EXT_WINDOW = (0.0, 40.0)
FLEX_WINDOW = (45.0, 85.0)


@lru_cache(maxsize=8)
def _fit_weight_prior(n_basis: int, overlap: float) -> tuple[np.ndarray, np.ndarray]:
    """Cache key is the basis's scalars -- ``VonMisesBasis`` holds arrays and is unhashable."""
    from .trapezoid import TrapezoidDesign

    from ..prior import sigma_from_design

    mu, sigma = sigma_from_design(TrapezoidDesign(), VonMisesBasis(n_basis, overlap),
                                  np.random.default_rng(0), n=20000)
    # The cache hands this same array out to every design built in a sweep worker.
    # Mutating one design's prior in place would corrupt every other design's.
    mu.flags.writeable = False
    sigma.flags.writeable = False
    return mu, sigma


def default_weight_prior(basis: VonMisesBasis) -> tuple[np.ndarray, np.ndarray]:
    """Weight prior shaped like the semi-trapezoid family -- the design's ``Sigma_param``.

    Sampling the parametric family and projecting onto the basis is what gives the ProMP
    "the shape of a standard parametric function": the prior concentrates on curves that
    family could have produced, while the representation stays free to leave it.

    Cached because fitting it projects thousands of curves, and every design in a sweep
    worker wants the same one.
    """
    return _fit_weight_prior(basis.n_basis, basis.overlap)


def _softplus(x: np.ndarray) -> np.ndarray:
    """Smooth, strictly positive, and numerically safe for large |x|."""
    return np.logaddexp(0.0, x)


class PrompS0(BoxDesign):
    """Unstructured ProMP: one weight per basis function.

    Draws come from the fitted prior ``N(mu_w, Sigma_w)``, not from the box. That is the
    difference between a ProMP and a box of basis weights, and it is not cosmetic: with
    box-uniform draws the best of forty random curves was measurably *worse* than the
    prior mean, because uniform weights on +/-14 Nm produce wildly oscillating curves
    while the mean is near-flat. A prior fitted to a parametric family instead produces
    plausible smooth two-bout curves, which is the space the search is supposed to
    explore.

    The box survives only as a feasibility clip for :meth:`project`; it no longer
    describes where samples come from.
    """

    def __init__(self, basis: VonMisesBasis | None = None, bound_nm: float = 14.0,
                 prior: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        self.basis = basis or VonMisesBasis()
        k = self.basis.n_basis
        super().__init__(np.full(k, -bound_nm), np.full(k, bound_nm))
        self.param_names = tuple(f"w{i}" for i in range(k))
        self.prior = prior if prior is not None else default_weight_prior(self.basis)

    def to_weights(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float)

    def curve(self, theta: np.ndarray) -> np.ndarray:
        return self.basis.curve(self.to_weights(theta))

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw from ``N(mu_w, Sigma_w)``, clipped into the feasibility box."""
        mu, sigma = self.prior
        return self.project(rng.multivariate_normal(mu, sigma, size=n))


class PrompS1(PrompS0):
    """Region- and sign-constrained ProMP, enforced by reparameterisation."""

    def to_weights(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        centres = self.basis.centres
        ext = (centres >= EXT_WINDOW[0]) & (centres < EXT_WINDOW[1])
        flex = (centres >= FLEX_WINDOW[0]) & (centres < FLEX_WINDOW[1])
        w = np.zeros_like(theta)
        w[..., ext] = -_softplus(theta[..., ext])
        w[..., flex] = +_softplus(theta[..., flex])
        return w


class PrompS2(BoxDesign):
    """Two-bout structure hard-coded; within-bout shape left free.

    Parameters are the six :class:`BumpDesign` parameters followed by ``n_shape``
    within-bout weights. With ``n_shape = 0`` the curve is exactly a bump curve.
    """

    def __init__(self, basis: VonMisesBasis | None = None, n_shape: int = 0,
                 shape_bound_nm: float = 4.0) -> None:
        self.basis = basis or VonMisesBasis()
        self.bump = BumpDesign()
        self.n_shape = int(n_shape)
        if self.n_shape > self.basis.n_basis:
            raise ValueError(
                f"n_shape ({self.n_shape}) cannot exceed the basis size "
                f"({self.basis.n_basis})"
            )
        low = np.concatenate([BUMP_LOW, np.full(self.n_shape, -shape_bound_nm)])
        high = np.concatenate([BUMP_HIGH, np.full(self.n_shape, +shape_bound_nm)])
        super().__init__(low, high)
        self.param_names = tuple(self.bump.param_names) + tuple(
            f"v{i}" for i in range(self.n_shape)
        )

    def curve(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        base = self.bump.curve(theta[:6])
        if self.n_shape == 0:
            return base
        weights = np.zeros(self.basis.n_basis)
        weights[: self.n_shape] = theta[6:]
        return base + self.basis.curve(weights)
