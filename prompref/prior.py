"""Priors over ProMP basis weights, and the normalized space they induce.

The prior is this project's independent variable. Three sources:
shaped to a parametric family, fitted to biological moments, or generically smooth --
and a mixture that interpolates between the first and third.

The mixture is renormalised to constant total variance. Without that, epsilon would
confound how *concentrated* the prior is with *which directions* it favours, and the
experiment would be uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi

from .basis import VonMisesBasis


def sigma_from_design(design, basis: VonMisesBasis, rng: np.random.Generator,
                      n: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of basis weights induced by sampling a profile family.

    This is how a ProMP is given "the shape of a standard parametric function": draw
    from the parametric family, project each curve onto the basis, and take the
    empirical moments of the resulting weights.
    """
    theta = design.sample_prior(rng, n)
    curves = np.array([design.curve(t) for t in theta])
    weights = basis.project(curves.T).T  # (n, K)
    return weights.mean(axis=0), np.cov(weights, rowvar=False)


def circular_second_difference(k: int) -> np.ndarray:
    """The wrap-around second-difference operator, ``(K, K)``."""
    d = np.zeros((k, k))
    for i in range(k):
        d[i, (i - 1) % k] = 1.0
        d[i, i] = -2.0
        d[i, (i + 1) % k] = 1.0
    return d


#: Relative tolerance for calling an eigenvalue of ``D^T D`` zero. The spectrum is
#: ``16 sin^4(pi k / K)``, so the gap between the exact zero and the smallest nonzero
#: eigenvalue is many orders of magnitude and this threshold is not delicate.
_NULL_EIGENVALUE_RTOL = 1e-10


def sigma_smooth(basis: VonMisesBasis, ridge: float = 1e-3) -> np.ndarray:
    """Generic curvature prior on the **zero-mean subspace**: ``(D^T D)^+``.

    ``D`` annihilates the constant vector, so ``D^T D`` is singular by construction and
    the curvature penalty says *nothing whatever* about a uniform torque offset. This
    builds the prior only where the penalty is informative: eigendecompose ``D^T D``,
    discard the null direction, invert the rest, and reassemble. The DC direction
    therefore gets **exactly zero** variance, which is the honest encoding of "a
    curvature penalty does not license a constant offset".

    ``ridge`` is retained in the signature for backwards compatibility and is
    **deliberately unused**. The previous implementation returned
    ``(D^T D + ridge I)^-1``, which handed the DC direction variance ``1 / ridge`` --
    measured at 1000 against 2.906 for the next eigenvalue. The ridge, meant only to
    regularise, silently became the dominant term: ``d_eff`` collapsed from 3.47 to 1.11
    at the first nonzero epsilon, and at every epsilon > 0 the top normalized direction was
    100 % a constant torque offset -- the one direction with no assistive role at all.
    Pricing DC is no longer the ridge's job, or anyone's; see our design notes
    """
    d = circular_second_difference(basis.n_basis)
    gram = d.T @ d
    vals, vecs = np.linalg.eigh(gram)
    inverse = np.zeros_like(vals)
    keep = vals > _NULL_EIGENVALUE_RTOL * max(float(vals.max()), 1.0)
    inverse[keep] = 1.0 / vals[keep]
    sigma = (vecs * inverse) @ vecs.T
    return 0.5 * (sigma + sigma.T)


def mix_priors(sigma_a: np.ndarray, sigma_b: np.ndarray, eps: float) -> np.ndarray:
    """``(1 - eps) A + eps B``, rescaled to the total variance of ``A``."""
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")
    mixed = (1.0 - eps) * sigma_a + eps * sigma_b
    return mixed * (np.trace(sigma_a) / np.trace(mixed))


def sigma_smooth_matched(basis: VonMisesBasis, sigma_param: np.ndarray) -> np.ndarray:
    """The curvature prior's *shape* with the parametric prior's *scale* and *offset*.

    design. :func:`sigma_smooth` is the right prior considered alone: a curvature
    penalty says nothing about a uniform torque offset, so that direction gets zero
    variance. Mixed against ``Sigma_param`` it is the wrong prior, because as epsilon
    rises the offset direction loses variance and the reachable set shrinks -- measured,
    the in-family target carries 0.2335 Nm of offset and A(eps=1) could get no closer
    than exactly that. Matching fixes both halves: on the zero-mean subspace the result
    is ``sigma_smooth`` rescaled to ``Sigma_param``'s trace there, and along the constant
    direction it takes ``Sigma_param``'s variance. Every epsilon then has identical
    offset variance and identical total variance, so ``mix_priors``' rescaling is a
    no-op and epsilon moves only the shape directions.
    """
    k = basis.n_basis
    sigma_param = np.asarray(sigma_param, dtype=float)
    if sigma_param.shape != (k, k):
        raise ValueError(f"sigma_param must be ({k}, {k}), got {sigma_param.shape}")
    dc = np.full(k, 1.0 / np.sqrt(k))
    proj = np.eye(k) - np.outer(dc, dc)
    shape = proj @ sigma_smooth(basis) @ proj
    shape *= np.trace(proj @ sigma_param @ proj) / np.trace(shape)
    matched = shape + float(dc @ sigma_param @ dc) * np.outer(dc, dc)
    return 0.5 * (matched + matched.T)


#: A normalization divides by the square root of every kept prior variance. Below this
#: fraction of the largest, that division amplifies round-off into nonsense rather than
#: into a meaningful scale.
RANK_RTOL = 1e-8


@dataclass
class Normalization:
    """Maps basis weights to a space where the prior is ``N(0, I)``.

    One z-unit is one population standard deviation, so the space is perceptually
    scaled by construction -- which is what allows a JND to be quoted per direction.
    """

    mu: np.ndarray
    sigma: np.ndarray
    d: int

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float)
        self.d = int(self.d)
        if not 1 <= self.d <= self.mu.size:
            # Out of range, d silently truncated to nothing (d <= 0 gives empty slices
            # and NaN round-trips) or padded past the spectrum, either way producing a
            # normalization that looks like it worked. Refuse instead.
            raise ValueError(
                f"d must be in [1, {self.mu.size}], got {self.d}"
            )
        vals, vecs = np.linalg.eigh(np.asarray(self.sigma, dtype=float))
        order = np.argsort(vals)[::-1]
        raw = vals[order]
        kept = raw[self.d - 1]
        if raw[0] <= 0.0 or kept < RANK_RTOL * raw[0]:
            # Clipping here instead silently turned a rank-deficient prior into
            # |z| = 2.1e7 rather than an error.
            raise ValueError(
                f"prior variance {self.d} of {raw.size} is {kept:.3e} against a largest of "
                f"{raw[0]:.3e}; whitening would divide by it. Lower d, or use a full-rank "
                f"prior such as sigma_smooth_matched"
            )
        self.eigenvalues = np.clip(raw, 1e-15, None)
        self.eigenvectors = vecs[:, order]
        self._u = self.eigenvectors[:, : self.d]
        self._scale = np.sqrt(self.eigenvalues[: self.d])

    def to_z(self, w: np.ndarray) -> np.ndarray:
        centred = np.asarray(w, dtype=float) - self.mu
        return (centred @ self._u) / self._scale

    def to_w(self, z: np.ndarray) -> np.ndarray:
        return (np.asarray(z, dtype=float) * self._scale) @ self._u.T + self.mu

    @property
    def d_eff(self) -> float:
        """Participation ratio: how many directions the prior effectively spans."""
        lam = self.eigenvalues
        return float(lam.sum() ** 2 / np.square(lam).sum())

    def reconstruction_error(self, weights: np.ndarray) -> float:
        """RMS error from projecting weights through the truncated basis and back."""
        w = np.atleast_2d(np.asarray(weights, dtype=float))
        return float(np.sqrt(np.mean((self.to_w(self.to_z(w)) - w) ** 2)))


def perceptual_resolution(jnd75_z: np.ndarray, r_prior: float = 2.5
                          ) -> tuple[np.ndarray, float]:
    """Distinguishable levels per direction, and their product.

    ``d_eff`` says how many directions the *prior* spans; this says how many a *person
    can feel*. If published JNDs are as coarse as the design's section 5.3 suggests, the
    product may be small enough that current profile families over-resolve the
    perceptual space entirely.
    """
    jnd = np.asarray(jnd75_z, dtype=float)
    if np.any(jnd <= 0.0):
        raise ValueError("every JND must be positive")
    levels = 2.0 * r_prior / jnd
    return levels, float(np.prod(levels))


def trust_radius(d: int, quantile: float = 0.99) -> float:
    """Radius of the search region holding ``quantile`` of an ``N(0, I_d)`` prior's mass.

    design. The norm of a d-dimensional standard normal draw concentrates near
    sqrt(d), so a fixed radius is a d-dependent handicap: 2.5 keeps 99 % of the mass at
    d=1 but only 38 % at d=8, where it excluded the sweep's own in-family target
    (|z*| = 2.70). A chi quantile keeps the same fraction at every d.
    """
    if d < 1:
        raise ValueError(f"d must be at least 1, got {d}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    return float(chi.ppf(quantile, d))


def learner_prior(design, eps: float = 0.0, n: int = 4000
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance a LEARNER assumes over ``design``'s parameters.

    The only place epsilon enters. The in-family target is drawn from
    ``design.sample_prior``, so had epsilon been a design argument, A(eps=0) and
    A(eps=1) would have chased different targets and the primary question would have
    been unanswerable from its own results.

    For ``PrompS0`` the parameters are basis weights, so the prior is the design's
    fitted ``Sigma_param`` mixed with :func:`sigma_smooth_matched`. For every other
    family epsilon has no meaning -- its parameters are not basis weights -- and the
    prior is the moments of the family's own sampler, drawn with a fixed seed so every
    learner in a sweep sees the same one.
    """
    from .designs.promp import PrompS0

    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")
    if type(design) is PrompS0:
        mu, param = (np.asarray(a, dtype=float) for a in design.prior)
        if eps == 0.0:
            return mu, param
        return mu, mix_priors(param, sigma_smooth_matched(design.basis, param), eps)
    if eps != 0.0:
        raise ValueError(
            f"eps applies only to PrompS0, whose parameters are basis weights; got "
            f"{type(design).__name__}"
        )
    theta = np.asarray(design.sample_prior(np.random.default_rng(0), n), dtype=float)
    return theta.mean(axis=0), np.cov(theta, rowvar=False)
