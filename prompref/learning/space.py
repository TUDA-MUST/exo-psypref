"""The space a learner reasons in: whitened design parameters inside a trust ball.

The harness speaks in a design's own parameters, theta. Learner A and D_z reason in
``z``, where the learner's prior is ``N(0, I_d)`` and one unit is one prior standard
deviation. This module is the only place the two meet, so every learner that reasons in
``z`` shares one definition of the space, the prior and the trust region -- which is
what makes "hold the search space fixed" literally true.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..designs.base import BoxDesign, Design
from ..prior import Whitening, learner_prior, trust_radius


@dataclass
class LearnerSpace:
    design: Design
    whitening: Whitening
    radius: float

    @classmethod
    def build(cls, design: Design, eps: float = 0.0, d: int | None = None,
              quantile: float = 0.99) -> "LearnerSpace":
        """``d`` defaults to every parameter; the radius is R10's."""
        mu, sigma = learner_prior(design, eps)
        d = design.n_params if d is None else int(d)
        return cls(design, Whitening(mu, sigma, d), trust_radius(d, quantile))

    @property
    def d(self) -> int:
        return self.whitening.d

    def to_theta(self, z: np.ndarray) -> np.ndarray:
        """Design parameters for ``z``, clipped to the design's own feasible set."""
        return self.design.project(self.whitening.to_w(z))

    def to_z(self, theta: np.ndarray) -> np.ndarray:
        return self.whitening.to_z(np.asarray(theta, dtype=float))

    def clip_to_ball(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        norm = float(np.linalg.norm(z))
        return z if norm <= self.radius else z * (self.radius / norm)

    def transfer(self, z: np.ndarray, to: "LearnerSpace") -> np.ndarray:
        """The same design parameters, in ``to``'s coordinates. Exact and unclipped.

        Two spaces of one design differ only in their whitening, so this is an affine map
        and needs no trip through the design's feasible set. needs it: learner
        A's posterior and its duels live in different spaces.

        Exact for full-rank whitening; refused otherwise. At
        d < n_params two spaces keep different eigen-subspaces, so the round trip through
        ``to``'s coordinates is a lossy projection rather than an exact map -- measured
        round-trip error 4.9 at d=6 on PrompS0 against 8.5e-15 at d=8.
        """
        if to is self:
            return np.asarray(z, dtype=float)
        if to.design is not self.design:
            raise ValueError("transfer needs spaces of one design")
        if to.d != self.d:
            raise ValueError(f"transfer needs equal dimensions, got {self.d} and {to.d}")
        if self.d < self.whitening.mu.size:
            raise ValueError(
                f"transfer is exact only for full-rank whitening (d = n_params); got "
                f"d={self.d} of {self.whitening.mu.size}"
            )
        return to.whitening.to_z(self.whitening.to_w(z))

    def as_design(self) -> "WhitenedDesign":
        return WhitenedDesign(self)


class WhitenedDesign(BoxDesign):
    """The trust ball, presented as a design, so a design-generic learner can search it.

    Its box is the cube ``[-r, r]^d`` and its feasible set is the ball inside it: a
    learner that treats "``project`` leaves the point unchanged" as feasibility -- as
    LineCoSpar does -- therefore searches exactly the ball.
    """

    def __init__(self, space: LearnerSpace) -> None:
        self.space = space
        r = space.radius
        super().__init__(np.full(space.d, -r), np.full(space.d, r))
        self.param_names = tuple(f"z{i}" for i in range(space.d))

    def project(self, z: np.ndarray) -> np.ndarray:
        return self.space.clip_to_ball(z)

    def curve(self, z: np.ndarray) -> np.ndarray:
        return self.space.design.curve(self.space.to_theta(z))

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.array([self.project(v) for v in rng.normal(size=(n, self.space.d))])


class SpacePriorDesign:
    """``space.design``, with ``sample_prior`` drawn from ``prior`` in the space's z.

    C2 in a Phase 5 arm. Ramella's learner keeps its batch in
    the family's own parameters, as in Phase 4. The batch is drawn from the arm's prior,
    heavy tails included, and clipped radially to the arm's radius like every other
    learner's duels, before being mapped to parameters. ``prior`` is a
    :class:`~prompref.learning.smc.BowlPrior`.
    """

    def __init__(self, space: LearnerSpace, prior) -> None:
        self.space = space
        self.prior = prior

    @property
    def n_params(self) -> int:
        return self.space.design.n_params

    @property
    def param_names(self) -> tuple[str, ...]:
        return self.space.design.param_names

    def curve(self, theta: np.ndarray) -> np.ndarray:
        return self.space.design.curve(theta)

    def project(self, theta: np.ndarray) -> np.ndarray:
        return self.space.design.project(theta)

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        z = self.prior.sample_zstar(rng, n, self.space.d)
        return np.array([self.space.to_theta(self.space.clip_to_ball(v)) for v in z])
