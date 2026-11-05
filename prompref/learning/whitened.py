"""Run a design-generic learner inside learner A's search space. This is D_z.

design found that LineCoSpar on PrompS0's +/-14 Nm box starts 7.9 Nm from the
target and never recovers, so D-on-box cannot test what A-versus-D is meant to test:
informed against random duel DIRECTIONS in the SAME space. This adapter gives any
design-generic learner A's space -- the same whitening, prior and trust ball, built by
the same :class:`LearnerSpace` -- by handing it a :class:`WhitenedDesign` and
translating at the boundary.
"""

from __future__ import annotations

import numpy as np

from ..designs.base import Design
from .linecospar import LineCoSparLearner
from .space import LearnerSpace


class WhitenedLearner:
    """An inner learner searching ``z``; the harness sees design parameters."""

    def __init__(self, space: LearnerSpace, inner) -> None:
        self.space = space
        self.inner = inner
        self._pending: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def linecospar(cls, design: Design, seed: int = 0, *, eps: float = 0.0,
                   d: int | None = None, space: LearnerSpace | None = None,
                   **kwargs) -> "WhitenedLearner":
        """D_z: LineCoSpar over A's trust ball, starting at the prior mean.

        With ``space``, the arm's space: the fold prior's whitening and
        the arm's radius. LineCoSpar's constants stay relative to that ball's box.
        """
        if space is None:
            space = LearnerSpace.build(design, eps=eps, d=d)
        elif design is not space.design or eps != 0.0 or d is not None:
            raise ValueError("space= must be a space of the given design, with eps and d unset")
        inner = LineCoSparLearner(space.as_design(), seed=seed, start=np.zeros(space.d),
                                  **kwargs)
        return cls(space, inner)

    @classmethod
    def gp_ucb(cls, design: Design, seed: int = 0, *, space: LearnerSpace | None = None,
               jnd: float | None = None, prior=None, **kwargs) -> "WhitenedLearner":
        """G: the preference GP over the arm's ball.

        The ball is the search region, so the kernel's length scale follows its span through ``GPUCBLearner``'s own default, and the JND gate measures
        delivered curves through the space's device-clipped design.

        ``prior`` is the arm's :class:`~prompref.learning.smc.BowlPrior`, and passing it is
        how G comes to search the *arm's* measure rather than a standard normal (review, item 1; the gp_ucb module docstring has the measurement). It is named
        explicitly rather than left to ``**kwargs`` because the ball this method builds --
        ``space.as_design`` -- is precisely what makes the design's own ``sample_prior``
        the wrong measure here, so the two belong in one signature.
        """
        from .gp_ucb import GPUCBLearner

        if space is None:
            space = LearnerSpace.build(design)
        elif design is not space.design:
            raise ValueError("space= must be a space of the given design")
        inner = GPUCBLearner(space.as_design(), seed=seed, jnd=jnd, prior=prior, **kwargs)
        return cls(space, inner)

    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        z_a, z_b = self.inner.propose()
        self._pending = (z_a, z_b)
        return self.space.to_theta(z_a), self.space.to_theta(z_b)

    def update(self, theta_a: np.ndarray, theta_b: np.ndarray, response: int) -> None:
        """Replays the proposed ``z`` rather than mapping ``theta`` back.

        The ``PreferenceLearner`` contract is that ``update`` receives the pair ``propose``
        returned, and replaying the cached ``z`` keeps that exact; a round trip through
        ``theta`` is exact only to ~1e-15. ``LineCoSparLearner`` happens not to read these
        arguments -- it resolves the point from its own pending proposal -- so for D_z
        this changes nothing today; it protects any inner learner that does read them.
        """
        if self._pending is None:
            raise RuntimeError("call propose() before update()")
        z_a, z_b = self._pending
        self._pending = None
        self.inner.update(z_a, z_b, response)

    def best(self) -> np.ndarray:
        return self.space.to_theta(self.inner.best())

    def posterior_summary(self) -> dict:
        summary = self.inner.posterior_summary()
        extras = dict(summary.get("extras", {}))
        extras.update(space_d=self.space.d, trust_radius=self.space.radius)
        return {**summary, "extras": extras}
