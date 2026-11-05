"""The Schaefer 4-timing data prior.

Unlike the ProMP and dual-Gaussian priors, this one is not fitted to biological curves: it
is the distribution of the preferred timings that Schaefer et al.'s 11 participants
actually chose, as summarised in ``out/schaefer/prior_inputs.json``. The covariance is the
**recommended** one, which is diagonal, because no correlation in that sample survived
shrinkage (see ``t_cov_note`` in the file). nu = 10 is the recommended tail, a judgement
recorded in the same file: 11 people cannot identify a tail.

It offers the part of ``DataPrior``'s interface the Phase 5 harness uses -- ``design``,
``normalization`` and ``space(radius)`` -- so ``arm_space`` treats it like any other prior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..designs.schaefer import SchaeferDesign
from ..prior import Normalization

#: The recommended Student-t tail for the timing prior (prior_inputs.json, student_t_nu).
TIMING_PRIOR_NU = 10.0

DEFAULT_INPUTS = Path("out/schaefer/prior_inputs.json")


@dataclass
class TimingPrior:
    """The preferred-timing prior of the Schaefer 4-timing family."""

    mu: np.ndarray
    sigma: np.ndarray
    #: The recommended tail, loaded and round-tripped -- but **nothing in the harness reads
    #: it**. A Phase 5 learner's tail comes from its arm
    #: (``ARMS[...].bowl_prior``, whose ``nu`` sets the Student-t), and this class
    #: contributes only ``mu`` and ``sigma`` through ``normalization``. So do not read the
    #: Schaefer prior as heavy-tailed in any cell: it is the arm that decides, and the
    #: ``tight`` arm is Gaussian. The field is kept because it is part of
    #: ``prior_inputs.json``'s recorded judgement and a reader of the file should find it
    #: here too, not because it is in force.
    nu: float = TIMING_PRIOR_NU
    source: str = ""
    family: str = field(default="schaefer", init=False)

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.mu.shape != (4,) or self.sigma.shape != (4, 4):
            raise ValueError(f"expected 4 timings, got mu {self.mu.shape}, sigma {self.sigma.shape}")

    @classmethod
    def load(cls, path=DEFAULT_INPUTS) -> "TimingPrior":
        summary = json.loads(Path(path).read_text(encoding="utf-8"))["summary"]
        if "t_cov_recommended" not in summary:
            raise KeyError(f"{path}: no t_cov_recommended in its summary")
        nu = float(summary.get("student_t_nu", {}).get("recommended", TIMING_PRIOR_NU))
        return cls(np.asarray(summary["t_mean"], dtype=float),
                   np.asarray(summary["t_cov_recommended"], dtype=float),
                   nu=nu, source=str(path))

    def design(self) -> SchaeferDesign:
        # an earlier step a later fix: without the fitted moments here, a data-prior Schaefer
        # cell's d_prior (n90's denominator) was drawn from the box-uniform default --
        # a knowledge-free quantity, the mirror image of the defect an earlier step's a later fix
        # corrected at this same line for promp_s0/dual_gauss.
        return SchaeferDesign(prior=(self.mu, self.sigma))

    def normalization(self) -> Normalization:
        return Normalization(self.mu, self.sigma, self.mu.size)

    def space(self, radius: float):
        """The learner space of this prior: its design, this normalization, and ``radius``."""
        from ..learning.space import LearnerSpace

        return LearnerSpace(self.design(), self.normalization(), float(radius))
