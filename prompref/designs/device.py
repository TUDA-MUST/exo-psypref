"""The device between a family and the simulated user: its peak-torque limit.

Every candidate and delivered curve in Phase 5 passes through the device, so
learners, the harness and the observer all see the clipped curve. Parameters, the feasible
set and the prior stay the family's own; only ``curve`` changes.

Of R29's other bounds, the torque-rate limit waits for a measured value, and landmark order
is reported rather than enforced.
"""

from __future__ import annotations

import numpy as np

from ..population.curves import TORQUE_LIMIT_NM, physics_clip


class DeviceDesign:
    def __init__(self, inner, limit: float = TORQUE_LIMIT_NM) -> None:
        if isinstance(inner, DeviceDesign):
            raise ValueError("this design is already behind a device")
        self.inner = inner
        self.limit = float(limit)

    @property
    def n_params(self) -> int:
        return self.inner.n_params

    @property
    def param_names(self) -> tuple[str, ...]:
        return self.inner.param_names

    def curve(self, theta: np.ndarray) -> np.ndarray:
        return physics_clip(self.inner.curve(theta), self.limit)

    def project(self, theta: np.ndarray) -> np.ndarray:
        return self.inner.project(theta)

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.inner.sample_prior(rng, n)

    def __getattr__(self, name: str):
        # Reached only for names not defined above (low, high, prior, basis, initial_guess),
        # so ``curve`` can never fall through to the family's unclipped one. "inner" is
        # refused so unpickling, which looks attributes up before __init__ runs, cannot recurse.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)
