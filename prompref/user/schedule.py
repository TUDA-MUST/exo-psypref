"""A wearer whose answer model changes during the session.

- **S3, fatigue.** The lapse rate rises linearly from 0.05 at the first comparison to 0.30 at the
  last.
- **S4, changing sensitivity.** The sensitivity ``s`` falls from 1.5 to 0.75 Nm (S4a, the wearer
  learns to discriminate) or rises from 0.75 to 1.5 Nm (S4b, the wearer is desensitised), linearly
  over the comparisons.

The preference itself (the utility, the ideal) never changes; only how the wearer answers does.
Like :class:`~prompref.user.drift.DriftingObserver`, comparison ``k`` (0-based) is answered with
the ``k``-th entry of each path, and past the end of a path its last entry holds. A path that is not
given leaves the observer's own value untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oracle import Observer


def linear_path(start: float, end: float, n: int) -> np.ndarray:
    """``n`` values from ``start`` at the first comparison to ``end`` at the last, linearly."""
    n = int(n)
    if n < 1:
        raise ValueError(f"a path needs at least one comparison, got {n}")
    return np.linspace(float(start), float(end), n)


@dataclass
class ScheduledObserver(Observer):
    """An :class:`Observer` that answers comparison ``k`` with ``lapse_path[k]`` and ``s_path[k]``.

    After an answer, ``lapse`` and ``s`` are the values that answer was given with."""

    lapse_path: np.ndarray | None = None
    s_path: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.lapse_path is None and self.s_path is None:
            raise ValueError("a ScheduledObserver needs lapse_path, s_path or both")
        if self.familiarity_weight:
            raise ValueError("a schedule is not combined with familiarity")
        for name in ("lapse_path", "s_path"):
            path = getattr(self, name)
            if path is None:
                continue
            path = np.asarray(path, dtype=float)
            if path.ndim != 1 or path.size < 1 or not np.all(np.isfinite(path)):
                raise ValueError(f"{name} must be a finite 1-D path, got shape {path.shape}")
            setattr(self, name, path)
        if self.lapse_path is not None:
            if not np.all((self.lapse_path >= 0.0) & (self.lapse_path <= 1.0)):
                raise ValueError("every lapse of lapse_path must lie in [0, 1]")
            self.lapse = float(self.lapse_path[0])
        if self.s_path is not None:
            if not np.all(self.s_path > 0.0):
                raise ValueError("every s of s_path must be positive")
            self.s = float(self.s_path[0])
        super().__post_init__()
        self._step = 0

    def respond(self, curve_a, curve_b, rng, order: int = 1) -> int:
        if self.lapse_path is not None:
            self.lapse = float(self.lapse_path[min(self._step, self.lapse_path.size - 1)])
        if self.s_path is not None:
            self.s = float(self.s_path[min(self._step, self.s_path.size - 1)])
        self._step += 1
        return super().respond(curve_a, curve_b, rng, order=order)
