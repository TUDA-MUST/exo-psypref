"""A simulated user, to try a setup before anyone walks with it. Its preferred profile is an assumption."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from prompref.user.simulated_user import Observer, rms_distance

from .prior import DATA

_ANSWERS = {0: "A", 1: "B", 2: None}


def demo_profile(subject: str = "AB30", peaks: tuple[float, float] = (8.0, 6.0)) -> np.ndarray:
    """A preferred profile as the paper simulates one: a subject's delayed hip moment, with the extension and flexion peak.

    Use ``Prior.from_walking(..., leave_out=subject)``, so that the prior does not contain this subject.
    """
    with np.load(DATA / "demo_users.npz") as z:
        if f"{subject}_minus" not in z.files:
            raise ValueError(f"subject must be one of AB30-AB34, got {subject!r}")
        return peaks[0] * z[f"{subject}_minus"] + peaks[1] * z[f"{subject}_plus"]


class SimulatedUser:
    """Chooses between two curves by their RMS distance to ``preferred``, with noise in Nm and a lapse rate."""

    def __init__(self, preferred, noise: float = 1.0, lapse: float = 0.1, seed: int = 0, rng=None) -> None:
        self.preferred = np.asarray(preferred, dtype=float)
        self.noise, self.lapse = float(noise), float(lapse)
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self._observer = Observer(ideal_nm=self.preferred, s=self.noise, lapse=self.lapse)

    def answer(self, a: np.ndarray, b: np.ndarray):
        """``"A"``, ``"B"``, or None for no preference, in a randomly drawn presentation order."""
        order = 1 if self.rng.random() < 0.5 else -1
        return _ANSWERS[int(self._observer.respond(a, b, self.rng, order=order))]

    def distance(self, curve: np.ndarray) -> float:
        """RMS distance to the preferred profile, in Nm. Only a simulation knows it."""
        return rms_distance(curve, self.preferred)

    def would_choose(self, x: np.ndarray, y: np.ndarray) -> float:
        """How often this user would choose ``x`` over ``y``."""
        return choice_probability(x, y, self.preferred, self.noise, self.lapse)


def choice_probability(x, y, preferred, noise: float = 1.0, lapse: float = 0.1) -> float:
    """The answer model: ``lapse/2 + (1 - lapse) Phi((d_y - d_x) / noise)``."""
    d_x, d_y = rms_distance(np.asarray(x, dtype=float), preferred), rms_distance(np.asarray(y, dtype=float), preferred)
    return float(lapse / 2.0 + (1.0 - lapse) * norm.cdf((d_y - d_x) / noise))
