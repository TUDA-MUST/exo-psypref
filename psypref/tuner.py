"""The tuning session: ask for a pair, tell the answer, read the identified curve."""

from __future__ import annotations

import numpy as np

from prompref.designs.base import BoxDesign
from prompref.designs.device import DeviceDesign
from prompref.experiment import amp_free as AF

from .prior import DATA, Prior

#: The gait cycle in %, at the 101 points of every curve.
PHASE = np.linspace(0.0, 100.0, 101)
_ANSWERS = {"A": 0, "B": 1, None: 2}
#: Name here -> the family's name in the range and at a fixed level.
_FAMILIES = {"schaefer": ("schaefer_ii", "schaefer_i"), "franks": ("franks", "franks"), "pchip": ("pchip_ii", "pchip")}
#: The stored priors of the families come from one data fold.
_FOLD = "loso-AB30"


class Family(BoxDesign):
    """Your own interpretable profile family, for a projection.

    ``curve(theta)`` gives the 101-point curve of the parameters ``theta``, within the bounds ``low`` and ``high``.
    ``initial_guess(target)``, optional, gives a starting point for fitting a target curve.
    """

    def __init__(self, curve, low, high, initial_guess=None, names=None) -> None:
        super().__init__(np.asarray(low, dtype=float), np.asarray(high, dtype=float))
        self._curve = curve
        self.param_names = tuple(names) if names is not None else tuple(f"p{i}" for i in range(len(low)))
        self.initial_guess = initial_guess if initial_guess is not None else (lambda target: 0.5 * (self.low + self.high))

    def curve(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(self._curve(self.project(theta)), dtype=float)


class PsyPref:
    """A tuning session.

    ``family`` shows every pair in an interpretable family: a name (``"schaefer"``, ``"franks"``, ``"pchip"``), a
    :class:`Family`, or None for the whole curve. ``seed`` fixes the session, so the same answers give the same curves.
    Present each pair in random order: the labels A and B say nothing about which curve comes first.
    """

    def __init__(self, prior: Prior, family=None, *, seed: int = 0) -> None:
        self.prior, self.seed = prior, int(seed)
        self.space = prior._space()
        arm = prior._arm()
        if prior.level is not None:
            inner = AF.psypref(self.space, self.seed, arm=arm, ties="band")
        else:
            inner = AF.metric_psypref(self.space, self.seed, arm=arm, ties="band")
            if prior.peak_range is not None:
                inner = AF.TwoPartGuard(inner, self.space, 0.5 * prior.peak_range[0], seed=self.seed)
        self.design = self.space.design
        self.learner = inner
        self.theta = None
        if family is not None:
            if prior.peak_range is None and prior.level is None:
                raise ValueError("a projection needs a prior with a peak range or a level")
            self.design = self._family_design(family)
            self.learner = AF.family_lens(inner, self.space, self.design, level=prior.level)
        self.family = family
        self._pair = None
        self._best = None
        self.answers: list = []

    def _family_design(self, family) -> DeviceDesign:
        limit = self.prior.peak_range[1] if self.prior.level is None else self.prior.level + 1e-9
        if isinstance(family, str):
            if family not in _FAMILIES:
                raise ValueError(f"family must be one of {sorted(_FAMILIES)}, or a Family; got {family!r}")
            name = _FAMILIES[family][0 if self.prior.level is None else 1]
            kw = dict(level=self.prior.level) if self.prior.level is not None else dict(peak_range=self.prior.peak_range)
            return AF.family_design(name, DATA, _FOLD, **kw)
        return DeviceDesign(family, limit=limit)

    @property
    def n_comparisons(self) -> int:
        return len(self.answers)

    def ask(self) -> tuple[np.ndarray, np.ndarray]:
        """The next pair of curves. Asking again before an answer returns the same pair."""
        if self._pair is None:
            self._pair = self.learner.propose()
        return self.design.curve(self._pair[0]), self.design.curve(self._pair[1])

    def tell(self, answer) -> None:
        """The user's answer to the last pair: ``"A"``, ``"B"``, or None for no preference."""
        if self._pair is None:
            raise RuntimeError("ask for a pair before telling an answer")
        if answer not in _ANSWERS:
            raise ValueError(f'the answer must be "A", "B" or None, got {answer!r}')
        self.learner.update(self._pair[0], self._pair[1], _ANSWERS[answer])
        self.answers.append(answer)
        self._pair = None
        self._best = None

    def identified(self) -> np.ndarray:
        """The identified assistance now: the curve at the posterior mean."""
        if self._best is None:
            self._best = self.design.curve(np.asarray(self.learner.best(), dtype=float))
        return self._best

    def parameters(self) -> np.ndarray:
        """The identified assistance in its own parameters: the family's, or the ProMP's weights."""
        return np.asarray(self.learner.best(), dtype=float)

    @classmethod
    def replay(cls, prior: Prior, answers, family=None, *, seed: int = 0) -> "PsyPref":
        """A session rebuilt from its answers, for example to continue one after a break."""
        tuner = cls(prior, family, seed=seed)
        for answer in answers:
            tuner.ask()
            tuner.tell(answer)
            tuner.identified()
        return tuner
