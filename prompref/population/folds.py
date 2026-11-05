"""Leave-one-subject-out folds over the Molinaro subjects.

With five subjects, a prior fitted to the same subjects its users come from would look
better than it is for a new simulated user. Every fold fits on four subjects and builds users
from the fifth; one in-sample fold is kept as a sanity bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from preftune.data.loader import Trial, load_trial, subject_path

SUBJECTS = ("AB30", "AB31", "AB32", "AB33", "AB34")
SIDES = ("l", "r")

#: Per-fold random streams: every fold gets its own SeedSequence
#: ``[seed, FOLD_INDEX[fold]]``, so folds share neither factor draws nor answer seeds, and
#: R31's selection folds and evaluation fold do not share noise realisations.
FOLD_INDEX = {"loso-AB30": 0, "loso-AB31": 1, "loso-AB32": 2, "loso-AB33": 3, "loso-AB34": 4,
              "in-sample": 5}

#: The four children of a fold's stream, in spawn order.
FOLD_STREAMS = ("training", "users", "floors", "extra")


def child_sequence(parent: np.random.SeedSequence, index: int) -> np.random.SeedSequence:
    """``parent.spawn(index + 1)[index]`` for a fresh parent, without advancing its counter.

    ``SeedSequence.spawn`` is stateful: a second call on the same object hands out new
    children. Deriving the child from ``(entropy, spawn_key + (index,))`` makes every
    consumer of a stream reproducible no matter how often the parent was used before.
    """
    return np.random.SeedSequence(parent.entropy, spawn_key=tuple(parent.spawn_key) + (int(index),),
                                  pool_size=parent.pool_size)


def fold_streams(seed: int, fold_name: str) -> dict[str, np.random.SeedSequence]:
    root = np.random.SeedSequence([int(seed), FOLD_INDEX[fold_name]])
    return {name: child_sequence(root, i) for i, name in enumerate(FOLD_STREAMS)}


@dataclass(frozen=True)
class Fold:
    name: str
    held_out: tuple[str, ...]
    train: tuple[str, ...]

    @property
    def in_sample(self) -> bool:
        return set(self.held_out) <= set(self.train)


def loso_folds() -> list[Fold]:
    return [Fold(f"loso-{s}", (s,), tuple(x for x in SUBJECTS if x != s)) for s in SUBJECTS]


def in_sample_fold() -> Fold:
    return Fold("in-sample", SUBJECTS, SUBJECTS)


def check_fold(fold: Fold, prior_subjects) -> None:
    """A leave-one-subject-out prior must never have seen its held-out subject."""
    if fold.in_sample:
        return
    leaked = sorted(set(prior_subjects) & set(fold.held_out))
    if leaked:
        raise ValueError(f"fold {fold.name}: held-out subject(s) {leaked} fed the prior")


def load_trials(subjects, root: Path | None = None) -> list[Trial]:
    return [load_trial(subject, side, root=root) for subject in subjects for side in SIDES]


def data_available(root: Path | None = None) -> bool:
    try:
        return subject_path(SUBJECTS[0], "r", root).exists()
    except OSError:
        return False
