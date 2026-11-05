"""Phase 5c's grid, block by block.

Every block runs on the five leave-one-subject-out folds and every exported user of each
(40 population and 4 stress), unless a block says otherwise. G always states its JND level; its candidate count is 2048 except in the attribution block's S7 cell.
"""

from __future__ import annotations

from ..learning.arms import ARMS
from ..population.export import load_fold_users
from ..user.jnd import true_jnd
from .phase5 import C1_ARM, C1_FAMILY, Phase5Point

ALL_ARMS = tuple(ARMS)
PAIR = ("tight", "t4")
LOSO = tuple(f"loso-AB{n}" for n in range(30, 35))
JND = true_jnd(0.5)
G_JND_SCALE = 1.0
BIG_CANDIDATES = 32768

#: our design notes S6, then S16 and S17: each cell changes one thing from the reference (lapse
#: 0.1, s 0.5). S6's cells come first, so they finish first.
NOISE_CELLS = {
    "lapse0": dict(lapse=0.0),
    "lapse02": dict(lapse=0.2),
    "lapse03": dict(lapse=0.3),
    "s1": dict(s=1.0),
    "order": dict(order_bias=0.3),
    "overdisp": dict(overdispersion=0.4),
    "plateau1": dict(plateau_nm=JND),
    "plateau2": dict(plateau_nm=2 * JND),
    "drift2": dict(drift_nm=2 * JND),
    "drift4": dict(drift_nm=4 * JND),
}

#: our design notes S16a: the preferred region anchored on each user's floor, k JND wide.
REGION_CELLS = {
    "region1": dict(plateau_over_floor_jnd=1.0),
    "region2": dict(plateau_over_floor_jnd=2.0),
}

#: The attribution cells a Phase 5c run adds; the ProMP data-prior cells are
#: block 1's.
ATTRIBUTION_CELLS = (("promp_s0", "flat"), ("schaefer", "data"), ("schaefer", "flat"))

BLOCKS = ("main-promp", "noise", "region", "attribution", "attribution-followup", "dual-smoke",
          "main-dual", "baseline")


def _g(learner: str) -> dict:
    return {"jnd_scale": G_JND_SCALE} if learner == "g" else {}


def _user_ids(run_dir: str, fold: str, smoke: bool = False) -> list[str]:
    users = load_fold_users(run_dir, fold)
    if not smoke:
        return list(users)
    population = [uid for uid, user in users.items() if user.kind == "population"]
    stress = [uid for uid, user in users.items() if user.kind == "stress"]
    return population[:5] + stress[:1]


def _main(run_dir, folds, family, n):
    points = []
    for fold in folds:
        uids = _user_ids(run_dir, fold)
        for arm in ALL_ARMS:
            for learner in ("a", "g", "d_z", "c2"):
                if learner == "c2" and arm not in PAIR:
                    continue      # our design notes S5
                points += [Phase5Point(run_dir, fold, family, arm, learner, uid,
                                       n_comparisons=n, **_g(learner)) for uid in uids]
    return points


def _noise(run_dir, folds, n, cells=NOISE_CELLS):
    points = []
    for fields in cells.values():
        for fold in folds:
            uids = _user_ids(run_dir, fold)
            for arm in PAIR:
                for learner in ("a", "g"):
                    points += [Phase5Point(run_dir, fold, "promp_s0", arm, learner, uid,
                                           n_comparisons=n, **fields, **_g(learner))
                               for uid in uids]
    return points


def _attribution(run_dir, folds, n, arm):
    points = []
    for fold in folds:
        uids = _user_ids(run_dir, fold)
        for family, prior in ATTRIBUTION_CELLS:
            for learner in ("a", "g"):
                points += [Phase5Point(run_dir, fold, family, arm, learner, uid,
                                       n_comparisons=n, prior=prior, **_g(learner))
                           for uid in uids]
    return points


def _attribution_block(run_dir, folds, n):
    points = _attribution(run_dir, folds, n, "tight")
    for fold in folds:
        uids = _user_ids(run_dir, fold)
        for arm in PAIR:     # our design notes S7
            points += [Phase5Point(run_dir, fold, "promp_s0", arm, "g", uid, n_comparisons=n,
                                   jnd_scale=G_JND_SCALE, n_candidates=BIG_CANDIDATES)
                       for uid in uids]
        points += [Phase5Point(run_dir, fold, C1_FAMILY, C1_ARM, "c1", uid, n_comparisons=n,
                               prior="flat") for uid in uids]     # our design notes S9
    return points


def block_points(block: str, run_dir: str, *, folds=LOSO, arm: str | None = None,
                 n_comparisons: int = 150) -> list[Phase5Point]:
    """The points of one block of our design notes's grid, in the order they should run."""
    n = int(n_comparisons)
    if block == "main-promp":
        return _main(run_dir, folds, "promp_s0", n)
    if block == "main-dual":
        return _main(run_dir, folds, "dual_gauss", n)
    if block == "noise":
        return _noise(run_dir, folds, n)
    if block == "region":
        return _noise(run_dir, folds, n, REGION_CELLS)
    if block == "attribution":
        return _attribution_block(run_dir, folds, n)
    if block == "attribution-followup":
        if arm is None:
            raise ValueError("attribution-followup needs an arm: the one R31 recommends")
        return _attribution(run_dir, folds, n, arm)
    if block == "dual-smoke":
        uids = _user_ids(run_dir, folds[0], smoke=True)
        return [Phase5Point(run_dir, folds[0], "dual_gauss", "tight", learner, uid,
                            n_comparisons=n, **_g(learner))
                for learner in ("a", "d_z", "g", "c2") for uid in uids]
    if block == "baseline":
        return [Phase5Point(run_dir, fold, "promp_s0", arm_name, learner, uid, n_comparisons=n,
                            **_g(learner))
                for fold in folds for arm_name in PAIR for learner in ("a", "g")
                for uid in _user_ids(run_dir, fold)]
    raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")
