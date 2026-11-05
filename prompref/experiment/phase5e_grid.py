"""Phase 5e's grid: how far the short-budget claims carry.

Every session runs 40 comparisons in the tight arm with the data prior, except e-drift, which
runs 24: the drift path spans the session, so it cannot be read off by truncation.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..user.jnd import true_jnd
from .phase5 import Phase5Point
from .phase5c_grid import G_JND_SCALE, LOSO, _user_ids

BLOCKS_5E = ("e-main", "e-s", "e-observer", "e-noise", "e-drift", "e-lapse", "e-tune",
             "e-tuned", "e-oracle")
ARM = "tight"
N_COMPARISONS = 40
N_DRIFT = 24
#: our design notes: the comparators are tuned on this fold and evaluated on the other four.
TUNE_FOLD = "loso-AB30"
EVAL_FOLDS = tuple(f for f in LOSO if f != TUNE_FOLD)
#: our design notes's tuning grids, as the point fields each setting sets (LineCoSpar is ``d_z``).
TUNE_SETTINGS = (
    [("cmaes", dict(cma_popsize=p, cma_sigma_mult=m)) for p in (4, 6, 10) for m in (0.5, 1.0, 2.0)]
    + [("g", dict(g_kappa0=k)) for k in (0.5, 1.5, 3.0)]
    + [("d_z", dict(lengthscale_mult=ls, step_mult=st))
       for ls, st in ((1.0, 0.5), (2.0, 0.5), (2.0, 1.0))])
#: a change, the oracle-tuned bound: every e-tune setting plus G at kappa0 4.5 and
#: 6.5 (6.5 is G's cap) and SortCMA at sigma multiplier 0.25 for each popsize, run on the
#: four evaluation folds. The report picks each comparator's best setting in sample there:
#: an upper bound on what tuning could buy, not a fair comparison against A.
#: Five of the 20 settings repeat sessions other blocks already ran (880 of 3,520 sessions):
#: 528 are key-identical to the e-tuned settings, and 352 are the e-main defaults (G kappa0 1.5
#: and SortCMA popsize 10, sigma multiplier 1.0). LineCoSpar (2, 0.5) is both a default and the
#: tuned choice. The repeats are identical sessions, so they change neither selection nor bound.
ORACLE_SETTINGS = (
    list(TUNE_SETTINGS)
    + [("g", dict(g_kappa0=k)) for k in (4.5, 6.5)]
    + [("cmaes", dict(cma_popsize=p, cma_sigma_mult=0.25)) for p in (4, 6, 10)])
#: The fields each tuned optimizer may carry in ``tuned.json``.
TUNED_FIELDS = {"cmaes": ("cma_popsize", "cma_sigma_mult"), "g": ("g_kappa0",),
                "d_z": ("lengthscale_mult", "step_mult")}

#: our design notes: (family, learner) cells per scope.
FULL = [("promp_s0", "a"), ("promp_s0", "g"), ("promp_s0", "d_z"), ("promp_s0", "cmaes"),
        ("franks", "a_lens"), ("franks", "a"), ("franks", "g"), ("franks", "d_z"),
        ("franks", "cmaes")]
CORE = [("promp_s0", "a"), ("promp_s0", "g"), ("promp_s0", "d_z"), ("promp_s0", "cmaes"),
        ("franks", "a_lens"), ("franks", "g")]
DRIFT = [("promp_s0", "a"), ("promp_s0", "g"), ("promp_s0", "d_z"), ("promp_s0", "cmaes")]

#: Magnitude drift ends k true JNDs of the s = 1.0 wearer away. Computed as
#: ``true_jnd(1.0) * k`` exactly, because the floor sidecar's target check compares at four
#: decimals.
DRIFT_JNDS = (1, 2)
#: a change: the walking-calibrated landmark cell's magnitude JND, 25 % of the
#: ideal's peak (walking thresholds are 3-4x the static 7-10 %).
WALKING_LANDMARK_FRACTION = 0.25


def _scenarios(block: str) -> list[tuple[list, dict]]:
    """(optimizer cells, extra point fields) per scenario of a block; G's JND level is added
    per point."""
    if block == "e-main":
        return [(FULL, dict(s=1.0))]
    if block == "e-s":
        return [(FULL, dict(s=2.0)), (FULL, dict(s=1.0, s_user_logsd=0.35))]
    if block == "e-observer":
        # The static landmark cell (10 % magnitude JND) and the walking-calibrated one (25 %,
        # a change) bracket the range of magnitude JNDs, not the range of plausible
        # wearers: both are more decisive than the s = 1.0 RMS wearer (report section 9).
        return [(CORE, dict(s=1.0, observer_metric="landmark")),
                (CORE, dict(s=1.0, observer_metric="landmark",
                            landmark_mag_fraction=WALKING_LANDMARK_FRACTION)),
                (CORE, dict(s=1.0, observer_metric="weber")),
                (CORE, dict(s=1.0, asymmetry=2.0)),
                (CORE, dict(s=1.0, familiarity_weight=0.25))]
    if block == "e-noise":
        return [(CORE, dict(s=1.0, device_noise_nm=0.6))]
    if block == "e-drift":
        return [(DRIFT, dict(s=1.0, drift_axis="magnitude", drift_nm=true_jnd(1.0) * k,
                             n_comparisons=N_DRIFT)) for k in DRIFT_JNDS]
    if block == "e-lapse":
        return [(CORE, dict(s=1.0, lapse=0.2)), (CORE, dict(s=1.0, lapse=0.3))]
    raise ValueError(block)


def load_tuned(tuned_path) -> dict:
    """``tuned.json``: optimizer -> its point fields, as ``phase5e_tune_select.py`` wrote it."""
    path = Path(tuned_path)
    if not path.exists():
        raise FileNotFoundError(
            f"e-tuned needs {path}, the comparator settings chosen on {TUNE_FOLD}: run e-tune, "
            f"then scripts/phase5e_tune_select.py --dir <out-dir>")
    tuned = json.loads(path.read_text(encoding="utf-8"))
    if set(tuned) != set(TUNED_FIELDS):
        raise ValueError(f"{path} must hold exactly {sorted(TUNED_FIELDS)}, got {sorted(tuned)}")
    for learner, fields in tuned.items():
        if set(fields) != set(TUNED_FIELDS[learner]):
            raise ValueError(f"{path}: {learner} must set exactly {TUNED_FIELDS[learner]}, "
                             f"got {sorted(fields)}")
    return tuned


def block_points(block: str, run_dir: str, *, folds=LOSO, arm=None,
                 n_comparisons: int = N_COMPARISONS, tuned_path=None) -> list[Phase5Point]:
    """The points of one block.

    ``folds`` restricts the real folds; e-tune always runs on :data:`TUNE_FOLD`, e-tuned and
    e-oracle on :data:`EVAL_FOLDS`, each intersected with ``folds``. ``e-drift`` ignores
    ``n_comparisons`` (it states 24). ``e-tuned`` needs ``tuned_path``, the ``tuned.json`` that
    ``phase5e_tune_select.py`` wrote (the runner passes ``<out-dir>/tuned.json``); without it,
    or if the file is missing, it raises.
    """
    if block not in BLOCKS_5E:
        raise ValueError(f"block must be one of {BLOCKS_5E}, got {block!r}")
    specs = []          # (fold, family, learner, fields)
    if block == "e-tune":
        for learner, fields in TUNE_SETTINGS:
            specs += [(TUNE_FOLD, "promp_s0", learner, dict(s=1.0, **fields))]
        fold_list = (TUNE_FOLD,) if TUNE_FOLD in folds else ()
    elif block == "e-tuned":
        if tuned_path is None:
            raise ValueError("e-tuned needs tuned_path (<out-dir>/tuned.json)")
        tuned = load_tuned(tuned_path)
        for learner, fields in tuned.items():
            specs += [(None, "promp_s0", learner, dict(s=1.0, **fields))]
        fold_list = tuple(f for f in EVAL_FOLDS if f in folds)
    elif block == "e-oracle":
        for learner, fields in ORACLE_SETTINGS:
            specs += [(None, "promp_s0", learner, dict(s=1.0, **fields))]
        fold_list = tuple(f for f in EVAL_FOLDS if f in folds)
    else:
        for cells, fields in _scenarios(block):
            specs += [(None, family, learner, dict(fields)) for family, learner in cells]
        fold_list = tuple(folds)
    points = []
    for fold in fold_list:
        uids = _user_ids(run_dir, fold)
        for _, family, learner, cell in specs:
            extra = dict(cell)
            n = int(extra.pop("n_comparisons", n_comparisons))
            g = {"jnd_scale": G_JND_SCALE} if learner == "g" else {}
            points += [Phase5Point(run_dir, fold, family, ARM, learner, uid, n_comparisons=n,
                                   **extra, **g) for uid in uids]
    return points
