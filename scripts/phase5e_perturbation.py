"""Perturbation strength of each Phase 5e cell.

How differently does a cell's simulated user answer than the plain s = 1.0 simulated user? The measure is the
mean |p_cell - p_plain| over duel pairs, where p is the probability of preferring the first
curve of the pair:

- **Pairs.** For each fold, its first ``n_users`` population users (sorted ids), and for each
  user ``n_pairs`` pairs from the fold's ProMP data prior in the tight arm: standard-normal
  normalized draws clipped to the arm's search region, mapped to curves through the device clip, as a
  learner's duels are. The draws are seeded from the user id, so every cell sees the same
  pairs.
- **Observers.** The cell's simulated user is built exactly as its sessions build it
  (:func:`prompref.experiment.phase5.build_observer`, at the user's own s); the plain simulated user
  is s = 1.0 against the user's ideal. Both are compared **without lapse**, so the lapse they
  share drops out -- except in the lapse cells, whose lapse is their only perturbation: there
  each simulated user keeps its own lapse (0.2 or 0.3 against 0.1), and the row says so.
- **Device noise** reaches the comparison as in a session: the noise cell's simulated user judges
  each pair through ``perceived``, from its own salted generator.
- **Drift.** A magnitude-drift cell's ideal moves over its 24 comparisons; pair i is judged
  against the ideal of comparison i mod 24, so the value is a session average.

Nothing here runs a session or writes a file; the report caches the result.
"""

from __future__ import annotations

import zlib

import numpy as np

from prompref.experiment import phase5
from prompref.experiment.phase5 import Phase5Point, arm_space, build_observer, per_user_s
from prompref.experiment.phase5e_grid import ARM, N_DRIFT
from prompref.population.export import load_fold_users
from prompref.user.drift import path_for_scale
from prompref.user.simulated_user import Observer

FAMILY = "promp_s0"
N_USERS = 20
N_PAIRS = 200
SEED = 0x5E9E27
PLAIN_LAPSE = 0.1


def pair_seed(user_id: str) -> list[int]:
    """The pairs' seed: fixed per user id, shared by every cell."""
    return [SEED, zlib.crc32(user_id.encode("utf-8"))]


def population_users(run_dir: str, fold: str, n_users: int = N_USERS) -> list[str]:
    users = load_fold_users(run_dir, fold)
    return sorted(uid for uid, u in users.items()
                  if u.kind == "population" and "/pop" in uid)[:n_users]


def duel_pairs(run_dir: str, fold: str, user_id: str, n_pairs: int = N_PAIRS) -> np.ndarray:
    """(n_pairs, 2, 101) device-clipped curves of normalized draws in the tight arm's search region."""
    space = arm_space(phase5._prior(run_dir, fold, FAMILY), ARM)
    design = space.as_design()
    z = np.random.default_rng(pair_seed(user_id)).standard_normal((n_pairs, 2, space.d))
    return np.array([[design.curve(space.clip_to_search_region(v)) for v in pair] for pair in z])


def _is_lapse_cell(fields: dict) -> bool:
    return float(fields.get("lapse", PLAIN_LAPSE)) != PLAIN_LAPSE


def _cell_judge(run_dir: str, fold: str, user, fields: dict):
    """A function (curve_a, curve_b, pair index) -> p for this cell's simulated user."""
    fields = dict(fields)
    cell_lapse = float(fields.pop("lapse", PLAIN_LAPSE))
    lapse = cell_lapse if cell_lapse != PLAIN_LAPSE else 0.0
    point = Phase5Point(run_dir, fold, FAMILY, ARM, "a", user.user_id, lapse=lapse,
                        n_comparisons=N_DRIFT if fields.get("drift_nm", 0.0) > 0.0 else 40,
                        **fields)
    s_user = per_user_s(point.s, point.s_user_logsd, user.answer_seed)
    if point.drift_nm > 0.0:
        if point.drift_axis != "magnitude":
            raise ValueError("only magnitude drift is a Phase 5e cell")
        path = path_for_scale(user.curve_nm, phase5._drift_entry(point)["scale"], N_DRIFT)
        steps = [Observer(ideal_nm=ideal, s=s_user, lapse=lapse) for ideal in path]
        return lambda a, b, i: steps[i % len(steps)].prob_prefers_a(a, b)
    observer, _ = build_observer(point, user, s_user=s_user, plateau=0.0)
    return lambda a, b, i: observer.prob_prefers_a(*observer.perceived(a, b))


def compute(run_dir: str, cells: dict[str, dict], *, folds, n_users: int = N_USERS,
            n_pairs: int = N_PAIRS) -> dict:
    """``{cell id: {mean_abs_dp, n_users, n_pairs, lapse_included, note}}`` for the cells
    ``{cell id: point fields}``. A cell that cannot be built (a missing sidecar, say) gets
    ``mean_abs_dp`` None and the reason in ``note``; it never raises."""
    run_dir = str(run_dir)
    sums = {cid: [] for cid in cells}
    users_seen = {cid: 0 for cid in cells}
    notes: dict[str, str] = {}
    for fold in folds:
        users = load_fold_users(run_dir, fold)
        for uid in population_users(run_dir, fold, n_users):
            user = users[uid]
            pairs = duel_pairs(run_dir, fold, uid, n_pairs)
            plain = {lapse: Observer(ideal_nm=user.curve_nm, s=1.0, lapse=lapse)
                     for lapse in (0.0, PLAIN_LAPSE)}
            for cid, fields in cells.items():
                if cid in notes:
                    continue
                try:
                    judge = _cell_judge(run_dir, fold, user, fields)
                except Exception as exc:          # noqa: BLE001 - reported, not raised
                    notes[cid] = f"{type(exc).__name__}: {exc}"
                    continue
                reference = plain[PLAIN_LAPSE if _is_lapse_cell(fields) else 0.0]
                sums[cid] += [abs(judge(a, b, i) - reference.prob_prefers_a(a, b))
                              for i, (a, b) in enumerate(pairs)]
                users_seen[cid] += 1
    out = {}
    for cid, fields in cells.items():
        if cid in notes:
            out[cid] = dict(mean_abs_dp=None, n_users=0, n_pairs=0,
                            lapse_included=_is_lapse_cell(fields), note=notes[cid])
            continue
        values = sums[cid]
        out[cid] = dict(mean_abs_dp=float(np.mean(values)) if values else None,
                        n_users=users_seen[cid], n_pairs=len(values),
                        lapse_included=_is_lapse_cell(fields), note="")
    return out


# --------------------------------------------------------------------------- decisiveness

def decisiveness_ratio(delta_cell, delta_rms) -> float:
    """Median over duel pairs of |du_cell| / |du_rms|, the cell's utility difference of the two
    curves against the plain s = 1.0 RMS simulated user's. Pairs the RMS simulated user cannot tell apart
    (du_rms = 0) have no ratio and drop out."""
    cell, rms = np.abs(np.asarray(delta_cell, float)), np.abs(np.asarray(delta_rms, float))
    keep = rms > 0.0
    return float(np.median(cell[keep] / rms[keep]))


def compute_decisiveness(run_dir: str, cells: dict[str, dict], *, folds,
                         n_users: int = N_USERS, n_pairs: int = N_PAIRS) -> dict:
    """``{cell id: {ratio, s_equiv, n_users, n_pairs, note}}`` for the landmark cells among
    ``cells``, on the same pairs and users as :func:`compute`. ``ratio`` is the median over all
    pairs of |du_landmark| / |du_rms(s = 1.0)|, u being each simulated user's utility (the landmark
    simulated user's is -D x true_jnd(s), the RMS simulated user's -RMS distance, so both are in Nm and the
    probit scale s is the same); ``s_equiv`` = 1 / ratio is the RMS simulated user who would be as
    decisive. A cell that cannot be built is reported with ratio None, never raised."""
    run_dir = str(run_dir)
    cells = {cid: f for cid, f in cells.items() if f.get("observer_metric") == "landmark"}
    deltas = {cid: ([], []) for cid in cells}
    users_seen = {cid: 0 for cid in cells}
    notes: dict[str, str] = {}
    for fold in folds:
        users = load_fold_users(run_dir, fold)
        for uid in population_users(run_dir, fold, n_users):
            user = users[uid]
            pairs = duel_pairs(run_dir, fold, uid, n_pairs)
            plain = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.0)
            d_rms = np.array([abs(plain.utility(a) - plain.utility(b)) for a, b in pairs])
            for cid, fields in cells.items():
                if cid in notes:
                    continue
                try:
                    point = Phase5Point(run_dir, fold, FAMILY, ARM, "a", user.user_id,
                                        lapse=0.0, **fields)
                    observer, _ = build_observer(point, user, plateau=0.0,
                                                 s_user=per_user_s(point.s, point.s_user_logsd,
                                                                   user.answer_seed))
                except Exception as exc:          # noqa: BLE001 - reported, not raised
                    notes[cid] = f"{type(exc).__name__}: {exc}"
                    continue
                deltas[cid][0].extend(abs(observer.utility(a) - observer.utility(b))
                                      for a, b in pairs)
                deltas[cid][1].extend(d_rms)
                users_seen[cid] += 1
    out = {}
    for cid in cells:
        if cid in notes or not deltas[cid][0]:
            out[cid] = dict(ratio=None, s_equiv=None, n_users=0, n_pairs=0,
                            note=notes.get(cid, "no pairs"))
            continue
        ratio = decisiveness_ratio(*deltas[cid])
        out[cid] = dict(ratio=ratio, s_equiv=1.0 / ratio, n_users=users_seen[cid],
                        n_pairs=len(deltas[cid][0]), note="")
    return out
