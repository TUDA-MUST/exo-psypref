"""The second review round's users: XD, users from other datasets, and HT (a) and (b), a
heavy-tailed generator and a population-family mixture.

Everything here builds **new** user sets beside a Phase 5a run, stored as the out-of-generator users
are (``RUN/<set>/users/<fold>.{json,npz}``, the exported users' schema with their promp_s0 best-profile distances,
:func:`prompref.experiment.robust.write_ood_users`). ``prompref/population`` and
``prompref/priors/data_prior.py`` are read only; the generator's own functions (``user_curve``,
``sample_draw``) are called, never changed.

**XD.** The unchanged generator pipeline
(``user_curve``: scales, filter, lag, landmark warp, device clip, the run's own factor distributions,
``sample_draw``) applied to an external subject's own hip moment, Nm/kg times body mass
(``prompref.data_external``). 200 users per cell, drawn over subjects and legs with replacement;
user ``i`` (0-199) is served by fold ``LOSO[i % 5]``'s Molinaro prior (round-robin):

- ``xd-fy``: Fukuchi et al. 2018, young adults, treadmill at comfortable speed (T05, 100 %);
- ``xd-fo``: Fukuchi et al. 2018, older adults (50-84 years), T05;
- ``xd-r``: Reznick et al. 2021, 1.0 m/s level walking and the -10, -5, +5 and +10 degree inclines,
  equal shares (users ``40 c`` to ``40 c + 39`` walk condition ``c``, so every fold sees each
  condition 8 times).

A draw that the generator rejects (a landmark-order violation) is redrawn on the same leg, as the
training curves are (``generate_training_curves``).

**HT (a), the heavy-tailed generator.** The run's generator with every Student-t factor at nu = 1.5
instead of 3 (the two magnitude scales) and 10 (the lag and the landmark offsets); every scale
parameter is the run's (``factors.ShiftSpread``, ``SCALE_LOG_SPREAD``, ``TimingSpread``'s scale
matrix). ``factors.FACTOR_NU`` is a module constant of the protected population code, so the draw is
:func:`sample_draw_nu`, which with the run's own nus reproduces ``factors.sample_draw`` exactly
(tested). Users are made as ``users.make_users`` makes them, from each fold's held-out Molinaro
trials: a trial and a draw per attempt, an attempt the generator cannot turn into a curve (a
landmark-order violation, or no cycle clear of the lag) is redrawn, both trial and factors. User
``j`` of a fold keeps the answer seed of the fold's population user ``j`` (pop0jj), so its answers
come from the stream that user's did.

**HT (b), the mixture.** Per fold 80 % population users and 20 % family ideals: the fold's population
users pop000-pop031 as exported (curve, answer seed, best-profile distances), and four Schaefer (ii) and four Franks
ideals sampled within their published ranges as in X (``xexpr.published_ideal`` with X's keys
``<fold>/schaefer000``-``003`` and ``<fold>/franks000``-``003``).
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..population.curves import LANDMARKS
from ..population.factors import (SCALE_CENTRE, SCALE_LOG_SPREAD, FactorDraw, FactorSwitches,
                                  ShiftSpread, TimingSpread, sample_draw)
from ..population.folds import child_sequence
from ..population.generator import OrderViolation, user_curve

#: Root entropy of every R2 user set; the set's CRC-32 is the second word (and a fold's the third).
R2_SEED = 0x0E2E2
XD_SETS = ("xd-fy", "xd-fo", "xd-r")
HT_GEN = "ht-gen"
HT_MIX = "ht-mix"
#: Users per XD cell and per fold of every R2 set.
XD_N_USERS = 200
N_PER_FOLD = 40
#: Fukuchi's comfortable treadmill speed (100 %).
FUKUCHI_CONDITION = "T05"
#: XD-Fy's and XD-Fo's age groups (``ExternalTrial.age_group``).
XD_AGE = {"xd-fy": "young", "xd-fo": "older"}
#: XD-R's conditions in Reznick's labels (``speed_incline``): 1.0 m/s level, then the inclines.
REZNICK_CONDITIONS = ("s1_i0", "s1_in10", "s1_in5", "s1_i5", "s1_i10")
#: Give up on an XD leg after this many consecutive rejected draws (``generate_training_curves``).
MAX_CONSECUTIVE_REJECTIONS = 200
#: HT (a)'s planned tail.
HT_NU = 1.5
#: HT (a): give up on a fold after this many rejected attempts in all.
HT_MAX_REJECTIONS = 20_000
#: HT (b): population users kept per fold (pop000-pop031) and family ideals per family and fold.
HT_MIX_POPULATION = 32
HT_MIX_PER_FAMILY = 4
HT_MIX_FAMILIES = ("schaefer", "franks")


@dataclass(frozen=True, eq=False)
class R2User:
    """A user of an R2 set, with the fields ``user_floors`` and ``write_ood_users`` need."""

    user_id: str
    kind: str
    subject: str
    side: str
    answer_seed: int
    draw: dict
    curve_nm: np.ndarray


def set_sequence(name: str, *more: str) -> np.random.SeedSequence:
    """``[R2_SEED, crc32(name), crc32(more[0]), ...]``: the root stream of one set (or fold of it).
    Child 0 holds the users' draws (user i: child i), child 1 their answer seeds (user i: child i)."""
    words = [R2_SEED, zlib.crc32(name.encode("utf-8"))]
    words += [zlib.crc32(m.encode("utf-8")) for m in more]
    return np.random.SeedSequence(words)


def answer_seed(root: np.random.SeedSequence, index: int) -> int:
    """``make_users``'s rule: ``child_1_i.generate_state(1, uint64)``."""
    return int(child_sequence(child_sequence(root, 1), index).generate_state(1, dtype=np.uint64)[0])


def _draw_json(draw: FactorDraw) -> dict:
    return json.loads(json.dumps(draw.to_dict()))


# ------------------------------------------------------------------------------------------- XD
def xd_fold(index: int, folds) -> str:
    """Round-robin: user ``index`` is served by ``folds[index % len(folds)]``."""
    return folds[int(index) % len(folds)]


def xd_condition(name: str, index: int) -> str:
    """The condition user ``index`` of XD cell ``name`` walks (XD-R: 40 users per condition)."""
    if name == "xd-r":
        return REZNICK_CONDITIONS[int(index) // (XD_N_USERS // len(REZNICK_CONDITIONS))]
    return FUKUCHI_CONDITION


def xd_legs(name: str, trials) -> dict[str, list]:
    """The cell's legs per condition, each sorted by (subject, side): Fukuchi T05 of the cell's age
    group, or Reznick's five 1.0 m/s conditions."""
    out: dict[str, list] = {}
    for trial in trials:
        if name in XD_AGE:
            ok = (trial.dataset == "fukuchi2018" and trial.condition == FUKUCHI_CONDITION
                  and trial.age_group == XD_AGE[name])
        else:
            ok = trial.dataset == "reznick2021" and trial.condition in REZNICK_CONDITIONS
        if ok:
            out.setdefault(trial.condition, []).append(trial)
    for legs in out.values():
        legs.sort(key=lambda t: (t.subject, t.side))
    return out


def make_xd_users(name: str, trials, folds, *, switches: FactorSwitches, timing: TimingSpread,
                  shift: ShiftSpread, n_users: int = XD_N_USERS) -> list[R2User]:
    """XD cell ``name``'s users in index order (see the module docstring)."""
    if name not in XD_SETS:
        raise ValueError(f"XD cell must be one of {XD_SETS}, got {name!r}")
    legs = xd_legs(name, trials)
    root = set_sequence(name)
    draws = child_sequence(root, 0)
    users = []
    for i in range(int(n_users)):
        condition = xd_condition(name, i)
        pool = legs.get(condition, [])
        if not pool:
            raise ValueError(f"{name}: no leg walks {condition}")
        rng = np.random.default_rng(child_sequence(draws, i))
        trial = pool[int(rng.integers(len(pool)))]
        rejected = 0
        while True:
            draw = sample_draw(rng, switches, timing, shift)
            try:
                curve = user_curve(trial, draw)
                break
            except OrderViolation:
                rejected += 1
                if rejected > MAX_CONSECUTIVE_REJECTIONS:
                    raise RuntimeError(f"{name} user {i}: {rejected} draws rejected on "
                                       f"{trial.subject}_{trial.side}")
        fold = xd_fold(i, folds)
        users.append(R2User(
            f"{fold}/{name}{i:03d}", "population", f"{trial.dataset}/{trial.subject}", trial.side,
            answer_seed(root, i),
            dict(factors=_draw_json(draw), dataset=trial.dataset, condition=trial.condition,
                 incline_deg=float(trial.incline_deg), age_group=trial.age_group,
                 age_years=float(trial.age_years), mass_kg=float(trial.mass_kg),
                 speed_mps=float(trial.speed_mps), curve_source=trial.curve_source,
                 n_legs_in_condition=len(pool), rejected_draws=int(rejected), index=i),
            curve))
    return users


# ----------------------------------------------------------------------------------------- HT (a)
def sample_draw_nu(rng: np.random.Generator, switches: FactorSwitches, timing: TimingSpread,
                   shift: ShiftSpread, *, nu_shift: float, nu_scale: float,
                   nu_timing: float) -> FactorDraw:
    """``factors.sample_draw`` with the three tails as arguments: the lag's (``shift.nu`` there), the
    two magnitude scales' (``factors.FACTOR_NU`` there) and the landmark offsets' (``timing.nu``
    there). Same random calls in the same order, same arithmetic; every scale is the given one's."""
    t_shift = rng.standard_t(nu_shift)
    t_ext = rng.standard_t(nu_scale)
    t_flex = rng.standard_t(nu_scale)
    normal = rng.standard_normal(len(LANDMARKS))
    chi2 = rng.chisquare(nu_timing)

    lag = shift.centre_ms + (shift.scale_ms * t_shift if switches.shift else 0.0)
    ext = SCALE_CENTRE * float(np.exp(SCALE_LOG_SPREAD * t_ext)) if switches.ext_scale else SCALE_CENTRE
    flex = (SCALE_CENTRE * float(np.exp(SCALE_LOG_SPREAD * t_flex)) if switches.flex_scale
            else SCALE_CENTRE)
    if switches.feature_timing:
        jitter = 1e-9 * np.trace(timing.scale_matrix) * np.eye(len(LANDMARKS))
        chol = np.linalg.cholesky(timing.scale_matrix + jitter)
        offsets = (chol @ normal) * np.sqrt(nu_timing / chi2)
    else:
        offsets = np.zeros(len(LANDMARKS))
    return FactorDraw(float(lag), float(ext), float(flex), tuple(float(o) for o in offsets))


def make_ht_gen_users(fold: str, held_out_trials, population_seeds: list[int], *,
                      switches: FactorSwitches, timing: TimingSpread, shift: ShiftSpread,
                      nu: float = HT_NU, n_population: int = N_PER_FOLD) -> tuple[list[R2User], dict]:
    """HT (a)'s users of one fold, ``<fold>/ht000`` onward, and the rejection counts."""
    if len(population_seeds) < n_population:
        raise ValueError(f"{fold}: {len(population_seeds)} population answer seeds for "
                         f"{n_population} users")
    rng = np.random.default_rng(child_sequence(set_sequence(HT_GEN, fold), 0))
    users: list[R2User] = []
    rejected = dict(order=0, no_cycle=0, not_finite=0)
    while len(users) < n_population:
        trial = held_out_trials[int(rng.integers(len(held_out_trials)))]
        draw = sample_draw_nu(rng, switches, timing, shift, nu_shift=nu, nu_scale=nu,
                              nu_timing=nu)
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                curve = user_curve(trial, draw)
        except OrderViolation:
            rejected["order"] += 1
        except ValueError as exc:
            if "no cycle clear" not in str(exc):
                raise
            rejected["no_cycle"] += 1
        else:
            if np.all(np.isfinite(curve)):
                index = len(users)
                users.append(R2User(f"{fold}/ht{index:03d}", "population", trial.subject,
                                    trial.side, int(population_seeds[index]),
                                    dict(factors=_draw_json(draw), nu=float(nu)), curve))
                continue
            rejected["not_finite"] += 1
        if sum(rejected.values()) > HT_MAX_REJECTIONS:
            raise RuntimeError(f"{fold}: {rejected} HT (a) attempts rejected")
    return users, rejected


# ----------------------------------------------------------------------------------------- HT (b)
def make_ht_mix_users(fold: str, population: dict, franks_prior) -> list[R2User]:
    """HT (b)'s users of one fold: pop000-pop031 as exported, then four Schaefer (ii) and four Franks
    ideals with X's keys and seeds (``xexpr.published_ideal``)."""
    from .xexpr import ideal_key, published_ideal

    pops = sorted(uid for uid, u in population.items() if u.kind == "population")
    users = []
    for uid in pops[:HT_MIX_POPULATION]:
        u = population[uid]
        users.append(R2User(uid, "population", u.subject, u.side, int(u.answer_seed),
                            dict(source="exported population user", draw=u.draw),
                            np.asarray(u.curve_nm, dtype=float)))
    if len(users) != HT_MIX_POPULATION:
        raise ValueError(f"{fold}: {len(users)} population users, expected {HT_MIX_POPULATION}")
    root = set_sequence(HT_MIX, fold)
    n = 0
    for generator in HT_MIX_FAMILIES:
        for j in range(HT_MIX_PER_FAMILY):
            key = ideal_key(fold, generator, j)
            theta, curve = published_ideal(generator, key,
                                           franks_prior if generator == "franks" else None)
            users.append(R2User(key, "population", f"{generator}-published", "-",
                                answer_seed(root, n),
                                dict(source=f"xexpr.published_ideal({generator!r}, {key!r})",
                                     generator=generator, theta=[float(v) for v in theta]),
                                np.asarray(curve, dtype=float)))
            n += 1
    return users


def exported_floor_entry(user) -> dict:
    """An exported user's promp_s0 best-profile distance entry in ``user_floors``' shape (for the mixture's
    population users, whose best-profile distances are the exported ones)."""
    fam = user.families["promp_s0"]
    return dict(floor_nm=float(fam["floor_nm"]), reach_050=float(fam["reach_050"]),
                reach_025=float(fam["reach_025"]), radius_floor=float(fam["radius"]))


def r2_config(**fields) -> dict:
    """A user set's config block: the test plan and the fields given."""
    return dict(schema_note="ICORR reframe section 9a, R2 (second review round)", **fields)


def write_set(run_dir, fold: str, name: str, users: list[R2User], floors: list[dict],
              config: dict) -> Path:
    from .robust import write_ood_users

    return write_ood_users(run_dir, fold, users, floors, config, user_set=name)
