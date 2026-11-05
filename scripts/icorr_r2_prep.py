"""Inputs of the second review round's cells (our design notes, R2; XD's test plan after conversion, before any
session). Each step writes only new files and its checks to ``OUT/r2_prep_<step>.json``.

Usage:
    python scripts/icorr_r2_prep.py --run <main>/out/phase5-schaefer --step STEP
        [--workers 4] [--out <main>/out/icorr]

Steps:

- ``xd-fy``, ``xd-fo``, ``xd-r`` (XD): the users of other datasets
  (:func:`prompref.experiment.r2.make_xd_users`; Fukuchi 2018 and Reznick 2021 from
  ``prompref.data_external``, raw data outside the repository), their promp_s0 best-profile distances and reach
  radii by 5a's ``user_floors`` with the serving fold's prior and 20 starts, into ``RUN/<set>/``.
- ``ht-gen`` (HT (a)): the generator with every Student-t factor at nu = 1.5, from each fold's
  held-out Molinaro trials (needs the S: share or ``PREFTUNE_DATA``), best-profile distances as above.
- ``ht-mix`` (HT (b)): per fold pop000-pop031 as exported and four Schaefer (ii) and four Franks
  ideals sampled within their published ranges as in X; best-profile distances of the family ideals as above.
- ``links`` (HT (c)): the Cauchy link's scale, matched as S5's (``scripts/icorr_stress_prep.py``):
  the median |p - 1/2| over REPORT section 9's 40,000 prior-scale pairs of the 200 population users
  equals the plain s = 1.0 probit simulated user's; checked through ``build_observer`` on those pairs and
  on fresh ones.
- ``pess`` (PESS): S1's phase weights (SD of log w 1.0, seeded from each user's answer seed) for the
  "ood-pb" users: the matched scale, the own generic distance and the own best-profile distance
  (``icorr_wph_prep.user_entry``), into ``RUN/ood-pb/phase_weight/pref-sd1/``.
- ``base-select --learner {cmaes,gp_bald}`` (BASE): after the tuning fold's sessions of
  ``e-base-cma`` or ``e-base-gp``, the setting with the lowest median error of the identified
  assistance at k = 24 over that fold's 40 population users (section E8's rule), into
  ``OUT/base_tuned.json``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompref.experiment import icorr_grid, phase5 as p5, r2  # noqa: E402
from prompref.experiment.floors5c import fold_seed  # noqa: E402
from prompref.experiment.metrics import prior_mean_curve  # noqa: E402
from prompref.experiment.phase5c_grid import LOSO  # noqa: E402
from prompref.experiment.phase_weighted import (PHASE_WEIGHT_SCHEMA, cell_constants,  # noqa: E402
                                                phase_weight_path)
from prompref.experiment.robust import load_ood_users, ood_floors_cache_path  # noqa: E402
from prompref.population.export import population_from_config  # noqa: E402
from prompref.population.users import FloorCache, prior_digest, user_floors  # noqa: E402
from prompref.priors.data_prior import DataPrior  # noqa: E402

MAX_WORKERS = 8
K_SELECT = 24


def stamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def _q(values, qs=(0.0, 0.1, 0.5, 0.9, 1.0)) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(values, dtype=float), qs)]


def _fold_config(run: str, fold: str) -> dict:
    return json.loads((Path(run) / "folds" / f"{fold}.json").read_text(encoding="utf-8"))["config"]


def run_population(run: str):
    """The run's (shift, switches, timing) and best-profile distance starts, the same in every LOSO fold (checked)."""
    configs = [_fold_config(run, fold) for fold in LOSO]
    keys = ("timing", "shift", "switches", "floor_starts")
    for c in configs[1:]:
        if any(c[k] != configs[0][k] for k in keys):
            raise ValueError("the LOSO folds' population configs differ")
    shift, switches, timing = population_from_config(configs[0])
    return shift, switches, timing, int(configs[0]["floor_starts"])


def floors_for(run: str, fold: str, name: str, users, workers: int, starts: int) -> list[dict]:
    prior = DataPrior.load(Path(run) / "priors" / f"{fold}-promp_s0")
    return user_floors(prior.design(), prior.normalization(), users, n_starts=starts,
                       seed_sequence=fold_seed(fold, name, "promp_s0"), workers=workers,
                       cache=FloorCache(ood_floors_cache_path(run, fold, name)),
                       family=f"promp_s0-{name}", prior_hash=prior_digest(prior.mu, prior.sigma))


def _set_summary(run: str, name: str, extra: dict) -> dict:
    """Checks every set step writes: per fold the user count and best-profile distance and size quantiles, all
    loaded back through ``load_ood_users``, against the run's population users."""
    rms = lambda c: float(np.sqrt(np.mean(np.asarray(c, dtype=float) ** 2)))  # noqa: E731
    per_fold, floors, sizes, d_gen = {}, [], [], []
    for fold in LOSO:
        loaded = load_ood_users(run, fold, name)
        generic = prior_mean_curve(p5.arm_space(p5._prior(run, fold, "promp_s0"), "tight").design)
        f = [u.families["promp_s0"]["floor_nm"] for u in loaded.values()]
        g = [rms(u.curve_nm - generic) for u in loaded.values()]
        per_fold[fold] = dict(n=len(loaded), floor_median=float(np.median(f)))
        floors += f
        sizes += [rms(u.curve_nm) for u in loaded.values()]
        d_gen += g
    pop = [u for fold in LOSO for u in p5._users(run, fold).values() if u.kind == "population"]
    return dict(when=stamp(), run=run, set=name, per_fold=per_fold,
                n_users=int(sum(v["n"] for v in per_fold.values())),
                floor_nm=_q(floors), ideal_rms_nm=_q(sizes), generic_distance_nm=_q(d_gen),
                room_nm=_q(np.asarray(d_gen) - np.asarray(floors)),
                population_floor_nm=_q([u.families["promp_s0"]["floor_nm"] for u in pop]),
                population_ideal_rms_nm=_q([rms(u.curve_nm) for u in pop]), **extra)


# ------------------------------------------------------------------------------------- XD
def load_external(name: str):
    if name == "xd-r":
        from prompref.data_external import reznick

        return reznick.load_trials()
    from prompref.data_external import fukuchi

    return fukuchi.load_trials(conditions=[r2.FUKUCHI_CONDITION])


def step_xd(run: str, name: str, out: Path, workers: int) -> int:
    started = time.perf_counter()
    shift, switches, timing, starts = run_population(run)
    loaded = load_external(name)
    users = r2.make_xd_users(name, loaded.trials, LOSO, switches=switches, timing=timing,
                             shift=shift)
    legs = r2.xd_legs(name, loaded.trials)
    for fold in LOSO:
        mine = [u for u in users if u.user_id.startswith(fold + "/")]
        floors = floors_for(run, fold, name, mine, workers, starts)
        config = r2.r2_config(
            cell={"xd-fy": "XD-Fy", "xd-fo": "XD-Fo", "xd-r": "XD-R"}[name],
            generator="prompref.experiment.r2.make_xd_users: generator.user_curve on the "
                      "external subject's own hip moment (Nm/kg x body mass), the run's factor "
                      "distributions (factors.sample_draw)",
            legs={c: [f"{t.subject}_{t.side}" for t in v] for c, v in legs.items()},
            folds="round-robin: user i is served by LOSO[i % 5]", seed=r2.R2_SEED,
            floor_starts=starts, floor_seed=f"floors5c.fold_seed({fold!r}, {name!r}, 'promp_s0')",
            prior=f"priors/{fold}-promp_s0")
        r2.write_set(run, fold, name, mine, floors, config)
        print(f"{name} {fold}: {len(mine)} users [{time.perf_counter() - started:.0f} s]",
              flush=True)
    p5._ood_users.cache_clear()
    conds: dict = {}
    for u in users:
        conds[u.draw["condition"]] = conds.get(u.draw["condition"], 0) + 1
    subjects = sorted({u.subject for u in users})
    extra = dict(seconds=round(time.perf_counter() - started, 1), conditions=conds,
                 n_subjects_drawn=len(subjects),
                 n_legs_available={c: len(v) for c, v in legs.items()},
                 rejected_draws=int(sum(u.draw["rejected_draws"] for u in users)),
                 skipped_by_converter=len(loaded.skipped))
    _write(out / f"r2_prep_{name}.json", _set_summary(run, name, extra))
    return 0


# ----------------------------------------------------------------------------------- HT (a)
def step_ht_gen(run: str, out: Path, workers: int) -> int:
    from prompref.population.folds import data_available, load_trials, loso_folds

    if not data_available():
        print("Molinaro data not available (S: drive or PREFTUNE_DATA)")
        return 2
    started = time.perf_counter()
    shift, switches, timing, starts = run_population(run)
    folds = {f.name: f for f in loso_folds()}
    rejected = {}
    for fold in LOSO:
        trials = load_trials(folds[fold].held_out)
        pops = sorted(uid for uid, u in p5._users(run, fold).items() if u.kind == "population")
        seeds = [int(p5._users(run, fold)[uid].answer_seed) for uid in pops]
        users, rej = r2.make_ht_gen_users(fold, trials, seeds, switches=switches, timing=timing,
                                          shift=shift)
        rejected[fold] = rej
        floors = floors_for(run, fold, r2.HT_GEN, users, workers, starts)
        config = r2.r2_config(
            cell="HT-a", nu=r2.HT_NU,
            generator="prompref.experiment.r2.make_ht_gen_users: users.make_users' procedure with "
                      "r2.sample_draw_nu at nu = 1.5 for the lag, both magnitude scales and the "
                      "landmark offsets; scales as the run's",
            answer_seeds="user j = the fold's exported population user j's answer seed",
            seed=r2.R2_SEED, floor_starts=starts,
            floor_seed=f"floors5c.fold_seed({fold!r}, 'ht-gen', 'promp_s0')",
            prior=f"priors/{fold}-promp_s0", rejected=rej)
        r2.write_set(run, fold, r2.HT_GEN, users, floors, config)
        print(f"ht-gen {fold}: {len(users)} users, rejected {rej} "
              f"[{time.perf_counter() - started:.0f} s]", flush=True)
    p5._ood_users.cache_clear()
    users = [u for fold in LOSO for u in load_ood_users(run, fold, r2.HT_GEN).values()]
    f = np.array([[u.draw["factors"]["shift_ms"], u.draw["factors"]["ext_scale"],
                   u.draw["factors"]["flex_scale"]] for u in users])
    at_limit = [float(np.mean(np.abs(u.curve_nm) >= 12.0 - 1e-9)) for u in users]
    extra = dict(seconds=round(time.perf_counter() - started, 1), rejected=rejected,
                 shift_ms=_q(f[:, 0]), ext_scale=_q(f[:, 1]), flex_scale=_q(f[:, 2]),
                 share_of_cycle_at_device_limit=_q(at_limit))
    _write(out / "r2_prep_ht-gen.json", _set_summary(run, r2.HT_GEN, extra))
    return 0


# ----------------------------------------------------------------------------------- HT (b)
def step_ht_mix(run: str, out: Path, workers: int) -> int:
    from prompref.priors.franks_prior import FranksPrior

    started = time.perf_counter()
    _, _, _, starts = run_population(run)
    for fold in LOSO:
        population = p5._users(run, fold)
        franks = FranksPrior.load(Path(run) / "priors" / f"{fold}-franks")
        users = r2.make_ht_mix_users(fold, population, franks)
        pop = users[:r2.HT_MIX_POPULATION]
        fam = users[r2.HT_MIX_POPULATION:]
        floors = [r2.exported_floor_entry(population[u.user_id]) for u in pop]
        floors += floors_for(run, fold, r2.HT_MIX, fam, workers, starts)
        config = r2.r2_config(
            cell="HT-b", population=f"{fold}/pop000-pop0{r2.HT_MIX_POPULATION - 1}, as exported",
            families="xexpr.published_ideal: Schaefer (ii) and Franks, X's keys and seeds, "
                     f"{r2.HT_MIX_PER_FAMILY} each per fold",
            seed=r2.R2_SEED, floor_starts=starts,
            floor_seed=f"floors5c.fold_seed({fold!r}, 'ht-mix', 'promp_s0') for the family ideals",
            prior=f"priors/{fold}-promp_s0")
        r2.write_set(run, fold, r2.HT_MIX, users, floors, config)
        print(f"ht-mix {fold}: {len(users)} users [{time.perf_counter() - started:.0f} s]",
              flush=True)
    p5._ood_users.cache_clear()
    same = all(np.array_equal(load_ood_users(run, fold, r2.HT_MIX)[uid].curve_nm,
                              p5._users(run, fold)[uid].curve_nm)
               for fold in LOSO for uid in load_ood_users(run, fold, r2.HT_MIX)
               if "/pop" in uid)
    extra = dict(seconds=round(time.perf_counter() - started, 1),
                 population_curves_identical_to_export=bool(same))
    _write(out / "r2_prep_ht-mix.json", _set_summary(run, r2.HT_MIX, extra))
    return 0 if same else 1


# ------------------------------------------------------------------------------------- links
def step_links(run: str, out: Path) -> int:
    import icorr_robust_prep as rp
    import icorr_stress_prep as sp
    import phase5e_perturbation as pert

    started = time.perf_counter()
    du, users, measure_pairs = sp.abs_du(run, pert.SEED)
    _, _, fresh_pairs = sp.abs_du(run, rp.OOS_SEED)
    result = dict(schema="r2-links-v1", when=stamp(), run=run,
                  spec="ICORR reframe section 9a, R2, HT (c): Cauchy answers, decisiveness-matched "
                       "as in S5", method="icorr_stress_prep.matched_scale and .achieved",
                  users=f"{len(users)} population users (5 LOSO folds x 40)", links={})
    for link in p5.R2_LINKS:
        kappa = sp.matched_scale(du, link)
        result["links"][link] = dict(
            link_scale=kappa, link_scale_repr=repr(kappa),
            measurement_pairs=sp.achieved(run, users, measure_pairs, link, kappa),
            fresh_pairs=sp.achieved(run, users, fresh_pairs, link, kappa))
        m = result["links"][link]
        print(f"{link}: kappa {kappa!r}; achieved ratio {m['measurement_pairs']['ratio']:.12f} "
              f"(measurement), {m['fresh_pairs']['ratio']:.4f} (fresh)", flush=True)
    result["seconds"] = round(time.perf_counter() - started, 1)
    _write(out / "r2_prep_links.json", result)
    return 0


# -------------------------------------------------------------------------------------- PESS
def pess_entry(run: str, fold: str, user, weight, generic, design, theta_stored) -> dict:
    """``icorr_wph_prep.user_entry`` with one more start for both best-profile distances: the weights of the user's
    stored promp_s0 best-profile distance (the user set's best-profile distance cache). The 'ood-pb' ideals are Schaefer profiles
    whose clipped best-profile distance has local minima the 20 prior-draw starts can miss (one user's w = 1 best-profile distance
    missed the stored one by 8e-5 Nm in the first prep run)."""
    import icorr_wph_prep as wp
    import phase5e_perturbation as pert
    import zlib

    from prompref.experiment.phase_weighted import weighted_floor
    from prompref.user.simulated_user import PHASE_WEIGHTED, Observer, weighted_rms_distance

    duels = pert.duel_pairs(run, fold, user.user_id, pert.N_PAIRS)
    plain = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.0)
    d_rms = np.array([abs(plain.utility(a) - plain.utility(b)) for a, b in duels])
    d_w = np.array([abs(weighted_rms_distance(a, user.curve_nm, weight)
                        - weighted_rms_distance(b, user.curve_nm, weight)) for a, b in duels])
    raw_ratio = pert.decisiveness_ratio(d_w, d_rms)
    scale = 1.0 / raw_ratio
    simulated_user = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.1, metric=PHASE_WEIGHTED,
                      phase_weight=weight, utility_scale=scale)
    seed = [wp.FLOOR_SEED, zlib.crc32(user.user_id.encode("utf-8"))]
    rms_check = weighted_floor(design, user.curve_nm, np.ones_like(weight),
                               n_starts=wp.FLOOR_STARTS, seed=seed, extra_starts=[theta_stored])
    floor = weighted_floor(design, user.curve_nm, weight, n_starts=wp.FLOOR_STARTS, seed=seed,
                           extra_starts=[rms_check["theta"], theta_stored])
    return dict(answer_seed=int(user.answer_seed), raw_ratio=float(raw_ratio), scale=float(scale),
                generic_w_nm=float(weighted_rms_distance(generic, user.curve_nm, weight)),
                own_generic=float(simulated_user.own_distance(generic)),
                floor_w_nm=floor["floor_nm"], own_floor=float(scale * floor["floor_nm"]),
                floor_theta=floor["theta"], proj_nm=floor["proj_nm"],
                proj_clipped_nm=floor["proj_clipped_nm"], proj_in_box=floor["proj_in_box"],
                proj_clip_free=floor["proj_clip_free"], refine_gain_nm=floor["refine_gain_nm"],
                rms_floor_5a_nm=float(user.families["promp_s0"]["floor_nm"]),
                rms_floor_this_routine_nm=rms_check["floor_nm"],
                weight=[float(v) for v in weight])


def step_pess(run: str, out: Path) -> int:
    import icorr_wph_prep as wp
    import phase5e_perturbation as pert
    from prompref.user.simulated_user import Observer
    from prompref.user.phase_weight import pref_weight

    cell, user_set = icorr_grid.PESS_CELL["phase_weighting"], icorr_grid.PESS_CELL["user_set"]
    log_sd = 1.0
    started = time.perf_counter()
    entries_all, ratios = [], []
    for fold in LOSO:
        prior = p5._prior(run, fold, "promp_s0")
        design = prior.design()
        generic = prior_mean_curve(p5.arm_space(prior, pert.ARM).design)
        users = load_ood_users(run, fold, user_set)
        cache = FloorCache(ood_floors_cache_path(run, fold, user_set))
        starts = int(_fold_config(run, fold)["floor_starts"])
        entries = {}
        for uid, user in users.items():
            weight = pref_weight(user.answer_seed, log_sd)
            stored = cache.get(f"promp_s0-{user_set}", uid, user.curve_nm, floor_starts=starts,
                               prior_hash=prior_digest(prior.mu, prior.sigma))
            if stored is None:
                raise ValueError(f"{uid}: no stored floor in the {user_set} floor cache")
            entries[uid] = pess_entry(run, fold, user, weight, generic, design,
                                      np.asarray(stored["theta"], dtype=float))
        sidecar = dict(schema=PHASE_WEIGHT_SCHEMA, cell=cell, fold=fold, when=stamp(),
                       spec="ICORR reframe section 9a, R2, PESS: S1's phase weights on the "
                            "'ood-pb' users",
                       constants=cell_constants(cell),
                       weight_source=f"pref_weight(answer_seed, log_sd={log_sd}), seeded per user",
                       scale="1 / median over REPORT section 9's 200 prior-scale pairs of "
                             "|d_w(a) - d_w(b)| / |d_rms(a) - d_rms(b)|, the user's own ideal",
                       generic="prior_mean_curve of the tight arm's device-clipped design",
                       floor="phase_weighted.weighted_floor over the fold's 8-basis ProMP data "
                             "prior design (weight box +-30), behind the device clip",
                       users=entries)
        path = phase_weight_path(run, cell, fold, user_set)
        _write(path, sidecar)
        p5.phase_weight_sidecar.cache_clear()
        # The achieved decisiveness through build_observer, as a session builds the simulated user
        # (attentive, s = 1.0, the plain RMS simulated user as reference): 1 by construction.
        for uid, user in users.items():
            point = p5.Phase5Point(run, fold, "promp_s0", "tight", "a", uid, s=1.0, lapse=0.0,
                                   n_comparisons=40, phase_weighting=cell, user_set=user_set)
            observer, _ = p5.build_observer(point, user, s_user=1.0, plateau=0.0)
            plain = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.0)
            duels = pert.duel_pairs(run, fold, uid, pert.N_PAIRS)
            d_cell = [abs(observer.utility(a) - observer.utility(b)) for a, b in duels]
            d_rms = [abs(plain.utility(a) - plain.utility(b)) for a, b in duels]
            ratios.append(pert.decisiveness_ratio(np.asarray(d_cell), np.asarray(d_rms)))
        entries_all += list(entries.values())
        print(f"pess {fold}: {len(entries)} users [{time.perf_counter() - started:.0f} s]",
              flush=True)
    excess = np.array([e["rms_floor_this_routine_nm"] - e["rms_floor_5a_nm"] for e in entries_all])
    result = dict(when=stamp(), run=run, cell=cell, user_set=user_set,
                  seconds=round(time.perf_counter() - started, 1),
                  achieved_ratio_per_user=_q(ratios),
                  floor_check_w1=dict(n=int(excess.size), max_excess_over_stored_nm=float(excess.max()),
                                      n_above_by_tol=int(np.sum(excess > wp.FLOOR_TOL_NM))),
                  own_generic=_q([e["own_generic"] for e in entries_all]),
                  own_floor=_q([e["own_floor"] for e in entries_all]),
                  own_room=_q([e["own_generic"] - e["own_floor"] for e in entries_all]),
                  scale=_q([e["scale"] for e in entries_all]))
    _write(out / "r2_prep_pess.json", result)
    return 0 if result["floor_check_w1"]["n_above_by_tol"] == 0 else 1


# ------------------------------------------------------------------------------- BASE select
def step_base_select(run: str, out: Path, learner: str) -> int:
    block = {v: k for k, v in icorr_grid.BASE_BLOCK_LEARNER.items()}[learner]
    path = out / f"{block}.jsonl"
    settings = icorr_grid.BASE_SETTINGS[learner]
    fold = icorr_grid.BASE_TUNE_FOLD
    uids = icorr_grid._population_ids(run, fold)
    found = {i: {} for i in range(len(settings))}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") or row["fold"] != fold or row["learner"] != learner:
                continue
            for i, s in enumerate(settings):
                if all(row.get(k, "tight" if k == "arm" else None) == v for k, v in s.items()):
                    found[i][row["user_id"]] = float(row["distances"][K_SELECT - 1])
    missing = {i: len(set(uids) - set(v)) for i, v in found.items()}
    if any(missing.values()):
        print(f"{block}: tuning fold incomplete, missing users per setting {missing}")
        return 1
    med = {i: float(np.median([found[i][u] for u in uids])) for i in found}
    best = min(med, key=lambda i: (med[i], i))
    target = out / icorr_grid.BASE_TUNED_NAME
    tuned = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    tuned[learner] = dict(setting=settings[best], when=stamp(), fold=fold, n_users=len(uids),
                          rule="lowest median error of the identified assistance at k = 24 "
                               "(distances[23]) over the tuning fold's population users; ties to "
                               "the first in grid order",
                          median_err_k24=[dict(setting=settings[i], median=med[i]) for i in med])
    _write(target, tuned)
    print(f"{learner}: chose {settings[best]} (median errors {med})", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", required=True,
                    choices=r2.XD_SETS + (r2.HT_GEN, r2.HT_MIX, "links", "pess", "base-select"))
    ap.add_argument("--learner", choices=tuple(icorr_grid.BASE_SETTINGS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/../icorr")
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        ap.error(f"--workers must lie in 1..{MAX_WORKERS}: the machine is shared")
    run = str(Path(args.run).resolve())
    out = args.out or Path(run).parent / "icorr"
    if args.step in r2.XD_SETS:
        return step_xd(run, args.step, out, args.workers)
    if args.step == r2.HT_GEN:
        return step_ht_gen(run, out, args.workers)
    if args.step == r2.HT_MIX:
        return step_ht_mix(run, out, args.workers)
    if args.step == "links":
        return step_links(run, out)
    if args.step == "pess":
        return step_pess(run, out)
    if args.learner is None:
        ap.error("--step base-select needs --learner")
    return step_base_select(run, out, args.learner)


if __name__ == "__main__":
    sys.exit(main())
