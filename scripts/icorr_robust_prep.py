"""Inputs and checks for the robustness blocks (our design notes:
"Home-advantage test at matched decisiveness" and "Out-of-generator and random-utility
simulated users").

Usage:
    python scripts/icorr_robust_prep.py --run <main>/out/phase5-schaefer --step STEP
        [--workers 8] [--out <main>/out/icorr]

Steps (each writes only new files, and its checks to ``OUT/robust_<step>.json``):

- ``landmark``: re-measures the two landmark cells' decisiveness ratios with
  ``scripts/phase5e_perturbation.py``'s ``compute_decisiveness`` (REPORT section 9: 5 folds x
  the first 20 population users x 200 prior-scale pairs) and requires them to equal the stored
  ``out/phase5e/perturbation_strength_decisiveness.json`` and ``icorr_grid.DECISIVENESS`` bit for
  bit. Then the achieved ratio of the matched cells (utility divided by the ratio): on the same
  pairs, where it is 1 by construction, and out of sample (population users 20-39, pairs from
  another seed), where it is a check.
- ``ood``: the out-of-generator users (5 folds x 40, :mod:`prompref.experiment.robust`), their
  promp_s0 best-profile distances and reach radii by 5a's ``user_floors`` with the fold's own prior and 20
  starts, written to ``RUN/ood/``; and the timing check against Schaefer's distribution.
- ``random``: per fold, regenerates the 20,000 training ideals (Molinaro trials on S:, as
  ``icorr_k_priors.py`` does) and checks them by refitting the stored 8-basis prior bit for bit;
  standardises their Fourier features; then per population user the centre, the
  matched-decisiveness scale and the utility of generic assistance, written to
  ``RUN/random_utility/``; then the achieved decisiveness ratio in and out of sample.
- ``umax`` (after ``random``): u_max per user, the best of 2-6 independent multi-start searches
  (more until two of them reach the best), written into the same sidecars.
- ``refine`` (after the e-rand block): more, diversified searches, also started from every
  session's delivered weights and a first pass's maxima (``--first-pass DIR``); u_max becomes
  the best of everything found, with its source, and the check counts sessions that delivered
  more than every search found.
- ``ood --magnitude per part``: the ruled primary OOD users: the same timings, z and answer seeds as the common-peak users,
  each part's peak its own population median x (1 + 0.26 z); best-profile distances as before; ``RUN/ood-pb/``.
- ``converge`` (after ``refine``): one more refinement search per user and round until the last
  two searches of every user agree within 1 % of its headroom (``--tol``), or ``--max-searches``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase5e_perturbation as pert  # noqa: E402
from prompref.experiment import icorr_grid, phase5 as p5  # noqa: E402
from prompref.experiment.floors5c import fold_seed  # noqa: E402
from prompref.experiment.metrics import prior_mean_curve  # noqa: E402
from prompref.experiment.phase5 import Phase5Point, build_observer, per_user_s  # noqa: E402
from prompref.experiment.phase5c_grid import LOSO  # noqa: E402
from prompref.experiment.robust import (MAGNITUDE_RULE, MAGNITUDE_RULE_PER_PART,  # noqa: E402
                                        OOD_MAGNITUDE_CV, OOD_N_POPULATION, OOD_NU,
                                        OOD_PER_PART_USER_SET, OOD_USER_SET,
                                        population_median_peaks,
                                        OOD_SEED, RANDOM_SCHEMA, UMAX_MAX_SEARCHES,
                                        UMAX_MIN_SEARCHES, TimingDistribution,
                                        combine_searches, device_ok, load_ood_users,
                                        make_ood_users, maximise_utility,
                                        ood_floors_cache_path, population_median_peak,
                                        random_utility_path, same_value,
                                        training_features_path, user_utility,
                                        write_ood_users)
from prompref.population.users import FloorCache, prior_digest, user_floors  # noqa: E402
from prompref.priors.data_prior import DataPrior  # noqa: E402
from prompref.user.simulated_user import Observer  # noqa: E402
from prompref.user.smooth_utility import (CENTRE_SALT, LENGTHSCALE, N_FEATURES, RFF_SALT, RHO,  # noqa: E402
                                          RandomSmoothUtility, draw_centre_index,
                                          fourier_coefficients)

MAX_WORKERS = 8
#: Out-of-sample decisiveness: another pair seed, and the population users the REPORT section
#: 9 measurement did not use.
OOS_SEED = 0x0D15C0
OOS_USERS = slice(20, 40)
#: u_max searches: search i of a user is seeded [UMAX_SEED, crc32(user id), i].
UMAX_SEED = 0x0A3A11


def stamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def pairs(run: str, fold: str, user_id: str, n_pairs: int, seed: int) -> np.ndarray:
    """``phase5e_perturbation.duel_pairs`` with the pair seed ``[seed, crc32(user_id)]``; at
    ``seed = pert.SEED`` it is ``duel_pairs`` itself (checked in :func:`step_landmark`)."""
    space = p5.arm_space(p5._prior(run, fold, pert.FAMILY), pert.ARM)
    design = space.as_design()
    z = np.random.default_rng([seed, zlib.crc32(user_id.encode("utf-8"))]).standard_normal(
        (n_pairs, 2, space.d))
    return np.array([[design.curve(space.clip_to_search_region(v)) for v in pair] for pair in z])


def measure(run: str, cells: dict, *, folds, users: slice, seed: int,
            n_pairs: int = pert.N_PAIRS) -> dict:
    """REPORT section 9's decisiveness, generalised to any user slice, pair seed and observer
    metric: per cell, the median over all pairs of |du_cell| / |du_rms(s = 1.0)|
    (``pert.decisiveness_ratio``), and also the ratio of the median |du|s."""
    deltas = {cid: ([], []) for cid in cells}
    per_user = {cid: [] for cid in cells}
    for fold in folds:
        all_users = p5._users(run, fold)
        uids = sorted(uid for uid, u in all_users.items()
                      if u.kind == "population" and "/pop" in uid)[users]
        for uid in uids:
            user = all_users[uid]
            duels = pairs(run, fold, uid, n_pairs, seed)
            plain = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.0)
            d_rms = np.array([abs(plain.utility(a) - plain.utility(b)) for a, b in duels])
            for cid, fields in cells.items():
                point = Phase5Point(run, fold, pert.FAMILY, pert.ARM, "a", uid, lapse=0.0, **fields)
                observer, _ = build_observer(point, user, plateau=0.0,
                                             s_user=per_user_s(point.s, point.s_user_logsd,
                                                               user.answer_seed))
                d_cell = np.array([abs(observer.utility(a) - observer.utility(b))
                                   for a, b in duels])
                deltas[cid][0].extend(d_cell)
                deltas[cid][1].extend(d_rms)
                per_user[cid].append(pert.decisiveness_ratio(d_cell, d_rms))
    out = {}
    for cid, (cell, rms) in deltas.items():
        cell, rms = np.asarray(cell), np.asarray(rms)
        out[cid] = dict(ratio=pert.decisiveness_ratio(cell, rms),
                        ratio_of_medians=float(np.median(cell) / np.median(rms)),
                        per_user_ratio_quantiles=[float(q) for q in np.quantile(
                            per_user[cid], [0.0, 0.1, 0.5, 0.9, 1.0])],
                        n_users=len(per_user[cid]), n_pairs=int(cell.size))
    return out


# ------------------------------------------------------------------------------- landmark

def step_landmark(run: str, out: Path) -> int:
    stored_path = Path(run).parent / "phase5e" / "perturbation_strength_decisiveness.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    cells = stored["params"]["cells"]
    folds = tuple(stored["params"]["folds"])
    started = time.perf_counter()
    plain = pert.compute_decisiveness(run, cells, folds=folds, n_users=stored["params"]["n_users"],
                                      n_pairs=stored["params"]["n_pairs"])
    grid_key = {"landmark": None, "landmark25": 0.25}
    reproduced = {cid: dict(measured=plain[cid]["ratio"], stored=stored["values"][cid]["ratio"],
                            grid=icorr_grid.DECISIVENESS[grid_key[cid]],
                            identical=plain[cid]["ratio"] == stored["values"][cid]["ratio"]
                            == icorr_grid.DECISIVENESS[grid_key[cid]])
                  for cid in cells}
    matched = {f"{cid}-matched": dict(fields, decisiveness_ratio=icorr_grid.DECISIVENESS[grid_key[cid]])
               for cid, fields in cells.items()}
    in_sample = pert.compute_decisiveness(run, matched, folds=folds,
                                          n_users=stored["params"]["n_users"],
                                          n_pairs=stored["params"]["n_pairs"])
    # The pairs helper at the REPORT's own seed is duel_pairs itself.
    uid = pert.population_users(run, folds[0], 1)[0]
    same_pairs = bool(np.array_equal(pairs(run, folds[0], uid, 5, pert.SEED),
                                     pert.duel_pairs(run, folds[0], uid, 5)))
    both = dict(cells, **matched)
    in_sample_full = measure(run, both, folds=folds, users=slice(0, pert.N_USERS), seed=pert.SEED)
    out_of_sample = measure(run, both, folds=folds, users=OOS_USERS, seed=OOS_SEED)
    result = dict(when=stamp(), run=run, folds=folds, seconds=round(time.perf_counter() - started, 1),
                  reproduced=reproduced, pairs_helper_is_duel_pairs=same_pairs,
                  matched_in_sample_compute_decisiveness=in_sample,
                  in_sample=in_sample_full, out_of_sample=out_of_sample,
                  out_of_sample_definition=f"population users {OOS_USERS.start}-{OOS_USERS.stop - 1} "
                  f"per fold, 200 pairs each, pair seed [{OOS_SEED}, crc32(user id)]")
    _write(out / "robust_decisiveness.json", result)
    ok = all(v["identical"] for v in reproduced.values()) and same_pairs
    for cid in cells:
        print(f"{cid}: ratio {plain[cid]['ratio']!r} (stored identical: {reproduced[cid]['identical']}); "
              f"matched in sample {in_sample[cid + '-matched']['ratio']:.6f}, out of sample "
              f"{out_of_sample[cid + '-matched']['ratio']:.4f} (unmatched "
              f"{out_of_sample[cid]['ratio']:.4f})", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------------------------ ood

def _population_curves(run: str, folds) -> np.ndarray:
    return np.array([u.curve_nm for fold in folds for u in p5._users(run, fold).values()
                     if u.kind == "population"])


def step_ood(run: str, out: Path, workers: int) -> int:
    inputs = Path(run).parent / "schaefer" / "prior_inputs.json"
    dist = TimingDistribution.load(inputs)
    if dist.nu != OOD_NU:
        raise ValueError(f"{inputs} recommends nu {dist.nu}, the planned design says {OOD_NU}")
    population = _population_curves(run, LOSO)
    median_peak = population_median_peak(population)
    summary = json.loads(inputs.read_text(encoding="utf-8"))["summary"]
    started = time.perf_counter()
    per_fold = {}
    for fold in LOSO:
        prior = DataPrior.load(Path(run) / "priors" / f"{fold}-promp_s0")
        fold_config = json.loads((Path(run) / "folds" / f"{fold}.json").read_text(encoding="utf-8"))["config"]
        starts = int(fold_config["floor_starts"])
        users = make_ood_users(fold, dist, median_peak)
        floors = user_floors(prior.design(), prior.normalization(), users, n_starts=starts,
                             seed_sequence=fold_seed(fold, "ood", "promp_s0"), workers=workers,
                             cache=FloorCache(ood_floors_cache_path(run, fold)),
                             family="promp_s0-ood", prior_hash=prior_digest(prior.mu, prior.sigma))
        config = dict(schema_note="ICORR reframe section 9a, O: out-of-generator ideals",
                      generator="prompref.experiment.robust.make_ood_users",
                      profile="SchaeferDesign(peak, peak).curve(t), the four-half-cosine profile",
                      timing_source=str(inputs), timing_mean=dist.mean.tolist(),
                      timing_cov=dist.cov.tolist(), nu=dist.nu,
                      scatter="cov * (nu - 2) / nu (t covariance = cov)",
                      device_limits="gaps t2+100-t1, t3-t2, t4-t3, t1-t4 in [10, 40] %ST; "
                                    "t2 <= 50; t4 >= 50; t1 in [50, 150)",
                      magnitude_rule=MAGNITUDE_RULE, median_peak_nm=median_peak,
                      median_peak_source=f"{len(population)} population users of {', '.join(LOSO)}",
                      magnitude_cv=OOD_MAGNITUDE_CV, seed=OOD_SEED,
                      n_population=OOD_N_POPULATION, floor_starts=starts,
                      floor_seed=f"floors5c.fold_seed({fold!r}, 'ood', 'promp_s0')",
                      prior=f"priors/{fold}-promp_s0", prior_hash=prior_digest(prior.mu, prior.sigma))
        write_ood_users(run, fold, users, floors, config)
        loaded = load_ood_users(run, fold)
        per_fold[fold] = dict(n=len(loaded),
                              floor_median=float(np.median([u.families["promp_s0"]["floor_nm"]
                                                            for u in loaded.values()])),
                              rejected_draws=int(sum(u.draw["rejected_timing_draws"]
                                                     for u in users)))
        print(f"{fold}: {len(users)} OOD users, floor median {per_fold[fold]['floor_median']:.3f} Nm "
              f"[{time.perf_counter() - started:.0f} s]", flush=True)
    # The timing check against Schaefer.
    t = np.array([u.draw["t_pct_st"] for fold in LOSO for u in load_ood_users(run, fold).values()])
    peaks = np.array([u.draw["peak_nm"] for fold in LOSO for u in load_ood_users(run, fold).values()])
    curves = np.array([u.curve_nm for fold in LOSO for u in load_ood_users(run, fold).values()])
    rng = np.random.default_rng(20261019)
    big = dist.draw(rng, 200_000)
    keep = device_ok(big)
    rms = lambda c: np.sqrt(np.mean(np.asarray(c) ** 2, axis=-1))  # noqa: E731
    floors_pop = [u.families["promp_s0"]["floor_nm"] for fold in LOSO
                  for u in p5._users(run, fold).values() if u.kind == "population"]
    floors_ood = [u.families["promp_s0"]["floor_nm"] for fold in LOSO
                  for u in load_ood_users(run, fold).values()]
    result = dict(
        when=stamp(), run=run, seconds=round(time.perf_counter() - started, 1),
        schaefer=dict(mean=summary["t_mean"], sd=summary["t_sd"],
                      reported_device_rejection_diag_nu10=0.141),
        users_200=dict(mean=t.mean(axis=0).tolist(), sd=t.std(axis=0, ddof=1).tolist(),
                       all_device_ok=bool(device_ok(t).all()),
                       rejected_share=float(sum(v["rejected_draws"] for v in per_fold.values())
                                            / (sum(v["rejected_draws"] for v in per_fold.values()) + len(t)))),
        draws_200k=dict(rejected_share=float(1.0 - keep.mean()),
                        unconstrained_mean=big.mean(axis=0).tolist(),
                        unconstrained_sd=big.std(axis=0, ddof=1).tolist(),
                        accepted_mean=big[keep].mean(axis=0).tolist(),
                        accepted_sd=big[keep].std(axis=0, ddof=1).tolist()),
        magnitude=dict(median_peak_rule_nm=median_peak, peaks_median=float(np.median(peaks)),
                       peaks_min=float(peaks.min()), peaks_max=float(peaks.max()),
                       peaks_at_12=int(np.sum(peaks >= 12.0 - 1e-12)),
                       population_ext_peak_median=float(np.median(-population[:, :-1].min(axis=1))),
                       population_flex_peak_median=float(np.median(population[:, :-1].max(axis=1))),
                       population_rms_median=float(np.median(rms(population))),
                       ood_rms_median=float(np.median(rms(curves))),
                       alternative_per_bout_medians_nm=[
                           float(np.median(-population[:, :-1].min(axis=1))),
                           float(np.median(population[:, :-1].max(axis=1)))]),
        floors=dict(population_median=float(np.median(floors_pop)), ood_median=float(np.median(floors_ood)),
                    ood_quantiles=[float(q) for q in np.quantile(floors_ood, [0.1, 0.5, 0.9])]),
        per_fold=per_fold)
    _write(out / "robust_ood.json", result)
    return 0


def step_ood_per_part(run: str, out: Path, workers: int) -> int:
    """The ruled primary OOD users: each part's peak its own population median x (1 + 0.26 z), one z per user.
    Same timings, z and answer seeds as the common-peak users, which must exist already and
    are checked against; best-profile distances as before, into ``RUN/ood-pb/``."""
    inputs = Path(run).parent / "schaefer" / "prior_inputs.json"
    dist = TimingDistribution.load(inputs)
    population = _population_curves(run, LOSO)
    median_ext, median_flex = population_median_peaks(population)
    started = time.perf_counter()
    per_fold, same = {}, dict(timings=True, z=True, answer_seeds=True, rejected=True, ids=True)
    for fold in LOSO:
        common = load_ood_users(run, fold, OOD_USER_SET)
        prior = DataPrior.load(Path(run) / "priors" / f"{fold}-promp_s0")
        fold_config = json.loads((Path(run) / "folds" / f"{fold}.json").read_text(encoding="utf-8"))["config"]
        starts = int(fold_config["floor_starts"])
        users = make_ood_users(fold, dist, None, median_peaks_per_part=(median_ext, median_flex))
        same["ids"] &= [u.user_id for u in users] == list(common)
        for user in users:
            ref = common[user.user_id].draw
            same["timings"] &= user.draw["t_pct_st"] == ref["t_pct_st"]
            same["z"] &= user.draw["z_magnitude"] == ref["z_magnitude"]
            same["rejected"] &= user.draw["rejected_timing_draws"] == ref["rejected_timing_draws"]
            same["answer_seeds"] &= user.answer_seed == common[user.user_id].answer_seed
        floors = user_floors(prior.design(), prior.normalization(), users, n_starts=starts,
                             seed_sequence=fold_seed(fold, "ood", "promp_s0"), workers=workers,
                             cache=FloorCache(ood_floors_cache_path(run, fold, OOD_PER_PART_USER_SET)),
                             family="promp_s0-ood-pb", prior_hash=prior_digest(prior.mu, prior.sigma))
        config = dict(schema_note="ICORR reframe section 9a, O: out-of-generator ideals, per-bout "
                                  "peaks (decision of 2026-10-19, the primary O analysis)",
                      generator="prompref.experiment.robust.make_ood_users(median_peaks_per_bout=...)",
                      profile="SchaeferDesign(peak_ext, peak_flex).curve(t), the four-half-cosine profile",
                      timing_source=str(inputs), timing_mean=dist.mean.tolist(),
                      timing_cov=dist.cov.tolist(), nu=dist.nu,
                      scatter="cov * (nu - 2) / nu (t covariance = cov)",
                      device_limits="gaps t2+100-t1, t3-t2, t4-t3, t1-t4 in [10, 40] %ST; "
                                    "t2 <= 50; t4 >= 50; t1 in [50, 150)",
                      magnitude_rule=MAGNITUDE_RULE_PER_PART, median_ext_peak_nm=median_ext,
                      median_flex_peak_nm=median_flex,
                      median_peak_source=f"{len(population)} population users of {', '.join(LOSO)}",
                      magnitude_cv=OOD_MAGNITUDE_CV, seed=OOD_SEED,
                      same_draws_as="the common-peak users of RUN/ood (user set 'ood')",
                      n_population=OOD_N_POPULATION, floor_starts=starts,
                      floor_seed=f"floors5c.fold_seed({fold!r}, 'ood', 'promp_s0')",
                      prior=f"priors/{fold}-promp_s0", prior_hash=prior_digest(prior.mu, prior.sigma))
        write_ood_users(run, fold, users, floors, config, OOD_PER_PART_USER_SET)
        loaded = load_ood_users(run, fold, OOD_PER_PART_USER_SET)
        per_fold[fold] = dict(n=len(loaded), floor_median=float(np.median(
            [u.families["promp_s0"]["floor_nm"] for u in loaded.values()])))
        print(f"{fold}: {len(users)} per-bout OOD users, floor median "
              f"{per_fold[fold]['floor_median']:.3f} Nm [{time.perf_counter() - started:.0f} s]",
              flush=True)
    rms = lambda c: np.sqrt(np.mean(np.asarray(c) ** 2, axis=-1))  # noqa: E731
    pb = [u for fold in LOSO for u in load_ood_users(run, fold, OOD_PER_PART_USER_SET).values()]
    common = [u for fold in LOSO for u in load_ood_users(run, fold, OOD_USER_SET).values()]
    curves = np.array([u.curve_nm for u in pb])
    floors_pb = [u.families["promp_s0"]["floor_nm"] for u in pb]
    floors_pop = [u.families["promp_s0"]["floor_nm"] for fold in LOSO
                  for u in p5._users(run, fold).values() if u.kind == "population"]
    result = dict(
        when=stamp(), run=run, seconds=round(time.perf_counter() - started, 1),
        same_draws_as_common_peak=same,
        magnitude=dict(rule=MAGNITUDE_RULE_PER_PART, median_ext_peak_nm=median_ext,
                       median_flex_peak_nm=median_flex,
                       ext_peaks_median=float(np.median([u.draw["peak_ext_nm"] for u in pb])),
                       flex_peaks_median=float(np.median([u.draw["peak_flex_nm"] for u in pb])),
                       ext_at_12=int(sum(u.draw["peak_ext_nm"] >= 12.0 - 1e-12 for u in pb)),
                       flex_at_12=int(sum(u.draw["peak_flex_nm"] >= 12.0 - 1e-12 for u in pb)),
                       ood_pb_rms_median=float(np.median(rms(curves))),
                       ood_common_rms_median=float(np.median(rms([u.curve_nm for u in common]))),
                       population_rms_median=float(np.median(rms(population))),
                       ood_pb_ext_peak_on_grid_median=float(np.median(-curves[:, :-1].min(axis=1))),
                       ood_pb_flex_peak_on_grid_median=float(np.median(curves[:, :-1].max(axis=1)))),
        floors=dict(population_median=float(np.median(floors_pop)),
                    ood_pb_median=float(np.median(floors_pb)),
                    ood_pb_quantiles=[float(q) for q in np.quantile(floors_pb, [0.1, 0.5, 0.9])]),
        per_fold=per_fold)
    _write(out / "robust_ood_pb.json", result)
    print(f"per-bout OOD: median RMS {result['magnitude']['ood_pb_rms_median']:.3f} Nm "
          f"(population {result['magnitude']['population_rms_median']:.3f}); same draws as the "
          f"common-peak users: {all(same.values())}", flush=True)
    return 0 if all(same.values()) else 1


# --------------------------------------------------------------------------------- random

def _training_curves(run: Path, fold_name: str) -> np.ndarray:
    """The fold's 20,000 training ideals, regenerated as ``icorr_k_priors.run_fold`` does, and
    checked by refitting the stored 8-basis prior bit for bit."""
    import icorr_k_priors as kp
    from prompref.population.export import population_from_config
    from prompref.population.folds import check_fold, fold_streams, loso_folds
    from prompref.priors.basis_prior import common_fields, fit_basis_prior
    from prompref.priors.data_prior import fold_training_set

    fold = {f.name: f for f in loso_folds()}[fold_name]
    config = json.loads((run / "folds" / f"{fold.name}.json").read_text(encoding="utf-8"))["config"]
    shift, switches, timing = population_from_config(config)
    streams = fold_streams(int(config["seed"]), fold.name)
    train = kp._trials(fold.train)
    check_fold(fold, [t.subject for t in train])
    training = fold_training_set(train, int(config["n_curves"]), int(config["seed"]), switches,
                                 timing, seed_sequence=streams["training"], shift=shift)
    stored = DataPrior.load(run / "priors" / f"{fold.name}-promp_s0")
    refit = fit_basis_prior(training.curves, 8, common_fields(stored))
    if not (np.array_equal(refit.mu, stored.mu) and np.array_equal(refit.sigma, stored.sigma)):
        raise RuntimeError(f"{fold.name}: the regenerated training ideals do not refit the stored "
                           f"prior; refusing to standardise by them")
    return np.asarray(training.curves, dtype=float)


def step_random(run: str, out: Path, workers: int) -> int:
    started = time.perf_counter()
    summary = {}
    for fold in LOSO:
        curves = _training_curves(Path(run), fold)
        coefficients = fourier_coefficients(curves)
        mean, sd = coefficients.mean(axis=0), coefficients.std(axis=0, ddof=1)
        standardised = (coefficients - mean) / sd
        feat_path = training_features_path(run, fold)
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(feat_path, standardised=standardised, mean=mean, sd=sd)
        prior = p5._prior(run, fold, "promp_s0")
        generic = prior_mean_curve(p5.arm_space(prior, pert.ARM).design)
        users = {uid: u for uid, u in p5._users(run, fold).items() if u.kind == "population"}
        entries = {}
        for uid, user in users.items():
            index = draw_centre_index(user.answer_seed, len(standardised))
            raw = RandomSmoothUtility.for_user(user.answer_seed, mean, sd, standardised[index])
            duels = pert.duel_pairs(run, fold, uid, pert.N_PAIRS)
            plain = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.0)
            d_rms = np.array([abs(plain.utility(a) - plain.utility(b)) for a, b in duels])
            d_raw = np.abs(raw.raw(duels[:, 0]) - raw.raw(duels[:, 1]))
            raw_ratio = pert.decisiveness_ratio(d_raw, d_rms)
            scaled = raw.with_scale(1.0 / raw_ratio)
            entries[uid] = dict(answer_seed=int(user.answer_seed), centre_index=index,
                                centre=[float(v) for v in standardised[index]],
                                raw_ratio=float(raw_ratio), scale=float(1.0 / raw_ratio),
                                u_generic=float(scaled(generic)))
        sidecar = dict(schema=RANDOM_SCHEMA, fold=fold, when=stamp(),
                       spec="ICORR reframe section 9a, R: random smooth utilities",
                       constants=dict(lengthscale=LENGTHSCALE, rho=RHO, n_features=N_FEATURES,
                                      rff_salt=RFF_SALT, centre_salt=CENTRE_SALT,
                                      n_harmonics=6, gp_variance=1.0,
                                      standardisation="mean and SD (ddof 1) of the fold's training ideals",
                                      centre="a uniformly drawn training ideal's standardised features",
                                      scale="1 / median over REPORT section 9's 200 prior-scale pairs "
                                            "of |du_raw| / |du_rms(s = 1.0, the user's own ideal)|",
                                      u_max="best of 2-6 independent searches (robust.maximise_utility: "
                                            "a 120,000-vector screen, L-BFGS-B from the best 320 and the "
                                            "prior mean, over the ProMP weight box behind the device "
                                            "clip), more searches until two reach the best; step umax",
                                      umax_seed=UMAX_SEED),
                       n_training=int(len(standardised)), feature_mean=mean.tolist(),
                       feature_sd=sd.tolist(), generic="prior_mean_curve of the tight arm's design",
                       users=entries)
        _write(random_utility_path(run, fold), sidecar)
        p5.random_utility_sidecar.cache_clear()
        summary[fold] = dict(n_users=len(entries),
                             raw_ratio_quantiles=[float(q) for q in np.quantile(
                                 [e["raw_ratio"] for e in entries.values()], [0, 0.1, 0.5, 0.9, 1])])
        print(f"{fold}: {len(entries)} users [{time.perf_counter() - started:.0f} s]", flush=True)
    cell = {"random": dict(observer_metric="random_smooth", s=1.0)}
    in_sample = measure(run, cell, folds=LOSO, users=slice(0, pert.N_USERS), seed=pert.SEED)
    all_in = measure(run, cell, folds=LOSO, users=slice(0, 40), seed=pert.SEED)
    out_of_sample = measure(run, cell, folds=LOSO, users=slice(0, 40), seed=OOS_SEED)
    result = dict(when=stamp(), run=run, seconds=round(time.perf_counter() - started, 1),
                  per_fold=summary, achieved_in_sample_report_users=in_sample["random"],
                  achieved_in_sample_all_200=all_in["random"],
                  achieved_out_of_sample_all_200=out_of_sample["random"],
                  out_of_sample_definition=f"all 40 population users per fold, 200 fresh pairs "
                  f"each, pair seed [{OOS_SEED}, crc32(user id)]")
    _write(out / "robust_random.json", result)
    print(f"random: achieved ratio in sample {in_sample['random']['ratio']:.4f} (REPORT users), "
          f"{all_in['random']['ratio']:.4f} (all 200); out of sample "
          f"{out_of_sample['random']['ratio']:.4f}", flush=True)
    return 0


def _maximise_job(job) -> dict:
    utility, design, seed = job
    return maximise_utility(utility, design, seed=seed)


def step_umax(run: str, out: Path, workers: int) -> int:
    """u_max per population user: independent :func:`maximise_utility` searches, seeded
    ``[UMAX_SEED, crc32(user id), search]``, at least ``UMAX_MIN_SEARCHES`` per user and more,
    in rounds of two, until the best value has been reached by two searches or
    ``UMAX_MAX_SEARCHES`` ran. Rewrites the u_max fields of each fold's sidecar."""
    started = time.perf_counter()
    sidecars = {fold: json.loads(random_utility_path(run, fold).read_text(encoding="utf-8"))
                for fold in LOSO}
    utilities, designs = {}, {}
    for fold, sidecar in sidecars.items():
        designs[fold] = p5._prior(run, fold, "promp_s0").design()
        for uid, entry in sidecar["users"].items():
            utilities[fold, uid] = user_utility(sidecar, uid, entry["answer_seed"])
    results = {key: [] for key in utilities}
    pending = list(utilities)
    rounds = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending:
            jobs = []
            for key in pending:
                fold, uid = key
                n = len(results[key])
                extra = UMAX_MIN_SEARCHES if n == 0 else 2
                for search in range(n, min(n + extra, UMAX_MAX_SEARCHES)):
                    seed = [UMAX_SEED, zlib.crc32(uid.encode("utf-8")), search]
                    jobs.append((key, (utilities[key], designs[fold], seed)))
            for (key, _), res in zip(jobs, pool.map(_maximise_job, [job for _, job in jobs])):
                results[key].append(res)
            pending = [key for key in pending
                       if combine_searches(results[key])["n_searches_at_best"] < 2
                       and len(results[key]) < UMAX_MAX_SEARCHES]
            rounds.append(dict(jobs=len(jobs), still_pending=len(pending),
                               seconds=round(time.perf_counter() - started, 1)))
            print(f"umax round {len(rounds)}: {len(jobs)} searches, {len(pending)} users pending "
                  f"[{time.perf_counter() - started:.0f} s]", flush=True)
    combined = {key: combine_searches(res) for key, res in results.items()}
    for fold, sidecar in sidecars.items():
        for uid, entry in sidecar["users"].items():
            best = combined[fold, uid]
            entry.update(u_max=best["u_max"], theta_max=best["theta"],
                         umax_searches=best["searches"], umax_n_searches_at_best=best["n_searches_at_best"],
                         umax_gap_to_next=best["gap_to_next"])
        sidecar["umax_when"] = stamp()
        _write(random_utility_path(run, fold), sidecar)
    p5.random_utility_sidecar.cache_clear()
    rows = list(combined.items())
    head = np.array([c["u_max"] - sidecars[f]["users"][u]["u_generic"] for (f, u), c in rows])
    gap = np.array([c["gap_to_next"] for _, c in rows])
    at_best = np.array([c["n_searches_at_best"] for _, c in rows])
    n_search = np.array([c["n_searches"] for _, c in rows])
    # The pessimistic precision of u_max: where only one search reached it, the next search fell
    # short by gap; relative to the user's headroom u_max - u_gen.
    rel = gap / head
    result = dict(when=stamp(), run=run, seconds=round(time.perf_counter() - started, 1),
                  rounds=rounds, n_users=len(rows),
                  searches_per_user=dict(zip(*[v.tolist() for v in np.unique(n_search, return_counts=True)])),
                  users_best_reached_by_2_or_more=int(np.sum(at_best >= 2)),
                  users_best_reached_once=int(np.sum(at_best == 1)),
                  gap_to_next_over_headroom_where_once=[float(q) for q in np.quantile(
                      rel[at_best == 1], [0.5, 0.9, 1.0])] if np.any(at_best == 1) else [],
                  headroom_quantiles=[float(q) for q in np.quantile(head, [0, 0.1, 0.5, 0.9, 1])],
                  min_headroom=float(head.min()))
    _write(out / "robust_umax.json", result)
    print(f"umax: best reached by >= 2 searches for {result['users_best_reached_by_2_or_more']} "
          f"of {len(rows)} users; headroom median {np.median(head):.3f}", flush=True)
    return 0


#: The refinement after the e-rand block: diversified searches per user.
REFINE_SEARCHES = 2
REFINE_RANDOM = 128
REFINE_OFFSET = 100


def _session_bests(path: Path) -> dict:
    """Per (fold, user): the largest own utility any e-rand session delivered (-own_distances,
    over every comparison) and the final weights of every session."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("error") or row.get("observer_metric") != "random_smooth":
            continue
        key = (row["fold"], row["user_id"])
        best, thetas = out.get(key, (-np.inf, []))
        out[key] = (max(best, max(-d for d in row["own_distances"])), thetas + [row["best"]])
    return out


def step_refine(run: str, out: Path, workers: int, first_pass: Path | None) -> int:
    """After e-rand: u_max can only rise. Per user, ``REFINE_SEARCHES`` more searches with
    ``REFINE_RANDOM`` unscreened starts each (the first also starts from the current maximum,
    the first-pass maximum if given, and every e-rand session's final weights); then u_max is
    the best of every search, the first pass and every delivered curve, with its source. Counts
    the sessions that delivered more than every search found."""
    started = time.perf_counter()
    sidecars = {fold: json.loads(random_utility_path(run, fold).read_text(encoding="utf-8"))
                for fold in LOSO}
    sessions = _session_bests(out / "e-rand.jsonl")
    firsts = {}
    if first_pass is not None:
        for fold in LOSO:
            path = first_pass / f"{fold}.json"
            if path.exists():
                for uid, e in json.loads(path.read_text(encoding="utf-8"))["users"].items():
                    firsts[fold, uid] = e
    jobs = []
    for fold, sidecar in sidecars.items():
        design = p5._prior(run, fold, "promp_s0").design()
        for uid, entry in sidecar["users"].items():
            utility = user_utility(sidecar, uid, entry["answer_seed"])
            crc = zlib.crc32(uid.encode("utf-8"))
            extra = [entry["theta_max"]] + sessions.get((fold, uid), (None, []))[1]
            if (fold, uid) in firsts:
                extra.append(firsts[fold, uid]["theta_max"])
            # A maximum taken from a session or a first pass's check run has no weights.
            extra = [start for start in extra if start is not None]
            for i in range(REFINE_SEARCHES):
                jobs.append(((fold, uid), (utility, design, [UMAX_SEED, crc, REFINE_OFFSET + i],
                                           dict(n_random=REFINE_RANDOM,
                                                extra_starts=extra if i == 0 else ()))))
    refined = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (key, _), res in zip(jobs, pool.map(_refine_job, [job for _, job in jobs])):
            refined.setdefault(key, []).append(res)
    summary = dict(improved_by_refine=0, from_first_pass=0, from_session=0,
                   sessions_above_every_search=0, users_with_sessions=0, rel_rise=[])
    for fold, sidecar in sidecars.items():
        for uid, entry in sidecar["users"].items():
            key = (fold, uid)
            before = float(entry["u_max"])
            searches = list(entry["umax_searches"]) + [r["u_max"] for r in refined[key]]
            best_search = combine_searches([dict(u_max=entry["u_max"], theta=entry["theta_max"])]
                                           + refined[key])
            first = firsts.get(key)
            first_best = max(first["u_max"], first.get("u_max_check", -np.inf)) if first else -np.inf
            session_best = sessions.get(key, (-np.inf, []))[0]
            u_max, source, theta = best_search["u_max"], "search", best_search["theta"]
            if first_best > u_max and not same_value(first_best, u_max):
                u_max, source = first_best, "first pass"
                theta = first["theta_max"] if same_value(first["u_max"], first_best) else None
            if session_best > u_max and not same_value(session_best, u_max):
                u_max, source, theta = session_best, "session", None
            if key in sessions:
                summary["users_with_sessions"] += 1
                summary["sessions_above_every_search"] += int(
                    session_best > max(searches) and not same_value(session_best, max(searches)))
            summary["improved_by_refine"] += int(not same_value(best_search["u_max"], before))
            summary["from_first_pass"] += int(source == "first pass")
            summary["from_session"] += int(source == "session")
            summary["rel_rise"].append((u_max - before) / (before - entry["u_generic"]))
            entry.update(u_max_before_refine=before, umax_refine_searches=[r["u_max"] for r in refined[key]],
                         u_max_first_pass=None if first is None else first_best,
                         u_max_session_best=None if key not in sessions else session_best,
                         u_max=u_max, theta_max=theta, u_max_source=source,
                         umax_n_searches_at_best=int(sum(same_value(u_max, v) for v in searches)))
        sidecar["refine_when"] = stamp()
        _write(random_utility_path(run, fold), sidecar)
    rise = np.array(summary.pop("rel_rise"))
    result = dict(when=stamp(), run=run, seconds=round(time.perf_counter() - started, 1),
                  searches_per_user=REFINE_SEARCHES, random_starts=REFINE_RANDOM,
                  first_pass_dir=None if first_pass is None else str(first_pass),
                  e_rand_rows=str(out / "e-rand.jsonl"), **summary,
                  rise_over_headroom_quantiles=[float(q) for q in np.quantile(rise, [0.5, 0.9, 1.0])])
    _write(out / "robust_umax_refine.json", result)
    print(f"refine: {summary['improved_by_refine']} users rose; first pass best for "
          f"{summary['from_first_pass']}, a session for {summary['from_session']}; "
          f"{summary['sessions_above_every_search']} users with a session above every search", flush=True)
    return 0


def _refine_job(job) -> dict:
    utility, design, seed, kwargs = job
    return maximise_utility(utility, design, seed=seed, **kwargs)


#: Convergence of u_max: the last two *independent* searches (no start taken from an earlier
#: maximum, a session or the first pass) both reach within this share of the user's headroom
#: u_max - u_gen of u_max, the best found by anything.
CONVERGE_TOL = 0.01
#: At most this many independent convergence searches per user.
CONVERGE_MAX_SEARCHES = 10
#: Search i of a user's convergence searches is seeded [UMAX_SEED, crc32(user id), 200 + i].
CONVERGE_OFFSET = 200


def refine_converged(searches, u_max: float, u_generic: float, tol: float = CONVERGE_TOL) -> bool:
    """Two successive searches agree: the last two of ``searches`` both lie within ``tol`` x
    (u_max - u_generic) of u_max. Then the latest search did not raise u_max by more than
    that, and the one before reached the same height."""
    if len(searches) < 2:
        return False
    slack = float(tol) * (float(u_max) - float(u_generic))
    return all(float(u_max) - float(s) <= slack for s in list(searches)[-2:])


def step_converge(run: str, out: Path, workers: int, first_pass: Path | None,
                  tol: float = CONVERGE_TOL, max_searches: int = CONVERGE_MAX_SEARCHES) -> int:
    """Independent searches until u_max is stable: per round, each unconverged user gets one
    more search (two in its first round), seeded ``[UMAX_SEED, crc32(user id),
    CONVERGE_OFFSET + i]`` for its i-th, with ``REFINE_RANDOM`` unscreened starts and **no**
    start taken from an earlier maximum, a session or the first pass, so that two searches
    agreeing is evidence and not an echo of the incumbent. Until :func:`refine_converged` holds
    on these searches for every user, or each has ``max_searches`` of them. u_max only rises
    (any search value is a lower bound). Writes the sidecars after every round, and the summary
    to ``robust_umax_converge.json``. ``first_pass`` is accepted for the launcher and unused."""
    started = time.perf_counter()
    sidecars = {fold: json.loads(random_utility_path(run, fold).read_text(encoding="utf-8"))
                for fold in LOSO}
    entries, utilities, designs = {}, {}, {}
    before = {}
    for fold, sidecar in sidecars.items():
        designs[fold] = p5._prior(run, fold, "promp_s0").design()
        for uid, entry in sidecar["users"].items():
            key = (fold, uid)
            entries[key] = entry
            entry.setdefault("umax_independent_searches", [])
            before[key] = float(entry["u_max"])
            utilities[key] = user_utility(sidecar, uid, entry["answer_seed"])

    def done(e) -> bool:
        return refine_converged(e["umax_independent_searches"], e["u_max"], e["u_generic"], tol)

    converged_at_start = sum(done(e) for e in entries.values())
    rounds = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            pending = [key for key, e in entries.items()
                       if not done(e) and len(e["umax_independent_searches"]) < max_searches]
            if not pending:
                break
            jobs = []
            for key in pending:
                n = len(entries[key]["umax_independent_searches"])
                for i in range(n, min(n + (2 if n == 0 else 1), max_searches)):
                    seed = [UMAX_SEED, zlib.crc32(key[1].encode("utf-8")), CONVERGE_OFFSET + i]
                    jobs.append((key, (utilities[key], designs[key[0]], seed,
                                       dict(n_random=REFINE_RANDOM))))
            raised = 0
            for (key, _), res in zip(jobs, pool.map(_refine_job, [job for _, job in jobs])):
                entry = entries[key]
                entry["umax_independent_searches"] = list(entry["umax_independent_searches"]) + [res["u_max"]]
                if res["u_max"] > entry["u_max"] and not same_value(res["u_max"], entry["u_max"]):
                    entry.update(u_max=res["u_max"], theta_max=res["theta"], u_max_source="search")
                    raised += 1
            for fold, sidecar in sidecars.items():
                _write(random_utility_path(run, fold), sidecar)
            still = sum(not done(e) for e in entries.values())
            rounds.append(dict(searches=len(jobs), raised=raised, unconverged_after=still,
                               seconds=round(time.perf_counter() - started, 1)))
            print(f"converge round {len(rounds)}: {len(jobs)} searches, u_max raised for {raised}, "
                  f"{still} users unconverged [{time.perf_counter() - started:.0f} s]", flush=True)
    for key, entry in entries.items():
        every = (list(entry["umax_searches"]) + list(entry["umax_refine_searches"])
                 + list(entry["umax_independent_searches"]))
        entry.update(umax_converged=bool(done(entry)), umax_converge_tol=float(tol),
                     umax_converge_criterion="the last two independent searches both within "
                                             "tol x (u_max - u_gen) of u_max",
                     umax_n_searches_at_best=int(sum(same_value(entry["u_max"], v) for v in every)))
    for fold, sidecar in sidecars.items():
        sidecar["converge_when"] = stamp()
        _write(random_utility_path(run, fold), sidecar)
    p5.random_utility_sidecar.cache_clear()
    rise = np.array([(e["u_max"] - before[k]) / (before[k] - e["u_generic"]) for k, e in entries.items()])
    unconverged = [k for k, e in entries.items() if not e["umax_converged"]]
    n_ref = np.array([len(e["umax_independent_searches"]) for e in entries.values()])
    gaps = {f"{f}/{u.split('/')[-1]}": [float((e["u_max"] - s) / (e["u_max"] - e["u_generic"]))
                                         for s in e["umax_independent_searches"][-2:]]
            for (f, u), e in entries.items() if not e["umax_converged"]}
    result = dict(when=stamp(), run=run, seconds=round(time.perf_counter() - started, 1), tol=tol,
                  max_searches=max_searches, criterion="the last two independent searches (no "
                  "start from an earlier maximum, a session or the first pass) both within tol x "
                  "(u_max - u_gen) of u_max", n_users=len(entries),
                  converged_at_start=int(converged_at_start), converged=len(entries) - len(unconverged),
                  unconverged=[f"{f}/{u.split('/')[-1]}" for f, u in unconverged],
                  unconverged_last_two_shortfall_over_headroom=gaps, rounds=rounds,
                  independent_searches_per_user=dict(zip(*[v.tolist() for v in np.unique(n_ref, return_counts=True)])),
                  u_max_raised=int(np.sum(rise > 1e-9)),
                  rise_over_headroom_quantiles=[float(q) for q in np.quantile(rise, [0.5, 0.9, 1.0])],
                  sources={s: sum(e["u_max_source"] == s for e in entries.values())
                           for s in {e["u_max_source"] for e in entries.values()}},
                  theta_missing=sum(e["theta_max"] is None for e in entries.values()))
    _write(out / "robust_umax_converge.json", result)
    print(f"converge: {result['converged']} of {len(entries)} users converged (tol {tol}); "
          f"u_max raised for {result['u_max_raised']}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", required=True,
                    choices=("landmark", "ood", "random", "umax", "refine", "converge"))
    ap.add_argument("--first-pass", default=None,
                    help="refine, converge: a directory of first-pass sidecars whose maxima count too")
    ap.add_argument("--magnitude", choices=("common", "per-bout"), default="common",
                    help="ood: one common peak (user set 'ood') or per-bout peaks ('ood-pb')")
    ap.add_argument("--tol", type=float, default=CONVERGE_TOL)
    ap.add_argument("--max-searches", type=int, default=CONVERGE_MAX_SEARCHES)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--out", default=None, help="default <run>/../icorr")
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        ap.error(f"--workers must lie in 1..{MAX_WORKERS}: the machine is shared")
    run = str(Path(args.run).resolve())
    out = Path(args.out) if args.out else Path(run).parent / "icorr"
    if args.step == "landmark":
        return step_landmark(run, out)
    if args.step == "ood":
        if args.magnitude == "per-bout":
            return step_ood_per_part(run, out, args.workers)
        return step_ood(run, out, args.workers)
    if args.step == "converge":
        return step_converge(run, out, args.workers,
                             Path(args.first_pass) if args.first_pass else None,
                             tol=args.tol, max_searches=args.max_searches)
    if args.step == "umax":
        return step_umax(run, out, args.workers)
    if args.step == "refine":
        return step_refine(run, out, args.workers,
                           Path(args.first_pass) if args.first_pass else None)
    return step_random(run, out, args.workers)


if __name__ == "__main__":
    sys.exit(main())
