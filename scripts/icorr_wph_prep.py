"""Per-user scales and floors of the phase-weighted wearers (our design notes, "Phase-weighted wearers"; decision of 2026-10-19:
W-pref runs now, W-bio once its hip-angle data can be read, with no substitute weighting).

Usage:
    python scripts/icorr_wph_prep.py --run <main>/out/phase5-schaefer --cell {pref,bio,pref-sd1}
        [--out <main>/out/icorr]

For each LOSO fold and each population user, written to ``RUN/phase_weight/<cell>/<fold>.json``
(:mod:`prompref.experiment.phase_weighted`):

- the weight: W-pref's per-user curve (seeded from the answer seed; "pref-sd1" is the stress test
  S1, the same curve shape with the SD of log w 1.0 instead of 0.5), or W-bio's common curve read
  from ``OUT/wph_bio_weight.json`` (``scripts/icorr_wph_bio_weight.py``; refused if missing);
- the matched scale: kappa = 1 / REPORT section 9's decisiveness ratio, the median over the user's
  200 prior-scale pairs (``phase5e_perturbation.duel_pairs``) of |Delta d_w| / |Delta d_rms|
  against the plain s = 1.0 wearer, as R's ``step_random`` scales its utility;
- the wearer's own distance of generic assistance, built exactly as a session builds it;
- the floor in the wearer's own metric, the weighted projection behind the device
  (``phase_weighted.weighted_floor``), unscaled and scaled.

Checks, in ``OUT/wph_prep_<cell>.json``:

- the same floor routine with w = 1 against the stored 5a RMS floors of all users;
- the achieved decisiveness ratio of the cell, through ``build_observer`` as a session builds the
  wearer: on the measurement pairs (1 per user by construction) and on fresh pairs (another seed);
- the weights: mean, periodicity, the SD of log w over the 101-point grid (and over the 100
  distinct points, for context).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import icorr_robust_prep as rp  # noqa: E402
import phase5e_perturbation as pert  # noqa: E402
from prompref.experiment import phase5 as p5  # noqa: E402
from prompref.experiment.metrics import prior_mean_curve  # noqa: E402
from prompref.experiment.phase5c_grid import LOSO  # noqa: E402
from prompref.experiment.phase_weighted import (PHASE_WEIGHT_SCHEMA,  # noqa: E402
                                                cell_constants, phase_weight_path, weighted_floor)
from prompref.user.oracle import PHASE_WEIGHTED, Observer, weighted_rms_distance  # noqa: E402
from prompref.user.phase_weight import (PHASE_WEIGHTINGS, PREF_CELLS,  # noqa: E402
                                        bio_weight_path, load_bio_weight, pref_weight)

#: The w = 1 floor check: the routine may beat 5a's multi-start floor, never miss it by more.
FLOOR_TOL_NM = 1e-6
#: The floor's prior-draw starts (5a's floors use 20), seeded [FLOOR_SEED, crc32(user id)]; the
#: weighted floor also starts from the w = 1 floor's weights.
FLOOR_STARTS = 20
FLOOR_SEED = 0x3F100E


def stamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _q(values, qs=(0.0, 0.1, 0.5, 0.9, 1.0)) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(values, dtype=float), qs)]


def user_entry(run: str, fold: str, user, weight: np.ndarray, generic: np.ndarray, design,
               *, store_weight: bool, n_pairs: int = pert.N_PAIRS) -> dict:
    """One user's sidecar entry (see the module docstring)."""
    duels = pert.duel_pairs(run, fold, user.user_id, n_pairs)
    plain = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.0)
    d_rms = np.array([abs(plain.utility(a) - plain.utility(b)) for a, b in duels])
    d_w = np.array([abs(weighted_rms_distance(a, user.curve_nm, weight)
                        - weighted_rms_distance(b, user.curve_nm, weight)) for a, b in duels])
    raw_ratio = pert.decisiveness_ratio(d_w, d_rms)
    scale = 1.0 / raw_ratio
    wearer = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.1, metric=PHASE_WEIGHTED,
                      phase_weight=weight, utility_scale=scale)
    seed = [FLOOR_SEED, zlib.crc32(user.user_id.encode("utf-8"))]
    rms_check = weighted_floor(design, user.curve_nm, np.ones_like(weight), n_starts=FLOOR_STARTS,
                               seed=seed)
    floor = weighted_floor(design, user.curve_nm, weight, n_starts=FLOOR_STARTS, seed=seed,
                           extra_starts=[rms_check["theta"]])
    entry = dict(answer_seed=int(user.answer_seed), raw_ratio=float(raw_ratio), scale=float(scale),
                 generic_w_nm=float(weighted_rms_distance(generic, user.curve_nm, weight)),
                 own_generic=float(wearer.own_distance(generic)),
                 floor_w_nm=floor["floor_nm"], own_floor=float(scale * floor["floor_nm"]),
                 floor_theta=floor["theta"], proj_nm=floor["proj_nm"],
                 proj_clipped_nm=floor["proj_clipped_nm"], proj_in_box=floor["proj_in_box"],
                 proj_clip_free=floor["proj_clip_free"], refine_gain_nm=floor["refine_gain_nm"],
                 rms_floor_5a_nm=float(user.families["promp_s0"]["floor_nm"]),
                 rms_floor_this_routine_nm=rms_check["floor_nm"])
    if store_weight:
        entry["weight"] = [float(v) for v in weight]
    return entry


def build(run: str, cell: str, out: Path) -> int:
    started = time.perf_counter()
    if cell == "bio":
        common, sha = load_bio_weight(run)       # refuses loudly when the file is missing
        source = dict(weight_source=str(bio_weight_path(run)), weight_sha256=sha)
    else:
        # W-pref (SD of log w 0.5) or the stress test S1 (1.0): the same coefficients per user.
        log_sd = PREF_CELLS[cell]
        common, source = None, dict(weight_source="pref_weight(answer_seed), seeded per user"
                                    if cell == "pref" else
                                    f"pref_weight(answer_seed, log_sd={log_sd}), seeded per user")
    per_fold, entries_all = {}, []
    for fold in LOSO:
        prior = p5._prior(run, fold, "promp_s0")
        design = prior.design()
        generic = prior_mean_curve(p5.arm_space(prior, pert.ARM).design)
        users = {uid: u for uid, u in p5._users(run, fold).items() if u.kind == "population"}
        entries = {}
        for uid, user in users.items():
            weight = common if common is not None else pref_weight(user.answer_seed, log_sd)
            entries[uid] = user_entry(run, fold, user, weight, generic, design,
                                      store_weight=common is None)
        sidecar = dict(schema=PHASE_WEIGHT_SCHEMA, cell=cell, fold=fold, when=stamp(),
                       spec="ICORR reframe section 9a, phase-weighted wearers"
                            + (" (stress test S1)" if cell == "pref-sd1" else ""),
                       constants=cell_constants(cell), **source,
                       scale="1 / median over REPORT section 9's 200 prior-scale pairs of "
                             "|d_w(a) - d_w(b)| / |d_rms(a) - d_rms(b)|, the user's own ideal",
                       generic="prior_mean_curve of the tight arm's device-clipped design",
                       floor="phase_weighted.weighted_floor over the fold's 8-basis ProMP data "
                             "prior design (weight box +-30), behind the device clip",
                       users=entries)
        path = phase_weight_path(run, cell, fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
        p5.phase_weight_sidecar.cache_clear()
        per_fold[fold] = dict(n_users=len(entries),
                              raw_ratio_quantiles=_q([e["raw_ratio"] for e in entries.values()]))
        entries_all += list(entries.values())
        print(f"{fold}: {len(entries)} users [{time.perf_counter() - started:.0f} s]", flush=True)

    # -- checks
    fields = dict(phase_weighting=cell, s=1.0)
    achieved = dict(
        in_sample_report_users=rp.measure(run, {cell: fields}, folds=LOSO,
                                          users=slice(0, pert.N_USERS), seed=pert.SEED)[cell],
        in_sample_all_200=rp.measure(run, {cell: fields}, folds=LOSO, users=slice(0, 40),
                                     seed=pert.SEED)[cell],
        out_of_sample_all_200=rp.measure(run, {cell: fields}, folds=LOSO, users=slice(0, 40),
                                         seed=rp.OOS_SEED)[cell])
    floor_5a = np.array([e["rms_floor_5a_nm"] for e in entries_all])
    floor_me = np.array([e["rms_floor_this_routine_nm"] for e in entries_all])
    excess = floor_me - floor_5a
    weights = ([np.asarray(e["weight"]) for e in entries_all] if common is None else [common])
    log_sd_101 = [float(np.std(np.log(w))) for w in weights]
    log_sd_100 = [float(np.std(np.log(w[:-1]))) for w in weights]
    room = [e["own_generic"] - e["own_floor"] for e in entries_all]
    result = dict(
        when=stamp(), run=run, cell=cell, seconds=round(time.perf_counter() - started, 1),
        **source, per_fold=per_fold, achieved_decisiveness=achieved,
        out_of_sample_definition=f"all 40 population users per fold, 200 fresh pairs each, pair "
                                 f"seed [{rp.OOS_SEED}, crc32(user id)]",
        floor_check_w1=dict(n=int(excess.size), max_excess_over_5a_nm=float(excess.max()),
                            n_above_5a_by_tol=int(np.sum(excess > FLOOR_TOL_NM)),
                            n_below_5a_by_tol=int(np.sum(excess < -FLOOR_TOL_NM)),
                            min_excess_nm=float(excess.min()), tolerance_nm=FLOOR_TOL_NM),
        floors=dict(floor_w_nm=_q([e["floor_w_nm"] for e in entries_all]),
                    own_floor=_q([e["own_floor"] for e in entries_all]),
                    proj_in_box=int(sum(e["proj_in_box"] for e in entries_all)),
                    proj_clip_free=int(sum(e["proj_clip_free"] for e in entries_all)),
                    refine_gain_nm=_q([e["refine_gain_nm"] for e in entries_all]),
                    n_refined_below_projection=int(sum(e["refine_gain_nm"] > 1e-9
                                                       for e in entries_all))),
        generic=dict(generic_w_nm=_q([e["generic_w_nm"] for e in entries_all]),
                     own_generic=_q([e["own_generic"] for e in entries_all])),
        own_room=_q(room),
        scale=_q([e["scale"] for e in entries_all]),
        raw_ratio=_q([e["raw_ratio"] for e in entries_all]),
        weights=dict(n=len(weights), log_sd_101=_q(log_sd_101), log_sd_100_distinct=_q(log_sd_100),
                     mean_minus_1_max=float(max(abs(float(np.mean(w)) - 1.0) for w in weights)),
                     periodic=all(w[-1] == w[0] for w in weights),
                     min=float(min(w.min() for w in weights)),
                     max=float(max(w.max() for w in weights))))
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"wph_prep_{cell}.json"
    target.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {target}", flush=True)
    print(f"{cell}: achieved ratio in sample {achieved['in_sample_all_200']['ratio']:.6f}, out of "
          f"sample {achieved['out_of_sample_all_200']['ratio']:.4f}; w = 1 floor check: max "
          f"excess over 5a {excess.max():.2e} Nm ({result['floor_check_w1']['n_above_5a_by_tol']} "
          f"above by > {FLOOR_TOL_NM})", flush=True)
    return 0 if result["floor_check_w1"]["n_above_5a_by_tol"] == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cell", required=True, choices=PHASE_WEIGHTINGS)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/../icorr")
    args = ap.parse_args(argv)
    run = str(Path(args.run).resolve())
    out = args.out or Path(run).parent / "icorr"
    return build(run, args.cell, out)


if __name__ == "__main__":
    sys.exit(main())
