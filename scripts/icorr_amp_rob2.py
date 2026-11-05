"""Test ROB2: the earlier robustness conditions in the amplitude-free design.

Usage:
  python scripts/icorr_amp_rob2.py prep
  python scripts/icorr_amp_rob2.py run [--workers 8]
  python scripts/icorr_amp_rob2.py report

Settings: wide 3-35 Nm (AM) and fixed 12 Nm (B_fixed),
FULL's 200 users, answer seeds and peaks; 40 comparisons. ``prep`` writes the matched-
decisiveness calibrations (B5-B8) and the C users' shapes. Outputs in <out>/icorr/amp_free/:
rob2-calib-<setting>.json, rob2-shapes-<fold>.npz, rob2_sessions.jsonl, amp_rob2_report.md.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import replace  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_full as F  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.experiment.run import run_session  # noqa: E402
from prompref.population.folds import child_sequence, load_trials, loso_folds  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.oracle import PHASE_WEIGHTED, RANDOM_SMOOTH, Observer, rms_distance  # noqa: E402

SETTINGS = ("wide", "fixed 12")
CANDS = ("psypref", "cmaes_sch4", "g", "d_z", "cmaes")
LINK_SCALES = {"logistic": 0.5916008501683337, "student_t": 0.8100468122288004,
               "cauchy": 0.4799604532549037}
#: Section 18: the conditions with the earlier values.
CONDITIONS = {
    "A1 individual s": dict(s_logsd=0.35),
    "A2 lapse 0.2": dict(lapse=0.2),
    "A3 device noise": dict(noise=0.6),
    "A4 order bias": dict(order_bias=0.3),
    "A5 no preference": dict(no_pref=0.5),
    "A6a logistic link": dict(link="logistic"),
    "A6b Student-t link": dict(link="student_t"),
    "A6c Cauchy link": dict(link="cauchy"),
    "A7 fatigue": dict(lapse_path=(0.05, 0.30)),
    "A8a s 1.5 to 0.75": dict(s_path=(1.5, 0.75)),
    "A8b s 0.75 to 1.5": dict(s_path=(0.75, 1.5)),
    "B1 Weber": dict(metric="weber"),
    "B2 asymmetry": dict(asymmetry=2.0),
    "B3 familiarity": dict(familiarity=0.25),
    "B4a drift 1 JND": dict(drift_jnd=1.0),
    "B4b drift 2 JND": dict(drift_jnd=2.0),
    "B5a features static": dict(metric="landmark", mag_fraction=0.10),
    "B5b features walking": dict(metric="landmark", mag_fraction=0.25),
    "B6a phase individual": dict(phase="pref"),
    "B6b phase hip speed": dict(phase="bio"),
    "B7 anisotropy": dict(phase="pref-sd1"),
    "B8 random utility": dict(metric=RANDOM_SMOOTH),
    "C1 Schaefer shapes": dict(shapes="schaefer"),
    "C2 heavy tails": dict(shapes="heavy"),
    "C3 family mixture": dict(shapes="mix"),
    "C4 pessimistic": dict(s=2.0, lapse_path=(0.05, 0.30), order_bias=0.3, no_pref=0.5,
                           phase="pref-sd1", shapes="schaefer"),
}
N_PAIRS = 200


def _dir(out: Path) -> Path:
    return out / "icorr" / "amp_free"


def _space(setting: str, ppath: str, fold_index: int):
    """PsyPref's space of the setting (its pairs calibrate B5-B8)."""
    d = R._fold(ppath, fold_index)
    if setting == "wide":
        mu, sigma = R._a_moments(ppath, fold_index, "wide")
        return AF.promp_space(mu, sigma, limit=F.RANGES["wide"][1])
    return AF.fixed_level_space(d["w_minus"], d["w_plus"], F.LEVEL)


def _pairs(space, seed) -> np.ndarray:
    z = np.random.default_rng(seed).standard_normal((N_PAIRS, 2, space.d))
    return np.array([[space.design.curve(space.to_theta(space.clip_to_ball(v))) for v in p] for p in z])


def _ratio(du_cell, du_rms) -> float:
    cell, rms = np.abs(np.asarray(du_cell, float)), np.abs(np.asarray(du_rms, float))
    keep = rms > 0.0
    return float(np.median(cell[keep] / rms[keep]))


def prep(out: Path) -> None:
    from prompref.experiment.r2 import sample_draw_nu
    from prompref.experiment.robust import TimingDistribution, draw_timing
    from prompref.designs.schaefer import SchaeferDesign
    from prompref.population.generator import OrderViolation, user_curve
    from prompref.user.phase_weight import load_bio_weight, pref_weight
    from prompref.user.smooth_utility import RandomSmoothUtility, draw_centre_index, fourier_coefficients

    d = _dir(out)
    run_dir = out / "phase5-schaefer"
    bio, _ = load_bio_weight(run_dir)
    dist = TimingDistribution.load(out / "schaefer" / "prior_inputs.json")
    # C users' shapes, per fold (FULL's users otherwise)
    for fi, fold in enumerate(loso_folds()):
        z = F._users(str(d / f"full-{fold.name}.npz"))
        root = np.random.SeedSequence([AF.SEED_FULL, 1800 + fi])
        rng = np.random.default_rng(child_sequence(root, 0))
        sch = [AF.unit_bouts(SchaeferDesign(1.0, 1.0).curve(draw_timing(rng, dist)[0]))
               for _ in range(AF.FULL_USERS_PER_FOLD)]
        shift, switches, timing = AF.population_config(run_dir, fold.name)
        held = load_trials(fold.held_out)
        rng_h = np.random.default_rng(child_sequence(root, 1))
        heavy = []
        while len(heavy) < AF.FULL_USERS_PER_FOLD:
            trial = held[int(rng_h.integers(len(held)))]
            draw = replace(sample_draw_nu(rng_h, switches, timing, shift, nu_shift=1.5, nu_scale=3.0,
                                          nu_timing=1.5),
                           ext_scale=AF.SHAPE_SCALE, flex_scale=AF.SHAPE_SCALE)
            try:
                heavy.append(AF.unit_bouts(user_curve(trial, draw)))
            except (OrderViolation, ValueError):
                continue
        from prompref.experiment.phase5 import _franks_prior

        franks = _franks_prior(str(run_dir), fold.name).design()
        rng_m = np.random.default_rng(child_sequence(root, 2))
        mix = [(z["user_minus"][u], z["user_plus"][u]) for u in range(AF.FULL_USERS_PER_FOLD)]
        chosen = rng_m.choice(AF.FULL_USERS_PER_FOLD, size=AF.FULL_USERS_PER_FOLD // 5, replace=False)
        for j, u in enumerate(chosen):
            if j % 2 == 0:
                mix[u] = sch[u]
            else:
                for _ in range(1000):
                    try:
                        mix[u] = AF.unit_bouts(franks.curve(franks.sample_prior(rng_m, 1)[0]))
                        break
                    except OrderViolation:
                        continue
        np.savez(d / f"rob2-shapes-{fold.name}.npz",
                 **{f"{k}_minus": np.array([s[0] for s in v]) for k, v in
                    (("schaefer", sch), ("heavy", heavy), ("mix", mix))},
                 **{f"{k}_plus": np.array([s[1] for s in v]) for k, v in
                    (("schaefer", sch), ("heavy", heavy), ("mix", mix))},
                 mix_replaced=np.sort(chosen))
        print(f"{fold.name}: C shapes", flush=True)
    # B5-B8 calibrations, per setting, on FULL's users
    for setting in SETTINGS:
        calib = dict(landmark={}, users={})
        du = {"rms": [], 0.10: [], 0.25: []}
        for fi, fold in enumerate(loso_folds()):
            ppath = str(d / f"fold-{fold.name}.npz")
            space = _space(setting, ppath, fi)
            pairs = _pairs(space, [AF.SEED_FULL, 1900, fi, SETTINGS.index(setting)])
            prior_curves = np.array([space.design.curve(space.to_theta(space.clip_to_ball(v)))
                                     for v in np.random.default_rng([AF.SEED_FULL, 1901, fi]).standard_normal((20_000, space.d))])
            coef = fourier_coefficients(prior_curves)
            mean, sd = coef.mean(axis=0), coef.std(axis=0, ddof=1)
            standardised = (coef - mean) / sd
            z = F._users(str(d / f"full-{fold.name}.npz"))
            for u in range(AF.FULL_USERS_PER_FOLD):
                amps = [F.LEVEL, F.LEVEL] if setting == "fixed 12" else z["amps_wide"][u]
                ideal = AF.compose(amps[0], amps[1], z["user_minus"][u], z["user_plus"][u])
                seed = int(z["answer_seeds"][u]) + F.SETTINGS.index(setting)
                plain = Observer(ideal_nm=ideal, s=1.0, lapse=0.0)
                d_rms = np.array([plain.utility(a) - plain.utility(b) for a, b in pairs])
                du["rms"].append(d_rms)
                for frac in (0.10, 0.25):
                    lm = Observer(ideal_nm=ideal, s=1.0, lapse=0.0, metric="landmark", landmark_mag_fraction=frac)
                    du[frac].append(np.array([lm.utility(a) - lm.utility(b) for a, b in pairs]))
                entry = {}
                for cell, w in (("pref", pref_weight(seed, 0.5)), ("pref-sd1", pref_weight(seed, 1.0)), ("bio", bio)):
                    pw = Observer(ideal_nm=ideal, s=1.0, lapse=0.0, metric=PHASE_WEIGHTED, phase_weight=w, utility_scale=1.0)
                    entry[cell] = 1.0 / _ratio([pw.utility(a) - pw.utility(b) for a, b in pairs], d_rms)
                index = draw_centre_index(seed, len(standardised))
                raw = RandomSmoothUtility.for_user(seed, mean, sd, standardised[index])
                d_raw = raw.raw(pairs[:, 0]) - raw.raw(pairs[:, 1])
                entry["random"] = dict(scale=1.0 / _ratio(d_raw, d_rms), centre=standardised[index].tolist(),
                                       mean=mean.tolist(), sd=sd.tolist())
                calib["users"][f"{fold.name}/f{u}"] = entry
        for frac in (0.10, 0.25):
            calib["landmark"][str(frac)] = _ratio(np.concatenate(du[frac]), np.concatenate(du["rms"]))
        (d / f"rob2-calib-{setting.replace(' ', '')}.json").write_text(json.dumps(calib), encoding="utf-8")
        print(f"{setting}: landmark ratios {calib['landmark']}", flush=True)


@lru_cache(maxsize=None)
def _calib(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _shapes(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def jobs(out: Path) -> list[dict]:
    d = _dir(out)
    js = []
    for cond in CONDITIONS:
        for setting in SETTINGS:
            for fi, fold in enumerate(loso_folds()):
                z = F._users(str(d / f"full-{fold.name}.npz"))
                for u in range(AF.FULL_USERS_PER_FOLD):
                    amps = [F.LEVEL, F.LEVEL] if setting == "fixed 12" else [float(a) for a in z["amps_wide"][u]]
                    for cand in CANDS:
                        js.append(dict(condition=cond, setting=setting, fold=fold.name, fold_index=fi,
                                       index=u, user=f"{fold.name}/f{u}", candidate=cand, amps=amps,
                                       answer_seed=int(z["answer_seeds"][u]) + F.SETTINGS.index(setting)))
    return js


def _observer(spec, ideal, seed, user, setting, out):
    from prompref.experiment.phase5 import NOISE_SALT, per_user_s
    from prompref.user.drift import DriftingObserver, path_for_scale, scale_for_displacement
    from prompref.user.phase_weight import load_bio_weight, pref_weight
    from prompref.user.schedule import ScheduledObserver, linear_path
    from prompref.user.smooth_utility import RandomSmoothUtility

    s = float(spec.get("s", AF.S))
    if "s_logsd" in spec:
        s = per_user_s(s, spec["s_logsd"], seed)
    lapse = float(spec.get("lapse", AF.LAPSE))
    knobs = dict(s=s, lapse=lapse, order_bias=float(spec.get("order_bias", 0.0)))
    calib = _calib(str(_dir(out) / f"rob2-calib-{setting.replace(' ', '')}.json"))
    if spec.get("metric") == "weber":
        knobs["metric"] = "weber"
    if spec.get("metric") == "landmark":
        knobs.update(metric="landmark", landmark_mag_fraction=spec["mag_fraction"],
                     utility_divisor=float(calib["landmark"][str(spec["mag_fraction"])]))
    if spec.get("metric") == RANDOM_SMOOTH:
        e = calib["users"][user]["random"]
        knobs.update(metric=RANDOM_SMOOTH, smooth_utility=RandomSmoothUtility.for_user(
            seed, np.array(e["mean"]), np.array(e["sd"]), np.array(e["centre"]), scale=float(e["scale"])))
    if "phase" in spec:
        w = (load_bio_weight(out / "phase5-schaefer")[0] if spec["phase"] == "bio"
             else pref_weight(seed, 0.5 if spec["phase"] == "pref" else 1.0))
        knobs.update(metric=PHASE_WEIGHTED, phase_weight=w,
                     utility_scale=float(calib["users"][user][spec["phase"]]))
    if "asymmetry" in spec:
        knobs["asymmetry"] = float(spec["asymmetry"])
    if "familiarity" in spec:
        knobs.update(familiarity_weight=float(spec["familiarity"]), habitual_nm=np.zeros_like(ideal))
    if "noise" in spec:
        knobs.update(device_noise_nm=float(spec["noise"]), noise_rng=np.random.default_rng([int(seed), NOISE_SALT]))
    if "link" in spec:
        knobs.update(link=spec["link"], link_scale=LINK_SCALES[spec["link"]])
    if "no_pref" in spec:
        knobs.update(allow_no_preference=True, no_pref_threshold=float(spec["no_pref"]) * true_jnd(1.0),
                     no_pref_hard=True)
    if "lapse_path" in spec or "s_path" in spec:
        paths = {}
        if "lapse_path" in spec:
            paths["lapse_path"] = linear_path(*spec["lapse_path"], F.N_COMP)
        if "s_path" in spec:
            paths["s_path"] = linear_path(*spec["s_path"], F.N_COMP)
        end_s = spec["s_path"][1] if "s_path" in spec else s
        end_l = spec["lapse_path"][1] if "lapse_path" in spec else lapse
        return ScheduledObserver(ideal_nm=ideal, **knobs, **paths), end_s, end_l
    if "drift_jnd" in spec:
        g = scale_for_displacement(ideal, spec["drift_jnd"] * true_jnd(1.0))
        path = path_for_scale(ideal, g, F.N_COMP)
        return DriftingObserver(ideal_nm=path[0], ideal_path=path, **knobs), s, lapse
    return Observer(ideal_nm=ideal, **knobs), s, lapse


def run_one(job: dict, out: str) -> dict:
    try:
        out_p = Path(out)
        spec = CONDITIONS[job["condition"]]
        d = _dir(out_p)
        ppath = str(d / f"fold-{job['fold']}.npz")
        u, setting, cand = job["index"], job["setting"], job["candidate"]
        if "shapes" in spec:
            sh = _shapes(str(d / f"rob2-shapes-{job['fold']}.npz"))
            shape = (sh[f"{spec['shapes']}_minus"][u], sh[f"{spec['shapes']}_plus"][u])
        else:
            z = F._users(str(d / f"full-{job['fold']}.npz"))
            shape = (z["user_minus"][u], z["user_plus"][u])
        ideal = AF.compose(job["amps"][0], job["amps"][1], *shape)
        seed = int(job["answer_seed"])
        observer, s_end, l_end = _observer(spec, ideal, seed, job["user"], setting, out_p)
        dfold = R._fold(ppath, job["fold_index"])
        g_minus, g_plus = AF.unit_bouts(dfold["generic"])
        run_dir = out_p / "phase5-schaefer"
        lseed = seed % 2**32
        if setting == "wide":
            lo, hi = F.RANGES["wide"]
            centre = math.sqrt(lo * hi)
            space = _space(setting, ppath, job["fold_index"])
            if cand == "psypref":
                learner = AF.TwoBoutGuard(AF.metric_psypref(space, lseed), space, 0.5 * lo, seed=lseed)
            elif cand.startswith("cmaes_sch"):
                space = AF.schaefer_range_space(run_dir, job["fold"], lo, hi)
                learner = AF.comparator(cand, space, lseed, s=observer.s)
            else:
                learner = AF.comparator(cand, space, lseed, s=observer.s)
        else:
            centre = F.LEVEL
            space = _space(setting, ppath, job["fold_index"])
            if cand == "psypref":
                learner = AF.psypref(space, lseed)
            elif cand.startswith("cmaes_sch"):
                space = AF.schaefer_fixed_space(run_dir, job["fold"], F.LEVEL)
                learner = AF.comparator(cand, space, lseed, s=observer.s)
            else:
                learner = AF.comparator(cand, space, lseed, s=observer.s)
        generic = AF.compose(centre, centre, g_minus, g_plus)
        rng = np.random.default_rng([seed, AF.OBSERVER_SALT])
        try:
            rec = run_session(space.design, learner, observer, rng, n_comparisons=F.N_COMP,
                              randomise_order=True, d_prior=rms_distance(generic, ideal), d_floor=0.0,
                              scores={"own": observer.own_distance})
        finally:
            inner = getattr(learner, "inner", None) or getattr(learner, "learner", None)
            inner = getattr(inner, "inner", inner)
            if inner is not None and hasattr(inner, "close"):
                inner.close()
        return dict(job, s_score=float(s_end), lapse_score=float(l_end),
                    own_d0=float(observer.own_distance(generic)), own=list(rec["own"]),
                    distances=list(rec["distances"]), error=None)
    except Exception as exc:  # one bad session must not stop the run
        return dict(job, error=f"{type(exc).__name__}: {exc}")


def _key(row):
    return row["user"], row["condition"], row["setting"], row["candidate"]


def run(out: Path, workers: int) -> None:
    path = _dir(out) / "rob2_sessions.jsonl"
    done = set()
    if path.exists():
        done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = [j for j in jobs(out) if _key(j) not in done]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", _key(row), row["error"], flush=True)
            if n % 1000 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def _pref(rows, k=24):
    d0 = np.array([r["own_d0"] for r in rows])
    dk = np.array([r["own"][k - 1] for r in rows])
    s = np.array([r["s_score"] for r in rows])
    lam = np.array([r["lapse_score"] for r in rows])
    return float(np.mean(lam / 2 + (1 - lam) * norm.cdf((d0 - dk) / s)))


def report(out: Path) -> None:
    d = _dir(out)
    raw = [json.loads(x) for x in (d / "rob2_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in rows]
    full = {(r["user"], r["setting"]): r for r in
            map(json.loads, (d / "full_sessions.jsonl").read_text(encoding="utf-8").splitlines())
            if r.get("error") is None and r["candidate"] == "psypref"}
    rng = np.random.default_rng(AF.SEED_FULL + 18)
    L = ["# Registration ROB2: the earlier robustness conditions in the amplitude-free design", "",
         f"Spec section 18 (fb6563b). {len(rows)} sessions, {len(errors)} errors. Preference over generic "
         f"assistance in the user's own utility at k = 24 (s and λ at the session's end), 95 % intervals "
         f"from {R.N_BOOT} draws, users resampled within folds.", "",
         "| condition | setting | PsyPref | CMA-ES (Schäfer) | best published | PsyPref − CMA-ES (Schäfer) | "
         "PsyPref − best published | PsyPref − FULL |", "|---|---|---|---|---|---|---|---|"]
    res = {}
    for cond in CONDITIONS:
        for setting in SETTINGS:
            by = {}
            for (u, c, s, cc), r in rows.items():
                if c == cond and s == setting:
                    by.setdefault(cc, {})[u] = r
            if "psypref" not in by or len(by["psypref"]) < 8:
                continue
            pub = [c for c in ("g", "d_z", "cmaes") if c in by and len(by[c]) >= 8]
            best = max(pub, key=lambda c: _pref(list(by[c].values()))) if pub else None

            def diff(other):
                users = sorted(set(by["psypref"]) & set(other))
                ps = [dict(fold=by["psypref"][u]["fold"], a=by["psypref"][u], b=other[u]) for u in users]
                return R._boot(ps, lambda x: _pref([p["a"] for p in x]) - _pref([p["b"] for p in x]), rng)

            p = R._boot(list(by["psypref"].values()), _pref, rng)
            dc = diff(by["cmaes_sch4"]) if "cmaes_sch4" in by else None
            db = diff(by[best]) if best else None
            fu = {u: full[u, setting] for u in by["psypref"] if (u, setting) in full}
            for r in fu.values():
                r.setdefault("own_d0", r["d0"]), r.setdefault("own", r["distances"])
                r.setdefault("s_score", AF.S), r.setdefault("lapse_score", AF.LAPSE)
            dfull = diff(fu) if len(fu) >= 8 else None
            res[cond, setting] = dict(p=p, dc=dc, db=db, dfull=dfull)
            fmt = lambda t: "—" if t is None else f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"
            L.append(f"| {cond} | {setting} | {R._f(p, 3)} | "
                     f"{R._f(R._boot(list(by['cmaes_sch4'].values()), _pref, rng), 3) if 'cmaes_sch4' in by else '—'} | "
                     f"{F.NAMES.get(best, '—')} | {fmt(dc)} | {fmt(db)} | {fmt(dfull)} |")
    L += ["", "## Predictions", ""]
    cells = [v for v in res.values() if v["dc"] is not None]
    share = np.mean([v["dc"][1] > 0 for v in cells]) if cells else float("nan")
    L.append(f"- PRB1 (ahead of CMA-ES on Schäfer in ≥ 80 % of cells): {'held' if share >= 0.8 else 'failed'} — {100 * share:.0f} %")
    drops = {k: v["dfull"][0] for k, v in res.items() if k[0].startswith("A") and v["dfull"] is not None}
    L.append(f"- PRB2 (no answer condition lowers PsyPref by more than 5 points): "
             f"{'held' if all(x >= -0.05 for x in drops.values()) else 'failed'} — worst "
             f"{min(drops, key=drops.get) if drops else '—'} {100 * min(drops.values()) if drops else float('nan'):+.1f}")
    bad = [k for k, v in res.items() if v["db"] is not None and not v["db"][1] > 0 and not k[0].startswith("B8")]
    L.append(f"- PRB3 (ahead of the best published optimizer everywhere except B8): {'held' if not bad else 'failed — ' + str(bad)}")
    pes = [res[("C4 pessimistic", s)]["p"][0] for s in SETTINGS if ("C4 pessimistic", s) in res]
    if len(pes) == 2:
        L.append(f"- PRB4 (pessimistic user below 0.60 in both settings): {'held' if max(pes) < 0.6 else 'failed'} — "
                 + ", ".join(f"{x:.3f}" for x in pes))
    b5 = [res[(c, s)]["dc"] for c in ("B5a features static", "B5b features walking") for s in SETTINGS if (c, s) in res]
    if b5:
        L.append(f"- PRB5 (feature-judging: ahead of CMA-ES on Schäfer): "
                 f"{'held' if all(t[1] > 0 for t in b5) else 'failed or uncertain'}")
    if errors:
        L += ["", "## Errors", ""] + [f"- {_key(e)}: {e['error']}" for e in errors[:30]]
    (d / "amp_rob2_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("prep", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"prep": lambda: prep(args.out), "run": lambda: run(args.out, args.workers),
     "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
