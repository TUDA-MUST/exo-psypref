"""Test ROB: robustness of the full comparison (R1-R7).

Usage:
  python scripts/icorr_amp_robust.py prep
  python scripts/icorr_amp_robust.py run [--workers 8]
  python scripts/icorr_amp_robust.py report

Settings: wide 3-35 Nm (AM) and fixed 12 Nm (B_fixed),
FULL's users (scripts/icorr_amp_full.py) unless stated, 40 comparisons. Outputs in
<out>/icorr/amp_free/: xd-shapes.npz, robust_sessions.jsonl (resumable), amp_robust_report.md.
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
from prompref.population.folds import child_sequence, loso_folds  # noqa: E402
from prompref.user.simulated_user import Observer, rms_distance  # noqa: E402

SETTINGS = ("wide", "fixed 12")
CHEAP = ("psypref", "cmaes_sch4", "g", "d_z", "cmaes")   # cmaes_sch4: a change
#: Section 17: the conditions, their knobs and their candidates.
CONDITIONS = {
    "R1 s=0.5": dict(s=0.5, cands=CHEAP),
    "R2 s=2.0": dict(s=2.0, cands=CHEAP + ("gp_bald",)),
    "R3 lapse=0.3": dict(lapse=0.3, cands=CHEAP),
    "R4 negative power": dict(np=True, cands=CHEAP),
    "R5 K=12": dict(k12=True, cands=("psypref",)),
    "R6 other datasets": dict(users="xd", cands=CHEAP + ("gp_bald",)),
    "R7 surrogates": dict(users="surrogate", cands=CHEAP),
}
XD_CELLS = ("xd-fy", "xd-fo", "xd-r")
XD_PER_CELL = 40


def _dir(out: Path) -> Path:
    return out / "icorr" / "amp_free"


def prep(out: Path) -> None:
    """R6's users: 40 per XD cell, XD's leg choice and factor draws at the small scale."""
    from dataclasses import replace

    from icorr_r2_prep import load_external
    from prompref.experiment import r2
    from prompref.population.factors import sample_draw
    from prompref.population.generator import OrderViolation, user_curve

    shift, switches, timing = AF.population_config(out / "phase5-schaefer", "loso-AB30")
    arrays = dict(minus=[], plus=[], cell=[], subject=[], fold=[], seed=[], amps=[])
    for c, cell in enumerate(XD_CELLS):
        trials = load_external(cell).trials
        legs = r2.xd_legs(cell, trials)
        root = np.random.SeedSequence([AF.SEED_FULL, 900 + c])
        amp_rng = np.random.default_rng(child_sequence(root, 2))
        amps = AF._loguniform(amp_rng, *F.RANGES["wide"], (XD_PER_CELL, 2))
        for i in range(XD_PER_CELL):
            index = i * (r2.XD_N_USERS // XD_PER_CELL)   # spread over the cell's conditions
            pool = legs[r2.xd_condition(cell, index)]
            rng = np.random.default_rng(child_sequence(child_sequence(root, 0), i))
            trial = pool[int(rng.integers(len(pool)))]
            for _ in range(AF.MAX_CONSECUTIVE_REJECTIONS + 1):
                draw = replace(sample_draw(rng, switches, timing, shift),
                               ext_scale=AF.SHAPE_SCALE, flex_scale=AF.SHAPE_SCALE)
                try:
                    s_minus, s_plus = AF.unit_parts(user_curve(trial, draw))
                    break
                except (OrderViolation, ValueError):
                    continue
            else:
                raise RuntimeError(f"{cell} user {i}: no accepted draw")
            arrays["minus"].append(s_minus)
            arrays["plus"].append(s_plus)
            arrays["cell"].append(cell)
            arrays["subject"].append(f"{trial.dataset}/{trial.subject}_{trial.side}")
            arrays["fold"].append(i % len(loso_folds()))
            arrays["seed"].append(child_sequence(child_sequence(root, 1), i).generate_state(1, dtype=np.uint64)[0])
            arrays["amps"].append(amps[i])
        print(f"{cell}: {XD_PER_CELL} users", flush=True)
    dest = _dir(out) / "xd-shapes.npz"
    np.savez(dest, **{k: np.array(v) for k, v in arrays.items()})
    print("wrote", dest)


@lru_cache(maxsize=None)
def _xd(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def jobs(out: Path) -> list[dict]:
    d = _dir(out)
    js = []
    for cond, spec in CONDITIONS.items():
        for si, setting in enumerate(SETTINGS):
            base = dict(condition=cond, setting=setting)
            if spec.get("users") == "xd":
                z = _xd(str(d / "xd-shapes.npz"))
                for i in range(len(z["cell"])):
                    fi = int(z["fold"][i])
                    fold = loso_folds()[fi].name
                    amps = [F.LEVEL, F.LEVEL] if setting == "fixed 12" else [float(a) for a in z["amps"][i]]
                    for cand in spec["cands"]:
                        js.append(dict(base, fold=fold, fold_index=fi, index=i, user=f"{z['cell'][i]}/{i:02d}",
                                       candidate=cand, amps=amps, answer_seed=int(z["seed"][i]) + si,
                                       _parts=str(d / f"fold-{fold}.npz")))
                continue
            for fi, fold in enumerate(loso_folds()):
                z = F._users(str(d / f"full-{fold.name}.npz"))
                for u in range(AF.FULL_USERS_PER_FOLD):
                    amps = ([F.LEVEL, F.LEVEL] if setting == "fixed 12"
                            else [float(a) for a in z["amps_wide"][u]])
                    for cand in spec["cands"]:
                        js.append(dict(base, fold=fold.name, fold_index=fi, index=u,
                                       user=f"{fold.name}/f{u}", candidate=cand, amps=amps,
                                       # FULL's seeds: wide is FULL's setting 1, fixed 12 setting 2
                                       answer_seed=int(z["answer_seeds"][u]) + F.SETTINGS.index(setting),
                                       _parts=str(d / f"fold-{fold.name}.npz"),
                                       _users=str(d / f"full-{fold.name}.npz")))
    # GP-BALD last: the cheap conditions finish first, and an interrupted run resumes GP-BALD
    return sorted(js, key=lambda j: j["candidate"] == "gp_bald")


def _shape_and_subject(job, out):
    if job["condition"].startswith("R6"):
        z = _xd(str(_dir(Path(out)) / "xd-shapes.npz"))
        return (z["minus"][job["index"]], z["plus"][job["index"]]), str(z["subject"][job["index"]])
    z = F._users(job["_users"])
    return (z["user_minus"][job["index"]], z["user_plus"][job["index"]]), str(z["user_subject"][job["index"]])


def run_one(job: dict, out: str) -> dict:
    ppath = job.pop("_parts")
    try:
        spec = CONDITIONS[job["condition"]]
        s, lapse = float(spec.get("s", AF.S)), float(spec.get("lapse", AF.LAPSE))
        d = R._fold(ppath, job["fold_index"])
        shape, subject = _shape_and_subject(job, out)
        job.pop("_users", None)
        setting, cand = job["setting"], job["candidate"]
        run_dir = Path(out) / "phase5-schaefer"
        ideal = AF.compose(job["amps"][0], job["amps"][1], *shape)
        if job["condition"].startswith("R7"):
            # XS's phase randomization of the user's preferred curve, at the device's top
            from prompref.experiment.xsource import phase_randomize

            top = F.RANGES["wide"][1] if setting == "wide" else F.LEVEL
            rng_s = np.random.default_rng(np.random.SeedSequence([AF.SEED_FULL, 700, job["fold_index"], job["index"]]))
            ideal = np.clip(phase_randomize(ideal, rng_s), -top, top)
        g_minus, g_plus = AF.unit_parts(d["generic"])
        seed = int(job["answer_seed"]) % 2**32
        # test LONG (section 27): the budget, and PsyPref-T's tie handling by name
        n_comp, ties = int(job.get("n_comp", F.N_COMP)), job.get("ties")
        if ties is not None and cand != "psypref":
            raise ValueError(f"ties {ties} is for PsyPref only")
        tkw = {} if ties is None else dict(ties=ties)   # only when set, as in FULL's run_one
        if setting == "wide":
            lo, hi = F.RANGES["wide"]
            centre = math.sqrt(lo * hi)
            if spec.get("k12"):
                mu, sigma = R._a12_moments(ppath, job["fold_index"], "wide")
                space = AF.promp_space(mu, sigma, limit=hi, basis=AF.VonMisesBasis(12))
            else:
                mu, sigma = R._a_moments(ppath, job["fold_index"], "wide")
                space = AF.promp_space(mu, sigma, limit=hi)
            if cand == "psypref":
                learner = AF.TwoPartGuard(AF.metric_psypref(space, seed, **tkw), space, 0.5 * lo,
                                          seed=seed)
            elif cand.startswith("cmaes_sch"):
                space = AF.schaefer_range_space(run_dir, job["fold"], lo, hi)
                learner = AF.comparator(cand, space, seed, s=s)
            else:
                learner = AF.comparator(cand, space, seed, s=s)
        else:
            centre = F.LEVEL
            if spec.get("k12"):
                p12 = R._parts12(ppath)
                space = AF.fixed_level_space(p12["w_minus"], p12["w_plus"], F.LEVEL, basis=AF.VonMisesBasis(12))
            else:
                space = AF.fixed_level_space(d["w_minus"], d["w_plus"], F.LEVEL)
            if cand == "psypref":
                learner = AF.psypref(space, seed, **tkw)
            elif cand.startswith("cmaes_sch"):
                space = AF.schaefer_fixed_space(run_dir, job["fold"], F.LEVEL)
                learner = AF.comparator(cand, space, seed, s=s)
            else:
                learner = AF.comparator(cand, space, seed, s=s)
        generic = AF.compose(centre, centre, g_minus, g_plus)
        if spec.get("np"):
            theta_dot = np.load(Path(out) / "icorr" / "amp_free" / "hip_velocity.npz")[subject]
            kappa = float(AF.np_kappas(1000 + job["fold_index"], AF.FULL_USERS_PER_FOLD)[job["index"]])
            observer = AF.NegativePowerObserver(ideal_nm=ideal, s=s, lapse=lapse, theta_dot=theta_dot, kappa=kappa)
            job = dict(job, kappa=kappa)
        else:
            observer = Observer(ideal_nm=ideal, s=s, lapse=lapse)
        rng = np.random.default_rng([int(job["answer_seed"]), AF.OBSERVER_SALT])
        try:
            rec = run_session(space.design, learner, observer, rng, n_comparisons=n_comp,
                              randomise_order=True, d_prior=rms_distance(generic, ideal), d_floor=0.0,
                              scores={"own": observer.own_distance})
        finally:
            inner = getattr(learner, "inner", None) or getattr(learner, "learner", None)
            inner = getattr(inner, "inner", inner)
            if inner is not None and hasattr(inner, "close"):
                inner.close()
        delivered = space.design.curve(np.asarray(rec["best"], dtype=float))
        ident = AF.peaks(delivered)
        return dict(job, s=s, lapse=lapse, d0=rms_distance(generic, ideal),
                    own_d0=observer.own_distance(generic), distances=list(rec["distances"]),
                    own=list(rec["own"]), identified=[max(ident[0], 0.0), max(ident[1], 0.0)],
                    preferred=list(AF.peaks(ideal)), error=None)
    except Exception as exc:  # one bad session must not stop the run
        job.pop("_users", None)
        return dict(job, error=f"{type(exc).__name__}: {exc}")


def _key(row):
    return row["user"], row["condition"], row["setting"], row["candidate"]


def run(out: Path, workers: int) -> None:
    path = _dir(out) / "robust_sessions.jsonl"
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
            if n % 200 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- report

def _pref(rows, k, own=False):
    d0 = np.array([r["own_d0" if own else "d0"] for r in rows])
    dk = np.array([r["own" if own else "distances"][k - 1] for r in rows])
    s = np.array([r["s"] for r in rows])
    lam = np.array([r["lapse"] for r in rows])
    return float(np.mean(lam / 2 + (1 - lam) * norm.cdf((d0 - dk) / s)))


def report(out: Path) -> None:
    d = _dir(out)
    raw = [json.loads(x) for x in (d / "robust_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in rows]
    full = {(r["user"], r["setting"], r["candidate"]): r for r in
            map(json.loads, (d / "full_sessions.jsonl").read_text(encoding="utf-8").splitlines())
            if r.get("error") is None}
    for r in full.values():
        r.setdefault("s", AF.S), r.setdefault("lapse", AF.LAPSE)
    rng = np.random.default_rng(AF.SEED_FULL + 17)
    L = ["# Test plan ROB: robustness of the full comparison", "",
         f"Design note 17 (8256b0b). {len(rows)} sessions, {len(errors)} errors. 40 comparisons; "
         f"preference over generic assistance (R4: in the user's own utility), 95 % intervals from "
         f"{R.N_BOOT} draws, users resampled within folds.", ""]
    ahead = {}
    for cond, spec in CONDITIONS.items():
        own = bool(spec.get("np"))
        L += [f"## {cond}", "", "| setting | optimizer | n | k = 24 | k = 40 | PsyPref − it, k = 24 (points) |",
              "|---|---|---|---|---|---|"]
        for setting in SETTINGS:
            for cand in spec["cands"]:
                rs = [r for (u, c, s, cc), r in rows.items() if c == cond and s == setting and cc == cand]
                if len(rs) < 8:
                    continue
                cells = [R._f(R._boot(rs, lambda x, k=k: _pref(x, k, own), rng), 3) for k in (24, 40)]
                diff = ""
                if cand != "psypref":
                    a = {r["user"]: r for (u, c, s, cc), r in rows.items() if c == cond and s == setting and cc == "psypref"}
                    b = {r["user"]: r for r in rs}
                    ps = [dict(fold=a[u]["fold"], a=a[u], b=b[u]) for u in sorted(set(a) & set(b))]
                    t = R._boot(ps, lambda x: _pref([p["a"] for p in x], 24, own) - _pref([p["b"] for p in x], 24, own), rng)
                    ahead[cond, setting, cand] = t
                    diff = f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"
                L.append(f"| {setting} | {F.NAMES[cand]} | {len(rs)} | {cells[0]} | {cells[1]} | {diff} |")
        L.append("")

    def held(t):
        return "held" if t[1] > 0 else ("failed" if t[2] < 0 else "uncertain")

    def paired_full(cond, setting, k, own):
        a = {r["user"]: r for (u, c, s, cc), r in rows.items() if c == cond and s == setting and cc == "psypref"}
        b = {u: r for (u, s, cc), r in full.items() if s == setting and cc == "psypref"}
        ps = [dict(fold=a[u]["fold"], a=a[u], b=b[u]) for u in sorted(set(a) & set(b))]
        return None if len(ps) < 8 else R._boot(
            ps, lambda x: _pref([p["a"] for p in x], k, own) - _pref([p["b"] for p in x], k), rng)

    L += ["## Predictions", ""]
    v = {(c, s): held(ahead[c, s, "cmaes_sch4"]) for c in CONDITIONS for s in SETTINGS if (c, s, "cmaes_sch4") in ahead}
    L.append(f"- PR1 (ahead of CMA-ES on Schäfer everywhere): {'held' if all(x == 'held' for x in v.values()) else v}")
    t = [ahead.get(("R2 s=2.0", s, "gp_bald")) for s in SETTINGS]
    if all(x is not None for x in t):
        L.append(f"- PR2 (s = 2.0: PsyPref − GP-BALD includes 0 in at least one setting): "
                 f"{'held' if any(x[1] <= 0 <= x[2] for x in t) else 'failed'} — "
                 + ", ".join(f"{s} {100 * x[0]:+.1f} [{100 * x[1]:+.1f}, {100 * x[2]:+.1f}]" for s, x in zip(SETTINGS, t)))
    v = {(c, s, cc): held(ahead[c, s, cc]) for c in CONDITIONS for s in SETTINGS for cc in ("g", "d_z", "cmaes")
         if (c, s, cc) in ahead}
    L.append(f"- PR3 (ahead of Arens' GP, LineCoSpar, CMA-ES on the ProMP everywhere): "
             f"{'held' if all(x == 'held' for x in v.values()) else {k: x for k, x in v.items() if x != 'held'}}")
    t = [paired_full("R4 negative power", s, 24, True) for s in SETTINGS]
    if all(x is not None for x in t):
        L.append(f"- PR4 (negative power: within 5 points of FULL): "
                 f"{'held' if all(abs(x[0]) <= 0.05 for x in t) else 'failed'} — "
                 + ", ".join(f"{s} {100 * x[0]:+.1f}" for s, x in zip(SETTINGS, t)))
    rs = [r for (u, c, s, cc), r in rows.items() if c == "R6 other datasets" and s == "wide" and cc == "psypref"]
    if rs:
        p = _pref(rs, 24)
        L.append(f"- PR5 (other datasets, wide: PsyPref ≥ 0.70): {'held' if p >= 0.70 else 'failed'} — {p:.3f}")
    t = paired_full("R5 K=12", "wide", 40, False)
    if t is not None:
        L.append(f"- PR6 (K = 12 at k = 40 not below K = 8 by more than 2 points, wide): "
                 f"{'held' if t[1] > -0.02 else ('failed' if t[2] < -0.02 else 'uncertain')} — "
                 f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]")
    v = [ahead.get(("R7 surrogates", s, "cmaes_sch4")) for s in SETTINGS]
    if all(x is not None for x in v):
        L.append(f"- PR7 (surrogates: ahead of CMA-ES on Schäfer in both settings): "
                 f"{'held' if all(x[1] > 0 for x in v) else [held(x) for x in v]}")
    if errors:
        L += ["", "## Errors", ""] + [f"- {_key(e)}: {e['error']}" for e in errors[:30]]
    (d / "amp_robust_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
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
