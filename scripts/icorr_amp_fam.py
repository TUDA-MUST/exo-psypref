"""Test FAM: interpretable families' floors, ceilings and PsyPref with projection.

Usage:
  python scripts/icorr_amp_fam.py floors [--workers 8]
  python scripts/icorr_amp_fam.py run [--workers 8]
  python scripts/icorr_amp_fam.py report

FULL's 200 users, preferred profiles and answer seeds;
the wide range 3-35 Nm and the fixed 12 Nm level. ``floors`` writes every user's floor per family
(fam_floors.jsonl), ``run`` the projection sessions (fam_sessions.jsonl; both resumable),
``report`` fam_report.md and fam_numbers.json (the figure's numbers). FULL's rows
(full_sessions.jsonl) and ROB R5's (robust_sessions.jsonl) are read, never rerun.
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
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
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
from prompref.population.folds import loso_folds  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.oracle import Observer, rms_distance  # noqa: E402

SETTINGS = ("wide", "fixed 12")
LABEL = {"schaefer_i": "Schäfer (i)", "pchip_i": "PCHIP (i)", "schaefer_ii": "Schäfer (ii)",
         "trapezoid": "trapezoid", "franks": "Franks", "pchip_ii": "PCHIP (ii)", "pchip": "PCHIP",
         **{f"promp{k}": f"ProMP {k}" for k in AF.PROMP_K}}
#: Parameters searched per family and setting (the fixed level holds both peaks).
N_PARAMS = {"wide": {"schaefer_i": 4, "pchip_i": 5, "schaefer_ii": 6, "trapezoid": 6, "franks": 8,
                     "pchip_ii": 6, **{f"promp{k}": k for k in AF.PROMP_K}},
            "fixed 12": {"schaefer_i": 4, "trapezoid": 4, "franks": 6, "pchip": 4,
                         **{f"promp{k}": k for k in AF.PROMP_K}}}
JND = true_jnd(AF.S)   # 0.674 Nm


def _families(setting: str) -> tuple:
    named = AF.FAM_RANGE if setting in F.RANGES else AF.FAM_LEVEL
    return named + tuple(f"promp{k}" for k in AF.PROMP_K)


def _kw(setting: str) -> dict:
    return dict(peak_range=F.RANGES[setting]) if setting in F.RANGES else dict(level=F.LEVEL)


def _user(job: dict):
    """The job's preferred profile and generic assistance, exactly as FULL's run_one builds them."""
    z, d = F._users(job["_users"]), R._fold(job["_parts"], job["fold_index"])
    u = job["index"]
    ideal = AF.compose(job["amps"][0], job["amps"][1], z["user_minus"][u], z["user_plus"][u])
    centre = math.sqrt(F.RANGES[job["setting"]][0] * F.RANGES[job["setting"]][1]) \
        if job["setting"] in F.RANGES else F.LEVEL
    generic = AF.compose(centre, centre, *AF.unit_bouts(d["generic"]))
    return ideal, generic, d


def _base_jobs(out: Path) -> list[dict]:
    """FULL's jobs of PsyPref in FAM's settings: users, peaks and answer seeds."""
    return [dict(j) for j in F.jobs(out) if j["candidate"] == "psypref" and j["setting"] in SETTINGS]


# ----------------------------------------------------------------------------- floors

def _floor_key(r):
    return r["user"], r["setting"], r["family"]


def floor_one(job: dict, out: str) -> list[dict]:
    """Every family's floor for one user and setting."""
    rows = []
    try:
        ideal, generic, _ = _user(job)
        run_dir = Path(out) / "phase5-schaefer"
        base = {k: v for k, v in job.items() if not k.startswith("_") and k != "candidate"}
        for fam in job["_families"]:
            try:
                if fam.startswith("promp"):
                    floor = AF.promp_floor(ideal, int(fam[5:]), **_kw(job["setting"]))
                    theta = None
                else:
                    design = AF.family_design(fam, run_dir, job["fold"], **_kw(job["setting"]))
                    seed = [AF.SEED_FULL, zlib.crc32(f"{job['user']}|{job['setting']}|{fam}".encode())]
                    theta, floor = AF.family_floor(design, ideal, seed)
                    theta = [float(x) for x in theta]
                rows.append(dict(base, family=fam, floor=floor, theta=theta,
                                 d0=rms_distance(generic, ideal), error=None))
            except Exception as exc:  # one bad fit must not stop the run
                rows.append(dict(base, family=fam, error=f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        rows.append(dict(job, family=None, error=f"{type(exc).__name__}: {exc}"))
    return rows


def floors(out: Path, workers: int) -> None:
    path = F._dir(out) / "fam_floors.jsonl"
    done = set()
    if path.exists():
        done = {_floor_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = []
    for j in _base_jobs(out):
        fams = [f for f in _families(j["setting"]) if (j["user"], j["setting"], f) not in done]
        if fams:
            todo.append(dict(j, _families=fams))
    print(f"{len(todo)} user-settings with floors to fit", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(floor_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            for row in fut.result():
                f.write(json.dumps(row) + "\n")
                if row.get("error"):
                    print("ERROR", row.get("user"), row.get("family"), row["error"], flush=True)
            f.flush()
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- projection sessions

def proj_jobs(out: Path) -> list[dict]:
    js = []
    for j in _base_jobs(out):
        fams = AF.PROJ_RANGE if j["setting"] in F.RANGES else AF.PROJ_LEVEL
        js += [dict(j, candidate=f"proj_{fam}") for fam in fams]
    return js


def run_one(job: dict, out: str) -> dict:
    try:
        ideal, generic, d = _user(job)
        ppath = job["_parts"]
        job = {k: v for k, v in job.items() if not k.startswith("_")}
        setting, fam = job["setting"], job["candidate"][len("proj_"):]
        run_dir = Path(out) / "phase5-schaefer"
        seed = int(job["answer_seed"]) % 2**32
        family = AF.family_design(fam, run_dir, job["fold"], **_kw(setting))
        if setting in F.RANGES:
            lo, hi = F.RANGES[setting]
            mu, sigma = R._a_moments(ppath, job["fold_index"], setting)
            space = AF.promp_space(mu, sigma, limit=hi)
            inner = AF.TwoBoutGuard(AF.metric_psypref(space, seed), space, 0.5 * lo, seed=seed)
            lens = AF.family_lens(inner, space, family)
        else:
            space = AF.fixed_level_space(d["w_minus"], d["w_plus"], F.LEVEL)
            lens = AF.family_lens(AF.psypref(space, seed), space, family, level=F.LEVEL)
        observer = Observer(ideal_nm=ideal, s=AF.S, lapse=AF.LAPSE)
        rng = np.random.default_rng([int(job["answer_seed"]), AF.OBSERVER_SALT])
        rec = run_session(family, lens, observer, rng, n_comparisons=F.N_COMP, randomise_order=True,
                          d_prior=rms_distance(generic, ideal), d_floor=0.0)
        delivered = family.curve(np.asarray(rec["best"], dtype=float))
        ident = AF.peaks(delivered)
        return dict(job, identified=[max(ident[0], 0.0), max(ident[1], 0.0)],
                    d0=rms_distance(generic, ideal), distances=list(rec["distances"]),
                    lens_gap_median=rec.get("lens_gap_median"), lens_gap_p95=rec.get("lens_gap_p95"),
                    error=None)
    except Exception as exc:  # one bad session must not stop the run
        return dict({k: v for k, v in job.items() if not k.startswith("_")},
                    error=f"{type(exc).__name__}: {exc}")


def run(out: Path, workers: int) -> None:
    path = F._dir(out) / "fam_sessions.jsonl"
    done = set()
    if path.exists():
        done = {F._key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = [j for j in proj_jobs(out) if F._key(j) not in done]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", F._key(row), row["error"], flush=True)
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- report

def _ceil(rows) -> float:
    d0 = np.array([r["d0"] for r in rows])
    fl = np.array([r["floor"] for r in rows])
    return float(np.mean(AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((d0 - fl) / AF.S)))


def _load(path: Path, key) -> dict:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("error") is None:
            rows[key(r)] = r
    return rows


def _verdict(t, side: str, bound: float = 0.0):
    """Held when the interval lies wholly on the predicted side of ``bound``, failed when wholly
    on the other, uncertain otherwise."""
    if t is None:
        return None
    est, lo, hi = t
    if side == "above":
        return "held" if lo > bound else ("failed" if hi < bound else "uncertain")
    return "held" if hi < bound else ("failed" if lo > bound else "uncertain")


def _all(vs):
    vs = list(vs)
    if any(v is None for v in vs):
        return "incomplete"
    return "held" if all(v == "held" for v in vs) else ("failed" if any(v == "failed" for v in vs)
                                                        else "uncertain")


def report(out: Path) -> None:
    d = F._dir(out)
    fl = _load(d / "fam_floors.jsonl", _floor_key)
    ses = _load(d / "fam_sessions.jsonl", F._key)
    full = _load(d / "full_sessions.jsonl", F._key)
    rob = _load(d / "robust_sessions.jsonl", lambda r: (r["user"], r.get("condition"), r["setting"], r["candidate"]))
    rng = np.random.default_rng(AF.SEED_FULL + 19)
    boot = R._boot
    L = ["# Registration FAM: interpretable families in the amplitude-free design", "",
         f"Spec section 19 (c8ef8e8). {len(fl)} floors, {len(ses)} projection sessions. FULL's 200 "
         f"users; s = {AF.S} Nm, λ = {AF.LAPSE}. Ceiling: preference over generic assistance at "
         f"the user's floor. Floors in Nm and in JND ({JND:.3f} Nm). 95 % intervals from "
         f"{R.N_BOOT} draws, users resampled within folds.", ""]
    numbers = {"generated": __import__("datetime").datetime.now().isoformat(timespec="minutes"),
               "settings": {}}
    ceil, proj = {}, {}

    def fam_rows(setting, fam):
        return [r for (u, s, f), r in fl.items() if s == setting and f == fam]

    def paired(a, b):
        ua = {r["user"]: r for r in a}
        ub = {r["user"]: r for r in b}
        return [dict(fold=ua[u]["fold"], a=ua[u], b=ub[u]) for u in sorted(set(ua) & set(ub))]

    def sess(setting, cand, cond=None):
        src = full if cond is None else {k[:1] + k[2:]: v for k, v in rob.items() if k[1] == cond}
        if cond is None and cand.startswith("proj_"):
            src = ses
        return [r for (u, s, c), r in src.items() if s == setting and c == cand]

    for setting in SETTINGS:
        nums = numbers["settings"][setting] = {}
        L += [f"## {setting}", "", "| family | parameters | n | ceiling | PsyPref, k = 24 | PsyPref, k = 40 "
              "| median floor (Nm) | median floor (JND) |", "|---|---|---|---|---|---|---|---|"]
        for fam in _families(setting):
            rs = fam_rows(setting, fam)
            if len(rs) < 8:
                L.append(f"| {LABEL[fam]} | {N_PARAMS[setting][fam]} | {len(rs)} | — | | | | |")
                continue
            c = ceil[setting, fam] = boot(rs, _ceil, rng)
            fnm = boot(rs, lambda x: float(np.median([r["floor"] for r in x])), rng)
            e = dict(label=LABEL[fam], n_params=N_PARAMS[setting][fam], n=len(rs),
                     ceiling=dict(zip(("est", "lo", "hi"), c)),
                     floor_nm=dict(zip(("est", "lo", "hi"), fnm)),
                     floor_jnd=dict(zip(("est", "lo", "hi"), [v / JND for v in fnm])))
            cells = ["", ""]
            if fam.startswith("promp"):
                cand, cond = ("psypref", None) if fam == "promp8" else ("psypref", "R5 K=12")
                ps = sess(setting, cand, cond) if fam in ("promp8", "promp12") else []
            else:
                ps = sess(setting, f"proj_{fam}")
            if len(ps) >= 8:
                for i, k in enumerate((24, 40)):
                    t = proj[setting, fam, k] = boot(ps, lambda x, k=k: F._pref_k(x, k), rng)
                    cells[i] = R._f(t, 3)
                    e[f"psypref_k{k}"] = dict(zip(("est", "lo", "hi"), t))
                gaps = [r["lens_gap_median"] for r in ps if r.get("lens_gap_median") is not None]
                if gaps:
                    e["lens_gap_median_nm"] = float(np.median(gaps))
            nums[fam] = e
            L.append(f"| {LABEL[fam]} | {N_PARAMS[setting][fam]} | {len(rs)} | {R._f(c, 3)} | {cells[0]} | "
                     f"{cells[1]} | {R._f(fnm, 2)} | {fnm[0] / JND:.2f} |")
        L += ["", "PsyPref with projection minus PsyPref on the ProMP (FULL) and minus CMA-ES (tuned) "
              "on Schäfer's family (FULL), k = 24, points, paired:", ""]
        fams = AF.PROJ_RANGE if setting in F.RANGES else AF.PROJ_LEVEL
        for fam in fams:
            a = sess(setting, f"proj_{fam}")
            if len(a) < 8:
                continue
            for other in ("psypref", "cmaes_sch4"):
                pr = paired(a, sess(setting, other))
                if len(pr) >= 8:
                    proj[setting, fam, "minus", other] = t = boot(
                        pr, lambda x: F._pref_k([p["a"] for p in x], 24) - F._pref_k([p["b"] for p in x], 24), rng)
                    L.append(f"- {LABEL[fam]} − {F.NAMES[other]}: {100 * t[0]:+.1f} [{100 * t[1]:+.1f}, "
                             f"{100 * t[2]:+.1f}]")
            gaps = [r["lens_gap_median"] for r in a if r.get("lens_gap_median") is not None]
            if gaps:
                L.append(f"  - shown curve vs PsyPref's proposal, median RMS {np.median(gaps):.2f} Nm")
        L.append("")

    def cdiff(setting, f1, f2):
        pr = paired(fam_rows(setting, f1), fam_rows(setting, f2))
        if len(pr) < 8:
            return None
        return boot(pr, lambda x: _ceil([p["a"] for p in x]) - _ceil([p["b"] for p in x]), rng)

    diffs = {(f1, f2): cdiff("wide", f1, f2) for f1, f2 in
             (("schaefer_ii", "schaefer_i"), ("schaefer_ii", "pchip_i"), ("trapezoid", "schaefer_ii"))}
    diffs.update({("fixed", f): cdiff("fixed 12", "promp8", f) for f in ("schaefer_i", "trapezoid", "pchip")})
    L += ["## Ceiling differences (points, paired)", ""]
    for k, t in diffs.items():
        if t is not None:
            L.append(f"- {k}: {100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]")
    L += ["", "## Predictions", ""]
    pfm1 = [_verdict(ceil.get(("wide", f)), "below", 0.70) for f in ("schaefer_i", "pchip_i")]
    pfm1 += [_verdict(diffs[("schaefer_ii", f)], "above", 0.15) for f in ("schaefer_i", "pchip_i")]
    L.append(f"- PFM1 (wide: Schäfer (i), PCHIP (i) ceilings < 0.70 and ≥ 15 points below Schäfer (ii)): "
             f"{_all(pfm1)} {pfm1}")
    pfm2 = [_verdict(ceil.get(("wide", f)), "above", 0.85) for f in ("schaefer_ii", "franks", "pchip_ii")]
    pfm2.append(_verdict(diffs[("trapezoid", "schaefer_ii")], "below"))
    L.append(f"- PFM2 (wide: Schäfer (ii), Franks, PCHIP (ii) ceilings ≥ 0.85; trapezoid below Schäfer (ii)): "
             f"{_all(pfm2)} {pfm2}")
    v = _verdict(proj.get(("wide", "schaefer_ii", "minus", "cmaes_sch4")), "above")
    L.append(f"- PFM3 (wide: projection onto Schäfer (ii) ahead of CMA-ES on Schäfer (ii)): {v}")
    v = _verdict(proj.get(("fixed 12", "schaefer_i", "minus", "cmaes_sch4")), "above")
    L.append(f"- PFM4 (fixed 12: projection onto Schäfer (i) ahead of CMA-ES on Schäfer (i)): {v}")
    pfm5 = []
    for f in ("schaefer_ii", "franks", "pchip_ii"):
        t = proj.get(("wide", f, "minus", "psypref"))
        pfm5.append(None if t is None else ("held" if t[0] >= -0.05 else "failed"))
    L.append(f"- PFM5 (wide: projection within 5 points of PsyPref on the ProMP, paired estimate): "
             f"{_all(pfm5)} {pfm5}")
    pfm6 = [_verdict(diffs[("fixed", f)], "above") for f in ("schaefer_i", "trapezoid", "pchip")]
    L.append(f"- PFM6 (fixed 12: ProMP 8 ceiling above Schäfer (i), trapezoid, PCHIP): {_all(pfm6)} {pfm6}")
    numbers["verdicts"] = {"PFM1": _all(pfm1), "PFM2": _all(pfm2), "PFM5": _all(pfm5), "PFM6": _all(pfm6)}
    (d / "fam_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (d / "fam_numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("floors", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"floors": lambda: floors(args.out, args.workers), "run": lambda: run(args.out, args.workers),
     "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
