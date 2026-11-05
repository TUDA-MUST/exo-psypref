"""Test KS: more parameters at the same budget.

Usage:
  python scripts/icorr_amp_ks.py prep [--workers 6]
  python scripts/icorr_amp_ks.py floors [--workers 8]
  python scripts/icorr_amp_ks.py smoke [--workers 6]
  python scripts/icorr_amp_ks.py run [--workers 8]
  python scripts/icorr_amp_ks.py report

FULL's users with index 0-19 per fold and their answer
seeds, main condition (s = 1.0 Nm, lambda = 0.1), wide and fixed 12, each session FULL's run_one
with the job key ``n_basis`` (FULL's priors and spaces at 8, unchanged): PsyPref as LONG's (PsyPref-T's
factories, ties="band") at K = 12, 16 and 24 to 150 comparisons, and GP-BALD and CMA-ES on the ProMP
with population 4 (CMP's ``cmaes4``) at K = 16 to 40, on the same K-16 space and prior; 1,000
sessions. The priors are the training shapes projected onto K von Mises bases
(``AF.build_fold_parts``, ``fold-<name>-K<n>.npz``, as K12's). K = 8 is LONG's PsyPref rows and
FULL's GP-BALD / CMP's cmaes4 rows of the same users. Each session records when every answer was
given, the machine and the search dimension d, as LONG's. ``prep`` builds the K = 16 and 24 parts
and checks K = 12 against the stored parts R5 used; ``floors`` fits each user's reachable floor at
K = 12, 16, 24 (LONG's definition and starts) and checks the K = 8 refit against LONG's floors;
``smoke`` checks the defaults and K = 12 against the stored rows and times one user per arm.
``run`` starts only on a machine where smoke's checks passed. Outputs in <out>/icorr/amp_free/:
fold-<name>-K16/K24.npz, ks_sessions.jsonl and ks_floors.jsonl (both resumable), ks_smoke.jsonl,
amp_ks_report.md. FULL's, CMP's, ROB's and LONG's rows are read, never replaced.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_abl as ABL  # noqa: E402
import icorr_amp_full as F  # noqa: E402
# LONG's import wraps FULL's run_session, which then records when every answer was given
import icorr_amp_long as LONG  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.population.folds import loso_folds  # noqa: E402

#: Section 28: the settings, the users (index 0-19 per fold), the arms (candidate, K) and budgets.
SETTINGS = ("wide", "fixed 12")
N_USERS = LONG.N_USERS
N_LONG, N_COMP = LONG.N_LONG, F.N_COMP     # PsyPref 150, the comparators 40
TIES = LONG.TIES                           # PsyPref-T's tie handling, as LONG's
BASES = (12, 16, 24)                       # PsyPref's K; K = 8 is LONG's
K_CMP = 16                                 # the comparators' K
COMPARATORS = ("gp_bald", "cmaes4")
ARMS = tuple(("psypref", k) for k in BASES) + tuple((c, K_CMP) for c in COMPARATORS)
NAMES = {"psypref": "PsyPref", "gp_bald": "GP-BALD", "cmaes4": "CMA-ES (ProMP, population 4)"}
#: The reports' comparisons (k) and the share of the prior's variance for its directions.
KS_PSY, KS_CMP = (24, 40, 150), (24, 40)
JND = LONG.JND                             # 0.674 Nm
VAR_SHARE = 0.9
R5_COND = "R5 K=12"
#: The fields a session's result is compared on (R5's rows carry ROB's fields besides).
RESULT_FIELDS = ("distances", "d0", "identified", "shape_error", "guard_fired", "guard_final")


def _key(row):
    return row["candidate"], row["n_basis"], row["user"], row["setting"]


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _same(a, b) -> bool:
    """Bit for bit: dtype, shape and bytes."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ----------------------------------------------------------------------------- prep

def _prior_cov(ppath: str, fold_index: int, setting: str, n_basis: int) -> np.ndarray:
    """The covariance of the prior PsyPref whitens with: in the range the training shapes at AR's
    amplitudes, at the fixed level the unit training shapes (FULL's parts at K = 8)."""
    if setting in F.RANGES:
        return (R._a_moments(ppath, fold_index, setting) if n_basis == 8
                else R._ak_moments(ppath, fold_index, setting, n_basis))[1]
    p = R._fold(ppath, fold_index) if n_basis == 8 else R._parts_k(ppath, n_basis)
    return np.cov(p["w_minus"] + p["w_plus"], rowvar=False)


def _eff_dim(sigma: np.ndarray) -> tuple[float, int, float]:
    """The participation ratio (sum lambda)^2 / sum lambda^2 of a covariance, the number of
    directions that carry VAR_SHARE of its variance, and its smallest over largest eigenvalue."""
    ev = np.clip(np.linalg.eigvalsh(sigma)[::-1], 0.0, None)
    share = np.cumsum(ev) / ev.sum()
    return (float(ev.sum() ** 2 / np.sum(ev ** 2)), int(np.searchsorted(share, VAR_SHARE) + 1),
            float(ev[-1] / ev[0]))


def _eff_dims(out: Path) -> list[dict]:
    """Each fold's prior per setting and K: effective dimension, search dimension d and radius."""
    d, rows = F._dir(out), []
    for i, fold in enumerate(loso_folds()):
        ppath = str(d / f"fold-{fold.name}.npz")
        job = next(j for j in F.jobs(out) if j["fold_index"] == i and j["candidate"] == "psypref")
        for s in SETTINGS:
            for k in (8,) + BASES:
                pr, n90, cond = _eff_dim(_prior_cov(ppath, i, s, k))
                space, _ = _space_and_ideal(dict(job, setting=s, n_basis=k))
                rows.append(dict(fold=fold.name, setting=s, n_basis=k, pr=pr, n90=n90, cond=cond,
                                 d=int(space.d), radius=float(space.radius)))
    return rows


def prep(out: Path, workers: int) -> None:
    """The K = 16 and 24 parts (``AF.build_fold_parts``, skipped where present); K = 12 rebuilt
    into a temporary folder and compared with the stored parts R5 used; the K-generic helpers at
    K = 12 against R5's (``R._parts12``, ``R._a12_moments`` and its space); the priors' effective
    dimensions."""
    d, run_dir = F._dir(out), out / "phase5-schaefer"
    folds = loso_folds()
    ok = True
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(AF.build_fold_parts, i, run_dir, Path(tmp), 12): (i, 12, "check")
                for i in range(len(folds))}
        for k in (16, 24):
            for i, fold in enumerate(folds):
                if (d / f"fold-{fold.name}-K{k}.npz").exists():
                    print(f"exists: fold-{fold.name}-K{k}.npz", flush=True)
                else:
                    futs[pool.submit(AF.build_fold_parts, i, run_dir, d, k)] = (i, k, "build")
        t0 = time.time()
        for fut in as_completed(futs):
            i, k, what = futs[fut]
            path = fut.result()
            if what == "build":
                print(f"wrote {path} ({time.time() - t0:.0f} s)", flush=True)
                continue
            with np.load(path) as new, np.load(d / path.name) as old:   # closed: the folder is removed
                same = all(_same(new[x], old[x]) for x in ("w_minus", "w_plus"))
            ok &= same
            print(f"K = 12 rebuilt, {folds[i].name}: identical to the stored {path.name}: {same} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    rng = np.random.default_rng(AF.SEED_FULL + 28)
    for i, fold in enumerate(folds):
        ppath = str(d / f"fold-{fold.name}.npz")
        same = all(_same(R._parts_k(ppath, 12)[x], R._parts12(ppath)[x]) for x in ("w_minus", "w_plus"))
        for name in R.RANGES:
            same &= all(_same(a, b) for a, b in zip(R._ak_moments(ppath, i, name, 12),
                                                     R._a12_moments(ppath, i, name)))
        # FULL's K-basis space at K = 12 against R5's construction (ROB's run_one), on 20 points
        p12, (mu, sigma) = R._parts12(ppath), R._a12_moments(ppath, i, "wide")
        r5 = {"wide": AF.promp_space(mu, sigma, limit=F.RANGES["wide"][1], basis=AF.VonMisesBasis(12)),
              "fixed 12": AF.fixed_level_space(p12["w_minus"], p12["w_plus"], F.LEVEL,
                                               basis=AF.VonMisesBasis(12))}
        for s in SETTINGS:
            ks = F._k_space(ppath, i, s, 12)
            same &= ks.d == r5[s].d and ks.radius == r5[s].radius
            for z in rng.normal(size=(20, 12)):
                same &= _same(ks.design.curve(ks.to_theta(z)), r5[s].design.curve(r5[s].to_theta(z)))
        ok &= same
        print(f"{fold.name}: the K-generic helpers and space at K = 12 equal R5's bit for bit: {same}",
              flush=True)
    print(f"\nEffective dimension of each prior (participation ratio; directions for "
          f"{100 * VAR_SHARE:.0f} % of the variance; smallest / largest eigenvalue), search dimension d", flush=True)
    for r in _eff_dims(out):
        print(f"  {r['fold']} {r['setting']:8s} K {r['n_basis']:2d}: PR {r['pr']:.2f}, {r['n90']} directions, "
              f"eigenvalue ratio {r['cond']:.1e} | d {r['d']}, radius {r['radius']:.2f}", flush=True)
    if not ok:
        raise SystemExit("K = 12 does not reproduce what R5 used")


# ----------------------------------------------------------------------------- sessions

def jobs(out: Path) -> list[dict]:
    """FULL's PsyPref jobs in wide and fixed 12 (users with index 0-19 per fold, FULL's peaks and
    answer seeds), crossed with the arms; GP-BALD first, then PsyPref by falling K, CMA-ES last."""
    base = [j for j in F.jobs(out) if j["candidate"] == "psypref" and j["index"] < N_USERS
            and j["setting"] in SETTINGS]
    js = []
    for cand, k in ARMS:
        for j in base:
            if cand == "psypref":
                js.append(dict(j, n_basis=k, n_comp=N_LONG, ties=TIES))
            else:
                js.append(dict(j, candidate=cand, n_basis=k, n_comp=N_COMP))
    order = {("gp_bald", K_CMP): 0, ("psypref", 24): 1, ("psypref", 16): 2, ("psypref", 12): 3,
             ("cmaes4", K_CMP): 4}
    return sorted(js, key=lambda j: order[j["candidate"], j["n_basis"]])


def run_one(job: dict, out: str) -> dict:
    """FULL's run_one (the job's ``n_basis``) with LONG's record: when every answer was given,
    the session's seconds and the machine."""
    LONG._CAP.clear()
    t0 = time.perf_counter()
    row = F.run_one(dict(job), out)
    row["seconds"] = round(time.perf_counter() - t0, 3)
    row.update(LONG._CAP)
    row.update(LONG._machine())
    return row


def _smoke_passed(d: Path) -> bool:
    """Whether smoke's checks (the defaults reproduce FULL's and CMP's rows, K = 12 reproduces
    R5's) passed on this machine."""
    m = LONG._machine()
    return any(r["defaults_same"] and r["k12_same"] and all(r.get(k) == v for k, v in m.items())
               for r in _read(d / "ks_smoke.jsonl"))


def run(out: Path, workers: int) -> None:
    if not _smoke_passed(F._dir(out)):
        raise SystemExit(f"smoke's checks have not passed on {LONG._machine_name(LONG._machine())}; run "
                         f"smoke on this machine first (as LONG's, section 27: the stored rows reproduce "
                         f"bit for bit only on the machine that produced them)")
    path = F._dir(out) / "ks_sessions.jsonl"
    done = {_key(r) for r in _read(path) if r.get("error") is None}
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
            if n % 10 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- floors

def _space_and_ideal(job: dict):
    """The space PsyPref searches at the job's K and the preferred curve, as FULL's run_one builds
    them (K = 8: LONG's, FULL's own space)."""
    k = int(job.get("n_basis", 8))
    if k == 8:
        return LONG._space_and_ideal(job)
    z, u = F._users(job["_users"]), job["index"]
    ideal = AF.compose(job["amps"][0], job["amps"][1], z["user_minus"][u], z["user_plus"][u])
    return F._k_space(job["_parts"], job["fold_index"], job["setting"], k), ideal


def floor_one(job: dict, out: str) -> dict:
    """One user's reachable floor in one setting at the job's K: LONG's ``floor_one`` (section 27's
    target and starts, the same seed per user) on the K-basis space, the unconstrained refit from
    the same starts plus the in-ball optimum, and FAM's ProMP-K floor (AF.promp_floor) alongside.
    At K = 8 it is LONG's floor."""
    base = {k: v for k, v in job.items()
            if not k.startswith("_") and k not in ("candidate", "n_comp", "ties")}
    try:
        k = int(job.get("n_basis", 8))
        space, ideal = _space_and_ideal(job)
        inner, setting = space.design.inner, job["setting"]
        if setting in F.RANGES:
            theta_ls = inner.basis.project(ideal)
            kw = dict(peak_range=F.RANGES[setting])
        else:
            theta_ls = inner.project(inner.basis.project(ideal / F.LEVEL))
            kw = dict(level=F.LEVEL)
        z_ls = space.to_z(theta_ls)
        rng = np.random.default_rng([AF.SEED_FULL, 27, zlib.crc32(f"{job['user']}|{setting}".encode())])
        g = rng.normal(size=(LONG.N_DRAWS, space.d))
        u = rng.random(LONG.N_DRAWS)
        draws = g / np.linalg.norm(g, axis=1, keepdims=True) * (space.radius * u ** (1.0 / space.d))[:, None]
        starts = [space.clip_to_ball(z_ls), np.zeros(space.d)] + list(draws)

        def mse(z):
            return float(np.mean((space.design.curve(space.to_theta(z)) - ideal) ** 2))

        z_b, v_b, i_b = LONG._fit(mse, starts, space)
        z_f, v_f, i_f = LONG._fit(mse, [z_ls, z_b] + starts)
        return dict(base, n_basis=k, d=int(space.d), floor=float(np.sqrt(v_b)), z=[float(x) for x in z_b],
                    z_norm=float(np.linalg.norm(z_b)), start=i_b, radius=float(space.radius),
                    floor_ls=float(np.sqrt(mse(starts[0]))), z_ls_norm=float(np.linalg.norm(z_ls)),
                    floor_free=float(np.sqrt(v_f)), z_free=[float(x) for x in z_f],
                    z_free_norm=float(np.linalg.norm(z_f)), start_free=i_f,
                    outside=bool(np.linalg.norm(z_f) > space.radius + 1e-9),
                    fam_promp=float(AF.promp_floor(ideal, k, **kw)), error=None)
    except Exception as exc:  # one bad fit must not stop the run
        return dict(base, error=f"{type(exc).__name__}: {exc}")


def _floor_key(r):
    return r["n_basis"], r["user"], r["setting"]


def _floor_jobs(out: Path, bases=BASES) -> list[dict]:
    """One job per K, user and setting (the PsyPref jobs' users and peaks)."""
    users = [j for j in jobs(out) if j["candidate"] == "psypref" and j["n_basis"] == BASES[0]]
    return [dict(j, n_basis=k) for k in bases for j in users]


def floors(out: Path, workers: int) -> None:
    """The reachable floors at K = 12, 16, 24 (ks_floors.jsonl, resumable); then the K = 8 refit
    of every user-setting against LONG's long_floors.jsonl (not written)."""
    d = F._dir(out)
    path = d / "ks_floors.jsonl"
    done = {_floor_key(r) for r in _read(path) if r.get("error") is None}
    todo = [j for j in _floor_jobs(out) if _floor_key(j) not in done]
    print(f"{len(todo)} user-settings and K with floors to fit ({len(done)} done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        with path.open("a", encoding="utf-8") as f:
            futures = [pool.submit(floor_one, j, str(out)) for j in todo]
            for n, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                f.write(json.dumps(row) + "\n")
                f.flush()
                if row.get("error"):
                    print("ERROR", _floor_key(row), row["error"], flush=True)
                if n % 50 == 0:
                    print(f"{n}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
        long_fl = {(r["user"], r["setting"]): r for r in _read(d / "long_floors.jsonl") if r.get("error") is None}
        eight = _floor_jobs(out, (8,))
        res = list(pool.map(floor_one, eight, [str(out)] * len(eight), chunksize=8))
    fields = ("floor", "z", "floor_free", "z_free", "floor_ls", "start", "start_free")
    same = [r for r in res if r.get("error") is None and (r["user"], r["setting"]) in long_fl
            and all(r[x] == long_fl[r["user"], r["setting"]][x] for x in fields)
            and r["fam_promp"] == long_fl[r["user"], r["setting"]]["fam_promp8"]]
    print(f"K = 8 refit against LONG's floors: {len(same)}/{len(res)} identical in {', '.join(fields)} "
          f"and FAM's ProMP-8 floor ({time.time() - t0:.0f} s)", flush=True)
    fl = [r for r in _read(path) if r.get("error") is None]
    for k in BASES:
        for s in SETTINGS:
            fs = [r for r in fl if r["n_basis"] == k and r["setting"] == s]
            if fs:
                med = lambda x: float(np.median([r[x] for r in fs])) / JND   # noqa: E731
                print(f"  K {k:2d} {s:8s}: n {len(fs)} | median floor {med('floor'):.2f} JND, free {med('floor_free'):.2f}, "
                      f"FAM ProMP-{k} {med('fam_promp'):.2f} | free fit outside the ball "
                      f"{100 * np.mean([r['outside'] for r in fs]):.0f} %", flush=True)


# ----------------------------------------------------------------------------- smoke

def _reference(d: Path) -> dict:
    """K = 8 on the same users: LONG's PsyPref rows (main cell, 150 comparisons), FULL's GP-BALD
    rows and CMP's cmaes4 rows (40), keyed (candidate, user, setting)."""
    ref = {("psypref", r["user"], r["setting"]): r for r in _read(d / "long_sessions.jsonl")
           if r.get("error") is None and r["cell"] == "main" and r["setting"] in SETTINGS}
    for name, cand in (("full_sessions.jsonl", "gp_bald"), ("cmp_sessions.jsonl", "cmaes4")):
        ref.update({(cand, r["user"], r["setting"]): r for r in _read(d / name)
                    if r.get("error") is None and r["candidate"] == cand and r["setting"] in SETTINGS})
    return ref


def _reference_check(out: Path, ref: dict) -> dict:
    """Per (candidate, setting): how many of KS's users have a K = 8 row with the same answer seed
    and peaks (all three candidates of section 28's K = 8 values)."""
    n = {}
    for j in jobs(out):
        if j["n_basis"] != (BASES[0] if j["candidate"] == "psypref" else K_CMP):
            continue
        r = ref.get((j["candidate"], j["user"], j["setting"]))
        ok = r is not None and r["answer_seed"] == j["answer_seed"] and r["amps"] == j["amps"]
        t = n.setdefault((j["candidate"], j["setting"]), [0, 0])
        t[0] += ok
        t[1] += 1
    return n


def smoke(out: Path, workers: int) -> None:
    """(i) FULL's run_one with the defaults (no ``n_basis``) must reproduce FULL's PsyPref, GP-BALD
    and CMA-ES rows and CMP's cmaes4 rows bit for bit (2 users per setting); (i-b) at n_basis 12,
    otherwise FULL's defaults, it must reproduce ROB's R5 rows (the parts R5 used); (ii) one user
    per setting for each arm at its budget: seconds per comparison, and at K = 12 the first 40
    distances against R5's; (iii) floors and excess of (ii)'s users; the projected wall time at 8
    workers. Appends the machine and the results to ks_smoke.jsonl, which ``run`` checks."""
    d = F._dir(out)
    print(f"Machine: {LONG._machine_name(LONG._machine())}", flush=True)
    full = {(r["user"], r["setting"], r["candidate"]): r for name in ("full_sessions.jsonl", "cmp_sessions.jsonl")
            for r in _read(d / name) if r.get("error") is None}
    r5 = {(r["user"], r["setting"]): r for r in _read(d / "robust_sessions.jsonl")
          if r.get("error") is None and r["condition"] == R5_COND and r["candidate"] == "psypref"}
    ref = _reference(d)
    print("K = 8 rows of KS's users with the same answer seed and peaks:", flush=True)
    for (c, s), (a, b) in _reference_check(out, ref).items():
        print(f"  {NAMES[c]}, {s}: {a}/{b}", flush=True)
    base = [j for j in F.jobs(out) if j["candidate"] == "psypref" and j["fold_index"] == 0
            and j["index"] < 2 and j["setting"] in SETTINGS]
    plain = [dict(j, candidate=c, part="i") for c in ("gp_bald", "psypref", "cmaes", "cmaes4") for j in base]
    k12 = [dict(j, n_basis=12, part="i-b") for j in base]
    new = [dict(j, part="ii") for j in jobs(out) if j["fold_index"] == 0 and j["index"] == 0]
    todo = sorted(plain + k12 + new, key=lambda j: j["candidate"] != "gp_bald")   # the slow ones first
    print(f"{len(todo)} sessions, {workers} in parallel", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(run_one, [{k: v for k, v in j.items() if k != "part"} for j in todo],
                            [str(out)] * len(todo)))
    print(f"all sessions done in {(time.time() - t0) / 60:.1f} min", flush=True)
    for j, r in zip(todo, res):
        r["part"] = j["part"]

    ok, k12_ok, first40 = True, True, True
    print("(i) the defaults against FULL's and CMP's stored rows (every stored field)", flush=True)
    for r in (r for r in res if r["part"] == "i"):
        st = full.get((r["user"], r["setting"], r["candidate"]))
        same = r.get("error") is None and st is not None and all(r.get(k) == v for k, v in st.items())
        ok &= same
        print(f"  {r['candidate']:8s} {r['setting']:8s} {r['user']}: err {r.get('error')} | {len(r.get('distances', []))} "
              f"distances | identical {same} | {r['seconds']:.1f} s", flush=True)
    print("(i-b) n_basis 12, otherwise the defaults, against ROB's R5 rows", flush=True)
    for r in (r for r in res if r["part"] == "i-b"):
        st = r5.get((r["user"], r["setting"]))
        same = (r.get("error") is None and st is not None
                and all(r.get(k) == st[k] for k in RESULT_FIELDS if k in st))
        k12_ok &= same
        print(f"  psypref  {r['setting']:8s} {r['user']}: err {r.get('error')} | d {r.get('d')} | identical in "
              f"{', '.join(k for k in RESULT_FIELDS if st is not None and k in st)}: {same} | {r['seconds']:.1f} s",
              flush=True)
    print(f"(ii) one user per setting and arm (PsyPref {N_LONG} comparisons, ties={TIES!r}; comparators "
          f"{N_COMP})", flush=True)
    long_rows = [r for r in res if r["part"] == "ii"]
    for r in long_rows:
        if r.get("error"):
            first40 = False
            print(f"  {r['candidate']} K {r['n_basis']} {r['setting']}: ERROR {r['error']}", flush=True)
            continue
        n = len(r["distances"])
        extra = ""
        if r["candidate"] == "psypref" and r["n_basis"] == 12:
            st = r5.get((r["user"], r["setting"]))
            same = st is not None and r["distances"][:40] == st["distances"]
            first40 &= same
            extra = f" | first 40 == R5 {same}"
        last = "" if n == 40 else f", d_{n} {r['distances'][-1]:.3f}"
        print(f"  {r['candidate']:8s} K {r['n_basis']:2d} {r['setting']:8s}: d {r['d']} | {n} distances, d0 "
              f"{r['d0']:.3f}, d_24 {r['distances'][23]:.3f}, d_40 {r['distances'][39]:.3f}{last} | "
              f"{r['learner_cls']}, guard {r.get('guard_fired', '-')}, no-preference "
              f"{r['nopref']} | {r['seconds']:.1f} s | s per comparison near k 24 {LONG._sec_near(r, 24):.3f}, "
              f"near k {n} {LONG._sec_near(r, n):.3f}{extra}", flush=True)

    print("(iii) floors and excess of (ii)'s users (K = 8: LONG's rows)", flush=True)
    fjobs = [dict(j, n_basis=k) for k in (8,) + BASES for j in base if j["index"] == 0]
    with ProcessPoolExecutor(max_workers=min(workers, len(fjobs))) as pool:
        fl = {_floor_key(f): f for f in pool.map(floor_one, fjobs, [str(out)] * len(fjobs))}
    for s in SETTINGS:
        for k in (8,) + BASES:
            f = fl.get((k, base[0]["user"], s))
            r = (ref.get(("psypref", base[0]["user"], s)) if k == 8 else
                 next((x for x in long_rows if x["candidate"] == "psypref" and x["n_basis"] == k
                       and x["setting"] == s and not x.get("error")), None))
            if f is None or f.get("error") or r is None:
                print(f"  K {k} {s}: no floor or session ({None if f is None else f.get('error')})", flush=True)
                continue
            e = lambda kk: (r["distances"][kk - 1] - f["floor"]) / JND   # noqa: E731
            print(f"  K {k:2d} {s:8s}: floor {f['floor']:.3f} Nm (free {f['floor_free']:.3f}, FAM ProMP-{k} "
                  f"{f['fam_promp']:.3f}; |z| {f['z_norm']:.2f} of {f['radius']:.2f}) | d_40 / JND "
                  f"{r['distances'][39] / JND:.2f}, d_150 / JND {r['distances'][149] / JND:.2f} | E_40 {e(40):.2f}, "
                  f"E_150 {e(150):.2f}", flush=True)

    per = {(r["candidate"], r["n_basis"], r["setting"]): r["seconds"] for r in long_rows if not r.get("error")}
    n_per = len(jobs(out)) // (len(ARMS) * len(SETTINGS))
    psy = sum(n_per * v for (c, k, s), v in per.items() if c == "psypref")
    cmp_ = sum(n_per * v for (c, k, s), v in per.items() if c != "psypref")
    print(f"Projected ({n_per} users per arm and setting, the smoke's seconds at {workers} in parallel): PsyPref "
          f"{psy / 3600:.1f} h of sessions, comparators {cmp_ / 3600:.1f} h; at 8 workers {psy / 8 / 3600:.1f} h + "
          f"{cmp_ / 8 / 3600:.1f} h = {(psy + cmp_) / 8 / 3600:.1f} h", flush=True)
    for (c, k, s), v in sorted(per.items()):
        print(f"  {c:8s} K {k:2d} {s:8s}: {v:.1f} s per session -> {n_per * v / 8 / 3600:.2f} h at 8 workers", flush=True)
    with (d / "ks_smoke.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(LONG._machine(), defaults_same=bool(ok), k12_same=bool(k12_ok),
                                k12_first40_same=bool(first40), workers=workers,
                                seconds={f"{c} K{k} {s}": v for (c, k, s), v in per.items()},
                                projected_h_8_workers=dict(psypref=round(psy / 8 / 3600, 2),
                                                           comparators=round(cmp_ / 8 / 3600, 2)),
                                when=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
    if not (ok and k12_ok):
        raise SystemExit("the defaults or K = 12 did not reproduce the stored rows")


# ----------------------------------------------------------------------------- report

def _pairs(a: dict, b: dict) -> list[dict]:
    """Paired rows of two {user: row} cells, sorted by user (the bootstrap does not depend on
    the order in which sessions finished)."""
    return [dict(fold=a[u]["fold"], a=a[u], b=b[u]) for u in sorted(set(a) & set(b))]


def _dk(r, k) -> float:
    return r["distances"][k - 1] / JND


def report(out: Path) -> None:
    d = F._dir(out)
    raw = _read(d / "ks_sessions.jsonl")
    ok = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in ok]
    ref = _reference(d)
    fl = {_floor_key(r): r for r in _read(d / "ks_floors.jsonl") if r.get("error") is None}
    fl.update({(8, r["user"], r["setting"]): dict(r, fam_promp=r["fam_promp8"])
               for r in _read(d / "long_floors.jsonl") if r.get("error") is None and r["setting"] in SETTINGS})
    fam = {(int(r["family"][5:]), r["user"], r["setting"]): r["floor"] for r in _read(d / "fam_floors.jsonl")
           if r.get("error") is None and r["family"].startswith("promp")}
    r5 = {(r["user"], r["setting"]): r for r in _read(d / "robust_sessions.jsonl")
          if r.get("error") is None and r["condition"] == R5_COND and r["candidate"] == "psypref"}
    rng = np.random.default_rng(AF.SEED_FULL + 28)
    users = {(j["user"], j["setting"]): j for j in jobs(out) if j["candidate"] == "psypref" and j["n_basis"] == BASES[0]}

    def cell(cand, k, s) -> dict:
        """{user: row} of one arm at one K in one setting; K = 8 from the reference rows, only
        where they carry the KS job's answer seed and peaks."""
        if k == 8:
            return {u: r for (c, u, ss), r in ref.items() if c == cand and ss == s and (u, s) in users
                    and r["answer_seed"] == users[u, s]["answer_seed"] and r["amps"] == users[u, s]["amps"]}
        return {u: r for (c, kk, u, ss), r in ok.items() if c == cand and kk == k and ss == s}

    def boot(rows, stat):
        return R._boot(sorted(rows, key=lambda r: r["user"]), stat, rng)

    L = ["# Registration KS: more parameters at the same budget", "",
         f"Spec section 28 (84def93). {len(ok)} sessions, {len(errors)} errors. FULL's users with index 0-19 "
         f"per fold, main condition (s = {AF.S} Nm, λ = {AF.LAPSE}), wide and fixed 12. PsyPref as LONG's "
         f"(PsyPref-T's factories) at K = {', '.join(map(str, BASES))} to {N_LONG} comparisons; GP-BALD and "
         f"CMA-ES on the ProMP with population 4 at K = {K_CMP} to {N_COMP}, on the same K-{K_CMP} space and "
         f"prior. K = 8: LONG's PsyPref rows, FULL's GP-BALD rows and CMP's cmaes4 rows of the same users. "
         f"Distances d_k in JND ({JND:.3f} Nm); preference over generic assistance as FULL's. 95 % intervals "
         f"from {R.N_BOOT} draws, users resampled within folds.", ""]

    # Checks: machines, the K = 8 rows, K = 12 against R5
    smokes = _read(d / "ks_smoke.jsonl")
    passed = {LONG._machine_name(m) for m in smokes if m["defaults_same"] and m["k12_same"]}
    L += ["## Checks", ""]
    for name in sorted({LONG._machine_name(r) for r in ok.values()}):
        n = sum(LONG._machine_name(r) == name for r in ok.values())
        L.append(f"- {name}: {n} sessions; smoke's checks passed here: {name in passed}")
    for (c, s), (a, b) in _reference_check(out, ref).items():
        L.append(f"- K = 8 rows ({NAMES[c]}, {s}) with the KS job's answer seed and peaks: {a}/{b}")
    for s in SETTINGS:
        rs = cell("psypref", 12, s)
        same = sum(r["distances"][:40] == r5[u, s]["distances"] for u, r in rs.items() if (u, s) in r5)
        L.append(f"- PsyPref K = 12, {s}: first 40 distances equal R5's (K = 12, 40 comparisons) in {same}/{len(rs)}")
    nopref = sum(r.get("nopref", 0) for r in ok.values())
    L += [f"- \"No preference\" answers over all sessions: {nopref}.", ""]

    # The priors' effective dimension and the search dimension
    L += ["## The priors' effective dimension", "",
          f"Participation ratio (Σλ)² / Σλ² of the covariance PsyPref whitens with (wide: the training shapes at "
          f"AR's amplitudes; fixed 12: the unit training shapes), the directions that carry "
          f"{100 * VAR_SHARE:.0f} % of its variance, and the search dimension d with the ball's radius. "
          "Median over the five folds [range].", ""]
    try:
        eds = _eff_dims(out)
    except FileNotFoundError as exc:
        eds = []
        L += [f"Parts missing ({exc}); run prep.", ""]
    L += ["| setting | K | participation ratio | directions for 90 % | d | radius |", "|---|---|---|---|---|---|"]
    for s in SETTINGS:
        for k in (8,) + BASES:
            e = [r for r in eds if r["setting"] == s and r["n_basis"] == k]
            if e:
                pr, n90 = [r["pr"] for r in e], [r["n90"] for r in e]
                L.append(f"| {s} | {k} | {np.median(pr):.2f} [{min(pr):.2f}, {max(pr):.2f}] | {np.median(n90):.0f} "
                         f"[{min(n90)}, {max(n90)}] | {', '.join(str(x) for x in sorted({r['d'] for r in e}))} | "
                         f"{np.median([r['radius'] for r in e]):.2f} |")
    L.append("")

    # PsyPref: distance to the preferred curve, paired against K = 8
    res = {}
    L += ["## PsyPref: distance to the preferred curve (d_k / JND)", "",
          "Median d_k, and the paired mean difference against K = 8 on the same users (negative: closer).", "",
          "| setting | K | n | median d_24 | median d_40 | median d_150 | Δ d_24 vs K = 8 | Δ d_40 vs K = 8 | "
          "Δ d_150 vs K = 8 |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        c8 = cell("psypref", 8, s)
        for k in (8,) + BASES:
            ck = cell("psypref", k, s)
            if len(ck) < 8:
                continue
            meds = [R._f(boot(list(ck.values()), lambda x, kk=kk: float(np.median([_dk(r, kk) for r in x]))))
                    for kk in KS_PSY]
            diffs = []
            for kk in KS_PSY:
                ps = _pairs(ck, c8)
                if k == 8 or len(ps) < 8:
                    diffs.append("–")
                    continue
                t = R._boot(ps, lambda x, kk=kk: float(np.mean([_dk(p["a"], kk) - _dk(p["b"], kk) for p in x])), rng)
                res["d", s, k, kk] = t
                diffs.append(R._f(t))
            L.append(f"| {s} | {k} | {len(ck)} | " + " | ".join(meds + diffs) + " |")
    L.append("")
    L += ["Paired mean difference K = 24 − K = 16 in d_150 (JND):", ""]
    for s in SETTINGS:
        ps = _pairs(cell("psypref", 24, s), cell("psypref", 16, s))
        if len(ps) >= 8:
            res["d24-16", s] = R._boot(ps, lambda x: float(np.mean([_dk(p["a"], 150) - _dk(p["b"], 150) for p in x])), rng)
            L.append(f"- {s}: {R._f(res['d24-16', s])}")
    L.append("")

    # PsyPref: preference over generic assistance, paired against K = 8
    L += ["## PsyPref: preference over generic assistance", "",
          "Mean preference (s = 1, λ = 0.1), and the paired difference against K = 8 in points.", "",
          "| setting | K | n | k = 24 | k = 40 | Δ k = 24 vs K = 8 | Δ k = 40 vs K = 8 |",
          "|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        c8 = cell("psypref", 8, s)
        for k in (8,) + BASES:
            ck = cell("psypref", k, s)
            if len(ck) < 8:
                continue
            prefs = [R._f(boot(list(ck.values()), lambda x, kk=kk: F._pref_k(x, kk)), 3) for kk in (24, 40)]
            diffs = []
            for kk in (24, 40):
                ps = _pairs(ck, c8)
                if k == 8 or len(ps) < 8:
                    diffs.append("–")
                    continue
                t = R._boot(ps, lambda x, kk=kk: 100 * (F._pref_k([p["a"] for p in x], kk)
                                                         - F._pref_k([p["b"] for p in x], kk)), rng)
                res["pref", s, k, kk] = t
                diffs.append(R._f(t, 1))
            L.append(f"| {s} | {k} | {len(ck)} | " + " | ".join(prefs + diffs) + " |")
    L.append("")

    # Floors and the excess over them
    L += ["## Reachable floors and the excess over them", "",
          "Floor f: LONG's definition and starts (the smallest RMS distance over PsyPref's search ball under "
          "the device limits) on each K's space; K = 8 is LONG's. Medians over users in JND; excess "
          "E_k = (d_k − f) / JND.", "",
          "| setting | K | n | reachable f | unconstrained refit | FAM's ProMP-K floor | the ball's cost (f − refit) | "
          "unconstrained fit outside the ball | median E_24 | median E_40 | median E_150 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    chk = []
    for s in SETTINGS:
        for k in (8,) + BASES:
            fs = [f for (kk, u, ss), f in fl.items() if kk == k and ss == s and (u, s) in users]
            if not fs:
                continue
            chk += [abs(f["fam_promp"] - fam[k, f["user"], s]) for f in fs if (k, f["user"], s) in fam]
            med = lambda x: float(np.median([f[x] for f in fs])) / JND   # noqa: E731
            cost = np.array([f["floor"] - f["floor_free"] for f in fs]) / JND
            ck = cell("psypref", k, s)
            xs = [dict(r, E=(np.asarray(r["distances"], dtype=float) - fl[k, u, s]["floor"]) / JND)
                  for u, r in ck.items() if (k, u, s) in fl]
            ex = (["–"] * len(KS_PSY) if len(xs) < 8 else
                  [R._f(boot(xs, lambda x, kk=kk: float(np.median([p["E"][kk - 1] for p in x])))) for kk in KS_PSY])
            L.append(f"| {s} | {k} | {len(fs)} | {med('floor'):.2f} | {med('floor_free'):.2f} | {med('fam_promp'):.2f} | "
                     f"{np.median(cost):.2f} | {100 * np.mean([f['outside'] for f in fs]):.0f} % | " + " | ".join(ex) + " |")
    if chk:
        L += ["", f"FAM's stored ProMP-K floors recomputed: largest difference {max(chk):.2e} Nm over {len(chk)} "
                  f"user-settings and K."]
    L.append("")

    # The comparators at K = 16
    L += [f"## The comparators at K = {K_CMP}", "",
          f"PsyPref (K = {K_CMP}) minus each comparator (K = {K_CMP}) in preference over generic assistance, "
          f"and each comparator's change from K = 8 to K = {K_CMP}, points, paired on the same users.", "",
          "| setting | comparator | n | preference K = 8, k = 24 / 40 | preference K = 16, k = 24 / 40 | "
          "PsyPref − comparator, k = 24 | PsyPref − comparator, k = 40 | change K = 8 → 16, k = 24 | "
          "change K = 8 → 16, k = 40 |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        psy = cell("psypref", K_CMP, s)
        for c in COMPARATORS:
            c16, c8 = cell(c, K_CMP, s), cell(c, 8, s)
            if len(c16) < 8:
                continue
            p8 = " / ".join(f"{F._pref_k(list(c8.values()), kk):.3f}" for kk in (24, 40)) if c8 else "–"
            p16 = " / ".join(f"{F._pref_k(list(c16.values()), kk):.3f}" for kk in (24, 40))
            cols = []
            for a, b, tag in ((psy, c16, "ahead"), (c16, c8, "change")):
                for kk in (24, 40):
                    ps = _pairs(a, b)
                    if len(ps) < 8:
                        cols.append("–")
                        continue
                    t = R._boot(ps, lambda x, kk=kk: 100 * (F._pref_k([p["a"] for p in x], kk)
                                                             - F._pref_k([p["b"] for p in x], kk)), rng)
                    res[tag, s, c, kk] = t
                    cols.append(R._f(t, 1))
            L.append(f"| {s} | {NAMES[c]} | {len(c16)} | {p8} | {p16} | " + " | ".join(cols) + " |")
    L.append("")

    # Seconds per comparison
    hosts = sorted({r.get("host", "?") for r in list(ok.values()) + [r for r in ref.values() if r.get("stamps")]})
    L += ["## Seconds per comparison (median over sessions; mean of the 5 comparisons around k)", "",
          "| arm | K | setting | machine | n | near k = 24 | near the last comparison | session (min) |",
          "|---|---|---|---|---|---|---|---|"]
    for cand, k in (("psypref", 8),) + ARMS:
        for s in SETTINGS:
            for h in hosts:
                rh = [r for r in cell(cand, k, s).values() if r.get("stamps") and r.get("host", "?") == h]
                if rh:
                    last = len(rh[0]["distances"])
                    L.append(f"| {NAMES[cand]} | {k} | {s} | {h} | {len(rh)} | "
                             f"{np.median([LONG._sec_near(r, 24) for r in rh]):.3f} | k = {last}: "
                             f"{np.median([LONG._sec_near(r, last) for r in rh]):.3f} | "
                             f"{np.median([r['seconds'] for r in rh]) / 60:.1f} |")
    L += ["", "K = 8's comparator rows (FULL, CMP) carry no answer times.", ""]

    # Predictions
    def v(t, side, bound=0.0):
        return None if t is None else ABL._verdict(t, side, bound)

    def both(*xs):
        xs = list(xs)
        return (None if any(x is None for x in xs) else "held" if all(x == "held" for x in xs)
                else "failed" if "failed" in xs else "uncertain")

    pks = {}
    for s in SETTINGS:
        t1 = res.get(("pref", s, K_CMP, 24))
        pks["PKS1", s] = None if t1 is None else ("held" if t1[0] >= -2.0 else "failed")
        pks["PKS2", s] = v(res.get(("d", s, K_CMP, 40)), "below")
        pks["PKS3", s] = both(v(res.get(("d", s, K_CMP, 150)), "below"), v(res.get(("d24-16", s)), "below"))
        pks["PKS4", s] = both(v(res.get(("ahead", s, "gp_bald", 24)), "above"),
                              v(res.get(("ahead", s, "cmaes4", 24)), "above"))

    def show(name):
        return ", ".join(f"{s} {pks[name, s]}" for s in SETTINGS)

    def pt(key, d=2):
        t = res.get(key)
        return "–" if t is None else R._f(t, d)

    L += ["## Predictions (each setting)", "",
          f"- PKS1 (k = 24: PsyPref K = 16 − K = 8 in preference, paired point estimate ≥ −2 points): {show('PKS1')} — "
          + ", ".join(f"{s} {pt(('pref', s, K_CMP, 24), 1)}" for s in SETTINGS),
          f"- PKS2 (k = 40: d_40 K = 16 − K = 8, paired interval below 0): {show('PKS2')} — "
          + ", ".join(f"{s} {pt(('d', s, K_CMP, 40))}" for s in SETTINGS),
          f"- PKS3 (k = 150: d_150 K = 16 − K = 8 and K = 24 − K = 16, paired intervals below 0): {show('PKS3')} — "
          + ", ".join(f"{s} {pt(('d', s, K_CMP, 150))} and {pt(('d24-16', s))}" for s in SETTINGS),
          f"- PKS4 (K = 16, k = 24: PsyPref ahead of GP-BALD and of CMA-ES (population 4), paired intervals "
          f"above 0): {show('PKS4')} — "
          + ", ".join(f"{s} {pt(('ahead', s, 'gp_bald', 24), 1)} / {pt(('ahead', s, 'cmaes4', 24), 1)}"
                      for s in SETTINGS)]
    gate = [pks[p, s] for p in ("PKS1", "PKS4") for s in SETTINGS]
    L += ["", "**Wording (section 28):** "
          + ("not decided: PKS1 or PKS4 lacks rows in a setting." if None in gate else
             "PKS1 and PKS4 hold in both settings — the paper may write that PsyPref uses 16 weights at the "
             "same budget." if all(x == "held" for x in gate) else
             "PKS1 and PKS4 do not both hold in both settings — the paper reports the trade-off between floor "
             "and learning speed with the numbers."), ""]
    if errors:
        L += ["## Errors", ""] + [f"- {_key(e)}: {e['error']}" for e in errors[:30]]
    (d / "amp_ks_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("prep", "floors", "smoke", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"prep": lambda: prep(args.out, args.workers), "floors": lambda: floors(args.out, args.workers),
     "smoke": lambda: smoke(args.out, args.workers), "run": lambda: run(args.out, args.workers),
     "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
