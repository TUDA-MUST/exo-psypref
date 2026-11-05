"""Test LONG: does PsyPref approach the best reachable profile?

Usage:
  python scripts/icorr_amp_long.py smoke
  python scripts/icorr_amp_long.py best-profile distances [--workers 8]
  python scripts/icorr_amp_long.py run [--workers 8]
  python scripts/icorr_amp_long.py report

PsyPref as in FULL (PsyPref-M with N1 and the two-part
guard in the ranges, B_fixed at 12 Nm), built through PsyPref-T's factories (ties="band", section
26a), to 150 comparisons on FULL's users with index 0-19 per fold and their answer seeds: the main
condition in medium, wide and fixed 12 (FULL's run_one, cell "main") and ROB's R2, s = 2.0 Nm, in
wide and fixed 12 (ROB's run_one, cell "R2"); 500 sessions. Each session also records when every
answer was given and the machine it ran on (host, CPU, numpy, OpenBLAS kernel). ``best-profile distances`` fits each
user's reachable best-profile distance: the smallest RMS distance to the preferred curve over PsyPref's search search region
under the setting's device limits (R2 reuses the main cell's: same preferred curves). ``smoke``
checks the defaults against the stored rows and times a few long sessions. The stored distances
reproduce bit for bit only on the machine that produced them (section 27), so ``smoke`` logs its
machine and ``run`` starts only on a machine where smoke's defaults reproduced the stored rows.
Outputs in <out>/icorr/amp_free/: long_sessions.jsonl and long_floors.jsonl (both resumable),
long_smoke.jsonl, amp_long_report.md. FULL's and ROB's rows are read, never replaced.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
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
import icorr_amp_range as R  # noqa: E402
import icorr_amp_robust as ROB  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402

#: Section 27: the cells, their settings, the users (index 0-19 per fold) and the budget.
CELLS = {"main": F.SETTINGS, "R2": ROB.SETTINGS}
R2_COND = "R2 s=2.0"
N_USERS = 20
N_LONG = 150
TIES = "band"            # PsyPref-T's tie handling (26a); no LONG user answers "no preference"
KS = (24, 40, 80, 150)
CURVE_KS = (1, 6, 12, 24, 40, 60, 80, 100, 120, 150)
JND = true_jnd(AF.S)     # 0.674 Nm
GOOD = 0.5               # JND: "within half a JND of the best-profile distance"
#: The best-profile distance fit: 20 uniform starts in the search region, SLSQP's settings.
N_DRAWS = 20
FIT_ITER, FIT_TOL = 500, 1e-10


def _key(row):
    return row["cell"], row["user"], row["setting"]


def _machine() -> dict:
    """The machine a session runs on: host, CPU, numpy's version and OpenBLAS's kernel (None
    without threadpoolctl). Section 27's bit-for-bit reproduction holds only where these match the
    machine that produced the stored rows."""
    blas = None
    try:
        from threadpoolctl import threadpool_info
        blas = next((i.get("architecture") for i in threadpool_info() if i.get("user_api") == "blas"), None)
    except Exception:  # the record must not stop a session
        pass
    return dict(host=platform.node(), cpu=platform.processor(), numpy=np.__version__, blas=blas)


def _machine_name(r) -> str:
    return f"{r.get('host', '?')} ({r.get('cpu', '?')}; numpy {r.get('numpy', '?')}, OpenBLAS {r.get('blas', '?')})"


# ----------------------------------------------------------------------------- per-session timing
_CAP: dict = {}


def _wrap(orig):
    """FULL's and ROB's run_session, recording when every answer was given (seconds since the
    session began, 3 decimals), the session's own seconds, the learner's class and its "no
    preference" answers. Feeds nothing back."""
    def _rs(design, learner, observer, rng, *a, **k):
        stamps, t0 = [], time.perf_counter()
        respond0 = observer.respond

        def respond(ca, cb, r, order=1):
            resp = respond0(ca, cb, r, order=order)
            stamps.append(round(time.perf_counter() - t0, 3))
            return resp

        observer.respond = respond
        rec = orig(design, learner, observer, rng, *a, **k)
        _CAP.update(stamps=stamps, session_seconds=round(time.perf_counter() - t0, 3),
                    learner_cls=type(getattr(learner, "learner", learner)).__name__,
                    nopref=int(sum(x == 2 for x in rec["answers"])))
        return rec
    return _rs


for _mod in (F, ROB):   # module top level: spawned workers re-run it
    if not getattr(_mod.run_session, "_long_wrapped", False):
        _mod.run_session = _wrap(_mod.run_session)
        _mod.run_session._long_wrapped = True


def run_one(job: dict, out: str) -> dict:
    _CAP.clear()
    t0 = time.perf_counter()
    row = (F if job["cell"] == "main" else ROB).run_one(dict(job), out)
    row["seconds"] = round(time.perf_counter() - t0, 3)
    row.update(_CAP)
    row.update(_machine())
    return row


def jobs(out: Path) -> list[dict]:
    """FULL's PsyPref jobs (cell "main") and ROB's R2 PsyPref jobs (cell "R2"), users with index
    0-19 per fold, at 150 comparisons with PsyPref-T's tie handling."""
    main = [dict(j, cell="main") for j in F.jobs(out)
            if j["candidate"] == "psypref" and j["index"] < N_USERS]
    r2 = [dict(j, cell="R2") for j in ROB.jobs(out)
          if j["condition"] == R2_COND and j["candidate"] == "psypref" and j["index"] < N_USERS]
    return [dict(j, n_comp=N_LONG, ties=TIES) for j in main + r2]


def _smoke_passed(d: Path) -> bool:
    """Whether smoke's part (i) (the defaults reproduce the stored rows) passed on this machine."""
    path, m = d / "long_smoke.jsonl", _machine()
    return path.exists() and any(
        r["defaults_same"] and all(r.get(k) == v for k, v in m.items())
        for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()))


def run(out: Path, workers: int) -> None:
    if not _smoke_passed(F._dir(out)):
        raise SystemExit(f"smoke's defaults have not reproduced the stored rows on {_machine_name(_machine())}; "
                         f"run smoke on this machine first (section 27: bit for bit only on the machine "
                         f"that produced them)")
    path = F._dir(out) / "long_sessions.jsonl"
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
            if n % 10 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- best-profile distances

def _space_and_ideal(job: dict):
    """The space PsyPref searches and the preferred curve, exactly as FULL's run_one builds them."""
    z, d = F._users(job["_users"]), R._fold(job["_parts"], job["fold_index"])
    u, setting = job["index"], job["setting"]
    ideal = AF.compose(job["amps"][0], job["amps"][1], z["user_minus"][u], z["user_plus"][u])
    if setting in F.RANGES:
        mu, sigma = R._a_moments(job["_parts"], job["fold_index"], setting)
        space = AF.promp_space(mu, sigma, limit=F.RANGES[setting][1])
    else:
        space = AF.fixed_level_space(d["w_minus"], d["w_plus"], F.LEVEL)
    return space, ideal


def _fit(f, starts, space=None):
    """The best of SLSQP runs on ``f`` (mean squared error in z) from ``starts``, inside
    ``space``'s search search region when it is given (None: unconstrained). Each start and each end point
    counts, end points pulled back onto the search region. Returns (z, f(z), index of the winning start)."""
    from scipy.optimize import minimize

    cons, clip = (), (lambda z: z)
    if space is not None:
        r2, clip = space.radius ** 2, space.clip_to_search_region
        cons = ({"type": "ineq", "fun": lambda z: r2 - z @ z, "jac": lambda z: -2.0 * z},)
    best = (None, np.inf, -1)
    for i, z0 in enumerate(starts):
        res = minimize(f, z0, method="SLSQP", constraints=cons,
                       options=dict(maxiter=FIT_ITER, ftol=FIT_TOL))
        for z in (z0, res.x):
            z = np.asarray(clip(z), dtype=float)
            v = f(z)
            if v < best[1]:
                best = (z, v, i)
    return best


def floor_one(job: dict, out: str) -> dict:
    """One user's reachable best-profile distance in one setting (section 27's target), the unconstrained refit
    from the same starts and FAM's ProMP-8 best-profile distance (AF.promp_floor) alongside."""
    base = {k: v for k, v in job.items()
            if not k.startswith("_") and k not in ("candidate", "cell", "n_comp", "ties")}
    try:
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
        g = rng.normal(size=(N_DRAWS, space.d))
        u = rng.random(N_DRAWS)
        draws = g / np.linalg.norm(g, axis=1, keepdims=True) * (space.radius * u ** (1.0 / space.d))[:, None]
        starts = [space.clip_to_search_region(z_ls), np.zeros(space.d)] + list(draws)

        def mse(z):
            return float(np.mean((space.design.curve(space.to_theta(z)) - ideal) ** 2))

        z_b, v_b, i_b = _fit(mse, starts, space)
        # Without the search region the least-squares start needs no clipping; it and the in-search region optimum are
        # added to the same starts, so the free best-profile distance is never above the reachable one
        z_f, v_f, i_f = _fit(mse, [z_ls, z_b] + starts)
        return dict(base, floor=float(np.sqrt(v_b)), z=[float(x) for x in z_b],
                    z_norm=float(np.linalg.norm(z_b)), start=i_b, radius=float(space.radius),
                    floor_ls=float(np.sqrt(mse(starts[0]))), z_ls_norm=float(np.linalg.norm(z_ls)),
                    floor_free=float(np.sqrt(v_f)), z_free=[float(x) for x in z_f],
                    z_free_norm=float(np.linalg.norm(z_f)), start_free=i_f,
                    outside=bool(np.linalg.norm(z_f) > space.radius + 1e-9),
                    fam_promp8=float(AF.promp_floor(ideal, 8, **kw)), error=None)
    except Exception as exc:  # one bad fit must not stop the run
        return dict(base, error=f"{type(exc).__name__}: {exc}")


def _floor_key(r):
    return r["user"], r["setting"]


def floors(out: Path, workers: int) -> None:
    path = F._dir(out) / "long_floors.jsonl"
    done = set()
    if path.exists():
        done = {_floor_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = [j for j in jobs(out) if j["cell"] == "main" and _floor_key(j) not in done]
    print(f"{len(todo)} user-settings with floors to fit ({len(done)} done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(floor_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", _floor_key(row), row["error"], flush=True)
            if n % 50 == 0:
                print(f"{n}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)


# ----------------------------------------------------------------------------- smoke

def _stored(d: Path) -> dict:
    """FULL's and ROB R2's stored PsyPref rows, per cell, keyed (user, setting)."""
    full = {(r["user"], r["setting"]): r for r in
            map(json.loads, (d / "full_sessions.jsonl").read_text(encoding="utf-8").splitlines())
            if r.get("error") is None and r["candidate"] == "psypref"}
    rob = {(r["user"], r["setting"]): r for r in
           map(json.loads, (d / "robust_sessions.jsonl").read_text(encoding="utf-8").splitlines())
           if r.get("error") is None and r["condition"] == R2_COND and r["candidate"] == "psypref"}
    return {"main": full, "R2": rob}


def _durations(r) -> np.ndarray:
    """Seconds from each answer to the next; the session's end is left out, since its last gap
    has no pair search."""
    return np.diff(np.array(r["stamps"], dtype=float))


def _sec_near(r, k: int, w: int = 5) -> float:
    """Mean seconds per comparison over the ``w`` comparisons around comparison ``k``."""
    dur = _durations(r)
    start = min(max(k - 1 - w // 2, 0), len(dur) - w)
    return float(np.mean(dur[start:start + w]))


def smoke(out: Path) -> None:
    """(i) FULL's and ROB's run_one with the jobs' defaults must reproduce the stored rows bit for
    bit; (ii) the same jobs at 150 comparisons with PsyPref-T: the first 40 distances, seconds per
    comparison; (iii) their best-profile distances and E_150; the projected wall time of the run. Appends the
    machine and the results of (i) and (ii) to long_smoke.jsonl, which ``run`` checks."""
    d = F._dir(out)
    stored = _stored(d)
    pick = [j for j in jobs(out) if j["fold_index"] == 0 and j["index"] == 0]
    plain = [{k: v for k, v in j.items() if k not in ("n_comp", "ties")} for j in pick]
    ok, first40 = True, True
    print(f"Machine: {_machine_name(_machine())}", flush=True)
    print(f"(i) defaults, {len(plain)} sessions in parallel, against the stored rows", flush=True)
    with ProcessPoolExecutor(max_workers=len(plain)) as pool:
        res = list(pool.map(run_one, plain, [str(out)] * len(plain)))
    for r in res:
        st = stored[r["cell"]].get((r["user"], r["setting"]))
        same = r.get("error") is None and st is not None and r["distances"] == st["distances"]
        ok &= same
        print(f"  {r['cell']:4s} {r['setting']:8s} {r['user']}: err {r.get('error')} | "
              f"{len(r.get('distances', []))} distances == stored {same} | {r.get('learner_cls')} | "
              f"{r['seconds']:.1f} s", flush=True)
    print(f"(ii) {N_LONG} comparisons, ties={TIES!r}, {len(pick)} sessions in parallel", flush=True)
    with ProcessPoolExecutor(max_workers=len(pick)) as pool:
        long_rows = list(pool.map(run_one, [dict(j) for j in pick], [str(out)] * len(pick)))
    for r in long_rows:
        if r.get("error"):
            first40 = False
            print(f"  {r['cell']} {r['setting']} {r['user']}: ERROR {r['error']}", flush=True)
            continue
        st = stored[r["cell"]].get((r["user"], r["setting"]))
        same = st is not None and r["distances"][:40] == st["distances"]
        first40 &= same
        print(f"  {r['cell']:4s} {r['setting']:8s} {r['user']}: {len(r['distances'])} distances, first 40 "
              f"== stored {same} | {r['learner_cls']}, no-preference answers {r['nopref']} | "
              f"{r['seconds']:.1f} s | s per comparison near k 24 {_sec_near(r, 24):.3f}, near k 150 "
              f"{_sec_near(r, 150):.3f}", flush=True)
    print("(iii) floors and excess", flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        fl = {_floor_key(f): f for f in pool.map(floor_one, [dict(j) for j in pick if j["cell"] == "main"],
                                                  [str(out)] * 3)}
    for r in long_rows:
        f = fl.get((r["user"], r["setting"]))
        if r.get("error") or f is None or f.get("error"):
            print(f"  {r['cell']} {r['setting']}: no floor ({None if f is None else f.get('error')})")
            continue
        e = lambda k: (r["distances"][k - 1] - f["floor"]) / JND   # noqa: E731
        print(f"  {r['cell']:4s} {r['setting']:8s}: floor {f['floor']:.3f} Nm (free {f['floor_free']:.3f}, "
              f"FAM ProMP-8 {f['fam_promp8']:.3f}; |z| {f['z_norm']:.2f} of {f['radius']:.2f}, free "
              f"{f['z_free_norm']:.2f}) | d0 {r['d0']:.3f} | d_40 {r['distances'][39]:.3f}, d_150 "
              f"{r['distances'][-1]:.3f} | E_40 {e(40):.2f}, E_150 {e(150):.2f} JND", flush=True)
    per = {(r["cell"], r["setting"]): r["seconds"] for r in long_rows if not r.get("error")}
    total = sum(100 * s for s in per.values())
    print(f"Projected: {total / 3600:.1f} h of sessions (100 per cell and setting, the smoke's "
          f"seconds), {total / 8 / 3600:.1f} h at 8 workers", flush=True)
    with (d / "long_smoke.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(_machine(), defaults_same=bool(ok), first40_same=bool(first40),
                                seconds={f"{c} {s}": v for (c, s), v in per.items()},
                                when=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
    if not ok:
        raise SystemExit("the defaults did not reproduce the stored rows")


# ----------------------------------------------------------------------------- report

def _rows(rs, fl):
    """Sessions with a best-profile distance, each with its excess E (JND) over k."""
    out = []
    for r in rs:
        f = fl.get((r["user"], r["setting"]))
        if f is not None:
            out.append(dict(r, E=(np.asarray(r["distances"], dtype=float) - f["floor"]) / JND))
    return out


def report(out: Path) -> None:
    d = F._dir(out)
    raw = [json.loads(x) for x in (d / "long_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in ok]
    fl = {_floor_key(r): r for r in
          map(json.loads, (d / "long_floors.jsonl").read_text(encoding="utf-8").splitlines())
          if r.get("error") is None}
    fam = {(r["user"], r["setting"]): r["floor"] for r in
           map(json.loads, (d / "fam_floors.jsonl").read_text(encoding="utf-8").splitlines())
           if r.get("error") is None and r["family"] == "promp8"}
    stored = _stored(d)
    rng = np.random.default_rng(AF.SEED_FULL + 27)
    # sorted by user, so the bootstrap does not depend on the order in which sessions finished
    cell_rows = {(c, s): sorted((r for (cc, u, ss), r in ok.items() if cc == c and ss == s),
                                key=lambda r: r["user"])
                 for c, settings in CELLS.items() for s in settings}
    L = ["# Test plan LONG: does PsyPref approach the best reachable profile?", "",
         f"Design note 27 (294859a). {len(ok)} sessions, {len(errors)} errors. FULL's users with index "
         f"0-19 per fold; PsyPref as in FULL with PsyPref-T's tie handling, {N_LONG} comparisons. "
         f"Excess E_k = (d_k − f) / {JND:.3f} Nm (JND), f the user's reachable floor in PsyPref's "
         f"search ball. 95 % intervals from {R.N_BOOT} draws, users resampled within folds.", ""]

    # The machines: the stored distances reproduce only on the machine that produced them (27)
    hosts = sorted({r.get("host", "?") for r in ok.values()})
    smoke_path = d / "long_smoke.jsonl"
    smokes = ([json.loads(x) for x in smoke_path.read_text(encoding="utf-8").splitlines()]
              if smoke_path.exists() else [])
    passed = {_machine_name(m) for m in smokes if m["defaults_same"]}
    L += ["## Machines", ""]
    for name in sorted({_machine_name(r) for r in ok.values()}):
        n = sum(_machine_name(r) == name for r in ok.values())
        L.append(f"- {name}: {n} sessions; smoke's defaults reproduced the stored rows here: {name in passed}")
    L.append("")

    # PLG4: reproduction of the stored first 40 distances
    L += ["## Reproduction of the stored rows (first 40 comparisons)", ""]
    same_all = True
    for (c, s), rs in cell_rows.items():
        st = stored[c]
        same = [r for r in rs if (r["user"], r["setting"]) in st
                and r["distances"][:40] == st[(r["user"], r["setting"])]["distances"]]
        same_all &= bool(rs) and len(same) == len(rs)
        per_host = "" if len(hosts) < 2 else " (" + ", ".join(
            f"{h} {sum(r.get('host', '?') == h for r in same)}/{sum(r.get('host', '?') == h for r in rs)}"
            for h in hosts) + ")"
        L.append(f"- {c}, {s}: {len(same)}/{len(rs)} identical{per_host}")
    nopref = sum(r.get("nopref", 0) for r in ok.values())
    L += ["", f"\"No preference\" answers over all sessions: {nopref}. Learners: "
          + ", ".join(sorted({r.get("learner_cls", "?") for r in ok.values()})) + ".", ""]

    # Best-profile distances
    L += ["## Floors (JND, median over users)", "",
          "| setting | n | reachable f | unconstrained refit | FAM's ProMP-8 floor | the ball's cost "
          "(f − refit) | unconstrained fit outside the ball | the ball costs > 0.05 JND | radius |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in F.SETTINGS:
        fs = [f for (u, ss), f in fl.items() if ss == s]
        if not fs:
            continue
        med = lambda k: float(np.median([f[k] for f in fs])) / JND   # noqa: E731
        cost = np.array([f["floor"] - f["floor_free"] for f in fs]) / JND
        L.append(f"| {s} | {len(fs)} | {med('floor'):.2f} | {med('floor_free'):.2f} | {med('fam_promp8'):.2f} | "
                 f"{np.median(cost):.2f} | {100 * np.mean([f['outside'] for f in fs]):.0f} % | "
                 f"{100 * np.mean(cost > 0.05):.0f} % | {fs[0]['radius']:.2f} |")
    chk = [abs(f["fam_promp8"] - fam[k]) for k, f in fl.items() if k in fam]
    L += ["", "At the fixed level the curve does not change when the shape weights are scaled, so an "
          "unconstrained fit outside the ball can have an equally good twin inside it; the ball's cost "
          "is the direct measure."]
    if chk:
        L += [f"FAM's stored ProMP-8 floors recomputed: largest difference {max(chk):.2e} Nm over {len(chk)} "
              f"user-settings."]
    L.append("")

    # Excess, preference, change
    L += ["## Excess over the floor", "",
          "| cell | setting | n | k | median E_k | 90th percentile E_k | share E_k ≤ 0.5 | preference over "
          "generic (s = 1, λ = 0.1) |", "|---|---|---|---|---|---|---|---|"]
    res = {}
    for (c, s), rs in cell_rows.items():
        xs = _rows(rs, fl)
        if len(xs) < 8:
            continue
        for k in KS:
            med = R._boot(xs, lambda x, k=k: float(np.median([p["E"][k - 1] for p in x])), rng)
            p90 = R._boot(xs, lambda x, k=k: float(np.percentile([p["E"][k - 1] for p in x], 90)), rng)
            good = R._boot(xs, lambda x, k=k: float(np.mean([p["E"][k - 1] <= GOOD for p in x])), rng)
            pref = R._boot(xs, lambda x, k=k: F._pref_k(x, k), rng)
            res[c, s, "med", k] = med
            L.append(f"| {c} | {s} | {len(xs)} | {k} | {R._f(med)} | {R._f(p90)} | {R._f(good)} | {R._f(pref, 3)} |")
        res[c, s, "change"] = R._boot(xs, lambda x: float(np.mean([p["E"][149] - p["E"][39] for p in x])), rng)
        res[c, s, "worse"] = R._boot(xs, lambda x: float(np.mean([p["distances"][149] > p["d0"] for p in x])), rng)
    L += ["", "Paired change of the excess from k = 40 to k = 150 (mean E_150 − E_40, JND) and the share "
          "of users worse off than generic assistance at k = 150 (d_150 > d_0):", ""]
    for (c, s) in cell_rows:
        if (c, s, "change") in res:
            L.append(f"- {c}, {s}: {R._f(res[c, s, 'change'])}; worse off {R._f(res[c, s, 'worse'])}")
    L += ["", "Median E_k over k:", "",
          "| cell | setting | " + " | ".join(f"k = {k}" for k in CURVE_KS) + " |",
          "|---|---|" + "---|" * len(CURVE_KS)]
    for (c, s), rs in cell_rows.items():
        xs = _rows(rs, fl)
        if xs:
            L.append(f"| {c} | {s} | " + " | ".join(f"{np.median([p['E'][k - 1] for p in xs]):.2f}"
                                                   for k in CURVE_KS) + " |")
    L.append("")

    # Medium: are the peaks identified?
    med_rows = cell_rows.get(("main", "medium"), [])
    if len(med_rows) >= 8:
        full40 = [stored["main"][(r["user"], "medium")] for r in med_rows]
        L += ["## Medium: R² of the identified peaks against the preferred ones", "",
              "| k | R² extension | R² flexion |", "|---|---|---|"]
        for k, rs in ((40, full40), (150, med_rows)):
            r2 = [R._boot(rs, lambda x, b=b: R._r2(x, b), rng) for b in (0, 1)]
            L.append(f"| {k} | {R._f(r2[0])} | {R._f(r2[1])} |")
        L += ["", "k = 40: FULL's stored rows of the same users; k = 150: LONG's.", ""]

    # Seconds per comparison
    L += ["## Seconds per comparison (median over sessions; mean of the 5 comparisons around k)", "",
          "| cell | setting | machine | n | near k = 24 | near k = 150 | session (min) |",
          "|---|---|---|---|---|---|---|"]
    for (c, s), rs in cell_rows.items():
        for h in hosts:
            rh = [r for r in rs if r.get("stamps") and r.get("host", "?") == h]
            if rh:
                L.append(f"| {c} | {s} | {h} | {len(rh)} | {np.median([_sec_near(r, 24) for r in rh]):.3f} | "
                         f"{np.median([_sec_near(r, 150) for r in rh]):.3f} | "
                         f"{np.median([r['seconds'] for r in rh]) / 60:.1f} |")
    L.append("")

    # Predictions and the convergence criterion
    def point_med(c, s):
        t = res.get((c, s, "med", 150))
        return None if t is None else t[0]

    v1 = {s: (None if point_med("main", s) is None else ("held" if point_med("main", s) <= GOOD else "failed"))
          for s in CELLS["main"]}
    v2 = {s: (ABL._verdict(res["main", s, "change"], "below") if ("main", s, "change") in res else None)
          for s in CELLS["main"]}
    v3 = {s: (ABL._verdict(res["R2", s, "change"], "below") if ("R2", s, "change") in res else None)
          for s in CELLS["R2"]}

    def overall(v):
        v = list(v.values())
        return "held" if v and all(x == "held" for x in v) else ("failed" if "failed" in v else "uncertain")

    L += ["## Predictions", "",
          f"- PLG1 (main, each setting: median E_150 ≤ 0.5 JND): {overall(v1)} — "
          + ", ".join(f"{s} {point_med('main', s):.2f}" for s in CELLS["main"] if point_med("main", s) is not None),
          f"- PLG2 (main, each setting: E_150 − E_40 interval below 0): {overall(v2)} {v2}",
          f"- PLG3 (s = 2, wide and fixed 12: E_150 − E_40 interval below 0): {overall(v3)} {v3}",
          f"- PLG4 (every session's first 40 distances equal the stored ones): "
          f"{'held' if same_all else 'failed'}"
          + ("" if all(_machine_name(r) in passed for r in ok.values()) else
             " (some sessions ran where smoke's defaults did not reproduce the stored rows: a "
             "difference there can be the machine's, not the budget's)")]
    conv = overall(v1) == "held" and overall(v3) == "held"
    L += ["", "**Convergence criterion (author, 2026-10-29):** "
          + ("met — \"converges toward the best reachable profile, in simulation and under the stated "
             "answer model\"." if conv else
             "not met — \"approaches it\", median excess at k = 150: "
             + ", ".join(f"{c} {s} {point_med(c, s):.2f} JND" for (c, s) in cell_rows
                         if point_med(c, s) is not None) + "."), ""]
    if errors:
        L += ["## Errors", ""] + [f"- {_key(e)}: {e['error']}" for e in errors[:30]]
    (d / "amp_long_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("smoke", "floors", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"smoke": lambda: smoke(args.out), "floors": lambda: floors(args.out, args.workers),
     "run": lambda: run(args.out, args.workers), "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
