"""Test PAN: the prior updated with n preferred curves of a new population.

Usage:
  python scripts/icorr_amp_pan.py run [--workers 8]
  python scripts/icorr_amp_pan.py report

The static prior (mu_0, Sigma_0) counts as N0 = 4 curves;
n curves x_i (mean xbar, scatter S) update it to
  mu_n = (4 mu_0 + n xbar) / (4 + n),
  Sigma_n = (4 Sigma_0 + S + 4n/(4 + n) (xbar - mu_0)(xbar - mu_0)^T) / (4 + n).
Sources: exact (rows of the population's own pool: S3 PRI's Schaefer-family draws, S0 the fold's
training shapes; log-uniform peaks in a range, unit shapes at the level) and identified (S3 only:
the curves PsyPref with the static prior delivered at k = 40 for n other users of the fold).
``run`` first reruns PAD's S3 static sessions to record the identified curves (pan_ident.jsonl),
then the sessions (pan_sessions.jsonl); both resumable. ``report`` writes amp_pan_report.md.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_full as F  # noqa: E402
import icorr_amp_pri as P  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.population.folds import loso_folds  # noqa: E402

NS = (4, 11, 30)
POPS = ("S0 users", "S3 Schaefer shapes")
SOURCES = ("exact", "identified")
SETTINGS = P.PAD_SETTINGS
S3 = "S3 Schaefer shapes"


def update(mu0: np.ndarray, s0: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The normal-inverse-Wishart update with kappa0 = nu0 = N0, scale normalized."""
    n, n0 = x.shape[0], P.N0
    xbar = x.mean(axis=0)
    scatter = (x - xbar).T @ (x - xbar)
    dm = (xbar - mu0)[:, None]
    mu = (n0 * mu0 + n * xbar) / (n0 + n)
    sigma = (n0 * s0 + scatter + n0 * n / (n0 + n) * (dm @ dm.T)) / (n0 + n)
    return mu, sigma


def _scale(setting: str) -> float:
    return 1.0 if setting in F.RANGES else 1.0 / P.LEVEL


# ----------------------------------------------------------------------------- identification reruns

def ident_jobs(out: Path) -> list[dict]:
    d = P._dir(out)
    js = []
    for fi, fold in enumerate(loso_folds()):
        z = F._users(str(d / f"full-{fold.name}.npz"))
        for setting in SETTINGS:
            js += [dict(P._base(fold, fi, u, setting, z), test="PAD", condition=S3, candidate="psypref_static")
                   for u in range(AF.FULL_USERS_PER_FOLD)]
    return js


def ident_one(job: dict, out: str) -> dict:
    try:
        row, delivered = P.session(job, Path(out))
        row["identified"] = P.BASIS.project(delivered * _scale(job["setting"])).tolist()
        return row
    except Exception as exc:  # one bad session must not stop the run
        return dict(job, error=f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------------- PAN sessions

def _exact(pop: str, setting: str, fi: int, out: Path, rng: np.random.Generator, n: int):
    d = P._dir(out)
    fold = loso_folds()[fi].name
    if pop == S3:
        p = P._prep(str(d / f"pri-prep-{fold}.npz"))
        wm, wp = p["S3_w_minus"], p["S3_w_plus"]
    else:
        dd = R._fold(str(d / f"fold-{fold}.npz"), fi)
        wm, wp = dd["w_minus"], dd["w_plus"]
    idx = rng.choice(wm.shape[0], size=n, replace=False)
    if setting in F.RANGES:
        a = AF._loguniform(rng, *F.RANGES[setting], (n, 2))
        return a[:, :1] * wm[idx] + a[:, 1:] * wp[idx], idx
    return wm[idx] + wp[idx], idx


def _key(r: dict) -> tuple:
    return r["condition"], r["setting"], r["source"], r["n"], r["user"]


def jobs(out: Path, ident: dict) -> list[dict]:
    """``ident``: (setting, user) -> identified parameters, from the reruns."""
    d = P._dir(out)
    js = []
    for fi, fold in enumerate(loso_folds()):
        z = F._users(str(d / f"full-{fold.name}.npz"))
        for setting in SETTINGS:
            si = F.SETTINGS.index(setting)
            for pop in POPS:
                pi = list(P.PAD_POPS).index(pop)
                for vi, source in enumerate(SOURCES):
                    if source == "identified" and pop != S3:
                        continue
                    for ni, n in enumerate(NS):
                        for u in range(AF.FULL_USERS_PER_FOLD):
                            rng = np.random.default_rng([AF.SEED_FULL, 2400, fi, u, si, pi, ni, vi])
                            if source == "exact":
                                x, idx = _exact(pop, setting, fi, out, rng, n)
                            else:
                                others = [v for v in range(AF.FULL_USERS_PER_FOLD) if v != u]
                                idx = rng.choice(others, size=n, replace=False)
                                x = np.array([ident[setting, f"{fold.name}/f{v}"] for v in idx])
                            js.append(dict(P._base(fold, fi, u, setting, z), test="PAN", condition=pop,
                                           candidate="psypref_pan", source=source, n=n,
                                           pool=[int(i) for i in idx], _x=x.tolist()))
    return js


def pan_one(job: dict, out: str) -> dict:
    out_p = Path(out)
    try:
        ppath = str(P._dir(out_p) / f"fold-{job['fold']}.npz")
        mu0, s0 = P.moments(job["setting"], job["fold_index"], ppath, out_p, "static")
        mu, sigma = update(mu0, s0, np.asarray(job["_x"], dtype=float))
        return P.session(job, out_p, mu_override=mu, sigma_override=sigma)[0]
    except Exception as exc:  # one bad session must not stop the run
        return dict({k: v for k, v in job.items() if not k.startswith("_")},
                    error=f"{type(exc).__name__}: {exc}")


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _pool(fn, todo: list[dict], out: Path, path: Path, workers: int, key) -> None:
    print(f"{path.name}: {len(todo)} sessions to run", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        for n, fut in enumerate(as_completed([pool.submit(fn, j, str(out)) for j in todo]), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", key(row), row["error"], flush=True)
            if n % 200 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def run(out: Path, workers: int) -> None:
    d = P._dir(out)
    ipath, spath = d / "pan_ident.jsonl", d / "pan_sessions.jsonl"
    ikey = lambda r: (r["setting"], r["user"])  # noqa: E731
    done = {ikey(r) for r in _read(ipath) if r.get("error") is None}
    _pool(ident_one, [j for j in ident_jobs(out) if ikey(j) not in done], out, ipath, workers, ikey)
    ident = {ikey(r): r["identified"] for r in _read(ipath) if r.get("error") is None}
    if len(ident) != 2 * len(loso_folds()) * AF.FULL_USERS_PER_FOLD:
        raise SystemExit(f"{len(ident)} identified curves; rerun to complete them first")
    done = {_key(r) for r in _read(spath) if r.get("error") is None}
    _pool(pan_one, [j for j in jobs(out, ident) if _key(j) not in done], out, spath, workers, _key)


# ----------------------------------------------------------------------------- report

def report(out: Path) -> None:
    d = P._dir(out)
    raw = _read(d / "pan_sessions.jsonl")
    rows = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in rows]
    pad = {(r["condition"], r["setting"], r["candidate"], r["user"]): r for r in _read(d / "pri_sessions.jsonl")
           if r["test"] == "PAD" and r.get("error") is None}
    full = {(r["setting"], r["user"]): r for r in _read(d / "full_sessions.jsonl")
            if r.get("error") is None and r["candidate"] == "psypref"}
    ident = [r for r in _read(d / "pan_ident.jsonl") if r.get("error") is None]
    same = sum(np.allclose(r["distances"], pad[S3, r["setting"], "psypref_static", r["user"]]["distances"])
               for r in ident)
    rng = np.random.default_rng(AF.SEED_FULL + 51)

    def ref(pop, setting, cand):
        if pop == "S0 users":
            return {u: r for (s, u), r in full.items() if s == setting} if cand == "psypref_static" else {}
        return {k[3]: r for k, r in pad.items() if k[:3] == (pop, setting, cand)}

    def arm(pop, setting, source, n):
        return {k[4]: r for k, r in rows.items() if k[:4] == (pop, setting, source, n)}

    def share(a, st, orc):
        items = [dict(fold=a[u]["fold"], a=a[u], s=st[u], o=orc[u]) for u in sorted(set(a) & set(st) & set(orc))]
        return R._boot(items, lambda x: float(np.mean([P._p(i["a"]) - P._p(i["s"]) for i in x])
                                         / np.mean([P._p(i["o"]) - P._p(i["s"]) for i in x])), rng)

    L = ["# Test plan PAN: the prior updated with n curves of a new population", "",
         f"Design note 22 (6cbc3d0). {len(rows)} sessions, {len(errors)} errors; identification reruns: "
         f"{same} of {len(ident)} reproduce PAD's static S3 rows. Preference over generic assistance at "
         f"k = 24; paired differences in points; 95 % intervals from {R.N_BOOT} draws, users resampled "
         "within folds.", "",
         "| population | setting | prior | preference | − static | share of the oracle's gain | − CMA-ES (Schäfer) |",
         "|---|---|---|---|---|---|---|"]
    m = lambda v: float(np.mean([P._p(r) for r in v.values()])) if v else float("nan")  # noqa: E731
    diffs = {}
    for pop in POPS:
        for setting in SETTINGS:
            st, orc, cm = (ref(pop, setting, c) for c in ("psypref_static", "psypref_oracle", "cmaes_sch4"))
            L.append(f"| {pop} | {setting} | static | {m(st):.3f} | – | – | "
                     f"{R._f(P._diff(P._pairs(st, cm), rng), 1) if cm else '–'} |")
            for source in SOURCES:
                for n in NS:
                    a = arm(pop, setting, source, n)
                    if not a:
                        continue
                    ds = P._diff(P._pairs(a, st), rng)
                    dc = P._diff(P._pairs(a, cm), rng) if cm else None
                    diffs[pop, setting, source, n] = ds, dc
                    L.append(f"| {pop} | {setting} | {source}, n = {n} | {m(a):.3f} | {R._f(ds, 1)} | "
                             f"{R._f(share(a, st, orc), 2) if orc else '–'} | {R._f(dc, 1) if dc else '–'} |")
            if orc:
                L.append(f"| {pop} | {setting} | oracle (2,000 curves) | {m(orc):.3f} | "
                         f"{R._f(P._diff(P._pairs(orc, st), rng), 1)} | 1 | {R._f(P._diff(P._pairs(orc, cm), rng), 1)} |")
    agg = lambda vs: "held" if vs and all(v == "held" for v in vs) else ("failed" if "failed" in vs else "uncertain")  # noqa: E731
    get = lambda *k: diffs.get(k, (None, None))  # noqa: E731
    pan1 = [P._verdict(get(S3, s, "exact", 30)[0]) for s in SETTINGS if get(S3, s, "exact", 30)[0]]
    pan2 = [P._verdict(t) for t in [get(S3, "fixed 12", "exact", 11)[0]] if t]
    pan3 = [P._verdict(t) for t in [get(S3, "fixed 12", "exact", 30)[1]] if t]
    pan4 = [P._verdict(t) for t in [get(S3, "fixed 12", "identified", 30)[0]] if t]
    pan5 = ["held" if t[0] >= -2.0 else "failed" for s in SETTINGS for n in (11, 30)
            for t in [get("S0 users", s, "exact", n)[0]] if t]
    L += ["", f"- PAN1 (S3: 30 exact curves ahead of static, both settings): {agg(pan1)} {pan1}",
          f"- PAN2 (S3, fixed 12: 11 exact curves ahead of static): {agg(pan2)} {pan2}",
          f"- PAN3 (S3, fixed 12: 30 exact curves ahead of CMA-ES on Schäfer's family): {agg(pan3)} {pan3}",
          f"- PAN4 (S3, fixed 12: 30 identified curves ahead of static): {agg(pan4)} {pan4}",
          f"- PAN5 (S0: 11 and 30 exact curves − static ≥ −2 points, both settings): {agg(pan5)} {pan5}", ""]
    if errors:
        L += ["## Errors", ""] + [f"- {_key(r)}: {r['error']}" for r in errors[:30]]
    (d / "amp_pan_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"run": lambda: run(args.out, args.workers), "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
