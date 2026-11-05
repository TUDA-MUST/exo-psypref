"""Test XSN: family floors on other sources of preferred profiles.

Usage:
  python scripts/icorr_amp_xsn.py floors [--workers 8]
  python scripts/icorr_amp_xsn.py report

Profile i of a fold is paired with FULL's user i: each
source's curve is reduced to its unit bouts and composed at the user's peaks (wide) or at 12 / 12
Nm (fixed). Sources: the phase-randomized surrogate of the user's preferred profile and the
earlier X's Schaefer, Franks, Ramella and Park generators (X's keys and seeds); FULL's users
are FAM's floors (fam_floors.jsonl), reused. Families and fit rules are FAM's.
Outputs in <out>/icorr/amp_free/: xsn_floors.jsonl (resumable), amp_xsn_report.md.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_fam as FAM  # noqa: E402
import icorr_amp_full as F  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.experiment import xexpr as X  # noqa: E402
from prompref.experiment.xsource import phase_randomize  # noqa: E402

SOURCES = ("surrogate", "schaefer", "franks", "ramella", "park")
#: Each published source's own families (PXN3 leaves them out).
OWN = {"schaefer": ("schaefer_i", "schaefer_ii"), "franks": ("franks",), "ramella": ("trapezoid",),
       "park": ("pchip_i", "pchip_ii", "pchip")}
JND = FAM.JND


@lru_cache(maxsize=None)
def _franks(run_dir: str, fold: str):
    from prompref.priors.franks_prior import FranksPrior

    return FranksPrior.load(Path(run_dir) / "priors" / f"{fold}-franks")


def profile(job: dict, source: str, out: Path) -> np.ndarray:
    """The source's preferred profile paired with FULL's user ``job``: unit bouts at its peaks."""
    if source == "surrogate":
        ideal, _, _ = FAM._user(job)
        rng = np.random.default_rng(X._seed("xsn-surrogate", job["user"], job["setting"]))
        curve = phase_randomize(ideal, rng)
    else:
        run_dir = out / "phase5-schaefer"
        franks = _franks(str(run_dir), job["fold"]) if source == "franks" else None
        _, curve = X.published_ideal(source, X.ideal_key(job["fold"], source, job["index"]), franks_prior=franks)
    return AF.compose(job["amps"][0], job["amps"][1], *AF.unit_bouts(curve))


def _key(r: dict) -> tuple:
    return r["user"], r["setting"], r["source"], r["family"]


def floor_one(job: dict, out: str) -> list[dict]:
    """Every family's floor for one user, setting and source."""
    out_p, rows = Path(out), []
    base = {k: v for k, v in job.items() if not k.startswith("_") and k != "candidate"}
    try:
        ideal = profile(job, job["source"], out_p)
    except Exception as exc:
        return [dict(base, family=f, error=f"{type(exc).__name__}: {exc}") for f in job["_families"]]
    kw = FAM._kw(job["setting"])
    for fam in job["_families"]:
        try:
            if fam.startswith("promp"):
                floor = AF.promp_floor(ideal, int(fam[5:]), **kw)
            else:
                design = AF.family_design(fam, out_p / "phase5-schaefer", job["fold"], **kw)
                seed = [AF.SEED_FULL, zlib.crc32(f"{job['user']}|{job['source']}|{job['setting']}|{fam}".encode())]
                floor = AF.family_floor(design, ideal, seed)[1]
            rows.append(dict(base, family=fam, floor=float(floor), error=None))
        except Exception as exc:  # one bad fit must not stop the run
            rows.append(dict(base, family=fam, error=f"{type(exc).__name__}: {exc}"))
    return rows


def floors(out: Path, workers: int) -> None:
    path = F._dir(out) / "xsn_floors.jsonl"
    done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r.get("error") is None} if path.exists() else set()
    todo = []
    for j in FAM._base_jobs(out):
        for source in SOURCES:
            fams = [f for f in FAM._families(j["setting"]) if (j["user"], j["setting"], source, f) not in done]
            if fams:
                todo.append(dict(j, source=source, _families=fams))
    print(f"{len(todo)} user × setting × source jobs to fit", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        for n, fut in enumerate(as_completed([pool.submit(floor_one, j, str(out)) for j in todo]), 1):
            for row in fut.result():
                f.write(json.dumps(row) + "\n")
                if row.get("error"):
                    print("ERROR", _key(row), row["error"], flush=True)
            f.flush()
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def report(out: Path) -> None:
    d = F._dir(out)
    raw = [json.loads(x) for x in (d / "xsn_floors.jsonl").read_text(encoding="utf-8").splitlines()]
    fl = {_key(r): r["floor"] for r in raw if r.get("error") is None}
    errors = {_key(r): r["error"] for r in raw if r.get("error") and _key(r) not in fl}
    for r in map(json.loads, (d / "fam_floors.jsonl").read_text(encoding="utf-8").splitlines()):
        if r.get("error") is None:
            fl[r["user"], r["setting"], "users", r["family"]] = r["floor"]
    users = sorted({k[0] for k in fl})
    fold = {u: u.split("/")[0] for u in users}
    rng = np.random.default_rng(AF.SEED_FULL + 71)

    def med(setting, source, fam):
        """Median floor in JND with its interval (users resampled within folds)."""
        items = [dict(fold=fold[u], v=fl[u, setting, source, fam] / JND) for u in users
                 if (u, setting, source, fam) in fl]
        return R._boot(items, lambda x: float(np.median([i["v"] for i in x])), rng)

    def dmed(setting, a, b):
        """Median of a minus median of b (each a (source, family)), the same users resampled."""
        items = [dict(fold=fold[u], a=fl[(u, setting) + a] / JND, b=fl[(u, setting) + b] / JND) for u in users
                 if (u, setting) + a in fl and (u, setting) + b in fl]
        return R._boot(items, lambda x: float(np.median([i["a"] for i in x]) - np.median([i["b"] for i in x])), rng)

    all_sources = ("users",) + SOURCES
    L = ["# Registration XSN: family floors on other sources of preferred profiles", "",
         f"Spec section 24 (c1a5c00). {len(fl)} floors ({len(errors)} failed fits). Median floor in JND "
         f"({JND:.3f} Nm) with 95 % intervals from {R.N_BOOT} draws, users resampled within folds. Users: "
         "FAM's floors (fam_floors.jsonl).", ""]
    pxn = {1: [], 2: [], 3: [], 4: []}
    for setting in FAM.SETTINGS:
        fams = FAM._families(setting)
        L += [f"## {setting}", "", "| family | " + " | ".join(all_sources) + " |",
              "|---|" + "---|" * len(all_sources)]
        for fam in fams:
            L.append(f"| {FAM.LABEL[fam]} | " + " | ".join(R._f(med(setting, s, fam), 2) for s in all_sources) + " |")
        L.append("")
        t = dmed(setting, ("surrogate", "promp8"), ("users", "promp8"))
        pxn[1].append("held" if -0.15 < t[1] and t[2] < 0.15 else ("failed" if t[2] < -0.15 or t[1] > 0.15 else "uncertain"))
        L.append(f"- PXN1, ProMP 8 surrogate − users: {R._f(t, 2)} JND")
        for fam in fams:
            if fam.startswith("promp"):
                continue
            t = dmed(setting, ("surrogate", fam), ("surrogate", "promp8"))
            pxn[2].append(FAM._verdict(t, "above"))
            L.append(f"- PXN2, surrogate, {FAM.LABEL[fam]} − ProMP 8: {R._f(t, 2)} JND")
        for source in OWN:
            for fam in fams:
                if fam.startswith("promp") or fam in OWN[source]:
                    continue
                t = dmed(setting, (source, "promp16"), (source, fam))
                pxn[3].append(FAM._verdict(t, "below"))
                L.append(f"- PXN3, {source}, ProMP 16 − {FAM.LABEL[fam]}: {R._f(t, 2)} JND")
        for source in all_sources:
            t = med(setting, source, "promp24")
            pxn[4].append("held" if t[2] <= 1.0 else ("failed" if t[1] > 1.0 else "uncertain"))
        L.append("")
    agg = FAM._all
    L += [f"- PXN1 (ProMP 8: surrogate within ±0.15 JND of users): {agg(pxn[1])} {pxn[1]}",
          f"- PXN2 (surrogate: every interpretable family above ProMP 8): {agg(pxn[2])} {pxn[2]}",
          f"- PXN3 (published sources: ProMP 16 below every other family): {agg(pxn[3])} {pxn[3]}",
          f"- PXN4 (ProMP 24 ≤ 1 JND on every source): {agg(pxn[4])} {pxn[4]}", ""]
    if errors:
        L += ["## Failed fits", ""] + [f"- {k}: {e}" for k, e in list(errors.items())[:30]]
    (d / "amp_xsn_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("floors", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"floors": lambda: floors(args.out, args.workers), "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
