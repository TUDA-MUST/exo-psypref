"""Post hoc per-user measures of FULL, and the cost of one comparison.

Usage:
  python scripts/icorr_amp_posthoc.py users            -> amp_posthoc_users.md
  python scripts/icorr_amp_posthoc.py timing [--workers 4] [--n 10]   -> amp_timing.jsonl, .md

``users`` reads full_sessions.jsonl and gives, per setting
and optimizer at k = 24 and 40:
  - the share of users at least one JND (0.674 Nm) closer to their preferred curve than generic
    assistance is, and the share at least one JND further away;
  - the median error (RMS distance to the preferred curve) in Nm, in Nm/kg of the body mass of
    the Molinaro subject whose walking shape the user's preferred curve comes from
    (``recover_mass``, as icorr_units.py), and in % of the RMS of the user's preferred curve.
The new design draws preferred peaks per range, not per kg, so Nm/kg only rescales by the
shape's source subject. 95 % intervals from 2000 draws, users resampled within folds.

``timing`` reruns FULL's sessions of the first ``n`` users of the first fold, every setting and
optimizer, and records the wall time per comparison (session time / 40) on this machine with
``workers`` processes; the rerun distances are checked against FULL's.
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
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_full as F  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402

JND = true_jnd(AF.S)
ORDER = ("psypref", "gp_bald", "g", "d_z", "cmaes", "cmaes_sch4", "cmaes_sch")


def _rows(out: Path) -> list[dict]:
    path = F._dir(out) / "full_sessions.jsonl"
    return [r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r.get("error") is None]


def _attach(rows: list[dict], out: Path, molinaro: Path) -> None:
    """Each row's preferred-curve RMS and its source subject's body mass."""
    from preftune.data.loader import recover_mass

    masses: dict[str, float] = {}
    for r in rows:
        z = F._users(str(F._dir(out) / f"full-{r['fold']}.npz"))
        u = r["index"]
        ideal = AF.compose(r["amps"][0], r["amps"][1], z["user_minus"][u], z["user_plus"][u])
        r["_rms"] = float(np.sqrt(np.mean(ideal ** 2)))
        leg = str(z["user_subject"][u])
        if leg not in masses:
            subject, side = leg.rsplit("_", 1)
            masses[leg] = float(recover_mass(subject, side, molinaro)[0])
        r["_mass"] = masses[leg]


def users(out: Path, molinaro: Path) -> None:
    rows = _rows(out)
    _attach(rows, out, molinaro)
    rng = np.random.default_rng(AF.SEED_FULL + 30)
    by = {}
    for r in rows:
        by.setdefault((r["setting"], r["candidate"]), []).append(r)
    L = ["# FULL, post hoc: per-user shares and errors", "",
         f"{len(rows)} sessions of full_sessions.jsonl. JND {JND:.3f} Nm (s = {AF.S} Nm). Better / "
         "worse off: at least one JND closer to / further from the preferred curve than generic "
         "assistance. Errors: medians over users of the RMS distance to the preferred curve; Nm/kg "
         "by the body mass of the shape's source subject, % of the preferred curve's RMS. 95 % "
         f"intervals from {R.N_BOOT} draws, users resampled within folds. Post hoc.", ""]
    masses = sorted({r["_mass"] for r in rows})
    L += [f"Source-subject masses: {len(masses)} legs, {masses[0]:.1f}–{masses[-1]:.1f} kg.", ""]
    for setting in F.SETTINGS:
        L += [f"## {setting}", "",
              "| optimizer | k | better off | worse off | error (Nm) | error (Nm/kg) | error (% of preferred) |",
              "|---|---|---|---|---|---|---|"]
        gen = by[setting, "psypref"]
        L.append(f"| generic assistance | – | – | – | "
                 f"{R._f(R._boot(gen, lambda x: float(np.median([r['d0'] for r in x])), rng), 2)} | "
                 f"{R._f(R._boot(gen, lambda x: float(np.median([r['d0'] / r['_mass'] for r in x])), rng), 4)} | "
                 f"{R._f(R._boot(gen, lambda x: float(np.median([100 * r['d0'] / r['_rms'] for r in x])), rng), 1)} |")
        for cand in ORDER:
            rs = by.get((setting, cand))
            if not rs:
                continue
            for k in (24, 40):
                better = R._boot(rs, lambda x: float(np.mean([r["d0"] - r["distances"][k - 1] >= JND for r in x])), rng)
                worse = R._boot(rs, lambda x: float(np.mean([r["distances"][k - 1] - r["d0"] >= JND for r in x])), rng)
                nm = R._boot(rs, lambda x: float(np.median([r["distances"][k - 1] for r in x])), rng)
                kg = R._boot(rs, lambda x: float(np.median([r["distances"][k - 1] / r["_mass"] for r in x])), rng)
                pc = R._boot(rs, lambda x: float(np.median([100 * r["distances"][k - 1] / r["_rms"] for r in x])), rng)
                L.append(f"| {F.NAMES[cand]} | {k} | {R._f(tuple(100 * v for v in better), 0)} % | "
                         f"{R._f(tuple(100 * v for v in worse), 0)} % | {R._f(nm, 2)} | {R._f(kg, 4)} | {R._f(pc, 1)} |")
        # paired: PsyPref - GP-BALD in the better-off share and the error, k = 24
        a = {r["user"]: r for r in by[setting, "psypref"]}
        b = {r["user"]: r for r in by.get((setting, "gp_bald"), [])}
        pairs = [dict(fold=a[u]["fold"], a=a[u], b=b[u]) for u in sorted(set(a) & set(b))]
        if pairs:
            share = R._boot(pairs, lambda x: float(np.mean([(p["a"]["d0"] - p["a"]["distances"][23] >= JND)
                                                            - (p["b"]["d0"] - p["b"]["distances"][23] >= JND)
                                                            for p in x])), rng)
            err = R._boot(pairs, lambda x: float(np.median([p["a"]["distances"][23] - p["b"]["distances"][23]
                                                            for p in x])), rng)
            L += ["", f"PsyPref − GP-BALD, k = 24, paired: better-off share "
                      f"{R._f(tuple(100 * v for v in share), 1)} points; median error difference "
                      f"{R._f(err, 2)} Nm"]
        L.append("")
    (F._dir(out) / "amp_posthoc_users.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# ----------------------------------------------------------------------------- timing

def timing_jobs(out: Path, n: int) -> list[dict]:
    """The first ``n`` users of the first fold, every setting and optimizer of FULL."""
    return [j for j in F.jobs(out) if j["fold_index"] == 0 and j["index"] < n]


def _timed(job: dict, out: str) -> dict:
    t0 = time.perf_counter()
    row = F.run_one(dict(job), out)
    row["seconds"] = time.perf_counter() - t0
    return row


def timing(out: Path, workers: int, n: int) -> None:
    js = timing_jobs(out, n)
    ref = {(r["user"], r["setting"], r["candidate"]): r for r in _rows(out)}
    path = F._dir(out) / "amp_timing.jsonl"
    print(f"{len(js)} sessions, {workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("w", encoding="utf-8") as f:
        for fut in as_completed([pool.submit(_timed, j, str(out)) for j in js]):
            r = fut.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            rows.append(r)
    L = ["# Cost of one comparison", "",
         f"{len(rows)} sessions (FULL's first {n} users of fold {js[0]['fold']}), {workers} worker "
         f"processes, one thread each, on {platform.processor() or platform.machine()} "
         f"({os.cpu_count()} logical CPUs), shared machine. Seconds per comparison = session wall "
         f"time / {F.N_COMP} (includes the final identification).", "",
         "| setting | optimizer | n | median s per comparison | range | reproduces FULL |",
         "|---|---|---|---|---|---|"]
    for setting in F.SETTINGS:
        for cand in ORDER:
            rs = [r for r in rows if r["setting"] == setting and r["candidate"] == cand and r["error"] is None]
            if not rs:
                continue
            per = np.array([r["seconds"] / F.N_COMP for r in rs])
            same = sum(np.allclose(r["distances"], ref[r["user"], setting, cand]["distances"])
                       for r in rs if (r["user"], setting, cand) in ref)
            L.append(f"| {setting} | {F.NAMES[cand]} | {len(rs)} | {np.median(per):.3f} | "
                     f"{per.min():.3f}–{per.max():.3f} | {same}/{len(rs)} |")
    errs = [r for r in rows if r["error"]]
    if errs:
        L += ["", "Errors: " + "; ".join(f"{r['candidate']} {r['setting']}: {r['error']}" for r in errs)]
    (F._dir(out) / "amp_timing.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    import icorr_units as U

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("users", "timing"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--molinaro", type=Path, default=U.MOLINARO)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    if args.command == "users":
        users(args.out, args.molinaro)
    else:
        timing(args.out, args.workers, args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
