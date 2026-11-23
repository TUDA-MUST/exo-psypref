"""Post hoc: CMA-ES on Schaefer et al.'s family, started at generic assistance (review 2026-11-09).

Usage:
  python scripts/icorr_amp_cmastart.py run [--workers 8]
  python scripts/icorr_amp_cmastart.py report

Why: Fig. 3b compares PsyPref with projection against FULL's cmaes_sch4, which starts at its own prior's
center (the population's timings, both peaks at the range's arithmetic center), while generic assistance and
PsyPref start at the geometric center. Two reviewers read the 18.8-point gap as start and search mixed. This
reruns cmaes_sch4 with one change: pycma's mean0 is generic assistance's least-squares fit in the family
(ball-clipped). Same users, peaks, answer seeds, sigma0 0.5 and population 4 as FULL, in 3-35 Nm and at
fixed 12 Nm. Outputs in <out>/icorr/amp_free/: cmastart_sessions.jsonl (resumable), amp_cmastart_report.md.
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
from prompref.user.simulated_user import Observer, rms_distance  # noqa: E402

SETTINGS = ("wide", "fixed 12")
PROJ = {"wide": "proj_schaefer_ii", "fixed 12": "proj_schaefer_i"}   # Fig. 3b's PsyPref with projection
KS = (12, 24, 40)
N_BOOT = 2000


def jobs(out: Path) -> list[dict]:
    return [j for j in F.jobs(out) if j["candidate"] == "cmaes_sch4" and j["setting"] in SETTINGS]


def _space(run_dir: Path, job: dict):
    if job["setting"] == "wide":
        lo, hi = F.RANGES["wide"]
        return AF.schaefer_range_space(run_dir, job["fold"], lo, hi), math.sqrt(lo * hi)
    return AF.schaefer_fixed_space(run_dir, job["fold"], F.LEVEL), F.LEVEL


def run_one(job: dict, out: str) -> dict:
    from prompref.learning.sortcma import SortCMALearner
    from prompref.learning.whitened import WhitenedLearner

    upath, ppath = job.pop("_users"), job.pop("_parts")
    try:
        z, d = F._users(upath), R._fold(ppath, job["fold_index"])
        u = job["index"]
        run_dir = Path(out) / "phase5-schaefer"
        shape = (z["user_minus"][u], z["user_plus"][u])
        ideal = AF.compose(job["amps"][0], job["amps"][1], *shape)
        g_minus, g_plus = AF.unit_bouts(d["generic"])
        seed = int(job["answer_seed"]) % 2**32
        space, centre = _space(run_dir, job)
        generic = AF.compose(centre, centre, g_minus, g_plus)
        theta_g, fit_rms = AF.fit_family(space.design, generic, AF._starts(space.design, generic))
        mean0 = space.clip_to_ball(space.to_z(theta_g))
        learner = WhitenedLearner(space, SortCMALearner(space.as_design(), seed=seed, sigma0=0.5,
                                                        popsize=4, mean0=mean0))
        observer = Observer(ideal_nm=ideal, s=AF.S, lapse=AF.LAPSE)
        rng = np.random.default_rng([int(job["answer_seed"]), AF.OBSERVER_SALT])
        try:
            rec = run_session(space.design, learner, observer, rng, n_comparisons=F.N_COMP,
                              randomise_order=True, d_prior=rms_distance(generic, ideal), d_floor=0.0)
        finally:
            learner.inner.close()
        start = {k: rms_distance(space.design.curve(space.to_theta(zz)), ideal)
                 for k, zz in (("own", np.zeros(space.d)), ("generic", mean0))}
        return dict(job, candidate="cmaes_sch4_gs", d0=rms_distance(generic, ideal),
                    distances=list(rec["distances"]), d_start_own=start["own"],
                    d_start_generic=start["generic"], generic_fit_rms=fit_rms, error=None)
    except Exception as exc:  # one bad session must not stop the run
        return dict(job, candidate="cmaes_sch4_gs", error=f"{type(exc).__name__}: {exc}")


def run(out: Path, workers: int) -> None:
    path = F._dir(out) / "cmastart_sessions.jsonl"
    done = set()
    if path.exists():
        done = {(r["user"], r["setting"]) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = [j for j in jobs(out) if (j["user"], j["setting"]) not in done]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", row["user"], row["setting"], row["error"], flush=True)
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- report

def _p(d0, dk) -> np.ndarray:
    return AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((np.asarray(d0) - np.asarray(dk)) / AF.S)


def _boot(folds: np.ndarray, x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Mean of x with a 95 % interval, users resampled within their walking subject (fold)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(folds == f) for f in np.unique(folds)]
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(g, size=len(g)) for g in groups])
        draws[b] = x[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return 100 * float(x.mean()), 100 * float(lo), 100 * float(hi)


def _fmt(t) -> str:
    return f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"


def report(out: Path) -> None:
    d = F._dir(out)
    def load(name):
        return [json.loads(x) for x in (d / name).read_text(encoding="utf-8").splitlines()]
    gs = {(r["user"], r["setting"]): r for r in load("cmastart_sessions.jsonl") if r.get("error") is None}
    own = {(r["user"], r["setting"]): r for r in load("full_sessions.jsonl")
           if r.get("error") is None and r["candidate"] == "cmaes_sch4"}
    proj = {(r["user"], r["setting"]): r for r in load("fam_sessions.jsonl")
            if r.get("error") is None and r["candidate"] == PROJ.get(r["setting"])}
    L = ["# Post hoc: CMA-ES on Schäfer et al.'s family, started at generic assistance", "",
         f"Sessions: {len(gs)} (errors are rerun by `run`). P_g in %, means over users with 95 % intervals "
         f"({N_BOOT} draws, users resampled within their walking subject, paired by user). "
         "Own start: FULL's cmaes_sch4 (population timings, arithmetic-center peaks). "
         "Generic start: the same with pycma's mean0 at generic assistance's fit in the family.", ""]
    for st in SETTINGS:
        users = sorted(u for (u, s) in gs if s == st and (u, s) in own and (u, s) in proj)
        if not users:
            continue
        folds = np.array([gs[u, st]["fold"] for u in users])
        a = [gs[u, st] for u in users]
        o = [own[u, st] for u in users]
        p = [proj[u, st] for u in users]
        d0 = np.array([r["d0"] for r in a])
        fit = np.array([r["generic_fit_rms"] for r in a])
        L += [f"## {st} (n = {len(users)})", "",
              f"Generic assistance's fit in the family: median RMS {np.median(fit):.2f} Nm.", "",
              "| | own start | generic start | generic − own | PsyPref with projection − generic start |",
              "|---|---|---|---|---|"]
        s_own, s_gen = _p(d0, [r["d_start_own"] for r in a]), _p(d0, [r["d_start_generic"] for r in a])
        L.append(f"| k = 0 | {_fmt(_boot(folds, s_own))} | {_fmt(_boot(folds, s_gen))} | "
                 f"{_fmt(_boot(folds, s_gen - s_own))} | |")
        for k in KS:
            pg = _p(d0, [r["distances"][k - 1] for r in a])
            po = _p(d0, [r["distances"][k - 1] for r in o])
            pp = _p(np.array([r["d0"] for r in p]), [r["distances"][k - 1] for r in p])
            L.append(f"| k = {k} | {_fmt(_boot(folds, po))} | {_fmt(_boot(folds, pg))} | "
                     f"{_fmt(_boot(folds, pg - po))} | {_fmt(_boot(folds, pp - pg))} |")
        L.append("")
    (d / "amp_cmastart_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "report"))
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(a.out)
    if a.cmd == "run":
        run(out, min(a.workers, 8))
    else:
        report(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
