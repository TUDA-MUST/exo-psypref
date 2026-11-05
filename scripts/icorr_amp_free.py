"""Test AF pilot: build the folds, run the sessions, write the report.

Usage:
  python scripts/icorr_amp_free.py prep   [--out <repository>/out] [--workers 5]
  python scripts/icorr_amp_free.py run    [--out <repository>/out] [--workers 8] [--limit N]
  python scripts/icorr_amp_free.py report [--out <repository>/out]

planned at dedbe8c before this script existed. Outputs in <out>/icorr/amp_free/: one npz per
fold (training parts, priors, users), sessions.jsonl (resumable), amp_free_report.md.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    # One BLAS thread per worker: eight workers at most on this shared machine.
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
sys.path.insert(0, str(ROOT))

from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.population.folds import loso_folds  # noqa: E402

N_BOOT = 2000


def af_dir(out: Path) -> Path:
    return out / "icorr" / "amp_free"


def _prep_one(args):
    i, run_dir, dest = args
    return str(AF.build_fold(i, Path(run_dir), Path(dest)))


def prep(out: Path, workers: int) -> None:
    run_dir = out / "phase5-schaefer"
    jobs = [(i, str(run_dir), str(af_dir(out))) for i in range(len(loso_folds()))]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for path in pool.map(_prep_one, jobs):
            print("wrote", path, flush=True)


@lru_cache(maxsize=None)
def _fold_data(path: str, fold_index: int) -> dict:
    return AF.load_fold_data(Path(path), fold_index)


def _run_one(job: dict) -> dict:
    data = _fold_data(job.pop("_path"), job["fold_index"])
    try:
        return AF.run_af_session(job, data)
    except Exception as exc:  # one bad session must not stop the pilot
        return dict({k: v for k, v in job.items() if k != "ideal"}, error=f"{type(exc).__name__}: {exc}")


def _key(row: dict) -> tuple:
    return (row["user"], row["scenario"], row["candidate"], int(row.get("k", AF.K)),
            float(row.get("s", AF.S)))


#: Change section 10: D1 (40 comparisons, S1-S3) and D2 (s = 0.5, S1) for A and B.
DIAGNOSTICS = (dict(k=40, s=AF.S, scenarios=AF.SCENARIOS), dict(k=AF.K, s=0.5, scenarios=("S1",)))


def _diag_jobs(jobs: list[dict]) -> list[dict]:
    return [dict(j, k=d["k"], s=d["s"]) for d in DIAGNOSTICS for j in jobs
            if j["candidate"] in ("a", "b") and j["scenario"] in d["scenarios"]]


def run(out: Path, workers: int, limit: int | None, diag: bool = False) -> None:
    d = af_dir(out)
    path = d / "sessions.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("error") is None:
                done.add(_key(row))
    jobs = []
    for i, fold in enumerate(loso_folds()):
        fpath = d / f"fold-{fold.name}.npz"
        fold_jobs = AF.jobs_for_fold(i, fold.name, AF.load_fold_data(fpath, i))
        for job in (_diag_jobs(fold_jobs) if diag else fold_jobs):
            if _key(job) not in done:
                jobs.append(dict(job, _path=str(fpath)))
    jobs = jobs[:limit] if limit else jobs
    print(f"{len(jobs)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        for n, fut in enumerate(as_completed([pool.submit(_run_one, j) for j in jobs]), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", _key(row), row["error"], flush=True)
            if n % 20 == 0:
                print(f"{n}/{len(jobs)}", flush=True)


# ----------------------------------------------------------------------------- report

def _pref(d0, dk):
    return AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((np.asarray(d0) - np.asarray(dk)) / AF.S)


def _slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def _boot(rows_by_fold: dict, stat, rng) -> tuple[float, float, float]:
    """Point estimate and 95 % interval, users resampled within each fold."""
    est = stat([r for rows in rows_by_fold.values() for r in rows])
    draws = []
    for _ in range(N_BOOT):
        sample = []
        for rows in rows_by_fold.values():
            idx = rng.integers(len(rows), size=len(rows))
            sample += [rows[i] for i in idx]
        draws.append(stat(sample))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return est, float(lo), float(hi)


def _by_fold(rows):
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r["fold"], []).append(r)
    return out


def _fmt(t, digits=2):
    return f"{t[0]:.{digits}f} [{t[1]:.{digits}f}, {t[2]:.{digits}f}]"


def report(out: Path) -> None:
    d = af_dir(out)
    rows = [json.loads(x) for x in (d / "sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [r for r in rows if r.get("error")]
    allrows = {_key(r): r for r in rows if not r.get("error")}
    # The planned comparison is at k = 24 and s = 1.0; the diagnostics are reported apart
    rows = {key[:3]: r for key, r in allrows.items() if key[3] == AF.K and key[4] == AF.S}
    rng = np.random.default_rng(AF.SEED)
    stats = {}
    lines = ["# Test plan AF pilot: amplitude-free prior", "",
             f"Spec: our design notes (dedbe8c). "
             f"{len(rows)} sessions, {len(errors)} errors. k = {AF.K}, s = {AF.S} Nm, "
             f"λ = {AF.LAPSE}, device ±{AF.LIMIT_NM:.0f} Nm, generic peaks {AF.GENERIC_PEAK_NM:.2f} Nm. "
             f"95 % intervals: {N_BOOT} draws, users resampled within folds.", "",
             "| scenario | candidate | n | slope ext | slope flex | median abs log error ext / flex "
             "| preference over generic | passes |", "|---|---|---|---|---|---|---|---|"]
    for sc in AF.SCENARIOS:
        for cand in AF.CANDIDATES:
            rs = [r for (u, s_, c), r in rows.items() if s_ == sc and c == cand]
            if len(rs) < 4:
                continue
            bf = _by_fold(rs)
            se = _boot(bf, lambda x: _slope([r["preferred_ext"] for r in x],
                                            [max(r["identified_ext"], 1e-3) for r in x]), rng)
            sf = _boot(bf, lambda x: _slope([r["preferred_flex"] for r in x],
                                            [max(r["identified_flex"], 1e-3) for r in x]), rng)
            ee = float(np.median([abs(math.log(max(r["identified_ext"], 1e-3) / r["preferred_ext"]))
                                  for r in rs]))
            ef = float(np.median([abs(math.log(max(r["identified_flex"], 1e-3) / r["preferred_flex"]))
                                  for r in rs]))
            pr = _boot(bf, lambda x: float(np.mean(_pref([r["d0"] for r in x],
                                                         [r["distances"][-1] for r in x]))), rng)
            ok = all(t[0] >= 0.8 and t[2] >= 0.9 for t in (se, sf))
            stats[sc, cand] = dict(se=se, sf=sf, ee=ee, ef=ef, pref=pr, ok=ok, n=len(rs))
            lines.append(f"| {sc} | {cand} | {len(rs)} | {_fmt(se)} | {_fmt(sf)} | {ee:.2f} / {ef:.2f} "
                         f"| {_fmt(pr, 3)} | {'yes' if ok else 'no'} |")

    def paired(sc, c1, c2):
        users = sorted({u for (u, s_, c) in rows if s_ == sc and c == c1}
                       & {u for (u, s_, c) in rows if s_ == sc and c == c2})
        pairs = [dict(fold=rows[u, sc, c1]["fold"],
                      diff=float(_pref(rows[u, sc, c1]["d0"], rows[u, sc, c1]["distances"][-1])
                                 - _pref(rows[u, sc, c2]["d0"], rows[u, sc, c2]["distances"][-1])))
                 for u in users]
        if len(pairs) < 4:
            return None
        return _boot(_by_fold(pairs), lambda x: float(np.mean([p["diff"] for p in x])), rng)

    lines += ["", "Paired differences in preference over generic assistance (points):", ""]
    diffs = {}
    for sc in AF.SCENARIOS:
        for c1, c2 in (("a", "ref"), ("c", "a"), ("b", "a")):
            t = paired(sc, c1, c2)
            if t is not None:
                diffs[sc, c1, c2] = t
                lines.append(f"- {sc}, {c1} − {c2}: {100 * t[0]:+.1f} [{100 * t[1]:+.1f}, "
                             f"{100 * t[2]:+.1f}]")

    def verdict(cond_held, cond_failed):
        return "held" if cond_held else ("failed" if cond_failed else "uncertain")

    lines += ["", "## Predictions", ""]
    if all((sc, "ref") in stats for sc in AF.SCENARIOS):
        mins = {sc: min(stats[sc, "ref"]["se"][0], stats[sc, "ref"]["sf"][0]) for sc in AF.SCENARIOS}
        low = min(mins.values())
        lines.append(f"- PAF1 (REF fails; minimum slope < 0.6, lowest in S2): "
                     f"{verdict(not any(stats[sc, 'ref']['ok'] for sc in AF.SCENARIOS) and low < 0.6 and min(mins, key=mins.get) == 'S2', False)}"
                     f" — minimum slopes {', '.join(f'{k} {v:.2f}' for k, v in mins.items())}")
    for cand, name in (("a", "PAF2 (A passes)"), ("c", "PAF4a (C passes)")):
        if all((sc, cand) in stats for sc in AF.SCENARIOS):
            ok = all(stats[sc, cand]["ok"] for sc in AF.SCENARIOS)
            lines.append(f"- {name}: {'held' if ok else 'failed'}")
    if ("S3", "a", "ref") in diffs:
        t = diffs["S3", "a", "ref"]
        lines.append(f"- PAF3 (S3, A − REF > 0): {verdict(t[1] > 0, t[2] < 0)}")
    if ("S3", "c", "a") in diffs:
        t = diffs["S3", "c", "a"]
        lines.append(f"- PAF4b (S3, C below A): {verdict(t[2] < 0, t[1] > 0)}")
    if ("S1", "a", "ref") in diffs:
        t = diffs["S1", "a", "ref"]
        lines.append(f"- PAF5 (S1, |A − REF| within 5 points): {verdict(t[1] > -0.05 and t[2] < 0.05, t[2] < -0.05 or t[1] > 0.05)}")
    # Change (section 9): candidate B
    if all((sc, c) in stats for sc in AF.SCENARIOS for c in ("a", "b")):
        def min_slope(c):
            return min(min(stats[sc, c]["se"][0], stats[sc, c]["sf"][0]) for sc in AF.SCENARIOS)
        lines.append(f"- PAF6 (B's minimum slope above A's): "
                     f"{'held' if min_slope('b') > min_slope('a') else 'failed'} — B {min_slope('b'):.2f}, "
                     f"A {min_slope('a'):.2f}")
        lines.append(f"- PAF7 (B passes in S2 and S3): "
                     f"{'held' if stats['S2', 'b']['ok'] and stats['S3', 'b']['ok'] else 'failed'}")
    if ("S3", "b", "a") in diffs:
        t = diffs["S3", "b", "a"]
        lines.append(f"- PAF8 (S3, B − A lower bound above −3 points): "
                     f"{verdict(t[1] > -0.03, t[2] < -0.03)}")
    # Change section 10: diagnostics D1 (k = 40) and D2 (s = 0.5)
    def diag_slopes(sc, cand, k, s):
        rs = [r for key, r in allrows.items() if key[1:] == (sc, cand, k, s)]
        if len(rs) < 4:
            return None
        bf = _by_fold(rs)
        se = _boot(bf, lambda x: _slope([r["preferred_ext"] for r in x],
                                        [max(r["identified_ext"], 1e-3) for r in x]), rng)
        sf = _boot(bf, lambda x: _slope([r["preferred_flex"] for r in x],
                                        [max(r["identified_flex"], 1e-3) for r in x]), rng)
        pr = _boot(bf, lambda x: float(np.mean(AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf(
            (np.array([r["d0"] for r in x]) - np.array([r["distances"][-1] for r in x])) / s))), rng)
        return dict(se=se, sf=sf, pref=pr, n=len(rs))

    lines += ["", "## Diagnostics (change, section 10)", "",
              "| diagnostic | scenario | candidate | n | slope ext | slope flex | preference over generic "
              "| mean slope change |", "|---|---|---|---|---|---|---|---|"]
    rises = {}
    for name, k, s, scs in (("D1, k = 40", 40, AF.S, AF.SCENARIOS), ("D2, s = 0.5", AF.K, 0.5, ("S1",))):
        for sc in scs:
            for cand in ("a", "b"):
                dg = diag_slopes(sc, cand, k, s)
                if dg is None or (sc, cand) not in stats:
                    continue
                base = stats[sc, cand]
                rise = 0.5 * ((dg["se"][0] - base["se"][0]) + (dg["sf"][0] - base["sf"][0]))
                rises[name, sc, cand] = rise
                lines.append(f"| {name} | {sc} | {cand} | {dg['n']} | {_fmt(dg['se'])} | {_fmt(dg['sf'])} "
                             f"| {_fmt(dg['pref'], 3)} | {rise:+.2f} |")
    d1 = [rises.get(("D1, k = 40", sc, c)) for sc in ("S2", "S3") for c in ("a", "b")]
    if all(v is not None for v in d1):
        lines.append(f"\n- PAF9 (D1: slopes in S2 and S3 rise by at least 0.1): "
                     f"{'held' if all(v >= 0.1 for v in d1) else 'failed'} — "
                     + ", ".join(f"{v:+.2f}" for v in d1))
    d2 = [rises.get(("D2, s = 0.5", "S1", c)) for c in ("a", "b")]
    if all(v is not None for v in d2):
        lines.append(f"- PAF10 (D2: S1 slopes rise by at least 0.15): "
                     f"{'held' if all(v >= 0.15 for v in d2) else 'failed'} — "
                     + ", ".join(f"{v:+.2f}" for v in d2))
    passing = [c for c in AF.CANDIDATES if all((sc, c) in stats and stats[sc, c]["ok"] for sc in AF.SCENARIOS)]
    lines += ["", "## Selection (rule of section 6)", "",
              f"Passing: {', '.join(passing) or 'none'}."]
    if errors:
        lines += ["", "## Errors", ""] + [f"- {_key(e)}: {e['error']}" for e in errors[:20]]
    (d / "amp_free_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("prep", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--diag", action="store_true", help="run the diagnostics D1 and D2 (change, section 10)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"prep": lambda: prep(args.out, args.workers), "run": lambda: run(args.out, args.workers, args.limit, args.diag),
     "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
