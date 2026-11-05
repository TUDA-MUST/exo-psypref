"""Test CMP: the comparators in their design regime.

Usage:
  python scripts/icorr_amp_cmp.py run [--workers 8]
  python scripts/icorr_amp_cmp.py report

FULL's users, peaks, answer seeds and generic assistance
in the three settings; each cell is a comparator of ``amp_free.comparator`` run by FULL's
``run_one``: GP-qEUBO, CMA-ES on the ProMP sized to the budget (population 4, sigma0 0.5), Arens'
GP with a generic start (with and without the true-JND gate) and Ramella et al.'s learner.
Outputs in <out>/icorr/amp_free/: cmp_sessions.jsonl (resumable), amp_cmp_report.md.
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

CELLS = ("gp_eubo", "cmaes4", "g_generic", "g_generic_nogate", "ramella")
NAMES = {"gp_eubo": "GP-qEUBO", "cmaes4": "CMA-ES (ProMP, population 4)",
         "g_generic": "Arens' GP, generic start", "g_generic_nogate": "Arens' GP, generic start, no gate",
         "ramella": "Ramella et al."}
PUBLISHED = ("g", "d_z", "cmaes")
STUCK = 0.1 * P.JND


def _key(r: dict) -> tuple:
    return r["user"], r["setting"], r["candidate"]


def jobs(out: Path) -> list[dict]:
    base = [j for j in F.jobs(out) if j["candidate"] == "psypref"]
    js = [dict(j, candidate=c) for c in CELLS for j in base]
    return sorted(js, key=lambda j: j["candidate"] != "gp_eubo")   # the slow GP first


def run(out: Path, workers: int) -> None:
    path = F._dir(out) / "cmp_sessions.jsonl"
    done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r.get("error") is None} if path.exists() else set()
    todo = [j for j in jobs(out) if _key(j) not in done]
    print(f"{len(todo)} sessions to run", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        for n, fut in enumerate(as_completed([pool.submit(F.run_one, j, str(out)) for j in todo]), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", _key(row), row["error"], flush=True)
            if n % 200 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def report(out: Path) -> None:
    d = F._dir(out)
    raw = [json.loads(x) for x in (d / "cmp_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in rows]
    full = {_key(r): r for r in map(json.loads, (d / "full_sessions.jsonl").read_text(
        encoding="utf-8").splitlines()) if r.get("error") is None}
    rng = np.random.default_rng(AF.SEED_FULL + 61)

    def cell(src, setting, cand):
        return {k[0]: r for k, r in src.items() if k[1:] == (setting, cand)}

    L = ["# Registration CMP: the comparators in their design regime", "",
         f"Spec section 23 (c1a5c00). {len(rows)} sessions, {len(errors)} errors. Preference over generic "
         f"assistance at k = 24 (s = {AF.S} Nm, λ = {AF.LAPSE}); PsyPref − cell in points, paired; 95 % "
         f"intervals from {R.N_BOOT} draws, users resampled within folds. Stuck: the identified curve at "
         f"k = 40 within 0.1 JND ({STUCK:.3f} Nm) of generic assistance.", ""]
    verdicts = {c: [] for c in CELLS}
    for setting in F.SETTINGS:
        ps = cell(full, setting, "psypref")
        mean = lambda v: float(np.mean([P._p(r) for r in v.values()])) if v else float("nan")  # noqa: E731
        pub = max(PUBLISHED, key=lambda c: mean(cell(full, setting, c)))
        L += [f"## {setting}", "",
              f"PsyPref {mean(ps):.3f}; FULL's best published: {F.NAMES[pub]} {mean(cell(full, setting, pub)):.3f}.", "",
              "| cell | n | preference | PsyPref − cell | cell − best published | stuck at generic |",
              "|---|---|---|---|---|---|"]
        for c in CELLS:
            v = cell(rows, setting, c)
            if not v:
                continue
            dp = P._diff(P._pairs(ps, v), rng)
            db = P._diff(P._pairs(v, cell(full, setting, pub)), rng)
            stuck = np.mean([r["d_generic"] <= STUCK for r in v.values()])
            verdicts[c].append(P._verdict(dp))
            L.append(f"| {NAMES[c]} | {len(v)} | {mean(v):.3f} | {R._f(dp, 1)} | {R._f(db, 1)} | {100 * stuck:.0f} % |")
        L.append("")
    agg = lambda vs: "held" if vs and all(v == "held" for v in vs) else ("failed" if "failed" in vs else "uncertain")  # noqa: E731
    g = verdicts["g_generic"] + verdicts["g_generic_nogate"]
    L += [f"- CMP1 (PsyPref ahead of GP-qEUBO, all settings): {agg(verdicts['gp_eubo'])} {verdicts['gp_eubo']}",
          f"- CMP2 (PsyPref ahead of CMA-ES on the ProMP, population 4): {agg(verdicts['cmaes4'])} {verdicts['cmaes4']}",
          f"- CMP3 (PsyPref ahead of Arens' GP with a generic start, gate and no gate): {agg(g)} {g}",
          f"- CMP4 (PsyPref ahead of Ramella et al.'s learner): {agg(verdicts['ramella'])} {verdicts['ramella']}", ""]
    if errors:
        L += ["## Errors", ""] + [f"- {_key(r)}: {r['error']}" for r in errors[:30]]
    (d / "amp_cmp_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
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
