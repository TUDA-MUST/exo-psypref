"""Post hoc: the robustness comparison at noise level s = 3.0 Nm, one step beyond ROB's R2.

Usage:
  python scripts/icorr_amp_s3.py run [--workers 8] [--limit N]
  python scripts/icorr_amp_s3.py report

Why: on the amplitude-free users the walking JND anchors imply s = 2.6-3.0 (paper/notes/
robustness_values_sources.md), and ROB stops at s = 2.0. Predictions: paper/notes/
figures_message_pass_2026-10-30.md, written before this script's first run. Same machinery as ROB
(icorr_amp_robust.run_one), FULL's users and answer seeds, ROB's cheap optimizers (no GP-BALD), wide
3-35 Nm and fixed 12 Nm, 40 comparisons. Outputs in <out>/icorr/amp_free/: s3_sessions.jsonl (resumable),
amp_s3_report.md.
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
import icorr_amp_range as R  # noqa: E402
import icorr_amp_robust as ROB  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402

COND = "R2b s=3.0"
S = 3.0
ROB.CONDITIONS[COND] = dict(s=S, cands=ROB.CHEAP)   # this process and its workers only


def run_one(job: dict, out: str) -> dict:
    return ROB.run_one(job, out)


def jobs(out: Path) -> list[dict]:
    return [j for j in ROB.jobs(out) if j["condition"] == COND]


def run(out: Path, workers: int, limit: int | None) -> None:
    path = ROB._dir(out) / "s3_sessions.jsonl"
    done = set()
    if path.exists():
        done = {ROB._key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = [j for j in jobs(out) if ROB._key(j) not in done]
    todo = todo if limit is None else todo[:limit]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", ROB._key(row), row["error"], flush=True)
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def report(out: Path) -> None:
    d = ROB._dir(out)
    raw = [json.loads(x) for x in (d / "s3_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {ROB._key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and ROB._key(r) not in rows]
    rng = np.random.default_rng(AF.SEED_FULL + 33)
    L = ["# Noise level s = 3.0 Nm (post hoc)", "",
         f"{len(rows)} sessions, {len(errors)} errors. 40 comparisons; preference over generic assistance with "
         f"s = {S:g} and lapse {AF.LAPSE:g}; 95 % intervals from {R.N_BOOT} draws, users resampled within folds.", "",
         "| setting | optimizer | n | k = 24 | k = 40 | PsyPref − it, k = 24 (points) |", "|---|---|---|---|---|---|"]
    for setting in ROB.SETTINGS:
        a = {r["user"]: r for (u, c, st, cc), r in rows.items() if st == setting and cc == "psypref"}
        for cand in ROB.CHEAP:
            rs = [r for (u, c, st, cc), r in rows.items() if st == setting and cc == cand]
            if not rs:
                continue
            cells = [R._f(R._boot(rs, lambda x, k=k: ROB._pref(x, k), rng), 3) for k in (24, 40)]
            diff = ""
            if cand != "psypref":
                b = {r["user"]: r for r in rs}
                ps = [dict(fold=a[u]["fold"], a=a[u], b=b[u]) for u in sorted(set(a) & set(b))]
                t = R._boot(ps, lambda x: ROB._pref([p["a"] for p in x], 24) - ROB._pref([p["b"] for p in x], 24), rng)
                diff = f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"
            L.append(f"| {setting} | {F.NAMES[cand]} | {len(rs)} | {cells[0]} | {cells[1]} | {diff} |")
    if errors:
        L += ["", "## Errors", ""] + [f"- {ROB._key(e)}: {e['error']}" for e in errors[:30]]
    (d / "amp_s3_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"run": lambda: run(args.out, args.workers, args.limit), "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
