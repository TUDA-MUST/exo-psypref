"""Post hoc: Fig. 2's comparators under the robustness conditions, so Fig. 4 measures against them everywhere.

Usage:
  python scripts/icorr_amp_var.py run [--workers 8] [--limit N]
  python scripts/icorr_amp_var.py report

Why: Fig. 4's strongest optimizer came from Arens' GP, LineCoSpar, CMA-ES and CMA-ES on Schaefer's family, while Fig. 2
measures against the stronger g_generic_nogate and cmaes4. Predictions: paper/notes/figures_message_pass_2026-10-30.md,
written before this script's first session. Both variants in both settings, in every condition Fig. 4 shows: ROB (without
R5), s = 3.0 Nm, ROB2, PAM at 3-35 Nm and PAD's two later flexion peaks. PAD ran only Arens' GP and CMA-ES on Schaefer's
family before, so LineCoSpar and CMA-ES join there. Same users, peaks and answer seeds, each condition's own run_one.
Outputs in <out>/icorr/amp_free/: var_sessions.jsonl (resumable), amp_var_report.md.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_pri as PRI  # noqa: E402
import icorr_amp_rob2 as ROB2  # noqa: E402
import icorr_amp_robust as ROB  # noqa: E402
import icorr_amp_s3 as S3  # noqa: E402

VARIANTS = ("g_generic_nogate", "cmaes4")
ROB_CONDS = ("R1 s=0.5", "R2 s=2.0", "R3 lapse=0.3", "R4 negative power", "R6 other datasets", "R7 surrogates", S3.COND)
PAD_CONDS = ("S1 flexion +6", "S2 flexion +12")
RUN = {"ROB": ROB.run_one, "ROB2": ROB2.run_one, "PRI": PRI.run_one}


def _key(r: dict) -> tuple:
    return r["block"], r["condition"], r["setting"], r["candidate"], r["user"]


def jobs(out: Path) -> list[dict]:
    js = []
    for j in ROB.jobs(out):
        if j["candidate"] == "psypref" and j["condition"] in ROB_CONDS:
            js += [dict(j, candidate=c, block="ROB") for c in VARIANTS]
    for j in ROB2.jobs(out):
        if j["candidate"] == "psypref":
            js += [dict(j, candidate=c, block="ROB2") for c in VARIANTS]
    for j in PRI.jobs(out):
        if j.get("chain") or j["candidate"] != "g":
            continue
        if j["test"] == "PAM" and j["setting"] == "wide":
            js += [dict(j, candidate=c, block="PRI") for c in VARIANTS]
        elif j["test"] == "PAD" and j["condition"] in PAD_CONDS:
            js += [dict(j, candidate=c, block="PRI") for c in VARIANTS + ("d_z", "cmaes")]
    return js


def run_one(job: dict, out: str) -> list[dict]:
    block = job["block"]
    t0 = time.perf_counter()
    got = RUN[block]({k: v for k, v in job.items() if k != "block"}, out)
    rows = got if isinstance(got, list) else [got]
    return [dict(r, block=block, session_seconds=round(time.perf_counter() - t0, 2)) for r in rows]


def _path(out: Path) -> Path:
    return out / "icorr" / "amp_free" / "var_sessions.jsonl"


def run(out: Path, workers: int, limit: int | None) -> None:
    path = _path(out)
    done = set()
    if path.exists():
        done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r.get("error") is None}
    todo = [j for j in jobs(out) if _key(j) not in done]
    todo = todo if limit is None else todo[:limit]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            for row in fut.result():
                f.write(json.dumps(row) + "\n")
                if row.get("error"):
                    print("ERROR", _key(row), row["error"], flush=True)
            f.flush()
            if n % 500 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def report(out: Path) -> None:
    rows = [json.loads(x) for x in _path(out).read_text(encoding="utf-8").splitlines()]
    want = {_key(j) for j in jobs(out)}
    ok = {_key(r) for r in rows if r.get("error") is None}
    errors = [r for r in rows if r.get("error")]
    cells = Counter((r["block"], r["candidate"]) for r in rows if r.get("error") is None)
    lines = ["# Fig. 2's comparators under the robustness conditions", "",
             f"{len(ok)} of {len(want)} sessions, {len(errors)} errors. The leads are in Fig. 4M's numbers "
             "(scripts/paper_v2_fig4m.py).", "", "| block | optimizer | sessions |", "|---|---|---|"]
    lines += [f"| {b} | {c} | {n} |" for (b, c), n in sorted(cells.items())]
    (out / "icorr" / "amp_free" / "amp_var_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.command == "run":
        run(args.out, args.workers, args.limit)
    else:
        report(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
