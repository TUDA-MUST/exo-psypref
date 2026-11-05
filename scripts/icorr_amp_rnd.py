"""Test RND: PsyPref with random pairs, the informative pairs' share of its gain.

Usage:
  python scripts/icorr_amp_rnd.py run [--workers 8]
  python scripts/icorr_amp_rnd.py report

FULL's PsyPref jobs (users, preferred profiles, answer
seeds, all three settings) with ``duels = "random"``; everything else is FULL's, so each session
is paired with FULL's PsyPref and FULL's GP-BALD. The report sets ABL's prior ablation beside it.
Outputs in <out>/icorr/amp_free/: rnd_sessions.jsonl (resumable), amp_rnd_report.md.
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

import icorr_amp_abl as ABL  # noqa: E402
import icorr_amp_full as F  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402

DUELS = "random"


def jobs(out: Path) -> list[dict]:
    return [dict(j, duels=DUELS) for j in F.jobs(out) if j["candidate"] == "psypref"]


def _key(row):
    return row["user"], row["setting"]


def run(out: Path, workers: int) -> None:
    path = F._dir(out) / "rnd_sessions.jsonl"
    done = set()
    if path.exists():
        done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = [j for j in jobs(out) if _key(j) not in done]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(F.run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", _key(row), row["error"], flush=True)
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- report

def _load(path: Path, cand: str | None = None) -> dict:
    return {(r["user"], r["setting"]): r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if r.get("error") is None and (cand is None or r["candidate"] == cand)}


def report(out: Path) -> None:
    d = F._dir(out)
    raw = [json.loads(x) for x in (d / "rnd_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    rnd = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in rnd]
    full = _load(d / "full_sessions.jsonl", "psypref")
    bald = _load(d / "full_sessions.jsonl", "gp_bald")
    abl = _load(d / "abl_sessions.jsonl")
    rng = np.random.default_rng(AF.SEED_FULL + 25)
    mean = ABL._mean
    L = ["# Test plan RND: the informative pairs' share of PsyPref's gain", "",
         f"Design note 25 (c945a5e). {len(rnd)} sessions, {len(errors)} errors. PsyPref with random "
         f"pairs against FULL's PsyPref and GP-BALD on the same users and answer seeds. s = {AF.S} Nm, "
         f"λ = {AF.LAPSE}. 95 % intervals from {R.N_BOOT} draws, users resampled within folds.", "",
         "| setting | k | PsyPref (FULL) | random pairs | random − PsyPref (points) | share of gain kept "
         "| GP-BALD − random (points) |", "|---|---|---|---|---|---|---|"]
    res = {}
    for s in F.SETTINGS:
        ps = [dict(fold=rnd[k]["fold"], a=rnd[k], b=full[k], c=bald[k])
              for k in sorted(rnd) if k[1] == s and k in full and k in bald]
        if not ps:
            continue
        for k in (24, 40):
            p_full = R._boot(ps, lambda x: mean(x, "b", k), rng)
            p_rnd = R._boot(ps, lambda x: mean(x, "a", k), rng)
            diff = R._boot(ps, lambda x: mean(x, "a", k) - mean(x, "b", k), rng)
            share = R._boot(ps, lambda x: (mean(x, "a", k) - 0.5) / (mean(x, "b", k) - 0.5), rng)
            vs_bald = R._boot(ps, lambda x: mean(x, "c", k) - mean(x, "a", k), rng)
            res[s, k] = dict(diff=diff, vs_bald=vs_bald)
            L.append(f"| {s} | {k} | {R._f(p_full, 3)} | {R._f(p_rnd, 3)} | "
                     f"{R._f(tuple(100 * v for v in diff), 1)} | {R._f(share, 2)} | "
                     f"{R._f(tuple(100 * v for v in vs_bald), 1)} |")
    # Decomposition at k = 24: each a paired difference against FULL's PsyPref (not additive)
    L += ["", "## Side by side at k = 24 (points, each against FULL's PsyPref; not additive)", "",
          "| setting | wide prior (ABL) | random pairs (RND) | GP-BALD (same prior, its pairs and model) |",
          "|---|---|---|---|"]
    for s in F.SETTINGS:
        cells = []
        for other in (abl, rnd, bald):
            ps = [dict(fold=full[k]["fold"], a=other[k], b=full[k]) for k in sorted(full)
                  if k[1] == s and k in other]
            cells.append(R._f(tuple(100 * v for v in R._boot(ps, lambda x: mean(x, "a", 24) - mean(x, "b", 24), rng)), 1)
                         if ps else "–")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    v1 = [ABL._verdict(res[s, 24]["diff"], "below") for s in F.SETTINGS if (s, 24) in res]
    v2 = [res[s, 24]["diff"][0] <= -0.05 for s in F.SETTINGS if (s, 24) in res]
    v3 = [ABL._verdict(res[s, 24]["vs_bald"], "above") for s in F.SETTINGS if (s, 24) in res]
    prn1 = "held" if v1 and all(v == "held" for v in v1) else (
        "failed" if any(v == "failed" for v in v1) else "uncertain")
    prn2 = "held" if sum(v2) >= 2 else "failed"
    prn3 = "held" if v3.count("held") >= 2 else ("failed" if v3.count("failed") >= 2 else "uncertain")
    L += ["", "## Predictions (k = 24)", "",
          f"- PRN1 (random pairs lower PsyPref's preference in each setting): {prn1} {v1}",
          f"- PRN2 (loss at least 5 points in at least two settings): {prn2} "
          f"{[round(100 * res[s, 24]['diff'][0], 1) for s in F.SETTINGS if (s, 24) in res]}",
          f"- PRN3 (GP-BALD ahead of PsyPref with random pairs in at least two settings): {prn3} {v3}", ""]
    if errors:
        L += ["## Errors", ""] + [f"- {_key(r)}: {r['error']}" for r in errors[:20]]
    (d / "amp_rnd_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
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
    run(args.out, args.workers) if args.command == "run" else report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
