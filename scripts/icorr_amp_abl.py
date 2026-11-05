"""Test ABL: PsyPref with the wide prior arm, the data prior's share of its gain.

Usage:
  python scripts/icorr_amp_abl.py run [--workers 8]
  python scripts/icorr_amp_abl.py report

FULL's PsyPref jobs (users, preferred profiles,
answer seeds, all three settings) with ``arm = "wide"``: z* ~ N(0, 9 I) and three times the
search radius; everything else is FULL's, so each session is paired with FULL's PsyPref.
Outputs in <out>/icorr/amp_free/: abl_sessions.jsonl (resumable), amp_abl_report.md.
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
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_full as F  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402

ARM = "wide"
JND = true_jnd(AF.S)


def jobs(out: Path) -> list[dict]:
    return [dict(j, arm=ARM) for j in F.jobs(out) if j["candidate"] == "psypref"]


def _key(row):
    return row["user"], row["setting"]


def run(out: Path, workers: int) -> None:
    path = F._dir(out) / "abl_sessions.jsonl"
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

def _pref(r, k: int) -> float:
    return float(AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((r["d0"] - r["distances"][k - 1]) / AF.S))


def _mean(pairs, side, k):
    return float(np.mean([_pref(p[side], k) for p in pairs]))


def _verdict(t, side: str, bound: float = 0.0) -> str:
    est, lo, hi = t
    if side == "above":
        return "held" if lo > bound else ("failed" if hi < bound else "uncertain")
    return "held" if hi < bound else ("failed" if lo > bound else "uncertain")


def report(out: Path) -> None:
    d = F._dir(out)
    raw = [json.loads(x) for x in (d / "abl_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    wide = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in wide]
    full = {(r["user"], r["setting"]): r
            for r in map(json.loads, (d / "full_sessions.jsonl").read_text(encoding="utf-8").splitlines())
            if r.get("error") is None and r["candidate"] == "psypref"}
    rng = np.random.default_rng(AF.SEED_FULL + 20)
    pairs = {s: [dict(fold=wide[k]["fold"], user=k[0], a=wide[k], b=full[k])
                 for k in sorted(wide) if k[1] == s and k in full] for s in F.SETTINGS}
    lines = ["# Registration ABL: the data prior's share of PsyPref's gain", "",
             f"Spec section 20 (714937c). {len(wide)} sessions, {len(errors)} errors. PsyPref with "
             f"the wide arm (z* ~ N(0, 9 I), three times the search radius) against FULL's PsyPref "
             f"on the same users and answer seeds. s = {AF.S} Nm, λ = {AF.LAPSE}. 95 % intervals "
             f"from {R.N_BOOT} draws, users resampled within folds.", "",
             "| setting | k | PsyPref (FULL) | wide prior | wide − PsyPref (points) | share of gain kept "
             "| median distance, wide prior (JND) | PsyPref (JND) |",
             "|---|---|---|---|---|---|---|---|"]
    res = {}
    for s, ps in pairs.items():
        if not ps:
            continue
        for k in (24, 40):
            p_full = R._boot(ps, lambda x: _mean(x, "b", k), rng)
            p_wide = R._boot(ps, lambda x: _mean(x, "a", k), rng)
            diff = R._boot(ps, lambda x: _mean(x, "a", k) - _mean(x, "b", k), rng)
            share = R._boot(ps, lambda x: (_mean(x, "a", k) - 0.5) / (_mean(x, "b", k) - 0.5), rng)
            med_w = R._boot(ps, lambda x: float(np.median([p["a"]["distances"][k - 1] for p in x])) / JND, rng)
            med_f = R._boot(ps, lambda x: float(np.median([p["b"]["distances"][k - 1] for p in x])) / JND, rng)
            res[s, k] = dict(diff=diff, share=share)
            lines.append(f"| {s} | {k} | {R._f(p_full, 3)} | {R._f(p_wide, 3)} | "
                         f"{R._f(tuple(100 * v for v in diff), 1)} | {R._f(share, 2)} | "
                         f"{R._f(med_w, 2)} | {R._f(med_f, 2)} |")
        g = [p["a"].get("guard_fired", 0) > 0 for p in ps if "guard_fired" in p["a"]]
        if g:
            lines.append(f"| {s} | guard fired in {np.mean(g):.0%} of sessions | | | | | | |")
    # PAB3: the loss at the fixed level minus the loss in the wide range, paired over users
    lf = {p["user"]: p for p in pairs.get("fixed 12", [])}
    lw = {p["user"]: p for p in pairs.get("wide", [])}
    both = [dict(fold=lf[u]["fold"], f=lf[u], w=lw[u]) for u in sorted(set(lf) & set(lw))]

    def _loss_gap(x):
        loss_f = np.mean([_pref(p["f"]["b"], 24) - _pref(p["f"]["a"], 24) for p in x])
        loss_w = np.mean([_pref(p["w"]["b"], 24) - _pref(p["w"]["a"], 24) for p in x])
        return float(loss_f - loss_w)

    gap = R._boot(both, _loss_gap, rng) if both else None
    v1 = [_verdict(res[s, 24]["diff"], "below") for s in F.SETTINGS if (s, 24) in res]
    v2 = [_verdict(res[s, 24]["share"], "below", 0.5) for s in F.SETTINGS if (s, 24) in res]
    pab1 = "held" if v1 and all(v == "held" for v in v1) else (
        "failed" if any(v == "failed" for v in v1) else "uncertain")
    n_held, n_failed = v2.count("held"), v2.count("failed")
    pab2 = "held" if n_held >= 2 else ("failed" if n_failed >= 2 else "uncertain")
    pab3 = _verdict(gap, "above") if gap else "incomplete"
    lines += ["", "## Predictions (k = 24)", "",
              f"- PAB1 (the wide prior lowers PsyPref's preference in each setting): {pab1} {v1}",
              f"- PAB2 (keeps less than half of the gain in at least two settings): {pab2} {v2}",
              f"- PAB3 (loss larger at fixed 12 Nm than in the wide range): {pab3}"
              + (f" — {R._f(tuple(100 * v for v in gap), 1)} points" if gap else ""), ""]
    if errors:
        lines += ["## Errors", ""] + [f"- {_key(r)}: {r['error']}" for r in errors[:20]]
    (d / "amp_abl_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


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
