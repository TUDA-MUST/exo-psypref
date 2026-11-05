"""Test PW: the prior's width against speed.

Usage:
  python scripts/icorr_amp_pw.py smoke [--workers 8]
  python scripts/icorr_amp_pw.py run [--workers 8]
  python scripts/icorr_amp_pw.py report

FULL's PsyPref jobs (users, preferred profiles, answer seeds, all three settings) with
z* ~ N(0, c^2 I) and c times the search radius, c = 1.5 and 2; everything else is FULL's, so each
session is paired with FULL's PsyPref (c = 1) and ABL's (c = 3). The arms are planned here at
run time, arms.py stays untouched. Outputs in <out>/icorr/amp_free/: pw_sessions.jsonl (resumable),
pw_smoke.jsonl, amp_pw_report.md.
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
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_full as F  # noqa: E402
import icorr_amp_long as LONG  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.learning.arms import ARMS, Arm  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402

WIDTHS = (1.5, 2.0)
ALL_WIDTHS = (1.0, 1.5, 2.0, 3.0)
JND = true_jnd(AF.S)
REPORT_K = (12, 24, 40)
COMPARE = ("distances", "d0", "identified", "shape_error")


def _arm(c: float) -> str:
    return f"c{c:g}"


for _c in WIDTHS:   # module top level: spawned workers re-run it
    ARMS.setdefault(_arm(_c), Arm(_arm(_c), scale=_c ** 2))


def run_one(job: dict, out: str) -> dict:
    t0 = time.perf_counter()
    row = F.run_one(dict(job), out)
    row["seconds"] = round(time.perf_counter() - t0, 3)
    row.update(LONG._machine())
    return row


def jobs(out: Path) -> list[dict]:
    return [dict(j, arm=_arm(c), width=c) for c in WIDTHS for j in F.jobs(out) if j["candidate"] == "psypref"]


def _key(row):
    return row["user"], row["setting"], row["width"]


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def _stored(d: Path) -> dict:
    """c = 1 (FULL's PsyPref) and c = 3 (ABL's), keyed (user, setting, width)."""
    st = {(r["user"], r["setting"], 1.0): r for r in _read(d / "full_sessions.jsonl")
          if r.get("error") is None and r["candidate"] == "psypref"}
    st.update({(r["user"], r["setting"], 3.0): r for r in _read(d / "abl_sessions.jsonl") if r.get("error") is None})
    return st


def _smoke_passed(d: Path) -> bool:
    m = LONG._machine()
    return any(r["identical"] and all(r.get(k) == v for k, v in m.items()) for r in _read(d / "pw_smoke.jsonl"))


def smoke(out: Path, workers: int) -> None:
    d = F._dir(out)
    st = _stored(d)
    print(f"Machine: {LONG._machine_name(LONG._machine())}", flush=True)
    few = [j for j in F.jobs(out) if j["candidate"] == "psypref" and j["fold_index"] == 0 and j["index"] < 2]
    known = [dict(j, arm="tight", width=1.0) for j in few] + [dict(j, arm="wide", width=3.0) for j in few]
    new = [j for j in jobs(out) if j["fold_index"] == 0 and j["index"] == 0]
    todo = known + new
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(run_one, todo, [str(out)] * len(todo)))
    identical, errors, seconds = True, 0, []
    for j, r in zip(todo, res):
        if r.get("error"):
            errors += 1
            print(f"  {j['setting']} c {j['width']:g}: ERROR {r['error']}", flush=True)
            continue
        seconds.append(r["seconds"])
        s = st.get(_key(j))
        if j["width"] in WIDTHS:
            print(f"  {j['setting']:8s} c {j['width']:g}: new, radius run, P_g(24) {100 * F._pref_k([r], 24):.1f} | {r['seconds']:.1f} s", flush=True)
        else:
            same = s is not None and all(r.get(k) == s[k] for k in COMPARE if k in s)
            identical &= same
            print(f"  {j['setting']:8s} c {j['width']:g} {j['user']}: identical {same} | {r['seconds']:.1f} s", flush=True)
    total = float(np.mean(seconds)) * len(jobs(out)) if seconds else float("nan")
    print(f"Projected: {total / 60:.0f} min of sessions, {total / workers / 60:.0f} min at {workers} workers", flush=True)
    with (d / "pw_smoke.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(LONG._machine(), identical=bool(identical) and errors == 0, errors=errors,
                                when=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
    if errors or not identical:
        raise SystemExit("smoke did not pass")


def run(out: Path, workers: int) -> None:
    d = F._dir(out)
    if not _smoke_passed(d):
        raise SystemExit(f"smoke has not passed on {LONG._machine_name(LONG._machine())}")
    path = d / "pw_sessions.jsonl"
    done = {_key(r) for r in _read(path) if r.get("error") is None}
    todo = [j for j in jobs(out) if _key(j) not in done]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", _key(row), row["error"], flush=True)
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- report

def _curve(r) -> np.ndarray:
    return AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((r["d0"] - np.asarray(r["distances"], dtype=float)) / AF.S)


def _first_reach(ref: np.ndarray, level: float) -> float:
    hit = np.nonzero(ref >= level)[0]
    return float(hit[0] + 1) if hit.size else float(ref.size + 1)


def report(out: Path) -> None:
    d = F._dir(out)
    raw = _read(d / "pw_sessions.jsonl")
    ok = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in ok]
    rows = {**_stored(d), **ok}
    rng = np.random.default_rng(AF.SEED_FULL + 31)
    L = ["# Test plan PW: the prior's width against speed", "",
         f"Design note 31. {len(ok)} sessions, {len(errors)} errors. PsyPref with z* ~ N(0, c² I) and c times the "
         f"search radius, on FULL's users and answer seeds; c = 1 is FULL's PsyPref, c = 3 ABL's. s = {AF.S} Nm, "
         f"λ = {AF.LAPSE}. 95 % intervals from {R.N_BOOT} draws, users resampled within folds. First k: the first k at "
         "which c = 1 reaches this width's level at k = 40.", "",
         "| setting | c | n | k = 12 | k = 24 | k = 40 | Δ k = 24 vs c = 1 | Δ k = 40 vs c = 1 | median distance k = 24 (JND) | first k |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for s in F.SETTINGS:
        base = {u: r for (u, ss, c), r in rows.items() if ss == s and c == 1.0}
        for c in ALL_WIDTHS:
            cell = {u: r for (u, ss, cc), r in rows.items() if ss == s and cc == c}
            if not cell:
                continue
            ps = [dict(fold=base[u]["fold"], a=_curve(cell[u]), b=_curve(base[u]), d=cell[u]["distances"][23])
                  for u in sorted(set(cell) & set(base))]
            a = np.array([p["a"] for p in ps])
            means = {k: float(a[:, k - 1].mean()) for k in REPORT_K}
            res["mean", s, c] = means
            dist = R._boot(ps, lambda x: float(np.median([p["d"] for p in x])) / JND, rng)
            if c == 1.0:
                L.append(f"| {s} | 1 | {len(ps)} | " + " | ".join(f"{means[k]:.3f}" for k in REPORT_K) + f" | – | – | {R._f(dist)} | – |")
                continue
            dk = {k: R._boot(ps, lambda x, k=k: 100 * float(np.mean([p["a"][k - 1] - p["b"][k - 1] for p in x])), rng) for k in (24, 40)}
            fk = R._boot(ps, lambda x: _first_reach(np.mean([p["b"] for p in x], axis=0), float(np.mean([p["a"][39] for p in x]))), rng)
            res["vs", s, c] = dk
            res["first", s, c] = fk
            L.append(f"| {s} | {c:g} | {len(ps)} | " + " | ".join(f"{means[k]:.3f}" for k in REPORT_K)
                     + f" | {R._f(dk[24], 1)} | {R._f(dk[40], 1)} | {R._f(dist)} | {fk[0]:.0f} [{fk[1]:.0f}, {fk[2]:.0f}] |")
    L += ["", "## Predictions (section 31, each setting, point estimates)", ""]
    if len(ok) == len(jobs(out)):
        def verdict(flag):
            return "held" if flag else "failed"

        pw1 = [all(res["mean", s, a][24] > res["mean", s, b][24] for a, b in zip(ALL_WIDTHS, ALL_WIDTHS[1:])) for s in F.SETTINGS]
        cost = {(s, c): -res["vs", s, c][24][0] for s in F.SETTINGS for c in WIDTHS}
        pw2 = [4 <= cost[s, 1.5] <= 10 and 8 <= cost[s, 2.0] <= 16 for s in F.SETTINGS]
        pw3 = [all(res["vs", s, c][40][0] > res["vs", s, c][24][0] for c in WIDTHS) for s in F.SETTINGS]
        pw4 = [10 <= res["first", s, 1.5][0] <= 30 and 7 <= res["first", s, 2.0][0] <= 20 for s in F.SETTINGS]
        L += [f"- PW1 (k = 24, preference falls with width): {verdict(all(pw1))} {pw1}",
              f"- PW2 (k = 24, c = 1.5 costs 4–10 points, c = 2 costs 8–16): {verdict(all(pw2))} {pw2} — "
              + "; ".join(f"{s} {cost[s, 1.5]:.1f}, {cost[s, 2.0]:.1f}" for s in F.SETTINGS),
              f"- PW3 (both widths cost less at k = 40 than at k = 24): {verdict(all(pw3))} {pw3}",
              f"- PW4 (first k within 10–30 for c = 1.5, 7–20 for c = 2): {verdict(all(pw4))} {pw4} — "
              + "; ".join(f"{s} {res['first', s, 1.5][0]:.0f}, {res['first', s, 2.0][0]:.0f}" for s in F.SETTINGS)]
    else:
        L.append(f"Not judged: {len(ok)} of {len(jobs(out))} sessions.")
    if errors:
        L += ["", "## Errors", ""] + [f"- {_key(r)}: {r['error']}" for r in errors[:20]]
    (d / "amp_pw_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("smoke", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    if args.command == "report":
        report(args.out)
    else:
        (smoke if args.command == "smoke" else run)(args.out, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
