"""Test KS2: the comparators at K = 12 and 24.

Usage (the pinned stack; at most 8 workers, 16 on TS01):
  python scripts/icorr_amp_ks2.py smoke [--workers 8]
  python scripts/icorr_amp_ks2.py check [--workers 8]
  python scripts/icorr_amp_ks2.py run [--workers 8]
  python scripts/icorr_amp_ks2.py report

KS's users (FULL's index 0-19 per fold, their peaks and answer seeds), main condition
(s = 1.0 Nm, lambda = 0.1), wide and fixed 12, each session KS's run_one: GP-BALD and CMA-ES on the
ProMP with population 4 at K = 12 and 24 to 40 comparisons, on the K-basis spaces and priors KS gave
PsyPref; 800 sessions. K = 8 is FULL's and CMP's rows, K = 16 KS's, PsyPref LONG's and KS's. ``smoke``
compares K = 8 and K = 16 sessions of 2 users per setting with the stored rows and times one user per
new arm. Where they differ (another machine), ``check`` reruns KS's K = 16 rows, and ``run`` starts
only if they agree within CHECK_POINTS. Outputs in <out>/icorr/amp_free/: ks2_sessions.jsonl and
ks2_check_sessions.jsonl (both resumable), ks2_smoke.jsonl, amp_ks2_report.md. Stored rows are read,
never replaced.
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
import icorr_amp_ks as KS  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402

#: Section 30: the new arms, and how far a rerun of KS's K = 16 rows may lie from them (points of
#: preference, paired mean, each arm and setting, k = 24 and 40).
BASES = (12, 24)
ALL_K = (8, 12, 16, 24)
COMPARATORS = KS.COMPARATORS
NAMES = KS.NAMES
SETTINGS = KS.SETTINGS
CHECK_POINTS = 1.5
KS_REPORT = (12, 24, 40)
COMPARE = ("distances", "d0", "identified", "shape_error")


def _base(out: Path) -> list[dict]:
    return [j for j in F.jobs(out) if j["candidate"] == "psypref" and j["index"] < KS.N_USERS
            and j["setting"] in SETTINGS]


def jobs(out: Path, bases=BASES) -> list[dict]:
    """KS's users crossed with the comparators and K, the slowest arm first."""
    js = [dict(j, candidate=c, n_basis=k, n_comp=KS.N_COMP) for k in bases for c in COMPARATORS for j in _base(out)]
    return sorted(js, key=lambda j: (j["candidate"] != "gp_bald", -j["n_basis"]))


def _worker_cap() -> int:
    return 16 if platform.node().upper().startswith("TS01") else 8


def _pref(r, k) -> float:
    return F._pref_k([r], k)


# ----------------------------------------------------------------------------- smoke, check

def _stored(d: Path) -> dict:
    """K = 8 and K = 16 rows, keyed (candidate, K, user, setting)."""
    st = {(c, 8, u, s): r for (c, u, s), r in KS._reference(d).items() if c in COMPARATORS}
    st.update({KS._key(r): r for r in KS._read(d / "ks_sessions.jsonl")
               if r.get("error") is None and r["candidate"] in COMPARATORS})
    return st


def _smoke_row(d: Path) -> dict | None:
    m = KS.LONG._machine()
    rows = [r for r in KS._read(d / "ks2_smoke.jsonl") if all(r.get(k) == v for k, v in m.items())]
    return rows[-1] if rows else None


def smoke(out: Path, workers: int) -> None:
    d = F._dir(out)
    st = _stored(d)
    print(f"Machine: {KS.LONG._machine_name(KS.LONG._machine())}", flush=True)
    few = [j for j in _base(out) if j["fold_index"] == 0 and j["index"] < 2]
    known = [dict(j, candidate=c, n_comp=KS.N_COMP, part="known", n_basis=k) for c in COMPARATORS for k in (8, 16) for j in few]
    new = [dict(j, part="new") for j in jobs(out) if j["fold_index"] == 0 and j["index"] == 0]
    todo = sorted(known + new, key=lambda j: j["candidate"] != "gp_bald")
    print(f"{len(todo)} sessions, {workers} in parallel", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(KS.run_one, [{k: v for k, v in j.items() if k != "part"} for j in todo],
                            [str(out)] * len(todo)))
    print(f"all sessions done in {(time.time() - t0) / 60:.1f} min", flush=True)
    errors = [r for r in res if r.get("error")]
    identical, largest = True, 0.0
    for j, r in zip(todo, res):
        if r.get("error"):
            print(f"  {j['candidate']} K {j['n_basis']} {j['setting']}: ERROR {r['error']}", flush=True)
            continue
        if j["part"] == "known":
            s = st.get((j["candidate"], j["n_basis"], j["user"], j["setting"]))
            same = s is not None and all(r.get(k) == s[k] for k in COMPARE if k in s)
            gap = max(abs(_pref(r, k) - _pref(s, k)) * 100 for k in (24, 40)) if s else float("nan")
            identical &= same
            largest = max(largest, gap)
            print(f"  {j['candidate']:8s} K {j['n_basis']:2d} {j['setting']:8s} {j['user']}: identical {same}, "
                  f"preference differs by {gap:.2f} points | {r['seconds']:.1f} s", flush=True)
        else:
            print(f"  {j['candidate']:8s} K {j['n_basis']:2d} {j['setting']:8s}: d {r['d']} | {len(r['distances'])} "
                  f"distances | {r['seconds']:.1f} s", flush=True)
    per = {f"{j['candidate']} K{j['n_basis']} {j['setting']}": r["seconds"] for j, r in zip(todo, res)
           if j["part"] == "new" and not r.get("error")}
    total = sum(per.values()) * len(_base(out)) / len(SETTINGS)
    print(f"Projected: {total / 3600:.1f} h of sessions, {total / workers / 3600:.1f} h at {workers} workers", flush=True)
    with (d / "ks2_smoke.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(KS.LONG._machine(), identical=bool(identical), errors=len(errors), largest_points=largest,
                                workers=workers, seconds=per, when=time.strftime("%Y-%m-%d %H:%M"))) + "\n")
    if errors:
        raise SystemExit(f"{len(errors)} smoke sessions failed")


def _sessions(jobs_: list[dict], path: Path, out: Path, workers: int) -> None:
    done = {KS._key(r) for r in KS._read(path) if r.get("error") is None}
    todo = [j for j in jobs_ if KS._key(j) not in done]
    print(f"{len(todo)} sessions to run ({len(done)} done), {path.name}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(KS.run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", KS._key(row), row["error"], flush=True)
            if n % 10 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def _check_table(d: Path) -> list[dict]:
    """A rerun of KS's K = 16 rows against the stored ones, paired, one row per arm, setting and k."""
    st = _stored(d)
    new = {KS._key(r): r for r in KS._read(d / "ks2_check_sessions.jsonl") if r.get("error") is None}
    rng = np.random.default_rng(AF.SEED_FULL + 30)
    rows = []
    for c in COMPARATORS:
        for s in SETTINGS:
            ps = [dict(fold=r["fold"], a=r, b=st[key]) for key, r in sorted(new.items(), key=str)
                  if key[0] == c and key[3] == s and key in st]
            if not ps:
                continue
            for k in (24, 40):
                t = R._boot(ps, lambda x, k=k: 100 * float(np.mean([_pref(p["a"], k) - _pref(p["b"], k) for p in x])), rng)
                rows.append(dict(candidate=c, setting=s, k=k, n=len(ps), diff=t,
                                 identical=sum(all(p["a"].get(f) == p["b"][f] for f in COMPARE if f in p["b"]) for p in ps)))
    return rows


def check(out: Path, workers: int) -> None:
    d = F._dir(out)
    row = _smoke_row(d)
    if row is None:
        raise SystemExit("run smoke on this machine first")
    if row["identical"]:
        print("smoke's sessions are identical to the stored rows; no check needed", flush=True)
        return
    _sessions(jobs(out, (KS.K_CMP,)), d / "ks2_check_sessions.jsonl", out, workers)
    for r in _check_table(d):
        print(f"  {r['candidate']:8s} {r['setting']:8s} k {r['k']}: this machine - stored {R._f(r['diff'])} points "
              f"(n {r['n']}, identical {r['identical']})", flush=True)


def _may_run(out: Path) -> bool:
    d = F._dir(out)
    row = _smoke_row(d)
    if row is None:
        return False
    if row["identical"]:
        return True
    table, n = _check_table(d), len(_base(out)) // len(SETTINGS)
    return (len(table) == len(COMPARATORS) * len(SETTINGS) * 2 and all(r["n"] == n for r in table)
            and all(abs(r["diff"][0]) <= CHECK_POINTS for r in table))


def run(out: Path, workers: int) -> None:
    d = F._dir(out)
    if not _may_run(out):
        raise SystemExit(f"smoke and check have not passed on {KS.LONG._machine_name(KS.LONG._machine())}")
    _sessions(jobs(out), d / "ks2_sessions.jsonl", out, workers)


# ----------------------------------------------------------------------------- report

def report(out: Path) -> None:
    d = F._dir(out)
    raw = KS._read(d / "ks2_sessions.jsonl")
    ok = {KS._key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and KS._key(r) not in ok]
    ks = {KS._key(r): r for r in KS._read(d / "ks_sessions.jsonl") if r.get("error") is None}
    ref = KS._reference(d)
    users = {(j["user"], j["setting"]): j for j in _base(out)}
    rng = np.random.default_rng(AF.SEED_FULL + 30)

    def cell(cand, k, s) -> dict:
        if k == 8:
            return {u: r for (c, u, ss), r in ref.items() if c == cand and ss == s and (u, s) in users
                    and r["answer_seed"] == users[u, s]["answer_seed"] and r["amps"] == users[u, s]["amps"]}
        rows = ok if (cand != "psypref" and k in BASES) else ks
        return {u: r for (c, kk, u, ss), r in rows.items() if c == cand and kk == k and ss == s}

    def paired(a, b, k):
        ps = KS._pairs(a, b)
        return R._boot(ps, lambda x: 100 * (F._pref_k([p["a"] for p in x], k) - F._pref_k([p["b"] for p in x], k)), rng)

    machines = sorted({KS.LONG._machine_name(r) for r in ok.values()})
    L = ["# Test plan KS2: the comparators at K = 12 and 24", "",
         f"Design note 30. {len(ok)} sessions, {len(errors)} errors. KS's users, main condition (s = {AF.S} Nm, "
         f"λ = {AF.LAPSE}), wide and fixed 12. K = 8 is FULL's and CMP's rows, K = 16 KS's, PsyPref LONG's and KS's. "
         "95 % intervals from 2,000 fold-stratified draws.", "", "## Checks", ""]
    L += [f"- Sessions run on: {m}" for m in machines]
    for r in KS._read(d / "ks2_smoke.jsonl"):
        L.append(f"- Smoke on {KS.LONG._machine_name(r)}, {r['when']}: identical to the stored rows {r['identical']}, "
                 f"largest difference {r['largest_points']:.2f} points, {r['errors']} errors")
    table = _check_table(d)
    if table:
        L += ["", "KS's K = 16 rows rerun on the new machine, paired, in points of preference:", "",
              "| comparator | setting | k | n | this machine − stored | sessions identical |", "|---|---|---|---|---|---|"]
        L += [f"| {NAMES[r['candidate']]} | {r['setting']} | {r['k']} | {r['n']} | {R._f(r['diff'])} | {r['identical']} |" for r in table]

    res = {}
    L += ["", "## Preference over generic assistance by K", "",
          "Mean preference (s = 1, λ = 0.1) and the paired difference against K = 8, in points.", "",
          "| setting | arm | K | n | k = 12 | k = 24 | k = 40 | Δ k = 24 vs K = 8 | Δ k = 40 vs K = 8 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in SETTINGS:
        for c in ("psypref",) + COMPARATORS:
            c8 = cell(c, 8, s)
            for k in ALL_K:
                ck = cell(c, k, s)
                if not ck:
                    L.append(f"| {s} | {NAMES[c]} | {k} | 0 | – | – | – | – | – |")
                    continue
                means = [F._pref_k(list(ck.values()), kk) for kk in KS_REPORT]
                res["mean", c, s, k] = dict(zip(KS_REPORT, means))
                diffs = []
                for kk in (24, 40):
                    if k == 8:
                        diffs.append("–")
                        continue
                    t = paired(ck, c8, kk)
                    res["vs8", c, s, k, kk] = t
                    diffs.append(R._f(t, 1))
                L.append(f"| {s} | {NAMES[c]} | {k} | {len(ck)} | " + " | ".join(f"{m:.3f}" for m in means) + " | " + " | ".join(diffs) + " |")

    L += ["", "## PsyPref minus each comparator at the same K", "", "Paired, in points of preference.", "",
          "| setting | comparator | K | n | k = 24 | k = 40 |", "|---|---|---|---|---|---|"]
    for s in SETTINGS:
        for c in COMPARATORS:
            for k in ALL_K:
                a, b = cell("psypref", k, s), cell(c, k, s)
                if not a or not b:
                    continue
                ts = [paired(a, b, kk) for kk in (24, 40)]
                res["lead", c, s, k] = dict(zip((24, 40), ts))
                L.append(f"| {s} | {NAMES[c]} | {k} | {len(KS._pairs(a, b))} | {R._f(ts[0], 1)} | {R._f(ts[1], 1)} |")

    L += ["", "## Seconds per comparison (median over sessions; mean of the 5 comparisons around k)", "",
          "| arm | K | setting | machine | n | near k = 24 | near k = 40 | session (min) |", "|---|---|---|---|---|---|---|---|"]
    for c in COMPARATORS:
        for k in BASES:
            for s in SETTINGS:
                rows = list(cell(c, k, s).values())
                if rows:
                    L.append(f"| {NAMES[c]} | {k} | {s} | {rows[0].get('host', '?')} | {len(rows)} | "
                             f"{np.median([KS.LONG._sec_near(r, 24) for r in rows]):.3f} | {np.median([KS.LONG._sec_near(r, 40) for r in rows]):.3f} | "
                             f"{np.median([r['seconds'] for r in rows]) / 60:.1f} |")

    def verdict(flag):
        return "held" if flag else "failed"

    L += ["", "## Predictions (section 30)", ""]
    if len(ok) == len(jobs(out)):
        pk1 = [res["vs8", "gp_bald", s, k, kk][0] >= -3 for s in SETTINGS for k in BASES for kk in (24, 40)]
        wide = {k: res["mean", "cmaes4", "wide", k][40] for k in ALL_K}
        fixed = {k: res["mean", "cmaes4", "fixed 12", k][40] for k in ALL_K}
        pk2 = wide[8] > wide[12] > wide[16] > wide[24] and fixed[24] < fixed[8]
        pk3 = [res["lead", c, s, k][24][1] > 0 for c in COMPARATORS for s in SETTINGS for k in BASES]
        lead_cma = [res["lead", "cmaes4", "wide", k][40][0] for k in ALL_K]
        lead_gp = {s: [res["lead", "gp_bald", s, k][40][0] for k in ALL_K] for s in SETTINGS}
        pk4 = all(x < y for x, y in zip(lead_cma, lead_cma[1:])) and all(max(v) - min(v) <= 3 for v in lead_gp.values())
        L += [f"- PK1 (GP-BALD at K = 12 and 24, at most 3 points below K = 8, k = 24 and 40, each setting): {verdict(all(pk1))} ({sum(pk1)}/{len(pk1)})",
              f"- PK2 (CMA-ES, population 4: wide, k = 40, K = 8 > 12 > 16 > 24; fixed 12, K = 24 below K = 8): {verdict(pk2)} — wide "
              + ", ".join(f"K {k} {wide[k]:.3f}" for k in ALL_K) + f"; fixed 12 K 8 {fixed[8]:.3f}, K 24 {fixed[24]:.3f}",
              f"- PK3 (PsyPref ahead of both at K = 12 and 24, k = 24, paired intervals above 0, each setting): {verdict(all(pk3))} ({sum(pk3)}/{len(pk3)})",
              f"- PK4 (wide, k = 40: the lead over CMA-ES rises with K; the lead over GP-BALD spans at most 3 points, each setting): {verdict(pk4)} — CMA-ES "
              + ", ".join(f"{x:+.1f}" for x in lead_cma) + "; GP-BALD " + "; ".join(f"{s} " + ", ".join(f"{x:+.1f}" for x in v) for s, v in lead_gp.items())]
    else:
        L.append(f"Not judged: {len(ok)} of {len(jobs(out))} sessions.")
    (d / "amp_ks2_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("smoke", "check", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > _worker_cap():
        raise SystemExit(f"at most {_worker_cap()} workers on this machine")
    {"smoke": lambda: smoke(args.out, args.workers), "check": lambda: check(args.out, args.workers),
     "run": lambda: run(args.out, args.workers), "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
