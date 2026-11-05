"""Test TIE: PsyPref-T ("no preference" answers as data) on users disjoint from its screen.

Usage:
  python scripts/icorr_amp_tie.py smoke
  python scripts/icorr_amp_tie.py run [--workers 7]
  python scripts/icorr_amp_tie.py report

ROB2's PsyPref jobs in C4 pessimistic and A5 no
preference, users with index 8-39, both settings, in four arms: plain PsyPref (rerun; must
reproduce the stored rows) and PsyPref-T (prompref.learning.ties, mode "band") with the band
prior's median at 0.5, 0.25 and 1.0. ROB2's run_one is unchanged; the learner
factories are swapped per job inside the worker. Each session is paired with ROB2's stored
PsyPref, CMA-ES (ProMP) and CMA-ES (Schäfer) sessions.
Outputs in <out>/icorr/amp_free/: tie_sessions.jsonl (resumable), amp_tie_report.md.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_abl as ABL  # noqa: E402
import icorr_amp_full as F  # noqa: E402
import icorr_amp_range as R  # noqa: E402
import icorr_amp_rob2 as RB  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.learning import ties as T  # noqa: E402

CONDS = ("C4 pessimistic", "A5 no preference")
#: Arm -> the band prior's median (noise units); None is plain PsyPref.
ARMS = {"base": None, "tie": 0.5, "tie_lo": 0.25, "tie_hi": 1.0}
MIN_INDEX = 8          # the screen used index < 8
_BASE = (AF.psypref, AF.metric_psypref)
_LOG_TAU0 = T.LOG_TAU0

# ----------------------------------------------------------------------------- per-session capture
_CAP: dict = {}


def _wrap(orig):
    """ROB2's run_session, recording per comparison [RMS Nm between the two curves, answer, exact
    repeat of the previous pair in either order] and the learner's end state. Feeds nothing back."""
    def _rs(design, learner, observer, rng, *a, **k):
        log, prev = [], [None]
        respond0 = observer.respond

        def respond(ca, cb, r, order=1):
            resp = respond0(ca, cb, r, order=order)
            ca_, cb_ = np.asarray(ca, dtype=float), np.asarray(cb, dtype=float)
            p = prev[0]
            rep = p is not None and ((np.array_equal(ca_, p[0]) and np.array_equal(cb_, p[1]))
                                     or (np.array_equal(ca_, p[1]) and np.array_equal(cb_, p[0])))
            prev[0] = (ca_.copy(), cb_.copy())
            log.append([float(np.sqrt(np.mean((ca_ - cb_) ** 2))), int(resp), int(bool(rep))])
            return resp

        observer.respond = respond
        rec = orig(design, learner, observer, rng, *a, **k)
        inner = getattr(learner, "learner", learner)
        post = inner.posterior
        _CAP.update(log=log, nopref=int(getattr(inner, "n_no_preference", 0)),
                    lam=float(post.lambda_mean()), ell=float(post.ell_mean()),
                    tau=(post.tau_mean() if hasattr(post, "tau_mean") else None),
                    tie_activated_at=getattr(inner, "tie_activated_at", None),
                    learner_cls=type(inner).__name__)
        return rec
    return _rs


if not getattr(RB.run_session, "_tie_wrapped", False):   # module top level: spawned workers re-run it
    RB.run_session = _wrap(RB.run_session)
    RB.run_session._tie_wrapped = True


def run_one(job: dict, out: str) -> dict:
    tau0 = ARMS[job["arm"]]
    _CAP.clear()
    try:
        if tau0 is not None:
            T.LOG_TAU0 = float(np.log(tau0))
            AF.psypref, AF.metric_psypref = T.factories("band")
        t0 = time.time()
        row = RB.run_one(job, out)
        row["seconds"] = time.time() - t0
    finally:
        AF.psypref, AF.metric_psypref = _BASE
        T.LOG_TAU0 = _LOG_TAU0
    row.update(_CAP)
    return row


def jobs(out: Path) -> list[dict]:
    base = [j for j in RB.jobs(out) if j["condition"] in CONDS and j["candidate"] == "psypref"
            and j["index"] >= MIN_INDEX]
    return [dict(j, arm=a) for a in ARMS for j in base]


def _key(row):
    return row["user"], row["condition"], row["setting"], row["arm"]


def run(out: Path, workers: int) -> None:
    path = RB._dir(out) / "tie_sessions.jsonl"
    done = set()
    if path.exists():
        done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
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
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- smoke

def smoke(out: Path) -> None:
    """The port: PsyPref-T on two screen users must equal the screen's rows bit for bit; plain
    PsyPref on two held-out users must equal ROB2's stored rows."""
    d = RB._dir(out)
    screen = Path(os.environ.get("TIE_SCREEN_ROWS", ""))
    stored = _rob2(d)
    js = [j for j in RB.jobs(out) if j["condition"] == "C4 pessimistic" and j["candidate"] == "psypref"
          and j["fold_index"] == 0]
    if screen.is_file():
        ref = {(r["user"], r["setting"]): r for r in map(json.loads, screen.read_text(encoding="utf-8").splitlines())
               if r.get("mode") == "band" and not r.get("error")}
        for j in [j for j in js if j["index"] == 0]:
            r = run_one(dict(j, arm="tie"), str(out))
            print(f"tie  {j['setting']:8s} {j['user']}: err {r['error']} | own == screen "
                  f"{r['own'] == ref[(j['user'], j['setting'])]['own']} | ties {r['nopref']} tau {r['tau']} "
                  f"| {r['learner_cls']} | {r['seconds']:.0f} s", flush=True)
    else:
        print("TIE_SCREEN_ROWS not set: port check skipped", flush=True)
    for j in [j for j in js if j["index"] == MIN_INDEX]:
        r = run_one(dict(j, arm="base"), str(out))
        print(f"base {j['setting']:8s} {j['user']}: err {r['error']} | own == stored "
              f"{r['own'] == stored[('C4 pessimistic', j['setting'], 'psypref')][j['user']]['own']} "
              f"| ties {r['nopref']} | {r['learner_cls']} | {r['seconds']:.0f} s", flush=True)


# ----------------------------------------------------------------------------- report

def _rob2(d: Path) -> dict:
    rows = {}
    for r in map(json.loads, (d / "rob2_sessions.jsonl").read_text(encoding="utf-8").splitlines()):
        if r.get("error") is None and r["condition"] in CONDS and r["index"] >= MIN_INDEX:
            rows.setdefault((r["condition"], r["setting"], r["candidate"]), {})[r["user"]] = r
    return rows


def _post_tie(rows):
    """(exact repeats of the previous pair right after a tie, comparisons right after a tie)."""
    rep = tot = 0
    for r in rows:
        lg = r["log"]
        for i in range(1, len(lg)):
            if lg[i - 1][1] == 2:
                tot += 1
                rep += lg[i][2]
    return rep, tot


def _med(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return float(np.median(v)) if v else float("nan")


def report(out: Path) -> None:
    d = RB._dir(out)
    raw = [json.loads(x) for x in (d / "tie_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in ok]
    arm = {a: {} for a in ARMS}
    for (u, c, s, a), r in ok.items():
        arm[a].setdefault((c, s), {})[u] = r
    stored = _rob2(d)
    rng = np.random.default_rng(AF.SEED_FULL + 26)
    pref = RB._pref
    pts = lambda t: tuple(100 * v for v in t)   # noqa: E731
    L = ["# Test plan TIE: \"no preference\" answers as data (PsyPref-T), users 8-39", "",
         f"Design note 26 (6671fa0). {len(ok)} sessions, {len(errors)} errors. ROB2's users with index "
         f"8-39 in C4 pessimistic and A5 no preference; paired with ROB2's stored sessions (same users "
         f"and answer seeds). Own-utility preference over generic assistance; 95 % intervals from "
         f"{R.N_BOOT} draws, users resampled within folds. Differences in points.", ""]
    # Reproduction of the stored PsyPref rows by the plain rerun
    L += ["## Plain PsyPref rerun against the stored rows", ""]
    for c in CONDS:
        for s in RB.SETTINGS:
            b, st = arm["base"].get((c, s), {}), stored.get((c, s, "psypref"), {})
            same = sum(b[u]["own"] == st[u]["own"] for u in b if u in st)
            L.append(f"- {c}, {s}: {same}/{len(b)} identical")
    L += ["", "## PsyPref-T against PsyPref and CMA-ES (k = 24 unless stated)", "",
          "| condition | setting | arm (band median) | n | PsyPref-T | PsyPref | − PsyPref | − PsyPref, k = 40 "
          "| − CMA-ES (ProMP) | − CMA-ES (Schäfer) | ties (PsyPref) | repeats after a tie | ℓ̂ (PsyPref) | λ̂ | τ̂ |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for c in CONDS:
        for s in RB.SETTINGS:
            base_st = stored.get((c, s, "psypref"), {})
            cm, sch = stored.get((c, s, "cmaes"), {}), stored.get((c, s, "cmaes_sch4"), {})
            plain = arm["base"].get((c, s), {})
            reg = arm["tie"].get((c, s), {})
            for a in ("tie", "tie_lo", "tie_hi"):
                mine = arm[a].get((c, s), {})
                us = sorted(u for u in mine if u in base_st and u in cm and u in sch)
                if not us:
                    continue
                ps = [dict(fold=mine[u]["fold"], a=mine[u], b=base_st[u], c=cm[u], e=sch[u],
                           r=reg.get(u)) for u in us]
                d24 = R._boot(ps, lambda x: pref([p["a"] for p in x]) - pref([p["b"] for p in x]), rng)
                d40 = R._boot(ps, lambda x: pref([p["a"] for p in x], 40) - pref([p["b"] for p in x], 40), rng)
                dcm = R._boot(ps, lambda x: pref([p["a"] for p in x]) - pref([p["c"] for p in x]), rng)
                dsch = R._boot(ps, lambda x: pref([p["a"] for p in x]) - pref([p["e"] for p in x]), rng)
                vs_reg = (pref([p["a"] for p in ps if p["r"]]) - pref([p["r"] for p in ps if p["r"]])
                          if a != "tie" and all(p["r"] for p in ps) else None)
                rows = [mine[u] for u in us]
                prow = [plain[u] for u in us if u in plain]
                rep, tot = _post_tie(rows)
                res[c, s, a] = dict(d24=d24, d40=d40, dsch=dsch, vs_reg=vs_reg, rep=rep, tot=tot,
                                    ties=_med(rows, "nopref"), ties_plain=_med(prow, "nopref"))
                L.append(f"| {c} | {s} | {a} ({ARMS[a]}) | {len(us)} | {pref(rows):.3f} | "
                         f"{pref([base_st[u] for u in us]):.3f} | {R._f(pts(d24), 1)} | {R._f(pts(d40), 1)} | "
                         f"{R._f(pts(dcm), 1)} | {R._f(pts(dsch), 1)} | {_med(rows, 'nopref'):.1f} "
                         f"({_med(prow, 'nopref'):.1f}) | {rep}/{tot} | {_med(rows, 'ell'):.2f} "
                         f"({_med(prow, 'ell'):.2f}) | {_med(rows, 'lam'):.2f} | {_med(rows, 'tau'):.2f} |")
    # Predictions
    get = lambda c, s, a="tie": res.get((c, s, a))   # noqa: E731
    v1 = [ABL._verdict(get("C4 pessimistic", s)["d24"], "above") for s in RB.SETTINGS if get("C4 pessimistic", s)]
    v2 = [ABL._verdict(get("A5 no preference", s)["d24"], "above") for s in RB.SETTINGS if get("A5 no preference", s)]
    p3 = get("C4 pessimistic", "fixed 12")
    v3 = ("held" if p3["dsch"][2] > 0 else "failed") if p3 else "–"
    off = [(c, s, a, res[c, s, a]) for c in CONDS for s in RB.SETTINGS for a in ("tie_lo", "tie_hi")
           if (c, s, a) in res]
    v4 = "held" if off and all(r["d24"][0] > 0 and r["vs_reg"] is not None and abs(r["vs_reg"]) < 0.02
                               for *_, r in off) else "failed"
    rep = sum(res[c, s, "tie"]["rep"] for c in CONDS for s in RB.SETTINGS if (c, s, "tie") in res)
    tot = sum(res[c, s, "tie"]["tot"] for c in CONDS for s in RB.SETTINGS if (c, s, "tie") in res)
    fewer = [res[c, "fixed 12", "tie"]["ties"] < res[c, "fixed 12", "tie"]["ties_plain"]
             for c in CONDS if (c, "fixed 12", "tie") in res]
    v5 = "held" if tot and rep / tot < 0.01 and fewer and all(fewer) else "failed"

    def overall(v):
        return "held" if v and all(x == "held" for x in v) else ("failed" if "failed" in v else "uncertain")

    L += ["", "## Predictions (k = 24, users with index 8-39)", "",
          f"- PTI1 (C4: PsyPref-T above PsyPref in each setting): {overall(v1)} {v1}",
          f"- PTI2 (A5: PsyPref-T above PsyPref in each setting): {overall(v2)} {v2}",
          f"- PTI3 (C4 fixed 12: not shown to trail CMA-ES on Schäfer's family): {v3}"
          + (f" {R._f(pts(p3['dsch']), 1)}" if p3 else ""),
          f"- PTI4 (off-centre band priors: gain > 0 and within 2 points of the planned arm in all "
          f"cells): {v4} " + str([(c.split()[0], s, a, round(100 * r['d24'][0], 1),
                                    None if r['vs_reg'] is None else round(100 * r['vs_reg'], 1))
                                   for c, s, a, r in off]),
          f"- PTI5 (repeats of a tied pair under 1 % of post-tie comparisons; fewer ties than PsyPref at "
          f"fixed 12 in C4 and A5): {v5} (repeats {rep}/{tot}; fewer ties {fewer})", ""]
    if errors:
        L += ["## Errors", ""] + [f"- {_key(r)}: {r['error']}" for r in errors[:20]]
    (d / "amp_tie_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("smoke", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"smoke": smoke, "run": lambda o: run(o, args.workers), "report": report}[args.command](args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
