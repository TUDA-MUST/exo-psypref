"""Post hoc: a fixed yardstick for the robustness rows (ROB, ROB2, TIE).

Usage:
  python scripts/icorr_posthoc_yardstick.py [--out <repository>/out]

Post hoc. Reads the stored session rows of FULL, ROB (robust_sessions.jsonl, section 17), ROB2 (rob2_sessions.jsonl, section 18) and TIE
(tie_sessions.jsonl, sections 26/26a). Nothing is rerun, and no planned verdict is changed or
re-judged: the planned yardstick stays the one the verdicts used.

Why: the reports score preference over generic assistance, lam/2 + (1 - lam) Phi((d0 - dk) / s),
with each condition's own s and lam at the session's end. Where these differ from the plain user's
(s = 1.0 Nm, lam = 0.1), a change against FULL mixes the method's loss with a change of the measure.
Per condition x setting x optimizer, at k = 24 and 40, this script gives preference over generic on
  (a) the planned yardstick, as each report computes it (ROB: the RMS distance with the row's s
      and lapse, R4 in the user's own utility; ROB2 and TIE: the own utility with s_score and
      lapse_score at the session's end), and
  (b) a fixed yardstick, s = 1.0 and lam = 0.1, on the same own-utility differences own_d0 - own[k-1].
Reported: the levels; PsyPref minus FULL's PsyPref on the same users (FULL: d with s = 1, lam = 0.1)
wherever the condition uses FULL's users; PsyPref minus each comparator in the condition; the share
of users worse off than generic assistance (own_k > own_d0) and at least one JND (0.674) worse; for
C4 and A5, PsyPref-T (TIE arm "tie", users 8-39) and plain PsyPref on the same users against CMA-ES
(ProMP) and CMA-ES (Schaefer, tuned) from ROB2; and every difference whose sign or whose interval's
relation to 0 differs between the two yardsticks. B5a and B5b at fixed 12 are bit-identical for
PsyPref (checked here) and count as one cell.

Intervals: 95 % from 2,000 fold-stratified draws (icorr_amp_range._boot), rows sorted by user. Each
statistic draws from a seed of its own label (which names the cell, the pair and k but not the
yardstick), so (a) and (b) use the same resamples and no interval depends on file order or on
which other statistics were computed. Self-check: the first cells of the ROB, ROB2 and TIE reports
are recomputed with each report's own seed and row order (intervals included), and every point
estimate in those reports' tables (and FULL's PsyPref rows) is recomputed from the rows here.
Outputs in <out>/icorr/amp_free/: amp_yardstick_report.md, amp_yardstick.json.
"""

from __future__ import annotations

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_full as F  # noqa: E402
import icorr_amp_range as R  # noqa: E402
import icorr_amp_rob2 as RB  # noqa: E402
import icorr_amp_robust as ROB  # noqa: E402
import icorr_amp_tie as TIE  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402

FIX_S, FIX_LAPSE = AF.S, AF.LAPSE          # the plain user's yardstick (FULL's)
JND = true_jnd(FIX_S)
KS = (24, 40)
SETTINGS = ROB.SETTINGS                    # ("wide", "fixed 12"), as ROB, ROB2 and TIE
COMPARATORS = ("cmaes_sch4", "g", "d_z", "cmaes", "gp_bald")
OPTIMIZERS = ("psypref_t", "psypref") + COMPARATORS
NAMES = dict(F.NAMES, psypref_t="PsyPref-T")
NAME_TO_CAND = {v: k for k, v in F.NAMES.items()}
SOURCES = ("FULL", "ROB", "ROB2", "TIE")
B5 = ("B5a features static", "B5b features walking")
SALT = 41                                  # this script's bootstrap seeds: [SEED_FULL, 41, crc32(label)]
_NUM = re.compile(r"[-+]?\d+\.\d+")


# ----------------------------------------------------------------------------- rows

def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def _rec(r: dict, measure_own: bool, s: float, lam: float) -> dict:
    """One session: the planned measure (d0, d, s, lam) and the own utility (own_d0, own)."""
    own_d0 = float(r["own_d0"] if "own_d0" in r else r["d0"])          # FULL: plain users, own = d
    own = np.asarray(r["own"] if "own" in r else r["distances"], dtype=float)
    d0, d = (own_d0, own) if measure_own else (float(r["d0"]), np.asarray(r["distances"], dtype=float))
    return dict(user=r["user"], fold=r["fold"], index=int(r["index"]), d0=d0, d=d, s=float(s),
                lam=float(lam), own_d0=own_d0, own=own)


def _uses_full_users(src: str, cond: str) -> bool:
    """FULL's 200 users (ROB R6 has users of its own; R7 keeps FULL's users, surrogate curves)."""
    return src in ("ROB2", "TIE") or (src == "ROB" and ROB.CONDITIONS[cond].get("users") != "xd")


def _cells(raw: dict) -> dict:
    """(source, condition, setting, optimizer) -> {user: record}."""
    cells: dict = {}

    def put(key, rec):
        cells.setdefault(key, {})[rec["user"]] = rec

    for r in raw["full"]:
        if r.get("error") is None and r["setting"] in SETTINGS and r["candidate"] == "psypref":
            put(("FULL", "main", r["setting"], "psypref"), _rec(r, False, AF.S, AF.LAPSE))
    for r in raw["rob"]:
        if r.get("error") is None:
            own = bool(ROB.CONDITIONS[r["condition"]].get("np"))   # as ROB's report: R4 in own utility
            put(("ROB", r["condition"], r["setting"], r["candidate"]), _rec(r, own, r["s"], r["lapse"]))
    for r in raw["rob2"]:
        if r.get("error") is None:
            rec = _rec(r, True, r["s_score"], r["lapse_score"])
            put(("ROB2", r["condition"], r["setting"], r["candidate"]), rec)
            if r["condition"] in TIE.CONDS and r["index"] >= TIE.MIN_INDEX and r["candidate"] in ("psypref", "cmaes", "cmaes_sch4"):
                put(("TIE", r["condition"], r["setting"], r["candidate"]), rec)
    for r in raw["tie"]:
        if r.get("error") is None and r["arm"] == "tie":
            put(("TIE", r["condition"], r["setting"], "psypref_t"), _rec(r, True, r["s_score"], r["lapse_score"]))
    return cells


def _order(key) -> tuple:
    src, cond, setting, opt = key
    conds = {"FULL": ["main"], "ROB": list(ROB.CONDITIONS), "ROB2": list(RB.CONDITIONS), "TIE": list(TIE.CONDS)}[src]
    return SOURCES.index(src), conds.index(cond), SETTINGS.index(setting), OPTIMIZERS.index(opt)


# ----------------------------------------------------------------------------- statistics

def _levels(recs: list[dict], k: int, fixed: bool) -> np.ndarray:
    """Per-user preference over generic at k on the planned (False) or fixed (True) yardstick."""
    if fixed:
        x = np.array([r["own_d0"] - r["own"][k - 1] for r in recs]) / FIX_S
        return FIX_LAPSE / 2 + (1 - FIX_LAPSE) * norm.cdf(x)
    lam = np.array([r["lam"] for r in recs])
    x = np.array([(r["d0"] - r["d"][k - 1]) / r["s"] for r in recs])
    return lam / 2 + (1 - lam) * norm.cdf(x)


def _rng(label: str) -> np.random.Generator:
    return np.random.default_rng([AF.SEED_FULL, SALT, zlib.crc32(label.encode("utf-8"))])


def _mean(x) -> float:
    return float(np.mean([p["v"] for p in x]))


def _ci(folds, values, label: str) -> tuple:
    return R._boot([dict(fold=f, v=float(v)) for f, v in zip(folds, values)], _mean, _rng(label))


def _both(folds, vr, vx, label):
    """Planned and fixed yardstick on the same resamples; the fixed one is copied when the
    per-user values are bit-identical."""
    same = bool(np.array_equal(vr, vx))
    tr = _ci(folds, vr, label)
    return tr, (tr if same else _ci(folds, vx, label)), same


def _level(cells, key, k) -> dict:
    us = sorted(cells[key])
    recs = [cells[key][u] for u in us]
    folds = [r["fold"] for r in recs]
    label = "|".join(("level",) + key + (str(k),))
    tr, tx, same = _both(folds, _levels(recs, k, False), _levels(recs, k, True), label)
    worse = [r["own"][k - 1] > r["own_d0"] for r in recs]
    worse_j = [r["own"][k - 1] - r["own_d0"] >= JND for r in recs]
    return dict(source=key[0], condition=key[1], setting=key[2], optimizer=key[3], k=k, n=len(us),
                registered=tr, fixed=tx, same_yardstick=same,
                worse_off=_ci(folds, worse, label + "|worse"),
                worse_off_1jnd=_ci(folds, worse_j, label + "|worse1jnd"))


def _diff(cells, kind, ka, kb, k) -> dict:
    A, B = cells[ka], cells[kb]
    us = sorted(set(A) & set(B))
    ra, rb = [A[u] for u in us], [B[u] for u in us]
    folds = [r["fold"] for r in ra]
    label = "|".join(("diff",) + ka + kb + (str(k),))
    vr = _levels(ra, k, False) - _levels(rb, k, False)
    vx = _levels(ra, k, True) - _levels(rb, k, True)
    tr, tx, same = _both(folds, vr, vx, label)
    return dict(kind=kind, source=ka[0], condition=ka[1], setting=ka[2], a=ka[3], b=kb[3],
                b_source=kb[0], k=k, n=len(us), registered=tr, fixed=tx, same_yardstick=same)


def _rel(t) -> str:
    return "above 0" if t[1] > 0 else ("below 0" if t[2] < 0 else "includes 0")


def _sign(t) -> str:
    return "+" if t[0] > 0 else ("−" if t[0] < 0 else "0")


def _pairs_to_compute(cells) -> list[tuple]:
    out = []
    for key in sorted(cells, key=_order):
        src, cond, setting, opt = key
        full = ("FULL", "main", setting, "psypref")
        if (src in ("ROB", "ROB2") and opt == "psypref") or (src == "TIE" and opt in ("psypref_t", "psypref")):
            if _uses_full_users(src, cond):
                out.append(("loss vs FULL", key, full))
            comps = ("cmaes", "cmaes_sch4") if src == "TIE" else COMPARATORS
            out += [("vs comparator", key, (src, cond, setting, c)) for c in comps if (src, cond, setting, c) in cells]
            if src == "TIE" and opt == "psypref_t":
                out.append(("PsyPref-T vs PsyPref", key, (src, cond, setting, "psypref")))
    return out


# ----------------------------------------------------------------------------- self-check

def _nums(s: str) -> list[float]:
    return [float(x) for x in _NUM.findall(s)]


def _parse(d: Path) -> dict:
    """The point estimates and intervals printed in the FULL, ROB, ROB2 and TIE reports."""
    rep = dict(full={}, rob={}, rob2={}, tie={}, rob_pr={})
    sec = None
    for line in (d / "amp_full_report.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            sec = line[3:].strip()
        elif line.startswith("| PsyPref |") and sec in SETTINGS and sec not in rep["full"]:
            c = [x.strip() for x in line.strip().strip("|").split("|")]   # optimizer, n, k = 12, 24, 40
            if len(c) == 5 and "[" in c[3]:
                rep["full"][sec] = {24: c[3], 40: c[4]}
    for line in (d / "amp_robust_report.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            sec = line[3:].strip()
        elif line.startswith("| ") and sec in ROB.CONDITIONS:
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) == 6 and c[1] in NAME_TO_CAND:
                rep["rob"][sec, c[0], NAME_TO_CAND[c[1]]] = dict(n=int(c[2]), k24=c[3], k40=c[4], diff=c[5])
        elif line.startswith(("- PR4", "- PR5", "- PR6")):
            rep["rob_pr"][line[2:5]] = line.split("—", 1)[1]
    for line in (d / "amp_rob2_report.md").read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(c) == 8 and c[0] in RB.CONDITIONS:
            rep["rob2"][c[0], c[1]] = dict(p=c[2], sch=c[3], best=c[4], dc=c[5], db=c[6], dfull=c[7])
    for line in (d / "amp_tie_report.md").read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(c) >= 10 and c[0] in TIE.CONDS:
            rep["tie"][c[0], c[1], c[2].split()[0]] = dict(n=int(c[3]), T=c[4], P=c[5], dP=c[6], dP40=c[7],
                                                           dcm=c[8], dsch=c[9])
    return rep


def _match(mine: tuple, reported: str, scale: float, dec: int, full: bool) -> bool:
    """Reported at its printed precision: the point estimate, or (full) all three numbers."""
    nums = _nums(reported)
    want = nums[:3] if full else nums[:1]
    got = [scale * v for v in (mine[:3] if full else mine[:1])]
    return len(want) == len(got) and all(abs(g - w) <= 0.5 * 10 ** -dec + 1e-9 for g, w in zip(got, want))


def _exact(raw: dict, rep: dict) -> list[dict]:
    """The first statistics of the ROB, ROB2 and TIE reports, with each report's seed and row order."""
    out = []

    def add(name, t, reported, scale, dec):
        out.append(dict(name=name, recomputed=[scale * v for v in t], reported=reported,
                        ok=_match(t, reported, scale, dec, True)))

    # ROB: rng SEED_FULL + 17; R1 wide: PsyPref k = 24, 40; CMA-ES (Schaefer) k = 24, 40; the difference
    rows = {ROB._key(r): r for r in raw["rob"] if r.get("error") is None}
    rng = np.random.default_rng(AF.SEED_FULL + 17)
    cond, setting = next(iter(ROB.CONDITIONS)), SETTINGS[0]
    by = {}
    for cand in ROB.CONDITIONS[cond]["cands"][:2]:
        rs = [r for (u, c, s, cc), r in rows.items() if c == cond and s == setting and cc == cand]
        by[cand] = {r["user"]: r for r in rs}
        for k in KS:
            add(f"ROB {cond}, {setting}, {NAMES[cand]}, k = {k}",
                R._boot(rs, lambda x, k=k: ROB._pref(x, k, False), rng), rep["rob"][cond, setting, cand][f"k{k}"], 1, 3)
        if cand != "psypref":
            a, b = by["psypref"], by[cand]
            ps = [dict(fold=a[u]["fold"], a=a[u], b=b[u]) for u in sorted(set(a) & set(b))]
            add(f"ROB {cond}, {setting}, PsyPref − {NAMES[cand]}, k = 24 (points)",
                R._boot(ps, lambda x: ROB._pref([p["a"] for p in x], 24) - ROB._pref([p["b"] for p in x], 24), rng),
                rep["rob"][cond, setting, cand]["diff"], 100, 1)
    # ROB2: rng SEED_FULL + 18; A1 wide: PsyPref, − CMA-ES (Schaefer), − strongest other optimizer, − FULL, CMA-ES (Schaefer)
    rows = {RB._key(r): r for r in raw["rob2"] if r.get("error") is None}
    full = {(r["user"], r["setting"]): dict(r) for r in raw["full"]
            if r.get("error") is None and r["candidate"] == "psypref"}
    rng = np.random.default_rng(AF.SEED_FULL + 18)
    cond, setting = next(iter(RB.CONDITIONS)), RB.SETTINGS[0]
    by = {}
    for (u, c, s, cc), r in rows.items():
        if c == cond and s == setting:
            by.setdefault(cc, {})[u] = r
    pub = [c for c in ("g", "d_z", "cmaes") if c in by]
    best = max(pub, key=lambda c: RB._pref(list(by[c].values())))

    def diff(other):
        users = sorted(set(by["psypref"]) & set(other))
        ps = [dict(fold=by["psypref"][u]["fold"], a=by["psypref"][u], b=other[u]) for u in users]
        return R._boot(ps, lambda x: RB._pref([p["a"] for p in x]) - RB._pref([p["b"] for p in x]), rng)

    r2 = rep["rob2"][cond, setting]
    add(f"ROB2 {cond}, {setting}, PsyPref, k = 24", R._boot(list(by["psypref"].values()), RB._pref, rng), r2["p"], 1, 3)
    add(f"ROB2 {cond}, {setting}, PsyPref − CMA-ES (Schäfer), k = 24 (points)", diff(by["cmaes_sch4"]), r2["dc"], 100, 1)
    add(f"ROB2 {cond}, {setting}, PsyPref − {NAMES[best]} (best published), k = 24 (points)", diff(by[best]), r2["db"], 100, 1)
    fu = {u: full[u, setting] for u in by["psypref"] if (u, setting) in full}
    for r in fu.values():
        r.setdefault("own_d0", r["d0"]), r.setdefault("own", r["distances"])
        r.setdefault("s_score", AF.S), r.setdefault("lapse_score", AF.LAPSE)
    add(f"ROB2 {cond}, {setting}, PsyPref − FULL, k = 24 (points)", diff(fu), r2["dfull"], 100, 1)
    add(f"ROB2 {cond}, {setting}, CMA-ES (Schäfer), k = 24",
        R._boot(list(by["cmaes_sch4"].values()), RB._pref, rng), r2["sch"], 1, 3)
    # TIE: rng SEED_FULL + 26; C4 wide, arm tie: − PsyPref k = 24, 40; − CMA-ES (ProMP); − CMA-ES (Schaefer)
    ok = {TIE._key(r): r for r in raw["tie"] if r.get("error") is None}
    arm = {}
    for (u, c, s, a), r in ok.items():
        arm.setdefault(a, {}).setdefault((c, s), {})[u] = r
    stored = {}
    for r in raw["rob2"]:
        if r.get("error") is None and r["condition"] in TIE.CONDS and r["index"] >= TIE.MIN_INDEX:
            stored.setdefault((r["condition"], r["setting"], r["candidate"]), {})[r["user"]] = r
    rng = np.random.default_rng(AF.SEED_FULL + 26)
    c, s = TIE.CONDS[0], RB.SETTINGS[0]
    mine, base_st = arm["tie"][c, s], stored[c, s, "psypref"]
    cm, sch = stored[c, s, "cmaes"], stored[c, s, "cmaes_sch4"]
    us = sorted(u for u in mine if u in base_st and u in cm and u in sch)
    ps = [dict(fold=mine[u]["fold"], a=mine[u], b=base_st[u], c=cm[u], e=sch[u]) for u in us]
    pref = RB._pref
    t = rep["tie"][c, s, "tie"]
    add(f"TIE {c}, {s}, PsyPref-T − PsyPref, k = 24 (points)",
        R._boot(ps, lambda x: pref([p["a"] for p in x]) - pref([p["b"] for p in x]), rng), t["dP"], 100, 1)
    add(f"TIE {c}, {s}, PsyPref-T − PsyPref, k = 40 (points)",
        R._boot(ps, lambda x: pref([p["a"] for p in x], 40) - pref([p["b"] for p in x], 40), rng), t["dP40"], 100, 1)
    add(f"TIE {c}, {s}, PsyPref-T − CMA-ES (ProMP), k = 24 (points)",
        R._boot(ps, lambda x: pref([p["a"] for p in x]) - pref([p["c"] for p in x]), rng), t["dcm"], 100, 1)
    add(f"TIE {c}, {s}, PsyPref-T − CMA-ES (Schäfer), k = 24 (points)",
        R._boot(ps, lambda x: pref([p["a"] for p in x]) - pref([p["e"] for p in x]), rng), t["dsch"], 100, 1)
    return out


def _points(levels: dict, diffs: dict, rep: dict, cells: dict) -> list[dict]:
    """Every point estimate printed in the reports' tables, from this script's planned values."""
    out = []

    def add(name, t, reported, scale, dec):
        out.append(dict(name=name, recomputed=scale * t[0], reported=_nums(reported)[0],
                        ok=_match(t, reported, scale, dec, False)))

    for setting, v in rep["full"].items():
        for k in KS:
            add(f"FULL {setting} PsyPref k = {k}", levels["FULL", "main", setting, "psypref", k]["registered"], v[k], 1, 3)
    for (cond, setting, cand), v in rep["rob"].items():
        for k in KS:
            add(f"ROB {cond} {setting} {NAMES[cand]} k = {k}", levels["ROB", cond, setting, cand, k]["registered"], v[f"k{k}"], 1, 3)
        if cand != "psypref":
            add(f"ROB {cond} {setting} PsyPref − {NAMES[cand]} k = 24",
                diffs["ROB", cond, setting, "psypref", cand, 24]["registered"], v["diff"], 100, 1)
    pr = rep["rob_pr"]
    for setting, val in re.findall(r"(wide|fixed 12) ([-+]\d+\.\d+)", pr.get("PR4", "")):
        add(f"ROB PR4 R4 negative power {setting} PsyPref − FULL k = 24",
            diffs["ROB", "R4 negative power", setting, "psypref", "FULL", 24]["registered"], val, 100, 1)
    if "PR5" in pr:
        add("ROB PR5 R6 wide PsyPref k = 24", levels["ROB", "R6 other datasets", "wide", "psypref", 24]["registered"], pr["PR5"], 1, 3)
    if "PR6" in pr:
        add("ROB PR6 R5 wide PsyPref − FULL k = 40", diffs["ROB", "R5 K=12", "wide", "psypref", "FULL", 40]["registered"],
            pr["PR6"], 100, 1)
    for (cond, setting), v in rep["rob2"].items():
        add(f"ROB2 {cond} {setting} PsyPref", levels["ROB2", cond, setting, "psypref", 24]["registered"], v["p"], 1, 3)
        add(f"ROB2 {cond} {setting} CMA-ES (Schäfer)", levels["ROB2", cond, setting, "cmaes_sch4", 24]["registered"], v["sch"], 1, 3)
        add(f"ROB2 {cond} {setting} PsyPref − CMA-ES (Schäfer)",
            diffs["ROB2", cond, setting, "psypref", "cmaes_sch4", 24]["registered"], v["dc"], 100, 1)
        if v["best"] in NAME_TO_CAND:
            add(f"ROB2 {cond} {setting} PsyPref − {v['best']} (best published)",
                diffs["ROB2", cond, setting, "psypref", NAME_TO_CAND[v["best"]], 24]["registered"], v["db"], 100, 1)
        add(f"ROB2 {cond} {setting} PsyPref − FULL",
            diffs["ROB2", cond, setting, "psypref", "FULL", 24]["registered"], v["dfull"], 100, 1)
    for (cond, setting, a), v in rep["tie"].items():
        if a != "tie":
            continue
        add(f"TIE {cond} {setting} PsyPref-T", levels["TIE", cond, setting, "psypref_t", 24]["registered"], v["T"], 1, 3)
        add(f"TIE {cond} {setting} PsyPref (users 8-39)", levels["TIE", cond, setting, "psypref", 24]["registered"], v["P"], 1, 3)
        for k, f in ((24, "dP"), (40, "dP40")):
            add(f"TIE {cond} {setting} PsyPref-T − PsyPref k = {k}",
                diffs["TIE", cond, setting, "psypref_t", "psypref", k]["registered"], v[f], 100, 1)
        add(f"TIE {cond} {setting} PsyPref-T − CMA-ES (ProMP)",
            diffs["TIE", cond, setting, "psypref_t", "cmaes", 24]["registered"], v["dcm"], 100, 1)
        add(f"TIE {cond} {setting} PsyPref-T − CMA-ES (Schäfer)",
            diffs["TIE", cond, setting, "psypref_t", "cmaes_sch4", 24]["registered"], v["dsch"], 100, 1)
    return out


def _identities(raw: dict, cells: dict) -> dict:
    """B5a = B5b at fixed 12 (per optimizer), TIE's plain rerun = ROB2's stored rows, d = own in ROB
    and ROB2 (per condition)."""
    out = dict(b5={}, tie_base={}, rob_own={}, rob2_own={})
    for cand in RB.CANDS:
        a, b = cells["ROB2", B5[0], "fixed 12", cand], cells["ROB2", B5[1], "fixed 12", cand]
        us = sorted(set(a) & set(b))
        same = sum(np.array_equal(a[u]["own"], b[u]["own"]) and a[u]["own_d0"] == b[u]["own_d0"]
                   and a[u]["s"] == b[u]["s"] and a[u]["lam"] == b[u]["lam"] for u in us)
        out["b5"][cand] = dict(identical=int(same), n=len(us),
                               max_abs_own_diff=float(max(np.max(np.abs(a[u]["own"] - b[u]["own"])) for u in us)))
    stored = {(r["user"], r["condition"], r["setting"]): r["own"] for r in raw["rob2"]
              if r.get("error") is None and r["candidate"] == "psypref"}
    for r in raw["tie"]:
        if r.get("error") is None and r["arm"] == "base":
            e = out["tie_base"].setdefault(f"{r['condition']}, {r['setting']}", [0, 0])
            e[0] += int(r["own"] == stored.get((r["user"], r["condition"], r["setting"])))
            e[1] += 1
    for r in raw["rob"]:
        if r.get("error") is None:
            e = out["rob_own"].setdefault(r["condition"], [0, 0])
            e[0] += int(r["own"] == r["distances"] and r["own_d0"] == r["d0"])
            e[1] += 1
    for r in raw["rob2"]:   # ROB2 rows store no RMS d0, so only the curves are compared
        if r.get("error") is None:
            e = out["rob2_own"].setdefault(r["condition"], [0, 0])
            e[0] += int(r["own"] == r["distances"])
            e[1] += 1
    return out


# ----------------------------------------------------------------------------- report

def _lv(t) -> str:
    return R._f(t, 3)


def _pt(t) -> str:
    return f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"


def _sh(t) -> str:
    return f"{100 * t[0]:.0f} [{100 * t[1]:.0f}, {100 * t[2]:.0f}] %"


def _yard(cells, src, cond, setting) -> dict:
    """The planned yardstick's s and lapse over all rows of a cell, and whether it is d or own."""
    recs = [r for (s_, c, st, o), us in cells.items() if (s_, c, st) == (src, cond, setting) for r in us.values()]
    s = np.array([r["s"] for r in recs])
    lam = np.array([r["lam"] for r in recs])
    return dict(n_rows=len(recs), s_median=float(np.median(s)), s_min=float(s.min()), s_max=float(s.max()),
                lapse_median=float(np.median(lam)), lapse_min=float(lam.min()), lapse_max=float(lam.max()))


def _span(lo, med, hi) -> str:
    return f"{med:.2f}" if lo == hi else f"{med:.2f} [{lo:.2f}, {hi:.2f}]"


def report(out: Path) -> None:
    d = F._dir(out)
    raw = dict(full=_load(d / "full_sessions.jsonl"), rob=_load(d / "robust_sessions.jsonl"),
               rob2=_load(d / "rob2_sessions.jsonl"), tie=_load(d / "tie_sessions.jsonl"))
    cells = _cells(raw)
    keys = sorted(cells, key=_order)
    print(f"{len(keys)} cells", flush=True)
    levels = {key + (k,): _level(cells, key, k) for key in keys for k in KS}
    print("levels done", flush=True)
    diffs = {}   # (source, condition, setting, a, b or "FULL", k)
    for kind, ka, kb in _pairs_to_compute(cells):
        for k in KS:
            diffs[ka + ("FULL" if kb[0] == "FULL" else kb[3], k)] = _diff(cells, kind, ka, kb, k)
    print("differences done", flush=True)
    rep = _parse(d)
    exact = _exact(raw, rep)
    points = _points(levels, diffs, rep, cells)
    ident = _identities(raw, cells)
    b5_psy = ident["b5"]["psypref"]
    b5_merged = b5_psy["identical"] == b5_psy["n"]

    def dup(src, cond, setting, opt_a, opt_b=None) -> bool:
        """B5b at fixed 12 repeats B5a's cell when every optimizer involved is bit-identical."""
        if not (src == "ROB2" and cond == B5[1] and setting == "fixed 12"):
            return False
        return all(ident["b5"][o]["identical"] == ident["b5"][o]["n"] for o in (opt_a, opt_b or opt_a))

    differs = {}   # (source, condition, setting) -> any optimizer's yardsticks differ
    for e in levels.values():
        if e["source"] != "FULL":
            kk = (e["source"], e["condition"], e["setting"])
            differs[kk] = differs.get(kk, False) or not e["same_yardstick"]
    diff_cells = [kk for kk, v in differs.items() if v]
    n_cells = sum(1 for kk in differs if not (b5_merged and dup(*kk, "psypref")))
    changes = []
    for key, e in diffs.items():
        if e["same_yardstick"] or dup(e["source"], e["condition"], e["setting"], e["a"], e["b"]):
            continue
        tr, tx = e["registered"], e["fixed"]
        if _sign(tr) != _sign(tx) or _rel(tr) != _rel(tx):
            changes.append(e)
    n_diff_tested = sum(1 for e in diffs.values() if not e["same_yardstick"])

    # ------------------------------------------------------------------ markdown
    L = ["# Fixed yardstick for the robustness rows (post hoc)", "",
         "Post hoc: stored session rows of FULL, ROB, ROB2 and TIE, nothing rerun. No "
         "planned verdict is changed or re-judged; the planned yardstick stays the one the "
         "verdicts used.", "",
         f"Preference over generic assistance at k = 24 and 40 on two yardsticks: **planned** (as each "
         f"report: ROB on the RMS distance with the row's s and λ, R4 in the own utility; ROB2 and TIE on "
         f"the own utility with s and λ at the session's end) and **fixed** ({FIX_LAPSE / 2:g} + "
         f"{1 - FIX_LAPSE:g} Φ((own_d0 − own_k) / {FIX_S:g}), the plain user's s = {FIX_S:g} and "
         f"λ = {FIX_LAPSE:g} on the same own-utility differences). FULL is scored on d with s = {AF.S:g}, "
         f"λ = {AF.LAPSE:g} in both. Worse off: own_k > own_d0; at least one JND worse: "
         f"own_k − own_d0 ≥ {JND:.3f} (the JND at s = {FIX_S:g}). 95 % intervals from {R.N_BOOT} "
         f"fold-stratified draws, rows sorted by user; the two yardsticks share each statistic's "
         f"resamples. Differences in points.", ""]
    # summary
    L += ["## Summary", "",
          f"- The yardsticks differ in {len(diff_cells)} of {n_cells} condition × setting cells: "
          + "; ".join(f"{src} {c}, {s}" for src, c, s in diff_cells) + ". In all other cells s and λ at the "
          f"end are {FIX_S:g} and {FIX_LAPSE:g} and the planned measure equals the own utility, so both "
          f"yardsticks give the same numbers (bit-identical per session).",
          f"- {len(changes)} of {n_diff_tested} differences (pair × k) in those cells change sign or their "
          f"interval's relation to 0 between the yardsticks (listed below): "
          f"{sum(e['b_source'] == 'FULL' for e in changes)} against FULL, "
          f"{sum(e['b_source'] != 'FULL' and e['b'] != 'psypref' for e in changes)} against a comparator, "
          f"{sum(e['b'] == 'psypref' and e['b_source'] != 'FULL' for e in changes)} PsyPref-T against PsyPref.", ""]
    for src, c, s in diff_cells:
        key = (src, c, s, "psypref_t" if src == "TIE" else "psypref")
        parts = []
        for k in KS:
            lv = levels[key + (k,)]
            fl = diffs.get(key + ("FULL", k))
            parts.append(f"k = {k}: {_lv(lv['registered'])} → {_lv(lv['fixed'])}"
                         + (f", − FULL {_pt(fl['registered'])} → {_pt(fl['fixed'])}" if fl else ""))
        L.append(f"- {src} {c}, {s}, {NAMES[key[3]]} (planned → fixed): " + "; ".join(parts))
    L.append("")
    # self-check
    n_ok = sum(p["ok"] for p in points)
    L += ["## Self-check against the planned reports", "",
          "Each report's first statistics, recomputed with that report's own seed and row order "
          "(point and interval at the printed precision):", ""]
    for e in exact:
        L.append(f"- {e['name']}: recomputed {e['recomputed'][0]:.4f} [{e['recomputed'][1]:.4f}, "
                 f"{e['recomputed'][2]:.4f}], reported {e['reported']} — {'match' if e['ok'] else 'MISMATCH'}")
    L += ["", f"Point estimates printed in the FULL (PsyPref rows), ROB, ROB2 and TIE (arm tie) reports, "
          f"recomputed from this script's planned yardstick: {n_ok} of {len(points)} match at the "
          f"printed precision."]
    L += [f"- MISMATCH {p['name']}: recomputed {p['recomputed']:.4f}, reported {p['reported']}"
          for p in points if not p["ok"]]
    L += ["", "Identities:", ""]
    for cand, v in ident["b5"].items():
        L.append(f"- B5a = B5b at fixed 12, {NAMES[cand]}: {v['identical']}/{v['n']} sessions bit-identical "
                 f"(largest own-utility difference {v['max_abs_own_diff']:.3g})")
    for cell, (a, n) in ident["tie_base"].items():
        L.append(f"- TIE plain rerun = ROB2's stored PsyPref, {cell}: {a}/{n}")
    L.append("- ROB rows with d = own utility: " + ", ".join(f"{c} {a}/{n}" for c, (a, n) in ident["rob_own"].items()))
    L.append("- ROB2 rows with d = own utility (curves): " + ", ".join(f"{c} {a}/{n}" for c, (a, n) in ident["rob2_own"].items()))
    L.append(f"- B5a and B5b at fixed 12 count as one cell where every optimizer involved is bit-identical "
             f"({'PsyPref: yes' if b5_merged else 'PsyPref: no'}).")
    L.append("")
    # yardsticks
    L += ["## Where the planned yardstick is not the fixed one", "",
          "| source | condition | setting | s at the end | λ at the end | sessions |", "|---|---|---|---|---|---|"]
    yard = {}
    for src, c, s in diff_cells:
        y = _yard(cells, src, c, s)
        yard[f"{src}|{c}|{s}"] = y
        L.append(f"| {src} | {c} | {s} | {_span(y['s_min'], y['s_median'], y['s_max'])} | "
                 f"{_span(y['lapse_min'], y['lapse_median'], y['lapse_max'])} | {y['n_rows']} |")
    L.append("")
    # PsyPref in the differing cells
    L += ["## PsyPref and its loss against FULL, cells whose yardstick differs", "",
          "FULL's PsyPref (s = 1, λ = 0.1, both yardsticks): "
          + "; ".join(f"{s} k = {k} {_lv(levels['FULL', 'main', s, 'psypref', k]['registered'])}"
                      for s in SETTINGS for k in KS) + ".", "",
          "| source | condition | setting | k | n | PsyPref, planned | PsyPref, fixed | − FULL, planned | "
          "− FULL, fixed | worse off | ≥ 1 JND worse |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for src, c, s in diff_cells:
        for opt in (("psypref_t", "psypref") if src == "TIE" else ("psypref",)):
            for k in KS:
                lv = levels[src, c, s, opt, k]
                fl = diffs.get((src, c, s, opt, "FULL", k))
                L.append(f"| {src} | {c}{' (PsyPref-T)' if opt == 'psypref_t' else ''} | {s} | {k} | {lv['n']} | "
                         f"{_lv(lv['registered'])} | {_lv(lv['fixed'])} | {_pt(fl['registered']) if fl else '—'} | "
                         f"{_pt(fl['fixed']) if fl else '—'} | {_sh(lv['worse_off'])} | {_sh(lv['worse_off_1jnd'])} |")
    L.append("")
    # comparators in the differing cells
    L += ["## PsyPref minus each comparator, cells whose yardstick differs", "",
          "| source | condition | setting | PsyPref − | n | k = 24, planned | k = 24, fixed | k = 40, planned | "
          "k = 40, fixed |", "|---|---|---|---|---|---|---|---|---|"]
    for src, c, s in diff_cells:
        if src == "TIE":
            continue
        for comp in COMPARATORS:
            e24, e40 = diffs.get((src, c, s, "psypref", comp, 24)), diffs.get((src, c, s, "psypref", comp, 40))
            if e24 is None:
                continue
            L.append(f"| {src} | {c} | {s} | {NAMES[comp]} | {e24['n']} | {_pt(e24['registered'])} | {_pt(e24['fixed'])} | "
                     f"{_pt(e40['registered'])} | {_pt(e40['fixed'])} |")
    L.append("")
    # TIE
    L += ["## C4 and A5: PsyPref-T and PsyPref against both CMA-ES variants (TIE, users 8-39)", "",
          "Paired on the same users and answer seeds; CMA-ES rows are ROB2's. A5's yardsticks coincide "
          "(s = 1, λ = 0.1).", "",
          "| condition | setting | k | method | n | level, planned | level, fixed | − CMA-ES (ProMP), planned | "
          "− CMA-ES (ProMP), fixed | − CMA-ES (Schäfer), planned | − CMA-ES (Schäfer), fixed | − FULL, planned | "
          "− FULL, fixed | worse off |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in TIE.CONDS:
        for s in SETTINGS:
            for k in KS:
                for opt in ("psypref_t", "psypref"):
                    lv = levels["TIE", c, s, opt, k]
                    g = lambda b: diffs["TIE", c, s, opt, b, k]   # noqa: E731
                    L.append(f"| {c} | {s} | {k} | {NAMES[opt]} | {lv['n']} | {_lv(lv['registered'])} | {_lv(lv['fixed'])} | "
                             f"{_pt(g('cmaes')['registered'])} | {_pt(g('cmaes')['fixed'])} | "
                             f"{_pt(g('cmaes_sch4')['registered'])} | {_pt(g('cmaes_sch4')['fixed'])} | "
                             f"{_pt(g('FULL')['registered'])} | {_pt(g('FULL')['fixed'])} | {_sh(lv['worse_off'])} |")
                e = diffs["TIE", c, s, "psypref_t", "psypref", k]
                L.append(f"| {c} | {s} | {k} | PsyPref-T − PsyPref | {e['n']} | {_pt(e['registered'])} | {_pt(e['fixed'])} "
                         f"| | | | | | | |")
    for c in TIE.CONDS:
        for s in SETTINGS:
            for opt in ("cmaes", "cmaes_sch4"):
                lv = [levels["TIE", c, s, opt, k] for k in KS]
                L.append(f"| {c} | {s} | 24 / 40 | {NAMES[opt]} | {lv[0]['n']} | {_lv(lv[0]['registered'])} / "
                         f"{_lv(lv[1]['registered'])} | {_lv(lv[0]['fixed'])} / {_lv(lv[1]['fixed'])} | | | | | | | "
                         f"{_sh(lv[0]['worse_off'])} / {_sh(lv[1]['worse_off'])} |")
    L.append("")
    # changes
    L += ["## Differences whose sign or interval relation to 0 changes between the yardsticks", ""]
    if not changes:
        L.append("- none")
    for e in changes:
        what = " and ".join(w for w, x in (("sign", _sign(e["registered"]) != _sign(e["fixed"])),
                                           ("interval", _rel(e["registered"]) != _rel(e["fixed"]))) if x)
        L.append(f"- {e['source']} {e['condition']}, {e['setting']}, {NAMES[e['a']]} − "
                 f"{'FULL' if e['b_source'] == 'FULL' else NAMES[e['b']]}, k = {e['k']} ({what}): planned "
                 f"{_pt(e['registered'])} ({_rel(e['registered'])}), fixed {_pt(e['fixed'])} ({_rel(e['fixed'])})")
    L.append("")
    # all optimizers in the differing cells
    L += ["## Every optimizer's level, cells whose yardstick differs", "",
          "| source | condition | setting | optimizer | n | k = 24, planned | k = 24, fixed | k = 40, planned | "
          "k = 40, fixed | worse off, k = 24 | worse off, k = 40 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in keys:
        if (key[0], key[1], key[2]) not in diff_cells:
            continue
        a, b = levels[key + (24,)], levels[key + (40,)]
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {NAMES[key[3]]} | {a['n']} | {_lv(a['registered'])} | "
                 f"{_lv(a['fixed'])} | {_lv(b['registered'])} | {_lv(b['fixed'])} | {_sh(a['worse_off'])} | "
                 f"{_sh(b['worse_off'])} |")
    L.append("")
    # all cells, PsyPref
    L += ["## All cells: PsyPref, its loss against FULL and the worse-off shares", "",
          "\"=\": the fixed yardstick gives the planned number (bit-identical per session). B5a and B5b at "
          "fixed 12 are one cell where bit-identical. ROB R6 has users of its own (no FULL pairing).", "",
          "| source | condition | setting | k | n | PsyPref, planned | PsyPref, fixed | − FULL, planned | "
          "− FULL, fixed | worse off | ≥ 1 JND worse |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in keys:
        src, c, s, opt = key
        if src == "FULL" or opt not in ("psypref", "psypref_t") or (src == "TIE" and opt == "psypref"):
            continue
        if b5_merged and dup(src, c, s, "psypref"):
            continue
        name = (f"{B5[0]} = {B5[1]}" if (b5_merged and src == "ROB2" and c == B5[0] and s == "fixed 12")
                else c) + (" (PsyPref-T)" if opt == "psypref_t" else "")
        for k in KS:
            lv = levels[key + (k,)]
            fl = diffs.get(key + ("FULL", k))
            fx = "=" if lv["same_yardstick"] else _lv(lv["fixed"])
            ffx = "—" if fl is None else ("=" if fl["same_yardstick"] else _pt(fl["fixed"]))
            L.append(f"| {src} | {name} | {s} | {k} | {lv['n']} | {_lv(lv['registered'])} | {fx} | "
                     f"{_pt(fl['registered']) if fl else '—'} | {ffx} | {_sh(lv['worse_off'])} | {_sh(lv['worse_off_1jnd'])} |")
    L += ["", "## Notes", "",
          "- Post hoc throughout; the fixed yardstick is a second reading of the same sessions, not a new "
          "test plan. Comparator differences and every optimizer's level in all cells are in "
          "amp_yardstick.json.",
          "- ROB R7 (surrogates) is paired with FULL on FULL's users and answer seeds although its preferred "
          "curves are the surrogates, as ROB2's C conditions are paired with FULL although their shapes "
          "differ. ROB R6 is not paired (users of its own).",
          "- TIE's plain PsyPref is ROB2's stored session on users 8-39 (TIE's rerun reproduces it, above).",
          "- \"Worse off\" uses the own utility; it does not depend on the yardstick. The own utility differs "
          "from the RMS distance d in "
          + ", ".join([f"ROB {c}" for c, (a, n) in ident["rob_own"].items() if a < n]
                      + [f"ROB2 {c}" for c, (a, n) in ident["rob2_own"].items() if a < n])
          + " (identities above); elsewhere it is d.",
          "- B5a and B5b at fixed 12: CMA-ES (Schäfer, tuned) is not bit-identical between them (above), so "
          "its differences are listed for both; PsyPref's cell is listed once." if not all(
              v["identical"] == v["n"] for v in ident["b5"].values()) else
          "- B5a and B5b at fixed 12 are bit-identical for every optimizer.", ""]
    (d / "amp_yardstick_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ json
    def tri(t):
        return [float(v) for v in t]

    js = dict(
        post_hoc=True,
        description="Fixed yardstick for the robustness rows: preference over generic on the planned "
                    "yardstick and on s = 1.0, lapse = 0.1 applied to own_d0 - own[k-1]. Proportions; "
                    "differences in proportion units (x100 for points). Triples are [estimate, 2.5 %, 97.5 %].",
        fixed_yardstick=dict(s=FIX_S, lapse=FIX_LAPSE), jnd=JND, n_boot=R.N_BOOT, ks=list(KS),
        bootstrap_seed=f"[{AF.SEED_FULL}, {SALT}, crc32(label)]",
        sources={"FULL": "full_sessions.jsonl", "ROB": "robust_sessions.jsonl", "ROB2": "rob2_sessions.jsonl",
                 "TIE": "tie_sessions.jsonl (arm tie); plain PsyPref and CMA-ES from rob2_sessions.jsonl, index >= 8"},
        self_check=dict(exact=[dict(e, recomputed=tri(e["recomputed"])) for e in exact],
                        points_matched=n_ok, points_total=len(points),
                        point_mismatches=[p for p in points if not p["ok"]], identities=ident,
                        b5_fixed12_one_cell=b5_merged),
        yardstick_differs=[dict(source=a, condition=b, setting=c, **yard[f"{a}|{b}|{c}"]) for a, b, c in diff_cells],
        levels=[dict(e, registered=tri(e["registered"]), fixed=tri(e["fixed"]), worse_off=tri(e["worse_off"]),
                     worse_off_1jnd=tri(e["worse_off_1jnd"]),
                     duplicate_of=(B5[0] if b5_merged and dup(e["source"], e["condition"], e["setting"], e["optimizer"]) else None))
                for e in levels.values()],
        differences=[dict(e, registered=tri(e["registered"]), fixed=tri(e["fixed"]),
                          registered_relation=_rel(e["registered"]), fixed_relation=_rel(e["fixed"]),
                          duplicate_of=(B5[0] if dup(e["source"], e["condition"], e["setting"], e["a"], e["b"]) else None))
                     for e in diffs.values()],
        changes=[dict(source=e["source"], condition=e["condition"], setting=e["setting"], a=e["a"],
                      b=("FULL" if e["b_source"] == "FULL" else e["b"]), k=e["k"], registered=tri(e["registered"]),
                      fixed=tri(e["fixed"]), registered_relation=_rel(e["registered"]), fixed_relation=_rel(e["fixed"]))
                 for e in changes],
    )
    (d / "amp_yardstick.json").write_text(json.dumps(js, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if (FIX_S, FIX_LAPSE) != (1.0, 0.1):
        raise SystemExit("the fixed yardstick is s = 1.0, lapse = 0.1")
    report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
