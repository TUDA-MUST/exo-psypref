"""Fig. 4 of the ICORR paper (v2), variant 4M: 4L as a measurement engineer would read it.

Usage:
  python scripts/paper_v2_fig4m.py [--out <repository>/out] [--dest figures/data]

Post hoc. Rows, markers and rating as 4L (scripts/paper_v2_fig4l.py), with s = 3 Nm (scripts/icorr_amp_s3.py) in the
row of s = 2 Nm and the shapes row split. Three changes:
1. Saturation: a cell whose lead is 0 in every draw has both methods at the measure's 95 % cap. It gets an
   open marker and "both at 95 %", no value and no bar.
2. Comparators: Fig. 2's g_generic_nogate and cmaes4 (CMP; under the deviations scripts/icorr_amp_var.py) join 4L's
   four in every cell, and all six also in the rows of tie handling and of a later flexion peak, where 4L has fewer.
   The caption gives the effect of the rating rule.
3. Coverage: every draw takes the strongest comparator again (the minimum over the comparators and, in a
   merged row, over the conditions). Thick bars: 95 % percentile interval of each row and setting. Thin bars:
   95 % for the 73 counted cells jointly, as a rank envelope (Besag et al., 1995): the per-cell percentile level
   at which 95 % of the draws lie inside all cells' intervals at once. Users are resampled jointly wherever
   cells share them, so the envelope uses the cells' correlation. A merged row's thin bar runs from the
   smallest lower to the smallest upper end of its cells, which joint coverage of the cells implies. Colour:
   icBlue with the value where the thin bar reaches 0, else icBlueLight, so a row is blue if and only if one of
   its cells is not clear.

Writes to <dest>: fig4m_rows.csv, fig4m_meta.csv, fig4m_provenance.csv, fig4m_text.tex.
Self-checks (stop on failure): every comparison's estimate equals amp_yardstick.json's; no row's lead above 4L's
fig4l_rows.csv, since more comparators can only lower it; the main row's comparison with Fig. 2's comparator equals
amp_table1.json; the envelope covers 95 % of the draws; a cell at the cap has PsyPref and the comparator at the cap;
every bar and label inside the axis and clear of everything drawn.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

import icorr_amp_pri as PRI
import icorr_amp_rob2 as RB
import icorr_amp_tie as TIE
import icorr_posthoc_yardstick as Y
import paper_v2_fig4 as F4
import paper_v2_fig4l as L
from paper_v2_fig4s import CLEAR, FIG_H, NUM_WORD, f1, sgn, text_w, tri
from prompref.experiment import amp_free as AF

K, SETTINGS, TAG = L.K, L.SETTINGS, L.TAG
N_DRAWS = 20000
BOOT_SEED = 20261103
COVERAGE = 0.95
RATING, FIXED = 0, 1
XMIN, XMAX = -13.0, 30.0
THICK_HALF, THIN_HALF = 0.5, 0.25
PAIR_SHIFT = 0.8                            # both of a row's labels on the left: each this far off its line (pt)
CAP_TEXT_W = 36.5                           # "both at 95 %" at \scriptsize, measured with \settowidth
CAP_LEVEL = AF.LAPSE / 2 + 1 - AF.LAPSE
SPLIT = ("SO", "SC")
OWN_FAMILY = "C1 Schaefer shapes"
S3_COND = "R2b s=3.0"                       # scripts/icorr_amp_s3.py, post hoc, the walking anchors' end
COMPARATORS = L.COMPARATORS + ("g_generic_nogate", "cmaes4")
NAMES = dict(L.NAMES, g_generic_nogate="Arens' GP (generic start)", cmaes4="CMA-ES (population 4)")


def _members(rk: str, conds: list) -> list:
    """4L's rows with two changes: s = 3 Nm joins the row of s = 2 Nm, and the shapes row is split, so that the
    exception, shapes from CMA-ES 4-6 param.'s own family, has its own."""
    if rk == "R2":
        return [(rk, conds + [("S3", S3_COND, "psypref")])]
    if rk == "SH":
        return [("SO", [c for c in conds if c[1] != OWN_FAMILY]), ("SC", [c for c in conds if c[1] == OWN_FAMILY])]
    return [(rk, conds)]


GROUPS = [(g, [m for rk, conds in members for m in _members(rk, conds)]) for g, members in L.GROUPS]


def levels_by_cell(amp: Path) -> dict:
    """(source, condition, setting, optimizer) -> {user: (fold, level on 4L's rating, on the fixed yardstick)}."""
    raw = dict(full=Y._load(amp / "full_sessions.jsonl"), rob=Y._load(amp / "robust_sessions.jsonl"),
               rob2=Y._load(amp / "rob2_sessions.jsonl"), tie=Y._load(amp / "tie_sessions.jsonl"))
    out = {}
    cells = Y._cells(raw)
    for r in Y._load(amp / "s3_sessions.jsonl"):
        if r.get("error") is None:
            cells.setdefault(("S3", r["condition"], r["setting"], r["candidate"]), {})[r["user"]] = Y._rec(r, False, r["s"], r["lapse"])
    var = Y._load(amp / "var_sessions.jsonl")

    def tie_cell(r, rec):
        # the rows of tie handling are on ROB2's users from TIE.MIN_INDEX on, against every comparator
        if r["condition"] in TIE.CONDS and r["index"] >= TIE.MIN_INDEX and r["candidate"] in COMPARATORS:
            cells.setdefault(("TIE", r["condition"], r["setting"], r["candidate"]), {})[r["user"]] = rec

    for r in raw["rob2"]:
        if r.get("error") is None:
            tie_cell(r, Y._rec(r, True, r["s_score"], r["lapse_score"]))
    for r in var:
        if r.get("error") is not None:
            raise SystemExit(f"var_sessions.jsonl has an error: {r['error']}")
        if r["block"] == "ROB2":
            rec = Y._rec(r, True, r["s_score"], r["lapse_score"])
            cells.setdefault(("ROB2", r["condition"], r["setting"], r["candidate"]), {})[r["user"]] = rec
            tie_cell(r, rec)
        elif r["block"] == "ROB":
            src = "S3" if r["condition"] == S3_COND else "ROB"
            own = r["condition"] == "R4 negative power"                  # as ROB's report: R4 in own utility
            cells.setdefault((src, r["condition"], r["setting"], r["candidate"]), {})[r["user"]] = Y._rec(r, own, r["s"], r["lapse"])
    for key, recs in cells.items():
        us = sorted(recs)
        rs = [recs[u] for u in us]
        reg, fix = Y._levels(rs, K, False), Y._levels(rs, K, True)
        out[key] = {u: (r["fold"], float(a), float(b)) for u, r, a, b in zip(us, rs, reg, fix)}

    def plain(key, r):
        p = PRI._p(r, K)
        out.setdefault(key, {})[r["user"]] = (r["fold"], p, p)

    for r in raw["full"]:
        if r.get("error") is None and r["setting"] in SETTINGS:
            plain(("FULL", "main", r["setting"], r["candidate"]), r)
    for x in (amp / "pri_sessions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r.get("error") is None:
            plain((r["test"], r["condition"], r["setting"], r["candidate"]), r)
    for x in (amp / "cmp_sessions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r.get("error") is None and r["setting"] in SETTINGS:
            plain(("CMP", "main", r["setting"], r["candidate"]), r)
            if r["candidate"] in COMPARATORS:
                plain(("FULL", "main", r["setting"], r["candidate"]), r)
    for r in var:
        if r["block"] == "PRI":
            plain((r["test"], r["condition"], r["setting"], r["candidate"]), r)
    return out


class Resampler:
    """Fold-stratified bootstrap counts, one matrix per set of users, so that cells on the same users share draws."""

    def __init__(self):
        self.w = {}

    def counts(self, users: tuple, folds: list) -> np.ndarray:
        if users not in self.w:
            rng = np.random.default_rng([BOOT_SEED, zlib.crc32("|".join(users).encode("utf-8"))])
            w = np.zeros((N_DRAWS, len(users)))
            folds = np.array(folds)
            for f in sorted(set(folds)):
                idx = np.flatnonzero(folds == f)
                w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=N_DRAWS)
            self.w[users] = w / len(users)
        return self.w[users]


def comparison(own: dict, other: dict, boot: Resampler) -> dict:
    """PsyPref minus one optimizer, in points: estimates and draws on both ratings."""
    us = tuple(sorted(set(own) & set(other)))
    w = boot.counts(us, [own[u][0] for u in us])
    res = dict(n=len(us))
    for col, name in ((1, "reg"), (2, "fix")):
        v = 100 * np.array([own[u][col] - other[u][col] for u in us])
        res[name], res[name + "_draws"] = float(v.mean()), w @ v
    return res


def strongest(comps: dict) -> dict:
    """A cell: the smallest lead over its comparators, in the data and in every draw, on both ratings."""
    b = min(comps, key=lambda c: comps[c]["reg"])
    bf = min(comps, key=lambda c: comps[c]["fix"])
    return dict(b=b, n=comps[b]["n"], est=comps[b]["reg"], draws=np.min([c["reg_draws"] for c in comps.values()], axis=0),
                b_fix=bf, fix=comps[bf]["fix"], fix_draws=np.min([c["fix_draws"] for c in comps.values()], axis=0),
                diff_chosen_fix=comps[b]["fix"], chosen_draws=comps[b]["reg_draws"])


def envelope_level(d: np.ndarray) -> float:
    """The lower tail of the rank envelope: draws x cells -> per-cell level with COVERAGE of the draws inside all."""
    n = d.shape[0]
    r = rankdata(d, axis=0, method="average")
    e = np.maximum(r, n + 1 - r).max(axis=1)
    return float((n + 1 - np.sort(e)[math.ceil(COVERAGE * n) - 1]) / n)


def read_rows(path: Path) -> dict:
    with open(path, encoding="utf-8", newline="") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def marker_4l(row: dict, tag: str) -> float:
    return float(row[f"in_{tag}"]) if row[f"in_{tag}"] != "nan" else float(row[f"out_{tag}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=F4.MAIN / "out")
    ap.add_argument("--dest", type=Path, default=F4.DEST)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    amp = args.out / "icorr" / "amp_free"
    js = json.loads((amp / "amp_yardstick.json").read_text(encoding="utf-8"))
    t1 = json.loads((amp / "amp_table1.json").read_text(encoding="utf-8"))["settings"]
    lev = levels_by_cell(amp)
    boot = Resampler()

    # ------------------------------------------------------------------ the cells
    def cell_comparisons(src, cond, st, opt):
        comps = {}
        for b in COMPARATORS:
            if (src, cond, st, b) in lev:
                comps[b] = comparison(lev[src, cond, st, opt], lev[src, cond, st, b], boot)
        return comps

    n_checked = 0
    rows, cells, y, first = [], [], 0.0, True
    for gkey, members in GROUPS:
        if gkey.startswith("G"):
            y += L.GROUP_GAP
            rows.append(dict(key=gkey, header=1, bold=0, y=y))
            y += L.HEADER_TO_ROW
        elif not first:
            y += L.GROUP_GAP
        first = False
        for i, (rk, conds) in enumerate(members):
            if i:
                y += 1.0
            row = dict(key=rk, header=0, bold=int(gkey.startswith("BOLD")), y=y, n_merged=len(conds),
                       tie=int(conds[0][2] == "psypref_t"), cells={})
            for st in SETTINGS:
                got = []
                for src, cond, opt in conds:
                    if (src, cond, st, opt) not in lev:
                        continue
                    comps = cell_comparisons(src, cond, st, opt)
                    for e in js["differences"]:
                        if (e["k"], e["kind"], e["source"], e["condition"], e["setting"], e["a"]) == (K, "vs comparator", src, cond, st, opt) \
                                and e["b"] in comps:
                            n_checked += 1
                            if abs(100 * e["registered"][0] - comps[e["b"]]["reg"]) > 1e-6 \
                                    or abs(100 * e["fixed"][0] - comps[e["b"]]["fix"]) > 1e-6:
                                raise SystemExit(f"{src} {cond} {st} {e['b']} differs from amp_yardstick.json")
                    c = dict(strongest(comps), row=rk, src=src, cond=cond, st=st, opt=opt,
                             counted=rk != "MAIN" and not (cond == "B5b features walking" and st == "fixed 12"), comps=comps)
                    got.append(c)
                    cells.append(c)
                if got:
                    row["cells"][st] = got
            rows.append(row)
            if rk == "MAIN":
                for st in SETTINGS:
                    cand = t1[st]["best_published"]["candidate"]
                    got = row["cells"][st][0]["comps"][cand]["reg"]
                    want = t1[st]["psypref_minus_points"]["best"][str(K)]["est"]
                    if abs(got - want) > 1e-6:
                        raise SystemExit(f"Fig. 2's comparator, {st}: {got} vs amp_table1.json's {want}")
    last_y = y
    family = [c for c in cells if c["counted"]]

    # ------------------------------------------------------------------ the bars
    gamma = envelope_level(np.column_stack([c["draws"] for c in family]))
    gamma_fix = envelope_level(np.column_stack([c["fix_draws"] for c in family]))
    for c in cells:
        c["lo"], c["hi"] = (float(v) for v in np.quantile(c["draws"], [0.025, 0.975]))
        c["jlo"], c["jhi"] = (float(v) for v in np.quantile(c["draws"], [gamma, 1 - gamma]))
        c["jlo_fix"], c["jhi_fix"] = (float(v) for v in np.quantile(c["fix_draws"], [gamma_fix, 1 - gamma_fix]))
        c["cap"] = abs(c["est"]) < 1e-9 and float(np.abs(c["draws"]).max()) < 1e-9
    inside = np.all([(c["draws"] >= c["jlo"]) & (c["draws"] <= c["jhi"]) for c in family], axis=0).mean()
    if not COVERAGE - 0.005 <= inside <= COVERAGE + 0.01:
        raise SystemExit(f"the envelope holds {inside:.4f} of the draws")
    for c in family:
        if c["cap"]:
            levels = [np.mean([v[1] for v in lev[c["src"], c["cond"], c["st"], o].values()]) for o in (c["opt"], c["b"])]
            if max(abs(v - CAP_LEVEL) for v in levels) > 1e-9:
                raise SystemExit(f"{c['cond']} {c['st']}: lead 0 in every draw but levels {levels}")

    fig4l = read_rows(args.dest / "fig4l_rows.csv")
    for r in rows:
        if r["header"]:
            continue
        for st in SETTINGS:
            got = r["cells"].get(st)
            if got is None:
                r[TAG[st]] = None
                continue
            draws = np.min([c["draws"] for c in got], axis=0)
            lo, hi = (float(v) for v in np.quantile(draws, [0.025, 0.975]))
            est = min(c["est"] for c in got)
            c = dict(fx=est, lo=lo, hi=hi, jlo=min(c["jlo"] for c in got), jhi=min(c["jhi"] for c in got), n=got[0]["n"],
                     cap=all(c["cap"] for c in got), of=min(got, key=lambda c: c["est"]))
            c["blue"] = c["jlo"] <= 0.0 and not c["cap"]
            if c["jlo"] > c["lo"] + 1e-9 or c["jhi"] < c["hi"] - 1e-9:
                raise SystemExit(f"{r['key']} {st}: the thin bar lies inside the thick one")
            r[TAG[st]] = c
            if r["key"] not in SPLIT:
                was = marker_4l(fig4l[r["key"]], TAG[st])
                own = est if r["key"] != "R2" else next(c["est"] for c in got if c["src"] != "S3")
                if own > was + 1e-6:
                    raise SystemExit(f"{r['key']} {st}: {own} above 4L's {was}")
    by_row = {r["key"]: r for r in rows}
    for st in SETTINGS:
        est, was = min(by_row[k][TAG[st]]["fx"] for k in SPLIT), marker_4l(fig4l["SH"], TAG[st])
        if est > was + 1e-6:
            raise SystemExit(f"split shapes row, {st}: {est} above 4L's {was}")

    # ------------------------------------------------------------------ layout and labels
    ymin, ymax, linetop = L.TOP_LABELS - L.TOP_PAD, last_y + L.BOTTOM_PAD, L.TOP_LABELS + 0.45
    scale = (L.AX_R - L.LAB_W) / (XMAX - XMIN)

    def xp(v):
        return L.LAB_W + (v - XMIN) * scale

    def yp(v):
        return (v - ymin) * L.PITCH

    drawn = []
    for r in rows:
        if r["header"]:
            continue
        for tag, off, rad in (("w", -L.OFF, L.MARK_R), ("f", L.OFF, L.MARK_S)):
            c = r[tag]
            if c is None:
                continue
            if not (XMIN < c["jlo"] and c["jhi"] < XMAX):
                raise SystemExit(f"{r['key']} {tag}: [{c['jlo']:.1f}, {c['jhi']:.1f}] outside the axis")
            yc = yp(r["y"] + off)
            if not c["cap"]:
                drawn.append((f"{r['key']} {tag} thin", xp(c["jlo"]), yc - THIN_HALF, xp(c["jhi"]), yc + THIN_HALF))
                drawn.append((f"{r['key']} {tag} thick", xp(c["lo"]), yc - THICK_HALF, xp(c["hi"]), yc + THICK_HALF))
            drawn.append((f"{r['key']} {tag} mark", xp(c["fx"]) - rad, yc - rad, xp(c["fx"]) + rad, yc + rad))
    drawn.append(("zero line", xp(0) - L.LINE_HALF, yp(linetop), xp(0) + L.LINE_HALF, yp(ymax)))

    def clear_box(x0, x1, yc, own, tag, start=0.0, clear=CLEAR):
        """The least vertical shift from start that clears everything drawn, a circle's label up, a square's down."""
        for step in range(16):
            shift = (-1 if tag == "w" else 1) * (start + 0.1 * step)
            box = (x0, yc + shift - FIG_H / 2, x1, yc + shift + FIG_H / 2)
            gaps = [(math.hypot(max(a0 - box[2], box[0] - a9, 0.0), max(b0 - box[3], box[1] - b9, 0.0)), name)
                    for name, a0, b0, a9, b9 in drawn if not name.startswith(own)]
            least, near = min(gaps)
            if least >= clear:
                return shift
        raise SystemExit(f"label {own}: {least:.2f} pt from {near}")

    labels = []
    for r in rows:
        if r["header"]:
            continue
        for tag, off, rad in (("w", -L.OFF, L.MARK_R), ("f", L.OFF, L.MARK_S)):
            c = r[tag]
            if c is None:
                continue
            yc = yp(r["y"] + off)
            if c["cap"]:
                x0 = xp(c["fx"]) + rad + L.VAL_GAPX
                # down, as a square's label: the row above's marker sits over the longer text, which leaves
                # 5.45 pt of room for its 4.73 pt, so 0.3 pt of clearance instead of CLEAR
                shift = clear_box(x0, x0 + CAP_TEXT_W, yc, f"{r['key']} {tag} mark", "f", clear=0.3)
                c.update(pos=4, vx=x0, vdy=-shift, lab="-", neg=0)
                labels.append((r["key"], tag, "both at 95 %"))
            elif c["blue"]:
                txt = f1(abs(c["fx"]))
                neg = c["fx"] < 0 and txt != "0.0"
                w = text_w(("-" if neg else "") + txt)
                x1 = min(xp(c["jlo"]), xp(c["fx"]) - rad) - L.VAL_GAPX
                if x1 - w < L.LAB_W + CLEAR:
                    raise SystemExit(f"value label {r['key']} {tag} leaves the axis")
                both = all(r[t] is not None and r[t]["blue"] for t in ("w", "f"))
                shift = clear_box(x1 - w, x1, yc, f"{r['key']} {tag} thin", tag, PAIR_SHIFT if both else 0.0)
                drawn.append((f"{r['key']} {tag} label", x1 - w, yc + shift - FIG_H / 2, x1, yc + shift + FIG_H / 2))
                c.update(pos=3, vx=x1, vdy=-shift, lab=txt, neg=int(neg))
                labels.append((r["key"], tag, ("-" if neg else "") + txt))

    # ------------------------------------------------------------------ numbers for the text
    clear = [c for c in family if c["jlo"] > 0.0]
    not_clear = [c for c in family if c["jlo"] <= 0.0]
    clear_cell = [c for c in family if c["lo"] > 0.0]
    clear_fix = [c for c in family if c["jlo_fix"] > 0.0]
    sign_changes = [c for c in family if (c["est"] > 0) != (c["fix"] > 0) and abs(c["est"]) > 1e-9]
    gap = max(abs(c["est"] - c["fix"]) for c in family)
    drops = {(e["source"], e["condition"], e["setting"], e["a"]): 100 * e["registered"][0] for e in js["differences"]
             if e["k"] == K and e["kind"] == "loss vs FULL" and not e.get("duplicate_of")}
    single = min(v for (s, c, st, a), v in drops.items() if c.startswith(("B5a", "B5b")))
    pess = min(v for (s, c, st, a), v in drops.items() if s == "TIE" and c.startswith("C4") and a == "psypref_t")
    plain = {(e["source"], e["condition"], e["setting"], e["a"], e["b"]): e for e in js["differences"]
             if e["k"] == K and e["kind"] == "vs comparator"}
    bald = [100 * plain["ROB", c, st, "psypref", "gp_bald"]["registered"][0] for c in ("R2 s=2.0", "R6 other datasets")
            for st in SETTINGS]
    cell = lambda rk, st: by_row[rk][TAG[st]]                                    # noqa: E731
    tr = lambda c: tri((c["fx"], c["lo"], c["hi"]))                              # noqa: E731
    s3 = {st: next(c for c in by_row["R2"]["cells"][st] if c["src"] == "S3") for st in SETTINGS}
    s3_level = {st: 100 * np.mean([v[1] for v in lev["S3", S3_COND, st, "psypref"].values()]) for st in SETTINGS}
    # how far the two methods are from users of CMA-ES 4-6 param.'s own family, after one comparison, median (Nm)
    own_family = {o: np.median([r["own"][0] for r in Y._load(amp / "rob2_sessions.jsonl") if r.get("error") is None
                                and r["condition"] == OWN_FAMILY and r["setting"] == "fixed 12" and r["candidate"] == o])
                  for o in ("cmaes_sch4", "psypref")}
    macro = {
        "fmSThreeW": tri((s3["wide"]["est"], s3["wide"]["lo"], s3["wide"]["hi"])),
        "fmSThreeF": tri((s3["fixed 12"]["est"], s3["fixed 12"]["lo"], s3["fixed 12"]["hi"])),
        "fmSThreeLevW": f1(s3_level["wide"]), "fmSThreeLevF": f1(s3_level["fixed 12"]),
        "fmStartC": f1(own_family["cmaes_sch4"]), "fmStartP": f1(own_family["psypref"]),
        "fmK": K, "fmNclear": len(clear), "fmNall": len(family), "fmNcell": len(clear_cell), "fmNfix": len(clear_fix),
        "fmNusers": cell("MAIN", "wide")["n"], "fmNother": cell("R6", "wide")["n"], "fmNtie": cell("C4T", "wide")["n"],
        "fmMainW": tr(cell("MAIN", "wide")), "fmMainF": tr(cell("MAIN", "fixed 12")),
        "fmSingleW": tr(cell("B5", "wide")), "fmSingleF": tr(cell("B5", "fixed 12")),
        "fmOtherW": tr(cell("R6", "wide")), "fmOtherF": tr(cell("R6", "fixed 12")),
        "fmPeaks": sgn(cell("PM", "wide")["fx"]), "fmLaterW": sgn(cell("PS", "wide")["fx"]),
        "fmLaterF": sgn(cell("PS", "fixed 12")["fx"]), "fmNoneF": tr(cell("B8", "fixed 12")),
        "fmShapesF": tr(cell("SC", "fixed 12")), "fmShapesW": tr(cell("SC", "wide")),
        "fmShapesNext": sgn(sorted(v["reg"] for v in cell("SC", "fixed 12")["of"]["comps"].values())[1]),
        "fmOtherShapesW": sgn(cell("SO", "wide")["fx"]), "fmOtherShapesF": sgn(cell("SO", "fixed 12")["fx"]),
        "fmPessW": tr(cell("C4T", "wide")), "fmPessF": tr(cell("C4T", "fixed 12")),
        "fmDropSingle": f1(-single), "fmDropPess": f1(-pess), "fmFixGap": f1(gap),
        "fmBaldLo": f1(min(bald)), "fmBaldHi": f1(max(bald)),
    }
    tex = ["% fig4m_text.tex: written by scripts/paper_v2_fig4m.py (amp-free worktree); do not edit by hand."]
    tex += [f"\\def\\{k_}{{{v}}}" for k_, v in macro.items()]
    (args.dest / "fig4m_text.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ write
    out_rows, prov = [], []
    for r in rows:
        o = [r["key"], r["header"], r["bold"], r["y"]]
        for tag, st in zip(("w", "f"), SETTINGS):
            c = r.get(tag)
            if c is None:
                o += [math.nan] * 11 + [0, "-", 0, 0, 0, 0, 0]
                continue
            t5, nan5 = [c["fx"], c["lo"], c["hi"], c["jlo"], c["jhi"]], [math.nan] * 5
            o += (nan5 if c["blue"] or c["cap"] else t5) + (t5 if c["blue"] else nan5) + [c["fx"] if c["cap"] else math.nan]
            o += [int(c["blue"]), c.get("lab", "-"), c.get("neg", 0), c.get("pos", 0), c.get("vx", 0), c.get("vdy", 0), c["n"]]
            prov.append([r["key"], st, c["of"]["src"], c["of"]["cond"], NAMES[c["of"]["b"]], c["n"], r["n_merged"],
                         c["fx"], c["lo"], c["hi"], c["jlo"], c["jhi"], min(x["fix"] for x in r["cells"][st]), int(c["cap"])])
        o += [r.get("n_merged", 0), r.get("tie", 0)]
        out_rows.append(o)
    head = ["key", "is_header", "bold", "y"]
    for t in ("w", "f"):
        for part in ("in", "out"):
            head += [f"{part}_{t}", f"{part}_lo_{t}", f"{part}_hi_{t}", f"{part}_jlo_{t}", f"{part}_jhi_{t}"]
        head += [f"cap_{t}", f"blue_{t}", f"lab_{t}", f"neg_{t}", f"pos_{t}", f"vx_{t}", f"vdy_{t}", f"n_{t}"]
    head += ["n_merged", "tie"]
    F4.write_csv(args.dest / "fig4m_rows.csv", head, out_rows)
    fab_lab = F4.label_values(js)
    meta = dict(xmin=XMIN, xmax=XMAX, ymin=ymin, ymax=ymax, ylabels=L.TOP_LABELS, linetop=linetop, lab_w=L.LAB_W, ax_r=L.AX_R,
                pitch=L.PITCH, bottom=L.BOTTOM, width=L.WIDTH, scale=scale, off=L.OFF, mark_r=L.MARK_R, mark_s=L.MARK_S,
                val_gapx=L.VAL_GAPX, height=L.BOTTOM + (ymax - ymin) * L.PITCH, n_rows=len(out_rows), s_r1=0.5, s_r2=2.0, s_r3=3.0,
                lapse_a2=float(fab_lab["lapse_a2"]), lapse_r3=float(fab_lab["lapse_r3"]),
                ndev_word=NUM_WORD[len(RB.CONDITIONS["C4 pessimistic"])],
                # the sizes in the row labels, from the conditions themselves
                a1_logsd=RB.CONDITIONS["A1 individual s"]["s_logsd"], a3_noise=RB.CONDITIONS["A3 device noise"]["noise"],
                a4_bias=RB.CONDITIONS["A4 order bias"]["order_bias"], a5_nopref=RB.CONDITIONS["A5 no preference"]["no_pref"],
                a7_from=RB.CONDITIONS["A7 fatigue"]["lapse_path"][0], a7_to=RB.CONDITIONS["A7 fatigue"]["lapse_path"][1],
                a8_from=RB.CONDITIONS["A8a s 1.5 to 0.75"]["s_path"][0], a8_to=RB.CONDITIONS["A8a s 1.5 to 0.75"]["s_path"][1],
                b4_a=RB.CONDITIONS["B4a drift 1 JND"]["drift_jnd"], b4_b=RB.CONDITIONS["B4b drift 2 JND"]["drift_jnd"])
    F4.write_csv(args.dest / "fig4m_meta.csv", list(meta), [list(meta.values())])
    F4.write_csv(args.dest / "fig4m_provenance.csv",
                 ["key", "setting", "source", "condition", "comparator", "n", "merged_from", "lead", "lo", "hi", "joint_lo", "joint_hi",
                  "lead_fixed", "at_cap"],
                 [[str(x).replace(" ", "_") if isinstance(x, str) else x for x in p] for p in prov])

    # ------------------------------------------------------------------ print
    print(f"{sum(not r['header'] for r in rows)} rows, figure {L.WIDTH:g} x {meta['height']:.1f} pt; scale {scale:.3f} pt per point")
    print(f"self-checks: {n_checked} comparisons equal amp_yardstick.json; no row above 4L; Fig. 2's comparator equals amp_table1.json")
    print(f"V3: level {gamma:.5f} per tail, coverage per cell {100 * (1 - 2 * gamma):.3f} %, normal equivalent z = {norm.isf(gamma):.2f}; "
          f"draws inside all cells {inside:.4f}")
    print(f"V4: clear jointly {len(clear)} of {len(family)}; not clear:")
    for c in not_clear:
        print(f"   {c['row']:<5}{c['cond']:<24}{c['st']:<9} vs {NAMES[c['b']]:<28}{c['est']:+6.1f} [{c['lo']:+6.1f}, {c['hi']:+6.1f}] "
              f"joint [{c['jlo']:+6.1f}, {c['jhi']:+6.1f}]{' CAP' if c['cap'] else ''}")
    below = [c for c in family if c["jhi"] < 0.0]
    print("joint upper end below 0: " + (", ".join(f"{c['cond']} {c['st']}" for c in below) or "no cell"))
    print(f"V2: clear with the selection inside, per cell: {len(clear_cell)} of {len(family)}; "
          f"not clear: {[(c['cond'], c['st']) for c in family if c['lo'] <= 0.0]}")
    falls = sorted(((float(np.quantile(c["chosen_draws"], 0.025)) - c["lo"], c["row"], c["cond"], c["st"]) for c in family), reverse=True)
    print(f"   fall of the lower end by the selection: median {np.median([f[0] for f in falls]):.2f}, largest {[(round(a, 2), d, e) for a, _, d, e in falls[:4]]}")
    print(f"V5: fixed rating: clear jointly {len(clear_fix)} of {len(family)}, level {gamma_fix:.5f}; sign changes: "
          f"{[(c['cond'], c['st'], round(c['est'], 1), round(c['fix'], 1)) for c in sign_changes] or 'none'}; largest gap {gap:.1f}")
    print(f"   not clear on the fixed rating: {[(c['cond'], c['st']) for c in family if c['jlo_fix'] <= 0.0]}")
    print("row    y     3-35 Nm                                  | fixed 12 Nm")
    for r in rows:
        if r["header"]:
            continue
        s = []
        for tag in ("w", "f"):
            c = r[tag]
            s.append("-" if c is None else f"{c['fx']:+6.1f} [{c['lo']:+6.1f},{c['hi']:+6.1f}] j[{c['jlo']:+6.1f},{c['jhi']:+6.1f}]"
                     + (" CAP" if c["cap"] else " BLUE" if c["blue"] else ""))
        print(f"  {r['key']:<5}{r['y']:6.2f} {s[0]:<42}| {s[1]}")
    print(f"labels: {labels}")
    who = {}
    for c in family:
        who.setdefault((c["st"], NAMES[c["b"]]), []).append(c["cond"])
    print("strongest comparator per cell: " + "; ".join(f"{st}, {n}: {len(v)}" for (st, n), v in sorted(who.items())))
    was_rows = {k_: {t: marker_4l(v, t) for t in ("w", "f") if v[f"in_{t}"] != "nan" or v[f"out_{t}"] != "nan"} for k_, v in fig4l.items()
                if not int(v["is_header"])}
    falls_4l = [(was_rows[r["key"]][t] - r[t]["fx"], r["key"], t) for r in rows if not r["header"] and r["key"] in was_rows
                for t in ("w", "f") if r[t] is not None and t in was_rows[r["key"]]]
    print(f"fall of the marker against 4L: median {np.median([f[0] for f in falls_4l]):.2f}, largest {sorted(falls_4l, reverse=True)[:6]}")
    near = sorted(((float(np.mean([v[1] for v in lev[c["src"], c["cond"], c["st"], c["opt"]].values()])), c["row"], c["cond"], c["st"])
                   for c in family), reverse=True)
    print(f"PsyPref's level, highest cells: {[(round(100 * a, 1), d, e) for a, _, d, e in near[:8]]}; "
          f"at least 90 %: {sum(a >= 0.90 for a, *_ in near)} of {len(near)}")
    for st in SETTINGS:
        c = by_row["MAIN"]["cells"][st][0]
        own, other = lev["FULL", "main", st, "psypref"], lev["FULL", "main", st, c["b"]]
        print(f"strongest in the main condition, {st}: {NAMES[c['b']]}")
        folds = sorted({v[0] for v in own.values()})
        per = [100 * np.mean([own[u][1] - other[u][1] for u in own if own[u][0] == f]) for f in folds]
        print(f"main, {st}, per held-out subject vs {NAMES[c['b']]}: {[round(float(p), 1) for p in per]}")
    print("macros: " + "; ".join(f"{k_} = {v}" for k_, v in macro.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
