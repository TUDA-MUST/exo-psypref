"""Fig. 4 of the ICORR paper (v2), two compact proposals: 4A (one column) and 4B (figure*).

Usage:
  python scripts/paper_v2_fig4ab.py
      [--out <repository>/out] [--dest figures/data] [--no-verify]

Post hoc, as scripts/paper_v2_fig4.py (whose helpers, yardstick check and verification this script
reuses; that script and its files stay untouched): PsyPref under a simulated deviation minus PsyPref
in FULL on the same users and answer seeds, preference over generic assistance at k = 24, in points,
3-35 Nm ("wide") and fixed 12 Nm, on the fixed yardstick (s = 1 Nm, lapse 0.1; filled markers, with
95 % intervals) and the planned scoring (hollow, drawn only where it differs from the filled one).
Every number is read from out/icorr/amp_free/amp_yardstick.json; unless --no-verify, every triple
written or quoted is recomputed from the stored session rows with the yardstick script's functions.

PsyPref's handling of "no preference" answers (PsyPref-T) replaced plain PsyPref in the paper, so the rows of the two conditions with such answers (A5
frequent "no preference", C4 pessimistic user) are TIE's arm "tie" on users 8-39 of each subject
(n = 160, disjoint from the screen); every other row is PsyPref on FULL's 200 users.

4A, one column: some conditions share a row. A merged row shows, per setting, the condition with
the lowest fixed-yardstick estimate (filled, with its interval) and that same condition's
planned estimate (hollow, where it differs), so filled and hollow are always the same sessions
scored two ways. Self-check: in every merged cell that condition is also the lowest on the
planned scoring, so the merge hides no planned result.
Merged: A6a-c (answer-curve shapes), B4a/B4b (drifts), B5a/B5b (judging single features; B5b's
fixed-12 cell is B5a's, bit-identical), B6a/B6b/B7 (gait phases weighted unequally), R7/C1/C2/C3
(Schaefer-family and other preferred shapes). A8a/A8b (discrimination improving / worsening) are
two rows: merged, the worst fixed estimate (A8a) and the worst planned one (A8b) are different
sessions. The pessimistic user is the last, bold row.
4B, figure*: every condition its own row (A6a-c separately; B5b's fixed-12 cell plotted once, in the
B5a row, as in fig4), the five groups in three columns; each column has its own x window, all at
the same scale (pt per point), written with the layout to fig4b_meta.csv.

Writes to <dest>:
  fig4a_rows.csv, fig4a_meta.csv                 4A: rows (y in row units) and axis/label values
  fig4b_rows_c0.csv, _c1.csv, _c2.csv, fig4b_meta.csv   4B: rows per column, per-column layout
  fig4b_groups.csv        4B: per column and group the header's y and the first and last row's y
  fig4ab_provenance.csv   per variant, row, setting and marker: the condition and entry plotted
  fig4ab_notes.csv        caption numbers (the PRB2 cells below -5 planned; the two cells where
                          PsyPref is not ahead of CMA-ES on Schaefer's family; PTI3's planned
                          value as amp_tie_report.md states it, the one to quote: the yardstick
                          JSON's own bootstrap of the same difference differs in the last digit);
                          the guide counts are printed
Self-checks (stop on failure): the JSON's own checks; every un-merged row equals fig4_rows.csv's
value for the same key; both variants cover the same conditions; in every cell the filled and the
hollow marker are the same condition, and in a merged cell it is the lowest on both yardsticks;
the five PRB2 cells below -5 points planned all appear as hollow markers in both; every plotted
value inside its axis; the rows where PsyPref is not ahead of CMA-ES on Schaefer's family are
exactly the ones the caption names; PTI3 in amp_tie_report.md agrees with the JSON's estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import paper_v2_fig4 as F4

DEST = F4.DEST
K, SETTINGS, TAG, N_BOOT, GUIDE, PRB2_LIMIT = F4.K, F4.SETTINGS, F4.TAG, F4.N_BOOT, F4.GUIDE, F4.PRB2_LIMIT


def C(src, *conds, opt="psypref"):
    return [(src, c, opt) for c in conds]


# ----------------------------------------------------------------------------- 4A rows
# (group key, [(row key, [(source, condition, optimizer), ...], bold)])
A_GROUPS = [
    ("G1", [("R1", C("ROB", "R1 s=0.5")), ("R2", C("ROB", "R2 s=2.0")),
            ("A2", C("ROB2", "A2 lapse 0.2")), ("R3", C("ROB", "R3 lapse=0.3")),
            ("A1", C("ROB2", "A1 individual s")),
            ("A6", C("ROB2", "A6a logistic link", "A6b Student-t link", "A6c Cauchy link")),
            ("A3", C("ROB2", "A3 device noise")), ("A4", C("ROB2", "A4 order bias")),
            ("A5T", C("TIE", "A5 no preference", opt="psypref_t"))]),
    ("G2", [("A7", C("ROB2", "A7 fatigue")), ("A8a", C("ROB2", "A8a s 1.5 to 0.75")),
            ("A8b", C("ROB2", "A8b s 0.75 to 1.5")),
            ("B4", C("ROB2", "B4a drift 1 JND", "B4b drift 2 JND")), ("B3", C("ROB2", "B3 familiarity"))]),
    ("G3", [("B1", C("ROB2", "B1 Weber")), ("B2", C("ROB2", "B2 asymmetry")),
            ("B5", C("ROB2", "B5a features static", "B5b features walking")),
            ("B67", C("ROB2", "B6a phase individual", "B6b phase hip speed", "B7 anisotropy")),
            ("B8", C("ROB2", "B8 random utility")), ("R4", C("ROB", "R4 negative power")),
            ("SH", C("ROB", "R7 surrogates") + C("ROB2", "C1 Schaefer shapes", "C2 heavy tails", "C3 family mixture"))]),
    ("BOLD", [("C4T", C("TIE", "C4 pessimistic", opt="psypref_t"))]),   # no header: the bold last row
]
# 4A y layout in row units, downwards
A_TOP_LABELS, A_HEADER_TO_ROW, A_GROUP_GAP, A_BOLD_GAP, A_BOTTOM_PAD, A_TOP_PAD = -1.15, 1.05, 1.3, 1.25, 0.72, 0.62
A_XMIN, A_XMAX = -23.0, 10.0   # +10: room for the "Main condition" label right of the 0 line

# ----------------------------------------------------------------------------- 4B rows
B_COLUMNS = [
    [("G1", [("R1", C("ROB", "R1 s=0.5")), ("R2", C("ROB", "R2 s=2.0")),
             ("A2", C("ROB2", "A2 lapse 0.2")), ("R3", C("ROB", "R3 lapse=0.3")),
             ("A1", C("ROB2", "A1 individual s")),
             ("A6a", C("ROB2", "A6a logistic link")), ("A6b", C("ROB2", "A6b Student-t link")),
             ("A6c", C("ROB2", "A6c Cauchy link")),
             ("A3", C("ROB2", "A3 device noise")), ("A4", C("ROB2", "A4 order bias")),
             ("A5T", C("TIE", "A5 no preference", opt="psypref_t"))])],
    [("G2", [("A7", C("ROB2", "A7 fatigue")), ("A8a", C("ROB2", "A8a s 1.5 to 0.75")),
             ("A8b", C("ROB2", "A8b s 0.75 to 1.5")), ("B4a", C("ROB2", "B4a drift 1 JND")),
             ("B4b", C("ROB2", "B4b drift 2 JND")), ("B3", C("ROB2", "B3 familiarity"))]),
     ("G4", [("R7", C("ROB", "R7 surrogates")), ("C1", C("ROB2", "C1 Schaefer shapes")),
             ("C2", C("ROB2", "C2 heavy tails")), ("C3", C("ROB2", "C3 family mixture"))])],
    [("G3", [("B1", C("ROB2", "B1 Weber")), ("B2", C("ROB2", "B2 asymmetry")),
             ("B5a", C("ROB2", "B5a features static")), ("B5b", C("ROB2", "B5b features walking")),
             ("B6a", C("ROB2", "B6a phase individual")), ("B6b", C("ROB2", "B6b phase hip speed")),
             ("B7", C("ROB2", "B7 anisotropy")), ("B8", C("ROB2", "B8 random utility")),
             ("R4", C("ROB", "R4 negative power"))]),
     ("G5", [("C4T", C("TIE", "C4 pessimistic", opt="psypref_t"))])],
]
# 4B y layout in row units (the group headers are footnotesize, so a little more room under them)
B_HEADER_TO_ROW, B_GROUP_GAP, B_BOTTOM_PAD, B_TOP_PAD = 1.12, 1.42, 0.7, 0.72
# 4B horizontal layout (pt): figure width, label column per column (measured in test_fig4b.pdf: the
# widest row labels end 89.1 / 95.0 / 112.5 pt from their column's start, the last "Judging single
# features, walking JND"; 4 pt clearance; the first column 3 pt more, for the key under its labels:
# its lines end at 92 pt), gap between a column's axis end and the next column's labels, x margin
# (points) around the data
B_WIDTH, B_LABEL_W, B_GAP, B_MARGIN = 516.0, (96.0, 99.0, 116.5), 9.0, 0.9

# the conditions every variant must cover: FULL's robustness rows of ROB (R5 and R6 left out, as
# fig4) and ROB2, with A5 and C4 from TIE
EXPECTED = ({("ROB", c) for c in ("R1 s=0.5", "R2 s=2.0", "R3 lapse=0.3", "R4 negative power", "R7 surrogates")}
            | {("ROB2", c) for c in (
                "A1 individual s", "A2 lapse 0.2", "A3 device noise", "A4 order bias", "A6a logistic link",
                "A6b Student-t link", "A6c Cauchy link", "A7 fatigue", "A8a s 1.5 to 0.75", "A8b s 0.75 to 1.5",
                "B1 Weber", "B2 asymmetry", "B3 familiarity", "B4a drift 1 JND", "B4b drift 2 JND",
                "B5a features static", "B5b features walking", "B6a phase individual", "B6b phase hip speed",
                "B7 anisotropy", "B8 random utility", "C1 Schaefer shapes", "C2 heavy tails", "C3 family mixture")}
            | {("TIE", "A5 no preference"), ("TIE", "C4 pessimistic")})
# where PsyPref (with its tie handling) is not ahead of CMA-ES on Schaefer's family at k = 24 on the
# fixed yardstick (interval not above 0); the caption names exactly these
NOT_AHEAD = {("ROB2", "C1 Schaefer shapes", "fixed 12"), ("TIE", "C4 pessimistic", "fixed 12")}


def cell(loss, members, setting, merge_rule):
    """One setting of one row: dict with the filled triple, the hollow value (nan if not drawn), the
    entries used, n. merge_rule 'worst fixed': the member with the lowest fixed estimate, filled
    and hollow both from it; it must also be the lowest on the planned scoring (4A); 'single'
    rows have one member."""
    cands = [loss[(s, c, setting, o)] for s, c, o in members if (s, c, setting, o) in loss]
    if len(cands) != len(members):
        raise SystemExit(f"missing entries for {members} {setting}")
    dups = [e for e in cands if e.get("duplicate_of")]
    live = [e for e in cands if not e.get("duplicate_of")]
    for e in dups:   # a duplicate must equal the condition it duplicates, and that one must be in the row
        twin = [x for x in live if x["condition"] == e["duplicate_of"]]
        if merge_rule == "single":
            return dict(fx=[math.nan] * 3, rg=[math.nan] * 3, show=math.nan, n=e["n"], fx_e=e, rg_e=e, dup=True)
        # bit-identical sessions: equal estimates (the intervals differ only by the bootstrap seed,
        # which hashes the condition's name); the row plots the twin's triple
        if not twin or abs(twin[0]["fixed"][0] - e["fixed"][0]) > 1e-12 \
                or abs(twin[0]["registered"][0] - e["registered"][0]) > 1e-12:
            raise SystemExit(f"duplicate {e['condition']} {setting} not equal to its twin in the row")
    ef = min(live, key=lambda d: d["fixed"][0])
    er = ef                      # filled and hollow: the same sessions scored two ways
    low_rg = min(d["registered"][0] for d in live)
    if er["registered"][0] > low_rg + 1e-12:
        raise SystemExit(f"merged cell {[m[1] for m in members]} {setting}: the lowest fixed estimate "
                         f"({ef['condition']}) is not the lowest planned one; split the row")
    fx, rg = F4.pts(ef["fixed"]), F4.pts(er["registered"])
    show = math.nan if (er is ef and ef["same_yardstick"]) or abs(rg[0] - fx[0]) < 1e-9 else rg[0]
    ns = {e["n"] for e in cands}
    assert len(ns) == 1, ns
    return dict(fx=fx, rg=rg, show=show, n=ns.pop(), fx_e=ef, rg_e=er, dup=False)


def build(groups, loss, rule, header_to_row, group_gap, bold_gap):
    """Rows of one column: [key, is_header, bold, y, (fx, lo, hi, rg, show, n) x 2, prb2, n_merged, tie]."""
    rows, prov, used, plotted, spans = [], [], [], [], []
    y = 0.0
    for gi, (gkey, members) in enumerate(groups):
        bold = gkey == "BOLD"
        if gi:
            y += bold_gap if bold else group_gap
        if not bold:
            rows.append([gkey, 1, 0, y] + [math.nan] * 5 + [0] + [math.nan] * 5 + [0] + [0, 0, 0])
            yy = y + header_to_row - 1.0
        else:
            yy = y - 1.0
        for key, conds in members:
            yy += 1.0
            r = [key, 0, int(bold), yy]
            for setting in SETTINGS:
                c = cell(loss, conds, setting, rule)
                r += c["fx"] + [c["rg"][0], c["show"], c["n"]]
                if not c["dup"]:
                    plotted += [c["fx"][1], c["fx"][2]] + ([] if math.isnan(c["show"]) else [c["show"]])
                    used += [c["fx_e"], c["rg_e"]]
                for mk, e in (("filled", c["fx_e"]), ("hollow", c["rg_e"])):
                    if mk == "hollow" and math.isnan(c["show"]):
                        continue
                    prov.append([key, setting, mk, e["source"], e["condition"], e["a"], e["n"],
                                 int(e["same_yardstick"]), e.get("duplicate_of") or "-",
                                 "|".join(m[1] for m in conds) if len(conds) > 1 else "-"])
            prb2 = int(all(s == "ROB2" and o == "psypref" and c[:1] == "A" for s, c, o in conds))
            r += [prb2, len(conds), int(conds[0][0] == "TIE")]
            rows.append(r)
        spans.append([gkey, y, yy - len(members) + 1.0, yy])   # header (or bold row), first and last row
        y = yy
    return rows, prov, used, plotted, y, spans


HEAD = (["key", "is_header", "bold", "y"]
        + [f"{c}_{t}" for t in ("w", "f") for c in ("fx", "fx_lo", "fx_hi", "rg", "rgshow", "n")]
        + ["prb2", "n_merged", "tie"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=F4.MAIN / "out")
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    src = args.out / "icorr" / "amp_free" / "amp_yardstick.json"
    js = json.loads(src.read_text(encoding="utf-8"))
    if not (js["post_hoc"] and js["n_boot"] == N_BOOT and js["fixed_yardstick"] == {"s": 1.0, "lapse": 0.1}
            and K in js["ks"]):
        raise SystemExit("amp_yardstick.json is not the post hoc fixed-yardstick file this figure expects")
    sc = js["self_check"]
    if sc["points_matched"] != sc["points_total"] or not all(e["ok"] for e in sc["exact"]):
        raise SystemExit("amp_yardstick.json's own self-check against the planned reports did not pass")
    loss, other = {}, {}
    for e in js["differences"]:
        if e["k"] != K:
            continue
        if e["kind"] == "loss vs FULL":
            loss[e["source"], e["condition"], e["setting"], e["a"]] = e
        else:
            other[e["source"], e["condition"], e["setting"], e["a"], e["b"]] = e

    # ------------------------------------------------------------------ 4A
    a_rows, a_prov, a_used, a_plot, a_last, _ = build(A_GROUPS, loss, "worst fixed", A_HEADER_TO_ROW,
                                                    A_GROUP_GAP, A_BOLD_GAP)
    a_ymin, a_ymax = A_TOP_LABELS - A_TOP_PAD, a_last + A_BOTTOM_PAD
    if not (A_XMIN < min(a_plot) and max(a_plot) < A_XMAX):
        raise SystemExit(f"4A values span [{min(a_plot):.2f}, {max(a_plot):.2f}], outside [{A_XMIN}, {A_XMAX}]")
    lab = F4.label_values(js)
    import icorr_amp_robust as ROB   # noqa: PLC0415 (scripts/ is on the path, as for paper_v2_fig4)
    lab.update(s_r1=float(ROB.CONDITIONS["R1 s=0.5"]["s"]), s_r2=float(ROB.CONDITIONS["R2 s=2.0"]["s"]))

    # ------------------------------------------------------------------ 4B
    b_cols, b_prov, b_used, b_last, b_plot, b_spans = [], [], [], [], [], []
    for i, groups in enumerate(B_COLUMNS):
        rows, prov, used, plotted, last, spans = build(groups, loss, "single", B_HEADER_TO_ROW, B_GROUP_GAP, 0.0)
        b_spans += [[i] + sp for sp in spans]
        b_cols.append(rows)
        b_prov.append(prov)
        b_last.append(last)
        b_plot.append(plotted)
        b_used += used
    b_ymin, b_ymax = -B_TOP_PAD, max(b_last) + B_BOTTOM_PAD
    # per-column x windows: the data plus a margin, outward to whole points; one scale for all
    wins = [(math.floor(min(p) - B_MARGIN), math.ceil(max(p) + B_MARGIN)) for p in b_plot]
    wins = [(min(lo, GUIDE - 1), max(hi, 1)) for lo, hi in wins]
    span = sum(hi - lo for lo, hi in wins)
    scale = (B_WIDTH - sum(B_LABEL_W) - 2 * B_GAP - 1.0) / span    # pt per point; 1 pt right margin
    b_meta, x = [], 0.0
    for (lo, hi), lw in zip(wins, B_LABEL_W):
        ax0 = x + lw
        axw = (hi - lo) * scale
        b_meta.append([x, ax0, axw, lo, hi, math.ceil(lo / 5) * 5, math.floor(hi / 5) * 5])
        x = ax0 + axw + B_GAP
    assert abs(b_meta[-1][1] + b_meta[-1][2] - (B_WIDTH - 1.0)) < 1e-9

    # ------------------------------------------------------------------ self-checks
    # (1) un-merged rows equal fig4_rows.csv (the current figure's data) for the same key
    cur = {}
    lines = (args.dest / "fig4_rows.csv").read_text(encoding="utf-8").splitlines()
    hd = lines[0].split(",")
    for ln in lines[1:]:
        d = dict(zip(hd, ln.split(",")))
        cur[d["key"]] = d
    n_same = 0
    for rows in [a_rows] + b_cols:
        for r in rows:
            if r[1] or r[0] not in cur:      # A6 (worst of A6a-c) is fig4's row too
                continue
            d = cur[r[0]]
            for i, t in enumerate(("w", "f")):
                b = 4 + 6 * i
                mine = [r[b], r[b + 1], r[b + 2], r[b + 4]]
                theirs = [float(d[f"fx_{t}"]), float(d[f"fx_{t}_lo"]), float(d[f"fx_{t}_hi"]), float(d[f"rgshow_{t}"])]
                for u, v in zip(mine, theirs):
                    if not ((math.isnan(u) and math.isnan(v)) or abs(u - v) < 1e-7 * max(1, abs(v))):
                        raise SystemExit(f"row {r[0]} {t} differs from fig4_rows.csv: {mine} vs {theirs}")
            n_same += 1
    # (2) both variants cover exactly the expected conditions
    def covered(groups_list):
        return {(s, c) for groups in groups_list for _, members in groups for _, conds in members for s, c, _ in conds}
    cov_a, cov_b = covered([A_GROUPS]), covered(B_COLUMNS)
    if not (cov_a == cov_b == EXPECTED):
        raise SystemExit(f"coverage differs: A-B {cov_a ^ EXPECTED}, B {cov_b ^ EXPECTED}")
    # (3) the PRB2 cells below the limit on the planned scoring are hollow markers in both variants
    a_conds = sorted({c for (s, c, st, a) in loss if s == "ROB2" and a == "psypref" and c[:1] == "A"})
    prb2 = [loss["ROB2", c, st, "psypref"] for c in a_conds for st in SETTINGS
            if 100 * loss["ROB2", c, st, "psypref"]["registered"][0] < PRB2_LIMIT]
    for name, prov in (("4A", a_prov), ("4B", [p for pc in b_prov for p in pc])):
        hollow = {(p[3], p[4], p[1]) for p in prov if p[2] == "hollow"}
        miss = [(e["condition"], e["setting"]) for e in prb2 if ("ROB2", e["condition"], e["setting"]) not in hollow]
        if miss:
            raise SystemExit(f"{name}: PRB2 cells below {PRB2_LIMIT} planned not drawn: {miss}")
    # (4) PsyPref (with its tie handling) against CMA-ES on Schaefer's family, every plotted cell
    vs_sch, not_ahead = [], set()
    for s, c in sorted(EXPECTED):
        opt = "psypref_t" if s == "TIE" else "psypref"
        for st in SETTINGS:
            e = other[s, c, st, opt, "cmaes_sch4"]
            vs_sch.append(e)
            if not e["fixed"][1] > 0:
                not_ahead.add((s, c, st))
    if not_ahead != NOT_AHEAD:
        raise SystemExit(f"cells not ahead of CMA-ES on Schaefer's family: {sorted(not_ahead)}, caption names {NOT_AHEAD}")
    # (5) in every cell of both variants, a hollow marker is the filled marker's own condition
    n_pairs = 0
    for prov in [a_prov] + b_prov:
        filled = {(p[0], p[1]): (p[3], p[4], p[5]) for p in prov if p[2] == "filled"}
        for p in prov:
            if p[2] == "hollow":
                if filled[p[0], p[1]] != (p[3], p[4], p[5]):
                    raise SystemExit(f"row {p[0]} {p[1]}: hollow {p[4]} is not the filled marker's {filled[p[0], p[1]]}")
                n_pairs += 1
    # (6) PTI3's planned value as amp_tie_report.md states it (the one to quote) agrees with the
    # JSON's estimate of the same difference (PsyPref-T minus CMA-ES on Schaefer's family, C4, fixed
    # 12 Nm); the intervals are two bootstraps and may differ in the last digit
    tie_md = (args.out / "icorr" / "amp_free" / "amp_tie_report.md").read_text(encoding="utf-8")
    m = re.search(r"PTI3 \(C4 fixed 12[^)]*\): held (-?\d+\.\d) \[(-?\d+\.\d), (-?\d+\.\d)\]", tie_md)
    if not m:
        raise SystemExit("PTI3 line not found in amp_tie_report.md")
    pti3 = [float(v) for v in m.groups()]
    e3 = other["TIE", "C4 pessimistic", "fixed 12", "psypref_t", "cmaes_sch4"]
    r3 = F4.pts(e3["registered"])
    if abs(pti3[0] - r3[0]) > 0.05 + 1e-9 or max(abs(pti3[1] - r3[1]), abs(pti3[2] - r3[2])) > 0.2:
        raise SystemExit(f"PTI3 in the TIE report {pti3} disagrees with amp_yardstick.json {r3}")

    # ------------------------------------------------------------------ write
    args.dest.mkdir(parents=True, exist_ok=True)
    F4.write_csv(args.dest / "fig4a_rows.csv", HEAD, a_rows)
    F4.write_csv(args.dest / "fig4a_meta.csv",
                 ["xmin", "xmax", "ymin", "ymax", "ylabels", "guide", "k", "n_boot"] + list(lab),
                 [[A_XMIN, A_XMAX, a_ymin, a_ymax, A_TOP_LABELS, GUIDE, K, N_BOOT] + list(lab.values())])
    for i, rows in enumerate(b_cols):
        F4.write_csv(args.dest / f"fig4b_rows_c{i}.csv", HEAD, rows)
    F4.write_csv(args.dest / "fig4b_meta.csv",
                 ["col", "x0", "ax0", "axw", "xmin", "xmax", "tick_lo", "tick_hi", "ymin", "ymax", "scale_pt",
                  "guide", "k"] + list(lab),
                 [[i] + m + [b_ymin, b_ymax, scale, GUIDE, K] + list(lab.values()) for i, m in enumerate(b_meta)])
    F4.write_csv(args.dest / "fig4b_groups.csv", ["col", "key", "y_header", "y_first", "y_last"], b_spans)
    prov_rows = [["A"] + p for p in a_prov] + [[f"B{i}"] + p for i, pc in enumerate(b_prov) for p in pc]
    F4.write_csv(args.dest / "fig4ab_provenance.csv",
                 ["variant", "key", "setting", "marker", "source", "condition", "optimizer", "n", "same_yardstick",
                  "duplicate_of", "merged_from"],
                 [[str(x).replace(" ", "_") for x in p] for p in prov_rows])

    notes = []

    def note(item, e, b):
        notes.append([item, e["source"], e["condition"].replace(" ", "_"), e["setting"].replace(" ", "_"), e["a"], b,
                      e["n"]] + F4.pts(e["registered"]) + F4.pts(e["fixed"]))

    for e in prb2:
        note("prb2_registered_below_-5", e, "FULL")
    for e in vs_sch:
        if (e["source"], e["condition"], e["setting"]) in NOT_AHEAD:
            note("not_ahead_of_cmaes_schaefer", e, "cmaes_sch4")
    notes.append(["registered_as_in_tie_report_pti3", "TIE", "C4_pessimistic", "fixed_12", "psypref_t", "cmaes_sch4",
                  e3["n"]] + pti3 + [math.nan] * 3)
    ahead = [100 * e["fixed"][1] for e in vs_sch if (e["source"], e["condition"], e["setting"]) not in NOT_AHEAD]
    F4.write_csv(args.dest / "fig4ab_notes.csv",
                 ["item", "source", "condition", "setting", "a", "b", "n", "registered", "registered_lo",
                  "registered_hi", "fixed", "fixed_lo", "fixed_hi"], notes)

    # ------------------------------------------------------------------ print
    def guide_count(rows):
        v = [r[4 + 6 * i] for r in rows if not r[1] for i in range(2) if not math.isnan(r[4 + 6 * i])]
        return sum(x >= GUIDE for x in v), len(v), sorted((r[0], TAG[SETTINGS[i]], round(r[4 + 6 * i], 1))
                                                           for r in rows if not r[1] for i in range(2)
                                                           if not math.isnan(r[4 + 6 * i]) and r[4 + 6 * i] < GUIDE)

    print(f"source: {src}\nwritten to {args.dest}: fig4a_rows.csv, fig4a_meta.csv, fig4b_rows_c0-2.csv, "
          "fig4b_meta.csv, fig4ab_provenance.csv, fig4ab_notes.csv")
    print(f"\n4A: {sum(not r[1] for r in a_rows)} rows, {sum(r[1] for r in a_rows)} headers; y {a_ymin:g} .. {a_ymax:g}; "
          f"axis {A_XMIN:g} .. {A_XMAX:g}, values {min(a_plot):+.2f} .. {max(a_plot):+.2f}")
    print("row   y      3-35 Nm fixed [95 %]  hollow | fixed 12 Nm fixed [95 %]  hollow | prb2 merged tie n")
    for r in a_rows:
        if r[1]:
            print(f"{r[0]}  y = {r[3]:g}")
            continue
        cells = []
        for i in range(2):
            b = 4 + 6 * i
            h = "" if math.isnan(r[b + 4]) else f"{r[b + 4]:+.1f}"
            cells.append(f"{r[b]:+6.1f} [{r[b + 1]:+6.1f}, {r[b + 2]:+6.1f}] {h:>6}")
        print(f"  {r[0]:<4}{r[3]:6.2f} {cells[0]} | {cells[1]} | {r[-3]} {r[-2]} {r[-1]} {r[9]}/{r[15]}")
    print("4A merged rows (filled from / hollow from):")
    for p in a_prov:
        if p[9] != "-":
            print("  ", p[0], p[1], p[2], p[4])
    gA = guide_count(a_rows)
    print(f"4A filled markers at or above {GUIDE:g}: {gA[0]} of {gA[1]}; below: {gA[2]}")
    print(f"\n4B: rows per column {[sum(not r[1] for r in c) for c in b_cols]}, y {b_ymin:g} .. {b_ymax:g}, "
          f"last row per column {b_last}; scale {scale:.3f} pt per point")
    for i, m in enumerate(b_meta):
        print(f"  column {i}: labels at {m[0]:.1f}, axis {m[1]:.1f} + {m[2]:.1f} pt, window {m[3]:g} .. {m[4]:g} "
              f"(data {min(b_plot[i]):+.2f} .. {max(b_plot[i]):+.2f}), ticks {m[5]:g} .. {m[6]:g}")
    gB = [guide_count(c) for c in b_cols]
    print(f"4B filled markers at or above {GUIDE:g}: {sum(g[0] for g in gB)} of {sum(g[1] for g in gB)}; "
          f"below: {[x for g in gB for x in g[2]]}")
    print(f"\nself-checks: {n_same} un-merged rows equal fig4_rows.csv; coverage {len(EXPECTED)} conditions in both; "
          f"{len(prb2)} PRB2 cells below {PRB2_LIMIT:g} planned drawn hollow in both "
          f"({', '.join(e['condition'] + ' ' + TAG[e['setting']] for e in prb2)}); "
          f"planned {max(100 * e['registered'][0] for e in prb2):+.1f} .. {min(100 * e['registered'][0] for e in prb2):+.1f}, "
          f"fixed {max(100 * e['fixed'][0] for e in prb2):+.1f} .. {min(100 * e['fixed'][0] for e in prb2):+.1f}")
    print(f"PsyPref ahead of CMA-ES (Schaefer) on the fixed yardstick in {len(ahead)} of {len(vs_sch)} plotted cells "
          f"(lowest lower bound {min(ahead):+.1f}); not ahead:")
    for n_ in notes:
        if n_[0] == "not_ahead_of_cmaes_schaefer":
            print(f"  {n_[1]} {n_[2]} {n_[3]} {n_[4]} (n = {n_[6]}): fixed {n_[10]:+.1f} [{n_[11]:+.1f}, {n_[12]:+.1f}], "
                  f"planned {n_[7]:+.1f} [{n_[8]:+.1f}, {n_[9]:+.1f}] (this script's bootstrap)")
    print(f"  planned value to quote for the pessimistic user (PTI3, amp_tie_report.md): "
          f"{pti3[0]:+.1f} [{pti3[1]:+.1f}, {pti3[2]:+.1f}]")
    print(f"same sessions: {n_pairs} hollow markers, each its filled marker's condition")
    print("label values: " + ", ".join(f"{k} = {v:.4g}" for k, v in lab.items()))

    if not args.no_verify:
        seen, todo = set(), []
        for e in a_used + b_used + prb2 + vs_sch:
            key = (e["kind"], e["source"], e["condition"], e["setting"], e["a"], e["b"], e["b_source"])
            if key not in seen:
                seen.add(key)
                todo.append(e)
        F4.verify(args.out, todo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
