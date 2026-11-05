"""Fig. 4 of the ICORR paper (v2), simplified column variant 4S: one message, one marker per setting.

Usage (Anaconda base python, as the earlier paper_v2 scripts; from the repository):
  python scripts/paper_v2_fig4s.py [--out <repository>/out] [--dest figures/data]
                                   [--no-verify]

Message (the caption's first sentence): after 24 comparisons in 3-35 and fixed 12 Nm, most broken
assumptions cost PsyPref less than 5 points, and only users who judge single features (11-17 points)
and the pessimistic user (6-10, with tie handling) cost more. The caption's second sentence ties the
points to chosen over generic assistance. The robustness runs did not include 3-12 Nm.

Post hoc. 4A's 22 rows and merges (scripts/paper_v2_fig4ab.py, whose cell/row builder and checks this
script reuses; that script and its files stay untouched), on the fixed yardstick only (s = 1 Nm,
lapse 0.1): circle 3-35 Nm, square fixed 12 Nm, 95 % intervals. The planned scoring is not drawn;
it moves to the caption and one Results sentence written here. Colour per cell by PRB2's rule (our design notes: "No answer condition (A) lowers PsyPref's preference by more than 5 points against FULL in
either setting (paired estimate)"), i.e. on the point estimate, applied post hoc to every row on the
fixed yardstick: estimate below -5 points = beyond the limit (icBlue, its value printed); otherwise
within (icBlueLight, the lighter PsyPref of Fig. 1's belief fan). A merged row shows, per setting, its
condition with the lowest fixed estimate (as 4A), so a light merged cell hides no condition beyond
the limit.

Value labels: every value sits on its own marker's line, left of its own interval (a forest plot's
reading order: value, bar). Where that place touches the row's other marker or bar, the label moves
away from it by the least amount that clears it (0.1-pt steps): a circle's label up, a square's
label down. A row whose square label moved down gets EXTRA_GAP rows of space below it, so that the
label's baseline stays clearly nearer its own row's baseline than the next row's.

Writes to <dest>:
  fig4s_rows.csv        one line per group header or row: y (row units); per setting the light (within)
                        and the blue (beyond) triple (nan where the cell has the other colour), the
                        value label (abs. value, one decimal), its place (0 none, 3 left of the
                        interval) and its vertical shift (pt, up positive); n_merged, tie (PsyPref with
                        tie handling, n = 160)
  fig4s_meta.csv        axis limits and the layout in pt (the figure reads it, the label check uses it),
                        the condition values the row labels print
  fig4s_provenance.csv  per row and setting: colour, source, condition, optimizer, n, merged from
  fig4s_notes.csv       every number the caption and the draft texts quote, with its source
  fig4s_text.tex        those numbers as macros and the draft texts (planned-scoring sentence,
                        Results paragraph, the shapes sentence for Table II's replacement), LaTeX,
                        \\input by test_fig4s.tex
Self-checks (stop on failure): the JSON's own checks; the rows equal 4A's filled values (fig4a_rows.csv)
key by key; coverage = 4A's 31 conditions; no estimate exactly at the limit; the cells beyond the
limit over all 31 conditions are exactly the single-feature and pessimistic-user cells the caption
names; no merged light cell hides a condition beyond the limit; every plotted value and every value
label inside the axis; every value label clear of every interval, marker, vertical line and header
(text size estimated from Times at 7 pt); every value label's centre within ROW_BAND pt of its own
row's centre and its baseline at least BASE_MIN pt from, and BASE_MARGIN pt farther from, every other
row's label baseline than from its own; PRB2's planned cells as amp_rob2_report.md states its
verdict (failed, worst cell and value); every failing PRB2 cell ends the session with a higher lapse
rate or noise level than the main condition (the reason the Results sentence gives); PTI3 as
amp_tie_report.md states it agrees with the JSON; the C1 cell at fixed 12 Nm (PsyPref - CMA-ES on
Schaefer's family) and PRB1 as amp_rob2_report.md states them agree with the JSON; the main condition's
level at k = 24 equals amp_table1.json's; the light cell closest to the limit is the one the Results
paragraph names; the pessimistic user combines six deviations; FULL has 200 users from 5 walking
subjects. Unless --no-verify, every quoted or plotted triple is recomputed from the stored session
rows (paper_v2_fig4.verify).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import paper_v2_fig4 as F4
import paper_v2_fig4ab as FAB

K, SETTINGS, TAG, N_BOOT, GUIDE, PRB2_LIMIT = F4.K, F4.SETTINGS, F4.TAG, F4.N_BOOT, F4.GUIDE, F4.PRB2_LIMIT
assert GUIDE == PRB2_LIMIT == -5.0

# rows: 4A's (same merges, same order); the labels change only in fig4s.tex
GROUPS = FAB.A_GROUPS
# y layout in row units, downwards (4A: header to row 1.05, group gap 1.3, bold gap 1.25). TOP_PAD: the
# top line's ascenders (4.8 pt above its baseline) stay inside the bounding box
TOP_LABELS, HEADER_TO_ROW, GROUP_GAP, BOLD_GAP, BOTTOM_PAD, TOP_PAD = -1.15, 1.05, 1.2, 1.2, 0.72, 0.70
EXTRA_GAP = 0.25                       # rows of extra space under a row whose square label moved down
# x layout (pt): label column 0..LAB_W, axis LAB_W..AX_R (246: the "+10" tick label ends inside the
# 252-pt column); row pitch; space under the axis (tick labels with the "<- Worse" cue, and a one-line
# x label whose baseline fig4s.tex sets 1.9 pt above the bottom: about 2.3 pt between the two lines);
# figure width
LAB_W, AX_R, PITCH, BOTTOM, WIDTH = 111.0, 246.0, 7.2, 17.4, 252.0
# -26: room for the -17.2 label left of its interval (-20.1); +10: room for the "Main condition" label
XMIN, XMAX = -26.0, 10.0
OFF, MARK_R, MARK_S = 0.25, 1.45, 1.25  # marker offset (row units), circle radius, square half side (pt)
VAL_GAPX = 1.5                         # gap between an interval's left end and its value label (pt)
NUDGE_MAX, NUDGE_STEP = 1.5, 0.1       # largest vertical shift of a value label away from its partner
# Times Roman at 7 pt (\scriptsize): figures 0.5 em, period 0.25 em, \textminus 0.564 em, figure
# height 0.676 em, x-height 0.45 em; widths padded by 6 %
EM = 7.0
CHAR_W = {**{d: 0.5 * EM for d in "0123456789"}, ".": 0.25 * EM, "-": 0.564 * EM}
FIG_H = 0.676 * EM
MID_TO_BASE = 0.5 * 0.45 * EM          # a row label's "mid" anchor lies 0.5 ex above its baseline
CLEAR = 0.5                            # least clearance (pt) between a value label and anything drawn
LINE_HALF, BAR_HALF = 0.25, 0.35       # half widths of the vertical lines and the interval bars (pt)
ROW_BAND = 3.0                         # a value label's centre lies within this of its row's centre (pt)
BASE_MIN, BASE_MARGIN = 2.0, 1.5       # baseline distances to other rows' label baselines (pt)

NUM_WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
SINGLE = ("B5a features static", "B5b features walking")       # judging single features
PESS = ("TIE", "C4 pessimistic", "psypref_t")                  # the pessimistic user, tie handling
CLOSEST = ("ROB", "R2 s=2.0", "fixed 12")                      # the light cell the Results text names
WHERE = {"wide": "in 3--35\\,Nm", "fixed 12": "at fixed 12\\,Nm"}


def text_w(s: str) -> float:
    return 1.06 * sum(CHAR_W[c] for c in s)


def f1(v: float) -> str:
    """One decimal, as printed (Python's correct rounding of the stored value)."""
    s = f"{v:.1f}"
    return "0.0" if s == "-0.0" else s


def sgn(v: float) -> str:
    """A signed number for LaTeX text: $-1.3$ / $+0.3$ / $0.0$."""
    s = f1(v)
    if s == "0.0":
        return "$0.0$"
    return f"$-{s[1:]}$" if s.startswith("-") else f"$+{s}$"


def tri(t) -> str:
    """A triple in points as LaTeX: $-3.1$ [$-4.7$, $-1.4$]."""
    return f"{sgn(t[0])} [{sgn(t[1])}, {sgn(t[2])}]"


def olist(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else (" and ".join(xs) if len(xs) == 2 else ", ".join(xs[:-1]) + ", and " + xs[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=F4.MAIN / "out")
    ap.add_argument("--dest", type=Path, default=F4.DEST)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    amp = args.out / "icorr" / "amp_free"
    src = amp / "amp_yardstick.json"
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

    # ------------------------------------------------------------------ rows (4A's builder)
    # (build's own list of plotted values includes 4A's hollow markers; 4S plots the fixed intervals only)
    rows, prov, used, _, last, _ = FAB.build(GROUPS, loss, "worst fixed", HEADER_TO_ROW, GROUP_GAP, BOLD_GAP)
    used = [e for e in used if (e["source"], e["condition"]) in FAB.EXPECTED]   # filled = hollow entries
    filled = {(p[0], p[1]): p for p in prov if p[2] == "filled"}
    last_key = [r[0] for r in rows if not r[1]][-1]

    # colour per cell (PRB2's rule on the point estimate)
    colour = {}
    for r in rows:
        if r[1]:
            continue
        for i, st in enumerate(SETTINGS):
            fx, lo, hi = r[4 + 6 * i: 7 + 6 * i]
            if any(math.isnan(v) for v in (fx, lo, hi)):
                raise SystemExit(f"row {r[0]} {st}: no value")
            if abs(fx - GUIDE) < 1e-9:
                raise SystemExit(f"row {r[0]} {st}: estimate at the limit, colour undefined")
            colour[r[0], st] = fx < GUIDE          # PRB2: more than 5 points lower, paired estimate
            if colour[r[0], st] and fx >= 0:
                raise SystemExit("a value beyond the limit must be negative (the label prints a minus)")

    scale = (AX_R - LAB_W) / (XMAX - XMIN)  # pt per point

    def xp(v):
        return LAB_W + (v - XMIN) * scale

    def layout(ys, ymin, ymax, linetop, nudge=None):
        """Cells, obstacles and the value-label boxes for row positions ys (row units). nudge: fixed
        shifts per (key, tag) in pt (down positive); None = find the least shift that clears."""
        def yp(y):      # pt from the axis top, downwards
            return (y - ymin) * PITCH

        cells = []
        for r in rows:
            if r[1]:
                continue
            for i, st in enumerate(SETTINGS):
                t = TAG[st]
                cells.append(dict(key=r[0], tag=t, st=st, y=ys[r[0]] + (-OFF if t == "w" else OFF), row_y=ys[r[0]],
                                  fx=r[4 + 6 * i], lo=r[5 + 6 * i], hi=r[6 + 6 * i], blue=colour[r[0], st],
                                  r=MARK_R if t == "w" else MARK_S))
        obstacles = []
        for c in cells:
            obstacles.append((f"{c['key']} {c['tag']} interval", c["key"], xp(c["lo"]), yp(c["y"]) - BAR_HALF,
                              xp(c["hi"]), yp(c["y"]) + BAR_HALF))
            obstacles.append((f"{c['key']} {c['tag']} marker", c["key"], xp(c["fx"]) - c["r"], yp(c["y"]) - c["r"],
                              xp(c["fx"]) + c["r"], yp(c["y"]) + c["r"]))
        for v, name in ((0.0, "main-condition line"), (GUIDE, "5-point line")):
            obstacles.append((name, "-", xp(v) - LINE_HALF, yp(linetop), xp(v) + LINE_HALF, yp(ymax)))
        for r in rows:
            if r[1]:    # a header's text runs from x = 0 toward the axis box: keep labels off its line
                obstacles.append((f"header {r[0]}", r[0], LAB_W, yp(ys[r[0]]) - FIG_H / 2 - 1.0, AX_R,
                                  yp(ys[r[0]]) + FIG_H / 2 + 1.0))

        def clearance(box):
            least, what = math.inf, ""
            for name, _, x0, y0, x1, y1 in obstacles:
                dx = max(x0 - box[2], box[0] - x1, 0.0)
                dy = max(y0 - box[3], box[1] - y1, 0.0)
                cl = math.hypot(dx, dy) if dx and dy else max(dx, dy)
                if cl < least:
                    least, what = cl, name
            return least, what

        boxes = []
        for c in cells:
            if not c["blue"]:
                continue
            c["lab"] = f1(-c["fx"])
            w = text_w("-" + c["lab"])
            away = -1.0 if c["tag"] == "w" else 1.0          # circle up (-), square down (+), in pt downwards
            x1 = xp(c["lo"]) - VAL_GAPX
            steps = [nudge[c["key"], c["tag"]]] if nudge else \
                [away * NUDGE_STEP * j for j in range(int(round(NUDGE_MAX / NUDGE_STEP)) + 1)]
            for dy in steps:
                cy = yp(c["y"]) + dy
                box = (x1 - w, cy - FIG_H / 2, x1, cy + FIG_H / 2)
                cl, what = clearance(box)
                if cl >= CLEAR:
                    break
            else:
                raise SystemExit(f"value label {c['key']} {c['tag']} finds no clear place (clearance {cl:.2f} pt "
                                 f"to {what})")
            c.update(dy=dy, box=box, clear=cl, near=what)
            boxes.append(c)
        return cells, boxes, yp

    # positions: 4A's, then EXTRA_GAP under any row whose square label moved down
    ys = {r[0]: r[3] for r in rows}
    ymin, linetop = TOP_LABELS - TOP_PAD, TOP_LABELS + 0.45   # the vertical lines start just under their labels
    cells, boxes, _ = layout(ys, ymin, last + BOTTOM_PAD, linetop)
    moved_down = {c["key"] for c in boxes if c["dy"] > 1e-9}
    order = [r[0] for r in rows]
    shift, ys2 = 0.0, {}
    for k_ in order:
        ys2[k_] = ys[k_] + shift
        if k_ in moved_down:
            shift += EXTRA_GAP
    ys, last = ys2, last + shift
    ymax = last + BOTTOM_PAD
    nudge = {(c["key"], c["tag"]): c["dy"] for c in boxes}
    cells, boxes, yp = layout(ys, ymin, ymax, linetop, nudge)

    # ------------------------------------------------------------------ self-checks
    # (1) the plotted values equal 4A's filled values (fig4a_rows.csv in --dest), key by key
    lines = (args.dest / "fig4a_rows.csv").read_text(encoding="utf-8").splitlines()
    hd = lines[0].split(",")
    a_rows = {d["key"]: d for d in (dict(zip(hd, ln.split(","))) for ln in lines[1:])}
    n_same = 0
    for r in rows:
        if r[1]:
            continue
        d = a_rows[r[0]]
        for i, t in enumerate(("w", "f")):
            mine = r[4 + 6 * i: 7 + 6 * i]
            theirs = [float(d[f"fx_{t}"]), float(d[f"fx_lo_{t}"]), float(d[f"fx_hi_{t}"])]
            if any(abs(u - v) > 1e-7 * max(1, abs(v)) for u, v in zip(mine, theirs)):
                raise SystemExit(f"row {r[0]} {t} differs from fig4a_rows.csv: {mine} vs {theirs}")
            n_same += 1
    if set(a_rows) != {r[0] for r in rows}:
        raise SystemExit("rows differ from 4A's")
    # (2) coverage: 4A's 31 conditions
    cov = {(s, c) for _, members in GROUPS for _, conds in members for s, c, _ in conds}
    if cov != FAB.EXPECTED:
        raise SystemExit(f"coverage differs: {cov ^ FAB.EXPECTED}")
    # (3) every condition-setting cell (B5b's fixed-12 cell is B5a's sessions, counted once): the
    # cells beyond the limit are exactly the single-feature and the pessimistic-user cells
    opt_of = {(s, c): o for _, members in GROUPS for _, conds in members for s, c, o in conds}
    all_cells = []
    for (s, c), o in sorted(opt_of.items()):
        for st in SETTINGS:
            e = loss[s, c, st, o]
            if e.get("duplicate_of"):
                continue
            all_cells.append(e)
    beyond = [e for e in all_cells if 100 * e["fixed"][0] < GUIDE]
    expect = {(e["source"], e["condition"]) for e in beyond}
    if expect != {("ROB2", SINGLE[0]), ("ROB2", SINGLE[1]), PESS[:2]}:
        raise SystemExit(f"conditions beyond the limit {sorted(expect)} are not the ones the caption names")
    n_cond = len(opt_of)
    n_cond_within = n_cond - len(expect)
    single = [100 * e["fixed"][0] for e in beyond if e["condition"] in SINGLE]
    pess = {e["setting"]: e for e in beyond if (e["source"], e["condition"]) == PESS[:2]}
    if set(pess) != set(SETTINGS):
        raise SystemExit("the pessimistic user is not beyond the limit in both settings")
    out_prov = []
    for r in rows:
        if r[1]:
            continue
        for st in SETTINGS:
            p = filled[r[0], st]
            out_prov.append([r[0], st.replace(" ", "_"), "blue" if colour[r[0], st] else "light", p[3], p[4], p[5],
                             p[6], p[9]])
    shown = {(p[0], p[1]): p for p in out_prov}
    hidden = [e for e in beyond if not any(p[3] == e["source"] and p[4] == e["condition"]
                                           and p[1] == e["setting"].replace(" ", "_") for p in out_prov)]
    for e in hidden:   # a hidden cell beyond the limit must sit in a merged row whose shown cell is blue
        rk = next(rk for _, members in GROUPS for rk, conds in members if (e["source"], e["condition"], e["a"]) in conds)
        if shown[rk, e["setting"].replace(" ", "_")][2] != "blue":
            raise SystemExit(f"merged row {rk} hides {e['condition']} {e['setting']} beyond the limit behind light")
    # the light cell closest to the limit, which the Results paragraph names
    within = [e for e in all_cells if 100 * e["fixed"][0] >= GUIDE]
    closest = max((e for e in within), key=lambda e: -e["fixed"][0])
    if (closest["source"], closest["condition"], closest["setting"]) != CLOSEST:
        raise SystemExit(f"the light cell closest to the limit is {closest['condition']} {closest['setting']}, "
                         f"not the one the Results paragraph names")
    # (4) plotted values and value labels inside the axis; labels clear of everything drawn and on
    # their own row
    plotted = [v for c in cells for v in (c["lo"], c["hi"])]
    if not (XMIN < min(plotted) and max(plotted) < XMAX):
        raise SystemExit(f"values span [{min(plotted):.2f}, {max(plotted):.2f}], outside [{XMIN}, {XMAX}]")
    row_mid = {r[0]: yp(ys[r[0]]) for r in rows}                      # centres (headers too), pt downwards
    row_base = {k_: v + MID_TO_BASE for k_, v in row_mid.items()}     # label baselines
    min_clear, assoc = math.inf, []
    for c in boxes:
        box = c["box"]
        if not (box[0] >= LAB_W + CLEAR and box[2] <= AX_R and box[1] >= yp(linetop) and box[3] <= yp(ymax) - CLEAR):
            raise SystemExit(f"value label {c['key']} {c['tag']} {box} outside the axis box")
        min_clear = min(min_clear, c["clear"])
        cy = (box[1] + box[3]) / 2
        base = box[3]                                                 # digits sit on the baseline
        d_mid = abs(cy - row_mid[c["key"]])
        d_own = abs(base - row_base[c["key"]])
        d_oth = min(abs(base - b) for k_, b in row_base.items() if k_ != c["key"])
        if d_mid > ROW_BAND or d_oth < BASE_MIN or d_oth < d_own + BASE_MARGIN:
            raise SystemExit(f"value label {c['key']} {c['tag']} does not sit on its own row: centre {d_mid:.2f} pt "
                             f"from its row's centre, baseline {d_own:.2f} pt from its row's and {d_oth:.2f} pt from "
                             f"the nearest other row's")
        assoc.append((c["key"], c["tag"], d_mid, d_own, d_oth))
    for i, b1 in enumerate(boxes):
        for b2 in boxes[i + 1:]:
            p, q = b1["box"], b2["box"]
            if not (p[2] + CLEAR <= q[0] or q[2] + CLEAR <= p[0] or p[3] + CLEAR <= q[1] or q[3] + CLEAR <= p[1]):
                raise SystemExit(f"value labels {b1['key']} {b1['tag']} and {b2['key']} {b2['tag']} overlap")
    # (5) PRB2 on the planned scoring, as amp_rob2_report.md states its verdict
    a_conds = [c for _, members in GROUPS for _, conds in members for s, c, o in conds
               if s == "ROB2" and o == "psypref" and c[:1] == "A"]
    a_conds.insert(a_conds.index("A6a logistic link"), "A5 no preference")   # PRB2 covered plain PsyPref here
    if sorted(a_conds) != sorted({c for (s, c, st, a) in loss if s == "ROB2" and a == "psypref" and c[:1] == "A"}):
        raise SystemExit("PRB2's conditions differ from the ROB2 answer conditions in the JSON")
    prb2_all = [loss["ROB2", c, st, "psypref"] for c in a_conds for st in SETTINGS]
    prb2 = [e for e in prb2_all if 100 * e["registered"][0] < PRB2_LIMIT]
    rob2_md = (amp / "amp_rob2_report.md").read_text(encoding="utf-8")
    m = re.search(r"PRB2 \([^)]*\): (\w+) — worst \('([^']+)', '([^']+)'\) (-?\d+\.\d)", rob2_md)
    worst = min(prb2_all, key=lambda e: e["registered"][0])
    if not m or m.group(1) != "failed" or (m.group(2), m.group(3)) != (worst["condition"], worst["setting"]) \
            or abs(float(m.group(4)) - 100 * worst["registered"][0]) > 0.05 + 1e-9:
        raise SystemExit(f"PRB2 in amp_rob2_report.md ({m and m.groups()}) disagrees with the JSON "
                         f"({worst['condition']} {worst['setting']} {100 * worst['registered'][0]:.2f})")
    # every failing cell ends the session noisier than the main condition (s > 1 or lapse > 0.1)
    differs = {(d["source"], d["condition"], d["setting"]): d for d in js["yardstick_differs"]}
    for e in prb2:
        d = differs.get((e["source"], e["condition"], e["setting"]))
        if not d or not (d["s_median"] > 1.0 or d["lapse_median"] > 0.1):
            raise SystemExit(f"PRB2 cell {e['condition']} {e['setting']} does not end with a higher lapse rate or "
                             f"noise level: {d}")
    # the failing cells by name (caption list; Results detail: both settings or one)
    lab = F4.label_values(js)
    import icorr_amp_robust as ROB   # noqa: PLC0415 (scripts/ is on the path)
    lab.update(s_r1=float(ROB.CONDITIONS["R1 s=0.5"]["s"]), s_r2=float(ROB.CONDITIONS["R2 s=2.0"]["s"]))
    words = {"A1 individual s": "a noise level differing per user",
             "A2 lapse 0.2": f"$\\lambda = {lab['lapse_a2']:g}$", "A3 device noise": "device noise",
             "A4 order bias": "a bias toward the first profile", "A5 no preference": "frequent ``no preference''",
             "A6a logistic link": "a logistic answer curve", "A6b Student-t link": "a Student-t answer curve",
             "A6c Cauchy link": "a Cauchy answer curve", "A7 fatigue": "fatigue",
             "A8a s 1.5 to 0.75": "a falling noise level", "A8b s 0.75 to 1.5": "a rising noise level"}
    fail_conds = [c for c in a_conds if any(e["condition"] == c for e in prb2)]
    fail_set = {c: [e["setting"] for e in prb2 if e["condition"] == c] for c in fail_conds}
    both = [words[c] for c in fail_conds if len(fail_set[c]) == 2]
    one = [f"{words[c]} {WHERE[fail_set[c][0]]}" for c in fail_conds if len(fail_set[c]) == 1]
    detail = ((olist(both) + " in both settings") if both else "") + (", and " if both and one else "") + \
        (olist(one) if one else "")
    plain_a5 = {e["setting"]: e for e in prb2_all if e["condition"] == "A5 no preference"}
    if any(100 * e["registered"][0] < PRB2_LIMIT for e in plain_a5.values()):
        raise SystemExit("plain PsyPref with frequent 'no preference' failed PRB2; the sentence says it held")
    # (6) PTI3 as amp_tie_report.md states it, against the JSON's estimate (as paper_v2_fig4ab)
    tie_md = (amp / "amp_tie_report.md").read_text(encoding="utf-8")
    m = re.search(r"PTI3 \(C4 fixed 12[^)]*\): (\w+) (-?\d+\.\d) \[(-?\d+\.\d), (-?\d+\.\d)\]", tie_md)
    if not m or m.group(1) != "held":
        raise SystemExit("PTI3 line not found in amp_tie_report.md, or not 'held'")
    pti3 = [float(v) for v in m.groups()[1:]]
    e3 = other["TIE", "C4 pessimistic", "fixed 12", "psypref_t", "cmaes_sch4"]
    r3, f3 = F4.pts(e3["registered"]), F4.pts(e3["fixed"])
    if abs(pti3[0] - r3[0]) > 0.05 + 1e-9 or max(abs(pti3[1] - r3[1]), abs(pti3[2] - r3[2])) > 0.2:
        raise SystemExit(f"PTI3 in the TIE report {pti3} disagrees with amp_yardstick.json {r3}")
    if not (f3[1] < 0 < f3[2]):
        raise SystemExit(f"PTI3 on the fixed yardstick {f3}: the text says 'not shown to trail' on both yardsticks")
    # (7) C1 at fixed 12 Nm (PsyPref - CMA-ES on Schaefer's family) and PRB1, as amp_rob2_report.md
    # states them, against the JSON
    row = next((ln for ln in rob2_md.splitlines() if ln.startswith("| C1 Schaefer shapes | fixed 12 |")), None)
    if row is None:
        raise SystemExit("C1 fixed 12 row not found in amp_rob2_report.md")
    hdr = next(ln for ln in rob2_md.splitlines() if ln.startswith("| condition | setting |"))
    col = [h.strip() for h in hdr.split("|")].index("PsyPref − CMA-ES (Schäfer)")
    mc = re.fullmatch(r"([+-]\d+\.\d) \[([+-]\d+\.\d), ([+-]\d+\.\d)\]", row.split("|")[col].strip())
    c1 = [float(v) for v in mc.groups()]
    e_c1 = other["ROB2", "C1 Schaefer shapes", "fixed 12", "psypref", "cmaes_sch4"]
    rc1, fc1 = F4.pts(e_c1["registered"]), F4.pts(e_c1["fixed"])
    if abs(c1[0] - rc1[0]) > 0.05 + 1e-9 or max(abs(c1[1] - rc1[1]), abs(c1[2] - rc1[2])) > 0.2 or not c1[2] < 0:
        raise SystemExit(f"C1 fixed 12 in amp_rob2_report.md {c1} disagrees with amp_yardstick.json {rc1}, or does "
                         f"not trail")
    if not (e_c1["same_yardstick"] and abs(rc1[0] - fc1[0]) < 1e-9):
        raise SystemExit("C1 fixed 12: the two yardsticks differ; the shapes sentence quotes one value")
    m1 = re.search(r"PRB1 \(ahead of CMA-ES on Schäfer in ≥ (\d+) % of cells\): (\w+) — (\d+) %", rob2_md)
    if not m1 or m1.group(2) != "held":
        raise SystemExit("PRB1 line not found in amp_rob2_report.md, or not 'held'")
    prb1_bar, prb1_share = m1.group(1), m1.group(3)
    # (8) the main condition's level at k = 24 (the caption's context) against amp_table1.json
    t1 = json.loads((amp / "amp_table1.json").read_text(encoding="utf-8"))
    main_lv = {}
    for st in SETTINGS:
        lv = [x for x in js["levels"] if x["source"] == "FULL" and x["condition"] == "main" and x["setting"] == st
              and x["optimizer"] == "psypref" and x["k"] == K]
        if len(lv) != 1 or not lv[0]["same_yardstick"]:
            raise SystemExit(f"main condition level {st}: {lv}")
        v = lv[0]["fixed"][0]
        if abs(v - t1["settings"][st]["rows"]["psypref"]["k"][str(K)]["est"]) > 1e-12:
            raise SystemExit(f"main condition level {st} {v} differs from amp_table1.json")
        main_lv[st] = 100 * v
    # (9) counts the caption states: the pessimistic user's deviations; FULL's users and subjects; TIE's n
    import icorr_amp_rob2 as RB   # noqa: PLC0415
    n_dev = len(RB.CONDITIONS["C4 pessimistic"])
    if n_dev != 6:
        raise SystemExit(f"the pessimistic user combines {n_dev} deviations, the label says six")
    full = [json.loads(ln) for ln in (amp / "full_sessions.jsonl").read_text(encoding="utf-8").splitlines() if ln]
    pp = [x for x in full if x["candidate"] == "psypref" and x["setting"] == "wide"]
    n_users, n_subj = len({x["user"] for x in pp}), len({x["fold"] for x in pp})
    n_main = {e["n"] for e in all_cells if e["source"] != "TIE"}
    n_tie = {e["n"] for e in all_cells if e["source"] == "TIE"}
    if n_main != {n_users} or len(n_tie) != 1:
        raise SystemExit(f"n: FULL {n_users} users, rows {n_main}, TIE {n_tie}")
    n_tie = n_tie.pop()

    # ------------------------------------------------------------------ numbers for the text
    reg = [100 * e["registered"][0] for e in prb2]
    fxd = [100 * e["fixed"][0] for e in prb2]
    sl, sh = min(-v for v in single), max(-v for v in single)
    pw, pf = F4.pts(pess["wide"]["fixed"]), F4.pts(pess["fixed 12"]["fixed"])
    pl, ph = min(-pw[0], -pf[0]), max(-pw[0], -pf[0])
    b5 = next(r for r in rows if r[0] == "B5")
    b5w, b5f = b5[4:7], b5[10:13]       # the row's cells: 3-35 Nm the worst, fixed 12 Nm the least bad
    if abs(b5w[0] + sh) > 1e-9 or abs(b5f[0] + sl) > 1e-9:
        raise SystemExit("the single-feature row's cells are not the worst (3-35 Nm) and the least bad (fixed "
                         "12 Nm) single-feature cells the text names")
    cl3 = F4.pts(closest["fixed"])
    nums = dict(
        k=str(K), limit=f"{-GUIDE:g}", sigma=f"{js['fixed_yardstick']['s']:g}", lapse=f"{js['fixed_yardstick']['lapse']:g}",
        n_users=str(n_users), n_subj=str(n_subj), n_tie=str(n_tie), ndev_word=NUM_WORD[n_dev],
        single_lo_int=f"{round(sl)}", single_hi_int=f"{round(sh)}", pess_lo_int=f"{round(pl)}",
        pess_hi_int=f"{round(ph)}", single_lo=f1(sl), single_hi=f1(sh), pess_w=f1(-pw[0]),
        pess_f=f1(-pf[0]), cond_within=str(n_cond_within), cond_all=str(n_cond),
        cells_within=str(len(all_cells) - len(beyond)), cells_all=str(len(all_cells)),
        prb2_fail=str(len(prb2)), prb2_cells=str(len(prb2_all)), prb2_reg_hi=f1(-max(reg)), prb2_reg_lo=f1(-min(reg)),
        prb2_fix_lo=f1(-max(fxd)), prb2_fix_hi=f1(-min(fxd)), prb2_fail_list=olist([words[c] for c in fail_conds]),
        pti3=tri(pti3), pti3_fixed=tri(f3), c1=tri(c1), prb1_bar=prb1_bar, prb1_share=prb1_share,
        main_w=f1(main_lv['wide']), main_f=f1(main_lv['fixed 12']), prb2_conds=str(len(a_conds)),
        single_w_tri=tri(b5w), single_f_tri=tri(b5f), pess_w_tri=tri(pw), pess_f_tri=tri(pf), closest_tri=tri(cl3),
        closest_sigma=f"{lab['s_r2']:g}", a5_w=f1(-100 * plain_a5["wide"]["registered"][0]),
        a5_f=f1(-100 * plain_a5["fixed 12"]["registered"][0]),
        markers_light=str(sum(not c["blue"] for c in cells)), markers_all=str(len(cells)))
    nums["prb2_fail_word"] = NUM_WORD[len(prb2)]
    sentence = (
        f"Rated instead with each user's noise level and lapse rate at the session's end, {nums['prb2_fail']} of "
        f"the {nums['prb2_cells']} cells for {nums['prb2_conds']} changes in how users answer, fatigue and a "
        f"changing noise level included, cost PsyPref more than {nums['limit']} points: {detail} "
        f"($-{nums['prb2_reg_hi']}$ to $-{nums['prb2_reg_lo']}$ points). On the fixed yardstick (post hoc), the same "
        f"cells cost {nums['prb2_fix_lo']}--{nums['prb2_fix_hi']} points. The end-of-session rating uses the higher "
        f"lapse rate or noise level that these users had at the session's end, and so it also counts their noisier "
        f"judgment. Frequent ``no preference'' answers cost plain PsyPref {nums['a5_w']} points in 3--35\\,Nm and "
        f"{nums['a5_f']} at fixed 12\\,Nm; Fig.~\\ref{{fig:robustness}} shows PsyPref with tie handling.")
    paragraph = (
        f"After {nums['k']} comparisons, {nums['cond_within']} of the {nums['cond_all']} simulated conditions cost "
        f"PsyPref less than {nums['limit']} points in both settings by the point estimate, on the fixed yardstick "
        f"(post hoc; Fig.~\\ref{{fig:robustness}}); a doubled noise level came closest ($\\sigma = "
        f"{nums['closest_sigma']}$\\,Nm: {nums['closest_tri']} points at fixed 12\\,Nm). Users who judged single "
        f"features cost it the most: {nums['single_hi']} points in 3--35\\,Nm with walking JNDs "
        f"({nums['single_w_tri']}) and {nums['single_lo']} at fixed 12\\,Nm ({nums['single_f_tri']}). The "
        f"pessimistic user, who combined {nums['ndev_word']} changes at once, cost PsyPref with tie handling "
        f"{nums['pess_w']} points in 3--35\\,Nm ({nums['pess_w_tri']}) and {nums['pess_f']} at fixed 12\\,Nm "
        f"({nums['pess_f_tri']}). For this user at fixed 12\\,Nm, PsyPref with tie handling was not shown to trail "
        f"CMA-ES on Sch\\\"afer et al.'s family ({nums['pti3']} points with the end-of-session rating; "
        f"{nums['pti3_fixed']} on the fixed yardstick, post hoc).")
    shapes = (
        f"At fixed 12\\,Nm, PsyPref trailed CMA-ES on Sch\\\"afer et al.'s family when the users' preferred "
        f"profiles had that family's shapes ({nums['c1']} points); it stayed ahead in "
        f"{nums['prb1_share']}\\,\\% of the robustness cells.")

    # ------------------------------------------------------------------ write
    args.dest.mkdir(parents=True, exist_ok=True)
    by_cell = {(c["key"], c["tag"]): c for c in cells}
    out_rows = []
    for r in rows:
        if r[1]:
            out_rows.append([r[0], 1, 0, ys[r[0]]] + ([math.nan] * 7 + [0, "-", 0, 0.0, 0]) * 2 + [0, 0, 0])
            continue
        o = [r[0], 0, r[2], ys[r[0]]]
        for i, st in enumerate(SETTINGS):
            c = by_cell[r[0], TAG[st]]
            t3 = [c["fx"], c["lo"], c["hi"]]
            nan3 = [math.nan] * 3
            o += (nan3 if c["blue"] else t3) + (t3 if c["blue"] else nan3) + [
                c["fx"], int(c["blue"]), c["lab"] if c["blue"] else "-", 3 if c["blue"] else 0,
                -c["dy"] if c["blue"] else 0.0, r[9 + 6 * i]]
        o += [r[17], r[18], int(r[0] == last_key)]
        out_rows.append(o)
    head = ["key", "is_header", "bold", "y"]
    for t in ("w", "f"):
        head += [f"in_{t}", f"in_lo_{t}", f"in_hi_{t}", f"out_{t}", f"out_lo_{t}", f"out_hi_{t}", f"fx_{t}",
                 f"blue_{t}", f"lab_{t}", f"pos_{t}", f"vdy_{t}", f"n_{t}"]
    head += ["n_merged", "tie", "last"]
    F4.write_csv(args.dest / "fig4s_rows.csv", head, out_rows)
    meta = dict(xmin=XMIN, xmax=XMAX, ymin=ymin, ymax=ymax, ylabels=TOP_LABELS, linetop=linetop, guide=GUIDE,
                k=K, n_boot=N_BOOT, lab_w=LAB_W, ax_r=AX_R, pitch=PITCH, bottom=BOTTOM, width=WIDTH, scale=scale,
                off=OFF, mark_r=MARK_R, mark_s=MARK_S, val_gapx=VAL_GAPX,
                height=BOTTOM + (ymax - ymin) * PITCH)
    meta.update({k_: float(v) for k_, v in lab.items()})
    meta["ndev_word"] = NUM_WORD[n_dev]
    meta["n_rows"] = len(out_rows)
    F4.write_csv(args.dest / "fig4s_meta.csv", list(meta), [list(meta.values())])
    F4.write_csv(args.dest / "fig4s_provenance.csv",
                 ["key", "setting", "colour", "source", "condition", "optimizer", "n", "merged_from"],
                 [[str(x).replace(" ", "_") for x in p] for p in out_prov])
    notes = [["k", K, "-", "-", "comparisons"],
             ["limit_points", -GUIDE, "spec 18 PRB2", "-", "paired estimate below -5 points = beyond"],
             ["conditions_within_both_settings", n_cond_within, "fixed yardstick", "-", f"of {n_cond}"],
             ["cells_within", len(all_cells) - len(beyond), "fixed yardstick", "-", f"of {len(all_cells)} (B5b fixed 12 = B5a)"],
             ["markers_light", sum(not c["blue"] for c in cells), "figure", "-", f"of {len(cells)}"]]
    for e in beyond:
        notes.append(["beyond_limit", 100 * e["fixed"][0], e["source"], e["condition"].replace(" ", "_"),
                      e["setting"].replace(" ", "_") + ("_hidden_by_merge" if e in hidden else "")])
    for nm, v in zip(("est", "lo", "hi"), cl3):
        notes.append([f"closest_within_{nm}", v, closest["source"], closest["condition"].replace(" ", "_"), "fixed_12"])
    notes.append(["prb2_conditions", len(a_conds), "spec 18 PRB2 (ROB2 answer conditions; plain PsyPref)", "-",
                  f"{len(prb2_all)} cells"])
    for e in prb2_all:
        notes.append(["prb2_registered" + ("_below_limit" if e in prb2 else ""), 100 * e["registered"][0], "ROB2",
                      e["condition"].replace(" ", "_"), e["setting"].replace(" ", "_")])
    for e in prb2:
        notes.append(["prb2_failed_cell_fixed", 100 * e["fixed"][0], "ROB2", e["condition"].replace(" ", "_"),
                      e["setting"].replace(" ", "_")])
    for v, nm in zip(pti3, ("est", "lo", "hi")):
        notes.append([f"pti3_registered_{nm}", v, "amp_tie_report.md", "C4_pessimistic", "fixed_12"])
    for v, nm in zip(f3, ("est", "lo", "hi")):
        notes.append([f"pti3_fixed_{nm}", v, "amp_yardstick.json (post hoc)", "C4_pessimistic", "fixed_12"])
    for v, nm in zip(c1, ("est", "lo", "hi")):
        notes.append([f"c1_vs_cmaes_sch_registered_{nm}", v, "amp_rob2_report.md", "C1_Schaefer_shapes", "fixed_12"])
    notes.append(["prb1_share_percent", float(prb1_share), "amp_rob2_report.md", "-", f"held; bar {prb1_bar} %"])
    for st in SETTINGS:
        notes.append(["main_condition_level_percent", main_lv[st], "amp_yardstick.json = amp_table1.json", "main",
                      st.replace(" ", "_")])
    notes += [["n_users", n_users, "full_sessions.jsonl", "-", "-"], ["n_subjects", n_subj, "full_sessions.jsonl", "-", "-"],
              ["n_tie", n_tie, "amp_yardstick.json", "-", "users 8-39"], ["pessimistic_deviations", n_dev,
                                                                       "icorr_amp_rob2.CONDITIONS", "-", "-"]]
    F4.write_csv(args.dest / "fig4s_notes.csv", ["item", "value", "source", "condition", "detail"], notes)
    macro = {"k": "fsK", "limit": "fsLimit", "sigma": "fsSigma", "lapse": "fsLapse", "n_users": "fsNusers",
             "n_subj": "fsNsubj", "n_tie": "fsNtie", "ndev_word": "fsNdevWord", "single_lo_int": "fsSingleLoInt",
             "single_hi_int": "fsSingleHiInt", "pess_lo_int": "fsPessLoInt", "pess_hi_int": "fsPessHiInt",
             "single_lo": "fsSingleLo", "single_hi": "fsSingleHi", "pess_w": "fsPessW", "pess_f": "fsPessF",
             "cond_within": "fsCondWithin", "cond_all": "fsCondAll", "cells_within": "fsCellsWithin",
             "cells_all": "fsCellsAll", "prb2_fail": "fsPrbFail", "prb2_fail_word": "fsPrbFailWord",
             "prb2_cells": "fsPrbCells", "prb2_reg_hi": "fsPrbRegHi", "prb2_reg_lo": "fsPrbRegLo",
             "prb2_fix_lo": "fsPrbFixLo", "prb2_fix_hi": "fsPrbFixHi", "prb2_fail_list": "fsPrbFailList",
             "pti3": "fsPtiThree", "pti3_fixed": "fsPtiThreeFixed", "c1": "fsShapesTrail", "prb1_bar": "fsPrbOneBar",
             "prb1_share": "fsPrbOneShare", "main_w": "fsMainW", "main_f": "fsMainF", "prb2_conds": "fsPrbConds",
             "single_w_tri": "fsSingleWTri", "single_f_tri": "fsSingleFTri",
             "pess_w_tri": "fsPessWTri", "pess_f_tri": "fsPessFTri", "closest_tri": "fsClosestTri",
             "closest_sigma": "fsClosestSigma", "a5_w": "fsNoPrefW", "a5_f": "fsNoPrefF",
             "markers_light": "fsMarkersLight", "markers_all": "fsMarkersAll"}
    tex = ["% fig4s_text.tex: written by scripts/paper_v2_fig4s.py (repository); do not edit by hand.",
           "% Every number for Fig. 4S's caption and the draft texts, from amp_yardstick.json, amp_table1.json,",
           "% amp_rob2_report.md and amp_tie_report.md (PTI3, C1 and PRB1 as those reports state them), checked",
           "% by the script."]
    tex += [f"\\def\\{macro[k_]}{{{nums[k_]}}}" for k_ in macro]
    tex += ["% draft Results sentences on the end-of-session rating (PRB2)",
            f"\\def\\fsRatingSentence{{{sentence}}}",
            "% draft Results paragraph (the figure's message)",
            f"\\def\\fsResultsParagraph{{{paragraph}}}",
            "% draft sentence for Table II's replacement (the C1 cell with its own interval, not a figure reference)",
            f"\\def\\fsShapesSentence{{{shapes}}}", ""]
    (args.dest / "fig4s_text.tex").write_text("\n".join(tex), encoding="utf-8")

    # ------------------------------------------------------------------ print
    print(f"source: {src}\nwritten to {args.dest}: fig4s_rows.csv, fig4s_meta.csv, fig4s_provenance.csv, "
          "fig4s_notes.csv, fig4s_text.tex")
    print(f"4S: {sum(not r[1] for r in rows)} rows, {sum(r[1] for r in rows)} headers; y {ymin:g} .. {ymax:g}; "
          f"axis {XMIN:g} .. {XMAX:g} ({scale:.3f} pt per point), values {min(plotted):+.2f} .. {max(plotted):+.2f}; "
          f"figure {WIDTH:g} x {meta['height']:.1f} pt; extra gap under {sorted(moved_down) or 'none'}")
    print("row    y      3-35 Nm fixed [95 %]        | fixed 12 Nm fixed [95 %]      | merged tie")
    for o in out_rows:
        if o[1]:
            print(f"{o[0]}  y = {o[3]:g}")
            continue
        cl = []
        for i in range(2):
            b = 4 + 12 * i
            v = o[b + 6]
            t3 = o[b + 3: b + 6] if o[b + 7] else o[b: b + 3]
            tag = f"BLUE {o[b + 8]} dy {o[b + 10]:+.1f}" if o[b + 7] else "light"
            cl.append(f"{v:+6.1f} [{t3[1]:+6.1f}, {t3[2]:+6.1f}] {tag:>17}")
        print(f"  {o[0]:<4}{o[3]:6.2f} {cl[0]} | {cl[1]} | {o[-3]} {o[-2]}")
    print(f"light markers {nums['markers_light']} of {nums['markers_all']}; over all {n_cond} conditions: "
          f"{n_cond_within} within the limit in both settings, cells {nums['cells_within']} of {nums['cells_all']}")
    print("beyond the limit (all conditions): " + ", ".join(
        f"{e['condition']} {TAG[e['setting']]} {100 * e['fixed'][0]:+.2f}{' (hidden by merge)' if e in hidden else ''}"
        for e in beyond))
    print(f"closest light cell: {closest['condition']} {closest['setting']} {cl3[0]:+.2f} [{cl3[1]:+.2f}, {cl3[2]:+.2f}]")
    print(f"value labels: least clearance {min_clear:.2f} pt (limit {CLEAR}); " + "; ".join(
        f"{c['key']} {c['tag']} x {c['box'][0]:.1f}-{c['box'][2]:.1f} y {c['box'][1]:.1f}-{c['box'][3]:.1f} "
        f"shift {c['dy']:+.1f} (nearest: {c['near']} {c['clear']:.2f})" for c in boxes))
    print("row association (centre from own row's centre; baseline from own / nearest other row's baseline, pt): "
          + "; ".join(f"{k_} {t_} {a:.2f}, {b:.2f} / {d:.2f}" for k_, t_, a, b, d in assoc))
    failed = ", ".join(f"{e['condition']} {TAG[e['setting']]} {v:+.2f}" for e, v in zip(prb2, reg))
    print(f"PRB2 planned: {len(prb2)} of {len(prb2_all)} cells below {PRB2_LIMIT:g} ({failed}); "
          f"fixed for the same cells {', '.join(f'{v:+.2f}' for v in fxd)}; amp_rob2_report.md verdict matches")
    print(f"PTI3 as amp_tie_report.md states it: {pti3}; JSON planned {r3[0]:+.2f} [{r3[1]:+.2f}, {r3[2]:+.2f}], "
          f"fixed {f3[0]:+.2f} [{f3[1]:+.2f}, {f3[2]:+.2f}]")
    print(f"C1 fixed 12 vs CMA-ES on Schaefer: report {c1}; JSON {rc1[0]:+.2f} [{rc1[1]:+.2f}, {rc1[2]:+.2f}]; "
          f"PRB1 held {prb1_share} % (bar {prb1_bar} %)")
    print(f"main condition at k = {K}: 3-35 Nm {main_lv['wide']:.2f} %, fixed 12 Nm {main_lv['fixed 12']:.2f} % "
          f"(= amp_table1.json)")
    print(f"n: {n_users} users from {n_subj} walking subjects; tie rows n = {n_tie}; pessimistic user {n_dev} deviations")
    print(f"self-checks: {n_same} cells equal fig4a_rows.csv's filled values; coverage {len(cov)} conditions")
    print("\nmacros: " + ", ".join(f"{macro[k_]} = {nums[k_]}" for k_ in macro))
    print("\nregistered-scoring sentences:\n  " + sentence + "\nResults paragraph:\n  " + paragraph
          + "\nshapes sentence:\n  " + shapes)

    if not args.no_verify:
        seen, todo = set(), []
        for e in used + all_cells + prb2_all + [e3, e_c1]:
            key = (e["kind"], e["source"], e["condition"], e["setting"], e["a"], e["b"], e["b_source"])
            if key not in seen:
                seen.add(key)
                todo.append(e)
        F4.verify(args.out, todo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
