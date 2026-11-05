"""Fig. 2 of the ICORR 2027 paper (claim 1, fewer comparisons): the CSVs for figures/fig2.tex
(variant v1) and figures/fig2r.tex (variant r).

Usage:
  python scripts/paper_v2_fig2.py [--variant v1|r|all]
v1 (default) writes the CSVs below, byte-identical to before variant r existed; r writes only
fig2r_panels.csv (fig2r.tex reads the v1 curves, users and meta files, whose data it shares);
all writes both.

Variant r (2026-10-30, the message pass: one message per figure, readable without the caption):
the same data and the same checks, plus the layout of the in-panel labels that name the
comparator and explain the arrow, each placed from the data and checked against every drawn line:
  - in the first panel, a gloss above the arrow's tail naming its start ("CMA-ES after 40", the
    marker of that panel's comparator before it), and "~20 min" right of the k = 24 rule at the
    bottom of the panel;
  - in a panel whose comparator differs from the first panel's (3-35 Nm: Arens' GP), the
    comparator's name with its marker under its curve;
  - the clipped Schaefer curve's start ("Start 23 %"), bottom left;
  - the direct labels right of the last panel spread for the comparator's two-line label (its
    name in that panel and "8 weights"), not the three-line "CMA-ES or Arens' GP, 8 weights".
The label boxes use the widths and heights LaTeX measured for their text at \scriptsize in
IEEEtran (LABEL_BOX); the script stops if a box comes closer than CLEAR_PT to a curve, the
comparator's markers, the arrow and its dot, the generic or the best-profile level line, or leaves the axis.

Variant c (2026-11-02): only fig2c_users.csv and fig2c_meta.csv, for figures/fig2c.tex, whose fourth
panel draws the share of users at least 1 JND better than generic assistance against comparisons,
pooled over the settings, instead of the bars at k = 24. Checked against fig2_users.csv at k = 24.

Post hoc. Preference over generic assistance ("chosen over generic assistance (%)") against
comparisons k = 0-40 in the three settings 3-12 Nm (medium), 3-35 Nm (wide), fixed 12 Nm, for
  - PsyPref and GP-BALD: FULL's rows (full_sessions.jsonl);
  - the strongest other optimizer comparator per setting: CMA-ES on the ProMP with population 4 (cmaes4) in medium and
    fixed 12, Arens' GP with a start at the prior's mean and no gate (g_generic_nogate) in wide, CMP's rows
    (cmp_sessions.jsonl);
  - CMA-ES on Schaefer's family in its planned configuration,
    FULL's rows.
Preference per user and k: 100 (lam/2 + (1 - lam) Phi((d0 - d_k) / s)), s and lam from
amp_table1.json (1.0 Nm, 0.1); the stored distances[i] is d after i + 1 comparisons. The curve is
the mean over the 200 users; its 95 % band from 2000 bootstrap draws, users resampled within
their fold (walking subject), rows sorted by user, one index draw per replicate shared by all
curves of a setting (the users are paired), fixed seed. k = 0 (the start curve, no session) is
amp_table1.json's estimate and interval; the best-profile level (preference at each user's best 8-weight
profile) and the first k at which PsyPref's mean reaches the comparator's k = 40 level are
amp_table1.json's too. The title's distance of generic assistance is the median d0 (Nm) of the
200 users.

The fourth panel ("Users after 24 comparisons", author 2026-10-29): per curve (PsyPref, GP-BALD,
the comparator, CMA-ES on Schaefer's family; the same rows as the curves) the share of users whose
identified assistance after 24 comparisons (distances[23]) is at least one JND closer to their
preferred profile than generic assistance (d0 - d24 >= JND: better), at least one JND farther
(d24 - d0 >= JND: worse), or neither (within); JND = Phi^-1(0.75) s = 0.674 Nm at s = 1 Nm, the
definition of amp_posthoc_users.md. Pooled over the three settings (3 x 200 users), the
per-setting shares written beside them for the caption; whole % rounded halves to even, as the
reports print them (one rounding rule across the paper). The pooled shares'
95 % intervals from the curves' index draws (per draw the mean of the three settings' shares); the
caption gives the largest distance of an interval end from its share, rounded up (users_ci_pm).

The decomposition that the fourth panel showed before (what carries PsyPref's gain) is computed
and written for the record and the Results text, not plotted: at k = 24, PsyPref with one part
replaced minus FULL's PsyPref, in points, paired by user (same users, folds, answer seeds,
amplitudes and start distance d0; checked), per setting:
  - abl: the three times wider prior (abl_sessions.jsonl, our design notes, arm "wide": same
    centre and directions, z* ~ N(0, 9 I), three times the search radius);
  - rnd: random pairs;
  - gpbald: GP-BALD's answer model, i.e. FULL's GP-BALD (same prior, also information-based
    pairs, a preference GP instead of the psychometric answer model).
Their 95 % intervals use the same index draws as the curves of that setting.

The curve panels show 40-100 %; a curve below 40 % is clipped and its k = 0 value and entry k
are written for the note (only CMA-ES on Schaefer's family may leave the axis; the script stops
if any other curve or band does).

Self-checks (the script stops if one fails): every curve's users identical and sorted, folds
equal across curves; the mean at k = 12, 24, 40 equals amp_table1.json's estimate to 1e-9; the
first k recomputed from the mean curves equals amp_table1.json's; the comparators equal our design notes's and are the highest at k = 24 of the published optimizers run (CMP's cells and
FULL's); the fourth panel's shares add to 100 % per user set, its JND equals the 0.674 Nm that
amp_posthoc_users.md prints, and its per-setting shares of PsyPref, GP-BALD and CMA-ES on
Schaefer's family (FULL's rows, as that report's) equal the report's (to the printed 1 %; the
comparator's CMP rows are not in it); the parts are paired with FULL's PsyPref and their point
estimates equal amp_abl_report.md's and amp_rnd_report.md's (to the printed 0.1 point) and, for
GP-BALD, amp_table1.json's psypref_minus_points (to 1e-9).

Outputs (figures/data/):
  fig2_curves_{a,b,c}.csv  k, then per curve the mean and (PsyPref, comparator) band, in %
  fig2_panels.csv          one row per panel: comparator code, Schaefer variant, median d0,
                           best-profile level, comparator's k = 40 level, first k and its interval, the
                           direct labels' vertical positions (spread so they do not overlap;
                           fig2.tex uses the last panel's, where the labels stand), the line ends,
                           and the clipped Schaefer curve's k = 0 value and entry k
  fig2_users.csv           the fourth panel: one row per bar, top to bottom (psypref, gpbald,
                           best, sch), the pooled n and shares better / within / worse (%), the
                           printed better and worse shares (whole %, halves to even), per setting
                           (m 3-12, w 3-35, f fixed 12 Nm) the shares better and worse (%), and
                           the pooled shares' 95 % intervals (%)
  fig2_parts.csv           not plotted (Results text): one row per part (abl, rnd, gpbald), per
                           setting the paired difference and its interval, in points
  fig2_meta.csv            seconds per comparison, n users, draws, seed, s, lapse, the y range
                           of the curve panels, the fourth panel's k, the JND (Nm) and the
                           largest distance of its intervals' ends (points, rounded up)
  fig2r_panels.csv         (variant r) fig2_panels.csv's columns, the direct labels spread for
                           variant r's label heights, and per panel the in-panel labels: a flag,
                           the anchor's k and its baseline (%) for the gloss (anchor base east),
                           the comparator's name, "~20 min" and "Start ..." (anchor base west)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8")

D = Path(__file__).resolve().parents[1] / "out" / "icorr" / "amp_free"
OUT = Path(__file__).resolve().parents[1] / "figures" / "data"

PANELS = (("a", "medium"), ("b", "wide"), ("c", "fixed 12"))   # 3-12 / 3-35 / fixed 12 Nm
SPEC_BEST = {"medium": "cmaes4", "wide": "g_generic_nogate", "fixed 12": "cmaes4"}  # section 23
CMP_CODE = {"cmaes4": 1, "g_generic_nogate": 2}       # 1 CMA-ES (square), 2 Arens' GP (triangle)
SCH_CODE = {"cmaes_sch": 1, "cmaes_sch4": 2}          # 1 default population, 2 tuned (16a)
N_COMP = 40
N_BOOT = 2000
SEED = 20271029
SEC_PER_COMPARISON = 50.0                             # story_concept: about 50 s per comparison
K24 = 24                                              # the fourth panel's k (and the parts')
USERS = ("psypref", "gpbald", "best", "sch")          # the fourth panel's bars, top to bottom
USERS_REPORT = {"psypref": "PsyPref", "gpbald": "GP-BALD",          # amp_posthoc_users.md's
                "sch": "CMA-ES (Schäfer, tuned)"}                   # names (FULL's rows only)
PARTS = ("abl", "rnd", "gpbald")                      # the parts (not plotted)

# Layout shared with fig2.tex (pt): the axis height for 40-100 % (author, 2026-10-29: from 20-100,
# so that the differences look larger) and the heights of the direct labels (\scriptsize,
# \linespread 0.9, inner sep 0.6 pt) plus a 0.8-pt gap, to spread them in the label column at the
# right of the last panel.
AXIS_H_PT, YMIN, YMAX = 70.0, 40.0, 100.0
LABEL_H_PT = {1: 7.0 + 0.8, 2: 13.3 + 0.8, 3: 19.6 + 0.8}
LABEL_LINES = {"psypref": 1, "gpbald": 1, "best": 3, "sch": 2, "generic": 2}
LABEL_TOP = 99.0                                      # highest label centre (%)

# Variant r (fig2r.tex). The comparator's direct label right of the last panel has two lines (its
# name in that panel, "8 weights"); the axes are 95 pt wide (40 comparisons) and AXIS_H_PT high.
LABEL_LINES_R = {"psypref": 1, "gpbald": 1, "best": 2, "sch": 2, "generic": 2}
AXIS_W_PT = 95.0
R_PANELS = "fig2r_panels.csv"
# the in-panel labels' boxes (pt): width, height, depth of their text at \scriptsize in IEEEtran
# (Times), as LaTeX typed them out (\sbox, \wd, \ht, \dp; 2026-10-30, the session's
# measure_labels.tex): the gloss "<marker> CMA-ES after 40", the comparator's name
# "<triangle> Arens' GP", "\approx 20 min", "Start 23 %"; the gloss's width is the CMA-ES text's
# (the first panel's comparator; checked below)
LABEL_BOX = {"gloss": (56.17, 4.77, 0.10), "cmplab": (35.71, 4.77, 0.10),
             "min20": (26.48, 4.77, 0.11), "start": (29.67, 4.77, 0.10)}
GLOSS_DY_PT = 3.4                     # the gloss's baseline above the arrow
GLOSS_RIGHT_K = 39.6                  # its right end (anchor base east), in comparisons
CMPLAB_AT = {"wide": (5.5, 62.0)}     # the comparator's name: left end (k), baseline (%)
MIN20_DX_PT, MIN20_BASE = 1.2, 41.4   # "~20 min": right of the k = 24 rule, baseline (%)
START_DX_PT, START_BASE = 0.6, 44.9   # "Start ...": from the y axis, baseline (%)
CLEAR_PT = 0.75                       # least distance of a label box to a drawn line (pt)
# half widths (pt) of the drawn lines (icorr_style.tex) and the comparator's markers' half sizes
# (square 1.6 pt, triangle 2 pt, plus half their 0.6-pt stroke)
LINE_HALF = {"psypref": 0.75, "gpbald": 0.45, "best": 0.6, "sch": 0.45, "generic": 0.3,
             "ceiling": 0.4, "arrow": 0.35}
MARK_HALF = {1: 1.9, 2: 2.3}
C_LABEL_PCT = 10.5                    # variant c: a value label's height (7.3 pt of 70 pt for 100 %)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def cell(src: list[dict], setting: str, cand: str) -> list[dict]:
    rs = sorted((r for r in src if r["setting"] == setting and r["candidate"] == cand and not r.get("error")),
                key=lambda r: r["user"])
    if not rs:
        raise SystemExit(f"no rows for {setting} {cand}")
    return rs


def pref(rs: list[dict], s: float, lam: float) -> np.ndarray:
    """Users x k = 1..40, in %."""
    d0 = np.array([r["d0"] for r in rs])[:, None]
    dk = np.array([r["distances"][:N_COMP] for r in rs])
    if dk.shape[1] != N_COMP:
        raise SystemExit("fewer than 40 stored distances")
    return 100.0 * (lam / 2 + (1 - lam) * norm.cdf((d0 - dk) / s))


def spread(desired: list[float], heights: list[float], top: float) -> list[float]:
    """Label centres (in %), ordered top-down as given, at least half their heights apart, as close
    to the desired ones as possible (least squares, by pooling adjacent violators), the top one
    at most `top`."""
    n = len(desired)
    gaps = [(heights[i] + heights[i + 1]) / 2 for i in range(n - 1)]
    off = np.concatenate([[0.0], np.cumsum(gaps)])
    z = np.array(desired) + off                     # constraint: z non-increasing
    blocks = [[z[i], 1, [i]] for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            a, b = blocks[i], blocks.pop(i + 1)
            a[0] = (a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1])
            a[1] += b[1]
            a[2] += b[2]
            i = max(i - 1, 0)
        else:
            i += 1
    zz = np.empty(n)
    for v, _, idx in blocks:
        zz[idx] = v
    y = zz - off
    y[0] = min(y[0], top)
    for j in range(1, n):
        y[j] = min(y[j], y[j - 1] - gaps[j - 1])
    return [float(v) for v in y]


def report_k24(path: Path) -> dict[str, str]:
    """A test plan report's first table: setting -> the paired difference's estimate at k = 24
    (its fifth column, '<est> [<lo>, <hi>]'), as printed."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 5 and c[1] == str(K24) and c[0] not in out:
            out[c[0]] = c[4].split(" ")[0]
    return out


def report_users(path: Path) -> tuple[str, dict[str, dict[str, tuple[str, str]]]]:
    """amp_posthoc_users.md: its printed JND (Nm) and, per setting section and optimizer at
    k = 24, the better-off and worse-off shares' estimates (%), as printed."""
    text = path.read_text(encoding="utf-8")
    jnd = re.search(r"JND (\d+\.\d+) Nm \(s = 1\.0 Nm\)", text)
    if not jnd:
        raise SystemExit(f"{path.name}: no 'JND ... Nm (s = 1.0 Nm)'")
    out: dict[str, dict[str, tuple[str, str]]] = {}
    setting = None
    for line in text.splitlines():
        if line.startswith("## "):
            setting = line[3:].strip()
            out[setting] = {}
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if setting and len(c) >= 4 and c[1] == str(K24):
            out[setting][c[0]] = (c[2].split(" ")[0], c[3].split(" ")[0])
    return jnd.group(1), out


def pct_round(count: int, n: int) -> int:
    """100 count / n rounded to a whole %, halves to even (exact), as the reports print it
    (f"{x:.0f}"); checked against that format below."""
    v = round(Fraction(100 * count, n))
    if f"{100 * count / n:.0f}" != str(v):
        raise SystemExit(f"rounding {count}/{n}: {v} vs {100 * count / n:.0f}")
    return v


def report_side_by_side(path: Path) -> dict[str, dict[str, str]]:
    """amp_rnd_report.md's side-by-side table at k = 24: setting -> {abl, rnd, gpbald: estimate}."""
    text = path.read_text(encoding="utf-8")
    sec = text.split("## Side by side at k = 24", 1)[1].split("\n## ", 1)[0]
    out = {}
    for line in sec.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*(-?[\d.]+) \[[^|]*\|\s*(-?[\d.]+) \[[^|]*\|\s*(-?[\d.]+) \[", line)
        if m:
            out[m.group(1)] = dict(zip(PARTS, m.group(2, 3, 4)))
    return out


def g(x: float) -> str:
    return f"{x:.10g}"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([g(v) if isinstance(v, float) else v for v in r])


def box_gap(box: tuple[float, float, float, float], px: np.ndarray, py: np.ndarray) -> float:
    """Least distance (pt) from the box (x0, x1, y0, y1; pt) to the points (pt)."""
    x0, x1, y0, y1 = box
    dx = np.maximum(np.maximum(x0 - px, 0.0), px - x1)
    dy = np.maximum(np.maximum(y0 - py, 0.0), py - y1)
    return float(np.hypot(dx, dy).min())


def layout_r(setting: str, cols: dict[str, np.ndarray], level40: float, first_k: int, ceiling: float,
             cmp_code: int, sch_clip: int, sch_k0: float, first_cmp_code: int | None) -> tuple[list, list[str]]:
    """Variant r's in-panel labels of one panel (first_cmp_code None: the first panel, else the
    first panel's comparator code): their flags and anchors, after checking every box against the
    panel's drawn lines (stops if one comes closer than CLEAR_PT or leaves the axis). Returns the
    CSV cells and the printed clearances."""
    xs, ys = AXIS_W_PT / N_COMP, AXIS_H_PT / (YMAX - YMIN)          # pt per comparison, per %
    kk = np.linspace(0.0, N_COMP, 40 * N_COMP + 1)
    k = np.arange(N_COMP + 1, dtype=float)

    def pt(kv, yv):
        return np.asarray(kv, dtype=float) * xs, (np.asarray(yv, dtype=float) - YMIN) * ys

    # the drawn lines: (name, x pt, y pt, half width); curves only where inside the axis
    lines = []
    for n in ("psypref", "gpbald", "best", "sch"):
        y = np.interp(kk, k, cols[n])
        m = (y >= YMIN) & (y <= YMAX)
        lines.append((n, *pt(kk[m], y[m]), LINE_HALF[n]))
    lines.append(("generic", *pt(kk, np.full_like(kk, 50.0)), LINE_HALF["generic"]))
    lines.append(("ceiling", *pt(kk, np.full_like(kk, ceiling)), LINE_HALF["ceiling"]))
    ka = np.linspace(first_k, N_COMP, 400)
    lines.append(("arrow", *pt(ka, np.full_like(ka, level40)), LINE_HALF["arrow"]))
    lines.append(("dot", *pt([first_k], [level40]), 1.5))
    mk = np.arange(4, N_COMP, 8)                                     # mark phase 5, repeat 8
    marks = [(f"mark{int(m)}", *pt([m], [cols["best"][int(m)]]), MARK_HALF[cmp_code]) for m in mk]

    def check(name: str, box: tuple[float, float, float, float]) -> str:
        x0, x1, y0, y1 = box
        if x0 < 0 or x1 > AXIS_W_PT or y0 < 0 or y1 > AXIS_H_PT:
            raise SystemExit(f"{setting} {name}: box {box} leaves the axis")
        gaps = {}
        for ln, px, py, half in lines:
            gaps[ln] = box_gap(box, px, py) - half
        for mn, px, py, half in marks:                               # markers as squares
            dx = max(x0 - px[0], px[0] - x1, 0.0) - half
            dy = max(y0 - py[0], py[0] - y1, 0.0) - half
            gaps[mn] = max(dx, dy)
        worst = min(gaps, key=gaps.get)
        if gaps[worst] < CLEAR_PT:
            raise SystemExit(f"{setting} {name}: {gaps[worst]:.2f} pt from {worst} (least {CLEAR_PT})")
        near = sorted(gaps.items(), key=lambda kv: kv[1])[:3]
        return f"{name} " + ", ".join(f"{n} {v:.2f}" for n, v in near) + " pt"

    out, notes = [], []
    # the gloss: first panel, above the arrow's tail, right end at GLOSS_RIGHT_K
    if first_cmp_code is None:
        if cmp_code != 1:
            raise SystemExit(f"{setting}: the gloss's measured width is CMA-ES's, comparator code {cmp_code}")
        w, h, d = LABEL_BOX["gloss"]
        gy = level40 + GLOSS_DY_PT / ys
        x1, y = pt(GLOSS_RIGHT_K, gy)
        notes.append(check("gloss", (float(x1) - w, float(x1), float(y) - d, float(y) + h)))
        out += [1, GLOSS_RIGHT_K, gy]
        # "~20 min", right of the k = 24 rule at the bottom
        w, h, d = LABEL_BOX["min20"]
        x0, y = pt(K24, MIN20_BASE)
        notes.append(check("min20", (float(x0) + MIN20_DX_PT, float(x0) + MIN20_DX_PT + w, float(y) - d, float(y) + h)))
        out += [1, K24, MIN20_BASE]
    else:
        out += [0, 0, 0, 0, 0, 0]
    # the comparator's name, where it differs from the first panel's
    if first_cmp_code is not None and cmp_code != first_cmp_code:
        if setting not in CMPLAB_AT or cmp_code != 2:
            raise SystemExit(f"{setting}: no measured place for comparator code {cmp_code}'s name")
        w, h, d = LABEL_BOX["cmplab"]
        k0, base = CMPLAB_AT[setting]
        x0, y = pt(k0, base)
        notes.append(check("cmplab", (float(x0), float(x0) + w, float(y) - d, float(y) + h)))
        out += [1, k0, base]
    else:
        out += [0, 0, 0]
    # the clipped Schaefer curve's start, bottom left
    if sch_clip:
        if not 10 <= round(sch_k0) <= 39:
            raise SystemExit(f"{setting}: start {sch_k0:.1f} % is not the measured two digits")
        w, h, d = LABEL_BOX["start"]
        x0, y = pt(0.0, START_BASE)
        notes.append(check("start", (float(x0) + START_DX_PT, float(x0) + START_DX_PT + w, float(y) - d, float(y) + h)))
        out += [1, 0, START_BASE]
    else:
        out += [0, 0, 0]
    return out, notes


def main(variant: str = "v1") -> None:
    write_v1, write_r = variant in ("v1", "all"), variant in ("r", "all")
    t1 = json.load(open(D / "amp_table1.json", encoding="utf-8"))
    s_nm, lam = float(t1["s_nm"]), float(t1["lapse"])
    if (s_nm, lam) != (1.0, 0.1):
        raise SystemExit(f"unexpected s, lapse {s_nm}, {lam}")
    full = load_jsonl(D / "full_sessions.jsonl")
    cmp_ = load_jsonl(D / "cmp_sessions.jsonl")
    abl = load_jsonl(D / "abl_sessions.jsonl")
    rnd = load_jsonl(D / "rnd_sessions.jsonl")
    rep_abl = report_k24(D / "amp_abl_report.md")
    rep_rnd = report_k24(D / "amp_rnd_report.md")
    rep_sbs = report_side_by_side(D / "amp_rnd_report.md")
    rep_jnd, rep_users = report_users(D / "amp_posthoc_users.md")
    jnd = float(norm.ppf(0.75)) * s_nm               # 1 JND, as prompref.user.jnd.true_jnd
    if f"{jnd:.3f}" != rep_jnd:
        raise SystemExit(f"JND {jnd} vs amp_posthoc_users.md's {rep_jnd}")
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    pct_per_pt = (YMAX - YMIN) / AXIS_H_PT
    panel_rows = []
    panel_rows_r, notes_r, first_cmp = [], [], None    # variant r
    parts = {p: {} for p in PARTS}                    # part -> setting -> (est, lo, hi), points
    counts = {u: {} for u in USERS}                   # bar -> setting -> (better, worse, n) users
    boots = {u: {} for u in USERS}                    # bar -> setting -> (better, worse) % per draw
    print(f"s = {s_nm} Nm, lapse = {lam}; {N_BOOT} draws, seed {SEED}; values in %; JND {jnd:.5f} Nm")

    for idx, (letter, setting) in enumerate(PANELS):
        T = t1["settings"][setting]
        best = T["best_published"]["candidate"]
        if best != SPEC_BEST[setting] or T["best_published"]["source"] != "CMP":
            raise SystemExit(f"{setting}: best published {best} is not section 23's {SPEC_BEST[setting]}")
        # the caption's statements: the comparator is the highest at k = 24 of the published
        # optimizers run (CMP's cells and FULL's)
        bp = T["best_published"]
        pub24 = {**{"FULL " + c: v for c, v in bp["full_published_k24"].items()}, **bp["cmp_cells_k24"]}
        ranked = sorted(pub24.items(), key=lambda cv: -cv[1])
        if ranked[0][0] != best:
            raise SystemExit(f"{setting}: {best} is not the highest at k = 24 ({ranked[0][0]})")
        # CMA-ES on Schaefer's family in its planned configuration
        # in every panel (author, 2026-10-29), not the post hoc stronger of the two variants
        sch = "cmaes_sch4"
        sch24 = T["schaefer_stronger_k24"]["k24"]          # both variants, printed for the record
        series = {"psypref": (cell(full, setting, "psypref"), "psypref"),
                  "gpbald": (cell(full, setting, "gp_bald"), "gp_bald"),
                  "best": (cell(cmp_, setting, best), "best"),
                  "sch": (cell(full, setting, sch), sch)}
        ref = series["psypref"][0]
        users = [r["user"] for r in ref]
        folds = np.array([r["fold"] for r in ref])
        for name, (rs, _) in series.items():
            if [r["user"] for r in rs] != users or [r["fold"] for r in rs] != list(folds):
                raise SystemExit(f"{setting} {name}: users or folds differ from PsyPref's")
        if users != sorted(users) or len(users) != T["n_users"]:
            raise SystemExit(f"{setting}: users not sorted or n differs")

        # one fold-stratified index draw per replicate, shared by the four curves and the parts
        groups = [np.flatnonzero(folds == f) for f in sorted(set(folds))]
        draws = np.stack([np.concatenate([gi[rng.integers(len(gi), size=len(gi))] for gi in groups])
                          for _ in range(N_BOOT)])
        cols, mean40, mats = {}, {}, {}
        for name, (rs, row) in series.items():
            m = pref(rs, s_nm, lam)
            mats[name] = m
            mean = m.mean(axis=0)
            boot = m[draws].mean(axis=1)                       # N_BOOT x 40
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            r1 = T["rows"][row]
            for k in (12, 24, 40):
                if abs(mean[k - 1] - 100 * r1["k"][str(k)]["est"]) > 1e-9:
                    raise SystemExit(f"{setting} {name} k = {k}: {mean[k - 1]} vs table {r1['k'][str(k)]}")
            k0 = r1["k"]["0"]
            cols[name] = np.r_[100 * k0["est"], mean]
            cols[name + "_lo"] = np.r_[100 * k0["lo"], lo]
            cols[name + "_hi"] = np.r_[100 * k0["hi"], hi]
            mean40[name] = mean[-1]

        # the y range 40-100 %: of the plotted curves and bands only CMA-ES on Schaefer's family may
        # leave it (at the bottom, at the start); it is clipped, and its k = 0 value and the k from
        # which it stays inside go to the note
        names = ["psypref", "psypref_lo", "psypref_hi", "best", "best_lo", "best_hi", "gpbald", "sch"]
        for n in names:
            c = cols[n]
            if c.max() > YMAX or (n != "sch" and c.min() < YMIN):
                raise SystemExit(f"{setting} {n}: leaves the y range {YMIN}-{YMAX} "
                                 f"({c.min():.1f}-{c.max():.1f}); fig2.tex notes only the Schaefer curve")
        below = np.flatnonzero(cols["sch"] < YMIN)
        if len(below):
            if list(below) != list(range(len(below))):
                raise SystemExit(f"{setting} sch: below {YMIN} % not only at the start ({list(below)})")
            sch_clip, sch_k0, sch_enter = 1, float(cols["sch"][0]), int(below[-1] + 1)
        else:
            sch_clip, sch_k0, sch_enter = 0, float(cols["sch"][0]), 0

        # first k at which PsyPref's mean reaches the comparator's k = 40 level, against the table
        level40 = mean40["best"]
        hit = np.nonzero(cols["psypref"][1:] >= level40)[0]
        fk_here = int(hit[0] + 1) if len(hit) else N_COMP + 1
        fk = T["first_k_reaching_k40_level"]["best"]
        if fk_here != int(fk["est"]):
            raise SystemExit(f"{setting}: first k {fk_here} vs table {fk['est']}")
        ceiling = 100 * T["rows"]["ceiling"]["k"]["24"]["est"]
        d0_med = float(np.median([r["d0"] for r in ref]))

        # the fourth panel: users better / worse than generic assistance by 1 JND or more after
        # k = 24, the same rows as the curves (d0 is the user's, shared by every curve: checked)
        for u in USERS:
            rs = series[u][0]
            d0 = np.array([r["d0"] for r in rs])
            if not np.array_equal(d0, np.array([r["d0"] for r in ref])):
                raise SystemExit(f"{setting} {u}: d0 differs from PsyPref's")
            dk = np.array([r["distances"][K24 - 1] for r in rs])
            better, worse = int(np.sum(d0 - dk >= jnd)), int(np.sum(dk - d0 >= jnd))
            within = int(np.sum((d0 - dk < jnd) & (dk - d0 < jnd)))
            if better + worse + within != len(rs):
                raise SystemExit(f"{setting} {u}: better + worse + within != n")
            counts[u][setting] = (better, worse, len(rs))
            # the shares under the curves' index draws of this setting (the pooled interval below)
            boots[u][setting] = (100 * (d0 - dk >= jnd)[draws].mean(axis=1),
                                 100 * (dk - d0 >= jnd)[draws].mean(axis=1))
            if u in USERS_REPORT:
                printed = rep_users[setting][USERS_REPORT[u]]
                mine = (str(pct_round(better, len(rs))), str(pct_round(worse, len(rs))))
                if mine != printed:
                    raise SystemExit(f"{setting} {u}: better / worse {mine} vs amp_posthoc_users.md's {printed}")

        # the decomposition (not plotted): PsyPref with one part replaced, paired by user with
        # FULL's PsyPref
        part_rows = {"abl": cell(abl, setting, "psypref"), "rnd": cell(rnd, setting, "psypref"),
                     "gpbald": series["gpbald"][0]}
        if any(r.get("arm") != "wide" for r in part_rows["abl"]) or \
                any(r.get("duels") != "random" for r in part_rows["rnd"]):
            raise SystemExit(f"{setting}: ABL rows not all arm 'wide' or RND rows not all duels 'random'")
        for p, rs in part_rows.items():
            for key in ("user", "fold", "answer_seed", "amps", "d0"):
                if [r[key] for r in rs] != [r[key] for r in ref]:
                    raise SystemExit(f"{setting} {p}: {key} differs from FULL's PsyPref")
            diff = pref(rs, s_nm, lam)[:, K24 - 1] - mats["psypref"][:, K24 - 1]
            est = float(diff.mean())
            lo, hi = np.percentile(diff[draws].mean(axis=1), [2.5, 97.5])
            parts[p][setting] = (est, float(lo), float(hi))
            # the reports' point estimates (printed to 0.1 point)
            printed = [rep_sbs[setting][p]] + ([rep_abl[setting]] if p == "abl" else []) + \
                      ([rep_rnd[setting]] if p == "rnd" else [])
            if any(f"{est:.1f}" != x for x in printed):
                raise SystemExit(f"{setting} {p}: {est:.3f} vs the reports' {printed}")
        gp_t1 = -T["psypref_minus_points"]["gp_bald"][str(K24)]["est"]
        if abs(parts["gpbald"][setting][0] - gp_t1) > 1e-9:
            raise SystemExit(f"{setting} gpbald: {parts['gpbald'][setting][0]} vs amp_table1 {gp_t1}")

        ends = {**mean40, "generic": 50.0}
        order = ["psypref", "gpbald", "best", "sch", "generic"]
        order.sort(key=lambda n: -ends[n])
        ys = spread([ends[n] for n in order], [LABEL_H_PT[LABEL_LINES[n]] * pct_per_pt for n in order],
                    LABEL_TOP)
        ylab = dict(zip(order, ys))

        if write_v1:
            write_csv(OUT / f"fig2_curves_{letter}.csv", ["k"] + names,
                      [[k] + [float(cols[n][k]) for n in names] for k in range(N_COMP + 1)])
        panel_rows.append([idx, CMP_CODE[best], SCH_CODE[sch], d0_med, ceiling, level40, int(fk["est"]),
                           int(fk["lo"]), int(fk["hi"])]
                          + [ylab[n] for n in ("psypref", "gpbald", "best", "sch", "generic")]
                          + [mean40[n] for n in ("psypref", "gpbald", "best", "sch")]
                          + [sch_clip, sch_k0, sch_enter])
        if write_r:
            # variant r: the direct labels spread for its label heights, and the in-panel labels
            ys_r = spread([ends[n] for n in order], [LABEL_H_PT[LABEL_LINES_R[n]] * pct_per_pt for n in order],
                          LABEL_TOP)
            ylab_r = dict(zip(order, ys_r))
            cells_r, notes = layout_r(setting, cols, level40, int(fk["est"]), ceiling, CMP_CODE[best],
                                      sch_clip, sch_k0, first_cmp)
            if first_cmp is None:
                first_cmp = CMP_CODE[best]
            panel_rows_r.append(panel_rows[-1][:9] + [ylab_r[n] for n in ("psypref", "gpbald", "best", "sch", "generic")]
                                + panel_rows[-1][14:] + cells_r)
            notes_r.append(f"[{letter}] {setting}: label centres (%) "
                           + ", ".join(f"{n} {ylab_r[n]:.1f}" for n in order)
                           + "; clearance of the in-panel labels (nearest three): " + "; ".join(notes))

        print(f"\n[{letter}] {setting}: comparator {best} (code {CMP_CODE[best]}), Schaefer variant {sch}; "
              f"n = {len(users)} users, folds {', '.join(sorted(set(folds)))}")
        print("  published at k = 24 (%): " + ", ".join(f"{c} {100 * v:.2f}" for c, v in ranked[:3])
              + f"; runner-up {100 * (ranked[0][1] - ranked[1][1]):.2f} points lower")
        print("  Schaefer's family at k = 24 (%): " + ", ".join(f"{c} {100 * v:.2f}" for c, v in sch24.items()))
        for n in ("psypref", "best", "gpbald", "sch"):
            c = cols[n]
            band = (f" [{cols[n + '_lo'][0]:.1f}, {cols[n + '_hi'][0]:.1f}] / [{cols[n + '_lo'][24]:.1f}, "
                    f"{cols[n + '_hi'][24]:.1f}] / [{cols[n + '_lo'][40]:.1f}, {cols[n + '_hi'][40]:.1f}]"
                    if n + "_lo" in cols else "")
            print(f"  {n:8s} k = 0 / 12 / 24 / 40: {c[0]:.1f} / {c[12]:.1f} / {c[24]:.1f} / {c[40]:.1f}"
                  + (f"   band k = 0 / 24 / 40:{band}" if band else ""))
        print(f"  ceiling (best 8-weight profile) {ceiling:.1f}; generic assistance 50.0, median d0 "
              f"{d0_med:.3f} Nm; comparator's k = 40 level {level40:.2f}, reached by PsyPref at k = "
              f"{int(fk['est'])} [{int(fk['lo'])}, {int(fk['hi'])}] (recomputed {fk_here})")
        print(f"  y range {YMIN:.0f}-{YMAX:.0f} %: " + (
            f"sch clipped, {sch_k0:.1f} at k = 0, inside from k = {sch_enter} ({cols['sch'][sch_enter]:.1f}; "
            f"k = {sch_enter - 1}: {cols['sch'][sch_enter - 1]:.1f})" if sch_clip else
            f"every curve inside (lowest {min(cols[n].min() for n in names):.1f})"))
        print("  label centres (%): " + ", ".join(f"{n} {ylab[n]:.1f} (end {ends[n]:.1f})" for n in order))
        print(f"  users at k = {K24} vs generic assistance (better / within / worse by 1 JND or more, "
              f"% of {len(users)}; amp_posthoc_users.md's):")
        for u in USERS:
            b, w, n = counts[u][setting]
            ci = [np.percentile(x, [2.5, 97.5]) for x in boots[u][setting]]
            print(f"    {u:8s} {100 * b / n:5.1f} / {100 * (n - b - w) / n:5.1f} / {100 * w / n:5.1f}"
                  f"  (better [{ci[0][0]:.1f}, {ci[0][1]:.1f}], worse [{ci[1][0]:.1f}, {ci[1][1]:.1f}])"
                  + (f"   report {' / '.join(rep_users[setting][USERS_REPORT[u]])}" if u in USERS_REPORT
                     else "   (CMP rows, not in the report)"))
        print(f"  parts at k = {K24} (points; paired with FULL's PsyPref; reports' estimates; not plotted):")
        for p in PARTS:
            e, lo, hi = parts[p][setting]
            print(f"    {p:7s} {e:6.2f} [{lo:6.2f}, {hi:6.2f}]   side by side {rep_sbs[setting][p]}"
                  + (f", ABL table {rep_abl[setting]}" if p == "abl" else "")
                  + (f", RND table {rep_rnd[setting]}" if p == "rnd" else "")
                  + (f", amp_table1 {-T['psypref_minus_points']['gp_bald'][str(K24)]['est']:.4f}"
                     if p == "gpbald" else ""))

    head_panels = ["panel", "cmp_code", "sch_code", "d0_median_nm", "ceiling", "level40", "first_k", "first_k_lo",
                   "first_k_hi", "ylab_psypref", "ylab_gpbald", "ylab_best", "ylab_sch", "ylab_generic",
                   "end_psypref", "end_gpbald", "end_best", "end_sch", "sch_clip", "sch_k0", "sch_enter_k"]
    if write_v1:
        write_csv(OUT / "fig2_panels.csv", head_panels, panel_rows)
    if write_r:
        print("\nvariant r (fig2r.tex): in-panel labels, each at least "
              f"{CLEAR_PT} pt from every drawn line (checked)")
        for line in notes_r:
            print("  " + line)
        write_csv(OUT / R_PANELS,
                  head_panels + ["gloss", "gloss_k", "gloss_y", "min20", "min20_k", "min20_y",
                                 "cmplab", "cmplab_k", "cmplab_y", "start", "start_k", "start_y"],
                  panel_rows_r)
    settings = [s for _, s in PANELS]
    # the fourth panel, pooled over the three settings (equal n, so the pooled share is the mean
    # of the three)
    user_rows = []
    ci_dev = 0.0                                      # the largest distance of an interval end (points)
    print(f"\nfourth panel, k = {K24}, pooled over {', '.join(settings)} (% of users, 95 % interval; per "
          "setting in parentheses, printed halves to even as the reports):")
    for i, u in enumerate(USERS):
        b = sum(counts[u][s][0] for s in settings)
        w = sum(counts[u][s][1] for s in settings)
        n = sum(counts[u][s][2] for s in settings)
        if n != 3 * t1["settings"]["medium"]["n_users"]:
            raise SystemExit(f"{u}: pooled n {n}")
        sb = [100 * counts[u][s][0] / counts[u][s][2] for s in settings]
        sw = [100 * counts[u][s][1] / counts[u][s][2] for s in settings]
        if abs(100 * b / n - np.mean(sb)) > 1e-9 or abs(100 * w / n - np.mean(sw)) > 1e-9:
            raise SystemExit(f"{u}: pooled share is not the mean of the settings'")
        # the pooled share's 95 % interval: per draw the mean of the three settings' shares (equal n;
        # users resampled within their fold in each setting, the curves' draws)
        bb = np.mean([boots[u][s][0] for s in settings], axis=0)
        wb = np.mean([boots[u][s][1] for s in settings], axis=0)
        (b_lo, b_hi), (w_lo, w_hi) = np.percentile(bb, [2.5, 97.5]), np.percentile(wb, [2.5, 97.5])
        ci_dev = max(ci_dev, 100 * b / n - b_lo, b_hi - 100 * b / n, 100 * w / n - w_lo, w_hi - 100 * w / n)
        user_rows.append([i, u, n, 100 * b / n, 100 * (n - b - w) / n, 100 * w / n,
                          pct_round(b, n), pct_round(w, n)] + sb + sw
                         + [float(b_lo), float(b_hi), float(w_lo), float(w_hi)])
        print(f"  {u:8s} better {100 * b / n:5.2f} [{b_lo:.1f}, {b_hi:.1f}] ({' / '.join(f'{v:.1f}' for v in sb)}), "
              f"within {100 * (n - b - w) / n:5.2f}, worse {100 * w / n:5.2f} [{w_lo:.1f}, {w_hi:.1f}] "
              f"({' / '.join(f'{v:.1f}' for v in sw)}); printed {pct_round(b, n)} / {pct_round(w, n)} "
              f"(per setting: {' / '.join(str(pct_round(counts[u][s][0], counts[u][s][2])) for s in settings)}"
              f"; {' / '.join(str(pct_round(counts[u][s][1], counts[u][s][2])) for s in settings)})")
    ci_pm = math.ceil(ci_dev - 1e-9)
    print(f"  every pooled share's 95 % interval within +-{ci_pm} points of it (largest distance {ci_dev:.2f})")
    if write_v1:
        write_csv(OUT / "fig2_users.csv",
                  ["row", "code", "n", "better", "within", "worse", "better_txt", "worse_txt",
                   "better_m", "better_w", "better_f", "worse_m", "worse_w", "worse_f",
                   "better_lo", "better_hi", "worse_lo", "worse_hi"], user_rows)
        write_csv(OUT / "fig2_parts.csv",
                  ["part", "code", "m", "m_lo", "m_hi", "w", "w_lo", "w_hi", "f", "f_lo", "f_hi"],
                  [[i, p] + [v for s in settings for v in parts[p][s]] for i, p in enumerate(PARTS)])
        write_csv(OUT / "fig2_meta.csv",
                  ["sec_per_comparison", "n_comp", "n_users", "n_boot", "seed", "s_nm", "lapse", "ymin", "ymax",
                   "k_users", "jnd_nm", "users_ci_pm"],
                  [[SEC_PER_COMPARISON, N_COMP, t1["settings"]["medium"]["n_users"], N_BOOT, SEED, s_nm, lam,
                    YMIN, YMAX, K24, jnd, ci_pm]])
    print("\nwrote", ", ".join(p.name for p in sorted(OUT.glob("fig2*.csv"))), "to", OUT)


def first_k(share: np.ndarray, level: float) -> int:
    """First k (1-40) at which the share reaches the level; -1 if it never does within 40."""
    hit = np.flatnonzero(share >= level)
    return int(hit[0] + 1) if len(hit) else -1


def main_c() -> None:
    """Variant c: the fourth panel as the share of users at least 1 JND better than generic assistance,
    against comparisons. Users resampled jointly across the settings, within their fold."""
    t1 = json.load(open(D / "amp_table1.json", encoding="utf-8"))
    s_nm = float(t1["s_nm"])
    full, cmp_ = load_jsonl(D / "full_sessions.jsonl"), load_jsonl(D / "cmp_sessions.jsonl")
    jnd = float(norm.ppf(0.75)) * s_nm
    settings = [s for _, s in PANELS]
    sources = {"psypref": lambda s: cell(full, s, "psypref"), "gpbald": lambda s: cell(full, s, "gp_bald"),
               "best": lambda s: cell(cmp_, s, SPEC_BEST[s]), "sch": lambda s: cell(full, s, "cmaes_sch4")}
    ref = cell(full, settings[0], "psypref")
    users, folds = [r["user"] for r in ref], np.array([r["fold"] for r in ref])
    groups = [np.flatnonzero(folds == f) for f in sorted(set(folds))]
    rng = np.random.default_rng(SEED + 1)
    draws = np.stack([np.concatenate([gi[rng.integers(len(gi), size=len(gi))] for gi in groups])
                      for _ in range(N_BOOT)])
    with open(OUT / "fig2_users.csv", encoding="utf-8") as fh:
        printed = {r["code"]: r for r in csv.DictReader(fh)}
    cols, rows = {}, []
    for code in USERS:
        per, boot = {}, []
        for s in settings:
            rs = sources[code](s)
            if [r["user"] for r in rs] != users:
                raise SystemExit(f"{s} {code}: users differ from PsyPref's")
            d0 = np.array([r["d0"] for r in rs])[:, None]
            better = d0 - np.array([r["distances"][:N_COMP] for r in rs]) >= jnd
            per[s] = 100.0 * better.mean(axis=0)
            boot.append(100.0 * better[draws].mean(axis=1))
        share = np.mean([per[s] for s in settings], axis=0)
        band = np.mean(boot, axis=0)                 # draws x k: the settings' shares, same users per draw
        lo, hi = np.percentile(band, [2.5, 97.5], axis=0)
        if abs(share[K24 - 1] - float(printed[code]["better"])) > 1e-6:
            raise SystemExit(f"{code}: {share[K24 - 1]} at k = {K24} vs fig2_users.csv")
        k50_draws = np.array([first_k(b, 50.0) for b in band])
        rows.append([code, first_k(share, 50.0), first_k(share, 75.0), float(share[K24 - 1]),
                     printed[code]["better_txt"], float(np.percentile(k50_draws, 2.5)),
                     float(np.percentile(k50_draws, 97.5)), float(np.mean(k50_draws < 0))]
                    + [first_k(per[s], 50.0) for s in settings])
        cols[code], cols[code + "_lo"], cols[code + "_hi"] = share, lo, hi
    # the column of the values at k = 24, right of the panel: spread so that they stay readable
    ys = spread([r[3] for r in rows], [C_LABEL_PCT] * len(rows), 100.0)
    write_csv(OUT / "fig2c_meta.csv",
              ["code", "k50", "k75", "at24", "at24_txt", "k50_lo", "k50_hi", "k50_none_share", "k50_m", "k50_w", "k50_f",
               "ylab"], [r + [y] for r, y in zip(rows, ys)])
    names = ["psypref", "psypref_lo", "psypref_hi", "gpbald", "best", "best_lo", "best_hi", "sch"]
    write_csv(OUT / "fig2c_users.csv", ["k"] + names,
              [[k] + [float(cols[n][k - 1]) for n in names] for k in range(1, N_COMP + 1)])
    # for the Results: PsyPref against each setting's comparator, paired by user, and how far both
    # generic assistance and PsyPref stay from the preferred profile, at k = 24, in JND
    paired = []
    for s in settings:
        p, c = sources["psypref"](s), sources["best"](s)
        dp, dc = (np.array([r["distances"][K24 - 1] for r in x]) for x in (p, c))
        paired.append([s.replace(" ", ""), 100.0 * np.mean(dp < dc), 100.0 * np.mean(dc - dp >= jnd), 100.0 * np.mean(dp - dc >= jnd),
                       float(np.median([r["d0"] for r in p])) / jnd, float(np.median(dp)) / jnd])
    paired.append(["mean"] + [float(np.mean([r[i] for r in paired])) for i in (1, 2, 3)] + [float("nan")] * 2)
    write_csv(OUT / "fig2c_paired.csv", ["setting", "closer", "closer_1jnd", "cmp_closer_1jnd", "d0_median_jnd",
                                         "d24_median_jnd"], paired)
    for r in rows:
        print(f"{r[0]:8s} half at k = {r[1]} [{r[5]:.0f}, {r[6]:.0f}], three quarters at {r[2]}, at {K24}: {r[3]:.1f}; "
              f"per setting (3-12 / 3-35 / fixed 12): half at {r[8]} / {r[9]} / {r[10]} (-1: not within {N_COMP})")
    print(f"wrote fig2c_users.csv, fig2c_meta.csv to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=("v1", "r", "all", "c", "d"), default="v1",
                    help="v1: fig2_*.csv (fig2.tex); r: fig2r_panels.csv (fig2r.tex); all: both; "
                         "c: fig2c_*.csv (fig2c.tex, reads fig2_users.csv); "
                         "d: fig2d_panels.csv (fig2d.tex, which also reads fig2c_*.csv and fig2_users.csv)")
    variant = ap.parse_args().variant
    if variant == "d":           # variant r's layout and checks for curve panels 85 pt wide; the
        AXIS_W_PT, R_PANELS = 85.0, "fig2d_panels.csv"          # gloss reads "CMA-ES at 40" there
        LABEL_BOX["gloss"] = (47.5, 4.77, 0.10)
        CMPLAB_AT["wide"] = (23.0, 58.0)
        LABEL_BOX["start"] = (22.7, 4.77, 0.10)          # "Start 23": the axis gives the unit
        START_DX_PT, START_BASE = 0.4, 45.0
        main("r")
    else:
        main_c() if variant == "c" else main(variant)
