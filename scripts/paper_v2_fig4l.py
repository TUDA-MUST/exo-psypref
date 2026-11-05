"""Fig. 4 of the ICORR paper (v2), variant 4L: PsyPref's lead under each broken assumption.

Usage:
  python scripts/paper_v2_fig4l.py [--out <repository>/out] [--dest figures/data]

Message (the caption's first sentence): after 24 comparisons, PsyPref stays ahead of the strongest other
optimizer in most condition x setting cells; it ties or trails with no single preferred profile, and at
fixed 12 Nm on Schaefer-family shapes and for the pessimistic user.

Post hoc, nothing rerun. Rows as 4S (scripts/paper_v2_fig4s.py), with lapse rate 0.2 / 0.3 and noise level
falling / rising merged, the main condition as the first row, and a new group for users who differ from
the prior: other datasets (ROB R6), peaks unlike the prior (PRI PAM, 3-35 Nm only, since fixed 12 Nm has
no peak law), a later flexion peak (PRI PAD S1, S2), and 4S's shapes row. PAD's S3 is C1's shapes, so it
counts once.

Each cell: PsyPref (with tie handling on the dagger rows) minus the strongest optimizer of that cell,
paired by user, chosen over generic assistance at k = 24. The optimizers are those run under the
deviations: Arens' GP, LineCoSpar and CMA-ES on the 8 weights, and CMA-ES on Schaefer's family (PAD: Arens'
GP and CMA-ES on Schaefer's family; TIE: both CMA-ES, checked to be ROB2's strongest there). Fig. 2's
stronger variants (CMP) and GP-BALD were not run under most deviations; the caption says so. Rating: as each
report, each user's own noise level and lapse rate at the session's end; the fixed yardstick (s = 1 Nm,
lapse 0.1) is computed beside it and the sign changes are counted. A merged row shows, per setting, its
smallest lead. Colour: icBlue with the value where the interval reaches 0 (lower end at or below 0), else
icBlueLight.

Writes to <dest>: fig4l_rows.csv, fig4l_meta.csv (the figure's), fig4l_provenance.csv (per row and
setting: source, condition, comparator, n, merged from, both ratings), fig4l_text.tex (the caption's and
the Results draft's numbers as macros).
Self-checks (stop on failure): amp_yardstick.json's own checks; coverage of 4S's conditions; PRI's
recomputed estimates equal amp_pri_report.md's; the main condition's levels equal amp_table1.json's; TIE's
comparators are ROB2's strongest; no merged row hides a cell whose interval reaches 0 behind a light marker;
every interval and value label inside the axis, every value label clear of everything drawn.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import icorr_amp_pri as PRI
import icorr_amp_range as RANGE
import icorr_amp_rob2 as RB
import paper_v2_fig4 as F4
import paper_v2_fig4ab as FAB
from paper_v2_fig4s import CLEAR, FIG_H, NUM_WORD, f1, sgn, text_w, tri
from prompref.experiment import amp_free as AF

K, SETTINGS, TAG, N_BOOT = F4.K, F4.SETTINGS, F4.TAG, F4.N_BOOT
COMPARATORS = ("g", "d_z", "cmaes", "cmaes_sch4")
NAMES = {"g": "Arens' GP", "d_z": "LineCoSpar", "cmaes": "CMA-ES", "cmaes_sch4": "CMA-ES on Schaefer's family"}
BOOT_SEED = 20261102
PAM_ROW = tuple(PRI.PAM_CONDS)
PAD_ROW = ("S1 flexion +6", "S2 flexion +12")


def C(source, *conds, opt="psypref"):
    return [(source, c, opt) for c in conds]


GROUPS = [
    ("BOLD0", [("MAIN", C("FULL", "main"))]),
    ("G1", [("R1", C("ROB", "R1 s=0.5")), ("R2", C("ROB", "R2 s=2.0")),
            ("AL", C("ROB2", "A2 lapse 0.2") + C("ROB", "R3 lapse=0.3")),
            ("A1", C("ROB2", "A1 individual s")),
            ("A6", C("ROB2", "A6a logistic link", "A6b Student-t link", "A6c Cauchy link")),
            ("A3", C("ROB2", "A3 device noise")), ("A4", C("ROB2", "A4 order bias")),
            ("A5T", C("TIE", "A5 no preference", opt="psypref_t"))]),
    ("G2", [("A7", C("ROB2", "A7 fatigue")), ("A8", C("ROB2", "A8a s 1.5 to 0.75", "A8b s 0.75 to 1.5")),
            ("B4", C("ROB2", "B4a drift 1 JND", "B4b drift 2 JND")), ("B3", C("ROB2", "B3 familiarity"))]),
    ("G3", [("B1", C("ROB2", "B1 Weber")), ("B2", C("ROB2", "B2 asymmetry")),
            ("B5", C("ROB2", "B5a features static", "B5b features walking")),
            ("B67", C("ROB2", "B6a phase individual", "B6b phase hip speed", "B7 anisotropy")),
            ("B8", C("ROB2", "B8 random utility")), ("R4", C("ROB", "R4 negative power"))]),
    ("G4", [("R6", C("ROB", "R6 other datasets")), ("PM", C("PAM", *PAM_ROW)), ("PS", C("PAD", *PAD_ROW, opt="psypref_static")),
            ("SH", C("ROB", "R7 surrogates") + C("ROB2", "C1 Schaefer shapes", "C2 heavy tails", "C3 family mixture"))]),
    ("BOLD", [("C4T", C("TIE", "C4 pessimistic", opt="psypref_t"))]),
]
# y layout in row units, downwards, and x layout in pt as 4S
TOP_LABELS, HEADER_TO_ROW, GROUP_GAP, BOTTOM_PAD, TOP_PAD = -1.15, 1.05, 1.2, 0.72, 0.70
LAB_W, AX_R, PITCH, BOTTOM, WIDTH = 111.0, 246.0, 7.2, 17.4, 252.0
XMIN, XMAX = -11.0, 30.0
OFF, MARK_R, MARK_S, VAL_GAPX = 0.25, 1.45, 1.25, 1.5
LINE_HALF, BAR_HALF = 0.25, 0.35


def pct(t):
    return [100 * v for v in t]


def paired(a: dict, b: dict, rng) -> list:
    """icorr_amp_pri._diff on the same pairs, with each user's difference computed once."""
    items = [dict(fold=a[u]["fold"], v=PRI._p(a[u]) - PRI._p(b[u])) for u in sorted(set(a) & set(b))]
    return list(RANGE._boot(items, lambda x: 100 * float(np.mean([i["v"] for i in x])), rng))


def pri_cells(amp: Path) -> dict:
    """PAM (3-35 Nm) and PAD (S1, S2) cells: PsyPref minus each optimizer, paired by user, as
    icorr_amp_pri's report scores them (s = 1 Nm, lapse 0.1)."""
    rows = {}
    for x in (amp / "pri_sessions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r.get("error") is None:
            rows.setdefault((r["test"], r["condition"], r["setting"], r["candidate"]), {})[r["user"]] = r
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for test, conds, settings, own, comps in (("PAM", PAM_ROW, ("wide",), "psypref", COMPARATORS),
                                              ("PAD", PAD_ROW, SETTINGS, "psypref_static", ("g", "cmaes_sch4"))):
        for cond in conds:
            for st in settings:
                for b in comps:
                    d = paired(rows[test, cond, st, own], rows[test, cond, st, b], rng)
                    n = len(rows[test, cond, st, own])
                    out[test, cond, st, b] = dict(a=own, b=b, n=n, reg=d, fix=d)
    return out


def full_cells(amp: Path) -> tuple[dict, dict]:
    """The main condition on the same optimizers, and PsyPref's and their levels at k = 24."""
    by = {}
    for x in (amp / "full_sessions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r.get("error") is None:
            by.setdefault((r["setting"], r["candidate"]), {})[r["user"]] = r
    rng = np.random.default_rng(BOOT_SEED + 1)
    out, levels = {}, {}
    for st in SETTINGS:
        levels[st, "psypref"] = float(np.mean([PRI._p(r) for r in by[st, "psypref"].values()]))
        for b in COMPARATORS:
            d = paired(by[st, "psypref"], by[st, b], rng)
            out["FULL", "main", st, b] = dict(a="psypref", b=b, n=len(by[st, "psypref"]), reg=d, fix=d)
            levels[st, b] = float(np.mean([PRI._p(r) for r in by[st, b].values()]))
    return out, levels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=F4.MAIN / "out")
    ap.add_argument("--dest", type=Path, default=F4.DEST)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    amp = args.out / "icorr" / "amp_free"
    js = json.loads((amp / "amp_yardstick.json").read_text(encoding="utf-8"))
    sc = js["self_check"]
    if sc["points_matched"] != sc["points_total"] or not all(e["ok"] for e in sc["exact"]):
        raise SystemExit("amp_yardstick.json's own self-check did not pass")

    # ------------------------------------------------------------------ the cells
    cells, loss, plain = {}, {}, {}
    for e in js["differences"]:
        if e["k"] != K:
            continue
        if e["kind"] == "loss vs FULL":
            loss[e["source"], e["condition"], e["setting"], e["a"]] = e
        elif e["kind"] == "vs comparator":
            plain[e["source"], e["condition"], e["setting"], e["a"], e["b"]] = e
            if e["b"] in COMPARATORS:
                cells[e["source"], e["condition"], e["setting"], e["a"], e["b"]] = dict(
                    a=e["a"], b=e["b"], n=e["n"], reg=pct(e["registered"]), fix=pct(e["fixed"]),
                    dup=e.get("duplicate_of"))
    pri = pri_cells(amp)
    full, levels = full_cells(amp)
    for (src, cond, st, b), c in {**pri, **full}.items():
        cells[src, cond, st, c["a"], b] = c

    def strongest(src, cond, st, opt):
        got = [c for (s, co, se, a, _), c in cells.items() if (s, co, se, a) == (src, cond, st, opt)]
        return min(got, key=lambda c: c["reg"][0]) if got else None

    # ------------------------------------------------------------------ rows
    rows, prov, all_cells, y = [], [], [], 0.0
    first = True
    for gkey, members in GROUPS:
        if gkey.startswith("G"):
            y += GROUP_GAP
            rows.append(dict(key=gkey, header=1, bold=0, y=y))
            y += HEADER_TO_ROW
        elif not first:
            y += GROUP_GAP
        first = False
        for i, (rk, conds) in enumerate(members):
            if i:
                y += 1.0
            row = dict(key=rk, header=0, bold=int(gkey.startswith("BOLD")), y=y, n_merged=len(conds),
                       tie=int(conds[0][2] == "psypref_t"))
            for st in SETTINGS:
                cand = []
                for src, cond, opt in conds:
                    c = strongest(src, cond, st, opt)
                    if c is not None:
                        cand.append((src, cond, c))
                        # counted: the deviations only, and B5b at fixed 12 Nm is B5a's PsyPref sessions (as 4S)
                        if rk != "MAIN" and not (cond == "B5b features walking" and st == "fixed 12"):
                            all_cells.append(dict(row=rk, src=src, cond=cond, st=st, **c))
                if not cand:
                    row[TAG[st]] = None
                    continue
                src, cond, c = min(cand, key=lambda t: t[2]["reg"][0])
                row[TAG[st]] = dict(fx=c["reg"][0], lo=c["reg"][1], hi=c["reg"][2], n=c["n"], blue=c["reg"][1] <= 0.0)
                prov.append([rk, st, src, cond, NAMES[c["b"]], c["n"], len(conds), *c["reg"], *c["fix"]])
            rows.append(row)
    last_y = y

    # ------------------------------------------------------------------ self-checks
    expected = {(s, c) for s, c in FAB.EXPECTED} | {("PAM", c) for c in PAM_ROW} | {("PAD", c) for c in PAD_ROW} \
        | {("ROB", "R6 other datasets"), ("FULL", "main")}
    cov = {(s, c) for _, m in GROUPS for _, conds in m for s, c, _ in conds}
    if cov != expected:
        raise SystemExit(f"coverage differs: {cov ^ expected}")
    # PRI's estimates, as its report prints them
    report = (amp / "amp_pri_report.md").read_text(encoding="utf-8")
    n_pri = 0
    for cond in PAM_ROW:
        m = re.search(rf"\| {re.escape(cond)} \| wide \|.*?\| ([\d.\-]+) \[[^]]*\] \| ([\d.\-]+) \[[^]]*\] \| [\d.\-]+ \[", report)
        pub = min((pri["PAM", cond, "wide", b]["reg"][0] for b in ("g", "d_z", "cmaes")))
        # the report's "strongest other optimizer" is the highest level, so the smallest paired difference
        if f"{pri['PAM', cond, 'wide', 'cmaes_sch4']['reg'][0]:.1f}" != m.group(1) or f"{pub:.1f}" != m.group(2):
            raise SystemExit(f"PAM {cond}: {pub:.1f} vs the report's {m.group(2)}")
        n_pri += 2
    for cond in PAD_ROW:
        for st in SETTINGS:
            m = re.search(rf"\| {re.escape(cond)} \| {st} \|.*\| ([\d.\-]+) \[[^]]*\] \|\n", report)
            if f"{pri['PAD', cond, st, 'cmaes_sch4']['reg'][0]:.1f}" != m.group(1):
                raise SystemExit(f"PAD {cond} {st}: differs from the report's {m.group(1)}")
            n_pri += 1
    # the main condition's levels against amp_table1.json
    t1 = json.loads((amp / "amp_table1.json").read_text(encoding="utf-8"))["settings"]
    for st in SETTINGS:
        for b, v in t1[st]["best_published"]["full_published_k24"].items():
            if abs(levels[st, b] - v) > 1e-9:
                raise SystemExit(f"main condition {st} {b}: {levels[st, b]} vs amp_table1.json's {v}")
    # TIE runs both CMA-ES only: they are ROB2's strongest in those cells
    for cond in ("A5 no preference", "C4 pessimistic"):
        for st in SETTINGS:
            best = min(COMPARATORS, key=lambda b: plain["ROB2", cond, st, "psypref", b]["registered"][0])
            if best not in ("cmaes", "cmaes_sch4"):
                raise SystemExit(f"{cond} {st}: ROB2's strongest is {best}, which TIE did not run")
    # no merged row hides a cell whose interval reaches 0 behind a light marker
    by_row = {r["key"]: r for r in rows}
    exceptions = [c for c in all_cells if c["reg"][1] <= 0.0]
    for c in exceptions:
        if not by_row[c["row"]][TAG[c["st"]]]["blue"]:
            raise SystemExit(f"{c['row']} {c['st']} hides {c['cond']} behind a light marker")
    # the two ratings: where the sign of the plotted lead, or of the lower end, changes
    sign_changes = [c for c in all_cells if (c["reg"][0] > 0) != (c["fix"][0] > 0) and abs(c["reg"][0]) > 0]
    low_changes = [c for c in all_cells if (c["reg"][1] > 0) != (c["fix"][1] > 0)]

    # layout and the value labels
    ymin, ymax, linetop = TOP_LABELS - TOP_PAD, last_y + BOTTOM_PAD, TOP_LABELS + 0.45
    scale = (AX_R - LAB_W) / (XMAX - XMIN)

    def xp(v):
        return LAB_W + (v - XMIN) * scale

    def yp(v):
        return (v - ymin) * PITCH

    drawn = []
    for r in rows:
        if r["header"]:
            continue
        for tag, off, rad in (("w", -OFF, MARK_R), ("f", OFF, MARK_S)):
            c = r[tag]
            if c is None:
                continue
            if not (XMIN < c["lo"] and c["hi"] < XMAX):
                raise SystemExit(f"{r['key']} {tag}: [{c['lo']:.1f}, {c['hi']:.1f}] outside the axis")
            yc = yp(r["y"] + off)
            drawn.append((f"{r['key']} {tag} bar", xp(c["lo"]), yc - BAR_HALF, xp(c["hi"]), yc + BAR_HALF))
            drawn.append((f"{r['key']} {tag} mark", xp(c["fx"]) - rad, yc - rad, xp(c["fx"]) + rad, yc + rad))
    drawn.append(("zero line", xp(0) - LINE_HALF, yp(linetop), xp(0) + LINE_HALF, yp(ymax)))
    labels = []
    for r in rows:
        if r["header"]:
            continue
        for tag, off, rad in (("w", -OFF, MARK_R), ("f", OFF, MARK_S)):
            c = r[tag]
            if c is None or not c["blue"]:
                continue
            txt = f1(abs(c["fx"]))
            neg = c["fx"] < 0 and txt != "0.0"
            w = text_w(("-" if neg else "") + txt)
            yc, x1 = yp(r["y"] + off), min(xp(c["lo"]), xp(c["fx"]) - rad) - VAL_GAPX
            c["vx"] = x1
            if x1 - w < LAB_W + CLEAR:
                raise SystemExit(f"value label {r['key']} {tag} leaves the axis")
            # as 4S: the least vertical shift that clears, a circle's label up, a square's down
            for step in range(16):
                shift = (-0.1 if tag == "w" else 0.1) * step
                box = (x1 - w, yc + shift - FIG_H / 2, x1, yc + shift + FIG_H / 2)
                gaps = [(math.hypot(max(x0 - box[2], box[0] - x9, 0.0), max(y0 - box[3], box[1] - y9, 0.0)), name)
                        for name, x0, y0, x9, y9 in drawn if not name.startswith(f"{r['key']} {tag} bar")]
                least, near = min(gaps)
                if least >= CLEAR:
                    break
            else:
                raise SystemExit(f"value label {r['key']} {tag}: {least:.2f} pt from {near}")
            c.update(lab=txt, neg=int(neg), vdy=-shift)
            labels.append((r["key"], tag, txt, neg))

    # ------------------------------------------------------------------ numbers for the text
    cell = lambda rk, st: by_row[rk][TAG[st]]                                    # noqa: E731
    tr = lambda c: tri((c["fx"], c["lo"], c["hi"]))                              # noqa: E731
    clear = [c for c in all_cells if c["reg"][1] > 0.0]
    drops = {k_: 100 * e["registered"][0] for k_, e in loss.items() if not e.get("duplicate_of")}
    single = min(v for (s, c, st, a), v in drops.items() if c.startswith(("B5a", "B5b")))
    pess = min(v for (s, c, st, a), v in drops.items() if s == "TIE" and c.startswith("C4") and a == "psypref_t")
    others = min(v for (s, c, st, a), v in drops.items() if not c.startswith(("B5a", "B5b", "C4")))
    if others <= single:
        raise SystemExit("another condition costs more than judging single features")
    bald = [100 * plain["ROB", c, st, "psypref", "gp_bald"]["registered"][0] for c in ("R2 s=2.0", "R6 other datasets")
            for st in SETTINGS]
    variants = {st: t1[st]["psypref_minus_points"]["best"][str(K)]["est"] for st in SETTINGS}
    if not (abs(next(e for e in js["levels"] if e["source"] == "ROB2" and e["condition"] == "B8 random utility"
                     and e["setting"] == "wide" and e["optimizer"] == "g" and e["k"] == K)["registered"][0] - 0.95) < 1e-9):
        raise SystemExit("Arens' GP is not at the cap with no single preferred profile in 3-35 Nm")
    macro = {
        "flK": K, "flNclear": len(clear), "flNall": len(all_cells), "flNusers": cell("MAIN", "wide")["n"],
        "flNother": cell("R6", "wide")["n"], "flNtie": cell("C4T", "wide")["n"],
        "flMainW": tr(cell("MAIN", "wide")), "flMainF": tr(cell("MAIN", "fixed 12")),
        "flVarW": f1(variants["wide"]), "flVarF": f1(variants["fixed 12"]),
        "flSingleW": tr(cell("B5", "wide")), "flSingleF": tr(cell("B5", "fixed 12")),
        "flOtherW": tr(cell("R6", "wide")), "flOtherF": tr(cell("R6", "fixed 12")),
        "flPeaks": sgn(cell("PM", "wide")["fx"]), "flLaterW": sgn(cell("PS", "wide")["fx"]),
        "flLaterF": sgn(cell("PS", "fixed 12")["fx"]), "flNoneF": tr(cell("B8", "fixed 12")),
        "flShapesF": tr(cell("SH", "fixed 12")), "flShapesW": tr(cell("SH", "wide")),
        "flPessW": tr(cell("C4T", "wide")), "flPessF": tr(cell("C4T", "fixed 12")),
        "flDropSingle": f1(-single), "flDropPess": f1(-pess),
        "flBaldLo": f1(min(bald)), "flBaldHi": f1(max(bald)),
    }
    tex = ["% fig4l_text.tex: written by scripts/paper_v2_fig4l.py (repository); do not edit by hand."]
    tex += [f"\\def\\{k_}{{{v}}}" for k_, v in macro.items()]
    (args.dest / "fig4l_text.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ write
    out_rows = []
    for r in rows:
        o = [r["key"], r["header"], r["bold"], r["y"]]
        for tag in ("w", "f"):
            c = r.get(tag)
            if c is None:
                o += [math.nan] * 6 + [0, "-", 0, 0, 0, 0, 0]
                continue
            t3, nan3 = [c["fx"], c["lo"], c["hi"]], [math.nan] * 3
            o += (nan3 if c["blue"] else t3) + (t3 if c["blue"] else nan3)
            o += [int(c["blue"]), c.get("lab", "-"), c.get("neg", 0), 3 if c["blue"] else 0, c.get("vx", 0),
                  c.get("vdy", 0), c["n"]]
        o += [r.get("n_merged", 0), r.get("tie", 0)]
        out_rows.append(o)
    head = ["key", "is_header", "bold", "y"]
    for t in ("w", "f"):
        head += [f"in_{t}", f"in_lo_{t}", f"in_hi_{t}", f"out_{t}", f"out_lo_{t}", f"out_hi_{t}",
                 f"blue_{t}", f"lab_{t}", f"neg_{t}", f"pos_{t}", f"vx_{t}", f"vdy_{t}", f"n_{t}"]
    head += ["n_merged", "tie"]
    F4.write_csv(args.dest / "fig4l_rows.csv", head, out_rows)
    fab_lab = F4.label_values(js)
    meta = dict(xmin=XMIN, xmax=XMAX, ymin=ymin, ymax=ymax, ylabels=TOP_LABELS, linetop=linetop, lab_w=LAB_W, ax_r=AX_R,
                pitch=PITCH, bottom=BOTTOM, width=WIDTH, scale=scale, off=OFF, mark_r=MARK_R, mark_s=MARK_S,
                val_gapx=VAL_GAPX, height=BOTTOM + (ymax - ymin) * PITCH, n_rows=len(out_rows), s_r1=0.5, s_r2=2.0,
                lapse_a2=float(fab_lab["lapse_a2"]), lapse_r3=float(fab_lab["lapse_r3"]), n_pam=len(PAM_ROW),
                ndev_word=NUM_WORD[len(RB.CONDITIONS["C4 pessimistic"])])
    F4.write_csv(args.dest / "fig4l_meta.csv", list(meta), [list(meta.values())])
    F4.write_csv(args.dest / "fig4l_provenance.csv",
                 ["key", "setting", "source", "condition", "comparator", "n", "merged_from", "lead", "lo", "hi",
                  "lead_fixed", "lo_fixed", "hi_fixed"],
                 [[str(x).replace(" ", "_") if isinstance(x, str) else x for x in p] for p in prov])

    # ------------------------------------------------------------------ print
    print(f"{sum(not r['header'] for r in rows)} rows, figure {WIDTH:g} x {meta['height']:.1f} pt; scale {scale:.3f} pt per point")
    print(f"self-checks: {n_pri} PRI estimates equal the report; main levels equal amp_table1.json; coverage {len(cov)} conditions")
    print("row    y     3-35 Nm                 | fixed 12 Nm")
    for r in rows:
        if r["header"]:
            print(f"{r['key']} y = {r['y']:g}")
            continue
        s = []
        for tag in ("w", "f"):
            c = r[tag]
            s.append("-" if c is None else f"{c['fx']:+6.1f} [{c['lo']:+6.1f}, {c['hi']:+6.1f}]{' BLUE' if c['blue'] else ''}")
        print(f"  {r['key']:<5}{r['y']:6.2f} {s[0]:<28}| {s[1]}")
    print(f"cells with the interval above 0: {len(clear)} of {len(all_cells)}; the others: " + "; ".join(
        f"{c['cond']} {c['st']} vs {NAMES[c['b']]} {c['reg'][0]:+.1f} [{c['reg'][1]:+.1f}, {c['reg'][2]:+.1f}]" for c in exceptions))
    print("rating with the fixed yardstick: lead changes sign in " + (", ".join(
        f"{c['cond']} {c['st']} {c['reg'][0]:+.1f} -> {c['fix'][0]:+.1f}" for c in sign_changes) or "no cell")
        + "; lower end changes sign in " + (", ".join(
        f"{c['cond']} {c['st']} {c['reg'][1]:+.1f} -> {c['fix'][1]:+.1f}" for c in low_changes) or "no cell"))
    print(f"largest gap between the ratings: {max(abs(c['reg'][0] - c['fix'][0]) for c in all_cells):.1f} points")
    print(f"PsyPref's own level: single features {single:+.1f}, pessimistic {pess:+.1f}, others at worst {others:+.1f}")
    print(f"main condition against Fig. 2's variants: {variants}; GP-BALD: {[f'{v:+.1f}' for v in bald]}")
    print(f"value labels: {labels}")
    print("macros: " + "; ".join(f"{k_} = {v}" for k_, v in macro.items()))
    print(f"AF lapse {AF.LAPSE}, s {AF.S}, boots {RANGE.N_BOOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
