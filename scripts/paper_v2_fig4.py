"""Fig. 4 of the ICORR paper (v2): what noisy answers and broken assumptions cost PsyPref (forest plot).

Usage:
  python scripts/paper_v2_fig4.py
      [--out <repository>/out] [--dest figures/data] [--no-verify]

Post hoc. One row per simulated deviation: PsyPref under the condition minus PsyPref in FULL on the
same users and answer seeds, preference over generic assistance at k = 24, in points, for 3-35 Nm
("wide") and fixed 12 Nm, on two yardsticks:
  fixed       s = 1.0 Nm, lapse = 0.1 on the user's own utility (post hoc; the filled markers),
  planned  as each planned report scored it, with the condition's s and lapse at the end of
              the 40-comparison session (the hollow markers, drawn only where the per-user values
              differ from the fixed ones; their 95 % intervals are written but not drawn).
Every number is read from out/icorr/amp_free/amp_yardstick.json, written by
scripts/icorr_posthoc_yardstick.py from the stored session rows of FULL, ROB, ROB2 and TIE: 95 %
intervals from 2,000 fold-stratified draws (users resampled within their fold), rows sorted by
user, seed [SEED_FULL, 41, crc32(label)]. Unless --no-verify, this script recomputes every
triple it writes from the stored session rows with that script's own functions and stops on any
difference (about a minute, one process).

Rows: R5 (K = 12, Fig. 3) and R6 (other datasets, not paired with FULL) are left out. "Other
answer-curve shapes" is, per setting, the worst (lowest fixed-yardstick estimate) of A6a-c; the
provenance file names it. B5a and B5b at fixed 12 Nm are bit-identical sessions (the JSON's
duplicate_of); B5b's fixed-12 cell is written as nan and plotted once, in the B5a row.
PsyPref-T rows (A5T, C4T) are TIE's arm "tie", users 8-39 of each subject (n = 160).
Column prb2 = 1 marks the rows covered by ROB2's planned prediction PRB2 (PsyPref under ROB2's
answer conditions A1-A8b; the same rule as icorr_amp_rob2.py's report: condition name starts "A").

Writes to <dest>:
  fig4_rows.csv        the plotted rows (key, header flag, y in row units, values in points, prb2)
  fig4_meta.csv        axis limits, the y of the line labels (layout, checked against the data) and
                       the condition values printed in row labels (lapse rates, drifts in Nm), read
                       from the planned condition definitions (icorr_amp_rob2 / icorr_amp_robust)
  fig4_provenance.csv  per row and setting: source, condition, optimizer, n, yardsticks equal?, duplicate
  fig4_notes.csv       numbers for the caption (PRB2 cells, C4 against CMA-ES on Schaefer's family, ...)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # the repository
MAIN = ROOT                                 # out/ lives here
DEST = ROOT / "figures" / "data"

K = 24
SETTINGS = ("wide", "fixed 12")                        # 3-35 Nm, fixed 12 Nm (the figure's order)
TAG = {"wide": "w", "fixed 12": "f"}
XMIN, XMAX = -26.0, 10.0                               # axis limits (points); checked against the data
GUIDE = -5.0                                           # the post hoc yardstick's 5-point guide
PRB2_LIMIT = -5.0                                      # PRB2: no answer condition (A) costs > 5 points
N_BOOT = 2000

# (group key, [(row key, source, [conditions; > 1 = the worst of them], optimizer)])
GROUPS = [
    ("G1", [("R1", "ROB", ["R1 s=0.5"], "psypref"),
            ("R2", "ROB", ["R2 s=2.0"], "psypref"),
            ("A2", "ROB2", ["A2 lapse 0.2"], "psypref"),
            ("R3", "ROB", ["R3 lapse=0.3"], "psypref"),
            ("A1", "ROB2", ["A1 individual s"], "psypref"),
            ("A6", "ROB2", ["A6a logistic link", "A6b Student-t link", "A6c Cauchy link"], "psypref"),
            ("A3", "ROB2", ["A3 device noise"], "psypref"),
            ("A4", "ROB2", ["A4 order bias"], "psypref"),
            ("A5", "ROB2", ["A5 no preference"], "psypref"),
            ("A5T", "TIE", ["A5 no preference"], "psypref_t")]),
    ("G2", [("A7", "ROB2", ["A7 fatigue"], "psypref"),
            ("A8a", "ROB2", ["A8a s 1.5 to 0.75"], "psypref"),
            ("A8b", "ROB2", ["A8b s 0.75 to 1.5"], "psypref"),
            ("B4a", "ROB2", ["B4a drift 1 JND"], "psypref"),
            ("B4b", "ROB2", ["B4b drift 2 JND"], "psypref"),
            ("B3", "ROB2", ["B3 familiarity"], "psypref")]),
    ("G3", [("B1", "ROB2", ["B1 Weber"], "psypref"),
            ("B2", "ROB2", ["B2 asymmetry"], "psypref"),
            ("B5a", "ROB2", ["B5a features static"], "psypref"),
            ("B5b", "ROB2", ["B5b features walking"], "psypref"),
            ("B6a", "ROB2", ["B6a phase individual"], "psypref"),
            ("B6b", "ROB2", ["B6b phase hip speed"], "psypref"),
            ("B7", "ROB2", ["B7 anisotropy"], "psypref"),
            ("B8", "ROB2", ["B8 random utility"], "psypref"),
            ("R4", "ROB", ["R4 negative power"], "psypref")]),
    ("G4", [("R7", "ROB", ["R7 surrogates"], "psypref"),
            ("C1", "ROB2", ["C1 Schaefer shapes"], "psypref"),
            ("C2", "ROB2", ["C2 heavy tails"], "psypref"),
            ("C3", "ROB2", ["C3 family mixture"], "psypref")]),
    ("G5", [("C4", "ROB2", ["C4 pessimistic"], "psypref"),
            ("C4T", "TIE", ["C4 pessimistic"], "psypref_t")]),
]
# y layout in row units (1 = one row's pitch), downwards: header, rows, a gap before the next header
Y_TOP_LABELS, Y_HEADER_TO_ROW, Y_GROUP_GAP, Y_BOTTOM_PAD, Y_TOP_PAD = -1.15, 1.05, 1.45, 0.75, 0.68


def g(v: float) -> str:
    """Ten significant digits, nan for a missing value (the tikz/README.md convention)."""
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.10g}"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    lines = [",".join(header)]
    for r in rows:
        cells = [g(x) if isinstance(x, float) or x is None else str(x) for x in r]
        assert all("," not in c for c in cells), cells
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pts(t) -> list[float]:
    return [100.0 * float(v) for v in t]


def label_values(js: dict) -> dict:
    """The condition values the row labels print, from the planned condition definitions.

    Also checks the words the labels use for them: "noise level halved / doubled" (R1, R2 against
    the main condition's s), "discrimination improving / worsening" (A8a, A8b: s falls / rises).
    """
    import icorr_amp_rob2 as RB     # noqa: PLC0415
    import icorr_amp_robust as ROB  # noqa: PLC0415

    s0 = js["fixed_yardstick"]["s"]
    assert ROB.CONDITIONS["R1 s=0.5"]["s"] == 0.5 * s0 and ROB.CONDITIONS["R2 s=2.0"]["s"] == 2.0 * s0
    a8a, a8b = RB.CONDITIONS["A8a s 1.5 to 0.75"]["s_path"], RB.CONDITIONS["A8b s 0.75 to 1.5"]["s_path"]
    assert a8a[1] < a8a[0] and a8b[1] > a8b[0]
    assert RB.CONDITIONS["A4 order bias"]["order_bias"] > 0   # > 0 favours the first-presented profile
    jnd = float(js["jnd"])                                    # 0.674 Nm, 1 JND at s = 1 Nm
    return dict(lapse_a2=float(RB.CONDITIONS["A2 lapse 0.2"]["lapse"]),
                lapse_r3=float(ROB.CONDITIONS["R3 lapse=0.3"]["lapse"]),
                drift_b4a_nm=RB.CONDITIONS["B4a drift 1 JND"]["drift_jnd"] * jnd,
                drift_b4b_nm=RB.CONDITIONS["B4b drift 2 JND"]["drift_jnd"] * jnd,
                jnd_nm=jnd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=MAIN / "out")
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

    loss = {}      # (source, condition, setting, a) -> PsyPref(-T) minus FULL at k = 24
    other = {}     # (source, condition, setting, a, b) -> other differences at k = 24
    for e in js["differences"]:
        if e["k"] != K:
            continue
        if e["kind"] == "loss vs FULL":
            loss[e["source"], e["condition"], e["setting"], e["a"]] = e
        else:
            other[e["source"], e["condition"], e["setting"], e["a"], e["b"]] = e

    # ------------------------------------------------------------------ rows
    rows, prov, used, missing = [], [], [], []
    y = 0.0
    plotted = []   # every plotted value (points), for the axis-limit check
    for gi, (gkey, members) in enumerate(GROUPS):
        if gi:
            y += Y_GROUP_GAP
        header_y = y
        body = []
        yy = header_y + Y_HEADER_TO_ROW - 1.0
        for key, source, conds, opt in members:
            vals, provs = {}, []
            for setting in SETTINGS:
                cands = [loss[source, c, setting, opt] for c in conds if (source, c, setting, opt) in loss]
                if not cands:
                    missing.append(f"{key} {setting}: none of {conds} in {source}")
                    continue
                e = min(cands, key=lambda d: d["fixed"][0])
                worst = "worst of " + " / ".join(conds) if len(conds) > 1 else ""
                dup = e.get("duplicate_of") or ""
                fx, rg = pts(e["fixed"]), pts(e["registered"])
                same = bool(e["same_yardstick"])
                vals[setting] = dict(fx=fx, rg=rg, same=same, dup=dup, n=e["n"])
                provs.append([key, setting, source, e["condition"], opt, e["n"], int(same), dup.replace(" ", "_") or "-",
                              worst.replace(" / ", "|").replace(" ", "_") or "-"])
                used.append(e)
            if not vals:
                continue
            yy += 1.0
            r = [key, 0, yy]
            for setting in SETTINGS:
                v = vals.get(setting)
                if v is None or v["dup"]:
                    r += [float("nan")] * 8 + [v["n"] if v else 0]
                    continue
                fx, rg = v["fx"], v["rg"]
                show = float("nan") if v["same"] else rg[0]
                r += [fx[0], fx[1], fx[2], rg[0], rg[1], rg[2], show, int(v["same"]), v["n"]]
                plotted += [fx[1], fx[2]] + ([] if v["same"] else [rg[0]])
            # PRB2's scope: PsyPref under ROB2's answer conditions (A1-A8b), as icorr_amp_rob2.py scores it
            r.append(int(source == "ROB2" and opt == "psypref" and all(c[:1] == "A" for c in conds)))
            body.append(r)
            prov += provs
        if body:
            rows.append([gkey, 1, header_y] + [float("nan")] * 8 + [0] + [float("nan")] * 8 + [0] + [0])
            rows += body
            y = yy
    ymin, ymax = Y_TOP_LABELS - Y_TOP_PAD, y + Y_BOTTOM_PAD
    lo_all, hi_all = min(plotted), max(plotted)
    if not (XMIN < lo_all and hi_all < XMAX):
        raise SystemExit(f"plotted values span [{lo_all:.2f}, {hi_all:.2f}], outside the axis [{XMIN}, {XMAX}]")

    head = ["key", "is_header", "y"]
    for setting in SETTINGS:
        t = TAG[setting]
        head += [f"fx_{t}", f"fx_{t}_lo", f"fx_{t}_hi", f"rg_{t}", f"rg_{t}_lo", f"rg_{t}_hi", f"rgshow_{t}",
                 f"same_{t}", f"n_{t}"]
    head += ["prb2"]
    lab = label_values(js)
    args.dest.mkdir(parents=True, exist_ok=True)
    write_csv(args.dest / "fig4_rows.csv", head, rows)
    write_csv(args.dest / "fig4_meta.csv", ["xmin", "xmax", "ymin", "ymax", "ylabels", "guide", "k", "n_boot"]
              + list(lab), [[XMIN, XMAX, ymin, ymax, Y_TOP_LABELS, GUIDE, K, N_BOOT] + list(lab.values())])
    write_csv(args.dest / "fig4_provenance.csv",
              ["key", "setting", "source", "condition", "optimizer", "n", "same_yardstick", "duplicate_of", "worst_of"],
              [[p[0], p[1].replace(" ", "_"), p[2], p[3].replace(" ", "_")] + p[4:] for p in prov])

    # ------------------------------------------------------------------ caption numbers
    notes = []

    def note(item, e, b_name):
        notes.append([item, e["source"], e["condition"].replace(" ", "_"), e["setting"].replace(" ", "_"),
                      e["a"], b_name, e["n"]] + pts(e["registered"]) + pts(e["fixed"]))

    a_conds = [c for (s, c, st, a) in loss if s == "ROB2" and a == "psypref" and c[:1] == "A"]
    prb2 = [loss["ROB2", c, st, "psypref"] for c in dict.fromkeys(a_conds) for st in SETTINGS
            if 100 * loss["ROB2", c, st, "psypref"]["registered"][0] < PRB2_LIMIT]
    for e in prb2:
        note("prb2_registered_below_-5", e, "FULL")
    for st in SETTINGS:
        note("c4_vs_cmaes_schaefer", other["ROB2", "C4 pessimistic", st, "psypref", "cmaes_sch4"], "cmaes_sch4")
        note("c4_psypref_t_vs_cmaes_schaefer", other["TIE", "C4 pessimistic", st, "psypref_t", "cmaes_sch4"],
             "cmaes_sch4")
        note("c4_psypref_users_8_39_vs_cmaes_schaefer", other["TIE", "C4 pessimistic", st, "psypref", "cmaes_sch4"],
             "cmaes_sch4")
        for c, tag in (("C4 pessimistic", "c4"), ("A5 no preference", "a5")):
            note(f"{tag}_psypref_t_vs_psypref_users_8_39", other["TIE", c, st, "psypref_t", "psypref"], "psypref")
            note(f"{tag}_psypref_users_8_39_vs_full", loss["TIE", c, st, "psypref"], "FULL")
    fixed_pts = [r[3 + 9 * i] for r in rows if r[1] == 0 for i in range(2) if not math.isnan(r[3 + 9 * i])]
    n_within = sum(v >= GUIDE for v in fixed_pts)
    write_csv(args.dest / "fig4_notes.csv",
              ["item", "source", "condition", "setting", "a", "b", "n", "registered", "registered_lo",
               "registered_hi", "fixed", "fixed_lo", "fixed_hi"], notes)
    write_csv(args.dest / "fig4_summary.csv",
              ["filled_markers", "filled_at_or_above_guide", "guide", "rows", "rows_registered_differs"],
              [[len(fixed_pts), n_within, GUIDE, sum(r[1] == 0 for r in rows),
                sum(1 for r in rows if r[1] == 0 and (r[10] == 0 or r[19] == 0))]])

    # ------------------------------------------------------------------ print what was written
    print(f"source: {src}")
    print(f"written to {args.dest}: fig4_rows.csv, fig4_meta.csv, fig4_provenance.csv, fig4_notes.csv, fig4_summary.csv")
    for m in missing:
        print("missing:", m)
    print(f"axis {XMIN:g} .. {XMAX:g} points; plotted values span {lo_all:+.1f} .. {hi_all:+.1f}; y {ymin:g} .. {ymax:g}")
    print("\nrow   y       3-35 Nm fixed [95 %]   planned | fixed 12 Nm fixed [95 %]  planned | n")

    def cell(r, i):
        b = 3 + 9 * i
        if math.isnan(r[b]):
            return f"{'(as B5a)' if r[0] == 'B5b' and i == 1 else '-':>34}"
        reg = "=" if r[b + 7] else f"{r[b + 3]:+.1f}"
        return f"{r[b]:+6.1f} [{r[b + 1]:+6.1f}, {r[b + 2]:+6.1f}]  {reg:>6}    "

    for r in rows:
        if r[1]:
            print(f"{r[0]}  y = {r[2]:g}")
        else:
            print(f"  {r[0]:<4}{'*' if r[21] else ' '}{r[2]:6.2f} {cell(r, 0)}| {cell(r, 1)}| {r[11]} / {r[20]}")
    print("  (* = prb2: in the scope of ROB2's planned prediction PRB2)")
    print("\nlabel values (fig4_meta.csv): " + ", ".join(f"{k} = {v:.4g}" for k, v in lab.items()))
    print("\nprovenance (worst-of choices and duplicates):")
    for p in prov:
        if p[7] != "-" or p[8] != "-":
            print("  ", p)
    print(f"\nfilled markers at or above {GUIDE:g} points: {n_within} of {len(fixed_pts)}")
    print("\ncaption numbers (points; planned | fixed):")
    for n_ in notes:
        print(f"  {n_[0]}: {n_[1]} {n_[2]} {n_[3]} {n_[4]} - {n_[5]} (n = {n_[6]}): "
              f"{n_[7]:+.1f} [{n_[8]:+.1f}, {n_[9]:+.1f}] | {n_[10]:+.1f} [{n_[11]:+.1f}, {n_[12]:+.1f}]")
    p_rg = [n_[7] for n_ in notes if n_[0] == "prb2_registered_below_-5"]
    p_fx = [n_[10] for n_ in notes if n_[0] == "prb2_registered_below_-5"]
    n_prb2_rows = sum(r[21] for r in rows if r[1] == 0)
    print(f"\nPRB2 in the figure: {n_prb2_rows} rows marked; {len(p_rg)} cells below {PRB2_LIMIT:g} planned "
          f"({max(p_rg):+.1f} to {min(p_rg):+.1f}); the same cells on the fixed yardstick "
          f"{max(p_fx):+.1f} to {min(p_fx):+.1f}")
    rows_by = {r[0]: r for r in rows if r[1] == 0}
    unmarked_below = [(r[0], TAG[st], r[3 + 9 * i + 3]) for r in rows_by.values() if not r[21]
                      for i, st in enumerate(SETTINGS) if not r[3 + 9 * i + 7] and r[3 + 9 * i + 3] < PRB2_LIMIT]
    print("planned below the limit outside PRB2's scope: "
          + ", ".join(f"{k} {t} {v:+.1f}" for k, t, v in unmarked_below))

    if not args.no_verify:
        verify(args.out, used + [other[k] for k in other if k[1] in ("C4 pessimistic", "A5 no preference")
                                 and k[0] in ("ROB2", "TIE") and k[4] in ("cmaes_sch4", "psypref")])
    return 0


def verify(out: Path, entries: list[dict]) -> None:
    """Recompute every written triple from the stored session rows with the yardstick script's functions."""
    import icorr_posthoc_yardstick as Y   # noqa: PLC0415 (heavy imports only when verifying)

    d = Y.F._dir(out)
    raw = dict(full=Y._load(d / "full_sessions.jsonl"), rob=Y._load(d / "robust_sessions.jsonl"),
               rob2=Y._load(d / "rob2_sessions.jsonl"), tie=Y._load(d / "tie_sessions.jsonl"))
    cells = Y._cells(raw)
    if Y.R.N_BOOT != N_BOOT:
        raise SystemExit(f"bootstrap draws {Y.R.N_BOOT}, expected {N_BOOT}")
    bad = 0
    seen = set()
    for e in entries:
        ka = (e["source"], e["condition"], e["setting"], e["a"])
        kb = ("FULL", "main", e["setting"], "psypref") if e["b_source"] == "FULL" else (e["b_source"], e["condition"],
                                                                                      e["setting"], e["b"])
        if (ka, kb) in seen:
            continue
        seen.add((ka, kb))
        r = Y._diff(cells, e["kind"], ka, kb, K)
        ok = (r["n"] == e["n"] and r["same_yardstick"] == e["same_yardstick"]
              and all(abs(float(x) - float(z)) < 1e-12 for x, z in zip(r["registered"], e["registered"]))
              and all(abs(float(x) - float(z)) < 1e-12 for x, z in zip(r["fixed"], e["fixed"])))
        bad += not ok
        if not ok:
            print("VERIFY MISMATCH", ka, kb, r["registered"], e["registered"], r["fixed"], e["fixed"])
    print(f"\nverify: {len(seen) - bad} of {len(seen)} written differences recomputed from the stored rows "
          f"(fold-stratified, {N_BOOT} draws, rows sorted by user, the yardstick script's seeds): "
          f"{'all identical' if not bad else f'{bad} MISMATCHES'}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
