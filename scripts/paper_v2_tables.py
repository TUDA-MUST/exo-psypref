"""Candidate tables of the ICORR paper (v2): main results, assumptions and their cost, prediction ledger.

Usage:
  python -B scripts/paper_v2_tables.py
      [--out <repository>/out] [--dest <repository>/paper/figures/figures/tables]

Post hoc tabulation; no session is run and no planned verdict is changed. Every number printed
in the three .tex files is read from a stored report (JSON where one exists, else the report's
Markdown table, parsed here) and checked against at least one second source:

  table_main.tex          Table I (candidate, recommended form): % chosen over generic assistance
                          after 24 comparisons (to 0.1, as the abstract quotes it) per setting for
                          PsyPref, GP-BALD, the comparator (the best of the five published
                          optimizers at k = 24, our design notes; Fig. 2's hollow square CMA-ES /
                          triangle Arens' GP by each value), CMA-ES on Schaefer et al.'s family and the best 8-weight profile
                          (best-profile level), all from amp_table1.json; generic assistance's median distance
                          from the preferred profiles (Fig. 2's panel titles; amp_posthoc_users.md);
                          the first k at which PsyPref reaches the comparator's and GP-BALD's
                          k = 40 level (amp_table1.json); the parts at k = 24 (three times wider
                          prior, random pairs, GP-BALD) from amp_rnd_report.md's side-by-side table.
                          Checks: GP-BALD's part = -(PsyPref - GP-BALD) of amp_table1.json, and
                          equal to the printed GP-BALD row minus the printed PsyPref row; the
                          ABL and RND parts = the first tables of amp_abl_report.md and
                          amp_rnd_report.md; all three = Fig. 2's fig2_parts.csv (recomputed from
                          the session rows by paper_v2_fig2.py) at the printed digit; the
                          comparators = paper_v2_fig2.SPEC_BEST; the generic distances =
                          fig2_panels.csv at the printed digit.
  table_main_k3.tex       Table I, alternative form: the same at k = 12 / 24 / 40, whole % (one
                          decimal does not fit nine columns in a column's width).
  table_assumptions.tex   Table II (candidate): assumptions of the simulation, the simulated
                          deviations that break each, and the most negative change in % chosen over
                          generic assistance after 24 comparisons (point estimates, points):
                          rows of Fig. 4 from amp_yardstick.json (fixed yardstick, post hoc; the
                          same selection as paper_v2_fig4.GROUPS, checked against fig4_rows.csv),
                          the peak law from amp_pri_report.md (PAM: PsyPref - matched prior), the
                          prior's population from amp_pri_report.md (PAD: static - simulated user), the
                          number of weights from amp_ks_report.md (K = 12, 16, 24 - K = 8).
                          The sentence recommended instead of the table (for the rows Fig. 4 does
                          not carry, with the one PAD cell where static PsyPref trails CMA-ES on
                          Schaefer et al.'s family, cited to Fig. 4's caption, which prints the
                          same cell from ROB2 (C1, inside PRB1's scope), and PAN3's planned
                          follow-up with 30 curves of that population, amp_pan_report.md) is
                          written as a comment in the file.
  table_ledger.tex        Table III (candidate): planned predictions held / failed / uncertain,
                          pilots and design selection against confirmatory, from amp_ledger.json's
                          tallies (checked against amp_ledger.md's tally table and its entries).
                          The one-sentence alternative is written as a comment in the same file;
                          it counts predictions, not judged parts (one pilot prediction was judged
                          in two parts).

Rounding: % chosen to 0.1 in table_main.tex, whole % in table_main_k3.tex (the intervals are 1-5
points wide on each side); points to 0.1, as the reports print them. Signs are printed (+/-) for
changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # the repository
MAIN = ROOT                                 # out/ lives here
DEST = ROOT / "figures" / "tables"
FIGDATA = ROOT / "figures" / "data"

sys.path.insert(0, str(HERE))
import paper_v2_fig2 as F2  # noqa: E402  (constants only: SPEC_BEST)
import paper_v2_fig4 as F4  # noqa: E402  (constants only: GROUPS, K)

SETTINGS = ("medium", "wide", "fixed 12")              # 3-12 / 3-35 / fixed 12 Nm
SETTING_TEX = {"medium": r"3--12\,Nm", "wide": r"3--35\,Nm", "fixed 12": r"Fixed 12\,Nm"}
SETTING_MD = {"medium": "medium", "wide": "wide", "fixed 12": "fixed 12"}   # amp_posthoc_users.md
KS = (12, 24, 40)
K24 = 24
# Fig. 2's comparator markers (fig2.tex's \ftmarkpic: hollow, icOrange, 0.6 pt), as text symbols
# centred on the x-height of the 8-pt table text; pgfpictures, as in fig2.tex
MARK_DEFS = (r"\providecommand{\icTabMark}[3]{\begin{pgfpicture}\pgfsetlinewidth{0.6pt}"
             r"\pgfsetstrokecolor{icOrange}\pgfsetfillcolor{white}\pgfsetplotmarksize{#2}"
             r"\pgfuseplotmark{#1}\pgfsetbaseline{#3}\end{pgfpicture}}%" "\n"
             r"\providecommand{\icTabSq}{\icTabMark{square*}{1.7pt}{-1.9pt}}%" "\n"
             r"\providecommand{\icTabTri}{\icTabMark{triangle*}{2.1pt}{-1.3pt}}%")
MARK = {"CMA-ES": r"\icTabSq", "Arens' GP": r"\icTabTri"}
CMP_LABEL = r"CMA-ES\,\icTabSq{} or Arens' GP\,\icTabTri"        # Fig. 2's line and bar label
SCH_LABEL = r"CMA-ES, 4--6 param."
ROWS_T1 = (("psypref", "PsyPref"),
           ("gp_bald", "GP-BALD"),
           ("best", CMP_LABEL),
           ("cmaes_sch4", SCH_LABEL))
PARTS = (("abl", r"Prior 3$\times$ wider"), ("rnd", "Random pairs"),
         ("gpbald", r"No psychometric model"))
FAILS: list[str] = []


def sup(letter: str) -> str:
    return rf"$^{{\mathrm{{{letter}}}}}$"


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILS.append(msg)


def pct(x: float) -> str:
    """A proportion as whole % (Python's round-half-even on the value)."""
    return f"{100.0 * x:.0f}"


def pts(x: float, sign: bool = True) -> str:
    """Points to 0.1 with a typeset minus; + for positive values when sign."""
    s = f"{x:+.1f}" if sign else f"{x:.1f}"
    if s in ("+0.0", "-0.0"):
        s = "0.0"
    return s.replace("-", r"$-$").replace("+", r"$+$")


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


TRIPLE = re.compile(r"^(-?\d+(?:\.\d+)?) \[(-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)\]$")


def triple(s: str) -> tuple[float, float, float]:
    m = TRIPLE.match(s.strip())
    if not m:
        raise SystemExit(f"not an 'est [lo, hi]' cell: {s!r}")
    return tuple(float(v) for v in m.groups())


def section(text: str, heading: str) -> str:
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


# ----------------------------------------------------------------------------------- Table I
def table_main(d: Path, dest: Path) -> dict:
    t1 = json.loads((d / "amp_table1.json").read_text(encoding="utf-8"))
    check(t1["post_hoc"] is True and t1["registered"] is False, "amp_table1.json is not post hoc")
    check(t1["s_nm"] == 1.0 and t1["lapse"] == 0.1, "amp_table1.json: s / lapse")
    check(all(c["match"] for c in t1["self_checks"]) if isinstance(t1["self_checks"], list) else True,
          "amp_table1.json's own self-checks")
    rnd = (d / "amp_rnd_report.md").read_text(encoding="utf-8")
    abl = (d / "amp_abl_report.md").read_text(encoding="utf-8")

    # parts: amp_rnd_report.md's side-by-side table at k = 24
    sbs = {}
    for line in section(rnd, "## Side by side at k = 24").splitlines():
        c = cells(line)
        if len(c) == 4 and c[0] in SETTINGS:
            sbs[c[0]] = {"abl": triple(c[1]), "rnd": triple(c[2]), "gpbald": triple(c[3])}
    check(set(sbs) == set(SETTINGS), f"side-by-side settings {sorted(sbs)}")

    def first_table_k24(text: str, col: int) -> dict[str, float]:
        out = {}
        for line in text.splitlines():
            c = cells(line)
            if len(c) > col and c[0] in SETTINGS and c[1] == str(K24) and c[0] not in out:
                out[c[0]] = triple(c[col])[0]
        return out

    abl_k24 = first_table_k24(abl, 4)   # "wide − PsyPref (points)"
    rnd_k24 = first_table_k24(rnd, 4)   # "random − PsyPref (points)"
    fig2_parts = {}
    with open(FIGDATA / "fig2_parts.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            fig2_parts[r["code"]] = {"medium": float(r["m"]), "wide": float(r["w"]), "fixed 12": float(r["f"])}

    S = t1["settings"]
    out_rows = {}
    part_halfwidths = []
    for st in SETTINGS:
        s = S[st]
        check(s["n_users"] == 200, f"{st}: n_users {s['n_users']}")
        check(s["best_published"]["candidate"] == F2.SPEC_BEST[st],
              f"{st}: comparator {s['best_published']['candidate']} vs Fig. 2's {F2.SPEC_BEST[st]}")
        for key, _ in ROWS_T1:
            r = s["rows"][key]
            check(r["n"] == 200, f"{st} {key}: n {r['n']}")
            for k in KS:
                out_rows[st, key, k] = r["k"][str(k)]["est"]
        ceil = s["rows"]["ceiling"]["k"]
        check(len({round(ceil[str(k)]["est"], 12) for k in KS}) == 1, f"{st}: ceiling depends on k")
        out_rows[st, "ceiling"] = ceil["24"]["est"]
        # PsyPref - GP-BALD at k = 24 from the table's paired differences = -(GP-BALD part)
        gp_t1 = s["psypref_minus_points"]["gp_bald"][str(K24)]["est"]
        check(f"{-gp_t1:.1f}" == f"{sbs[st]['gpbald'][0]:.1f}",
              f"{st}: GP-BALD part {sbs[st]['gpbald'][0]} vs amp_table1 {-gp_t1:.4f}")
        check(f"{abl_k24[st]:.1f}" == f"{sbs[st]['abl'][0]:.1f}", f"{st}: ABL {abl_k24[st]} vs side-by-side")
        check(f"{rnd_k24[st]:.1f}" == f"{sbs[st]['rnd'][0]:.1f}", f"{st}: RND {rnd_k24[st]} vs side-by-side")
        for p, _ in PARTS:
            check(f"{fig2_parts[p][st]:.1f}" == f"{sbs[st][p][0]:.1f}",
                  f"{st} {p}: fig2_parts.csv {fig2_parts[p][st]:.3f} vs report {sbs[st][p][0]}")
            e, lo, hi = sbs[st][p]
            part_halfwidths.append(max(e - lo, hi - e))
        # the comparator's k = 24 point = section 23's choice in the JSON's CMP cells
        bp = s["best_published"]
        if bp["source"] == "CMP":
            check(abs(bp["cmp_cells_k24"][bp["candidate"]] - s["rows"]["best"]["k"]["24"]["est"]) < 1e-9,
                  f"{st}: comparator row vs its CMP cell")
    cmp_name = {st: S[st]["best_published"]["label"] for st in SETTINGS}
    check(cmp_name["wide"].startswith("Arens' GP") and all(cmp_name[s].startswith("CMA-ES")
                                                            for s in ("medium", "fixed 12")),
          f"comparator names {cmp_name}: the row label assumes Arens' GP in 3-35, CMA-ES otherwise")
    mark = {}
    for st in SETTINGS:
        m = [v for n, v in MARK.items() if cmp_name[st].startswith(n)]
        check(len(m) == 1, f"{st}: no marker for comparator {cmp_name[st]}")
        mark[st] = m[0] if m else ""
    fk = {st: S[st]["first_k_reaching_k40_level"] for st in SETTINGS}

    # generic assistance's median distance from the preferred profiles: amp_posthoc_users.md's
    # 'generic assistance' row per setting, checked against Fig. 2's panel titles (fig2_panels.csv)
    pu = (d / "amp_posthoc_users.md").read_text(encoding="utf-8")
    d0 = {}
    for st in SETTINGS:
        for line in section(pu, f"## {SETTING_MD[st]}\n").splitlines():
            c = cells(line)
            if len(c) == 7 and c[0] == "generic assistance":
                d0[st] = triple(c[4])[0]
    check(set(d0) == set(SETTINGS), f"generic distances {d0}")
    with open(FIGDATA / "fig2_panels.csv", encoding="utf-8") as fh:
        panels = {int(r["panel"]): float(r["d0_median_nm"]) for r in csv.DictReader(fh)}
    for i, st in enumerate(SETTINGS):
        check(f"{panels[i]:.2f}" == f"{d0.get(st, float('nan')):.2f}",
              f"{st}: generic distance {d0.get(st)} vs fig2_panels.csv {panels[i]}")

    # the 'No psychometric model' row is the GP-BALD row minus the PsyPref row: equal at the
    # printed digit in the k = 24 form (one decimal)
    for st in SETTINGS:
        diff = round(100 * out_rows[st, "gp_bald", K24], 1) - round(100 * out_rows[st, "psypref", K24], 1)
        check(f"{diff:.1f}" == f"{sbs[st]['gpbald'][0]:.1f}",
              f"{st}: printed GP-BALD - PsyPref {diff:.1f} vs part {sbs[st]['gpbald'][0]:.1f}")

    phw = max(part_halfwidths)

    def hw_range(ks: tuple[int, ...]) -> tuple[float, float]:
        h = []
        for st in SETTINGS:
            for key, _ in ROWS_T1:
                for k in ks:
                    e = S[st]["rows"][key]["k"][str(k)]
                    h.append(100 * max(e["est"] - e["lo"], e["hi"] - e["est"]))
        return min(h), max(h)

    def fk_cell(st: str, key: str) -> str:
        e = fk[st][key]
        return f"{e['est']:.0f} [{e['lo']:.0f}, {e['hi']:.0f}]"

    def note(ks: tuple[int, ...], printed_digit: bool) -> str:
        lo, hi = hw_range(ks)
        d_note = (r"The GP-BALD row minus PsyPref's" if printed_digit
                  else r"The GP-BALD row minus PsyPref's at $k = 24$, before rounding")
        return (r"\parbox{\columnwidth}{\scriptsize Means over $n = 200$ simulated users per setting from 5 "
                r"walking subjects ($\sigma = 1$\,Nm, $\lambda = 0.1$); 95\,\% intervals "
                rf"$\pm${lo:.0f}--{hi:.0f} points (changes: within $\pm${phw:.1f}), users resampled within "
                r"their walking subject; first $k$: [95\,\% interval]. Levels are not comparable across "
                r"settings (Fig.~2). Post hoc: the comparator's choice, first $k$, best profile. "
                + sup("a") + r"Median distance from the preferred profiles. "
                + sup("b") + r"Of the five published optimizers run here, the highest at $k = 24$; "
                r"8 weights, PsyPref's prior and start. "
                + sup("c") + r"On Sch\"afer et al.'s family, tuned settings. "
                + sup("d") + d_note + r": the same prior and informative pairs, a preference GP instead "
                r"of the psychometric answer model, with users who answer by that model. Changes are "
                r"not additive.}")

    def head(fname: str, form: str) -> list[str]:
        return [f"% {fname}: Table I (candidate; {form}) of the ICORR paper v2. Written by the amp-free",
                "% repository's scripts/paper_v2_tables.py from out/icorr/amp_free (amp_table1.json,",
                "% amp_rnd_report.md, amp_posthoc_users.md); do not edit by hand. Post hoc tabulation.",
                "% Comparators: " + "; ".join(f"{st}: {cmp_name[st]}" for st in SETTINGS),
                "% Markers: Fig. 2's hollow orange square (CMA-ES) and triangle (Arens' GP), defined",
                "% inside the float; they need pgfplots and icorr_style.tex (icOrange), as the figures do.",
                r"\begin{table}[t]", MARK_DEFS]

    # ------------------------------------------------------------ k = 24 form (recommended)
    def v24(st: str, key: str) -> str:
        v = f"{100.0 * out_rows[st, key, K24]:.1f}"
        return v + (rf"\rlap{{\,{mark[st]}}}" if key == "best" else "")

    L = head("table_main.tex", "k = 24, one decimal; recommended")
    L.append(r"\caption{Chosen Over Generic Assistance After 24 Comparisons (\%), and What Carries "
             r"PsyPref's Gain}")
    L.append(r"\label{tab:main}")
    L.append(r"\centering\footnotesize")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}")
    L.append(r"\toprule")
    L.append(" & " + " & ".join(SETTING_TEX[st] for st in SETTINGS) + r" \\")
    L.append("Generic assistance away (Nm)" + sup("a") + " & "
             + " & ".join(f"{d0[st]:.1f}" for st in SETTINGS) + r" \\")
    L.append(r"\midrule")
    for key, lab in ROWS_T1:
        lab2 = lab + (sup("b") if key == "best" else sup("c") if key == "cmaes_sch4" else "")
        L.append(lab2 + " & " + " & ".join(v24(st, key) for st in SETTINGS) + r" \\")
    L.append("Best 8-weight profile & " + " & ".join(f"{100.0 * out_rows[st, 'ceiling']:.1f}"
                                                   for st in SETTINGS) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{4}{@{}l}{\textit{First $k$ at which PsyPref reaches the 40-comparison level of}} \\")
    L.append(CMP_LABEL + " & " + " & ".join(fk_cell(st, "best") + rf"\rlap{{\,{mark[st]}}}"
                                            for st in SETTINGS) + r" \\")
    L.append("GP-BALD & " + " & ".join(fk_cell(st, "gp_bald") for st in SETTINGS) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{4}{@{}l}{\textit{Change against PsyPref after 24 comparisons (points)}} \\")
    for p, lab in PARTS:
        L.append(lab + (sup("d") if p == "gpbald" else "") + " & "
                 + " & ".join(pts(sbs[st][p][0]) for st in SETTINGS) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular*}")
    L.append(r"\\[2pt]")
    L.append(note((K24,), True))
    L.append(r"\end{table}")
    (dest / "table_main.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ k = 12 / 24 / 40 form (alternative)
    L = head("table_main_k3.tex", "k = 12 / 24 / 40, whole %; alternative")
    L.append(r"\caption{Chosen Over Generic Assistance (\%) After $k$ Comparisons, and What Carries "
             r"PsyPref's Gain}")
    L.append(r"\label{tab:main}")
    L.append(r"\centering\footnotesize")
    L.append(r"\setlength{\tabcolsep}{2.1pt}")
    # equal numeric columns, wide enough that three of them carry "Fixed 12 Nm"
    L.append(r"\begin{tabular}{@{}l*{9}{>{\centering\arraybackslash}p{10.8pt}}@{}}")
    L.append(r"\toprule")
    L.append(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{SETTING_TEX[st]}}}" for st in SETTINGS) + r" \\")
    L.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}")
    L.append("Generic assistance away (Nm)" + sup("a") + " & "
             + " & ".join(rf"\multicolumn{{3}}{{c}}{{{d0[st]:.1f}}}" for st in SETTINGS) + r" \\")
    L.append("Comparisons $k$ & " + " & ".join(str(k) for _ in SETTINGS for k in KS) + r" \\")
    L.append(r"\midrule")
    for key, lab in ROWS_T1:
        lab2 = lab + (sup("b") if key == "best" else sup("c") if key == "cmaes_sch4" else "")
        # (no marker by the values: the 10.8-pt columns leave no room; the first-k cells carry them)
        L.append(lab2 + " & " + " & ".join(pct(out_rows[st, key, k]) for st in SETTINGS for k in KS) + r" \\")
    L.append("Best 8-weight profile & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{pct(out_rows[st, 'ceiling'])}}}"
                                                       for st in SETTINGS) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{10}{@{}l}{\textit{First $k$ at which PsyPref reaches the 40-comparison level of}} \\")
    L.append(CMP_LABEL + " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{fk_cell(st, 'best')}\rlap{{\,{mark[st]}}}}}"
                                            for st in SETTINGS) + r" \\")
    L.append("GP-BALD & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{fk_cell(st, 'gp_bald')}}}" for st in SETTINGS)
             + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{10}{@{}l}{\textit{Change against PsyPref after 24 comparisons (points)}} \\")
    for p, lab in PARTS:
        L.append(lab + (sup("d") if p == "gpbald" else "") + " & "
                 + " & ".join(rf"\multicolumn{{3}}{{c}}{{{pts(sbs[st][p][0])}}}" for st in SETTINGS) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\\[2pt]")
    L.append(note(KS, False))
    L.append(r"\end{table}")
    (dest / "table_main_k3.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    return dict(rows=out_rows, parts=sbs, first_k=fk, hw=hw_range((K24,)), hw3=hw_range(KS), phw=phw,
                cmp=cmp_name, d0=d0)


# ----------------------------------------------------------------------------------- Table II
def table_assumptions(d: Path, dest: Path) -> dict:
    js = json.loads((d / "amp_yardstick.json").read_text(encoding="utf-8"))
    check(js["post_hoc"] is True and js["fixed_yardstick"] == {"s": 1.0, "lapse": 0.1}, "yardstick file")
    sc = js["self_check"]
    check(sc["points_matched"] == sc["points_total"] and all(e["ok"] for e in sc["exact"]),
          "amp_yardstick.json's own self-check")
    loss = {}
    for e in js["differences"]:
        if e["k"] == K24 and e["kind"] == "loss vs FULL":
            loss[e["source"], e["condition"], e["setting"], e["a"]] = e
    check(F4.K == K24, "Fig. 4's k")
    tw = ("wide", "fixed 12")

    # Fig. 4's rows, with its selection (worst of several conditions, by the fixed estimate)
    fig4 = {}
    for gkey, members in F4.GROUPS:
        for key, source, conds, opt in members:
            for st in tw:
                cands = [loss[source, c, st, opt] for c in conds if (source, c, st, opt) in loss]
                if not cands:
                    continue
                e = min(cands, key=lambda x: x["fixed"][0])
                if e.get("duplicate_of"):
                    continue
                fig4[key, st] = (gkey, 100.0 * e["fixed"][0], e["n"], opt)
    rows4 = {}
    with open(FIGDATA / "fig4_rows.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["is_header"] == "0":
                rows4[r["key"]] = r
    n_checked = 0
    for (key, st), (_, v, _, _) in fig4.items():
        col = "fx_w" if st == "wide" else "fx_f"
        check(key in rows4 and abs(float(rows4[key][col]) - v) < 1e-6,
              f"Fig. 4 row {key} {st}: {v} vs fig4_rows.csv {rows4.get(key, {}).get(col)}")
        n_checked += 1

    def worst(keys: list[str]) -> dict[str, tuple[float, str]]:
        out = {}
        for st in tw:
            vals = [(fig4[k, st][1], k) for k in keys if (k, st) in fig4]
            out[st] = min(vals)
        return out

    g_keys = {g: [k for k, *_ in m] for g, m in F4.GROUPS}
    feat = ["B5a", "B5b"]
    rows = {
        "answer": worst([k for k in g_keys["G1"] if k != "A5T"]),
        "change": worst(g_keys["G2"]),
        "judge": worst([k for k in g_keys["G3"] if k not in feat]),
        "features": worst(feat),
        "shapes": worst(g_keys["G4"]),
        "all": worst(["C4"]),
        "all_t": worst(["C4T"]),
    }

    pri = (d / "amp_pri_report.md").read_text(encoding="utf-8")
    pam = section(pri, "## PAM")
    pam_rows = []
    for line in pam.splitlines():
        c = cells(line)
        if len(c) == 9 and c[0][:1] == "M" and c[1] in ("medium", "wide"):
            pam_rows.append((c[0], c[1], float(c[2]), float(c[3]), triple(c[8])))
    check(len(pam_rows) == 12, f"PAM rows {len(pam_rows)}")
    for name, st, ps, mt, (dm, _, _) in pam_rows:  # matched - PsyPref = 100 (matched - PsyPref) to the print
        check(abs((mt - ps) * 100 - dm) <= 0.11, f"PAM {name} {st}: {mt}-{ps} vs {dm}")
    pam_cost = {st: min((-r[4][0], r[0]) for r in pam_rows if r[1] == st) for st in ("medium", "wide")}

    pad = section(pri, "## PAD")
    pad_rows = []
    for line in pad.splitlines():
        c = cells(line)
        if len(c) == 8 and c[0][:1] == "S" and c[1] in tw and c[5] != "–":
            pad_rows.append((c[0], c[1], float(c[2]), float(c[3]), triple(c[5]), triple(c[7])))
    check(len(pad_rows) == 6, f"PAD rows {len(pad_rows)}")
    for name, st, stat, orc, (dv, _, _), _ in pad_rows:
        check(abs((orc - stat) * 100 - dv) <= 0.11, f"PAD {name} {st}: {orc}-{stat} vs {dv}")
    pad_cost = {st: min((-r[4][0], r[0]) for r in pad_rows if r[1] == st) for st in tw}
    check(all(pad_cost[st][1].startswith("S3") and "Schaefer" in pad_cost[st][1] for st in tw),
          f"PAD: the worst population is not S3 (Schaefer shapes) in both settings: {pad_cost}")
    # where static PsyPref trails CMA-ES on Schaefer et al.'s family (users with that family's
    # shapes): planned (PAD4) only for S1 and S2, so any S3 cell is post hoc
    trail = [r for r in pad_rows if r[5][0] < 0]
    check(len(trail) == 1 and trail[0][0].startswith("S3") and trail[0][1] == "fixed 12" and trail[0][5][2] < 0,
          f"PAD: PsyPref behind CMA-ES (Schaefer) expected only in S3 fixed 12, interval below 0: {trail}")
    pad4 = re.search(r"- PAD4 \(([^)]*)\)", pad)
    check(bool(pad4) and "S1, S2" in pad4.group(1) and "S3" not in pad4.group(1),
          f"PAD4's scope: {pad4.group(1) if pad4 else None}")
    # the same cell is ROB2's C1 at fixed 12 (inside PRB1's planned scope, one of its failing
    # cells) and Fig. 4's caption note (fig4ab_notes.csv): the sentence cites Fig. 4 for it, so it
    # must agree there at the printed digit; PAN adds 30 exact curves of that
    # population to the prior
    with open(FIGDATA / "fig4ab_notes.csv", encoding="utf-8") as fh:
        notes = [r for r in csv.DictReader(fh) if r["item"] == "not_ahead_of_cmaes_schaefer"
                 and r["condition"].startswith("C1") and r["setting"] == "fixed_12"]
    check(len(notes) == 1 and trail and f"{float(notes[0]['fixed']):.1f}" == f"{trail[0][5][0]:.1f}",
          f"Fig. 4's note {notes} vs PAD S3 fixed 12 {trail[0][5] if trail else None}")
    pan = (d / "amp_pan_report.md").read_text(encoding="utf-8")
    pan_rows = {}
    for line in pan.splitlines():
        c = cells(line)
        if len(c) == 7 and c[0].startswith("S3") and c[1] == "fixed 12":
            pan_rows[c[2]] = c
    check("static" in pan_rows and "exact, n = 30" in pan_rows, f"PAN S3 fixed 12 rows {sorted(pan_rows)}")
    pan_static_cmp = triple(pan_rows["static"][6]) if "static" in pan_rows else (float("nan"),) * 3
    pan30 = triple(pan_rows["exact, n = 30"][6]) if "exact, n = 30" in pan_rows else (float("nan"),) * 3
    check(trail and f"{pan_static_cmp[0]:.1f}" == f"{trail[0][5][0]:.1f}"
          and pan_rows["static"][3] == f"{trail[0][2]:.3f}",
          f"PAN static S3 fixed 12 {pan_rows.get('static')} vs PAD {trail[0] if trail else None}")
    pan3 = re.search(r"- PAN3 \(([^)]*)\): (\w+)", pan)
    check(bool(pan3) and "30 exact curves ahead of CMA-ES" in pan3.group(1) and pan3.group(2) == "held"
          and pan30[1] > 0, f"PAN3: {pan3.group(0) if pan3 else None}; 30 exact curves {pan30}")

    ks = (d / "amp_ks_report.md").read_text(encoding="utf-8")
    kpref = section(ks, "## PsyPref: preference over generic assistance")
    ks_rows = []
    for line in kpref.splitlines():
        c = cells(line)
        if len(c) == 7 and c[0] in tw and c[1] in ("12", "16", "24"):
            ks_rows.append((c[0], int(c[1]), triple(c[5]), int(c[2])))
    check(len(ks_rows) == 6, f"KS rows {len(ks_rows)}")
    check(len({r[3] for r in ks_rows}) == 1, "KS rows' n differ")
    ks_n = ks_rows[0][3]
    ks_pk1 = re.search(r"PKS1 .*?wide (-?[\d.]+) \[.*?fixed 12 (-?[\d.]+) \[", ks)
    check(bool(ks_pk1) and {float(ks_pk1.group(1)), float(ks_pk1.group(2))}
          == {r[2][0] for r in ks_rows if r[1] == 16}, "KS: K = 16 rows vs PKS1's line")
    ks_cost = {st: min((r[2][0], f"K = {r[1]}") for r in ks_rows if r[0] == st) for st in tw}

    def c2(v: dict) -> str:
        return " & ".join(pts(v[st][0]) for st in tw)

    lapse_a2 = lapse_r3 = None
    meta = {}
    with open(FIGDATA / "fig4_meta.csv", encoding="utf-8") as fh:
        meta = next(csv.DictReader(fh))
    lapse_a2, lapse_r3 = float(meta["lapse_a2"]), float(meta["lapse_r3"])
    d1, d2 = float(meta["drift_b4a_nm"]), float(meta["drift_b4b_nm"])

    # compact wording; the vocabulary of Fig. 4's row labels (fig4.tex), which carry the detail
    body = [
        (r"Probit answers, $\sigma = 1$\,Nm, $\lambda = 0.1$",
         rf"$\sigma$ halved, doubled or per user; $\lambda = {lapse_a2:g}$, ${lapse_r3:g}$; other answer-curve "
         r"shapes; device noise; order bias; frequent `no preference'",
         c2(rows["answer"])),
        (r"No change within the session",
         rf"Fatigue; discrimination improving or worsening; drift {d1:.2f} or {d2:.2f}\,Nm; familiarity",
         c2(rows["change"])),
        (r"Judged: RMS distance of the whole profile",
         r"Weber's law; asymmetry; gait-phase weights; anisotropy; no single preferred profile; "
         r"negative power", c2(rows["judge"])),
        (r"", r"Single profile features, static or walking", c2(rows["features"])),
        (r"Shapes: walking hip moments, as the prior",
         r"Surrogate, Sch\"afer-family or heavy-tailed shapes; family mixture",
         c2(rows["shapes"])),
        (r"Prior from the users' population",
         r"Flexion peak 6 or 12\,\%GC later; Sch\"afer-family shapes",
         c2(pad_cost)),
        (r"Peaks log-uniform, per bout",
         r"Tight cohorts; one bout larger; correlated peaks",
         pts(pam_cost["wide"][0]) + r" & n.a."),
        (r"8 weights", r"12, 16 or 24 weights", c2(ks_cost)),
        (r"Several at once", r"Pessimistic user: $\sigma = 2$\,Nm, fatigue, order bias, `no preference', "
         r"phase weights, Sch\"afer-family shapes", c2(rows["all"])),
        (r"", r"\hspace{4pt}with PsyPref-T", c2(rows["all_t"])),
    ]
    L = []
    L.append("% table_assumptions.tex: Table II (candidate) of the ICORR paper v2. Written by the amp-free")
    L.append("% repository's scripts/paper_v2_tables.py (amp_yardstick.json with Fig. 4's selection;")
    L.append("% amp_pri_report.md PAM and PAD; amp_ks_report.md); do not edit by hand. Post hoc.")
    for name, v in rows.items():
        L.append(f"%   {name}: " + "; ".join(f"{st} {v[st][0]:+.2f} ({v[st][1]})" for st in tw))
    L.append("%   peaks (PAM, matched - PsyPref, worst): " + "; ".join(
        f"{st} {pam_cost[st][0]:+.1f} ({pam_cost[st][1]})" for st in ("medium", "wide")))
    L.append("%   prior population (PAD, static - oracle, worst): " + "; ".join(
        f"{st} {pad_cost[st][0]:+.1f} ({pad_cost[st][1]})" for st in tw))
    L.append("%   weights (KS, K - 8, worst): " + "; ".join(f"{st} {ks_cost[st][0]:+.1f} ({ks_cost[st][1]})"
                                                          for st in tw))
    # the sentence recommended instead of the table: the three rows Fig. 4 does not carry
    tr = trail[0][5] if trail else (float("nan"),) * 3
    ks_k = sorted({r[1] for r in ks_rows})
    ks_tail = ("none at fixed 12\\,Nm" if ks_cost["fixed 12"][0] >= 0
               else f"{-ks_cost['fixed 12'][0]:.1f} at fixed 12\\,Nm")
    sentence = (f"A prior from another population than the users' cost PsyPref at most "
                f"{-pad_cost['wide'][0]:.1f} and {-pad_cost['fixed 12'][0]:.1f} points (users with "
                f"Sch\\\"afer-family shapes; 3--35 and fixed 12\\,Nm). At fixed 12\\,Nm, PsyPref then trailed "
                f"CMA-ES on Sch\\\"afer et al.'s family, whose shapes these users had, by {-tr[0]:.1f} points "
                f"(Fig.~4); with 30 curves of that population added to its prior, it led by {pan30[0]:.1f} "
                f"[{pan30[1]:.1f}, {pan30[2]:.1f}] points. A different law of preferred peaks "
                f"cost at most {-pam_cost['wide'][0]:.1f} points (3--35\\,Nm; {-pam_cost['medium'][0]:.1f} in "
                f"3--12\\,Nm), and {ks_k[0]}--{ks_k[-1]} weights instead of 8 at most "
                f"{-min(ks_cost['wide'][0], 0):.1f} (3--35\\,Nm; {ks_tail}).")
    L.append("% Recommended instead of the table (Results, after Fig. 4; the other rows are Fig. 4's):")
    L.append("%   " + sentence)
    L.append(r"\begin{table}[t]")
    L.append(r"\caption{Assumptions of the Simulation, How Each Was Broken, and Its Cost}")
    L.append(r"\label{tab:assumptions}")
    L.append(r"\centering\footnotesize")
    L.append(r"\setlength{\tabcolsep}{2.2pt}")
    L.append(r"\renewcommand{\arraystretch}{0.96}")
    L.append(r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.3cm}"
             r">{\raggedright\arraybackslash}p{3.85cm}rr@{}}")
    L.append(r"\toprule")
    L.append(r"Assumption & Broken by (simulated) & \multicolumn{2}{c@{}}{Change$^{\mathrm{a}}$} \\")
    L.append(r"\cmidrule(l){3-4}")
    L.append(r" & & 3--35 & Fixed 12 \\")
    L.append(r"\midrule")
    for i, (a, b, c) in enumerate(body):
        sep = r"\addlinespace[1.2pt]" if i and a else ""
        L.append(sep + f"{a} & {b} & {c} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\\[2pt]")
    L.append(r"\parbox{\columnwidth}{\scriptsize $^{\mathrm{a}}$Most negative change among the row's "
             r"conditions in \% chosen over generic assistance after 24 comparisons against the main "
             r"condition, same users, points, scored at $\sigma = 1$\,Nm and $\lambda = 0.1$ (Fig.~4's rows: "
             r"fixed yardstick, post hoc; intervals there). Prior: static minus a prior from the users' "
             r"population. Peaks: minus a prior with the users' peak law (3--12\,Nm: "
             rf"{pts(pam_cost['medium'][0])}); n.a.: fixed level. Weights: $K$ minus 8 weights "
             rf"($n = {ks_n}$).}}")
    L.append(r"\end{table}")
    (dest / "table_assumptions.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    return dict(rows=rows, pam=pam_cost, pad=pad_cost, ks=ks_cost, fig4_checked=n_checked,
                trail=trail, sentence=sentence)


# ----------------------------------------------------------------------------------- Table III
def table_ledger(d: Path, dest: Path) -> dict:
    lj = json.loads((d / "amp_ledger.json").read_text(encoding="utf-8"))
    check(lj["label"].startswith("post hoc"), "ledger label")
    G = lj["tallies"]["groups"]
    sh = lj["tallies"]["shares"]
    md = (d / "amp_ledger.md").read_text(encoding="utf-8")
    tally = {}
    for line in section(md, "## Tally").splitlines():
        c = cells(line)
        if len(c) == 11 and c[0] in ("pilots and design selection", "confirmatory"):
            tally["pilot" if c[0].startswith("pilots") else "confirmatory"] = c
    for g in ("pilot", "confirmatory"):
        t = G[g]
        check(t["held"] + t["failed"] + t["uncertain"] == t["judged"] == t["units"], f"ledger {g}: sums")
        check(t["not yet run"] == 0 and t["not judged"] == 0 and t["conflict"] == 0, f"ledger {g}: open units")
        c = tally[g]
        check([int(c[2]), int(c[3]), int(c[4]), int(c[5]), int(c[6])]
              == [t["registered"], t["units"], t["held"], t["failed"], t["uncertain"]], f"ledger {g} vs md")
        n_ent = {v: sum(1 for e in lj["entries"] if e["group"] == g and e["verdict"] == v)
                 for v in ("held", "failed", "uncertain")}
        check(n_ent == {v: t[v] for v in n_ent}, f"ledger {g}: entries {n_ent}")
        check(c[9].startswith(f"{100 * sh[g]['held'][0]:.0f} %"), f"ledger {g}: held share vs md {c[9]}")

    def share(g: str, v: str) -> str:
        e, lo, hi = sh[g][v]
        return f"{100 * e:.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"

    P, C = G["pilot"], G["confirmatory"]
    # predictions, not judged parts: a prediction judged in parts held in full when every part held,
    # in part when some did
    by_id: dict[str, dict[str, list[str]]] = {}
    for e in lj["entries"]:
        by_id.setdefault(e["group"], {}).setdefault(e["id"], []).append(e["verdict"])
    pred = {}
    for g in ("pilot", "confirmatory"):
        ids = by_id[g]
        check(len(ids) == G[g]["registered"], f"ledger {g}: {len(ids)} ids vs {G[g]['registered']} planned")
        split = {i: v for i, v in ids.items() if len(v) > 1}
        pred[g] = dict(n=len(ids), full=sum(all(x == "held" for x in v) for v in ids.values()),
                       part=sum(("held" in v) and not all(x == "held" for x in v) for v in ids.values()),
                       split=split)
    check(sum(len(v) for v in by_id["pilot"].values()) == P["units"], "ledger: pilot parts vs units")
    check(not pred["confirmatory"]["split"] and C["registered"] == C["units"],
          "ledger: a confirmatory prediction judged in parts (the sentence counts units as predictions)")
    pp = pred["pilot"]
    held_any = pp["full"] + pp["part"]
    part_txt = ("" if not pp["part"] else
                f", {'one' if pp['part'] == 1 else pp['part']} of them only in part")
    sentence = (f"Of {C['registered']} predictions fixed before the confirmatory runs, {C['held']} held, "
                f"{C['failed']} failed and {C['uncertain']} remained uncertain; in the pilots that selected "
                f"the design, {held_any} of {pp['n']} held{part_txt} (full ledger in the repository).")
    split_txt = "; ".join(f"{i}: " + ", ".join(v) for i, v in pp["split"].items())
    L = []
    L.append("% table_ledger.tex: Table III (candidate) of the ICORR paper v2. Written by the amp-free")
    L.append("% repository's scripts/paper_v2_tables.py from amp_ledger.json (post hoc ledger of the")
    L.append("% planned predictions; scripts/icorr_posthoc_ledger.py); do not edit by hand.")
    L.append(f"% Pilot predictions judged in parts: {split_txt}.")
    L.append("% Recommended instead of the table (one sentence, Methods' statistics paragraph or Results):")
    L.append("%   " + sentence)
    L.append(r"\begin{table}[t]")
    L.append(r"\caption{Predictions Fixed Before the Runs and Their Outcomes}")
    L.append(r"\label{tab:ledger}")
    L.append(r"\centering\footnotesize")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\begin{tabular}{@{}lrrrrc@{}}")
    L.append(r"\toprule")
    L.append(r"Runs & Predictions & Held & Failed & Uncertain & Held (\%) \\")
    L.append(r"\midrule")
    L.append(rf"Pilots, design selection & {P['registered']}$^{{\mathrm{{a}}}}$ & {P['held']} & {P['failed']} & "
             rf"{P['uncertain']} & {share('pilot', 'held')} \\")
    L.append(rf"Confirmatory & {C['registered']} & {C['held']} & {C['failed']} & {C['uncertain']} & "
             rf"{share('confirmatory', 'held')} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\\[2pt]")
    n_split = len(pp["split"])
    L.append(r"\parbox{\columnwidth}{\scriptsize Held: 95\,\% interval wholly on the predicted side; failed: "
             r"wholly on the other; uncertain: otherwise. Post hoc tally; brackets: 95\,\% interval over "
             rf"test plan ({sh['pilot']['clusters']} and {sh['confirmatory']['clusters']}), descriptive. "
             rf"$^{{\mathrm{{a}}}}${'One' if n_split == 1 else n_split} judged in two parts; counts and shares "
             rf"of the {P['units']} parts. Full ledger in the repository.}}")
    L.append(r"\end{table}")
    (dest / "table_ledger.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    return dict(pilot=P, confirmatory=C, shares=sh, pred=pred, sentence=sentence)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=MAIN / "out")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    d = args.out / "icorr" / "amp_free"
    args.dest.mkdir(parents=True, exist_ok=True)
    r1 = table_main(d, args.dest)
    r2 = table_assumptions(d, args.dest)
    r3 = table_ledger(d, args.dest)
    if FAILS:
        print("SELF-CHECKS FAILED:")
        for f in FAILS:
            print("  -", f)
        return 1
    print("Table I: PsyPref k = 24 (%):", ", ".join(f"{st} {100 * r1['rows'][st, 'psypref', 24]:.1f}"
                                                    for st in SETTINGS))
    print("  generic assistance's median distance (Nm):", r1["d0"])
    print("  first k (comparator):", {st: r1["first_k"][st]["best"]["est"] for st in SETTINGS})
    print("  parts:", {st: {p: v[0] for p, v in r1["parts"][st].items()} for st in SETTINGS})
    print(f"  interval half-widths k = 24: {r1['hw'][0]:.2f}-{r1['hw'][1]:.2f} points "
          f"(k = 12/24/40: {r1['hw3'][0]:.2f}-{r1['hw3'][1]:.2f}); parts within {r1['phw']:.1f}")
    print("Table II rows (worst, points):")
    for k, v in r2["rows"].items():
        print(f"  {k}: " + ", ".join(f"{st} {x[0]:+.2f} ({x[1]})" for st, x in v.items()))
    print("  peaks (PAM):", r2["pam"], " prior (PAD):", r2["pad"], " weights (KS):", r2["ks"])
    print(f"  Fig. 4 cells checked against fig4_rows.csv: {r2['fig4_checked']}")
    print("  PsyPref behind CMA-ES (Schaefer family), PAD:", [(r[0], r[1], r[5]) for r in r2["trail"]])
    print("  sentence instead of Table II:", r2["sentence"])
    print("Table III (units):", {g: (r3[g]['units'], r3[g]['held'], r3[g]['failed'], r3[g]['uncertain'])
                                 for g in ("pilot", "confirmatory")})
    print("  predictions:", {g: {k: v for k, v in r3["pred"][g].items()} for g in ("pilot", "confirmatory")})
    print("  sentence instead of Table III:", r3["sentence"])
    print("All self-checks passed; wrote", ", ".join(p.name for p in sorted(args.dest.glob("table_*.tex"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
