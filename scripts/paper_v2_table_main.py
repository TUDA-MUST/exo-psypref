"""Table I of the ICORR paper in its frame of 2026-11-04: what speeds up preference tuning.

Usage:
  python -B scripts/paper_v2_table_main.py
      [--out <repository>/out] --dest <paper>/figures/figures/tables

Post hoc tabulation; no session is run and no planned verdict is changed. Writes table_main.tex
from the stored sessions and reports and checks it against amp_table1.json at the printed digit:
preference and median distance after 24 comparisons for PsyPref, GP-BALD, the strongest other
optimizer per setting and CMA-ES on Schaefer et al.'s family, with the range of the other ProMP
configurations (FULL, CMP) in the note; the first k (amp_table1.json); the ablations (ABL, RND) and
the prior's width (PW). It also prints what the text quotes: distances by k, PsyPref's leads, and
preference by number of weights (FULL/CMP, KS, KS2, LONG; n = 100).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = ROOT
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_ks as KS  # noqa: E402
import paper_v2_tables as T  # noqa: E402

F = KS.F
SETTINGS = T.SETTINGS
K24 = 24
#: every optimizer's session file: FULL's and CMP's
SOURCE = {"psypref": "full", "gp_bald": "full", "g": "full", "d_z": "full", "cmaes": "full",
          "cmaes_sch4": "full", "cmaes4": "cmp", "g_generic": "cmp", "g_generic_nogate": "cmp",
          "gp_eubo": "cmp", "ramella": "cmp"}
#: the ProMP configurations of the five published optimizers, among which the strongest is picked
PROMP = ("cmaes4", "g_generic", "g_generic_nogate", "g", "cmaes", "gp_eubo", "d_z", "ramella")
WIDTHS = (1.5, 2.0)
ALL_K = (8, 12, 16, 24)
SWEEP = ("psypref", "gp_bald", "cmaes4")


def _read(path: Path) -> list[dict]:
    return [r for r in KS._read(path) if r.get("error") is None]


def table(d: Path, dest: Path) -> None:
    t1 = json.loads((d / "amp_table1.json").read_text(encoding="utf-8"))["settings"]
    full, cmp_ = _read(d / "full_sessions.jsonl"), _read(d / "cmp_sessions.jsonl")
    cell = {}
    for src, rows in (("full", full), ("cmp", cmp_)):
        for r in rows:
            cell.setdefault((src, r["setting"], r["candidate"]), []).append(r)
    pref = {(st, c): 100 * F._pref_k(cell[src, st, c], K24) for c, src in SOURCE.items() for st in SETTINGS}
    dist = {(st, c, k): float(np.median([r["distances"][k - 1] for r in cell[src, st, c]]))
            for c, src in SOURCE.items() for st in SETTINGS for k in (12, 24, 40)}
    strongest = {st: t1[st]["best_published"]["candidate"] for st in SETTINGS}

    def direct(st, other) -> float:
        """How often users would choose PsyPref's profile over the other's after 24 comparisons (%)."""
        ours = {r["user"]: r["distances"][K24 - 1] for r in cell["full", st, "psypref"]}
        return 100 * float(np.mean([F.AF.LAPSE / 2 + (1 - F.AF.LAPSE) * norm.cdf((r["distances"][K24 - 1] - ours[r["user"]]) / F.AF.S)
                                    for r in cell[SOURCE[other], st, other]]))

    for st in SETTINGS:
        T.check(all(len(cell[src, st, c]) == 200 for c, src in SOURCE.items()), f"{st}: n")
        for c, key in (("psypref", "psypref"), ("gp_bald", "gp_bald"), (strongest[st], "best"), ("cmaes_sch4", "cmaes_sch4")):
            T.check(f"{pref[st, c]:.1f}" == f"{100 * t1[st]['rows'][key]['k']['24']['est']:.1f}", f"{st} {c}: preference")
        T.check(max(pref[st, c] for c in PROMP) == pref[st, strongest[st]], f"{st}: the strongest is not the highest")
    others = {st: [pref[st, c] for c in PROMP if c != strongest[st]] for st in SETTINGS}
    mark = {st: T.MARK["CMA-ES" if strongest[st].startswith("cmaes") else "Arens' GP"] for st in SETTINGS}

    # parts: ABL and RND at k = 24, and PW
    rnd = (d / "amp_rnd_report.md").read_text(encoding="utf-8")
    parts = {}
    for line in T.section(rnd, "## Side by side at k = 24").splitlines():
        c = T.cells(line)
        if len(c) == 4 and c[0] in SETTINGS:
            parts[c[0]] = {"abl": T.triple(c[1]), "rnd": T.triple(c[2])}
    base = {(r["user"], r["setting"]): r for r in full if r["candidate"] == "psypref"}
    pw = {}
    for r in _read(d / "pw_sessions.jsonl"):
        pw.setdefault((r["setting"], r["width"]), []).append(r)
    width = {}
    for st in SETTINGS:
        for c in WIDTHS:
            rows = pw[st, c]
            T.check(len(rows) == 200, f"PW {st} {c}: n {len(rows)}")
            width[st, c] = 100 * (F._pref_k(rows, K24) - F._pref_k([base[r["user"], st] for r in rows], K24))

    # by number of weights, for the text: FULL's and CMP's K = 8, KS's, KS2's; the sweep's 100 users
    sets = ("wide", "fixed 12")
    ks = {KS._key(r): r for r in _read(d / "ks_sessions.jsonl")}
    ks2 = {KS._key(r): r for r in _read(d / "ks2_sessions.jsonl")}
    users = {(u, s) for (_, _, u, s) in ks2}

    def sweep(c, k, s):
        if k == 8:
            return [r for r in cell[SOURCE[c], s, c] if (r["user"], s) in users]
        rows = ks2 if (c != "psypref" and k in (12, 24)) else ks
        return [r for (cc, kk, _, ss), r in rows.items() if cc == c and kk == k and ss == s]

    by_k = {}
    for s in sets:
        for k in ALL_K:
            for c in SWEEP:
                rows = sweep(c, k, s)
                T.check(len(rows) == 100, f"sweep {s} K {k} {c}: n {len(rows)}")
                by_k[s, k, c] = 100 * F._pref_k(rows, K24)

    def fk(st, key) -> str:
        e = t1[st]["first_k_reaching_k40_level"][key]
        return f"{e['est']:.0f} [{e['lo']:.0f}, {e['hi']:.0f}]"

    def value(st, c) -> str:
        return f"{pref[st, c]:.1f} ({dist[st, c, K24]:.1f})"

    ceil = {st: t1[st]["rows"]["ceiling"]["k"]["24"]["est"] for st in SETTINGS}
    floor = {st: t1[st]["median_floor_nm"] for st in SETTINGS}
    hw = [100 * max(e["est"] - e["lo"], e["hi"] - e["est"]) for st in SETTINGS
          for e in (t1[st]["rows"][k]["k"]["24"] for k in ("psypref", "gp_bald", "best", "cmaes_sch4"))]
    ranges = ", ".join(f"{min(others[st]):.1f}--{max(others[st]):.1f}" for st in SETTINGS[:2]) + f" and {min(others[SETTINGS[2]]):.1f}--{max(others[SETTINGS[2]]):.1f}"
    L = ["% table_main.tex: Table I of the ICORR paper. Written by the amp-free worktree's",
         "% scripts/paper_v2_table_main.py from out/icorr/amp_free; do not edit by hand. Post hoc tabulation.",
         "% Markers: Fig. 2's hollow orange square (CMA-ES) and triangle (Arens' GP), defined inside the float.",
         r"\begin{table}[t]", T.MARK_DEFS,
         r"\caption{Results After 24 Comparisons}",
         r"\label{tab:main}", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lccc@{}}", r"\toprule",
         " & " + " & ".join(T.SETTING_TEX[st] for st in SETTINGS) + r" \\", r"\midrule",
         r"\multicolumn{4}{@{}l}{\textit{$P_\mathrm{g}$ (\%), and median distance from the preferred profile (Nm)}} \\",
         "Generic assistance & " + " & ".join(f"50 ({np.median([r['d0'] for r in cell['full', st, 'psypref']]):.1f})" for st in SETTINGS) + r" \\",
         "PsyPref & " + " & ".join(value(st, "psypref") for st in SETTINGS) + r" \\",
         "GP-BALD" + T.sup("a") + " & " + " & ".join(value(st, "gp_bald") for st in SETTINGS) + r" \\",
         T.CMP_LABEL + T.sup("b") + " & " + " & ".join(value(st, strongest[st]) + rf"\rlap{{\,{mark[st]}}}" for st in SETTINGS) + r" \\",
         T.SCH_LABEL + T.sup("c") + " & " + " & ".join(value(st, "cmaes_sch4") for st in SETTINGS) + r" \\", r"\midrule",
         r"\multicolumn{4}{@{}l}{\textit{PsyPref's profile chosen over (\%)}} \\",
         "The comparator's & " + " & ".join(f"{direct(st, strongest[st]):.1f}" + rf"\rlap{{\,{mark[st]}}}" for st in SETTINGS) + r" \\",
         T.SCH_LABEL + "'s & " + " & ".join(f"{direct(st, 'cmaes_sch4'):.1f}" for st in SETTINGS) + r" \\", r"\midrule",
         r"First $k$ at comparator's $k = 40$ & " + " & ".join(fk(st, "best") + rf"\rlap{{\,{mark[st]}}}" for st in SETTINGS) + r" \\", r"\midrule",
         r"\multicolumn{4}{@{}l}{\textit{Change against PsyPref (points)}} \\"]
    for c in (w for w in WIDTHS if w == 2):
        L.append(rf"Prior {c:g}$\times$ wider & " + " & ".join(T.pts(width[st, c]) for st in SETTINGS) + r" \\")
    L += [r"Prior 3$\times$ wider & " + " & ".join(T.pts(parts[st]["abl"][0]) for st in SETTINGS) + r" \\",
          "Random pairs & " + " & ".join(T.pts(parts[st]["rnd"][0]) for st in SETTINGS) + r" \\",
          r"\bottomrule", r"\end{tabular*}", r"\\[2pt]",
          rf"\parbox{{\columnwidth}}{{\scriptsize Means ($P_\mathrm{{g}}$) over $n = 200$ simulated users per setting "
          rf"($\sigma = 1$\,Nm, $\lambda = 0.1$); 95\,\% intervals of $P_\mathrm{{g}}$ $\pm${min(hw):.0f}--{max(hw):.0f} points; 1~JND: 0.67\,Nm. "
          r"Post hoc: the comparator's choice, the distances, direct choices, first $k$. Changes are not additive. "
          + T.sup("a") + r"Prior and informative pairs without the psychometric answer model; not a published configuration. "
          + T.sup("b") + rf"Highest of eight configurations of five published optimizers, with PsyPref's prior; the others reached {ranges}\,\%. "
          + T.sup("c") + r"On Sch\"afer et al.'s family.}", r"\end{table}"]
    (dest / "table_main.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    print("Text values, median distance (Nm) at k = 12 / 24 / 40:")
    for c in ("psypref", "gp_bald", "cmaes4", "g_generic_nogate", "cmaes_sch4"):
        print(f"  {c:18s}", {st: [round(dist[st, c, k], 2) for k in (12, 24, 40)] for st in SETTINGS})
    print("  leads over the strongest:", {st: t1[st]["psypref_minus_points"]["best"]["24"] for st in SETTINGS})
    print("  width, points:", {k: round(v, 1) for k, v in width.items()})
    print("  P_g by K:", {f"{s} {k}": [round(by_k[s, k, c], 1) for c in SWEEP] for s in sets for k in ALL_K})
    print("  the others' range:", ranges)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=MAIN / "out")
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    table(args.out / "icorr" / "amp_free", args.dest)
    if T.FAILS:
        print("SELF-CHECKS FAILED:")
        for f in T.FAILS:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
