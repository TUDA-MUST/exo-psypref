"""Post hoc: PsyPref's leads per walking subject (FULL, CMP and FAM rows).

Usage:
  python scripts/icorr_posthoc_subjects.py [--out <repository>/out]

Not planned. Reads stored session rows only (full_sessions.jsonl, cmp_sessions.jsonl,
fam_sessions.jsonl); nothing is rerun and no planned verdict is re-judged. FULL's 200 users
per setting are 40 per LOSO fold, and the users of fold loso-ABxx all take their walking shape
from the held-out subject ABxx (checked against full-<fold>.npz), so a fold is a subject. For
each setting (medium, wide, fixed 12), subject and k = 24, 40 the script gives the mean paired
lead in points of preference over generic assistance, lambda/2 + (1 - lambda) Phi((d0 - d_k)/s)
with s = 1.0 Nm and lambda = 0.1, of
  - PsyPref (FULL) over the strongest other optimizer optimizer of our design notes: the highest mean
    preference at k = 24 over all 200 users among FULL's Arens' GP, LineCoSpar and CMA-ES on the
    ProMP and the five CMP cells; the same optimizer is kept at k = 40 (the strongest at k = 40
    is added as a sensitivity row where it differs);
  - PsyPref over GP-BALD and over CMA-ES on Schaefer's family, tuned and with
    pycma's default population;
  - PsyPref with projection onto Franks et al.'s family (FAM; wide and fixed 12 only) over
    CMA-ES on Schaefer's family, tuned.
Per subject: the lead with a 95 % interval from 2,000 draws of the subject's 40 users; the
number of subjects with a positive lead; descriptive, a t-interval (df = 4) over the five
subject means; and the pooled lead with the fold-stratified interval (icorr_amp_range._boot).
Pairs are sorted by user, so the intervals do not depend on file order. Context per subject:
generic assistance's median distance to the preferred curve (Nm and JND, 0.674 Nm) and
PsyPref's preference. Self-check: the CMP report's first paired cell is redrawn with that
report's generator, and every pooled lead here that FULL's, CMP's or FAM's report prints is
compared with the printed value. Outputs in <out>/icorr/amp_free/: amp_subjects.md,
amp_subjects.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_cmp as C  # noqa: E402
import icorr_amp_full as F  # noqa: E402
import icorr_amp_pri as P  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402

KS = (24, 40)
JND = P.JND   # true_jnd(s = 1.0) = 0.674 Nm
FAM_SETTINGS = ("wide", "fixed 12")
NAMES = {**F.NAMES, **C.NAMES}
RNG_OFFSET = 101
COMPARISONS = ("best_published", "gp_bald", "cmaes_sch4", "cmaes_sch", "franks_cmaes_sch4")
LABELS = {"best_published": "PsyPref − best published",
          "gp_bald": "PsyPref − GP-BALD",
          "cmaes_sch4": "PsyPref − CMA-ES (Schäfer, tuned)",
          "cmaes_sch": "PsyPref − CMA-ES (Schäfer, default population)",
          "franks_cmaes_sch4": "Franks projection − CMA-ES (Schäfer, tuned)",
          "strongest_published_k40": "PsyPref − strongest published at k = 40"}


def _load(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("error") is None:
            rows[F._key(r)] = r
    return rows


def _cell(src: dict, setting: str, cand: str) -> dict:
    return {u: r for (u, s, c), r in src.items() if s == setting and c == cand}


def _subject(fold: str) -> str:
    return fold.split("-", 1)[1]


def _legs(out: Path, fold: str) -> dict:
    """The fold's users' source legs; every one must belong to the fold's held-out subject."""
    z = F._users(str(F._dir(out) / f"full-{fold}.npz"))
    legs = [str(x) for x in z["user_subject"]]
    return dict(l=sum(x.endswith("_l") for x in legs), r=sum(x.endswith("_r") for x in legs),
                other_subject=sum(x.rsplit("_", 1)[0] != _subject(fold) for x in legs))


def _mean_pref(v: dict, k: int) -> float:
    return float(np.mean([P._p(r, k) for r in v.values()]))


def _est(t) -> dict:
    return dict(zip(("est", "lo", "hi"), (float(x) for x in t)))


def _lead(a: dict, b: dict, k: int, rng) -> dict:
    """Paired lead of a over b at k, points: per subject, across subjects, pooled."""
    pairs = P._pairs(a, b)   # sorted by user
    for p in pairs:
        p["diff"] = 100 * (P._p(p["a"], k) - P._p(p["b"], k))

    def stat(x):
        return float(np.mean([p["diff"] for p in x]))

    by: dict[str, list] = {}
    for p in pairs:
        by.setdefault(p["fold"], []).append(p)
    per = {_subject(f): dict(_est(R._boot(ps, stat, rng)), n=len(ps)) for f, ps in sorted(by.items())}
    m = np.array([v["est"] for v in per.values()])
    sd = float(np.std(m, ddof=1))
    tc = float(student_t.ppf(0.975, len(m) - 1))
    half = tc * sd / np.sqrt(len(m))
    return dict(k=k, per_subject=per, n_positive=int(np.sum(m > 0)), n_subjects=len(m),
                subject_mean=dict(est=float(m.mean()), lo=float(m.mean() - half), hi=float(m.mean() + half),
                                  sd=sd, df=len(m) - 1, t_crit=tc),
                pooled=dict(_est(R._boot(pairs, stat, rng)), n=len(pairs)))


def _fs(t: dict, d: int = 1) -> str:
    return f"{t['est']:+.{d}f} [{t['lo']:+.{d}f}, {t['hi']:+.{d}f}]"


def _sections(text: str) -> dict:
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def _reported_full(sec: list, cand: str, k: int):
    pat = re.compile(rf"^- {re.escape(F.NAMES[cand])}: k = 24 ([+-]\d+\.\d) \[[^\]]*\], "
                     rf"k = 40 ([+-]\d+\.\d) ")
    for line in sec:
        m = pat.match(line)
        if m:
            return m.group(1 if k == 24 else 2)
    return None


def _reported_cmp_cell(sec: list, cand: str):
    """CMP's 'PsyPref − cell' cell (k = 24) for one cell, as printed."""
    for line in sec:
        if line.startswith(f"| {C.NAMES[cand]} |"):
            return [c.strip() for c in line.strip().strip("|").split("|")][3]
    return None


def _reported_fam(sec: list, fam_label: str, other: str):
    pat = re.compile(rf"^- {re.escape(fam_label)} − {re.escape(F.NAMES[other])}: ([+-]\d+\.\d) ")
    for line in sec:
        m = pat.match(line)
        if m:
            return m.group(1)
    return None


def report(out: Path) -> None:
    d = F._dir(out)
    full = _load(d / "full_sessions.jsonl")
    cmp_rows = _load(d / "cmp_sessions.jsonl")
    fam = _load(d / "fam_sessions.jsonl")
    folds = sorted({r["fold"] for r in full.values()})
    subjects = [_subject(f) for f in folds]
    legs = {_subject(f): _legs(out, f) for f in folds}
    if any(v["other_subject"] for v in legs.values()):
        raise SystemExit(f"a fold holds users of another subject: {legs}")
    rng = np.random.default_rng(AF.SEED_FULL + RNG_OFFSET)
    numbers = {"generated": datetime.datetime.now().isoformat(timespec="minutes"), "post_hoc": True,
               "sources": ["full_sessions.jsonl", "cmp_sessions.jsonl", "fam_sessions.jsonl"],
               "s_nm": AF.S, "lapse": AF.LAPSE, "jnd_nm": JND, "n_boot": R.N_BOOT,
               "rng_seed": AF.SEED_FULL + RNG_OFFSET, "subjects": subjects, "legs": legs,
               "settings": {}}

    for setting in F.SETTINGS:
        ps = _cell(full, setting, "psypref")
        pool = {c: _cell(full, setting, c) for c in C.PUBLISHED}
        pool.update({c: _cell(cmp_rows, setting, c) for c in C.CELLS})
        pool = {c: v for c, v in pool.items() if v}
        means = {k: {c: _mean_pref(v, k) for c, v in pool.items()} for k in KS}
        best = max(means[24], key=means[24].get)
        best_wo_eubo = max((c for c in means[24] if c != "gp_eubo"), key=means[24].get)
        best40 = max(means[40], key=means[40].get)
        ctx = {}
        for f in folds:
            us = {u: r for u, r in ps.items() if r["fold"] == f}
            d0 = float(np.median([r["d0"] for r in us.values()]))
            ctx[_subject(f)] = dict(n=len(us), d0_median_nm=d0, d0_median_jnd=d0 / JND,
                                    **{f"psypref_k{k}": _mean_pref(us, k) for k in KS})
        comps = {"best_published": (ps, pool[best]),
                 "gp_bald": (ps, _cell(full, setting, "gp_bald")),
                 "cmaes_sch4": (ps, _cell(full, setting, "cmaes_sch4")),
                 "cmaes_sch": (ps, _cell(full, setting, "cmaes_sch"))}
        if setting in FAM_SETTINGS:
            comps["franks_cmaes_sch4"] = (_cell(fam, setting, "proj_franks"), _cell(full, setting, "cmaes_sch4"))
        leads = {key: {k: _lead(a, b, k, rng) for k in KS} for key, (a, b) in comps.items()}
        if best40 != best:
            leads["strongest_published_k40"] = {40: _lead(ps, pool[best40], 40, rng)}
        numbers["settings"][setting] = dict(
            best_published=dict(choice=best, name=NAMES[best], rule="highest mean preference at k = 24, "
                                "FULL's Arens' GP, LineCoSpar, CMA-ES (ProMP) and the five CMP cells",
                                choice_without_gp_eubo=best_wo_eubo, strongest_k40=best40,
                                means={str(k): means[k] for k in KS}),
            subjects=ctx, leads={key: {str(k): v for k, v in by_k.items()} for key, by_k in leads.items()})

    # ---------------------------------------------------------------- self-check
    checks = []
    rng0 = np.random.default_rng(AF.SEED_FULL + 61)   # CMP's generator; its first draw is this cell
    t0 = P._diff(P._pairs(_cell(full, "medium", "psypref"), _cell(cmp_rows, "medium", "gp_eubo")), rng0)
    cmp_sec = _sections((d / "amp_cmp_report.md").read_text(encoding="utf-8"))
    rep = _reported_cmp_cell(cmp_sec["medium"], "gp_eubo")
    checks.append(dict(source="amp_cmp_report.md", setting="medium", quantity="PsyPref − GP-qEUBO, k = 24, "
                       "estimate and interval (CMP's generator redrawn)", reported=rep, here=R._f(t0, 1)))
    full_sec = _sections((d / "amp_full_report.md").read_text(encoding="utf-8"))
    fam_sec = _sections((d / "fam_report.md").read_text(encoding="utf-8"))
    for setting, s in numbers["settings"].items():
        for key in ("gp_bald", "cmaes_sch4", "cmaes_sch"):
            for k in KS:
                checks.append(dict(source="amp_full_report.md", setting=setting,
                                   quantity=f"{LABELS[key]}, k = {k}, pooled estimate",
                                   reported=_reported_full(full_sec[setting], key, k),
                                   here=f"{s['leads'][key][str(k)]['pooled']['est']:+.1f}"))
        best = s["best_published"]["choice"]
        est = s["leads"]["best_published"]["24"]["pooled"]["est"]
        if best in C.CELLS:
            cell = _reported_cmp_cell(cmp_sec[setting], best)
            checks.append(dict(source="amp_cmp_report.md", setting=setting,
                               quantity=f"PsyPref − {NAMES[best]}, k = 24, pooled estimate",
                               reported=None if cell is None else cell.split()[0], here=f"{est:.1f}"))
        else:
            checks.append(dict(source="amp_full_report.md", setting=setting,
                               quantity=f"PsyPref − {NAMES[best]}, k = 24, pooled estimate",
                               reported=_reported_full(full_sec[setting], best, 24), here=f"{est:+.1f}"))
        if "franks_cmaes_sch4" in s["leads"]:
            checks.append(dict(source="fam_report.md", setting=setting,
                               quantity=f"{LABELS['franks_cmaes_sch4']}, k = 24, pooled estimate",
                               reported=_reported_fam(fam_sec[setting], "Franks", "cmaes_sch4"),
                               here=f"{s['leads']['franks_cmaes_sch4']['24']['pooled']['est']:+.1f}"))
    for c in checks:
        c["match"] = c["reported"] == c["here"]
    gap = max(abs(v["subject_mean"]["est"] - v["pooled"]["est"]) for s in numbers["settings"].values()
              for by_k in s["leads"].values() for v in by_k.values())
    numbers["self_check"] = dict(checks=checks, all_match=all(c["match"] for c in checks),
                                 max_gap_subject_mean_vs_pooled_points=gap)

    # ---------------------------------------------------------------- summary
    summary = []
    for key in COMPARISONS:
        cells = [(setting, int(k), v) for setting, s in numbers["settings"].items()
                 for k, v in s["leads"].get(key, {}).items()]
        if not cells:
            continue
        per = [(v["per_subject"][subj]["est"], subj, setting, k) for setting, k, v in cells for subj in subjects]
        lo = min(per)
        t_above = sum(v["subject_mean"]["lo"] > 0 for _, _, v in cells)
        summary.append(f"- {LABELS[key]}: positive in {sum(e > 0 for e, *_ in per)} of {len(per)} subject "
                       f"cells ({len(cells)} setting × k cells × {len(subjects)} subjects); smallest "
                       f"{lo[0]:+.1f} ({lo[1]}, {lo[2]}, k = {lo[3]}); t-interval above 0 in {t_above} of "
                       f"{len(cells)} setting × k cells.")
    numbers["summary"] = summary

    # ---------------------------------------------------------------- markdown
    leg_text = ", ".join(f"{x} {legs[x]['l']} l / {legs[x]['r']} r legs" for x in subjects)
    L = ["# Post hoc: PsyPref's leads per walking subject", "",
         "Post hoc; no planned verdict is changed. Stored rows of FULL "
         f"({len(full)} sessions), CMP ({len(cmp_rows)}) and FAM's projection ({len(fam)}); nothing is rerun. "
         f"FULL's 200 users per setting are 40 per LOSO fold, all from the fold's held-out subject "
         f"({leg_text}). Lead: mean paired "
         f"difference in preference over generic assistance, λ/2 + (1 − λ) Φ((d0 − d_k)/s), s = {AF.S} Nm, "
         f"λ = {AF.LAPSE}, in points. Per subject: 95 % interval from {R.N_BOOT} draws of the subject's "
         f"users. Across subjects (descriptive): mean of the five subject means ± t(0.975, df = 4) × SD / √5; "
         f"with 40 users per subject this mean equals the pooled lead. Pooled: {R.N_BOOT} draws, users "
         "resampled within subjects (the reports' intervals; Monte Carlo differences in the last digit from "
         "the source reports are expected). Best published (design note 23): the highest mean preference at "
         "k = 24 over all 200 users among FULL's Arens' GP, LineCoSpar and CMA-ES (ProMP) and the five CMP "
         "cells, kept at k = 40.", "",
         "## Summary (post hoc, descriptive)", ""] + summary + [""]
    for setting, s in numbers["settings"].items():
        bp = s["best_published"]
        L += [f"## {setting}", "",
              f"Best published at k = 24: {bp['name']} ({bp['means']['24'][bp['choice']]:.3f}; without GP-qEUBO "
              f"in the pool: {NAMES[bp['choice_without_gp_eubo']]}); strongest at k = 40: "
              f"{NAMES[bp['strongest_k40']]} ({bp['means']['40'][bp['strongest_k40']]:.3f}).", "",
              "| subject | users | generic's median distance (Nm; JND) | PsyPref, k = 24 | PsyPref, k = 40 |",
              "|---|---|---|---|---|"]
        for subj, c in s["subjects"].items():
            L.append(f"| {subj} | {c['n']} | {c['d0_median_nm']:.2f}; {c['d0_median_jnd']:.2f} | "
                     f"{c['psypref_k24']:.3f} | {c['psypref_k40']:.3f} |")
        L += ["", "| lead (points) | k | " + " | ".join(subjects) + " | subjects > 0 | mean of subjects, "
              "t (df = 4) | pooled |", "|---|---|" + "---|" * len(subjects) + "---|---|---|"]
        for key, by_k in s["leads"].items():
            label = LABELS[key] + (f" ({bp['name']})" if key == "best_published" else "")
            if key == "strongest_published_k40":
                label += f" ({NAMES[bp['strongest_k40']]}, sensitivity)"
            for k, v in by_k.items():
                L.append(f"| {label} | {k} | " + " | ".join(_fs(v["per_subject"][x]) for x in subjects)
                         + f" | {v['n_positive']} of {v['n_subjects']} | {_fs(v['subject_mean'])} | "
                         f"{_fs(v['pooled'])} |")
        L.append("")
    sc = numbers["self_check"]
    L += ["## Self-check against already reported numbers", "",
          f"All match: {sc['all_match']}. Largest gap between the mean of subject means and the pooled "
          f"lead: {sc['max_gap_subject_mean_vs_pooled_points']:.1e} points.", "",
          "| source | setting | quantity | reported | here | match |", "|---|---|---|---|---|---|"]
    for c in sc["checks"]:
        L.append(f"| {c['source']} | {c['setting']} | {c['quantity']} | {c['reported']} | {c['here']} | "
                 f"{'yes' if c['match'] else 'NO'} |")
    (d / "amp_subjects.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (d / "amp_subjects.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
