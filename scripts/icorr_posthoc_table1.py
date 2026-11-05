"""Post hoc: Table I of the paper, the main condition in the three settings.

Usage:
  python scripts/icorr_posthoc_table1.py

Not planned. A tabulation of stored rows made after the planned verdicts were judged; it
changes and re-judges none of them. FULL's 200 users, s = 1.0 Nm, lambda = 0.1,
in medium (3-12 Nm), wide (3-35 Nm) and fixed 12 Nm, at k = 0, 12, 24 and 40. Rows:
  - PsyPref and GP-BALD: FULL's rows (full_sessions.jsonl);
  - the strongest other optimizer optimizer per setting, by test CMP's Consequence (section 23):
    FULL's strongest of Arens' GP, LineCoSpar and CMA-ES on the ProMP by mean preference at
    k = 24, unless CMP cells (cmp_sessions.jsonl) exceed it there; then the highest of those;
  - CMA-ES on Schaefer's family with pycma's default population (cmaes_sch) and tuned, FULL's rows; the stronger at
    k = 24 is marked;
  - the best-profile level: the mean preference at each user's ProMP-8 best-profile distance (FAM's definition, section 19).
    Wide and fixed 12: fam_floors.jsonl, family promp8. Medium, which FAM did not fit:
    AF.promp_floor(ideal, 8, peak_range=(3, 12)) for all 200 users, the preferred profile built
    by FAM's own _user (as FULL's run_one builds it).
k = 0 is the preference of an optimizer's starting curve, computed from its space without a
session: z = 0 for the optimizers in FULL's ProMP space (PsyPref's prior-mean curve, GP-BALD's
identified assistance before any answer, CMA-ES's initial mean, LineCoSpar's start, candidate 0 of
Arens' GP with a start at the prior's mean), and z = 0 of Schaefer's space for CMA-ES on Schaefer's family
(its initial mean: the fold's timing-prior mean with both peaks at the range's midpoint, or at
12 Nm). Arens' GP with Algorithm S1's random start and Ramella et al.'s prior batch have no
defined start curve (n/a).

Preference over generic assistance: lam/2 + (1 - lam) Phi((d0 - dk) / s); generic assistance
itself scores 0.5. PsyPref minus each row at k = 12, 24, 40 (points, paired by user), and the
first k at which PsyPref reaches the row's k = 40 level (FULL's rule, searched over k = 1-40;
41: not within 40). 95 % intervals from 2000 draws, users resampled within folds
(icorr_amp_range._boot), rows sorted by user so the intervals do not depend on file order.

Self-checks, every reported number parsed from its report: FULL's estimates (amp_full_report.md),
the chosen cells' estimates and margins (amp_cmp_report.md, amp_cmp_firstk.md), FAM's ProMP-8
best-profile levels (fam_report.md), and CMP's first interval redrawn with CMP's own generator; each CMA-ES
row's stored distance at k = 1 against its k = 0 start (CMA-ES delivers its initial mean until its
first generation is sorted); the pairing
of every row with PsyPref's (users, answer seeds, peaks, d0); d0 recomputed from the preferred
curves; the ProMP-8 best-profile distances refit in wide and fixed 12 against fam_floors.jsonl, and in medium
against LONG's (long_floors.jsonl, users 0-19 per fold).
Outputs in <out>/icorr/amp_free/: amp_table1.md, amp_table1.json.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_cmp as C  # noqa: E402
import icorr_amp_fam as FAM  # noqa: E402
import icorr_amp_full as F  # noqa: E402
import icorr_amp_pri as P  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.simulated_user import rms_distance  # noqa: E402

JND = true_jnd(AF.S)   # 0.674 Nm
KS = (0, 12, 24, 40)
NAMES = {**F.NAMES, **C.NAMES}
#: How each optimizer's k = 0 curve is derived ("z0": z = 0 of FULL's ProMP space; "sch0": z = 0
#: of Schaefer's space), with the documented reason; None: no defined start curve.
START = {
    "psypref": ("z0", "z = 0 of FULL's space, the prior-mean curve"),
    "gp_bald": ("z0", "z = 0, its identified assistance before any answer (gp_eubo.best)"),
    "gp_eubo": ("z0", "z = 0, its identified assistance before any answer (gp_eubo.best)"),
    "cmaes": ("z0", "z = 0, CMA-ES's initial mean (sortcma x0)"),
    "cmaes4": ("z0", "z = 0, CMA-ES's initial mean (sortcma x0)"),
    "d_z": ("z0", "z = 0, LineCoSpar's start (WhitenedLearner.linecospar)"),
    "g_generic": ("z0", "z = 0, candidate 0 (section 23's generic start)"),
    "g_generic_nogate": ("z0", "z = 0, candidate 0 (section 23's generic start)"),
    "cmaes_sch": ("sch0", "z = 0 of Schäfer's space, CMA-ES's initial mean"),
    "cmaes_sch4": ("sch0", "z = 0 of Schäfer's space, CMA-ES's initial mean"),
    "g": (None, "starts at a random prior draw (Algorithm S1's random first duel)"),
    "ramella": (None, "starts at a member of a batch drawn from the prior"),
}


def _load(path: Path, key=F._key) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("error") is None:
            rows[key(r)] = r
    return rows


def _cell(src: dict, setting: str, cand: str) -> dict:
    return {k[0]: r for k, r in src.items() if k[1:] == (setting, cand)}


def _pref(d0, dk):
    return AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((np.asarray(d0) - np.asarray(dk)) / AF.S)


def _t(t) -> dict:
    return dict(zip(("est", "lo", "hi"), (float(v) for v in t)))


def _pts(t) -> str:
    return f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"


# ----------------------------------------------------------------------------- geometry

def _geometry(out: Path) -> tuple[dict, dict]:
    """Per (user, setting): d0 from the rebuilt curves, the start distances, the start curve's gap
    to generic assistance, and the ProMP-8 best-profile distance refit (FAM's _user and _kw). Per (fold, setting):
    the start curves' peaks."""
    run_dir = out / "phase5-schaefer"
    geo, curves, peaks = {}, {}, {}
    for job in F.jobs(out):
        if job["candidate"] != "psypref":
            continue
        setting = job["setting"]
        ideal, generic, d = FAM._user(job)
        key = (job["fold"], setting)
        if key not in curves:
            if setting in F.RANGES:
                lo, hi = F.RANGES[setting]
                mu, sigma = R._a_moments(job["_parts"], job["fold_index"], setting)
                spaces = dict(z0=AF.promp_space(mu, sigma, limit=hi),
                              sch0=AF.schaefer_range_space(run_dir, job["fold"], lo, hi))
            else:
                spaces = dict(z0=AF.fixed_level_space(d["w_minus"], d["w_plus"], F.LEVEL),
                              sch0=AF.schaefer_fixed_space(run_dir, job["fold"], F.LEVEL))
            curves[key] = {k: sp.design.curve(sp.to_theta(np.zeros(sp.d))) for k, sp in spaces.items()}
            peaks[key] = {k: [float(v) for v in AF.peaks(c)] for k, c in curves[key].items()}
        cs = curves[key]
        geo[job["user"], setting] = dict(
            d0=rms_distance(generic, ideal),
            start={k: rms_distance(c, ideal) for k, c in cs.items()},
            z0_gap=rms_distance(cs["z0"], generic),
            floor=float(AF.promp_floor(ideal, 8, **FAM._kw(setting))))
    return geo, peaks


# ----------------------------------------------------------------------------- report parsing

def _sections(md: str) -> dict:
    parts = re.split(r"^## (.+)$", md, flags=re.M)
    return {t.strip(): b for t, b in zip(parts[1::2], parts[2::2])}


def _table(text: str, ncol_key: int = 1) -> dict:
    """The first table row per key (the first ``ncol_key`` cells) of every table in ``text``."""
    rows = {}
    for line in text.splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.setdefault(tuple(cols[:ncol_key]) if ncol_key > 1 else cols[0], cols)
    return rows


def _tok(s: str) -> str:
    return s.split()[0]


# ----------------------------------------------------------------------------- report

def report(out: Path) -> None:
    d = F._dir(out)
    full = _load(d / "full_sessions.jsonl")
    cmp_ = _load(d / "cmp_sessions.jsonl")
    fam = _load(d / "fam_floors.jsonl", FAM._floor_key)
    long_fl = _load(d / "long_floors.jsonl", lambda r: (r["user"], r["setting"]))
    md_full = _sections((d / "amp_full_report.md").read_text(encoding="utf-8"))
    md_cmp = _sections((d / "amp_cmp_report.md").read_text(encoding="utf-8"))
    md_fam = _sections((d / "fam_report.md").read_text(encoding="utf-8"))
    md_firstk = _table((d / "amp_cmp_firstk.md").read_text(encoding="utf-8"), ncol_key=2)
    geo, start_peaks = _geometry(out)
    rng = np.random.default_rng(AF.SEED_FULL + 81)
    checks: list[dict] = []

    def check(name: str, computed: str, reported: str | None) -> None:
        checks.append(dict(check=name, computed=computed, reported=reported,
                           match=reported is not None and computed == reported))

    numbers = {"generated": datetime.datetime.now().isoformat(timespec="minutes"), "post_hoc": True,
               "registered": False, "jnd_nm": JND, "s_nm": AF.S, "lapse": AF.LAPSE,
               "n_boot": R.N_BOOT, "ks": list(KS), "settings": {}}
    generic_score = float(_pref(0.0, 0.0))
    L = ["# Table I, post hoc: preference over generic assistance in the main condition", "",
         "**Post hoc.** A tabulation of stored rows (full_sessions.jsonl, "
         "cmp_sessions.jsonl, fam_floors.jsonl) and, in medium, of ProMP-8 floors fitted here; no "
         "session was run, and no planned verdict (FULL, CMP, FAM) is changed or re-judged. "
         f"FULL's 200 users per setting, s = {AF.S} Nm, λ = {AF.LAPSE}. Preference over generic "
         "assistance λ/2 + (1 − λ)Φ((d₀ − d_k)/s); generic assistance itself scores "
         f"{generic_score:.3f} at every k. JND {JND:.3f} Nm. 95 % intervals from {R.N_BOOT} draws, "
         "users resampled within folds, rows sorted by user.", ""]
    paper: dict[str, dict] = {}
    details: list[str] = []

    for setting in F.SETTINGS:
        S = numbers["settings"][setting] = {}
        ps = _cell(full, setting, "psypref")
        users = sorted(ps)
        S["n_users"] = len(users)
        mean24 = lambda v: float(np.mean([P._p(r, 24) for r in v.values()]))  # noqa: E731

        # -- strongest other optimizer, section 23's Consequence (mean preference at k = 24)
        pub_vals = {c: mean24(_cell(full, setting, c)) for c in C.PUBLISHED}
        pub = max(pub_vals, key=pub_vals.get)
        cell_vals = {c: mean24(_cell(cmp_, setting, c)) for c in C.CELLS}
        above = {c: v for c, v in cell_vals.items() if v > pub_vals[pub]}
        best = max(above, key=above.get) if above else pub
        best_src = cmp_ if above else full
        S["best_published"] = dict(candidate=best, label=NAMES[best], source="CMP" if above else "FULL",
                                   full_published_k24=pub_vals, full_best=pub,
                                   cmp_cells_k24=cell_vals, cmp_cells_above=sorted(above))
        sch = {c: mean24(_cell(full, setting, c)) for c in ("cmaes_sch", "cmaes_sch4")}
        stronger = max(sch, key=sch.get)
        S["schaefer_stronger_k24"] = dict(candidate=stronger, k24=sch)

        # -- per-user preference traces, index k = 0..40
        def trace(r, cand):
            kind = START[cand][0]
            s0 = geo[r["user"], setting]["start"][kind] if kind else np.nan
            return _pref(r["d0"], np.concatenate([[s0], r["distances"][:F.N_COMP]]))

        rows = {"psypref": (ps, "psypref"), "gp_bald": (_cell(full, setting, "gp_bald"), "gp_bald"),
                "best": (_cell(best_src, setting, best), best),
                "cmaes_sch": (_cell(full, setting, "cmaes_sch"), "cmaes_sch"),
                "cmaes_sch4": (_cell(full, setting, "cmaes_sch4"), "cmaes_sch4")}
        traces = {name: {u: dict(fold=r["fold"], p=trace(r, cand)) for u, r in src.items()}
                  for name, (src, cand) in rows.items()}
        if setting in F.RANGES and setting not in FAM.SETTINGS:
            floors = {u: geo[u, setting]["floor"] for u in users}
            ceil_src = "ProMP-8 floors fitted here (AF.promp_floor, peak range "f"{F.RANGES[setting]})"
        else:
            floors = {u: fam[u, setting, "promp8"]["floor"] for u in users if (u, setting, "promp8") in fam}
            ceil_src = "fam_floors.jsonl, family promp8"
        traces["ceiling"] = {u: dict(fold=ps[u]["fold"], p=np.full(F.N_COMP + 1, _pref(ps[u]["d0"], f)))
                             for u, f in floors.items()}

        # -- pairing with PsyPref's rows
        pairing = {}
        for name, (src, cand) in rows.items():
            if name == "psypref":
                continue
            common = sorted(set(ps) & set(src))
            pairing[name] = dict(row=cand, n=len(src), paired=len(common), same_users=set(ps) == set(src),
                                 answer_seed_diff=sum(ps[u]["answer_seed"] != src[u]["answer_seed"] for u in common),
                                 amps_diff=sum(ps[u]["amps"] != src[u]["amps"] for u in common),
                                 d0_diff=sum(ps[u]["d0"] != src[u]["d0"] for u in common),
                                 fold_diff=sum(ps[u]["fold"] != src[u]["fold"] for u in common))
        if setting in FAM.SETTINGS:
            fr = {u: fam[u, setting, "promp8"] for u in floors}
            pairing["ceiling"] = dict(row="fam_floors promp8", n=len(fr), paired=len(fr),
                                      same_users=set(fr) == set(ps),
                                      answer_seed_diff=sum(ps[u]["answer_seed"] != fr[u]["answer_seed"] for u in fr),
                                      amps_diff=sum(ps[u]["amps"] != fr[u]["amps"] for u in fr),
                                      d0_diff=sum(ps[u]["d0"] != fr[u]["d0"] for u in fr))
        else:
            pairing["ceiling"] = dict(row="floors fitted here", n=len(floors), paired=len(floors),
                                      same_users=set(floors) == set(ps))
        S["pairing"] = pairing
        d0_gap = max(abs(geo[u, setting]["d0"] - ps[u]["d0"]) for u in users)
        S["d0_rebuilt_max_abs_diff_nm"] = d0_gap

        # -- best-profile distances: refit against stored ones
        refit = {}
        if setting in FAM.SETTINGS:
            diffs = [abs(geo[u, setting]["floor"] - fam[u, setting, "promp8"]["floor"]) for u in floors]
            refit = dict(against="fam_floors.jsonl", n=len(diffs), max_abs_diff_nm=max(diffs))
        lf = [abs(geo[u, setting]["floor"] - long_fl[u, setting]["fam_promp8"])
              for u in users if (u, setting) in long_fl and long_fl[u, setting].get("fam_promp8") is not None]
        if lf:
            refit["long_floors"] = dict(n=len(lf), max_abs_diff_nm=max(lf))
        S["floor_refit"] = refit
        fl_arr = np.array([floors[u] for u in users])
        S["ceiling_source"] = ceil_src
        S["median_floor_nm"] = float(np.median(fl_arr))
        S["median_floor_jnd"] = float(np.median(fl_arr) / JND)
        if setting not in FAM.SETTINGS:
            S["promp8_floors_nm"] = {u: floors[u] for u in users}

        # -- levels per row and k (k = 0 shared by rows with the same start)
        def items(name):
            return [dict(fold=v["fold"], p=v["p"]) for _, v in sorted(traces[name].items())]

        levels, k0_cache = {}, {}
        for name in ("psypref", "gp_bald", "best", "cmaes_sch", "cmaes_sch4"):
            cand = rows[name][1]
            kind = START[cand][0]
            lv = {}
            for k in KS:
                if k == 0:
                    if kind is None:
                        lv[0] = None
                        continue
                    if kind not in k0_cache:
                        k0_cache[kind] = R._boot(items(name), lambda x: float(np.mean([i["p"][0] for i in x])), rng)
                    lv[0] = k0_cache[kind]
                else:
                    lv[k] = R._boot(items(name), lambda x, k=k: float(np.mean([i["p"][k] for i in x])), rng)
            levels[name] = lv
        c = R._boot(items("ceiling"), lambda x: float(np.mean([i["p"][0] for i in x])), rng)
        levels["ceiling"] = {k: c for k in KS}

        # -- PsyPref minus each row, paired, and the first k
        minus, first = {}, {}
        a = traces["psypref"]
        for name in ("gp_bald", "best", "cmaes_sch", "cmaes_sch4", "ceiling"):
            b = traces[name]
            common = sorted(set(a) & set(b))
            pr = [dict(fold=a[u]["fold"], dp=a[u]["p"] - b[u]["p"]) for u in common]
            minus[name] = {k: R._boot(pr, lambda x, k=k: float(np.mean([i["dp"][k] for i in x])), rng)
                           for k in (12, 24, 40)}
            pp = [dict(fold=a[u]["fold"], a=a[u]["p"], b=b[u]["p"]) for u in common]

            def reach(x):
                target = np.mean(np.array([i["b"] for i in x]), axis=0)[F.N_COMP]
                hit = np.nonzero(np.mean(np.array([i["a"] for i in x]), axis=0)[1:] >= target)[0]
                return float(hit[0] + 1) if len(hit) else float(F.N_COMP + 1)
            first[name] = R._boot(pp, reach, rng)

        # -- labels
        label = {"psypref": "PsyPref", "gp_bald": "GP-BALD",
                 "best": f"Best published: {NAMES[best]}" + (" (CMP)" if above else " (FULL)"),
                 "cmaes_sch": "CMA-ES (Schäfer, default population)",
                 "cmaes_sch4": "CMA-ES (Schäfer, tuned; planned, 16a)",
                 "ceiling": "Ceiling: preference at the ProMP-8 floor (any k)"}
        label[stronger] += " †"
        S["rows"] = {}
        for name, lv in levels.items():
            cand = rows[name][1] if name in rows else None
            e = dict(label=label[name], candidate=cand, n=len(traces[name]),
                     k={str(k): (None if v is None else _t(v)) for k, v in lv.items()})
            if cand is not None:
                e["start"] = START[cand][1]
            S["rows"][name] = e
        S["psypref_minus_points"] = {n: {str(k): _t(tuple(100 * v for v in t)) for k, t in m.items()}
                                     for n, m in minus.items()}
        S["first_k_reaching_k40_level"] = {n: _t(t) for n, t in first.items()}
        S["start_peaks_nm"] = {kind: {f: start_peaks[f, setting][kind] for f, s in start_peaks if s == setting}
                               for kind in ("z0", "sch0")}
        gaps = [geo[u, setting]["z0_gap"] for u in users]
        S["z0_gap_to_generic_nm"] = dict(median=float(np.median(gaps)), min=float(np.min(gaps)),
                                         max=float(np.max(gaps)))
        paper[setting] = dict(levels=levels, label=label, best=best, above=bool(above))

        # -- self-checks against the reports (point estimates; strings as printed)
        ft = _table(md_full.get(setting, ""))
        for name in ("psypref", "gp_bald", "cmaes_sch", "cmaes_sch4") + (("best",) if not above else ()):
            rep = ft.get(F.NAMES[rows[name][1]])
            for i, k in enumerate((12, 24, 40)):
                check(f"FULL {setting} {F.NAMES[rows[name][1]]} k = {k}", f"{levels[name][k][0]:.3f}",
                      _tok(rep[2 + i]) if rep else None)
        for name in ("gp_bald", "cmaes_sch", "cmaes_sch4") + (("best",) if not above else ()):
            nm = re.escape(F.NAMES[rows[name][1]])
            m = re.search(rf"^- {nm}: k = 24 ([+-]\d+\.\d) \[.*?\], k = 40 ([+-]\d+\.\d) \[.*?\]; reaches its "
                          rf"k = 40 level at k = (\d+) ", md_full.get(setting, ""), flags=re.M)
            for k, g in ((24, 1), (40, 2)):
                check(f"FULL {setting} PsyPref − {F.NAMES[rows[name][1]]} k = {k}",
                      f"{100 * minus[name][k][0]:+.1f}", m.group(g) if m else None)
            check(f"FULL {setting} first k reaching {F.NAMES[rows[name][1]]}'s k = 40 level",
                  f"{first[name][0]:.0f}", m.group(3) if m else None)
        if above:
            ct = _table(md_cmp.get(setting, ""))
            rep = ct.get(C.NAMES[best])
            check(f"CMP {setting} {C.NAMES[best]} preference k = 24", f"{levels['best'][24][0]:.3f}",
                  _tok(rep[2]) if rep else None)
            check(f"CMP {setting} PsyPref − {C.NAMES[best]} k = 24 (points)", f"{100 * minus['best'][24][0]:.1f}",
                  _tok(rep[3]) if rep else None)
            rep = md_firstk.get((setting, C.NAMES[best]))
            for i, k in enumerate((12, 24, 40)):
                check(f"CMP first-k table {setting} {C.NAMES[best]} k = {k}", f"{levels['best'][k][0]:.3f}",
                      rep[2 + i] if rep else None)
            check(f"CMP first-k table {setting} first k reaching {C.NAMES[best]}'s k = 40 level",
                  f"{first['best'][0]:.0f}", _tok(rep[5]) if rep else None)
        if setting in FAM.SETTINGS:
            rep = _table(md_fam.get(setting, "")).get("ProMP 8")
            check(f"FAM {setting} ProMP 8 ceiling", f"{c[0]:.3f}", _tok(rep[3]) if rep else None)
        # k = 0 against the rows: CMA-ES delivers its initial mean until its first generation is
        # sorted (at least three answers), so its stored distance at k = 1 is the start's distance
        for name, (src, cand) in rows.items():
            if cand.startswith("cmaes"):
                kind = START[cand][0]
                same = sum(abs(r["distances"][0] - geo[u, setting]["start"][kind]) < 1e-9 for u, r in src.items())
                check(f"{setting} {NAMES[cand]}: k = 0 start = stored curve at k = 1 (users)", f"{same}/{len(src)}",
                      f"{len(src)}/{len(src)}")

    # CMP's first interval, redrawn with CMP's generator in CMP's order (its first draw)
    rng_cmp = np.random.default_rng(AF.SEED_FULL + 61)
    dp = P._diff(P._pairs(_cell(full, "medium", "psypref"), _cell(cmp_, "medium", C.CELLS[0])), rng_cmp)
    rep = _table(md_cmp.get("medium", "")).get(C.NAMES[C.CELLS[0]])
    check(f"CMP medium PsyPref − {C.NAMES[C.CELLS[0]]} with its interval, CMP's generator", R._f(dp, 1),
          rep[3] if rep else None)
    numbers["self_checks"] = checks

    # ---------------------------------------------------------------- markdown
    order = ("psypref", "gp_bald", "best", "cmaes_sch", "cmaes_sch4", "ceiling")
    plain = {"psypref": "PsyPref", "gp_bald": "GP-BALD", "best": "Best published ‡",
             "cmaes_sch": "CMA-ES (Schäfer, default population)", "cmaes_sch4": "CMA-ES (Schäfer, tuned)",
             "ceiling": "Ceiling (ProMP-8 floor)"}
    head = " | ".join(f"{s[:6] if s != 'fixed 12' else 'fixed'} k = {k}" for s in F.SETTINGS for k in KS)
    L += ["## Table I (point estimates)", "", f"| optimizer | {head} |", "|---|" + "---|" * (3 * len(KS))]
    for name in order:
        cells = []
        for s in F.SETTINGS:
            lv = paper[s]["levels"][name]
            mark = " †" if paper[s]["label"][name].endswith("†") else ""
            cells += [("n/a" if lv[k] is None else f"{lv[k][0]:.3f}") + (mark if k == 24 else "") for k in KS]
        L.append(f"| {plain[name]} | " + " | ".join(cells) + " |")
    L += ["", "‡ Best published per setting (section 23's Consequence): " + "; ".join(
        f"{s}: {NAMES[paper[s]['best']]} ({'CMP' if paper[s]['above'] else 'FULL'})" for s in F.SETTINGS) + ".",
        "† The stronger CMA-ES on Schäfer's family at k = 24 in that setting (post hoc mark; the "
        "planned comparator is the tuned one, change 16a).",
        "k = 0: the start curve's preference (see each setting); generic assistance is "
        f"{generic_score:.3f}. The ceiling does not depend on k.", ""]

    for setting in F.SETTINGS:
        S = numbers["settings"][setting]
        bp = S["best_published"]
        lv = paper[setting]["levels"]
        L += [f"## {setting}", ""]
        above = bp["cmp_cells_above"]
        L.append(f"Best published (section 23's Consequence, mean preference at k = 24): FULL's strongest "
                 f"published is {NAMES[bp['full_best']]} ({bp['full_published_k24'][bp['full_best']]:.4f}); "
                 + ("CMP cells above it: " + ", ".join(
                     f"{NAMES[c]} {bp['cmp_cells_k24'][c]:.4f}" for c in sorted(above, key=lambda c: -bp['cmp_cells_k24'][c]))
                    + f". Chosen: **{NAMES[bp['candidate']]}**." if above else
                    "no CMP cell exceeds it, so it stays."))
        if len(above) > 1:
            ranked = sorted(above, key=lambda c: -bp["cmp_cells_k24"][c])
            gap = 100 * (bp["cmp_cells_k24"][ranked[0]] - bp["cmp_cells_k24"][ranked[1]])
            L.append(f"Runner-up {NAMES[ranked[1]]}, {gap:.2f} points lower (point estimates).")
        m = S["psypref_minus_points"]["best"]["24"]
        L += [f"PsyPref − {NAMES[bp['candidate']]} at k = 24: {m['est']:+.1f} [{m['lo']:+.1f}, {m['hi']:+.1f}] points.", ""]
        L += ["| row | n | k = 0 | k = 12 | k = 24 | k = 40 |", "|---|---|---|---|---|---|"]
        for name in order:
            e = S["rows"][name]
            L.append(f"| {e['label']} | {e['n']} | " + " | ".join(
                "n/a" if lv[name][k] is None else R._f(lv[name][k], 3) for k in KS) + " |")
        L += ["", "PsyPref minus each row (points, paired by user) and the first k at which PsyPref reaches the "
              f"row's k = 40 level ({F.N_COMP + 1}: not within {F.N_COMP}; for the ceiling, its level):", "",
              "| row | k = 12 | k = 24 | k = 40 | PsyPref reaches its k = 40 level at |", "|---|---|---|---|---|"]
        for name in order[1:]:
            mm, f = S["psypref_minus_points"][name], S["first_k_reaching_k40_level"][name]
            L.append(f"| {S['rows'][name]['label']} | " + " | ".join(
                f"{mm[str(k)]['est']:+.1f} [{mm[str(k)]['lo']:+.1f}, {mm[str(k)]['hi']:+.1f}]" for k in (12, 24, 40))
                + f" | {f['est']:.0f} [{f['lo']:.0f}, {f['hi']:.0f}] |")
        zp = S["start_peaks_nm"]
        z0_lo = [min(v[b] for v in zp["z0"].values()) for b in (0, 1)]
        z0_hi = [max(v[b] for v in zp["z0"].values()) for b in (0, 1)]
        s0 = next(iter(zp["sch0"].values()))
        g = S["z0_gap_to_generic_nm"]
        L += ["", "k = 0 starts: PsyPref, GP-BALD and the best published start at z = 0 of FULL's space "
              f"(one curve per fold; peaks extension {z0_lo[0]:.2f}–{z0_hi[0]:.2f}, flexion {z0_lo[1]:.2f}–"
              f"{z0_hi[1]:.2f} Nm over folds; RMS gap to generic assistance median {g['median']:.2f} Nm, "
              f"{g['min']:.2f}–{g['max']:.2f}); their k = 0 values are one computation. Both CMA-ES on "
              f"Schäfer's family start at z = 0 of Schäfer's space (peaks {s0[0]:.2f} / {s0[1]:.2f} Nm).",
              f"Ceiling: {S['ceiling_source']}; median floor {S['median_floor_nm']:.2f} Nm "
              f"({S['median_floor_jnd']:.2f} JND).", ""]
        pr = S["pairing"]
        bad = {n: p for n, p in pr.items() if not p["same_users"] or any(
            p.get(x, 0) for x in ("answer_seed_diff", "amps_diff", "d0_diff", "fold_diff"))}
        L.append("Pairing with PsyPref's rows: " + ("every row has PsyPref's " + f"{S['n_users']} users"
                 + "; answer seeds, peaks and d0 identical where stored (CMP and FULL rows, FAM's floors)."
                 if not bad else f"MISMATCH {bad}"))
        rf = S["floor_refit"]
        txt = []
        if "against" in rf:
            txt.append(f"refit against fam_floors.jsonl: largest difference {rf['max_abs_diff_nm']:.1e} Nm "
                       f"over {rf['n']} users")
        if "long_floors" in rf:
            txt.append(f"against LONG's FAM ProMP-8 floors: largest difference "
                       f"{rf['long_floors']['max_abs_diff_nm']:.1e} Nm over {rf['long_floors']['n']} users")
        L += [f"ProMP-8 floors: {'; '.join(txt)}. d0 rebuilt from the preferred curves: largest difference "
              f"{S['d0_rebuilt_max_abs_diff_nm']:.1e} Nm.", ""]

    n_ok = sum(c["match"] for c in checks)
    L += ["## Self-checks against the reports", "",
          f"{n_ok} of {len(checks)} checks reproduced at their printed digits (point estimates; the "
          "intervals here are drawn afresh, except the last check, which redraws CMP's first interval "
          "with CMP's own generator). The k = 0 checks compare per user, to 1e-9 Nm, the computed start "
          "with the stored CMA-ES distance at k = 1.", "", "| check | here | reported | match |",
          "|---|---|---|---|"]
    L += [f"| {c['check']} | {c['computed']} | {c['reported']} | {'yes' if c['match'] else 'NO'} |" for c in checks]
    L += ["", "## Definitions chosen here", "",
          "- Best published: section 23's Consequence applied to point estimates of the mean preference at "
          "k = 24 (CMP's planned k); when several CMP cells exceed FULL's best published, the highest is "
          "taken.",
          "- k = 0: the curve each optimizer would deliver before any answer, from its documented start; "
          "z = 0 is the same curve for every optimizer in FULL's space, so their k = 0 values coincide.",
          "- The first k is searched over k = 1–40 (FULL's rule), not k = 0.",
          "- Ceiling: one value per setting, shown at every k; PsyPref minus the ceiling is paired by user."]
    (d / "amp_table1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (d / "amp_table1.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
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
