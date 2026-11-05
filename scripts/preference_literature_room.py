"""Literature-informed room for personalisation.

Usage:
    python scripts/preference_literature_room.py [--out <project>/out]
        [--data <Schaefer share>] [--boot 2000]

Writes ``OUT/schaefer_grounding/literature_room.md`` (OUT is git-ignored; only aggregate numbers).

What it does:
a. converts each hip timing spread of the literature (SD in %GC) into RMS Nm with G2's
   landmark-shift conversion (schaefer_grounding.landmark_shift_rms on each of the 11 Schaefer
   analytic preferred profiles, median over participants, x kappa). Mapping: extension peak ->
   t2, extension rise -> t2 - t1, flexion peak -> t4. A rise time is converted by moving the
   onset t1 with the peak t2 held, because every source holds its peak time as a separate
   parameter. Any other parameter is listed, not converted.
b. converts each between-person CV of preferred magnitude into Nm as CV x median RMS of the 200
   simulated population ideals (recomputed here from load_sim, 4.816 Nm).
c. deflates by within-person scatter where both are given, sqrt(max(b^2 - w^2, 0)), and prints beside it the variant for a between-SD of trial means.
d. forms a low, a central and a high literature room as the RMS sum of a timing and a magnitude
   component; every choice is printed.
e. rescales Schaefer's amplitude-matched room distribution (b) so that its median equals each
   room level and applies the G1b KDE reweighting to every optimizer of the main cell (k = 24),
   with joint bootstrap intervals (simulated users within folds and participants, 2000 draws)
   and paired differences A (lens on Franks) minus each comparator.
f. gives the verdicts of P7a-P7c by the design's held / failed / uncertain rule.

The draw construction (seed 20261009, sim_draws then part_draws) is the one of
schaefer_grounding_optimizers.py, so the anchor row (Schaefer timing-only room, unrescaled)
must reproduce out/schaefer_grounding/optimizers_real_room.md; the script checks this.

Literature values below come from the extracted study records with their corrections applied
(docs/preference_studies_2026-10-08.md holds the full table with every tag).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import schaefer_grounding as G  # noqa: E402

N_BOOT = 2000
K = 24
SEED_SHARED = 20261009  # schaefer_grounding_optimizers.py, so the anchor reproduces
SEED_OWN = 20261010     # own stream: room-level bootstrap and Monte Carlo
SIM_ROOM_NM = 2.31      # planned simulated room

#: Planned predictions, section 7 (never edited after the run).
P7A_RATIO = 1.0
P7B_LOW, P7B_HIGH = 0.13, 0.30
P7C_ROOM_LOW, P7C_ROOM_HIGH = 1.0, 2.31
P7C_Q_LOW, P7C_Q_HIGH = 0.48, 0.82

#: optimizers_real_room.md, which the anchor row must reproduce (2 decimals).
ANCHOR_EXPECTED = {
    ("ProMP", "A"): (0.48, 0.25, 0.69, 0.18, 0.06, 0.33),
    ("ProMP", "preference GP"): (0.16, 0.05, 0.28, 0.53, 0.39, 0.68),
    ("ProMP", "LineCoSpar"): (0.10, 0.03, 0.18, 0.53, 0.41, 0.67),
    ("ProMP", "SortCMA"): (0.13, 0.04, 0.23, 0.53, 0.38, 0.69),
    ("Franks", "lens"): (0.44, 0.23, 0.63, 0.16, 0.06, 0.28),
    ("Franks", "A direct"): (0.29, 0.11, 0.46, 0.57, 0.36, 0.81),
    ("Franks", "preference GP"): (0.13, 0.04, 0.24, 0.56, 0.41, 0.71),
    ("Franks", "LineCoSpar"): (0.15, 0.06, 0.24, 0.54, 0.42, 0.68),
    ("Franks", "SortCMA"): (0.11, 0.04, 0.20, 0.57, 0.45, 0.71),
}
ANCHOR_CONTRASTS = {
    ("ProMP", "preference GP"): (0.32, 0.17, 0.47), ("ProMP", "LineCoSpar"): (0.38, 0.20, 0.56),
    ("ProMP", "SortCMA"): (0.35, 0.19, 0.51), ("Franks", "A direct"): (0.15, 0.04, 0.28),
    ("Franks", "preference GP"): (0.31, 0.15, 0.45), ("Franks", "LineCoSpar"): (0.30, 0.15, 0.44),
    ("Franks", "SortCMA"): (0.33, 0.18, 0.48),
}

ROWS = [("ProMP", "promp_s0", "a", "A"), ("ProMP", "promp_s0", "g", "preference GP"),
        ("ProMP", "promp_s0", "d_z", "LineCoSpar"), ("ProMP", "promp_s0", "cmaes", "SortCMA"),
        ("Franks", "franks", "a_lens", "lens"), ("Franks", "franks", "a", "A direct"),
        ("Franks", "franks", "g", "preference GP"), ("Franks", "franks", "d_z", "LineCoSpar"),
        ("Franks", "franks", "cmaes", "SortCMA")]
OURS = {"ProMP": "A", "Franks": "lens"}

# Schaefer landmark index moved by each mapping ("rise" = t2 - t1: move t1, hold t2).
MOVE = {"t1": 0, "t2": 1, "t3": 2, "t4": 3, "rise": 0}

# ============================================================================ literature data
# Hip timing spreads that convert (mapping in MOVE). sd in %GC (= %ST), n = people.
# Ramella: Table II [verified p.2505 Table II; the last two rows' labels are swapped in print,
# flexion peak time is 58.3 +- 2.2]; per-subject values reproduce it [derived from Fig. 4].
# Liu: per-subject values [read from figure, approx., Fig. 6a p.12, vector coordinates].
# Ding: optima [read from figure, vector coordinates, +-0.5 %; Fig. 3B p.3], 7 of 8 resolved.
# Franks: hip-only optima [derived from verified Suppl. sec. 5 p.7]. Park: [verified PMC12358144
# Sec. VI-B], "+-" read as SD [unverified: SD vs SE; full text not reachable on 2026-10-08].
# Within for Franks: across configurations [derived from verified Suppl. sec. 5 p.7]; not a
# retest, so no deflation is printed for Franks.
# Ding: the onset is fixed at maximum hip flexion, at 86.2 % of the heel-strike cycle, and "peak
# timing" is bounded 15-40 % "in this newly defined gait cycle" [verified Din18 p.3]. It is the
# extension peak, so the planned mapping is t2 (used for P7a); it is also the rise duration from
# a fixed onset, so t2 - t1 is shown as the alternative. The Nm conversion uses the rise mapping
# (0.774 vs 0.759 Nm via t2: no difference that matters); Ding enters no room.
TIMING = [
    dict(study="Ramella 2025", obj="preference", param="extension peak time f2", map="t2",
         sd=0.7, n=8, within=None, trunc=True,
         tag="[verified p.2505 Table II]; floor artefact: all 8 within 2.3 %GC of the box floor"),
    dict(study="Ramella 2025", obj="preference", param="extension rise time f3", map="rise",
         sd=3.3, n=8, within=None, trunc=True, tag="[verified p.2505 Table II]"),
    dict(study="Ramella 2025", obj="preference", param="flexion peak time f5", map="t4",
         sd=2.2, n=8, within=None, trunc=True,
         tag="[verified p.2505 Table II, printed in the row labelled 'flexion rise time']"),
    dict(study="Liu 2026 (preprint)", obj="preference", param="extension peak a1", map="t2",
         values=[16.4, 21.4, 12.4, 22.2, 11.4], n=5, within=None, trunc=False,
         tag="[read from figure, approx., Fig. 6a p.12]; 2-trial means, experienced users"),
    dict(study="Liu 2026 (preprint)", obj="preference", param="flexion peak a3", map="t4",
         values=[69.7, 60.8, 72.8, 63.7, 73.9], n=5, within=None, trunc=False,
         tag="[read from figure, approx., Fig. 6a p.12]"),
    dict(study="Ding 2018", obj="metabolic", param="peak timing from the fixed onset (rise)",
         map="rise", values=[26.55, 29.03, 33.67, 36.70, 40.0, 40.0, 40.0], n=7, within=None,
         trunc=True, tag="[read from figure, vector coordinates, +-0.5 %, Fig. 3B p.3]; "
                         "3 of 7 resolved optima at the 40 % bound"),
    dict(study="Franks 2021 (hip-only)", obj="metabolic", param="HE peak time", map="t2",
         sd=0.31, n=3, within=1.0, trunc=False,
         tag="[derived from verified Suppl. sec. 5 p.7]; seeded from P1's optimum"),
    dict(study="Franks 2021 (hip-only)", obj="metabolic", param="HE rise time", map="rise",
         sd=1.48, n=3, within=1.6, trunc=False, tag="[derived from verified Suppl. sec. 5 p.7]"),
    dict(study="Franks 2021 (hip-only)", obj="metabolic", param="HF peak time", map="t4",
         sd=0.50, n=3, within=1.2, trunc=False, tag="[derived from verified Suppl. sec. 5 p.7]"),
    dict(study="Franks 2021 (whole-leg)", obj="metabolic", param="HE peak time", map="t2",
         sd=1.40, n=4, within=1.0, trunc=False, tag="[derived from verified Suppl. sec. 5 p.7]"),
    dict(study="Franks 2021 (whole-leg)", obj="metabolic", param="HE rise time", map="rise",
         sd=0.73, n=4, within=1.6, trunc=False, tag="[derived from verified Suppl. sec. 5 p.7]"),
    dict(study="Franks 2021 (whole-leg)", obj="metabolic", param="HF peak time", map="t4",
         sd=1.68, n=4, within=1.2, trunc=False, tag="[derived from verified Suppl. sec. 5 p.7]"),
    dict(study="Park 2025 (stairs)", obj="metabolic", param="peak extension timing", map="t2",
         sd=3.8, n=10, within=None, trunc=False,
         tag="[verified PMC12358144 Sec. VI-B; values; unverified: SD vs SE]; stair climbing"),
    dict(study="Park 2025 (stairs)", obj="metabolic", param="peak flexion timing", map="t4",
         sd=3.7, n=10, within=None, trunc=False,
         tag="[verified PMC12358144 Sec. VI-B; values; unverified: SD vs SE]; stair climbing"),
]

#: Timing parameters that are not converted.
UNCONVERTED = [
    ("Ramella 2025", "flexion rise time f6", "SD 2.7 %GC, n = 8 [verified p.2505 Table II, printed "
     "in the row labelled 'flexion peak time']; analogue t4 - t3 exists but is not in the spec's "
     "mapping (unregistered sensitivity below)"),
    ("Ramella 2025", "extension / flexion onset (derived)", "SD 3.27 or 1.71 / 4.03 or 2.95 %GC "
     "depending on the undefined 'peak time' [derived from Fig. 4]; not used"),
    ("Liu 2026", "inflection point a2", "SD 5.3 %GC [read from figure, approx., Fig. 6a]; torque "
     "still positive there, not a zero crossing"),
    ("Ding 2018", "offset timing (end of extension assistance, near t3)", "SD 4.07 %, n = 7 "
     "[read from figure, vector coordinates]; 4 of 7 at the 55 % bound; t3 not in the mapping"),
    ("Franks 2021", "mid time, mid duration, HF fall time", "hip-only SD 1.58, 0.46, 2.12 % "
     "stride [derived from verified Suppl. sec. 5 p.7]"),
    ("Park 2025", "flexion duration, extension duration", "SD 3.6, 7.9 %GC [verified "
     "PMC12358144; unverified: SD vs SE]"),
    ("Bryan 2021 (speeds, loads)", "hip timings", "the 1.25 m/s rows and the no-load optima are "
     "Franks 2021 optima digit for digit [verified, corrections]; counted once, as Franks"),
    ("Ingraham 2022", "peak torque timing (ankle)", "between 1.5, within 1.5 %GC [verified p.2-3]; "
     "ankle, no hip landmark"),
    ("Diaz 2026", "push-off peak timing (ankle prosthesis)", "SD of means 1.0, within 1.79 %GC "
     "[read from figure, Fig. 3A/4; derived]; ankle"),
    ("Zhang 2017 / Poggensee 2021", "ankle peak, rise, fall time", "Poggensee SD 0.7, 4.5, 0.9 % "
     "stride [verified p.7]; Zhang spans only; ankle, metabolic"),
    ("Taddei 2026", "swing-flexion onset tSF (knee prosthesis)", "between 5.4, within 5.5 %GC "
     "[derived from verified Table 1]; knee"),
    ("Tucker 2020 (CoSpar, LineCoSpar), Alili 2023, Shepherd 2021", "gait or alignment "
     "parameters", "no torque timing; listed in the table only"),
    ("Lee 2023", "four ankle parameters", "between-person spread not obtainable (paywalled; "
     "Dryad needs a token) [verified abstract]"),
]

# Preferred-magnitude CVs. mean/b/w in the parameter's own units; m = trials per between value
# (b is an SD of m-trial means); w_cv = within as a CV where the source gives a CV.
MAGNITUDE = [
    dict(study="Ramella 2025", obj="preference", joint="hip",
         params=[("extension peak torque", [5.83, 5.07, 6.51, 5.73, 5.57, 6.17, 6.70, 7.57]),
                 ("flexion peak torque", [5.31, 6.86, 5.41, 7.32, 7.82, 7.28, 7.81, 6.60])],
         n=8, w=None, m=1, trunc=True, room=True,
         tag="[derived from Fig. 4 p.2505 vector extraction; reproduces verified Table II 6.1 +- 0.8,"
             " 6.8 +- 1.0]; 3-Nm box, lower bound"),
    dict(study="Liu 2026 (preprint)", obj="preference", joint="hip",
         params=[("extension peak b1", [5.63, 4.56, 6.48, 4.40, 6.71]),
                 ("flexion peak b3", dict(mean=9.64, sd=0.64))],
         n=5, w=None, m=2, trunc=False, room=True,
         tag="[read from figure, approx., Fig. 6a p.12]; 2-trial means, no within reported"),
    dict(study="Ingraham 2022", obj="preference", joint="ankle",
         params=[("peak torque", dict(mean=12.7, sd=3.7, w=1.7))], n=12, m=24, trunc=True, room=True,
         tag="between 3.7, within 1.7 Nm [verified p.2]; mean 12.7 Nm [read from figure, approx., "
             "Fig. 2A]"),
    dict(study="Arens 2025", obj="preference", joint="back (lifting)",
         params=[("lifting gain LIG", dict(mean=151.2, sd=44.7, icc_k=0.91, k=3)),
                 ("lowering gain LOG", dict(mean=104.3, sd=35.2, icc_k=0.80, k=3))],
         n=15, m=1, trunc=True, room=True,
         tag="mean, SD over 45 single BO endpoints [verified Table 1, PMC11980829]; within about "
             "21.4 / 23.0 [derived from verified ICC(2,k) 0.91 / 0.80, k = 3, PMC11980829], "
             "consistent with the Fig. 4A read (about 22 / 24); LOG 5/45 at the lower bound"),
    dict(study="Shepherd 2018", obj="preference", joint="ankle prosthesis (stiffness)",
         params=[("dorsiflexion stiffness", dict(mean=6.7, sd=2.0, w_cv=0.142))], n=8, m=5,
         trunc=False, room=True, stiffness=True,
         tag="6.7 +- 2.0 Nm/rad/kg (SD of 5-trial means, mass-normalised) [verified p.5]; within "
             "CV 14.2 % [verified p.5]"),
    dict(study="Shepherd 2020", obj="preference", joint="ankle prosthesis (stiffness)",
         params=[("stiffness PSE", dict(mean=597.0, sd=180.0))], n=7, m=1,
         trunc=False, room=True, stiffness=True,
         tag="597 +- 180 Nm/rad, absolute [read from figure, approx., Fig. 4, vector]; the 5.6 % "
             "is psychometric SD / PSE, not retest [verified PMC7527979], so no deflation"),
    dict(study="Clites 2021", obj="preference", joint="ankle prosthesis (stiffness)",
         params=[("stiffness, self-selected speed", dict(mean=13.0, sd=2.6))], n=7, m=3,
         trunc=False, room=False,
         tag="[read from figure, approx., Fig. 2B; unverified in pass 1: full text blocked]; "
             "table only"),
    dict(study="Diaz 2026", obj="preference", joint="ankle prosthesis",
         params=[("magnitude level (motor rev)", dict(mean=14.3, sd=2.94, w=7.9))], n=3, m=4,
         trunc=True, room=False,
         tag="[read from figure, Fig. 3A/4; derived]; n = 3, within > between: not separable"),
    dict(study="Franks 2021 (hip-only)", obj="metabolic", joint="hip",
         params=[("HE peak torque", dict(mean=0.404, sd=0.15, w=0.06)),
                 ("HF peak torque", dict(mean=0.286, sd=0.055))], n=3, m=1, trunc=False,
         room=False, tag="[derived from verified Suppl. sec. 5 p.7]; within across configurations"),
    dict(study="Park 2025 (stairs)", obj="metabolic", joint="hip",
         params=[("peak flexion magnitude", dict(mean=11.1, sd=1.5))], n=10, m=1, trunc=True,
         room=False, tag="[verified PMC12358144 Sec. VI-B; values; unverified: SD vs SE]; 13 Nm cap"),
    dict(study="Poggensee 2021", obj="metabolic", joint="ankle",
         params=[("peak torque", dict(mean=0.68, sd=0.06))], n=5, m=1, trunc=False, room=False,
         tag="[verified p.7-8]; not converged"),
]


# ============================================================================ helpers
def git_head() -> str:
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "scripts/preference_literature_room.py"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f"{out}{' (script uncommitted)' if dirty else ''}, script sha256 {sha}"
    except (OSError, subprocess.CalledProcessError):
        return f"unknown, script sha256 {sha}"


def sd_of(entry) -> float:
    if "values" in entry:
        return float(np.std(entry["values"], ddof=1))
    return float(entry["sd"])


def deflate(b, w):
    """Planned deflation sqrt(max(b^2 - w^2, 0))."""
    return float(np.sqrt(max(b * b - w * w, 0.0)))


def chi2_factors(n):
    """95 % chi-square interval of an SD as factors of the sample SD (df = n - 1)."""
    df = n - 1
    return (float(np.sqrt(df / stats.chi2.ppf(0.975, df))),
            float(np.sqrt(df / stats.chi2.ppf(0.025, df))))


def sd_ratio_ci(s1, n1, s2, n2):
    """95 % F interval of sigma1 / sigma2 from sample SDs."""
    F = (s1 / s2) ** 2
    d1, d2 = n1 - 1, n2 - 1
    return (float(np.sqrt(F / stats.f.ppf(0.975, d1, d2))),
            float(np.sqrt(F / stats.f.ppf(0.025, d1, d2))))


def f2(x, d=2):
    return G.fmt(x, d)


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--data", type=Path, default=G.DEFAULT_DATA)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    OUT, n_boot = args.out, args.boot
    L: list[str] = []
    p = L.append

    # ------------------------------------------------------------------ Schaefer and simulation
    logs = G.load_participants(args.data)
    P = np.array([logs[f]["t_pref"] for f in G.FILE_IDS])
    real = np.array([G.analytic_profile(t, G.PHASE_GRID) for t in P])
    d_loo = np.array([G.rms_distance(real[i], np.delete(real, i, axis=0).mean(axis=0))
                      for i in range(len(real))])
    sim = G.load_sim(OUT / "phase5-schaefer")
    sim_rms = np.array([G.rms(c) for c in sim["curves"]])
    real_rms = np.array([G.rms(c) for c in real])
    sim_med = float(np.median(sim_rms))
    kappa = float(sim_med / np.median(real_rms))
    y0 = kappa * d_loo
    med_y0 = float(np.median(y0))

    # Schaefer between SDs (n = 11, and n = 10 with a log) and late-exploration within (n = 10)
    sd11 = P.std(axis=0, ddof=1)
    rise11 = P[:, 1] - (P[:, 0] - 100.0)
    fids10 = [f for f in G.FILE_IDS if f != G.NO_TUNING_LOG]
    idx10 = [G.FILE_IDS.index(f) for f in fids10]
    sd10 = P[idx10].std(axis=0, ddof=1)
    late = np.array([G.late_sd(logs[f]) for f in fids10])
    within = np.sqrt(np.mean(late ** 2, axis=0))
    defl10 = np.array([deflate(b, w) for b, w in zip(sd10, within)])
    defl11 = np.array([deflate(b, w) for b, w in zip(sd11, within)])

    def conv(k, sd, profiles=P, kap=kappa):
        """G2 conversion x kappa: median over profiles of landmark_shift_rms(t, k, sd)."""
        if sd <= 0:
            return 0.0
        return kap * float(np.median([G.landmark_shift_rms(t, k, sd) for t in profiles]))

    # ------------------------------------------------------------------ main cell (copied cell)
    df_all = G.sb._n2048(G.sb._popn(G.sb._load(OUT / "phase5e", "e-main")))
    cond = G.er.BY_ID["main"]

    def cell(family, learner):
        """schaefer_grounding_optimizers.cell, with the e-main rows loaded once."""
        df = df_all
        mask = (df["family"] == family).to_numpy() & (df["learner"] == learner).to_numpy()
        want = dict(G.er.DEFAULTS)
        want.update(cond.fields)
        for name, value in want.items():
            mask &= G.er._match(df, name, value)
        df = df[mask].drop_duplicates(G.er.KEY, keep="last")
        idx = {(f, u): i for i, (f, u) in enumerate(zip(df["fold"], df["user_id"]))}
        df = df.iloc[[idx[k] for k in sim["keys"]]].reset_index(drop=True)
        dp = df["d_prior"].astype(float).to_numpy()
        dfl = df["d_floor"].astype(float).to_numpy()
        dk = np.array([float(d[K - 1]) for d in df["distances"]])
        return dp, dfl, dk

    dpA, dfA, _ = cell("promp_s0", "a")
    room = dpA - dfA
    folds = sim["folds"]
    flags = {}
    for fam_label, fam, lr, name in ROWS:
        dp, _, dk = cell(fam, lr)
        flags[(fam_label, name)] = ((dp - dk) / G.JND[1.0] >= 1.0 - G.TOL, dk > dp)

    # shared draws, exactly as schaefer_grounding_optimizers.py
    rng = np.random.default_rng(SEED_SHARED)
    sim_draws = G.fold_draws(folds, n_boot, rng)
    part_draws = rng.integers(0, len(y0), size=(n_boot, len(y0)))
    kappa_b = np.median(sim_rms[sim_draws], axis=1) / np.median(real_rms[part_draws], axis=1)

    # ------------------------------------------------------------------ a. timing conversions
    t_rows = []
    for e in TIMING:
        sd = sd_of(e)
        k = MOVE[e["map"]]
        nm = conv(k, sd)
        dfl = deflate(sd, e["within"]) if e["within"] is not None else None
        t_rows.append(dict(e, sd_used=sd, nm=nm, defl=dfl,
                           nm_defl=conv(k, dfl) if dfl is not None else None,
                           nm_alt=conv(1, sd) if e["map"] == "rise" else None))
    sch_rows = []
    for k, lab in enumerate(G.TIMINGS):
        sch_rows.append(dict(lab=lab, sd11=sd11[k], sd10=sd10[k], within=within[k],
                             defl10=defl10[k], defl11=defl11[k],
                             nm11=conv(k, sd11[k]), nm10=conv(k, sd10[k]),
                             nm_defl10=conv(k, defl10[k]), nm_defl11=conv(k, defl11[k])))
    T_sch_raw = float(np.sqrt(sum(r["nm11"] ** 2 for r in sch_rows)))
    T_sch_defl = float(np.sqrt(sum(r["nm_defl10"] ** 2 for r in sch_rows)))
    T_sch_defl11 = float(np.sqrt(sum(r["nm_defl11"] ** 2 for r in sch_rows)))
    T_sch_raw3 = float(np.sqrt(sch_rows[0]["nm11"] ** 2 + sch_rows[1]["nm11"] ** 2
                               + sch_rows[3]["nm11"] ** 2))
    ram = [r for r in t_rows if r["study"] == "Ramella 2025"]
    T_ram = float(np.sqrt(sum(r["nm"] ** 2 for r in ram)))
    T_ram_f6 = float(np.sqrt(T_ram ** 2 + conv(2, 2.7) ** 2))  # flexion rise -> move t3
    liu = [r for r in t_rows if r["study"].startswith("Liu")]
    T_liu = float(np.sqrt(sum(r["nm"] ** 2 for r in liu)))
    calib = med_y0 / T_sch_raw  # direct (b) x kappa median over the RMS-sum rule, Schaefer raw

    # ------------------------------------------------------------------ b, c. magnitude CVs
    m_rows = []
    for e in MAGNITUDE:
        per = []
        for pname, v in e["params"]:
            if isinstance(v, list):
                mean, b, w, w_cv = float(np.mean(v)), float(np.std(v, ddof=1)), None, None
            else:
                mean, b, w, w_cv = v["mean"], v["sd"], v.get("w"), v.get("w_cv")
                if "icc_k" in v:  # ICC(2,k) -> ICC(2,1) -> within variance (1 - ICC1) * SD^2
                    r_, k_ = v["icc_k"], v["k"]
                    icc1 = r_ / (k_ - (k_ - 1) * r_)
                    w = float(np.sqrt((1.0 - icc1) * b * b))
            if w is None and w_cv is not None:
                w = w_cv * mean
            cv = b / mean
            if w is not None:
                cv_defl = deflate(b, w) / mean                         # planned
                cv_means = (np.sqrt(max(b * b - w * w / e["m"], 0.0)) / mean)  # b = SD of m-trial means
                cv_single = (np.sqrt(b * b + w * w * (1 - 1 / e["m"])) / mean)  # one endpoint each
            else:
                cv_defl = cv_means = cv_single = None
            per.append(dict(param=pname, mean=mean, b=b, w=w, cv=cv, cv_defl=cv_defl,
                            cv_means=cv_means, cv_single=cv_single))

        def pool(key):
            vals = [q[key] if q[key] is not None else None for q in per]
            if any(v is None for v in vals):
                return None
            return float(np.sqrt(np.mean(np.square(vals))))
        cv_raw = pool("cv")
        cv_d = pool("cv_defl")
        m_rows.append(dict(e, per=per, cv_raw=cv_raw, cv_defl=cv_d, cv_means=pool("cv_means"),
                           cv_single=pool("cv_single") if cv_d is not None else cv_raw,
                           cv_room=cv_d if cv_d is not None else cv_raw,
                           cv_room_means=pool("cv_means") if cv_d is not None else cv_raw))
    room_studies = [r for r in m_rows if r["room"]]
    both = [r for r in room_studies if r["cv_defl"] is not None]
    cv_low = next(r for r in m_rows if r["study"] == "Ramella 2025")["cv_raw"]
    cv_central = float(np.median([r["cv_room"] for r in room_studies]))
    cv_central_means = float(np.median([r["cv_room_means"] for r in room_studies]))
    cv_high = float(np.median([r["cv_single"] for r in both]))
    cv_hip = float(np.mean([r["cv_raw"] for r in room_studies if r["joint"] == "hip"]))
    no_stiff = [r for r in room_studies if not r.get("stiffness")]
    cv_no_stiff = float(np.median([r["cv_room"] for r in no_stiff]))
    # uniform-on-box null for Ramella: SD = box width / sqrt(12) (10 %GC, 3 Nm boxes)
    ram_m = next(r for r in m_rows if r["study"] == "Ramella 2025")
    U_T, U_M = 10.0 / np.sqrt(12.0), 3.0 / np.sqrt(12.0)
    cv_unif = float(np.sqrt(np.mean([(U_M / q["mean"]) ** 2 for q in ram_m["per"]])))

    # ------------------------------------------------------------------ d. room levels
    def lvl(T, cv):
        M = cv * sim_med
        return dict(T=T, cv=cv, M=M, L=float(np.hypot(T, M)))
    levels = {
        "low": lvl(T_ram, cv_low),
        "central": lvl(T_sch_defl, cv_central),
        "high": lvl(T_sch_raw, cv_high),
    }

    # Monte Carlo median of the combined distance (sensitivity: the RMS sum is a 1-SD quantity,
    # Schaefer's (b) median is a median). Independent Gaussian landmark shifts and an amplitude
    # scaling of each kappa-matched Schaefer profile; draws that break the cyclic order rejected.
    mc_rng = np.random.default_rng(SEED_OWN + 7)

    def mc_median(sds_by_k, cv, n_per=400):
        out = []
        for t in P:
            base = G.analytic_profile(t, G.PHASE_GRID)
            for _ in range(n_per):
                q = np.array(t, float)
                for k, s in sds_by_k:
                    q[k] += mc_rng.normal(0.0, s)
                cyc = np.array([q[1], q[2], q[3], q[0], q[1] + 100.0])
                if not np.all(np.diff(cyc) > 0):
                    continue
                amp = 1.0 + mc_rng.normal(0.0, cv)
                out.append(kappa * G.rms_distance(amp * G.analytic_profile(q, G.PHASE_GRID), base))
        return float(np.median(out))
    ram_sds = [(MOVE[r["map"]], r["sd_used"]) for r in ram]
    mc = {
        "Schaefer raw, timing only": mc_median(list(enumerate(sd11)), 0.0),
        "low": mc_median(ram_sds, cv_low),
        "central": mc_median(list(enumerate(defl10)), cv_central),
        "high": mc_median(list(enumerate(sd11)), cv_high),
    }

    T_unif = float(np.sqrt(sum(conv(MOVE[r["map"]], U_T) ** 2 for r in ram)))
    sens = {
        "central, median-matched (Monte Carlo median)": mc["central"],
        f"central, both terms x calibration ({calib:.3f})": calib * levels["central"]["L"],
        "central, no stiffness studies (Shepherd 2018, 2020 out)": float(np.hypot(
            T_sch_defl, cv_no_stiff * sim_med)),
        "central, raw Schaefer timing (no deflation)": float(np.hypot(T_sch_raw,
                                                                      cv_central * sim_med)),
        "central, magnitude deflated as SDs of trial means": float(np.hypot(T_sch_defl,
                                                                            cv_central_means * sim_med)),
        "central, hip-only magnitude (Ramella, Liu)": float(np.hypot(T_sch_defl, cv_hip * sim_med)),
        "central, Schaefer deflation with n = 11 between-SDs": float(np.hypot(T_sch_defl11,
                                                                              cv_central * sim_med)),
        "low, Ramella flexion rise f6 added (t4 - t3, move t3)": float(np.hypot(T_ram_f6,
                                                                              cv_low * sim_med)),
        "low, median-matched (Monte Carlo median)": mc["low"],
        "low, uniform-on-box null (Ramella box widths)": float(np.hypot(T_unif,
                                                                         cv_unif * sim_med)),
        "high, median-matched (Monte Carlo median)": mc["high"],
        "hip preference only: Liu timing and magnitude": float(np.hypot(T_liu,
                                                                         next(r for r in m_rows if r["study"].startswith("Liu"))["cv_raw"] * sim_med)),
    }

    # ------------------------------------------------------------------ P7c(i): interval of the central room
    # (own generator SEED_OWN, so the SEED_SHARED draw order of the shares is untouched)
    rng2 = np.random.default_rng(SEED_OWN)
    sd2 = G.fold_draws(folds, n_boot, rng2)
    pd10 = rng2.integers(0, len(idx10), size=(n_boot, len(idx10)))
    P10 = P[idx10]
    T_b, M_b, L_b = [], [], []
    for b in range(n_boot):
        sel = pd10[b]
        bsd = P10[sel].std(axis=0, ddof=1)
        bw = np.sqrt(np.mean(late[sel] ** 2, axis=0))
        bdefl = [deflate(x, y) for x, y in zip(bsd, bw)]
        smed = float(np.median(sim_rms[sd2[b]]))
        kap = smed / float(np.median(real_rms[idx10][sel]))
        T = float(np.sqrt(sum(conv(k, s, P10[sel], kap) ** 2 for k, s in enumerate(bdefl))))
        cvs = []
        for r in room_studies:
            df_ = r["n"] - 1
            cvs.append(r["cv_room"] * np.sqrt(df_ / rng2.chisquare(df_)))
        M = float(np.median(cvs)) * smed
        T_b.append(T)
        M_b.append(M)
        L_b.append(float(np.hypot(T, M)))
    L_b = np.array(L_b)
    Lc_ci = np.percentile(L_b, [2.5, 97.5])
    T_ci = np.percentile(T_b, [2.5, 97.5])
    M_ci = np.percentile(M_b, [2.5, 97.5])

    # ------------------------------------------------------------------ e. shares per level
    def targets(Lval, Lvec=None):
        """Point target and per-draw targets. Lval None = anchor (unrescaled (b) x kappa).
        Lvec: per-draw room levels (propagates the room's own uncertainty; draw b of Lvec and
        part_draws[b] are independent draws paired by index)."""
        if Lval is None:
            return y0, [kappa_b[b] * d_loo[part_draws[b]] for b in range(n_boot)]
        yp = d_loo * Lval / np.median(d_loo)
        Ls = Lvec if Lvec is not None else np.full(n_boot, Lval)
        yb = [d_loo[part_draws[b]] * Ls[b] / np.median(d_loo[part_draws[b]]) for b in range(n_boot)]
        return yp, yb

    def point_share(Lval, key=("ProMP", "A")):
        """Point Q>=1 share at a fixed level (no bootstrap)."""
        yp = d_loo * Lval / np.median(d_loo)
        return G.weighted_share(G.kde_weights(room, yp), flags[key][0])

    def shares(Lval, Lvec=None):
        yp, yb = targets(Lval, Lvec)
        w = G.kde_weights(room, yp)
        wb = [G.kde_weights(room[sim_draws[b]], yb[b]) for b in range(n_boot)]
        res = {}
        for key, (q1, worse) in flags.items():
            bq = np.array([G.weighted_share(wb[b], q1[sim_draws[b]]) for b in range(n_boot)])
            bw = np.array([G.weighted_share(wb[b], worse[sim_draws[b]]) for b in range(n_boot)])
            res[key] = dict(q=G.weighted_share(w, q1), q_ci=np.percentile(bq, [2.5, 97.5]),
                            w=G.weighted_share(w, worse), w_ci=np.percentile(bw, [2.5, 97.5]),
                            bq=bq, bw=bw, uq=float(q1.mean()), uw=float(worse.mean()))
        con = {}
        for fam, ours in OURS.items():
            for (f, name), r in res.items():
                if f != fam or name == ours:
                    continue
                d = res[(fam, ours)]["q"] - r["q"]
                db = res[(fam, ours)]["bq"] - r["bq"]
                lo, hi = np.percentile(db, [2.5, 97.5])
                dw = r["w"] - res[(fam, ours)]["w"]
                dwb = r["bw"] - res[(fam, ours)]["bw"]
                con[(fam, name)] = dict(d=d, lo=lo, hi=hi, dw=dw,
                                        dw_ci=np.percentile(dwb, [2.5, 97.5]),
                                        verdict=G.verdict_lower(d, lo, hi, 0.0))
        med = float(np.median(yp))
        q1A = flags[("ProMP", "A")][0]
        raw_mask = room <= med
        return dict(res=res, con=con, ess=G.ess(w), median=med, ymin=float(yp.min()),
                    ymax=float(yp.max()),
                    n_in=int(np.sum((room >= yp.min()) & (room <= yp.max()))),
                    n_below=int(raw_mask.sum()),
                    raw_A=float(q1A[raw_mask].mean()) if raw_mask.any() else np.nan,
                    ess_b=np.percentile([G.ess(x) for x in wb], [2.5, 97.5]))

    S = {"anchor": shares(None)}
    for name, lv in levels.items():
        S[name] = shares(lv["L"])
    S_sens = {name: shares(v) for name, v in sens.items()}
    S_prop = shares(levels["central"]["L"], L_b)
    q_over_ci = (point_share(Lc_ci[0]), point_share(Lc_ci[1]))
    mm = {"low": "low, median-matched (Monte Carlo median)",
          "central": "central, median-matched (Monte Carlo median)",
          "high": "high, median-matched (Monte Carlo median)"}

    # anchor gate
    gate_ok = True
    gate_lines = []
    for key, exp in ANCHOR_EXPECTED.items():
        r = S["anchor"]["res"][key]
        got = (r["q"], *r["q_ci"], r["w"], *r["w_ci"])
        ok = all(abs(round(g, 2) - e) < 1e-9 for g, e in zip(got, exp))
        gate_ok &= ok
        gate_lines.append(f"| {key[0]} {key[1]} | {exp[0]:.2f} [{exp[1]:.2f}, {exp[2]:.2f}]; worse "
                          f"{exp[3]:.2f} [{exp[4]:.2f}, {exp[5]:.2f}] | {got[0]:.2f} [{got[1]:.2f}, "
                          f"{got[2]:.2f}]; worse {got[3]:.2f} [{got[4]:.2f}, {got[5]:.2f}] | "
                          f"{'yes' if ok else 'NO'} |")
    for key, exp in ANCHOR_CONTRASTS.items():
        c = S["anchor"]["con"][key]
        ok = all(abs(round(g, 2) - e) < 1e-9 for g, e in zip((c["d"], c["lo"], c["hi"]), exp))
        gate_ok &= ok
        gate_lines.append(f"| {OURS[key[0]]} − {key[1]} ({key[0]}) | {exp[0]:+.2f} [{exp[1]:+.2f}, "
                          f"{exp[2]:+.2f}] | {c['d']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] | "
                          f"{'yes' if ok else 'NO'} |")

    # ------------------------------------------------------------------ P7a
    sch = {"t2": (sd11[1], 11), "t4": (sd11[3], 11), "rise": (float(np.std(rise11, ddof=1)), 11)}
    p7a = []
    LMK = {"t2": "t2", "t4": "t4", "rise": "t2 − t1"}
    for r in t_rows:
        if r["study"].startswith("Franks 2021 (whole") or r["study"].startswith("Park"):
            registered = False
        elif r["study"].startswith("Liu"):
            registered = False
        else:
            registered = True
        if r["study"].startswith("Ding"):
            # planned mapping: Ding's "peak timing" is the extension peak -> t2; the rise
            # reading (t2 - t1) includes Schaefer's onset variation, which Ding fixed by design.
            maps = [("t2", "t2 (planned mapping)", True),
                    ("rise", "t2 − t1 (alternative)", False)]
        else:
            maps = [(r["map"], LMK[r["map"]], registered)]
        for mp, lab, reg_ in maps:
            s1, n1 = sch[mp]
            ratio = s1 / r["sd_used"]
            lo, hi = sd_ratio_ci(s1, n1, r["sd_used"], r["n"])
            v = G.verdict_lower(ratio, lo, hi, P7A_RATIO)
            shown = v
            if v == "held" and (r["study"].startswith("Ramella") or r["study"].startswith("Franks")):
                shown = "held against a capped/seeded spread (not a test)"
            p7a.append(dict(r, map=mp, lab=lab, s1=s1, ratio=ratio, lo=lo, hi=hi,
                            registered=reg_, verdict=v, shown=shown))
    reg = [x for x in p7a if x["registered"]]
    ding_t2 = next(x for x in p7a if x["study"].startswith("Ding") and x["map"] == "t2")
    ding_rise = next(x for x in p7a if x["study"].startswith("Ding") and x["map"] == "rise")
    liu_ratios = [x["ratio"] for x in p7a if x["study"].startswith("Liu")]
    if all(x["verdict"] == "held" for x in reg):
        p7a_overall = "held"
    elif any(x["verdict"] == "failed" for x in reg):
        p7a_overall = "failed"
    else:
        n_pt = sum(x["ratio"] > 1 for x in reg)
        p7a_overall = (f"uncertain (point estimate held in {n_pt} of {len(reg)}; "
                       f"{sum(x['verdict'] == 'held' for x in reg)} of {len(reg)} held)")
    p7a_line = (f"{p7a_overall}. Schäfer is wider than comparators whose spread is capped "
                f"(Ramella) or seeded (Franks, n = 3); it equals Liu (ratios "
                f"{', '.join(f'{x:.2f}' for x in liu_ratios)}); against Ding the ratio is "
                f"{ding_t2['ratio']:.2f} [{ding_t2['lo']:.2f}, {ding_t2['hi']:.2f}] at t2 "
                f"({ding_rise['ratio']:.2f} [{ding_rise['lo']:.2f}, {ding_rise['hi']:.2f}] at "
                f"t2 − t1). That Schäfer is at the wide end is not shown")

    # ------------------------------------------------------------------ P7b
    p7b = []
    for r in room_studies:
        lo_f, hi_f = chi2_factors(r["n"])
        v = r["cv_room"]
        p7b.append(dict(study=r["study"], v=v, raw=r["cv_raw"], joint=r["joint"],
                        lo=v * lo_f, hi=v * hi_f, trunc=r["trunc"],
                        kind="deflated" if r["cv_defl"] is not None else "raw",
                        per=[(q["param"], q["cv"], q["cv_defl"]) for q in r["per"]],
                        verdict=G.verdict_range(v, v * lo_f, v * hi_f, P7B_LOW, P7B_HIGH)))
    if all(x["verdict"] == "held" for x in p7b):
        p7b_overall = "held"
    elif any(x["verdict"] == "failed" for x in p7b):
        p7b_overall = "failed"
    else:
        n_in = sum(P7B_LOW <= x["v"] <= P7B_HIGH for x in p7b)
        p7b_overall = f"uncertain (point estimate held: {n_in} of {len(p7b)} studies inside 13-30 %)"
    n_in_used = sum(P7B_LOW <= x["v"] <= P7B_HIGH for x in p7b)
    n_in_raw = sum(P7B_LOW <= x["raw"] <= P7B_HIGH for x in p7b)
    raw_above = [x for x in p7b if x["raw"] > P7B_HIGH]
    hip_v = [x["v"] for x in p7b if x["joint"] == "hip"]
    oth_v = [x["v"] for x in p7b if x["joint"] != "hip"]
    p7b_line = (f"uncertain; point estimates inside 13-30 % for {n_in_used} of {len(p7b)} "
                f"({n_in_raw} of {len(p7b)} on raw CVs; "
                + ", ".join(f"{x['study'].split(' ')[0]} {x['study'].split(' ')[1]} raw {x['raw']:.3f}"
                            for x in raw_above)
                + f" above); hip studies {', '.join(f'{v:.2f}' for v in hip_v)}, others "
                f"{min(oth_v):.2f}-{max(oth_v):.2f}")
    per_params = [(r["study"], q["param"], q["cv"], q["cv_defl"]) for r in room_studies for q in r["per"]]

    # ------------------------------------------------------------------ P7c
    Lc = levels["central"]["L"]
    p7c1 = G.verdict_range(Lc, Lc_ci[0], Lc_ci[1], P7C_ROOM_LOW, P7C_ROOM_HIGH)
    rA = S["central"]["res"][("ProMP", "A")]
    p7c2 = G.verdict_range(rA["q"], rA["q_ci"][0], rA["q_ci"][1], P7C_Q_LOW, P7C_Q_HIGH)
    leads = {lv: S[lv]["con"] for lv in ("low", "central", "high")}
    all_v = [c["verdict"] for lv in leads.values() for c in lv.values()]
    if all(v == "held" for v in all_v):
        p7c3 = "held"
    elif any(v == "failed" for v in all_v):
        p7c3 = "failed"
    else:
        p7c3 = (f"uncertain ({sum(v == 'held' for v in all_v)} of {len(all_v)} contrasts held; "
                f"the rest straddle 0)")
    promp_v = [c["verdict"] for lv in leads.values() for (f, _), c in lv.items() if f == "ProMP"]
    p7c3_promp = "held" if all(v == "held" for v in promp_v) else (
        "failed" if any(v == "failed" for v in promp_v) else
        f"uncertain ({sum(v == 'held' for v in promp_v)} of {len(promp_v)} held)")
    not_held = [(lv, f, n_) for lv, cs in leads.items() for (f, n_), c in cs.items()
                if c["verdict"] != "held"]
    rP = S_prop["res"][("ProMP", "A")]
    p7c2_prop = G.verdict_range(rP["q"], rP["q_ci"][0], rP["q_ci"][1], P7C_Q_LOW, P7C_Q_HIGH)
    central_sens = {k_: v for k_, v in sens.items()
                    if k_.startswith("central") or k_.startswith("hip preference only")}
    cs_min, cs_max = min(central_sens.values()), max(central_sens.values())
    p7c1_sens = all(P7C_ROOM_LOW <= v <= P7C_ROOM_HIGH for v in central_sens.values())

    # ================================================================== report
    p("# Literature-informed room for personalisation")
    p("")
    p(f"Generated by `scripts/preference_literature_room.py` on {_dt.date.today().isoformat()} at "
      f"repository commit {git_head()}. Spec: `our design notes"
      "` §7 (predictions P7a-P7c planned before the extraction). Literature values "
      "and their tags: `docs/preference_studies_2026-10-08.md`. Bootstrap: "
      f"{n_boot} draws, 95 % percentile intervals.")
    p("")
    p("**Verdict rule** (spec §6). *held*: the 95 % interval lies wholly on the predicted side; "
      "*failed*: wholly on the other side; *uncertain*: it straddles (the point estimate's side "
      "is given).")
    p("")
    summary_at = len(L)
    p("## 0. Reproduction gates")
    p("")
    p(f"- Median RMS of the 200 simulated population ideals: **{sim_med:.3f} Nm** (spec: 4.82). "
      f"κ = {sim_med:.3f} / {np.median(real_rms):.3f} = **{kappa:.4f}** (spec §6: 0.659). "
      f"Schäfer (b) ×κ median {med_y0:.3f} Nm (§6: 1.26).")
    p(f"- Schäfer between-SD t1..t4, n = 11: {', '.join(f'{x:.2f}' for x in sd11)} %ST (REPORT §3.2: "
      f"4.8, 5.3, 3.2, 6.8); n = 10: {', '.join(f'{x:.2f}' for x in sd10)} (REPORT §4: 4.8, 5.6, "
      f"3.0, 7.1); late within RMS {', '.join(f'{x:.2f}' for x in within)} (2.7, 3.3, 2.8, 2.5); "
      f"deflated (n = 10) {', '.join(f'{x:.2f}' for x in defl10)} (REPORT §3.7: 4.0, 4.5, 1.0, 6.7). "
      f"t2 − t1: SD {np.std(rise11, ddof=1):.2f} %ST (REPORT §3.3: 7.8).")
    p(f"- Anchor: Schäfer's timing-only room, unrescaled, must reproduce "
      f"`optimizers_real_room.md` (same seed {SEED_SHARED}, same draw order). "
      f"**{'All rows reproduce.' if gate_ok else 'FAILED: see the table.'}**")
    p("")
    p("| row | optimizers_real_room.md | this script | ok |")
    p("|---|---|---|---|")
    L.extend(gate_lines)
    p("")

    # ---- a
    p("## 1. Timing spreads converted to RMS Nm (step a)")
    p("")
    p("Conversion: G2's landmark shift. Each of the 11 Schäfer analytic preferred profiles has the "
      "mapped landmark moved by +SD and by −SD; the RMS change (mean of the two) is taken per "
      "profile, the median over the 11 profiles is multiplied by κ. Mapping: extension peak → t2, "
      "extension rise → t2 − t1 (moving the onset t1 with the peak t2 held, because each source "
      "holds its peak time as a separate parameter; moving t2 instead is shown in the last "
      "column), flexion peak → t4. On these profiles 1 %ST is about "
      f"{conv(0, 1.0):.3f} / {conv(1, 1.0):.3f} / {conv(2, 1.0):.3f} / {conv(3, 1.0):.3f} Nm ×κ "
      "for t1 / t2 / t3 / t4, nearly linear up to the SDs used here.")
    p("")
    p("**Schäfer's own landmarks** (preference, self-tuning, n = 11; within = late-exploration RMS, "
      "n = 10; own computation from the raw data):")
    p("")
    p("| landmark | between SD n = 11 | n = 10 | within | deflated (n = 10) | RMS Nm ×κ raw (n = 11) | deflated |")
    p("|---|---|---|---|---|---|---|")
    for r in sch_rows:
        p(f"| {r['lab']} | {r['sd11']:.2f} | {r['sd10']:.2f} | {r['within']:.2f} | {r['defl10']:.2f} | "
          f"{r['nm11']:.3f} | {r['nm_defl10']:.3f} |")
    p(f"| **RMS sum (timing component)** | | | | | **{T_sch_raw:.3f}** | **{T_sch_defl:.3f}** |")
    p("")
    p(f"Calibration check: the RMS-sum rule gives {T_sch_raw:.2f} Nm for Schäfer raw; the "
      f"directly measured room (b) ×κ has median {med_y0:.2f} Nm (ratio {calib:.3f}); the Monte "
      f"Carlo median of independent Gaussian landmark shifts at the same SDs is "
      f"{mc['Schaefer raw, timing only']:.2f} Nm. The RMS-sum is a root-mean-square (1-SD) "
      "distance, (b) is a median: the rule overstates a median room by about "
      f"{(1 / calib - 1) * 100:.0f} % for timing alone. Rows in §3 show the effect.")
    p("")
    p("**Other hip studies:**")
    p("")
    p("| study | objective | parameter → landmark | SD %GC (n) | within | RMS Nm ×κ raw | deflated | rise via t2 | source tag |")
    p("|---|---|---|---|---|---|---|---|---|")
    for r in t_rows:
        tl = {"t2": "t2", "t4": "t4", "rise": "t2 − t1"}[r["map"]]
        franks = r["study"].startswith("Franks")
        wtxt = (f"{f2(r['within'])}†" if franks else f2(r["within"])) if r["within"] is not None \
            else "—"
        dtxt = "—" if (franks or r["nm_defl"] is None) else f2(r["nm_defl"], 3)
        p(f"| {r['study']} | {r['obj']} | {r['param']} → {tl} | {r['sd_used']:.2f} ({r['n']}) | "
          f"{wtxt} | {r['nm']:.3f} | {dtxt} | "
          f"{f2(r['nm_alt'], 3) if r['nm_alt'] is not None else '—'} | {r['tag']}"
          f"{'; box-truncated' if r['trunc'] else ''} |")
    p("")
    p("† Franks: within = across joint configurations, not a retest; no deflation is printed. "
      "Ding: the planned mapping for P7a is t2 (its peak sits at about 21 % of the heel-strike "
      "cycle, Schäfer's t2 at 20.0 %ST); the Nm conversion uses the rise reading, the t2 reading "
      "is in the last column (no material difference). None of the metabolic rows enters a room.")
    p("")
    p(f"Timing components (RMS sum over the converted parameters): Ramella {T_ram:.3f} Nm "
      f"(box-limited, not a bound); Ramella with f6 added {T_ram_f6:.3f}; Liu {T_liu:.3f} (n = 5); "
      f"Schäfer raw {T_sch_raw:.3f}, deflated {T_sch_defl:.3f}; Schäfer raw restricted to t1, t2, "
      f"t4 {T_sch_raw3:.3f}.")
    p("")
    p("**Not converted** (no landmark in the planned mapping, or not hip):")
    p("")
    for s, prm, why in UNCONVERTED:
        p(f"- {s}, {prm}: {why}.")
    p("")

    # ---- b, c
    p("## 2. Magnitude CVs converted to Nm (steps b, c)")
    p("")
    p(f"Nm = CV × {sim_med:.3f} Nm (median RMS of the simulated ideals). Deflated = the planned "
      "√max(b² − w², 0). Where b is an SD of m-trial means, √(b² − w²/m) is the matching "
      "deflation and √(b² + w²(1 − 1/m)) the single-endpoint equivalent; both are printed. "
      "A study with two magnitudes is pooled as the root-mean-square of its CVs.")
    p("")
    p("| study | objective, joint | parameter | mean | between SD | within | CV raw | deflated (planned) | b of means | single endpoint | Nm (raw / defl.) |")
    p("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in m_rows:
        for q in r["per"]:
            nm = f"{q['cv'] * sim_med:.2f} / {q['cv_defl'] * sim_med:.2f}" if q["cv_defl"] is not None \
                else f"{q['cv'] * sim_med:.2f} / —"
            p(f"| {r['study']} | {r['obj']}, {r['joint']} | {q['param']} | {q['mean']:.3g} | "
              f"{q['b']:.3g} | {f2(q['w']) if q['w'] is not None else '—'} | {q['cv']:.3f} | "
              f"{f2(q['cv_defl'], 3) if q['cv_defl'] is not None else '—'} | "
              f"{f2(q['cv_means'], 3) if q['cv_means'] is not None else '—'} | "
              f"{f2(q['cv_single'], 3) if q['cv_single'] is not None else '—'} | {nm} |")
    p("")
    p("Sources: " + "; ".join(f"{r['study']}: {r['tag']}" for r in m_rows) + ".")
    p("")
    p("Per study, the CV that enters the room (deflated where both spreads are given, else raw), "
      "preference studies with a verified magnitude spread only:")
    p("")
    for r in room_studies:
        p(f"- {r['study']}: {r['cv_room']:.3f} ({'deflated' if r['cv_defl'] is not None else 'raw'}"
          f"{', box-truncated' if r['trunc'] else ''}).")
    p("- Not used for the room: Clites 2021 (figure reads unverified), Díaz 2026 (n = 3, within > "
      "between), Taddei 2026 (n = 4 preprint, within ≥ between, heavily box-truncated), metabolic "
      "studies (context). Inclusion rule: preference studies with a verified magnitude spread, "
      "impedance (stiffness) parameters included; the room without the two stiffness studies is a "
      "sensitivity row in §3.")
    p("")

    # ---- d
    p("## 3. Room levels (step d)")
    p("")
    p("| level | timing component | Nm ×κ | magnitude component | CV | Nm | room, RMS sum |")
    p("|---|---|---|---|---|---|---|")
    p(f"| low: Ramella box scenario (not a bound) | Ramella f2, f3, f5 (box-limited, raw; no within reported) | "
      f"{levels['low']['T']:.3f} | Ramella, RMS of the two peak-torque CVs (box-limited) | "
      f"{cv_low:.3f} | {levels['low']['M']:.3f} | **{levels['low']['L']:.2f}** |")
    p(f"| central | Schäfer t1..t4 deflated by late exploration (n = 10) | {levels['central']['T']:.3f} | "
      f"median over the {len(room_studies)} preference studies (deflated where possible) | "
      f"{cv_central:.3f} | {levels['central']['M']:.3f} | **{levels['central']['L']:.2f}** "
      f"[{Lc_ci[0]:.2f}, {Lc_ci[1]:.2f}] |")
    p(f"| high | Schäfer t1..t4 raw (single stopping points, n = 11) | {levels['high']['T']:.3f} | "
      f"single-endpoint equivalent, undeflated: median over the {len(both)} studies reporting both "
      f"spreads | {cv_high:.3f} | {levels['high']['M']:.3f} | **{levels['high']['L']:.2f}** |")
    p("")
    p(f"Interval of the central room: bootstrap of the 10 logged participants (between and within "
      f"recomputed, profiles and κ redrawn), simulated users within folds (median RMS), and each "
      f"study's CV drawn from its χ² sampling distribution at its n before taking the median. "
      f"Timing component [{T_ci[0]:.2f}, {T_ci[1]:.2f}], magnitude component [{M_ci[0]:.2f}, "
      f"{M_ci[1]:.2f}] Nm. A percentile bootstrap of an SD at n = 10 is too narrow (REPORT §3.2). "
      "The interval covers sampling only, not the choices of studies, deflation and "
      "median-matching; the sensitivity rooms below cover those.")
    p("")
    p(f"Low level: Ramella's finals are near-uniform on its box (a uniform null gives "
      f"{sens['low, uniform-on-box null (Ramella box widths)']:.2f} Nm), so this level measures the "
      f"box widths, not wearers. It lies below the planned lower anchor (Schäfer's timing-only "
      f"room, {med_y0:.2f} Nm) and is kept as the planned low scenario for P7c(iii) only.")
    p("")
    p("Choices, each stated: (1) timing comes only from the hip preference studies, magnitude only "
      "from preference studies with a verified magnitude spread; metabolic optima are context. "
      "(2) The central magnitude CV comes mostly from ankle, back and prosthesis studies; the two "
      f"hip preference studies give {cv_hip:.3f} (Ramella box-limited, Liu n = 5 preprint). "
      "(3) Landmarks and magnitude are combined as independent (RMS sum), as the spec registers. "
      "(4) Magnitude is a CV of the whole profile; Ramella and Liu tuned extension and flexion "
      "peaks separately and are pooled. (5) Under the ProMP-family definition of the room (§6 "
      "context: real room 1.65 instead of 1.26 Nm) every level would sit about 0.4 Nm higher. "
      f"(6) The planned recipe pins the median of (b) to an RMS-sum; on Schäfer's data the "
      f"RMS-sum exceeds the median by {(1 / calib - 1) * 100:.0f} % ({T_sch_raw:.2f} vs "
      f"{med_y0:.2f} Nm); the median-matched rows are the like-for-like reading. "
      f"(7) Deflation uses late-exploration spread, a proxy for endpoint noise that probably "
      f"over-deflates (t3 {defl10[2]:.2f} vs {sd10[2]:.2f} %ST); with raw timing the central room "
      f"is {sens['central, raw Schaefer timing (no deflation)']:.2f} Nm. "
      "(8) Two of the six magnitude studies tune a prosthesis stiffness, not a torque; a stiffness "
      "CV is not a torque-magnitude CV, because deflection changes with stiffness. Shepherd "
      "2018's CV is mass-normalised (Nm/rad/kg), the others are absolute. Without the two "
      f"stiffness studies the central CV is {cv_no_stiff:.3f} and the room "
      f"{sens['central, no stiffness studies (Shepherd 2018, 2020 out)']:.2f} Nm.")
    p("")
    p("Sensitivity rooms (unregistered):")
    p("")
    p("| variant | room Nm |")
    p("|---|---|")
    for k_, v in sens.items():
        p(f"| {k_} | {v:.2f} |")
    p("")

    # ---- e
    def share_table(Sx, title, anchor=False):
        p(f"### {title}")
        p("")
        what = "unrescaled, median" if anchor else "rescaled to median"
        p(f"Target: Schäfer (b) ×κ {what} {Sx['median']:.2f} Nm (range {Sx['ymin']:.2f}–"
          f"{Sx['ymax']:.2f}); {Sx['n_in']} of 200 simulated users have a room inside that range; KDE "
          f"ESS {Sx['ess']:.0f} (bootstrap {Sx['ess_b'][0]:.0f}–{Sx['ess_b'][1]:.0f}); raw A share "
          f"among the {Sx['n_below']} simulated users with room ≤ the median: {f2(Sx['raw_A'])}.")
        if Sx["ess"] < 40:
            p("")
            p(f"*Thin support:* the target lies largely below the simulated rooms (simulated 5 % "
              f"quantile {np.percentile(room, 5):.2f} Nm), so these shares rest on about "
              f"{Sx['ess']:.0f} effectively weighted simulated users (fewer in some draws). They are "
              "estimable but coarse.")
        p("")
        p("| family | optimizer | Q≥1 unweighted | Q≥1 [95 % CI] | worse than generic, unweighted | worse [95 % CI] |")
        p("|---|---|---|---|---|---|")
        for (fam, name), r in Sx["res"].items():
            p(f"| {fam} | {name} | {r['uq']:.2f} | {r['q']:.2f} [{r['q_ci'][0]:.2f}, {r['q_ci'][1]:.2f}] | "
              f"{r['uw']:.2f} | {r['w']:.2f} [{r['w_ci'][0]:.2f}, {r['w_ci'][1]:.2f}] |")
        p("")
        p("| contrast | Q≥1 difference [95 % CI] | verdict (> 0) | worse: comparator − ours |")
        p("|---|---|---|---|")
        for (fam, name), c in Sx["con"].items():
            p(f"| {OURS[fam]} − {name} ({fam}) | {c['d']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] | "
              f"{c['verdict']} | {c['dw']:+.2f} [{c['dw_ci'][0]:+.2f}, {c['dw_ci'][1]:+.2f}] |")
        p("")

    p("## 4. Shares at each room (step e)")
    p("")
    p("KDE reweighting as G1b (weights per simulated user from the ProMP room d_prior − d_floor; "
      "Franks optimizers scored on their own distances with the same weights). Rescaling is "
      "multiplicative and pins the target median in every draw: y_b = (b)[draw] × L / "
      "median((b)[draw]), so κ_b cancels and the level L is a scenario, not an estimate. The anchor "
      "row lets the median float with the participants and κ, so its interval is wider. "
      "Intervals: simulated users within folds and participants resampled jointly, "
      f"{n_boot} draws; differences paired by draw.")
    p("")
    share_table(S["anchor"], "Anchor: Schäfer timing-only room, unrescaled (reproduces §6)",
                anchor=True)
    for name in ("low", "central", "high"):
        share_table(S[name], f"{name.capitalize()} literature room, {levels[name]['L']:.2f} Nm")
    p("### Sensitivity rooms (A and lens only)")
    p("")
    p("| variant | room Nm | ESS | A Q≥1 [CI] | A worse | lens Q≥1 [CI] | smallest ProMP lead | smallest Franks lead |")
    p("|---|---|---|---|---|---|---|---|")
    for k_, Sx in S_sens.items():
        a = Sx["res"][("ProMP", "A")]
        ln = Sx["res"][("Franks", "lens")]
        mp = min(((c["d"], c["lo"], c["hi"], n_) for (f, n_), c in Sx["con"].items() if f == "ProMP"),
                 key=lambda x: x[1])
        mf = min(((c["d"], c["lo"], c["hi"], n_) for (f, n_), c in Sx["con"].items() if f == "Franks"),
                 key=lambda x: x[1])
        p(f"| {k_} | {sens[k_]:.2f} | {Sx['ess']:.0f} | {a['q']:.2f} [{a['q_ci'][0]:.2f}, {a['q_ci'][1]:.2f}] | "
          f"{a['w']:.2f} | {ln['q']:.2f} [{ln['q_ci'][0]:.2f}, {ln['q_ci'][1]:.2f}] | "
          f"{mp[0]:+.2f} [{mp[1]:+.2f}, {mp[2]:+.2f}] ({mp[3]}) | {mf[0]:+.2f} [{mf[1]:+.2f}, {mf[2]:+.2f}] ({mf[3]}) |")
    a = S_prop["res"][("ProMP", "A")]
    ln = S_prop["res"][("Franks", "lens")]
    p(f"| central, room uncertainty propagated (L drawn per draw from its interval) | "
      f"{levels['central']['L']:.2f} [{Lc_ci[0]:.2f}, {Lc_ci[1]:.2f}] | {S_prop['ess']:.0f} | "
      f"{a['q']:.3f} [{a['q_ci'][0]:.3f}, {a['q_ci'][1]:.3f}] | {a['w']:.2f} | "
      f"{ln['q']:.2f} [{ln['q_ci'][0]:.2f}, {ln['q_ci'][1]:.2f}] | — | — |")
    p("")
    p(f"Propagated row: draw b rescales the participant draw part_draws[b] to L_b[b] from the "
      f"central room's interval; L_b (10 logged participants, own generator) and part_draws "
      f"(11 participants) are independent draws paired by index. At fixed L, A's point share "
      f"moves {q_over_ci[0]:.2f}-{q_over_ci[1]:.2f} over the room interval "
      f"[{Lc_ci[0]:.2f}, {Lc_ci[1]:.2f}].")
    p("")

    # ---- f, verdicts
    p("## 5. Verdicts")
    p("")
    p("### P7a. Schäfer's timing spread is wider than the hip timing spreads reported elsewhere")
    p("")
    p("Estimand: the ratio of Schäfer's between-SD (n = 11) to the other study's SD at the matched "
      "landmark; 95 % F interval from both sample sizes; predicted ratio > 1. Planned "
      "comparators: Ramella (preference), Franks hip-only and Ding (metabolic). Ding's 'peak "
      "timing' is the extension peak on a cycle that starts at the fixed onset (maximum hip "
      "flexion, 86.2 % of the heel-strike cycle [verified Din18 p.3]), so the planned mapping "
      "is t2; the rise reading t2 − t1 is shown as an alternative and is not counted. Liu (a hip "
      "preference study found in the sweep), Franks whole-leg and Park (stairs) are reported but "
      "not part of the planned verdict.")
    p("")
    p("| study | objective | landmark | Schäfer SD | other SD (n) | ratio [95 % CI] | verdict | planned |")
    p("|---|---|---|---|---|---|---|---|")
    for x in p7a:
        p(f"| {x['study']} | {x['obj']} | {x['lab']} | {x['s1']:.2f} | {x['sd_used']:.2f} ({x['n']}) | "
          f"{x['ratio']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}] | {x['shown']} | "
          f"{'yes' if x['registered'] else 'no'} |")
    p("")
    p(f"**P7a overall (planned comparators): {p7a_line}.** Ramella's SDs are box-limited "
      "(SD ≤ half the 10 %GC box; its finals are near-uniform on the box), and its t2 comparison "
      "is trivial: the extension peak time is a floor artefact (all 8 within 2.3 %GC of the box "
      "floor, which was also the familiarisation value of 10.0 %GC). Franks's optima were seeded "
      "from one participant (n = 3, the authors). The 'held' rows therefore compare Schäfer with "
      "spreads held down by design and are not a test. Ding's optima are boundary-truncated "
      "('Most of the optimal timings were on the boundaries' [verified Din18 p.3]), so its true "
      "spread is probably larger and the t2 ratio is an upper bound. Liu's hip preference spreads "
      "(n = 5, experienced users, 2-trial means) match Schäfer's.")
    p("")
    p("### P7b. Between-person CV of preferred magnitude lies in 13-30 %")
    p("")
    p("Estimand: one CV per preference study (deflated where both spreads are given, else raw); "
      "interval: χ² 95 % interval of an SD at the study's n, applied to the CV (mean treated as "
      "fixed; for deflated values the same relative factors). Overall rule: held if every study "
      "is held, failed if any study fails, else uncertain.")
    p("")
    p("P7b's text asks for the between-person CV, not a deflated one; the raw CV is printed "
      "beside the one used.")
    p("")
    p("| study | CV raw | CV used | kind | 95 % CI (used) | verdict |")
    p("|---|---|---|---|---|---|")
    for x in p7b:
        p(f"| {x['study']} | {x['raw']:.3f} | {x['v']:.3f} | {x['kind']}"
          f"{', box-truncated' if x['trunc'] else ''} | [{x['lo']:.3f}, {x['hi']:.3f}] | "
          f"{x['verdict']} |")
    p("")
    p("Per parameter (raw / deflated): " + "; ".join(
        f"{s} {prm} {cv:.3f}{' / ' + format(cd, '.3f') if cd is not None else ''}"
        for s, prm, cv, cd in per_params) + ".")
    p("")
    ar = next(x for x in p7b if x["study"].startswith("Arens"))
    s20 = next(x for x in p7b if x["study"] == "Shepherd 2020")
    p(f"**P7b overall: {p7b_line}.** Edge cases: Ramella extension 0.127 (box-limited) and Liu "
      f"flexion 0.066 lie below 13 % as single parameters; Arens LOG raw 0.337 and Arens pooled "
      f"raw {ar['raw']:.3f} lie above 30 % before deflation; Shepherd 2020 raw {s20['raw']:.3f} "
      "lies above 30 % and is not deflated (its 5.6 % is a psychometric width, not retest "
      "scatter).")
    p("")
    p("### P7c. Central room in 1.0-2.31 Nm; A's Q≥1 there in 0.48-0.82; A's lead positive at every room")
    p("")
    p(f"- (i) Central room {Lc:.2f} Nm [{Lc_ci[0]:.2f}, {Lc_ci[1]:.2f}]: **{p7c1}**. "
      f"(i) also {'holds' if p7c1_sens else 'does NOT hold'} across every central sensitivity "
      f"room ({cs_min:.2f}-{cs_max:.2f} Nm).")
    excess = rA["q_ci"][1] - P7C_Q_HIGH
    p(f"- (ii) A's Q≥1 at the central room, at fixed L, {rA['q']:.3f} [{rA['q_ci'][0]:.3f}, "
      f"{rA['q_ci'][1]:.3f}]: **{p7c2}**"
      + (f" (upper end exceeds 0.82 by {excess:.3f})" if excess > 0 else "")
      + f". The verdict is taken at fixed L, as G1b registers; with the room's own uncertainty "
      f"propagated the share is {rP['q']:.3f} [{rP['q_ci'][0]:.3f}, {rP['q_ci'][1]:.3f}] "
      f"({p7c2_prop}); over the room interval the fixed-L point share moves "
      f"{q_over_ci[0]:.2f}-{q_over_ci[1]:.2f}.")
    p(f"- (iii) Lead over every comparator at low, central and high: **{p7c3}**. Reading of "
      "\"A\": ours in both families (lens on Franks), the stricter reading. The literal ProMP-only "
      f"reading ({len(promp_v)} contrasts): {p7c3_promp} (not planned). Contrasts not held: "
      + ("; ".join(f"{OURS[f]} − {n_} at {lv}" for lv, f, n_ in not_held) if not_held else "none")
      + (" (all at the low, box-scenario room)" if not_held and all(lv == "low" for lv, _, _ in not_held)
         else "") + ".")
    for lv in ("low", "central", "high"):
        p(f"  - {lv}: " + "; ".join(f"{OURS[f]} − {n_} {c['d']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] "
                                    f"{c['verdict']}" for (f, n_), c in leads[lv].items()))
    p("")

    # ---- summary (inserted after the verdict rule); text ready for our design notes
    def qa(lv, key=("ProMP", "A")):
        r = S[lv]["res"][key]
        return f"{r['q']:.2f} [{r['q_ci'][0]:.2f}, {r['q_ci'][1]:.2f}]"

    def wa(lv, key=("ProMP", "A")):
        r = S[lv]["res"][key]
        return f"{r['w']:.2f} [{r['w_ci'][0]:.2f}, {r['w_ci'][1]:.2f}]"

    def lead_rng(lv, fam):
        cs = [c for (f, _), c in S[lv]["con"].items() if f == fam]
        return f"{min(c['d'] for c in cs):+.2f} to {max(c['d'] for c in cs):+.2f}"
    def mmc(lv):
        Sx = S_sens[mm[lv]]
        return f"{sens[mm[lv]]:.2f} → {Sx['res'][('ProMP', 'A')]['q']:.2f}"

    sm = [
        "## Summary (text for spec §7.1)",
        "",
        "Rooms are 1-SD (RMS-sum) distances, as planned; the 'median-matched' column gives the "
        "Monte Carlo median of the same spreads and A's Q≥1 there (the like-for-like reading of a "
        "median room). The share intervals are at fixed L (the room is a scenario); see P7c(ii) "
        "for the room's own uncertainty.",
        "",
        "| room | how it is built | Nm (1-SD) | A Q≥1 at fixed L | A worse than generic | lens Q≥1 | A's lead (ProMP) | lens's lead (Franks) | median-matched: Nm → A Q≥1 |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| Schäfer timing-only (anchor, §6; planned lower anchor) | (b) ×κ as measured | "
        f"{med_y0:.2f} (a median) | {qa('anchor')} | "
        f"{wa('anchor')} | {qa('anchor', ('Franks', 'lens'))} | {lead_rng('anchor', 'ProMP')} | "
        f"{lead_rng('anchor', 'Franks')} | — |",
        f"| central | Schäfer deflated timing ⊕ median preference CV {cv_central:.3f} | "
        f"{levels['central']['L']:.2f} [{Lc_ci[0]:.2f}, {Lc_ci[1]:.2f}] | "
        f"{rA['q']:.3f} [{rA['q_ci'][0]:.3f}, {rA['q_ci'][1]:.3f}] | "
        f"{wa('central')} | {qa('central', ('Franks', 'lens'))} | {lead_rng('central', 'ProMP')} | "
        f"{lead_rng('central', 'Franks')} | {mmc('central')} |",
        f"| high | Schäfer raw timing ⊕ single-endpoint CV {cv_high:.3f} | {levels['high']['L']:.2f} | "
        f"{qa('high')} | {wa('high')} | {qa('high', ('Franks', 'lens'))} | {lead_rng('high', 'ProMP')} | "
        f"{lead_rng('high', 'Franks')} | {mmc('high')} |",
        "| *sensitivity, below the planned lower anchor* | | | | | | | | |",
        f"| low: Ramella box scenario (not a bound) | Ramella timing ⊕ Ramella magnitude, both "
        f"box-limited | {levels['low']['L']:.2f} | {qa('low')} | {wa('low')} | "
        f"{qa('low', ('Franks', 'lens'))} | {lead_rng('low', 'ProMP')} | {lead_rng('low', 'Franks')} | "
        f"{mmc('low')} |",
        "",
        f"Headline range, anchor to high: A's Q≥1 from {qa('anchor')} to {qa('high')}. Over the "
        f"central room's interval the fixed-L point share moves {q_over_ci[0]:.2f}-{q_over_ci[1]:.2f}. "
        f"Ramella's finals are near-uniform on its box (a uniform null gives "
        f"{sens['low, uniform-on-box null (Ramella box widths)']:.2f} Nm, A's Q≥1 "
        f"{S_sens['low, uniform-on-box null (Ramella box widths)']['res'][('ProMP', 'A')]['q']:.2f}), "
        "so the low level measures the box widths, not wearers.",
        "",
        f"- **P7a: {p7a_line}.**",
        f"- **P7b: {p7b_line}.**",
        f"- **P7c:** (i) central room {Lc:.2f} [{Lc_ci[0]:.2f}, {Lc_ci[1]:.2f}] Nm in 1.0–2.31: {p7c1}, "
        f"and {'also' if p7c1_sens else 'NOT'} across every central sensitivity room "
        f"({cs_min:.2f}-{cs_max:.2f} Nm); "
        f"(ii) A's Q≥1 there {rA['q']:.3f} [{rA['q_ci'][0]:.3f}, {rA['q_ci'][1]:.3f}] in 0.48–0.82: "
        f"{p7c2}"
        + (f", upper end exceeds 0.82 by {rA['q_ci'][1] - P7C_Q_HIGH:.3f}" if rA['q_ci'][1] > P7C_Q_HIGH else "")
        + f" (room uncertainty propagated: {rP['q']:.3f} [{rP['q_ci'][0]:.3f}, {rP['q_ci'][1]:.3f}]); "
        f"(iii) leads at low, central and high, reading A as ours in both families (lens on "
        f"Franks): {p7c3}; ProMP-only reading (not planned): {p7c3_promp}"
        + ((f"; {'both' if len(not_held) == 2 else 'all'} exceptions at the low (box-scenario) room")
           if not_held and all(lv == "low" for lv, _, _ in not_held) else "") + ".",
        f"- The central room's magnitude half comes from ankle, back and prosthesis studies; with the "
        f"two hip preference studies' magnitude CV ({cv_hip:.3f}) the central room is "
        f"{sens['central, hip-only magnitude (Ramella, Liu)']:.2f} Nm and A's Q≥1 "
        f"{S_sens['central, hip-only magnitude (Ramella, Liu)']['res'][('ProMP', 'A')]['q']:.2f}. "
        f"Read as a median rather than a 1-SD distance (Monte Carlo), the central room is "
        f"{mc['central']:.2f} Nm and A's Q≥1 "
        f"{S_sens['central, median-matched (Monte Carlo median)']['res'][('ProMP', 'A')]['q']:.2f}.",
        "",
    ]
    L[summary_at:summary_at] = sm

    path = OUT / "schaefer_grounding" / "literature_room.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {path}; anchor gate {'ok' if gate_ok else 'FAILED'}")
    print(f"levels: " + ", ".join(f"{k}: T {v['T']:.3f} M {v['M']:.3f} L {v['L']:.3f}"
                                  for k, v in levels.items()))
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
