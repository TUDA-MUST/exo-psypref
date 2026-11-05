"""The ICORR reframe's grid: blocks ``e-k``, ``e-home``, ``e-ood`` and ``e-rand``.

Every session is e-main's wearer and users: the five leave-one-subject-out folds with all
exported users (40 population and 4 stress each), s = 1.0, lapse 0.1, the data prior, the tight
arm and 40 comparisons, read at k = 12, 24 and 40 by truncation. The cells are what e-main
does not already hold:

- W3, basis count: A (EIG duels) at K = 4 and 12, and G and SortCMA at K = 4 and 12. Their
  K = 8 cells are e-main's own rows (section 9a: "K = 8 reuses e-main bit for bit").
- W3, the random-duel ablation: A with random duels at K = 4, 8 and 12.
- W5: C2, Ramella's learner on ProMP (K = 8), at s = 1.0.
- C1, Ramella as published, in her own semi-trapezoid box (flat prior, no arm), at s = 1.0.

The robustness blocks of section 9a:

- ``e-home``, the home-advantage test at matched decisiveness: e-observer's two landmark cells
  (static 10 % and walking 25 % magnitude JND) with the utility divided by the cell's measured
  decisiveness ratio (:data:`DECISIVENESS`), on e-observer's six optimizer cells and users.
- ``e-ood``, out-of-generator ideals: Schaefer four-half-cosine users (``user_set="ood"``,
  5 folds x 40), the plain RMS wearer at s = 1.0, on :data:`ROBUST_CELLS`. One common peak for
  both bouts; after the 2026-10-19 decision the labelled sensitivity run.
- ``e-ood-pb``, the ruled primary O analysis: the same users, timings and seeds with each bout's
  own population-median peak (``user_set="ood-pb"``), on the same cells.
- ``e-rand``, random smooth utilities: e-main's 200 population users with the random smooth
  utility (``observer_metric="random_smooth"``), on :data:`ROBUST_CELLS`.

The calibration test of section 9a:

- ``e-cal``: e-main's A (EIG duels) sessions on its 200 population users, rerun with
  ``calibration_log=True``. The logging draws no random numbers, so every session must reproduce
  its stored e-main row in every stored field.

The phase-weighted wearers of section 9a ("Phase-weighted wearers", PW-1 and PW-2; decision of 2026-10-19: W-pref runs now, W-bio when its data can be read):

- ``e-wph-pref``: e-main's 200 population users, each judging by the phase-weighted RMS with its
  own random periodic weight (``phase_weighting="pref"``), on :data:`ROBUST_CELLS`;
- ``e-wph-bio``: the same with the common weight from the mean hip angular velocity
  (``phase_weighting="bio"``). It refuses to build without ``out/icorr/wph_bio_weight.json``.

Both need ``scripts/icorr_wph_prep.py --cell <cell>`` first (the per-user scales and floors).

The review response of section 9a ("Review response: comparator fairness, ablations and a sanity
check", PF1, PF2, PA1, PA2 and PS), on e-main's 200
population users, ProMP, 40 comparisons:

- ``e-rev-fair``, first: the sanity check PS, Ramella's learner C2 under an easy wearer (s = 0.5,
  lapse 0); SortCMA with pycma popsize 4 and 6; G with a generic start, with the true-JND gate
  (jnd_scale 1.0) and without a gate (jnd_scale None). Everything else as in e-main.
- ``e-rev-abl``, second: A in the existing wide arm (z* ~ N(0, 9 I)), and A without lapse
  (lambda fixed at 0 in A's likelihood, probes off) against the lapse-0.1 wearer.

- ``e-rev-abl2``: A without lapse
  against the lapse-0.3 wearer, and A without probes (lambda still inferred, p_probe 0) against the
  lapse-0.1 and lapse-0.3 wearers. A itself at lapse 0.3 is e-lapse's stored cell.

The same review response's preference GPs: two preference GPs built from BoTorch, one PairwiseGP asked by analytic EUBO (GP-EUBO, learner
``gp_eubo``, PF3) or by BoTorch's pairwise BALD (GP-BALD, learner ``gp_bald``, PF3b);
:mod:`prompref.learning.gp_eubo`, with botorch installed:

- ``e-gpeubo`` and ``e-gpbald``: e-main's 200 population users in the main condition, the plain
  RMS wearer;
- ``<block>-rand``, ``-ood-pb``, ``-wph-pref`` and ``-wph-bio``: the same cell under the R, O
  (per-bout peaks, the ruled primary set) and W wearers, on the users and sidecars those blocks
  already use ("if time remains").

The higher-parameter test of section 9a:

- ``e-k-hi``: e-main's 200 population users in the main condition at K = 16 and K = 24, each K
  with its own data prior, floors and arm radius (``scripts/icorr_k_priors.py --k 16,24``): A
  (EIG duels), A with random duels, G, LineCoSpar and SortCMA at pycma's default population.
  GP-BALD at K = 16 and 24 is block ``e-gpbald-k16`` and ``e-gpbald-k24`` (:data:`GP_K_BLOCKS`).

The stress tests of section 9a, on e-main's 200 population users in the main condition (s = 1.0,
lapse 0.1, data prior, tight arm, 40 comparisons), each assumption of the simulated user violated on
purpose and alone:

- ``e-stress``: PsyPref (A), Arens' GP (G), LineCoSpar and CMA-ES (SortCMA) in every stress cell of
  :data:`STRESS_CELLS` (S1 strong anisotropy, S3 fatigue, S4a/S4b changing sensitivity, S5a/S5b
  link, S6 order bias, S7 "no preference" by the hard rule and S7s by our design notes's soft band), and
  PsyPref-M (``a_metric``) under S1, W-pref and the main condition (:data:`METRIC_CELLS`). S1 needs
  ``scripts/icorr_wph_prep.py --cell pref-sd1`` first, PsyPref-M under W-pref the W-pref sidecars.

The acquisition variants of section 9a:

- ``e-acq``: e-main's 200 population users in the main condition at K = 8, 16 and 24, each K with
  its own data prior, floors and arm radius: A-curve, A-width and A-sub
  (``acq_variant`` "curve", "width", "sub"). A itself is e-main's stored cell at K = 8 and
  e-k-hi's at K = 16 and 24, so it is not repeated. The S1 cell (W-pref with log-w SD 1.0) waits
  for the stress lane's sidecars.

The response-time test and the population hyperprior of section 9a, on e-main's 200 population users in the main
condition with 40 comparisons:

- ``e-rt``: the wearer answers by a drift-diffusion model (:mod:`prompref.user.ddm`) with informative
  response times (RT-inf) or with response times drawn from the marginal (RT-null), each with
  PsyPref-RT (``a_rt``) and PsyPref (A). Needs ``scripts/icorr_rt_prep.py`` first.
- ``e-hyper-1``: H2's first pass, A's own sessions at lambda 0.1 (e-main) and 0.3 (e-lapse) with
  ``hyper_log``; each must reproduce its stored row. ``scripts/icorr_hyper_fit.py`` then fits the
  hyperpriors leave-fold-out.
- ``e-hyper``: PsyPref-hyper at lambda 0.1 and 0.3, each fold with the fit on the other four.

Test P of section 9a ("Interpretable families and how many parameters a profile needs",
PP1-PP3), on e-main's 200 population users in the main
condition:

- ``e-proj``: PsyPref on the ProMP (K = 8) with each proposal projected onto Schaefer's family with
  free peaks (``schaefer_peaks``) and onto Ramella's semi-trapezoid (``trapezoid_surface``)
  (:data:`PROJ_CELLS`). The Franks projection is e-main's stored cell. Needs
  ``scripts/icorr_proj_prep.py`` first (the surface priors and the Schaefer (ii) floors).
- ``e-proj-pchip``:
  the same with each proposal projected onto Park et al. 2025's PCHIP profile adapted to level
  walking (``pchip``); needs ``scripts/icorr_proj_prep.py --pchip`` first.

The second review round of section 9a, K = 8, 40 comparisons, the main condition unless
stated; every user set needs ``scripts/icorr_r2_prep.py --step <set>`` first
(:mod:`prompref.experiment.r2`):

- ``e-xd``: XD-Fy, XD-Fo and XD-R, 200 users each from Fukuchi 2018 and Reznick 2021, the Molinaro
  prior round-robin over the folds; PsyPref, Arens' GP, LineCoSpar and CMA-ES (:data:`XD_CELLS`).
- ``e-ht``: HT (a) the generator at nu = 1.5, (b) 80 % population users and 20 % Schaefer (ii) and
  Franks ideals, (c) Cauchy answers matched as S5 (:data:`R2_LINK_SCALES`); the same learners.
- ``e-pess``: the pessimistic combined user (:data:`PESS_CELL`, ``phase5.PESS_FIELDS``) on the
  "ood-pb" users with S1's phase weights; the same learners. ``e-pess-gp``: GP-BALD there, if the time box allows.
- ``e-base-cma`` and ``e-base-gp``: CMA-ES (popsize 4) and GP-BALD searching Schaefer (ii)'s six
  parameters in its own prior space; each first runs its three settings on :data:`BASE_TUNE_FOLD`,
  then, once ``out/icorr/base_tuned.json`` holds its choice, that setting on the other folds.
"""

from __future__ import annotations

from ..user.ddm import load_rt_params
from ..user.phase_weight import load_bio_weight
from .hyper import HYPER_LAPSES, hyper_fields
from .phase5 import C1_ARM, C1_FAMILY, Phase5Point
from .phase5c_grid import G_JND_SCALE, LOSO, _user_ids
from .phase5e_grid import CORE, WALKING_LANDMARK_FRACTION
from .phase_weighted import PREP_SCRIPT, phase_weight_path
from .robust import OOD_PER_BOUT_USER_SET, OOD_USER_SET, load_ood_users, random_utility_path

#: The BoTorch GP blocks (section 9a, review response; PF3 GP-EUBO, PF3b GP-BALD): block ->
#: (learner, the existing block whose wearer, users and sidecars it shares; ``None`` is the main
#: condition's plain RMS wearer on e-main's 200 population users).
GP_LEARNER_BLOCK = {"gp_eubo": "e-gpeubo", "gp_bald": "e-gpbald"}
GP_SHARED = {"": None, "-rand": "e-rand", "-ood-pb": "e-ood-pb", "-wph-pref": "e-wph-pref",
             "-wph-bio": "e-wph-bio"}
GP_BLOCKS = {stem + suffix: (learner, shared) for suffix, shared in GP_SHARED.items()
             for learner, stem in GP_LEARNER_BLOCK.items()}
#: Section 9a, "More parameters: K = 16 and 24": GP-BALD at K = 16,
#: with the K = 16 prior, floors and arm radius chi_16,0.99 (``scripts/icorr_k_priors.py``, run on
#:; its A cell is's block e-k-hi. block -> (learner, n_basis).
#: PA-c2: GP-BALD at
#: K = 24, the same way, against A-curve at K = 24.
GP_K_BLOCKS = {"e-gpbald-k16": ("gp_bald", 16), "e-gpbald-k24": ("gp_bald", 24)}
#: R2 / G-s2 (section 9a, second review round, PR2-1; or
#: run): GP-BALD at s = 2.0, K = 8, the main condition otherwise; its PsyPref cell is the stored
#: e-s block (out/phase5e/e-s.jsonl, A on promp_s0 at s = 2.0). block -> (learner, s).
GP_S_BLOCKS = {"e-gpbald-s2": ("gp_bald", 2.0)}
#: The second review round's blocks:
#: XD, HT (a)-(c), PESS (the fast learners, then GP-BALD in its own block) and BASE (CMA-ES and
#: GP-BALD, each tuned on one fold first). See :func:`r2_cells`.
R2_BLOCKS = ("e-xd", "e-ht", "e-pess", "e-pess-gp", "e-base-cma", "e-base-gp")
BLOCKS_ICORR = ("e-k", "e-home", "e-ood", "e-rand", "e-ood-pb", "e-cal", "e-wph-pref",
                "e-wph-bio", "e-rev-fair", "e-rev-abl", "e-rev-abl2", "e-k-hi",
                "e-stress", "e-acq", "e-rt", "e-hyper-1", "e-hyper", "e-proj",
                "e-proj-pchip") + tuple(GP_BLOCKS) + tuple(GP_K_BLOCKS) + tuple(GP_S_BLOCKS) \
    + R2_BLOCKS
#: The phase-weighted blocks and their cells (``Phase5Point.phase_weighting``).
WPH_BLOCKS = {"e-wph-pref": "pref", "e-wph-bio": "bio"}
ARM = "tight"
N_COMPARISONS = 40
S_MAIN = 1.0
#: W3's basis counts beside the run's own 8 (``None`` in a point).
K_VALUES = (4, 12)

#: (family, learner, extra point fields), in the order each fold runs them: W3's A cells
#: first, then the cheap comparators, then the slow batch learners.
CELLS = (
    [("promp_s0", "a", dict(n_basis=k)) for k in K_VALUES]
    + [("promp_s0", "a", dict(duel_mode="random", n_basis=k)) for k in (4, None, 12)]
    + [("promp_s0", learner, dict(n_basis=k)) for learner in ("g", "cmaes") for k in K_VALUES]
    + [("promp_s0", "c2", {}),
       (C1_FAMILY, "c1", dict(arm=C1_ARM, prior="flat"))]
)

#: The decisiveness ratio of each landmark cell (keyed by its magnitude fraction; ``None`` is
#: the static 10 % cell), at full precision: ``scripts/phase5e_perturbation.py``'s
#: ``compute_decisiveness`` on 5 folds x 20 population users x 200 prior-scale pairs, the
#: Phase 5e REPORT section 9 values (1.96 and 1.59), re-measured bit for bit before e-home
#: was built (``out/icorr/robust_decisiveness.json``).
DECISIVENESS = {None: 1.9630404137610769, WALKING_LANDMARK_FRACTION: 1.5875969510262529}

#: Section 9a, O and R: A (EIG duels), A with random duels, G, LineCoSpar and SortCMA, on ProMP.
ROBUST_CELLS = [("promp_s0", "a", {}), ("promp_s0", "a", dict(duel_mode="random")),
                ("promp_s0", "g", {}), ("promp_s0", "d_z", {}), ("promp_s0", "cmaes", {})]

#: The review response: (family, learner, extra point fields)
#: per cell, in the order each fold runs them. PS's easy wearer comes first in ``e-rev-fair``.
#: A G cell that states ``jnd_scale`` keeps it (None is the gate off); every other G cell gets
#: e-main's true-JND gate.
REVIEW_BLOCKS = {
    "e-rev-fair": [("promp_s0", "c2", dict(s=0.5, lapse=0.0)),
                   ("promp_s0", "cmaes", dict(cma_popsize=4)),
                   ("promp_s0", "cmaes", dict(cma_popsize=6)),
                   ("promp_s0", "g", dict(g_start="generic", jnd_scale=1.0)),
                   ("promp_s0", "g", dict(g_start="generic", jnd_scale=None))],
    "e-rev-abl": [("promp_s0", "a", dict(arm="wide")),
                  ("promp_s0", "a", dict(a_no_lapse=True))],
    # PA3 and PA4. A itself at lapse 0.3 is
    # e-lapse's stored cell, so it is not repeated here.
    "e-rev-abl2": [("promp_s0", "a", dict(a_no_lapse=True, lapse=0.3)),
                   ("promp_s0", "a", dict(a_no_probes=True)),
                   ("promp_s0", "a", dict(a_no_probes=True, lapse=0.3))],
}


#: Section 9a, "More parameters": the basis counts of block e-k-hi.
K_HI_VALUES = (16, 24)
#: Its cells per K, in the order each fold runs them: A (EIG duels), A with random duels, G,
#: LineCoSpar and SortCMA (pycma's default population: ``cma_popsize`` left at None).
K_HI_CELLS = [("promp_s0", learner, dict(extra, n_basis=k))
              for k in K_HI_VALUES
              for learner, extra in (("a", {}), ("a", dict(duel_mode="random")), ("g", {}),
                                     ("d_z", {}), ("cmaes", {}))]


#: Section 9a, stress test S5: each link's scale, matched so that the median
#: |p - 1/2| over REPORT section 9's prior-scale pairs of the 200 population users equals the plain
#: s = 1.0 probit wearer's, at full precision (``scripts/icorr_stress_prep.py`` ->
#: ``out/icorr/stress_prep_links.json``; achieved ratio 1 on those pairs, 1.0002 and 1.0004 on fresh
#: pairs).
LINK_SCALES = {"logistic": 0.5916008501683337, "student_t": 0.8100468122288004}

#: Section 9a, stress tests: cell id -> the point fields that violate one
#: assumption of the simulated user. Everything else is the main condition.
STRESS_CELLS = {
    "S1": dict(phase_weighting="pref-sd1"),
    "S3": dict(lapse_start=0.05, lapse_end=0.30),
    "S4a": dict(s_start=1.5, s_end=0.75),
    "S4b": dict(s_start=0.75, s_end=1.5),
    "S5a": dict(link="logistic", link_scale=LINK_SCALES["logistic"]),
    "S5b": dict(link="student_t", link_scale=LINK_SCALES["student_t"]),
    "S6": dict(order_bias=0.3),
    "S7": dict(no_pref_jnd=0.5, no_pref_hard=True),
    "S7s": dict(no_pref_jnd=0.5),
}
#: The learners of every stress cell: PsyPref, Arens' GP, LineCoSpar, CMA-ES.
STRESS_LEARNERS = ("a", "g", "d_z", "cmaes")
#: PsyPref-M's cells.
METRIC_CELLS = {"S1": dict(phase_weighting="pref-sd1"), "W-pref": dict(phase_weighting="pref"),
                "main": {}}


#: Test P (section 9a, "Interpretable families and how many parameters a profile needs",
#:): PsyPref on the ProMP (K = 8) with each proposal
#: projected onto Schaefer's family with free peaks (ii) and onto Ramella's semi-trapezoid. The
#: Franks projection is e-main's stored a_lens cell (same condition); the PCHIP family of the
#: test plan is not built (its parameterization could not be found in Park & Collins).
PROJ_CELLS = [("schaefer_peaks", "a_lens", {}), ("trapezoid_surface", "a_lens", {})]
#: The PCHIP projection (design correction of 2026-10-22, 4509e78 on, before any PCHIP
#: code or run): Park et al. 2025's PCHIP profile adapted to level walking.
PROJ_PCHIP_CELLS = [("pchip", "a_lens", {})]
PROJ_BLOCKS = {"e-proj": PROJ_CELLS, "e-proj-pchip": PROJ_PCHIP_CELLS}


# ------------------------------------------------------------------ R2, the second review round
#: The fast learners of every R2 cell: PsyPref (A), Arens' GP (G), LineCoSpar, CMA-ES.
R2_LEARNERS = ("a", "g", "d_z", "cmaes")
#: XD: one user set per cell.
XD_CELLS = {"XD-Fy": "xd-fy", "XD-Fo": "xd-fo", "XD-R": "xd-r"}
#: HT's Cauchy link scale, matched as S5's (``scripts/icorr_r2_prep.py --step links`` ->
#: ``out/icorr/r2_prep_links.json``), at full precision.
R2_LINK_SCALES = {"cauchy": 0.4799604532549037}
#: HT: (a) the heavy-tailed generator, (b) the mixture, (c) Cauchy answers.
HT_CELLS = {"HT-a": dict(user_set="ht-gen"), "HT-b": dict(user_set="ht-mix"),
            "HT-c": dict(link="cauchy")}
#: PESS: every violation together (``phase5.PESS_FIELDS``).
PESS_CELL = dict(s=2.0, lapse_start=0.05, lapse_end=0.30, order_bias=0.3, no_pref_jnd=0.5,
                 no_pref_hard=True, phase_weighting="pref-sd1", user_set="ood-pb")
#: BASE: each method gets 3 settings on one fold (as section E8's comparators got
#: 3-9), the setting with the lowest median error at k = 24 on that fold's 40 population users is
#: then run on the other folds. CMA-ES keeps its planned population of 4 and tunes E8's sigma
#: multipliers; GP-BALD tunes its search region, the arm (the ball radius and the prior its pool is
#: drawn from). The choice is written to ``out/icorr/base_tuned.json`` by
#: ``scripts/icorr_r2_prep.py --step base-select``.
BASE_TUNE_FOLD = "loso-AB30"
BASE_SETTINGS = {"cmaes": [dict(cma_popsize=4, cma_sigma_mult=m) for m in (0.5, 1.0, 2.0)],
                 "gp_bald": [dict(arm=a) for a in ("tight", "t4", "wide")]}
BASE_BLOCK_LEARNER = {"e-base-cma": "cmaes", "e-base-gp": "gp_bald"}
BASE_TUNED_NAME = "base_tuned.json"


def base_tuned_path(run_dir) -> "Path":
    from pathlib import Path

    return Path(run_dir).parent / "icorr" / BASE_TUNED_NAME


def load_base_tuned(run_dir) -> dict:
    """``out/icorr/base_tuned.json``: learner -> its chosen setting (absent before the selection)."""
    import json

    path = base_tuned_path(run_dir)
    if not path.exists():
        return {}
    tuned = json.loads(path.read_text(encoding="utf-8"))
    for learner, entry in tuned.items():
        if learner not in BASE_SETTINGS or entry["setting"] not in BASE_SETTINGS[learner]:
            raise ValueError(f"{path}: {learner}'s setting {entry.get('setting')} is not one of "
                             f"its registered settings")
    return tuned


def r2_specs(block: str, run_dir: str, fold: str) -> list[tuple[str, str, dict, list[str]]]:
    """(family, learner, extra fields, user ids) per cell of one fold of an R2 block."""
    from .phase5 import BASE_FAMILY

    if block == "e-xd":
        specs = []
        for user_set in XD_CELLS.values():
            uids = list(load_ood_users(run_dir, fold, user_set))
            specs += [("promp_s0", learner, dict(user_set=user_set), uids) for learner in R2_LEARNERS]
        return specs
    if block == "e-ht":
        specs = []
        for cid, extra in HT_CELLS.items():
            if "user_set" in extra:
                uids = list(load_ood_users(run_dir, fold, extra["user_set"]))
            else:
                if R2_LINK_SCALES[extra["link"]] is None:
                    raise ValueError("HT (c) needs its matched link scale (icorr_r2_prep.py --step "
                                     "links) in icorr_grid.R2_LINK_SCALES")
                extra = dict(extra, link_scale=R2_LINK_SCALES[extra["link"]])
                uids = _population_ids(run_dir, fold)
            specs += [("promp_s0", learner, dict(extra), uids) for learner in R2_LEARNERS]
        return specs
    if block in ("e-pess", "e-pess-gp"):
        path = phase_weight_path(run_dir, PESS_CELL["phase_weighting"], fold, PESS_CELL["user_set"])
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run scripts/icorr_r2_prep.py --run "
                                    f"{run_dir} --step pess first")
        uids = list(load_ood_users(run_dir, fold, PESS_CELL["user_set"]))
        learners = R2_LEARNERS if block == "e-pess" else ("gp_bald",)
        return [("promp_s0", learner, dict(PESS_CELL), uids) for learner in learners]
    if block in BASE_BLOCK_LEARNER:
        learner = BASE_BLOCK_LEARNER[block]
        uids = _population_ids(run_dir, fold)
        if fold == BASE_TUNE_FOLD:
            settings = BASE_SETTINGS[learner]
        else:
            tuned = load_base_tuned(run_dir).get(learner)
            settings = [] if tuned is None else [tuned["setting"]]
        return [(BASE_FAMILY, learner, dict(s), uids) for s in settings]
    raise ValueError(block)


def stress_cells() -> list[tuple[str, str, str, dict]]:
    """(cell id, family, learner, extra point fields) of block e-stress, in the order each fold runs
    them: the slow PsyPref-M and PsyPref cells first, then the comparators."""
    cells = [(cid, "promp_s0", "a", dict(extra, a_metric=True)) for cid, extra in METRIC_CELLS.items()]
    for learner in STRESS_LEARNERS:
        cells += [(cid, "promp_s0", learner, dict(extra)) for cid, extra in STRESS_CELLS.items()]
    return cells


#: Section 9a, part A: the variants and basis counts of block e-acq.
#: ``None`` is the run's own K = 8.
ACQ_VARIANT_NAMES = ("curve", "width", "sub")
ACQ_K_VALUES = (None, 16, 24)
#: Its cells, in the order each fold runs them: K = 8 first, then 16 and 24.
ACQ_CELLS = [("promp_s0", "a", dict(acq_variant=v) if k is None else dict(acq_variant=v, n_basis=k))
             for k in ACQ_K_VALUES for v in ACQ_VARIANT_NAMES]


#: Section 9a, "RT": cell id -> the DDM wearer, and each cell's learners in
#: the order a fold runs them: PsyPref-RT (the slower) first, then PsyPref (A).
RT_CELLS = {"RT-inf": dict(rt_user="inf"), "RT-null": dict(rt_user="null")}
RT_LEARNERS = {"a_rt": dict(a_rt=True), "a": {}}


def rt_cells() -> list[tuple[str, str, dict]]:
    """(cell id, learner key, extra point fields) of block e-rt."""
    return [(cid, key, dict(wearer, **extra)) for cid, wearer in RT_CELLS.items()
            for key, extra in RT_LEARNERS.items()]


def _population_ids(run_dir: str, fold: str) -> list[str]:
    from .phase5 import _users

    return [uid for uid, user in _users(run_dir, fold).items() if user.kind == "population"]


def _specs(block: str, run_dir: str, fold: str) -> list[tuple[str, str, dict, list[str]]]:
    """(family, learner, extra fields, user ids) per cell of one fold."""
    if block in GP_S_BLOCKS:
        # The main condition on e-main's 200 population users, at the block's wearer sensitivity s.
        learner, s_cell = GP_S_BLOCKS[block]
        return [("promp_s0", learner, dict(s=s_cell), _population_ids(run_dir, fold))]
    if block in GP_K_BLOCKS:
        # The main condition on e-main's 200 population users, at the block's basis count.
        learner, k = GP_K_BLOCKS[block]
        return [("promp_s0", learner, dict(n_basis=k), _population_ids(run_dir, fold))]
    if block in GP_BLOCKS:
        # A GP cell in the main condition, or with the wearer, users and sidecars of the
        # robustness block it shares: that block's first cell (A, EIG duels, which carries the
        # wearer's fields alone) with its learner replaced.
        learner, shared = GP_BLOCKS[block]
        if shared is None:
            return [("promp_s0", learner, {}, _population_ids(run_dir, fold))]
        family, first, fields, uids = _specs(shared, run_dir, fold)[0]
        assert (family, first) == ("promp_s0", "a") and "duel_mode" not in fields
        return [(family, learner, dict(fields), uids)]
    if block == "e-k":
        uids = _user_ids(run_dir, fold)
        return [(family, learner, dict(extra), uids) for family, learner, extra in CELLS]
    if block == "e-home":
        uids = _user_ids(run_dir, fold)
        specs = []
        for fraction, ratio in DECISIVENESS.items():
            wearer = dict(observer_metric="landmark", decisiveness_ratio=ratio)
            if fraction is not None:
                wearer["landmark_mag_fraction"] = fraction
            specs += [(family, learner, dict(wearer), uids) for family, learner in CORE]
        return specs
    if block in ("e-ood", "e-ood-pb"):
        user_set = OOD_USER_SET if block == "e-ood" else OOD_PER_BOUT_USER_SET
        uids = list(load_ood_users(run_dir, fold, user_set))
        return [(family, learner, dict(extra, user_set=user_set), uids)
                for family, learner, extra in ROBUST_CELLS]
    if block == "e-rand":
        path = random_utility_path(run_dir, fold)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run scripts/icorr_robust_prep.py "
                                    f"--run {run_dir} --step random first")
        uids = _population_ids(run_dir, fold)
        return [(family, learner, dict(extra, observer_metric="random_smooth"), uids)
                for family, learner, extra in ROBUST_CELLS]
    if block == "e-cal":
        # e-main's A cell (ProMP, data prior, tight arm, s = 1.0, 40 comparisons), population
        # users only, with the calibration logging on.
        return [("promp_s0", "a", dict(calibration_log=True), _population_ids(run_dir, fold))]
    if block in WPH_BLOCKS:
        cell = WPH_BLOCKS[block]
        if cell == "bio":
            # W-bio has no fallback weighting: no weight file, no block.
            load_bio_weight(run_dir)
        path = phase_weight_path(run_dir, cell, fold)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run {PREP_SCRIPT} --run {run_dir} "
                                    f"--cell {cell} first")
        uids = _population_ids(run_dir, fold)
        return [(family, learner, dict(extra, phase_weighting=cell), uids)
                for family, learner, extra in ROBUST_CELLS]
    if block in REVIEW_BLOCKS:
        uids = _population_ids(run_dir, fold)
        return [(family, learner, dict(extra), uids) for family, learner, extra in REVIEW_BLOCKS[block]]
    if block == "e-k-hi":
        # Section 9a names the 200 population users (e-k ran e-main's 220, stress users too).
        uids = _population_ids(run_dir, fold)
        return [(family, learner, dict(extra), uids) for family, learner, extra in K_HI_CELLS]
    if block == "e-stress":
        for cell in ("pref-sd1", "pref"):
            path = phase_weight_path(run_dir, cell, fold)
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing: run {PREP_SCRIPT} --run {run_dir} "
                                        f"--cell {cell} first")
        uids = _population_ids(run_dir, fold)
        return [(family, learner, dict(extra), uids) for _, family, learner, extra in stress_cells()]
    if block == "e-acq":
        uids = _population_ids(run_dir, fold)
        return [(family, learner, dict(extra), uids) for family, learner, extra in ACQ_CELLS]
    if block == "e-rt":
        # The DDM wearer's marginal and constants must exist before any point is built.
        load_rt_params(str(run_dir))
        uids = _population_ids(run_dir, fold)
        return [("promp_s0", "a", dict(extra), uids) for _, _, extra in rt_cells()]
    if block == "e-hyper-1":
        # H2's first pass: A's own sessions of e-main (lambda 0.1) and e-lapse (0.3), with logging.
        uids = _population_ids(run_dir, fold)
        return [("promp_s0", "a", dict(lapse=lapse, hyper_log=True), uids)
                for lapse in HYPER_LAPSES]
    if block == "e-hyper":
        uids = _population_ids(run_dir, fold)
        return [("promp_s0", "a", dict(lapse=lapse, **hyper_fields(str(run_dir), lapse, fold)),
                 uids) for lapse in HYPER_LAPSES]
    if block in PROJ_BLOCKS:
        from .phase5 import PROJ_SCRIPT, _proj_floor_path
        from ..priors.surface_prior import surface_prior_path

        cells = PROJ_BLOCKS[block]
        for family, _, _ in cells:
            for path in (surface_prior_path(run_dir, fold, family),
                         _proj_floor_path(run_dir, fold, family)):
                if not path.exists():
                    raise FileNotFoundError(f"{path} is missing: run {PROJ_SCRIPT} --run "
                                            f"{run_dir} first")
        uids = _population_ids(run_dir, fold)
        return [(family, learner, dict(extra), uids) for family, learner, extra in cells]
    if block in R2_BLOCKS:
        return r2_specs(block, run_dir, fold)
    raise ValueError(block)


def block_points(block: str, run_dir: str, *, folds=LOSO, arm=None,
                 n_comparisons: int = N_COMPARISONS, **_ignored) -> list[Phase5Point]:
    """The points of one block, fold by fold. ``arm`` is the runner's attribution-follow-up
    option and must stay ``None`` here; ``n_comparisons`` defaults to 40."""
    if block not in BLOCKS_ICORR:
        raise ValueError(f"block must be one of {BLOCKS_ICORR}, got {block!r}")
    if arm is not None:
        raise ValueError("the ICORR blocks state their own arms: tight, and C1's box")
    points = []
    for fold in folds:
        for family, learner, extra, uids in _specs(block, run_dir, fold):
            fields = dict(extra)
            cell_arm = fields.pop("arm", ARM)
            # Only the review response's PS cell, G-s2 and R2's PESS state their own s, and only the
            # review response's G cells their gate. BASE's GP-BALD settings state their arm.
            cell_s = fields.pop("s", S_MAIN)
            g = {"jnd_scale": G_JND_SCALE} if learner == "g" and "jnd_scale" not in fields else {}
            points += [Phase5Point(run_dir, fold, family, cell_arm, learner, uid, s=cell_s,
                                   n_comparisons=int(n_comparisons), **fields, **g)
                       for uid in uids]
    return points
