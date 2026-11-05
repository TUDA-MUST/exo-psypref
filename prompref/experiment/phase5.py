"""One Phase 5 session: a held-out user, a fold's data prior, an arm and a learner.

design sections 4 and 5.1. A session needs two things from a Phase 5a run directory:
- the fold's fitted prior, ``RUN/priors/<fold>-<family>``;
- the exported users, ``RUN/users/<fold>``, written by ``scripts/phase5_export_users.py``.

No session touches the Molinaro drive.

Pairing: a user's answer seed drives the observer in every arm and for every
learner. The observer's stream is salted so no learner shares its random numbers; in Phase 4, C2's batch and the observer both drew from ``default_rng(seed)``. Learners
take the answer seed modulo 2^32, because C2's wrapped learner seeds numpy's legacy
generator, which accepts nothing larger.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..basis import DEFAULT_N_BASIS
from ..designs.device import DeviceDesign
from ..designs.promp import PrompS0
from ..designs.schaefer import SchaeferDesign
from ..designs.trapezoid import TrapezoidDesign
from ..learning.acq_variants import VARIANTS as ACQ_VARIANTS
from ..learning.arms import ARMS
from ..learning.batch_mi import BatchMILearner
from ..learning.gp_ucb import KAPPA_MAX, N_CANDIDATES
from ..learning.lens import LensLearner, SurfaceFitter
from ..learning.linecospar import LENGTHSCALE, STEP
from ..learning.psychometric import DUEL_MODES, PsychometricLearner
from ..learning.space import LearnerSpace, SpacePriorDesign
from ..learning.whitened import WhitenedLearner
from ..population.export import load_fold_users
from ..prior import Whitening
from ..priors.basis_prior import (BASIS_SCRIPT, BasisDataPrior, basis_floors_path,
                                  basis_prior_stem)
from ..priors.data_prior import FAMILIES, DataPrior
from ..priors.franks_prior import FranksPrior
from ..priors.surface_prior import SurfacePrior, surface_prior_path
from ..priors.timing_prior import TimingPrior
from ..user.ddm import RT_MODES, DDMObserver, load_rt_params, pyddm_version
from ..user.drift import DriftingObserver, path_for_scale, path_for_shift
from ..user.jnd import JND_SCALES, gate_value, true_jnd
from ..user.oracle import (LANDMARK_MAG_FRACTION, METRICS, PHASE_WEIGHTED, RANDOM_SMOOTH,
                           Observer, rms_distance)
from ..user.phase_weight import PHASE_WEIGHTINGS, PREF_CELLS, load_bio_weight, pref_weight
from ..user.schedule import ScheduledObserver, linear_path
from .calibration import CalibrationMonitor
from .hyper import posterior_sample
from .landmarks import two_bout_plausible
from .metrics import prior_mean_curve, prior_mean_distance
from .phase_weighted import load_phase_weight_sidecar, sidecar_entry
from .review import G_STARTS, NoLapseLearner, generic_start_fields, generic_start_g
from .robust import (OOD_USER_SET, USER_SETS, load_ood_users, load_random_utility,
                     user_utility)
from .rt import RTLearner
from .run import run_session
from .stress import MetricLearner
from .sweep import run_sweep

#: our design notes's learners. G arrived with plan 5b-ii; C1 -- Ramella as published, on her
#: semi-trapezoid box and with no arm -- with Phase 5c.
LEARNERS = ("a", "d_z", "c2", "g", "c1", "a_lens", "cmaes")

#: Opt-in learners, kept out of :data:`LEARNERS` so every loop over the established learners
#: (smoke runs, grids, tests) is unchanged. "gp_eubo" and "gp_bald" are the ICORR review
#: response's preference GPs, built from BoTorch
#: (:mod:`prompref.learning.gp_eubo`): one model, asked by analytic EUBO or by BoTorch's pairwise
#: BALD. They need and run only on promp_s0 with its data prior, whose
#: whitened coordinates their bowl mean is defined in.
OPT_IN_LEARNERS = ("gp_eubo", "gp_bald")
#: Each GP learner's acquisition (:data:`prompref.learning.gp_eubo.ACQUISITIONS`).
GP_ACQUISITION = {"gp_eubo": "eubo", "gp_bald": "bald"}

#: C1's family and its stand-in for an arm. Only C1 runs them, and C1 runs nothing else.
C1_FAMILY = "trapezoid"
C1_ARM = "box"

#: R35's prior factor. "flat" is the family's own box, with no population knowledge.
PRIORS = ("data", "flat")

#: The designs a flat cell searches: the family's own defaults, not the
#: fold's data-prior design, whose bounds already carry population knowledge.
FLAT_DESIGNS = {"promp_s0": PrompS0, "schaefer": SchaeferDesign}

#: Families a Phase 5 session can run. "schaefer" arrived with plan 5b-ii (R35) and keeps
#: its floors in a sidecar, because the exported users and the population code
#: they come from are hash-guarded against the 5a run.
FAMILIES_5B2 = tuple(FAMILIES) + ("schaefer",)

#: Phase 5d adds the Franks family; a_lens runs only on a surface family (L2).
FAMILIES_5D = FAMILIES_5B2 + ("franks",)
LENS_FAMILIES = ("franks", "dual_gauss")
#: ICORR reframe, test P (our design notes
#: or run): further surfaces PsyPref's proposals are projected onto -- Schaefer et al.'s family with
#: free peaks, family (ii), and Ramella et al.'s semi-trapezoid. Only a_lens runs them, with the
#: data prior. Their priors (``prompref.priors.surface_prior``) and the Schaefer (ii) floors come
#: from :data:`PROJ_SCRIPT`; the semi-trapezoid is scored against C1's floors over the same box.
#: ``pchip`` (added after the design's correction of 2026-10-22, 4509e78 on, before any
#: PCHIP code or run): Park et al. 2025's PCHIP profile adapted to level walking
#: (``prompref.designs.pchip_park``).
PROJ_FAMILIES = ("schaefer_peaks", "trapezoid_surface", "pchip")
PROJ_SCRIPT = "scripts/icorr_proj_prep.py"
#: The second review round's established-practice baseline (our design notes, R2, BASE,
#:): GP-BALD and CMA-ES search Schaefer (ii)'s six
#: interpretable parameters directly, in the family's own prior space (test P's surface
#: prior, ``scripts/icorr_proj_prep.py``), scored against its own floors.
BASE_FAMILY = "schaefer_peaks"
BASE_LEARNERS = ("cmaes", "gp_bald")


def _schaefer_sidecar_path(run_dir: str, fold: str) -> Path:
    return Path(run_dir) / "users" / f"{fold}-schaefer.json"


def _flat_sidecar_path(run_dir: str, fold: str, family: str) -> Path:
    return Path(run_dir) / "users" / f"{fold}-flat-{family}.json"


@lru_cache(maxsize=16)
def schaefer_floors(run_dir: str, fold: str) -> dict:
    """Per-user floors for the Schaefer family, written by scripts/phase5_schaefer_floors.py."""
    path = _schaefer_sidecar_path(run_dir, fold)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run scripts/phase5_schaefer_floors.py --run {run_dir} first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def flat_floors(run_dir: str, fold: str, family: str) -> dict:
    """Per-user floors for a flat cell's own box, written by scripts/phase5_schaefer_floors.py.

    This is the canonical explanation of why this sidecar exists; other places in this
    task cross-reference it rather than repeating it. A flat cell's learner searches
    ``FLAT_DESIGNS[family]``'s box, not the fold's data-prior design -- for ``promp_s0``
    that box is narrower (``bound_nm=14``, against the data prior's
    ``PROMP_DATA_BOUND_NM=30``), so scoring a flat cell against ``family["floor_nm"]``
    (5a's floor over the *wider* box) can be unreachable for it: a minimum over the wider
    box is at most the minimum over the narrower one. Measured on one fold's 44 users, the
    gap reached 0.213 Nm for one user, whose n90 threshold sat only ~0.19 Nm above the floor
    -- a flat cell there could never reach it at all. This sidecar is the flat box's own
    floor, so a flat cell's ``delivered_over_floor`` is never inflated by an unreachable
    target (the flat-floors addition to plan 5b-ii).
    """
    path = _flat_sidecar_path(run_dir, fold, family)
    if not path.exists():
        script = ("scripts/phase5c_floors.py" if family == C1_FAMILY
                  else "scripts/phase5_schaefer_floors.py")
        raise FileNotFoundError(f"{path} is missing: run {script} --run {run_dir} first")
    return json.loads(path.read_text(encoding="utf-8"))


def _sidecar_entry(sidecar: dict, user_id: str, path: Path, run_dir: str,
                   script: str = "scripts/phase5_schaefer_floors.py") -> dict:
    """One user's entry from a floors sidecar: a stale or partial
    sidecar -- one exported after a population change added or renamed a user -- used to
    fail with a bare ``KeyError: 'pop042'`` two functions removed from the missing-file
    message above, naming neither the sidecar nor the fix. This names both.
    """
    try:
        return sidecar[user_id]
    except KeyError:
        raise KeyError(
            f"{user_id} is missing from {path} (stale against the run's current exported "
            f"users); delete it and run {script} --run {run_dir} again") from None


DRIFT_SCRIPT = "scripts/phase5c_floors.py"


def drift_label(drift_nm: float) -> str:
    """The sidecar name of a drift displacement, to four decimals: 0.6745 -> drift0p6745."""
    return "drift" + f"{float(drift_nm):.4f}".replace(".", "p")


def _drift_sidecar_path(run_dir: str, fold: str, label: str) -> Path:
    return Path(run_dir) / "users" / f"{fold}-{label}.json"


@lru_cache(maxsize=32)
def drift_floors(run_dir: str, fold: str, label: str, script: str = DRIFT_SCRIPT) -> dict:
    """Per-user drift entries -- the shift reaching the displacement, whether it was capped,
    and the family's floor for the *final* ideal -- written by scripts/phase5c_floors.py.
    A magnitude sidecar holds the growth ``scale`` instead of shift and capped."""
    path = _drift_sidecar_path(run_dir, fold, label)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run {script} --run {run_dir} first")
    return json.loads(path.read_text(encoding="utf-8"))


def magnitude_label(drift_nm: float) -> str:
    """The sidecar name of a magnitude drift: 0.6745 -> drift-mag-0p6745."""
    return "drift-mag-" + f"{float(drift_nm):.4f}".replace(".", "p")


def _drift_entry(point: Phase5Point) -> dict:
    magnitude = point.drift_axis == "magnitude"
    label = magnitude_label(point.drift_nm) if magnitude else drift_label(point.drift_nm)
    path = _drift_sidecar_path(point.run_dir, point.fold, label)
    script = DRIFT_SCRIPT + (" --axis magnitude" if magnitude else "")
    entry = _sidecar_entry(drift_floors(point.run_dir, point.fold, label, script), point.user_id,
                           path, point.run_dir, script=script)
    if abs(float(entry["target_nm"]) - float(point.drift_nm)) > 1e-9:
        # Labels round to four decimals, so two displacements can share a file; scoring
        # against the other one's final ideal would be complete, plausible and wrong.
        raise ValueError(f"{path} was computed for a drift of {entry['target_nm']} Nm, "
                         f"not {point.drift_nm} Nm")
    return entry


#: Phase 4's tuned D_z multipliers; used when a point leaves them unset.
DZ_TUNED = dict(lengthscale_mult=2.0, step_mult=0.5)

#: Success tolerances above the user's floor, Nm RMS.
TAUS = (0.25, 0.5, 0.75)

OBSERVER_SALT = 0x5E55105

#: Phase 5e's two further streams per user (plan Global Constraints): the per-user
#: sensitivity draw and the device noise (E5). Each is seeded from the answer
#: seed with its own salt, so a cell that turns either on leaves the answer stream exactly
#: as it was, and a cell that leaves it off creates no generator at all.
S_USER_SALT = 0x5E05E5
NOISE_SALT = 0x5E0715E

#: The axes along which an ideal may drift: S17's time shift, or E6's magnitude scaling.
DRIFT_AXES = ("time", "magnitude")

#: The ICORR stress tests' point fields: the flags, False by
#: default, and the values, None by default. Both stay out of the key and the row at the default.
STRESS_FLAGS = ("a_metric", "no_pref_hard")
STRESS_VALUES = ("lapse_start", "lapse_end", "s_start", "s_end", "link", "link_scale",
                 "no_pref_jnd")
#: S5's links (``Phase5Point.link``); the probit is the plain wearer, ``link=None``.
STRESS_LINKS = ("logistic", "student_t")
#: The second review round's heavy-tailed answers (section 9a, R2, HT (c)): the Cauchy link, matched
#: as S5 (``scripts/icorr_r2_prep.py --step links``). A single stressor, checked as S5's links are.
R2_LINKS = ("cauchy",)

#: The second review round's pessimistic combined user (section 9a, R2, PESS
#: before any code or run): s = 2.0, fatigue (lapse 0.05 -> 0.30), order bias 0.3 s, the hard
#: "no difference" rule at 0.5 JND, phase weights with SD of log w 1.0 (S1's "pref-sd1") and ideals
#: from Schaefer's family by the per-bout rule of O (user set "ood-pb"). Exactly these values, all
#: together: this combination is the only multi-stressor point the checks accept, and it needs no
#: field of its own (:func:`is_pess`).
PESS_FIELDS = dict(s=2.0, lapse_start=0.05, lapse_end=0.30, order_bias=0.3, no_pref_jnd=0.5,
                   no_pref_hard=True, phase_weighting="pref-sd1", user_set="ood-pb")


def is_pess(point) -> bool:
    """Whether ``point`` carries the planned PESS combination (:data:`PESS_FIELDS`), all of it."""
    for name, value in PESS_FIELDS.items():
        have = getattr(point, name)
        if isinstance(value, bool) or isinstance(value, str):
            if have != value or type(have) is not type(value):
                return False
        elif not _finite(have) or float(have) != float(value):
            return False
    return True

#: The response-time test and the population hyperprior (our design notes, "RT" and "H2",
#:): the flags, False by default, and the values, None
#: by default. Both stay out of the key and the row at the default.
RTH_FLAGS = ("a_rt", "hyper_log")
HYPER_FIELDS = ("hyper_log_ell0", "hyper_ell_sd", "hyper_lam_a", "hyper_lam_b")
RTH_VALUES = ("rt_user",) + HYPER_FIELDS


class _Unstated:
    """The default of :attr:`Phase5Point.jnd_scale`: the level was never mentioned.

    It exists because ``None`` is a *legal* R34 level -- the gate turned off -- so it
    cannot double as "unset". A G point built without a level would otherwise silently be
    the JND-unaware variant rather than the published method, which is how every smoke row
    published before the review came to be gate-off.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "UNSTATED"


#: See :class:`_Unstated`. Never survives ``Phase5Point.__post_init__``, so it reaches
#: neither ``key`` nor a sweep row.
UNSTATED = _Unstated()


@dataclass(frozen=True)
class Phase5Point:
    run_dir: str
    fold: str
    family: str
    arm: str
    learner: str
    user_id: str
    lapse: float = 0.1
    s: float = 0.5
    n_comparisons: int = 150
    #: R35's prior factor: the fold's data prior, or the family's own box.
    prior: str = "data"
    #: R34's JND factor for G: a multiple of the wearer's true JND, or ``None`` for gate
    #: off. A G point must state it -- see :class:`_Unstated`; every other learner leaves
    #: it alone and gets ``None``, exactly as before.
    jnd_scale: float | None = UNSTATED
    #: D_z only; ``None`` means Phase 4's tuned value (:data:`DZ_TUNED`).
    lengthscale_mult: float | None = None
    step_mult: float | None = None
    #: G only: the size of its candidate set. ``None`` resolves to
    #: :data:`~prompref.learning.gp_ucb.N_CANDIDATES` for G in ``__post_init__``.
    n_candidates: int | None = None
    #: The observer's answer model: the time-order bias in units of
    #: ``s``, the trial-to-trial overdispersion of ``s``, and the half-width in Nm of a
    #: region of equally preferred curves. All zero reproduces every earlier session.
    order_bias: float = 0.0
    overdispersion: float = 0.0
    plateau_nm: float = 0.0
    #: Total RMS displacement in Nm by which the ideal drifts over the session (our design notes
    #: S17): a time shift growing linearly from the first comparison to the last. Zero
    #: holds the preference still, as every earlier session did.
    drift_nm: float = 0.0
    #: our design notes S16a: the wearer is indifferent among every curve within
    #: ``floor + k * true_jnd(s)`` of the ideal, with k this field. S16's ``plateau_nm`` put
    #: the region around the ideal, which the family cannot reach, so it changed nothing.
    plateau_over_floor_jnd: float = 0.0
    #: Learner A's posterior stopping rule: stop once the
    #: largest posterior SD of z* is below stop_sd x ell-hat. None runs the full budget.
    stop_sd: float | None = None
    #: Phase 5e: heterogeneous sensitivity. Each user's sensitivity is
    #: ``s_user = s * exp(s_user_logsd * z)``, z a standard normal drawn from the answer seed
    #: with :data:`S_USER_SALT` (:func:`per_user_s`). Zero is ``s`` itself, with no draw.
    #: Everything that models this wearer reads s_user: the observer; G's JND gate (a
    #: decision: R34's jnd_scale is a multiple of each wearer's own true JND, which
    #: is oracle information and favours G); the anchored plateau of
    #: ``plateau_over_floor_jnd``; and ``tracking_error_jnd``. ``s`` stays the cell's identity.
    s_user_logsd: float = 0.0
    #: Phase 5e: the observer's comparison, one of
    #: :data:`~prompref.user.oracle.METRICS`. "rms" is A's own model.
    observer_metric: str = "rms"
    #: O3's over-assistance weight. 1.0 is the plain observer.
    asymmetry: float = 1.0
    #: O6's familiarity weight w, with a zero-torque habitual
    #: profile. The optimizers then see a preference biased toward tau* / (1 + w), which is
    #: 0.8 tau* at w = 0.25, while the floor and the score stay against the true tau*. The
    #: biased target sits w / (1 + w) RMS(tau*) from tau*, 0.2 RMS(tau*) at w = 0.25; each
    #: row records it as ``familiarity_target_gap_nm``. 0.0 is the plain observer.
    familiarity_weight: float = 0.0
    #: Phase 5e: RMS in Nm of the smooth noise on each curve the wearer feels.
    device_noise_nm: float = 0.0
    #: Which way ``drift_nm`` moves the ideal: S17's time shift, or E6's magnitude scaling.
    drift_axis: str = "time"
    #: SortCMA only: pycma's population size, and a multiplier on sigma0.
    #: ``None`` keeps pycma's default and sigma0 as it is.
    cma_popsize: int | None = None
    cma_sigma_mult: float | None = None
    #: G only: the initial UCB weight. ``None`` keeps Arens' 1.5.
    g_kappa0: float | None = None
    #: Landmark observer only: its magnitude JND as a fraction of the
    #: ideal's peak. A landmark point resolves ``None`` to the static
    #: :data:`~prompref.user.oracle.LANDMARK_MAG_FRACTION` in ``__post_init__``, so an
    #: explicit 0.10 is the same identity; 0.25 is the walking-calibrated cell. Every other
    #: point keeps ``None``, and then the field appears in neither its key nor its row: the
    #: keys and rows of every cell without it stay byte-identical to those already stored.
    landmark_mag_fraction: float | None = None
    #: ICORR reframe, W3: the number K of von Mises bases of the
    #: ProMP family. ``None`` is the run's own 8-basis prior and floors. An explicit 8 resolves
    #: to ``None`` in ``__post_init__``, so K = 8 has one identity, the stored one; a ``None``
    #: stays out of the key and the row, so every earlier point keeps both byte for byte. Any
    #: other K reads the fold's K prior and K floors (``prompref.priors.basis_prior``, written
    #: by ``scripts/icorr_k_priors.py``); only promp_s0 with its data prior has them.
    n_basis: int | None = None
    #: ICORR reframe, W3: learner A's duels. "eig" is Phase 4's expected information gain, the
    #: method; "random" is Phase 3's random duels in the trust ball with the same posterior and
    #: prior, the acquisition ablation. "eig" stays out of the key and the row, as above.
    duel_mode: str = "eig"
    #: ICORR reframe, home-advantage test at matched decisiveness:
    #: the landmark wearer's utility is divided by this, its cell's decisiveness ratio measured
    #: as the Phase 5e REPORT section 9 measures it, so the wearer answers as decisively as the
    #: plain s = 1.0 RMS wearer. ``None`` is e-observer's wearer; an explicit 1.0 resolves to
    #: ``None`` (one identity), and ``None`` stays out of the key and the row.
    decisiveness_ratio: float | None = None
    #: ICORR reframe, out-of-generator ideals (section 9a, O): which users ``user_id`` names.
    #: ``None`` is the run's exported users; "ood" the Schaefer-profile users with one common
    #: peak (``RUN/ood/users``) and "ood-pb" those with
    #: per-bout peaks (``RUN/ood-pb/users``; :mod:`prompref.experiment.robust`). ``None`` stays
    #: out of key and row.
    user_set: str | None = None
    #: ICORR reframe, calibration of A's confidence: log A's
    #: predicted win probability against generic assistance, its predicted gain in Nm and its
    #: curve-space spread after every comparison (:mod:`prompref.experiment.calibration`). The
    #: logging draws no random numbers, so the session is otherwise the one without it. False
    #: stays out of the key and the row.
    calibration_log: bool = False
    #: ICORR reframe, phase-weighted wearers (our design notes, planned before any
    #: code or run): "bio" (W-bio, the weight from the mean hip angular velocity, read from
    #: ``out/icorr/wph_bio_weight.json``) or "pref" (W-pref, a random periodic weight per user).
    #: The wearer then judges by the phase-weighted RMS on its per-user matched scale
    #: (:mod:`prompref.experiment.phase_weighted`). ``None``, the plain wearer, stays out of the
    #: key and the row.
    phase_weighting: str | None = None
    #: ICORR reframe, review response (our design notes before any
    #: code or run): G's start. ``None`` is Arens' random first duel; "generic" puts z = 0, generic
    #: assistance, in candidate 0 and opens with it against a random candidate
    #: (:mod:`prompref.experiment.review`). ``None`` stays out of the key and the row.
    g_start: str | None = None
    #: ICORR review response, the lapse ablation of A: lambda fixed at 0 in A's likelihood and no
    #: intensity probes, against a wearer who still lapses at ``lapse``
    #: (:class:`~prompref.experiment.review.NoLapseLearner`). False stays out of the key and the row.
    a_no_lapse: bool = False
    #: ICORR review response, the probe ablation of A (PA4 before the cell
    #: existed): A with its own lapse model, lambda still inferred, and no intensity probes
    #: (``p_probe = 0``). False stays out of the key and the row.
    a_no_probes: bool = False
    #: ICORR stress tests (our design notes, "Week to Fri 30 Oct", part S, planned
    #: 2026-10-21 before any code or run; :mod:`prompref.experiment.stress`). Every field below is
    #: off at its default and then stays out of the key and the row.
    #: S2, PsyPref-M: learner A with a learned phase-weighted metric
    #: (:class:`~prompref.experiment.stress.MetricLearner`).
    a_metric: bool = False
    #: S3, fatigue: the wearer's lapse runs linearly from ``lapse_start`` at the first comparison to
    #: ``lapse_end`` at the last (:class:`~prompref.user.schedule.ScheduledObserver`). ``lapse``
    #: stays the cell's identity (0.1, the main condition's) and is read by nothing else.
    lapse_start: float | None = None
    lapse_end: float | None = None
    #: S4, changing sensitivity: the wearer's s runs linearly from ``s_start`` to ``s_end``. ``s``
    #: stays 1.0, the main condition's, so every comparator keeps its main configuration (G's JND
    #: gate included); only the wearer changes.
    s_start: float | None = None
    s_end: float | None = None
    #: S5, link misspecification: "logistic" or "student_t" (nu = 3) instead of the probit, at the
    #: scale ``s * link_scale``, with ``link_scale`` matched to the main wearer's median |dp| on
    #: prior-scale pairs (``scripts/icorr_stress_prep.py``).
    link: str | None = None
    link_scale: float | None = None
    #: S7, "no preference" answers below ``no_pref_jnd`` true JNDs of the wearer: our design notes's soft
    #: band (``allow_no_preference``), or with ``no_pref_hard`` the hard rule on the true utility
    #: difference (:class:`~prompref.user.oracle.Observer`).
    no_pref_jnd: float | None = None
    no_pref_hard: bool = False
    #: ICORR reframe, smarter comparisons (our design notes, "A"
    #: before any code or run): one change to learner A's acquisition. "curve" ranks the candidates
    #: by the expected reduction of trace(M Cov(z*)) (A-curve), "width" sets each candidate's width
    #: to a predicted probability of the more likely answer of 0.80 (A-width), "sub" keeps only the
    #: posterior axes carrying 95 % of its curve-space variance (A-sub);
    #: :mod:`prompref.learning.acq_variants`. ``None``, A itself, stays out of the key and the row.
    acq_variant: str | None = None
    #: ICORR response-time test (our design notes, "RT" before any
    #: code or run; :mod:`prompref.user.ddm`): the main condition's wearer answers by a drift-diffusion
    #: model, its response time informative ("inf", cell RT-inf) or drawn from the marginal
    #: independently of the pair ("null", cell RT-null). ``None``, the probit wearer, stays out of the
    #: key and the row.
    rt_user: str | None = None
    #: PsyPref-RT: learner A with the joint likelihood of choice and response time
    #: (:class:`~prompref.experiment.rt.RTLearner`). Needs ``rt_user``. False stays out of the key
    #: and the row.
    a_rt: bool = False
    #: ICORR population hyperprior: the first pass, A itself
    #: logging draws of its final posterior over (log ell, logit lambda) for the empirical-Bayes fit
    #: (:mod:`prompref.experiment.hyper`). The logging draws no random numbers. False stays out of
    #: the key and the row.
    hyper_log: bool = False
    #: PsyPref-hyper: learner A whose priors are log ell ~ N(hyper_log_ell0, hyper_ell_sd^2) and
    #: lambda ~ Beta(hyper_lam_a, hyper_lam_b), the fit on the other folds
    #: (``scripts/icorr_hyper_fit.py``). Set together; ``None`` stays out of the key and the row.
    hyper_log_ell0: float | None = None
    hyper_ell_sd: float | None = None
    hyper_lam_a: float | None = None
    hyper_lam_b: float | None = None

    def __post_init__(self) -> None:
        """Resolve :data:`UNSTATED`, so no sentinel ever leaves the constructor.

        For A, D_z and C2 it becomes ``None``, which is the value they have always carried:
        their ``key`` and their sweep rows are unchanged to the byte, so no completed row
        is re-run and double-counted on resume. For G it is refused, here rather than in
        :func:`check_point`, because ``key``'s float normalisation and the JSON row
        writer would both choke on the sentinel -- and because failing at construction
        stops a mis-specified 13,400-session sweep before it starts rather than filling it
        with error rows.

        ``sweep._row_key`` needs no widening for this: it rebuilds a point from a stored
        row, and every row ever written carries ``jnd_scale`` explicitly (the field is
        serialised from ``point.__dict__``, with ``null`` for the published gate-off G
        rows), so reconstruction never reaches this default.
        """
        if self.jnd_scale is UNSTATED:
            if self.learner == "g":
                raise ValueError(
                    "a G point must state jnd_scale: one of "
                    f"{JND_SCALES} (R34's levels; None is the gate turned off, which is a "
                    "level and not a default). It was left unstated.")
            object.__setattr__(self, "jnd_scale", None)
        if self.learner == "g" and self.n_candidates is None:
            # G's own default, stated so the point's identity says what G searched. Rows
            # written before 5c carry n_candidates as a learner extra with this same value,
            # so their resume key is unchanged (plan Review Focus 1).
            object.__setattr__(self, "n_candidates", N_CANDIDATES)
        if self.observer_metric == "landmark" and self.landmark_mag_fraction is None:
            # E4b: the static landmark cell states its fraction, as G states n_candidates.
            object.__setattr__(self, "landmark_mag_fraction", LANDMARK_MAG_FRACTION)
        if self.n_basis is not None and not isinstance(self.n_basis, (bool, str)) \
                and self.n_basis == DEFAULT_N_BASIS:
            # W3: K = 8 is the run's own prior; one identity for it, the one stored in e-main.
            object.__setattr__(self, "n_basis", None)
        ratio = self.decisiveness_ratio
        if ratio is not None and not isinstance(ratio, (bool, str)) \
                and isinstance(ratio, (int, float, np.integer, np.floating)) and ratio == 1.0:
            # A divisor of 1 is e-observer's own wearer: one identity, the stored one.
            object.__setattr__(self, "decisiveness_ratio", None)
        if isinstance(self.calibration_log, np.bool_):
            # json.dumps refuses numpy's bool; True and np.True_ are one identity.
            object.__setattr__(self, "calibration_log", bool(self.calibration_log))
        if isinstance(self.a_no_lapse, np.bool_):
            object.__setattr__(self, "a_no_lapse", bool(self.a_no_lapse))
        if isinstance(self.a_no_probes, np.bool_):
            object.__setattr__(self, "a_no_probes", bool(self.a_no_probes))
        for name in STRESS_FLAGS:
            if isinstance(getattr(self, name), np.bool_):
                object.__setattr__(self, name, bool(getattr(self, name)))
        for name in RTH_FLAGS:
            if isinstance(getattr(self, name), np.bool_):
                object.__setattr__(self, name, bool(getattr(self, name)))

    def row_fields(self) -> dict:
        """The point's fields as a row carries them: every field, except
        ``landmark_mag_fraction``, ``n_basis``, ``decisiveness_ratio`` and ``user_set`` when
        they are ``None``, ``duel_mode`` when it is "eig", ``calibration_log`` when it is
        False, ``phase_weighting`` and ``g_start`` when they are ``None``, ``a_no_lapse`` and
        ``a_no_probes`` when they are False, the stress tests' :data:`STRESS_FLAGS` when False
        and :data:`STRESS_VALUES` when ``None``, and ``acq_variant`` when it is ``None`` (see
        those fields)."""
        fields = dict(self.__dict__)
        if fields["landmark_mag_fraction"] is None:
            del fields["landmark_mag_fraction"]
        if fields["n_basis"] is None:
            del fields["n_basis"]
        if fields["duel_mode"] == "eig":
            del fields["duel_mode"]
        if fields["decisiveness_ratio"] is None:
            del fields["decisiveness_ratio"]
        if fields["user_set"] is None:
            del fields["user_set"]
        if fields["calibration_log"] is False:
            del fields["calibration_log"]
        if fields["phase_weighting"] is None:
            del fields["phase_weighting"]
        if fields["g_start"] is None:
            del fields["g_start"]
        if fields["a_no_lapse"] is False:
            del fields["a_no_lapse"]
        if fields["a_no_probes"] is False:
            del fields["a_no_probes"]
        # The stress tests' fields: out at their defaults.
        for name in STRESS_FLAGS:
            if fields[name] is False:
                del fields[name]
        for name in STRESS_VALUES:
            if fields[name] is None:
                del fields[name]
        if fields["acq_variant"] is None:
            del fields["acq_variant"]
        # The response-time test's and the hyperprior's fields (section 9a, RT and H2): likewise.
        for name in RTH_FLAGS:
            if fields[name] is False:
                del fields[name]
        for name in RTH_VALUES:
            if fields[name] is None:
                del fields[name]
        return fields

    def key(self) -> str:
        fields = self.row_fields()
        # json.dumps distinguishes 1 from 1.0; normalise so a resume key cannot fork.
        for name in ("lapse", "s", "lengthscale_mult", "step_mult", "jnd_scale",
                     "order_bias", "overdispersion", "plateau_nm", "drift_nm",
                     "plateau_over_floor_jnd", "stop_sd", "s_user_logsd", "asymmetry",
                     "familiarity_weight", "device_noise_nm", "cma_sigma_mult", "g_kappa0",
                     "landmark_mag_fraction", "decisiveness_ratio", "lapse_start", "lapse_end",
                     "s_start", "s_end", "link_scale", "no_pref_jnd") + HYPER_FIELDS:
            if fields.get(name) is not None:
                fields[name] = float(fields[name])
        fields["n_comparisons"] = int(fields["n_comparisons"])
        for name in ("n_candidates", "cma_popsize"):
            if fields[name] is not None:
                fields[name] = int(fields[name])
        k = fields.get("n_basis")
        if k is not None and not isinstance(k, (bool, str)) and float(k) == int(k):
            # 4, 4.0 and np.int64(4) are one identity; a fractional K keeps its own (and
            # check_point refuses it), so int never merges two identities.
            fields["n_basis"] = int(k)
        return json.dumps(fields, sort_keys=True)


def check_point(point: Phase5Point) -> None:
    if point.learner not in LEARNERS + OPT_IN_LEARNERS:
        raise ValueError(f"learner must be one of {LEARNERS + OPT_IN_LEARNERS}, got "
                         f"{point.learner!r}")
    if point.learner in GP_ACQUISITION and (point.family, point.prior) != ("promp_s0", "data") \
            and not _is_base(point):
        # Any basis count with its own K prior (section 9a, "More parameters": GP-BALD at
        # K = 16); _check_icorr validates n_basis itself. R2's BASE: GP-BALD on Schaefer (ii)'s
        # own prior space, whose whitened coordinates the bowl mean is then defined in.
        raise ValueError(f"{point.learner} runs on promp_s0 with its data prior: its prior mean "
                         "is that prior's bowl in its whitened coordinates (section 9a); got "
                         f"family {point.family!r}, prior {point.prior!r}")
    c1 = (point.learner == "c1", point.family == C1_FAMILY, point.arm == C1_ARM)
    if any(c1) and not all(c1):
        raise ValueError(f"C1 is Ramella as published (spec 5.4 S9): learner 'c1' runs only "
                         f"family {C1_FAMILY!r} in arm {C1_ARM!r}, and nothing else runs "
                         f"them; got learner {point.learner!r}, family {point.family!r}, "
                         f"arm {point.arm!r}")
    if all(c1):
        if point.prior != "flat":
            raise ValueError("C1 searches its own box with no population knowledge: "
                             "prior must be 'flat'")
    else:
        if point.arm not in ARMS:
            raise ValueError(f"arm must be one of {tuple(ARMS)}, got {point.arm!r}")
        if point.family not in FAMILIES_5D and not (point.learner == "a_lens"
                                                    and point.family in PROJ_FAMILIES) \
                and not _is_base(point):
            raise ValueError(f"family must be one of {FAMILIES_5D}, got {point.family!r}")
    multipliers = (point.lengthscale_mult, point.step_mult)
    if point.learner != "d_z" and multipliers != (None, None):
        raise ValueError(f"learner {point.learner!r} does not read lengthscale_mult or step_mult")
    if any(m is not None and m <= 0.0 for m in multipliers):
        raise ValueError(f"lengthscale_mult and step_mult must be positive; got {multipliers}")
    if point.prior not in PRIORS:
        raise ValueError(f"prior must be one of {PRIORS}, got {point.prior!r}")
    # Before the flat-family check below, so a flat franks cell is refused for its own reason.
    if point.learner == "a_lens" and (point.family not in LENS_FAMILIES + PROJ_FAMILIES
                                      or point.prior != "data"):
        raise ValueError(f"a_lens needs a surface family {LENS_FAMILIES + PROJ_FAMILIES} with the "
                         f"data prior (Phase 5d spec L2); got {point.family!r}, prior "
                         f"{point.prior!r}")
    if point.family == "franks" and point.prior != "data":
        raise ValueError("the franks family has a data prior only (Phase 5d spec L7)")
    if point.prior == "flat" and point.family not in FLAT_DESIGNS and point.learner != "c1":
        raise ValueError(
            f"family must be one of {tuple(FLAT_DESIGNS)} for a flat prior, got {point.family!r}")
    if point.learner == "g":
        if point.jnd_scale not in JND_SCALES:
            raise ValueError(f"jnd_scale must be one of {JND_SCALES}, got {point.jnd_scale}")
    elif point.jnd_scale is not None:
        raise ValueError(f"learner {point.learner!r} does not read jnd_scale")
    if point.n_candidates is not None:
        if point.learner != "g":
            raise ValueError(f"learner {point.learner!r} does not read n_candidates")
        if int(point.n_candidates) < 2:
            raise ValueError(f"n_candidates must be at least 2, got {point.n_candidates}")
    for name in ("overdispersion", "plateau_nm"):
        if getattr(point, name) < 0.0:
            raise ValueError(f"{name} must be non-negative, got {getattr(point, name)}")
    if point.drift_nm < 0.0:
        raise ValueError(f"drift_nm must be non-negative, got {point.drift_nm}")
    if point.drift_nm > 0.0 and (point.family, point.prior) != ("promp_s0", "data"):
        raise ValueError("drift cells run on promp_s0 with its data prior (spec 5.4 S17): "
                         "the final-ideal floors exist only there")
    if point.plateau_over_floor_jnd < 0.0:
        raise ValueError(f"plateau_over_floor_jnd must be non-negative, got "
                         f"{point.plateau_over_floor_jnd}")
    if point.plateau_over_floor_jnd > 0.0 and point.plateau_nm > 0.0:
        raise ValueError("a point states one kind of preferred region: plateau_nm (around "
                         "the ideal, S16) or plateau_over_floor_jnd (above the floor, S16a)")
    if point.plateau_over_floor_jnd > 0.0 and point.drift_nm > 0.0:
        raise ValueError("the anchored region is not combined with drift: its floor moves")
    if point.stop_sd is not None:
        if point.learner not in ("a", "a_lens"):
            raise ValueError(f"learner {point.learner!r} does not read stop_sd")
        if point.stop_sd <= 0.0:
            raise ValueError(f"stop_sd must be positive, got {point.stop_sd}")
    _check_phase5e(point)
    _check_icorr(point)
    _check_robust(point)
    _check_calibration(point)
    _check_phase_weighting(point)
    _check_review(point)
    _check_stress(point)
    _check_acq(point)
    _check_rt_hyper(point)
    _check_proj(point)
    _check_base(point)


def _is_base(point: Phase5Point) -> bool:
    """Whether ``point`` is an R2 BASE session: a BASE learner on Schaefer (ii) with the data prior."""
    return (point.learner in BASE_LEARNERS and point.family == BASE_FAMILY
            and point.prior == "data")


#: BASE's planned CMA-ES population (section 9a, R2: "CMA-ES (popsize 4)").
BASE_CMA_POPSIZE = 4


def _check_base(point: Phase5Point) -> None:
    """R2's BASE (section 9a): the main condition's plain wearer of the run's exported users, s = 1.0
    and lapse 0.1, 40 comparisons' worth of the main condition; CMA-ES at popsize 4. The arm and
    CMA-ES's sigma multiplier are the tuned settings; nothing else is changed."""
    if not _is_base(point):
        return
    if point.n_basis is not None or point.duel_mode != "eig":
        raise ValueError("BASE searches Schaefer (ii)'s six parameters: no basis count or duel mode")
    if float(point.s) != 1.0 or float(point.lapse) != 0.1:
        raise ValueError(f"BASE runs the main condition's wearer (s = 1.0, lapse 0.1); got s "
                         f"{point.s}, lapse {point.lapse}")
    if point.learner == "cmaes":
        if point.cma_popsize is None or int(point.cma_popsize) != BASE_CMA_POPSIZE:
            raise ValueError(f"BASE's CMA-ES has the registered population {BASE_CMA_POPSIZE}; got "
                             f"{point.cma_popsize!r}")
    if (point.user_set is not None or point.phase_weighting is not None
            or point.observer_metric != "rms" or point.order_bias != 0.0
            or point.overdispersion > 0.0 or point.drift_nm > 0.0 or point.plateau_nm > 0.0
            or point.plateau_over_floor_jnd > 0.0 or point.asymmetry != 1.0
            or point.familiarity_weight > 0.0 or point.device_noise_nm > 0.0
            or point.s_user_logsd > 0.0 or point.calibration_log
            or any(getattr(point, name) is not False for name in STRESS_FLAGS)
            or any(getattr(point, name) is not None for name in STRESS_VALUES)):
        raise ValueError("BASE runs the main condition's plain wearer of the run's exported users")


def _main_condition_deviations(point: Phase5Point) -> list[str]:
    """How a point departs from the main condition's plain A session (promp_s0, data prior, tight
    arm, 8 bases, s = 1.0, the plain RMS wearer of the exported users, A with EIG duels and no
    variant). The lapse is not checked here."""
    out = []
    if (point.family, point.prior, point.arm) != ("promp_s0", "data", "tight") \
            or point.n_basis is not None:
        out.append(f"family {point.family!r}, prior {point.prior!r}, arm {point.arm!r}, n_basis "
                   f"{point.n_basis!r}")
    if float(point.s) != 1.0:
        out.append(f"s {point.s}")
    if point.learner != "a" or point.duel_mode != "eig":
        out.append(f"learner {point.learner!r}, duel_mode {point.duel_mode!r}")
    if (point.observer_metric != "rms" or point.user_set is not None
            or point.phase_weighting is not None or point.drift_nm > 0.0
            or point.plateau_nm > 0.0 or point.plateau_over_floor_jnd > 0.0
            or point.asymmetry != 1.0 or point.familiarity_weight > 0.0
            or point.device_noise_nm > 0.0 or point.s_user_logsd > 0.0
            or point.overdispersion > 0.0 or point.order_bias != 0.0):
        out.append("another wearer, user set, drift, preferred region, asymmetry, familiarity, "
                   "device noise, heterogeneous s, overdispersion or order bias")
    if any(getattr(point, name) is not False for name in STRESS_FLAGS) \
            or any(getattr(point, name) is not None for name in STRESS_VALUES):
        out.append("a stress-test field")
    if point.a_no_lapse or point.a_no_probes or point.stop_sd is not None or point.calibration_log:
        out.append("the lapse or probe ablation, a stopping rule or the calibration logging")
    if point.acq_variant is not None:
        out.append(f"the acquisition variant {point.acq_variant!r}")
    return out


def _check_rt_hyper(point: Phase5Point) -> None:
    """The response-time test and the population hyperprior. Each runs on the main condition's plain A session and changes one thing: the
    wearer's answer model (RT), PsyPref-RT's likelihood (RT), the first pass's logging (H2) or
    PsyPref-hyper's priors on log ell and lambda (H2)."""
    for name in RTH_FLAGS:
        if not isinstance(getattr(point, name), bool):
            raise ValueError(f"{name} must be True or False, got {getattr(point, name)!r}")
    hyper = [getattr(point, name) for name in HYPER_FIELDS]
    if any(v is None for v in hyper) and any(v is not None for v in hyper):
        raise ValueError(f"{', '.join(HYPER_FIELDS)} are set together (PsyPref-hyper)")
    has_hyper = hyper[0] is not None
    if has_hyper:
        if not all(_finite(v) for v in hyper):
            raise ValueError(f"the hyperprior must be finite numbers, got {hyper}")
        if not (float(point.hyper_ell_sd) > 0.0 and float(point.hyper_lam_a) > 0.0
                and float(point.hyper_lam_b) > 0.0):
            raise ValueError(f"hyper_ell_sd, hyper_lam_a and hyper_lam_b must be positive, got "
                             f"{hyper[1:]}")
    if point.rt_user is not None and point.rt_user not in RT_MODES:
        raise ValueError(f"rt_user must be one of {RT_MODES} or None, got {point.rt_user!r}")
    if point.a_rt and point.rt_user is None:
        raise ValueError("PsyPref-RT (a_rt) reads the response times of a DDM wearer: it needs "
                         "rt_user")
    used = [name for name, on in (("rt_user", point.rt_user is not None),
                                  ("hyper_log", point.hyper_log), ("PsyPref-hyper", has_hyper))
            if on]
    if not used:
        return
    if len(used) > 1:
        raise ValueError(f"one of the response-time test, the first pass and PsyPref-hyper per "
                         f"point: got {', '.join(used)}")
    off = _main_condition_deviations(point)
    if off:
        raise ValueError(f"{used[0]} runs on the main condition's plain A session; got {off[0]}")
    if point.rt_user is not None and float(point.lapse) != 0.1:
        raise ValueError(f"the DDM wearer has the registered lapse 0.1, got {point.lapse}")


def _check_proj(point: Phase5Point) -> None:
    """Test P's projection cells (section 9a, 2026-10-22): PsyPref (A with EIG duels) on
    the fold's 8-basis ProMP, each proposal projected onto a surface of :data:`PROJ_FAMILIES`, for
    the main condition's plain wearer and the run's exported users, run to its budget."""
    if point.family not in PROJ_FAMILIES:
        return
    if point.stop_sd is not None:
        raise ValueError(f"the {point.family} projection runs to its budget: no stop_sd")
    if (point.observer_metric != "rms" or point.user_set is not None
            or point.phase_weighting is not None or point.calibration_log):
        raise ValueError(f"the {point.family} projection was registered for the plain RMS wearer "
                         "of the run's exported users, without calibration logging")


def _finite(value) -> bool:
    return (not isinstance(value, (bool, str, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value)))


def _check_stress(point: Phase5Point) -> None:
    """The stress tests. Each stressor runs alone on the
    main condition's plain wearer (S3-S7), and PsyPref-M on the plain or a phase-weighted wearer
    (S2): nothing else was planned."""
    for name in STRESS_FLAGS:
        if not isinstance(getattr(point, name), bool):
            raise ValueError(f"{name} must be True or False, got {getattr(point, name)!r}")
    if is_pess(point):
        _check_pess(point)
        return
    stressors = []
    for start, end, what in (("lapse_start", "lapse_end", "the lapse schedule (S3)"),
                             ("s_start", "s_end", "the sensitivity schedule (S4)")):
        a, b = getattr(point, start), getattr(point, end)
        if (a is None) != (b is None):
            raise ValueError(f"{start} and {end} are set together: {what}")
        if a is None:
            continue
        if not (_finite(a) and _finite(b)):
            raise ValueError(f"{start} and {end} must be finite numbers, got {a!r} and {b!r}")
        if start == "lapse_start" and not (0.0 <= float(a) <= 1.0 and 0.0 <= float(b) <= 1.0):
            raise ValueError(f"a lapse schedule lies in [0, 1], got {a} to {b}")
        if start == "s_start" and not (float(a) > 0.0 and float(b) > 0.0):
            raise ValueError(f"a sensitivity schedule is positive, got {a} to {b}")
        stressors.append(what)
    if (point.link is None) != (point.link_scale is None):
        raise ValueError("link and link_scale are set together (S5)")
    if point.link is not None:
        if point.link not in STRESS_LINKS + R2_LINKS:
            raise ValueError(f"link must be one of {STRESS_LINKS + R2_LINKS} or None, got "
                             f"{point.link!r}")
        if not _finite(point.link_scale) or float(point.link_scale) <= 0.0:
            raise ValueError(f"link_scale must be positive and finite, got {point.link_scale!r}")
        stressors.append("the link (S5)")
    if point.no_pref_jnd is not None:
        if not _finite(point.no_pref_jnd) or float(point.no_pref_jnd) <= 0.0:
            raise ValueError(f"no_pref_jnd must be positive and finite, got {point.no_pref_jnd!r}")
        stressors.append("the no-preference answer (S7)")
    elif point.no_pref_hard:
        raise ValueError("no_pref_hard is a rule for the no-preference answer: it needs no_pref_jnd")
    if point.a_metric:
        if point.learner != "a" or point.duel_mode != "eig":
            raise ValueError("a_metric is learner A with EIG duels (PsyPref-M); got learner "
                             f"{point.learner!r}, duel_mode {point.duel_mode!r}")
        if point.a_no_lapse or point.a_no_probes or point.stop_sd is not None \
                or point.calibration_log:
            raise ValueError("a_metric is A itself plus the metric: not combined with the lapse or "
                             "probe ablations, a stopping rule or the calibration logging")
        if (point.family, point.prior, point.arm) != ("promp_s0", "data", "tight") \
                or point.n_basis is not None:
            raise ValueError("PsyPref-M runs on promp_s0 with its data prior, the tight arm and 8 "
                             f"bases; got family {point.family!r}, prior {point.prior!r}, arm "
                             f"{point.arm!r}, n_basis {point.n_basis!r}")
        if stressors:
            raise ValueError("PsyPref-M was registered under the main condition and the phase-"
                             f"weighted wearers only, not under {stressors[0]}")
    if not stressors:
        return
    if len(stressors) > 1:
        raise ValueError(f"one stressor per point: got {', '.join(stressors)}")
    if (point.family, point.prior, point.arm) != ("promp_s0", "data", "tight") \
            or point.n_basis is not None:
        raise ValueError("the stress tests run on promp_s0 with its data prior, the tight arm and 8 "
                         f"bases; got family {point.family!r}, prior {point.prior!r}, arm "
                         f"{point.arm!r}, n_basis {point.n_basis!r}")
    if float(point.s) != 1.0:
        raise ValueError(f"the stress tests keep the main condition's s = 1.0 as the cell's "
                         f"identity; got s {point.s}")
    if (point.observer_metric != "rms" or point.user_set is not None
            or point.phase_weighting is not None or point.drift_nm > 0.0
            or point.plateau_nm > 0.0 or point.plateau_over_floor_jnd > 0.0
            or point.asymmetry != 1.0 or point.familiarity_weight > 0.0
            or point.device_noise_nm > 0.0 or point.s_user_logsd > 0.0
            or point.overdispersion > 0.0 or point.order_bias != 0.0 or point.calibration_log):
        raise ValueError(f"{stressors[0]} is violated alone, on the main condition's plain wearer: "
                         "no other wearer, user set, drift, preferred region, asymmetry, "
                         "familiarity, device noise, heterogeneous s, overdispersion, order bias or "
                         "calibration logging")


#: The learners of the PESS cell (planned: PsyPref, the fast comparators, and GP-BALD if the
#: time box allows).
PESS_LEARNERS = ("a", "g", "d_z", "cmaes", "gp_bald")


def _check_pess(point: Phase5Point) -> None:
    """The pessimistic combined user (section 9a, R2, PESS): :data:`PESS_FIELDS` together, on
    promp_s0 with its data prior, the tight arm and 8 bases, the main condition's plain learners,
    and nothing else changed."""
    if point.learner not in PESS_LEARNERS or point.duel_mode != "eig":
        raise ValueError(f"PESS runs {PESS_LEARNERS} (A with EIG duels); got learner "
                         f"{point.learner!r}, duel_mode {point.duel_mode!r}")
    if (point.family, point.prior, point.arm) != ("promp_s0", "data", "tight") \
            or point.n_basis is not None:
        raise ValueError("PESS runs on promp_s0 with its data prior, the tight arm and 8 bases; got "
                         f"family {point.family!r}, prior {point.prior!r}, arm {point.arm!r}, "
                         f"n_basis {point.n_basis!r}")
    if (point.s_start is not None or point.link is not None or point.observer_metric != "rms"
            or point.drift_nm > 0.0 or point.plateau_nm > 0.0
            or point.plateau_over_floor_jnd > 0.0 or point.asymmetry != 1.0
            or point.familiarity_weight > 0.0 or point.device_noise_nm > 0.0
            or point.s_user_logsd > 0.0 or point.overdispersion > 0.0):
        raise ValueError("PESS combines exactly its registered violations: no sensitivity "
                         "schedule, other link, other wearer, drift, preferred region, asymmetry, "
                         "familiarity, device noise, heterogeneous s or overdispersion")
    if (point.a_metric or point.a_no_lapse or point.a_no_probes or point.stop_sd is not None
            or point.calibration_log or point.acq_variant is not None or point.a_rt
            or point.hyper_log or point.hyper_log_ell0 is not None or point.rt_user is not None
            or point.g_start is not None or point.cma_popsize is not None
            or point.cma_sigma_mult is not None or point.g_kappa0 is not None):
        raise ValueError("PESS runs every learner in its main configuration: no variant, "
                         "ablation, stopping rule, logging or tuned setting")


def _check_acq(point: Phase5Point) -> None:
    """The acquisition variants (section 9a, "A"): learner A with EIG duels, on promp_s0 with its
    data prior (A-curve and A-sub need the ProMP basis's curve metric), one single change each:
    not combined with A's other ablations, a stopping rule or the calibration logging."""
    variant = point.acq_variant
    if variant is None:
        return
    if variant not in ACQ_VARIANTS:
        raise ValueError(f"acq_variant must be one of {tuple(ACQ_VARIANTS)} or None, got "
                         f"{variant!r}")
    if point.learner != "a" or point.duel_mode != "eig":
        raise ValueError("acq_variant is learner A with EIG duels; got learner "
                         f"{point.learner!r}, duel_mode {point.duel_mode!r}")
    if (point.family, point.prior) != ("promp_s0", "data"):
        raise ValueError("the acquisition variants run on promp_s0 with its data prior, whose basis "
                         f"gives the curve metric; got family {point.family!r}, prior "
                         f"{point.prior!r}")
    if point.a_no_lapse or point.a_no_probes or point.stop_sd is not None or point.calibration_log:
        raise ValueError("an acquisition variant is one change to A: it is not combined with "
                         "a_no_lapse, a_no_probes, a stopping rule or the calibration logging")
    if point.a_metric:
        # PsyPref-M is another learner A, which build_learner would build instead of the variant.
        raise ValueError("an acquisition variant is one change to A: it is not combined with "
                         "PsyPref-M (a_metric)")


def _check_review(point: Phase5Point) -> None:
    """The review response's two learner variants: G's generic
    start, on promp_s0 with its data prior and 8 bases, where z = 0 is generic assistance; and A
    without lapse, which is A with EIG duels (the probes it switches off exist only there)."""
    if point.g_start is not None:
        if point.g_start not in G_STARTS:
            raise ValueError(f"g_start must be one of {G_STARTS} or None, got {point.g_start!r}")
        if point.learner != "g":
            raise ValueError(f"only learner 'g' reads g_start; got learner {point.learner!r}")
        if (point.family, point.prior) != ("promp_s0", "data") or point.n_basis is not None:
            raise ValueError("G's generic start is z = 0 of promp_s0's data prior with 8 bases; got "
                             f"family {point.family!r}, prior {point.prior!r}, n_basis "
                             f"{point.n_basis!r}")
    flag = point.a_no_lapse
    if not isinstance(flag, bool):
        raise ValueError(f"a_no_lapse must be True or False, got {flag!r}")
    if flag:
        if point.learner != "a" or point.duel_mode != "eig":
            raise ValueError("a_no_lapse is learner A with EIG duels; got learner "
                             f"{point.learner!r}, duel_mode {point.duel_mode!r}")
        if point.stop_sd is not None or point.calibration_log:
            raise ValueError("a_no_lapse is not combined with a stopping rule or the calibration "
                             "logging")
    flag = point.a_no_probes
    if not isinstance(flag, bool):
        raise ValueError(f"a_no_probes must be True or False, got {flag!r}")
    if flag:
        if point.learner != "a" or point.duel_mode != "eig":
            raise ValueError("a_no_probes is learner A with EIG duels; got learner "
                             f"{point.learner!r}, duel_mode {point.duel_mode!r}")
        if point.a_no_lapse:
            # A without lapse has no probes already: one identity for that learner.
            raise ValueError("a_no_lapse already switches the probes off; a_no_probes is A with "
                             "its own lapse model")
        if point.stop_sd is not None or point.calibration_log:
            raise ValueError("a_no_probes is not combined with a stopping rule or the "
                             "calibration logging")


def _check_phase_weighting(point: Phase5Point) -> None:
    """The phase-weighted wearers (section 9a, W-bio and W-pref): the plain s = 1.0 wearer of
    e-main with only the weighting changed, on promp_s0 with its data prior and 8 bases, where the
    per-user scales and floors were computed."""
    pw = point.phase_weighting
    if pw is None:
        return
    if pw not in PHASE_WEIGHTINGS:
        raise ValueError(f"phase_weighting must be one of {PHASE_WEIGHTINGS} or None, got {pw!r}")
    if is_pess(point):
        # The second review round's PESS user: S1's weights on the "ood-pb" users at s = 2.0, with
        # the other planned violations (_check_pess); the sidecar lives below that user set.
        return
    if (point.family, point.prior) != ("promp_s0", "data") or point.n_basis is not None:
        raise ValueError("the phase-weighted wearers run on promp_s0 with its data prior and 8 "
                         f"bases, where their floors were computed; got family {point.family!r}, "
                         f"prior {point.prior!r}, n_basis {point.n_basis!r}")
    if point.observer_metric != "rms" or point.user_set is not None:
        raise ValueError("the phase-weighted wearer replaces the plain RMS wearer of the run's "
                         f"exported users; got observer_metric {point.observer_metric!r}, "
                         f"user_set {point.user_set!r}")
    if float(point.s) != 1.0:
        raise ValueError(f"the phase-weighted wearer's scale is matched to the plain s = 1.0 "
                         f"wearer; got s {point.s}")
    if (point.drift_nm > 0.0 or point.plateau_nm > 0.0 or point.plateau_over_floor_jnd > 0.0
            or point.asymmetry != 1.0 or point.familiarity_weight > 0.0
            or point.device_noise_nm > 0.0 or point.s_user_logsd > 0.0
            or point.calibration_log):
        raise ValueError("the phase-weighted wearer is not combined with drift, a preferred "
                         "region, asymmetry, familiarity, device noise, heterogeneous s or the "
                         "calibration logging: only the weighting changes")


def _check_calibration(point: Phase5Point) -> None:
    """The calibration logging (section 9a): A's own posterior on ProMP with its data prior."""
    flag = point.calibration_log
    if not isinstance(flag, bool):
        raise ValueError(f"calibration_log must be True or False, got {flag!r}")
    if flag and (point.learner, point.family, point.prior) != ("a", "promp_s0", "data"):
        # a_lens delivers a projection, not A's curve; the other learners have no posterior.
        raise ValueError("calibration_log reads learner A's posterior on promp_s0 with its data "
                         f"prior; got learner {point.learner!r}, family {point.family!r}, prior "
                         f"{point.prior!r}")


def _check_robust(point: Phase5Point) -> None:
    """The ICORR reframe's robustness fields (section 9a): the matched-decisiveness divisor, the
    out-of-generator users and the random smooth utility."""
    ratio = point.decisiveness_ratio
    if ratio is not None:
        if isinstance(ratio, (bool, str)) or not isinstance(
                ratio, (int, float, np.integer, np.floating)) or not math.isfinite(float(ratio)) \
                or float(ratio) <= 0.0:
            raise ValueError(f"decisiveness_ratio must be a positive, finite number, got {ratio!r}")
        if point.observer_metric != "landmark":
            raise ValueError("only the landmark observer reads decisiveness_ratio (the home-"
                             f"advantage test); got observer_metric {point.observer_metric!r}")
    if point.user_set is not None:
        if point.user_set not in USER_SETS:
            raise ValueError(f"user_set must be one of {USER_SETS} or None, got {point.user_set!r}")
        if (point.family, point.prior) != ("promp_s0", "data") or point.n_basis is not None:
            # The OOD users carry floors for the fold's 8-basis ProMP data prior only.
            raise ValueError("the out-of-generator users run on promp_s0 with its data prior "
                             f"and 8 bases; got family {point.family!r}, prior {point.prior!r}, "
                             f"n_basis {point.n_basis!r}")
        if point.drift_nm > 0.0 or point.observer_metric != "rms":
            raise ValueError("the out-of-generator users answer as the plain RMS wearer (section "
                             "9a, O): no drift and observer_metric 'rms'")
    if point.observer_metric == RANDOM_SMOOTH:
        if (point.family, point.prior) != ("promp_s0", "data") or point.n_basis is not None:
            raise ValueError("the random smooth utility runs on promp_s0 with its data prior and "
                             "8 bases, where its u_max was found")
        if point.user_set is not None:
            raise ValueError("the random smooth utility belongs to the run's exported users")
        if (point.drift_nm > 0.0 or point.plateau_nm > 0.0 or point.plateau_over_floor_jnd > 0.0
                or point.asymmetry != 1.0 or point.familiarity_weight > 0.0
                or point.device_noise_nm > 0.0 or point.s_user_logsd > 0.0):
            raise ValueError("the random smooth utility is not combined with drift, a preferred "
                             "region, asymmetry, familiarity, device noise or heterogeneous s: "
                             "its scale is matched to the plain s = 1.0 wearer")


def _check_icorr(point: Phase5Point) -> None:
    """The ICORR reframe's fields (W3): the ProMP basis count and A's duel mode."""
    if point.duel_mode not in DUEL_MODES:
        raise ValueError(f"duel_mode must be one of {DUEL_MODES}, got {point.duel_mode!r}")
    if point.duel_mode != "eig":
        if point.learner != "a":
            # a_lens's inner A is the method itself; no other learner duels by EIG at all.
            raise ValueError(f"only learner 'a' reads duel_mode; got learner {point.learner!r} "
                             f"with duel_mode {point.duel_mode!r}")
        if point.stop_sd is not None:
            # PsychometricLearner refuses it too, but as an error row mid-sweep.
            raise ValueError("random duels take no stopping rule: stop_sd belongs to "
                             "duel_mode 'eig' (Phase 4, R18 item 12)")
    if point.n_basis is None:
        return
    k = point.n_basis
    if isinstance(k, (bool, str)) or not isinstance(k, (int, float, np.integer, np.floating)) \
            or float(k) != int(k):
        raise ValueError(f"n_basis must be a whole number, got {k!r}")
    if int(k) < 2:
        raise ValueError(f"n_basis must be at least 2 (VonMisesBasis), got {k}")
    if (point.family, point.prior) != ("promp_s0", "data"):
        # The K priors and floors exist for the ProMP data prior only; a flat box, the other
        # families, C1's trapezoid and the lens's surface families have no basis count.
        raise ValueError(f"n_basis applies to family 'promp_s0' with prior 'data' (W3); got "
                         f"family {point.family!r}, prior {point.prior!r}, learner "
                         f"{point.learner!r}")
    if point.drift_nm > 0.0:
        raise ValueError("drift cells score against final-ideal floors computed for the "
                         "8-basis prior only; n_basis is not combined with drift")


def _check_phase5e(point: Phase5Point) -> None:
    """Phase 5e's fields: the wearer, the device and the comparator knobs."""
    if point.observer_metric not in METRICS:
        raise ValueError(f"observer_metric must be one of {METRICS}, got {point.observer_metric!r}")
    if point.drift_axis not in DRIFT_AXES:
        raise ValueError(f"drift_axis must be one of {DRIFT_AXES}, got {point.drift_axis!r}")
    if point.drift_axis != "time" and point.drift_nm <= 0.0:
        # Otherwise a no-drift session would run under a second identity.
        raise ValueError(f"drift_axis {point.drift_axis!r} needs a drift: drift_nm must be "
                         f"positive, got {point.drift_nm}")
    if point.asymmetry <= 0.0:
        raise ValueError(f"asymmetry must be positive, got {point.asymmetry}")
    if not 0.0 <= point.familiarity_weight < 1.0:
        raise ValueError(f"familiarity_weight must lie in [0, 1), got {point.familiarity_weight}")
    for name in ("device_noise_nm", "s_user_logsd"):
        if getattr(point, name) < 0.0:
            raise ValueError(f"{name} must be non-negative, got {getattr(point, name)}")
    if point.observer_metric == "landmark" and (
            point.asymmetry != 1.0 or point.plateau_nm > 0.0
            or point.plateau_over_floor_jnd > 0.0):
        raise ValueError("the landmark observer judges features, not an RMS distance: it "
                         "takes neither asymmetry nor a preferred region (spec E4)")
    if point.landmark_mag_fraction is not None:
        if point.observer_metric != "landmark":
            raise ValueError("only the landmark observer reads landmark_mag_fraction (spec "
                             f"amendment E4b); got observer_metric {point.observer_metric!r}")
        if point.landmark_mag_fraction <= 0.0:
            raise ValueError(f"landmark_mag_fraction must be positive, got "
                             f"{point.landmark_mag_fraction}")
    for name in ("cma_popsize", "cma_sigma_mult"):
        if getattr(point, name) is not None and point.learner != "cmaes":
            raise ValueError(f"learner {point.learner!r} does not read {name}")
    if point.cma_popsize is not None:
        if float(point.cma_popsize) != int(point.cma_popsize):
            # key stores int(cma_popsize), so 4.5 would share the identity of 4.
            raise ValueError(f"cma_popsize must be a whole number, got {point.cma_popsize}")
        if int(point.cma_popsize) < 2:
            raise ValueError(f"cma_popsize must be at least 2, got {point.cma_popsize}")
    if point.cma_sigma_mult is not None and point.cma_sigma_mult <= 0.0:
        raise ValueError(f"cma_sigma_mult must be positive, got {point.cma_sigma_mult}")
    if point.g_kappa0 is not None:
        if point.learner != "g":
            raise ValueError(f"learner {point.learner!r} does not read g_kappa0")
        if point.g_kappa0 < 0.0:
            # A negative weight turns the upper confidence bound into a lower one.
            raise ValueError(f"g_kappa0 must be non-negative, got {point.g_kappa0}")
        if point.g_kappa0 > KAPPA_MAX:
            # G steps kappa up from kappa0 to its cap; above the cap it cannot.
            raise ValueError(f"g_kappa0 must be at most {KAPPA_MAX} (G's cap), got "
                             f"{point.g_kappa0}")


def per_user_s(s: float, logsd: float, answer_seed: int) -> float:
    """The wearer's own sensitivity: ``s * exp(logsd * z)``, z a standard normal
    from ``default_rng([answer_seed, S_USER_SALT])``. The same user gets the same ``s_user``
    in every arm and for every learner, as the answers are paired.

    ``logsd == 0`` returns ``s`` itself and creates no generator.
    """
    if logsd == 0.0:
        return s
    z = np.random.default_rng([int(answer_seed), S_USER_SALT]).standard_normal()
    return float(s * np.exp(logsd * z))


def arm_space(prior: DataPrior | TimingPrior, arm: str) -> LearnerSpace:
    """The arm's space behind the device.

    It combines the prior's own whitening (``DataPrior.space``, 5a review; or,
    for a data-prior Schaefer cell, ``TimingPrior.space`` -- both offer the same
    ``design``/``whitening``/``space(radius)`` shape, so this function does not care
    which one it was handed), the arm's 99 % radius, and a design whose curves
    are device-clipped (R29).

    The order matters: build the space from the prior first and
    wrap its design afterwards. ``DeviceDesign`` is not a ``PrompS0`` instance, so it defeats
    ``learner_prior``'s PrompS0 fast path, and ``LearnerSpace.build(DeviceDesign(...))`` would
    silently whiten differently from ``DataPrior.space``, with radii that no longer match the
    5a diagnostics. This note lives here rather than in ``data_prior.py`` on purpose: that file
    is population code whose hash the users export checks against the 5a run, so even a
    docstring edit there makes the export refuse.
    """
    space = prior.space(ARMS[arm].radius(prior.mu.size))
    return LearnerSpace(DeviceDesign(space.design), space.whitening, space.radius)


def flat_whitening(design) -> Whitening:
    """The moment-matched Gaussian of a design's box: mean at the centre,
    variance (high - low)^2 / 12 per parameter. It is the closest Gaussian analogue of "no
    prior knowledge", not an uninformed prior, and §4.3 says so.

    Shared by ``flat_space`` and ``scripts/phase5_schaefer_floors.py``'s flat-box floors
    (the flat-floors addition to plan 5b-ii), so a flat cell's search and its floor agree
    on exactly what "no prior knowledge" means for that box.
    """
    low = np.asarray(design.low, dtype=float)
    high = np.asarray(design.high, dtype=float)
    mu = 0.5 * (low + high)
    sigma = np.diag((high - low) ** 2 / 12.0)
    return Whitening(mu, sigma, mu.size)


def flat_space(family: str, arm: str) -> LearnerSpace:
    """The arm's space over a family's own box, with no population knowledge.

    ``family`` is checked again here although ``check_point`` already rejects an unsupported
    family-and-``"flat"``-prior combination first for every caller that goes through
    ``run_phase5_point``: this function is public and callable on its
    own, so it keeps its own guard.
    """
    if family not in FLAT_DESIGNS:
        raise ValueError(f"family must be one of {tuple(FLAT_DESIGNS)}, got {family!r}")
    design = FLAT_DESIGNS[family]()
    whitening = flat_whitening(design)
    return LearnerSpace(DeviceDesign(design), whitening, ARMS[arm].radius(whitening.mu.size))


def learner_seed(answer_seed: int) -> int:
    return int(answer_seed) % 2**32


def build_learner(point: Phase5Point, space: LearnerSpace | None, seed: int, design=None, *,
                  s_user: float | None = None):
    """``s_user`` is the wearer's own sensitivity (:func:`per_user_s`); ``None`` means
    ``point.s``. Only G reads it: R34's ``jnd_scale`` is a multiple of the wearer's true JND."""
    if point.learner == "c1":
        # Phase 4's C1: a batch of 40 on the box, with linear reward in its six parameters.
        return BatchMILearner(design, seed=seed)
    prior = ARMS[point.arm].bowl_prior()
    if point.learner == "a" and point.hyper_log_ell0 is not None:
        # H2, PsyPref-hyper: A with the fitted priors on log ell and lambda, nothing else changed.
        prior = replace(prior, log_ell0=float(point.hyper_log_ell0),
                        ell_sd=float(point.hyper_ell_sd), lam_a=float(point.hyper_lam_a),
                        lam_b=float(point.hyper_lam_b))
    if point.learner == "a" and point.a_rt:
        # RT, PsyPref-RT: A with the joint likelihood of choice and response time.
        return RTLearner(space.design, seed=seed, space=space, prior=prior,
                         params=load_rt_params(point.run_dir))
    if point.learner == "a" and point.a_no_lapse:
        # The review response's lapse ablation: A with lambda = 0 in its model and no probes.
        return NoLapseLearner(space.design, seed=seed, space=space, prior=prior)
    if point.learner == "a" and point.a_no_probes:
        # The probe ablation (PA4): A's own posterior and lapse model, no intensity probes.
        return PsychometricLearner(space.design, seed=seed, duels="eig", space=space,
                                   prior=prior, p_probe=0.0)
    if point.learner == "a" and point.a_metric:
        # Stress test S2, PsyPref-M: A with the phase-weighted metric in its particle state.
        return MetricLearner(space.design, seed=seed, space=space, prior=prior)
    if point.learner == "a" and point.acq_variant is not None:
        # ICORR section 9a, "A": one change to A's acquisition, everything else A's own.
        return PsychometricLearner(space.design, seed=seed, duels="eig", space=space,
                                   prior=prior, **ACQ_VARIANTS[point.acq_variant])
    if point.learner == "a":
        # W3's ablation: duel_mode "random" keeps the posterior, the prior and the space and
        # changes only how duels are chosen (and, as Phase 3 defined it, that the delivered
        # curve is the posterior mean unclipped to the trust ball; rows record
        # best_outside_trust_region).
        return PsychometricLearner(space.design, seed=seed, duels=point.duel_mode, space=space,
                                   prior=prior, stop_sd=point.stop_sd)
    if point.learner == "a_lens":
        inner_space = arm_space(_prior(point.run_dir, point.fold, "promp_s0"), point.arm)
        inner = PsychometricLearner(inner_space.design, seed=seed, duels="eig", space=inner_space,
                                    prior=prior, stop_sd=point.stop_sd)
        fitter = SurfaceFitter(space.design, start_mean=space.whitening.mu)
        return LensLearner(inner, inner_space, space.design, fitter)
    if point.learner in GP_ACQUISITION:
        from ..learning.gp_eubo import GPEUBOLearner  # with botorch installed

        # The arm's ball and whitening, as A and G search them; the arm's prior is the measure
        # its raw pool is drawn from (as G's candidates). WhitenedLearner maps z to the arm's
        # device-clipped design, so its identified assistance is delivered like the others'.
        return WhitenedLearner(space, GPEUBOLearner(
            space, seed=seed, prior=prior, acquisition=GP_ACQUISITION[point.learner]))
    if point.learner == "cmaes":
        from ..learning.sortcma import KUTULAKOS_SIGMA_FRACTION, SortCMALearner  # venv only

        # L10: the prior's own scale. L12's uninformed start: Kutulakos & Slade's 0.3 x the
        # box range per coordinate, which in the flat box's moment-matched whitening is
        # 0.3 x 2 sqrt(3).
        sigma0 = KUTULAKOS_SIGMA_FRACTION * 2.0 * math.sqrt(3.0) if point.prior == "flat" else 1.0
        if point.cma_sigma_mult is not None:
            sigma0 *= float(point.cma_sigma_mult)   # E8's short-budget tuning
        return WhitenedLearner(space, SortCMALearner(space.as_design(), seed=seed, sigma0=sigma0,
                                                     popsize=point.cma_popsize))
    if point.learner == "d_z":
        ls = DZ_TUNED["lengthscale_mult"] if point.lengthscale_mult is None else point.lengthscale_mult
        st = DZ_TUNED["step_mult"] if point.step_mult is None else point.step_mult
        return WhitenedLearner.linecospar(space.design, seed=seed, space=space,
                                          lengthscale=LENGTHSCALE * ls, step=STEP * st)
    if point.learner == "g":
        # `prior` here for the same reason A and C2 get it: without
        # it G's candidates are `WhitenedDesign.sample_prior`'s standard normal in every
        # arm, so in four of five arms G searched a strictly narrower region than the
        # learners it is compared against.
        tuned = {} if point.g_kappa0 is None else {"kappa0": float(point.g_kappa0)}  # E8
        s = point.s if s_user is None else s_user
        if point.g_start == "generic":
            # The review response: G opens from generic assistance (prompref.experiment.review).
            return generic_start_g(space, seed, prior=prior, jnd=gate_value(s, point.jnd_scale),
                                   n_candidates=int(point.n_candidates), **tuned)
        return WhitenedLearner.gp_ucb(space.design, seed=seed, space=space, prior=prior,
                                      jnd=gate_value(s, point.jnd_scale),
                                      n_candidates=int(point.n_candidates), **tuned)
    return BatchMILearner(SpacePriorDesign(space, prior), seed=seed)


def reach_outcomes(distances, floor_nm: float, taus=TAUS) -> dict:
    """Comparisons until the delivered curve is within ``floor + tau``.

    ``first_*`` counts until the curve first gets there; ``sustained_*`` counts until it gets
    there and stays for the rest of the session. Both are 1-based, like ``n90``; ``None``
    means censored at the session's length.
    """
    d = np.asarray(distances, dtype=float)
    out = {}
    for tau in taus:
        label = f"{int(round(tau * 100)):03d}"
        inside = d <= np.asarray(floor_nm, dtype=float) + tau + 1e-12
        first = int(np.argmax(inside)) + 1 if inside.any() else None
        if inside.size and inside[-1]:
            outside = np.flatnonzero(~inside)
            sustained = int(outside[-1]) + 2 if outside.size else 1
        else:
            sustained = None
        out[f"first_{label}"], out[f"sustained_{label}"] = first, sustained
    return out


def region_outcomes(distances, floor_nm, plateau_nm: float, taus=TAUS) -> dict:
    """our design notes S16's region criterion: comparisons until the delivered curve is within
    ``tau`` of the best reachable distance *to the preferred region*, reached and sustained.

    The region is every curve within ``plateau_nm`` RMS of the ideal, so a curve's
    distance to it is ``max(0, d - plateau)``, and the family's best is
    ``max(0, floor - plateau)``. Where the floor lies outside the region this is R25
    exactly; where it lies inside, it is more lenient, which is why R25 is reported beside it.
    """
    p = float(plateau_nm)
    to_region = np.maximum(0.0, np.asarray(distances, dtype=float) - p)
    best = np.maximum(0.0, np.asarray(floor_nm, dtype=float) - p)
    return {f"region_{name}": value
            for name, value in reach_outcomes(to_region, best, taus).items()}


def candidate_floor(learner, ideal_nm) -> float:
    """G's best possible delivery: the smallest RMS distance between any of
    its device-clipped candidate curves and ``ideal_nm``. Reads the curves G computed at
    construction; draws no random numbers."""
    curves = np.asarray(learner.inner.curves, dtype=float)
    diff = curves - np.asarray(ideal_nm, dtype=float)
    return float(np.sqrt(np.mean(diff**2, axis=1)).min())


@lru_cache(maxsize=16)
def _users(run_dir: str, fold: str):
    return load_fold_users(run_dir, fold)


@lru_cache(maxsize=16)
def _ood_users(run_dir: str, fold: str, user_set: str = OOD_USER_SET):
    return load_ood_users(run_dir, fold, user_set)


def point_users(point: Phase5Point) -> dict:
    """The users ``point.user_id`` is looked up in: the run's exported users, or an
    out-of-generator user set: "ood", the common-peak users, or
    "ood-pb", the per-bout-peak users of the 2026-10-19 decision."""
    if point.user_set in USER_SETS:
        return _ood_users(point.run_dir, point.fold, point.user_set)
    return _users(point.run_dir, point.fold)


@lru_cache(maxsize=16)
def random_utility_sidecar(run_dir: str, fold: str) -> dict:
    """The fold's random-utility sidecar."""
    return load_random_utility(run_dir, fold)


@lru_cache(maxsize=16)
def phase_weight_sidecar(run_dir: str, cell: str, fold: str, user_set: str | None = None) -> dict:
    """The fold's phase-weight sidecar of one cell, for
    the run's exported users or, with ``user_set``, for that user set's users (R2's PESS)."""
    return load_phase_weight_sidecar(run_dir, cell, fold, user_set)


@lru_cache(maxsize=4)
def bio_weight(run_dir: str) -> tuple[np.ndarray, str]:
    """W-bio's weight curve and its sidecar's sha256; raises when the sidecar is missing."""
    return load_bio_weight(run_dir)


def phase_weight_knobs(point: Phase5Point, user) -> dict:
    """The observer fields of a phase-weighted wearer: the metric, the user's weight curve and
    matched scale. W-pref's weight is rebuilt from the answer seed and must equal the one the
    sidecar's scale was computed with; W-bio's comes from ``out/icorr/wph_bio_weight.json``, and
    its sha256 must be the one the sidecar records."""
    if point.user_set is None:
        sidecar = phase_weight_sidecar(point.run_dir, point.phase_weighting, point.fold)
    else:
        sidecar = phase_weight_sidecar(point.run_dir, point.phase_weighting, point.fold,
                                       point.user_set)
    entry = sidecar_entry(sidecar, user.user_id, user.answer_seed)
    if point.phase_weighting in PREF_CELLS:
        # W-pref, or the stress test S1 ("pref-sd1"): the same coefficients, the cell's SD.
        if point.phase_weighting == "pref":
            weight = pref_weight(user.answer_seed)
        else:
            weight = pref_weight(user.answer_seed, PREF_CELLS[point.phase_weighting])
        if not np.array_equal(weight, np.asarray(entry["weight"], dtype=float)):
            raise ValueError(f"{user.user_id}: the W-pref weight rebuilt from the answer seed "
                             "differs from the sidecar's; the sidecar is stale")
    else:
        weight, sha = bio_weight(point.run_dir)
        if sha != sidecar["weight_sha256"]:
            raise ValueError("the W-bio weight file changed after the sidecar's scales were "
                             f"computed (sha256 {sha[:12]} vs {sidecar['weight_sha256'][:12]}); "
                             "rerun scripts/icorr_wph_prep.py --cell bio")
    return dict(metric=PHASE_WEIGHTED, phase_weight=weight, utility_scale=float(entry["scale"]))


@lru_cache(maxsize=16)
def _prior(run_dir: str, fold: str, family: str) -> DataPrior:
    return DataPrior.load(Path(run_dir) / "priors" / f"{fold}-{family}")


FRANKS_SCRIPT = "scripts/phase5d_franks.py"


@lru_cache(maxsize=16)
def _franks_prior(run_dir: str, fold: str) -> FranksPrior:
    stem = Path(run_dir) / "priors" / f"{fold}-franks"
    if not Path(f"{stem}.json").exists():
        raise FileNotFoundError(f"{stem}.json is missing: run {FRANKS_SCRIPT} --run {run_dir} first")
    return FranksPrior.load(stem)


@lru_cache(maxsize=16)
def _surface_prior(run_dir: str, fold: str, family: str) -> SurfacePrior:
    """A projection surface's prior, written by :data:`PROJ_SCRIPT`."""
    path = surface_prior_path(run_dir, fold, family)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run {PROJ_SCRIPT} --run {run_dir} first")
    prior = SurfacePrior.load(path)
    if prior.family != family or prior.fold != fold:
        raise ValueError(f"{path} holds the {prior.family} prior of {prior.fold!r}, not the "
                         f"{family} prior of {fold!r}")
    return prior


@lru_cache(maxsize=16)
def proj_floors(run_dir: str, fold: str, family: str) -> dict:
    """Per-user floors of a projection surface: Schaefer (ii)'s own sidecar, written by
    :data:`PROJ_SCRIPT`; the semi-trapezoid's are C1's floors over the same box and curve."""
    if family == "trapezoid_surface":
        return flat_floors(run_dir, fold, C1_FAMILY)
    path = Path(run_dir) / "users" / f"{fold}-{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run {PROJ_SCRIPT} --run {run_dir} first")
    return json.loads(path.read_text(encoding="utf-8"))


def _proj_floor_path(run_dir: str, fold: str, family: str) -> Path:
    if family == "trapezoid_surface":
        return _flat_sidecar_path(run_dir, fold, C1_FAMILY)
    return Path(run_dir) / "users" / f"{fold}-{family}.json"


@lru_cache(maxsize=16)
def _basis_prior(run_dir: str, fold: str, n_basis: int) -> BasisDataPrior:
    stem = basis_prior_stem(run_dir, fold, n_basis)
    if not Path(f"{stem}.json").exists():
        raise FileNotFoundError(f"{stem}.json is missing: run {BASIS_SCRIPT} --run {run_dir} "
                                f"--k {n_basis} first")
    prior = BasisDataPrior.load(stem)
    if prior.n_basis != int(n_basis) or prior.fold != fold:
        raise ValueError(f"{stem} holds the {prior.n_basis}-basis prior of {prior.fold}, not "
                         f"the {n_basis}-basis prior of {fold}")
    return prior


@lru_cache(maxsize=32)
def basis_floors(run_dir: str, fold: str, n_basis: int) -> dict:
    """Per-user ProMP floors at ``n_basis`` bases (W3), written by scripts/icorr_k_priors.py."""
    path = basis_floors_path(run_dir, fold, n_basis)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run {BASIS_SCRIPT} --run {run_dir} "
                                f"--k {n_basis} first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def franks_floors(run_dir: str, fold: str) -> dict:
    path = Path(run_dir) / "users" / f"{fold}-franks.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run {FRANKS_SCRIPT} --run {run_dir} first")
    return json.loads(path.read_text(encoding="utf-8"))


def _family_floor(point: Phase5Point, user) -> dict:
    """The one place a session's floor entry comes from (the flat-floors addition).

    Three branches, in order: a flat cell always reads its own family's flat-box sidecar,
    regardless of family. A data-prior Schaefer cell reads the Schaefer sidecar instead. Anything else reads the fold's own exported floor. Schaefer's flat and
    data-prior floors happen to agree in value -- its box (``TIMING_LOW``/``TIMING_HIGH``)
    never widens for a data prior the way ``promp_s0``'s does -- but a flat Schaefer cell
    still reads the flat sidecar rather than the Schaefer one: one predicate, not a
    per-family exception.
    """
    if point.prior == "flat":
        sidecar = flat_floors(point.run_dir, point.fold, point.family)
        path = _flat_sidecar_path(point.run_dir, point.fold, point.family)
        return _sidecar_entry(sidecar, point.user_id, path, point.run_dir)
    if point.family == "schaefer":
        sidecar = schaefer_floors(point.run_dir, point.fold)
        path = _schaefer_sidecar_path(point.run_dir, point.fold)
        return _sidecar_entry(sidecar, point.user_id, path, point.run_dir)
    if point.family == "franks":
        path = Path(point.run_dir) / "users" / f"{point.fold}-franks.json"
        return _sidecar_entry(franks_floors(point.run_dir, point.fold), point.user_id, path,
                              point.run_dir, script=FRANKS_SCRIPT)
    if point.family in PROJ_FAMILIES:
        # Test P: the surface's own floor, the target the projection can at best deliver.
        return _sidecar_entry(proj_floors(point.run_dir, point.fold, point.family), point.user_id,
                              _proj_floor_path(point.run_dir, point.fold, point.family),
                              point.run_dir, script=PROJ_SCRIPT)
    if point.n_basis is not None:
        # W3: a K cell is scored against the K family's own floor, never the 8-basis one.
        k = int(point.n_basis)
        return _sidecar_entry(basis_floors(point.run_dir, point.fold, k), point.user_id,
                              basis_floors_path(point.run_dir, point.fold, k), point.run_dir,
                              script=BASIS_SCRIPT)
    return user.families[point.family]


def build_observer(point: Phase5Point, user, *, s_user: float, plateau: float):
    """The session's simulated wearer, and a drift cell's sidecar entry (``None`` otherwise).

    The observer runs at ``s_user`` (:func:`per_user_s`) with the preferred region
    ``plateau`` already resolved. Phase 5e's knobs reach it only when they
    differ from the plain observer, so a session that leaves them off constructs exactly
    the observer it always did: familiarity brings its zero-torque habitual profile, and
    device noise its own generator, salted from the answer seed.
    """
    knobs = dict(s=s_user, lapse=point.lapse, order_bias=point.order_bias,
                 overdispersion=point.overdispersion, plateau_nm=plateau)
    if point.observer_metric != "rms":
        knobs["metric"] = point.observer_metric
    if point.landmark_mag_fraction is not None:
        knobs["landmark_mag_fraction"] = float(point.landmark_mag_fraction)
    if point.decisiveness_ratio is not None:
        knobs["utility_divisor"] = float(point.decisiveness_ratio)
    if point.observer_metric == RANDOM_SMOOTH:
        knobs["smooth_utility"] = user_utility(random_utility_sidecar(point.run_dir, point.fold),
                                               user.user_id, user.answer_seed)
    if point.phase_weighting is not None:
        knobs.update(phase_weight_knobs(point, user))
    if point.asymmetry != 1.0:
        knobs["asymmetry"] = float(point.asymmetry)
    if point.familiarity_weight > 0.0:
        knobs.update(familiarity_weight=float(point.familiarity_weight),
                     habitual_nm=np.zeros_like(user.curve_nm))
    if point.device_noise_nm > 0.0:
        knobs.update(device_noise_nm=float(point.device_noise_nm),
                     noise_rng=np.random.default_rng([int(user.answer_seed), NOISE_SALT]))
    if point.link is not None:
        # Stress test S5: the probit replaced, at the matched scale.
        knobs.update(link=point.link, link_scale=float(point.link_scale))
    if point.no_pref_jnd is not None:
        # Stress test S7: "no preference" below no_pref_jnd true JNDs; the threshold is in units
        # of s, and true_jnd(s) / s is the same Phi^-1(0.75) at every s.
        knobs.update(allow_no_preference=True,
                     no_pref_threshold=float(point.no_pref_jnd) * true_jnd(1.0),
                     no_pref_hard=bool(point.no_pref_hard))
    if point.rt_user is not None:
        # The response-time test (section 9a, RT): the same wearer answering by a DDM.
        return DDMObserver(ideal_nm=user.curve_nm, **knobs, rt_mode=point.rt_user,
                           ddm=load_rt_params(point.run_dir)), None
    if point.lapse_start is not None or point.s_start is not None:
        # Stress tests S3 and S4: the answer model changes linearly over the session.
        paths = {}
        if point.lapse_start is not None:
            paths["lapse_path"] = linear_path(point.lapse_start, point.lapse_end,
                                              point.n_comparisons)
        if point.s_start is not None:
            paths["s_path"] = linear_path(point.s_start, point.s_end, point.n_comparisons)
        return ScheduledObserver(ideal_nm=user.curve_nm, **knobs, **paths), None
    if point.drift_nm > 0.0:
        drift = _drift_entry(point)
        if point.drift_axis == "magnitude":
            path = path_for_scale(user.curve_nm, drift["scale"], point.n_comparisons)
        else:
            path = path_for_shift(user.curve_nm, drift["shift_pct"], point.n_comparisons)
        return DriftingObserver(ideal_nm=path[0], ideal_path=path, **knobs), drift
    return Observer(ideal_nm=user.curve_nm, **knobs), None


def is_observer_variant(point: Phase5Point) -> bool:
    """Whether the wearer answers by a distance other than A's own RMS model: the
    landmark or Weber metric, asymmetry, or familiarity. Noise, lapse and sensitivity change
    how the wearer answers, not what they prefer."""
    return (point.observer_metric != "rms" or point.asymmetry != 1.0
            or point.familiarity_weight > 0.0 or point.phase_weighting is not None)


def own_scores(point: Phase5Point, observer: Observer) -> dict:
    """The wearer's own outcome per comparison, as ``run_session``
    scores: nothing for a plain wearer, whose own distance is the RMS ``distances``.

    For an observer variant, ``own_distances`` is the observer's own distance of the
    delivered curve, ``-utility``, noise-free like ``distances``: in Nm-equivalent for the
    landmark observer (feature distance x true_jnd(s)), the weighted RMS for asymmetry, the
    RMS to the biased target for familiarity. The Weber metric changes only the probit
    scale, so its own distance is the RMS distance itself. A landmark cell also records
    ``own_feature_jnd``, the feature distance D in feature-JNDs.
    """
    if not is_observer_variant(point):
        return {}
    scores = {"own_distances": observer.own_distance}
    if point.observer_metric == "landmark":
        scores["own_feature_jnd"] = observer.feature_distance
    return scores


def run_phase5_point(point: Phase5Point, *, snapshot=None) -> dict:
    """One session. Never raises, so one bad point cannot abort a long sweep.

    ``snapshot`` (calibration-logging points only; the video logs of section 9a) is called as
    ``snapshot(k, learner)`` before the first comparison (k = 0) and after the comparisons in
    :data:`~prompref.experiment.calibration.SNAPSHOT_KS`. It must only read the learner."""
    try:
        if snapshot is not None and not point.calibration_log:
            raise ValueError("snapshot= needs a calibration_log point")
        check_point(point)
        user = point_users(point)[point.user_id]
        if point.learner == "c1":
            space = None
            design = DeviceDesign(TrapezoidDesign())
        elif point.prior == "flat":
            space = flat_space(point.family, point.arm)
        elif point.family == "schaefer":
            # a change anchored on the run dir, not TimingPrior.load's CWD-relative
            # default -- out/ is git-ignored and absent from a detached repository, where 5c
            # launches from.
            inputs = Path(point.run_dir).parent / "schaefer" / "prior_inputs.json"
            space = arm_space(TimingPrior.load(inputs), point.arm)
        elif point.family == "franks":
            space = arm_space(_franks_prior(point.run_dir, point.fold), point.arm)
        elif point.family in PROJ_FAMILIES:
            # Test P: the surface the projection fits to, in its own prior's space (the
            # arm sets only a radius, which the projection's fitter never reads).
            space = arm_space(_surface_prior(point.run_dir, point.fold, point.family), point.arm)
        elif point.n_basis is not None:
            # W3: the fold's K prior; the arm's radius follows its dimension K.
            space = arm_space(_basis_prior(point.run_dir, point.fold, int(point.n_basis)),
                              point.arm)
        else:
            space = arm_space(_prior(point.run_dir, point.fold, point.family), point.arm)
        if space is not None:
            design = space.design
        family = _family_floor(point, user)
        # The wearer's own sensitivity. Everything that models their perception
        # reads it -- the observer, its anchored region, G's JND gate, the drift tracking
        # error in JND -- while point.s stays the cell's identity. Without a log-SD it is
        # point.s itself.
        s_user = per_user_s(point.s, point.s_user_logsd, user.answer_seed)
        plateau = point.plateau_nm
        if point.plateau_over_floor_jnd > 0.0:
            plateau = family["floor_nm"] + point.plateau_over_floor_jnd * true_jnd(s_user)
        observer, drift = build_observer(point, user, s_user=s_user, plateau=plateau)
        learner = build_learner(point, space, learner_seed(user.answer_seed), design=design,
                                s_user=s_user)
        if point.a_rt:
            # PsyPref-RT reads each answer's response time from the wearer that gave it.
            learner.bind_response_times(observer)
        scores = own_scores(point, observer)
        if point.calibration_log:
            # Section 9a: A's confidence after every comparison. Reads the posterior only, and
            # takes its k = 0 snapshot here, before the first proposal.
            monitor = CalibrationMonitor(learner, space, snapshot=snapshot)
            scores = dict(scores, **monitor.scores())
        # A flat cell's N90 denominator must come from the flat prior the learner actually
        # searches (the whitened design, truncated to the arm's ball), not the family's own
        # fitted prior -- else the baseline quietly carries back the population knowledge
        # the flat cell exists to withhold.
        d_prior = prior_mean_distance(
            space.as_design() if (point.prior == "flat" and space is not None) else design,
            user.curve_nm)
        started = time.perf_counter()
        try:
            record = run_session(design, learner, observer,
                                 np.random.default_rng([int(user.answer_seed), OBSERVER_SALT]),
                                 n_comparisons=point.n_comparisons, randomise_order=True,
                                 d_prior=d_prior, d_floor=family["floor_nm"],
                                 scores=scores)
        finally:
            # A CMA session usually ends mid-generation; end its sort thread now rather than
            # at garbage collection (a sweep runs thousands of them per worker).
            inner = getattr(learner, "inner", learner)
            if hasattr(inner, "close"):
                inner.close()
        elapsed = time.perf_counter() - started
        floor = family["floor_nm"]
        if drift is not None:
            # The floor at comparison k, interpolated from the start ideal's to the final
            # ideal's: exact at both ends, an approximation between.
            steps = np.arange(len(record["distances"])) / max(point.n_comparisons - 1, 1)
            floor = family["floor_nm"] + steps * (float(drift["floor_nm"]) - family["floor_nm"])
            tracking = record["distances"][-1] - float(drift["floor_nm"])
            if point.drift_axis == "magnitude":
                record.update(drift_scale=float(drift["scale"]))
            else:
                record.update(drift_shift_pct=float(drift["shift_pct"]),
                              drift_capped=bool(drift["capped"]))
            record.update(final_floor_nm=float(drift["floor_nm"]),
                          tracking_error_nm=tracking,
                          tracking_error_jnd=tracking / true_jnd(s_user))
            # n90/censored/delivered_over_floor came from summarise_run's d_prior/d_floor
            # of the START ideal, while distances/delivered track the moving one -- a
            # complete, plausible, wrong number against a target that never held still.
            # tracking_error_* and the (interpolated-floor) first_*/sustained_* below are
            # the drift outcomes instead.
            record.update(n90=None, censored=None, delivered_over_floor=None)
        record.update(reach_outcomes(record["distances"], floor))
        if plateau > 0.0:
            record.update(region_outcomes(record["distances"], floor, plateau))
        if point.plateau_over_floor_jnd > 0.0:
            # Not "plateau_nm": that is a point field, and a record key of that name would
            # overwrite it in the row and break the resume key.
            record["plateau_nm_used"] = float(plateau)
        if point.familiarity_weight > 0.0:
            # The optimizers chase the familiarity-biased target, the score is against the
            # true ideal: how far apart the two sit.
            record["familiarity_target_gap_nm"] = rms_distance(observer.effective_ideal_nm,
                                                               observer.ideal_nm)
        if point.learner == "g":
            record["g_candidate_floor"] = candidate_floor(learner, observer.ideal_nm)
        if point.g_start is not None:
            record.update(generic_start_fields(learner, design, observer.ideal_nm))
        if point.decisiveness_ratio is not None or point.observer_metric == RANDOM_SMOOTH:
            # ICORR robustness: the wearer's own distance (-utility) of generic assistance, the
            # mean curve d_prior is the distance of, so the own-utility win rate reads the row.
            record["own_generic_distance"] = observer.own_distance(prior_mean_curve(design))
        if point.observer_metric == RANDOM_SMOOTH:
            entry = random_utility_sidecar(point.run_dir, point.fold)["users"][point.user_id]
            # The matched-decisiveness scale the utility carries. u_max is not copied into the
            # row: it is a per-user constant, the best any search has found, and lives in the
            # fold's sidecar, where later searches may raise it.
            record["utility_scale"] = float(entry["scale"])
        if point.phase_weighting is not None:
            # ICORR phase-weighted wearers: the own distance of generic assistance, on the matched
            # scale, and the scale itself. The floor lives in the sidecar, as R's u_max does.
            record["own_generic_distance"] = observer.own_distance(prior_mean_curve(design))
            record["utility_scale"] = float(observer.utility_scale)
        if point.rt_user is not None:
            # The response-time test: every answer's response time (s), whether it was a lapse,
            # the PyDDM redraws, and the DDM's constants and library.
            ddm = observer.ddm
            record.update(response_times=list(observer.response_times),
                          rt_lapsed=list(observer.lapsed), rt_n_redraws=int(observer.n_redraws),
                          rt_boundary=float(ddm.boundary), rt_t0=float(ddm.t0),
                          rt_choice_scale=float(ddm.choice_scale),
                          rt_marginal_sha256=ddm.marginal_sha256, pyddm_version=pyddm_version())
        if point.hyper_log:
            # H2's first pass: draws of the final posterior over (log ell, logit lambda); reads it.
            record["hyper_particles"] = posterior_sample(learner.posterior)
        record.update(session_s=elapsed, s_user=float(s_user), user_kind=user.kind,
                      user_reach_050=family["reach_050"], user_radius=family["radius"],
                      arm_radius=None if space is None else space.radius, draw=user.draw,
                      delivered_two_bout=two_bout_plausible(
                          design.curve(np.asarray(record["best"], dtype=float))))
        return dict(point.row_fields(), **record, error=None)
    except Exception as exc:  # pragma: no cover - defensive
        return dict(point.row_fields(), error=f"{type(exc).__name__}: {exc}")


def run_phase5_sweep(points: list[Phase5Point], path, workers: int | None = None,
                     resume: bool = True) -> Path:
    return run_sweep(points, path, workers=workers, resume=resume, runner=run_phase5_point)
