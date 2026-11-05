"""The simulated observer: a utility over curves plus a psychometric response model.

Utility levels are selected by fields rather than subclasses:
``asymmetry`` and ``plateau_nm`` give the wrong-family simulated user O3, ``familiarity_weight``
plus ``habitual_nm`` give the familiarity-bias simulated user O6, and the plain settings give the
curve-L2 simulated user O2.

Three modelling decisions worth stating:

- **Beta is not a parameter.** Beta and the utility scale are not separately
  identifiable, so ``s`` carries the slope -- which makes ``s`` literally the JND scale,
  in Nm.
- **Utility is linear in distance, not squared.** That keeps ``s`` in Nm so the
  published anchors in our design notes apply directly. The quadratic structure the
  learner assumes lives in the *distance*, and :func:`squared_distance_hessian` is what
  pins it.
- **Familiarity shifts the target, it does not add a second distance term.** An earlier
  version of O6 computed ``d(curve, ideal) + w * d(curve, habitual)`` -- adding a second
  distance penalty on top of the first. Because a sum of two norms is *affine* along the
  segment joining their two targets, that formula is bang-bang: its optimum sits at
  ``ideal`` for any ``w < 1``, is flat (a tie) along the *entire* segment at ``w == 1``,
  and jumps straight to ``habitual`` for any ``w > 1`` -- never anywhere in between. That
  is not the graded familiarity bias O6 exists to model. Instead, O6 shifts the *target*
  itself to ``(ideal_nm + w * habitual_nm) / (1 + w)`` and takes a single distance to
  that blended target, computed once in ``__post_init__``. This interpolates
  continuously: the utility-maximising point on the ideal-habitual segment sits at
  fraction ``w / (1 + w)`` of the way from ideal to habitual, for every ``w >= 0``. The
  utility is still linear in the (single) distance, so ``s`` still carries the slope and
  stays in Nm.

Phase 5e adds two simulated users who do not judge by A's own model, and a device that
does not deliver exactly what it is told:

- ``metric="landmark"`` judges four features of a curve -- the extension and flexion peaks'
  magnitude and timing -- each in literature JNDs, and maps one feature-JND to one RMS-JND.
- ``metric="weber"`` keeps the RMS utility but widens the per-duel scale with the torque
  shown, after the Hatzfeld form JND = k (RMS + a).
- ``device_noise_nm`` perturbs each curve the simulated user feels with smooth noise from its own
  generator. Only the observer sees it: learners propose parameters, and the session scores
  the identified curve against the ideal without it.

``metric="rms"`` with no noise is the observer every earlier session used, unchanged.

The ICORR reframe's robustness tests add two more:

- ``utility_divisor`` divides the landmark simulated user's utility by its cell's decisiveness ratio,
  so it answers as decisively as the plain s = 1.0 RMS simulated user (the home-advantage test at
  matched decisiveness). 1.0 leaves the utility untouched, and every earlier path skips it.
- ``metric="random_smooth"`` answers by a random smooth utility of the curve's Fourier
  features (:mod:`prompref.user.smooth_utility`), given as ``smooth_utility``.

The phase-weighted simulated users of the same section (W-bio, W-pref) add one more, off by default:

- ``metric="phase_weighted"`` judges by the phase-weighted RMS distance
  ``d_w = sqrt(mean_phi w(phi) (tau - tau*)^2)`` to the target, with ``phase_weight`` the weight
  curve on the curve's grid (mean 1 over all its points, as ``rms_distance`` averages them), and
  multiplies it by ``utility_scale``, the per-user matched-decisiveness scale. With ``w = 1`` and a
  scale of 1 it is the RMS simulated user. Every other metric skips both fields.

The stress tests of the same section add two
answer models, both off by default:

- ``link`` replaces the probit of a duel by a logistic ("logistic"), a Student-t with 3 degrees
  of freedom ("student_t") or, for the second review round's HT (c), a Cauchy ("cauchy", Student-t
  with 1 degree of freedom) at the scale ``s * link_scale``. ``link_scale`` is measured once so that
  the median |p - 1/2| on prior-scale pairs equals the plain s = 1.0 simulated user's. "probit" (the
  default) never reads ``link_scale``.
- ``no_pref_hard`` turns the "no preference" band of ``allow_no_preference`` into a hard rule on
  the simulated user's true utility difference: an attentive simulated user answers "no preference" exactly when
  ``|u_A - u_B| < no_pref_threshold * s``, and otherwise answers by the probit; a lapse is a coin
  flip between A and B, as without the band. False (the default) keeps our design notes's soft band, where
  the noisy difference falls inside the band and the lapse is spread over all three answers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from preftune.profile import N_PHASE
from scipy.special import expit
from scipy.stats import norm
from scipy.stats import t as student_t

from ..basis import VonMisesBasis
from .jnd import true_jnd

#: Response codes.
PREFER_A, PREFER_B, NO_PREFERENCE = 0, 1, 2

#: How the simulated user compares two curves. "rms" is A's own model. "random_smooth" is
#: the random smooth utility.
METRICS = ("rms", "landmark", "weber", "random_smooth", "phase_weighted")
RANDOM_SMOOTH = "random_smooth"
#: The ICORR reframe's phase-weighted simulated users.
PHASE_WEIGHTED = "phase_weighted"
#: A phase weight's mean must be 1 to within this.
PHASE_WEIGHT_MEAN_TOL = 1e-9

#: Landmark observer (E4): magnitude JND as a fraction of the ideal's peak (static force
#: Weber fractions 7-10 %, Jones 1989; Pang 1991); timing JND in % of the gait cycle (Peng
#: 2022, measured while walking). The fraction is the observer's ``landmark_mag_fraction``:
#: this static value by default, 0.25 in the walking-calibrated cell,
#: because walking thresholds are 3-4x the static ones.
LANDMARK_MAG_FRACTION = 0.10
#: The magnitude JND never falls below this, in Nm (E4b): max(|peak| x fraction, best-profile distance). A
#: best-profile distance on the JND itself, so a near-zero peak cannot get a vanishing JND. It replaces the
#: earlier 0.1 Nm best-profile distance on the *peak*, which left a JND of 0.01 Nm.
LANDMARK_MAG_FLOOR_NM = 0.5
LANDMARK_TIME_JND_PCT = 2.8
#: Points within this of a curve's extremum are tied with it. A peak clipped at the device
#: limit is a plateau: 91 of the 264 users exported for the six Phase 5 folds touch +-12 Nm,
#: with up to 37 of the cycle's 100 points at the limit.
LANDMARK_TIE_NM = 1e-9
#: A curve whose maximum is at most this has no flexion part, and one whose minimum is at
#: least minus this has no extension part: that part's peak has no timing.
LANDMARK_PART_NM = 0.05

#: Weber observer (E4): s_eff = s (RMS_pair + a) / (RMS_typ + a). RMS_typ is the
#: population-median RMS of the ideals, so s_eff equals s at typical torque.
WEBER_A_NM = 1.0
#: The design's value: the population-median RMS over the 100 distinct points of the cycle.
#: The code takes RMS over all 101 points, where the median is 4.816 Nm. The difference is
#: negligible, and the design constant is kept.
WEBER_RMS_TYP_NM = 4.83

#: Device noise (E5): a random Fourier series over these harmonics of the gait cycle.
NOISE_HARMONICS = (1, 2, 3, 4)

#: ICORR stress test S5: the duel's link. "probit" is every earlier simulated user's. "cauchy" is the
#: second review round's heavy-tailed answers (our design notes, R2, HT (c), planned
#: 2026-10-22 before any code or run): Student-t noise with one degree of freedom.
LINKS = ("probit", "logistic", "student_t", "cauchy")
#: S5b's degrees of freedom.
STUDENT_T_DF = 3.0
#: R2 HT (c)'s degrees of freedom.
CAUCHY_DF = 1.0


def link_cdf(link: str, x):
    """The link's cdf at ``x`` (already divided by its scale): Phi, the logistic, Student-t with
    :data:`STUDENT_T_DF` degrees of freedom, or the Cauchy (Student-t with :data:`CAUCHY_DF`)."""
    if link == "probit":
        return norm.cdf(x)
    if link == "logistic":
        return expit(x)
    if link == "student_t":
        return student_t.cdf(x, STUDENT_T_DF)
    if link == "cauchy":
        return student_t.cdf(x, CAUCHY_DF)
    raise ValueError(f"link must be one of {LINKS}, got {link!r}")


def rms_distance(curve: np.ndarray, ideal: np.ndarray) -> float:
    """Root-mean-square difference over the gait cycle, in Nm."""
    diff = np.asarray(curve, dtype=float) - np.asarray(ideal, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def weighted_rms_distance(curve: np.ndarray, ideal: np.ndarray, weight: np.ndarray) -> float:
    """The phase-weighted RMS difference ``sqrt(mean(w (curve - ideal)^2))`` over the same points
    :func:`rms_distance` averages, in Nm. ``weight`` has mean 1, so ``w = 1`` is the RMS."""
    diff = np.asarray(curve, dtype=float) - np.asarray(ideal, dtype=float)
    return float(np.sqrt(np.mean(np.asarray(weight, dtype=float) * diff**2)))


def check_phase_weight(weight, n_points: int | None = None) -> np.ndarray:
    """A phase weight as an array, refused unless it is finite, positive, of ``n_points`` values
    (when given), periodic (the last point, 100 %GC, equals the first) and of mean 1 to within
    :data:`PHASE_WEIGHT_MEAN_TOL`."""
    w = np.asarray(weight, dtype=float)
    if w.ndim != 1 or w.size < 2:
        raise ValueError(f"a phase weight is a 1-D curve, got shape {w.shape}")
    if n_points is not None and w.size != int(n_points):
        raise ValueError(f"a phase weight needs {n_points} points, got {w.size}")
    if not np.all(np.isfinite(w)) or not np.all(w > 0.0):
        raise ValueError("a phase weight must be finite and positive")
    if w[-1] != w[0]:
        raise ValueError("a phase weight is periodic: its last point (100 %GC) must equal its "
                         "first (0 %GC)")
    if abs(float(np.mean(w)) - 1.0) > PHASE_WEIGHT_MEAN_TOL:
        raise ValueError(f"a phase weight has mean 1, got {float(np.mean(w))!r}")
    return w


def _plateau_centre(circle: np.ndarray, extremum: float) -> float:
    """The circular centre, as a fractional index, of the run of points tied with
    ``extremum`` (within :data:`LANDMARK_TIE_NM`) on the closed cycle ``circle``.

    A run may cross the seam from the last point to the first. If several separate runs
    tie, the longest wins, and among equally long runs the one starting at the lowest
    index. A curve tied everywhere has no peak to time; it gets index 0.
    """
    m = circle.size
    tied = np.abs(circle - extremum) <= LANDMARK_TIE_NM
    if tied.all():
        return 0.0
    # Rotate so the sequence starts on an untied point: then no run crosses the seam.
    shift = int(np.argmin(tied))
    rotated = np.roll(tied, -shift).astype(np.int8)
    edges = np.diff(np.concatenate([[0], rotated, [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    lengths = stops - starts + 1
    longest = np.flatnonzero(lengths == lengths.max())
    best = min(longest, key=lambda i: (starts[i] + shift) % m)
    return float((0.5 * (starts[best] + stops[best]) + shift) % m)


def landmark_features(curve: np.ndarray) -> tuple[float, float, float, float]:
    """``(extension peak, its phase, flexion peak, its phase)``, in Nm and % gait cycle.

    The features are read on the cycle's distinct points: the last point duplicates the
    first (0 and 100 %GC are one instant), so it is left out, as
    :func:`prompref.population.curves.landmarks` leaves it out. The extension peak is the
    minimum, the flexion peak the maximum. Its phase is the circular centre of the run of
    points tied with it: a peak clipped at the device limit is a plateau,
    and its onset would time it early by up to half the plateau.
    """
    curve = np.asarray(curve, dtype=float)
    circle = curve[:-1]
    to_pct = 100.0 / circle.size
    ext, flex = float(circle.min()), float(circle.max())
    return (ext, _plateau_centre(circle, ext) * to_pct,
            flex, _plateau_centre(circle, flex) * to_pct)


def _wrap_pct(delta: float) -> float:
    """A phase difference in %GC, wrapped round the cycle into [-50, 50)."""
    return (delta + 50.0) % 100.0 - 50.0


def landmark_distance(curve: np.ndarray, target: np.ndarray,
                      mag_fraction: float = LANDMARK_MAG_FRACTION) -> float:
    """The landmark observer's distance in feature-JNDs.

    ``sqrt(sum((delta / JND)^2))`` over the four features of :func:`landmark_features`.
    Each magnitude JND is ``max(mag_fraction x |target's peak|, LANDMARK_MAG_FLOOR_NM)``, so a target with little torque in one direction has a JND of at
    least half a newton metre; timing differences wrap round the cycle, so a peak at 98 %GC
    and one at 1 %GC are 3 %GC apart.

    A part's timing term needs that part in both curves: a curve whose
    maximum is at most :data:`LANDMARK_PART_NM` has no flexion part, and one whose minimum
    is at least ``-LANDMARK_PART_NM`` has no extension part. Without it, the part's
    magnitude term stays, against the target's peak with the same JND.
    """
    c_ext, c_ext_pct, c_flex, c_flex_pct = landmark_features(curve)
    t_ext, t_ext_pct, t_flex, t_flex_pct = landmark_features(target)
    ext_jnd = max(abs(t_ext) * mag_fraction, LANDMARK_MAG_FLOOR_NM)
    flex_jnd = max(abs(t_flex) * mag_fraction, LANDMARK_MAG_FLOOR_NM)
    terms = [(c_ext - t_ext) / ext_jnd]
    if c_ext < -LANDMARK_PART_NM and t_ext < -LANDMARK_PART_NM:
        terms.append(_wrap_pct(c_ext_pct - t_ext_pct) / LANDMARK_TIME_JND_PCT)
    terms.append((c_flex - t_flex) / flex_jnd)
    if c_flex > LANDMARK_PART_NM and t_flex > LANDMARK_PART_NM:
        terms.append(_wrap_pct(c_flex_pct - t_flex_pct) / LANDMARK_TIME_JND_PCT)
    return float(np.sqrt(sum(term * term for term in terms)))


def weber_scale(s: float, curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """The Weber observer's per-duel scale ``s (RMS_pair + a) / (RMS_typ + a)``,
    with ``RMS_pair`` the mean RMS torque of the two curves shown."""
    rms_a = float(np.sqrt(np.mean(np.asarray(curve_a, dtype=float) ** 2)))
    rms_b = float(np.sqrt(np.mean(np.asarray(curve_b, dtype=float) ** 2)))
    rms_pair = 0.5 * (rms_a + rms_b)
    return float(s) * (rms_pair + WEBER_A_NM) / (WEBER_RMS_TYP_NM + WEBER_A_NM)


def smooth_noise(rng: np.random.Generator, rms_nm: float, n_points: int = N_PHASE) -> np.ndarray:
    """Device tracking error: a Fourier series over :data:`NOISE_HARMONICS`
    with standard-normal sine and cosine coefficients, scaled to RMS ``rms_nm`` over the
    ``n_points`` grid of one gait cycle (0 to 100 %GC).

    Draws one ``(2, len(NOISE_HARMONICS))`` block from ``rng``: the sine coefficients, then
    the cosine ones.
    """
    t = np.arange(n_points) / (n_points - 1)
    k = np.asarray(NOISE_HARMONICS, dtype=float)
    sin_c, cos_c = rng.standard_normal((2, k.size))
    phase = 2.0 * np.pi * np.outer(t, k)
    noise = np.sin(phase) @ sin_c + np.cos(phase) @ cos_c
    return noise * (float(rms_nm) / float(np.sqrt(np.mean(noise**2))))


def squared_distance_hessian(basis: VonMisesBasis) -> np.ndarray:
    """``2 Phi^T Phi / N`` -- the Hessian of ``d^2(w)`` in basis-weight space.

    A curve distance is exactly a quadratic ideal-point model in weight space. That is the geometric
    fact the ideal-point model-shaped preference model rests on, and it is what the Phase 1 gate
    checks. Note it is the *squared* distance that is quadratic; the Hessian of the norm
    itself is not the Gram matrix, which is why the utility stays linear in distance.
    """
    return 2.0 * basis.gram


@dataclass
class Observer:
    """A simulated simulated user who prefers whichever curve sits closer to their ideal.

    ``order_bias`` and ``no_pref_threshold`` (its ``c``)
    are both **in units of ``s``**, not in Nm, and both are multiplied by ``s`` before
    use: ``Delta = u_A - u_B + delta * s * o``. That convention is what makes them mean
    the same thing at every sensitivity. ``s`` is swept across more than an order of
    magnitude, so adding ``order_bias`` raw -- as an earlier version did -- made the
    bias's effective size vary 20-fold across the sweep: at ``s`` = 0.1 with
    ``order_bias`` = 0.3 the response probability swung 0.0014 -> 0.9987, the order bias
    alone deciding every answer, while at ``s`` = 2.0 it swung only 0.44 -> 0.56. The
    ``confounds`` design would have reported a hugely overstated effect at small ``s``.
    """

    ideal_nm: np.ndarray
    s: float = 0.5
    lapse: float = 0.1
    #: Time-order error, in units of ``s``. Positive favours whichever curve was
    #: presented first.
    order_bias: float = 0.0
    overdispersion: float = 0.0
    familiarity_weight: float = 0.0
    habitual_nm: np.ndarray | None = None
    allow_no_preference: bool = False
    no_pref_threshold: float = 0.5
    asymmetry: float = 1.0
    plateau_nm: float = 0.0
    #: Phase 5e: how the simulated user compares curves, one of :data:`METRICS`.
    metric: str = "rms"
    #: Phase 5e: RMS in Nm of the smooth noise added to each curve the simulated user
    #: feels, drawn from ``noise_rng`` -- a generator of its own, so the answer stream is
    #: consumed exactly as without noise.
    device_noise_nm: float = 0.0
    noise_rng: np.random.Generator | None = None
    #: a change: the landmark observer's magnitude JND as a fraction of the
    #: target's peak. Only ``metric="landmark"`` reads it.
    landmark_mag_fraction: float = LANDMARK_MAG_FRACTION
    #: ICORR reframe, home-advantage test at matched decisiveness: the landmark utility is
    #: divided by this, its cell's measured decisiveness ratio. Only ``metric="landmark"``
    #: reads it; 1.0 skips the division entirely.
    utility_divisor: float = 1.0
    #: ICORR reframe, random smooth utility: a callable curve -> utility (already scaled),
    #: required by ``metric="random_smooth"`` and read by nothing else.
    smooth_utility: object | None = None
    #: ICORR reframe, phase-weighted simulated users: the weight curve w(phi) on the curve's grid (mean 1,
    #: periodic), required by ``metric="phase_weighted"`` and read by nothing else.
    phase_weight: np.ndarray | None = None
    #: ICORR reframe, phase-weighted simulated users: the per-user matched-decisiveness scale the weighted
    #: distance is multiplied by. Only ``metric="phase_weighted"`` reads it; 1.0 elsewhere.
    utility_scale: float = 1.0
    #: ICORR stress test S5: the duel's link, one of :data:`LINKS`, at the scale
    #: ``s * link_scale``. "probit" reads neither and is every earlier simulated user.
    link: str = "probit"
    link_scale: float = 1.0
    #: ICORR stress test S7: the hard "no preference" rule (module docstring). Needs
    #: ``allow_no_preference``.
    no_pref_hard: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.lapse <= 1.0:
            raise ValueError(f"lapse must be in [0, 1], got {self.lapse}")
        if self.s <= 0.0:
            raise ValueError(f"s must be positive, got {self.s}")
        if self.no_pref_threshold < 0.0:
            # A negative threshold makes the two symmetric bounds cross, so p_a + p_b
            # exceeds 1 and p_none is clamped to 0 -- the three returned probabilities
            # then sum to more than 1 and the observer is silently biased. Nothing
            # downstream can detect that, so refuse it here.
            raise ValueError(
                f"no_pref_threshold must be non-negative, got {self.no_pref_threshold}"
            )
        if self.familiarity_weight and self.habitual_nm is None:
            raise ValueError("familiarity_weight needs habitual_nm")
        if self.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {self.metric!r}")
        if self.metric == "landmark" and (self.asymmetry != 1.0 or self.plateau_nm > 0.0):
            # Both reshape an RMS distance; the landmark observer has none to reshape.
            raise ValueError("the landmark observer takes neither asymmetry nor plateau_nm")
        if self.landmark_mag_fraction <= 0.0:
            raise ValueError(f"landmark_mag_fraction must be positive, got "
                             f"{self.landmark_mag_fraction}")
        if self.metric != "landmark" and self.landmark_mag_fraction != LANDMARK_MAG_FRACTION:
            raise ValueError("only the landmark observer reads landmark_mag_fraction")
        if self.device_noise_nm < 0.0:
            raise ValueError(f"device_noise_nm must be non-negative, got {self.device_noise_nm}")
        if self.device_noise_nm > 0.0 and self.noise_rng is None:
            raise ValueError("device_noise_nm needs its own noise_rng")
        if self.utility_divisor != 1.0:
            divisor = float(self.utility_divisor)
            if not np.isfinite(divisor) or divisor <= 0.0:
                raise ValueError(f"utility_divisor must be positive and finite, got "
                                 f"{self.utility_divisor}")
            if self.metric != "landmark":
                raise ValueError("only the landmark observer reads utility_divisor")
        if (self.metric == RANDOM_SMOOTH) != (self.smooth_utility is not None):
            raise ValueError("metric 'random_smooth' needs smooth_utility, and only it reads it")
        if self.metric == RANDOM_SMOOTH and (self.asymmetry != 1.0 or self.plateau_nm > 0.0
                                             or self.familiarity_weight):
            raise ValueError("the random smooth utility takes neither asymmetry, plateau_nm "
                             "nor familiarity: it has no ideal to reshape")
        if (self.metric == PHASE_WEIGHTED) != (self.phase_weight is not None):
            raise ValueError("metric 'phase_weighted' needs phase_weight, and only it reads it")
        if self.utility_scale != 1.0:
            scale = float(self.utility_scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"utility_scale must be positive and finite, got "
                                 f"{self.utility_scale}")
            if self.metric != PHASE_WEIGHTED:
                raise ValueError("only the phase-weighted observer reads utility_scale")
        if self.metric == PHASE_WEIGHTED:
            if self.asymmetry != 1.0 or self.plateau_nm > 0.0 or self.familiarity_weight:
                raise ValueError("the phase-weighted wearer takes neither asymmetry, plateau_nm "
                                 "nor familiarity: only the weighting changes")
            self.phase_weight = check_phase_weight(self.phase_weight,
                                                   np.asarray(self.ideal_nm).size)
        if self.link not in LINKS:
            raise ValueError(f"link must be one of {LINKS}, got {self.link!r}")
        if self.link != "probit":
            scale = float(self.link_scale)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"link_scale must be positive and finite, got {self.link_scale}")
            if self.allow_no_preference or self.metric == "weber" or self.overdispersion > 0.0:
                raise ValueError("a non-probit link is not combined with the no-preference band, "
                                 "the Weber scale or overdispersion: only the link changes")
        elif self.link_scale != 1.0:
            raise ValueError("only a non-probit link reads link_scale")
        if self.no_pref_hard and not self.allow_no_preference:
            raise ValueError("no_pref_hard is a rule for the no-preference answer: it needs "
                             "allow_no_preference")
        self.ideal_nm = np.asarray(self.ideal_nm, dtype=float)
        if self.habitual_nm is not None:
            self.habitual_nm = np.asarray(self.habitual_nm, dtype=float)
        if self.familiarity_weight:
            w = self.familiarity_weight
            self.effective_ideal_nm = (self.ideal_nm + w * self.habitual_nm) / (1.0 + w)
        else:
            # No arithmetic here: familiarity_weight=0 must reduce to ideal_nm exactly,
            # bit-for-bit, since every other simulated user variant (O2, O3) depends on it.
            self.effective_ideal_nm = self.ideal_nm

    def utility(self, curve: np.ndarray) -> float:
        """Negated distance to the (possibly familiarity-shifted) target.

        O3's asymmetry/plateau and O6's familiarity shift both apply here: O6 moves the
        target itself (see the module docstring for why), and O3's asymmetric weighting
        and plateau are then computed against that same target -- exactly as they would
        be against the plain ``ideal_nm`` when familiarity_weight is 0.

        The landmark observer measures its feature distance against that same
        target, read at call time because :class:`~prompref.user.drift.DriftingObserver`
        moves it every comparison, and converts it to the probit scale as distance x
        ``true_jnd(s)``: one feature-JND then answers like one RMS-JND.
        """
        curve = np.asarray(curve, dtype=float)
        target = self.effective_ideal_nm
        if self.metric == "landmark":
            if self.utility_divisor != 1.0:
                # Matched decisiveness: the same utility on a flatter scale.
                return -self.feature_distance(curve) * true_jnd(self.s) / self.utility_divisor
            return -self.feature_distance(curve) * true_jnd(self.s)
        if self.metric == RANDOM_SMOOTH:
            return float(self.smooth_utility(curve))
        if self.metric == PHASE_WEIGHTED:
            # ICORR reframe, W-bio / W-pref: the weighted RMS on the matched scale.
            return -self.utility_scale * weighted_rms_distance(curve, target, self.phase_weight)
        diff = curve - target
        if self.asymmetry != 1.0:
            # Over-assistance means more torque magnitude than wanted, in either
            # direction of the sign convention.
            excess = np.abs(curve) - np.abs(target)
            weight = np.where(excess > 0.0, self.asymmetry, 1.0)
            distance = float(np.sqrt(np.mean(weight * diff**2)))
        else:
            distance = float(np.sqrt(np.mean(diff**2)))
        if self.plateau_nm > 0.0:
            distance = max(0.0, distance - self.plateau_nm)
        return -distance

    def feature_distance(self, curve: np.ndarray) -> float:
        """The landmark observer's distance of ``curve`` in feature-JNDs, against the current
        (possibly familiarity-shifted or drifting) target, at its own magnitude fraction."""
        if self.metric != "landmark":
            raise ValueError("only the landmark observer has a feature distance")
        return landmark_distance(curve, self.effective_ideal_nm, self.landmark_mag_fraction)

    def own_distance(self, curve: np.ndarray) -> float:
        """The simulated user's own distance of ``curve``: ``-utility``, in Nm or
        Nm-equivalent. Noise-free, like ``utility``.

        It is the distance this simulated user answers by, which need not be the RMS distance to the
        ideal that every session is scored in: the landmark observer's feature distance x
        ``true_jnd(s)``, the asymmetric weighted RMS, or the RMS to the familiarity-shifted
        target. The Weber observer changes only the probit scale of a duel, not the utility,
        so its own distance is the RMS distance itself.
        """
        return -self.utility(curve)

    def _scale(self, rng: np.random.Generator | None, base: float | None = None) -> float:
        """The duel's probit scale: ``base`` (``s`` unless given) times the trial's
        overdispersion factor."""
        s = self.s if base is None else base
        if self.overdispersion <= 0.0 or rng is None:
            return s
        return float(s * np.exp(rng.normal(0.0, self.overdispersion)))

    def prob_prefers_a(self, curve_a: np.ndarray, curve_b: np.ndarray, order: int = 1,
                       rng: np.random.Generator | None = None):
        """Response probabilities. Returns a scalar, or three values if no-preference.

        Noise-free: these are the probabilities for exactly the curves passed in. Only
        :meth:`respond` applies device noise, through :meth:`perceived`.

        The Weber observer replaces ``s`` by :func:`weber_scale` of the two
        curves before overdispersion multiplies on top; the order bias and no-preference
        threshold stay in units of ``s``.
        """
        scale = self._scale(rng, weber_scale(self.s, curve_a, curve_b)
                            if self.metric == "weber" else None)
        # order_bias is in units of s, so scale it by s -- exactly as no_pref_threshold
        # is scaled below. Note it is `self.s`, not the overdispersed `scale`: design
        # section 5.2 puts `exp(eta)` in the denominator only, so overdispersion widens
        # the psychometric slope on a given trial without also moving the bias.
        du = self.utility(curve_a) - self.utility(curve_b)
        delta = du + self.order_bias * self.s * order
        if not self.allow_no_preference:
            if self.link != "probit":
                # ICORR stress test S5: the same utility difference through another link.
                clean = float(link_cdf(self.link, delta / (scale * self.link_scale)))
            else:
                clean = float(norm.cdf(delta / scale))
            return self.lapse / 2.0 + (1.0 - self.lapse) * clean
        c = self.no_pref_threshold * self.s
        if self.no_pref_hard:
            # ICORR stress test S7, the hard rule: "no preference" exactly when the true utility
            # difference lies inside the band; a lapse is a coin flip between A and B.
            half = self.lapse / 2.0
            if abs(du) < c:
                return (half, 1.0 - self.lapse, half)
            p_a = half + (1.0 - self.lapse) * float(norm.cdf(delta / scale))
            return (p_a, 0.0, 1.0 - p_a)
        p_a = float(norm.cdf((delta - c) / scale))
        p_b = float(norm.cdf((-delta - c) / scale))
        p_none = max(0.0, 1.0 - p_a - p_b)
        third = self.lapse / 3.0
        return (
            third + (1.0 - self.lapse) * p_a,
            third + (1.0 - self.lapse) * p_none,
            third + (1.0 - self.lapse) * p_b,
        )

    def perceived(self, curve_a: np.ndarray, curve_b: np.ndarray) -> tuple:
        """The two curves as the simulated user feels them.

        Without device noise these are the curves themselves. With it, each gets its own
        :func:`smooth_noise` draw from ``noise_rng`` -- A's first, then B's -- added to a
        copy, so the caller's arrays are untouched and nothing outside the observer ever
        sees the noise.
        """
        if self.device_noise_nm <= 0.0:
            return curve_a, curve_b
        curve_a = np.asarray(curve_a, dtype=float)
        curve_b = np.asarray(curve_b, dtype=float)
        noise_a = smooth_noise(self.noise_rng, self.device_noise_nm, curve_a.size)
        noise_b = smooth_noise(self.noise_rng, self.device_noise_nm, curve_b.size)
        return curve_a + noise_a, curve_b + noise_b

    def respond(self, curve_a: np.ndarray, curve_b: np.ndarray,
                rng: np.random.Generator, order: int = 1) -> int:
        """One answer: 0 prefers A, 1 prefers B, 2 no preference."""
        curve_a, curve_b = self.perceived(curve_a, curve_b)
        probs = self.prob_prefers_a(curve_a, curve_b, order=order, rng=rng)
        if not self.allow_no_preference:
            return PREFER_A if rng.random() < probs else PREFER_B
        p_a, p_none, _ = probs
        draw = rng.random()
        if draw < p_a:
            return PREFER_A
        if draw < p_a + p_none:
            return NO_PREFERENCE
        return PREFER_B
