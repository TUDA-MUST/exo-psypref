"""Phase weights of the phase-weighted wearers.

The wearer judges by ``d_w(tau, tau*) = sqrt(mean_phi w(phi) (tau - tau*)^2)`` with
``mean_phi w = 1`` (:func:`prompref.user.oracle.weighted_rms_distance`). Two weightings:

- **W-pref, individual preference.** ``log w_u`` is a random periodic function per user: Fourier
  harmonics 1-3 with independent normal coefficients, scaled so that the SD of ``log w`` over
  the phase grid is exactly 0.5, then ``w`` normalised to mean 1 (:func:`pref_weight`). Seeded per
  user from the answer seed with :data:`PREF_SALT`, so every cell of a user sees the same weight.
- **W-bio, biomechanical impact.** ``w ∝ |theta_dot_hip| + 0.2 mean |theta_dot_hip|`` from the
  mean hip angular velocity of the Molinaro trials the population was generated from
  (:func:`bio_weight`). The same for every user. The curve is not computed here: it is read from
  the sidecar ``out/icorr/wph_bio_weight.json`` (:func:`load_bio_weight`), which
  ``scripts/icorr_wph_bio_weight.py`` writes from the trials. Without that file W-bio fails; there
  is no fallback weighting.

**Which phase points.** Every phase average (the weight's mean, the mean ``|theta_dot|`` of the
floor term) and the SD of ``log w`` are over the 101-point grid 0, 1, ..., 100 %GC, the points
``rms_distance`` averages and the test plan's ``M_w = ... / 101`` sums. Every curve is
periodic: it is evaluated on the 100 distinct points 0-99 %GC and its point 100 is a copy of point
0, so ``w[100] == w[0]`` exactly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from preftune.profile import N_PHASE

from .oracle import check_phase_weight

#: Values of ``Phase5Point.phase_weighting``. "pref-sd1" is the ICORR stress test S1 (section 9a,
#:): W-pref with the SD of log w raised from 0.5 to 1.0, the same
#: coefficients per user, so its log weight is exactly twice W-pref's shape.
PHASE_WEIGHTINGS = ("bio", "pref", "pref-sd1")
#: W-pref: harmonics 1-3, SD of log w over phi 0.5.
PREF_HARMONICS = (1, 2, 3)
PREF_LOG_SD = 0.5
#: The individual-preference cells and their SD of log w.
PREF_CELLS = {"pref": PREF_LOG_SD, "pref-sd1": 1.0}
#: W-pref's own stream per user: ``default_rng([answer_seed, PREF_SALT])``.
PREF_SALT = 0x3A5E7F
#: W-bio: the floor term's fraction of the mean |theta_dot|.
BIO_FLOOR_FRACTION = 0.2
#: The W-bio sidecar, relative to the project's ``out`` directory (``Path(run_dir).parent``).
BIO_WEIGHT_NAME = "wph_bio_weight.json"
BIO_SCHEMA = "wph-bio-weight-v1"
BIO_SCRIPT = "scripts/icorr_wph_bio_weight.py"
#: The distinct points of the periodic grid (0-99 %GC).
N_DISTINCT = N_PHASE - 1


def periodic(values_distinct: np.ndarray) -> np.ndarray:
    """A curve on the 100 distinct points, closed onto the 101-point grid (point 100 = point 0)."""
    values = np.asarray(values_distinct, dtype=float)
    if values.shape != (N_DISTINCT,):
        raise ValueError(f"expected {N_DISTINCT} distinct points, got shape {values.shape}")
    return np.concatenate([values, values[:1]])


def normalise(weight: np.ndarray) -> np.ndarray:
    """``weight`` divided by its mean over all its points."""
    weight = np.asarray(weight, dtype=float)
    return weight / float(np.mean(weight))


def harmonic_basis(harmonics=PREF_HARMONICS) -> np.ndarray:
    """(100, 2 len(harmonics)): cos and sin of each harmonic on the distinct points, interleaved
    (cos h1, sin h1, cos h2, ...)."""
    phi = np.arange(N_DISTINCT) / N_DISTINCT
    cols = []
    for h in harmonics:
        cols += [np.cos(2.0 * np.pi * h * phi), np.sin(2.0 * np.pi * h * phi)]
    return np.stack(cols, axis=1)


def pref_coefficients(answer_seed: int) -> np.ndarray:
    """The user's six standard-normal Fourier coefficients (cos h1, sin h1, ..., sin h3)."""
    rng = np.random.default_rng([int(answer_seed), PREF_SALT])
    return rng.standard_normal(2 * len(PREF_HARMONICS))


def pref_log_weight(answer_seed: int, log_sd: float = PREF_LOG_SD) -> np.ndarray:
    """``log w_u`` on the 101-point grid before the mean-1 shift: the random Fourier series scaled
    to SD ``log_sd`` (W-pref: :data:`PREF_LOG_SD`) over the 101 points (population SD, ddof 0)."""
    raw = periodic(harmonic_basis() @ pref_coefficients(answer_seed))
    return raw * (log_sd / float(np.std(raw)))


def pref_weight(answer_seed: int, log_sd: float = PREF_LOG_SD) -> np.ndarray:
    """W-pref's weight for the user with ``answer_seed``: ``exp(log w_u)`` normalised to mean 1.
    The normalisation shifts ``log w`` by a constant, so its SD stays ``log_sd`` (0.5 for W-pref,
    1.0 for the stress test S1)."""
    weight = normalise(np.exp(pref_log_weight(answer_seed, log_sd)))
    # Normalising both copies of the periodic point the same way keeps w[100] == w[0].
    return check_phase_weight(weight, N_PHASE)


def periodic_derivative(curve: np.ndarray) -> np.ndarray:
    """``d curve / d phi`` per unit phase (one gait cycle) on the 101-point grid, by the central
    difference on the 100 distinct points with wrap-around, closed periodically."""
    curve = np.asarray(curve, dtype=float)
    if curve.shape != (N_PHASE,):
        raise ValueError(f"expected a {N_PHASE}-point curve, got shape {curve.shape}")
    distinct = curve[:-1]
    step = 1.0 / N_DISTINCT
    return periodic((np.roll(distinct, -1) - np.roll(distinct, 1)) / (2.0 * step))


def bio_weight(theta_dot: np.ndarray, floor_fraction: float = BIO_FLOOR_FRACTION) -> np.ndarray:
    """W-bio's weight from the mean hip angular velocity on the 101-point grid:
    ``|theta_dot| + floor_fraction * mean |theta_dot|``, normalised to mean 1. Units and sign of
    ``theta_dot`` cancel."""
    speed = np.abs(np.asarray(theta_dot, dtype=float))
    if speed.shape != (N_PHASE,) or not np.all(np.isfinite(speed)):
        raise ValueError("theta_dot must be a finite 101-point curve")
    if speed[-1] != speed[0]:
        raise ValueError("theta_dot must be periodic (point 100 = point 0)")
    if not float(np.mean(speed)) > 0.0:
        raise ValueError("theta_dot is zero everywhere: the hip does not move")
    return check_phase_weight(normalise(speed + floor_fraction * float(np.mean(speed))), N_PHASE)


def bio_weight_path(run_dir) -> Path:
    """``<out>/icorr/wph_bio_weight.json``, beside the ICORR rows (``out`` is the run's parent)."""
    return Path(run_dir).parent / "icorr" / BIO_WEIGHT_NAME


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bio_weight(run_dir) -> tuple[np.ndarray, str]:
    """W-bio's weight curve and the sha256 of its sidecar. Refuses loudly when the sidecar is
    missing or malformed: W-bio has no fallback weighting."""
    path = bio_weight_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: W-bio's weight comes only from the Molinaro trials. Point "
            f"PREFTUNE_DATA at a readable copy and run {BIO_SCRIPT} --run {run_dir} first; there "
            "is no substitute weighting")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != BIO_SCHEMA:
        raise ValueError(f"{path} has schema {payload.get('schema')!r}, expected {BIO_SCHEMA!r}")
    if float(payload.get("floor_fraction", np.nan)) != BIO_FLOOR_FRACTION:
        raise ValueError(f"{path} was built with floor fraction {payload.get('floor_fraction')}, "
                         f"not the registered {BIO_FLOOR_FRACTION}")
    return check_phase_weight(payload["weight"], N_PHASE), file_sha256(path)
