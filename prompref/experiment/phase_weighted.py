"""The phase-weighted wearers' per-user sidecars (our design notes,
"Phase-weighted wearers"; W-bio and W-pref, PW-1 and PW-2).

The wearer's own distance is ``kappa_u d_w(tau, tau*)``: the phase-weighted RMS
(:func:`prompref.user.oracle.weighted_rms_distance`) times a per-user scale ``kappa_u`` matched to
the plain s = 1.0 wearer's decisiveness on prior-scale pairs, as in H and R. Everything per user
that the sessions and the report need lives in ``RUN/phase_weight/<cell>/<fold>.json``, written by
``scripts/icorr_wph_prep.py --cell <cell>``:

- ``scale`` (kappa_u) and ``raw_ratio`` (1 / kappa_u): REPORT section 9's decisiveness statistic,
  the median over the user's 200 prior-scale pairs of |Delta d_w| / |Delta d_rms(s = 1.0)|;
- ``own_generic``: the wearer's own distance of generic assistance (the mean curve ``d_prior`` is
  the distance of), exactly as a session records it;
- the floor in the wearer's own metric (:func:`weighted_floor`), unscaled and scaled;
- for W-pref the user's weight curve, which a session rebuilds from the answer seed and checks
  against it; for W-bio the sha256 of the weight sidecar the scales were computed with, which a
  session checks against the weight file it reads.

**The floor is the weighted projection behind the device.** On the linear ProMP family the
minimum of ``d_w`` is the weighted least-squares projection of the ideal,
``argmin_theta sum_phi w (Phi theta - tau*)^2``, computed exactly (``numpy.linalg.lstsq`` on
``sqrt(w) Phi`` and ``sqrt(w) tau*``; bounded-variable least squares when it leaves the weight
box). The delivered curve is ``physics_clip(Phi theta)``, and with ``|tau*| <= 12 Nm`` the clip can
only bring a curve closer to the ideal pointwise, so the device-clipped family can reach below the
projection where the ideal reaches the limit. The floor is therefore the smallest of the clipped
projection and L-BFGS-B descents of the clipped weighted mean square (exact gradient) started from
the projection, the prior mean, prior draws (as 5a's floors start) and any extra starts. With
``w = 1`` the same routine is the RMS floor, which the prep step checks against the stored 5a
floors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear, minimize

from ..population.curves import TORQUE_LIMIT_NM, physics_clip
from ..user.oracle import weighted_rms_distance
from ..user.phase_weight import (BIO_FLOOR_FRACTION, PHASE_WEIGHTINGS, PREF_CELLS,
                                 PREF_HARMONICS, PREF_LOG_SD, PREF_SALT)

PHASE_WEIGHT_DIR = "phase_weight"
PHASE_WEIGHT_SCHEMA = "phase-weight-v1"
PREP_SCRIPT = "scripts/icorr_wph_prep.py"
#: The constants a sidecar must have been built with.
CONSTANTS = dict(pref_harmonics=list(PREF_HARMONICS), pref_log_sd=PREF_LOG_SD,
                 pref_salt=PREF_SALT, bio_floor_fraction=BIO_FLOOR_FRACTION)


def cell_constants(cell: str) -> dict:
    """The constants a cell's sidecar must have been built with: :data:`CONSTANTS` for W-bio and
    W-pref (their stored sidecars carry exactly these), and for the stress test S1 ("pref-sd1")
    the same with its own SD of log w."""
    if cell not in PHASE_WEIGHTINGS:
        raise ValueError(f"phase weighting must be one of {PHASE_WEIGHTINGS}, got {cell!r}")
    if cell in ("bio", "pref"):
        return CONSTANTS
    return dict(CONSTANTS, pref_log_sd=PREF_CELLS[cell])


def phase_weight_path(run_dir, cell: str, fold: str, user_set: str | None = None) -> Path:
    """``RUN/phase_weight/<cell>/<fold>.json`` for the run's exported users; for another user set
    (the second review round's PESS user, on the "ood-pb" users) the same below that set's base
    directory, ``RUN/<set dir>/phase_weight/<cell>/<fold>.json``."""
    if cell not in PHASE_WEIGHTINGS:
        raise ValueError(f"phase weighting must be one of {PHASE_WEIGHTINGS}, got {cell!r}")
    if user_set is None:
        return Path(run_dir) / PHASE_WEIGHT_DIR / cell / f"{fold}.json"
    from .robust import ood_root

    return ood_root(run_dir, user_set) / PHASE_WEIGHT_DIR / cell / f"{fold}.json"


def load_phase_weight_sidecar(run_dir, cell: str, fold: str, user_set: str | None = None) -> dict:
    path = phase_weight_path(run_dir, cell, fold, user_set)
    if not path.exists():
        if user_set is not None:
            raise FileNotFoundError(f"{path} is missing: run scripts/icorr_r2_prep.py --run "
                                    f"{run_dir} --step pess first")
        raise FileNotFoundError(f"{path} is missing: run {PREP_SCRIPT} --run {run_dir} --cell "
                                f"{cell} first")
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if sidecar.get("schema") != PHASE_WEIGHT_SCHEMA:
        raise ValueError(f"{path} has schema {sidecar.get('schema')!r}, expected "
                         f"{PHASE_WEIGHT_SCHEMA!r}")
    if sidecar.get("cell") != cell or sidecar.get("fold") != fold:
        raise ValueError(f"{path} holds cell {sidecar.get('cell')!r} of {sidecar.get('fold')!r}")
    constants = cell_constants(cell)
    if sidecar.get("constants") != constants:
        raise ValueError(f"{path} was built with {sidecar.get('constants')}, not with this code's "
                         f"{constants}")
    return sidecar


def sidecar_entry(sidecar: dict, user_id: str, answer_seed: int) -> dict:
    try:
        entry = sidecar["users"][user_id]
    except KeyError:
        raise KeyError(f"{user_id} is missing from the {sidecar.get('cell')} phase-weight sidecar "
                       f"of {sidecar.get('fold')}") from None
    if int(entry["answer_seed"]) != int(answer_seed):
        raise ValueError(f"{user_id}: the phase-weight sidecar was built for answer seed "
                         f"{entry['answer_seed']}, not {answer_seed}")
    return entry


def _clipped_weighted_mse(phi: np.ndarray, target: np.ndarray, weight: np.ndarray, limit: float):
    n = phi.shape[0]

    def value_and_gradient(theta: np.ndarray):
        raw = phi @ theta
        diff = np.clip(raw, -limit, limit) - target
        inside = (raw > -limit) & (raw < limit)
        return float(np.mean(weight * diff**2)), (2.0 / n) * (phi.T @ (weight * diff * inside))

    return value_and_gradient


def weighted_floor(design, target: np.ndarray, weight: np.ndarray, *,
                   limit: float = TORQUE_LIMIT_NM, n_starts: int = 0, seed=0,
                   extra_starts=()) -> dict:
    """The floor of a ProMP ``design`` (``design.basis.design`` the curve matrix Phi, weights in
    ``[design.low, design.high]``) for ``target`` in the metric ``d_w``, behind the device clip.

    The clipped descent starts from the projection, the prior mean, ``n_starts`` prior draws
    (``design.sample_prior`` with ``default_rng(seed)``, as 5a's floors draw theirs) and any
    ``extra_starts``; the clipped objective has local minima where the ideal reaches the limit.

    Returns ``floor_nm`` (unscaled, Nm) and its ``theta``; the weighted projection ``proj_theta``,
    its distance before (``proj_nm``) and after (``proj_clipped_nm``) the clip, whether it lies in
    the box (``proj_in_box``) and inside the device limit (``proj_clip_free``); and
    ``refine_gain_nm``, how far the clipped descent got below the clipped projection."""
    phi = np.asarray(design.basis.design, dtype=float)
    target = np.asarray(target, dtype=float)
    weight = np.asarray(weight, dtype=float)
    low, high = np.asarray(design.low, dtype=float), np.asarray(design.high, dtype=float)
    root = np.sqrt(weight)
    a, b = phi * root[:, None], target * root
    theta, *_ = np.linalg.lstsq(a, b, rcond=None)
    in_box = bool(np.all((theta >= low) & (theta <= high)))
    if not in_box:
        theta = np.asarray(lsq_linear(a, b, bounds=(low, high), method="bvls").x, dtype=float)
    raw = phi @ theta
    proj_nm = weighted_rms_distance(raw, target, weight)
    proj_clipped_nm = weighted_rms_distance(physics_clip(raw, limit), target, weight)
    objective = _clipped_weighted_mse(phi, target, weight, limit)
    best_value, best_theta = proj_clipped_nm**2, theta
    mu = np.asarray(design.prior[0], dtype=float)
    starts = [theta, np.clip(mu, low, high)]
    if n_starts > 0:
        starts += list(design.sample_prior(np.random.default_rng(seed), int(n_starts)))
    starts += [np.clip(np.asarray(s, dtype=float), low, high) for s in extra_starts]
    for start in starts:
        result = minimize(objective, np.asarray(start, dtype=float), jac=True, method="L-BFGS-B",
                          bounds=list(zip(low, high)))
        cand = np.clip(np.asarray(result.x, dtype=float), low, high)
        value = weighted_rms_distance(physics_clip(phi @ cand, limit), target, weight) ** 2
        if np.isfinite(value) and value < best_value:
            best_value, best_theta = value, cand
    floor_nm = float(np.sqrt(best_value))
    return dict(floor_nm=floor_nm, theta=[float(v) for v in best_theta], n_starts=len(starts),
                proj_theta=[float(v) for v in theta], proj_nm=float(proj_nm),
                proj_clipped_nm=float(proj_clipped_nm), proj_in_box=in_box,
                proj_clip_free=bool(np.all(np.abs(raw) <= limit)),
                refine_gain_nm=float(proj_clipped_nm - floor_nm))
