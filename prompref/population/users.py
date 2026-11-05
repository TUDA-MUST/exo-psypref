"""Held-out users and their best-profile distances.

A user's best-profile distance is the best a family can do for them within the device limit. It is not
restricted to any prior radius: whether a prior can *reach* that best is exactly what
the diagnostics and the robustness study measure.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .curves import physics_clip
from .factors import FactorDraw, FactorSwitches, ShiftSpread, TimingSpread, sample_draw
from .folds import child_sequence
from .generator import OrderViolation, user_curve

_ZERO = (0.0, 0.0, 0.0, 0.0)

def stress_draws(shift: ShiftSpread | None = None) -> dict[str, FactorDraw]:
    """Fixed extreme users; they show where a prior breaks and never enter P95.

    "shift +250ms" and "combined" are +250 ms from the run's shift centre (decision: extension x2 and flexion x0.3 are relative to their centre too, so the shift
    must be), not an absolute 250 ms. The scales stay relative to Molinaro's 0.20.
    """
    centre = float((shift or ShiftSpread.default()).centre_ms)
    return {
        "shift+250ms": FactorDraw(centre + 250.0, 0.20, 0.20, _ZERO),
        "ext_x2": FactorDraw(centre, 0.40, 0.20, _ZERO),
        "flex_x0.3": FactorDraw(centre, 0.20, 0.06, _ZERO),
        "combined": FactorDraw(centre + 250.0, 0.40, 0.06, _ZERO),
    }


#: The stress users at R21's 100 ms centre.
STRESS_DRAWS = stress_draws()


@dataclass(frozen=True, eq=False)
class HeldOutUser:
    user_id: str
    fold: str
    kind: str
    subject: str
    side: str
    draw: FactorDraw
    answer_seed: int
    curve_nm: np.ndarray


def make_users(fold_name: str, held_out_trials, seed: int | None = None, n_population: int = 40,
               switches: FactorSwitches = FactorSwitches(),
               timing: TimingSpread | None = None,
               seed_sequence: np.random.SeedSequence | None = None,
               shift: ShiftSpread | None = None) -> list[HeldOutUser]:
    """Population users, then the stress users.

    With ``seed_sequence``, population draws come from its child 0, and user ``i``'s answer seed is
    ``child_1_i.generate_state(1, uint64)`` -- independent across folds and runs. Without
    it, the int ``seed`` path is unchanged: draws from ``default_rng(seed)`` and answer
    seeds ``seed * 1000 + index``.
    """
    if not held_out_trials:
        raise ValueError("no held-out trials")
    if seed_sequence is None and seed is None:
        raise ValueError("make_users needs a seed or a seed_sequence")
    if seed_sequence is None:
        rng = np.random.default_rng(seed)

        def answer_seed(index: int) -> int:
            return seed * 1000 + index
    else:
        rng = np.random.default_rng(child_sequence(seed_sequence, 0))
        answers = child_sequence(seed_sequence, 1)

        def answer_seed(index: int) -> int:
            return int(child_sequence(answers, index).generate_state(1, dtype=np.uint64)[0])
    timing = timing or TimingSpread.published()
    users: list[HeldOutUser] = []
    rejected = 0
    while len(users) < n_population:
        trial = held_out_trials[int(rng.integers(len(held_out_trials)))]
        draw = sample_draw(rng, switches, timing, shift)
        try:
            curve = user_curve(trial, draw)
        except OrderViolation:
            rejected += 1
            if rejected > 10 * n_population:
                raise RuntimeError(f"{rejected} user draws rejected for fold {fold_name}")
            continue
        index = len(users)
        users.append(HeldOutUser(f"{fold_name}/pop{index:03d}", fold_name, "population",
                                 trial.subject, trial.side, draw, answer_seed(index), curve))
    stress_trial = held_out_trials[0]
    for offset, (name, draw) in enumerate(stress_draws(shift).items()):
        users.append(HeldOutUser(f"{fold_name}/stress-{name}", fold_name, "stress",
                                 stress_trial.subject, stress_trial.side, draw,
                                 answer_seed(n_population + offset),
                                 user_curve(stress_trial, draw)))
    return users


def _clipped_mse(design, target: np.ndarray):
    def objective(theta: np.ndarray) -> float:
        # Mean squared error, not RMS: RMS is a cone at a perfect fit, where its gradient
        # never vanishes and finite differences turn noisy.
        diff = physics_clip(design.curve(design.project(theta))) - target
        return float(np.mean(diff ** 2))
    return objective


def _radius_sq(normalization):
    """``theta -> ||normalization.to_z(theta)||^2`` and its exact gradient (to_z is affine)."""
    m = normalization.eigenvectors[:, :normalization.d] / np.sqrt(normalization.eigenvalues[:normalization.d])
    mu = np.asarray(normalization.mu, dtype=float)

    def value(theta: np.ndarray) -> float:
        z = (np.asarray(theta, dtype=float) - mu) @ m
        return float(z @ z)

    def gradient(theta: np.ndarray) -> np.ndarray:
        return 2.0 * (m @ ((np.asarray(theta, dtype=float) - mu) @ m))

    return value, gradient


def physics_floor(design, target: np.ndarray, n_starts: int = 20, seed=0, normalization=None
                  ) -> tuple[float, np.ndarray]:
    """The user's best-profile distance in a family: the smallest RMS between the device-clipped curve and
    the target, over the design's box. Returns ``(floor_nm, theta)``.

    With ``normalization``, ``theta`` is tie-broken toward the prior: where
    the model curve exceeds the device limit the objective is flat, so the multi-start
    argmin lands wherever the optimiser stops, and its normalized radius is an optimiser
    artefact. SLSQP then takes the smallest-radius point on the best-profile distance's level set,
    ``mse <= best * (1 + 1e-6) + 1e-12``, inside the box. The constraint is hard, not a
    penalty -- a ``1e-8 ||z||^2`` penalty's gradient sits at finite-difference noise.
    The best-profile distance value stays the multi-start minimum. ``seed`` may be an int or a
    ``SeedSequence``.
    """
    target = np.asarray(target, dtype=float)
    rng = np.random.default_rng(seed)
    bounds = list(zip(design.low, design.high))
    objective = _clipped_mse(design, target)

    try:
        starts = list(design.sample_prior(rng, n_starts))
    except ValueError:  # a family without a prior yet
        starts = list(rng.uniform(design.low, design.high, size=(n_starts, design.n_params)))
    if hasattr(design, "initial_guess"):
        starts.insert(0, design.initial_guess(target))
    best_value, best_theta = np.inf, None
    for start in starts:
        result = minimize(objective, np.asarray(start, dtype=float), method="L-BFGS-B", bounds=bounds)
        # Do not gate on result.success: L-BFGS-B often reports success=False with a
        # perfectly usable finite optimum. Only a non-finite value -- a pathological
        # target/design combination, or every start failing -- disqualifies a start.
        if np.isfinite(result.fun) and result.fun < best_value:
            best_value, best_theta = float(result.fun), design.project(result.x)
    if best_theta is None:
        raise RuntimeError(
            f"physics_floor: no start converged to a finite value for "
            f"{type(design).__name__} ({len(starts)} starts)")
    best_theta = np.asarray(best_theta, dtype=float)
    if normalization is not None:
        best_theta = _tie_break(design, objective, normalization, best_value, best_theta, bounds)
    return float(np.sqrt(best_value)), best_theta


def _pull_back(feasible: np.ndarray, candidate: np.ndarray, is_feasible, steps: int = 30
               ) -> np.ndarray:
    """The point nearest ``candidate`` on the segment from ``feasible`` that is feasible.

    SLSQP ends on the constraint's boundary, often a hair outside it: on a flat device-limit
    valley it stops about 1e-9 above the level (relative 2e-11), which a strict feasibility
    check rejects although the radius it found is right. Bisection along the segment toward
    a known-feasible point restores feasibility at a negligible cost in radius.
    """
    if is_feasible(candidate):
        return candidate
    lo, hi = 0.0, 1.0  # fraction of the way from feasible to candidate; lo is feasible
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if is_feasible(feasible + mid * (candidate - feasible)):
            lo = mid
        else:
            hi = mid
    return feasible + lo * (candidate - feasible)


def _tie_break(design, objective, normalization, best_value: float, theta: np.ndarray, bounds
               ) -> np.ndarray:
    level = best_value * (1.0 + 1e-6) + 1e-12
    radius_sq, radius_grad = _radius_sq(normalization)
    result = minimize(radius_sq, theta, jac=radius_grad, method="SLSQP", bounds=bounds,
                      constraints=[{"type": "ineq", "fun": lambda th: level - objective(th)}],
                      options=dict(maxiter=300, ftol=1e-12))
    candidate = np.asarray(design.project(result.x), dtype=float)
    if not np.all(np.isfinite(candidate)):
        return theta
    candidate = _pull_back(theta, candidate, lambda th: objective(th) <= level)
    if objective(candidate) <= level and radius_sq(candidate) < radius_sq(theta):
        return candidate
    return theta


#: The two success tolerances of our design notes whose reach radii the diagnostics report.
REACH_TAUS = (0.5, 0.25)


def reach_key(tau: float) -> str:
    return f"reach_{int(round(tau * 100)):03d}"


def reach_radius(design, normalization, target: np.ndarray, floor_nm: float, tau: float,
                 theta_floor: np.ndarray | None = None, return_z: bool = False):
    """The reach radius R_tau.

    ``min ||z||`` subject to ``RMS(physics_clip(curve(project(to_w(z)))) - target) <=
    floor_nm + tau``: the smallest prior's search region that holds a delivered curve inside this user's
    R25 success region. Zero when the prior mean already is. Otherwise, from the best-profile distance's
    parameters (feasible by definition), a 30-step ray bisection toward z = 0 and SLSQP in
    z started from the bisection point; the smaller feasible radius wins.

    ``theta_floor`` is the best-profile distance's parameters from :func:`physics_floor`; without it the
    best-profile distance is recomputed (20 starts, tie-broken). ``return_z`` also returns the point.
    """
    target = np.asarray(target, dtype=float)
    goal = (float(floor_nm) + float(tau)) ** 2

    def mse_z(z: np.ndarray) -> float:
        theta = design.project(normalization.to_w(z))
        return float(np.mean((physics_clip(design.curve(theta)) - target) ** 2))

    def done(z: np.ndarray):
        radius = float(np.linalg.norm(z))
        return (radius, np.asarray(z, dtype=float)) if return_z else radius

    zero = np.zeros(normalization.d)
    if mse_z(zero) <= goal:
        return done(zero)
    if theta_floor is None:
        _, theta_floor = physics_floor(design, target, normalization=normalization)
    z0 = normalization.to_z(np.asarray(theta_floor, dtype=float))
    if not mse_z(z0) <= goal:
        raise ValueError(f"reach_radius: the floor's parameters miss floor + tau "
                         f"({np.sqrt(mse_z(z0)):.6f} > {np.sqrt(goal):.6f} Nm)")
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if mse_z(mid * z0) <= goal:
            hi = mid
        else:
            lo = mid
    best = hi * z0
    result = minimize(lambda z: float(z @ z), best, jac=lambda z: 2.0 * z, method="SLSQP",
                      constraints=[{"type": "ineq", "fun": lambda z: goal - mse_z(z)}],
                      options=dict(maxiter=400, ftol=1e-10))
    if np.all(np.isfinite(result.x)):
        candidate = _pull_back(best, np.asarray(result.x, dtype=float), lambda z: mse_z(z) <= goal)
        if mse_z(candidate) <= goal and np.linalg.norm(candidate) < np.linalg.norm(best):
            best = candidate
    return done(best)


def floor_and_reach(design, normalization, target: np.ndarray, n_starts: int, seed,
                    taus=REACH_TAUS) -> dict:
    """One user's tie-broken best-profile distance, its normalized radius, and the reach radii, with timings."""
    started = time.perf_counter()
    floor, theta = physics_floor(design, target, n_starts=n_starts, seed=seed, normalization=normalization)
    floor_s = time.perf_counter() - started
    started = time.perf_counter()
    reach = {reach_key(tau): float(reach_radius(design, normalization, target, floor, tau, theta_floor=theta))
             for tau in taus}
    return dict(floor_nm=float(floor), theta=[float(v) for v in theta],
                radius_floor=float(np.linalg.norm(normalization.to_z(theta))), **reach,
                floor_s=float(floor_s), reach_s=float(time.perf_counter() - started))


def _floor_job(job) -> dict:
    design, normalization, target, n_starts, seed = job
    return floor_and_reach(design, normalization, target, n_starts, seed)


#: Result fields cached next to a best-profile distance's value and parameters.
_CACHED_EXTRAS = ("reach_050", "reach_025", "floor_s", "reach_s")


def user_floors(design, normalization, users, *, n_starts: int, seed_sequence: np.random.SeedSequence,
                workers: int = 1, cache: "FloorCache | None" = None, family: str | None = None,
                prior_hash: str | None = None, save_every: int = 10) -> list[dict]:
    """Best-profile distance, best-profile distance radius and reach radii of every user in one family, in user order.

    User ``i``'s best-profile distance starts come from ``child_sequence(seed_sequence, i)``, never from its
    answer seed. Missing users run in a ``ProcessPoolExecutor(workers)``;
    each job carries its own seed, so results do not depend on ``workers``. With a
    ``cache`` (and ``family``/``prior_hash``), hits are reused and the cache is saved every
    ``save_every`` new users.
    """
    if cache is not None and (family is None or prior_hash is None):
        raise ValueError("a cache needs family and prior_hash")
    key = dict(floor_starts=n_starts, prior_hash=prior_hash)
    entries: list[dict | None] = [None] * len(users)
    todo = []
    for index, user in enumerate(users):
        if cache is not None:
            entries[index] = cache.get(family, user.user_id, user.curve_nm, **key)
        if entries[index] is None:
            todo.append(index)
    jobs = [(design, normalization, users[i].curve_nm, n_starts, child_sequence(seed_sequence, i)) for i in todo]

    def store(index: int, result: dict, new: int) -> None:
        user = users[index]
        if cache is None:
            entries[index] = dict(result, theta=np.asarray(result["theta"], dtype=float))
            return
        cache.put(family, user.user_id, user.curve_nm, result["floor_nm"], result["theta"], **key,
                  **{name: result[name] for name in _CACHED_EXTRAS})
        entries[index] = cache.get(family, user.user_id, user.curve_nm, **key)
        if new % save_every == 0:
            cache.save()

    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for new, (index, result) in enumerate(zip(todo, pool.map(_floor_job, jobs)), start=1):
                store(index, result, new)
    else:
        for new, (index, job) in enumerate(zip(todo, jobs), start=1):
            store(index, _floor_job(job), new)
    if cache is not None and jobs:
        cache.save()
    for entry in entries:
        entry["radius_floor"] = float(np.linalg.norm(normalization.to_z(entry["theta"])))
    return entries


#: Bump when the best-profile distance's definition changes, so older cache entries are never reused.
FLOOR_VERSION = "tiebreak-v1"


def prior_digest(mu: np.ndarray, sigma: np.ndarray) -> str:
    """Short hash of a prior's moments: best-profile distance starts and the tie-break both depend on it."""
    h = hashlib.sha1(np.ascontiguousarray(mu, dtype=float).tobytes())
    h.update(np.ascontiguousarray(sigma, dtype=float).tobytes())
    return h.hexdigest()[:12]


class FloorCache:
    """Best-profile distances are expensive (multi-start per user and family); cache them across runs.

    The key holds everything an entry depends on: family, user id, the
    user's curve, the number of best-profile distance starts, the prior's moments, and ``FLOOR_VERSION``.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self._data = {}
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # Losing this file loses every previously cached best-profile distance, not just the latest
            # write -- fail loudly and name the file, rather than silently starting over.
            raise RuntimeError(
                f"FloorCache: {self.path} is corrupt and cannot be parsed as JSON "
                f"({exc}). Delete or repair the file before reusing it.") from exc

    @staticmethod
    def _key(family: str, user_id: str, curve: np.ndarray, floor_starts: int, prior_hash: str) -> str:
        # The curve's hash is part of the key: a rerun with other factors or timing reuses
        # user ids for different users, and must not reuse their best-profile distances.
        digest = hashlib.sha1(np.ascontiguousarray(curve, dtype=float).tobytes()).hexdigest()[:16]
        return f"{FLOOR_VERSION}::{family}::{user_id}::{digest}::starts{int(floor_starts)}::prior{prior_hash}"

    def get(self, family: str, user_id: str, curve: np.ndarray, *, floor_starts: int,
            prior_hash: str) -> dict | None:
        """The cached entry -- ``floor_nm``, ``theta`` (an array) and any extra values
        stored with it, such as ``reach_050`` and ``reach_025`` -- or None."""
        entry = self._data.get(self._key(family, user_id, curve, floor_starts, prior_hash))
        if entry is None:
            return None
        out = dict(entry)
        out["floor_nm"] = float(entry["floor_nm"])
        out["theta"] = np.asarray(entry["theta"], dtype=float)
        return out

    def put(self, family: str, user_id: str, curve: np.ndarray, floor_nm: float,
            theta: np.ndarray, *, floor_starts: int, prior_hash: str, **extra: float) -> None:
        entry = {"floor_nm": float(floor_nm), "theta": [float(v) for v in theta]}
        entry.update({name: float(value) for name, value in extra.items()})
        self._data[self._key(family, user_id, curve, floor_starts, prior_hash)] = entry

    def __len__(self) -> int:
        return len(self._data)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file and rename into place: os.replace is atomic on
        # both POSIX and Windows, so an interruption mid-write never leaves a truncated
        # cache file behind.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
