"""The ICORR reframe's robustness wearers: out-of-generator ideals (O) and random smooth utilities (R).

Everything here writes or reads **new** files beside a Phase 5a run; the exported users, the
priors and the population code (``prompref/population``, ``prompref/priors/data_prior.py``)
are read only.

**O. Out-of-generator ideals.** Users whose ideal is a Schaefer four-half-cosine profile, not a
Molinaro-generated curve:

- timings t1..t4 from Schaefer's per-timing distribution (``out/schaefer/prior_inputs.json``:
  ``t_mean``, the recommended diagonal covariance, Student-t with nu = 10 whose scatter is
  scaled to that covariance, ``cov (nu - 2) / nu``), rejecting draws that break the device
  limits exactly as the Schaefer report section 3.6 counts them: every gap between adjacent
  landmarks within 10-40 %ST, the extension peak t2 at most 50 and the flexion peak t4 at
  least 50, with t1 unwrapped to [50, 150);
- one peak magnitude for both bouts, as Schaefer's family has one (12 Nm there): the simulated
  population's median ideal peak ``max |tau*|`` times ``1 + 0.26 z``, z ~ N(0, 1), clipped to
  [0, 12] Nm (:data:`MAGNITUDE_RULE`);
- the profile is ``SchaeferDesign(peak, peak).curve(t)``, the analytic profile of
  ``scripts/schaefer_grounding.py``.

They are written as ``RUN/ood/users/<fold>.{json,npz}`` in the exported users' own schema
(``users-v1``), so :func:`~prompref.population.export.load_fold_users` reads them, and outside
``RUN/users``, where every ``*.json`` with a ``.npz`` beside it is taken for a fold. Their
``families`` hold the promp_s0 floor and reach radii, computed with the fold's own prior by
5a's ``user_floors`` (``scripts/icorr_robust_prep.py``).

**R. Random smooth utilities.** The fold's sidecar ``RUN/random_utility/<fold>.json`` holds the
feature standardisation (mean and SD over the fold's 20,000 training ideals) and, per
population user, the centre c, the matched-decisiveness scale, u_max over the ProMP family
within the device limit and the utility of generic assistance. The user's g is rebuilt from
the answer seed (:mod:`prompref.user.smooth_utility`).
"""

from __future__ import annotations

import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..designs.schaefer import SchaeferDesign
from ..population.curves import TORQUE_LIMIT_NM, physics_clip
from ..population.export import EXPORT_SCHEMA, load_fold_users
from ..population.folds import child_sequence
from ..priors.timing_prior import TimingPrior
from ..user.smooth_utility import (CENTRE_SALT, LENGTHSCALE, N_FEATURES, RFF_SALT, RHO,
                                   RandomSmoothUtility)

# ----------------------------------------------------------------------- O: out of generator

OOD_USER_SET = "ood"
#: The decision of 2026-10-19: the
#: primary O analysis gives each bout its own population-median peak. Same timings, z and
#: answer seeds as "ood"; only the magnitude rule differs. "ood" stays as the labelled
#: common-peak sensitivity run.
OOD_PER_BOUT_USER_SET = "ood-pb"
#: The second review round's user sets (our design notes, R2
#: before any code or run; built by ``scripts/icorr_r2_prep.py``, :mod:`prompref.experiment.r2`):
#: XD's users from other datasets (Fukuchi young and older adults, Reznick level walking and
#: inclines) and HT's heavy-tailed generator (a) and population-family mixture (b). Each is stored
#: like the OOD users, in the exported users' schema with its promp_s0 floors.
R2_USER_SETS = ("xd-fy", "xd-fo", "xd-r", "ht-gen", "ht-mix")
#: Values of ``Phase5Point.user_set``; ``None`` is the run's exported users.
USER_SETS = (OOD_USER_SET, OOD_PER_BOUT_USER_SET) + R2_USER_SETS
OOD_DIR = "ood"
#: Each OOD user set's base directory under the run (an R2 set's is its own name).
OOD_DIRS = {OOD_USER_SET: OOD_DIR, OOD_PER_BOUT_USER_SET: "ood-pb",
            **{name: name for name in R2_USER_SETS}}
#: Population users per fold.
OOD_N_POPULATION = 40
#: Root entropy of the out-of-generator users; the fold's CRC-32 is the second word.
OOD_SEED = 0x00D0CE
#: Planned: Student-t tail nu = 10 (Schaefer report section 3.4's recommendation).
OOD_NU = 10.0
#: Planned: the literature magnitude CV of grounding our design notes
OOD_MAGNITUDE_CV = 0.26
#: Device limits of the Schaefer report section 3.6 (paper, Methods), %ST.
GAP_MIN, GAP_MAX = 10.0, 40.0
#: How the peak magnitude is set: one magnitude for both bouts, the population median of
#: max |tau*| times (1 + CV z). Written into each OOD users file's config.
MAGNITUDE_RULE = "single-peak: median over population ideals of max|tau*|, x (1 + cv z), clip [0, 12]"
#: The ruled primary rule (user set "ood-pb"): extension and flexion peaks each their own
#: population median, times the same (1 + CV z).
MAGNITUDE_RULE_PER_BOUT = ("per-bout: extension peak = median over population ideals of -min tau*, "
                           "flexion peak = median of max tau*, each x (1 + cv z) with one z per "
                           "user, clip [0, 12]")
#: Give up on a user after this many rejected timing draws (the expected share is 14 %).
MAX_TIMING_TRIES = 1000


def unwrap_t1(t: np.ndarray) -> np.ndarray:
    """t1 onto [50, 150) %ST, as the Schaefer analysis keeps it (``sources.unwrap_t1``)."""
    t = np.array(t, dtype=float)
    t[..., 0] = (t[..., 0] - 50.0) % 100.0 + 50.0
    return t


def landmark_gaps(t: np.ndarray) -> np.ndarray:
    """``[t2 + 100 - t1, t3 - t2, t4 - t3, t1 - t4]`` with t1 in [50, 150): the Schaefer
    analysis's ``landmark_intervals``; each row sums to 100."""
    t = np.asarray(t, dtype=float)
    t1, t2, t3, t4 = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
    return np.stack([t2 + 100.0 - t1, t3 - t2, t4 - t3, t1 - t4], axis=-1)


def device_ok(t: np.ndarray) -> np.ndarray:
    """The device limits of the Schaefer report section 3.6, as its ``violation_rates``
    applies them: every gap in [10, 40] %ST, t2 <= 50 and t4 >= 50 (t1 in [50, 150))."""
    t = np.asarray(t, dtype=float)
    gaps = landmark_gaps(t)
    return (np.all((gaps >= GAP_MIN) & (gaps <= GAP_MAX), axis=-1)
            & (t[..., 1] <= 50.0) & (t[..., 3] >= 50.0))


@dataclass(frozen=True)
class TimingDistribution:
    """Schaefer's per-timing distribution: mean, recommended (diagonal) covariance, nu."""

    mean: np.ndarray
    cov: np.ndarray
    nu: float = OOD_NU
    source: str = ""

    @classmethod
    def load(cls, inputs) -> "TimingDistribution":
        prior = TimingPrior.load(inputs)
        return cls(np.asarray(prior.mu, dtype=float), np.asarray(prior.sigma, dtype=float),
                   float(prior.nu), str(inputs))

    def scatter(self) -> np.ndarray:
        """The t scatter whose covariance is ``cov``: ``cov (nu - 2) / nu``."""
        return self.cov * (self.nu - 2.0) / self.nu

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """``n`` unconstrained draws: ``mean + L e / sqrt(chi2_nu / nu)``, ``L L' = scatter``."""
        chol = np.linalg.cholesky(self.scatter())
        z = rng.standard_normal((int(n), 4)) @ chol.T
        g = rng.chisquare(self.nu, size=(int(n), 1)) / self.nu
        return self.mean + z / np.sqrt(g)


def draw_timing(rng: np.random.Generator, dist: TimingDistribution) -> tuple[np.ndarray, int]:
    """One accepted timing vector (t1 in [50, 150)) and how many draws were rejected."""
    for rejected in range(MAX_TIMING_TRIES):
        t = dist.draw(rng, 1)[0]
        if device_ok(t):
            return t, rejected
    raise RuntimeError(f"{MAX_TIMING_TRIES} timing draws in a row broke the device limits")


def draw_peak(rng: np.random.Generator, median_peak_nm: float,
              cv: float = OOD_MAGNITUDE_CV) -> tuple[float, float]:
    """``(peak, z)``: ``median (1 + cv z)`` clipped to [0, TORQUE_LIMIT_NM]."""
    z = float(rng.standard_normal())
    peak = float(np.clip(float(median_peak_nm) * (1.0 + float(cv) * z), 0.0, TORQUE_LIMIT_NM))
    return peak, z


def draw_peaks_per_bout(rng: np.random.Generator, median_ext_nm: float, median_flex_nm: float,
                        cv: float = OOD_MAGNITUDE_CV) -> tuple[float, float, float]:
    """``(extension peak, flexion peak, z)``: each bout's median times the same ``1 + cv z``,
    clipped to [0, TORQUE_LIMIT_NM]. z is the first and only draw, as in :func:`draw_peak`,
    so the same generator gives the same z under both rules."""
    z = float(rng.standard_normal())
    factor = 1.0 + float(cv) * z
    ext = float(np.clip(float(median_ext_nm) * factor, 0.0, TORQUE_LIMIT_NM))
    flex = float(np.clip(float(median_flex_nm) * factor, 0.0, TORQUE_LIMIT_NM))
    return ext, flex, z


def schaefer_profile(t: np.ndarray, peak_nm: float, peak_flex_nm: float | None = None) -> np.ndarray:
    """The four-half-cosine ideal on the 101-point grid, behind the device limit (a no-op for
    peaks within it): extension peak ``peak_nm`` and flexion peak ``peak_flex_nm``, which is
    ``peak_nm`` too unless given."""
    flex = peak_nm if peak_flex_nm is None else peak_flex_nm
    return physics_clip(SchaeferDesign(peak_ext_nm=peak_nm, peak_flex_nm=flex).curve(t))


def population_median_peak(curves) -> float:
    """The median over ideals of ``max |tau*|``, read on the 100 distinct points."""
    curves = np.asarray(curves, dtype=float)
    return float(np.median(np.abs(curves[:, :-1]).max(axis=1)))


def population_median_peaks(curves) -> tuple[float, float]:
    """``(extension, flexion)``: the medians over ideals of ``-min tau*`` and ``max tau*``, read
    on the 100 distinct points."""
    curves = np.asarray(curves, dtype=float)[:, :-1]
    return float(np.median(-curves.min(axis=1))), float(np.median(curves.max(axis=1)))


@dataclass(frozen=True, eq=False)
class OodUser:
    """An out-of-generator user, with the fields ``user_floors`` and the export need."""

    user_id: str
    kind: str
    subject: str
    side: str
    answer_seed: int
    draw: dict
    curve_nm: np.ndarray


def fold_sequence(fold: str) -> np.random.SeedSequence:
    """The fold's root stream: ``[OOD_SEED, crc32(fold)]``. Child 0 holds the users' draws
    (user i: child i, whose child 0 draws the timing and child 1 the magnitude); child 1 the
    answer seeds (user i: child i), as ``make_users`` separates them."""
    return np.random.SeedSequence([OOD_SEED, zlib.crc32(fold.encode("utf-8"))])


def make_ood_users(fold: str, dist: TimingDistribution, median_peak_nm: float | None,
                   n_population: int = OOD_N_POPULATION, *,
                   median_peaks_per_bout: tuple[float, float] | None = None) -> list[OodUser]:
    """The fold's out-of-generator population users, ``<fold>/ood000`` onward, each seeded
    from its own child stream (so user i does not depend on how many draws user i - 1
    rejected).

    The magnitude rule: one common peak ``median_peak_nm`` for both bouts (user set "ood"), or,
    with ``median_peaks_per_bout = (extension, flexion)``, each bout its own median (user set
    "ood-pb"). Timings, z and answer seeds are the same under both rules."""
    if (median_peak_nm is None) == (median_peaks_per_bout is None):
        raise ValueError("give either median_peak_nm (common peak) or median_peaks_per_bout")
    root = fold_sequence(fold)
    draws, answers = child_sequence(root, 0), child_sequence(root, 1)
    users = []
    for i in range(int(n_population)):
        own = child_sequence(draws, i)
        t, rejected = draw_timing(np.random.default_rng(child_sequence(own, 0)), dist)
        magnitude = np.random.default_rng(child_sequence(own, 1))
        seed = int(child_sequence(answers, i).generate_state(1, dtype=np.uint64)[0])
        if median_peaks_per_bout is None:
            peak, z = draw_peak(magnitude, median_peak_nm)
            draw = dict(t_pct_st=[float(v) for v in t], peak_nm=peak, z_magnitude=z,
                        rejected_timing_draws=int(rejected))
            curve = schaefer_profile(t, peak)
        else:
            ext, flex, z = draw_peaks_per_bout(magnitude, *median_peaks_per_bout)
            draw = dict(t_pct_st=[float(v) for v in t], peak_ext_nm=ext, peak_flex_nm=flex,
                        z_magnitude=z, rejected_timing_draws=int(rejected))
            curve = schaefer_profile(t, ext, flex)
        users.append(OodUser(f"{fold}/ood{i:03d}", "population", "schaefer-ood", "-", seed,
                             draw, curve))
    return users


def ood_root(run_dir, user_set: str = OOD_USER_SET) -> Path:
    """The base directory ``load_fold_users`` reads an OOD user set from: ``RUN/ood`` or
    ``RUN/ood-pb``."""
    if user_set not in OOD_DIRS:
        raise ValueError(f"user_set must be one of {tuple(OOD_DIRS)}, got {user_set!r}")
    return Path(run_dir) / OOD_DIRS[user_set]


def ood_floors_cache_path(run_dir, fold: str, user_set: str = OOD_USER_SET) -> Path:
    return ood_root(run_dir, user_set) / "floors" / f"{fold}-promp_s0.json"


def write_ood_users(run_dir, fold: str, users: list[OodUser], floors: list[dict],
                    config: dict, user_set: str = OOD_USER_SET) -> Path:
    """``RUN/<ood dir>/users/<fold>.{json,npz}`` in the ``users-v1`` schema. ``floors`` are
    ``user_floors`` entries in user order; ``families`` gets promp_s0's four values."""
    if len(floors) != len(users):
        raise ValueError(f"{len(floors)} floors for {len(users)} users")
    records = []
    for user, entry in zip(users, floors):
        family = {"floor_nm": float(entry["floor_nm"]), "reach_050": float(entry["reach_050"]),
                  "reach_025": float(entry["reach_025"]), "radius": float(entry["radius_floor"])}
        records.append(dict(user_id=user.user_id, kind=user.kind, subject=user.subject,
                            side=user.side, answer_seed=int(user.answer_seed),
                            draw=json.loads(json.dumps(user.draw)),
                            families={"promp_s0": family}))
    out = ood_root(run_dir, user_set) / "users"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{fold}.npz", curves=np.array([user.curve_nm for user in users]))
    path = out / f"{fold}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(schema=EXPORT_SCHEMA, fold=fold, config=config, users=records),
                              indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_ood_users(run_dir, fold: str, user_set: str = OOD_USER_SET) -> dict:
    """The fold's out-of-generator users by id, as ``load_fold_users`` returns exported ones."""
    root = ood_root(run_dir, user_set)
    path = root / "users" / f"{fold}.json"
    if not path.exists() and user_set in R2_USER_SETS:
        raise FileNotFoundError(f"{path} is missing: run scripts/icorr_r2_prep.py --run {run_dir} "
                                f"--step {user_set} first")
    if not path.exists():
        magnitude = " --magnitude per-bout" if user_set == OOD_PER_BOUT_USER_SET else ""
        raise FileNotFoundError(f"{path} is missing: run scripts/icorr_robust_prep.py --run "
                                f"{run_dir} --step ood{magnitude} first")
    return load_fold_users(root, fold)


# --------------------------------------------------------------------- R: random utilities

RANDOM_DIR = "random_utility"
RANDOM_SCHEMA = "random-utility-v1"
#: One u_max search screens this many weight vectors and starts L-BFGS-B from the best
#: :data:`UMAX_KEEP` of them (and from the prior mean). The screen is half prior draws (at 1x
#: and 2x the prior's scale) and half least-squares preimages of feature vectors drawn around
#: the user's centre at :data:`UMAX_SPREADS` SD, where the envelope keeps the optimum.
UMAX_SCREEN = 120_000
UMAX_KEEP = 320
UMAX_SPREADS = (1.0, 2.0, 3.0)
#: Independent searches per user: at least this many, and more (up to the maximum) until the
#: best value has been reached by two of them.
UMAX_MIN_SEARCHES = 2
UMAX_MAX_SEARCHES = 6


def random_utility_path(run_dir, fold: str) -> Path:
    return Path(run_dir) / RANDOM_DIR / f"{fold}.json"


def training_features_path(run_dir, fold: str) -> Path:
    return Path(run_dir) / RANDOM_DIR / f"{fold}-training-features.npz"


def load_random_utility(run_dir, fold: str) -> dict:
    path = random_utility_path(run_dir, fold)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run scripts/icorr_robust_prep.py --run "
                                f"{run_dir} --step random first")
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if sidecar.get("schema") != RANDOM_SCHEMA:
        raise ValueError(f"{path} has schema {sidecar.get('schema')!r}, expected {RANDOM_SCHEMA!r}")
    constants = sidecar["constants"]
    expected = dict(lengthscale=LENGTHSCALE, rho=RHO, n_features=N_FEATURES, rff_salt=RFF_SALT,
                    centre_salt=CENTRE_SALT)
    if {k: constants.get(k) for k in expected} != expected:
        raise ValueError(f"{path} was built with {constants}, not with this code's {expected}")
    return sidecar


def user_utility(sidecar: dict, user_id: str, answer_seed: int, *, scaled: bool = True
                 ) -> RandomSmoothUtility:
    """The user's utility from the fold's sidecar and the answer seed (scale 1 if not
    ``scaled``)."""
    try:
        entry = sidecar["users"][user_id]
    except KeyError:
        raise KeyError(f"{user_id} is missing from the random-utility sidecar of "
                       f"{sidecar.get('fold')}") from None
    if int(entry["answer_seed"]) != int(answer_seed):
        raise ValueError(f"{user_id}: the sidecar was built for answer seed "
                         f"{entry['answer_seed']}, not {answer_seed}")
    return RandomSmoothUtility.for_user(answer_seed, sidecar["feature_mean"],
                                        sidecar["feature_sd"], entry["centre"],
                                        scale=float(entry["scale"]) if scaled else 1.0)


def same_value(a: float, b: float) -> bool:
    """Two utility maxima agree: within 1e-6, relative to max(1, |a|)."""
    return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)))


def maximise_utility(utility: RandomSmoothUtility, design, *, seed, n_screen: int = UMAX_SCREEN,
                     keep: int = UMAX_KEEP, spreads=UMAX_SPREADS, n_random: int = 0,
                     extra_starts=(), limit: float = TORQUE_LIMIT_NM) -> dict:
    """One search for u_max over a ProMP family within the device limit: the largest
    ``utility`` of ``physics_clip(design.curve(theta))`` over the design's weight box.

    The utility is a random function with many local maxima, so the search screens widely and
    then climbs. All draws come from ``default_rng(seed)``:

    - a screen of ``n_screen`` weight vectors, clipped into the box: a quarter from the prior
      ``N(mu, Sigma)``, a quarter from ``N(mu, 4 Sigma)``, and half split over ``spreads`` as
      least-squares preimages (unclipped feature map) of features ``centre + spread * N(0, I)``;
    - L-BFGS-B with the exact gradient (the clip's derivative is 0 where the curve is clipped)
      from the prior mean and the ``keep`` best screened vectors;
    - for diversity, from ``n_random`` unscreened draws as well (half at 1x, half at 2x the
      prior's scale, drawn after the screen, so ``n_random = 0`` is the search without them),
      and from any ``extra_starts`` (earlier maxima, delivered curves' weights).

    Returns ``u_max``, its ``theta``, and how many of the starts reached it (:func:`same_value`).
    """
    from ..user.smooth_utility import feature_matrix

    phi = np.asarray(design.basis.design, dtype=float)
    k = phi.shape[1]
    bounds = list(zip(design.low, design.high))
    mu, sigma = (np.asarray(m, dtype=float) for m in design.prior)
    chol = np.linalg.cholesky(sigma)
    # Standardised features of the unclipped curve phi theta: A theta + b.
    a = (feature_matrix() @ phi) / utility.feature_sd[:, None]
    b = -utility.feature_mean / utility.feature_sd
    pinv = np.linalg.pinv(a)

    def negative(theta: np.ndarray):
        raw = phi @ theta
        curve = np.clip(raw, -limit, limit)
        value, grad_curve = utility.value_and_gradient(curve)
        inside = (raw > -limit) & (raw < limit)
        return -value, -(phi.T @ (grad_curve * inside))

    rng = np.random.default_rng(seed)
    quarter = int(n_screen) // 4
    parts = [mu + rng.standard_normal((quarter, k)) @ chol.T,
             mu + 2.0 * rng.standard_normal((quarter, k)) @ chol.T]
    per = (int(n_screen) - 2 * quarter) // len(spreads)
    for spread in spreads:
        x = utility.centre + float(spread) * rng.standard_normal((per, a.shape[0]))
        parts.append((x - b) @ pinv.T)
    screen = np.clip(np.vstack(parts), design.low, design.high)
    values = utility.raw(physics_clip(screen @ phi.T, limit)) * utility.scale
    starts = [design.project(mu)] + list(screen[np.argsort(values)[::-1][:int(keep)]])
    if n_random > 0:
        half = int(n_random) // 2
        starts += list(np.clip(np.vstack([
            mu + rng.standard_normal((half, k)) @ chol.T,
            mu + 2.0 * rng.standard_normal((int(n_random) - half, k)) @ chol.T]),
            design.low, design.high))
    starts += [np.clip(np.asarray(s, dtype=float), design.low, design.high) for s in extra_starts]
    reached, best_value, best_theta = [], -np.inf, None
    for start in starts:
        result = minimize(negative, np.asarray(start, dtype=float), jac=True, method="L-BFGS-B",
                          bounds=bounds)
        theta = design.project(result.x)
        value = utility(physics_clip(design.curve(theta), limit))
        reached.append(float(value))
        if np.isfinite(value) and value > best_value:
            best_value, best_theta = float(value), np.asarray(theta, dtype=float)
    if best_theta is None:
        raise RuntimeError("maximise_utility: no start reached a finite utility")
    return dict(u_max=best_value, theta=[float(v) for v in best_theta], n_starts=len(starts),
                n_at_best=int(sum(same_value(best_value, v) for v in reached)))


def combine_searches(results: list[dict]) -> dict:
    """The best of several independent :func:`maximise_utility` searches: ``u_max`` and its
    ``theta``, every search's value, how many searches reached the best, and the gap from the
    best to the best search that did not reach it (0 when all agree)."""
    values = [float(r["u_max"]) for r in results]
    best = int(np.argmax(values))
    agree = [same_value(values[best], v) for v in values]
    others = [v for v, ok in zip(values, agree) if not ok]
    return dict(u_max=values[best], theta=results[best]["theta"], searches=values,
                n_searches=len(values), n_searches_at_best=int(sum(agree)),
                gap_to_next=float(values[best] - max(others)) if others else 0.0)
