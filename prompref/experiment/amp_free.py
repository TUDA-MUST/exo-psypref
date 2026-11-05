"""Test AF: removing the amplitude assumption (pilot, 2026-10-27).

A preferred profile is ``a_ext * s_minus + a_flex * s_plus``: the main population's shape
(lag, landmark spread) with each bout normalized to a unit peak, times amplitudes from a user
scenario. The population code is called, never changed: shapes come from its own generator at
a small common scale (no curve reaches any clip) and are then normalized. Because a ProMP
projection is linear, each training shape is stored as the weights of its two unit bouts, and
any amplitude distribution gives its prior's moments from them without regenerating curves.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..basis import VonMisesBasis
from ..designs.base import BoxDesign
from ..designs.device import DeviceDesign
from ..designs.promp import PrompS0
from ..learning.arms import ARMS
from ..learning.psychometric import PsychometricLearner
from ..learning.space import LearnerSpace
from ..population.export import population_from_config
from ..population.folds import child_sequence, load_trials, loso_folds
from ..population.generator import (MAX_CONSECUTIVE_REJECTIONS, OrderViolation, _finish,
                                    assistance_series, clear_cycles, user_curve)
from ..population.curves import stride_curves
from ..population.factors import sample_draw
from ..priors.data_prior import DataPrior
from ..prior import Whitening
from ..user.oracle import Observer, rms_distance
from .run import run_session
from dataclasses import dataclass, field

#: Median mass of the population users (UNITS sec. 0); amplitudes are in Nm at this mass.
MASS_KG = 68.86
#: Band B, the published range of chosen or optimal hip peaks (literature_amplitude.md).
BAND_NMKG = (0.04, 0.5)
BAND_NM = (BAND_NMKG[0] * MASS_KG, BAND_NMKG[1] * MASS_KG)
#: Generic assistance: both peaks at the band's geometric centre.
GENERIC_PEAK_NM = math.sqrt(BAND_NM[0] * BAND_NM[1])
#: Device clip of every candidate: the Utah hip exoskeletons' capacity, above the band.
LIMIT_NM = 45.0
#: PrompS0's weight box; wide enough for every weight the 45 Nm device can show.
BOUND_NM = 60.0
#: The common scale the shapes are generated at, so no curve reaches the 12 Nm clip.
SHAPE_SCALE = 0.05
N_TRAIN = 20_000
USERS_PER_FOLD = 8
SCENARIOS = ("S1", "S2", "S3")
CANDIDATES = ("ref", "a", "c", "b")   # "b" since the change (section 9)
S, LAPSE, K = 1.0, 0.1, 24
C_STAGE1, C_LOGSD = 8, 0.1
SEED = 20261027
OBSERVER_SALT = 0x5E55105   # phase5's, so the answer stream is drawn the same way
STREAMS = ("train_shapes", "train_amp", "user_shapes", "user_amp", "answers", "c_amp")


def fold_streams(fold_index: int) -> dict[str, np.random.SeedSequence]:
    root = np.random.SeedSequence([SEED, int(fold_index)])
    return {name: child_sequence(root, i) for i, name in enumerate(STREAMS)}


def unit_bouts(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The extension part with peak -1 and the flexion part with peak +1."""
    curve = np.asarray(curve, dtype=float)
    neg, pos = np.minimum(curve, 0.0), np.maximum(curve, 0.0)
    if neg.min() >= 0.0 or pos.max() <= 0.0:
        raise OrderViolation("a curve without both bouts")
    return neg / -neg.min(), pos / pos.max()


def compose(a_ext: float, a_flex: float, s_minus: np.ndarray, s_plus: np.ndarray) -> np.ndarray:
    return float(a_ext) * s_minus + float(a_flex) * s_plus


def peaks(curve: np.ndarray) -> tuple[float, float]:
    return float(-np.min(curve)), float(np.max(curve))


def _shape_draw(rng, switches, timing, shift):
    draw = sample_draw(rng, switches, timing, shift)
    return replace(draw, ext_scale=SHAPE_SCALE, flex_scale=SHAPE_SCALE)


def population_config(run_dir: Path, fold: str):
    config = json.loads((run_dir / "folds" / f"{fold}.json").read_text(encoding="utf-8"))["config"]
    return population_from_config(config)


def training_parts(trials, n: int, rng, switches, timing, shift):
    """``n`` per-stride unit-bout pairs, as ``generate_training_curves`` draws its strides."""
    minus, plus = [], []
    for _ in range(n):
        trial = trials[int(rng.integers(len(trials)))]
        for _attempt in range(MAX_CONSECUTIVE_REJECTIONS + 1):
            draw = _shape_draw(rng, switches, timing, shift)
            cycles = clear_cycles(trial, draw.shift_ms)
            if not cycles:
                continue
            stride = int(rng.integers(len(cycles)))
            raw = stride_curves(assistance_series(trial, draw), trial.gait_phase,
                                [cycles[stride]])[0]
            try:
                curve = _finish(raw, draw)
                s_minus, s_plus = unit_bouts(curve)
            except OrderViolation:
                continue
            if np.max(np.abs(curve)) >= 11.9:
                raise RuntimeError("a shape reached the clip; lower SHAPE_SCALE")
            minus.append(s_minus)
            plus.append(s_plus)
            break
        else:
            raise RuntimeError(f"{trial.subject}_{trial.side}: no accepted stride")
    return np.array(minus), np.array(plus)


def user_shapes(trials, n: int, rng, switches, timing, shift):
    """``n`` held-out users' unit-bout pairs (mean over clear cycles, as ``user_curve``)."""
    out = []
    while len(out) < n:
        trial = trials[int(rng.integers(len(trials)))]
        draw = _shape_draw(rng, switches, timing, shift)
        try:
            curve = user_curve(trial, draw)
            out.append((trial.subject, trial.side, *unit_bouts(curve)))
        except (OrderViolation, ValueError):
            continue
    return out


def _loguniform(rng, low, high, size=None):
    return np.exp(rng.uniform(math.log(low), math.log(high), size))


def scenario_amplitudes(scenario: str, rng, n: int) -> np.ndarray:
    """``(n, 2)`` preferred peaks (extension, flexion) in Nm; test AF section 3."""
    if scenario == "S1":
        ext = _loguniform(rng, 0.05 * MASS_KG, 0.14 * MASS_KG, n)
        return np.column_stack([ext, ext * np.exp(math.log(1.1) + math.log(1.3) * rng.standard_normal(n))])
    if scenario == "S2":
        ext = _loguniform(rng, 0.15 * MASS_KG, 0.5 * MASS_KG, n)
        return np.column_stack([ext, ext * np.exp(math.log(0.7) + math.log(1.3) * rng.standard_normal(n))])
    if scenario == "S3":
        return np.column_stack([_loguniform(rng, *BAND_NM, n), _loguniform(rng, *BAND_NM, n)])
    raise ValueError(scenario)


def weight_moments(w_minus: np.ndarray, w_plus: np.ndarray, amps: np.ndarray):
    """Mean and covariance of ``a_ext w_minus + a_flex w_plus`` over the training shapes."""
    w = amps[:, :1] * w_minus + amps[:, 1:] * w_plus
    return w.mean(axis=0), np.cov(w, rowvar=False)


def generic_curve(s_minus: np.ndarray, s_plus: np.ndarray) -> np.ndarray:
    """The fold's mean training shape, bouts renormalized, both peaks at GENERIC_PEAK_NM."""
    m_minus = s_minus.mean(axis=0)
    m_plus = s_plus.mean(axis=0)
    return compose(GENERIC_PEAK_NM, GENERIC_PEAK_NM, m_minus / -m_minus.min(), m_plus / m_plus.max())


def promp_space(mu: np.ndarray, sigma: np.ndarray, limit: float = LIMIT_NM,
                basis: VonMisesBasis | None = None, arm: str = "tight") -> LearnerSpace:
    whitening = Whitening(mu, sigma, mu.size)
    design = DeviceDesign(PrompS0(basis=basis, prior=(mu, sigma), bound_nm=BOUND_NM), limit=limit)
    return LearnerSpace(design, whitening, ARMS[arm].radius(mu.size))


class AmplitudeDesign(BoxDesign):
    """Candidate C's first stage: the generic shape with free log-peaks per bout."""

    param_names = ("log_a_ext", "log_a_flex")

    def __init__(self, s_minus: np.ndarray, s_plus: np.ndarray) -> None:
        super().__init__(np.full(2, math.log(0.5)), np.full(2, math.log(LIMIT_NM)))
        self.s_minus, self.s_plus = s_minus, s_plus

    def curve(self, theta: np.ndarray) -> np.ndarray:
        theta = self.project(theta)
        return compose(math.exp(theta[0]), math.exp(theta[1]), self.s_minus, self.s_plus)


class FactorizedDesign(BoxDesign):
    """Candidate B: a ProMP shape and two free log-peaks.

    The ProMP curve's bouts are renormalized to unit peaks, so its weights carry shape only
    and the last two parameters set the peaks; a bout below ``MIN_BOUT_NM`` is divided by
    ``MIN_BOUT_NM`` instead, so a vanishing bout stays small rather than being blown up."""

    MIN_BOUT_NM = 0.05

    def __init__(self, basis: VonMisesBasis | None = None, shape_bound: float = 5.0,
                 peak_box: tuple[float, float] = (0.5, LIMIT_NM)) -> None:
        self.basis = basis or VonMisesBasis()
        k = self.basis.n_basis
        lo, hi = math.log(peak_box[0]), math.log(peak_box[1])
        super().__init__(np.concatenate([np.full(k, -shape_bound), np.full(2, lo)]),
                         np.concatenate([np.full(k, shape_bound), np.full(2, hi)]))
        self.param_names = tuple(f"w{i}" for i in range(k)) + ("log_a_ext", "log_a_flex")

    def curve(self, theta: np.ndarray) -> np.ndarray:
        theta = self.project(theta)
        shape = self.basis.design @ theta[:-2]
        neg, pos = np.minimum(shape, 0.0), np.maximum(shape, 0.0)
        neg = neg / max(-neg.min(), self.MIN_BOUT_NM)
        pos = pos / max(pos.max(), self.MIN_BOUT_NM)
        return compose(math.exp(theta[-2]), math.exp(theta[-1]), neg, pos)


def factorized_space(w_minus: np.ndarray, w_plus: np.ndarray, band=BAND_NM,
                     peak_box=(0.5, LIMIT_NM), limit: float = LIMIT_NM) -> LearnerSpace:
    """B's prior: the unit-peak training shapes' weights, and log-uniform ``band`` per
    log-peak. Test AR passes its range as ``band``, ``peak_box`` and ``limit``."""
    w = w_minus + w_plus
    lo, hi = math.log(band[0]), math.log(band[1])
    k = w.shape[1]
    mu = np.concatenate([w.mean(axis=0), np.full(2, 0.5 * (lo + hi))])
    sigma = np.zeros((k + 2, k + 2))
    sigma[:k, :k] = np.cov(w, rowvar=False)
    sigma[k:, k:] = np.diag(np.full(2, (hi - lo) ** 2 / 12.0))
    design = DeviceDesign(FactorizedDesign(peak_box=peak_box), limit=limit)
    return LearnerSpace(design, Whitening(mu, sigma, k + 2), ARMS["tight"].radius(k + 2))


class FixedLevelDesign(BoxDesign):
    """Test AR's fixed level: B's shape with both peaks clamped (shape only)."""

    def __init__(self, level_nm: float, basis: VonMisesBasis | None = None,
                 shape_bound: float = 5.0) -> None:
        self.basis = basis or VonMisesBasis()
        k = self.basis.n_basis
        super().__init__(np.full(k, -shape_bound), np.full(k, shape_bound))
        self.param_names = tuple(f"w{i}" for i in range(k))
        self.level = float(level_nm)

    def curve(self, theta: np.ndarray) -> np.ndarray:
        shape = self.basis.design @ self.project(theta)
        neg, pos = np.minimum(shape, 0.0), np.maximum(shape, 0.0)
        neg = neg / max(-neg.min(), FactorizedDesign.MIN_BOUT_NM)
        pos = pos / max(pos.max(), FactorizedDesign.MIN_BOUT_NM)
        return compose(self.level, self.level, neg, pos)


def fixed_level_space(w_minus: np.ndarray, w_plus: np.ndarray, level_nm: float,
                      basis: VonMisesBasis | None = None, arm: str = "tight") -> LearnerSpace:
    w = w_minus + w_plus
    mu, sigma = w.mean(axis=0), np.cov(w, rowvar=False)
    design = DeviceDesign(FixedLevelDesign(level_nm, basis=basis), limit=level_nm + 1e-9)
    return LearnerSpace(design, Whitening(mu, sigma, mu.size), ARMS[arm].radius(mu.size))


def amplitude_space(generic: np.ndarray) -> LearnerSpace:
    s_minus, s_plus = unit_bouts(generic)
    lo, hi = math.log(BAND_NM[0]), math.log(BAND_NM[1])
    mu = np.full(2, 0.5 * (lo + hi))
    sigma = np.diag(np.full(2, (hi - lo) ** 2 / 12.0))
    design = DeviceDesign(AmplitudeDesign(s_minus, s_plus), limit=LIMIT_NM)
    return LearnerSpace(design, Whitening(mu, sigma, 2), ARMS["tight"].radius(2))


def _tie_factories(ties: str):
    """PsyPref-T's factories. Imported here: prompref.learning.ties imports this module."""
    from ..learning.ties import MODES, factories

    if ties not in MODES:
        raise ValueError(f"ties must be None or one of {MODES}, got {ties!r}")
    return factories(ties)


def psypref(space: LearnerSpace, seed: int, arm: str = "tight",
            ties: str | None = None) -> PsychometricLearner:
    """PsyPref (B_fixed's learner at a fixed level). ``ties`` None is PsyPref as before; "band"
    is PsyPref-T (section 26a), "split" the cheap variant. Both equal PsyPref for a user who
    never answers "no preference"."""
    if ties is not None:
        return _tie_factories(ties)[0](space, seed, arm=arm)
    return PsychometricLearner(space.design, seed=seed, duels="eig", space=space,
                               prior=ARMS[arm].bowl_prior())


def _session(design, learner, observer, rng, n, generic):
    return run_session(design, learner, observer, rng, n_comparisons=n, randomise_order=True,
                       d_prior=rms_distance(generic, observer.ideal_nm), d_floor=0.0)


def run_af_session(job: dict, fold_data: dict) -> dict:
    """One PsyPref session of candidate ``job['candidate']`` for one user and scenario."""
    ideal = np.asarray(job["ideal"], dtype=float)
    generic = fold_data["generic"]
    seed = int(job["answer_seed"]) % 2**32
    # Diagnostics D1 and D2 change only the budget and the sensitivity
    k, s = int(job.get("k", K)), float(job.get("s", S))
    job = dict(job, k=k, s=s)
    observer = Observer(ideal_nm=ideal, s=s, lapse=LAPSE)
    rng = np.random.default_rng([int(job["answer_seed"]), OBSERVER_SALT])
    cand = job["candidate"]
    if cand in ("ref", "a"):
        mu, sigma = fold_data["ref" if cand == "ref" else "a"]
        space = promp_space(mu, sigma)
        rec = _session(space.design, psypref(space, seed), observer, rng, k, generic)
        delivered = space.design.curve(np.asarray(rec["best"], dtype=float))
        distances = list(rec["distances"])
    elif cand == "b":
        space = factorized_space(fold_data["w_minus"], fold_data["w_plus"])
        rec = _session(space.design, psypref(space, seed), observer, rng, k, generic)
        delivered = space.design.curve(np.asarray(rec["best"], dtype=float))
        distances = list(rec["distances"])
    elif cand == "c":
        s1 = amplitude_space(generic)
        r1 = _session(s1.design, psypref(s1, seed), observer, rng, C_STAGE1, generic)
        a_hat = np.exp(s1.design.inner.project(np.asarray(r1["best"], dtype=float)))
        crng = np.random.default_rng(child_sequence(fold_data["c_amp"], int(job["index"])))
        n = fold_data["w_minus"].shape[0]
        amps = a_hat[None, :] * np.exp(C_LOGSD * crng.standard_normal((n, 2)))
        mu, sigma = weight_moments(fold_data["w_minus"], fold_data["w_plus"], amps)
        s2 = promp_space(mu, sigma)
        r2 = _session(s2.design, psypref(s2, seed + 1), observer, rng, K - C_STAGE1, generic)
        delivered = s2.design.curve(np.asarray(r2["best"], dtype=float))
        distances = list(r1["distances"]) + list(r2["distances"])
        job = dict(job, c_stage1_peaks=[float(x) for x in a_hat])
    else:
        raise ValueError(cand)
    ident = peaks(delivered)
    return dict({k: v for k, v in job.items() if k != "ideal"},
                preferred_ext=float(job["amps"][0]), preferred_flex=float(job["amps"][1]),
                identified_ext=ident[0], identified_flex=ident[1],
                d0=rms_distance(generic, ideal), distances=distances,
                delivered=[float(x) for x in delivered])


def build_fold(fold_index: int, run_dir: Path, out_dir: Path) -> Path:
    """Training parts, generic assistance and the scenario users of one LOSO fold."""
    fold = loso_folds()[fold_index]
    shift, switches, timing = population_config(run_dir, fold.name)
    streams = fold_streams(fold_index)
    train = load_trials(fold.train)
    held = load_trials(fold.held_out)
    s_minus, s_plus = training_parts(train, N_TRAIN, np.random.default_rng(streams["train_shapes"]),
                                     switches, timing, shift)
    basis = VonMisesBasis()
    w_minus = basis.project(s_minus.T).T
    w_plus = basis.project(s_plus.T).T
    # Candidate A: the training shapes at amplitudes log-uniform over the band, per bout
    amps_a = _loguniform(np.random.default_rng(streams["train_amp"]), *BAND_NM, (N_TRAIN, 2))
    mu_a, sigma_a = weight_moments(w_minus, w_plus, amps_a)
    ref = DataPrior.load(run_dir / "priors" / f"{fold.name}-promp_s0")
    users = user_shapes(held, USERS_PER_FOLD, np.random.default_rng(streams["user_shapes"]),
                        switches, timing, shift)
    amp_rng = np.random.default_rng(streams["user_amp"])
    user_amps = {sc: scenario_amplitudes(sc, amp_rng, USERS_PER_FOLD) for sc in SCENARIOS}
    answers = child_sequence(streams["answers"], 0)
    out = out_dir / f"fold-{fold.name}.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, w_minus=w_minus, w_plus=w_plus, mu_a=mu_a, sigma_a=sigma_a,
             mu_ref=ref.mu, sigma_ref=ref.sigma, generic=generic_curve(s_minus, s_plus),
             user_minus=np.array([u[2] for u in users]), user_plus=np.array([u[3] for u in users]),
             user_subject=np.array([f"{u[0]}_{u[1]}" for u in users]),
             answer_seeds=np.array([child_sequence(answers, i).generate_state(1, dtype=np.uint64)[0]
                                    for i in range(USERS_PER_FOLD)], dtype=np.uint64),
             **{f"amps_{sc}": user_amps[sc] for sc in SCENARIOS})
    return out


def load_fold_data(path: Path, fold_index: int) -> dict:
    z = np.load(path)
    return dict(w_minus=z["w_minus"], w_plus=z["w_plus"], a=(z["mu_a"], z["sigma_a"]),
                ref=(z["mu_ref"], z["sigma_ref"]), generic=z["generic"],
                user_minus=z["user_minus"], user_plus=z["user_plus"],
                user_subject=z["user_subject"], answer_seeds=z["answer_seeds"],
                amps={sc: z[f"amps_{sc}"] for sc in SCENARIOS},
                c_amp=fold_streams(fold_index)["c_amp"])


def jobs_for_fold(fold_index: int, fold_name: str, data: dict) -> list[dict]:
    jobs = []
    for sc in SCENARIOS:
        for i in range(USERS_PER_FOLD):
            amps = data["amps"][sc][i]
            ideal = np.clip(compose(amps[0], amps[1], data["user_minus"][i], data["user_plus"][i]),
                            -LIMIT_NM, LIMIT_NM)
            for cand in CANDIDATES:
                jobs.append(dict(fold=fold_name, fold_index=fold_index, scenario=sc, index=i,
                                 user=f"{fold_name}/u{i}", subject=str(data["user_subject"][i]),
                                 candidate=cand, amps=[float(amps[0]), float(amps[1])],
                                 ideal=ideal.tolist(),
                                 # paired: the same user answers alike in every candidate and
                                 # scenario only through the seed; scenarios get their own
                                 answer_seed=int(data["answer_seeds"][i]) + SCENARIOS.index(sc)))
    return jobs


# --------------------------------------------------------------------------- test AM

class FixedCurveMetric:
    """Test AM's N1: a fixed metric ``M`` in PsyPref's bowl, in PsyPref-M's interface
    (:class:`~prompref.experiment.stress.PhaseMetric`) with no coefficients.

    ``correction`` is ``Delta_M - Delta_I`` = 1/2 [(z_B - z*)^T E (z_B - z*) - (z_A - z*)^T E
    (z_A - z*)] with ``E = M - I``, so the likelihood stays linear in z*."""

    n_coef = 0

    def __init__(self, matrix: np.ndarray) -> None:
        self.m = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        self.d = self.m.shape[0]
        self.e = self.m - np.eye(self.d)

    def correction(self, z_star, coef, z_a, z_b) -> np.ndarray:
        z_a, z_b = np.atleast_2d(z_a), np.atleast_2d(z_b)
        quad = 0.5 * (np.sum(z_b * (z_b @ self.e), axis=1) - np.sum(z_a * (z_a @ self.e), axis=1))
        return quad[None, :] - (z_star @ self.e) @ (z_b - z_a).T


def curve_metric_at_centre(space: LearnerSpace, h: float = 1e-4) -> np.ndarray:
    """``M = J^T J / 101`` at z = 0 (the unclipped curve), scaled to trace d."""
    inner = space.design.inner if isinstance(space.design, DeviceDesign) else space.design
    d = space.d
    cols = []
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        cols.append((inner.curve(space.to_theta(e)) - inner.curve(space.to_theta(-e))) / (2 * h))
    jac = np.column_stack(cols)
    m = jac.T @ jac / jac.shape[0]
    return m * d / np.trace(m)


def metric_psypref(space: LearnerSpace, seed: int, arm: str = "tight", ties: str | None = None):
    """PsyPref with N1: PsyPref-M's learner with the fixed curve metric and no free coefficients.
    ``ties`` as in :func:`psypref`."""
    from .stress import MetricLearner

    if ties is not None:
        return _tie_factories(ties)[1](space, seed, arm=arm)
    return MetricLearner(space.design, seed, space=space, prior=ARMS[arm].bowl_prior(),
                         metric=FixedCurveMetric(curve_metric_at_centre(space)), free_metric=False)


class ProbitPeakDesign(FactorizedDesign):
    """N2: B's design with each log-peak at ``log lo + (log hi - log lo) Phi(u)``, so that
    u ~ N(0, 1) is log-uniform over the range [lo, hi]."""

    def __init__(self, peak_range: tuple[float, float], basis: VonMisesBasis | None = None,
                 shape_bound: float = 5.0, u_bound: float = 6.0) -> None:
        super().__init__(basis, shape_bound, peak_box=peak_range)
        self.low[-2:], self.high[-2:] = -u_bound, u_bound
        self.log_lo, self.log_span = math.log(peak_range[0]), math.log(peak_range[1] / peak_range[0])
        self.param_names = self.param_names[:-2] + ("u_ext", "u_flex")

    def curve(self, theta: np.ndarray) -> np.ndarray:
        from scipy.special import ndtr

        theta = np.array(self.project(theta), dtype=float)
        theta[-2:] = self.log_lo + self.log_span * ndtr(theta[-2:])
        shape = self.basis.design @ theta[:-2]
        neg, pos = np.minimum(shape, 0.0), np.maximum(shape, 0.0)
        neg = neg / max(-neg.min(), self.MIN_BOUT_NM)
        pos = pos / max(pos.max(), self.MIN_BOUT_NM)
        return compose(math.exp(theta[-2]), math.exp(theta[-1]), neg, pos)


def probit_space(w_minus: np.ndarray, w_plus: np.ndarray, peak_range) -> LearnerSpace:
    w = w_minus + w_plus
    k = w.shape[1]
    mu = np.concatenate([w.mean(axis=0), np.zeros(2)])
    sigma = np.zeros((k + 2, k + 2))
    sigma[:k, :k] = np.cov(w, rowvar=False)
    sigma[k:, k:] = np.eye(2)
    design = DeviceDesign(ProbitPeakDesign(peak_range), limit=peak_range[1])
    return LearnerSpace(design, Whitening(mu, sigma, k + 2), ARMS["tight"].radius(k + 2))


# --------------------------------------------------------------------------- test FP

class ProjectedFixedLearner:
    """FP (section 14): AM on the linear ProMP, the user feeling each proposal renormalized to
    the fixed peaks, AM told that curve's ProMP projection (the paper's projection)."""

    def __init__(self, inner, display, basis: VonMisesBasis) -> None:
        self.inner, self.display, self.basis = inner, display, basis

    def propose(self):
        return self.inner.propose()

    def update(self, theta_a, theta_b, response: int) -> None:
        w_a = self.basis.project(self.display.curve(theta_a))
        w_b = self.basis.project(self.display.curve(theta_b))
        self.inner.update(w_a, w_b, response)

    def best(self):
        return self.inner.best()

    def __getattr__(self, name):   # posterior_summary, should_stop, ... are AM's
        return getattr(self.inner, name)


def projected_fixed(w_minus, w_plus, level_nm: float, seed: int):
    """``(display design, learner)`` of AM_fixed at ``level_nm``."""
    w = level_nm * (w_minus + w_plus)
    mu, sigma = w.mean(axis=0), np.cov(w, rowvar=False)
    inner_space = promp_space(mu, sigma, limit=BOUND_NM)
    display = DeviceDesign(FixedLevelDesign(level_nm, shape_bound=BOUND_NM), limit=level_nm + 1e-9)
    return display, ProjectedFixedLearner(metric_psypref(inner_space, seed), display,
                                          inner_space.design.inner.basis)


# --------------------------------------------------------------------------- test NP

def negative_power_nm(curve: np.ndarray, theta_dot: np.ndarray) -> float:
    """``mean_phi max(0, -tau theta_dot) / mean |theta_dot|``: negative power as Nm."""
    return float(np.mean(np.maximum(0.0, -np.asarray(curve) * theta_dot)) / np.mean(np.abs(theta_dot)))


def np_kappas(fold_index: int, n: int) -> np.ndarray:
    """NP's kappa per user: 0 with probability 1/2, else log-uniform on [0.5, 2]."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(fold_index), 300]))
    on = rng.random(n) >= 0.5
    return np.where(on, _loguniform(rng, 0.5, 2.0, n), 0.0)


def np_floor(ideal: np.ndarray, theta_dot: np.ndarray, kappa: float,
             basis: VonMisesBasis | None = None) -> float:
    """The user's best ProMP curve's own distance: RMS(Phi w - tau*) + kappa P-(Phi w), convex
    in w; L-BFGS on a softplus-smoothed max from the least-squares start."""
    from scipy.optimize import minimize

    basis = basis or VonMisesBasis()
    phi = np.asarray(basis.design, dtype=float)
    w0 = basis.project(ideal)
    if kappa == 0.0:
        return rms_distance(phi @ w0, ideal)
    omega = float(np.mean(np.abs(theta_dot)))

    def own(w, beta=None):
        c = phi @ w
        x = -c * theta_dot
        neg = (np.logaddexp(0.0, beta * x) / beta) if beta else np.maximum(0.0, x)
        return float(np.sqrt(np.mean((c - ideal) ** 2)) + kappa * np.mean(neg) / omega)

    w = w0
    for beta in (2.0, 10.0, 50.0):
        w = minimize(lambda v: own(v, beta), w, method="L-BFGS-B").x
    return own(w)


@dataclass
class NegativePowerObserver(Observer):
    """NP's user: the plain ideal-point utility minus ``kappa`` times the curve's negative power
    as Nm (:func:`negative_power_nm`). ``kappa = 0`` is exactly the plain user."""

    theta_dot: np.ndarray = field(default=None)
    kappa: float = 0.0

    def utility(self, curve: np.ndarray) -> float:
        base = super().utility(curve)
        if self.kappa == 0.0:
            return base
        return base - self.kappa * negative_power_nm(curve, self.theta_dot)


# --------------------------------------------------------------------------- test K12

def build_fold_parts(fold_index: int, run_dir: Path, out_dir: Path, n_basis: int) -> Path:
    """The fold's training shapes regenerated from the same stream as :func:`build_fold`,
    projected onto ``n_basis`` von Mises bases (``fold-<name>-K<n>.npz``). At K = 8 the
    projection reproduces the stored parts bit for bit (checked 2026-10-27, fold loso-AB30)."""
    fold = loso_folds()[fold_index]
    shift, switches, timing = population_config(run_dir, fold.name)
    s_minus, s_plus = training_parts(load_trials(fold.train), N_TRAIN,
                                     np.random.default_rng(fold_streams(fold_index)["train_shapes"]),
                                     switches, timing, shift)
    basis = VonMisesBasis(n_basis)
    out = out_dir / f"fold-{fold.name}-K{n_basis}.npz"
    np.savez(out, w_minus=basis.project(s_minus.T).T, w_plus=basis.project(s_plus.T).T)
    return out


class TwoBoutGuard:
    """Test K12's guard: when the delivered curve lacks a bout (a peak below
    ``min_peak``), deliver the two-bout curve nearest in z to the posterior mean among
    ``n`` posterior particles instead. ``fired`` counts the calls that needed it."""

    def __init__(self, learner, space: LearnerSpace, min_peak: float, n: int = 500,
                 seed: int = 0) -> None:
        self.learner, self.space, self.min_peak, self.n = learner, space, float(min_peak), int(n)
        self.rng = np.random.default_rng([int(seed), 0x2B0])
        self.fired = 0
        self.fired_last = False

    def propose(self):
        return self.learner.propose()

    def update(self, theta_a, theta_b, response: int) -> None:
        self.learner.update(theta_a, theta_b, response)

    def _two_bout(self, theta) -> bool:
        return min(peaks(self.space.design.curve(theta))) >= self.min_peak

    def best(self):
        theta = self.learner.best()
        self.fired_last = not self._two_bout(theta)
        if not self.fired_last:
            return theta
        self.fired += 1
        post = self.learner.posterior
        d = self.space.d
        idx = self.rng.choice(len(post.x), size=min(self.n, len(post.x)), p=post.weights())
        zs = post.x[idx, :d]
        z_mean = post.zstar_mean()
        ok = [z for z in zs if self._two_bout(self.space.to_theta(z))]
        if not ok:
            return theta
        z = min(ok, key=lambda v: float(np.sum((v - z_mean) ** 2)))
        return self.space.to_theta(z)

    def __getattr__(self, name):
        return getattr(self.learner, name)


# --------------------------------------------------------------------------- full runs (FULL)

#: The full runs' users are new (seed of their own), so the evaluation users are independent of
#: the 40 pilot users the methods were chosen on.
SEED_FULL = 20261028
FULL_USERS_PER_FOLD = 40
#: Arens' GP: the main condition's settings (e-main: jnd_scale 1.0, 2048 candidates).
G_JND_SCALE, G_CANDIDATES = 1.0, 2048


def comparator(name: str, space: LearnerSpace, seed: int, s: float = S):
    """A comparator on ``space`` with the main condition's settings (phase5.build_learner)."""
    from ..learning.linecospar import LENGTHSCALE, STEP
    from ..learning.whitened import WhitenedLearner
    from ..user.jnd import gate_value

    prior = ARMS["tight"].bowl_prior()
    if name == "gp_bald":
        from ..learning.gp_eubo import GPEUBOLearner  # with botorch installed

        return WhitenedLearner(space, GPEUBOLearner(space, seed=seed, prior=prior,
                                                    acquisition="bald"))
    if name == "g":
        return WhitenedLearner.gp_ucb(space.design, seed=seed, space=space, prior=prior,
                                      jnd=gate_value(s, G_JND_SCALE), n_candidates=G_CANDIDATES)
    if name == "d_z":
        return WhitenedLearner.linecospar(space.design, seed=seed, space=space,
                                          lengthscale=LENGTHSCALE * 2.0, step=STEP * 0.5)
    if name in ("cmaes", "cmaes_sch"):
        from ..learning.sortcma import SortCMALearner  # venv only

        return WhitenedLearner(space, SortCMALearner(space.as_design(), seed=seed, sigma0=1.0,
                                                     popsize=None))
    if name in ("cmaes_sch4", "cmaes4"):
        # a change: the earlier comparison's short-budget tuning (e-base-cma: population 4,
        # sigma0 0.5 x the prior's scale); cmaes4 is the same on the ProMP
        from ..learning.sortcma import SortCMALearner  # venv only

        return WhitenedLearner(space, SortCMALearner(space.as_design(), seed=seed, sigma0=0.5,
                                                     popsize=4))
    # test CMP (section 23): the comparators in their design regime
    if name == "gp_eubo":
        from ..learning.gp_eubo import GPEUBOLearner  # with botorch installed

        return WhitenedLearner(space, GPEUBOLearner(space, seed=seed, prior=prior,
                                                    acquisition="eubo"))
    if name in ("g_generic", "g_generic_nogate"):
        from .review import generic_start_g

        jnd = gate_value(s, G_JND_SCALE) if name == "g_generic" else None
        return generic_start_g(space, seed, prior=prior, jnd=jnd, n_candidates=G_CANDIDATES)
    if name == "ramella":
        from ..learning.batch_mi import BatchMILearner
        from ..learning.space import SpacePriorDesign

        return BatchMILearner(SpacePriorDesign(space, prior), seed=seed)
    raise ValueError(name)


def _schaefer_timing(run_dir: Path, fold: str):
    from ..priors.surface_prior import SurfacePrior, surface_prior_path

    p = SurfacePrior.load(surface_prior_path(run_dir, fold, "schaefer_peaks"))
    return np.asarray(p.mu, float)[:4], np.asarray(p.sigma, float)[:4, :4]


def schaefer_range_space(run_dir: Path, fold: str, lo: float, hi: float) -> LearnerSpace:
    """Schaefer et al.'s family (ii) as established practice: four timings (the fold's surface
    prior's timing block) and two free peaks with the range's moment-matched uniform prior."""
    from ..designs.schaefer import TIMING_HIGH, TIMING_LOW
    from ..designs.schaefer_peaks import SchaeferPeaksDesign

    m4, s4 = _schaefer_timing(run_dir, fold)
    mu = np.concatenate([m4, np.full(2, 0.5 * (lo + hi))])
    sigma = np.zeros((6, 6))
    sigma[:4, :4] = s4
    sigma[4, 4] = sigma[5, 5] = (hi - lo) ** 2 / 12.0
    design = SchaeferPeaksDesign(prior=(mu, sigma), low=np.concatenate([TIMING_LOW, [lo, lo]]),
                                 high=np.concatenate([TIMING_HIGH, [hi, hi]]))
    return LearnerSpace(DeviceDesign(design, limit=hi), Whitening(mu, sigma, 6), ARMS["tight"].radius(6))


def schaefer_fixed_space(run_dir: Path, fold: str, level: float) -> LearnerSpace:
    """Schaefer et al.'s family (i): four timings, both peaks fixed at ``level``."""
    from ..designs.schaefer import SchaeferDesign

    m4, s4 = _schaefer_timing(run_dir, fold)
    design = SchaeferDesign(level, level, prior=(m4, s4))
    return LearnerSpace(DeviceDesign(design, limit=level + 1e-9), Whitening(m4, s4, 4),
                        ARMS["tight"].radius(4))


def build_full_users(fold_index: int, run_dir: Path, out_dir: Path, ranges: dict) -> Path:
    """The full runs' users of one fold: 40 new shapes, answer seeds, and preferred peaks per
    range (log-uniform per bout), from streams of their own (SEED_FULL)."""
    fold = loso_folds()[fold_index]
    shift, switches, timing = population_config(run_dir, fold.name)
    root = np.random.SeedSequence([SEED_FULL, int(fold_index)])
    shape_rng = np.random.default_rng(child_sequence(root, 0))
    users = user_shapes(load_trials(fold.held_out), FULL_USERS_PER_FOLD, shape_rng, switches,
                        timing, shift)
    answers = child_sequence(root, 1)
    amps = {name: _loguniform(np.random.default_rng(child_sequence(root, 2 + r)), lo, hi,
                              (FULL_USERS_PER_FOLD, 2))
            for r, (name, (lo, hi)) in enumerate(ranges.items())}
    out = out_dir / f"full-{fold.name}.npz"
    np.savez(out, user_minus=np.array([u[2] for u in users]), user_plus=np.array([u[3] for u in users]),
             user_subject=np.array([f"{u[0]}_{u[1]}" for u in users]),
             answer_seeds=np.array([child_sequence(answers, i).generate_state(1, dtype=np.uint64)[0]
                                    for i in range(FULL_USERS_PER_FOLD)], dtype=np.uint64),
             **{f"amps_{name}": a for name, a in amps.items()})
    return out


# --------------------------------------------------------------------------- test FAM

#: FAM's families (section 19): in a range, then at a fixed level; and the ones PsyPref is
#: projected onto. ProMP rows are "promp<K>".
FAM_RANGE = ("schaefer_i", "pchip_i", "schaefer_ii", "trapezoid", "franks", "pchip_ii")
FAM_LEVEL = ("schaefer_i", "trapezoid", "franks", "pchip")
PROJ_RANGE = ("schaefer_ii", "trapezoid", "franks", "pchip_ii")
PROJ_LEVEL = FAM_LEVEL
PROMP_K = (4, 8, 12, 16, 24)
#: Schaefer et al.'s fixed peaks, as published, for family (i) in a range.
SCHAEFER_PEAK_NM = 12.0
FLOOR_DRAWS = 8


class FixedPeaksDesign(BoxDesign):
    """A family with some parameters held at given values (FAM: both peaks at the level)."""

    def __init__(self, inner, fixed: dict) -> None:
        self.inner, self.fixed = inner, {int(i): float(v) for i, v in fixed.items()}
        self.free = [i for i in range(inner.n_params) if i not in self.fixed]
        super().__init__(np.asarray(inner.low)[self.free], np.asarray(inner.high)[self.free])
        self.param_names = tuple(inner.param_names[i] for i in self.free)

    def full(self, theta: np.ndarray) -> np.ndarray:
        th = np.empty(self.inner.n_params)
        th[self.free] = self.project(theta)
        for i, v in self.fixed.items():
            th[i] = v
        return th

    def curve(self, theta: np.ndarray) -> np.ndarray:
        return self.inner.curve(self.full(theta))

    def initial_guess(self, target: np.ndarray) -> np.ndarray:
        return self.project(self.inner.initial_guess(target)[self.free])


def _schaefer_i(level: float, prior):
    """Family (i) with family (ii)'s initial guess for its four timings."""
    from ..designs.schaefer import SchaeferDesign
    from ..designs.schaefer_peaks import SchaeferPeaksDesign

    design = SchaeferDesign(level, level, prior=prior)
    design.initial_guess = lambda target: design.project(SchaeferPeaksDesign().initial_guess(target)[:4])
    return design


def _with_peaks(design, idx, lo: float, hi: float):
    low, high = np.array(design.low, dtype=float), np.array(design.high, dtype=float)
    low[list(idx)], high[list(idx)] = lo, hi
    design.low, design.high = low, high
    return design


def family_design(name: str, run_dir: Path, fold: str, peak_range=None, level=None) -> DeviceDesign:
    """FAM's family ``name``, device-clipped at the range's top or the level, with both peaks
    free in ``peak_range`` or both at ``level`` (fixed-peak families keep their published ones)."""
    from ..designs.franks import FIT_HIGH, FIT_LOW, FranksDesign
    from ..designs.pchip_park import PchipDesign
    from ..designs.trapezoid_surface import TrapezoidSurface
    from ..priors.surface_prior import SurfacePrior, surface_prior_path

    if (peak_range is None) == (level is None):
        raise ValueError("give a peak range or a level")
    limit = peak_range[1] if level is None else level + 1e-9
    timing = _schaefer_timing(run_dir, fold)
    if name == "schaefer_i":
        design = _schaefer_i(SCHAEFER_PEAK_NM if level is None else level, timing)
    elif name == "pchip_i":
        if level is not None:
            raise ValueError("pchip_i is the range's published variant; at a level use pchip")
        design = PchipDesign.published()
    elif name == "schaefer_ii":
        design = schaefer_range_space(run_dir, fold, *peak_range).design.inner
    elif name == "trapezoid":
        design = TrapezoidSurface()
        design = (_with_peaks(design, (0, 3), *peak_range) if level is None
                  else FixedPeaksDesign(design, {0: level, 3: level}))
    elif name == "franks":
        if level is None:
            design = _with_peaks(FranksDesign(FIT_LOW, FIT_HIGH), (6, 7),
                                 math.log(peak_range[0]), math.log(peak_range[1]))
        else:
            design = FixedPeaksDesign(FranksDesign(FIT_LOW, FIT_HIGH),
                                      {6: math.log(level), 7: math.log(level)})
    elif name in ("pchip_ii", "pchip"):
        box = SurfacePrior.load(surface_prior_path(run_dir, fold, "pchip"))
        design = PchipDesign(box.low, box.high)
        if name == "pchip_ii":
            if level is not None:
                raise ValueError("pchip_ii has free peaks; at a level use pchip")
            design = _with_peaks(design, (4, 5), *peak_range)
        else:
            if level is None:
                raise ValueError("pchip holds both peaks at a level")
            design = FixedPeaksDesign(design, {4: level, 5: level})
    else:
        raise ValueError(name)
    return DeviceDesign(design, limit=limit)


def fit_family(design: DeviceDesign, target: np.ndarray, starts) -> tuple[np.ndarray, float]:
    """``prompref.priors.franks_prior.fit_curve`` with the design's own device clip in place of
    the 12 Nm one: the best bounded least-squares fit of the clipped curve to ``target`` from
    ``starts``. Returns (theta, RMS in Nm)."""
    from scipy.optimize import least_squares

    target = np.asarray(target, dtype=float)
    best, failed = None, []
    for start in starts:
        try:
            res = least_squares(lambda th: design.curve(th) - target,
                                design.project(np.asarray(start, dtype=float)),
                                bounds=(design.low, design.high), max_nfev=400)
        except ValueError as exc:   # scipy's "`x` is not within the trust region", as fit_curve
            if "trust region" not in str(exc):
                raise
            failed.append(str(exc))
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise ValueError(f"every start of the family fit failed: {failed}")
    return best.x, float(np.sqrt(2.0 * best.cost / target.size))


def _starts(design, target) -> list:
    s = [design.initial_guess(target)] if hasattr(design, "initial_guess") else []
    return s + [0.5 * (np.asarray(design.low) + np.asarray(design.high))]


def family_floor(design: DeviceDesign, target: np.ndarray, seed) -> tuple[np.ndarray, float]:
    """A user's floor in a family: the fit from the initial guess, the box centre and
    ``FLOOR_DRAWS`` uniform box draws."""
    rng = np.random.default_rng(seed)
    draws = rng.uniform(design.low, design.high, size=(FLOOR_DRAWS, len(design.low)))
    return fit_family(design, target, _starts(design, target) + list(draws))


def promp_floor(target: np.ndarray, n_basis: int, peak_range=None, level=None) -> float:
    """A user's floor in the K-basis ProMP: in a range the least-squares projection refined
    under the device clip; at a level the renormalized design refined from it."""
    from scipy.optimize import least_squares

    basis = VonMisesBasis(n_basis)
    phi = np.asarray(basis.design, dtype=float)
    target = np.asarray(target, dtype=float)
    if level is None:
        hi = float(peak_range[1])
        w0 = basis.project(target)
        res = least_squares(lambda w: np.clip(phi @ w, -hi, hi) - target, w0)
    else:
        design = FixedLevelDesign(level, basis=basis)
        res = least_squares(lambda th: design.curve(th) - target,
                            design.project(basis.project(target / level)),
                            bounds=(design.low, design.high))
    return float(np.sqrt(2.0 * res.cost / target.size))


class FamilyLens:
    """FAM's projection (section 19; Phase 5d's ``LensLearner`` with the setting's clip):
    PsyPref proposes in the ProMP, the wearer sees each proposal's best family fit, PsyPref is
    told ``to_inner`` of the curve shown, and the delivered curve is the fit to its optimum."""

    def __init__(self, inner, inner_design, family: DeviceDesign, to_inner) -> None:
        self.inner, self.inner_design, self.family, self.to_inner = inner, inner_design, family, to_inner
        self.last = None
        self.gaps: list[float] = []

    def _fit(self, curve: np.ndarray) -> np.ndarray:
        starts = _starts(self.family, curve) + ([self.last] if self.last is not None else [])
        theta, rms = fit_family(self.family, curve, starts)
        self.last = theta
        self.gaps.append(rms)
        return theta

    def propose(self):
        a, b = self.inner.propose()
        return self._fit(self.inner_design.curve(a)), self._fit(self.inner_design.curve(b))

    def update(self, theta_a, theta_b, response: int) -> None:
        self.inner.update(self.to_inner(self.family.curve(theta_a)),
                          self.to_inner(self.family.curve(theta_b)), response)

    def best(self):
        return self._fit(self.inner_design.curve(self.inner.best()))

    def posterior_summary(self) -> dict:
        summary = self.inner.posterior_summary()
        extras = dict(summary.get("extras", {}),
                      lens_gap_median=float(np.median(self.gaps)) if self.gaps else None,
                      lens_gap_p95=float(np.percentile(self.gaps, 95)) if self.gaps else None)
        return {**summary, "extras": extras}

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def family_lens(inner, space: LearnerSpace, family: DeviceDesign, level=None) -> FamilyLens:
    """The lens around PsyPref ``inner`` on ``space`` (AM's ProMP, or B_fixed's at ``level``)."""
    basis = space.design.inner.basis
    scale = 1.0 if level is None else 1.0 / float(level)
    return FamilyLens(inner, space.design, family,
                      lambda curve: space.design.project(basis.project(np.asarray(curve) * scale)))
