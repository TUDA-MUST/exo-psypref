"""A simulated user who answers by a drift-diffusion model (DDM): a choice and a response time (our design notes, "RT. Response time, isolated").

**The answer model.** A comparison is a Wiener diffusion with unit noise between two boundaries a
apart, started midway (no response bias). The upper boundary is "prefers A". The drift is
proportional to the utility difference in units of s,

    v = (u_A - u_B) / (s * CHOICE_SCALE * a),

so that the attentive choice probability is P(A) = expit(v a) = expit((u_A - u_B) / (s *
CHOICE_SCALE)). That is the logistic simulated user of stress test S5a at its matched scale: its decisiveness
(the median |p - 1/2| on prior-scale pairs) equals the main probit simulated user's, as planned ("as in
S5"). The response time is the first-passage time plus a non-decision time T0.

- **Lapse.** With probability ``lapse`` the answer is a coin flip, and its response time is drawn
  from the marginal distribution: the response time of a randomly chosen prior-scale pair
  (:func:`marginal_drifts`), whatever the pair on screen.
- **RT-null.** The choice is the DDM's, but every response time is drawn from that same marginal,
  independently of the pair on screen.

The two cells consume the answer stream identically (five draws per comparison), so for the same
duels they give the same choices: a learner that ignores response times runs the same session in
both.

**Assumptions**:

- T0 = :data:`T0_S` = 0.44 s, the location of the non-decision-time prior in Tran, van Maanen,
  Heathcote & Matzke (2021, Front. Psychol. 11:608287, Table 1: a truncated t with location 0.44 s
  and scale 0.08 s, fitted to 857 published estimates).
- The boundary separation a (noise 1) is calibrated so that the median response time over the 200
  population users' prior-scale pairs is :data:`TARGET_MEDIAN_RT_S` = 1.5 s, the planned value
  (``scripts/icorr_rt_prep.py``; :data:`BOUNDARY` holds it at full precision): a = 2.82. The same
  review's boundary-separation estimates (rescaled to noise 1) span 0.11 to 7.47, with the bulk in a
  gamma of mean 1.40, so a = 2.82 is a wide but published value: a median response time of 1.5 s
  with T0 = 0.44 s needs a cautious decision, slower than most perceptual tasks the review covers.
- No across-trial variability of drift, start point or non-decision time: the plain DDM.

**The library.** PyDDM (Shinn, Lam & Murray 2020, eLife 9:e56938), version recorded in every RT row.
Sampling is PyDDM's: an analytic solution of the model on a time grid of :data:`DT_S` up to
:data:`T_DUR_S`, then ``Solution.sample`` with a seed drawn from the answer stream. The density is
PyDDM's analytic first-passage density (``pyddm.analytic.analytic_ddm``, Anderson 1960) at zero
drift; a drift enters through the exact factor of the Wiener density (Navarro & Fuss 2009, J. Math.
Psychol. 53:222, eq. 4),

    f(t | upper, v) = exp(+v a/2 - v^2 t/2) f(t | upper, 0),
    f(t | lower, v) = exp(-v a/2 - v^2 t/2) f(t | lower, 0)        (start midway),

so the response-time density given the choice is the same at both boundaries and depends on |v|
only: h(t | v) = cosh(v a/2) exp(-v^2 t/2) h(t | 0). The tests check this against PyDDM's density
at nonzero drift. PyDDM is imported lazily, so this module imports without it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from .simulated_user import Observer

#: Non-decision time in seconds (assumption; module docstring).
T0_S = 0.44
#: The planned median response time.
TARGET_MEDIAN_RT_S = 1.5
#: PyDDM's time grid for sampling: step and horizon, in seconds. The mass beyond the horizon is
#: below 1e-6 even at zero drift; an undecided draw is redrawn (:func:`sample_decision`).
DT_S = 0.001
T_DUR_S = 30.0
#: PyDDM's space step. The analytic solver reads it only to centre the start point.
DX = 0.01
#: S5a's matched logistic scale (``icorr_grid.LINK_SCALES["logistic"]``, from
#: ``scripts/icorr_stress_prep.py``): the DDM choice is expit(du / (s * CHOICE_SCALE)).
CHOICE_SCALE = 0.5916008501683337
#: The boundary separation a (noise 1), calibrated by ``scripts/icorr_rt_prep.py`` so that the
#: median response time on the prior-scale pairs is TARGET_MEDIAN_RT_S at T0_S.
BOUNDARY = 2.8222737007239345
#: Answer models of the DDM simulated user: response time informative ("inf") or drawn from the marginal
#: independently of the pair ("null").
RT_MODES = ("inf", "null")
#: Where ``scripts/icorr_rt_prep.py`` writes the marginal and the calibration, under the run.
RT_DIR = "rt"
RT_PREP_SCHEMA = "rt-prep-v1"
#: Draws per comparison from the answer stream (lapse, coin, DDM seed, marginal pair, its seed).
DRAWS_PER_ANSWER = 5
_SEED_HIGH = 2**31 - 1


def log_cosh(y):
    """``log cosh(y)``, stable for large |y|."""
    y = np.abs(np.asarray(y, dtype=float))
    return y + np.log1p(np.exp(-2.0 * y)) - np.log(2.0)


def log_tilt(v, tau, a: float):
    """``log[h(tau | v) / h(tau | 0)] = log cosh(v a / 2) - v^2 tau / 2``: the drift's factor on the
    response-time density given the choice, at decision time ``tau`` (module docstring)."""
    v = np.asarray(v, dtype=float)
    return log_cosh(0.5 * v * float(a)) - 0.5 * v * v * np.asarray(tau, dtype=float)


def zero_drift_density(tau, a: float) -> np.ndarray:
    """PyDDM's analytic density of the decision time at zero drift, boundary separation ``a``, start
    midway, summed over both boundaries (each holds half): ``h(tau | 0)``, the density given either
    choice. ``tau`` in seconds, positive."""
    from pyddm.analytic import analytic_ddm

    t = np.atleast_1d(np.array(tau, dtype=float))
    if np.any(~np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("decision times must be positive and finite")
    upper, lower = analytic_ddm(0.0, 1.0, 0.5 * float(a), t.copy(), force_python=True)
    return np.asarray(upper, dtype=float) + np.asarray(lower, dtype=float)


def log_zero_drift_density(tau: float, a: float) -> float:
    """``log h(tau | 0)``, or 0.0 where PyDDM's value underflows (a decision faster than a few ms,
    far below any sample): the term is the same for every particle, so dropping it there changes no
    posterior."""
    value = float(zero_drift_density(tau, a)[0])
    return float(np.log(value)) if value > 0.0 else 0.0


@dataclass(frozen=True, eq=False)
class DDMParams:
    """The DDM answer model's constants: boundary separation ``boundary`` (noise 1), non-decision
    time ``t0`` (s), ``choice_scale`` and the marginal: ``marginal_abs_du``, the |u_A - u_B| (Nm) of
    the prior-scale pairs, answered at sensitivity ``s``."""

    marginal_abs_du: np.ndarray
    boundary: float = BOUNDARY
    t0: float = T0_S
    choice_scale: float = CHOICE_SCALE
    s: float = 1.0
    source: str = ""
    marginal_sha256: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        du = np.asarray(self.marginal_abs_du, dtype=float).ravel()
        if du.size < 1 or not np.all(np.isfinite(du)) or np.any(du < 0.0):
            raise ValueError("the marginal needs at least one finite, non-negative |du|")
        for name in ("boundary", "choice_scale", "s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if not np.isfinite(self.t0) or self.t0 < 0.0:
            raise ValueError(f"t0 must be non-negative and finite, got {self.t0}")
        object.__setattr__(self, "marginal_abs_du", du)
        if not self.marginal_sha256:
            object.__setattr__(self, "marginal_sha256", array_sha256(du))

    def drift(self, du):
        """The drift of a utility difference (Nm) at this model's s."""
        return np.asarray(du, dtype=float) / (self.s * self.choice_scale * self.boundary)

    def drift_of_probit(self, x):
        """The drift of a probit argument ``x = Delta / ell^2`` (A's units of s)."""
        return np.asarray(x, dtype=float) / (self.choice_scale * self.boundary)

    def marginal_drifts(self) -> np.ndarray:
        return self.drift(self.marginal_abs_du)

    def log_marginal_factor(self, tau: float) -> float:
        """``log[g(tau) / h(tau | 0)]``, g the marginal density of the decision time: the mean over
        the marginal's pairs of the drift's factor."""
        lt = log_tilt(self.marginal_drifts(), float(tau), self.boundary)
        return float(logsumexp(lt) - np.log(lt.size))

    def choice_probability(self, du):
        """The attentive P(A) of a utility difference: expit(v a)."""
        return expit(np.asarray(du, dtype=float) / (self.s * self.choice_scale))


def array_sha256(values) -> str:
    """sha256 of an array's float64 bytes (little-endian, C order)."""
    a = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(a.tobytes()).hexdigest()


# --------------------------------------------------------------------------------- PyDDM

def pyddm_version() -> str:
    import pyddm

    return str(pyddm.__version__)


def solve(v: float, a: float, *, dt: float = DT_S, t_dur: float = T_DUR_S):
    """PyDDM's analytic solution of the DDM with drift ``v``, noise 1, boundaries at +-a/2 and a
    start midway, no non-decision time: a ``pyddm.Solution`` over the decision time."""
    import pyddm
    from pyddm import Model
    from pyddm.models import (BoundConstant, DriftConstant, ICPointSourceCenter, NoiseConstant,
                              OverlayNone)

    pyddm.set_log_level("ERROR")
    model = Model(drift=DriftConstant(drift=float(v)), noise=NoiseConstant(noise=1.0),
                  bound=BoundConstant(B=0.5 * float(a)), IC=ICPointSourceCenter(),
                  overlay=OverlayNone(), dx=DX, dt=float(dt), T_dur=float(t_dur))
    return model.solve()


#: Redraws allowed for undecided samples (mass beyond T_DUR_S) before giving up.
MAX_REDRAWS = 100


def sample_decision(v: float, a: float, seed: int) -> tuple[bool, float, int]:
    """One DDM decision by PyDDM's ``Solution.sample``: ``(upper, decision time in s, redraws)``.
    An undecided draw (beyond the horizon) is redrawn with seed + 1, + 2, ...; ``redraws`` counts
    them."""
    sol = solve(v, a)
    for redraw in range(MAX_REDRAWS):
        sample = sol.sample(1, seed=int(seed) + redraw)
        if len(sample.choice_upper):
            return True, float(sample.choice_upper[0]), redraw
        if len(sample.choice_lower):
            return False, float(sample.choice_lower[0]), redraw
    raise RuntimeError(f"{MAX_REDRAWS} undecided DDM draws in a row (v={v}, a={a})")


def decision_time_grid(a: float, *, dt: float = DT_S, t_dur: float = T_DUR_S):
    """``(t, h0)``: PyDDM's grid and its zero-drift decision-time density, both boundaries."""
    sol = solve(0.0, a, dt=dt, t_dur=t_dur)
    return np.asarray(sol.t_domain, dtype=float), np.asarray(sol.pdf("upper") + sol.pdf("lower"),
                                                             dtype=float)


def marginal_decision_cdf(params: DDMParams, *, dt: float = DT_S, t_dur: float = T_DUR_S,
                          chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """``(t, F)``: the cumulative distribution of the decision time under the marginal, on PyDDM's
    grid: the zero-drift density times the mean drift factor, accumulated left point by left point
    as ``Solution.sample`` places its mass."""
    t, h0 = decision_time_grid(params.boundary, dt=dt, t_dur=t_dur)
    drifts = params.marginal_drifts()
    factor = np.zeros_like(t)
    for start in range(0, drifts.size, chunk):
        v = drifts[start:start + chunk, None]
        factor += np.exp(log_tilt(v, t[None, :], params.boundary)).sum(axis=0)
    density = h0 * factor / drifts.size
    return t, np.cumsum(density * dt)


def marginal_median_rt(params: DDMParams, **kwargs) -> float:
    """The marginal median response time, t0 plus the decision time's median (the grid's mass sits
    uniformly within each step, as ``Solution.sample`` spreads it)."""
    t, cdf = marginal_decision_cdf(params, **kwargs)
    dt = float(t[1] - t[0])
    total = float(cdf[-1])
    i = int(np.searchsorted(cdf, 0.5 * total))
    below = float(cdf[i - 1]) if i > 0 else 0.0
    mass = float(cdf[i]) - below
    frac = (0.5 * total - below) / mass if mass > 0.0 else 0.0
    return params.t0 + float(t[i]) + frac * dt


def calibrate_boundary(abs_du, *, t0: float = T0_S, target: float = TARGET_MEDIAN_RT_S,
                       s: float = 1.0, choice_scale: float = CHOICE_SCALE, low: float = 0.3,
                       high: float = 8.0, xtol: float = 1e-12, **kwargs) -> float:
    """The boundary separation at which the marginal median response time is ``target``."""
    from scipy.optimize import brentq

    def gap(a: float) -> float:
        params = DDMParams(np.asarray(abs_du, dtype=float), boundary=a, t0=t0,
                           choice_scale=choice_scale, s=s)
        return marginal_median_rt(params, **kwargs) - target

    return float(brentq(gap, low, high, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=200))


# --------------------------------------------------------------------------------- sidecar

def rt_prep_path(run_dir) -> Path:
    return Path(run_dir) / RT_DIR / "rt_prep.json"


@lru_cache(maxsize=4)
def load_rt_params(run_dir: str) -> DDMParams:
    """The DDM model of a run: the marginal from ``RUN/rt/rt_prep.json``, refused unless its
    boundary, non-decision time and choice scale are this module's constants bit for bit."""
    path = rt_prep_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run scripts/icorr_rt_prep.py --run {run_dir} "
                                "first")
    side = json.loads(path.read_text(encoding="utf-8"))
    if side.get("schema") != RT_PREP_SCHEMA:
        raise ValueError(f"{path} has schema {side.get('schema')!r}, expected {RT_PREP_SCHEMA!r}")
    want = dict(boundary=BOUNDARY, t0=T0_S, choice_scale=CHOICE_SCALE, s=1.0)
    got = {k: side["constants"].get(k) for k in want}
    if got != want:
        raise ValueError(f"{path} was written for {got}, not this code's {want}")
    du = np.asarray(side["marginal_abs_du"], dtype=float)
    if array_sha256(du) != side["marginal_sha256"]:
        raise ValueError(f"{path}: the marginal's sha256 differs from the one it records")
    return DDMParams(du, source=str(path), marginal_sha256=side["marginal_sha256"])


# --------------------------------------------------------------------------------- the simulated user

@dataclass
class DDMObserver(Observer):
    """The RMS simulated user of the main condition answering by a DDM (module docstring). ``respond`` also
    records the response time in s: ``last_response_time`` and the list ``response_times``, with
    ``lapsed`` (0/1 per answer) and ``n_redraws`` (undecided PyDDM draws redrawn)."""

    rt_mode: str = "inf"
    ddm: DDMParams | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rt_mode not in RT_MODES:
            raise ValueError(f"rt_mode must be one of {RT_MODES}, got {self.rt_mode!r}")
        if self.ddm is None:
            raise ValueError("a DDM wearer needs its DDMParams")
        if (self.metric != "rms" or self.link != "probit" or self.allow_no_preference
                or self.overdispersion > 0.0 or self.device_noise_nm > 0.0
                or self.order_bias != 0.0 or self.asymmetry != 1.0 or self.plateau_nm > 0.0
                or self.familiarity_weight):
            raise ValueError("the DDM wearer is the main condition's RMS wearer with only the answer "
                             "model changed")
        if float(self.s) != float(self.ddm.s):
            raise ValueError(f"the wearer's s {self.s} differs from the DDM model's {self.ddm.s}")
        self.response_times: list[float] = []
        self.lapsed: list[int] = []
        self.n_redraws = 0
        self.last_response_time: float | None = None

    def prob_prefers_a(self, curve_a, curve_b, order: int = 1, rng=None):
        """P(A): lapse / 2 + (1 - lapse) expit(v a), the DDM's exact choice probability."""
        du = self.utility(curve_a) - self.utility(curve_b)
        return self.lapse / 2.0 + (1.0 - self.lapse) * float(self.ddm.choice_probability(du))

    def _marginal_decision_time(self, j: int, seed: int) -> float:
        v = float(self.ddm.marginal_drifts()[j])
        _, dt, redraws = sample_decision(v, self.ddm.boundary, seed)
        self.n_redraws += redraws
        return dt

    def respond(self, curve_a, curve_b, rng: np.random.Generator, order: int = 1) -> int:
        """0 prefers A, 1 prefers B. Draws :data:`DRAWS_PER_ANSWER` numbers from ``rng`` whatever
        happens, so both cells and every lapse pattern consume the stream alike."""
        du = self.utility(curve_a) - self.utility(curve_b)
        v = float(self.ddm.drift(du))
        lapse_draw = rng.random()
        coin = rng.random()
        seed_ddm = int(rng.integers(0, _SEED_HIGH))
        j = int(rng.integers(0, self.ddm.marginal_abs_du.size))
        seed_marginal = int(rng.integers(0, _SEED_HIGH))
        lapsed = lapse_draw < self.lapse
        if lapsed:
            prefers_a = coin < 0.5
            decision = self._marginal_decision_time(j, seed_marginal)
        else:
            prefers_a, decision, redraws = sample_decision(v, self.ddm.boundary, seed_ddm)
            self.n_redraws += redraws
            if self.rt_mode == "null":
                decision = self._marginal_decision_time(j, seed_marginal)
        rt = float(self.ddm.t0 + decision)
        self.last_response_time = rt
        self.response_times.append(rt)
        self.lapsed.append(int(lapsed))
        return 0 if prefers_a else 1
