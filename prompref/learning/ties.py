"""PsyPref that uses "no preference" answers as data.

The code of the exploratory screen's tie_model.py (MD5 e31ba872a8e4833c406e286a16c6518d), unchanged
apart from this docstring. Two modes, both inert until the first "no preference" answer
(response 2). Before it, every call
goes to the unchanged parent class, so a user who never answers "no preference" (FULL's normal
users) gets PsyPref's session bit for bit. Gated code (smc.py, acquisition.py, psychometric.py,
stress.py) is imported, never changed.

mode="band" (PsyPref-T, the main candidate): a two-stage indecision model on PsyPref's probit
argument D = Delta / ell^2. An attentive user first decides whether the pair is distinguishable,
then chooses by the probit:

    q(D)     = Phi((tau - |D|) / R_SOFT)                      probability of "cannot tell"
    P(tie)   = (1 - lam) [EPS_TIE + (1 - EPS_TIE) q(D)]
    P(A)     = lam/2 + (1 - lam)(1 - EPS_TIE)(1 - q(D)) Phi(D)
    P(B)     = lam/2 + (1 - lam)(1 - EPS_TIE)(1 - q(D)) Phi(-D)

tau (the band half-width, in units of the choice noise) joins the particle state as log tau with
prior N(LOG_TAU0, TAU_SD^2). At the first tie the particles get log-tau draws from the prior (own
generator), are importance-reweighted from the binary to the ternary likelihood over every answer
so far, and are resampled and moved (10 moves). From then on the update, the MCMC target and the
pair choice (expected information gain over three outcomes) all use the ternary model.

mode="split" (the cheap variant): a tie counts as half an answer for each side,
log L_tie = (log P(A) + log P(B)) / 2 under PsyPref's own binary likelihood; no new parameter,
pair choice unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_expit, log_ndtr

from prompref.experiment.amp_free import FixedCurveMetric, curve_metric_at_centre
from prompref.experiment.stress import (MetricLearner, metric_log_likelihood,
                                        metric_per_duel_log_likelihood)
from prompref.learning.acquisition import P_FLOOR, _log_det_cov
from prompref.learning.arms import ARMS
from prompref.learning.candidates import (candidate_duels, clip_into_region, coordinate_directions,
                                          step_region, truncate_duel)
from prompref.learning.psychometric import PROBE_LEVELS, PsychometricLearner
from prompref.learning.smc import SMCPosterior, log_likelihood, log_prior, per_duel_log_likelihood

TIE_SALT = 0x71E5
#: Band half-width prior, in units of the choice noise (D units): median 0.5, 95 % in [0.10, 2.4].
LOG_TAU0, TAU_SD = float(np.log(0.5)), 0.8
#: Softness of the band edge (D units) and the floor on P(tie) for an attentive answer.
R_SOFT = 0.3
EPS_TIE = 0.02
#: MCMC moves at activation: two resample-move passes of the posterior's n_moves each.
ACTIVATION_PASSES = 2
MODES = ("band", "split")


# ----------------------------------------------------------------------------- likelihoods

def _delta(x, d, z_a, z_b, metric):
    """PsyPref's Delta = 1/2 (|z_B - z*|^2 - |z_A - z*|^2) (plus the fixed metric's correction),
    shape (N, C). Identical arithmetic to smc.per_duel_log_likelihood / stress's metric version."""
    z_star = x[:, :d]
    delta = (z_star @ (z_a - z_b).T
             - 0.5 * (np.sum(z_a * z_a, axis=1) - np.sum(z_b * z_b, axis=1))[None, :])
    if metric is not None:
        delta = delta + metric.correction(z_star, x[:, d + 2:d + 2 + metric.n_coef], z_a, z_b)
    return delta


def band_log_probs(x, d, z_a, z_b, metric=None):
    """(log P(A), log P(tie), log P(B)), each (N, C), for particles x = (z*, log ell, logit lam,
    [metric coefs], log tau). The three probabilities sum to 1 for every particle and pair."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    z_a, z_b = np.atleast_2d(z_a), np.atleast_2d(z_b)
    dd = _delta(x, d, z_a, z_b, metric) * np.exp(-2.0 * x[:, d])[:, None]
    t = x[:, d + 1][:, None]
    tau = np.exp(x[:, -1])[:, None]
    absd = np.abs(dd)
    log_q = log_ndtr((tau - absd) / R_SOFT)
    log_1mq = log_ndtr((absd - tau) / R_SOFT)
    log_half_lapse = log_expit(t) - np.log(2.0)
    log_att = log_expit(-t)
    log_decide = log_att + np.log1p(-EPS_TIE) + log_1mq
    lpa = np.logaddexp(log_half_lapse, log_decide + log_ndtr(dd))
    lpb = np.logaddexp(log_half_lapse, log_decide + log_ndtr(-dd))
    lpt = log_att + np.logaddexp(np.log(EPS_TIE), np.log1p(-EPS_TIE) + log_q)
    return lpa, lpt, lpb


def _pick(lpa, lpt, lpb, resp):
    resp = np.asarray(resp)[None, :]
    return np.where(resp == 0, lpa, np.where(resp == 1, lpb, lpt)).sum(axis=1)


def band_log_likelihood(x, d, z_a, z_b, resp, metric=None):
    """(N,) ternary log-likelihood of answers ``resp`` (0 A, 1 B, 2 no preference)."""
    return _pick(*band_log_probs(x, d, z_a, z_b, metric), resp)


def _binary_per_duel(x, z_a, z_b, prefers_a, metric):
    if metric is None:
        return per_duel_log_likelihood(x, z_a, z_b, prefers_a)
    return metric_per_duel_log_likelihood(x, z_a, z_b, prefers_a, metric)


def split_log_likelihood(x, d, z_a, z_b, resp, metric=None):
    """(N,) PsyPref's binary log-likelihood, with a tie counted as half an answer each way."""
    z_a, z_b = np.atleast_2d(z_a), np.atleast_2d(z_b)
    ones = np.ones(len(z_a), dtype=bool)
    lla = _binary_per_duel(x, z_a, z_b, ones, metric)
    llb = _binary_per_duel(x, z_a, z_b, ~ones, metric)
    return _pick(lla, 0.5 * (lla + llb), llb, resp)


def binary_log_likelihood(x, z_a, z_b, prefers_a, metric=None):
    """(N,) PsyPref's own binary log-likelihood (the parent posterior's)."""
    if metric is None:
        return log_likelihood(x, z_a, z_b, prefers_a)
    return metric_log_likelihood(x, z_a, z_b, prefers_a, metric)


def band_eig(x, w, d, z_a, z_b, metric=None):
    """acquisition.expected_information_gain's Gaussian proxy on z*, over three outcomes, (C,)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    w = np.asarray(w, dtype=float)
    zs = np.ascontiguousarray(x[:, :d])
    outer = (zs[:, :, None] * zs[:, None, :]).reshape(len(zs), d * d)
    eig = np.full(len(np.atleast_2d(z_a)), 0.5 * _log_det_cov(zs, outer, w[:, None])[0])
    for lp in band_log_probs(x, d, z_a, z_b, metric):
        weighted = w[:, None] * np.exp(lp)
        p = np.clip(weighted.sum(axis=0), P_FLOOR, 1.0)
        eig -= p * 0.5 * _log_det_cov(zs, outer, weighted / p)
    return eig


# ----------------------------------------------------------------------------- posterior

class TiePosterior(SMCPosterior):
    """The parent posterior's particles and weights, taken over at the first tie, with every
    answer so far (ties included) in the target. Built from the old posterior, never from scratch."""

    def __init__(self, old: SMCPosterior, answers, tau_rng: np.random.Generator, metric=None,
                 mode: str = "band") -> None:
        if mode not in MODES:
            raise ValueError(mode)
        # Adopt the old state (no super.__init__: it would draw new particles).
        self.d, self.rng, self.n, self.prior = old.d, old.rng, old.n, old.prior
        self.n_moves, self.ess_fraction = old.n_moves, old.ess_fraction
        self.n_resample, self.acceptance = old.n_resample, list(old.acceptance)
        self.metric, self.mode = metric, mode
        self.free_metric = False   # MetricLearner.posterior_summary reads it
        za = np.array([a[0] for a in answers], dtype=float)
        zb = np.array([a[1] for a in answers], dtype=float)
        resp = np.array([a[2] for a in answers], dtype=int)
        self._za, self._zb, self._resp = list(za), list(zb), [int(r) for r in resp]
        self._z_a, self._z_b = [], []          # the parent's binary lists stay unused
        self._prefers_a = [r == 0 for r in self._resp]   # n_answers counts every answer
        old_x = np.ascontiguousarray(old.x[:, :self.d + 2])
        if old.x.shape[1] != self.d + 2:
            raise ValueError("only learners without free metric coefficients are supported")
        self.log_w = old.log_w.copy()
        binary = resp != 2
        if binary.any():
            self.log_w -= binary_log_likelihood(old_x, za[binary], zb[binary], resp[binary] == 0,
                                                metric)
        if mode == "band":
            log_tau = tau_rng.normal(LOG_TAU0, TAU_SD, size=self.n)
            self.x = np.column_stack([old_x, log_tau])
        else:
            self.x = old_x.copy()
        self.log_w += self._ll(self.x, za, zb, resp)
        if mode == "band":
            for _ in range(ACTIVATION_PASSES):
                self._resample_move()
        elif self.ess < self.ess_fraction * self.n:
            self._resample_move()

    def _ll(self, x, za, zb, resp):
        if self.mode == "band":
            return band_log_likelihood(x, self.d, za, zb, resp, self.metric)
        return split_log_likelihood(x, self.d, za, zb, resp, self.metric)

    def log_target(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        lp = log_prior(np.ascontiguousarray(x[:, :self.d + 2]), self.d, self.prior)
        if self.mode == "band":
            lp = lp - 0.5 * ((x[:, -1] - LOG_TAU0) / TAU_SD) ** 2
        if self._resp:
            lp = lp + self._ll(x, np.array(self._za), np.array(self._zb), np.array(self._resp))
        return lp

    def update_response(self, z_a, z_b, resp: int) -> None:
        z_a, z_b = np.asarray(z_a, dtype=float), np.asarray(z_b, dtype=float)
        self._za.append(z_a)
        self._zb.append(z_b)
        self._resp.append(int(resp))
        self._prefers_a.append(resp == 0)
        self.log_w += self._ll(self.x, z_a[None], z_b[None], [int(resp)])
        if self.ess < self.ess_fraction * self.n:
            self._resample_move()

    def update(self, z_a, z_b, prefers_a: bool) -> None:   # the learner's binary path
        self.update_response(z_a, z_b, 0 if prefers_a else 1)

    def tau_mean(self) -> float | None:
        if self.mode != "band":
            return None
        return float(self.weights() @ np.exp(self.x[:, -1]))


# ----------------------------------------------------------------------------- learners

class TieAware:
    """Mixin placed before PsychometricLearner or MetricLearner. Inert until the first tie."""

    tie_mode = "band"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        seed = kwargs.get("seed", args[1] if len(args) > 1 else 0)
        self._tie_rng = np.random.default_rng([int(seed), TIE_SALT])
        self._answers: list = []
        self.tie_active = False
        self.tie_activated_at: int | None = None
        self._tau_trace: list = []

    def _tie_metric(self):
        return getattr(self, "metric", None)

    def update(self, theta_a, theta_b, response: int) -> None:
        z_a, z_b = self.space.to_z(theta_a), self.space.to_z(theta_b)
        self._answers.append((z_a, z_b, int(response)))
        # The parent records its traces; a binary answer updates the posterior (TiePosterior's
        # update once active); a tie leaves it untouched and counts n_no_preference.
        super().update(theta_a, theta_b, response)
        if response == 2:
            if not self.tie_active:
                self.posterior = TiePosterior(self.posterior, self._answers, self._tie_rng,
                                              self._tie_metric(), self.tie_mode)
                self.tie_active = True
                self.tie_activated_at = len(self._answers)
            else:
                self.posterior.update_response(z_a, z_b, 2)
            self._ell_trace[-1] = self.posterior.ell_mean()
            self._lambda_trace[-1] = self.posterior.lambda_mean()
        if self.tie_active:
            self._tau_trace.append(self.posterior.tau_mean())

    def _propose_eig(self):
        if not (self.tie_active and self.tie_mode == "band"):
            return super()._propose_eig()
        # MetricLearner._propose_eig / PsychometricLearner._propose_eig (default options) line for
        # line, with the three-outcome information gain.
        own, duel, post, rng = self.space, self.duel_space, self.posterior, self._duel_rng
        ell = post.ell_mean()
        z_hat = post.zstar_mean()
        z_worn = self._z_worn()
        region = step_region(z_worn, self.r_step)
        z_a = z_b = None
        if rng.random() < self.p_probe:
            if self.duel_directions == "free":
                u = rng.normal(size=own.d)
                u /= np.linalg.norm(u)
            else:
                dirs = coordinate_directions(own)
                u = dirs[int(rng.integers(len(dirs)))]
            sign = 1.0 if rng.random() < 0.5 else -1.0
            k = float(rng.choice(PROBE_LEVELS))
            c_own = duel.transfer(
                clip_into_region(own.transfer(z_hat - sign * k * ell * u, duel), duel, **region),
                own)
            z_a, z_b = truncate_duel(c_own, u, 0.5 * ell, own, duel, **region)
            if np.linalg.norm(z_a - z_b) < 1e-9:
                z_a = z_b = None
        if z_a is not None:
            kind, multiple = "probe", k
        else:
            cands = candidate_duels(np.ascontiguousarray(post.x[:, :own.d]), post.weights(), own,
                                    duel, rng, ell_hat=ell, max_mult=self.max_mult,
                                    directions=self.duel_directions, **region)
            best = int(np.argmax(band_eig(post.x, post.weights(), own.d, cands["z_a"],
                                          cands["z_b"], self._tie_metric())))
            z_a, z_b = cands["z_a"][best], cands["z_b"][best]
            kind, multiple = "acq", float(cands["multiple"][best])
        self._pending_multiple = multiple
        self._duel_kind.append(kind)
        delta = 0.5 * (float(np.dot(z_b - z_hat, z_b - z_hat))
                       - float(np.dot(z_a - z_hat, z_a - z_hat)))
        self._duel_intensity.append(delta / (ell * ell))
        return own.to_theta(z_a), own.to_theta(z_b)

    def posterior_summary(self) -> dict:
        summary = super().posterior_summary()
        summary["extras"].update(tie_mode=self.tie_mode, tie_active=self.tie_active,
                                 tie_activated_at=self.tie_activated_at,
                                 tau_trace=list(self._tau_trace))
        return summary


class TiePsyPref(TieAware, PsychometricLearner):
    """Fixed level (B_fixed): AF.psypref's learner with the tie model."""


class TieMetricPsyPref(TieAware, MetricLearner):
    """Range (AM): AF.metric_psypref's learner with the tie model."""


class SplitPsyPref(TiePsyPref):
    tie_mode = "split"


class SplitMetricPsyPref(TieMetricPsyPref):
    tie_mode = "split"


def factories(mode: str = "band"):
    """Drop-in replacements for (AF.psypref, AF.metric_psypref), same signatures."""
    fixed_cls, range_cls = ((TiePsyPref, TieMetricPsyPref) if mode == "band"
                            else (SplitPsyPref, SplitMetricPsyPref))

    def psypref(space, seed, arm="tight"):
        return fixed_cls(space.design, seed=seed, duels="eig", space=space,
                         prior=ARMS[arm].bowl_prior())

    def metric_psypref(space, seed, arm="tight"):
        return range_cls(space.design, seed, space=space, prior=ARMS[arm].bowl_prior(),
                         metric=FixedCurveMetric(curve_metric_at_centre(space)), free_metric=False)

    return psypref, metric_psypref
