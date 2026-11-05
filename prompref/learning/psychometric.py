"""Learner A -- psychometric duel design.

Phase 3 built the posterior with random duels; Phase 4 adds duels chosen by expected
information gain about z*, an intensity probe ladder, and a stopping rule.
Random duels stay available unchanged: they are the baseline the acquisition must beat.

A reasons in ``z`` (:class:`LearnerSpace`) and speaks design parameters to the harness.

Duels and the trust region live in the eps = 0 space, the posterior in the learner's own. Drawing duels in each learner's own space and clipping them to a common
ball would push 74 % of eps = 1's duel centres onto the boundary; drawing them in one
space means every eps is shown the same curves and differs in its prior alone.
"""

from __future__ import annotations

import numpy as np

from ..designs.base import Design
from .acq_variants import (ACQUISITIONS, DIRECTION_SETS, WIDTHS, curve_metric,
                           curve_subspace_axes, expected_curve_variance_reduction,
                           target_width_duel)
from .acquisition import TRIM_QUANTILE, expected_information_gain
from .candidates import (DIRECTIONS, candidate_duels, clip_into_region, coordinate_directions,
                         step_region, truncate_duel)
from .smc import BowlPrior, SMCPosterior, log_likelihood
from .space import LearnerSpace

_STREAM_SALT = 0xA11CE

#: Duel separations as multiples of the prior's ell0. makes ell a length in z,
#: so a separation can be quoted in the same unit.
SEPARATIONS = (0.5, 1.0, 2.0)

#: Where duels are drawn and the trust region is judged. "reference" is the eps = 0 space
#:; "own" is the learner's own-prior ball, kept as one sensitivity row.
DUEL_SPACES = ("reference", "own")

DUEL_MODES = ("random", "eig")

#: Probe intensities: the probe's centre sits k * ell-hat from z-hat along its direction, at
#: width ell-hat, so Delta / ell^2 ~ k. A centred probe identifies nothing.
PROBE_LEVELS = (0.5, 1.0, 2.0)

_PHASE4_DEFAULTS = dict(p_probe=0.15, max_mult=4.0, r_step=None, duel_directions="free",
                        stop_sd=None)
#: The EIG-mode options at their defaults: Phase 4's, kept equal to the sweep's own copy
#: (tests/test_run_and_sweep.py), and the ICORR acquisition variants' (section 9a, part A).
_EIG_DEFAULTS = dict(_PHASE4_DEFAULTS, acquisition="eig", widths="ladder",
                     direction_set="posterior")


class PsychometricLearner:
    """The bowl model's SMC posterior, fed either by random duels inside the trust ball
    (``duels="random"``) or by duels chosen to maximise expected information gain about
    z*, interleaved with an intensity probe ladder.

    ``acquisition``, ``widths`` and ``direction_set`` are the ICORR section 9a acquisition
    variants (A-curve, A-width, A-sub; :mod:`prompref.learning.acq_variants`), EIG mode only and
    off by default. Each changes how acquisition duels are scored or built; the probes, the
    posterior and the delivered curve are untouched."""

    def __init__(self, design: Design, seed: int = 0, *, eps: float = 0.0,
                 d: int | None = None, duels: str = "random", duel_space: str = "reference",
                 p_probe: float = 0.15, max_mult: float = 4.0, r_step: float | None = None,
                 duel_directions: str = "free", stop_sd: float | None = None,
                 n_particles: int = 4000, prior: BowlPrior = BowlPrior(),
                 space: LearnerSpace | None = None, acquisition: str = "eig",
                 widths: str = "ladder", direction_set: str = "posterior") -> None:
        if duels not in DUEL_MODES:
            raise ValueError(f"duels must be one of {DUEL_MODES}, got {duels!r}")
        options = dict(p_probe=p_probe, max_mult=max_mult, r_step=r_step,
                       duel_directions=duel_directions, stop_sd=stop_sd,
                       acquisition=acquisition, widths=widths, direction_set=direction_set)
        if duels == "random" and options != _EIG_DEFAULTS:
            changed = sorted(k for k in options if options[k] != _EIG_DEFAULTS[k])
            raise ValueError(f"{changed} apply to duels='eig' only")
        if acquisition not in ACQUISITIONS:
            raise ValueError(f"acquisition must be one of {ACQUISITIONS}, got {acquisition!r}")
        if widths not in WIDTHS:
            raise ValueError(f"widths must be one of {WIDTHS}, got {widths!r}")
        if direction_set not in DIRECTION_SETS:
            raise ValueError(f"direction_set must be one of {DIRECTION_SETS}, got "
                             f"{direction_set!r}")
        if direction_set != "posterior" and duel_directions != "free":
            raise ValueError("direction_set='curve_subspace' replaces the posterior's axes, which "
                             "only duel_directions='free' uses")
        if not 0.0 <= p_probe <= 1.0:
            raise ValueError(f"p_probe must be in [0, 1], got {p_probe}")
        if max_mult <= 0.0:
            raise ValueError(f"max_mult must be positive, got {max_mult}")
        if r_step is not None and r_step <= 0.0:
            raise ValueError(f"r_step must be positive or None, got {r_step}")
        if duel_directions not in DIRECTIONS:
            raise ValueError(f"duel_directions must be one of {DIRECTIONS}, got {duel_directions!r}")
        if stop_sd is not None and stop_sd <= 0.0:
            raise ValueError(f"stop_sd must be positive or None, got {stop_sd}")
        if duel_space not in DUEL_SPACES:
            raise ValueError(f"duel_space must be one of {DUEL_SPACES}, got {duel_space!r}")
        self.duels = duels
        self.p_probe, self.max_mult, self.r_step = float(p_probe), float(max_mult), r_step
        self.duel_directions, self.stop_sd = duel_directions, stop_sd
        if space is not None:
            # Phase 5: the arm's space is the learner's own and its duel
            # space; its whitening and radius come from the fold's data prior, never from eps.
            if design is not space.design:
                raise ValueError("space= must be a space of the given design")
            if eps != 0.0 or d is not None or duel_space != "reference":
                raise ValueError("a given space fixes eps, d and the duel space: Phase 5 draws "
                                 "duels in the arm's own space (spec 4.2)")
            self.space = self.duel_space = space
        else:
            self.space = LearnerSpace.build(design, eps=eps, d=d)
            # At eps = 0 the reference space is built from identical inputs, so reusing
            # self.space changes no number -- and it sidesteps transfer's full-rank
            # requirement, which a truncated eps = 0 space would otherwise trip.
            if duel_space == "own" or eps == 0.0:
                self.duel_space = self.space
            else:
                if self.space.d < design.n_params:
                    raise ValueError(
                        f"duel_space='reference' at eps={eps} needs full-rank whitening "
                        f"(d = n_params, Ruling R15); got d={self.space.d} of "
                        f"{design.n_params}"
                    )
                self.duel_space = LearnerSpace.build(design, eps=0.0, d=d)
        self.prior = prior
        # Separate streams: resampling is triggered by the answers, so a shared stream
        # would make later "random" duels depend on the data.
        self._duel_rng = np.random.default_rng([int(seed), _STREAM_SALT, 0])
        self.posterior = SMCPosterior(self.space.d,
                                      np.random.default_rng([int(seed), _STREAM_SALT, 1]),
                                      n_particles=n_particles, prior=prior)
        self._ell_trace: list[float] = []
        self._lambda_trace: list[float] = []
        self._kappa_trace: list[float] = []
        self.n_no_preference = 0
        self._duel_trace: list[list] = []
        self._duel_kind: list[str] = []
        #: Delta / ell_hat**2 for each eig-mode duel, aligned with `_duel_kind` (F3).
        #: Random duels never append here.
        self._duel_intensity: list[float] = []
        self._pending_multiple: float | None = None
        self.acquisition, self.widths, self.direction_set = acquisition, widths, direction_set
        #: Any of the three acquisition variants is on (ICORR section 9a).
        self._variant = (acquisition, widths, direction_set) != ("eig", "ladder", "posterior")
        # The curve metric in the posterior's own z (A-curve and A-sub need it; A-width not).
        self._metric = (curve_metric(self.space)
                        if acquisition == "curve" or direction_set == "curve_subspace" else None)
        self._n_candidates_trace: list[int] = []
        self._n_axes_trace: list[int] = []
        self._n_capped_trace: list[int] = []

    def _propose_random(self) -> tuple[np.ndarray, np.ndarray]:
        """A prior centre clipped to the ball, a random direction, a prescribed separation.

        All three are drawn in the duel space, so under the same seed shows the
        same curves at every eps. ``update`` maps what was shown into the posterior's space.
        """
        self._duel_kind.append("random")
        space, rng = self.duel_space, self._duel_rng
        centre = space.clip_to_ball(rng.normal(size=space.d))
        direction = rng.normal(size=space.d)
        direction /= np.linalg.norm(direction)
        multiple = float(rng.choice(SEPARATIONS))
        self._pending_multiple = multiple
        separation = multiple * np.exp(self.prior.log_ell0)
        z_a = space.clip_to_ball(centre + 0.5 * separation * direction)
        z_b = space.clip_to_ball(centre - 0.5 * separation * direction)
        return space.to_theta(z_a), space.to_theta(z_b)

    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._propose_random() if self.duels == "random" else self._propose_eig()

    def _z_worn(self) -> np.ndarray:
        """The curve currently delivered: z-hat, clipped into the trust ball, duel coordinates."""
        return self.duel_space.clip_to_ball(
            self.space.transfer(self.posterior.zstar_mean(), self.duel_space))

    def _propose_eig(self) -> tuple[np.ndarray, np.ndarray]:
        own, duel, post, rng = self.space, self.duel_space, self.posterior, self._duel_rng
        ell = post.ell_mean()
        z_hat = post.zstar_mean()
        z_worn = self._z_worn()
        region = step_region(z_worn, self.r_step)
        # A probe whose truncated ends coincide becomes an acquisition duel (R18 item 11,
        # revised again: ends are truncated along the duel's own line, see
        # candidates.truncate_duel, rather than clipped individually).
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
            hooks = self._variant_hooks(ell, region) if self._variant else {}
            cands = candidate_duels(post.x, post.weights(), own, duel, rng, ell_hat=ell,
                                    max_mult=self.max_mult, directions=self.duel_directions,
                                    **region, **hooks)
            # a change decision T: under a heavy-tailed prior the proxy is scored on the bulk.
            trim = TRIM_QUANTILE if self.prior.nu is not None else None
            if self.acquisition == "curve":
                score = expected_curve_variance_reduction(post.x, post.weights(), own.d,
                                                          cands["z_a"], cands["z_b"],
                                                          self._metric, trim=trim)
            else:
                score = expected_information_gain(post.x, post.weights(), own.d,
                                                  cands["z_a"], cands["z_b"], trim=trim)
            best = int(np.argmax(score))
            z_a, z_b = cands["z_a"][best], cands["z_b"][best]
            kind, multiple = "acq", float(cands["multiple"][best])
            if self._variant:
                self._n_candidates_trace.append(int(len(cands["z_a"])))
                if "n_axes" in cands:
                    self._n_axes_trace.append(int(cands["n_axes"]))
                if "n_capped" in cands:
                    self._n_capped_trace.append(int(cands["n_capped"]))
        self._pending_multiple = multiple
        self._duel_kind.append(kind)
        # F3: the realised intensity of this eig-mode duel, Delta / ell_hat**2, both z_hat
        # and ell_hat taken at proposal time -- before to_theta.
        delta = 0.5 * (float(np.dot(z_b - z_hat, z_b - z_hat))
                       - float(np.dot(z_a - z_hat, z_a - z_hat)))
        self._duel_intensity.append(delta / (ell * ell))
        return own.to_theta(z_a), own.to_theta(z_b)

    def _variant_hooks(self, ell: float, region: dict) -> dict:
        """candidate_duels' hooks for A-width and A-sub (ICORR section 9a)."""
        hooks = {}
        if self.direction_set == "curve_subspace":
            metric = self._metric
            hooks["axes_rule"] = lambda cov: curve_subspace_axes(cov, metric)
        if self.widths == "target":
            post, own, duel = self.posterior, self.space, self.duel_space
            x, w = post.x, post.weights()

            def width_rule(c_own, u):
                pair = target_width_duel(x, w, c_own, u, own, duel, **region)
                if pair is None:
                    return None
                a, b, capped = pair
                # `multiple` is the realised width in ell-hat, where the ladder records the
                # requested multiple; both are widths in units of ell-hat.
                return a, b, float(np.linalg.norm(a - b)) / ell, capped

            hooks["width_rule"] = width_rule
        return hooks

    def should_stop(self) -> bool:
        """our design notes: every posterior SD of z* is below stop_sd * ell-hat."""
        if self.stop_sd is None or self.posterior.n_answers == 0:
            return False
        sd = self.posterior.sd()[: self.space.d]
        return bool(np.max(sd) < self.stop_sd * self.posterior.ell_mean())

    def update(self, theta_a: np.ndarray, theta_b: np.ndarray, response: int) -> None:
        """Map what was actually shown back to ``z``; 0 prefers A, 1 prefers B, 2 neither."""
        z_a, z_b = self.space.to_z(theta_a), self.space.to_z(theta_b)
        # One-step-ahead predictive P(A > B), from the posterior BEFORE this answer. Draws
        # no random numbers. Recorded for every comparison, aligned with the run record's
        # `answers`, so the analysis can check calibration split by separation (R8).
        # For random duels `multiple` is in duel-space units. For eig duels it is instead
        # the requested ladder multiple k of ell-hat (kind "acq") or the probe intensity
        # level k (kind "probe") -- see `_duel_kind`. The recorded separation is always in
        # the posterior's own units.
        predicted_p_a = float(self.posterior.weights() @ np.exp(
            log_likelihood(self.posterior.x, z_a[None], z_b[None], [True])))
        self._duel_trace.append([self._pending_multiple,
                                 float(np.linalg.norm(z_a - z_b)), predicted_p_a])
        self._pending_multiple = None
        if response == 2:
            self.n_no_preference += 1
        elif response in (0, 1):
            self.posterior.update(z_a, z_b, prefers_a=(response == 0))
        else:
            raise ValueError(f"response must be 0, 1 or 2, got {response!r}")
        self._ell_trace.append(self.posterior.ell_mean())
        self._lambda_trace.append(self.posterior.lambda_mean())
        if self.prior.nu is not None:
            # Phase 5 our design notes: E[kappa | answers] after every comparison.
            self._kappa_trace.append(self.posterior.kappa_mean())

    def best(self) -> np.ndarray:
        """Random duels: the posterior mean of z*. EIG duels: that mean clipped into the
        trust ball, so the worn curve the step region is centred on is the one delivered."""
        if self.duels == "random":
            return self.space.to_theta(self.posterior.zstar_mean())
        return self.space.to_theta(self.duel_space.transfer(self._z_worn(), self.space))

    def posterior_summary(self) -> dict:
        post = self.posterior
        z_hat = post.zstar_mean()
        lambda_sd = post.lambda_sd()
        extras = {
            "ell_mean": post.ell_mean(),
            "lambda_mean": post.lambda_mean(),
            "lambda_sd": lambda_sd,
            # our design notes: say so rather than report a lapse rate the data never
            # moved from its prior.
            "lambda_identified": bool(lambda_sd < 0.8 * self.prior.lambda_sd),
            "zstar_sd_max": float(np.max(post.sd()[: self.space.d])),
            "zstar_hat": z_hat.tolist(),
            "best_outside_trust_region": bool(
                np.linalg.norm(self.space.transfer(z_hat, self.duel_space))
                > self.duel_space.radius),
            "ess_final": post.ess,
            "n_resample": post.n_resample,
            "acceptance_mean": float(np.mean(post.acceptance)) if post.acceptance else None,
            "n_no_preference": self.n_no_preference,
            "duel_trace": list(self._duel_trace),
            "duel_kind": list(self._duel_kind),
            "ell_trace": list(self._ell_trace),
            "lambda_trace": list(self._lambda_trace),
            "space_d": self.space.d,
            "trust_radius": self.duel_space.radius,
        }
        if self.prior.nu is not None:
            extras["kappa_trace"] = list(self._kappa_trace)
            extras["kappa_mean"] = post.kappa_mean()
        if self.duels == "eig":
            # F3: realised intensity Delta / ell_hat**2 per eig-mode duel, aligned with
            # `duel_kind`. Random mode never computes this and never gets the key.
            extras["duel_intensity"] = list(self._duel_intensity)
        if self._variant:
            # ICORR section 9a: per acquisition duel (kind "acq", in order), the size of the
            # candidate set, and A-sub's kept axes or A-width's pairs capped by the region.
            # Only a variant learner carries these keys, so A's rows are unchanged.
            extras["n_candidates_trace"] = list(self._n_candidates_trace)
            if self.direction_set == "curve_subspace":
                extras["subspace_axes_trace"] = list(self._n_axes_trace)
            if self.widths == "target":
                extras["width_capped_trace"] = list(self._n_capped_trace)
        return {"extras": extras}
