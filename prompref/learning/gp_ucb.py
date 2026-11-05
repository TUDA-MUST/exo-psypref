"""Comparator G: preference-GP optimisation with a JND-gated UCB.

Algorithm S1 of Arens et al. 2025, with the things the paper leaves open fixed by this
plan's decision:

- **GP3** caps the exploration step-up at ``KAPPA_MAX``. The published loop raises kappa
  until the proposal is perceivable and never stops; a 13,400-session sweep cannot run
  that. At the cap, G searches the UCB ranking restricted to candidates that already
  clear the gate -- the cap is our stopping rule, not Arens', so a perceivable candidate
  the ranking simply had not yet surfaced is not "exhausted" -- and falls back to the
  single furthest candidate, logging ``n_gate_exhausted``, only when *no* candidate in
  the set clears the gate at all.
- **GP4** measures the gate in delivered curves, in Nm RMS, not in parameters. Arens' two
  parameters were both newtons; eight basis weights have no common unit.
- **GP5** makes the search domain a fixed candidate set, the finite domain the paper's own
  library assumes. Candidates are one draw from the search region's own prior measure per
  session, not a uniform grid over the design's box. A first draft drew a scrambled Sobol
  sequence over the box instead; on the normalized path an earlier step wires, ``design.project`` clips
  radially onto a search region, and in eight dimensions the search region inscribed in a cube holds only
  ~1.6 % of its volume, so ~98 % of a box-uniform draw lands on the search region's *surface* -- a
  shell the arm's own prior mass does not sit on. The prior measure is also the one
  comparator C2 draws its batch from, so every comparator searches under the
  same measure. See the an earlier step report's a later fix for the candidate-best-profile distance and gate
  measurements this replacement is based on.
- **The measure is the arm's, not the design's**. ``design.sample_prior``
  is the default, but on every Phase 5 harness path that design is a
  :class:`~prompref.learning.space.WhitenedDesign`, whose ``sample_prior`` is a *standard
  normal* clipped to the search region -- the same draw in every arm, while the arm's own prior
  carries its Student-t tail through ``nu`` and ``scale``. Measured over 200k draws per arm
  at d = 8, the share of the arm's own prior mass beyond the radius a standard normal
  actually fills is 1.0 % at ``tight``, 10.7 % at t8, 19.4 % at t4, 31.5 % at t2 and 97.3 %
  at ``wide``: in four of five arms G searched a strictly narrower region than learner A's
  particle filter and C2's batch, and it was narrowest on exactly the atypical simulated users the
  heavy-tailed arms exist to represent. A baseline that loses for a reason we introduced is
  worth less than nothing, so ``prior=`` hands the learner the arm's own prior -- the same
  :meth:`~prompref.learning.smc.BowlPrior.sample_zstar` draw A and C2 use -- and the result
  is clipped through ``design.project``, so a candidate is always feasible. At ``tight`` the
  arm's prior *is* ``rng.normal(size=(n, d))``, so that arm's candidate set is unchanged.

Everything else follows the paper: fixed hyperparameters (GP1) scaled to the search region
(GP2), the incumbent duelled against the UCB argmax (GP6), and an indifferent answer entered
as two opposing comparisons (GP7).
"""

from __future__ import annotations

import numpy as np

from ..designs.base import Design
from ..user.simulated_user import NO_PREFERENCE, PREFER_A, PREFER_B, rms_distance
from .preference_gp import OUTPUT_SCALE, PREFERENCE_NOISE, PreferenceGP

#: Arens et al.: the exploration weight and its step. The cap is.
KAPPA0 = 1.5
KAPPA_STEP = 0.1
KAPPA_MAX = 6.5
#: the size of the per-session candidate draw from the prior measure.
#:
#: **Measured, not assumed**.
#: The case for raising this to 32768 was that G's candidate best-profile distance -- the best its own
#: domain can do -- exceeds its convergence threshold for every user measured, so the
#: discretisation and not the learning bounded its headline number. Sixteen times the
#: candidates does lower the best-profile distance about as spacing predicts (n^(-1/8)): 0.96 -> 0.60 Nm at
#: ``tight``, and 8-38 % across three real users in every arm, averaged over six candidate
#: seeds.
#:
#: **But the best-profile distance is not what bounds the delivered curve, so buying best-profile distance buys nothing.**
#: At 150 comparisons, 2048 -> 32768 on real users: at ``tight`` the delivered curve got
#: *worse* (1.56 -> 1.77 Nm median over three users here; 1.72 -> 2.03 Nm over the
#: coordinator's five-user smoke), while at t4 all three users improved (2.14 -> 1.45,
#: 1.92 -> 1.63, 1.47 -> 1.46). Note the two measures move independently in both arms: at
#: ``tight`` the best-profile distance improved 0.36 Nm while delivered worsened, and at t4 the best-profile distance got
#: *worse* (0.70 -> 0.80) while delivered improved. So the effect of a denser domain at this
#: budget is arm-dependent, and it is not mediated by the best-profile distance.
#:
#: Left at the plan's 2048 because the evidence does not point one way -- not because 32768
#: was shown to be worse. What bounds G's result is two things, both properties of the
#: published method in eight dimensions, and the paper should report both: the domain, whose
#: best-profile distance is above G's threshold at *either* size, and the budget, which gives the GP about
#: twenty distinct observations however many candidates it is offered. ``n_candidates`` stays
#: a constructor argument precisely so a 5c cell can vary it deliberately and say so.
#:
#: One tempting explanation does *not* survive the measurement: that the extra candidates are
#: mostly unexplored decoys a kappa = 1.5 upper bound chases, diluting the budget. At t4 the
#: denser set produced *more* distinct observations, not fewer (``n_shown`` 26 -> 41,
#: 30 -> 31, 26 -> 32), which is the opposite of dilution.
#:
#: Cost, if a cell does raise it, measured solo on one real user at 150 comparisons
#: (``tight``/t2): construction 0.39 s and session 9.7-12.4 s at 32768, against 0.03 s and
#: 0.7-1.1 s at 2048, and against learner A's 17.6-18.2 s in the same arms. Memory is linear:
#: ``curves`` is n x 101 x 8 bytes, and the transient peak is ``PreferenceGP._kernel``'s
#: ``n x n_shown x d x 8`` difference array -- +102-117 MB of RSS per worker at 32768 against
#: +6-7 MB at 2048, so ~0.9 GB of peak headroom across the sweep's eight workers.
N_CANDIDATES = 2048
#: Arens' length scale is 0.2 of his parameter span.
LENGTHSCALE_FRACTION = 0.2

_STREAM_SALT = 0x6B11CE


class GPUCBLearner:
    """A preference GP over a fixed candidate set, queried by JND-gated UCB."""

    def __init__(self, design: Design, seed: int = 0, *, jnd: float | None = None,
                 lengthscale: float | None = None, n_candidates: int = N_CANDIDATES,
                 prior=None, kappa0: float = KAPPA0, kappa_step: float = KAPPA_STEP,
                 kappa_max: float = KAPPA_MAX, output_scale: float = OUTPUT_SCALE,
                 preference_noise: float = PREFERENCE_NOISE) -> None:
        """``prior`` is the measure the candidate set is drawn from (see the module
        docstring's second GP5 note). It is a :class:`~prompref.learning.smc.BowlPrior`
        -- anything offering ``sample_zstar(rng, n, d)`` -- and its draw is clipped
        through ``design.project``, exactly as ``NormalizedDesign.sample_prior`` clips its
        own. ``None`` keeps ``design.sample_prior``, which is what a design searched in
        its own parameters (rather than in an arm's normalized search region) wants.
        """
        if kappa_max < kappa0:
            raise ValueError(f"kappa_max {kappa_max} is below kappa0 {kappa0}")
        if kappa_step <= 0.0:
            raise ValueError(f"kappa_step must be positive, got {kappa_step}")
        if n_candidates < 2:
            raise ValueError(f"n_candidates must be at least 2, got {n_candidates}")
        self.design = design
        self.jnd = None if jnd is None else float(jnd)
        self.kappa0 = float(kappa0)
        self.kappa_step = float(kappa_step)
        self.kappa_max = float(kappa_max)
        # Determinism (a global plan constraint): the one Generator below is the only
        # randomness this learner ever draws, seeded and salted so it cannot collide
        # with another comparator's stream under a shared ``seed`` -- see
        # ``test_the_candidate_draw_is_deterministic_and_ignores_the_legacy_generator``.
        self._rng = np.random.default_rng([int(seed), _STREAM_SALT])

        # one draw from the search region's own prior measure per session --
        # see the module docstring for why this replaced a uniform Sobol grid, and why
        # `prior` (the arm's measure) beats the design's in a Phase 5 arm.
        self.candidates = np.asarray(
            design.sample_prior(self._rng, int(n_candidates)) if prior is None
            else self._prior_candidates(design, prior, int(n_candidates)), dtype=float)
        self.curves = np.array([design.curve(theta) for theta in self.candidates])
        # the published length scale as a fraction of the search span.
        span = np.asarray(design.high, dtype=float) - np.asarray(design.low, dtype=float)
        self.lengthscale = (float(lengthscale) if lengthscale is not None
                            else LENGTHSCALE_FRACTION * float(np.max(span)))
        self.gp = PreferenceGP(self.lengthscale, output_scale=output_scale,
                               preference_noise=preference_noise)

        self._shown: list[int] = []
        self._pairs: list[tuple[int, int]] = []
        self._pending: tuple[int, int] | None = None
        self._fitted_n_pairs: int | None = None
        self.n_gate_blocked = 0
        self.n_gate_exhausted = 0
        self.kappa_max_used = self.kappa0

    # -- internals ---------------------------------------------------------------------

    def _prior_candidates(self, design: Design, prior, n: int) -> np.ndarray:
        """``n`` draws from ``prior``, clipped into ``design``'s feasible set.

        The clip is ``design.project`` rather than an explicit radial formula, for the
        same reason ``NormalizedDesign.sample_prior`` and ``SpacePriorDesign.sample_prior``
        use it: it is the one definition of "feasible" the whole space module shares, so
        a candidate here is feasible by exactly the test LineCoSpar and A's duels use. On
        the normalized path it is the radial clip onto the arm's search region; on a design searched
        in its own parameters it is that design's box.
        """
        z = prior.sample_zstar(self._rng, n, design.n_params)
        return np.array([design.project(v) for v in z])

    def _index_of_shown(self, index: int) -> int:
        if index not in self._shown:
            self._shown.append(index)
        return self._shown.index(index)

    def _fit(self) -> None:
        """Fits once per distinct ``_pairs`` state.

        The saving is not ``propose`` and a ``best`` called right after it in the
        same comparison -- ``run.py`` calls ``update`` between them, and that is
        exactly what changes ``_pairs``. It is ``best`` and the *next* comparison's
        ``propose``: ``run.py`` calls ``best`` once per comparison, right after that
        comparison's ``update``, and nothing touches ``_pairs`` between that call and
        the following ``propose``, so the second of those two fits is free. The
        Laplace MAP is the expensive part, and ``_pairs`` only ever grows, so its length
        is an exact, cheap key for "has anything changed since the last fit".
        """
        n_pairs = len(self._pairs)
        if self._fitted_n_pairs == n_pairs:
            return
        points = (self.candidates[self._shown] if self._shown
                  else np.zeros((0, self.candidates.shape[1])))
        self.gp.fit(points, self._pairs)
        self._fitted_n_pairs = n_pairs

    def _incumbent_index(self) -> int:
        """The posterior-mean argmax over ``_shown``. Assumes ``_fit`` has just run."""
        return int(self._shown[int(np.argmax(self.gp.mean_at_points()))])

    def _perceivable(self, a: int, b: int) -> bool:
        if self.jnd is None:
            return True
        return rms_distance(self.curves[a], self.curves[b]) >= self.jnd

    def _curve_distances(self, reference: int) -> np.ndarray:
        """RMS distance, in Nm, from every candidate's curve to ``reference``'s.

        Vectorised, not a Python loop over ``self.curves``: the gate's cap path needs
        this over the whole candidate set, and at the exhaustion rates a discrete
        candidate set can reach, a per-candidate Python loop here was measured to add
        real overhead across a 13,400-session sweep.
        """
        return np.sqrt(np.mean((self.curves - self.curves[reference]) ** 2, axis=1))

    @staticmethod
    def _argmax_excluding(scores: np.ndarray, exclude: int) -> int:
        """argmax of ``scores``, refusing ``exclude``: a duel needs two distinct designs.

        Arens duels an incumbent against a proposal, never a design against itself. On
        Algorithm S1's continuous domain a UCB argmax coincides with a specific existing
        point with probability zero; a fixed candidate set reintroduces that
        possibility, so it is excluded explicitly rather than left to the gate to catch
        (the gate only catches it when ``jnd`` is set, and even then only by accident).
        """
        masked = np.array(scores, dtype=float, copy=True)
        masked[exclude] = -np.inf
        return int(np.argmax(masked))

    @staticmethod
    def _argmax_where(scores: np.ndarray, mask: np.ndarray) -> int:
        """argmax of ``scores`` restricted to where ``mask`` is True.

        ``mask`` must have at least one ``True`` entry -- callers check that first,
        since an empty mask is exactly the genuinely-exhausted case, handled
        separately rather than by this returning a meaningless answer.
        """
        masked = np.where(mask, np.asarray(scores, dtype=float), -np.inf)
        return int(np.argmax(masked))

    # -- the learner protocol ------------------------------------------------------------

    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pending is not None:
            raise RuntimeError("update() the previous proposal before proposing again")
        if not self._pairs:
            # Algorithm S1 line 2: the first comparison is random, and shown as drawn
            # even when it fails the gate -- there is no posterior yet, so there is no
            # kappa to step up. A gate failure here is therefore already at the cap
            #, logged as such rather than corrected.
            a, b = self._rng.choice(len(self.candidates), size=2, replace=False)
            a, b = int(a), int(b)
            if not self._perceivable(a, b):
                self.n_gate_blocked += 1
                self.n_gate_exhausted += 1
            self._pending = (a, b)
            return self.candidates[a].copy(), self.candidates[b].copy()

        self._fit()
        mean, sd = self.gp.predict(self.candidates)
        incumbent = self._incumbent_index()
        kappa = self.kappa0
        proposal = self._argmax_excluding(mean + kappa * sd, incumbent)
        while not self._perceivable(incumbent, proposal):
            self.n_gate_blocked += 1
            if kappa >= self.kappa_max:
                gaps = self._curve_distances(incumbent)
                perceivable = gaps >= self.jnd
                perceivable[incumbent] = False
                if np.any(perceivable):
                    # A perceivable candidate exists; the capped step-up (,
                    # ours, not Arens') just had not surfaced it. Keep the UCB ranking,
                    # restricted to what already clears the gate, at the capped kappa --
                    # not exhaustion, so n_gate_exhausted is not touched.
                    proposal = self._argmax_where(mean + kappa * sd, perceivable)
                else:
                    # Genuinely exhausted: no candidate in the whole set is perceivably
                    # different from the incumbent. Excluded explicitly, not just
                    # incidentally: the incumbent's own gap is 0 and so is never the
                    # argmax of "farthest", but this is the one candidate the duel
                    # must never return regardless.
                    self.n_gate_exhausted += 1
                    proposal = self._argmax_excluding(gaps, incumbent)
                break
            kappa = min(kappa + self.kappa_step, self.kappa_max)
            self.kappa_max_used = max(self.kappa_max_used, kappa)
            proposal = self._argmax_excluding(mean + kappa * sd, incumbent)
        self._pending = (incumbent, proposal)
        return self.candidates[incumbent].copy(), self.candidates[proposal].copy()

    def update(self, theta_a: np.ndarray, theta_b: np.ndarray, response: int) -> None:
        """``PREFER_A`` (0), ``PREFER_B`` (1) or ``NO_PREFERENCE``."""
        if self._pending is None:
            raise RuntimeError("call propose() before update()")
        # Validated before anything mutates: a raise here must leave `_pending`,
        # `_shown` and `_pairs` -- and so the fit cache's key -- exactly as they were.
        response = int(response)
        if response not in (PREFER_A, PREFER_B, NO_PREFERENCE):
            raise ValueError(
                f"response must be PREFER_A (0), PREFER_B (1) or NO_PREFERENCE (2), "
                f"got {response}")
        a, b = self._pending
        self._pending = None
        ia, ib = self._index_of_shown(a), self._index_of_shown(b)
        if response == PREFER_A:
            self._pairs.append((ia, ib))
        elif response == PREFER_B:
            self._pairs.append((ib, ia))
        else:
            self._pairs.append((ia, ib))
            self._pairs.append((ib, ia))

    def best(self) -> np.ndarray:
        if not self._shown:
            return self.candidates[0].copy()
        self._fit()
        return self.candidates[self._incumbent_index()].copy()

    def posterior_summary(self) -> dict:
        """``n_gate_blocked`` counts rejected proposals (once per failed gate check, so
        one comparison's step-up can add several); ``n_gate_exhausted`` counts
        comparisons *shown* although no candidate in the whole set was perceivably
        different from the incumbent -- genuine exhaustion, not merely that the capped
        step-up had not yet surfaced one -- or the first, ungated
        comparison, where no step-up exists to surface one at all.
        """
        return {"extras": {
            "n_answers": len(self._pairs),
            "n_candidates": int(len(self.candidates)),
            "n_shown": len(self._shown),
            "n_gate_blocked": int(self.n_gate_blocked),
            "n_gate_exhausted": int(self.n_gate_exhausted),
            "kappa_max_used": float(self.kappa_max_used),
            "gp_lengthscale": float(self.lengthscale),
        }}
