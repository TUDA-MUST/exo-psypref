"""Comparators GP-EUBO and GP-BALD: preference GPs built from BoTorch.

design ``2026-10-17-icorr-user-tuning-reframe-design.md`` section 9a, "Review response", GP-EUBO
(PF3) and GP-BALD (PF3b). One learner; the two
differ only in the acquisition that picks each pair (``acquisition="eubo"`` or ``"bald"``):

- **Model.** BoTorch's ``PairwiseGP`` (Chu & Ghahramani's probit likelihood with the Laplace
  approximation) on the arm's whitened coordinates ``z``. The utility's prior mean is the data
  prior's bowl, ``-c/2 ||z||^2``, with ``c`` learned (:class:`BowlMean`). The kernel is
  PairwiseGP's own default (a scaled ARD RBF with BoTorch's priors).
- **Refit after every answer**, as BoTorch's preference-BO tutorial does: a fresh model on all
  answers so far, its hyperparameters (lengthscales, output scale and ``c``) fitted by
  ``fit_gpytorch_mll`` on the Laplace evidence.
- **Acquisition.** GP-EUBO: analytic EUBO (``AnalyticExpectedUtilityOfBestOption``; for a pair
  under the Laplace posterior it is exactly the quantity qEUBO of Astudillo et al. 2023 estimates
  by Monte Carlo), decision-theoretic. GP-BALD: BoTorch's
  ``PairwiseBayesianActiveLearningByDisagreement``, information-based, with its Monte Carlo draws
  held fixed within each acquisition (:class:`FixedDrawBALD`). Either is maximised over pairs in
  A's search region, the arm's ball ``||z|| <= r`` (r the chi quantile at 0.99), the way BoTorch
  handles a nonlinear region: SLSQP with the ball as an intra-point nonlinear inequality
  constraint (``gen_candidates_scipy``, see :meth:`GPEUBOLearner._optimize`), started from the
  best raw pairs (``initialize_q_batch_topn``) of a per-comparison pool of prior draws inside the
  ball and pairs with the incumbent.
- **Start** at generic assistance, ``z = 0``: it is the identified assistance before any answer,
  and the first duel shows it against one prior draw inside the ball (analytic EUBO needs a model
  with data; BoTorch's tutorials, like Arens' Algorithm S1 for G, start from a random duel).
- **Identified assistance**: the maximiser of the posterior mean within the ball
  (``optimize_acqf`` on ``PosteriorMean`` with the same constraint, started from the shown points,
  ``z = 0`` and prior draws). The harness delivers it through the arm's device-clipped design,
  like every other optimizer's (:class:`~prompref.learning.whitened.WhitenedLearner`).

Two deviations from BoTorch's defaults, each needed for the planned model and stated as a
build decision:

1. **Inputs are normalised** with BoTorch's ``Normalize`` and *fixed* bounds, the cube
   ``[-r, r]^d`` around the ball (the tutorial normalises too; PairwiseGP's default lengthscale
   prior, Gamma(2.4, 2.7), is meant for unit-cube inputs). This is an isotropic affine map of
   ``z``, so the GP still lives on the whitened coordinates; :class:`BowlMean` maps back to ``z``.
2. **The evidence counts the prior mean.** PairwiseGP subtracts its mean module in the MAP solve
   and in prediction, but ``PairwiseLaplaceMarginalLogLikelihood`` forms its prior term from
   ``f_map`` alone, which is right only for a zero mean. :class:`BowlLaplaceMLL` adds the omitted
   term, ``m' K^-1 f - m' K^-1 m / 2`` (so that the prior term is ``(f - m)' K^-1 (f - m) / 2``),
   to BoTorch's value; at ``m = 0`` it is BoTorch's MLL exactly.

Determinism (a global plan constraint): PairwiseGP's MAP solve perturbs its start with numpy's
*global* generator, and ``fit_gpytorch_mll`` resamples hyperparameters from their priors with
torch's on a failed attempt. Every model build, fit and optimisation therefore runs inside
:meth:`GPEUBOLearner._seeded`, which seeds both from this learner's own stream and restores the
global states afterwards. Torch runs single-threaded (``torch.set_num_threads(1)``), which the
constructor sets in the worker process, so rows do not depend on the machine's thread count.

Only has torch and BoTorch; this module is imported lazily, by the
``gp_eubo`` learner alone (as SortCMA's pycma), so no other learner, test or row needs either.
"""

from __future__ import annotations

import contextlib
import time
import warnings

import botorch
import gpytorch
import numpy as np
import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.preference import (AnalyticExpectedUtilityOfBestOption,
                                            PairwiseBayesianActiveLearningByDisagreement)
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation.gen import gen_candidates_scipy
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.optim.initializers import initialize_q_batch_topn
from gpytorch.constraints import GreaterThan
from gpytorch.means import Mean
from gpytorch.priors import LogNormalPrior

from ..user.oracle import NO_PREFERENCE, PREFER_A, PREFER_B

#: Raw pairs scored per comparison before the restarts, and the restarts: BoTorch's
#: preference-BO tutorial's RAW_SAMPLES = 512 and NUM_RESTARTS = 3.
N_RAW = 512
N_RESTARTS = 3
#: SLSQP's iteration cap per restart: ``None`` keeps gen_candidates_scipy's default (2000),
#: which the tutorial's ``optimize_acqf`` call also leaves in place.
MAXITER = None
#: The mean scale ``c``: its start and BoTorch-style prior, LogNormal(0, 1) (median 1, the
#: natural scale: A's bowl at ell = 1 is ``c = sqrt 2`` in PairwiseGP's probit units).
MEAN_SCALE_INIT = 1.0
MEAN_SCALE_PRIOR = (0.0, 1.0)
#: Its lower bound, PairwiseGP's own lower bound on a lengthscale.
MEAN_SCALE_LOWER = 1e-4
#: Initial conditions sit this fraction inside the ball, so SLSQP starts feasible.
SHRINK = 1.0 - 1e-9

#: The acquisitions: analytic EUBO (GP-EUBO, PF3) and BoTorch's pairwise BALD (GP-BALD, PF3b).
ACQUISITIONS = ("eubo", "bald")
#: What a failed hyperparameter fit or acquisition climb may raise (BoTorch's ModelFittingError
#: after its retries; a ValueError or RuntimeError, linear_operator's NotPSDError among them, from a
#: numerical failure inside a step). Each is counted in the row, and the session goes on.
FIT_FAILURES = (ModelFittingError, ValueError, RuntimeError)

_STREAM_SALT = 0x6E0B0
DTYPE = torch.float64


class FixedDrawBALD(PairwiseBayesianActiveLearningByDisagreement):
    """BoTorch's pairwise BALD with its Monte Carlo draws held fixed from call to call.

    BoTorch's BALD draws fresh normal samples (1024 by default) in every forward pass, so the
    objective SLSQP climbs would change between iterations. Re-seeding torch's generator with one
    seed per acquisition before each pass makes it a sample-average approximation, as BoTorch's
    MC samplers fix their base samples: deterministic and smooth in ``X``. Nothing else changes,
    including BoTorch's defaults (``num_samples=1024``, ``std_noise=0``)."""

    def __init__(self, pref_model, seed: int) -> None:
        super().__init__(pref_model=pref_model)
        self._draw_seed = int(seed)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self._draw_seed)
            return super().forward(X)


class BowlMean(Mean):
    """``m(x) = -c/2 ||z||^2`` with ``z = r (2 x - 1)``: the data prior's bowl in the whitened
    coordinates, evaluated on BoTorch's normalised inputs ``x = (z + r) / (2 r)``; ``c`` learned,
    with a LogNormal prior.

    ``c`` is bounded below as PairwiseGP bounds its own lengthscales: ``GreaterThan(1e-4,
    transform=None)``, which the scipy fit enforces as an L-BFGS-B bound. A softplus transform
    (``Positive``) let a long line-search step underflow ``c`` to exactly 0, where the LogNormal
    prior refuses to evaluate (one session of the first launch, superseded)."""

    def __init__(self, radius: float, init: float = MEAN_SCALE_INIT,
                 prior: tuple[float, float] | None = MEAN_SCALE_PRIOR) -> None:
        super().__init__()
        self.radius = float(radius)
        self.register_parameter("raw_scale", torch.nn.Parameter(torch.zeros((), dtype=DTYPE)))
        self.register_constraint("raw_scale", GreaterThan(MEAN_SCALE_LOWER, transform=None,
                                                          initial_value=float(init)))
        if prior is not None:
            self.register_prior("scale_prior", LogNormalPrior(*prior),
                                lambda m: m.scale, lambda m, v: m._set_scale(v))
        self._set_scale(init)

    @property
    def scale(self) -> torch.Tensor:
        return self.raw_scale_constraint.transform(self.raw_scale)

    def _set_scale(self, value) -> None:
        value = torch.as_tensor(value, dtype=DTYPE).to(self.raw_scale)
        self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.radius * (2.0 * x - 1.0)
        return -0.5 * self.scale * (z ** 2).sum(dim=-1)


class BowlLaplaceMLL(PairwiseLaplaceMarginalLogLikelihood):
    """BoTorch's Laplace evidence with the prior mean in its prior term (module docstring,
    deviation 2): BoTorch's value plus ``m' K^-1 f - m' K^-1 m / 2``."""

    def forward(self, post, comp):
        mll = super().forward(post, comp)
        model = self.model
        f = post.mean.squeeze(-1)
        m = model._prior_mean(model.transform_inputs(model.datapoints))
        kinv_m = torch.cholesky_solve(m.unsqueeze(-1), model.covar_chol).squeeze(-1)
        return mll + (kinv_m * f).sum() - 0.5 * (kinv_m * m).sum()


class GPEUBOLearner:
    """PairwiseGP with a learned bowl mean, queried by analytic EUBO inside the arm's ball.

    Works in the whitened coordinates of ``space`` (a
    :class:`~prompref.learning.space.LearnerSpace`); :class:`~prompref.learning.whitened.WhitenedLearner`
    translates to design parameters. ``prior`` is the arm's
    :class:`~prompref.learning.smc.BowlPrior`, the measure the raw pool is drawn from.
    """

    def __init__(self, space, seed: int = 0, *, prior, acquisition: str = "eubo",
                 n_raw: int = N_RAW, n_restarts: int = N_RESTARTS,
                 maxiter: int | None = MAXITER) -> None:
        if acquisition not in ACQUISITIONS:
            raise ValueError(f"acquisition must be one of {ACQUISITIONS}, got {acquisition!r}")
        torch.set_num_threads(1)
        self.acquisition = acquisition
        self.space = space
        self.d = int(space.d)
        self.radius = float(space.radius)
        self.prior = prior
        self.n_raw, self.n_restarts = int(n_raw), int(n_restarts)
        self.maxiter = None if maxiter is None else int(maxiter)
        self._rng = np.random.default_rng([int(seed), _STREAM_SALT])
        r = self.radius
        self._bounds = torch.tensor([[-r] * self.d, [r] * self.d], dtype=DTYPE)
        self._X: list[np.ndarray] = []
        self._comps: list[tuple[int, int]] = []
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self._model = None
        self._fitted_n = None
        self._best = np.zeros(self.d)
        self._best_n = 0
        self.n_no_preference = 0
        self.n_fit_failed = 0
        self.n_fit_warnings = 0
        self.n_climb_failed = 0
        self.n_acq_failed = 0
        self._last_warnings = 0
        self.timing = {"fit_s": 0.0, "acq_s": 0.0, "best_s": 0.0}
        self.traces: dict[str, list] = {"mean_scale": [], "outputscale": [], "lengthscale_mean": [],
                                        "eubo": [], "best_mean": []}

    # -- determinism ---------------------------------------------------------------------
    @contextlib.contextmanager
    def _seeded(self):
        """numpy's global and torch's generators seeded from this learner's stream for the
        duration of a BoTorch call, then restored (module docstring)."""
        seed = int(self._rng.integers(0, 2**31 - 1))
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    yield
                self._last_warnings = len(caught)
        finally:
            np.random.set_state(state)

    # -- the pool ------------------------------------------------------------------------
    def _draws(self, n: int) -> np.ndarray:
        """``n`` draws from the arm's prior, clipped radially into the ball (the clip every
        learner in the space shares), then pulled just inside it."""
        z = self.prior.sample_zstar(self._rng, n, self.d)
        return np.array([self.space.clip_to_ball(v) for v in z]) * SHRINK

    @staticmethod
    def _t(x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=float), dtype=DTYPE)

    def _ball(self):
        r2 = self.radius ** 2
        return [(lambda x: r2 - (x ** 2).sum(dim=-1), True)]

    # -- the model -----------------------------------------------------------------------
    def _build(self, X: torch.Tensor, comps: torch.Tensor):
        """PairwiseGP on the shown points and answers, with the bowl mean, its Laplace
        approximation solved at the starting hyperparameters."""
        model = PairwiseGP(X, comps, input_transform=Normalize(d=self.d, bounds=self._bounds))
        model.mean_module = BowlMean(self.radius)
        model.set_train_data(X, comps, update_model=True)
        return model

    def _fit(self) -> None:
        """A fresh model on every answer so far, hyperparameters refitted (the tutorial's
        ``init_and_fit_model``). Once per data state: ``best`` after an answer and the next
        ``propose`` share it."""
        n = len(self._comps)
        if self._fitted_n == n or n == 0:
            return
        started = time.perf_counter()
        X = self._t(np.array(self._X))
        comps = torch.as_tensor(np.array(self._comps), dtype=torch.long)
        with self._seeded():
            model = self._build(X, comps)
            mll = BowlLaplaceMLL(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except FIT_FAILURES:
                # BoTorch's retries all failed, or a numerical error escaped them: the model
                # with PairwiseGP's and BowlMean's starting hyperparameters, unfitted.
                self.n_fit_failed += 1
                model = self._build(X, comps)
            model.eval()
        self.n_fit_warnings += self._last_warnings
        self._model = model
        self._fitted_n = n
        kernel = model.covar_module
        self.traces["mean_scale"].append(float(model.mean_module.scale.detach()))
        self.traces["outputscale"].append(float(kernel.outputscale.detach()))
        self.traces["lengthscale_mean"].append(float(kernel.base_kernel.lengthscale.detach().mean()))
        self.timing["fit_s"] += time.perf_counter() - started

    def _optimize(self, acq, raw: np.ndarray, q: int) -> tuple[np.ndarray, float]:
        """Maximise ``acq`` inside the ball from the best raw candidates (``raw`` is
        ``b x q x d``): BoTorch's ``initialize_q_batch_topn`` picks the starts, and BoTorch's
        ``gen_candidates_scipy`` (the generator ``optimize_acqf`` runs; SLSQP with the ball as an
        intra-point nonlinear constraint, one start at a time as ``optimize_acqf`` requires for
        nonlinear constraints) climbs from each.

        Called directly rather than through ``optimize_acqf`` because ``optimize_acqf`` then
        checks feasibility at an absolute tolerance of 1e-8 on ``r^2 - ||z||^2`` (r^2 = 20 at
        d = 8), below SLSQP's own, and refuses ends that touch the boundary. Each end is instead
        clipped radially onto the ball, the clip every learner in the space shares. Every end and
        every start is then scored one at a time (a batch of one, as in the climb), and the best
        is returned, so a failed climb can never do worse than its start."""
        Xraw = self._t(raw)
        with torch.no_grad():
            vals = acq(Xraw)
        ics, _ = initialize_q_batch_topn(Xraw, vals, n=min(self.n_restarts, len(raw)))
        lo, hi = self._bounds[0], self._bounds[1]
        ends = []
        for ic in ics:
            try:
                cand, _ = gen_candidates_scipy(ic[None], acq, lower_bounds=lo, upper_bounds=hi,
                                               nonlinear_inequality_constraints=self._ball(),
                                               options={} if self.maxiter is None
                                               else {"maxiter": self.maxiter})
            except FIT_FAILURES:
                self.n_climb_failed += 1        # this start stays a candidate below
                continue
            z = cand.detach().cpu().numpy().reshape(q, self.d)
            ends.append(np.array([self.space.clip_to_ball(v) for v in z]))
        options = ends + [ic.detach().cpu().numpy().reshape(q, self.d) for ic in ics]
        with torch.no_grad():
            scores = np.array([float(acq(self._t(z)[None])) for z in options])
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        best = int(np.argmax(scores))
        return options[best], float(scores[best])

    # -- the learner protocol ------------------------------------------------------------
    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pending is not None:
            raise RuntimeError("update() the previous proposal before proposing again")
        if not self._comps:
            # The first duel: generic assistance against one prior draw in the ball.
            z_a, z_b = np.zeros(self.d), self._draws(1)[0]
            self.traces["eubo"].append(None)
        else:
            self._fit()
            started = time.perf_counter()
            half = self.n_raw // 2
            pool = self._draws(2 * half + half)
            pairs = pool[: 2 * half].reshape(half, 2, self.d)
            with_incumbent = np.stack([np.repeat(self._best[None] * SHRINK, half, axis=0),
                                       pool[2 * half:]], axis=1)
            raw = np.concatenate([pairs, with_incumbent], axis=0)
            try:
                with self._seeded():
                    if self.acquisition == "eubo":
                        acq = AnalyticExpectedUtilityOfBestOption(pref_model=self._model)
                    else:
                        acq = FixedDrawBALD(self._model,
                                            seed=int(self._rng.integers(0, 2**31 - 1)))
                    (z_a, z_b), value = self._optimize(acq, raw, q=2)
            except FIT_FAILURES:
                # The posterior itself failed: the incumbent against a fresh prior draw.
                self.n_acq_failed += 1
                (z_a, z_b), value = with_incumbent[0], None
            self.traces["eubo"].append(value)
            self.timing["acq_s"] += time.perf_counter() - started
        self._pending = (np.asarray(z_a, dtype=float), np.asarray(z_b, dtype=float))
        return self._pending[0].copy(), self._pending[1].copy()

    def update(self, z_a: np.ndarray, z_b: np.ndarray, response: int) -> None:
        """``PREFER_A`` (0), ``PREFER_B`` (1); ``NO_PREFERENCE`` (2) carries no comparison for
        a probit pairwise likelihood and is skipped, as A skips it."""
        if self._pending is None:
            raise RuntimeError("call propose() before update()")
        response = int(response)
        if response not in (PREFER_A, PREFER_B, NO_PREFERENCE):
            raise ValueError(f"response must be 0, 1 or 2, got {response}")
        a, b = self._pending
        self._pending = None
        if response == NO_PREFERENCE:
            self.n_no_preference += 1
            return
        ia, ib = len(self._X), len(self._X) + 1
        self._X += [a, b]
        self._comps.append((ia, ib) if response == PREFER_A else (ib, ia))

    def best(self) -> np.ndarray:
        """The posterior-mean maximiser in the ball; ``z = 0`` (generic) before any answer."""
        n = len(self._comps)
        if n == 0:
            return np.zeros(self.d)
        if self._best_n != n:
            self._fit()
            started = time.perf_counter()
            shown = np.array(self._X) * SHRINK
            raw = np.concatenate([shown, np.zeros((1, self.d)), self._draws(self.n_raw)],
                                 axis=0)[:, None, :]
            try:
                with self._seeded():
                    acq = PosteriorMean(self._model)
                    z, value = self._optimize(acq, raw, q=1)
            except FIT_FAILURES:
                # The posterior itself failed: the identified assistance stays as it was.
                self.n_acq_failed += 1
                z, value = self._best[None], None
            self._best, self._best_n = z[0], n
            self.traces["best_mean"].append(value)
            self.timing["best_s"] += time.perf_counter() - started
        return self._best.copy()

    def posterior_summary(self) -> dict:
        extras = {
            "gp_acquisition": self.acquisition,
            "n_answers": len(self._comps),
            "n_shown": len(self._X),
            "n_no_preference": int(self.n_no_preference),
            "n_fit_failed": int(self.n_fit_failed),
            "n_fit_warnings": int(self.n_fit_warnings),
            "n_climb_failed": int(self.n_climb_failed),
            "n_acq_failed": int(self.n_acq_failed),
            "gp_versions": dict(torch=torch.__version__, botorch=botorch.__version__,
                                gpytorch=gpytorch.__version__),
            "gp_mean_scale_trace": list(self.traces["mean_scale"]),
            "gp_outputscale_trace": list(self.traces["outputscale"]),
            "gp_lengthscale_mean_trace": list(self.traces["lengthscale_mean"]),
            "eubo_trace": list(self.traces["eubo"]),
            "best_mean_trace": list(self.traces["best_mean"]),
            "gp_fit_s": float(self.timing["fit_s"]),
            "gp_acq_s": float(self.timing["acq_s"]),
            "gp_best_s": float(self.timing["best_s"]),
            "zstar_hat": self._best.tolist(),
        }
        if self._model is not None:
            extras["gp_lengthscale"] = (self._model.covar_module.base_kernel.lengthscale
                                        .detach().cpu().numpy().ravel().tolist())
        return {"extras": extras}
