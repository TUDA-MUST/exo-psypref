"""Priors fitted to generated users.

Sample, then project: generated curves are projected onto the ProMP basis, or fitted
with the dual split-Gaussian, and the prior is the empirical mean and covariance of the
resulting parameters. Dual-Gaussian fits whose RMS exceeds 0.5 Nm are counted and
reported, but kept: most generated curves lie that far from either family (81 % for the
dual Gaussian, 86 % for the ProMP projection), and a poorly fitted curve's parameters
are still the family's closest point to that user. a later fix: the ProMP projection's
own RMS is now counted the same way, so `n_fit_failures` is comparable across families.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..basis import VonMisesBasis
from ..designs.dual_gauss import HIGH, LOW, PARAM_NAMES, DualGaussDesign, reference_centres
from ..designs.promp import PrompS0
from ..population.factors import FactorSwitches, ShiftSpread, TimingSpread
from ..population.folds import child_sequence
from ..population.generator import TrainingSet, generate_training_curves
from ..prior import Whitening

FAMILIES = ("promp_s0", "dual_gauss")
MAX_FIT_RMS_NM = 0.5

#: PrompS0's own default box (bound_nm=14) is not a physical limit -- it is a
#: feasibility clip left over from box-uniform sampling (PrompS0's docstring), from
#: before this project had a fitted weight prior at all. The physical limit is the
#: device's +/-12 Nm curve clip (prompref.population.curves.TORQUE_LIMIT_NM); a basis
#: *weight*, unlike a curve value, is not held to that bound by the partition-of-unity
#: design. About 5 % of training curves' projected weights need |w| > 14 to represent a
#: curve within the device's own physical limit, and the largest observed was 15.4. A
#: data prior built against bound_nm=14 would clip real, physically-reachable weights
#: (see also PROMP_DATA_BOUND_NM's use in DataPrior.design and share_at_bound), so the
#: fitted promp_s0 prior gets this wider box instead.
PROMP_DATA_BOUND_NM = 30.0


@dataclass(eq=False)
class DataPrior:
    family: str
    fold: str
    mu: np.ndarray
    sigma: np.ndarray
    n_curves: int
    n_fit_failures: int
    fit_rms_median_nm: float
    n_rejected_draws: int
    switches: dict
    timing_source: str
    seed: int
    source_hashes: dict = field(default_factory=dict)
    share_at_bound: dict = field(default_factory=dict)
    #: Provenance: {source, nu, scale_matrix, file_sha256} and
    #: {centre_ms, scale_ms, nu, source}, so a regenerated Schaefer file or another shift
    #: is visible in the prior itself.
    timing: dict = field(default_factory=dict)
    shift: dict = field(default_factory=dict)
    #: Constructor arguments of the family's design; the dual Gaussian's references follow
    #: the population's shift centre. Empty means the family defaults.
    design_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        self.mu = np.asarray(self.mu, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)

    def whitening(self) -> Whitening:
        return Whitening(self.mu, self.sigma, self.mu.size)

    def design(self) -> PrompS0 | DualGaussDesign:
        if self.family == "promp_s0":
            return PrompS0(prior=(self.mu, self.sigma), bound_nm=PROMP_DATA_BOUND_NM, **self.design_kwargs)
        return DualGaussDesign(prior=(self.mu, self.sigma), **self.design_kwargs)

    def space(self, radius: float):
        """The learner space of this prior: its design, *this* whitening, and ``radius``.

        Final review I2: 5b and 5c must build learners here, never with
        ``LearnerSpace.build(prior.design)``. For every family but PrompS0,
        ``learner_prior`` takes the moments of 4000 box-clipped prior draws, so the
        dual-Gaussian space would get a shrunken mu and Sigma (mu off by up to 0.19, eigenvalue
        ratios 0.81-0.99) and radii that do not match the 5a diagnostics. One whitening per
        fold and family serves 5a, 5b and 5c.
        """
        from ..learning.space import LearnerSpace

        return LearnerSpace(self.design(), self.whitening(), float(radius))

    def save(self, stem) -> None:
        # Build file names by appending, not by Path.with_suffix:
        # with_suffix replaces text after the *last* dot, so a stem such as
        # "loso-AB30.promp_s0" and "loso-AB30.dual_gauss" both resolved to
        # "loso-AB30.npz"/".json" and silently overwrote each other.
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        npz_path = stem.parent / (stem.name + ".npz")
        json_path = stem.parent / (stem.name + ".json")
        np.savez(npz_path, mu=self.mu, sigma=self.sigma)
        meta = {k: v for k, v in asdict(self).items() if k not in ("mu", "sigma")}
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, stem) -> "DataPrior":
        stem = Path(stem)
        npz_path = stem.parent / (stem.name + ".npz")
        json_path = stem.parent / (stem.name + ".json")
        arrays = np.load(npz_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        return cls(mu=arrays["mu"], sigma=arrays["sigma"], **meta)


def population_source_hashes() -> dict[str, str]:
    """Hashes of the code that generated a prior's training curves, and the fitting code
    of both families. A change to the population
    generator, either family's fit function, the ProMP basis, or the dual-Gaussian model
    must be visible here -- not just a change to curve generation.
    """
    from ..validation.stamp import ROOT, source_hash

    # folds.py holds FOLD_INDEX and the stream derivation the priors' seeds come from.
    files = ("prompref/population/curves.py", "prompref/population/factors.py",
             "prompref/population/folds.py",
             "prompref/population/generator.py", "prompref/designs/dual_gauss.py",
             "prompref/basis.py", "prompref/designs/promp.py", "prompref/priors/data_prior.py")
    return {rel: source_hash(ROOT / rel) for rel in files}


def share_at_bound(thetas: np.ndarray, low, high, names) -> dict[str, float]:
    """Share of each parameter's fitted values at or beyond either bound.

    "At or beyond" rather than "close to": a bounded fit (the dual-Gaussian's
    ``least_squares(bounds=...)``) can only ever reach a bound from inside, so this is
    exactly "pinned at the bound" there. An unconstrained fit (the ProMP projection,
    checked here against a box it was never fitted inside) can land past a bound
    entirely, and that must count too -- ``|w| >= PROMP_DATA_BOUND_NM`` is not "close to"
    anything in particular. The tolerance ``1e-6 * (high - low)`` only catches numerical
    near-misses at the inside edge of a bounded fit.
    """
    thetas = np.asarray(thetas, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    tol = 1e-6 * (high - low)
    at_bound = (thetas <= low + tol) | (thetas >= high - tol)
    return {name: float(at_bound[:, i].mean()) for i, name in enumerate(names)}


def _project_weights(curves: np.ndarray, basis: VonMisesBasis) -> np.ndarray:
    return basis.project(np.asarray(curves, dtype=float).T).T


def fit_promp_prior(curves: np.ndarray, basis: VonMisesBasis | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = basis or VonMisesBasis()
    curves = np.asarray(curves, dtype=float)
    weights = _project_weights(curves, basis)
    rms = np.sqrt(np.mean((weights @ basis.design.T - curves) ** 2, axis=1))
    return weights.mean(axis=0), np.cov(weights, rowvar=False), rms


def _fit_one_dual_gauss(job: tuple[np.ndarray, np.random.SeedSequence, dict]) -> tuple[np.ndarray, float]:
    curve, seed_sequence, design_kwargs = job
    return DualGaussDesign(**design_kwargs).fit_curve(curve, np.random.default_rng(seed_sequence))


def fit_dual_gauss_prior(curves: np.ndarray, seed: int | np.random.SeedSequence, workers: int = 1,
                         max_rms: float = MAX_FIT_RMS_NM, design_kwargs: dict | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``design_kwargs`` are the references the fits are made against; the prior's
    ``design`` must use the same ones, or its peak offsets mean something else."""
    curves = np.asarray(curves, dtype=float)
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    # Each curve's generator is built from its own spawned SeedSequence, passed whole
    #: truncating a child to one 32-bit state word made two of 20,000
    # curves share random starts with probability ~4.6 %.
    kwargs = dict(design_kwargs or {})
    jobs = [(curve, child_sequence(root, i), kwargs) for i, curve in enumerate(curves)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_fit_one_dual_gauss, jobs, chunksize=64))
    else:
        results = [_fit_one_dual_gauss(job) for job in jobs]
    thetas = np.array([theta for theta, _ in results])
    rms = np.array([r for _, r in results])
    if len(thetas) < thetas.shape[1] + 2:
        raise ValueError(f"need at least {thetas.shape[1] + 2} curves for a full-rank prior, "
                         f"got {len(thetas)}")
    # Every curve is kept; RMS above max_rms is reported, not excluded (see module docstring).
    within = rms <= max_rms
    return thetas.mean(axis=0), np.cov(thetas, rowvar=False), rms, within, thetas


def fold_training_set(train_trials, n_curves: int, seed: int,
                      switches: FactorSwitches = FactorSwitches(),
                      timing: TimingSpread | None = None,
                      seed_sequence: np.random.SeedSequence | None = None,
                      shift: ShiftSpread | None = None) -> TrainingSet:
    """The generated curves a fold's priors are fitted on -- regenerated identically when
    saved priors are loaded instead of refitted. With ``seed_sequence`` (the fold's
    "training" stream) curves come from its child 0; otherwise from ``default_rng(seed)``."""
    timing = timing or TimingSpread.published()
    rng = np.random.default_rng(child_sequence(seed_sequence, 0) if seed_sequence is not None else seed)
    return generate_training_curves(train_trials, n_curves, rng, switches, timing, shift)


def fit_fold_priors(fold_name: str, train_trials, n_curves: int, seed: int,
                    switches: FactorSwitches = FactorSwitches(),
                    timing: TimingSpread | None = None, workers: int = 1,
                    seed_sequence: np.random.SeedSequence | None = None,
                    return_training: bool = False, shift: ShiftSpread | None = None):
    """Both families' priors for a fold; ``(priors, training)`` with ``return_training``.

    With ``seed_sequence`` the training curves use its child 0 and the dual-Gaussian fits
    its child 1; ``seed`` is then only provenance. Without it, both use ``seed`` as before.
    The dual-Gaussian references follow ``shift``'s centre, for the fits and the design.
    """
    timing = timing or TimingSpread.published()
    shift = shift or ShiftSpread.default()
    training = fold_training_set(train_trials, n_curves, seed, switches, timing, seed_sequence, shift)
    dual_seed = child_sequence(seed_sequence, 1) if seed_sequence is not None else seed
    dual_kwargs = reference_centres(shift.centre_ms)
    switches_dict = asdict(switches)
    hashes = population_source_hashes()

    def _common() -> dict:
        # Fresh dict copies per family: unpacking one shared dict into both
        # DataPrior calls handed both families the *same* switches/source_hashes object,
        # so mutating one family's copy silently corrupted the other's.
        return dict(fold=fold_name, n_curves=n_curves, n_rejected_draws=training.n_rejected,
                    switches=dict(switches_dict), timing_source=timing.source, seed=seed,
                    source_hashes=dict(hashes), timing=timing.to_dict(), shift=shift.to_dict())

    basis = VonMisesBasis()
    mu, sigma, rms = fit_promp_prior(training.curves, basis)
    weights = _project_weights(training.curves, basis)
    promp_names = tuple(f"w{i}" for i in range(basis.n_basis))
    promp_bound = np.full(basis.n_basis, PROMP_DATA_BOUND_NM)
    priors = {"promp_s0": DataPrior(
        "promp_s0", mu=mu, sigma=sigma,
        n_fit_failures=int((rms > MAX_FIT_RMS_NM).sum()),
        fit_rms_median_nm=float(np.median(rms)),
        share_at_bound=share_at_bound(weights, -promp_bound, promp_bound, promp_names),
        **_common())}

    mu, sigma, rms, within, thetas = fit_dual_gauss_prior(training.curves, dual_seed, workers,
                                                          design_kwargs=dual_kwargs)
    priors["dual_gauss"] = DataPrior(
        "dual_gauss", mu=mu, sigma=sigma,
        n_fit_failures=int((~within).sum()),
        fit_rms_median_nm=float(np.median(rms)),
        share_at_bound=share_at_bound(thetas, LOW, HIGH, PARAM_NAMES),
        design_kwargs=dict(dual_kwargs), **_common())
    return (priors, training) if return_training else priors
