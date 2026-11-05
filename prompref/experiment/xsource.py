"""Source dependence of best-profile distances and best-profile levels. Extends test X (``prompref.experiment.xexpr``);
no session is run.

Sources (rows), 200 ideals each, device-clipped to +/-12 Nm: X's five (``xexpr.GENERATORS``), plus
- ``surrogate``: a phase-randomized surrogate (Theiler et al. 1992) of each population ideal. The
  DFT of the device-clipped ideal over the 101-point cycle keeps its DC term and the magnitudes of
  harmonics 1-50; their phases are drawn uniformly on [0, 2 pi), one seed per key, with conjugate
  symmetry (``irfft``); the curve is then device-clipped. Key ``<fold>/surrogateNNN`` for the parent
  ``<fold>/popNNN``, so every surrogate is paired with its parent ideal;
- ``schaefer_i``: Schaefer et al.'s study setting, the timings from X's Schaefer sampler
  (``xexpr.sample_schaefer``, uniform on the set their device allowed), both peaks at 12 Nm.

Representations (columns): X's eight, plus
- ``promp4``: the ProMP with K = 4 von Mises bases, linear least squares, device-clipped;
- ``schaefer_fixed``: Schaefer (i), prompref's ``SchaeferDesign`` (four timings, both peaks fixed at
  12 Nm) in X's RX9 timing box, the hull of Schaefer's device constraints, with its cycle-order
  repair; starts: Schaefer (ii)'s target-read initial guess (timings) and 20 draws of P's timing
  prior (``TimingPrior``);
- ``pchip_pub``: PCHIP (i), ``PchipDesign.published`` (Park et al. 2025 as published: five
  parameters in their box, uext 15 Nm, clipped by the device to 12 Nm).
Every fit is X's (``xexpr.floor_job``, X's seed rule), so a new cell is what X's code computes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..designs.pchip_park import PchipDesign
from ..designs.schaefer import SchaeferDesign
from ..designs.schaefer_peaks import SchaeferPeaksDesign
from ..population.curves import TORQUE_LIMIT_NM, physics_clip
from . import xexpr as X

NEW_SOURCES = ("surrogate", "schaefer_i")
SOURCES = X.GENERATORS + NEW_SOURCES
NEW_REPS = ("promp4", "schaefer_fixed", "pchip_pub")
#: The columns in the figure's order: interpretable families by parameter count, then the ProMP.
REPRESENTATIONS = ("schaefer_fixed", "schaefer_peaks", "trapezoid_surface", "franks", "pchip_pub",
                   "pchip", "promp4", "promp8", "promp12", "promp16", "promp24")
INTERPRETABLE = REPRESENTATIONS[:6]
N_PARAMS = {"schaefer_fixed": 4, "schaefer_peaks": 6, "trapezoid_surface": 6, "franks": 8,
            "pchip_pub": 5, "pchip": 6, "promp4": 4, "promp8": 8, "promp12": 12, "promp16": 16,
            "promp24": 24}
#: The 101-point cycle of the test plan: harmonics 1..50 of a 101-sample DFT.
N_CYCLE = 101
#: Schaefer (i)'s fixed peaks (their study), Nm.
FIXED_PEAK_NM = 12.0


# ---------------------------------------------------------------- the two new sources
def surrogate_key(parent: str) -> str:
    """``<fold>/popNNN`` -> ``<fold>/surrogateNNN``."""
    fold, uid = parent.split("/", 1)
    if not uid.startswith("pop"):
        raise ValueError(f"not a population key: {parent!r}")
    return f"{fold}/surrogate{uid[3:]}"


def phase_randomize(curve: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The 101-point DFT's DC term and harmonic magnitudes kept, phases of harmonics 1..50
    uniform on [0, 2 pi); ``irfft`` imposes the conjugate symmetry. Not clipped."""
    c = np.asarray(curve, dtype=float)
    if c.shape != (N_CYCLE,):
        raise ValueError(f"expected {N_CYCLE} samples, got {c.shape}")
    spec = np.fft.rfft(c)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=spec.size - 1)
    out = spec.copy()
    out[1:] = np.abs(spec[1:]) * np.exp(1j * phases)
    return np.fft.irfft(out, n=N_CYCLE)


def surrogate_ideal(parent_curve: np.ndarray, key: str) -> np.ndarray:
    """The device-clipped surrogate of one population ideal, seeded by its own key."""
    rng = np.random.default_rng(X._seed("xsource-surrogate", key))
    return physics_clip(phase_randomize(parent_curve, rng))


def schaefer_i_ideal(key: str) -> tuple[np.ndarray, np.ndarray]:
    """(parameters, device-clipped curve) of one Schaefer (i) published ideal: X's Schaefer
    sampler under this source's seed (``published_rng("schaefer_i", key)``), peaks set to 12 Nm
    (the sampler's peak draws are discarded)."""
    theta = X.sample_schaefer(X.published_rng("schaefer_i", key))
    theta[4:] = FIXED_PEAK_NM
    return theta, physics_clip(SchaeferPeaksDesign().curve(theta))


# ---------------------------------------------------------------- the new representations
class SchaeferFixedFit(SchaeferDesign):
    """Schaefer (i) for XS: four timings, both peaks at 12 Nm, in RX9's timing box (the hull of
    Schaefer's device constraints) with SchaeferDesign's cycle-order repair. It is the face of
    X's Schaefer (ii) fit box where both peaks sit at 12 Nm."""

    def __init__(self, prior: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        super().__init__(peak_ext_nm=FIXED_PEAK_NM, peak_flex_nm=FIXED_PEAK_NM, prior=prior)
        self.low = np.asarray(X.SCHAEFER_FIT_LOW[:4], dtype=float).copy()
        self.high = np.asarray(X.SCHAEFER_FIT_HIGH[:4], dtype=float).copy()

    def initial_guess(self, target: np.ndarray) -> np.ndarray:
        """Schaefer (ii)'s target-read guess in the same timing box, timings only."""
        wide = SchaeferPeaksDesign(low=X.SCHAEFER_FIT_LOW, high=X.SCHAEFER_FIT_HIGH)
        return self.project(wide.initial_guess(target)[:4])


def timing_prior_path(run: Path) -> Path:
    """P's Schaefer (i) prior inputs, anchored on the run as phase5_schaefer_floors.py does."""
    return Path(run).parent / "schaefer" / "prior_inputs.json"


def fold_designs(run: Path, fold: str, timing_prior=None) -> dict:
    """X's designs of the fold (RX9's fit boxes) plus Schaefer (i) and PCHIP (i)."""
    from ..priors.timing_prior import TimingPrior

    tp = timing_prior if timing_prior is not None else TimingPrior.load(timing_prior_path(run))
    designs = X.fold_designs(run, fold)
    designs["schaefer_fixed"] = SchaeferFixedFit(prior=(np.asarray(tp.mu, dtype=float),
                                                        np.asarray(tp.sigma, dtype=float)))
    designs["pchip_pub"] = PchipDesign.published()
    return designs


def new_cells() -> list[tuple[str, str]]:
    """The (source, representation) cells XS fits: the two new sources in every column, X's
    five sources in the three new columns. X's 40 cells are reused as stored."""
    cells = [(s, r) for s in NEW_SOURCES for r in REPRESENTATIONS]
    cells += [(s, r) for s in X.GENERATORS for r in NEW_REPS]
    return cells


# ---------------------------------------------------------------- baselines and best-profile levels
def rms_rows(curves: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per row, the RMS over the cycle of ``curves - reference``."""
    d = np.asarray(curves, dtype=float) - np.asarray(reference, dtype=float)
    return np.sqrt(np.mean(d ** 2, axis=-1))


def own_generic(curves: np.ndarray, folds: np.ndarray) -> dict:
    """Baseline (b): per fold, the device-clipped pointwise mean of the source's ideals in the
    other folds (leave one fold out, as the fold priors are)."""
    curves = np.asarray(curves, dtype=float)
    folds = np.asarray(folds)
    return {f: physics_clip(curves[folds != f].mean(axis=0)) for f in dict.fromkeys(folds)}


def ceiling(generic_nm, floor_nm, s: float = 1.0, lam: float = 0.1) -> np.ndarray:
    """C_i = lambda/2 + (1 - lambda) Phi((g_i - f_i)/s) (R2), per ideal."""
    from scipy.stats import norm

    return lam / 2.0 + (1.0 - lam) * norm.cdf((np.asarray(generic_nm, dtype=float)
                                               - np.asarray(floor_nm, dtype=float)) / float(s))


def endpoint_bound(curve: np.ndarray) -> float:
    """The smallest RMS error (Nm) any curve with equal first and last samples can reach at the
    two end samples of ``curve`` alone: |c_0 - c_100| / sqrt(2 * 101). Every representation here
    repeats its first sample at 100 %GC; a surrogate, periodic over 101 samples, need not."""
    c = np.asarray(curve, dtype=float)
    return float(abs(c[0] - c[-1]) / np.sqrt(2.0 * c.size))


__all__ = ["NEW_SOURCES", "SOURCES", "NEW_REPS", "REPRESENTATIONS", "INTERPRETABLE", "N_PARAMS",
           "TORQUE_LIMIT_NM", "surrogate_key", "phase_randomize", "surrogate_ideal",
           "schaefer_i_ideal", "SchaeferFixedFit", "fold_designs", "new_cells", "rms_rows",
           "own_generic", "ceiling", "endpoint_bound", "timing_prior_path"]
