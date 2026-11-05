"""Cross-expressiveness: how well each representation fits ideals from each generator. No session is run.

Generators (rows), 200 ideals each, every one device-clipped (R29, +/-12 Nm):
- ``population``: the run's 200 population users' ideals;
- ``schaefer``: Schaefer et al. (2026)'s four-half-cosine profile with free peaks, the timings
  uniform within their device constraints (npj Biomed. Innov., PDF p. 8, Methods, user-tuning:
  "each distance between two adjacent control parameters had to be in the interval of 10%ST to
  40%ST. The extension and flexion peak timings were limited to the first and second half of the
  gait cycle"), both peak magnitudes uniform in [0, 12] Nm (the device limit);
- ``franks``: Franks et al. (2021). Their parameter ranges are in their Supplementary Material,
  Section 5, which could not be retrieved; assumption: uniform in the Franks family's learner box
  of Phase 5d, the fold's box in internal coordinates, derived from fits to that fold's
  ProMP prior draws, so this generator is population-informed;
- ``ramella``: Ramella et al. (2025), Table I (PDF p. 2): peak torques 5-8 Nm, extension peak time
  10-20 %GC, flexion peak time 55-65 %GC, rise times 10-20 %GC; uniform, not rounded;
- ``park``: Park et al. (2025), p. 2151: tflexp 65-92.5 %, tflexd 20-47.5 %, textp 15-45 %,
  textd 20-60 %, uflex 6-13 Nm, uniform; uext fixed at 15 Nm (p. 2149), so the device clips the
  extension part to 12 Nm.

Ideals of a family generator are assigned to the folds 40 at a time, in fold order, so every ideal
has a fold whose priors and boxes the representations use. Each ideal's draw has its own seed,
``published_rng(generator, key)``, from its key ("<fold>/<generator>NNN"), so any subset can be
regenerated alone (the published-range samplers are reused by later cells).

Representations (columns): the ProMP with K = 8, 12, 16 and 24 von Mises bases, fitted by linear
least squares on the basis without the prior, then device-clipped; and the interpretable families
Franks (the fold's Phase 5d box), Schaefer (ii) (``schaefer_peaks``), the semi-trapezoid
(``TrapezoidSurface``, C1's box) and PCHIP (ii) (``pchip``, the fold's level-walking box), each by
the best-profile distances' bounded multi-start fit (``physics_floor``: L-BFGS-B on the device-clipped mean squared
error, the family's initial guess plus 20 prior draws).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from preftune.profile import FEATURE_HIGH, FEATURE_LOW

from ..basis import VonMisesBasis
from ..designs.pchip_park import PUBLISHED_HIGH as PARK_HIGH
from ..designs.pchip_park import PUBLISHED_LOW as PARK_LOW
from ..designs.pchip_park import PchipDesign
from ..designs.schaefer_peaks import SchaeferPeaksDesign
from ..designs.trapezoid_surface import TrapezoidSurface
from ..population.curves import TORQUE_LIMIT_NM, physics_clip
from ..population.users import physics_floor

GENERATORS = ("population", "schaefer", "franks", "ramella", "park")
FAMILY_GENERATORS = GENERATORS[1:]
PROMP_K = (8, 12, 16, 24)
FAMILY_REPS = ("franks", "schaefer_peaks", "trapezoid_surface", "pchip")
REPRESENTATIONS = tuple(f"promp{k}" for k in PROMP_K) + FAMILY_REPS
N_PER_FOLD = 40
N_STARTS = 20
#: Schaefer et al.'s device constraints (PDF p. 8): adjacent timings 10-40 %ST apart, the extension
#: peak (t2) in the first half of the cycle and the flexion peak (t4) in the second.
SCHAEFER_GAP = (10.0, 40.0)


def _seed(*parts: str) -> np.random.SeedSequence:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).digest()
    return np.random.SeedSequence(int.from_bytes(digest[:8], "big"))


def published_rng(generator: str, key: str) -> np.random.Generator:
    """The fixed seed rule of the published-range samplers: one generator per (generator, key)."""
    return np.random.default_rng(_seed("published-ideal", generator, key))


# ---------------------------------------------------------------- the published-range samplers
def sample_schaefer(rng: np.random.Generator) -> np.ndarray:
    """One Schaefer (ii) parameter vector (t1, t2, t3, t4, ext_peak, flex_peak): timings uniform on
    the set their device allowed, by rejection from t2 ~ U[0, 50), t4 ~ U[50, 100) and the two
    gaps around t2 ~ U[10, 40] (a unit-Jacobian map of the timings, so uniform in them); peaks
    U[0, 12] Nm."""
    lo, hi = SCHAEFER_GAP
    while True:
        t2, t4 = rng.uniform(0.0, 50.0), rng.uniform(50.0, 100.0)
        g1, g2 = rng.uniform(lo, hi), rng.uniform(lo, hi)
        t1, t3 = t2 - g1, t2 + g2
        g3, g4 = t4 - t3, t1 + 100.0 - t4
        if lo <= g3 <= hi and lo <= g4 <= hi:
            peaks = rng.uniform(0.0, TORQUE_LIMIT_NM, size=2)
            return np.array([t1 % 100.0, t2, t3, t4, peaks[0], peaks[1]])


def sample_franks(rng: np.random.Generator, franks_prior) -> np.ndarray:
    """One Franks internal-coordinate vector, uniform in the fold's Phase 5d learner box
    (the stated assumption; Franks' own ranges are in their unretrieved supplement)."""
    return rng.uniform(franks_prior.low, franks_prior.high)


def sample_ramella(rng: np.random.Generator) -> np.ndarray:
    """Ramella et al.'s six features, uniform in their Table I ranges."""
    return rng.uniform(FEATURE_LOW, FEATURE_HIGH)


def sample_park(rng: np.random.Generator) -> np.ndarray:
    """Park et al.'s five optimized parameters, uniform in their published box (uext fixed)."""
    return rng.uniform(PARK_LOW, PARK_HIGH)


def published_ideal(generator: str, key: str, franks_prior=None) -> tuple[np.ndarray, np.ndarray]:
    """(parameters, device-clipped ideal curve) of one ideal from a family generator."""
    rng = published_rng(generator, key)
    if generator == "schaefer":
        theta = sample_schaefer(rng)
        curve = SchaeferPeaksDesign().curve(theta)
    elif generator == "franks":
        if franks_prior is None:
            raise ValueError("the franks generator needs the fold's FranksPrior")
        theta = sample_franks(rng, franks_prior)
        curve = franks_prior.design().curve(theta)
    elif generator == "ramella":
        theta = sample_ramella(rng)
        curve = TrapezoidSurface().curve(theta)
    elif generator == "park":
        theta = sample_park(rng)
        curve = PchipDesign.published().curve(theta)
    else:
        raise ValueError(f"generator must be one of {FAMILY_GENERATORS}, got {generator!r}")
    return theta, physics_clip(curve)


def ideal_key(fold: str, generator: str, index: int) -> str:
    return f"{fold}/{generator}{index:03d}"


# ---------------------------------------------------------------- representations
def promp_floor(ideal, n_basis: int) -> float:
    """Linear least squares on the K von Mises bases (no prior), device-clipped: RMS in Nm."""
    basis = VonMisesBasis(int(n_basis)).design
    w, *_ = np.linalg.lstsq(basis, np.asarray(ideal, dtype=float), rcond=None)
    return float(np.sqrt(np.mean((physics_clip(basis @ w) - ideal) ** 2)))


def family_floor(design, ideal, seed: np.random.SeedSequence) -> float:
    """The best-profile distances' bounded multi-start fit (``physics_floor``), its value in Nm (no tie-break: the
    value is the multi-start minimum either way)."""
    floor, _ = physics_floor(design, np.asarray(ideal, dtype=float), n_starts=N_STARTS, seed=seed)
    return float(floor)


def floor_job(job) -> dict:
    """One (ideal, representation) fit; ``job`` = (key, generator, representation, ideal, design)."""
    key, generator, rep, ideal, design = job
    if rep.startswith("promp"):
        value = promp_floor(ideal, int(rep[5:]))
    else:
        value = family_floor(design, ideal, _seed("xexpr-floor", generator, rep, key))
    return dict(key=key, generator=generator, representation=rep, floor_nm=value,
                ideal_rms=float(np.sqrt(np.mean(np.asarray(ideal, dtype=float) ** 2))))


#: (after the self-fit check, before the rerun): X's fit boxes cover each family's own
#: generator's published ranges. Schaefer (ii): the hull of Schaefer's device constraints (t1, the
#: zero crossing before extension, as t2 - 10..40 %ST around the cycle; t2 in the first half; t3 as
#: t2 + 10..40; t4 in the second half), peaks 0-12 Nm. PCHIP (ii): magnitudes 0-15 Nm (Park's uext
#: 15 Nm and uflex up to 13 Nm; the device clips at 12), timings as the fold's box (which already
#: covers Park's published ranges). Franks and the semi-trapezoid keep their boxes: their
#: generators sample inside them.
SCHAEFER_FIT_LOW = np.array([60.0, 0.0, 10.0, 50.0, 0.0, 0.0])
SCHAEFER_FIT_HIGH = np.array([140.0, 50.0, 90.0, 100.0, TORQUE_LIMIT_NM, TORQUE_LIMIT_NM])
PCHIP_FIT_PEAK_HIGH = 15.0


def fold_designs(run: Path, fold: str, fit_boxes: bool = True) -> dict:
    """The fold's representation designs for the interpretable families, in X's fit boxes."""
    from ..priors.franks_prior import FranksPrior
    from ..priors.surface_prior import SurfacePrior, surface_prior_path

    schaefer = SurfacePrior.load(surface_prior_path(run, fold, "schaefer_peaks")).design()
    pchip = SurfacePrior.load(surface_prior_path(run, fold, "pchip")).design()
    if fit_boxes:
        schaefer = SchaeferPeaksDesign(prior=schaefer.prior, low=SCHAEFER_FIT_LOW,
                                       high=SCHAEFER_FIT_HIGH)
        pchip = PchipDesign(np.r_[pchip.low[:4], 0.0, 0.0],
                            np.r_[pchip.high[:4], PCHIP_FIT_PEAK_HIGH, PCHIP_FIT_PEAK_HIGH],
                            prior=pchip.prior)
    return {"franks": FranksPrior.load(Path(run) / "priors" / f"{fold}-franks").design(),
            "schaefer_peaks": schaefer, "trapezoid_surface": TrapezoidSurface(), "pchip": pchip}
