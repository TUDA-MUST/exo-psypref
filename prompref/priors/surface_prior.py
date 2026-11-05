"""Priors of the projection surfaces added for test P.

Each surface family PsyPref's proposals can be projected onto needs, beside its box, a Gaussian
prior: the projection's fitter starts one of its three fits at the prior mean,
and the floors (``prompref.population.users.user_floors``) draw their starts from it and
tie-break toward it. They offer the ``design`` / ``whitening`` / ``space(radius)`` interface
of ``FranksPrior`` and ``TimingPrior``, so ``arm_space`` takes them as it takes those.

- ``schaefer_peaks``, Schaefer et al.'s family with free peaks (P, family (ii)): the four
  timings' prior is family (i)'s, the distribution of the timings Schaefer's 11 participants
  preferred (``out/schaefer/prior_inputs.json``, ``TimingPrior``). Schaefer's data cannot inform
  the peaks (their peaks were fixed at 12 Nm; Schaefer report section 7), so the two peak
  magnitudes' mean and covariance come from the fold's population, as the fold's ProMP data prior
  encodes it: per draw of ``n_draws`` device-clipped ProMP prior curves, the extension peak
  max(0, -min) and the flexion peak max(0, max), each at most 12 Nm. Timings and peaks are
  independent in the prior (block-diagonal covariance).
- ``trapezoid_surface``, Ramella et al.'s semi-trapezoid: the moment-matched Gaussian of its own
  box (``flat_whitening``), the whitening C1's floors were computed with. Ramella's box is the
  family as published; no population knowledge is added.

``prompref/priors/data_prior.py`` and ``prompref/population/`` are not touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..designs.pchip_park import PUBLISHED_HIGH, PUBLISHED_LOW, PchipDesign, bout_width
from ..designs.schaefer_peaks import PEAK_HIGH, SchaeferPeaksDesign, representative
from ..designs.trapezoid_surface import TrapezoidSurface
from ..population.curves import physics_clip
from ..prior import Whitening

#: Families whose box is stored with their prior.
BOXED = ("pchip",)

#: Surface family name -> its design class (each takes ``prior=(mu, sigma)``). The PCHIP family
#: (``pchip``) also carries its box per fold, so it is built from ``low`` and ``high``.
SURFACE_DESIGNS = {"schaefer_peaks": SchaeferPeaksDesign, "trapezoid_surface": TrapezoidSurface,
                   "pchip": None}
N_PARAMS = {"schaefer_peaks": 6, "trapezoid_surface": 6, "pchip": 6}


@dataclass
class SurfacePrior:
    family: str
    mu: np.ndarray
    sigma: np.ndarray
    fold: str = ""
    source: str = ""
    stats: dict | None = field(default=None)
    #: The box, for a family whose box is set per fold (``pchip``); ``None`` otherwise.
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.family not in SURFACE_DESIGNS:
            raise ValueError(f"family must be one of {tuple(SURFACE_DESIGNS)}, got {self.family!r}")
        self.mu = np.asarray(self.mu, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        n = N_PARAMS[self.family]
        if self.mu.shape != (n,) or self.sigma.shape != (n, n):
            raise ValueError(f"{self.family}: expected mu ({n},) and sigma ({n}, {n}), got "
                             f"{self.mu.shape} and {self.sigma.shape}")
        if (self.low is None) != (self.family not in BOXED):
            raise ValueError(f"{self.family}: a box is given exactly for {BOXED}")
        if self.low is not None:
            self.low = np.asarray(self.low, dtype=float)
            self.high = np.asarray(self.high, dtype=float)

    def design(self):
        if self.family == "pchip":
            return PchipDesign(self.low, self.high, prior=(self.mu, self.sigma))
        return SURFACE_DESIGNS[self.family](prior=(self.mu, self.sigma))

    def whitening(self) -> Whitening:
        return Whitening(self.mu, self.sigma, self.mu.size)

    def space(self, radius: float):
        from ..learning.space import LearnerSpace

        return LearnerSpace(self.design(), self.whitening(), float(radius))

    def save(self, path) -> None:
        payload = dict(family=self.family, mu=self.mu.tolist(), sigma=self.sigma.tolist(),
                       fold=self.fold, source=self.source, stats=self.stats)
        if self.low is not None:
            payload.update(low=self.low.tolist(), high=self.high.tolist())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "SurfacePrior":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(p["family"], p["mu"], p["sigma"], p.get("fold", ""), p.get("source", ""),
                   p.get("stats"), p.get("low"), p.get("high"))


def surface_prior_path(run_dir, fold: str, family: str) -> Path:
    return Path(run_dir) / "priors" / f"{fold}-{family}.json"


def peak_moments(promp_prior, n_draws: int = 2000, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of (extension peak, flexion peak) over ``n_draws`` device-clipped
    curves of the fold's ProMP data prior, each peak in [0, 12] Nm."""
    rng = np.random.default_rng(seed)
    design = promp_prior.design()
    curves = np.array([physics_clip(design.curve(w)) for w in design.sample_prior(rng, int(n_draws))])
    peaks = np.stack([np.clip(-curves.min(axis=1), 0.0, PEAK_HIGH),
                      np.clip(curves.max(axis=1), 0.0, PEAK_HIGH)], axis=1)
    return peaks.mean(axis=0), np.cov(peaks, rowvar=False)


def schaefer_peaks_prior(promp_prior, timing_prior, fold: str, *, n_draws: int = 2000,
                         seed=0) -> SurfacePrior:
    peak_mu, peak_cov = peak_moments(promp_prior, n_draws, seed)
    mu = np.concatenate([timing_prior.mu, peak_mu])
    sigma = np.zeros((6, 6))
    sigma[:4, :4] = timing_prior.sigma
    sigma[4:, 4:] = peak_cov
    stats = dict(n_draws=int(n_draws), peak_mean_nm=peak_mu.tolist(),
                 peak_sd_nm=np.sqrt(np.diag(peak_cov)).tolist(),
                 peak_corr=float(peak_cov[0, 1] / np.sqrt(peak_cov[0, 0] * peak_cov[1, 1])))
    return SurfacePrior("schaefer_peaks", mu, sigma, fold=fold,
                        source=f"timings: {timing_prior.source}; peaks: {n_draws} ProMP prior draws",
                        stats=stats)


def trapezoid_surface_prior(fold: str = "") -> SurfacePrior:
    """The moment-matched Gaussian of Ramella's box: mean at the centre, variance range^2 / 12
    (``prompref.experiment.phase5.flat_whitening``)."""
    design = TrapezoidSurface()
    low, high = np.asarray(design.low, dtype=float), np.asarray(design.high, dtype=float)
    return SurfacePrior("trapezoid_surface", 0.5 * (low + high), np.diag((high - low) ** 2 / 12.0),
                        fold=fold, source="moment-matched Gaussian of Ramella's box")


#: Percentiles of the prior draws' features the PCHIP box is extended to cover (Franks' L6 range).
PCHIP_COVER = (0.5, 99.5)


def pchip_features(curves) -> np.ndarray:
    """Per device-clipped curve, Park et al.'s six quantities read off the curve, in their order
    (tflexp, tflexd, textp, textd, uflex, uext): the flexion and extension peak times (argmax and
    argmin; the extension one in [-50, 50), the flexion one in the cycle after it), the widths of
    the positive and negative regions around them, and the peak magnitudes, at most 12 Nm."""
    out = []
    for curve in np.asarray(curves, dtype=float):
        c = curve[:-1]
        i_ext, i_flex = int(np.argmin(c)), int(np.argmax(c))
        textp = representative(float(i_ext), -50.0, 49.999)
        tflexp = float(i_flex) if float(i_flex) > textp else float(i_flex) + 100.0
        if tflexp >= textp + 100.0:
            tflexp -= 100.0
        out.append([tflexp, bout_width(c, i_flex, 1.0), textp, bout_width(c, i_ext, -1.0),
                    min(max(float(c[i_flex]), 0.0), PEAK_HIGH),
                    min(max(-float(c[i_ext]), 0.0), PEAK_HIGH)])
    return np.array(out)


#: a bout is readable when its region is 4-95 %GC wide and its peak at least 0.5 Nm.
READABLE_WIDTH = (4.0, 95.0)
READABLE_PEAK_NM = 0.5


def two_bouts_readable(feats) -> np.ndarray:
    """Per row of :func:`pchip_features`, whether both bouts can be read off the curve."""
    f = np.atleast_2d(np.asarray(feats, dtype=float))
    widths, peaks = f[:, [1, 3]], f[:, [4, 5]]
    return (np.all((widths >= READABLE_WIDTH[0]) & (widths <= READABLE_WIDTH[1]), axis=1)
            & np.all(peaks >= READABLE_PEAK_NM, axis=1))


def pchip_prior(promp_prior, fold: str, *, n_draws: int = 2000, seed=0) -> SurfacePrior:
    """Park et al.'s PCHIP family adapted to level walking: the six
    parameters with uext free; the published timing and duration ranges (p. 2151) extended, never
    narrowed, to cover the 0.5-99.5th percentiles of the same quantities over ``n_draws`` device-
    clipped draws of the fold's ProMP data prior; both magnitudes in [0, 12] Nm (the device limit).
    The prior is the mean and covariance of those quantities, clipped into the box."""
    rng = np.random.default_rng(seed)
    design = promp_prior.design()
    curves = [physics_clip(design.curve(w)) for w in design.sample_prior(rng, int(n_draws))]
    all_feats = pchip_features(curves)
    # quantities are read only off draws in which both bouts are readable.
    readable = two_bouts_readable(all_feats)
    feats = all_feats[readable]
    lo_q, hi_q = np.percentile(feats, PCHIP_COVER, axis=0)
    low = np.minimum(np.r_[PUBLISHED_LOW[:4], 0.0, 0.0], np.r_[lo_q[:4], 0.0, 0.0])
    high = np.maximum(np.r_[PUBLISHED_HIGH[:4], PEAK_HIGH, PEAK_HIGH],
                      np.r_[hi_q[:4], PEAK_HIGH, PEAK_HIGH])
    high[4:] = PEAK_HIGH
    # The peaks must keep their order (textp < tflexp < textp + 100) over the whole box. Where the
    # extended ranges would let them cross, they meet 1 %GC apart at their midpoint, held outside
    # the published ranges, which are never narrowed (R13a).
    split = []
    if high[2] >= low[0]:
        mid = float(np.clip(0.5 * (high[2] + low[0]), PUBLISHED_HIGH[2] + 0.5,
                            PUBLISHED_LOW[0] - 0.5))
        high[2], low[0] = mid - 0.5, mid + 0.5
        split.append("textp high / tflexp low")
    if high[0] >= low[2] + 100.0:
        mid = float(np.clip(0.5 * (high[0] + low[2] + 100.0), PUBLISHED_HIGH[0] + 0.5,
                            PUBLISHED_LOW[2] + 100.0 - 0.5))
        high[0], low[2] = mid - 0.5, mid + 0.5 - 100.0
        split.append("tflexp high / textp low + 100")
    if np.any(low[:4] > PUBLISHED_LOW[:4]) or np.any(high[:4] < PUBLISHED_HIGH[:4]):
        raise ValueError(f"{fold}: the PCHIP box narrows a published range: {low} {high}")
    clipped = np.clip(feats, low, high)
    stats = dict(n_draws=int(n_draws), n_readable=int(readable.sum()), cover=list(PCHIP_COVER),
                 feature_percentiles={name: [float(a), float(b)] for name, a, b in
                                      zip(("tflexp", "tflexd", "textp", "textd", "uflex", "uext"),
                                          lo_q, hi_q)},
                 extended_low=[bool(a) for a in (low[:4] < PUBLISHED_LOW[:4])],
                 extended_high=[bool(a) for a in (high[:4] > PUBLISHED_HIGH[:4])],
                 split=split)
    return SurfacePrior("pchip", clipped.mean(axis=0),
                        np.cov(clipped, rowvar=False) + 1e-9 * np.eye(6), fold=fold,
                        source=f"Park et al. 2025 box extended to {n_draws} ProMP prior draws",
                        stats=stats, low=low, high=high)
