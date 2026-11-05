"""Personal factors of a simulated user.

Every factor draws its random numbers whether or not it is switched on, so switching
one factor off never changes another factor's draws -- which is what makes the later
fixed-parameter ablations comparable, user by user.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from .curves import LANDMARKS

#: R21: lag of the preferred assistance behind the biological moment, ms.
SHIFT_CENTRE_MS = 100.0
SHIFT_SCALE_MS = 25.0

#: R22: Molinaro's assistance scale and the personal log-spread around it.
SCALE_CENTRE = 0.20
SCALE_LOG_SPREAD = 0.26

#: Tail parameter of the personal factors (Student-t).
FACTOR_NU = 3.0

#: R23 fallback. Schaefer et al. (2026) report only the t3 IQR (5.5 %ST) and that it is
#: 38-44 % smaller than the other three; those are set to the midpoint of
#: 5.5/(1-0.38) .. 5.5/(1-0.44). LANDMARKS order: t2 (extension peak), t3 (extension to
#: flexion), t4 (flexion peak), t1 (flexion to extension).
PUBLISHED_TIMING_IQR_PCT = (9.3, 5.5, 9.3, 9.3)

#: Paper order t1..t4 -> LANDMARKS order.
_PAPER_TO_LANDMARK = [1, 2, 3, 0]


@dataclass(frozen=True)
class FactorSwitches:
    shift: bool = True
    ext_scale: bool = True
    flex_scale: bool = True
    feature_timing: bool = True


@dataclass(frozen=True)
class ShiftSpread:
    """The personal lag of preferred assistance behind biology: ``centre + scale * t_nu`` ms.

    The defaults are 's assumption. Final review I5: a parameter object rather
    than module constants, so a Schaefer-centred population is a CLI flag, not a code fork
    (``load_schaefer_shift``).
    """

    centre_ms: float = SHIFT_CENTRE_MS
    scale_ms: float = SHIFT_SCALE_MS
    nu: float = FACTOR_NU
    source: str = "spec R21 assumption"

    @classmethod
    def default(cls) -> "ShiftSpread":
        return _default_shift_spread()

    def to_dict(self) -> dict:
        return dict(centre_ms=float(self.centre_ms), scale_ms=float(self.scale_ms), nu=float(self.nu),
                    source=str(self.source))


@lru_cache(maxsize=1)
def _default_shift_spread() -> ShiftSpread:
    return ShiftSpread()


@dataclass(frozen=True, eq=False)
class TimingSpread:
    """Multivariate Student-t over the four landmark offsets (%GC, LANDMARKS order)."""

    scale_matrix: np.ndarray
    nu: float
    source: str
    file_sha256: str | None = None

    @classmethod
    def published(cls) -> "TimingSpread":
        return _published_timing_spread()

    def to_dict(self) -> dict:
        """Provenance: the matrix and tail themselves, not only a path."""
        return dict(source=str(self.source), nu=float(self.nu),
                    scale_matrix=np.asarray(self.scale_matrix, dtype=float).tolist(),
                    file_sha256=self.file_sha256)


@lru_cache(maxsize=1)
def _published_timing_spread() -> TimingSpread:
    # A Student-t's IQR is 2 t_{0.75}(nu) scale units, so scale = IQR / (2 t_{0.75}).
    # Cached: computing this scipy.stats.t.ppf call inside every sample_draw(timing=None)
    # cost about 20 ms per 100 draws (~100 s over the 200,000-draw tests below).
    scale = np.asarray(PUBLISHED_TIMING_IQR_PCT) / (2.0 * student_t.ppf(0.75, FACTOR_NU))
    return TimingSpread(np.diag(scale ** 2), FACTOR_NU, "Schaefer et al. 2026, published IQRs")


def _finite_or_none(value) -> float | None:
    try:
        number = float(value)  # the analysis writes an unbounded value as the string "inf"
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _schaefer_nu(summary: dict) -> float:
    """``student_t_nu.mle`` when finite; otherwise ``recommended`` (11 people cannot identify
    the tail, so the MLE is "inf" and the analysis recommends a judgement value)."""
    info = summary.get("student_t_nu") or {}
    if not info:
        return FACTOR_NU
    for key in ("mle", "recommended"):
        nu = _finite_or_none(info.get(key))
        if nu is not None:
            return nu
    raise ValueError(f"student_t_nu has neither a finite mle nor a finite recommended value: {info}")


def load_schaefer_timing(path, residual_shift_sd_ms: float | None = None, stride_s: float | None = None,
                         min_sd_pct: float = 1.0) -> TimingSpread:
    """The joint timing spread from the Schaefer data analysis.

    See ``docs/prompts/2026-09-14-schaefer-data-analysis.md``. The covariance is
    ``summary.t_cov_recommended`` when present, else ``summary.t_cov``, in the paper's t1..t4
    order and %ST, used here as %GC. Its tail parameter is ``student_t_nu.mle`` when finite,
    else ``student_t_nu.recommended``.

    ``residual_shift_sd_ms`` removes a personal global shift that is modelled separately: Schaefer's landmark SDs already contain each
    person's global shift, so adding a shift factor on top would count it twice. The shift SD
    is converted to %GC at ``stride_s`` (default: the median participant stride time) and its
    square subtracted from every variance; each SD is floored at ``min_sd_pct``. That needs a
    diagonal matrix. The matrix is then reordered to LANDMARKS order and converted from
    covariance to Student-t scale by ``(nu - 2) / nu``.
    """
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    summary = data["summary"]
    key = "t_cov_recommended" if "t_cov_recommended" in summary else "t_cov"
    cov = np.asarray(summary[key], dtype=float)
    if cov.shape != (len(LANDMARKS), len(LANDMARKS)):
        raise ValueError(f"{key} must be 4x4, got {cov.shape}")
    nu = _schaefer_nu(summary)
    residual = "residual shift not applied"
    if residual_shift_sd_ms is not None:
        if np.any(cov[~np.eye(len(LANDMARKS), dtype=bool)] != 0.0):
            raise ValueError(f"removing a residual shift needs a diagonal covariance; {key} is not diagonal")
        if stride_s is None:
            stride_s = float(np.median([p["stride_time_s"] for p in data["participants"]]))
        shift_pct = float(residual_shift_sd_ms) / (float(stride_s) * 1000.0) * 100.0
        sd = np.sqrt(np.clip(np.diag(cov) - shift_pct ** 2, 0.0, None))
        cov = np.diag(np.maximum(sd, float(min_sd_pct)) ** 2)
        residual = (f"residual after a {float(residual_shift_sd_ms):g} ms shift SD at stride "
                    f"{float(stride_s):.3f} s ({shift_pct:.2f} %GC), SD floor {float(min_sd_pct):g} %GC")
    order = _PAPER_TO_LANDMARK
    cov = cov[np.ix_(order, order)]
    sha = hashlib.sha256(raw).hexdigest()
    detail = f"{Path(path)} ({key}, nu {nu:g}, {residual})"
    # A Student-t's covariance is nu / (nu - 2) times its scale matrix. For nu <= 2 no
    # covariance exists; the reported matrix is then used as the scale, and the source says so.
    if nu > 2.0:
        return TimingSpread(cov * (nu - 2.0) / nu, nu, f"Schaefer data analysis: {detail}", sha)
    return TimingSpread(cov, nu, f"Schaefer data analysis (nu <= 2, covariance used as scale): {detail}", sha)


def load_schaefer_shift(path) -> ShiftSpread:
    """The personal shift from the Schaefer global cross-correlation lag against biology.

    Centre ``summary.shift_vs_bio.global.mean`` (ms); tail as in :func:`load_schaefer_timing`;
    the between-person SD converted to a Student-t scale, ``sd * sqrt((nu - 2) / nu)``.
    """
    summary = json.loads(Path(path).read_text(encoding="utf-8"))["summary"]
    lag = summary["shift_vs_bio"]["global"]
    nu = _schaefer_nu(summary)
    if nu <= 2.0:
        raise ValueError(f"a shift scale needs nu > 2 to convert an SD, got {nu}")
    return ShiftSpread(float(lag["mean"]), float(lag["sd"]) * float(np.sqrt((nu - 2.0) / nu)), nu,
                       str(Path(path)))


@dataclass(frozen=True)
class FactorDraw:
    shift_ms: float
    ext_scale: float
    flex_scale: float
    landmark_offsets: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)


NEUTRAL_DRAW = FactorDraw(SHIFT_CENTRE_MS, SCALE_CENTRE, SCALE_CENTRE, (0.0, 0.0, 0.0, 0.0))


def neutral_draw(shift: ShiftSpread | None = None) -> FactorDraw:
    """Every factor at its centre, the shift at ``shift``'s centre (``NEUTRAL_DRAW`` by default)."""
    if shift is None:
        return NEUTRAL_DRAW
    return FactorDraw(float(shift.centre_ms), SCALE_CENTRE, SCALE_CENTRE, (0.0, 0.0, 0.0, 0.0))


def sample_draw(rng: np.random.Generator, switches: FactorSwitches = FactorSwitches(),
                timing: TimingSpread | None = None, shift: ShiftSpread | None = None) -> FactorDraw:
    """One user's factors. With the shift switched off, the lag is ``shift.centre_ms``.

    The default ``shift`` (R21) reproduces the draws from before ``ShiftSpread`` existed
    exactly: same random calls in the same order, same arithmetic.
    """
    timing = timing or TimingSpread.published()
    spread = shift or ShiftSpread.default()
    t_shift = rng.standard_t(spread.nu)
    t_ext = rng.standard_t(FACTOR_NU)
    t_flex = rng.standard_t(FACTOR_NU)
    normal = rng.standard_normal(len(LANDMARKS))
    chi2 = rng.chisquare(timing.nu)

    shift = spread.centre_ms + (spread.scale_ms * t_shift if switches.shift else 0.0)
    ext = SCALE_CENTRE * float(np.exp(SCALE_LOG_SPREAD * t_ext)) if switches.ext_scale else SCALE_CENTRE
    flex = (SCALE_CENTRE * float(np.exp(SCALE_LOG_SPREAD * t_flex)) if switches.flex_scale
            else SCALE_CENTRE)
    if switches.feature_timing:
        jitter = 1e-9 * np.trace(timing.scale_matrix) * np.eye(len(LANDMARKS))
        chol = np.linalg.cholesky(timing.scale_matrix + jitter)
        offsets = (chol @ normal) * np.sqrt(timing.nu / chi2)
    else:
        offsets = np.zeros(len(LANDMARKS))
    return FactorDraw(float(shift), float(ext), float(flex), tuple(float(o) for o in offsets))
