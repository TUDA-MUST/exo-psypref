"""The PCHIP hip assistance profile of Park, An, Lee, Kang & Young (2025), as a Design.

Park et al., "Human-in-the-Loop Optimization of Hip Exoskeleton Assistance During Stair Climbing",
IEEE TBME 72(7):2147-2156, 2025 (doi 10.1109/TBME.2025.3536516); our design notes,
test P, as corrected at 4509e78 before any PCHIP code or run.

What the paper states (journal pages):
- p. 2149: the gait phase is the time since the last ipsilateral foot contact over the mean of the
  last five strides (0 % at foot contact, as here); "five parameters": tflexp, tflexd (peak timing
  and duration of flexion assistance), textp, textd (the same for extension), uflex (peak flexion
  magnitude, Nm); uext, the peak extension magnitude, "set to the maximum value of 15 Nm", the
  hardware limit, because stair ascent demands more extension than the device gives.
- p. 2150, eq. (1): T(x) = PCHIP(t, u, x) with t = [tflexp - tflexd/2, tflexp, tflexp + tflexd/2,
  textp - textd/2, textp, textp + textd/2] and u = [0, uflex, 0, 0, uext, 0]: two parts, each
  symmetric about its peak, through six nodes.
- p. 2151: bounds, from stair-climbing pilots: tflexp 65-92.5 %, tflexd 20-47.5 %, textp 15-45 %,
  textd 20-60 % of the gait cycle, uflex 6-13 Nm; uext fixed at 15 Nm.

Implementation:
- One scipy ``PchipInterpolator`` through the six nodes sorted in time (eq. (1) lists the flexion
  nodes first; a sorting interpolator such as MATLAB's ``pchip`` puts them in time order), made
  periodic over the cycle by a zero node at the previous flexion end and one at the next extension
  onset. Torque is zero outside the parts, and every part starts and ends with zero slope (the
  PCHIP derivative at a node next to a zero segment). Torque is flexion positive here, so the
  extension part enters as -uext.
- The paper leaves overlapping parts undefined (eq. (1) needs increasing nodes, and the published
  box allows overlap). Where the two parts would overlap on either side of the cycle, both
  half-durations are scaled down by one common factor until the parts touch; a touching pair of
  zero nodes is merged. ``project`` clips to the box only, so the reported parameters are the ones
  set; the curve uses the effective durations.

Two variants:
- ``PchipDesign.published``: Park's five parameters and box, uext fixed at 15 Nm (the device
  clips it to 12 Nm; R29), as published for stair climbing;
- ``PchipDesign(low, high)``: the six parameters with uext free, in a box per fold (the level-walking
  adaptation of ``prompref.priors.surface_prior.pchip_prior``).

Opt-in: nothing that existed before this module reads it.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .base import BoxDesign
from .schaefer_peaks import representative

PHASE = np.linspace(0.0, 100.0, 101)
#: Park et al.'s parameter order (p. 2149), with uext last.
PARAM_NAMES = ("tflexp", "tflexd", "textp", "textd", "uflex", "uext")
#: The published box (p. 2151) of the five optimized parameters, and the fixed uext (p. 2149).
PUBLISHED_LOW = np.array([65.0, 20.0, 15.0, 20.0, 6.0])
PUBLISHED_HIGH = np.array([92.5, 47.5, 45.0, 60.0, 13.0])
PUBLISHED_UEXT = 15.0
#: Two nodes closer than this (%GC) are one node.
MERGE_TOL = 1e-9


def effective_durations(tflexp, tflexd, textp, textd) -> tuple[float, float]:
    """The durations the curve uses: both scaled by one factor where the parts would overlap,
    so that they touch (the extension part precedes the flexion part, which precedes the next
    cycle's extension part). Needs textp < tflexp < textp + 100."""
    gap_1 = tflexp - textp
    gap_2 = textp + 100.0 - tflexp
    if gap_1 <= 0.0 or gap_2 <= 0.0:
        raise ValueError(f"peaks out of order: textp {textp}, tflexp {tflexp}")
    half = 0.5 * (textd + tflexd)
    factor = min(1.0, gap_1 / half, gap_2 / half)
    return tflexd * factor, textd * factor


def nodes(theta6) -> tuple[np.ndarray, np.ndarray, float]:
    """Periodic nodes from the previous flexion end to the next extension onset, in time order,
    and the extension onset, where the curve's one-cycle window starts."""
    tflexp, tflexd, textp, textd, uflex, uext = (float(v) for v in theta6)
    tflexd, textd = effective_durations(tflexp, tflexd, textp, textd)
    e_on, e_off = textp - textd / 2.0, textp + textd / 2.0
    f_on, f_off = tflexp - tflexd / 2.0, tflexp + tflexd / 2.0
    x = [f_off - 100.0, e_on, textp, e_off, f_on, tflexp, f_off, e_on + 100.0]
    y = [0.0, 0.0, -uext, 0.0, 0.0, uflex, 0.0, 0.0]
    xs, ys = [x[0]], [y[0]]
    for xi, yi in zip(x[1:], y[1:]):
        if xi - xs[-1] <= MERGE_TOL:
            continue          # a touching pair of zero nodes (or the closing anchor): one node
        xs.append(xi)
        ys.append(yi)
    return np.asarray(xs), np.asarray(ys), e_on


class PchipDesign(BoxDesign):
    """Park et al.'s two-part PCHIP profile; six parameters, or five with uext fixed."""

    def __init__(self, low, high, prior=None, fixed_uext: float | None = None) -> None:
        super().__init__(np.asarray(low, dtype=float), np.asarray(high, dtype=float))
        self.fixed_uext = None if fixed_uext is None else float(fixed_uext)
        want = 5 if self.fixed_uext is not None else 6
        if self.low.size != want:
            raise ValueError(f"expected {want} bounds, got {self.low.size}")
        if not (self.high[2] < self.low[0] and self.high[0] < self.low[2] + 100.0):
            raise ValueError("the box must keep textp < tflexp < textp + 100")
        self.prior = prior
        self.param_names = PARAM_NAMES[:want]

    @classmethod
    def published(cls) -> "PchipDesign":
        return cls(PUBLISHED_LOW, PUBLISHED_HIGH, fixed_uext=PUBLISHED_UEXT)

    def full(self, theta) -> np.ndarray:
        th = np.asarray(theta, dtype=float)
        return th if self.fixed_uext is None else np.append(th, self.fixed_uext)

    def curve(self, theta: np.ndarray) -> np.ndarray:
        x, y, onset = nodes(self.full(theta))
        local = np.clip(np.mod(PHASE - onset, 100.0) + onset, x[0], x[-1])
        return PchipInterpolator(x, y)(local)

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """With ``prior``: N(mu, sigma) clipped into the box; without: uniform in the box."""
        if self.prior is None:
            return rng.uniform(self.low, self.high, size=(int(n), self.n_params))
        mu, sigma = self.prior
        return self.project(rng.multivariate_normal(mu, sigma, size=int(n)))

    def initial_guess(self, target: np.ndarray) -> np.ndarray:
        """Peaks and their timings read off the target's minimum and maximum; each duration the
        width of the region of the target's sign around that peak; clipped into the box."""
        c = np.asarray(target, dtype=float)[:-1]
        i_ext, i_flex = int(np.argmin(c)), int(np.argmax(c))
        textp = representative(float(PHASE[i_ext]), self.low[2], self.high[2])
        tflexp = representative(float(PHASE[i_flex]), self.low[0], self.high[0])
        tflexd, textd = part_width(c, i_flex, 1.0), part_width(c, i_ext, -1.0)
        # Start 2 % clear of touching parts: where they overlap, the curve is flat in the sum of
        # the durations, and a least-squares start there stays there.
        half = 0.5 * (tflexd + textd)
        if half > 0.0 and tflexp > textp:
            scale = min(1.0, 0.98 * (tflexp - textp) / half,
                        0.98 * (textp + 100.0 - tflexp) / half)
            tflexd, textd = tflexd * scale, textd * scale
        guess = [tflexp, tflexd, textp, textd, max(float(c[i_flex]), 0.0)]
        if self.fixed_uext is None:
            guess.append(max(-float(c[i_ext]), 0.0))
        return self.project(np.array(guess))


def part_width(c: np.ndarray, i: int, sign: float) -> float:
    """The width (%GC) of the region of ``sign`` around sample ``i`` of the one-cycle curve ``c``
    (no duplicated end point): the contiguous samples of that sign plus one step on each side,
    from the last sample before the part to the first after it."""
    c = np.asarray(c, dtype=float)
    n = c.size
    left = right = 0
    while left < n - 1 and sign * c[(i - left - 1) % n] > 0.0:
        left += 1
    while right < n - 1 and sign * c[(i + right + 1) % n] > 0.0:
        right += 1
    return float(min(left + right + 2, n))
