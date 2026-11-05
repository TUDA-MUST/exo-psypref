"""Schaefer et al. (2026) four-timing hip assistance profile.

Four half-cosine segments joined at the two zero-crossings and the two peaks. The
parameters are timings only -- peak torques were held fixed at 12 Nm in that study, and
are fixed here too, which is what makes this a four-parameter family.

This is the primary rigid baseline for the project: it is hip rather than ankle, it
comes from this group, and it has published preferred values. Those values set a
realistic operating range and are deliberately NOT used as a recovery target -- see
our design notes on why a self-tuned stopping point is not ground truth.

Cycle order. The four timings must stay in the order t1 -> t2 -> t3 -> t4 -> t1 going
forward around the 100 %GC cycle. ``_t2_ordered`` and ``_t4_ordered`` are the two
sub-predicates (t2 strictly inside the forward arc t1->t3; t4 strictly inside the
forward arc t3->t1), and ``_cycle_ordered`` is exactly their conjunction -- nothing
else in this module re-derives either comparison. ``sample_prior`` rejection-samples
against ``_cycle_ordered``; ``project`` (via ``_project_ordered``) calls the same
``_t2_ordered``/``_t4_ordered`` sub-predicates to decide what, if anything, needs
repairing; the tests exercise all three. So each order comparison is written down in
exactly one place, and the modulo arithmetic itself is written down in exactly one
place below that (``_forward_gap``, which all three predicates are built on).
"""

from __future__ import annotations

import numpy as np
from preftune.profile import PHASE_GRID

from .base import BoxDesign

#: Published mean preferred timings, %ST: zero-crossing, extension peak,
#: zero-crossing, flexion peak.
PREFERRED_MEAN = np.array([95.3, 20.0, 44.2, 68.1])

#: Generous ranges around the published preferences, wide enough to contain the
#: reported between-participant spread of up to 22.5 %ST.
TIMING_LOW = np.array([85.0, 10.0, 34.0, 58.0])
TIMING_HIGH = np.array([105.0, 30.0, 54.0, 78.0])


def _half_cosine_rise(u: np.ndarray) -> np.ndarray:
    """0 at u = 0, 1 at u = 1, zero derivative at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * u))


def _forward_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Forward distance from ``a`` to ``b`` around the 100 %GC cycle, in [0, 100).

    The one place the "modulo 100" arithmetic is written down; every ordering check
    in this module (the predicate, the repair, and -- indirectly -- the tests) goes
    through this function rather than re-deriving ``% 100.0`` on its own.
    """
    return (np.asarray(b, dtype=float) - np.asarray(a, dtype=float)) % 100.0


def _t2_ordered(t1: np.ndarray, t2: np.ndarray, t3: np.ndarray) -> np.ndarray:
    """True where ``t2`` lies strictly inside the forward arc ``t1 -> t3`` (mod 100).

    This is the one place this comparison is written; ``_cycle_ordered`` and
    ``_project_ordered`` both call it rather than re-deriving it.
    """
    return _forward_gap(t1, t2) < _forward_gap(t1, t3)


def _t4_ordered(t3: np.ndarray, t4: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """True where ``t4`` lies strictly inside the forward arc ``t3 -> t1`` (mod 100).

    This is the one place this comparison is written; ``_cycle_ordered`` and
    ``_project_ordered`` both call it rather than re-deriving it.
    """
    return _forward_gap(t3, t4) < _forward_gap(t3, t1)


def _cycle_ordered(theta: np.ndarray) -> np.ndarray:
    """True where ``(t1, t2, t3, t4)`` respect the forward cycle order.

    ``theta`` is any array whose last axis has length 4, holding ``(t1, t2, t3,
    t4)``. Returns a boolean array of ``theta``'s leading shape (a scalar bool for a
    single ``(4,)`` vector). This is exactly the conjunction of ``_t2_ordered`` and
    ``_t4_ordered`` -- the single shared definition used by ``sample_prior``,
    ``project`` and the tests; see the module docstring.
    """
    theta = np.asarray(theta, dtype=float)
    t1, t2, t3, t4 = theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3]
    return _t2_ordered(t1, t2, t3) & _t4_ordered(t3, t4, t1)


class SchaeferDesign(BoxDesign):
    """Timings of the peaks and zero-crossings; peak torques fixed."""

    param_names = ("t1", "t2", "t3", "t4")

    def __init__(self, peak_ext_nm: float = 12.0, peak_flex_nm: float = 12.0,
                 prior: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        super().__init__(TIMING_LOW, TIMING_HIGH)
        self.peak_ext_nm = float(peak_ext_nm)
        self.peak_flex_nm = float(peak_flex_nm)
        #: The timing prior's fitted moments (``TimingPrior.design``), or ``None`` for the
        #: box-uniform default every existing caller still gets: with
        #: no prior, ``sample_prior`` is exactly the rejection sampler it always was.
        self.prior = prior

    def curve(self, theta: np.ndarray) -> np.ndarray:
        t1, t2, t3, t4 = (np.asarray(theta, dtype=float) % 100.0)
        out = np.zeros_like(PHASE_GRID)

        # Offsets of each sample from each segment start, going forward around the cycle.
        def span(start: float, end: float) -> float:
            width = (end - start) % 100.0
            return width if width > 0.0 else 100.0

        segments = (
            (t1, t2, -self.peak_ext_nm, True),    # rise into extension
            (t2, t3, -self.peak_ext_nm, False),   # fall out of extension
            (t3, t4, +self.peak_flex_nm, True),   # rise into flexion
            (t4, t1, +self.peak_flex_nm, False),  # fall out of flexion
        )
        for start, end, amplitude, rising in segments:
            width = span(start, end)
            offset = (PHASE_GRID - start) % 100.0
            inside = offset <= width
            u = np.where(inside, offset / width, 0.0)
            shape = _half_cosine_rise(u) if rising else _half_cosine_rise(1.0 - u)
            out = np.where(inside, amplitude * shape, out)
        return out

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Without ``prior`` (the default): uniform in the box, rejecting draws that
        scramble the cycle order -- unchanged since before this parameter existed.

        With ``prior``: draw
        ``N(mu, sigma)`` and repair through :meth:`project`, exactly
        ``PrompS0.sample_prior``'s pattern. ``project`` repairs order rather than
        rejecting it, which is the right treatment here: unlike the box-uniform case,
        where every draw is equally plausible so rejection costs nothing but a retry, a
        fitted prior's own probability mass should not be thrown away merely because a
        draw's ``t2`` or ``t4`` landed a little outside the required arc.
        """
        if self.prior is None:
            kept: list[np.ndarray] = []
            while len(kept) < n:
                batch = rng.uniform(self.low, self.high, size=(4 * n, 4))
                kept.extend(batch[_cycle_ordered(batch)])
            return np.array(kept[:n])
        mu, sigma = self.prior
        return self.project(rng.multivariate_normal(mu, sigma, size=n))

    def project(self, theta: np.ndarray) -> np.ndarray:
        """Nearest feasible vector: box-clip, then repair the cycle order.

        ``BoxDesign.project`` clips each of t1..t4 independently, which is not
        enough on its own to guarantee the same cycle order ``sample_prior``
        enforces (t2 strictly between t1 and t3, t4 strictly between t3 and t1,
        going forward mod 100). See ``_project_ordered`` for how the repair works
        and what it does with a badly out-of-order input; for this design's
        published ``TIMING_LOW``/``TIMING_HIGH`` the repair is provably a no-op
        (box membership alone already implies cycle order here -- see the test
        suite), so this override exists as insurance for callers such as a
        numerical optimiser that might pass in an arbitrary, not-necessarily-nice
        vector, rather than a path this design normally exercises.
        """
        return _project_ordered(theta, self.low, self.high)


def _snap_into_arc(
    value: np.ndarray,
    arc_start: np.ndarray,
    arc_length: np.ndarray,
    lo: float,
    hi: float,
    margin: float = 1e-6,
) -> np.ndarray:
    """Move ``value`` to the nearest point strictly inside the forward arc
    ``[arc_start, arc_start + arc_length)`` (mod 100), then clip into ``[lo, hi]``.

    If ``value`` already lies in the arc's interior it is returned unchanged --
    zero movement. Otherwise it is snapped to whichever end of the arc is
    circularly closer to its current offset (ties broken toward the forward/far
    end), backed off by ``margin`` %GC so the containment stays strict rather than
    landing exactly on the boundary. The final clip to ``[lo, hi]`` is a safety net
    for a box whose own range does not fully contain the snapped target -- for
    SchaeferDesign's published bounds this clip is always a no-op, but a box wide
    enough to only partially overlap the required arc would come back clipped to
    the nearest edge of its own box, which may not itself be strictly inside the
    arc; that would mean the box needs revisiting, not this function.
    """
    value = np.asarray(value, dtype=float)
    arc_start = np.asarray(arc_start, dtype=float)
    arc_length = np.asarray(arc_length, dtype=float)

    offset = _forward_gap(arc_start, value)
    inside = offset < arc_length

    dist_to_far_end = offset - arc_length  # how far past the arc's far end
    dist_to_near_end = 100.0 - offset  # how far, wrapping, back to the arc's near end
    snap_to_far_end = dist_to_far_end <= dist_to_near_end

    far_end_offset = np.maximum(arc_length - margin, 0.0)
    near_end_offset = np.minimum(margin, arc_length)
    snapped_offset = np.where(snap_to_far_end, far_end_offset, near_end_offset)

    new_offset = np.where(inside, offset, snapped_offset)
    new_value = (arc_start + new_offset) % 100.0
    return np.clip(new_value, lo, hi)


def _project_ordered(theta: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Clip ``theta`` into ``[low, high]`` elementwise, then repair its cycle order.

    This is what ``SchaeferDesign.project`` delegates to; it is also called
    directly by the test suite with bounds other than the design's own published
    ones, to exercise the repair branch that the design's real, disjoint box can
    never actually trigger (see the docstring of ``SchaeferDesign.project``).

    The repair treats the ``(t1, t2, t3)`` triple and the ``(t3, t4, t1)`` triple
    independently, deciding whether each needs fixing via the exact same
    ``_t2_ordered``/``_t4_ordered`` sub-predicates ``_cycle_ordered`` is built from
    (never a re-derived comparison), and, in each, moves only the offending timing
    (``t2`` or ``t4``); ``t1`` and ``t3`` are never touched by this function. Each
    offending timing is snapped to the *nearest* point of the arc it is required to
    sit inside (see ``_snap_into_arc``), so the repair is the smallest nudge that
    restores order, not a jump to a canonical position such as the arc's midpoint.

    A badly out-of-order input -- for example ``t2`` having lapped almost all the
    way around past ``t3`` -- is still handled deterministically: nearest-point
    projection onto a circular arc is well defined everywhere on the circle, with
    ties (an input exactly opposite the arc) broken toward the arc's forward/far
    end. There is no iteration and no randomness, so repeated calls on the same
    input always return the same output.
    """
    theta = np.clip(np.asarray(theta, dtype=float), low, high)
    single = theta.ndim == 1
    if single:
        theta = theta[np.newaxis, :]

    t1, t2, t3, t4 = theta[:, 0], theta[:, 1].copy(), theta[:, 2], theta[:, 3].copy()

    ok_t2 = _t2_ordered(t1, t2, t3)
    if not np.all(ok_t2):
        arc_t1_t3 = _forward_gap(t1, t3)
        fixed = _snap_into_arc(t2, t1, arc_t1_t3, low[1], high[1])
        t2 = np.where(ok_t2, t2, fixed)

    ok_t4 = _t4_ordered(t3, t4, t1)
    if not np.all(ok_t4):
        arc_t3_t1 = _forward_gap(t3, t1)
        fixed = _snap_into_arc(t4, t3, arc_t3_t1, low[3], high[3])
        t4 = np.where(ok_t4, t4, fixed)

    out = np.stack([t1, t2, t3, t4], axis=-1)
    return out[0] if single else out
