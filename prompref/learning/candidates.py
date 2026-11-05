"""The our design notes candidate set, built under.

Candidates are built in the posterior's own space, where the posterior covariance and
the Thompson draws live. Feasibility is judged in the duel space, where the trust search region
(and, optionally, the step search region around the worn curve) is defined. A candidate's centre
is transferred to the duel space, clipped there, and transferred back, exactly as
before. Each end is then truncated along the duel's own line: :func:`truncate_duel`
walks out from that centre toward each end and stops as soon as going further would
leave the trust search region, or, under a step region, the step search region -- so a duel keeps its
centre line and its direction, and never turns into a multi-parameter move. A candidate
whose truncated ends coincide (width below ``1e-9``) carries no information and is
dropped. ``width`` is the actual separation, and truncation only ever shortens a duel
toward its own centre, so ``width`` never exceeds the requested multiple of ell-hat.

Three earlier geometries are recorded here because each failed for reasons worth
remembering. our design notes clipped centres into the search regions and then capped each
candidate's width at the widest symmetric segment that still fit; every out-of-search region
centre then sat exactly on a boundary, where the only symmetric duel that fits has
width 0 (measured: 48-66 % of candidates at eps=1, some sessions with none at all).
Item 11's first cut moved the whole centre inward until the full-width duel fit
("shrink-to-fit"); that removed the zero-width failure but translated wide duels toward
the prior mean by up to about ell-hat, and a pre-merge spot check showed it made
learner A's posterior *worse* as comparisons accumulated. Item 11's second cut clipped
each end radially and independently into the region; that reproduced the prototype's
monotone improvement but turned coordinate duels into multi-parameter moves and could
widen duels under a step region. Truncation along the duel's line keeps its direction
and never widens it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .space import LearnerSpace

#: Duel widths as multiples of the posterior's ell-hat.
MAGNITUDES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DIRECTIONS = ("free", "coordinate")

SearchRegion = tuple[np.ndarray, float]


def shrink_toward(origin: np.ndarray, point: np.ndarray, search_regions: Sequence[SearchRegion]) -> np.ndarray:
    """The farthest point from ``origin`` along the segment to ``point`` inside every search region."""
    origin, point = np.asarray(origin, dtype=float), np.asarray(point, dtype=float)
    step = point - origin
    length_sq = float(step @ step)
    if length_sq == 0.0:
        return point
    t = 1.0
    for centre, radius in search_regions:
        a = origin - np.asarray(centre, dtype=float)
        ab = float(a @ step)
        disc = ab * ab - length_sq * (float(a @ a) - radius * radius)
        t = min(t, (-ab + np.sqrt(max(disc, 0.0))) / length_sq)
    return origin + max(t, 0.0) * step


def clip_into_region(z_duel: np.ndarray, duel: LearnerSpace, *,
                     z_worn: np.ndarray | None = None,
                     r_step: float | None = None) -> np.ndarray:
    """One duel-space point: radially into the trust search region, or, with a step region, shrunk
    toward ``z_worn`` into the trust search region and the step search region (``z_worn`` must lie in both).
    """
    if r_step is not None and z_worn is None:
        raise ValueError("a step region needs z_worn, the duel-space point it is centred on")
    if r_step is None:
        return duel.clip_to_search_region(z_duel)
    z_worn = np.asarray(z_worn, dtype=float)
    search_regions: list[SearchRegion] = [(np.zeros(duel.d), duel.radius), (z_worn, float(r_step))]
    return shrink_toward(z_worn, z_duel, search_regions)


def step_region(z_worn: np.ndarray | None, r_step: float | None) -> dict:
    """The step-region keyword arguments shared by :func:`clip_into_region` and
    :func:`truncate_duel`, built once so callers do not each repeat the
    "z_worn only when r_step is set" gate."""
    return dict(z_worn=z_worn if r_step is not None else None, r_step=r_step)


def coordinate_directions(own: LearnerSpace) -> list[np.ndarray]:
    """One unit move per design parameter, mapped into ``own``'s z and normalised."""
    n = own.design.n_params
    moves = own.normalization.to_z(np.eye(n)) - own.normalization.to_z(np.zeros(n))
    return [m / np.linalg.norm(m) for m in moves]


def _exit(c: np.ndarray, v: np.ndarray, search_region: SearchRegion) -> float:
    """How far ``c`` can move along ``v`` before leaving ``search region``, at least 0."""
    centre, radius = search_region
    a = c - np.asarray(centre, dtype=float)
    av = float(a @ v)
    vv = float(v @ v)
    disc = av * av - vv * (float(a @ a) - radius * radius)
    return max(0.0, (-av + np.sqrt(max(disc, 0.0))) / vv)


def truncate_duel(c_own: np.ndarray, u: np.ndarray, half: float, own: LearnerSpace,
                  duel: LearnerSpace, *, z_worn: np.ndarray | None = None,
                  r_step: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The duel's ends ``c_own +/- t * u``, each ``t <= half`` and as large as keeps that
    end inside every search region: the trust search region, plus the step search region around ``z_worn`` when
    ``r_step`` is given. The duel keeps its line and its direction and is never wider
    than ``2 * half``.

    ``c_own`` must already lie in the region -- the caller clips it with
    :func:`clip_into_region`. ``u`` is a unit vector in own coordinates. The midpoint
    moves off ``c_own`` when the two truncations differ; that is intended.
    """
    if r_step is not None and z_worn is None:
        raise ValueError("a step region needs z_worn, the duel-space point it is centred on")
    c_duel = own.transfer(c_own, duel)
    v = own.transfer(u, duel) - own.transfer(np.zeros(own.d), duel)
    search_regions: list[SearchRegion] = [(np.zeros(duel.d), duel.radius)]
    if r_step is not None:
        search_regions.append((np.asarray(z_worn, dtype=float), float(r_step)))
    t_plus = min([half] + [_exit(c_duel, v, search_region) for search_region in search_regions])
    t_minus = min([half] + [_exit(c_duel, -v, search_region) for search_region in search_regions])
    return c_own + t_plus * u, c_own - t_minus * u


def candidate_duels(x: np.ndarray, w: np.ndarray, own: LearnerSpace, duel: LearnerSpace,
                    rng: np.random.Generator, *, ell_hat: float, max_mult: float = 4.0,
                    directions: str = "free", n_thompson: int = 3, n_random: int = 3,
                    z_worn: np.ndarray | None = None, r_step: float | None = None,
                    axes_rule=None, width_rule=None) -> dict:
    """The candidate set: centres (the posterior mean and Thompson draws) x directions x widths.

    Two opt-in hooks for the acquisition variants of ICORR section 9a
    (:mod:`prompref.learning.acq_variants`); left at ``None``, the set is built exactly as
    before. ``axes_rule(cov)`` replaces the posterior's d principal axes with the directions it
    returns (A-sub); the random directions are still drawn after them. ``width_rule(c_own, u)``
    replaces the ladder of widths with one duel per centre and direction, returning
    ``(z_a, z_b, multiple, capped)`` or ``None`` to drop the pair (A-width). With a hook, the
    result also carries ``n_axes`` or ``n_capped``.
    """
    if directions not in DIRECTIONS:
        raise ValueError(f"directions must be one of {DIRECTIONS}, got {directions!r}")
    if r_step is not None and z_worn is None:
        raise ValueError("a step region needs z_worn, the duel-space point it is centred on")
    if axes_rule is not None and directions != "free":
        raise ValueError("axes_rule replaces the posterior's axes, which only 'free' directions use")
    d = own.d
    zs = np.asarray(x, dtype=float)[:, :d]
    w = np.asarray(w, dtype=float)
    mean = w @ zs
    cov = ((zs - mean) * w[:, None]).T @ (zs - mean)
    region = step_region(z_worn, r_step)

    raw_centres = [mean] + [zs[i] for i in rng.choice(len(w), size=n_thompson, p=w)]
    if directions == "free":
        if axes_rule is None:
            _, vecs = np.linalg.eigh(cov)
            dirs = [vecs[:, i] for i in range(d)]
        else:
            dirs = list(axes_rule(cov))
        n_axes = len(dirs)
        for _ in range(n_random):
            u = rng.normal(size=d)
            dirs.append(u / np.linalg.norm(u))
    else:
        dirs = coordinate_directions(own)

    # Descending, so a truncated duel is identical under every smaller label too, and
    # np.argmax's first-maximum tie-break picks the largest one (our design notes, revised
    # again).
    ladder = sorted((k for k in MAGNITUDES if k <= max_mult), reverse=True)
    z_a, z_b, multiple, width = [], [], [], []
    n_capped = 0
    for c_own in raw_centres:
        c_own = duel.transfer(clip_into_region(own.transfer(c_own, duel), duel, **region), own)
        for u in dirs:
            if width_rule is not None:
                pair = width_rule(c_own, u)
                if pair is None:
                    continue
                a, b, k, capped = pair
                n_capped += bool(capped)
                z_a.append(a)
                z_b.append(b)
                multiple.append(k)
                width.append(float(np.linalg.norm(a - b)))
                continue
            for k in ladder:
                a, b = truncate_duel(c_own, u, 0.5 * k * ell_hat, own, duel, **region)
                w_ab = float(np.linalg.norm(a - b))
                if w_ab < 1e-9:
                    continue
                z_a.append(a)
                z_b.append(b)
                multiple.append(k)
                width.append(w_ab)
    if not z_a:
        raise ValueError("no candidate duel fits inside the trust region and step region")
    out = {"z_a": np.array(z_a), "z_b": np.array(z_b),
           "multiple": np.array(multiple), "width": np.array(width)}
    if axes_rule is not None:
        out["n_axes"] = n_axes
    if width_rule is not None:
        out["n_capped"] = n_capped
    return out
