"""Expected information gain about z* for many candidate duels at once.

Design-free, like ``smc.py``: it sees particles and whitened duel ends only. For a binary
answer the gain is exact to reweight both ways; the entropy of the ``z*`` marginal is
approximated by a Gaussian's, ``0.5 log det Cov``. That proxy is never negative -- the
law of total covariance and the concavity of log det -- which the Phase 4 gate checks.
It ranks duels; it is not a mutual information in nats, and a wide duel through the middle
of a skewed cloud scores above ln 2 by construction.

Two guards: the answer probability is floored at ``P_FLOOR``, and each branch
covariance gets a ``RIDGE``. A branch holding a few effective particles would otherwise
have a collapsed log det and score a useless duel as hugely informative.

A third guard for heavy-tailed priors: ``trim`` keeps
the particles nearest the weighted z* mean. One far particle inflates the cloud's covariance
along its direction, and a duel far out on that side upweights it in one branch; untrimmed,
the proxy scored duels 16 posterior SDs out 1.2-2.4 times higher than central ones, while
the exact information there was near zero.
"""

from __future__ import annotations

import numpy as np

from .smc import per_duel_log_likelihood

P_FLOOR = 1e-6
RIDGE = 1e-9

#: Weighted share of particles ``trim`` keeps under heavy-tailed priors.
#: At 0.95 the acquisition gate passes under t4, and t2 agreement passes; at 0.99 t2 agreement
#: still failed (minimum ratio 0.793).
TRIM_QUANTILE = 0.95


def trim_mask(x: np.ndarray, w: np.ndarray, d: int, q: float) -> np.ndarray:
    """Particles whose ``z*`` lies within the weighted ``q`` quantile of Euclidean distance
    from the weighted mean, as a boolean mask of shape ``(N,)``."""
    if not 0.0 < q <= 1.0:
        raise ValueError(f"trim must be in (0, 1]; got {q}")
    zs = np.asarray(x, dtype=float)[:, :d]
    w = np.asarray(w, dtype=float)
    r = np.linalg.norm(zs - w @ zs, axis=1)
    order = np.argsort(r)
    cut = r[order][min(int(np.searchsorted(np.cumsum(w[order]), q)), len(r) - 1)]
    return r <= cut


def answer_probabilities(x: np.ndarray, z_a: np.ndarray, z_b: np.ndarray) -> np.ndarray:
    """P(A preferred) under every particle for every candidate, shape ``(N, C)``."""
    z_a, z_b = np.atleast_2d(z_a), np.atleast_2d(z_b)
    return np.exp(per_duel_log_likelihood(x, z_a, z_b, np.ones(len(z_a), dtype=bool)))


def _log_det_cov(zs: np.ndarray, outer: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``log det`` of the weighted covariance of ``zs`` for each weight column, ``(C,)``."""
    d = zs.shape[1]
    mean = zs.T @ weights  # (d, C)
    second = (outer.T @ weights).reshape(d, d, -1)  # (d, d, C)
    cov = np.moveaxis(second, -1, 0) - np.einsum("ic,jc->cij", mean, mean)
    return np.linalg.slogdet(cov + RIDGE * np.eye(d))[1]


def expected_information_gain(x: np.ndarray, w: np.ndarray, d: int, z_a: np.ndarray,
                              z_b: np.ndarray, *, trim: float | None = None) -> np.ndarray:
    """Gaussian-proxy EIG on ``z*`` for each candidate pair, shape ``(C,)``.

    ``trim`` (heavy-tailed priors only) computes it on ``trim_mask(x, w, d, trim)`` with the
    kept weights renormalised. ``None`` is the untrimmed Phase 4 proxy, unchanged."""
    if trim is not None:
        keep = trim_mask(x, w, d, trim)
        x, w = np.asarray(x, dtype=float)[keep], np.asarray(w, dtype=float)[keep]
        w = w / w.sum()
    zs = np.asarray(x, dtype=float)[:, :d]
    outer = (zs[:, :, None] * zs[:, None, :]).reshape(len(zs), d * d)
    weighted = np.asarray(w, dtype=float)[:, None] * answer_probabilities(x, z_a, z_b)
    p = np.clip(weighted.sum(axis=0), P_FLOOR, 1.0 - P_FLOOR)
    h_prior = 0.5 * _log_det_cov(zs, outer, np.asarray(w, dtype=float)[:, None])[0]
    h_a = 0.5 * _log_det_cov(zs, outer, weighted / p)
    h_b = 0.5 * _log_det_cov(zs, outer, (np.asarray(w, dtype=float)[:, None] - weighted) / (1.0 - p))
    return h_prior - p * h_a - (1.0 - p) * h_b
