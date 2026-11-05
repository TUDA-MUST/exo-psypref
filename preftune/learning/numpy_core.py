"""A vectorised reimplementation of the APReL algorithm Ramella et al. used.

Written only because APReL costs 7.8 s per query — 5.9 s of it scoring all 780
candidate pairs against 100 posterior samples in a Python loop — which puts the
parameter sweep out of reach (PLAN.md §5.2). The algorithm is unchanged:

- reward linear in the batch-normalised features, ``R(xi) = w. phi(xi)``
- softmax response model with rationality ``beta = 1.0``
- uniform prior over ``||w|| <= 1``
- Metropolis-Hastings with a gaussian proposal of standard deviation 0.05 that
  renormalises ``w`` to the unit sphere, 200 burn-in, thinning 20, 100 samples
- exhaustive mutual-information search over all pairs

The uniform prior is *not* vacuous, despite the proposal renormalising ``w`` to unit
length: APReL tests ``||w|| <= 1`` on the already-normalised vector, so rounding to
``1 + eps`` rejects the proposal outright. That happens for about 2 % of proposals
and has nothing to do with the posterior. Small as it is, Metropolis-Hastings is
chaotic -- one differing accept/reject decision changes the whole remaining chain --
so reproducing it is what makes the two implementations agree bit for bit.

The proposal draws ``np.random.randn(6)`` and the acceptance test draws
``np.random.rand``, in that order, from numpy's legacy global generator — exactly
as APReL does. Seeded identically, the two implementations walk the same chain, so
:func:`preftune.learning.equivalence.compare` can assert exact agreement rather than
merely statistical agreement.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .base import Learner

BURNIN = 200
THIN = 20
NUM_SAMPLES = 100
PROPOSAL_STD = 0.05
BETA = 1.0

ACQUISITIONS = ("mutual_information", "volume_removal", "random")


def _pair_indices(n: int) -> np.ndarray:
    """All unordered pairs, in ``itertools.combinations`` order — APReL's enumeration."""
    i, j = np.triu_indices(n, k=1)
    return np.column_stack([i, j])


def mutual_information(pair_rewards: np.ndarray) -> np.ndarray:
    """APReL's mutual-information objective, evaluated for every pair at once.

    ``pair_rewards`` is ``(n_pairs, n_samples, 2)``. Mirrors APReL's

        np.sum(probs * np.log2(M * probs / np.sum(probs, axis=0))) / M

    with ``probs`` the per-sample softmax over the two options.
    """
    probs = np.exp(pair_rewards - logsumexp(pair_rewards, axis=2, keepdims=True))
    m = probs.shape[1]
    denom = probs.sum(axis=1, keepdims=True)  # sum over samples
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = probs * np.log2(m * probs / denom)
    return np.nansum(terms, axis=(1, 2)) / m


def volume_removal(pair_rewards: np.ndarray) -> np.ndarray:
    """APReL's volume-removal objective: minimise the expected remaining volume."""
    probs = np.exp(pair_rewards - logsumexp(pair_rewards, axis=2, keepdims=True))
    return float(probs.shape[1]) - probs.sum(axis=1).max(axis=1)


class NumpyLearner(Learner):
    """Ramella's learner, vectorised. Interchangeable with :class:`AprelLearner`."""

    def __init__(self, batch: np.ndarray, seed: int = 0,
                 acquisition: str = "mutual_information", num_samples: int = NUM_SAMPLES,
                 burnin: int = BURNIN, thin: int = THIN, beta: float = BETA) -> None:
        super().__init__(batch, seed)
        if acquisition not in ACQUISITIONS:
            raise ValueError(f"acquisition must be one of {ACQUISITIONS}, got {acquisition!r}")
        self.acquisition = acquisition
        self.num_samples = num_samples
        self.burnin = burnin
        self.thin = thin
        self.beta = beta

        np.random.seed(seed)  # APReL uses the legacy global generator; match it
        self._pairs = _pair_indices(self.n_candidates)
        self._dataset: list[tuple[int, int, int]] = []
        self._data_features = None
        d = self.features.shape[1]
        self._initial = np.ones(d) / np.sqrt(d)
        self.samples = np.repeat(self._initial[None, :], num_samples, axis=0)
        self._pending: tuple[int, int] | None = None
        self._resample()

    # --- posterior ------------------------------------------------------------------
    def _rebuild_cache(self) -> None:
        """Precompute the dataset's feature slabs so the MH loop stays allocation-free."""
        if not self._dataset:
            self._data_features = None
            return
        idx = np.asarray(self._dataset, dtype=int)  # (n, 3): i, j, response
        self._data_features = self.features[idx[:, :2]]  # (n, 2, d)
        self._data_rows = np.arange(len(idx))
        self._data_response = idx[:, 2]

    def _loglikelihood(self, w: np.ndarray) -> float:
        """Sum of preference log-likelihoods under weights ``w``."""
        if self._data_features is None:
            return 0.0
        rewards = self.beta * (self._data_features @ w)  # (n, 2)
        chosen = rewards[self._data_rows, self._data_response]
        return float(np.sum(chosen - logsumexp(rewards, axis=1)))

    def _resample(self) -> None:
        """Metropolis-Hastings, in APReL's exact draw order.

        The chain restarts from the previous posterior *mean* on every update, which
        is APReL's default (``initial_point = self.mean``) — not from the last sample.
        """
        curr = self._initial.copy() if not self._dataset else self.weights()
        curr_logprob = self._loglikelihood(curr)
        chain = [curr]
        for _ in range(self.burnin + self.thin * self.num_samples - 1):
            proposal = curr + np.random.randn(curr.size) * PROPOSAL_STD
            proposal = proposal / np.linalg.norm(proposal)
            # APReL's uniform prior tests ||w|| <= 1 *after* the proposal has already
            # normalised w to unit length, so rounding to 1+eps rejects it outright --
            # about 2 % of proposals, for no posterior reason. Reproduced deliberately:
            # MH is chaotic, so without it the chains diverge from APReL's entirely.
            prop_logprob = (self._loglikelihood(proposal)
                            if np.linalg.norm(proposal) <= 1.0 else -np.inf)
            if np.log(np.random.rand()) < prop_logprob - curr_logprob:
                curr, curr_logprob = proposal, prop_logprob
            chain.append(curr)
        self.samples = np.array(chain[self.burnin :: self.thin], dtype=float)

    # --- Learner interface ----------------------------------------------------------
    def query(self) -> tuple[int, int]:
        if self.acquisition == "random":
            k = np.random.randint(len(self._pairs))
        else:
            rewards = self.beta * (self.features @ self.samples.T)  # (n_candidates, n_samples)
            pair_rewards = np.stack(
                [rewards[self._pairs[:, 0]], rewards[self._pairs[:, 1]]], axis=2
            )  # (n_pairs, n_samples, 2)
            objective = (mutual_information if self.acquisition == "mutual_information"
                         else volume_removal)(pair_rewards)
            k = int(np.argmax(objective))
        i, j = self._pairs[k]
        self._pending = (int(i), int(j))
        return self._pending

    def update(self, i: int, j: int, response: int) -> None:
        if self._pending is None:
            raise RuntimeError("call query() before update()")
        self._dataset.append((int(i), int(j), int(response)))
        self._pending = None
        self._rebuild_cache()
        self._resample()
        self.history.append({
            "i": int(i), "j": int(j), "response": int(response),
            "weights": self.weights().tolist(),
            "posterior_std": self.samples.std(axis=0).tolist(),
        })

    def weights(self) -> np.ndarray:
        w = self.samples.mean(axis=0)
        norm = np.linalg.norm(w)
        return w / norm if norm > 0 else w

    def posterior_samples(self) -> np.ndarray:
        return self.samples
