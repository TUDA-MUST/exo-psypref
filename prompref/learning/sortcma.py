"""SortCMA: CMA-ES ranked by the simulated user's pairwise answers.

After "Optimizing Algorithms From Pairwise User Preferences" (arXiv 2308.04571, IROS 2023):
each generation's lambda offspring are sorted with the user as comparator, and the order is
handed to CMA-ES as ranks. CMA-ES is pycma (Hansen's reference implementation) with its
defaults; the sort is Python's built-in Timsort through ``functools.cmp_to_key``.

``sorted`` calls its comparator synchronously, while a session asks for one duel at a time,
so the sort runs in a daemon thread that hands each comparison out through a queue and blocks
for its answer. A session that ends mid-sort leaves that thread waiting; it gives up after
``_TIMEOUT`` seconds. pycma is kept off numpy's global generator (``seed=nan`` and a ``randn``
drawing from our own Generator) and writes no files.
"""

from __future__ import annotations

import functools
import queue
import threading
import weakref

import cma
import numpy as np

#: Kutulakos & Slade (2024)'s default step size, as a fraction of the normalised range (L12).
KUTULAKOS_SIGMA_FRACTION = 0.3
_SALT = 0x50C7C4A
_TIMEOUT = 600.0
_CLOSED = object()


class _SessionClosed(Exception):
    pass


class SortCMALearner:
    def __init__(self, design, seed: int = 0, *, sigma0: float = 1.0, mean0=None,
                 popsize: int | None = None) -> None:
        """``popsize`` is pycma's lambda; ``None`` leaves pycma's default, 4 + floor(3 ln d)."""
        self.design = design
        d = int(design.n_params)
        self._rng = rng = np.random.default_rng([int(seed), _SALT])
        # The lambda captures rng, not self: self -> es -> opts -> lambda -> self is a cycle that
        # only the cyclic GC would break, so the finalizer would not fire on a plain `del`.
        options = {"bounds": [list(map(float, design.low)), list(map(float, design.high))],
                   "seed": np.nan, "randn": lambda lam, n: rng.standard_normal((lam, n)),
                   "verbose": -9, "verb_log": 0, "verb_disp": 0}
        if popsize is not None:
            options["popsize"] = int(popsize)
        x0 = np.zeros(d) if mean0 is None else np.asarray(mean0, dtype=float)
        self.es = cma.CMAEvolutionStrategy(list(x0), float(sigma0), options)
        self.n_generations = 0
        self._pending: tuple[int, int] | None = None
        # One answers queue for the whole session: a generation's sort has consumed every answer it
        # asked for before it finishes, so nothing stale carries over. The finalizer captures the
        # queue only (never self), so dropping the learner releases a blocked sort thread at once.
        self._answers: queue.Queue = queue.Queue()
        self._finalizer = weakref.finalize(self, self._answers.put, _CLOSED)
        self._start_generation()

    def close(self) -> None:
        """End the session: a sort thread waiting for an answer exits immediately."""
        self._finalizer()

    def _start_generation(self) -> None:
        # Each offspring is clipped into the design (the trust search region) before it is shown, and CMA-ES
        # is later told these same clipped vectors, so it ranks what the simulated user saw (review
        # I2). pycma takes a told vector it did not send through its bounds' inverse and its
        # Mahalanobis-length repair, as for any injected solution.
        self._offspring = [self.design.project(np.asarray(x, dtype=float)) for x in self.es.ask()]
        self._asks: queue.Queue = queue.Queue()
        asks, answers, n = self._asks, self._answers, len(self._offspring)

        def compare(i: int, j: int) -> int:
            asks.put(("duel", i, j))
            answer = answers.get(timeout=_TIMEOUT)
            if answer is _CLOSED:
                raise _SessionClosed
            return -1 if answer else 1

        def sort() -> None:
            try:
                order = sorted(range(n), key=functools.cmp_to_key(compare))
            except (queue.Empty, _SessionClosed):
                return
            asks.put(("done", order, None))

        self._thread = threading.Thread(target=sort, daemon=True)
        self._thread.start()

    def propose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._pending is not None:      # unanswered duel: show it again, never wait on the sort
            a, b = self._pending
            return self._offspring[a].copy(), self._offspring[b].copy()
        kind, a, b = self._asks.get(timeout=_TIMEOUT)
        if kind == "done":
            ranks = np.empty(len(a))
            ranks[np.asarray(a)] = np.arange(len(a))
            self.es.tell(self._offspring, list(ranks))
            self.n_generations += 1
            self._start_generation()
            kind, a, b = self._asks.get(timeout=_TIMEOUT)
        self._pending = (a, b)
        return self._offspring[a].copy(), self._offspring[b].copy()

    def update(self, z_a, z_b, response: int) -> None:
        if self._pending is None:
            raise RuntimeError("call propose() before update()")
        self._pending = None
        # 0 prefers the first-shown offspring; Phase 5 sessions never answer "no preference".
        self._answers.put(response == 0)

    def best(self) -> np.ndarray:
        """pycma's documented estimate, ``result.xfavorite``: the distribution mean in phenotype
        space (``es.mean`` is the genotype, which can lie outside the bounds), search region-clipped."""
        return self.design.project(np.asarray(self.es.result.xfavorite, dtype=float))

    def posterior_summary(self) -> dict:
        return {"extras": {"cmaes_generations": int(self.n_generations),
                           "cmaes_sigma": float(self.es.sigma),
                           "cmaes_popsize": int(self.es.popsize)}}
