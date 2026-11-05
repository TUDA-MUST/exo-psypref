"""Drive APReL -- the library Ramella et al. used -- over a fixed batch of profiles.

APReL is built around gym rollouts: ``Trajectory.__init__`` calls
``env.features(trajectory)`` with no null check. Our candidates are already
six-dimensional feature vectors, so instead of building a gym environment we pass a
stub whose ``features`` is a lookup. That is the whole adapter.

Install the reference implementation separately (it is not on PyPI, and it needs an
old moviepy)::

    pip install "moviepy<2"
    pip install "git+https://github.com/Stanford-ILIAD/APReL.git@13a1e7f7"

Without ``moviepy<2`` the import fails outright: moviepy 2.0 removed
``moviepy.editor``, which APReL imports at module scope.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from .base import Learner

APREL_INSTALL_HINT = (
    "APReL is not importable. It is not on PyPI; install with\n"
    '    pip install "moviepy<2"\n'
    '    pip install "git+https://github.com/Stanford-ILIAD/APReL.git@13a1e7f7"\n'
    "The moviepy pin is required: moviepy>=2 removed moviepy.editor."
)


def import_aprel():
    """Import APReL quietly.

    gym 0.26 prints an unconditional deprecation notice straight to ``sys.stderr`` at
    import time, with no environment guard. Windows PowerShell turns any stderr from a
    native command into an ErrorRecord, so the notice reads as a failure. Swallow it.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        import aprel
    return aprel


def aprel_available() -> bool:
    try:
        import_aprel()
    except Exception:
        return False
    return True


class _FeatureLookupEnv:
    """Stands in for a gym environment; APReL only ever calls ``features``."""

    def __init__(self, features: np.ndarray) -> None:
        self._features = features

    def features(self, index):  # noqa: D102 - APReL's expected hook
        return self._features[index]


class AprelLearner(Learner):
    """Ramella's learner: linear reward, softmax likelihood, Metropolis-Hastings belief.

    Defaults mirror APReL's own: a gaussian proposal, 200 burn-in samples, thinning
    of 20, and 100 retained samples, with the weight vector on the unit sphere.
    """

    def __init__(self, batch: np.ndarray, seed: int = 0,
                 acquisition: str = "mutual_information", num_samples: int = 100) -> None:
        super().__init__(batch, seed)
        try:
            aprel = import_aprel()
        except Exception as exc:  # pragma: no cover - exercised only without APReL
            raise ImportError(APREL_INSTALL_HINT) from exc

        self._aprel = aprel
        self.acquisition = acquisition
        np.random.seed(seed)  # APReL samples from numpy's global generator

        env = _FeatureLookupEnv(self.features)
        self._tset = aprel.TrajectorySet(
            [aprel.Trajectory(env, i) for i in range(self.n_candidates)]
        )
        self._optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(self._tset)
        params = {"weights": np.ones(self.features.shape[1]) / np.sqrt(self.features.shape[1])}
        self._belief = aprel.SamplingBasedBelief(
            aprel.SoftmaxUser(params), [], params, num_samples=num_samples
        )
        self._template = aprel.PreferenceQuery(self._tset[:2])
        self._pending = None

    def query(self) -> tuple[int, int]:
        queries, _ = self._optimizer.optimize(self.acquisition, self._belief, self._template)
        self._pending = queries[0]
        i, j = (t.trajectory for t in self._pending.slate)
        return int(i), int(j)

    def update(self, i: int, j: int, response: int) -> None:
        if self._pending is None:
            raise RuntimeError("call query() before update()")
        self._belief.update(self._aprel.Preference(self._pending, int(response)))
        self._pending = None
        # Track the posterior's own spread, not just its mean: a wandering mean over
        # few comparisons is the posterior being honestly wide, which is a different
        # claim from the learner failing to converge.
        self.history.append({
            "i": int(i), "j": int(j), "response": int(response),
            "weights": self.weights().tolist(),
            "posterior_std": self.posterior_samples().std(axis=0).tolist(),
        })

    def weights(self) -> np.ndarray:
        w = np.asarray(self._belief.mean["weights"], dtype=float)
        norm = np.linalg.norm(w)
        return w / norm if norm > 0 else w

    def posterior_samples(self) -> np.ndarray:
        """The retained Metropolis-Hastings samples, ``(num_samples, 6)``."""
        return np.array([s["weights"] for s in self._belief.samples], dtype=float)
