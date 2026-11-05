"""ProMP data priors at a basis count K other than 8.

A Phase 5a run's K = 8 prior, ``RUN/priors/<fold>-promp_s0``, was fitted by
``fit_fold_priors``: project the fold's generated training curves onto ``VonMisesBasis``
and take the mean and covariance of the weights. A K-basis prior is the same recipe through
the unchanged ``fit_promp_prior`` with ``VonMisesBasis(K)``, on the same training curves,
regenerated from the fold's own "training" stream (``scripts/icorr_k_priors.py`` refuses to
write one unless its K = 8 refit reproduces the stored prior bit for bit). It is stored
beside the K = 8 prior as ``RUN/priors/<fold>-promp_s0-k<K>``, and its best-profile distances as
``RUN/users/<fold>-promp_s0-k<K>.json``.

Nothing here edits ``data_prior.py`` or the population package: their hashes are part of
every prior's provenance and of the users export (HANDOFF, Phase 5b-i post-merge checks),
and ``prompref/basis.py`` and ``prompref/designs/promp.py`` are hashed with them. K enters
through a subclass whose ``design`` builds the K basis; ``DataPrior.design`` builds
``PrompS0``'s default 8-basis design and has no way to carry a basis through
``design_kwargs``, which must stay JSON.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..basis import DEFAULT_N_BASIS, VonMisesBasis
from ..designs.promp import PrompS0
from .data_prior import (MAX_FIT_RMS_NM, PROMP_DATA_BOUND_NM, DataPrior, _project_weights,
                         fit_promp_prior, share_at_bound)

#: The script that writes K priors and their best-profile distances.
BASIS_SCRIPT = "scripts/icorr_k_priors.py"

#: The provenance a K prior copies from its fold's K = 8 prior (``fit_fold_priors``'s
#: ``_common``): everything that describes the training set rather than the fit.
COMMON_FIELDS = ("fold", "n_curves", "n_rejected_draws", "switches", "timing_source", "seed",
                 "source_hashes", "timing", "shift")


@dataclass(eq=False)
class BasisDataPrior(DataPrior):
    """A ``promp_s0`` data prior over ``n_basis`` von Mises weights.

    Saved and loaded like a :class:`DataPrior` (``n_basis`` is one more JSON field). A plain
    ``DataPrior.load`` of such a file fails on the unknown field rather than building an
    8-basis design around a K-dimensional prior.
    """

    n_basis: int = DEFAULT_N_BASIS

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family != "promp_s0":
            raise ValueError(f"a basis-count prior is a promp_s0 prior, got {self.family!r}")
        if float(self.n_basis) != int(self.n_basis) or int(self.n_basis) < 2:
            raise ValueError(f"n_basis must be a whole number of at least 2, got {self.n_basis}")
        self.n_basis = int(self.n_basis)
        k = self.n_basis
        if self.mu.shape != (k,) or self.sigma.shape != (k, k):
            raise ValueError(f"a {k}-basis prior needs mu ({k},) and sigma ({k}, {k}); got "
                             f"{self.mu.shape} and {self.sigma.shape}")

    def basis(self) -> VonMisesBasis:
        return VonMisesBasis(self.n_basis)

    def design(self) -> PrompS0:
        """``DataPrior.design`` for ``promp_s0``, with this prior's basis: the same
        ``PROMP_DATA_BOUND_NM`` box, so only K differs between the K priors' designs."""
        return PrompS0(basis=self.basis(), prior=(self.mu, self.sigma),
                       bound_nm=PROMP_DATA_BOUND_NM, **self.design_kwargs)


def basis_label(n_basis: int) -> str:
    """The family label of a K prior's files: ``promp_s0-k4``."""
    return f"promp_s0-k{int(n_basis)}"


def basis_prior_stem(run_dir, fold: str, n_basis: int) -> Path:
    return Path(run_dir) / "priors" / f"{fold}-{basis_label(n_basis)}"


def basis_floors_path(run_dir, fold: str, n_basis: int) -> Path:
    return Path(run_dir) / "users" / f"{fold}-{basis_label(n_basis)}.json"


def common_fields(prior: DataPrior) -> dict:
    """The training-set provenance of ``prior``, as a K prior of the same fold carries it.
    Deep copies, so no two priors share a mutable dict."""
    return {name: copy.deepcopy(getattr(prior, name)) for name in COMMON_FIELDS}


def fit_basis_prior(curves: np.ndarray, n_basis: int, common: dict) -> BasisDataPrior:
    """``fit_fold_priors``' promp_s0 branch with ``VonMisesBasis(n_basis)``.

    ``common`` is the fold's training-set provenance (:func:`common_fields` of its K = 8
    prior). The fit statistics -- fit failures above ``MAX_FIT_RMS_NM``, the median fit RMS
    and the share of weights at the +/-``PROMP_DATA_BOUND_NM`` box -- are computed exactly as
    there, so at K = 8 the result equals the stored prior field for field.
    """
    basis = VonMisesBasis(int(n_basis))
    curves = np.asarray(curves, dtype=float)
    mu, sigma, rms = fit_promp_prior(curves, basis)
    weights = _project_weights(curves, basis)
    names = tuple(f"w{i}" for i in range(basis.n_basis))
    bound = np.full(basis.n_basis, PROMP_DATA_BOUND_NM)
    return BasisDataPrior(
        "promp_s0", mu=mu, sigma=sigma,
        n_fit_failures=int((rms > MAX_FIT_RMS_NM).sum()),
        fit_rms_median_nm=float(np.median(rms)),
        share_at_bound=share_at_bound(weights, -bound, bound, names),
        n_basis=basis.n_basis, **{name: common[name] for name in COMMON_FIELDS})
