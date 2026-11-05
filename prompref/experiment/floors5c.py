"""Floors Phase 5c needs beyond 5a's and 5b-ii's.

Both use 5a's floor machinery unchanged -- same multi-start, tie-break and cache -- and write
sidecars beside the exported users, which stay untouched:

- **C1's box.** Ramella's semi-trapezoid, scored against every user, with the box's
  moment-matched whitening, as a flat cell's floors are.
- **Drifted final ideals.** For each user and displacement, the time shift that reaches it
  and the ProMP data prior's floor for the shifted curve. A drift session interpolates its
  floor between the user's own and this one.
- **Basis counts**. Each user's ProMP floor at K bases, against the
  fold's K prior (``prompref.priors.basis_prior``); written by ``scripts/icorr_k_priors.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from ..designs.trapezoid import TrapezoidDesign
from ..population.users import FloorCache, prior_digest, user_floors
from ..priors.basis_prior import BasisDataPrior, basis_label
from ..priors.data_prior import DataPrior
from ..user.drift import scale_for_displacement, shift_for_displacement, time_shift
from .phase5 import C1_FAMILY, drift_label, flat_whitening, magnitude_label


def fold_seed(*parts: str) -> np.random.SeedSequence:
    """A deterministic seed for one fold's floor computation. Not ``hash(fold)``: Python
    salts a str's hash per interpreter process, so that would reseed differently on every
    run. ``scripts/phase5_schaefer_floors.py`` uses this same one."""
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).digest()
    return np.random.SeedSequence(int.from_bytes(digest[:4], "big"))


def _write(path: Path, sidecar: dict) -> None:
    path.write_text(json.dumps(sidecar, indent=1), encoding="utf-8")


def trapezoid_sidecar(run: Path, fold: str, users: list, *, n_starts: int,
                      workers: int) -> dict:
    """C1's floor for every user, over the semi-trapezoid box."""
    design = TrapezoidDesign()
    whitening = flat_whitening(design)
    family = f"flat-{C1_FAMILY}"
    entries = user_floors(design, whitening, users, n_starts=n_starts,
                          seed_sequence=fold_seed(fold, "flat", C1_FAMILY), workers=workers,
                          cache=FloorCache(run / "floors" / f"{fold}-{family}.json"),
                          family=family, prior_hash=prior_digest(whitening.mu, whitening.sigma))
    sidecar = {user.user_id: {"floor_nm": float(entry["floor_nm"]),
                              "reach_050": float(entry["reach_050"]),
                              "radius": float(entry["radius_floor"])}
               for user, entry in zip(users, entries)}
    _write(run / "users" / f"{fold}-{family}.json", sidecar)
    return sidecar


def basis_sidecar(run: Path, fold: str, users: list, prior: BasisDataPrior, *, n_starts: int,
                  workers: int) -> dict:
    """Every user's floor in the ProMP family at ``prior.n_basis`` bases, with that prior's
    whitening for the tie-break and the reach radii. The same machinery and seeding as the other sidecars here; the entries carry the
    exported users' four per-family values, so a K cell reads one exactly as a K = 8 cell
    reads ``user.families["promp_s0"]``."""
    label = basis_label(prior.n_basis)
    entries = user_floors(prior.design(), prior.whitening(), users, n_starts=n_starts,
                          seed_sequence=fold_seed(fold, label), workers=workers,
                          cache=FloorCache(run / "floors" / f"{fold}-{label}.json"),
                          family=label, prior_hash=prior_digest(prior.mu, prior.sigma))
    sidecar = {user.user_id: {"floor_nm": float(entry["floor_nm"]),
                              "reach_050": float(entry["reach_050"]),
                              "reach_025": float(entry["reach_025"]),
                              "radius": float(entry["radius_floor"]),
                              "n_basis": int(prior.n_basis)}
               for user, entry in zip(users, entries)}
    _write(run / "users" / f"{fold}-{label}.json", sidecar)
    return sidecar


def drift_sidecar(run: Path, fold: str, users: list, target_nm: float, *, n_starts: int,
                  workers: int) -> dict:
    """Every user's drift to ``target_nm`` and the ProMP data prior's floor at its end."""
    prior = DataPrior.load(run / "priors" / f"{fold}-promp_s0")
    label = drift_label(target_nm)
    shifts = {user.user_id: shift_for_displacement(user.curve_nm, target_nm) for user in users}
    moved = [dataclasses.replace(user, curve_nm=time_shift(user.curve_nm,
                                                           shifts[user.user_id][0]))
             for user in users]
    family = f"promp_s0-{label}"
    entries = user_floors(prior.design(), prior.whitening(), moved, n_starts=n_starts,
                          seed_sequence=fold_seed(fold, label), workers=workers,
                          cache=FloorCache(run / "floors" / f"{fold}-{label}.json"),
                          family=family, prior_hash=prior_digest(prior.mu, prior.sigma))
    sidecar = {user.user_id: {"floor_nm": float(entry["floor_nm"]),
                              "shift_pct": float(shifts[user.user_id][0]),
                              "capped": bool(shifts[user.user_id][1]),
                              "target_nm": float(target_nm)}
               for user, entry in zip(users, entries)}
    _write(run / "users" / f"{fold}-{label}.json", sidecar)
    return sidecar


def magnitude_sidecar(run: Path, fold: str, users: list, target_nm: float, *, n_starts: int,
                      workers: int) -> dict:
    """Every user's magnitude drift to ``target_nm`` and the ProMP data prior's floor for the
    final ideal ``curve * (1 + scale)``."""
    prior = DataPrior.load(run / "priors" / f"{fold}-promp_s0")
    label = magnitude_label(target_nm)
    scales = {user.user_id: scale_for_displacement(user.curve_nm, target_nm) for user in users}
    grown = [dataclasses.replace(user, curve_nm=user.curve_nm * (1.0 + scales[user.user_id]))
             for user in users]
    family = f"promp_s0-{label}"
    entries = user_floors(prior.design(), prior.whitening(), grown, n_starts=n_starts,
                          seed_sequence=fold_seed(fold, label), workers=workers,
                          cache=FloorCache(run / "floors" / f"{fold}-{label}.json"),
                          family=family, prior_hash=prior_digest(prior.mu, prior.sigma))
    sidecar = {user.user_id: {"floor_nm": float(entry["floor_nm"]),
                              "scale": float(scales[user.user_id]),
                              "target_nm": float(target_nm)}
               for user, entry in zip(users, entries)}
    _write(run / "users" / f"{fold}-{label}.json", sidecar)
    return sidecar
