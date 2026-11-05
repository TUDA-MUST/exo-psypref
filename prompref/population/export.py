"""Held-out users on disk, so Phase 5 sessions never need the Molinaro drive.

A Phase 5a fold file holds each user's factor draw, answer seed, floors and reach radii,
but not the curve, and rebuilding a curve needs the raw trials on S:. The export runs once
per fold:
1. regenerate the fold's users from its own stream and config;
2. check that every regenerated draw and answer seed is the fold file's;
3. write the curves beside them.

A fold whose population code has changed since it ran is refused, because its users can no
longer be regenerated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..priors.data_prior import FAMILIES, population_source_hashes
from ..validation.stamp import ROOT, source_hash
from .factors import FactorSwitches, ShiftSpread, TimingSpread
from .folds import fold_streams
from .users import make_users

EXPORT_SCHEMA = "users-v1"

#: Per-family values copied from the fold file for every user.
FAMILY_FIELDS = ("floor_nm", "reach_050", "reach_025", "radius")


def run_source_hashes() -> dict[str, str]:
    """The source hashes a Phase 5a fold config records (the diagnostics' run_source_hashes)."""
    hashes = population_source_hashes()
    hashes["prompref/population/users.py"] = source_hash(ROOT / "prompref/population/users.py")
    return hashes


def population_from_config(config: dict) -> tuple[ShiftSpread, FactorSwitches, TimingSpread]:
    timing = config["timing"]
    return (ShiftSpread(**config["shift"]), FactorSwitches(**config["switches"]),
            TimingSpread(np.asarray(timing["scale_matrix"], dtype=float), float(timing["nu"]),
                         timing["source"], timing["file_sha256"]))


def _as_json(obj):
    return json.loads(json.dumps(obj))


def export_fold_users(result: dict, trials, out_dir) -> Path:
    """Write ``OUT/users/<fold>.npz`` (curves) and ``OUT/users/<fold>.json`` (the rest).

    ``result`` is a Phase 5a fold file's content; ``trials`` are its held-out trials, loaded
    as the diagnostics loaded them. Returns the JSON path.
    """
    config, fold = result["config"], result["fold"]
    current = run_source_hashes()
    if config["source_hashes"] != current:
        changed = sorted(name for name in set(config["source_hashes"]) | set(current)
                         if config["source_hashes"].get(name) != current.get(name))
        raise ValueError(f"{fold}: population code changed since this fold ran ({changed}); "
                         f"its users cannot be regenerated")
    shift, switches, timing = population_from_config(config)
    users = make_users(fold, trials, n_population=int(config["n_population"]), switches=switches,
                       timing=timing, seed_sequence=fold_streams(int(config["seed"]), fold)["users"],
                       shift=shift)
    rows = {(row["family"], row["user_id"]): row for row in result["users"]}
    if len(rows) != len(users) * len(FAMILIES):
        raise ValueError(f"{fold}: the fold file has {len(rows)} user rows, but {len(users)} users "
                         f"x {len(FAMILIES)} families were regenerated")
    records = []
    for user in users:
        draw = _as_json(user.draw.to_dict())
        families = {}
        for family in FAMILIES:
            row = rows.get((family, user.user_id))
            if row is None:
                raise ValueError(f"{fold}: {user.user_id} has no {family} row in the fold file")
            if row["answer_seed"] != user.answer_seed or row["draw"] != draw:
                raise ValueError(f"{fold}: regenerated {user.user_id} differs from the fold file")
            families[family] = {name: float(row[name]) for name in FAMILY_FIELDS}
        records.append(dict(user_id=user.user_id, kind=user.kind, subject=user.subject,
                            side=user.side, answer_seed=int(user.answer_seed), draw=draw,
                            families=families))
    out = Path(out_dir) / "users"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{fold}.npz", curves=np.array([user.curve_nm for user in users]))
    path = out / f"{fold}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(schema=EXPORT_SCHEMA, fold=fold, config=config, users=records),
                              indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


@dataclass(frozen=True, eq=False)
class ExportedUser:
    user_id: str
    kind: str
    subject: str
    side: str
    answer_seed: int
    draw: dict
    families: dict
    curve_nm: np.ndarray


def load_fold_users(run_dir, fold: str) -> dict[str, ExportedUser]:
    """A fold's exported users by id, in export order."""
    base = Path(run_dir) / "users"
    meta = json.loads((base / f"{fold}.json").read_text(encoding="utf-8"))
    if meta.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"{base / fold}.json has schema {meta.get('schema')!r}, "
                         f"expected {EXPORT_SCHEMA!r}")
    curves = np.load(base / f"{fold}.npz")["curves"]
    if len(curves) != len(meta["users"]):
        raise ValueError(f"{fold}: {len(curves)} curves for {len(meta['users'])} users")
    return {record["user_id"]: ExportedUser(curve_nm=curves[i], **record)
            for i, record in enumerate(meta["users"])}
