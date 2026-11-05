"""ICORR W1 and the cheap part of W2: JND-free outcomes from the stored sessions.

sections 4 (W1, W2)
and 5 (Measures). No session is run.

Usage:
    python scripts/icorr_jnd_free.py [--out <project>/out]
        [--data <Schaefer share>] [--boot 2000] [--report <out>/icorr/jnd_free.md]

Reads, all read-only: OUT/phase5e/{e-main,e-s,e-observer,e-noise,e-drift,e-lapse}.jsonl,
OUT/phase5c/main-promp.jsonl, OUT/phase5d/{cmaes,lens-franks}.jsonl (each streamed line by line,
population rows only, only the fields used here), the simulated users and priors under
OUT/phase5-schaefer, and the Schaefer self-tuning logs (``NN-selfOpt.mat`` only, through
schaefer_grounding.load_selfopt; aggregates only, no per-participant number is written).

Cells are phase5e_report's (``CONDS``, ``DEFAULTS``, ``_match``, ``SCOPES``); the s = 0.5
reference is the 5c/5d rows truncated, selected as short_budget_report.Rows selects them.
Per user i and comparison k (identified assistance = the curve the optimizer would deliver now):

  win rate     P_i(k) = lambda_i / 2 + (1 - lambda_i) Phi((g_i - e_i(k)) / s_i), the probability
               that the cell's wearer prefers the identified assistance over generic assistance
               in one duel. g_i, e_i(k): the generic and identified distances in the metric the
               wearer answers by (RMS ``d_prior`` and ``distances``; ``own_distances`` and the
               rebuilt own generic distance in the landmark, asymmetry and familiarity cells).
  share closed c_i(k) = (d_prior_i - d_i(k)) / (d_prior_i - d_floor_i), RMS.
  error        d_i(k) = distances[k - 1], RMS Nm to the ideal.
  within m JND d_i(k) - d_floor_i <= m * JND_cell (+1e-12), m = 1, 2; JND_cell as phase5e_report.

Rooms: as stored (simulated), and reweighted with the G1b / literature-room KDE weights
(schaefer_grounding.kde_weights on each user's ProMP room d_prior - d_floor of e-main A) to the
Schaefer timing-only anchor (kappa * d_loo) and to the literature-informed central and high
rooms (d_loo rescaled to the room level), exactly as preference_literature_room.py does, with its
draw construction (seed 20261009, sim_draws then part_draws). The room levels are recomputed
here from that module's literature records (its steps a-d), not typed in. Before anything new is
reported, A's Q>=1 at k = 24 must reproduce 0.48 at the anchor and 0.70 at the central room.

Section 9 (added for the W3/W5 analysis) gives the main condition at one extra room, the
median-matched central room of grounding our design notes (1.47 Nm, recomputed from
preference_literature_room's Monte Carlo draws); A's Q>=1 at k = 24 must reproduce 0.59 there.
Every other section is unchanged by it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import benchmark_report as bm  # noqa: E402
import phase5c_report as c5r  # noqa: E402
import phase5e_own_generic as og  # noqa: E402
import phase5e_report as er  # noqa: E402
import preference_literature_room as plr  # noqa: E402
import schaefer_grounding as G  # noqa: E402
import short_budget_report as sb  # noqa: E402

from prompref.experiment import phase5 as p5  # noqa: E402
from prompref.experiment.phase5 import Phase5Point  # noqa: E402
from prompref.user.drift import path_for_scale  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.oracle import rms_distance  # noqa: E402

KMAX = 40
KS = (12, 24, 40)
K_GATE = 24
N_BOOT = 2000
SEED = plr.SEED_SHARED  # 20261009: the literature-room draws, so the gate reproduces
TOL = bm.TOL
ROOMS = ("simulated", "anchor", "central", "high")
#: Reproduction targets (checks, not results): optimizers_real_room.md via
#: preference_literature_room.ANCHOR_EXPECTED, and grounding our design notes for the central room.
GATE_ANCHOR = plr.ANCHOR_EXPECTED[("ProMP", "A")][:3]
GATE_CENTRAL = (0.699, 0.538, 0.825)
GATE_TOL = 0.01
#: The median-matched central room: an extra room level,
#: reported for the main condition only (section 9). ROOMS and every table built on it are
#: unchanged. Reproduction target (a check, not a result): A's Q>=1 at k = 24 there,
#: literature_room.md "central, median-matched (Monte Carlo median)", 0.59 [0.36, 0.72].
EXTRA_ROOMS = ("median",)
ALL_ROOMS = ROOMS + EXTRA_ROOMS
GATE_MEDIAN = (0.59, 0.36, 0.72)
GATE_MEDIAN_LEVEL = 1.47
#: Phase 5e our design notes and ICORR our design notes, printed beside the recomputed values (not gated).
SANITY_5E = {"A error k=24": 1.712, "A Q_opt (= within 2 JND) k=24": 0.72, "A Q>=1 k=24": 0.82,
             "A error k=12": 2.009}

#: Reading levels for the first k (fixed before the tables were read): (measure, direction, level).
LEVELS = (("win", ">=", 0.70), ("win", ">=", 0.75), ("win", ">=", 0.80),
          ("share", ">=", 0.25), ("share", ">=", 0.50),
          ("err", "<=", 2.5), ("err", "<=", 2.0),
          ("jnd1", ">=", 0.25), ("jnd2", ">=", 0.50), ("jnd2", ">=", 0.75))
MEASURE_NAME = {"win": "mean win rate", "maj3": "P(at least 2 of 3 won)",
                "share": "median share closed", "err": "median error (Nm)",
                "jnd1": "share within 1 JND", "jnd2": "share within 2 JND"}
CONTRAST_MEASURES = ("win", "share", "err", "jnd1", "jnd2")
OURS = {"promp_s0": "a", "franks": "a_lens"}

#: Row fields kept from the jsonl rows: cell identity and filters, and the values used here.
KEEP = ("fold", "user_id", "user_kind", "family", "learner", "arm", "prior", "n_candidates",
        "lapse", "s", "s_user", "s_user_logsd", "observer_metric", "asymmetry",
        "familiarity_weight", "device_noise_nm", "drift_nm", "drift_axis",
        "landmark_mag_fraction", "order_bias", "overdispersion", "plateau_nm",
        "plateau_over_floor_jnd", "n_comparisons", "d_prior", "d_floor", "final_floor_nm",
        "drift_scale")
LISTS = ("distances", "own_distances")
E_BLOCKS = ("e-main", "e-s", "e-observer", "e-noise", "e-drift", "e-lapse")

#: How each cell's win rate relates to the wearer that answered the session.
CELL_NOTE = {
    "s05": "exact (RMS wearer, s = 0.5); 5c/5d rows truncated",
    "main": "exact (RMS wearer)",
    "s2": "exact (RMS wearer, s = 2.0)",
    "het": "exact (RMS wearer at each user's own s_user)",
    "landmark": "exact in the wearer's own metric (own_distances, own generic distance rebuilt)",
    "landmark25": "exact in the wearer's own metric (own_distances, own generic distance rebuilt)",
    "weber": "approximate: own distance = RMS, but the probit scale is s, not the per-duel "
             "Weber scale s_eff (the identified curves are not stored)",
    "asym": "exact in the wearer's own metric (own_distances, own generic distance rebuilt)",
    "famil": "exact in the wearer's own metric (own_distances, own generic distance rebuilt)",
    "noise": "approximate: noise-free curves (the device noise of a real validation duel is "
             "not modelled)",
    "drift1": "k = 24 only (end of the n = 24 session), against the final ideal: generic distance "
              "to the final ideal rebuilt, final floor",
    "drift2": "k = 24 only (end of the n = 24 session), against the final ideal: generic distance "
              "to the final ideal rebuilt, final floor",
    "lapse2": "exact (RMS wearer, lapse 0.2)",
    "lapse3": "exact (RMS wearer, lapse 0.3)",
}
OWN_CELLS = ("landmark", "landmark25", "asym", "famil")


# ============================================================================ streaming
def _pop_row(r: dict) -> bool:
    return (not r.get("error") and r.get("user_kind") == "population"
            and "/pop" in str(r.get("user_id", "")))


def stream(path: Path, keep=_pop_row, extra=None, fields=KEEP, lists=LISTS) -> pd.DataFrame | None:
    """The rows of a jsonl file that pass ``keep``, read line by line, holding only ``fields``,
    the per-comparison ``lists`` truncated to KMAX, and the scalars ``extra(row)`` returns
    (computed from the full row before it is dropped)."""
    if not path.exists():
        return None
    out = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(f"{path}: skipping unparseable line {number} ({exc})")
                continue
            if not keep(r):
                continue
            slim = {f: r[f] for f in fields if f in r}
            for name in lists:
                if isinstance(r.get(name), list):
                    slim[name] = [float(x) for x in r[name][:KMAX]]
            if extra is not None:
                slim.update(extra(r))
            out.append(slim)
    return pd.DataFrame(out) if out else None


def _point(r: dict) -> Phase5Point:
    return Phase5Point(**{f: r[f] for f in og.FIELDS if f in r})


def own_generic_extra(r: dict) -> dict:
    """phase5e_own_generic.collect: the wearer's own distance of the generic assistance (the
    prior's mean curve), and the check that this curve reproduces the stored RMS d_prior."""
    if r.get("observer_metric", "rms") == "rms" and r.get("asymmetry", 1.0) == 1.0 \
            and not r.get("familiarity_weight"):
        return {}
    point = _point(r)
    user = p5._users(point.run_dir, point.fold)[point.user_id]
    observer, _ = p5.build_observer(point, user, s_user=r["s_user"], plateau=point.plateau_nm)
    generic = og.generic_curve(point)
    return {"own_generic": observer.own_distance(generic),
            "generic_check": abs(rms_distance(generic, user.curve_nm) - float(r["d_prior"]))}


def drift_extra(r: dict) -> dict:
    """A drift row's generic distance to the final ideal, ``path_for_scale(...)[-1]``."""
    point = _point(r)
    user = p5._users(point.run_dir, point.fold)[point.user_id]
    generic = og.generic_curve(point)
    final = path_for_scale(user.curve_nm, float(r["drift_scale"]), int(r["n_comparisons"]))[-1]
    return {"generic_final": rms_distance(generic, final),
            "generic_check": abs(rms_distance(generic, user.curve_nm) - float(r["d_prior"]))}


def load_blocks(out: Path) -> dict:
    blocks = {}
    for name in E_BLOCKS:
        extra = own_generic_extra if name == "e-observer" else drift_extra if name == "e-drift" \
            else None
        df = stream(out / "phase5e" / f"{name}.jsonl", extra=extra)
        blocks[name] = None if df is None else sb._n2048(sb._popn(df))
    blocks["5c/main-promp"] = stream(out / "phase5c" / "main-promp.jsonl")
    blocks["5d/cmaes"] = stream(out / "phase5d" / "cmaes.jsonl")
    blocks["5d/lens-franks"] = stream(out / "phase5d" / "lens-franks.jsonl")
    return blocks


# ============================================================================ cells
def select(blocks: dict, cond, family: str, learner: str):
    """phase5e_report.Ctx._find on the slim frames (and short_budget_report.Rows for s = 0.5)."""
    if cond.cid == "s05":
        if family == "promp_s0" and learner != "cmaes":
            df = sb._n2048(sb._popn(c5r._reference(c5r._cell(
                blocks["5c/main-promp"], learner=learner, family=family, arm="tight",
                prior="data"))))
        elif learner == "cmaes":
            df = sb._popn(c5r._reference(c5r._cell(blocks["5d/cmaes"], family=family,
                                                   prior="data")))
        else:
            df = sb._n2048(sb._popn(c5r._reference(c5r._cell(blocks["5d/lens-franks"],
                                                             learner=learner))))
        return df if df is not None and len(df) else None
    df = blocks[cond.block]
    if df is None:
        return None
    mask = (df["family"] == family).to_numpy() & (df["learner"] == learner).to_numpy()
    want = dict(er.DEFAULTS)
    want.update(cond.fields)
    for name, value in want.items():
        mask &= er._match(df, name, value)
    out = df[mask].drop_duplicates(er.KEY, keep="last")
    return out if len(out) else None


def in_order(df: pd.DataFrame, keys) -> pd.DataFrame:
    """The rows in the 200-user order of load_sim (schaefer_grounding_optimizers.cell)."""
    if df.duplicated(er.KEY).any():
        raise ValueError("a cell holds more than one row per user")
    idx = {(f, u): i for i, (f, u) in enumerate(zip(df["fold"], df["user_id"]))}
    missing = [k for k in keys if k not in idx]
    if missing:
        raise ValueError(f"{len(missing)} of the 200 population users are missing from a cell")
    return df.iloc[[idx[k] for k in keys]].reset_index(drop=True)


class Arm:
    """One optimizer in one cell, aligned to the 200 users."""

    def __init__(self, cond, family, learner, df):
        self.cond, self.family, self.learner = cond, family, learner
        self.name = er._label(family, learner)
        self.fam_name = er.FAMILY_NAME[family]
        n_k = int(df["distances"].map(len).min())
        drift = cond.cid.startswith("drift")
        self.kmax = min(KMAX, n_k)
        self.ks = (24,) if drift else tuple(k for k in KS if k <= self.kmax)
        self.D = np.array([d[:self.kmax] for d in df["distances"]], dtype=float)
        self.dp = df["d_prior"].astype(float).to_numpy()
        self.floor = df["d_floor"].astype(float).to_numpy()
        s = df["s"].astype(float).to_numpy()
        if "s_user" in df and df["s_user"].notna().all():
            s_user = df["s_user"].astype(float).to_numpy()
            if cond.cid != "het" and not np.allclose(s_user, s, atol=1e-12):
                raise ValueError(f"{cond.cid}: s_user differs from s outside the heterogeneous cell")
            s = s_user
        self.s = s
        self.lam = df["lapse"].astype(float).to_numpy()
        self.jnd = float(cond.jnd)
        self.generic_check = None
        # the wearer's metric for the win rate
        self.E, self.gen = self.D, self.dp
        if cond.cid in OWN_CELLS:
            self.E = np.array([d[:self.kmax] for d in df["own_distances"]], dtype=float)
            self.gen = df["own_generic"].astype(float).to_numpy()
            self.generic_check = float(df["generic_check"].max())
        if drift:
            # validation at the end of the session, against the final ideal
            self.gen = df["generic_final"].astype(float).to_numpy()
            self.dp = self.gen
            self.floor = df["final_floor_nm"].astype(float).to_numpy()
            self.generic_check = float(df["generic_check"].max())
        self.folds = df["fold"].to_numpy()

    def per_user(self, k: int) -> dict:
        """Per-user values at comparison k (1-based)."""
        e, d = self.E[:, k - 1], self.D[:, k - 1]
        p = win_probability(self.gen, e, self.s, self.lam)
        room = self.dp - self.floor
        ok = room > bm.NO_ROOM
        share = np.where(ok, (self.dp - d) / np.where(ok, room, 1.0), np.nan)
        over = d - self.floor
        return {"win": p, "maj3": majority_of_three(p), "share": share, "err": d,
                "jnd1": (over <= self.jnd + TOL).astype(float),
                "jnd2": (over <= 2 * self.jnd + TOL).astype(float)}


def win_probability(generic, identified, s, lam):
    """P(identified preferred over generic) for the probit wearer with lapse lam (oracle.py)."""
    return lam / 2.0 + (1.0 - lam) * norm.cdf((np.asarray(generic) - np.asarray(identified))
                                               / np.asarray(s))


def majority_of_three(p):
    """P(at least 2 of 3 independent duels won) = p^3 + 3 p^2 (1 - p)."""
    p = np.asarray(p, dtype=float)
    return p ** 3 + 3 * p ** 2 * (1 - p)


# ============================================================================ statistics
def wquantile(x, w, q):
    """Weighted Hazen quantile along the last axis: each value's mass is centred at its
    cumulative-weight midpoint, (C_j - w_j / 2) / sum(w), with linear interpolation between and
    clipping at the ends. NaN values get no weight. With equal weights it equals
    np.percentile(..., method="hazen"), whose median is the usual median."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    w = np.atleast_2d(np.asarray(w, dtype=float))
    w = np.where(np.isnan(x), 0.0, np.broadcast_to(w, x.shape))
    x = np.where(np.isnan(x), np.inf, x)
    order = np.argsort(x, axis=-1, kind="stable")
    xs = np.take_along_axis(x, order, axis=-1)
    ws = np.take_along_axis(w, order, axis=-1)
    c = np.cumsum(ws, axis=-1)
    p = np.where(ws > 0, (c - ws / 2.0) / c[..., -1:], np.inf)
    m = (ws > 0).sum(axis=-1)
    j = (p < q).sum(axis=-1)
    rows = np.arange(x.shape[0])
    lo, hi = np.clip(j - 1, 0, m - 1), np.clip(j, 0, m - 1)
    plo, phi = p[rows, lo], p[rows, hi]
    xlo, xhi = xs[rows, lo], xs[rows, hi]
    t = np.where(phi > plo, np.clip((q - plo) / np.where(phi > plo, phi - plo, 1.0), 0, 1), 0.0)
    return xlo + t * (xhi - xlo)


def wmean(x, w):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    w = np.broadcast_to(np.atleast_2d(np.asarray(w, dtype=float)), x.shape)
    return (w * x).sum(axis=-1) / w.sum(axis=-1)


QUANTILE_MEASURES = ("share", "err")


class Rooms:
    """Per-user weights at every room, for the original users and for every bootstrap draw.
    The rooms are ROOMS, then each of EXTRA_ROOMS that ``targets_point`` holds."""

    def __init__(self, room, targets_point, targets_draw, sim_draws):
        self.sim_draws = sim_draws
        self.point, self.draw, self.ess = {}, {}, {}
        n_boot = sim_draws.shape[0]
        for name in ROOMS + tuple(r for r in EXTRA_ROOMS if r in targets_point):
            if name == "simulated":
                self.point[name] = np.ones(len(room))
                self.draw[name] = np.ones(sim_draws.shape)
            else:
                self.point[name] = G.kde_weights(room, targets_point[name])
                self.draw[name] = np.array([G.kde_weights(room[sim_draws[b]], targets_draw[name][b])
                                            for b in range(n_boot)])
            ess_b = [G.ess(w) for w in self.draw[name]]
            self.ess[name] = (G.ess(self.point[name]), *np.percentile(ess_b, [2.5, 97.5]))

    def stat(self, measure: str, x, room: str, q: float = 0.5):
        """(estimate, bootstrap values) of ``measure``'s population statistic of x."""
        w, wd = self.point[room], self.draw[room]
        xd = np.asarray(x, dtype=float)[self.sim_draws]
        if measure in QUANTILE_MEASURES:
            return float(wquantile(x, w, q)[0]), wquantile(xd, wd, q)
        return float(wmean(x, w)[0]), wmean(xd, wd)


def ci(boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def first_k(curve, direction: str, level: float):
    """Smallest k (1-based) whose cross-sectional statistic meets the level; inf if none. Not
    sustained and not a per-user first hit (as benchmark_report.n90)."""
    curve = np.atleast_2d(curve)
    ok = curve >= level - TOL if direction == ">=" else curve <= level + TOL
    hit = ok.any(axis=-1)
    return np.where(hit, ok.argmax(axis=-1) + 1, np.inf)


def inverted_cdf_ci(values):
    """2.5 / 97.5 % order statistics (inverted CDF), safe with inf."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    return float(v[math.ceil(0.025 * n) - 1]), float(v[math.ceil(0.975 * n) - 1])


# ============================================================================ the rooms
def literature_levels(logs: dict, P: np.ndarray, kappa: float, sim_med: float) -> dict:
    """The central and high literature rooms, preference_literature_room.main steps a-d (its
    lines on the Schaefer SDs, the G2 conversion, the magnitude CVs and ``lvl``), recomputed
    from that module's literature records."""
    fids10 = [f for f in G.FILE_IDS if f != G.NO_TUNING_LOG]
    idx10 = [G.FILE_IDS.index(f) for f in fids10]
    sd11 = P.std(axis=0, ddof=1)
    sd10 = P[idx10].std(axis=0, ddof=1)
    late = np.array([G.late_sd(logs[f]) for f in fids10])
    within = np.sqrt(np.mean(late ** 2, axis=0))
    defl10 = np.array([plr.deflate(b, w) for b, w in zip(sd10, within)])

    def conv(k, sd):
        if sd <= 0:
            return 0.0
        return kappa * float(np.median([G.landmark_shift_rms(t, k, sd) for t in P]))

    t_raw = float(np.sqrt(sum(conv(k, sd11[k]) ** 2 for k in range(4))))
    t_defl = float(np.sqrt(sum(conv(k, defl10[k]) ** 2 for k in range(4))))
    rows = []
    for e in plr.MAGNITUDE:
        per = []
        for _, v in e["params"]:
            if isinstance(v, list):
                mean, b, w, w_cv = float(np.mean(v)), float(np.std(v, ddof=1)), None, None
            else:
                mean, b, w, w_cv = v["mean"], v["sd"], v.get("w"), v.get("w_cv")
                if "icc_k" in v:
                    r_, k_ = v["icc_k"], v["k"]
                    icc1 = r_ / (k_ - (k_ - 1) * r_)
                    w = float(np.sqrt((1.0 - icc1) * b * b))
            if w is None and w_cv is not None:
                w = w_cv * mean
            if w is not None:
                cv_defl = plr.deflate(b, w) / mean
                cv_single = np.sqrt(b * b + w * w * (1 - 1 / e["m"])) / mean
            else:
                cv_defl = cv_single = None
            per.append(dict(cv=b / mean, cv_defl=cv_defl, cv_single=cv_single))

        def pool(key, per=per):
            vals = [q[key] for q in per]
            if any(v is None for v in vals):
                return None
            return float(np.sqrt(np.mean(np.square(vals))))
        cv_raw, cv_d = pool("cv"), pool("cv_defl")
        rows.append(dict(room=e["room"], cv_defl=cv_d,
                         cv_room=cv_d if cv_d is not None else cv_raw,
                         cv_single=pool("cv_single") if cv_d is not None else cv_raw))
    studies = [r for r in rows if r["room"]]
    cv_central = float(np.median([r["cv_room"] for r in studies]))
    cv_high = float(np.median([r["cv_single"] for r in studies if r["cv_defl"] is not None]))

    def lvl(t, cv):
        m = cv * sim_med
        return dict(T=t, cv=cv, M=m, L=float(np.hypot(t, m)))
    return {"central": lvl(t_defl, cv_central), "high": lvl(t_raw, cv_high)}


def ramella_cv_raw() -> float:
    """Ramella's pooled between-person magnitude CV, preference_literature_room's ``cv_raw``
    (the low room's CV): the RMS over its peak-torque parameters of SD / mean."""
    entry = next(e for e in plr.MAGNITUDE if e["study"] == "Ramella 2025")
    cvs = []
    for _, v in entry["params"]:
        if isinstance(v, list):
            cvs.append(float(np.std(v, ddof=1)) / float(np.mean(v)))
        else:
            cvs.append(v["sd"] / v["mean"])
    return float(np.sqrt(np.mean(np.square(cvs))))


def median_matched_central(logs: dict, P: np.ndarray, kappa: float, cv_central: float) -> dict:
    """The median-matched central room, preference_literature_room.main's ``mc["central"]``: the
    Monte Carlo median of kappa x the RMS distance between each Schaefer analytic preferred
    profile and a copy with Gaussian landmark shifts (Schaefer timing SDs deflated by late
    exploration, n = 10) and an amplitude scaling with the central CV; draws that break the
    cyclic order are rejected. That module draws its Monte Carlo rooms from one stream (seed
    SEED_OWN + 7) in a fixed order, "Schaefer raw, timing only", then "low", then "central";
    the first two are replayed here (draws only, distances not computed) so that the central
    room uses exactly its draws."""
    fids10 = [f for f in G.FILE_IDS if f != G.NO_TUNING_LOG]
    idx10 = [G.FILE_IDS.index(f) for f in fids10]
    sd11 = P.std(axis=0, ddof=1)
    sd10 = P[idx10].std(axis=0, ddof=1)
    late = np.array([G.late_sd(logs[f]) for f in fids10])
    within = np.sqrt(np.mean(late ** 2, axis=0))
    defl10 = np.array([plr.deflate(b, w) for b, w in zip(sd10, within)])
    ram_sds = [(plr.MOVE[e["map"]], plr.sd_of(e)) for e in plr.TIMING
               if e["study"] == "Ramella 2025"]
    rng = np.random.default_rng(plr.SEED_OWN + 7)

    def mc_median(sds_by_k, cv, n_per=400, keep=True):
        out = []
        for t in P:
            base = G.analytic_profile(t, G.PHASE_GRID) if keep else None
            for _ in range(n_per):
                q = np.array(t, float)
                for k, s in sds_by_k:
                    q[k] += rng.normal(0.0, s)
                cyc = np.array([q[1], q[2], q[3], q[0], q[1] + 100.0])
                if not np.all(np.diff(cyc) > 0):
                    continue
                amp = 1.0 + rng.normal(0.0, cv)
                if keep:
                    out.append(kappa * G.rms_distance(amp * G.analytic_profile(q, G.PHASE_GRID),
                                                      base))
        return float(np.median(out)) if keep else None

    mc_median(list(enumerate(sd11)), 0.0, keep=False)      # "Schaefer raw, timing only"
    mc_median(ram_sds, ramella_cv_raw(), keep=False)        # "low"
    level = mc_median(list(enumerate(defl10)), cv_central)  # "central"
    return dict(L=level, cv=cv_central)


def room_setup(out: Path, data: Path, n_boot: int, room: np.ndarray, folds) -> dict:
    """The targets of the three reweighted rooms and the shared draws of
    preference_literature_room.main (same seed, same order)."""
    logs = G.load_participants(data)
    P = np.array([logs[f]["t_pref"] for f in G.FILE_IDS])
    real = np.array([G.analytic_profile(t, G.PHASE_GRID) for t in P])
    d_loo = np.array([G.rms_distance(real[i], np.delete(real, i, axis=0).mean(axis=0))
                      for i in range(len(real))])
    sim = G.load_sim(out / "phase5-schaefer")
    sim_rms = np.array([G.rms(c) for c in sim["curves"]])
    real_rms = np.array([G.rms(c) for c in real])
    sim_med = float(np.median(sim_rms))
    kappa = float(sim_med / np.median(real_rms))
    levels = literature_levels(logs, P, kappa, sim_med)
    levels["median"] = median_matched_central(logs, P, kappa, levels["central"]["cv"])
    rng = np.random.default_rng(SEED)
    sim_draws = G.fold_draws(folds, n_boot, rng)
    part_draws = rng.integers(0, len(d_loo), size=(n_boot, len(d_loo)))
    kappa_b = np.median(sim_rms[sim_draws], axis=1) / np.median(real_rms[part_draws], axis=1)
    tp = {"anchor": kappa * d_loo}
    td = {"anchor": [kappa_b[b] * d_loo[part_draws[b]] for b in range(n_boot)]}
    for name in ("central", "high") + EXTRA_ROOMS:
        L = levels[name]["L"]
        tp[name] = d_loo * L / np.median(d_loo)
        td[name] = [d_loo[part_draws[b]] * L / np.median(d_loo[part_draws[b]])
                    for b in range(n_boot)]
    level = {"anchor": float(np.median(tp["anchor"])), "central": levels["central"]["L"],
             "high": levels["high"]["L"], "median": levels["median"]["L"]}
    return dict(sim=sim, kappa=kappa, sim_med=sim_med, levels=levels, level=level,
                rooms=Rooms(room, tp, td, sim_draws), n_part=len(d_loo))


# ============================================================================ computation
def compute_arm(arm: Arm, rooms: Rooms, room_names=ROOMS) -> dict:
    """Every statistic of one optimizer: at the reporting ks with CIs, the IQR at those ks, the
    curves k = 1..kmax (estimate and per-draw) for the first-k levels, and the per-draw values
    at the reporting ks for the contrasts, at each room of ``room_names``."""
    res = {"at": {}, "iqr": {}, "boot": {}, "first": {}}
    curve_k = range(1, arm.kmax + 1) if len(arm.ks) > 1 else arm.ks
    curves = {room: {m: ([], []) for m in MEASURE_NAME} for room in room_names}
    for k in curve_k:
        vals = arm.per_user(k)
        for room in room_names:
            for m, x in vals.items():
                est, boot = rooms.stat(m, x, room)
                curves[room][m][0].append(est)
                curves[room][m][1].append(boot)
                if k in arm.ks:
                    res["at"][(room, m, k)] = (est, *ci(boot))
                    res["boot"][(room, m, k)] = boot.astype(np.float32)
                    if m in QUANTILE_MEASURES:
                        w = rooms.point[room]
                        res["iqr"][(room, m, k)] = (float(wquantile(x, w, 0.25)[0]),
                                                   float(wquantile(x, w, 0.75)[0]))
    if len(arm.ks) > 1:
        for room in room_names:
            for m, direction, level in LEVELS:
                est_curve = np.array(curves[room][m][0])
                boot_curve = np.array(curves[room][m][1]).T  # (draws, k)
                point = float(first_k(est_curve, direction, level)[0])
                draws = first_k(boot_curve, direction, level)
                res["first"][(room, m, direction, level)] = (point, *inverted_cdf_ci(draws),
                                                            float(np.isfinite(draws).mean()))
    return res


GENERIC_MEASURES = ("err", "jnd1", "jnd2")


def compute_generic(arm: Arm, rooms: Rooms, room_names=ROOMS) -> dict:
    """Generic assistance itself (k = 0) in the family of ``arm``: its error (d_prior) and the
    share of users for whom it already lies within 1 or 2 JND of the family optimum. Its win rate
    is 0.5 and its share closed 0 by definition."""
    over = arm.dp - arm.floor
    vals = {"err": arm.dp, "jnd1": (over <= arm.jnd + TOL).astype(float),
            "jnd2": (over <= 2 * arm.jnd + TOL).astype(float)}
    res = {"at": {}, "iqr": {}}
    for room in room_names:
        for m, x in vals.items():
            est, boot = rooms.stat(m, x, room)
            res["at"][(room, m)] = (est, *ci(boot))
            if m in QUANTILE_MEASURES:
                w = rooms.point[room]
                res["iqr"][(room, m)] = (float(wquantile(x, w, 0.25)[0]),
                                         float(wquantile(x, w, 0.75)[0]))
    return res


def generic_meets(gen: dict, room: str, measure: str, direction: str, level: float) -> bool:
    if measure in ("win", "maj3"):
        value = 0.5
    elif measure == "share":
        value = 0.0
    else:
        value = gen["at"][(room, measure)][0]
    return value >= level - TOL if direction == ">=" else value <= level + TOL


def gate(arm_a: Arm, rooms: Rooms, room_names=ROOMS) -> dict:
    """A's Q>=1 at k = 24 (benchmark_report._flags, true_jnd(1.0)) at every room."""
    d = arm_a.D[:, K_GATE - 1]
    q1 = ((arm_a.dp - d) / G.JND[1.0] >= 1.0 - G.TOL).astype(float)
    out = {}
    for room in room_names:
        est, boot = rooms.stat("q1", q1, room)
        out[room] = (est, *ci(boot))
    return out


# ============================================================================ W2
W2_FIELDS = ("zstar_sd_max", "lambda_sd", "ell_mean", "lambda_mean", "lambda_identified",
             "ess_final", "n_resample", "acceptance_mean")
W2_LISTS = ("distances", "ell_trace", "lambda_trace")


def w2_rows(out: Path, population_only: bool):
    keep = (lambda r: _pop_row(r) and r.get("family") == "promp_s0" and r.get("learner") == "a") \
        if population_only else \
        (lambda r: not r.get("error") and r.get("family") == "promp_s0" and r.get("learner") == "a")
    df = stream(out / "phase5e" / "e-main.jsonl", keep=keep, fields=KEEP + W2_FIELDS,
                lists=W2_LISTS)
    cond = er.BY_ID["main"]
    mask = np.ones(len(df), dtype=bool)
    want = dict(er.DEFAULTS)
    want.update(cond.fields)
    for name, value in want.items():
        mask &= er._match(df, name, value)
    return df[mask].drop_duplicates(er.KEY, keep="last")


def w2_fields_present(out: Path) -> dict:
    """Field names of e-main's A rows (ProMP), split into scalars and per-comparison lists."""
    with open(out / "phase5e" / "e-main.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("learner") == "a" and r.get("family") == "promp_s0" and not r.get("error"):
                lists = sorted(k for k, v in r.items() if isinstance(v, list)
                               and len(v) == r.get("n_comparisons"))
                return dict(lists=lists)
    return dict(lists=[])


def spearman_boot(x, y, draws):
    """Spearman's rho on the users and on each fold-stratified bootstrap draw (average ranks)."""
    rx, ry = rankdata(x), rankdata(y)
    est = float(np.corrcoef(rx, ry)[0, 1])
    bx, by = rankdata(x[draws], axis=1), rankdata(y[draws], axis=1)
    bx = bx - bx.mean(axis=1, keepdims=True)
    by = by - by.mean(axis=1, keepdims=True)
    boot = (bx * by).sum(axis=1) / np.sqrt((bx ** 2).sum(axis=1) * (by ** 2).sum(axis=1))
    return est, *ci(boot)


def w2(out: Path, keys, draws) -> dict:
    df = in_order(w2_rows(out, True), keys)
    D = np.array([d for d in df["distances"]], dtype=float)
    floor = df["d_floor"].astype(float).to_numpy()
    dp = df["d_prior"].astype(float).to_numpy()
    excess40 = D[:, -1] - floor
    excess24 = D[:, 23] - floor
    res = {"n": len(df), "rows": []}
    spread = {"zstar_sd_max": df["zstar_sd_max"].astype(float).to_numpy(),
              "lambda_sd": df["lambda_sd"].astype(float).to_numpy()}
    spread["zstar_sd_max / ell_mean (the stop-rule statistic)"] = (
        spread["zstar_sd_max"] / df["ell_mean"].astype(float).to_numpy())
    other = {"ell_mean": df["ell_mean"], "lambda_mean": df["lambda_mean"],
             "lambda_identified": df["lambda_identified"].astype(float),
             "ess_final": df["ess_final"], "n_resample": df["n_resample"],
             "acceptance_mean": df["acceptance_mean"]}
    for kind, group in (("posterior spread", spread), ("posterior summary, not a spread", other)):
        for name, x in group.items():
            x = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x)):
                res.setdefault("skipped", []).append(f"{name} ({int((~np.isfinite(x)).sum())} "
                                                     "rows without a value)")
                continue
            res["rows"].append(dict(kind=kind, field=name,
                                    excess=spearman_boot(x, excess40, draws),
                                    room=spearman_boot(x, dp - floor, draws)))
    per_k = {}
    for name in ("ell_trace", "lambda_trace"):
        x = np.array([v[23] for v in df[name]], dtype=float)
        per_k[name] = spearman_boot(x, excess24, draws)
    res["per_k"] = per_k
    full = w2_rows(out, False)
    Df = np.array([d for d in full["distances"]], dtype=float)
    xf = full["zstar_sd_max"].astype(float).to_numpy()
    yf = Df[:, -1] - full["d_floor"].astype(float).to_numpy()
    res["all_rows"] = (len(full), float(np.corrcoef(rankdata(xf), rankdata(yf))[0, 1]))
    res["fields"] = w2_fields_present(out)
    res["final_fields"] = [f for f in W2_FIELDS if f in df]
    return res


# ============================================================================ formatting
def f(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{d}f}"


def fs(x, d=2):
    if not np.isfinite(x):
        return "—"
    return "0." + "0" * d if round(x, d) == 0 else f"{x:+.{d}f}"


def fci(t, d=2, signed=False):
    g = fs if signed else f
    return f"{g(t[0], d)} [{g(t[1], d)}, {g(t[2], d)}]"


def fk(x):
    return "> 40" if not np.isfinite(x) else str(int(x))


def git_head() -> str:
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "scripts/icorr_jnd_free.py"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f"{head}{' (script uncommitted)' if dirty else ''}, script sha256 {sha}"
    except (OSError, subprocess.CalledProcessError):
        return f"unknown, script sha256 {sha}"


DIGITS = {"win": 2, "maj3": 2, "share": 2, "err": 3, "jnd1": 2, "jnd2": 2}


def room_head(setup) -> list[str]:
    lv = setup["level"]
    return ["simulated (as stored)", f"anchor, Schäfer timing-only, {lv['anchor']:.2f} Nm",
            f"central literature room, {lv['central']:.2f} Nm",
            f"high literature room, {lv['high']:.2f} Nm"]


def measure_table(cells, results, setup, measure, title, definition) -> list[str]:
    head = room_head(setup)
    out = [f"## {title}", "", definition, ""]
    for cond, arms in cells:
        out += [f"### {cond.title}", "", f"Win-rate basis: {CELL_NOTE[cond.cid]}.", "",
                "| family | optimizer | k | " + " | ".join(head) + " |",
                "|---|---|---|" + "---|" * len(head)]
        seen = set()
        for arm in arms:
            if measure in GENERIC_MEASURES and arm.family not in seen:
                seen.add(arm.family)
                gen = results[(cond.cid, arm.family, "generic")]
                txt = []
                for room in ROOMS:
                    t = fci(gen["at"][(room, measure)], DIGITS[measure])
                    if measure in QUANTILE_MEASURES:
                        q = gen["iqr"][(room, measure)]
                        t += f"; IQR {f(q[0], DIGITS[measure])}–{f(q[1], DIGITS[measure])}"
                    txt.append(t)
                out.append(f"| {arm.fam_name} | *generic assistance* | 0 | " + " | ".join(txt)
                           + " |")
            r = results[(cond.cid, arm.family, arm.learner)]
            for k in arm.ks:
                cells_txt = []
                for room in ROOMS:
                    t = r["at"][(room, measure, k)]
                    txt = fci(t, DIGITS[measure])
                    if measure == "win":
                        txt += f"; 3·P̄ {3 * t[0]:.2f} [{3 * t[1]:.2f}, {3 * t[2]:.2f}]"
                    if measure in QUANTILE_MEASURES:
                        q = r["iqr"][(room, measure, k)]
                        txt += f"; IQR {f(q[0], DIGITS[measure])}–{f(q[1], DIGITS[measure])}"
                    cells_txt.append(txt)
                out.append(f"| {arm.fam_name} | {arm.name} | {k} | " + " | ".join(cells_txt) + " |")
        out.append("")
    return out


def first_k_tables(cells, results, setup) -> list[str]:
    head = [f"{MEASURE_NAME[m]} {d} {lv:g}" for m, d, lv in LEVELS]
    out = ["## 5. Comparisons to reach a level (first k on the learning curve)", "",
           "For each optimizer, the smallest k in 1..40 at which the population statistic (the "
           "mean win rate, the median share closed, the median error, or the share within 1 or 2 "
           "JND, each at that room) first meets the level. It is read on the cross-sectional curve, "
           "as N90 is: not a per-user first hit and not sustained. \"> 40\" means not reached by "
           "k = 40. The interval is the 2.5 and 97.5 % order statistics of the first k over the "
           "bootstrap draws (inverted CDF; \"> 40\" where those draws do not reach it), and the "
           "last figure in each cell is the share of draws that reach the level by 40. The levels "
           "are reading aids fixed in the script (`LEVELS`). The *generic assistance* row says "
           "whether generic assistance itself already meets a level at that room (point "
           "estimate); where it does, a first k says nothing about learning. Drift cells are not "
           "here: they run 24 comparisons with a moving ideal, so their curve has no fixed "
           "target.", ""]
    for cond, arms in cells:
        if any(len(a.ks) < 2 for a in arms):
            continue
        for room, rh in zip(ROOMS, room_head(setup)):
            out += [f"### {cond.title}, {rh}", "",
                    "| family | optimizer | " + " | ".join(head) + " |",
                    "|---|---|" + "---|" * len(head)]
            seen = set()
            for arm in arms:
                if arm.family not in seen:
                    seen.add(arm.family)
                    gen = results[(cond.cid, arm.family, "generic")]
                    out.append(f"| {arm.fam_name} | *generic assistance (k = 0)* | " + " | ".join(
                        "**met at k = 0**" if generic_meets(gen, room, m, d, lv) else "not met"
                        for m, d, lv in LEVELS) + " |")
                r = results[(cond.cid, arm.family, arm.learner)]["first"]
                txt = []
                for m, d, lv in LEVELS:
                    p, lo, hi, reach = r[(room, m, d, lv)]
                    txt.append(f"{fk(p)} [{fk(lo)}, {fk(hi)}] ({reach:.2f})")
                out.append(f"| {arm.fam_name} | {arm.name} | " + " | ".join(txt) + " |")
            out.append("")
    return out


def contrast(results, cid, fam, ours, other, measure, k_ours, k_other, room):
    a = results[(cid, fam, ours)]
    b = results[(cid, fam, other)]
    est = a["at"][(room, measure, k_ours)][0] - b["at"][(room, measure, k_other)][0]
    boot = a["boot"][(room, measure, k_ours)].astype(float) - \
        b["boot"][(room, measure, k_other)].astype(float)
    return (est, *ci(boot))


def contrast_tables(cells, results, setup, k_ours, k_other, title, intro) -> list[str]:
    head = room_head(setup)
    out = [f"## {title}", "", intro, ""]
    for cond, arms in cells:
        by = {(a.family, a.learner): a for a in arms}
        if any(k_ours not in a.ks or k_other not in a.ks for a in arms):
            continue
        out += [f"### {cond.title}", "", "| family | contrast | measure | " + " | ".join(head)
                + " |", "|---|---|---|" + "---|" * len(head)]
        for fam in ("promp_s0", "franks"):
            ours = OURS[fam]
            if (fam, ours) not in by:
                continue
            for (f_, lr), arm in by.items():
                if f_ != fam or lr == ours:
                    continue
                label = f"{by[(fam, ours)].name} (k = {k_ours}) − {arm.name} (k = {k_other})"
                for m in CONTRAST_MEASURES:
                    txt = [fci(contrast(results, cond.cid, fam, ours, lr, m, k_ours, k_other, room),
                               DIGITS[m], signed=True) for room in ROOMS]
                    out.append(f"| {arm.fam_name} | {label} | {MEASURE_NAME[m]} | "
                               + " | ".join(txt) + " |")
        out.append("")
    return out


def headline(cells, results, setup) -> list[str]:
    """Main condition at k = 24: ours and every comparator, simulated and central room."""
    cond, arms = next((c, a) for c, a in cells if c.cid == "main")
    out = ["## 1. Headline: main condition (s = 1.0, λ = 0.1), k = 12 / 24 / 40", "",
           "Each cell: estimate [95 % CI]. Win rate = mean probability of preferring the "
           "identified assistance over generic assistance in one duel; share closed and error "
           "are medians; within m JND is the share of users with d_k − d_floor ≤ m × 0.674 Nm.",
           ""]
    for room, rh in zip(("simulated", "central", "anchor", "high"),
                        [room_head(setup)[i] for i in (0, 2, 1, 3)]):
        out += [f"### {rh}", "", "| family | optimizer | k | win rate (3·P̄) | share closed | "
                "error, Nm | within 1 JND | within 2 JND |", "|---|---|---|---|---|---|---|---|"]
        seen = set()
        for arm in arms:
            if arm.family not in seen:
                seen.add(arm.family)
                g = results[(cond.cid, arm.family, "generic")]["at"]
                out.append(f"| {arm.fam_name} | *generic assistance* | 0 | 0.50 (1.50) | 0 | "
                           f"{fci(g[(room, 'err')], 3)} | {fci(g[(room, 'jnd1')])} | "
                           f"{fci(g[(room, 'jnd2')])} |")
            r = results[(cond.cid, arm.family, arm.learner)]["at"]
            for k in KS:
                w = r[(room, "win", k)]
                out.append(f"| {arm.fam_name} | {arm.name} | {k} | {fci(w)} ({3 * w[0]:.2f}) | "
                           f"{fci(r[(room, 'share', k)])} | {fci(r[(room, 'err', k)], 3)} | "
                           f"{fci(r[(room, 'jnd1', k)])} | {fci(r[(room, 'jnd2', k)])} |")
        out.append("")
    return out


def w2_section(res, n_boot: int) -> list[str]:
    out = ["## 8. W2 (cheap part): does A know how close it is?", "",
           f"e-main, ProMP, A, the {res['n']} population users. Spearman's ρ between each stored "
           "posterior field after the last comparison (k = 40) and A's true excess error, "
           "distances[39] − d_floor (Nm), with a 95 % fold-stratified bootstrap interval (the "
           f"same {n_boot} user draws as above; average ranks for ties). Beside it, ρ with the user's "
           "room d_prior − d_floor, as context: a field that tracks the room rather than the "
           "excess does not say how close A is. Simulated room only.", "",
           "| kind | field (final) | ρ with excess error [95 % CI] | ρ with the room [95 % CI] |",
           "|---|---|---|---|"]
    for r in res["rows"]:
        out.append(f"| {r['kind']} | {r['field']} | {fci(r['excess'], 2, True)} | "
                   f"{fci(r['room'], 2, True)} |")
    n_all, rho_all = res["all_rows"]
    out += ["", f"Reproduction of the spec's first count (W2: Spearman 0.29): on all {n_all} e-main "
            f"A rows of the main cell, stress users included, ρ(zstar_sd_max, excess) = "
            f"{rho_all:+.2f}.", "",
            "**Per-comparison fields stored** (lists of length n_comparisons in A's rows): "
            + ", ".join(f"`{x}`" for x in res["fields"]["lists"]) + ". `ell_trace` and "
            "`lambda_trace` are posterior means after each comparison; `duel_trace` holds the "
            "requested multiple, the duel separation and the one-step predicted P(A); "
            "`distances` and `answers` are outcomes. **No per-comparison posterior spread is "
            "stored** (`zstar_sd_max` and `lambda_sd` exist only after the last comparison), so "
            "the k = 24 correlation with a posterior spread cannot be computed from the stored rows. "
            "For reference, the per-k posterior means at k = 24 against the excess at k = 24:", ""]
    for name, t in res["per_k"].items():
        out.append(f"- `{name}[23]`: ρ = {fci(t, 2, True)}")
    out += ["", "Final scalar fields present: " + ", ".join(f"`{x}`" for x in res["final_fields"])
            + "."]
    if res.get("skipped"):
        out += ["Not correlated (missing values): " + "; ".join(res["skipped"]) + "."]
    return out + [""]


def median_room_section(cells, results, setup, g) -> list[str]:
    """Section 9: the main condition at the median-matched central room (EXTRA_ROOMS), in the
    formats of sections 1, 5, 6 and 7. Every other section is untouched by this room."""
    room = "median"
    rooms = setup["rooms"]
    lv = setup["levels"]["median"]
    cond, arms = next((c, a) for c, a in cells if c.cid == "main")
    ess = rooms.ess[room]
    out = [f"## 9. The median-matched central room ({lv['L']:.2f} Nm), main condition", "",
           "The central literature room of §0 is a 1-SD level (the RMS sum of a timing and a "
           "magnitude spread). Read as a median, the like-for-like reading of a room whose "
           "anchor is a median, the same spreads give the median-matched central room of the "
           "grounding spec §7.1 (1.47 Nm). It is recomputed here as "
           "`preference_literature_room.py` computes it: the Monte Carlo median of κ × the RMS "
           "distance between each Schäfer analytic preferred profile and a copy with Gaussian "
           "landmark shifts (Schäfer timing SDs deflated by late exploration) and an amplitude "
           f"scaling with the central CV {lv['cv']:.3f}, from that module's own draws (seed "
           f"{plr.SEED_OWN + 7}, replayed in its order): **{lv['L']:.3f} Nm**. The users' "
           "weights are §0's KDE ratio against d_loo rescaled to a median of this level; "
           f"effective sample size {ess[0]:.0f} (bootstrap 95 % {ess[1]:.0f}–{ess[2]:.0f}). "
           "Intervals as in §0, from the same draws, so they pair with every other number in "
           "this report. Only the main condition is computed at this room.", "",
           "**Reproduction gate.** A's Q≥1 at k = 24 (as §0a): "
           f"{fci(g[room])}; expected {GATE_MEDIAN[0]:.2f} [{GATE_MEDIAN[1]:.2f}, "
           f"{GATE_MEDIAN[2]:.2f}] (`out/schaefer_grounding/literature_room.md`, central, "
           "median-matched). "
           f"{'Reproduced' if abs(g[room][0] - GATE_MEDIAN[0]) <= GATE_TOL else 'NOT reproduced'}"
           f" to {GATE_TOL}.", "",
           "### 9.1 Measures at k = 12 / 24 / 40", "",
           "As §1: win rate (3·P̄), median share closed, median error, share within 1 and 2 JND "
           "(0.674 Nm), estimate [95 % CI].", "",
           "| family | optimizer | k | win rate (3·P̄) | share closed | error, Nm | within 1 JND "
           "| within 2 JND |", "|---|---|---|---|---|---|---|---|"]
    seen = set()
    for arm in arms:
        if arm.family not in seen:
            seen.add(arm.family)
            gen = results[(cond.cid, arm.family, "generic")]["at"]
            out.append(f"| {arm.fam_name} | *generic assistance* | 0 | 0.50 (1.50) | 0 | "
                       f"{fci(gen[(room, 'err')], 3)} | {fci(gen[(room, 'jnd1')])} | "
                       f"{fci(gen[(room, 'jnd2')])} |")
        r = results[(cond.cid, arm.family, arm.learner)]["at"]
        for k in KS:
            w = r[(room, "win", k)]
            out.append(f"| {arm.fam_name} | {arm.name} | {k} | {fci(w)} ({3 * w[0]:.2f}) | "
                       f"{fci(r[(room, 'share', k)])} | {fci(r[(room, 'err', k)], 3)} | "
                       f"{fci(r[(room, 'jnd1', k)])} | {fci(r[(room, 'jnd2', k)])} |")
    head = [f"{MEASURE_NAME[m]} {d} {x:g}" for m, d, x in LEVELS]
    out += ["", "### 9.2 Comparisons to reach a level (first k)", "",
            "As §5, at this room: first k [2.5, 97.5 % order statistics over the draws] (share "
            "of draws reaching the level by 40).", "",
            "| family | optimizer | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
    seen = set()
    for arm in arms:
        if arm.family not in seen:
            seen.add(arm.family)
            gen = results[(cond.cid, arm.family, "generic")]
            out.append(f"| {arm.fam_name} | *generic assistance (k = 0)* | " + " | ".join(
                "**met at k = 0**" if generic_meets(gen, room, m, d, x) else "not met"
                for m, d, x in LEVELS) + " |")
        first = results[(cond.cid, arm.family, arm.learner)]["first"]
        txt = []
        for m, d, x in LEVELS:
            p, lo, hi, reach = first[(room, m, d, x)]
            txt.append(f"{fk(p)} [{fk(lo)}, {fk(hi)}] ({reach:.2f})")
        out.append(f"| {arm.fam_name} | {arm.name} | " + " | ".join(txt) + " |")
    by = {(a.family, a.learner): a for a in arms}
    for title, k_ours, k_other in (("9.3 Ours minus each comparator at k = 24", 24, 24),
                                   ("9.4 Cross-budget, derived: ours at k = 12 against each "
                                    "comparator at k = 40", 12, 40)):
        out += ["", f"### {title}", "", "Signs and pairing as §6 and §7.", "",
                "| family | contrast | " + " | ".join(MEASURE_NAME[m] for m in CONTRAST_MEASURES)
                + " |", "|---|---|" + "---|" * len(CONTRAST_MEASURES)]
        for fam in ("promp_s0", "franks"):
            ours = OURS[fam]
            if (fam, ours) not in by:
                continue
            for (f_, lr), arm in by.items():
                if f_ != fam or lr == ours:
                    continue
                label = f"{by[(fam, ours)].name} (k = {k_ours}) − {arm.name} (k = {k_other})"
                out.append(f"| {arm.fam_name} | {label} | " + " | ".join(
                    fci(contrast(results, cond.cid, fam, ours, lr, m, k_ours, k_other, room),
                        DIGITS[m], signed=True) for m in CONTRAST_MEASURES) + " |")
    return out + [""]


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--data", type=Path, default=G.DEFAULT_DATA)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    out, n_boot = args.out, args.boot
    report = args.report or out / "icorr" / "jnd_free.md"
    sys.stdout.reconfigure(encoding="utf-8")

    print("streaming rows ...", flush=True)
    blocks = load_blocks(out)
    sim_keys = G.load_sim(out / "phase5-schaefer")["keys"]
    main_a = in_order(select(blocks, er.BY_ID["main"], "promp_s0", "a"), sim_keys)
    room = main_a["d_prior"].astype(float).to_numpy() - main_a["d_floor"].astype(float).to_numpy()
    folds = main_a["fold"].to_numpy()

    print("rooms and weights ...", flush=True)
    setup = room_setup(out, args.data, n_boot, room, folds)
    if setup["sim"]["keys"] != sim_keys or not np.array_equal(setup["sim"]["folds"], folds):
        raise RuntimeError("user order differs between load_sim and the main cell")
    rooms = setup["rooms"]

    # cells
    cells, checks = [], []
    dp_main = main_a["d_prior"].astype(float).to_numpy()
    for cond in er.CONDS:
        arms = []
        for family in ("promp_s0", "franks"):
            for learner in er.SCOPES[cond.scope][family]:
                df = select(blocks, cond, family, learner)
                if df is None:
                    raise RuntimeError(f"cell {cond.cid} {family} {learner} has no rows")
                arm = Arm(cond, family, learner, in_order(df, sim_keys))
                arms.append(arm)
                if family == "promp_s0" and not cond.cid.startswith("drift"):
                    checks.append((cond.cid, arm.name, float(np.max(np.abs(arm.dp - dp_main)))))
                if arm.generic_check is not None:
                    checks.append((cond.cid, f"{arm.fam_name} {arm.name}, rebuilt generic curve",
                                   arm.generic_check))
        cells.append((cond, arms))
    bad = [c for c in checks if c[2] > 1e-9]
    if bad:
        raise RuntimeError(f"d_prior / generic-curve checks failed: {bad[:5]}")

    # the gate, before anything new
    arm_a = next(a for c, arms in cells if c.cid == "main" for a in arms
                 if a.family == "promp_s0" and a.learner == "a")
    g = gate(arm_a, rooms, ALL_ROOMS)
    ok_anchor = abs(g["anchor"][0] - GATE_ANCHOR[0]) <= GATE_TOL
    ok_central = abs(g["central"][0] - GATE_CENTRAL[0]) <= GATE_TOL
    print(f"gate: anchor {fci(g['anchor'])} (expected {GATE_ANCHOR}), central "
          f"{fci(g['central'], 3)} (expected {GATE_CENTRAL})", flush=True)
    if not (ok_anchor and ok_central):
        print("REPRODUCTION GATE FAILED: nothing written", flush=True)
        return 1
    level_median = setup["level"]["median"]
    ok_median = (abs(g["median"][0] - GATE_MEDIAN[0]) <= GATE_TOL
                 and round(level_median, 2) == GATE_MEDIAN_LEVEL)
    print(f"median-matched room {level_median:.3f} Nm (expected {GATE_MEDIAN_LEVEL}): A's Q>=1 "
          f"{fci(g['median'])} (expected {GATE_MEDIAN})", flush=True)
    if not ok_median:
        print("MEDIAN-ROOM GATE FAILED: nothing written", flush=True)
        return 1

    print("statistics ...", flush=True)
    results = {}
    for cond, arms in cells:
        names = ALL_ROOMS if cond.cid == "main" else ROOMS
        for arm in arms:
            results[(cond.cid, arm.family, arm.learner)] = compute_arm(arm, rooms, names)
            key = (cond.cid, arm.family, "generic")
            if key not in results:
                results[key] = compute_generic(arm, rooms, names)
                results[key]["arm"] = arm
            else:
                first = results[key]["arm"]
                if not (np.allclose(first.dp, arm.dp, atol=1e-9)
                        and np.allclose(first.floor, arm.floor, atol=1e-9)):
                    raise RuntimeError(f"{cond.cid} {arm.fam_name}: optimizers differ in the "
                                       "generic distance or the floor")
        print(f"  {cond.cid} done", flush=True)
    w2_res = w2(out, sim_keys, rooms.sim_draws)

    # ------------------------------------------------------------------ report
    lv = setup["levels"]
    ra = results[("main", "promp_s0", "a")]["at"]
    L: list[str] = []
    p = L.append
    p("# ICORR W1: JND-free outcomes from the stored sessions (and the cheap part of W2)")
    p("")
    p(f"Generated by `scripts/icorr_jnd_free.py` on {_dt.date.today().isoformat()} at worktree "
      f"commit {git_head()}. Spec: `docs/superpowers/specs/2026-10-17-icorr-user-tuning-reframe-"
      "design.md` §4 (W1, W2) and §5. No session was run; every number is computed from the "
      "stored rows. Post hoc on cells fixed before this analysis.")
    p("")
    p("## 0. Definitions, rooms and intervals")
    p("")
    p("**Users.** The 200 population users of the five LOSO folds (`user_kind == 'population'`), in "
      "every cell the same users, in the order of `schaefer_grounding.load_sim`. The identified "
      "assistance after k comparisons is the curve the optimizer would deliver then; its error is "
      "`distances[k − 1]`, the RMS distance in Nm to the simulated wearer's ideal (exact under "
      "truncation, B3). Cells and their row filters are `phase5e_report`'s (`CONDS`, `DEFAULTS`, "
      "`_match`, `SCOPES`); the s = 0.5 reference is the 5c/5d rows (150 comparisons) truncated, "
      "selected as `short_budget_report.Rows` does.")
    p("")
    p("**Measures** (per user i at comparison k):")
    p("")
    p("- **Win rate** P_i(k) = λ_i/2 + (1 − λ_i)·Φ((g_i − e_i(k))/s_i): the probability that the "
      "cell's wearer, answering as in the session, prefers the identified assistance over generic "
      "assistance (the prior's mean curve, `d_prior`) in one duel. λ_i and s_i are the row's lapse "
      "and sensitivity (`s_user` in the heterogeneous cell). g_i and e_i(k) are in the metric the "
      "wearer answers by: RMS (`d_prior`, `distances`) for RMS wearers; `own_distances` and the "
      "wearer's own distance of the generic curve, rebuilt as `phase5e_own_generic.py` does, in "
      "the landmark, asymmetry and familiarity cells. Reported: the mean over users P̄, the "
      "expected number of wins in a 3-comparison validation block 3·P̄ (its interval is 3 × that "
      "of P̄), and, in §2, P(at least 2 of 3 won) = mean of P³ + 3P²(1 − P), duels independent.")
    p("- **Share of the achievable improvement closed** c_i(k) = (d_prior − d_k)/(d_prior − "
      "d_floor), RMS, the family's own generic distance and floor: median, IQR and a CI for the "
      "median. A user with d_prior − d_floor ≤ 1e-9 is left out (none here).")
    p("- **Error** d_k in Nm: median, IQR, CI for the median.")
    p("- **Within m JND of the family optimum**: the share of users with d_k − d_floor ≤ m·JND, "
      "m = 1, 2, JND = true_jnd(s) = 0.674·s Nm of the cell (phase5e_report E2): 0.337 at s = 0.5, "
      "0.674 at s = 1.0 and in every observer, noise, drift, lapse and heterogeneous cell, 1.349 at "
      "s = 2.0. In the observer cells the share closed, the error and the JND shares are RMS "
      "measures (A's model class); no floor in the wearer's own metric is stored.")
    p("- **Generic assistance rows (k = 0)**: the same error and JND shares for generic "
      "assistance itself (d_prior in place of d_k), per family. Its win rate is 0.5 and its share "
      "closed 0 by definition. At small rooms many users' generic assistance already lies within "
      "2 JND of the optimum, so the JND shares and the error must be read against this row.")
    p("")
    p("**Quantiles.** Medians and quartiles are weighted Hazen quantiles: each user's weight is "
      "centred at its cumulative-weight midpoint, with linear interpolation. With equal weights "
      "(the simulated room) this is `np.percentile(method='hazen')`, whose median is the usual "
      "median.")
    p("")
    p("**Rooms.** *Simulated*: the users as stored (equal weights). *Reweighted*: each user's "
      "weight is the G1b kernel-density ratio `schaefer_grounding.kde_weights` (Scott bandwidth) of "
      "their ProMP room (d_prior − d_floor of e-main A; Franks optimizers carry the same user "
      "weights, as in `schaefer_grounding_optimizers.py`) against a target room distribution: "
      f"the anchor is Schäfer's amplitude-matched timing-only room, κ·d_loo (κ = "
      f"{setup['kappa']:.4f}, median {setup['level']['anchor']:.3f} Nm); the central and high "
      "literature rooms rescale d_loo to a median of the room level, as "
      "`preference_literature_room.py` does. The two levels are recomputed here from that "
      "module's literature records (its steps a–d): central = Schäfer timing deflated by late "
      f"exploration ({lv['central']['T']:.3f} Nm) ⊕ the median preference magnitude CV "
      f"{lv['central']['cv']:.3f} × {setup['sim_med']:.3f} Nm ({lv['central']['M']:.3f} Nm) = "
      f"**{lv['central']['L']:.3f} Nm**; high = Schäfer raw timing ({lv['high']['T']:.3f}) ⊕ the "
      f"single-endpoint CV {lv['high']['cv']:.3f} ({lv['high']['M']:.3f}) = "
      f"**{lv['high']['L']:.3f} Nm**. Effective sample sizes (point, bootstrap 95 %): "
      + "; ".join(f"{room} {rooms.ess[room][0]:.0f} ({rooms.ess[room][1]:.0f}–"
                  f"{rooms.ess[room][2]:.0f})" for room in ROOMS) + ".")
    p("")
    p(f"**Intervals.** 95 % percentile intervals over {n_boot} bootstrap draws: simulated users "
      "resampled within folds (cluster bootstrap stratified by fold) and, at the reweighted rooms, "
      f"the {setup['n_part']} Schäfer participants resampled jointly with the densities refitted "
      f"per draw. This is `preference_literature_room.py`'s construction (seed {SEED}, "
      "`fold_draws` then the participant draws), so its numbers reproduce; one set of draws serves "
      "every cell, optimizer, k and room, so differences are paired by draw. At the reweighted "
      "rooms the intervals are at a fixed room level; the room's own uncertainty is not "
      "propagated.")
    p("")
    p("## 0a. Reproduction gate (before anything new)")
    p("")
    p("A's Q≥1 at k = 24 (share of users with (d_prior − d_24)/0.674 ≥ 1), main cell, ProMP:")
    p("")
    p("| room | this script | expected | ok |")
    p("|---|---|---|---|")
    p(f"| anchor | {fci(g['anchor'])} | {GATE_ANCHOR[0]:.2f} [{GATE_ANCHOR[1]:.2f}, "
      f"{GATE_ANCHOR[2]:.2f}] (`optimizers_real_room.md`) | {'yes' if ok_anchor else 'NO'} |")
    p(f"| central | {fci(g['central'], 3)} | {GATE_CENTRAL[0]:.3f} [{GATE_CENTRAL[1]:.3f}, "
      f"{GATE_CENTRAL[2]:.3f}] (grounding spec §7.1) | {'yes' if ok_central else 'NO'} |")
    p(f"| simulated | {fci(g['simulated'])} | 0.82 (5e spec §8.1) | — |")
    p(f"| high | {fci(g['high'])} | 0.80 [0.66, 0.90] (grounding spec §7.1) | — |")
    p("")
    p("Other checks at the simulated room against the 5e spec §8.1 and ICORR spec C2 (not gated): "
      f"A's median error at k = 24 {ra[('simulated', 'err', 24)][0]:.3f} Nm (expected "
      f"{SANITY_5E['A error k=24']}), at k = 12 {ra[('simulated', 'err', 12)][0]:.3f} "
      f"({SANITY_5E['A error k=12']}); A's within 2 JND at k = 24 "
      f"{ra[('simulated', 'jnd2', 24)][0]:.2f} (Q_opt {SANITY_5E['A Q_opt (= within 2 JND) k=24']}). "
      f"Every cell's ProMP d_prior equals e-main's per user (max |Δ| "
      f"{max(c[2] for c in checks if 'rebuilt' not in c[1]):.1e} Nm), so the s = 0.5 rows hold the "
      "same users; the rebuilt generic curve reproduces the stored d_prior in every observer and "
      f"drift row (max |Δ| {max([c[2] for c in checks if 'rebuilt' in c[1]] or [0.0]):.1e} Nm).")
    p("")
    L.extend(headline(cells, results, setup))
    L.extend(measure_table(
        cells, results, setup, "win", "2. Validation win rate against generic assistance",
        "Mean over users of P_i(k), estimate [95 % CI], then the expected wins of a "
        "3-comparison validation block, 3·P̄. A wearer for whom the identified assistance equals "
        "generic assistance wins with P = 0.5. The basis of the win rate in each cell is stated "
        "under its heading. The majority-of-3 probability is in the next table."))
    L.extend(measure_table(
        cells, results, setup, "maj3", "2a. Majority of a 3-comparison validation block",
        "Mean over users of P_i³ + 3P_i²(1 − P_i): the probability that the identified assistance "
        "wins at least 2 of 3 independent validation duels against generic assistance."))
    L.extend(measure_table(
        cells, results, setup, "share", "3. Share of the achievable improvement closed",
        "Median of (d_prior − d_k)/(d_prior − d_floor) [95 % CI for the median]; IQR = the "
        "weighted quartiles at that room. 1 = at the family optimum, 0 = no better than generic "
        "assistance, negative = worse. RMS in every cell."))
    L.extend(measure_table(
        cells, results, setup, "err", "3a. Error of the identified assistance (Nm)",
        "Median of d_k = distances[k − 1], RMS Nm to the ideal [95 % CI for the median]; IQR = "
        "the weighted quartiles. Each family has its own floor (median ProMP floor 0.66 Nm in the "
        "grounding spec)."))
    L.extend(measure_table(
        cells, results, setup, "jnd1", "4. Within 1 JND of the family optimum",
        "Share of users with d_k − d_floor ≤ 1 × JND of the cell [95 % CI]."))
    L.extend(measure_table(
        cells, results, setup, "jnd2", "4a. Within 2 JND of the family optimum",
        "Share of users with d_k − d_floor ≤ 2 × JND of the cell [95 % CI] (Q_opt of the B7 "
        "tables)."))
    L.extend(first_k_tables(cells, results, setup))
    L.extend(contrast_tables(
        cells, results, setup, 24, 24, "6. Ours minus each comparator at k = 24",
        "A on ProMP, the lens on Franks, minus each comparator of the same family at the same "
        "k = 24, paired by bootstrap draw (same users, same weights). Win rate, share and JND "
        "shares: positive favours ours; error: negative favours ours. A difference of medians, "
        "not a median of differences."))
    L.extend(contrast_tables(
        cells, results, setup, 12, 40, "7. Cross-budget, derived: ours at k = 12 against each "
        "comparator at k = 40",
        "**Derived** (compares two budgets: ours after 12 comparisons, the comparator after 40). "
        "Same pairing and signs as §6."))
    L.extend(w2_section(w2_res, n_boot))
    L.extend(median_room_section(cells, results, setup, g))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
