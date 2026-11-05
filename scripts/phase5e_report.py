"""Phase 5e robustness report.

Usage:  python scripts/phase5e_report.py [--dir5e out/phase5e] [--dir5c out/phase5c]
            [--dir5d out/phase5d] [--out out/phase5e/REPORT.md] [--boot 10000]
            [--run <run dir>] [--perturbation-cache <path>]

No session is run. Every number is read from the stored rows: the error of the identified
assistance after k comparisons is ``distances[k - 1]`` (exact under truncation, B3). The measures
are benchmark_report's (B7) and the intervals are phase5c_stats.paired_bootstrap's (S11); the
one change against the benchmark is the yardstick. Quality levels use each cell's own JND (E2):
true_jnd(s) in s cells and true_jnd(1.0) = 0.674 Nm in the observer, noise, drift, lapse and
heterogeneous-s cells. The s = 0.5 reference comes from the 5c/5d rows (150 comparisons),
truncated.

Two parts read the run directory the sessions ran on (``--run``, by default the ``run_dir`` the
e-main rows record), read-only: the drift split by the device limit needs each user's ideal,
and the perturbation strength (``phase5e_perturbation.py``) its priors and users. The latter is
computed once and cached as JSON (``--perturbation-cache``, by default
``<dir5e>/perturbation_strength.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import benchmark_report as bm  # noqa: E402
import phase5e_perturbation as pt  # noqa: E402
import short_budget_report as sb  # noqa: E402

from prompref.experiment.phase5 import DZ_TUNED  # noqa: E402
from prompref.experiment.phase5c_grid import LOSO  # noqa: E402
from prompref.experiment.phase5c_stats import paired_bootstrap  # noqa: E402
from prompref.experiment.phase5e_grid import (EVAL_FOLDS, TUNE_FOLD,  # noqa: E402
                                              TUNED_FIELDS)
from prompref.experiment.sweep import _parse_rows  # noqa: E402
from prompref.learning.gp_ucb import KAPPA0  # noqa: E402
from prompref.population.curves import TORQUE_LIMIT_NM  # noqa: E402
from prompref.population.export import load_fold_users  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402

K_MAIN = 24
KS = (12, 24, 40)
N90_MAX = 40
BLOCKS = ("e-main", "e-s", "e-observer", "e-noise", "e-drift", "e-lapse", "e-tune", "e-tuned",
          "e-oracle")
NOT_RUN = "not run yet"
#: The wearer's own distance per comparison, and the landmark
#: observer's feature distance in feature-JNDs.
OWN = "own_distances"
FEATURE = "own_feature_jnd"
PERTURB_CACHE = "perturbation_strength.json"
#: The estimand of every Δ median (the paper's wording).
ESTIMAND = ("difference of the two medians on paired users (not the median of paired "
            "differences)")
KEY = ["fold", "user_id"]
#: Population users a complete cell holds (5 folds x 40); a verdict, row or contrast on fewer
#: is provisional. e-tuned holds 4 folds x 40 per learner, e-tune 40 users per setting.
EXPECTED_N = 200
EXPECTED_N_TUNED = 160
EXPECTED_N_TUNE = 40
SEED = 20261007

PROMP_FULL = ("a", "g", "d_z", "cmaes")
FRANKS_FULL = ("a_lens", "a", "g", "d_z", "cmaes")
SCOPES = {"full": {"promp_s0": PROMP_FULL, "franks": FRANKS_FULL},
          "core": {"promp_s0": PROMP_FULL, "franks": ("a_lens", "g")},
          "drift": {"promp_s0": PROMP_FULL, "franks": ()}}
IDS = {"a": "A", "g": "G", "d_z": "D_z", "cmaes": "SortCMA", "a_lens": "lens"}
FAMILY_NAME = {"promp_s0": "ProMP", "franks": "Franks"}
#: point fields at their defaults; a cell's own fields override them.
#: landmark_mag_fraction: a static landmark row carries 0.1, every other row none (E4b).
DEFAULTS = dict(lapse=0.1, s_user_logsd=0.0, observer_metric="rms", asymmetry=1.0,
                familiarity_weight=0.0, device_noise_nm=0.0, drift_nm=0.0,
                landmark_mag_fraction=0.10)
JND_MAIN = true_jnd(1.0)
TUNED_ORDER = ("cmaes", "g", "d_z")


class Cond:
    def __init__(self, cid, title, block, fields, jnd, scope):
        self.cid, self.title, self.block = cid, title, block
        self.fields, self.jnd, self.scope = fields, jnd, scope


CONDS = [
    Cond("s05", "s = 0.5 reference", None, {}, true_jnd(0.5), "full"),
    Cond("main", "main (s = 1.0)", "e-main", dict(s=1.0), JND_MAIN, "full"),
    Cond("s2", "s = 2.0", "e-s", dict(s=2.0), true_jnd(2.0), "full"),
    # the quality levels of the heterogeneous cell use the population s, not each user's own
    Cond("het", "heterogeneous s (population s = 1.0)", "e-s",
         dict(s=1.0, s_user_logsd=0.35), JND_MAIN, "full"),
    Cond("landmark", "landmark observer, static JND (10 %)", "e-observer",
         dict(s=1.0, observer_metric="landmark"), JND_MAIN, "core"),
    Cond("landmark25", "landmark observer, walking JND (25 %)", "e-observer",
         dict(s=1.0, observer_metric="landmark", landmark_mag_fraction=0.25), JND_MAIN, "core"),
    Cond("weber", "Weber observer", "e-observer", dict(s=1.0, observer_metric="weber"),
         JND_MAIN, "core"),
    Cond("asym", "asymmetry 2", "e-observer", dict(s=1.0, asymmetry=2.0), JND_MAIN, "core"),
    Cond("famil", "familiarity 0.25", "e-observer", dict(s=1.0, familiarity_weight=0.25),
         JND_MAIN, "core"),
    Cond("noise", "device noise 0.6 Nm", "e-noise", dict(s=1.0, device_noise_nm=0.6),
         JND_MAIN, "core"),
    Cond("drift1", "magnitude drift 1 JND, n = 24", "e-drift",
         dict(s=1.0, drift_nm=true_jnd(1.0) * 1, drift_axis="magnitude"), JND_MAIN, "drift"),
    Cond("drift2", "magnitude drift 2 JND, n = 24", "e-drift",
         dict(s=1.0, drift_nm=true_jnd(1.0) * 2, drift_axis="magnitude"), JND_MAIN, "drift"),
    Cond("lapse2", "lapse 0.2", "e-lapse", dict(s=1.0, lapse=0.2), JND_MAIN, "core"),
    Cond("lapse3", "lapse 0.3", "e-lapse", dict(s=1.0, lapse=0.3), JND_MAIN, "core"),
]
BY_ID = {c.cid: c for c in CONDS}
COND_JND = {c.cid: c.jnd for c in CONDS}
#: our design notes-E7: the cells QE2 requires A to lead in, with E4b's walking landmark cell
QE2_CELLS = ("s2", "het", "landmark", "landmark25", "weber", "asym", "famil", "noise", "drift1",
             "drift2", "lapse2", "lapse3")
#: the observer-model cells, whose rows carry the wearer's own distance (E4a)
VARIANT_CELLS = ("landmark", "landmark25", "weber", "asym", "famil")


def _label(family: str, learner: str) -> str:
    """The paper's name: "A" on ProMP, "A direct" on Franks."""
    return bm.label("ProMP" if family == "promp_s0" else "Franks", IDS[learner])


# --------------------------------------------------------------------------- the data

def _match(df, name, want):
    if name not in df:
        col = pd.Series([DEFAULTS.get(name)] * len(df), index=df.index)
    else:
        col = df[name]
    if isinstance(want, str):
        default = DEFAULTS.get(name)
        return ((col if default is None else col.fillna(default)) == want).to_numpy()
    return np.isclose(pd.to_numeric(col, errors="coerce").fillna(DEFAULTS.get(name, 0.0)),
                      want, atol=1e-6)


def _med(x, axis=None):
    """The median as phase5c_stats takes it (S11): the lower middle value, ``inverted_cdf``."""
    return np.quantile(np.asarray(x, dtype=float), 0.5, axis=axis, method="inverted_cdf")


def has_field(df, field: str, k: int = K_MAIN) -> bool:
    """Every row holds a per-comparison list ``field`` of at least k values."""
    if field == "distances":
        return True
    if df is None or field not in df:
        return False
    return bool(df[field].map(lambda v: isinstance(v, list) and len(v) >= k).all())


def values_at(df, k: int, field: str = "distances") -> pd.Series:
    """The per-comparison ``field`` after comparison k: ``distances`` (the RMS error of the
    identified assistance) or, in an observer-variant cell, the wearer's own distance (E4a)."""
    if field == "distances":
        return sb.delivered_at(df, k)
    if not has_field(df, field, k):
        raise ValueError(f"not every row holds {field} up to comparison {k}")
    return pd.Series([float(v[k - 1]) for v in df[field]], index=df.index, dtype=float)


def prov(n: int, dropped: int = 0, expected: int | None = None) -> str:
    """The provisional suffix: n users of the expected count, or users dropped by pairing."""
    expected = EXPECTED_N if expected is None else expected
    return f" (provisional, n = {n} of {expected})" if n < expected or dropped else ""


def align(dfs):
    """The frames restricted to the users all of them hold, sorted by (fold, user_id):
    (frames, number of users held by only some of them)."""
    dfs = [d.drop_duplicates(KEY, keep="last") for d in dfs]
    keys = [set(zip(d["fold"], d["user_id"])) for d in dfs]
    common = set.intersection(*keys)
    out = []
    for d in dfs:
        keep = np.array([key in common for key in zip(d["fold"], d["user_id"])], dtype=bool)
        out.append(d[keep].sort_values(KEY).reset_index(drop=True))
    return out, len(set.union(*keys)) - len(common)


def joint_bootstrap(arrays, strata, stat, n_boot: int, seed: int = SEED) -> dict:
    """A joint cluster bootstrap of ``stat`` over aligned per-user arrays: users are resampled
    once per draw, within each fold, and every array is indexed by the same draw. ``stat``
    takes one (draws, users) matrix per array and returns one value per draw. Returns the
    estimate on the original users and the 95 % percentile interval, its ends observed
    bootstrap values (``inverted_cdf``, as paired_bootstrap takes them, S11)."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    strata = np.asarray(strata)
    estimate = float(stat(*[a[None, :] for a in arrays])[0])
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == f) for f in pd.unique(strata)]
    draws = np.concatenate([g[rng.integers(0, len(g), size=(n_boot, len(g)))] for g in groups],
                           axis=1)
    boot = stat(*[a[draws] for a in arrays])
    lo, hi = np.percentile(boot, [2.5, 97.5], method="inverted_cdf")
    return dict(estimate=estimate, lo=float(lo), hi=float(hi), n=int(strata.size))


@lru_cache(maxsize=16)
def _fold_users(run_dir: str, fold: str):
    return load_fold_users(run_dir, fold)


def _touches_from(run_dir):
    """``(fold, user_id) -> bool``: whether the user's ideal reaches the device limit
    anywhere (|tau*| >= 12 - 1e-9 Nm), read from the run directory; None without one."""
    if run_dir is None or not (Path(run_dir) / "users").is_dir():
        return None

    def touches(fold: str, user_id: str) -> bool:
        curve = _fold_users(str(run_dir), fold)[user_id].curve_nm
        return bool(np.any(np.abs(curve) >= TORQUE_LIMIT_NM - 1e-9))

    return touches


class Ctx:
    """Every block's rows, loaded once, with the cell filters and a cache for contrasts."""

    def __init__(self, dir5e, dir5c, dir5d, n_boot: int, *, run_dir=None, touches=None,
                 perturbation_cache=None):
        self.dir5e, self.n_boot = Path(dir5e), n_boot
        self.ref = sb.Rows(Path(dir5c), Path(dir5d))
        self.blocks = {}
        for block in BLOCKS:
            df = sb._load(self.dir5e, block)
            self.blocks[block] = None if df is None else sb._n2048(sb._popn(df))
        self._df, self._con = {}, {}
        main = self.blocks["e-main"]
        if run_dir is None and main is not None and len(main) and "run_dir" in main:
            run_dir = main["run_dir"].iloc[0]
        self.run_dir = None if run_dir is None else str(run_dir)
        self.touches = touches if touches is not None else _touches_from(self.run_dir)
        self.perturbation_cache = (Path(perturbation_cache) if perturbation_cache is not None
                                   else self.dir5e / PERTURB_CACHE)
        self._perturbation = None
        self._decisiveness = None

    def df(self, cond: str, family: str, learner: str):
        key = (cond, family, learner)
        if key not in self._df:
            self._df[key] = self._find(BY_ID[cond], family, learner)
        return self._df[key]

    def _find(self, cond, family, learner):
        if cond.cid == "s05":
            r = self.ref
            if family == "promp_s0":
                df = r.sortcma() if learner == "cmaes" else r.main(learner)
            else:
                df = (r.sortcma("franks") if learner == "cmaes"
                      else r.lens("a_lens" if learner == "a_lens" else learner))
            return df if df is not None and len(df) else None
        df = self.blocks[cond.block]
        if df is None:
            return None
        mask = (df["family"] == family).to_numpy() & (df["learner"] == learner).to_numpy()
        want = dict(DEFAULTS)
        want.update(cond.fields)
        for name, value in want.items():
            mask &= _match(df, name, value)
        out = df[mask].drop_duplicates(KEY, keep="last")
        return out if len(out) else None

    def contrast(self, a, b, k: int, key=None, field: str = "distances"):
        """First cell minus second at comparison k: users, medians and the Δ-median CI, read in
        ``field`` (the RMS error by default, or the wearer's own distance); ``None`` when no
        user is common or a cell does not hold ``field``."""
        if key is not None and key in self._con:
            return self._con[key]
        (a, b), dropped = align([a, b])
        out = None
        if len(a) and has_field(a, field, k) and has_field(b, field, k):
            da, db = values_at(a, k, field).to_numpy(), values_at(b, k, field).to_numpy()
            out = dict(n=len(a), dropped=dropped, med_a=float(_med(da)),
                       med_b=float(_med(db)),
                       diff=paired_bootstrap(da, db, a["fold"].to_numpy(),
                                             n_boot=self.n_boot)["diff_median"])
        if key is not None:
            self._con[key] = out
        return out

    def cell_contrast(self, cond, family, a_lr, b_lr, k=K_MAIN, field: str = "distances"):
        a, b = self.df(cond, family, a_lr), self.df(cond, family, b_lr)
        if a is None or b is None:
            return None
        return self.contrast(a, b, k, key=(cond, family, a_lr, b_lr, k, field), field=field)

    def median(self, cond, family, learner, k=K_MAIN):
        df = self.df(cond, family, learner)
        return None if df is None else float(_med(sb.delivered_at(df, k)))

    def joint(self, dfs, k: int, stat, key=None, fields=None):
        """``stat`` of the values at k of several cells on the users all hold, with its joint
        bootstrap interval; ``None`` when a cell is missing, does not hold its field, or no
        user is common. ``fields`` names each cell's field (default: the RMS error)."""
        if key is not None and key in self._con:
            return self._con[key]
        fields = fields or ["distances"] * len(dfs)
        out = None
        if all(_present(d) and has_field(d, f, k) for d, f in zip(dfs, fields)):
            aligned, dropped = align(dfs)
            if len(aligned[0]):
                out = joint_bootstrap([values_at(d, k, f).to_numpy()
                                       for d, f in zip(aligned, fields)],
                                      aligned[0]["fold"].to_numpy(), stat, self.n_boot)
                out["dropped"] = dropped
        if key is not None:
            self._con[key] = out
        return out


def _present(df) -> bool:
    return df is not None and len(df) > 0


def _ahead(c) -> bool:
    """The first cell is lower with an interval that excludes 0."""
    return c is not None and c["diff"]["hi"] < 0


def _behind(c) -> bool:
    return c is not None and c["diff"]["lo"] > 0


def _cstr(name: str, c) -> str:
    return f"{name} {sb._ci(c['diff'])}"


# --------------------------------------------------------------------------- B7 rows

B7_HEAD = ("| {first} | n | median error (Nm) | median improvement (Nm) | Q≥1 | Q≥2 | Q_opt "
           "| lower decile of gain, JND [95 % CI] | median share of the achievable improvement "
           "| N90 Q≥1 / Q≥2 / Q_opt |\n|---|---|---|---|---|---|---|---|---|---|")


def b7_row(name: str, df, k: int, jnd, n_boot: int) -> str:
    """One optimizer's B7 row at k, the levels in ``jnd`` Nm, with the physical improvement.
    ``jnd`` is one yardstick for the cell, or one per row (each wearer's own JND)."""
    if not _present(df):
        return f"| {name} | {NOT_RUN} | — | — | — | — | — | — | — | — |"
    per_user = np.ndim(jnd) > 0
    if per_user:
        jnd = np.asarray(jnd, dtype=float)
    s = bm.reliability(df, k, n_boot, jnd=jnd)
    d = s["decile"]
    delivered = sb.delivered_at(df, k)
    improvement = float((df["d_prior"].astype(float) - delivered).median())
    kmax = int(min(N90_MAX, df["distances"].map(len).min()))
    n90 = []
    for key, _ in bm.LEVELS:
        # share_curve's matrix is (users, k): a per-user yardstick is a column
        value = bm.n90(bm.share_curve(df, key, kmax=kmax, jnd=jnd[:, None] if per_user else jnd))
        n90.append(f"> {kmax}" if value is None else str(value))
    return (f"| {name} | {s['n']}{prov(s['n'])} | {float(delivered.median()):.3f} | {improvement:.3f} | "
            f"{s['q1']:.2f} | {s['q2']:.2f} | {s['qopt']:.2f} | "
            f"{sb._sg(d['estimate'], 2)} [{sb._sg(d['lo'], 2)}, {sb._sg(d['hi'], 2)}] | "
            f"{sb._num(s['median_share'], 2)} ({s['n_excluded']} excluded) | "
            f"{' / '.join(n90)} |")


def _b7_tables(ctx: Ctx, cond: Cond, k: int, heading: str = "###") -> list[str]:
    out = []
    for family in ("promp_s0", "franks"):
        learners = SCOPES[cond.scope][family]
        if not learners:
            continue
        out += [f"{heading} {cond.title}, {FAMILY_NAME[family]}, k = {k}", "",
                B7_HEAD.format(first="optimizer")]
        out += [b7_row(_label(family, lr), ctx.df(cond.cid, family, lr), k, cond.jnd, ctx.n_boot)
                for lr in learners] + [""]
    return out


# --------------------------------------------------------------------------- sections

ASYMMETRY_NOTE = ("Asymmetry 2 applies to the squared error of over-assistance, so it weighs "
                  "about 1.41× (√2) in RMS.")
GATE_NOTE = ("G's true-JND gate (R34, oracle information no experimenter has) is exact only "
             "for RMS wearers; in the landmark and Weber cells it is a fixed 0.674 Nm, "
             "true_jnd(1.0), not the JND those wearers have.")


def _intro() -> list[str]:
    return ["# Phase 5e: how far the short-budget claims carry", "",
            "Spec E1-E10 and the amendments of its §4. Population users; the error of the "
            "identified assistance after k comparisons is `distances[k − 1]`, the RMS distance "
            "in Nm to the simulated wearer's ideal. Every statement is about the simulator, "
            "conditional on the assumptions of each cell. Δ median is the first cell minus the "
            f"second: the {ESTIMAND}, with a 95 % cluster bootstrap over users stratified by "
            "fold, so a negative value favours the first.", "",
            "**Yardstick (E2).** Q≥1, Q≥2 and Q_opt are in the cell's own JND: true_jnd(s) in "
            "the s cells (0.337, 0.674 and 1.349 Nm at s = 0.5, 1.0 and 2.0) and true_jnd(1.0) = "
            "0.674 Nm in the observer, noise, drift, lapse and heterogeneous cells. In the "
            "heterogeneous cell this is the population s = 1.0, not each wearer's own s_user: "
            "the levels then measure one fixed physical improvement across wearers (section 4 "
            "also gives them in each wearer's own JND). Every row also prints the improvement "
            "in Nm.", "",
            "**Assumptions of the cells.**", "",
            "- The simulated wearer judges by the RMS distance to one ideal profile, A's own "
            "model class, except in the observer cells (landmark, Weber, asymmetry, "
            "familiarity). Section 8 scores those cells in the wearer's own distance as well.",
            f"- {ASYMMETRY_NOTE}",
            f"- {GATE_NOTE}",
            "- A's lapse prior is Beta(1, 9): the lapse prior has mean 0.1, the same as the "
            "reference lapse of the simulated wearer. The lapse cells (0.2, 0.3) are where "
            "that prior is off.",
            "- Familiarity and asymmetry are sensitivity cases without perceptual measurements "
            "behind them (spec §6).", ""]


def _main_section(ctx: Ctx) -> list[str]:
    cond = BY_ID["main"]
    out = ["## 1. The main condition, s = 1.0", "",
           f"B7 at k = 12, 24 and 40, levels in {cond.jnd:.3f} Nm (E2). N90 is read up to "
           f"{N90_MAX} comparisons (\"> {N90_MAX}\" if never reached).", ""]
    for k in KS:
        out += _b7_tables(ctx, cond, k)
    return out


def _contrast_pairs(cond: Cond):
    out = []
    for family in ("promp_s0", "franks"):
        scope = SCOPES[cond.scope][family]
        first = "a" if family == "promp_s0" else "a_lens"
        if first not in scope:
            continue
        out += [(family, first, other) for other in scope if other != first]
    return out


CONTRAST_HEAD = ("| cell | contrast | users | median error (Nm), first / second "
                 "| Δ median (Nm) [95 % CI] |\n|---|---|---|---|---|")


def _contrast_section(ctx: Ctx) -> list[str]:
    out = [f"## 2. Paired contrasts at k = {K_MAIN}", "",
           "A minus each comparator on ProMP and the lens minus each optimizer on Franks, per "
           "cell. \"(k dropped)\" counts users held by only one of the two cells.", "",
           CONTRAST_HEAD]
    for cond in CONDS:
        for family, a_lr, b_lr in _contrast_pairs(cond):
            name = f"{_label(family, a_lr)} − {_label(family, b_lr)} ({FAMILY_NAME[family]})"
            c = ctx.cell_contrast(cond.cid, family, a_lr, b_lr)
            if c is None:
                out.append(f"| {cond.title} | {name} | {NOT_RUN} | — | — |")
                continue
            users = (f"{c['n']}" + (f" ({c['dropped']} dropped)" if c["dropped"] else "")
                     + prov(c["n"], c["dropped"]))
            out.append(f"| {cond.title} | {name} | {users} | {c['med_a']:.3f} / {c['med_b']:.3f} "
                       f"| {sb._ci(c['diff'])} |")
    return out + [""]


def _rank_family(ctx: Ctx, cond: Cond, family: str):
    """(first place, margin text, n, dropped) for one family, on the users common to every
    optimizer present, with medians as S11 takes them. The first place carries "(not
    separable, CI includes 0)" against the runner-up when the paired interval includes 0, and
    exact ties are named. The margin is the first optimizer (A, or the lens on Franks) minus
    the best of the others, chosen the same way."""
    scope = SCOPES[cond.scope][family]
    if not scope:
        return "—", "—", None, 0
    present = [lr for lr in scope if ctx.df(cond.cid, family, lr) is not None]
    if not present:
        return NOT_RUN, NOT_RUN, None, 0
    aligned, dropped = align([ctx.df(cond.cid, family, lr) for lr in present])
    n = len(aligned[0])
    if not n:
        return NOT_RUN, NOT_RUN, None, 0
    err = {lr: sb.delivered_at(d, K_MAIN).to_numpy() for lr, d in zip(present, aligned)}
    strata = aligned[0]["fold"].to_numpy()
    meds = {lr: float(_med(v)) for lr, v in err.items()}

    def diff(a, b):
        return paired_bootstrap(err[a], err[b], strata, n_boot=ctx.n_boot)["diff_median"]

    def rank(names):
        """names ordered by (median, scope order), and the group tied with the best."""
        order = sorted(names, key=lambda lr: (meds[lr], scope.index(lr)))
        return order, [lr for lr in order if meds[lr] == meds[order[0]]]

    def named(group):
        return " = ".join(_label(family, lr) for lr in group) + (" (tie)" if len(group) > 1 else "")

    order, tied = rank(present)
    first = named(tied)
    if len(tied) == 1 and len(order) > 1:
        c = diff(order[0], order[1])
        if c["hi"] >= 0:
            first = f"{_label(family, order[0])} / {_label(family, order[1])} " \
                    f"(not separable, CI includes 0)"
    missing = [lr for lr in scope if lr not in present]
    if missing:
        first += " (not run: " + ", ".join(_label(family, lr) for lr in missing) + ")"
    lead = "a" if family == "promp_s0" else "a_lens"
    others = [lr for lr in present if lr != lead]
    if lead not in present or not others:
        return first, NOT_RUN, n, dropped
    _, best = rank(others)
    margin = f"{named(best)} {sb._ci(diff(lead, best[0]))}"
    return first, margin, n, dropped


def _ranking_section(ctx: Ctx) -> list[str]:
    out = [f"## 3. Ranking at k = {K_MAIN}", "",
           "One row per condition: the optimizer with the lowest median error, and the margin of "
           "A over the best comparator on ProMP (of the lens over the best direct optimizer on "
           "Franks) as Δ median in Nm with its 95 % CI, naming that comparator. A negative "
           "margin favours A (the lens). Optimizers are ranked on the users common to all of them in "
           "the cell, with medians as S11 takes them; a first place whose interval against the "
           "runner-up includes 0 is printed as \"first / runner-up (not separable)\". The last "
           "column is the cell's perturbation strength (section 9): how far its wearer's "
           "answers move from the plain s = 1.0 wearer's.", "",
           "| condition | ProMP: first place | A − best comparator (Nm) [95 % CI] "
           "| Franks: first place | lens − best direct optimizer (Nm) [95 % CI] "
           "| mean abs Δp vs the plain wearer |",
           "|---|---|---|---|---|---|"]
    strength = perturbation(ctx)
    for cond in CONDS:
        p_first, p_margin, p_n, p_drop = _rank_family(ctx, cond, "promp_s0")
        f_first, f_margin, f_n, f_drop = _rank_family(ctx, cond, "franks")
        ns = [n for n in (p_n, f_n) if n is not None]
        mark = prov(min(ns), p_drop + f_drop) if ns else ""
        out.append(f"| {cond.title}{mark} | {p_first} | {p_margin} | {f_first} | {f_margin} "
                   f"| {_dp(strength, cond.cid)} |")
    return out + [""]


def _s_user_line(ctx: Ctx) -> list[str]:
    df = ctx.df("het", "promp_s0", "a")
    if not _present(df) or "s_user" not in df or df["s_user"].isna().all():
        return ["The s_user distribution of the heterogeneous cell: not run yet or not recorded.",
                ""]
    s = df["s_user"].astype(float)
    q = np.percentile(s, [5, 50, 95])
    return [f"Heterogeneous cell, s_user = 1.0 · exp(0.35 z) over {len(s)} users{prov(len(s))}: "
            f"median {q[1]:.3f}, 5th to 95th percentile {q[0]:.3f} to {q[2]:.3f} (a "
            f"{q[2] / q[0]:.1f}-fold range), minimum {s.min():.3f}, maximum {s.max():.3f}. The "
            f"quality levels of the first heterogeneous tables use the population s = 1.0 (JND "
            f"{JND_MAIN:.3f} Nm), because each wearer's own JND differs; the last tables give "
            f"them in each wearer's own JND, true_jnd(s_user).", ""]


def _generic_distance(ctx: Ctx, cond: Cond) -> str:
    df = ctx.df(cond.cid, "promp_s0", "a")
    if not _present(df):
        return NOT_RUN
    room = (df["d_prior"].astype(float) - df["d_floor"].astype(float)).median()
    return f"{room:.2f} Nm = {room / cond.jnd:.1f} JND{prov(len(df))}"


def _het_own_tables(ctx: Ctx) -> list[str]:
    """The heterogeneous cell's B7 levels in each wearer's own JND, true_jnd(s_user)."""
    cond = BY_ID["het"]
    out = []
    for family in ("promp_s0", "franks"):
        out += [f"### heterogeneous s, levels in each wearer's own JND, {FAMILY_NAME[family]}, "
                f"k = {K_MAIN}", "", B7_HEAD.format(first="optimizer")]
        for lr in SCOPES[cond.scope][family]:
            df, name = ctx.df(cond.cid, family, lr), _label(family, lr)
            if _present(df) and ("s_user" not in df or df["s_user"].isna().any()):
                out.append(f"| {name} | s_user not recorded | — | — | — | — | — | — | — | — |")
                continue
            jnd = (JND_MAIN * df["s_user"].astype(float).to_numpy()) if _present(df) else JND_MAIN
            out.append(b7_row(name, df, K_MAIN, jnd, ctx.n_boot))
        out.append("")
    return out


def _s_sweep_section(ctx: Ctx) -> list[str]:
    out = [f"## 4. The sensitivity sweep at k = {K_MAIN}", "",
           "s = 0.5 is the optimistic reference (5c/5d rows truncated), 1.0 the main condition, "
           "2.0 the coarse wearer, and the heterogeneous cell draws s_user around 1.0. Levels "
           "are in each cell's own JND (E2).", "",
           "How far generic assistance lies from the optimum, on ProMP (median of d_prior − "
           "d_floor, ideal-profile RMS in Nm and in the cell's JND):", "",
           "| cell | JND (Nm) | generic assistance to optimum |", "|---|---|---|"]
    sweep = [BY_ID[c] for c in ("s05", "main", "s2", "het")]
    out += [f"| {c.title} | {c.jnd:.3f} | {_generic_distance(ctx, c)} |" for c in sweep]
    out += ["", "How much personalisation a wearer can perceive depends on their sensitivity: "
            "the same physical improvement is more JNDs for the fine wearer.", ""]
    out += _s_user_line(ctx)
    for cond in sweep:
        out += _b7_tables(ctx, cond, K_MAIN)
    return out + _het_own_tables(ctx)


def _gap_table(ctx: Ctx, cond: Cond) -> list[str]:
    out = ["Target gap of the familiarity cell: the optimizers see a preference biased away "
           "from the true ideal, so part of the error is that bias, not optimizer failure. "
           "The target gap is the median distance in Nm between the biased target and the "
           "true ideal.", "",
           "| family | optimizer | users | median error (Nm) | median target gap (Nm) |",
           "|---|---|---|---|---|"]
    for family in ("promp_s0", "franks"):
        for lr in SCOPES[cond.scope][family]:
            df = ctx.df(cond.cid, family, lr)
            name = _label(family, lr)
            if not _present(df):
                out.append(f"| {FAMILY_NAME[family]} | {name} | {NOT_RUN} | — | — |")
                continue
            gap = (pd.to_numeric(df["familiarity_target_gap_nm"], errors="coerce").median()
                   if "familiarity_target_gap_nm" in df else float("nan"))
            out.append(f"| {FAMILY_NAME[family]} | {name} | {len(df)}{prov(len(df))} | "
                       f"{float(sb.delivered_at(df, K_MAIN).median()):.3f} | {sb._num(gap)} |")
    return out + [""]


def _median_rows(x):
    return _med(x, axis=1)


def _drift_table(ctx: Ctx, cond: Cond) -> list[str]:
    out = [f"Drift cell, {cond.title}. The levels (gain, Q, N90, share) are not reported: the "
           f"start ideal and its floor are not the ideal the wearer ends at. The tracking error "
           f"is the distance of the identified assistance at n = 24 to the final ideal, minus "
           f"that ideal's floor (`tracking_error_nm`, in Nm). The excess is each user's tracking "
           f"error minus the same optimizer's e-main error above the floor at n = 24 "
           f"(distances[23] − d_floor, same user), as a median with a joint cluster bootstrap. "
           f"A drift row's own d_floor is the floor of the start ideal and is not used. The last "
           f"column is Q_opt against the final floor: tracking error ≤ 2 JND "
           f"({2 * cond.jnd:.3f} Nm).", "",
           "Each optimizer's users are also split by whether their ideal touches the device "
           f"limit (|τ*| ≥ {TORQUE_LIMIT_NM:g} Nm − 1e-9 anywhere). The final ideal τ*(1 + g) is "
           "not clipped: where the ideal touches the limit, part of the drift lies beyond the "
           "device, where no delivered curve can follow, so part of that group's excess is out "
           "of reach rather than a tracking failure. The split rows print their user counts "
           "plainly: a subgroup is smaller than a complete cell by construction."
           + ("" if ctx.touches is not None else
              f" The split needs the run directory, which was not found ({ctx.run_dir})."), "",
           "| optimizer | n | median tracking error (Nm) | e-main error above floor (Nm) "
           "| excess over e-main, Nm [95 % CI] | within 2 JND of the final floor |",
           "|---|---|---|---|---|---|"]
    for lr in SCOPES[cond.scope]["promp_s0"]:
        name = _label("promp_s0", lr)
        df, ref = ctx.df(cond.cid, "promp_s0", lr), ctx.df("main", "promp_s0", lr)
        if not _present(df) or "tracking_error_nm" not in df:
            out.append(f"| {name} | {NOT_RUN} | — | — | — | — |")
            continue
        within = float((pd.to_numeric(df["tracking_error_nm"]) <= 2 * cond.jnd + 1e-12).mean())
        tracking = float(pd.to_numeric(df["tracking_error_nm"]).median())
        if not _present(ref):
            out.append(f"| {name} | {len(df)}{prov(len(df))} | {tracking:.3f} | {NOT_RUN} | — "
                       f"| {within:.2f} |")
            continue
        (d, r), dropped = align([df, ref])
        if not len(d):
            out.append(f"| {name} | {len(df)}{prov(len(df))} | {tracking:.3f} | {NOT_RUN} | — "
                       f"| {within:.2f} |")
            continue
        above = (sb.delivered_at(r, K_MAIN) - r["d_floor"].astype(float)).to_numpy()
        excess = pd.to_numeric(d["tracking_error_nm"]).to_numpy() - above
        ci = joint_bootstrap([excess], d["fold"].to_numpy(), _median_rows, ctx.n_boot)
        out.append(f"| {name} | {len(d)}{prov(len(d), dropped)} | {tracking:.3f} | "
                   f"{float(_med(above)):.3f} | {sb._ci(ci)} | {within:.2f} |")
        if ctx.touches is not None:
            out += _drift_split_rows(ctx, cond, name, d, above, excess)
    return out + [""]


def _drift_split_rows(ctx: Ctx, cond: Cond, name: str, d, above, excess) -> list[str]:
    """The drift row of one optimizer, split by whether the ideal touches the device limit."""
    touch = np.array([ctx.touches(f, u) for f, u in zip(d["fold"], d["user_id"])], dtype=bool)
    tracking = pd.to_numeric(d["tracking_error_nm"]).to_numpy()
    folds = d["fold"].to_numpy()
    out = []
    for label, mask in (("ideal touches ±12 Nm", touch), ("ideal within ±12 Nm", ~touch)):
        if not mask.any():
            out.append(f"| {name}, {label} | 0 | — | — | — | — |")
            continue
        ci = joint_bootstrap([excess[mask]], folds[mask], _median_rows, ctx.n_boot)
        within = float((tracking[mask] <= 2 * cond.jnd + 1e-12).mean())
        out.append(f"| {name}, {label} | {int(mask.sum())} | {np.median(tracking[mask]):.3f} | "
                   f"{float(_med(above[mask])):.3f} | {sb._ci(ci)} | {within:.2f} |")
    return out


def _robust_section(ctx: Ctx) -> list[str]:
    out = [f"## 5. Observer models, device noise, drift and lapse at k = {K_MAIN}", "",
           "All at s = 1.0, levels in 0.674 Nm (not for the drift cells). The paired contrasts "
           "of these cells are in section 2. Every error here is the RMS distance to the "
           "ideal, A's own model class; section 8 scores the observer cells in the wearer's own "
           f"distance. The static landmark cell uses a 10 % magnitude JND, the walking one 25 % "
           f"(spec amendment E4b), both floored at 0.5 Nm. {ASYMMETRY_NOTE} {GATE_NOTE}", ""]
    for cid in ("landmark", "landmark25", "weber", "asym", "famil", "noise", "drift1", "drift2",
                "lapse2", "lapse3"):
        cond = BY_ID[cid]
        if cid.startswith("drift"):
            out += _drift_table(ctx, cond)
            continue
        out += _b7_tables(ctx, cond, K_MAIN)
        if cid == "famil":
            out += _gap_table(ctx, cond)
    return out


# --------------------------------------------------------------------------- tuning

def _tune_rows(ctx: Ctx) -> list[dict]:
    return _parse_rows(ctx.dir5e / "e-tune.jsonl")


def _selection(ctx: Ctx):
    """(table, error) from e-tune via phase5e_tune_select.select, which picks on loso-AB30."""
    from phase5e_tune_select import select

    rows = _tune_rows(ctx)
    if not rows:
        return None, NOT_RUN
    try:
        return select(rows)[1], None
    except SystemExit as exc:
        return None, f"{NOT_RUN} ({exc})"


def _tuned_df(ctx: Ctx, learner: str):
    df = ctx.blocks["e-tuned"]
    if df is None:
        return None
    sel = df[(df["learner"] == learner) & (df["family"] == "promp_s0")]
    sel = sel[sel["fold"].isin(EVAL_FOLDS)].drop_duplicates(KEY, keep="last")
    return sel if len(sel) else None


def _held_out(df):
    """The e-main rows of the four evaluation folds."""
    return None if df is None else df[df["fold"].isin(EVAL_FOLDS)]


def _against_default_and_a(ctx: Ctx, learner: str, tuned, tag: str) -> dict | None:
    """A tuned comparator's rows on the held-out folds against its default (e-main) and A
    (e-main), all on the users the three cells hold in common: ``gain`` is the joint bootstrap
    of median(default) − median(tuned), ``lead`` the paired contrast A − tuned (each None when
    its cell is missing), ``n`` and ``dropped`` the common and the dropped users, and the
    three medians on them."""
    if not _present(tuned):
        return None
    default = _held_out(ctx.df("main", "promp_s0", learner))
    a_main = _held_out(ctx.df("main", "promp_s0", "a"))
    frames = [tuned] + [d for d in (default, a_main) if _present(d)]
    aligned, dropped = align(frames)
    tuned = aligned[0]
    default = aligned[1] if _present(default) else None
    a_main = aligned[-1] if _present(a_main) else None
    gain = (ctx.joint([default, tuned], K_MAIN, lambda d, t: _med(d, 1) - _med(t, 1),
                      key=(tag + "-gain", learner)) if _present(default) else None)
    lead = (ctx.contrast(a_main, tuned, K_MAIN, key=(tag + "-vs-a", learner))
            if _present(a_main) else None)

    def med(df):
        return float(_med(sb.delivered_at(df, K_MAIN))) if _present(df) else None

    return dict(tuned=tuned, gain=gain, lead=lead, n=len(tuned), dropped=dropped,
                med_tuned=med(tuned), med_default=med(default), med_a=med(a_main))


def _tuning(ctx: Ctx, learner: str):
    """(tuned rows, gain, lead) of one comparator on the held-out folds, on the users common
    to the tuned, default and A cells (:func:`_against_default_and_a`)."""
    res = _against_default_and_a(ctx, learner, _tuned_df(ctx, learner), "tune")
    if res is None:
        return None, None, None
    return res["tuned"], res["gain"], res["lead"]


def _users_cell(n: int, dropped: int, expected: int) -> str:
    return f"{n}" + (f" ({dropped} dropped)" if dropped else "") + prov(n, dropped, expected)


#: The setting each comparator runs with in e-main: LineCoSpar at Phase 4's tuned multipliers,
#: G at Arens' kappa0, SortCMA at pycma's popsize for d = 8 (4 + floor(3 ln 8) = 10) and sigma0.
DEFAULT_SETTINGS = {"d_z": dict(DZ_TUNED), "g": dict(g_kappa0=KAPPA0),
                    "cmaes": dict(cma_popsize=10, cma_sigma_mult=1.0)}


def _is_default(learner: str, values: dict) -> bool:
    default = DEFAULT_SETTINGS[learner]
    return all(f in values and np.isclose(float(values[f]), float(default[f])) for f in default)


def _setting_text(learner: str, values: dict) -> str:
    """``field = value`` pairs, marked when they are the comparator's e-main default (its
    tuning gain is then 0 by construction)."""
    text = ", ".join(f"{f} = {float(values[f]):g}" for f in TUNED_FIELDS[learner] if f in values)
    return text + (" (= its e-main default)" if _is_default(learner, values) else "")


def _tuning_section(ctx: Ctx) -> list[str]:
    out = ["## 6. Fair tuning of the comparators (E8)", "",
           f"Each comparator's setting is chosen on {TUNE_FOLD} only (its 40 population users; "
           f"the fold's 4 stress users are not used; s = 1.0, ProMP) by the lowest median error "
           f"at k = {K_MAIN}, and evaluated on the other four folds against A, which is not "
           f"retuned. The gain from tuning is the median error with the default setting "
           f"(e-main) minus the tuned one, with a joint cluster bootstrap. Every number of a "
           f"held-out row is on the users the tuned, default and A cells hold in common.", ""]
    table, err = _selection(ctx)
    if table is None:
        out += [f"Selection table: {err}.", ""]
    else:
        out += ["| optimizer | setting | n | median error (Nm) | p75 (Nm) | chosen |",
                "|---|---|---|---|---|---|"]
        for t in sorted(table, key=lambda t: (t["learner"], t["median"], t["p75"])):
            fields = ", ".join(f"{f} = {t[f]:g}" for f in TUNED_FIELDS[t["learner"]])
            out.append(f"| {_label('promp_s0', t['learner'])} | {fields} | "
                       f"{t['n']}{prov(t['n'], expected=EXPECTED_N_TUNE)} | {t['median']:.3f} | "
                       f"{t['p75']:.3f} | {'yes' if t['chosen'] else ''} |")
        out.append("")
    out += [f"Held-out folds ({len(EVAL_FOLDS)}), error at k = {K_MAIN}:", "",
            "| optimizer | tuned setting | users | default median (Nm) | tuned median (Nm) "
            "| gain from tuning (default − tuned) [95 % CI] | A − tuned [95 % CI] |",
            "|---|---|---|---|---|---|---|"]
    for learner in TUNED_ORDER:
        name = _label("promp_s0", learner)
        rows = _tuned_df(ctx, learner)
        res = _against_default_and_a(ctx, learner, rows, "tune")
        if res is None:
            out.append(f"| {name} | {NOT_RUN} | — | — | — | — | — |")
            continue
        gain, lead = res["gain"], res["lead"]
        setting = _setting_text(learner, rows.iloc[0].to_dict())
        out.append(
            f"| {name} | {setting} | {_users_cell(res['n'], res['dropped'], EXPECTED_N_TUNED)} | "
            f"{sb._num(res['med_default'])} | {sb._num(res['med_tuned'])} | "
            f"{'—' if gain is None else sb._ci(gain)} | "
            f"{'—' if lead is None else sb._ci(lead['diff'])} |")
    return out + [""]


# --------------------------------------------------------------------------- oracle bound

ORACLE_LABEL = "upper bound on what tuning could buy; not a fair comparison"


def _setting_df(df, learner: str, fields: dict):
    """One setting's rows of a tuning block on the evaluation folds."""
    if df is None:
        return None
    sel = df[(df["learner"] == learner).to_numpy() & (df["family"] == "promp_s0").to_numpy()]
    for name, value in fields.items():
        sel = sel[np.isclose(pd.to_numeric(sel[name], errors="coerce"), float(value))]
    sel = sel[sel["fold"].isin(EVAL_FOLDS)].drop_duplicates(KEY, keep="last")
    return sel if len(sel) else None


def _oracle_section(ctx: Ctx) -> list[str]:
    from phase5e_tune_select import select

    out = ["## 7. Oracle-tuned bound (E8a)", "",
           f"**An {ORACLE_LABEL}.** Every e-tune setting, plus G at κ₀ = 4.5 and 6.5 and SortCMA "
           f"at σ multiplier 0.25 for popsize 4, 6 and 10, runs on the four evaluation folds. "
           f"Each comparator's best setting is then picked *in sample*, on those same folds and "
           f"users, by the lowest median error at k = {K_MAIN} over population users (ties to "
           f"the lower p75, as for e-tune). Selection and evaluation share their users, so the "
           f"bound is optimistic by construction, and A is not tuned. The one-fold tuning of "
           f"section 6 stays the fair comparison.", "",
           "The 20 oracle settings are not all new: 5 of them repeat settings that other "
           "blocks already ran (880 of the 3,520 sessions). 528 are key-identical to the "
           "e-tuned settings, and 352 are the e-main defaults, G at κ₀ = 1.5 and SortCMA at "
           "(popsize 10, σ multiplier 1.0). LineCoSpar at (2, 0.5) is both a default and the "
           "tuned choice. The repeats are the same sessions under the same keys, so they "
           "cost compute and change neither the selection nor the bound.", ""]
    rows = _parse_rows(ctx.dir5e / "e-oracle.jsonl")
    if not rows:
        return out + [f"e-oracle: {NOT_RUN}.", ""]
    try:
        tuned, table = select(rows)
    except SystemExit as exc:
        return out + [f"e-oracle: {NOT_RUN} ({exc}).", ""]
    out += ["| optimizer | setting | n | median error (Nm) | p75 (Nm) | best in sample |",
            "|---|---|---|---|---|---|"]
    for t in sorted(table, key=lambda t: (t["learner"], t["median"], t["p75"])):
        fields = ", ".join(f"{f} = {t[f]:g}" for f in TUNED_FIELDS[t["learner"]])
        out.append(f"| {_label('promp_s0', t['learner'])} | {fields} | "
                   f"{t['n']}{prov(t['n'], expected=EXPECTED_N_TUNED)} | {t['median']:.3f} | "
                   f"{t['p75']:.3f} | {'yes' if t['chosen'] else ''} |")
    out += ["", f"Best setting in sample, error at k = {K_MAIN}, on the users the oracle-tuned, "
            f"default (e-main) and A (e-main) cells hold in common ({ORACLE_LABEL}):", "",
            "| optimizer | best setting in sample | users | default median (Nm) "
            "| oracle-tuned median (Nm) | gain over the default (default − oracle-tuned) "
            "[95 % CI] | A − oracle-tuned [95 % CI] |",
            "|---|---|---|---|---|---|---|"]
    for learner in TUNED_ORDER:
        name = _label("promp_s0", learner)
        fields = tuned.get(learner, {})
        setting = _setting_text(learner, fields) if fields else ""
        res = _against_default_and_a(
            ctx, learner, _setting_df(ctx.blocks["e-oracle"], learner, fields), "oracle")
        if res is None:
            out.append(f"| {name} | {setting or NOT_RUN} | {NOT_RUN} | — | — | — | — |")
            continue
        gain, lead = res["gain"], res["lead"]
        out.append(
            f"| {name} | {setting} | {_users_cell(res['n'], res['dropped'], EXPECTED_N_TUNED)} | "
            f"{sb._num(res['med_default'])} | {sb._num(res['med_tuned'])} | "
            f"{'—' if gain is None else sb._ci(gain)} | "
            f"{'—' if lead is None else sb._ci(lead['diff'])} |")
    return out + [""]


# --------------------------------------------------------------------------- own metric (E4a)

def _own_label(family: str, learner: str) -> str:
    return f"{_label(family, learner)} ({FAMILY_NAME[family]})"


def _own_medians(df, field: str) -> str:
    return " / ".join(f"{float(_med(values_at(df, k, field))):.3f}" for k in KS)


def _best_own_comparator(ctx: Ctx, cid: str):
    """The ProMP comparator with the lowest median own distance at k = 24, on the users A and
    every comparator present hold in common (ties to the scope order); None if none."""
    scope = SCOPES[BY_ID[cid].scope]["promp_s0"]
    others = [lr for lr in scope if lr != "a"
              and _present(ctx.df(cid, "promp_s0", lr))
              and has_field(ctx.df(cid, "promp_s0", lr), OWN)]
    a = ctx.df(cid, "promp_s0", "a")
    if not others or not _present(a) or not has_field(a, OWN):
        return None
    aligned, _ = align([a] + [ctx.df(cid, "promp_s0", lr) for lr in others])
    if not len(aligned[0]):
        return None
    meds = {lr: float(_med(values_at(d, K_MAIN, OWN))) for lr, d in zip(others, aligned[1:])}
    return min(others, key=lambda lr: (meds[lr], scope.index(lr)))


OWN_CONTRAST_HEAD = ("| contrast | users | own distance (Nm-eq.), first / second "
                     f"| Δ median, own distance (Nm-eq.) [95 % CI] "
                     f"| Δ median, RMS error (Nm) [95 % CI] |\n|---|---|---|---|---|")


def _own_contrast_row(ctx: Ctx, cid: str, family: str, first: str, second: str,
                      name: str) -> str:
    own = ctx.cell_contrast(cid, family, first, second, field=OWN)
    if own is None:
        return f"| {name} | {NOT_RUN} or not recorded | — | — | — |"
    rms = ctx.cell_contrast(cid, family, first, second)
    users = _users_cell(own["n"], own["dropped"], EXPECTED_N)
    return (f"| {name} | {users} | {own['med_a']:.3f} / {own['med_b']:.3f} | "
            f"{sb._ci(own['diff'])} | {'—' if rms is None else sb._ci(rms['diff'])} |")


def _own_section(ctx: Ctx) -> list[str]:
    jnd = JND_MAIN
    out = ["## 8. The wearer's own outcome (E4a)", "",
           "The RMS distance to the ideal is A's own model class, and a wearer who judges "
           "otherwise does not feel it. Each observer cell therefore also records, per "
           "comparison and noise-free, the wearer's own distance of the identified assistance, "
           "−utility: the landmark wearer's feature distance D × true_jnd(1.0) (Nm-equivalent; "
           f"one feature-JND answers like {jnd:.3f} Nm of RMS) and D itself in feature-JNDs; the "
           "weighted RMS of the asymmetric wearer; the RMS to the biased target 0.8 τ* of the "
           "familiar wearer. The Weber wearer changes only the probit scale, so its own distance "
           "is its RMS error. Medians are lower middle values (inverted_cdf); the contrasts are "
           f"at k = {K_MAIN}, A − best comparator against the comparator with the lowest median "
           "own distance. own_feature_jnd is heavy-tailed, because the landmark flexion peak "
           "can switch between humps, and medians are reported for that reason.", ""]
    for cid in VARIANT_CELLS:
        cond = BY_ID[cid]
        landmark = cond.fields.get("observer_metric") == "landmark"
        out += [f"### {cond.title}", "",
                "| optimizer | n | own distance, k = 12 / 24 / 40 (Nm-eq.) "
                + ("| own distance, k = 12 / 24 / 40 (feature-JND) " if landmark else "")
                + f"| RMS error, k = {K_MAIN} (Nm) |",
                "|---|---|---|" + ("---|" if landmark else "") + "---|"]
        for family in ("promp_s0", "franks"):
            for lr in SCOPES[cond.scope][family]:
                df, name = ctx.df(cid, family, lr), _own_label(family, lr)
                blank = " — |" if landmark else ""
                if not _present(df):
                    out.append(f"| {name} | {NOT_RUN} | — |{blank} — |")
                    continue
                if not has_field(df, OWN, max(KS)):
                    out.append(f"| {name} | {len(df)} | not recorded (rows predate E4a) |"
                               f"{blank} — |")
                    continue
                feature = (f" {_own_medians(df, FEATURE)} |"
                           if landmark and has_field(df, FEATURE, max(KS)) else blank)
                out.append(f"| {name} | {len(df)}{prov(len(df))} | {_own_medians(df, OWN)} |"
                           f"{feature} {float(_med(sb.delivered_at(df, K_MAIN))):.3f} |")
        g = _label("promp_s0", "g")
        out += ["", OWN_CONTRAST_HEAD,
                _own_contrast_row(ctx, cid, "promp_s0", "a", "g", f"A − {g} (ProMP)"),
                _own_contrast_row(ctx, cid, "franks", "a_lens", "g",
                                  f"lens − {_label('franks', 'g')} (Franks)")]
        best = _best_own_comparator(ctx, cid)
        if best is None:
            out.append(f"| A − best comparator (ProMP) | {NOT_RUN} or not recorded | — | — | — |")
        else:
            out.append(_own_contrast_row(ctx, cid, "promp_s0", "a", best,
                                         f"A − best comparator ({_label('promp_s0', best)}) "
                                         f"(ProMP)"))
        out.append("")
    return out


# --------------------------------------------------------------------------- perturbation

def _perturbation_cells() -> dict:
    """The point fields of each condition's wearer; the s = 0.5 reference is s = 0.5."""
    return {c.cid: (dict(s=0.5) if c.cid == "s05" else dict(c.fields)) for c in CONDS}


def perturbation_params(run_dir) -> dict:
    """What a cached perturbation strength was computed with; a cache with other parameters
    is not read."""
    return json.loads(json.dumps(dict(run_dir=str(run_dir), folds=list(LOSO),
                                      n_users=pt.N_USERS, n_pairs=pt.N_PAIRS, seed=pt.SEED,
                                      cells=_perturbation_cells())))


def write_perturbation_cache(path, params: dict, values: dict) -> None:
    Path(path).write_text(json.dumps(dict(params=params, values=values), indent=1,
                                     sort_keys=True), encoding="utf-8")


def perturbation(ctx: Ctx):
    """``{cell id: {mean_abs_dp, ...}}`` from the cache, or computed and cached; a string
    saying why when it cannot be had."""
    if ctx._perturbation is not None:
        return ctx._perturbation
    if ctx.run_dir is None:
        result = "not computed: no e-main rows name the run directory"
    else:
        params = perturbation_params(ctx.run_dir)
        cached = None
        if ctx.perturbation_cache.exists():
            cached = json.loads(ctx.perturbation_cache.read_text(encoding="utf-8"))
        if cached is not None and cached.get("params") == params:
            result = cached["values"]
        elif not (Path(ctx.run_dir) / "users").is_dir():
            result = f"not computed: run directory {ctx.run_dir} not found"
        else:
            result = pt.compute(ctx.run_dir, params["cells"], folds=params["folds"],
                                n_users=params["n_users"], n_pairs=params["n_pairs"])
            write_perturbation_cache(ctx.perturbation_cache, params, result)
    ctx._perturbation = result
    return result


LANDMARK_CELLS = ("landmark", "landmark25")


def decisiveness_cache_path(perturbation_cache) -> Path:
    """The decisiveness cache, next to the perturbation cache: ``<stem>_decisiveness.json``."""
    p = Path(perturbation_cache)
    return p.with_name(f"{p.stem}_decisiveness{p.suffix}")


def decisiveness_params(run_dir) -> dict:
    """What a cached decisiveness was computed with (the landmark cells, on the perturbation
    section's users and pairs); a cache with other parameters is not read."""
    cells = {cid: fields for cid, fields in _perturbation_cells().items()
             if cid in LANDMARK_CELLS}
    return json.loads(json.dumps(dict(run_dir=str(run_dir), folds=list(LOSO),
                                      n_users=pt.N_USERS, n_pairs=pt.N_PAIRS, seed=pt.SEED,
                                      cells=cells)))


def decisiveness(ctx: Ctx):
    """``{cell id: {ratio, s_equiv, ...}}`` of the landmark cells from the cache, or computed
    and cached; a string saying why when it cannot be had."""
    if ctx._decisiveness is not None:
        return ctx._decisiveness
    if ctx.run_dir is None:
        result = "not computed: no e-main rows name the run directory"
    else:
        params = decisiveness_params(ctx.run_dir)
        path = decisiveness_cache_path(ctx.perturbation_cache)
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if cached is not None and cached.get("params") == params:
            result = cached["values"]
        elif not (Path(ctx.run_dir) / "users").is_dir():
            result = f"not computed: run directory {ctx.run_dir} not found"
        else:
            result = pt.compute_decisiveness(ctx.run_dir, params["cells"],
                                             folds=params["folds"], n_users=params["n_users"],
                                             n_pairs=params["n_pairs"])
            write_perturbation_cache(path, params, result)
    ctx._decisiveness = result
    return result


def _qe3_decisiveness(ctx: Ctx) -> str:
    """The one QE3 sentence on what the landmark cells do and do not control for."""
    head = "  - **QE3: neither landmark cell isolates the home advantage.** "
    dec = decisiveness(ctx)
    values = []
    if not isinstance(dec, str):
        for cid in LANDMARK_CELLS:
            entry = dec.get(cid) or {}
            values.append(entry.get("s_equiv"))
    if isinstance(dec, str) or any(v is None for v in values):
        return (head + "Their decisiveness against the plain s = 1.0 wearer was not computed "
                f"here ({dec if isinstance(dec, str) else 'cache incomplete'}); see section 9.")
    return (head + "Neither landmark cell isolates the home advantage from sensitivity: their "
            f"decisiveness corresponds to RMS wearers of s ≈ {values[0]:.2f} (static) and "
            f"{values[1]:.2f} (walking), so the s = 0.5 reading is the closer control for both.")


def _dp(strength, cid: str) -> str:
    if isinstance(strength, str) or cid not in strength:
        return "—"
    value = strength[cid]["mean_abs_dp"]
    return "n/a" if value is None else f"{value:.3f}"


def _perturbation_section(ctx: Ctx) -> list[str]:
    out = ["## 9. Perturbation strength", "",
           "How differently each cell's wearer answers than the plain s = 1.0 wearer: the mean "
           "absolute difference |Δp| in the probability of preferring the first curve, over "
           f"{pt.N_PAIRS} duel pairs per user from the fold's ProMP prior (whitened draws in the "
           f"tight arm, device-clipped, seeded per user), {pt.N_USERS} population users per "
           "fold, five folds. Both wearers are compared without lapse, except in the lapse "
           "cells, whose lapse is their only perturbation (0.2 or 0.3 against 0.1). The noise "
           "cell's wearer judges the noisy curves; a drift cell is averaged over its 24 "
           "comparisons' ideals. A cell whose |Δp| is small barely changes the answers, so A "
           "keeping its rank there says little about robustness.", ""]
    strength = perturbation(ctx)
    if isinstance(strength, str):
        return out + [f"Perturbation strength {strength}.", ""]
    out += ["| cell | mean abs Δp | users | pairs | note |", "|---|---|---|---|---|"]
    for cond in CONDS:
        entry = strength.get(cond.cid)
        if entry is None:
            out.append(f"| {cond.title} | — | — | — | not in the cache |")
            continue
        note = entry.get("note") or ("lapse included: it is the cell's perturbation"
                                     if entry.get("lapse_included") else "")
        out.append(f"| {cond.title} | {_dp(strength, cond.cid)} | {entry['n_users']} | "
                   f"{entry['n_pairs']} | {note} |")
    out += ["", "**Decisiveness of the landmark cells.** The landmark wearer's utility is "
            "−D × true_jnd(1.0), the plain RMS wearer's −RMS distance, both in Nm on the same "
            "probit scale. The ratio is the median, over the same duel pairs, of the "
            "landmark wearer's |Δu| over the plain s = 1.0 wearer's |Δu| (pairs the RMS wearer "
            "cannot tell apart drop out); the RMS-equivalent sensitivity is s ≈ 1 / ratio. A "
            "ratio above 1 means the landmark wearer answers more decisively than the plain "
            "s = 1.0 wearer, so part of A's changed advantage over G is sensitivity and not "
            "the home advantage. The two landmark cells bracket the range of magnitude JNDs "
            "(10 % and 25 % of the peak), not the range of plausible wearers.", ""]
    dec = decisiveness(ctx)
    if isinstance(dec, str):
        return out + [f"Decisiveness {dec}.", ""]
    out += ["| cell | decisiveness ratio | RMS-equivalent s ≈ 1 / ratio | users | pairs | note |",
            "|---|---|---|---|---|---|"]
    for cid in LANDMARK_CELLS:
        entry, title = dec.get(cid), BY_ID[cid].title
        if entry is None or entry.get("ratio") is None:
            out.append(f"| {title} | — | — | — | — | "
                       f"{'not in the cache' if entry is None else entry.get('note', '')} |")
            continue
        out.append(f"| {title} | {entry['ratio']:.2f} | {entry['s_equiv']:.2f} | "
                   f"{entry['n_users']} | {entry['n_pairs']} | |")
    return out + [""]


# --------------------------------------------------------------------------- gain over generic

def _gain_section(ctx: Ctx) -> list[str]:
    out = [f"## 10. Gain over generic assistance at k = {K_MAIN}", "",
           "Robustness framed by benefit rather than rank: each optimizer's median gain over "
           "generic assistance (the prior's mean curve, d_prior − error at k), in Nm and in the "
           "cell's JND (E2), and the share of users whose identified assistance ends farther "
           "from the ideal than generic assistance. The drift cells are left out: their ideal "
           "moves, and d_prior belongs to the start ideal. Medians are the usual ones, as in "
           "the B7 tables.", "",
           "| cell | family | optimizer | users | median gain (Nm) | median gain (cell JND) "
           "| share ending worse than generic assistance |",
           "|---|---|---|---|---|---|---|"]
    for cond in CONDS:
        if cond.cid.startswith("drift"):
            continue
        for family in ("promp_s0", "franks"):
            for lr in SCOPES[cond.scope][family]:
                df = ctx.df(cond.cid, family, lr)
                head = f"| {cond.title} | {FAMILY_NAME[family]} | {_label(family, lr)} |"
                if not _present(df):
                    out.append(f"{head} {NOT_RUN} | — | — | — |")
                    continue
                error = sb.delivered_at(df, K_MAIN)
                start = df["d_prior"].astype(float)
                gain = float((start - error).median())
                worse = float((error > start).mean())
                out.append(f"{head} {len(df)}{prov(len(df))} | {gain:.3f} | "
                           f"{gain / cond.jnd:.2f} | {worse:.2f} |")
    return out + [""]


# --------------------------------------------------------------------------- the predictions

def _verdict(number: int, title: str, fails: list[str], missing: list[str], facts: list[str],
             consequence: list[str], uncertain: list[str] | None = None, n: int | None = None,
             expected: int | None = None, head: str | None = None) -> str:
    """"failed." as soon as a part fails (with the consequence, and any missing parts too);
    otherwise "not run yet." while a part is missing, "uncertain." while an interval straddles
    its threshold, and "held." when every part was computed and came true. A verdict on fewer
    than the expected users carries "(provisional, n = X of N)". ``head`` replaces the line's
    "- **QEn (title).** " opening (a second reading of one prediction)."""
    uncertain = uncertain or []
    head = head or f"- **QE{number} ({title}).** "
    mark = prov(n, expected=expected) if n is not None else ""
    also = [f"Also not run yet: {'; '.join(missing)}."] if missing else []
    if fails:
        return head + "failed." + mark + " " + " ".join(facts + fails + uncertain + also
                                                         + consequence)
    if missing:
        return head + " ".join([f"{NOT_RUN}."] + facts + uncertain
                               + ["Missing: " + "; ".join(missing) + "."])
    if uncertain:
        return head + "uncertain." + mark + " " + " ".join(facts + uncertain)
    return head + "held." + mark + " " + " ".join(facts)


def _n_min(cs) -> int | None:
    ns = [c["n"] for c in cs if c is not None]
    return min(ns) if ns else None


def _n_range(cs) -> str:
    ns = [c["n"] for c in cs if c is not None]
    if not ns:
        return "n = 0"
    return f"n = {min(ns)}" if min(ns) == max(ns) else f"n = {min(ns)} to {max(ns)}"


def _side(ci: dict, threshold: float, want_low: bool):
    """"held" when the interval lies wholly on the predicted side of ``threshold`` (below it
    if ``want_low``, above it otherwise), "failed" when wholly on the other side, else None."""
    if want_low:
        return "held" if ci["hi"] <= threshold else "failed" if ci["lo"] > threshold else None
    return "held" if ci["lo"] > threshold else "failed" if ci["hi"] < threshold else None


def _ci3(ci: dict) -> str:
    return sb._ci(ci)


ANALYST = "Analyst's reading, not a registered consequence:"
KILL = "Registered kill criterion:"
#: QE1 contrasts that rows existing before the design could already answer
PRE_REGISTRATION = ("g", "cmaes")
PRE_REGISTRATION_TAG = "computable before registration (prefix of 5c/5d rows)"


def _qe1_contrasts(ctx: Ctx):
    """QE1's contrasts as."""
    out = []
    for family, lead, others in (("promp_s0", "a", ("g", "d_z", "cmaes")),
                                 ("franks", "a_lens", ("a", "g", "d_z", "cmaes"))):
        for other in others:
            name = f"{_label(family, lead)} − {_label(family, other)} ({FAMILY_NAME[family]})"
            out.append((name, ctx.cell_contrast("main", family, lead, other),
                        family == "promp_s0" and other in PRE_REGISTRATION))
    return out


def _qe1(ctx: Ctx) -> str:
    """QE1's verdict over the new predictions only (A − LineCoSpar on ProMP and the Franks
    contrasts); the contrasts that rows existing before the design could answer are in
    :func:`_qe1_before`."""
    fails, missing, facts, cs = [], [], [], []
    for name, c, before in _qe1_contrasts(ctx):
        if before:
            continue
        if c is None:
            missing.append(name)
            continue
        cs.append(c)
        facts.append(f"{_cstr(name, c)};")
        if not _ahead(c):
            fails.append(f"{name} does not exclude 0 on the favourable side.")
    if facts:
        facts[-1] = facts[-1][:-1] + "."
    facts = ([f"{_n_range(cs)}, k = {K_MAIN}, Δ median in Nm, the new predictions only "
              "(A − LineCoSpar on ProMP and the Franks contrasts):"] + facts)
    consequence = [ANALYST + " claim 1 at the main condition is restricted to the contrasts "
                   "whose interval excludes 0."]
    return _verdict(1, "main condition, k = 24", fails, missing, facts, consequence,
                    n=_n_min(cs))


def _qe1_before(ctx: Ctx) -> str:
    """The QE1 contrasts computable before test plan, apart from the verdict: e-main is an
    exact prefix of the 5c/5d s = 1.0 rows, so these are not confirmed predictions."""
    parts = []
    for name, c, before in _qe1_contrasts(ctx):
        if not before:
            continue
        if c is None:
            parts.append(f"{name} {NOT_RUN}")
        else:
            parts.append(f"{_cstr(name, c)}, "
                         f"{'consistent' if _ahead(c) else 'not consistent'} with the prediction")
    return ("  - **QE1, computed before registration (not confirmed predictions, not in the "
            f"verdict above; k = {K_MAIN}, Δ median in Nm).** " + "; ".join(parts) + ". "
            "e-main is an exact prefix of the 5c/5d s = 1.0 rows, so these two could be "
            "computed from rows that existed before the spec.")


def _qe2(ctx: Ctx) -> str:
    fails, missing, facts, cs, killed = [], [], [], [], []
    closest = None
    for cid in QE2_CELLS:
        title = BY_ID[cid].title
        for other in ("g", "d_z", "cmaes"):
            name = _label("promp_s0", other)
            c = ctx.cell_contrast(cid, "promp_s0", "a", other)
            if c is None:
                missing.append(f"{title}: A − {name}")
                continue
            cs.append(c)
            est = c["diff"]["estimate"]
            if closest is None or est > closest[0]:
                closest = (est, f"{title}, A − {name}", c)
            if est >= 0:
                fails.append(f"{title}: {name} is not above A ({_cstr('A − ' + name, c)}).")
                if _behind(c):
                    killed.append(title)
    if closest is not None:
        facts.append(f"{_n_range(cs)} per contrast, k = {K_MAIN}; closest comparator: "
                     f"{closest[1]} {sb._ci(closest[2]['diff'])}.")
    facts.append("The registered cell list was extended, before any landmark cell ran, to "
                 "include the walking landmark cell (E4b); this is stricter than the original "
                 "criterion.")
    if killed:
        consequence = [KILL + " a comparator lower than A with an interval excluding 0 in "
                       f"{', '.join(sorted(set(killed)))}: claim 1 restricted to the cells "
                       "where it holds."]
    else:
        consequence = [KILL + " not met: no comparator is lower than A with an interval "
                       "excluding 0."]
    return _verdict(2, "ranking stability, k = 24", fails, missing, facts, consequence,
                    n=_n_min(cs))


def _qe3_stat(la, lg, ma, mg):
    """QE3(b)'s D: A's advantage over G in the landmark cell minus 2/3 of it in the reference."""
    return (_med(lg, 1) - _med(la, 1)) - 2 / 3 * (_med(mg, 1) - _med(ma, 1))


def _qe3_read(ctx: Ctx, cid: str, field: str, ref: str) -> dict:
    """QE3's parts on landmark cell ``cid``, its errors read in ``field`` (the RMS error, or
    the wearer's own distance), against reference cell ``ref``, read in the RMS error -- the
    reference wearer judges by RMS, so that is its own distance too. ``lm`` is (a)'s contrast,
    ``adv`` the reference advantage over G, ``diff`` D; ``present`` whether all four cells hold
    what they need."""
    lm = ctx.cell_contrast(cid, "promp_s0", "a", "g", field=field)
    cells = [ctx.df(c, "promp_s0", lr) for c, lr in ((cid, "a"), (cid, "g"), (ref, "a"),
                                                      (ref, "g"))]
    present = (all(_present(d) for d in cells)
               and all(has_field(d, field) for d in cells[:2]))
    adv = diff = None
    if present:
        adv = ctx.joint(cells[2:], K_MAIN, lambda a, g: _med(g, 1) - _med(a, 1),
                        key=("qe3-adv", ref))
        diff = ctx.joint(cells, K_MAIN, _qe3_stat, key=("qe3-b", cid, field, ref),
                         fields=[field, field, "distances", "distances"])
    return dict(lm=lm, adv=adv, diff=diff, present=present)


def _qe3_b_state(adv: dict, diff: dict) -> tuple[str, str]:
    """QE3(b)'s state from its two intervals, by the amended verdict rule, and why."""
    if adv["hi"] <= 0:
        return "failed", "A has no advantage over G in the reference cell"
    if diff["lo"] > 0:
        return "failed", "the advantage shrinks by less than a third"
    if diff["hi"] > 0 or adv["lo"] <= 0:
        return "uncertain", "an interval straddles its threshold"
    return "held", ""


def _metric_name(field: str) -> str:
    return "the RMS error" if field == "distances" else "the wearer's own distance"


def _qe3(ctx: Ctx, field: str = "distances", head: str | None = None) -> str:
    """QE3 on the planned (static) landmark cell, read in ``field``."""
    g_name = _label("promp_s0", "g")
    metric = _metric_name(field)
    title = f"landmark observer, k = {K_MAIN}, read in the RMS error to the ideal"
    fails, missing, facts, uncertain, consequence = [], [], [], [], []
    r = _qe3_read(ctx, "landmark", field, "main")
    lm = r["lm"]
    if lm is None:
        missing.append(f"landmark A − {g_name} in {metric}")
    else:
        facts.append(f"n = {lm['n']}; (a) {_cstr(f'A − {g_name}, landmark, in {metric}', lm)};")
        if not _ahead(lm):
            fails.append(f"(a) does not hold: A is not ahead of {g_name}.")
            consequence.append(ANALYST + (
                " claim 1 (A beats G) is restricted to wearers who judge by RMS distance."
                if field == "distances" else
                " the landmark wearer does not feel A's lead over G: claim 1 (A beats G) is "
                "restricted to wearers who judge by RMS distance."))
    if not r["present"]:
        missing.append(f"landmark and main cells of A and {g_name}"
                       + ("" if field == "distances" else " with the wearer's own distance"))
        return _verdict(3, title, fails, missing, facts, consequence,
                        n=lm["n"] if lm else None, head=head)
    adv, diff = r["adv"], r["diff"]
    if adv is None or diff is None:
        missing.append("users common to the landmark and main cells")
        return _verdict(3, title, fails, missing, facts, consequence,
                        n=lm["n"] if lm else None, head=head)
    facts.append(f"(b) D = advantage over {g_name} in landmark (in {metric}) − 2/3 · advantage "
                 f"in main (in the RMS error) = {_ci3(diff)} Nm (n = {diff['n']}), predicted ≤ 0; "
                 f"advantage over {g_name} in main {_ci3(adv)} Nm, required > 0 with an interval "
                 f"excluding 0.")
    state, why = _qe3_b_state(adv, diff)
    if state == "failed":
        fails.append(f"(b) does not hold: {why}.")
        if adv["hi"] > 0:
            consequence.append(ANALYST + " the home-advantage share is smaller than predicted.")
    elif state == "uncertain":
        uncertain.append(f"(b) is uncertain: {why}.")
    return _verdict(3, title, fails, missing, facts, consequence, uncertain,
                    n=min(lm["n"] if lm else diff["n"], diff["n"]), head=head)


def _qe3_lines(ctx: Ctx) -> list[str]:
    """QE3's planned verdict in the RMS error, then three sub-lines: the same criteria in
    the wearer's own distance, QE3(b) against the s = 0.5 reference, and the walking cell."""
    g_name = _label("promp_s0", "g")
    lines = [_qe3(ctx),
             _qe3(ctx, OWN, head=f"  - **QE3 in the wearer's own metric (landmark observer, "
                                 f"k = {K_MAIN}).** ")]
    cells = []
    for cid, label in (("landmark", "static landmark cell (10 %)"),
                       ("landmark25", "walking landmark cell (25 %)")):
        parts = []
        for field in ("distances", OWN):
            r = _qe3_read(ctx, cid, field, "s05")
            if r["adv"] is None or r["diff"] is None:
                parts.append(f"in {_metric_name(field)}: {NOT_RUN}")
                continue
            state, _ = _qe3_b_state(r["adv"], r["diff"])
            parts.append(f"in {_metric_name(field)}: D = {_ci3(r['diff'])} Nm "
                         f"(n = {r['diff']['n']}), {state} by the same rule, advantage over "
                         f"{g_name} at s = 0.5 {_ci3(r['adv'])} Nm")
        cells.append(f"{label}, " + "; ".join(parts))
    lines.append("  - **QE3(b) against the s = 0.5 reference.** The same D, with the advantage "
                 f"over {g_name} taken from the truncated s = 0.5 rows (5c) instead of main. "
                 "For the walking cell this is the analyst's reading, as below: "
                 + "; ".join(cells) + "; sensitivity alone moves A's advantage; not a pure "
                 "home-advantage share.")
    parts = []
    for field in ("distances", OWN):
        r = _qe3_read(ctx, "landmark25", field, "main")
        a = f"(a) {NOT_RUN}" if r["lm"] is None else f"(a) A − {g_name} {_ci3(r['lm']['diff'])}"
        if r["adv"] is None or r["diff"] is None:
            b = f"(b) {NOT_RUN}"
        else:
            b = f"(b) D = {_ci3(r['diff'])} Nm, {_qe3_b_state(r['adv'], r['diff'])[0]}"
        parts.append(f"in {_metric_name(field)}: {a}; {b}")
    lines.append("  - **Walking-calibrated landmark cell (25 %), analyst's reading, not a "
                 "registered verdict.** QE3's criteria applied to the cell of spec amendment "
                 "E4b (magnitude JND 25 % of the ideal's peak, timing JND 2.8 %GC; its "
                 "perturbation strength is in section 9): " + "; ".join(parts) + ".")
    lines.append(_qe3_decisiveness(ctx))
    return lines


def _qe4(ctx: Ctx) -> str:
    fails, missing, facts, uncertain, consequence = [], [], [], [], []
    cells = [ctx.df(c, "promp_s0", lr) for c, lr in (("noise", "a"), ("main", "a"),
                                                      ("noise", "g"), ("main", "g"))]
    did = None
    if all(_present(d) for d in cells):
        did = ctx.joint(cells, K_MAIN,
                        lambda an, am, gn, gm: (_med(an, 1) - _med(am, 1))
                        - (_med(gn, 1) - _med(gm, 1)), key=("qe4-a",))
    g_name = _label("promp_s0", "g")
    if did is None:
        missing.append(f"noise and main cells of A and {g_name}")
    else:
        facts.append(f"(a) rise in error under noise, A minus {g_name}: {_ci3(did)} Nm "
                     f"(n = {did['n']}), predicted > 0;")
        side = _side(did, 0.0, want_low=False)
        if side == "failed":
            fails.append(f"(a) does not hold: A's error rises less than {g_name}'s.")
        elif side is None:
            uncertain.append("(a) is uncertain: the interval straddles 0.")
    c = ctx.cell_contrast("noise", "promp_s0", "a", "g")
    if c is None:
        missing.append(f"noise A − {g_name}")
    else:
        facts.append(f"n = {c['n']}; (b) {_cstr(f'A − {g_name}, noise', c)}.")
        if not _ahead(c):
            fails.append(f"(b) does not hold: A is not ahead of {g_name} under noise.")
            consequence.append(ANALYST + " claim 1 restricted to noise-free presentations.")
    ns = [x["n"] for x in (did, c) if x is not None]
    return _verdict(4, "device noise, k = 24", fails, missing, facts, consequence, uncertain,
                    n=min(ns) if ns else None)


def _qe5(ctx: Ctx) -> str:
    fails, missing, facts, ns = [], [], [], []
    jnd = BY_ID["s2"].jnd
    for family in ("promp_s0", "franks"):
        for lr in SCOPES["full"][family]:
            df = ctx.df("s2", family, lr)
            name = f"{_label(family, lr)} ({FAMILY_NAME[family]})"
            if not _present(df):
                missing.append(name)
                continue
            ns.append(len(df))
            share = float(bm.levels_at(df, K_MAIN, jnd)["q1"].mean())
            facts.append(f"{name} {share:.2f} (n = {len(df)});")
            if share >= 0.5:
                fails.append(f"Q≥1 of {name} is {share:.2f}, not below 0.5.")
    if ns:
        facts = [f"Q≥1 at k = {K_MAIN} in {jnd:.3f} Nm:"] + facts
    consequence = [ANALYST + " at s = 2.0 most users still become perceptibly better off "
                   "within 24 comparisons: the limit-of-personalisation statement is dropped "
                   "for the optimizers listed."]
    return _verdict(5, "s = 2.0, k = 24", fails, missing, facts, consequence,
                    n=min(ns) if ns else None)


def _qe6(ctx: Ctx) -> str:
    fails, missing, facts, uncertain, consequence, ns = [], [], [], [], [], []
    for learner in TUNED_ORDER:
        name = _label("promp_s0", learner)
        _, gain, lead = _tuning(ctx, learner)
        if gain is None or lead is None:
            missing.append(f"{name} tuned and default on the held-out folds")
            continue
        ns.append(min(gain["n"], lead["n"]))
        vacuous = (" (its tuned setting is its e-main default, so (a) is 0 by construction)"
                   if _is_default(learner, _tuned_df(ctx, learner).iloc[0].to_dict()) else "")
        facts.append(f"{name}: (a) tuning gains {_ci3(gain)} Nm (n = {gain['n']}){vacuous}, "
                     f"predicted ≤ 0.2; (b) {_cstr('A − tuned', lead)};")
        side = _side(gain, 0.2, want_low=True)
        if side == "failed":
            fails.append(f"(a) does not hold for {name}: tuning gains more than 0.2 Nm.")
        elif side is None:
            uncertain.append(f"(a) is uncertain for {name}: the interval straddles 0.2 Nm.")
        if _behind(lead):
            fails.append(f"(b) does not hold for {name}: A is behind it.")
            consequence.append(f"{KILL} beaten by tuned {name}: claim 1 does not hold against "
                               f"tuned comparators.")
        elif not _ahead(lead):
            fails.append(f"(b) does not hold for {name}: A is not separable from it.")
            consequence.append(f"{KILL} matches tuned {name}: claim 1 is stated as \"matches "
                               f"tuned comparators\" rather than \"beats\".")
    return _verdict(6, "fair tuning, k = 24", fails, missing, facts, consequence, uncertain,
                    n=min(ns) if ns else None, expected=EXPECTED_N_TUNED)


def _qe_section(ctx: Ctx) -> list[str]:
    return ["## 11. Predictions QE1-QE6", "",
            "Registered 2026-10-07 before any 5e session ran. Each line gives its verdict, the "
            "values and, when a prediction does not come true, its consequence: the registered "
            "kill criterion for QE2 and QE6, the analyst's reading for QE1, QE3, QE4 and QE5. "
            "QE3(b), QE4(a) and QE6(a) compare changes between cells and are judged, by the "
            "amendment at the end of spec §4, on a joint cluster-bootstrap interval of the "
            "difference: users resampled once per draw within each fold, across every array, "
            "common users only. The verdict is wholly on one side of the threshold or "
            "\"uncertain\". A verdict on fewer users than a complete cell is marked "
            "provisional. By the amendment after the whole-branch review, QE3 is read in the "
            "RMS error (its registered verdict) and in the landmark wearer's own distance, and "
            "QE3(b) is also read against the s = 0.5 reference.", "",
            _qe1(ctx), _qe1_before(ctx), _qe2(ctx), *_qe3_lines(ctx), _qe4(ctx), _qe5(ctx), _qe6(ctx), ""]


def _methods_note() -> list[str]:
    return ["## 12. Reading notes", "",
            "The s = 0.5 reference rows come from 5c/5d (150 comparisons) truncated at k, which "
            "is exact (B3). Drift cells run 24 comparisons and are not truncated, so their "
            "excess is read at n = 24 against the same optimizer's e-main row. Quality levels "
            "are post hoc on the stored sessions; the intervals are cluster bootstraps over "
            "users within folds. Everything is conditional on the simulator and on the "
            "assumptions of each cell.", "",
            f"The estimand of every Δ median is the {ESTIMAND}: the first cell's median minus "
            "the second's, both on the users the two cells hold in common.", "",
            "Median conventions. The usual median (the mean of the two middle values for an "
            "even count) is printed in the B7 tables (sections 1, 4 and 5), the generic "
            "distance of section 4, the familiarity gap table, the drift tables' tracking error, "
            "the e-tune and e-oracle selection tables (which also choose by it, with a linear "
            "p75) and the gain over generic assistance (section 10). The lower middle value "
            "(`inverted_cdf`, as phase5c_stats takes it, S11) is used by the contrasts, the "
            "ranking, the drift tables' e-main error above floor and excess, the held-out and "
            "oracle-tuned medians and gains (sections 6 and 7), the own-metric section 8 and "
            "every verdict; a printed median can then differ from a contrast's by half a gap "
            "between neighbouring users. Every bootstrap interval of a Δ median, a gain, an "
            "excess or a verdict is an `inverted_cdf` percentile of its draws; the lower decile "
            "of gain in the B7 tables and its interval are linear percentiles "
            "(benchmark_report).", ""]


def build_report(dir5e, dir5c, dir5d, n_boot: int = 10_000, *, run_dir=None, touches=None,
                 perturbation_cache=None) -> str:
    """The report. ``run_dir`` defaults to the run directory the e-main rows record;
    ``touches(fold, user_id)`` (whether the ideal reaches the device limit) to one read from
    it; ``perturbation_cache`` to ``<dir5e>/perturbation_strength.json``."""
    ctx = Ctx(dir5e, dir5c, dir5d, n_boot, run_dir=run_dir, touches=touches,
              perturbation_cache=perturbation_cache)
    lines = _intro()
    for section in (_main_section, _contrast_section, _ranking_section, _s_sweep_section,
                    _robust_section, _tuning_section, _oracle_section, _own_section,
                    _perturbation_section, _gain_section, _qe_section):
        lines += section(ctx)
    lines += _methods_note()
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir5e", default="out/phase5e")
    ap.add_argument("--dir5c", default="out/phase5c")
    ap.add_argument("--dir5d", default="out/phase5d")
    ap.add_argument("--out", default="out/phase5e/REPORT.md")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--run", default=None,
                    help="the run directory the sessions ran on (default: as the rows record)")
    ap.add_argument("--perturbation-cache", default=None,
                    help="default <dir5e>/perturbation_strength.json")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(args.dir5e, args.dir5c, args.dir5d, n_boot=args.boot,
                                run_dir=args.run, perturbation_cache=args.perturbation_cache),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
