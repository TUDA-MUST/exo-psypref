"""ICORR second review round (R2): users from other datasets (XD), heavy tails (HT), the
pessimistic combined user (PESS) and the established-practice baseline (BASE); PXD-1, PXD-2 and
PR2-2 to PR2-6 of our design notes (PR2-1, G-s2, is reported elsewhere).

Usage:
    python scripts/icorr_r2_report.py [--out <main>/out] [--data <Schaefer logs>]
        [--simulated-only] [--boot 2000] [--boot-verdict 10000] [--report <path>] [--json <path>]

Reads ``out/icorr/e-xd.jsonl``, ``e-ht.jsonl``, ``e-pess.jsonl``, ``e-pess-gp.jsonl``,
``e-base-cma.jsonl``, ``e-base-gp.jsonl`` and ``base_tuned.json``, ``out/phase5e/e-main.jsonl``, the
R2 user sets and sidecars under ``out/phase5-schaefer``, and the Schaefer logs (aggregates only)
through ``icorr_jnd_free.room_setup`` for the median-matched room. Writes
``out/icorr/r2_report.md`` and ``out/icorr/r2_numbers.json``. No session is run here.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import icorr_jnd_free as jf  # noqa: E402
import icorr_k_report as kr  # noqa: E402
import phase5e_report as er  # noqa: E402
import schaefer_grounding as G  # noqa: E402

from prompref.experiment import icorr_grid  # noqa: E402
from prompref.experiment.phase5 import PESS_FIELDS, Phase5Point  # noqa: E402
from prompref.experiment.phase5c_grid import LOSO  # noqa: E402
from prompref.experiment.phase_weighted import load_phase_weight_sidecar  # noqa: E402
from prompref.experiment.robust import load_ood_users  # noqa: E402
from prompref.experiment.sweep import _row_key  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.oracle import link_cdf  # noqa: E402
from prompref.user.schedule import linear_path  # noqa: E402

K_MAIN = 24
KS = (12, 24, 40)
KMAX = 40
S_MAIN, LAM = 1.0, 0.1
S_PESS = float(PESS_FIELDS["s"])
N_BOOT_VERDICT = 10_000
FAVOUR = ("failed", "uncertain", "held")
COMPARATORS = ("g", "d_z", "cmaes")
NAME = {"a": "PsyPref", "g": "Arens' GP", "d_z": "LineCoSpar", "cmaes": "CMA-ES",
        "gp_bald": "GP-BALD"}
PXD2_LEVEL = 0.70
PR23_MARGIN = -0.10
#: The main condition's preference at k = 24 as stress_report.md and ceiling.md give it.
EXPECT_MAIN = {("simulated", "a"): 0.815, ("simulated", "g"): 0.618, ("simulated", "d_z"): 0.578,
               ("simulated", "cmaes"): 0.629, ("median", "a"): 0.721, ("median", "g"): 0.521,
               ("median", "d_z"): 0.480, ("median", "cmaes"): 0.552}

#: The cells: id -> (block file, learners, how its rows are recognised, users, yardstick).
CELLS = {
    "XD-Fy": dict(block="e-xd", learners=icorr_grid.R2_LEARNERS, fields=dict(user_set="xd-fy"),
                  users="xd-fy", yard="rms"),
    "XD-Fo": dict(block="e-xd", learners=icorr_grid.R2_LEARNERS, fields=dict(user_set="xd-fo"),
                  users="xd-fo", yard="rms"),
    "XD-R": dict(block="e-xd", learners=icorr_grid.R2_LEARNERS, fields=dict(user_set="xd-r"),
                 users="xd-r", yard="rms"),
    "HT-a": dict(block="e-ht", learners=icorr_grid.R2_LEARNERS, fields=dict(user_set="ht-gen"),
                 users="ht-gen", yard="rms"),
    "HT-b": dict(block="e-ht", learners=icorr_grid.R2_LEARNERS, fields=dict(user_set="ht-mix"),
                 users="ht-mix", yard="rms"),
    "HT-c": dict(block="e-ht", learners=icorr_grid.R2_LEARNERS,
                 fields=dict(link="cauchy", link_scale=icorr_grid.R2_LINK_SCALES["cauchy"]),
                 users=None, yard="cauchy"),
    "PESS": dict(block="e-pess", learners=icorr_grid.R2_LEARNERS, fields=dict(PESS_FIELDS),
                 users="ood-pb", yard="pess"),
}
#: GP-BALD in PESS, its own block.
PESS_GP = dict(block="e-pess-gp", learners=("gp_bald",), fields=dict(PESS_FIELDS),
               users="ood-pb", yard="pess")
BASE = {"cmaes": "e-base-cma", "gp_bald": "e-base-gp"}

RULINGS = (
    ("R0. Timing",
     "These rulings were written and committed with this script before any R2 session row existed "
     "(blocks e-xd, e-ht, e-pess, e-pess-gp, e-base-cma, e-base-gp). Seen before them: the build's "
     "prep summaries (out/icorr/r2_prep_*.json: the R2 users' floors, ideal sizes and rooms, the "
     "Cauchy link's matched scale), and short smoke sessions run on the real run directory and not "
     "stored (their error field, comparison count and seconds). Before the commit, the script ran "
     "end to end on stand-ins built from stored e-main rows only (each relabelled with an R2 cell's "
     "fields and users), whose numbers are e-main's. No outcome of any R2 cell."),
    ("R1. Users and completeness",
     "A cell is judged only if every learner holds exactly its registered users once, with 40 "
     "comparisons, no unresolved error row (an error row is allowed only beside a successful row of "
     "the same key, the sweep's retry), and every point field as registered (the cell's fields, the "
     "main condition otherwise: s = 1.0 (PESS 2.0), lambda 0.1, data prior, tight arm (BASE GP-BALD: "
     "the tuned arm), 8 bases, G's true-JND gate and 2,048 candidates, CMA-ES at pycma's default "
     "population (BASE: 4)). Users: XD-Fy, XD-Fo, XD-R, HT-a, HT-b and PESS their user set's 200 "
     "(RUN/<set>/users, PESS the 'ood-pb' users); HT-c and BASE the 200 population users of the five "
     "LOSO folds. Within a cell every learner's d_prior and d_floor are equal user by user (BASE: "
     "d_floor equal to the Schaefer (ii) floor sidecar); HT-c's and HT-b's population users' equal "
     "e-main A's bit for bit; PESS's own_generic_distance and utility_scale equal the PESS sidecar's "
     "own_generic and scale. A cell with no rows reads 'not run'; one that fails a check reads "
     "'incomplete' with the reason, and none of its outcomes is printed. A cell not complete by Wed "
     "28 Oct 2026, 20:00, is reported as not run (section 9a)."),
    ("R2. The preference (Y1)",
     "P_i(k) = lambda/2 + (1 - lambda) Phi((g_i - e_i(k)) / s), averaged over users, lambda = 0.1. XD, "
     "HT and BASE: the main condition's wearer, s = 1.0, g_i = d_prior (generic assistance: the fold's "
     "ProMP K = 8 prior-mean curve), e_i(k) = distances[k - 1] (RMS, Nm, as delivered). BASE: g_i is "
     "e-main A's d_prior of the same user (generic assistance is one curve for every family, "
     "proj_report R1); its rows' own d_prior (the Schaefer (ii) prior mean) is a labelled "
     "sensitivity. PESS changes what the wearer prefers, so it is read in the wearer's own metric on "
     "its matched scale, as S1: g_i = own_generic_distance, e_i(k) = own_distances[k - 1], with the "
     "PESS wearer's s = 2.0."),
    ("R3. The stressed wearer's yardstick (Y2), and the lower of the two",
     "HT-c and PESS answer differently from the main wearer, so each of their parts is also computed "
     "with the validation duel as that wearer would answer it after comparison k, and the part takes "
     "the lower verdict of Y1 and Y2 (failed < uncertain < held), as the stress report's R3. HT-c: "
     "lambda/2 + (1 - lambda) F_Cauchy(Delta / kappa), kappa the matched scale. PESS at k: lambda_k "
     "on the fatigue path (linear 0.05 -> 0.30 over the 40 comparisons; 0.197 at k = 24); 'no "
     "difference' (counted as half a win) when |Delta| < 0.5 true_jnd(2.0) = 0.674 (hard rule); "
     "otherwise the order bias averaged over the two orders, (1/2)[Phi(Delta/s + 0.3) + Phi(Delta/s - "
     "0.3)], s = 2.0; Delta = g_i - e_i(k) in the own metric. Both sides of a contrast are scored "
     "with the same yardstick."),
    ("R4. Statistics as simulated",
     "phase5e_report's fold-stratified joint cluster bootstrap, 10,000 draws, its seed; every array "
     "of a statistic indexed by the same draw of users (each cell's users are 40 per fold in LOSO "
     "order, so every cell is drawn identically); 95 % ends are observed bootstrap values "
     "(inverted_cdf). The best fast comparator is the largest of Arens' GP's, LineCoSpar's and "
     "CMA-ES's mean preference, taken within each draw (stress R5). PR2-3 contrasts two user sets "
     "(HT-a against e-main's population users); they are aligned by fold and position (HT-a user j "
     "of a fold carries the answer seed of population user j) and drawn jointly, a difference of "
     "two means."),
    ("R5. The median-matched room (K = 8 room weights)",
     "icorr_jnd_free.room_setup's KDE reweighting to the median-matched room (1.47 Nm), with 2,000 "
     "draws and its seed, applied to each cell's own users with room = d_prior - d_floor of their "
     "ProMP K = 8 rows (A's): e-main A's for the population cells (HT-c, BASE, and PsyPref of "
     "PR2-3 and PR2-6), the set's own for XD, HT-a, HT-b and PESS (PESS: the RMS room of the "
     "'ood-pb' users; its preference stays on its own yardstick). Weighted means; intervals the "
     "2.5 / 97.5 % percentiles over the draws; a contrast is formed draw by draw (two user sets are "
     "drawn alike, R4). The Schaefer logs must match median_room_data_check.json, and e-main A's "
     "Q>=1 at k = 24 must reproduce jnd_free's 0.59 at 1.47 Nm (to 0.01), or the room is not "
     "computed and every part that needs it reads 'not computed'."),
    ("R6. Verdicts",
     "A part is held only if its interval lies wholly and strictly on the predicted side of its "
     "threshold, failed if wholly and strictly on the other side, uncertain otherwise; a prediction "
     "takes its lowest part. Rooms: the registration's preamble ('unless stated otherwise ... as "
     "simulated and at the median room') decides. PXD-1 and PXD-2 are judged as simulated (XD's "
     "registration: the median room is context) and PR2-4 as simulated (stated). PR2-2, PR2-3 and "
     "PR2-5 state no room, so each part is judged as simulated and at the median room, both "
     "required; PR2-6 states both."),
    ("R7. The predictions",
     "PXD-1: per XD cell, PsyPref minus the best fast comparator at k = 24 (> 0). PXD-2: per XD "
     "cell, PsyPref's mean preference at k = 24 (lower end > 0.70). PXD-3: not run (the author's "
     "ruling of 2026-10-22, before any XD session). PR2-2: per HT cell (a), (b), (c), PsyPref minus "
     "the best fast comparator (> 0; (c) the lower of Y1 and Y2). PR2-3: HT-a's PsyPref minus "
     "e-main's PsyPref, difference of means (lower end > -0.10). PR2-4: PESS PsyPref's mean "
     "preference (lower end > 0.5). PR2-5: PESS PsyPref minus the best fast comparator (> 0). PR2-6: "
     "PsyPref on the ProMP (e-main A) minus BASE's CMA-ES and minus BASE's GP-BALD (each > 0), as "
     "simulated and at the median room: four parts. A part whose cell is not run reads 'not run'; a "
     "prediction with such a part is judged on the parts that ran, labelled partial: a failed part "
     "fails it, otherwise it cannot be held as registered and the parts' verdict is given."),
    ("R8. Floors and ceilings (context)",
     "Per cell, the floor (d_floor, the device-clipped ProMP K = 8 curve closest to the ideal; BASE: "
     "Schaefer (ii)'s; PESS: the own-metric floor of the sidecar, on its matched scale) as the median "
     "over users, and the attainable ceiling C_i = the preference (R2's yardstick) of that floor "
     "curve, its mean over users, as simulated (R4) and at the median room (R5); PsyPref's share of "
     "the ceiling (P - 0.5) / (C - 0.5) as simulated."),
    ("R9. BASE's tuning",
     "Each BASE method's three settings ran on loso-AB30's 40 population users first; "
     "scripts/icorr_r2_prep.py --step base-select took the lowest median error at k = 24 "
     "(section E8's rule) into out/icorr/base_tuned.json, and that setting ran on the other folds. "
     "PR2-6 uses all 200 users (the tuning fold in-sample, which can only favour BASE); the 160 users "
     "of the other folds are reported beside it, no verdict. Every setting's tuning-fold result is "
     "listed."),
    ("R10. Context, no verdicts",
     "Per cell and learner: the mean preference at k = 12, 24 and 40 (Y1), Y2 at 24 where it exists, "
     "and the median error at k = 24 (Nm; own metric for PESS), as simulated and at the median room; "
     "GP-BALD in PESS if it ran; BASE's time per comparison. XD is reported as 'users from two "
     "independent datasets', not as a population claim (the author's ruling of 2026-10-22)."),
    ("R11. A failed prediction is reported as found", "Section 9a."),
)


# ============================================================================ utilities
def stamp() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lowest(states) -> str:
    states = [s for s in states if s in FAVOUR]
    return min(states, key=FAVOUR.index) if states else "not run"


def state(ci: dict, threshold: float) -> str:
    return kr.state(ci, threshold, want_low=False)


def ci3(d: dict, digits: int = 3) -> str:
    f = f"{{:+.{digits}f}}"
    return f"{f.format(d['estimate'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


def p3(d: dict, digits: int = 3) -> str:
    f = f"{{:.{digits}f}}"
    return f"{f.format(d['estimate'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]"


def no_conflicts(path: Path) -> list[str]:
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    return sorted(p.name for p in path.parent.glob(f"{stem}*conflicted*"))


# ============================================================================ rows
FIELDS = tuple(Phase5Point.__dataclass_fields__)
SCALARS = ("d_prior", "d_floor", "own_generic_distance", "utility_scale", "session_s",
           "n_fit_failed", "gp_fit_s", "gp_acq_s")
LISTS = ("distances", "own_distances", "answers")


def read_block(path: Path) -> dict:
    """Every row of one block file: successful rows by key, with the point's fields, the scalars and
    the lists; counts of lines, error rows and unresolved errors; pCloud conflict copies."""
    out = dict(path=str(path), exists=path.exists(), rows={}, lines=0, error_keys=set(),
               conflicts=no_conflicts(path) if path.parent.exists() else [], bad=0)
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            out["lines"] += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                out["bad"] += 1
                continue
            key = _row_key(r, Phase5Point)
            if r.get("error"):
                out["error_keys"].add(key)
                continue
            slim = {f: r.get(f) for f in FIELDS}
            slim.update({f: r.get(f) for f in SCALARS})
            for name in LISTS:
                if isinstance(r.get(name), list):
                    slim[name] = [float(x) for x in r[name][:KMAX]]
            out["rows"][key] = slim
    out["unresolved"] = len(out["error_keys"] - set(out["rows"]))
    return out


def population_keys(run: Path) -> list[tuple[str, str]]:
    return list(G.load_sim(run)["keys"])


def set_keys(run: Path, user_set: str) -> list[tuple[str, str]]:
    return [(fold, uid) for fold in LOSO for uid in load_ood_users(str(run), fold, user_set)]


def registered_points(block: str, run_dir: str) -> list[Phase5Point]:
    """The block's planned points (R1): exactly what the grid builds, every field included."""
    return icorr_grid.block_points(block, run_dir)


def _has(point: Phase5Point, fields: dict) -> bool:
    for name, value in fields.items():
        have = getattr(point, name)
        if isinstance(value, float) and not isinstance(have, (bool, str)) and have is not None:
            if float(have) != value:
                return False
        elif have != value:
            return False
    return True


def collect(block: dict, points: list[Phase5Point], cell_fields: dict, learner: str,
            keys) -> tuple[dict | None, str]:
    """One (cell, learner): the rows of the planned points with ``cell_fields`` and ``learner``,
    looked up by resume key, in the order of ``keys``; or
    (None, reason)."""
    wanted = set(keys)
    mine = [p for p in points if p.learner == learner and _has(p, cell_fields)
            and (p.fold, p.user_id) in wanted]
    by_user = {(p.fold, p.user_id): p for p in mine}
    if len(by_user) != len(mine):
        return None, "the grid names a user twice"
    if sorted(by_user) != sorted(keys):
        return None, f"the grid holds {len(by_user)} users, the registration {len(keys)}"
    rows, missing = [], 0
    for k in keys:
        row = block["rows"].get(by_user[k].key())
        if row is None:
            missing += 1
        rows.append(row)
    if missing == len(keys):
        return None, "not run"
    if missing:
        return None, f"incomplete: {len(keys) - missing} of {len(keys)} users"
    for r in rows:
        if len(r.get("distances") or []) < KMAX:
            return None, "a row holds fewer than 40 distances"
    return dict(rows=rows, keys=list(keys)), "ok"


def with_run_dir(points: list[Phase5Point], block: dict) -> list[Phase5Point]:
    """The points with the run directory string the block's rows carry (part of the key)."""
    from dataclasses import replace

    dirs = {r["run_dir"] for r in block["rows"].values()}
    if len(dirs) > 1:
        raise SystemExit(f"{block['path']}: rows from {len(dirs)} run directories")
    if not dirs:
        return points
    run_dir = dirs.pop()
    return [replace(p, run_dir=run_dir) for p in points]


def arr(cell: dict, name: str) -> np.ndarray:
    return np.array([float(r[name]) for r in cell["rows"]], dtype=float)


def mat(cell: dict, name: str) -> np.ndarray:
    return np.array([r[name][:KMAX] for r in cell["rows"]], dtype=float)


# ============================================================================ yardsticks
def y1(yard: str, g, e) -> np.ndarray:
    s = S_PESS if yard == "pess" else S_MAIN
    return jf.win_probability(g, e, s, LAM)


def y2(yard: str, g, e, k: int) -> np.ndarray:
    delta = np.asarray(g, dtype=float) - np.asarray(e, dtype=float)
    if yard == "cauchy":
        kappa = icorr_grid.R2_LINK_SCALES["cauchy"]
        return LAM / 2.0 + (1.0 - LAM) * link_cdf("cauchy", delta / kappa)
    if yard == "pess":
        lam = float(linear_path(PESS_FIELDS["lapse_start"], PESS_FIELDS["lapse_end"], KMAX)[k - 1])
        b = float(PESS_FIELDS["order_bias"])
        c = float(PESS_FIELDS["no_pref_jnd"]) * true_jnd(S_PESS)
        x = delta / S_PESS
        ans = np.where(np.abs(delta) < c, 0.5, 0.5 * (norm.cdf(x + b) + norm.cdf(x - b)))
        return lam / 2.0 + (1.0 - lam) * ans
    raise ValueError(yard)


def yardsticks(yard: str) -> tuple[str, ...]:
    return ("Y1",) if yard == "rms" else ("Y1", "Y2")


def generic_error(yard: str, cell: dict, generic=None) -> tuple[np.ndarray, np.ndarray]:
    if yard == "pess":
        return arr(cell, "own_generic_distance"), mat(cell, "own_distances")
    g = arr(cell, "d_prior") if generic is None else np.asarray(generic, dtype=float)
    return g, mat(cell, "distances")


def pref(yard: str, cell: dict, k: int, y: str = "Y1", generic=None) -> np.ndarray:
    g, e = generic_error(yard, cell, generic)
    return y1(yard, g, e[:, k - 1]) if y == "Y1" else y2(yard, g, e[:, k - 1], k)


def best_lead(a, *others):
    return a.mean(axis=1) - np.max(np.stack([o.mean(axis=1) for o in others]), axis=0)


def mean(p):
    return p.mean(axis=1)


def diff_mean(x, y):
    return x.mean(axis=1) - y.mean(axis=1)


# ============================================================================ rooms
class RoomSet:
    """jnd_free.room_setup per user set (R5), computed lazily and cached by the room array."""

    def __init__(self, out: Path, data: Path | None, n_boot: int, folds):
        self.out, self.data, self.n_boot, self.folds = out, data, n_boot, np.asarray(folds)
        self.cache, self.levels = {}, None

    def get(self, name: str, room: np.ndarray):
        if self.data is None:
            return None
        if name not in self.cache:
            setup = jf.room_setup(self.out, self.data, self.n_boot, np.asarray(room, dtype=float),
                                  self.folds)
            self.levels = setup["level"]
            self.cache[name] = setup["rooms"]
        return self.cache[name]

    def stat(self, name: str, room, p) -> dict | None:
        rooms = self.get(name, room)
        if rooms is None or "median" not in rooms.point:
            return None
        est, b = rooms.stat("win", p, "median")
        lo, hi = jf.ci(b)
        return dict(estimate=float(est), lo=lo, hi=hi, boot=b)

    def stat_q(self, name: str, room, x, q: float = 0.5) -> dict | None:
        rooms = self.get(name, room)
        if rooms is None or "median" not in rooms.point:
            return None
        est, b = rooms.stat("err", x, "median", q)
        lo, hi = jf.ci(b)
        return dict(estimate=float(est), lo=lo, hi=hi, boot=b)


def room_diff(a: dict | None, b: dict | None) -> dict | None:
    if a is None or b is None:
        return None
    d = a["boot"] - b["boot"]
    lo, hi = jf.ci(d)
    return dict(estimate=a["estimate"] - b["estimate"], lo=lo, hi=hi, boot=d)


def room_best_lead(a: dict | None, others: list) -> dict | None:
    if a is None or any(o is None for o in others):
        return None
    est = a["estimate"] - max(o["estimate"] for o in others)
    d = a["boot"] - np.max(np.stack([o["boot"] for o in others]), axis=0)
    lo, hi = jf.ci(d)
    return dict(estimate=est, lo=lo, hi=hi, boot=d)


def strip(d):
    if d is None:
        return None
    return {k: v for k, v in d.items() if k != "boot"}


# ============================================================================ loading and checks
def load_all(out: Path, run: Path, pop_keys) -> dict:
    """Every R2 cell (R1) and e-main's four learners, each a dict of learner -> rows in key order,
    with the cell's state; and the checks made."""
    from prompref.experiment import phase5e_grid
    from prompref.experiment.phase5 import proj_floors

    run_str = str(run)
    icorr = out / "icorr"
    blocks = {name: read_block(icorr / f"{name}.jsonl") for name in icorr_grid.R2_BLOCKS}
    res = dict(cells={}, state={}, keys={}, checks=[], blocks={}, tuning={})
    for name, b in blocks.items():
        res["blocks"][name] = dict(exists=b["exists"], lines=b["lines"], ok=len(b["rows"]),
                                   unresolved=b.get("unresolved", 0), bad=b["bad"],
                                   conflicts=b["conflicts"])
        if b["conflicts"]:
            raise SystemExit(f"{name}.jsonl: pCloud conflict copies beside it: {b['conflicts']}")
        if b["bad"]:
            raise SystemExit(f"{name}.jsonl: {b['bad']} lines do not parse")
    # e-main, the main condition's four learners on the 200 population users
    mb = read_block(out / "phase5e" / "e-main.jsonl")
    mpts = with_run_dir(phase5e_grid.block_points("e-main", run_str), mb)
    main = {}
    for learner in icorr_grid.R2_LEARNERS:
        c, why = collect(mb, mpts, dict(family="promp_s0"), learner, pop_keys)
        if c is None:
            raise SystemExit(f"e-main {learner}: {why}")
        main[learner] = c
    res["main"] = main
    res["pop_keys"] = list(pop_keys)
    gen_main = arr(main["a"], "d_prior")
    floor_main = arr(main["a"], "d_floor")

    specs = dict(CELLS)
    specs["PESS-GP"] = PESS_GP
    for cid, spec in specs.items():
        b = blocks[spec["block"]]
        if b.get("unresolved"):
            res["state"][cid] = f"incomplete ({b['unresolved']} unresolved error rows in the block)"
            continue
        try:
            keys = pop_keys if spec["users"] is None else set_keys(run, spec["users"])
            pts = with_run_dir(registered_points(spec["block"], run_str), b)
        except (FileNotFoundError, ValueError) as exc:
            res["state"][cid] = f"not run ({type(exc).__name__}: {str(exc)[:80]})"
            continue
        res["keys"][cid] = keys
        cell, states = {}, {}
        for learner in spec["learners"]:
            c, why = collect(b, pts, spec["fields"], learner, keys)
            states[learner] = why
            if c is not None:
                cell[learner] = c
        bad = {lr: w for lr, w in states.items() if w != "ok"}
        if bad:
            if len(bad) == len(spec["learners"]) and all(w == "not run" for w in bad.values()):
                res["state"][cid] = "not run"
            else:
                res["state"][cid] = "incomplete: " + "; ".join(f"{NAME[lr]} {w}"
                                                               for lr, w in bad.items())
            continue
        # R1: d_prior and d_floor equal across learners, user by user
        ref = cell[spec["learners"][0]]
        for learner in spec["learners"][1:]:
            for name in ("d_prior", "d_floor"):
                if not np.array_equal(arr(cell[learner], name), arr(ref, name)):
                    res["state"][cid] = (f"incomplete ({NAME[learner]}'s {name} differs from "
                                         f"{NAME[spec['learners'][0]]}'s)")
        if cid == "PESS-GP" and "PESS" in res["cells"]:
            for name in ("d_prior", "d_floor", "own_generic_distance"):
                if not np.array_equal(arr(cell["gp_bald"], name),
                                      arr(res["cells"]["PESS"]["a"], name)):
                    res["state"][cid] = f"incomplete ({name} differs from the PESS cell's)"
        if cid in res["state"]:
            continue
        if cid == "HT-c":
            for name, want in (("d_prior", gen_main), ("d_floor", floor_main)):
                if not np.array_equal(arr(ref, name), want):
                    res["state"][cid] = f"incomplete ({name} differs from e-main A's)"
            res["checks"].append("HT-c: every learner's d_prior and d_floor equal e-main A's bit "
                                 "for bit, user by user.")
        if cid == "HT-b":
            idx = {k: i for i, k in enumerate(pop_keys)}
            pops = [(j, k) for j, k in enumerate(keys) if "/pop" in k[1]]
            same = all(float(ref["rows"][j]["d_prior"]) == gen_main[idx[k]]
                       and float(ref["rows"][j]["d_floor"]) == floor_main[idx[k]] for j, k in pops)
            if not same:
                res["state"][cid] = ("incomplete (a population user's d_prior or d_floor differs "
                                     "from e-main A's)")
            ident = sum(cell[lr]["rows"][j]["distances"] == main[lr]["rows"][idx[k]]["distances"]
                        for lr in spec["learners"] for j, k in pops)
            res["checks"].append(f"HT-b: d_prior and d_floor of its {len(pops)} population users "
                                 f"equal e-main A's; {ident} of {len(pops) * len(spec['learners'])}"
                                 " of their sessions reproduce e-main's 40 distances exactly.")
        if spec["yard"] == "pess":
            side = {}
            for fold in LOSO:
                sc = load_phase_weight_sidecar(run_str, "pref-sd1", fold, "ood-pb")
                for uid, e in sc["users"].items():
                    side[(fold, uid)] = e
            for learner in spec["learners"]:
                for k, r in zip(keys, cell[learner]["rows"]):
                    e = side[k]
                    if float(r["own_generic_distance"]) != float(e["own_generic"]) or \
                            float(r["utility_scale"]) != float(e["scale"]):
                        res["state"][cid] = ("incomplete (own generic distance or scale differs "
                                             "from the sidecar)")
            res["pess_floor"] = np.array([float(side[k]["own_floor"]) for k in keys])
            res["checks"].append(f"{cid}: every row's own_generic_distance and utility_scale equal "
                                 "the PESS sidecar (ood-pb/phase_weight/pref-sd1) bit for bit.")
        if cid not in res["state"]:
            res["state"][cid] = "ok"
            res["cells"][cid] = cell
    # BASE: the tuning fold's settings, then the chosen setting on all 200 users
    tuned = icorr_grid.load_base_tuned(run_str)
    res["base_tuned"] = tuned
    tune_keys = [k for k in pop_keys if k[0] == icorr_grid.BASE_TUNE_FOLD]
    for learner, bname in BASE.items():
        cid = f"BASE-{learner}"
        b = blocks[bname]
        pts = with_run_dir(registered_points(bname, run_str), b)
        tun = []
        for setting in icorr_grid.BASE_SETTINGS[learner]:
            fields = dict(setting, family="schaefer_peaks")
            c, why = collect(b, [p for p in pts if p.fold == icorr_grid.BASE_TUNE_FOLD], fields,
                             learner, tune_keys)
            tun.append(dict(setting=setting, state=why,
                            median_err=None if c is None else
                            float(np.median(mat(c, "distances")[:, K_MAIN - 1])),
                            chosen=bool(learner in tuned and tuned[learner]["setting"] == setting)))
        res["tuning"][learner] = tun
        if b.get("unresolved"):
            res["state"][cid] = f"incomplete ({b['unresolved']} unresolved error rows in the block)"
            continue
        if learner not in tuned:
            res["state"][cid] = "not run" if not b["rows"] else "incomplete (no setting chosen yet)"
            continue
        fields = dict(tuned[learner]["setting"], family="schaefer_peaks")
        c, why = collect(b, pts, fields, learner, pop_keys)
        if c is None:
            res["state"][cid] = why
            continue
        floors = np.array([float(proj_floors(run_str, f, "schaefer_peaks")[u]["floor_nm"])
                           for f, u in pop_keys])
        if not np.array_equal(arr(c, "d_floor"), floors):
            res["state"][cid] = "incomplete (d_floor differs from the Schaefer (ii) floor sidecar)"
            continue
        res["checks"].append(f"BASE {NAME[learner]}: d_floor equals the Schaefer (ii) floor "
                             "sidecar (users/<fold>-schaefer_peaks.json) bit for bit.")
        res["state"][cid] = "ok"
        res["cells"][cid] = {learner: c}
    return res


# ============================================================================ statistics
def cell_yard(cid: str) -> str:
    if cid.startswith("BASE"):
        return "rms"
    if cid == "PESS-GP":
        return "pess"
    return CELLS[cid]["yard"]


def cell_pref(res: dict, cid: str, learner: str, k: int, y: str = "Y1") -> np.ndarray:
    """Per-user preference of one learner of a cell (R2/R3); BASE against e-main's generic."""
    c = res["cells"][cid][learner]
    generic = arr(res["main"]["a"], "d_prior") if cid.startswith("BASE") else None
    return pref(cell_yard(cid), c, k, y, generic)


def cell_room(res: dict, cid: str) -> np.ndarray:
    """R5: the K = 8 room of the cell's users, from PsyPref's rows (or e-main A's)."""
    if cid.startswith("BASE") or cid == "HT-c" or cid == "main":
        a = res["main"]["a"]
    elif cid == "PESS-GP":
        a = res["cells"]["PESS"]["a"]
    else:
        a = res["cells"][cid]["a"]
    return arr(a, "d_prior") - arr(a, "d_floor")


def room_name(cid: str) -> str:
    """Cells sharing users share their room weights."""
    if cid.startswith("BASE") or cid in ("HT-c", "main"):
        return "population"
    if cid == "PESS-GP":
        return "PESS"
    return cid


def verdicts(res: dict, boot, rs: RoomSet | None) -> dict:
    v = {}
    ok = lambda cid: res["state"].get(cid) == "ok"  # noqa: E731

    def sim_best(cid, y):
        arrays = [cell_pref(res, cid, "a", K_MAIN, y)] + [cell_pref(res, cid, o, K_MAIN, y)
                                                           for o in COMPARATORS]
        ci = boot(arrays, best_lead)
        return dict(ci=ci, state=state(ci, 0.0))

    def med_best(cid, y):
        if rs is None:
            return dict(ci=None, state="not computed")
        room = cell_room(res, cid)
        a = rs.stat(room_name(cid), room, cell_pref(res, cid, "a", K_MAIN, y))
        others = [rs.stat(room_name(cid), room, cell_pref(res, cid, o, K_MAIN, y))
                  for o in COMPARATORS]
        ci = room_best_lead(a, others)
        return dict(ci=strip(ci), state=state(ci, 0.0) if ci else "not computed")

    # PXD-1 and PXD-2: as simulated
    v["PXD-1"], v["PXD-2"] = {"parts": {}}, {"parts": {}}
    for cid in ("XD-Fy", "XD-Fo", "XD-R"):
        if not ok(cid):
            v["PXD-1"]["parts"][cid] = dict(state="not run", note=res["state"].get(cid))
            v["PXD-2"]["parts"][cid] = dict(state="not run", note=res["state"].get(cid))
            continue
        v["PXD-1"]["parts"][cid] = sim_best(cid, "Y1")
        ci = boot([cell_pref(res, cid, "a", K_MAIN)], mean)
        v["PXD-2"]["parts"][cid] = dict(ci=ci, state=state(ci, PXD2_LEVEL))
    v["PXD-3"] = dict(parts={}, state="not run",
                      note="GP-BALD in XD-Fo is not run (the author's ruling of 2026-10-22)")
    # PR2-2: per HT cell, both yardsticks, both rooms
    v["PR2-2"] = {"parts": {}}
    for cid in ("HT-a", "HT-b", "HT-c"):
        if not ok(cid):
            v["PR2-2"]["parts"][cid] = dict(state="not run", note=res["state"].get(cid))
            continue
        sub = {}
        for y in yardsticks(cell_yard(cid)):
            sub[(y, "simulated")] = sim_best(cid, y)
            sub[(y, "median")] = med_best(cid, y)
        v["PR2-2"]["parts"][cid] = dict(sub=sub, state=lowest_or_nc(sub))
    # PR2-3: HT-a minus the main condition, both rooms
    v["PR2-3"] = {"parts": {}}
    if ok("HT-a"):
        pa, pm = cell_pref(res, "HT-a", "a", K_MAIN), pref("rms", res["main"]["a"], K_MAIN)
        ci = boot([pa, pm], diff_mean)
        sub = {("Y1", "simulated"): dict(ci=ci, state=state(ci, PR23_MARGIN))}
        if rs is not None:
            d = room_diff(rs.stat("HT-a", cell_room(res, "HT-a"), pa),
                          rs.stat("population", cell_room(res, "main"), pm))
            sub[("Y1", "median")] = dict(ci=strip(d), state=state(d, PR23_MARGIN))
        else:
            sub[("Y1", "median")] = dict(ci=None, state="not computed")
        v["PR2-3"]["parts"]["HT-a"] = dict(sub=sub, state=lowest_or_nc(sub))
    else:
        v["PR2-3"]["parts"]["HT-a"] = dict(state="not run", note=res["state"].get("HT-a"))
    # PR2-4 (as simulated) and PR2-5 (both rooms), both yardsticks
    v["PR2-4"], v["PR2-5"] = {"parts": {}}, {"parts": {}}
    if ok("PESS"):
        sub4, sub5 = {}, {}
        for y in ("Y1", "Y2"):
            ci = boot([cell_pref(res, "PESS", "a", K_MAIN, y)], mean)
            sub4[(y, "simulated")] = dict(ci=ci, state=state(ci, 0.5))
            sub5[(y, "simulated")] = sim_best("PESS", y)
            sub5[(y, "median")] = med_best("PESS", y)
        v["PR2-4"]["parts"]["PESS"] = dict(sub=sub4, state=lowest_or_nc(sub4))
        v["PR2-5"]["parts"]["PESS"] = dict(sub=sub5, state=lowest_or_nc(sub5))
    else:
        for name in ("PR2-4", "PR2-5"):
            v[name]["parts"]["PESS"] = dict(state="not run", note=res["state"].get("PESS"))
    # PR2-6: PsyPref on the ProMP minus each BASE method, both rooms
    v["PR2-6"] = {"parts": {}}
    pa = pref("rms", res["main"]["a"], K_MAIN)
    for learner in BASE:
        cid = f"BASE-{learner}"
        if not ok(cid):
            v["PR2-6"]["parts"][cid] = dict(state="not run", note=res["state"].get(cid))
            continue
        pb = cell_pref(res, cid, learner, K_MAIN)
        ci = boot([pa, pb], diff_mean)
        sub = {("Y1", "simulated"): dict(ci=ci, state=state(ci, 0.0))}
        if rs is not None:
            room = cell_room(res, "main")
            d = room_diff(rs.stat("population", room, pa), rs.stat("population", room, pb))
            sub[("Y1", "median")] = dict(ci=strip(d), state=state(d, 0.0))
        else:
            sub[("Y1", "median")] = dict(ci=None, state="not computed")
        v["PR2-6"]["parts"][cid] = dict(sub=sub, state=lowest_or_nc(sub))
    for name in ("PXD-1", "PXD-2", "PR2-2", "PR2-3", "PR2-4", "PR2-5", "PR2-6"):
        v[name]["state"] = overall([p["state"] for p in v[name]["parts"].values()])
    return v


def lowest_or_nc(sub: dict) -> str:
    states = [p["state"] for p in sub.values()]
    if "not computed" in states:
        judged = [s for s in states if s in FAVOUR]
        low = lowest(judged)
        return "failed" if low == "failed" else "not computed"
    return lowest(states)


def overall(states: list[str]) -> str:
    """R7: the lowest part; with a part not run or not computed, the parts' verdict, partial."""
    judged = [s for s in states if s in FAVOUR]
    missing = [s for s in states if s not in FAVOUR]
    if not judged:
        return "not run" if all(s == "not run" for s in missing) else "not computed"
    low = lowest(judged)
    if not missing:
        return low
    if low == "failed":
        return "failed (partial)"
    return f"{low} on the parts that ran (partial)"


def context(res: dict, boot, rs: RoomSet | None) -> dict:
    """R8, R10: per cell and learner, preference and error, floors and ceilings, both rooms."""
    ctx = {}
    cids = [c for c in list(CELLS) + ["PESS-GP", "BASE-cmaes", "BASE-gp_bald"]
            if res["state"].get(c) == "ok"]
    cids = ["main"] + cids
    for cid in cids:
        cell = res["main"] if cid == "main" else res["cells"][cid]
        yard = "rms" if cid == "main" else cell_yard(cid)
        r = {"pref": {}, "err": {}, "room": {}}
        for learner, c in cell.items():
            generic = arr(res["main"]["a"], "d_prior") if cid.startswith("BASE") else None
            for k in KS:
                r["pref"][(learner, "Y1", k)] = boot([pref(yard, c, k, "Y1", generic)], mean)
            if yard != "rms":
                r["pref"][(learner, "Y2", K_MAIN)] = boot([pref(yard, c, K_MAIN, "Y2", generic)],
                                                          mean)
            e = mat(c, "own_distances" if yard == "pess" else "distances")[:, K_MAIN - 1]
            r["err"][learner] = boot([e], lambda x: er._med(x, axis=1))
            if rs is not None:
                room = cell_room(res, cid)
                r["room"][(learner, "pref")] = strip(rs.stat(room_name(cid), room,
                                                             pref(yard, c, K_MAIN, "Y1", generic)))
                r["room"][(learner, "err")] = strip(rs.stat_q(room_name(cid), room, e))
        # floors and ceilings (R8), on the cell's yardstick
        first = next(iter(cell.values()))
        if yard == "pess":
            g, f = arr(first, "own_generic_distance"), res["pess_floor"]
        elif cid.startswith("BASE"):
            g, f = arr(res["main"]["a"], "d_prior"), arr(first, "d_floor")
        else:
            g, f = arr(first, "d_prior"), arr(first, "d_floor")
        r["floor"] = boot([f], lambda x: er._med(x, axis=1))
        r["generic"] = boot([g], lambda x: er._med(x, axis=1))
        r["room_nm"] = boot([g - f], lambda x: er._med(x, axis=1))
        r["ceiling"] = boot([y1(yard, g, f)], mean)
        if yard != "rms":
            r["ceiling_y2"] = boot([y2(yard, g, f, K_MAIN)], mean)
        if "a" in cell:
            r["share"] = boot([pref(yard, cell["a"], K_MAIN, "Y1"), y1(yard, g, f)],
                              lambda p, cc: (p.mean(axis=1) - 0.5) / (cc.mean(axis=1) - 0.5))
        if rs is not None:
            room = cell_room(res, cid)
            r["room"]["floor"] = strip(rs.stat_q(room_name(cid), room, f))
            r["room"]["ceiling"] = strip(rs.stat(room_name(cid), room, y1(yard, g, f)))
        if cid.startswith("BASE"):
            learner = next(iter(cell))
            own = pref("rms", cell[learner], K_MAIN, "Y1")  # its own family's prior mean
            r["own_generic_pref"] = boot([own], mean)
            held_out = np.array([k[0] != icorr_grid.BASE_TUNE_FOLD for k in res["pop_keys"]])
            pa = pref("rms", res["main"]["a"], K_MAIN)
            pb = pref("rms", cell[learner], K_MAIN, "Y1", arr(res["main"]["a"], "d_prior"))
            r["held_out"] = dict(n=int(held_out.sum()),
                                 a=float(pa[held_out].mean()), base=float(pb[held_out].mean()),
                                 diff=float(pa[held_out].mean() - pb[held_out].mean()))
            per_cmp = [float(r_["session_s"]) / KMAX for r_ in cell[learner]["rows"]]
            r["s_per_comparison"] = [float(np.quantile(per_cmp, q)) for q in (0.1, 0.5, 0.9)]
        fast = [o for o in COMPARATORS if o in cell]
        if fast:
            r["best_fast"] = max(fast, key=lambda o: r["pref"][(o, "Y1", K_MAIN)]["estimate"])
        if cid == "PESS-GP" and res["state"].get("PESS") == "ok":
            for y in ("Y1", "Y2"):
                r[f"a_minus_gp_{y}"] = boot([pref("pess", res["cells"]["PESS"]["a"], K_MAIN, y),
                                             pref("pess", cell["gp_bald"], K_MAIN, y)], diff_mean)
        if yard == "pess":
            r["no_difference_share"] = {
                learner: float(np.mean([[a == 2 for a in (row.get("answers") or [])[:K_MAIN]]
                                        for row in c["rows"]]))
                for learner, c in cell.items()}
        ctx[cid] = r
    return ctx


# ============================================================================ the report
PREDICTIONS = (
    ("PXD-1", "In each XD cell, PsyPref's preference at k = 24 exceeds the best fast comparator's "
              "(interval excluding 0); as simulated."),
    ("PXD-2", "In each XD cell, PsyPref's preference at k = 24 is ≥ 0.70 as simulated (interval's "
              "lower end above 0.70)."),
    ("PXD-3", "In XD-Fo, PsyPref − GP-BALD > 0, if GP-BALD runs."),
    ("PR2-2", "In each HT cell, PsyPref leads the best fast comparator (interval excluding 0); as "
              "simulated and at the median room (R6)."),
    ("PR2-3", "In HT (a), PsyPref is at most 0.10 below its main-condition preference; as simulated "
              "and at the median room (R6)."),
    ("PR2-4", "PESS: PsyPref's preference exceeds 0.5 as simulated (interval above 0.5)."),
    ("PR2-5", "PESS: PsyPref leads the best fast comparator (interval excluding 0); as simulated and "
              "at the median room (R6)."),
    ("PR2-6", "BASE: PsyPref on the ProMP exceeds GP-BALD and CMA-ES on Schäfer (ii) in preference, "
              "as simulated and at the median room (each interval excluding 0)."),
)
CELL_TEXT = {
    "main": "Main condition (e-main), the 200 population users",
    "XD-Fy": "XD-Fy: Fukuchi et al. 2018, young adults, treadmill at comfortable speed",
    "XD-Fo": "XD-Fo: Fukuchi et al. 2018, older adults (50–84 years), treadmill at comfortable speed",
    "XD-R": "XD-R: Reznick et al. 2021, 1.0 m/s level and ±5°, ±10° inclines, equal shares",
    "HT-a": "HT (a): the generator with every Student-t factor at ν = 1.5",
    "HT-b": "HT (b): 80 % population users, 20 % Schäfer (ii) and Franks ideals (published ranges)",
    "HT-c": "HT (c): Cauchy answers (Student-t, ν = 1), decisiveness-matched as S5",
    "PESS": "PESS: s = 2.0, fatigue 0.05 → 0.30, order bias 0.3·s, hard 'no difference' at 0.5 JND, "
            "phase weights SD 1.0, Schäfer-family ideals (per-bout rule); own metric",
    "PESS-GP": "PESS, GP-BALD (own block, if the time box allowed)",
    "BASE-cmaes": "BASE: CMA-ES (popsize 4) on Schäfer (ii), tuned on one fold",
    "BASE-gp_bald": "BASE: GP-BALD on Schäfer (ii), tuned on one fold",
}


def fmt_part(p: dict) -> str:
    if "sub" in p:
        bits = []
        for (y, room), s in p["sub"].items():
            ci = s.get("ci")
            txt = "not computed" if ci is None else ci3(ci)
            bits.append(f"{y} {room}: {txt} ({s['state']})")
        return "; ".join(bits)
    if "ci" in p:
        return f"{ci3(p['ci'])} ({p['state']})"
    return f"{p['state']}" + (f" — {p['note']}" if p.get("note") else "")


def write_report(path: Path, res: dict, v: dict, ctx: dict, meta: dict) -> None:
    L = []
    p = L.append
    verdict_line = "; ".join(f"{name} {v[name]['state']}" for name, _ in PREDICTIONS)
    p(f"**{verdict_line}.**")
    p("")
    p("# ICORR second review round (R2): XD, HT, PESS and BASE")
    p("")
    p(f"Generated by `scripts/icorr_r2_report.py`, run started {meta['started']}, at worktree commit "
      f"{meta['head']}, script sha256 {meta['script_sha']}. Spec: "
      "`docs/superpowers/specs/2026-10-17-icorr-user-tuning-reframe-design.md` §9a, R2 (registered "
      "2026-10-22 before any code or run; XD's registration after conversion, before any session; "
      "the XD ruling of 2026-10-22). PR2-1 (G-s2) is reported elsewhere. No session was run here.")
    p("")
    p("## 0. Rulings (committed before any R2 session row existed)")
    p("")
    for head, text in RULINGS:
        p(f"- **{head}.** {text}")
    p("")
    p("## 1. Registered verdicts")
    p("")
    p("| # | Prediction | Verdict | Parts |")
    p("|---|---|---|---|")
    for name, text in PREDICTIONS:
        parts = "<br>".join(f"{cid}: {fmt_part(pt)}" for cid, pt in v[name]["parts"].items()) \
            or v[name].get("note", "")
        p(f"| {name} | {text} | **{v[name]['state']}** | {parts} |")
    p("")
    p("Cell states (R1): " + "; ".join(f"{cid} {s}" for cid, s in res["state"].items()) + ".")
    p("")
    if meta.get("room_note"):
        p(meta["room_note"])
        p("")
    p("### Floors, ceilings and PsyPref per cell (R8, context)")
    p("")
    p("| cell | floor, median (Nm) | ceiling, as simulated | ceiling, median room | PsyPref k = 24, as "
      "simulated | PsyPref k = 24, median room |")
    p("|---|---|---|---|---|---|")
    for cid, r in ctx.items():
        lead = "a" if ("a", "Y1", K_MAIN) in r["pref"] else next(iter(r["err"]))
        rp = r["room"].get((lead, "pref"))
        rc = r["room"].get("ceiling")
        p(f"| {cid} | {p3(r['floor'])} | {p3(r['ceiling'])} | {p3(rc) if rc else '—'} | "
          f"{NAME[lead]} {p3(r['pref'][(lead, 'Y1', K_MAIN)])} | {p3(rp) if rp else '—'} |")
    p("")
    p("## 2. Each cell (context, no verdicts)")
    p("")
    p("Preference over generic assistance (R2's yardstick; PESS in the wearer's own metric at "
      "s = 2.0) and median error at k = 24 (Nm; PESS own metric, matched scale). As simulated: "
      "the joint bootstrap (R4); median-matched room: R5.")
    for cid, r in ctx.items():
        p("")
        p(f"### {cid}")
        p("")
        p(CELL_TEXT.get(cid, cid))
        p("")
        y2col = any(key[1] == "Y2" for key in r["pref"])
        head = "| optimizer | k = 12 | k = 24 | k = 40 |" + (" Y2 k = 24 |" if y2col else "") + \
            " error k = 24 | median room: k = 24 | median room: error k = 24 |"
        p(head)
        p("|" + "---|" * (head.count("|") - 1))
        for learner in r["err"]:
            row = f"| {NAME[learner]} | " + " | ".join(p3(r["pref"][(learner, "Y1", k)])
                                                        for k in KS) + " |"
            if y2col:
                row += f" {p3(r['pref'][(learner, 'Y2', K_MAIN)])} |"
            row += f" {p3(r['err'][learner])} |"
            rp, re_ = r["room"].get((learner, "pref")), r["room"].get((learner, "err"))
            row += f" {p3(rp) if rp else '—'} | {p3(re_) if re_ else '—'} |"
            p(row)
        p("")
        bits = [f"floor (median) {p3(r['floor'])}", f"generic distance (median) {p3(r['generic'])}",
                f"room generic − floor (median) {p3(r['room_nm'])}",
                f"ceiling {p3(r['ceiling'])}"]
        if "ceiling_y2" in r:
            bits.append(f"ceiling Y2 {p3(r['ceiling_y2'])}")
        if r["room"].get("floor"):
            bits.append(f"median room: floor {p3(r['room']['floor'])}, ceiling "
                        f"{p3(r['room']['ceiling'])}")
        if "share" in r:
            bits.append(f"PsyPref's share of the ceiling (P − 0.5)/(C − 0.5) {p3(r['share'], 2)}")
        p("Floors and ceilings (R8): " + "; ".join(bits) + ".")
        if "best_fast" in r:
            p("")
            p(f"Best fast comparator by point estimate at k = 24: {NAME[r['best_fast']]}.")
        if "a_minus_gp_Y1" in r:
            p("")
            p(f"PsyPref (e-pess) minus GP-BALD, k = 24, as simulated (context): Y1 "
              f"{ci3(r['a_minus_gp_Y1'])}; Y2 {ci3(r['a_minus_gp_Y2'])}.")
        if "no_difference_share" in r:
            p("")
            p("Share of 'no difference' answers among the first 24 comparisons: " + "; ".join(
                f"{NAME[lr]} {x:.2f}" for lr, x in r["no_difference_share"].items()) + ".")
        if "held_out" in r:
            h = r["held_out"]
            p("")
            p(f"Users of the four folds not used for tuning ({h['n']}; R9, no verdict): PsyPref "
              f"{h['a']:.3f}, BASE {h['base']:.3f}, difference {h['diff']:+.3f}. Against its own "
              f"family's prior-mean curve as generic assistance (sensitivity, R2): "
              f"{p3(r['own_generic_pref'])}. Seconds per comparison (10/50/90 %): "
              + ", ".join(f"{x:.2f}" for x in r["s_per_comparison"]) + ".")
    p("")
    p("## 3. BASE's tuning (R9)")
    p("")
    p("| method | setting | state on loso-AB30 | median error k = 24 (Nm) | chosen |")
    p("|---|---|---|---|---|")
    for learner, rows in res["tuning"].items():
        for t in rows:
            err = "—" if t["median_err"] is None else f"{t['median_err']:.3f}"
            p(f"| {NAME[learner]} | {json.dumps(t['setting'])} | {t['state']} | {err} | "
              f"{'yes' if t['chosen'] else ''} |")
    p("")
    p("## 4. Checks")
    p("")
    for name, b in res["blocks"].items():
        p(f"- `{name}.jsonl`: {'present' if b['exists'] else 'absent'}, {b['lines']} lines, "
          f"{b['ok']} successful rows, {b['unresolved']} unresolved error rows, no conflict copy.")
    for c in res["checks"] + meta.get("checks", []):
        p(f"- {c}")
    p("")
    p("## 5. Caveats")
    p("")
    p("- **The preference is a model probability** for the wearer that answered (Y1: the main "
      "wearer's validation duel; Y2: the stressed wearer's). It is not a measured preference.")
    p("- **XD** uses users from two independent datasets with the Molinaro prior; it decouples the "
      "prior from the users and is not a population claim (the author's ruling). Fukuchi trials "
      "repeat one ensemble-mean cycle, so those users have no stride-to-stride variability. "
      "Molinaro's moments are biological moments during assisted walking, the external ones "
      "unassisted (external_data_report.md).")
    p("- **HT (a)'s users are drawn afresh** at ν = 1.5 (the generator's rejection of landmark-order "
      "violations kept); PR2-3 compares two user sets, not the same users.")
    p("- **PESS** is read in the wearer's own metric on its matched scale, as S1; its ideals are the "
      "'ood-pb' users, whose room differs from the population's (robust_report.md §1).")
    p("- **BASE's generic assistance** is the ProMP prior-mean curve for every method (R2); its own "
      "family's prior mean is a sensitivity. The tuning fold is in-sample for BASE (R9).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def jsonable(x):
    if isinstance(x, dict):
        return {(k if isinstance(k, str) else " ".join(map(str, k))): jsonable(v)
                for k, v in x.items() if k != "boot"}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--simulated-only", action="store_true")
    ap.add_argument("--boot", type=int, default=jf.N_BOOT)
    ap.add_argument("--boot-verdict", type=int, default=N_BOOT_VERDICT)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = args.out
    run = out / "phase5-schaefer"
    report = args.report or out / "icorr" / "r2_report.md"
    numbers = args.json or out / "icorr" / "r2_numbers.json"
    if args.boot_verdict < 2000:
        raise SystemExit("at least 2000 bootstrap draws")
    meta = dict(started=stamp(), head=git_head(),
                script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12], checks=[])
    merge = out / "icorr" / "r2_merge_check.json"
    if merge.exists():
        m = json.loads(merge.read_text(encoding="utf-8"))
        meta["checks"].append(m["summary"])
    pop_keys = population_keys(run)
    folds = np.asarray([f for f, _ in pop_keys])
    print("reading rows ...", flush=True)
    res = load_all(out, run, pop_keys)
    for cid, keys in res["keys"].items():
        if [f for f, _ in keys] != list(folds):
            raise SystemExit(f"{cid}: its users are not 40 per fold in LOSO order")
    boot = kr.Boot(folds, args.boot_verdict)
    rs = None
    meta["room_note"] = "Median-matched room not computed (--simulated-only)."
    if not args.simulated_only:
        check = json.loads((out / "icorr" / "median_room_data_check.json").read_text(
            encoding="utf-8"))
        data = args.data or Path(check["data"])
        for f in check["files"]:
            if sha256(data / f["name"]) != f["sha256"]:
                raise SystemExit(f"{f['name']}: sha256 differs from median_room_data_check.json")
        rs = RoomSet(out, data, args.boot, folds)
        a = res["main"]["a"]
        gen = arr(a, "d_prior")
        q1 = ((gen - mat(a, "distances")[:, K_MAIN - 1]) / G.JND[1.0] >= 1.0 - G.TOL).astype(float)
        rooms = rs.get("population", cell_room(res, "main"))
        gate = float(rooms.stat("q1", q1, "median")[0])
        if abs(gate - jf.GATE_MEDIAN[0]) > jf.GATE_TOL:
            raise SystemExit(f"room gate failed: e-main A's Q>=1 at k = 24 is {gate:.3f} at the "
                             f"median-matched room, expected {jf.GATE_MEDIAN[0]}")
        meta["room_note"] = (f"Median-matched room (R5): level {rs.levels['median']:.3f} Nm; gate "
                             f"passed (e-main A's Q≥1 at k = 24 is {gate:.3f}, expected "
                             f"{jf.GATE_MEDIAN[0]}); Schäfer logs from `{data}`, sha256 as in "
                             "median_room_data_check.json.")
    print("verdicts ...", flush=True)
    v = verdicts(res, boot, rs)
    print("context ...", flush=True)
    ctx = context(res, boot, rs)
    if rs is not None:
        meta["room_note"] += " Effective sample size per user set: " + ", ".join(
            f"{name} {r.ess['median'][0]:.1f}" for name, r in rs.cache.items()) + "."
    # Reproduction gate: the main condition as the stress report and ceiling.md give it.
    repro = []
    for (room, learner), want in EXPECT_MAIN.items():
        if room == "simulated":
            got = ctx["main"]["pref"][(learner, "Y1", K_MAIN)]["estimate"]
        elif rs is None:
            continue
        else:
            got = ctx["main"]["room"][(learner, "pref")]["estimate"]
        if abs(round(got, 3) - want) > 1e-9:
            raise SystemExit(f"main condition {learner} at the {room} room: {got:.4f}, the "
                             f"reports' {want:.3f}; the yardstick or the room differs")
        repro.append(f"{NAME[learner]} {room} {want:.3f}")
    meta["checks"].append("Reproduction: the main condition's preference at k = 24 equals "
                          "stress_report.md and ceiling.md at three decimals ("
                          + ", ".join(repro) + ").")
    write_report(report, res, v, ctx, meta)
    numbers.write_text(json.dumps(jsonable(dict(meta=meta, state=res["state"], verdicts=v,
                                                 context=ctx, tuning=res["tuning"],
                                                 base_tuned=res["base_tuned"])), indent=1),
                       encoding="utf-8")
    print(f"wrote {report} and {numbers}", flush=True)
    for name, _ in PREDICTIONS:
        print(f"{name}: {v[name]['state']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

