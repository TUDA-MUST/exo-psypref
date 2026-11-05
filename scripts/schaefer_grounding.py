"""Grounding the simulation in the Schaefer self-tuning data.

Usage:  python scripts/schaefer_grounding.py --out <project>/out [--data <share>] [--boot 4000]

Reads, all read-only:
- the Schaefer share (``NN-selfOpt.mat`` only: preferred setting and tuning log; the
  ``NN-met.mat`` files, which hold the subject codes, are never opened);
- the extraction cache ``OUT/schaefer/cache/*.npz`` (built on our analysis of those data);
- the simulated users and fold priors ``OUT/phase5-schaefer`` (users/, priors/);
- the Phase 5e rows ``OUT/phase5e/e-main.jsonl``.

Writes ``OUT/schaefer_grounding/REPORT.md`` and ``OUT/schaefer_grounding/figures/*.png``.
``OUT`` is git-ignored: per-participant numbers (P1..P11 only) never enter the repository.

Conventions (see schaefer_grounding_NOTES.md for every judgement call):
- Profiles that are compared with the simulation (G1, G1b, G2, G4, kappa) live on the
  101-point PHASE_GRID including the duplicated endpoint, because d_prior, d_floor and the
  session distances are RMS values on that grid (prompref.user.oracle.rms_distance).
- Numbers that reproduce the existing Schaefer report (the gates and G3(a)) use its own
  1000-point grid, HI_GRID; G3(b, c) use the cache's per-stride 101-point curves.
- Torques are flexion positive; the cache is already negated and already drops the first
  10 strides after each condition change (extract.py N_SKIP), so neither is redone here.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import phase5e_report as er  # noqa: E402
import short_budget_report as sb  # noqa: E402
from preftune.profile import PHASE_GRID  # noqa: E402

from prompref.experiment.metrics import prior_mean_distance  # noqa: E402
from prompref.experiment.phase5 import _prior, _users, arm_space  # noqa: E402
from prompref.learning.arms import ARMS  # noqa: E402
from prompref.population.curves import TORQUE_LIMIT_NM  # noqa: E402
from prompref.population.users import physics_floor  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.oracle import rms_distance  # noqa: E402

DEFAULT_DATA = Path("data/schaefer_not_included")
FOLDS = ("loso-AB30", "loso-AB31", "loso-AB32", "loso-AB33", "loso-AB34")
K_MAIN = 24
S_LEVELS = (0.5, 1.0, 2.0)
JND = {s: true_jnd(s) for s in S_LEVELS}
SEED = 20261008
TOL = 1e-12  # benchmark_report.TOL, so the unweighted Q>=1 reproduces bit for bit

#: Planned predictions; never edited after the run.
SIM_ROOM_SPEC_NM = 2.31
G2_LOW, G2_HIGH = 0.34, 1.35
NOISE_CELL_NM = 0.6
G1B_LOW, G1B_HIGH = 0.50, 0.82
#: Aggregate numbers of out/schaefer/REPORT.md that the gates must reproduce.
REPORT_STEP_MEDIANS = np.array([3.75, 4.38, 3.62, 3.75])
REPORT_LATE_RMS_SD = np.array([2.7, 3.3, 2.8, 2.5])
REPORT_MAX_CMD_RMS = 0.51
REPORT_ALL_COND_RMS = (0.58, 0.10)


# ============================================================================ copied code
# --- from analysis/schaefer/sources.py (our analysis of those data)
FILE_IDS = ("02", "03", "05", "06", "07", "09", "10", "11", "13", "14", "15")
PSEUDONYM = {fid: f"P{i + 1}" for i, fid in enumerate(FILE_IDS)}
SESSIONS = ("02", "03", "04", "05")
TIMINGS = ("t1", "t2", "t3", "t4")
NO_TUNING_LOG = "10"
INITIAL_SETTING = np.array([95.0, 25.0, 50.0, 70.0])
CACHE_VERSION = 1  # analysis/schaefer/extract.py


def unwrap_t1(pct):
    """Map a t1 value in %ST onto [50, 150). (sources.py)"""
    return (np.asarray(pct, dtype=float) - 50.0) % 100.0 + 50.0


def to_pct_st(frac):
    """Fractions of stride (last axis t1..t4) to %ST with t1 unwrapped. (sources.py)"""
    pct = np.round(np.asarray(frac, dtype=float) * 100.0, 6)
    pct[..., 0] = unwrap_t1(pct[..., 0])
    return pct


def _read_mat(path: Path, names: list[str]) -> dict:
    """loadmat restricted to ``names``; the S: share is flaky, so one retry."""
    for attempt in (1, 2):
        try:
            return sio.loadmat(path, variable_names=names)
        except OSError as exc:
            if attempt == 2:
                raise
            print(f"read of {path.name} failed ({exc}); retrying once", flush=True)
            time.sleep(2.0)
    raise AssertionError("unreachable")


def load_selfopt(data_dir: Path, fid: str) -> dict:
    """Preferred setting and, where it survives, the tuning log. (sources.load_selfopt,
    reduced to the variables used here; ``author`` and ``info`` are not read.)"""
    m = _read_mat(data_dir / f"{fid}-selfOpt.mat",
                  ["tDurOpt", "tParSelfOpt", "datTParMan", "timStaXpcMan"])
    out = {"duration_s": float(np.asarray(m["tDurOpt"], float).ravel()[0]),
           "t_pref": to_pct_st(np.asarray(m["tParSelfOpt"], float).ravel()),
           "settings": None, "timestamps_s": None}
    if "datTParMan" in m:
        out["settings"] = to_pct_st(np.asarray(m["datTParMan"], float))
        out["timestamps_s"] = np.asarray(m["timStaXpcMan"], float).ravel()
    return out


# --- from analysis/schaefer/shapes.py (our analysis of those data)
def analytic_profile(t, phase, peak_ext_nm: float = 12.0, peak_flex_nm: float = 12.0):
    """Schaefer profile at arbitrary phases (%), flexion positive. (shapes.py)"""
    t1, t2, t3, t4 = np.asarray(t, dtype=float) % 100.0
    phase = np.asarray(phase, dtype=float) % 100.0
    out = np.zeros_like(phase)
    segments = (
        (t1, t2, -peak_ext_nm, True),
        (t2, t3, -peak_ext_nm, False),
        (t3, t4, +peak_flex_nm, True),
        (t4, t1, +peak_flex_nm, False),
    )
    for start, end, amplitude, rising in segments:
        width = (end - start) % 100.0
        width = width if width > 0.0 else 100.0
        offset = (phase - start) % 100.0
        inside = offset <= width
        u = np.where(inside, offset / width, 0.0)
        u = u if rising else 1.0 - u
        out = np.where(inside, amplitude * 0.5 * (1.0 - np.cos(np.pi * u)), out)
    return out


def circ_diff(a, b):
    """a - b on the 100 % cycle, in (-50, 50]. (shapes.py)"""
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 100.0
    return np.where(d > 50.0, d - 100.0, d)


# --- from analysis/schaefer/tuning.py (our analysis of those data)
EPS = 1e-6


def collapse_repeats(settings, timestamps):
    """Merge consecutive identical rows. (tuning.py)"""
    keep = np.ones(len(settings), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(settings, axis=0)) > EPS, axis=1)
    return settings[keep], timestamps[keep]


def median_steps(log: dict) -> np.ndarray:
    """Median absolute step per timing over the changes of that timing, collapsed rows.
    (tuning.participant_procedure, 'median_step_per_param')"""
    S, _ = collapse_repeats(log["settings"], log["timestamps_s"])
    steps = np.diff(S, axis=0)
    changed = np.abs(steps) > EPS
    return np.array([np.median(np.abs(steps[changed[:, k], k])) if changed[:, k].any()
                     else np.nan for k in range(4)])


def _window_rows(T, start_time):
    """Rows active at any time in [start_time, T[-1]]. (tuning.py)"""
    first = max(int(np.searchsorted(T, start_time, side="right")) - 1, 0)
    return np.arange(first, len(T))


def late_sd(log: dict) -> np.ndarray:
    """SD of the settings in the last third of the participant's own tuning time.
    (tuning.late_exploration, window 'last_third')"""
    S, T = collapse_repeats(log["settings"], log["timestamps_s"])
    rows = _window_rows(T, T[-1] - (T[-1] - T[0]) / 3.0)
    W = S[rows]
    return W.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(4)


# --- grid of analysis/schaefer/delivered.py
N_HI = 1001
HI_GRID = np.linspace(0.0, 100.0, N_HI)[:-1]


# ============================================================================ helpers
def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=float)))))


def hazen(x, q):
    return np.percentile(np.asarray(x, float), q, method="hazen")


def fmt(x, d: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{d}f}"


def ci_str(est, lo, hi, d: int = 2) -> str:
    return f"{fmt(est, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def verdict_upper(est, lo, hi, bound) -> str:
    """Prediction 'value < bound'."""
    if hi < bound:
        return "held"
    if lo >= bound:
        return "failed"
    return f"uncertain (point estimate {'held' if est < bound else 'failed'})"


def verdict_lower(est, lo, hi, bound) -> str:
    """Prediction 'value > bound'."""
    if lo > bound:
        return "held"
    if hi <= bound:
        return "failed"
    return f"uncertain (point estimate {'held' if est > bound else 'failed'})"


def verdict_range(est, lo, hi, low, high) -> str:
    """Prediction 'low <= value <= high'."""
    if low <= lo and hi <= high:
        return "held"
    if hi < low or lo > high:
        return "failed"
    inside = low <= est <= high
    return f"uncertain (point estimate {'held' if inside else 'failed'})"


def fold_draws(folds, n_boot: int, rng) -> np.ndarray:
    """Cluster-free bootstrap of users, resampled within each fold (as the 5e report)."""
    folds = np.asarray(folds)
    groups = [np.flatnonzero(folds == f) for f in dict.fromkeys(folds)]
    return np.concatenate([g[rng.integers(0, len(g), size=(n_boot, len(g)))] for g in groups],
                          axis=1)


def git_head() -> str:
    """HEAD, which of the script and its notes are uncommitted, and the script's sha256, so
    that a report from an uncommitted script can still be tied to the code that wrote it."""
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = [name for name in ("scripts/schaefer_grounding.py",
                                   "scripts/schaefer_grounding_NOTES.md")
                 if subprocess.run(["git", "status", "--porcelain", name], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip()]
        note = f" ({' and '.join(Path(d).name for d in dirty)} uncommitted)" if dirty else ""
        return f"{out}{note}, script sha256 {sha}"
    except (OSError, subprocess.CalledProcessError):
        return f"unknown, script sha256 {sha}"


# ============================================================================ loading
def load_cached(cache_dir: Path, fid: str, ss: str) -> dict:
    with np.load(cache_dir / f"{fid}-{ss}.npz") as z:
        data = {k: z[k] for k in z.files}
    if int(data["version"]) != CACHE_VERSION:
        raise RuntimeError(f"stale cache {fid}-{ss}")
    return data


def load_participants(data_dir: Path) -> dict:
    return {fid: load_selfopt(data_dir, fid) for fid in FILE_IDS}


def load_sim(run_dir: Path) -> dict:
    """The 200 population ideals of the five folds, in (fold, user_id) order."""
    keys, curves, folds, radius = [], [], [], []
    for fold in FOLDS:
        users = _users(str(run_dir), fold)
        for uid, user in users.items():
            if user.kind != "population" or "/pop" not in uid:
                continue
            keys.append((fold, uid))
            curves.append(np.asarray(user.curve_nm, float))
            folds.append(fold)
            radius.append(float(user.families["promp_s0"]["radius"]))
    return dict(keys=keys, curves=np.array(curves), folds=np.array(folds),
                radius=np.array(radius))


def load_main_cell(dir5e: Path):
    """e-main, ProMP, learner A, the 'main' cell exactly as phase5e_report selects it."""
    df = sb._load(Path(dir5e), "e-main")
    df = sb._n2048(sb._popn(df))
    cond = er.BY_ID["main"]
    mask = (df["family"] == "promp_s0").to_numpy() & (df["learner"] == "a").to_numpy()
    want = dict(er.DEFAULTS)
    want.update(cond.fields)
    for name, value in want.items():
        mask &= er._match(df, name, value)
    return df[mask].drop_duplicates(er.KEY, keep="last")


# ============================================================================ gates
def gate_commanded(cache_dir: Path) -> dict:
    """The copied analytic profile against the cached commanded torque, every section."""
    worst, worst_vs_cache = 0.0, 0.0
    n = 0
    for fid in FILE_IDS:
        for ss in SESSIONS:
            c = load_cached(cache_dir, fid, ss)
            for side in "LR":
                for j in range(1, 6):
                    ana = analytic_profile(c["conditions_pct_st"][j - 1], HI_GRID)
                    value = rms(c[f"{side}{j}_cmd_hi_mean"][:-1] - ana)
                    worst = max(worst, value)
                    worst_vs_cache = max(worst_vs_cache,
                                         abs(value - float(c[f"{side}_map_rms"][j - 1, j - 1])))
                    n += 1
    return dict(n=n, max_rms=worst, max_diff_vs_cached=worst_vs_cache)


def gate_tuning(logs: dict) -> dict:
    fids = [f for f in FILE_IDS if f != NO_TUNING_LOG]
    med = np.array([median_steps(logs[f]) for f in fids])
    sd = np.array([late_sd(logs[f]) for f in fids])
    return dict(fids=fids, median_step=med, late_sd=sd,
                median_of_medians=np.nanmedian(med, axis=0),
                late_rms_sd=np.sqrt((sd ** 2).mean(axis=0)))


# ============================================================================ G3 data
def participant_device(cache_dir: Path, fid: str, t_pref: np.ndarray) -> dict:
    """Per leg: the preferred condition's sessions (in session-file order) and the
    pooled mean of every condition, for the tracking reproduction."""
    pref = {"L": [], "R": []}
    conds: dict = {}
    for k, ss in enumerate(SESSIONS):
        c = load_cached(cache_dir, fid, ss)
        rows = c["conditions_pct_st"]
        is_pref = np.all(np.abs(circ_diff(rows, t_pref)) < 1e-6, axis=1)
        if is_pref.sum() != 1:
            raise RuntimeError(f"{PSEUDONYM[fid]} session {ss}: {is_pref.sum()} preferred rows")
        for side in "LR":
            for j in range(1, 6):
                key = f"{side}{j}"
                n = len(c[f"{key}_stride_s"])
                part = dict(n=n, meas101=c[f"{key}_meas101"].astype(float),
                            meas_hi=c[f"{key}_meas_hi_mean"])
                cond = "pref" if is_pref[j - 1] else tuple(np.round(rows[j - 1], 6))
                conds.setdefault(cond, {"L": [], "R": [], "t": rows[j - 1]})[side].append(part)
                if is_pref[j - 1]:
                    pref[side].append(part)
    all_rms = []
    for cond in conds.values():
        for side in "LR":
            parts = cond[side]
            n = np.array([p["n"] for p in parts], float)
            mean_hi = np.average([p["meas_hi"] for p in parts], axis=0, weights=n)
            all_rms.append(rms(mean_hi[:-1] - analytic_profile(cond["t"], HI_GRID)))
    return dict(pref=pref, all_cond_rms=all_rms, n_conditions=len(conds))


def device_numbers(pref_parts: list, t_pref: np.ndarray, block_sizes=(20, 45)) -> dict:
    """G3 for one participant and leg. ``pref_parts`` are the four sessions' preferred
    sections in session-file order, each with its strides in time order."""
    n = np.array([p["n"] for p in pref_parts], float)
    mean_hi = np.average([p["meas_hi"] for p in pref_parts], axis=0, weights=n)
    a = rms(mean_hi[:-1] - analytic_profile(t_pref, HI_GRID))
    X = np.vstack([p["meas101"] for p in pref_parts])
    b_pooled = rms(X - X.mean(axis=0))
    b_session = rms(np.vstack([p["meas101"] - p["meas101"].mean(axis=0) for p in pref_parts]))
    within = []
    for p in pref_parts:
        Y = p["meas101"]
        h = len(Y) // 2
        within.append(rms(Y[:h].mean(axis=0) - Y[-h:].mean(axis=0)))
    c_within = float(np.sqrt(np.mean(np.square(within))))
    h = len(X) // 2
    c_pooled = rms(X[:h].mean(axis=0) - X[-h:].mean(axis=0))
    means = [p["meas101"].mean(axis=0) for p in pref_parts]
    pairs = [rms(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))]
    c_between = float(np.sqrt(np.mean(np.square(pairs))))
    blocks = {}
    for B in block_sizes:
        diffs = []
        for p in pref_parts:
            Y = p["meas101"]
            m = len(Y) // B
            bm = Y[:m * B].reshape(m, B, -1).mean(axis=1)
            diffs += [rms(bm[i + 1] - bm[i]) for i in range(m - 1)]
        blocks[B] = float(np.sqrt(np.mean(np.square(diffs))))
    return dict(a=a, b_pooled=b_pooled, b_session=b_session, c_within=c_within,
                c_pooled=c_pooled, c_between=c_between, blocks=blocks, n_strides=int(n.sum()))


# ============================================================================ G1, G1b
def kde_weights(r_sim, y_real):
    return gaussian_kde(y_real)(r_sim) / gaussian_kde(r_sim)(r_sim)


def bin_weights(r_sim, y_real, n_bins: int = 4):
    """Quantile-bin weights: interior edges at the real quartiles (Hazen), outer edges at the
    real minimum and maximum widened by half the neighbouring bin. Weight = real share of the
    bin / number of simulated users in it; simulated users outside get zero."""
    q = hazen(y_real, np.linspace(0, 100, n_bins + 1))
    edges = q.copy()
    edges[0] = q[0] - 0.5 * (q[1] - q[0])
    edges[-1] = q[-1] + 0.5 * (q[-1] - q[-2])
    sim_bin = np.digitize(r_sim, edges) - 1
    real_bin = np.clip(np.digitize(y_real, edges) - 1, 0, n_bins - 1)
    w = np.zeros(len(r_sim))
    counts, lost = [], 0.0
    for b in range(n_bins):
        share = float(np.mean(real_bin == b))
        m = sim_bin == b
        counts.append(int(m.sum()))
        if m.any():
            w[m] = share / m.sum()
        else:
            lost += share
    return w, edges, counts, lost


def weighted_share(w, flags) -> float:
    return float(np.sum(w * flags) / np.sum(w))


def ess(w) -> float:
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


#: Room bands for the Q>=1 profile of the simulated users (Nm, d_prior - d_floor).
ROOM_BANDS = (1.0, 1.25, 1.5, 1.75)
#: Hatzfeld form of the 5e design's Weber observer (E4): JND proportional to RMS + a.
WEBER_A_NM = 1.0
KNN_K = range(3, 21)


def g1b_sensitivities(R: dict, room, q1, y, d_loo, kappa: float, n_boot: int) -> dict:
    """Unregistered sensitivities of the G1b share:
    the Q>=1 profile over room bands, the KDE's support, nearest-neighbour matching,
    the room measured in the ProMP family, an affine Weber kappa and an absolute JND."""
    rng = np.random.default_rng(SEED + 1)
    out: dict = {}
    # Q>=1 by room band, and where the real rooms fall
    edges = np.array((0.0,) + ROOM_BANDS + (np.inf,))
    band = np.digitize(room, edges) - 1
    real_band = np.digitize(y, edges) - 1
    out["bands"] = [dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=int(np.sum(band == b)),
                         q1=float(q1[band == b].mean()) if np.any(band == b) else np.nan,
                         n_real=int(np.sum(real_band == b))) for b in range(len(edges) - 1)]
    out["below_jnd_q1"] = float(q1[room < JND[1.0]].mean()) if np.any(room < JND[1.0]) else np.nan
    out["below_jnd_n"] = int(np.sum(room < JND[1.0]))
    # support of the KDE target
    kr = gaussian_kde(y)
    w = kde_weights(room, y)
    wn = w / w.sum()
    lo_y, hi_y = float(y.min()), float(y.max())
    out["mass_below_min"] = float(kr.integrate_box_1d(-np.inf, lo_y))
    out["mass_below_jnd"] = float(kr.integrate_box_1d(-np.inf, JND[1.0]))
    out["mass_above_max"] = float(kr.integrate_box_1d(hi_y, np.inf))
    below = room < lo_y
    out["sim_below_min_n"] = int(below.sum())
    out["sim_below_min_weight"] = float(wn[below].sum())
    out["sim_below_min_q1"] = float(q1[below].mean()) if below.any() else np.nan
    inside = (room >= lo_y) & (room <= hi_y)
    wr = np.where(inside, w, 0.0)
    out["restricted"] = weighted_share(wr, q1)
    out["restricted_ess"] = ess(wr)
    out["support"] = (lo_y, hi_y)
    # nearest-neighbour matching: per participant the k simulated users with the closest room
    knn = []
    for kk in KNN_K:
        idx = np.argsort(np.abs(room[None, :] - y[:, None]), axis=1, kind="stable")[:, :kk]
        knn.append(float(q1[idx].mean(axis=1).mean()))
    out["knn"] = np.array(knn)
    # (i) the real room measured in the ProMP family, pooled over folds (11 x 5 values)
    M = np.array([R["g4"]["per_fold"][f]["room_promp"] for f in FOLDS]).T  # (11, 5)
    out["promp_target_median"] = float(np.median(M))
    out["promp"] = weighted_share(kde_weights(room, M.ravel()), q1)
    out["promp_ess"] = ess(kde_weights(room, M.ravel()))
    pdraw = rng.integers(0, M.shape[0], size=(n_boot, M.shape[0]))
    out["promp_ci"] = np.percentile([weighted_share(kde_weights(room, M[d].ravel()), q1)
                                     for d in pdraw], [2.5, 97.5])
    fold_mean = M.mean(axis=1)
    out["promp_fold_mean_median"] = float(np.median(fold_mean))
    out["promp_fold_mean_ci"] = np.percentile(np.median(fold_mean[pdraw], axis=1), [2.5, 97.5])
    # (ii) affine Weber kappa
    kw = float((np.median(R["sim_rms"]) + WEBER_A_NM) / (np.median(R["real_rms"]) + WEBER_A_NM))
    out["kappa_weber"] = kw
    out["weber"] = weighted_share(kde_weights(room, kw * d_loo), q1)
    out["weber_ess"] = ess(kde_weights(room, kw * d_loo))
    # (iii) absolute JND: the device's Nm taken as they are (no amplitude matching)
    out["absolute"] = weighted_share(kde_weights(room, d_loo), q1)
    out["absolute_ess"] = ess(kde_weights(room, d_loo))
    # real share in the first quantile bin of the planned bin method
    bedges = R["g1b"]["bin_edges"]
    out["bin0_real_share"] = float(np.mean(np.clip(np.digitize(y, bedges) - 1, 0,
                                                    len(bedges) - 2) == 0))
    return out


# ============================================================================ G2
def landmark_shift_rms(t: np.ndarray, k: int, delta: float) -> float:
    """Mean over +delta and -delta of the RMS change of the analytic profile at ``t`` when
    landmark k moves by delta (101-point grid). Asserts the cyclic order survives."""
    base = analytic_profile(t, PHASE_GRID)
    out = []
    for sign in (+1.0, -1.0):
        moved = np.array(t, dtype=float)
        moved[k] += sign * delta
        cyc = np.array([moved[1], moved[2], moved[3], moved[0], moved[1] + 100.0])
        if not np.all(np.diff(cyc) > 0.0):
            raise RuntimeError(f"moving t{k + 1} by {sign * delta} breaks the cyclic order")
        out.append(rms_distance(analytic_profile(moved, PHASE_GRID), base))
    return float(np.mean(out))


def to_nm(t_pref: dict, per_timing: dict) -> dict:
    """Per participant: mean over timings (finite values) of the landmark-shift RMS."""
    out = {}
    for fid, vals in per_timing.items():
        nm = [landmark_shift_rms(t_pref[fid], k, float(v)) if np.isfinite(v) else np.nan
              for k, v in enumerate(vals)]
        out[fid] = dict(per_timing=np.array(nm), mean=float(np.nanmean(nm)),
                        n_timings=int(np.isfinite(nm).sum()))
    return out


# ============================================================================ main analysis
def analyse(out_root: Path, data_dir: Path, n_boot: int) -> dict:
    rng = np.random.default_rng(SEED)
    cache_dir = out_root / "schaefer" / "cache"
    run_dir = out_root / "phase5-schaefer"
    R: dict = {"provenance": dict(data=str(data_dir), cache=str(cache_dir), run=str(run_dir),
                                  rows=str(out_root / "phase5e" / "e-main.jsonl"),
                                  date=_dt.date.today().isoformat(), commit=git_head(),
                                  n_boot=n_boot)}

    # ---------------------------------------------------------------- participants
    logs = load_participants(data_dir)
    t_pref = {f: logs[f]["t_pref"] for f in FILE_IDS}
    P = np.array([t_pref[f] for f in FILE_IDS])
    R["pref_means"] = P.mean(axis=0)
    real = np.array([analytic_profile(t_pref[f], PHASE_GRID) for f in FILE_IDS])
    R["real_rms"] = np.array([rms(c) for c in real])

    # ---------------------------------------------------------------- gates
    R["gate_cmd"] = gate_commanded(cache_dir)
    R["gate_tuning"] = gate_tuning(logs)

    # ---------------------------------------------------------------- simulation
    sim = load_sim(run_dir)
    main = load_main_cell(out_root / "phase5e")
    row_index = {(f, u): i for i, (f, u) in enumerate(zip(main["fold"], main["user_id"]))}
    order = np.array([row_index[k] for k in sim["keys"]])
    main = main.iloc[order].reset_index(drop=True)
    d_prior = main["d_prior"].astype(float).to_numpy()
    d_floor = main["d_floor"].astype(float).to_numpy()
    d24 = np.array([float(d[K_MAIN - 1]) for d in main["distances"]])
    room = d_prior - d_floor
    q1 = (d_prior - d24) / JND[1.0] >= 1.0 - TOL
    folds = sim["folds"]
    R["sim"] = dict(n=len(room), room=room, q1=q1, d_floor=d_floor, folds=folds,
                    arm_radius=main.groupby("fold")["arm_radius"].unique().to_dict())
    sim_rms = np.array([rms(c) for c in sim["curves"]])
    kappa = float(np.median(sim_rms) / np.median(R["real_rms"]))
    R["kappa"] = kappa
    R["sim_rms"] = sim_rms
    R["sign_corr"] = float(np.corrcoef(sim["curves"].mean(axis=0), real.mean(axis=0))[0, 1])

    # common bootstrap draws: simulated users within folds, participants
    sim_draws = fold_draws(folds, n_boot, rng)
    part_draws = rng.integers(0, len(FILE_IDS), size=(n_boot, len(FILE_IDS)))
    kappa_b = np.median(sim_rms[sim_draws], axis=1) / np.median(R["real_rms"][part_draws], axis=1)

    # ---------------------------------------------------------------- G1
    start = analytic_profile(INITIAL_SETTING, PHASE_GRID)
    d_start = np.array([rms_distance(c, start) for c in real])
    d_loo = np.array([rms_distance(real[i], np.delete(real, i, axis=0).mean(axis=0))
                      for i in range(len(real))])
    loo_timing = np.array([rms_distance(real[i], analytic_profile(np.delete(P, i, axis=0).mean(axis=0),
                                                                 PHASE_GRID))
                           for i in range(len(real))])
    diff_b = (np.median(kappa_b[:, None] * d_loo[part_draws], axis=1)
              - np.median(room[sim_draws], axis=1))
    diff = kappa * np.median(d_loo) - np.median(room)
    lo, hi = np.percentile(diff_b, [2.5, 97.5])
    med_b = np.median(kappa_b[:, None] * d_loo[part_draws], axis=1)
    R["g1"] = dict(d_start=d_start, d_loo=d_loo, loo_timing=loo_timing,
                   sim_room_median=float(np.median(room)),
                   sim_room_q=hazen(room, [5, 10, 25, 50, 75, 90, 95]),
                   matched_median=kappa * float(np.median(d_loo)),
                   matched_median_ci=np.percentile(med_b, [2.5, 97.5]),
                   diff=float(diff), diff_ci=(float(lo), float(hi)),
                   share_sim_below_real_max=float(np.mean(room <= kappa * d_loo.max())),
                   verdict=verdict_upper(diff, lo, hi, 0.0))

    # ---------------------------------------------------------------- G1b
    y = kappa * d_loo
    w = kde_weights(room, y)
    wb, edges, counts, lost = bin_weights(room, y)
    raw_mask = room <= np.median(y)
    g1b = dict(unweighted=float(q1.mean()), kde=weighted_share(w, q1), kde_ess=ess(w),
               bin=weighted_share(wb, q1) if wb.sum() > 0 else np.nan,
               bin_ess=ess(wb) if wb.sum() > 0 else np.nan, bin_edges=edges, bin_counts=counts,
               bin_lost=lost, raw_n=int(raw_mask.sum()),
               raw=float(q1[raw_mask].mean()) if raw_mask.any() else np.nan,
               kde_bw_real=float(gaussian_kde(y).factor * np.std(y, ddof=1)),
               kde_bw_sim=float(gaussian_kde(room).factor * np.std(room, ddof=1)))
    boot_sim, boot_part, boot_joint, ess_sim = [], [], [], []
    for b in range(n_boot):
        rs, qs = room[sim_draws[b]], q1[sim_draws[b]]
        yp = kappa_b[b] * d_loo[part_draws[b]]
        ws = kde_weights(rs, y)
        boot_sim.append(weighted_share(ws, qs))
        ess_sim.append(ess(ws))
        boot_part.append(weighted_share(kde_weights(room, kappa * d_loo[part_draws[b]]), q1))
        boot_joint.append(weighted_share(kde_weights(rs, yp), qs))
    g1b["ci_sim"] = np.percentile(boot_sim, [2.5, 97.5])
    g1b["ci_part"] = np.percentile(boot_part, [2.5, 97.5])
    g1b["ci_joint"] = np.percentile(boot_joint, [2.5, 97.5])
    g1b["ess_sim_range"] = np.percentile(ess_sim, [2.5, 97.5])
    g1b["verdict"] = verdict_range(g1b["kde"], *g1b["ci_sim"], G1B_LOW, G1B_HIGH)
    g1b["verdict_joint"] = verdict_range(g1b["kde"], *g1b["ci_joint"], G1B_LOW, G1B_HIGH)
    g1b["weights"] = w
    R["g1b"] = g1b

    # ---------------------------------------------------------------- G2
    gt = R["gate_tuning"]
    steps = {f: gt["median_step"][i] for i, f in enumerate(gt["fids"])}
    lates = {f: gt["late_sd"][i] for i, f in enumerate(gt["fids"])}
    step_nm = to_nm(t_pref, steps)
    late_nm = to_nm(t_pref, lates)
    sv = np.array([step_nm[f]["mean"] for f in gt["fids"]])
    lv = np.array([late_nm[f]["mean"] for f in gt["fids"]])
    pd_ = rng.integers(0, len(sv), size=(n_boot, len(sv)))
    step_med_b = np.median(sv[pd_], axis=1)
    late_med_b = np.median(lv[pd_], axis=1)
    g2 = dict(step_nm=step_nm, late_nm=late_nm, step=sv, late=lv,
              step_med=float(np.median(sv)), step_med_ci=np.percentile(step_med_b, [2.5, 97.5]),
              step_med_matched=kappa * float(np.median(sv)),
              step_med_matched_ci=np.percentile(kappa_b * step_med_b, [2.5, 97.5]),
              late_med=float(np.median(lv)), late_med_ci=np.percentile(late_med_b, [2.5, 97.5]),
              late_med_matched=kappa * float(np.median(lv)),
              late_med_matched_ci=np.percentile(kappa_b * late_med_b, [2.5, 97.5]),
              per_timing_step=np.nanmedian(np.array([step_nm[f]["per_timing"] for f in gt["fids"]]),
                                           axis=0),
              per_timing_late=np.nanmedian(np.array([late_nm[f]["per_timing"] for f in gt["fids"]]),
                                           axis=0))
    g2["verdict_matched"] = verdict_range(g2["step_med_matched"], *g2["step_med_matched_ci"],
                                          G2_LOW, G2_HIGH)
    g2["verdict_unscaled"] = verdict_lower(g2["step_med"], *g2["step_med_ci"], JND[1.0])
    g2["late_zero"] = int(np.sum(gt["late_sd"] == 0.0))  # timings no longer moved in the window
    g2["late_n"] = int(gt["late_sd"].size)
    R["g2"] = g2

    # ---------------------------------------------------------------- G3
    dev, all_cond = {}, []
    for fid in FILE_IDS:
        d = participant_device(cache_dir, fid, t_pref[fid])
        all_cond += d["all_cond_rms"]
        dev[fid] = {side: device_numbers(d["pref"][side], t_pref[fid]) for side in "LR"}
        dev[fid]["n_conditions"] = d["n_conditions"]
    R["g3_all_cond"] = (float(np.mean(all_cond)), float(np.std(all_cond, ddof=1)), len(all_cond))

    def legs(field, sub=None):
        return np.array([[dev[f][s][field] if sub is None else dev[f][s][field][sub]
                          for s in "LR"] for f in FILE_IDS])  # (11, 2)

    fields = {"a": legs("a"), "b_pooled": legs("b_pooled"), "b_session": legs("b_session"),
              "c_within": legs("c_within"), "c_pooled": legs("c_pooled"),
              "c_between": legs("c_between"), "block20": legs("blocks", 20),
              "block45": legs("blocks", 45)}
    pdraw = rng.integers(0, len(FILE_IDS), size=(n_boot, len(FILE_IDS)))
    g3 = {"dev": dev}
    for name, M in fields.items():
        flat = M.ravel()
        boot = np.median(M[pdraw].reshape(n_boot, -1), axis=1)
        g3[name] = dict(values=M, median=float(np.median(flat)), mean=float(flat.mean()),
                        sd=float(flat.std(ddof=1)), min=float(flat.min()), max=float(flat.max()),
                        ci=np.percentile(boot, [2.5, 97.5]))
    g3["b_gt_c_legs"] = int(np.sum(fields["b_pooled"] > fields["c_within"]))
    paired = fields["b_pooled"] - fields["c_within"]
    cdiff = np.median(paired[pdraw].reshape(n_boot, -1), axis=1)
    g3["b_minus_c"] = float(np.median(paired))  # median of the paired (per-leg) differences
    g3["b_minus_c_ci"] = np.percentile(cdiff, [2.5, 97.5])
    # (c') reading for the report only, after every planned draw: reuses pdraw
    g3["b_gt_cpooled_legs"] = int(np.sum(fields["b_pooled"] > fields["c_pooled"]))
    paired_p = fields["b_pooled"] - fields["c_pooled"]
    g3["b_minus_cpooled"] = float(np.median(paired_p))
    g3["b_minus_cpooled_ci"] = np.percentile(
        np.median(paired_p[pdraw].reshape(n_boot, -1), axis=1), [2.5, 97.5])
    g3["verdict_c"] = verdict_upper(g3["c_within"]["median"], *g3["c_within"]["ci"], NOISE_CELL_NM)
    g3["verdict_bc"] = verdict_lower(g3["b_minus_c"], *g3["b_minus_c_ci"], 0.0)
    R["g3"] = g3

    # ---------------------------------------------------------------- G4
    tight = ARMS["tight"].radius(8)
    g4 = {"tight": tight, "per_fold": {}}
    matched = kappa * real
    for fold in FOLDS:
        prior = _prior(str(run_dir), fold, "promp_s0")
        W = prior.whitening()
        if W.d != 8:
            raise RuntimeError(f"{fold}: whitening keeps {W.d} directions, expected 8")
        design = prior.design()
        basis = design.basis
        rows_radius = R["sim"]["arm_radius"][fold]
        if len(rows_radius) != 1 or abs(float(rows_radius[0]) - tight) > 1e-9:
            raise RuntimeError(f"{fold}: e-main arm radius {rows_radius} != tight {tight}")
        radius, resid, wmax, floor_r, floor_nm, unscaled = [], [], [], [], [], []
        for i, fid in enumerate(FILE_IDS):
            yv = matched[i]
            wv = basis.project(yv)
            radius.append(float(np.linalg.norm(W.to_z(wv))))
            resid.append(rms_distance(basis.curve(wv), yv))
            wmax.append(float(np.max(np.abs(wv))))
            fl, theta = physics_floor(design, yv, n_starts=8, seed=SEED, whitening=W)
            floor_nm.append(float(fl))
            floor_r.append(float(np.linalg.norm(W.to_z(theta))))
            unscaled.append(float(np.linalg.norm(W.to_z(basis.project(real[i])))))
        # context, not planned: the real preferences' room in the ProMP family, with
        # d_prior computed exactly as the sessions do (prior_mean_distance on the arm design)
        arm_design = arm_space(prior, "tight").design
        d_prior_real = np.array([prior_mean_distance(arm_design, matched[i])
                                 for i in range(len(FILE_IDS))])
        in_fold = folds == fold
        check_idx = np.flatnonzero(in_fold)[:5]
        d_prior_check = max(abs(prior_mean_distance(arm_design, sim["curves"][j]) - d_prior[j])
                            for j in check_idx)
        # pipeline check on the fold's own simulated users
        sim_lsq = np.array([np.linalg.norm(W.to_z(basis.project(c))) for c in sim["curves"][in_fold]])
        # an ideal "touches" the device where |tau*| reaches 12 Nm (phase5e_report._touches_from)
        sim_touch = np.array([bool(np.any(np.abs(c) >= TORQUE_LIMIT_NM - 1e-9))
                              for c in sim["curves"][in_fold]])
        g4["per_fold"][fold] = dict(
            radius=np.array(radius), resid=np.array(resid), wmax=max(wmax),
            floor_radius=np.array(floor_r), floor_nm=np.array(floor_nm),
            unscaled=np.array(unscaled), inside=int(np.sum(np.array(radius) <= tight)),
            unscaled_inside=int(np.sum(np.array(unscaled) <= tight)),
            sim_stored=sim["radius"][in_fold], sim_lsq=sim_lsq, sim_touch=sim_touch,
            sim_maxabs=np.abs(sim["curves"][in_fold]).max(axis=1),
            d_prior_real=d_prior_real, room_promp=d_prior_real - np.array(resid),
            d_prior_check=float(d_prior_check),
            sim_inside=float(np.mean(sim["radius"][in_fold] <= tight)))
    g4["max_abs_matched_nm"] = float(np.max(np.abs(matched)))
    g4["min_inside"] = min(v["inside"] for v in g4["per_fold"].values())
    g4["verdict"] = "held" if g4["min_inside"] >= 10 else "failed"
    R["g4"] = g4

    # ---------------------------------------------------------------- G1b sensitivities
    # Not planned (added after verification, 2026-10-08). Own RNG stream, so that every
    # planned interval above keeps its draws.
    R["g1b_sens"] = g1b_sensitivities(R, room, q1, y, d_loo, kappa, n_boot)
    R["fids"] = FILE_IDS
    R["t_pref"] = t_pref
    return R


# ============================================================================ figure
def make_figure(R: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SIM, REAL, AXIS = "#004488", "#DDAA33", "#666666"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": AXIS, "xtick.color": AXIS,
                         "ytick.color": AXIS, "axes.labelcolor": "black"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8), constrained_layout=True)

    room = R["sim"]["room"]
    w = R["g1b"]["weights"]
    y = R["kappa"] * R["g1"]["d_loo"]
    bins = np.linspace(0.0, max(room.max(), y.max()) * 1.02, 30)
    ax1.hist(room, bins=bins, density=True, color=SIM, alpha=0.35, edgecolor="white",
             linewidth=0.5, label="simulated (200 users)")
    ax1.hist(room, bins=bins, weights=w, density=True, histtype="step", color=SIM,
             linewidth=1.5, linestyle="--", label="simulated, reweighted")
    ymax = ax1.get_ylim()[1] * 1.05
    ax1.plot(y, np.full_like(y, -0.06 * ymax), "D", color=REAL, markersize=5,
             markeredgecolor="black", markeredgewidth=0.4, clip_on=False,
             label="real (b) ×κ (n = 11)")
    ax1.axvline(np.median(room), color=SIM, linewidth=1, label="median, simulated")
    ax1.axvline(np.median(y), color=REAL, linewidth=1, label="median, real")
    ax1.axvline(JND[1.0], color=AXIS, linewidth=0.8, linestyle=":")
    ax1.text(JND[1.0], ymax * 0.5, "1 JND (s = 1) ", color=AXIS, va="center", ha="right",
             rotation=90, fontsize=6.5)
    ax1.set_ylim(-0.1 * ymax, ymax)
    ax1.set_xlabel("room for personalisation (Nm RMS)")
    ax1.set_ylabel("density")
    ax1.set_title("G1/G1b: real vs simulated room", fontsize=8)
    ax1.legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    g4 = R["g4"]
    x = np.arange(len(R["fids"]))
    markers = ("o", "s", "^", "v", "D")
    for j, (fold, v) in enumerate(g4["per_fold"].items()):
        ax2.plot(x + (j - 2) * 0.1, v["radius"], markers[j], color=REAL, markersize=3.5,
                 markeredgecolor="black", markeredgewidth=0.3, linestyle="none",
                 label=fold.replace("loso-", "fold "))
    sim_r = np.concatenate([v["sim_stored"] for v in g4["per_fold"].values()])
    xs = len(x) + 0.8
    ax2.boxplot([sim_r], positions=[xs], widths=0.6, showfliers=False,
                medianprops=dict(color=SIM), boxprops=dict(color=SIM),
                whiskerprops=dict(color=SIM), capprops=dict(color=SIM))
    ax2.axhline(g4["tight"], color=AXIS, linestyle="--", linewidth=0.9)
    ax2.text(-0.5, g4["tight"], " tight arm radius", color=AXIS, va="bottom", fontsize=7)
    ax2.set_xticks(list(x) + [xs])
    ax2.set_xticklabels([PSEUDONYM[f] for f in R["fids"]] + ["sim.\nusers"], rotation=0,
                        fontsize=6.5)
    ax2.set_ylabel("whitened radius |z|")
    ax2.set_title("G4: real preferences in A's search region", fontsize=8)
    ax2.legend(frameon=False, fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================================ report
def build_report(R: dict) -> str:
    fids = R["fids"]
    k = R["kappa"]
    pv = R["provenance"]
    L: list[str] = []
    L += ["# Grounding the simulation in the Schäfer self-tuning data", "",
          f"Generated by `scripts/schaefer_grounding.py` on {pv['date']} at worktree commit "
          f"{pv['commit']}. Spec: `docs/superpowers/specs/2026-10-08-schaefer-grounding-design.md` "
          "(predictions registered before this run). Participants P1..P11 in file-id order "
          "(out/schaefer/REPORT.md §1.1). Torques flexion positive. Bootstrap: "
          f"{pv['n_boot']} draws, 95 % percentile intervals.", "",
          f"- Share (read only): `{pv['data']}` (NN-selfOpt.mat only)",
          f"- Cache: `{pv['cache']}`", f"- Simulated users and priors: `{pv['run']}`",
          f"- Rows: `{pv['rows']}`", "",
          "**Verdict rule.** *held*: the point estimate meets the prediction and the 95 % "
          "interval lies wholly on the predicted side; *failed*: the interval lies wholly on the "
          "other side; *uncertain*: the interval straddles the boundary (the point estimate's "
          "side is given).", ""]

    # ------------------------------------------------------------ summary
    g1, g1b, g2, g3, g4 = R["g1"], R["g1b"], R["g2"], R["g3"], R["g4"]
    L += ["## Summary", "",
          "| check | key number | prediction | verdict |", "|---|---|---|---|",
          f"| G1 room | real LOO room ×κ median {fmt(g1['matched_median'])} Nm "
          f"[{fmt(g1['matched_median_ci'][0])}, {fmt(g1['matched_median_ci'][1])}] vs simulated "
          f"{fmt(g1['sim_room_median'])} Nm; difference {ci_str(g1['diff'], *g1['diff_ci'])} | "
          f"real (b) ×κ median < simulated room | {g1['verdict']} |",
          f"| G1b reliability at the real room | Q≥1 reweighted (KDE) {fmt(g1b['kde'])} "
          f"[{fmt(g1b['ci_joint'][0])}, {fmt(g1b['ci_joint'][1])}] (participants and simulated "
          f"users resampled; simulated users alone [{fmt(g1b['ci_sim'][0])}, "
          f"{fmt(g1b['ci_sim'][1])}]), ESS {g1b['kde_ess']:.0f} of 200 simulated users, target "
          f"distribution from 11 participants; unweighted {fmt(g1b['unweighted'])} | 0.50–0.82 | "
          f"{g1b['verdict']} (both intervals{'' if g1b['verdict'] == g1b['verdict_joint'] else '; joint: ' + g1b['verdict_joint']}) |",
          f"| G2 step size | median step ×κ {ci_str(g2['step_med_matched'], *g2['step_med_matched_ci'])} Nm; "
          f"unscaled {ci_str(g2['step_med'], *g2['step_med_ci'])} Nm; late-exploration SD ×κ "
          f"{ci_str(g2['late_med_matched'], *g2['late_med_matched_ci'])} Nm (not registered) | "
          f"×κ in 0.34–1.35; unscaled > 0.674 | ×κ: {g2['verdict_matched']}; unscaled: "
          f"{g2['verdict_unscaled']} (one measurement under two scalings, see G2) |",
          f"| G3 device | (c) presentation-level median {ci_str(g3['c_within']['median'], *g3['c_within']['ci'])} Nm; "
          f"(b) stride-to-stride {fmt(g3['b_pooled']['median'])} Nm | (c) < 0.6; (b) > (c) | "
          f"(c): {g3['verdict_c']}; (b) > (c): {g3['verdict_bc']} |",
          f"| G4 coverage | inside the tight radius: "
          + ", ".join(f"{v['inside']}/11" for v in g4["per_fold"].values())
          + f" (folds AB30..AB34) | ≥ 10/11 in every fold | {g4['verdict']} |", "",
          "G1 and G4 reuse data the simulated timing population was fitted to (Phase 5 spec "
          "§3.5a): they check the room and the change of family, not the timing population "
          "independently. G2 and G3 use parts of the data the simulation did not use (tuning "
          "logs, measured torque). See Limitations.", ""]

    # ------------------------------------------------------------ gates
    gc, gt = R["gate_cmd"], R["gate_tuning"]
    m_all = R["g3_all_cond"]
    L += ["## 0. Reproduction gates", "",
          "| gate | existing report | this script | ok |", "|---|---|---|---|",
          f"| analytic profile vs commanded torque, max RMS over {gc['n']} section-legs | ≤ "
          f"{REPORT_MAX_CMD_RMS} Nm | {gc['max_rms']:.3f} Nm (largest difference from the cached "
          f"value {gc['max_diff_vs_cached']:.1e} Nm) | "
          f"{'yes' if gc['max_rms'] <= REPORT_MAX_CMD_RMS + 5e-3 else 'NO'} |",
          f"| median of participant median steps t1..t4 (%ST), n = {len(gt['fids'])} | "
          f"{', '.join(f'{v:.2f}' for v in REPORT_STEP_MEDIANS)} | "
          f"{', '.join(f'{v:.3f}' for v in gt['median_of_medians'])} | "
          f"{'yes' if np.allclose(gt['median_of_medians'], REPORT_STEP_MEDIANS, atol=0.006) else 'NO'} |",
          f"| RMS late-exploration SD, last third (%ST) | "
          f"{', '.join(f'{v:.1f}' for v in REPORT_LATE_RMS_SD)} | "
          f"{', '.join(f'{v:.2f}' for v in gt['late_rms_sd'])} | "
          f"{'yes' if np.allclose(gt['late_rms_sd'], REPORT_LATE_RMS_SD, atol=0.051) else 'NO'} |",
          f"| measured vs analytic, all condition-leg means (mean ± SD; 17 conditions — 16 offsets and "
          f"the pooled preferred one — × 22 legs, n = {m_all[2]}) | "
          f"{REPORT_ALL_COND_RMS[0]} ± {REPORT_ALL_COND_RMS[1]} Nm | {m_all[0]:.3f} ± {m_all[1]:.3f} Nm | "
          f"{'yes' if abs(m_all[0] - REPORT_ALL_COND_RMS[0]) < 0.006 and abs(m_all[1] - REPORT_ALL_COND_RMS[1]) < 0.006 else 'NO'} |",
          f"| e-main main cell, ProMP, A: users / median room / Q≥1 at k = 24 | 200 / 2.31 Nm / 0.82 | "
          f"{R['sim']['n']} / {R['g1']['sim_room_median']:.3f} Nm / {R['g1b']['unweighted']:.3f} | "
          f"{'yes' if R['sim']['n'] == 200 and abs(R['g1']['sim_room_median'] - SIM_ROOM_SPEC_NM) < 0.005 and abs(R['g1b']['unweighted'] - 0.82) < 0.005 else 'NO'} |",
          f"| preferred means t1..t4 (%ST) | 95.3, 20.0, 44.2, 68.1 | "
          f"{', '.join(f'{v:.2f}' for v in R['pref_means'])} | "
          f"{'yes' if np.allclose(R['pref_means'], [95.27, 20.05, 44.23, 68.14], atol=0.01) else 'NO'} |",
          ""]
    L += [f"**Amplitude matching.** κ = median RMS of the 200 simulated population ideals "
          f"({np.median(R['sim_rms']):.3f} Nm; 5–95 %: {hazen(R['sim_rms'], 5):.2f}–"
          f"{hazen(R['sim_rms'], 95):.2f}) / median RMS of the 11 analytic preferred profiles "
          f"({np.median(R['real_rms']):.3f} Nm; all 11 within "
          f"{R['real_rms'].min():.3f}–{R['real_rms'].max():.3f}, i.e. 12·√(3/8) = 7.35 Nm by "
          f"construction) = **κ = {k:.3f}**. κ is set entirely by the simulated ideals. "
          f"The mean simulated ideal and the mean real profile correlate at r = {R['sign_corr']:.2f} "
          "(sign conventions agree). Amplitude matching assumes Weber-like scaling (spec §2): a "
          "perceptible difference proportional to torque level. Under that assumption the "
          f"wearer's JND on the 12 Nm device is true_jnd(s)/κ ({JND[1.0] / k:.2f} Nm at s = 1), and "
          "the ×κ columns are the comparable ones. The JND columns of the unscaled numbers instead "
          "assume that the JND in Nm is the same at 12 Nm as at the simulated torque level (an "
          "absolute JND).", "",
          f"JNDs: true_jnd(0.5) = {JND[0.5]:.3f}, true_jnd(1.0) = {JND[1.0]:.3f}, "
          f"true_jnd(2.0) = {JND[2.0]:.3f} Nm.", ""]

    # ------------------------------------------------------------ G1
    def jnd_cells(x):
        return " / ".join(f"{x / JND[s]:.1f}" for s in S_LEVELS)

    rows = [("(a) to the study's starting profile (95, 25, 50, 70 %ST)", g1["d_start"]),
            ("(b) to the leave-one-out mean of the other ten profiles", g1["d_loo"]),
            ("sensitivity, not registered: to the profile at the other ten's mean timings",
             g1["loo_timing"])]
    L += ["## G1. Real room for personalisation", "",
          "RMS distance (101-point grid) of each participant's analytic preferred profile. "
          "Median [range]; JND columns at s = 0.5 / 1.0 / 2.0.", "",
          "| distance | unscaled, Nm | in JND | amplitude-matched (×κ), Nm | in JND |",
          "|---|---|---|---|---|"]
    for name, v in rows:
        m = float(np.median(v))
        L.append(f"| {name} | {m:.2f} [{v.min():.2f}, {v.max():.2f}] | {jnd_cells(m)} | "
                 f"{k * m:.2f} [{k * v.min():.2f}, {k * v.max():.2f}] | {jnd_cells(k * m)} |")
    sq = g1["sim_room_q"]
    L += ["", f"Simulated room d_prior − d_floor (e-main, ProMP, A, 200 users): median "
          f"{g1['sim_room_median']:.2f} Nm ({jnd_cells(g1['sim_room_median'])} JND); "
          f"5/10/25/75/90/95 % quantiles {sq[0]:.2f}, {sq[1]:.2f}, {sq[2]:.2f}, {sq[4]:.2f}, "
          f"{sq[5]:.2f}, {sq[6]:.2f} Nm. Share of simulated users whose room is at most the "
          f"largest real amplitude-matched room (b): {g1['share_sim_below_real_max']:.2f}.", "",
          "Per participant (Nm, unscaled; ×κ for amplitude-matched):", "",
          "| id | (a) start | (b) LOO mean | (b) ×κ |", "|---|---|---|---|"]
    for i, f in enumerate(fids):
        L.append(f"| {PSEUDONYM[f]} | {g1['d_start'][i]:.2f} | {g1['d_loo'][i]:.2f} | "
                 f"{k * g1['d_loo'][i]:.2f} |")
    L += ["", f"**Verdict: {g1['verdict']}.** Amplitude-matched median (b) "
          f"{ci_str(g1['matched_median'], *g1['matched_median_ci'])} Nm against the simulated "
          f"median room {g1['sim_room_median']:.2f} Nm; difference "
          f"{ci_str(g1['diff'], *g1['diff_ci'])} Nm (bootstrap of participants and of simulated "
          "users within folds, κ recomputed per draw). The real family's floor is zero (each "
          "preference is itself a four-half-cosine profile), so (b) is the room with nothing to "
          "subtract.", "",
          _promp_room_line(R), "",
          "**Meaning for the paper.** " + _g1_meaning(R), ""]

    # ------------------------------------------------------------ G1b
    L += ["## G1b. Reliability at the real room", "",
          "A's Q≥1 at k = 24 (s = 1.0, e-main main cell, ProMP, 200 population users), with the "
          "simulated users reweighted so that their room d_prior − d_floor follows the real "
          "amplitude-matched room (b).", "",
          "| method | Q≥1 | effective sample size | 95 % interval |", "|---|---|---|---|",
          f"| unweighted (reproduces the headline) | {g1b['unweighted']:.3f} | 200 | — |",
          f"| KDE ratio f_real(r)/f_sim(r), Scott bandwidths (real {g1b['kde_bw_real']:.3f} Nm, "
          f"sim {g1b['kde_bw_sim']:.3f} Nm) | {g1b['kde']:.3f} | {g1b['kde_ess']:.1f} "
          f"(bootstrap 2.5–97.5 %: {g1b['ess_sim_range'][0]:.1f}–{g1b['ess_sim_range'][1]:.1f}) | "
          f"simulated users: [{g1b['ci_sim'][0]:.3f}, {g1b['ci_sim'][1]:.3f}]; participants: "
          f"[{g1b['ci_part'][0]:.3f}, {g1b['ci_part'][1]:.3f}]; both: "
          f"[{g1b['ci_joint'][0]:.3f}, {g1b['ci_joint'][1]:.3f}] |",
          f"| quantile bins (real quartiles; edges {', '.join(f'{e:.2f}' for e in g1b['bin_edges'])} Nm; "
          f"simulated users per bin {', '.join(str(c) for c in g1b['bin_counts'])}; real mass with "
          f"no simulated user {g1b['bin_lost']:.2f}) | {fmt(g1b['bin'], 3)} | {fmt(g1b['bin_ess'], 1)} | — |",
          f"| raw: simulated users with room ≤ the real median ({np.median(R['kappa'] * g1['d_loo']):.2f} Nm) | "
          f"{fmt(g1b['raw'], 3)} | {g1b['raw_n']} users | — |", ""]
    sens = R["g1b_sens"]
    band_txt = "; ".join(
        (f"below {b['hi']:.2f}" if b["lo"] == 0.0 else
         f"above {b['lo']:.2f}" if not np.isfinite(b["hi"]) else f"{b['lo']:.2f}–{b['hi']:.2f}")
        + f": {fmt(b['q1'])} (n = {b['n']}; real rooms {b['n_real']})" for b in sens["bands"])
    knn = sens["knn"]
    am = [g1b["kde"], sens["promp"], sens["weber"]]
    L += [f"*Q≥1 by simulated room, not registered* (Nm, room d_prior − d_floor): {band_txt}. "
          f"Below one JND ({JND[1.0]:.3f} Nm): {fmt(sens['below_jnd_q1'])} (n = "
          f"{sens['below_jnd_n']}).", "",
          f"*Support.* The real rooms ({sens['support'][0]:.2f}–{sens['support'][1]:.2f} Nm ×κ) lie "
          "inside the simulated range, but their lowest quarter is matched by few simulated users "
          f"({g1b['bin_counts'][0]} in the first quantile bin, which carries "
          f"{round(sens['bin0_real_share'] * 11)}/11 of the weight). The KDE spreads "
          f"{100 * sens['mass_below_min']:.0f} % of the real density below the smallest real room "
          f"({100 * sens['mass_below_jnd']:.0f} % below one JND), onto {sens['sim_below_min_n']} "
          f"simulated users ({100 * sens['sim_below_min_weight']:.0f} % of the normalised weight; "
          f"Q≥1 {fmt(sens['sim_below_min_q1'])}), and {100 * sens['mass_above_max']:.0f} % above "
          f"the largest. Restricted to the real support the share is {sens['restricted']:.2f} (ESS "
          f"{sens['restricted_ess']:.0f}), and with {min(KNN_K)}–{max(KNN_K)} nearest neighbours "
          f"per participant it is {knn.min():.2f}–{knn.max():.2f}.", "",
          "*Sensitivity, not registered.* The share moves with choices the spec did not fix. "
          "(i) Room definition: with the target measured as the simulation measures its own room "
          "(the fold ProMP prior mean minus the ProMP floor; G1 context row, median "
          f"{sens['promp_target_median']:.2f} Nm; 11 participants × 5 folds pooled), the "
          f"reweighted share is {ci_str(sens['promp'], *sens['promp_ci'])} with participants "
          f"resampled (κ fixed; ESS {sens['promp_ess']:.0f}). (ii) Weber form: with the 5e spec's "
          f"own Weber observer (JND ∝ RMS + {WEBER_A_NM:.0f} Nm, E4), κ = "
          f"{sens['kappa_weber']:.3f} and the share is {sens['weber']:.2f} (ESS "
          f"{sens['weber_ess']:.0f}). (iii) With an absolute JND (the device's Nm, unscaled) the "
          f"share is {sens['absolute']:.2f} (ESS {sens['absolute_ess']:.0f}). The verdict stands "
          f"on the registered definition. The paper quotes {g1b['kde']:.2f} only together with "
          f"the range {min(am):.2f}–{max(am):.2f} under amplitude matching, and "
          f"{sens['absolute']:.2f} under an absolute JND.", "",
          f"**Verdict: {g1b['verdict']}** over simulated users "
          f"[{g1b['ci_sim'][0]:.2f}, {g1b['ci_sim'][1]:.2f}]; with the 11 participants also "
          f"resampled [{g1b['ci_joint'][0]:.2f}, {g1b['ci_joint'][1]:.2f}]: "
          f"{g1b['verdict_joint']}. The spec registers only the range; the three-way interval rule "
          "is the project's verdict form (NOTES item 8). Users whose room is below one JND "
          f"({JND[1.0]:.3f} Nm) cannot reach Q≥1, because the gain cannot exceed the room.", "",
          "**Meaning for the paper.** " + _g1b_meaning(R), ""]

    # ------------------------------------------------------------ G2
    sn, ln = g2["step_nm"], g2["late_nm"]
    L += ["## G2. Self-chosen step size as a sensitivity proxy", "",
          "Per participant (10 with a tuning log; P7 has none): the median step per timing "
          "(collapsed rows), converted to the RMS change of that participant's analytic preferred "
          "profile when that one landmark moves by the step (mean of +step and −step), then "
          "averaged over the four timings. The same for the late-exploration SD (last third of "
          "tuning time).", "",
          "| id | median step t1..t4 (%ST) | → RMS change (Nm) | late SD t1..t4 (%ST) | → RMS change (Nm) |",
          "|---|---|---|---|---|"]
    for i, f in enumerate(R["gate_tuning"]["fids"]):
        st = R["gate_tuning"]["median_step"][i]
        sd = R["gate_tuning"]["late_sd"][i]
        L.append(f"| {PSEUDONYM[f]} | {', '.join(fmt(v, 2) for v in st)} | {sn[f]['mean']:.2f} | "
                 f"{', '.join(fmt(v, 1) for v in sd)} | {ln[f]['mean']:.2f} |")
    L += ["", "| quantity | unscaled, Nm | in JND (s = 0.5 / 1 / 2) | amplitude-matched, Nm | in JND |",
          "|---|---|---|---|---|",
          f"| median step | {ci_str(g2['step_med'], *g2['step_med_ci'])} | {jnd_cells(g2['step_med'])} | "
          f"{ci_str(g2['step_med_matched'], *g2['step_med_matched_ci'])} | {jnd_cells(g2['step_med_matched'])} |",
          f"| late-exploration SD | {ci_str(g2['late_med'], *g2['late_med_ci'])} | {jnd_cells(g2['late_med'])} | "
          f"{ci_str(g2['late_med_matched'], *g2['late_med_matched_ci'])} | {jnd_cells(g2['late_med_matched'])} |",
          "", f"Per timing, median over participants of the RMS change (Nm, unscaled): step "
          f"{', '.join(fmt(v, 2) for v in g2['per_timing_step'])}; late SD "
          f"{', '.join(fmt(v, 2) for v in g2['per_timing_late'])} (t1..t4).", "",
          f"**Verdict.** Amplitude-matched median step within 0.34–1.35 Nm: "
          f"**{g2['verdict_matched']}**. Unscaled median step above 0.674 Nm: "
          f"**{g2['verdict_unscaled']}**. Intervals: bootstrap over the 10 participants (and κ). "
          "The two G2 verdicts read one measurement under two scalings; they are not independent "
          "confirmations. The unscaled verdict compares a step on the 12 Nm device with the "
          "simulation's JND and is meaningful only under an absolute JND. Under Weber scaling the "
          f"wearer's JND on the device is {JND[1.0] / k:.2f} Nm at s = 1, and the step is "
          f"{g2['step_med'] / (JND[1.0] / k):.1f} JND either way.", "",
          "**Meaning for the paper.** "
          + _g2_meaning(g2), ""]

    # ------------------------------------------------------------ G3
    L += ["## G3. Device variation", "",
          "Preferred condition (the offset-0 section of each of the four sessions), per "
          "participant and leg; the cache already drops 10 strides after each condition change. "
          "Summaries over the 22 participant-legs; intervals bootstrap the 11 participants "
          "(both legs together).", "",
          "| quantity | median | mean ± SD | range | 95 % interval of the median | ×κ (median) |",
          "|---|---|---|---|---|---|"]
    names = [("a", "(a) systematic: mean measured vs analytic (1000-point grid)"),
             ("b_pooled", "(b) stride to stride, about the pooled condition mean"),
             ("b_session", "(b′) stride to stride, about each session's own mean"),
             ("c_within", "(c) presentation level: first vs second half of each session's strides (primary)"),
             ("c_pooled", "(c′) first vs second half of the pooled strides (sessions 02+03 vs 04+05)"),
             ("c_between", "(c″) between sessions: RMS over the 6 session pairs"),
             ("block20", "sensitivity: consecutive 20-stride blocks (≈ 22 s, Schäfer's 18.7 s per setting)"),
             ("block45", "sensitivity: consecutive 45-stride blocks (≈ 50 s, the paper's 45–60 s)")]
    for key, name in names:
        v = g3[key]
        L.append(f"| {name} | {v['median']:.3f} | {v['mean']:.3f} ± {v['sd']:.3f} | "
                 f"{v['min']:.3f}–{v['max']:.3f} | [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | "
                 f"{k * v['median']:.3f} |")
    L += ["", "Per participant (L / R, Nm):", "",
          "| id | strides L/R | (a) | (b) | (c) | (c′) pooled halves | (c″) between sessions |",
          "|---|---|---|---|---|---|---|"]
    dev = g3["dev"]
    for f in fids:
        d = dev[f]
        L.append(f"| {PSEUDONYM[f]} | {d['L']['n_strides']}/{d['R']['n_strides']} | "
                 f"{d['L']['a']:.2f}/{d['R']['a']:.2f} | {d['L']['b_pooled']:.2f}/{d['R']['b_pooled']:.2f} | "
                 f"{d['L']['c_within']:.3f}/{d['R']['c_within']:.3f} | "
                 f"{d['L']['c_pooled']:.3f}/{d['R']['c_pooled']:.3f} | "
                 f"{d['L']['c_between']:.3f}/{d['R']['c_between']:.3f} |")
    c_med = g3["c_within"]["median"]
    L += ["", f"**Verdict.** (c) median below 0.6 Nm: **{g3['verdict_c']}** "
          f"({ci_str(c_med, *g3['c_within']['ci'], d=3)} Nm). (b) above (c): **{g3['verdict_bc']}** "
          f"(median of the per-leg differences (b) − (c) "
          f"{ci_str(g3['b_minus_c'], *g3['b_minus_c_ci'], d=3)} Nm; "
          f"(b) > (c) in {g3['b_gt_c_legs']} of 22 participant-legs). (a) on the preferred condition "
          f"alone is {g3['a']['mean']:.2f} ± {g3['a']['sd']:.2f} Nm (mean ± SD over 22 legs); the "
          "0.58 ± 0.10 Nm of the existing report pools every condition and is reproduced in §0.",
          "", f"The spec's literal reading, the first vs second half of the pooled preferred "
          f"strides (c′), gives {ci_str(g3['c_pooled']['median'], *g3['c_pooled']['ci'], d=3)} Nm; "
          "the within-session halves are used as primary because the pooled session order is not "
          f"known to be chronological (NOTES 30). (c′) below 0.6 Nm: "
          f"**{verdict_upper(g3['c_pooled']['median'], *g3['c_pooled']['ci'], NOISE_CELL_NM)}**; "
          f"(b) > (c′) in {g3['b_gt_cpooled_legs']} of 22 legs "
          f"(**{verdict_lower(g3['b_minus_cpooled'], *g3['b_minus_cpooled_ci'], 0.0)}**).", "",
          f"Like with like: the simulation's noise cell perturbs every presented curve with "
          f"independent smooth noise of RMS 0.6 Nm, so two presentations of one setting differ by "
          f"0.6·√2 = {0.6 * np.sqrt(2):.2f} Nm. The data's per-presentation equivalent is (c)/√2 = "
          f"{c_med / np.sqrt(2):.3f} Nm (primary), {g3['block45']['median'] / np.sqrt(2):.3f} Nm for "
          f"45-stride presentations, {g3['c_between']['median'] / np.sqrt(2):.3f} Nm between sessions.", "",
          "**Meaning for the paper.** " + _g3_meaning(g3), ""]

    # ------------------------------------------------------------ G4
    L += ["## G4. Coverage of real preferences by A's search region", "",
          f"Each amplitude-matched analytic preferred profile (×κ; largest |torque| "
          f"{g4['max_abs_matched_nm']:.2f} Nm, so no device clip is involved) is projected by least "
          "squares onto the fold's ProMP basis and whitened with the fold's prior "
          f"(`priors/<fold>-promp_s0`). Tight arm radius {g4['tight']:.3f} (= the e-main rows' "
          "arm_radius in every fold).", "",
          "| id | " + " | ".join(f.replace("loso-", "") for f in FOLDS) + " | residual RMS, Nm |",
          "|---|" + "---|" * len(FOLDS) + "---|"]
    for i, f in enumerate(fids):
        radii = [g4["per_fold"][fold]["radius"][i] for fold in FOLDS]
        res = [g4["per_fold"][fold]["resid"][i] for fold in FOLDS]
        res_txt = f"{min(res):.2f}" if max(res) - min(res) < 5e-3 else f"{min(res):.2f}–{max(res):.2f}"
        L.append(f"| {PSEUDONYM[f]} | " + " | ".join(
            (f"**{r:.2f}**" if r > g4["tight"] else f"{r:.2f}") for r in radii) + f" | {res_txt} |")
    L.append("| inside / 11 | " + " | ".join(str(g4["per_fold"][f]["inside"]) for f in FOLDS) + " | |")
    L.append("| simulated population users inside (stored floor radius), share of 40 | " + " | ".join(
        f"{g4['per_fold'][f]['sim_inside']:.2f}" for f in FOLDS) + " | |")
    L.append("| context, not registered: unscaled profiles (12 Nm peaks) inside / 11 | " + " | ".join(
        str(g4["per_fold"][f]["unscaled_inside"]) for f in FOLDS) + " | |")
    allres = np.concatenate([g4["per_fold"][f]["resid"] for f in FOLDS])
    allrad = np.concatenate([g4["per_fold"][f]["radius"] for f in FOLDS])
    fl_r = np.concatenate([g4["per_fold"][f]["floor_radius"] for f in FOLDS])
    fl_nm = np.concatenate([g4["per_fold"][f]["floor_nm"] for f in FOLDS])
    uns = np.concatenate([g4["per_fold"][f]["unscaled"] for f in FOLDS])
    s_st = np.concatenate([g4["per_fold"][f]["sim_stored"] for f in FOLDS])
    s_lsq = np.concatenate([g4["per_fold"][f]["sim_lsq"] for f in FOLDS])
    s_touch = np.concatenate([g4["per_fold"][f]["sim_touch"] for f in FOLDS])
    s_diff = np.abs(s_lsq - s_st)
    free = ~s_touch
    s_max = np.concatenate([g4["per_fold"][f]["sim_maxabs"] for f in FOLDS])
    odd = free & (s_diff > 0.01)
    gap = float(np.max(TORQUE_LIMIT_NM - s_max[odd])) if odd.any() else float("nan")
    sim_floor_med = float(np.median(R["sim"]["d_floor"]))
    L += ["", "Bold: outside the tight radius. The least-squares residual depends on the basis "
          "only, not on the fold's prior, so it is one value per participant.", "",
          f"- Residual RMS (the profile's optimum within the ProMP parameterization): median "
          f"{np.median(allres):.2f} Nm, range {allres.min():.2f}–{allres.max():.2f} "
          f"({np.median(allres) / JND[1.0]:.2f} JND at s = 1); the simulated users' median floor "
          f"d_floor is {sim_floor_med:.2f} Nm.",
          f"- Cross-check with prompref's tie-broken floor (`physics_floor`, device clip, 8 starts): "
          f"largest |floor − residual| {np.max(np.abs(fl_nm - allres)):.1e} Nm, largest |floor radius "
          f"− projection radius| {np.max(np.abs(fl_r - allrad)):.1e}.",
          f"- Pipeline check on the 200 simulated users: least-squares radius against the stored "
          f"floor radius. Of the {int(free.sum())} ideals that stay inside ±12 Nm, "
          f"{int(np.sum(s_diff[free] <= 0.01))} agree within 0.01 (median |difference| "
          f"{np.median(s_diff[free]):.1e}); the other {int(odd.sum())} peak within "
          f"{gap:.2f} Nm of the limit, where a clipped overshooting curve beats the least-squares fit, so "
          f"their floor is not the projection. The {int(s_touch.sum())} ideals that touch the limit "
          f"differ for the same reason (median |difference| "
          f"{(np.median(s_diff[s_touch]) if s_touch.any() else float('nan')):.2f}). The real "
          "profiles stay far from the limit, and the `physics_floor` cross-check above confirms "
          "that their floor is the projection.",
          f"- Unscaled (12 Nm peaks) for context: radius median {np.median(uns):.1f}, range "
          f"{uns.min():.1f}–{uns.max():.1f}; the largest projected weight of the matched profiles "
          f"is {max(g4['per_fold'][f]['wmax'] for f in FOLDS):.2f} Nm (box ±30).", "",
          f"**Verdict: {g4['verdict']}** (fewest inside in any fold: {g4['min_inside']} of 11).", "",
          "**Meaning for the paper.** " + _g4_meaning(g4), ""]

    L += _limitations(R)
    L += ["## Not computed", "",
          "Session time is not a computation (spec §3): Schäfer's participants self-tuned in "
          "10.9 ± 0.9 min (SE) over a median of 30.5 settings at 18.7 s each; the paper's 24 "
          "comparisons take 18–24 min at 45–60 s each.", "",
          "Judgement calls are listed in `scripts/schaefer_grounding_NOTES.md`.", "",
          "![Grounding](figures/grounding.png)", ""]
    return "\n".join(L)


def _promp_room_line(R: dict) -> str:
    """Context, not planned: the amplitude-matched real preferences' room in the ProMP
    family, d_prior (the fold prior's mean curve, as the sessions compute it) minus their
    ProMP floor (the G4 residual). It separates family from population in the G1 gap."""
    pf = R["g4"]["per_fold"]
    rooms = np.concatenate([v["room_promp"] for v in pf.values()])
    dpr = np.concatenate([v["d_prior_real"] for v in pf.values()])
    per_fold = ", ".join(f"{f.replace('loso-', '')} {np.median(v['room_promp']):.2f}"
                         for f, v in pf.items())
    check = max(v["d_prior_check"] for v in pf.values())
    return (f"*Context, not registered:* the same amplitude-matched preferences measured in the "
            f"simulation's own terms, d_prior (distance to the fold's ProMP prior-mean curve, "
            f"`prior_mean_distance` on the tight arm's design) minus their ProMP floor (the G4 "
            f"residual): median {np.median(rooms):.2f} Nm over 11 participants × 5 folds, range "
            f"{rooms.min():.2f}–{rooms.max():.2f} (fold medians {per_fold}); d_prior alone median "
            f"{np.median(dpr):.2f} Nm. Against the simulated median room "
            f"{R['g1']['sim_room_median']:.2f} Nm. (The same d_prior computation reproduces the "
            f"e-main rows' d_prior to {check:.1e} Nm on 5 simulated users per fold.)")


def _g1_meaning(R: dict) -> str:
    g1, sens = R["g1"], R["g1b_sens"]
    mm, sim = g1["matched_median"], g1["sim_room_median"]
    lo, hi = g1["matched_median_ci"]
    promp = sens["promp_target_median"]
    more = 100.0 * (1.0 - g1["share_sim_below_real_max"])
    held = g1["verdict"] == "held"
    text = (f"Scaled to the simulated torque level, real wearers' timing preferences lie a median "
            f"{mm:.2f} Nm [{lo:.2f}, {hi:.2f}] ({mm / JND[1.0]:.1f} JND at s = 1) from the mean of "
            f"the other participants' profiles, against the simulation's {sim:.2f} Nm, and "
            f"{more:.0f} % of simulated users have more room than any participant. The simulated "
            f"room is {'larger than' if held else 'not shown to be larger than'} the real "
            "timing-only room. ")
    text += (f"Measured as the simulation measures its own room (fold ProMP prior mean minus the "
             f"ProMP floor), the real room is {promp:.2f} Nm (median of the per-participant fold "
             f"means {sens['promp_fold_mean_median']:.2f} [{sens['promp_fold_mean_ci'][0]:.2f}, "
             f"{sens['promp_fold_mean_ci'][1]:.2f}]). ")
    if mm < promp < sim:
        text += (f"About {promp - mm:.2f} Nm of the {sim - mm:.2f} Nm gap therefore comes from how "
                 f"the room is defined, and {sim - promp:.2f} Nm remains. ")
    else:
        text += "On that definition the gap is " + f"{sim - promp:.2f} Nm. "
    return text + ("The real room is timing-only; the simulated room also contains magnitude "
                   "variation, which these data cannot test.")


def _g1b_where(g1b) -> str:
    """Where the reweighted share lies against the planned range (finding 1)."""
    est, (lo, hi) = g1b["kde"], g1b["ci_joint"]
    if G1B_LOW <= est <= G1B_HIGH:
        return f"inside the registered range {G1B_LOW:.2f}–{G1B_HIGH:.2f}"
    if est < G1B_LOW:
        return (f"below the registered range {G1B_LOW:.2f}–{G1B_HIGH:.2f}: the prediction failed "
                f"on its point estimate, and the interval "
                f"{'straddles' if hi >= G1B_LOW else 'lies below'} {G1B_LOW:.2f}")
    return (f"above the registered range {G1B_LOW:.2f}–{G1B_HIGH:.2f}: the prediction failed on "
            f"its point estimate, and the interval "
            f"{'straddles' if lo <= G1B_HIGH else 'lies above'} {G1B_HIGH:.2f}")


def _g1b_meaning(R: dict) -> str:
    g1b, sens = R["g1b"], R["g1b_sens"]
    if R["g1"]["verdict"] == "held":
        lead = ("G1 holds, so under the spec's rule the reweighted share replaces the unweighted "
                f"{g1b['unweighted']:.2f} as the reliability headline. ")
    else:
        lead = (f"G1 did not hold, so under the spec's rule the unweighted {g1b['unweighted']:.2f} "
                "stays the headline and the reweighted share is stated as found. ")
    lo, hi = g1b["ci_joint"]
    text = (lead + "With the simulated users reweighted to the timing-only room of the 11 real "
            f"participants, A reaches Q≥1 at k = {K_MAIN} in {g1b['kde']:.2f} of them (95 % "
            f"interval {lo:.2f}–{hi:.2f} with participants and simulated users resampled; ESS "
            f"{g1b['kde_ess']:.0f} of 200 simulated users). This is {_g1b_where(g1b)}. ")
    bands = sens["bands"]
    low = [b for b in bands if b["hi"] <= 1.5]
    high = [b for b in bands if b["lo"] >= 1.5]
    q_low = max(b["q1"] for b in low)
    q_high = min(b["q1"] for b in high)
    n_low = sum(b["n_real"] for b in low)
    n_high = sum(b["n_real"] for b in high)
    text += (f"In the simulation Q≥1 is close to a step function of the room: Q≥1 is at most "
             f"{q_low:.2f} in every band below 1.5 Nm and at least {q_high:.2f} above it, and the "
             f"11 real rooms split {n_low} below and {n_high} above 1.5 Nm. The share is therefore "
             "close to a count over 11 people, and it moves with the room's definition and the "
             "scaling (sensitivity above).")
    return text


def _g2_meaning(g2) -> str:
    m = g2["step_med_matched"]
    lo, hi = g2["step_med_matched_ci"]
    if m < G2_LOW:
        where = "below the sweep's JND range, so s = 0.5 is not the optimistic end"
    elif m > G2_HIGH:
        where = "above the sweep's JND range, so s = 2.0 is not the pessimistic end"
    else:
        where = f"inside the range of JNDs the sweep assumes ({G2_LOW:.2f}–{G2_HIGH:.2f} Nm)"
    late, (llo, lhi) = g2["late_med_matched"], g2["late_med_matched_ci"]
    if late < G2_LOW:
        late_where = f"below the sweep's most optimistic JND ({G2_LOW:.2f} Nm, s = 0.5)"
    elif late > G2_HIGH:
        late_where = f"above the sweep's most pessimistic JND ({G2_HIGH:.2f} Nm, s = 2.0)"
    else:
        late_where = "inside the sweep's JND range"
    n_zero = int(g2["late_zero"])
    return (f"Scaled to the simulated torque level, wearers' median self-chosen step changes the "
            f"profile by {m:.2f} Nm RMS [{lo:.2f}, {hi:.2f}], {where}; on the device it is "
            f"{g2['step_med']:.2f} Nm. Their late-exploration spread is "
            f"{'smaller' if late < m else 'larger'}: {late:.2f} Nm [{llo:.2f}, {lhi:.2f}] scaled, "
            f"{late_where}. No prediction was registered for it, and it partly counts timings that "
            f"participants had stopped moving ({n_zero} of {g2['late_n']} late SDs are 0). Neither "
            "number is a measured JND. A step is how far a participant chose to move in "
            "self-paced adjustment, which can be larger or smaller than what they can "
            "discriminate. The data are consistent with the sweep's JND range but do not test it.")


def _g3_meaning(g3) -> str:
    c = g3["c_within"]["median"]
    b45 = g3["block45"]["median"]
    cell_pair = NOISE_CELL_NM * np.sqrt(2.0)
    text = (f"Two 45-stride presentations of one setting on the real device differ by about "
            f"{b45:.2f} Nm RMS ({c:.2f} Nm between half-sessions of about 2 min), against "
            f"{cell_pair:.2f} Nm between two presentations in the {NOISE_CELL_NM} Nm noise cell. ")
    if c < NOISE_CELL_NM and b45 < cell_pair:
        text += ("That cell therefore overstates random variation between presentations and is a "
                 "conservative stress test for it. ")
    else:
        text += "That cell is therefore the realistic case for random variation, not a stress test. "
    return text + (f"The {NOISE_CELL_NM} Nm itself is a tracking-error figure (Franks 2021; 5e spec "
                   f"E5). On this device the mean delivered torque deviates from the intended "
                   f"analytic profile by a similar {g3['a']['median']:.2f} Nm RMS (median over "
                   "legs, preferred condition), but that deviation repeats from one presentation "
                   "to the next. The simulation does not model a repeatable deviation of this "
                   f"size. Stride-to-stride variation ({g3['b_pooled']['median']:.2f} Nm) is larger; "
                   "the presentation mean averages it out, but whether the wearer's judgement does "
                   "is not tested here.")


def _g4_meaning(g4) -> str:
    outside = {f.replace("loso-", ""): 11 - v["inside"] for f, v in g4["per_fold"].items()}
    worst = max(outside.values())
    uns = [v["unscaled_inside"] for v in g4["per_fold"].values()]
    resid = float(np.median(next(iter(g4["per_fold"].values()))["resid"]))
    if worst == 0:
        lead = ("With its amplitude matched to the simulated population (×κ), every real "
                "preference's best ProMP approximation lies inside A's main (tight) search region "
                "in every fold (11 of 11)")
    else:
        lead = (f"With its amplitude matched to the simulated population (×κ), up to {worst} of 11 "
                "real preferences' best ProMP approximations lie outside A's main search region, "
                "depending on the fold (" + ", ".join(f"{k}: {v}" for k, v in outside.items())
                + "); the paper states how many A's main region would not reach")
    return (lead + f"; a residual of {resid:.2f} Nm RMS (median) stays out of reach. At the "
            f"device's own 12 Nm peaks, {min(uns)}–{max(uns)} of 11 lie inside depending on the "
            "fold, so coverage depends on amplitude, which these data do not inform. The priors "
            "were generated with timing spread fitted to these participants, so this check tests "
            "the change of profile family more than the timing spread.")


def _limitations(R: dict) -> list[str]:
    k = R["kappa"]
    g4 = R["g4"]
    uns = [v["unscaled_inside"] for v in g4["per_fold"].values()]
    late = R["gate_tuning"]["late_rms_sd"]
    n_g2 = len(R["gate_tuning"]["fids"])
    return [
        "## Limitations", "",
        "1. **Timing only.** Schäfer's profiles are four half-cosines with peaks fixed at ±12 Nm. "
        "They measure timing preferences only, and magnitude preferences remain unmeasured, so "
        "the real room in G1 is a timing-only room.",
        f"2. **Amplitude matching.** Every ×κ number rests on κ = {k:.3f} and on Weber-like "
        "scaling, under which a perceptible difference is proportional to torque level (spec "
        "§2). G1b, G2 and G4 move with that assumption (see the G1b sensitivity; for G4, "
        f"{min(uns)}–{max(uns)} of 11 lie inside the tight radius when unscaled).",
        f"3. **Sample size.** n = 11 participants ({n_g2} for G2: P7 has no log). "
        "Percentile-bootstrap intervals of a median at this n are coarse and may undercover. In "
        "G1b most of the uncertainty comes from the participants.",
        "4. **Self-tuning is not pairwise comparison.** Participants adjusted the timings "
        "themselves at about 19 s per setting. A self-tuned endpoint is not the optimum a "
        "pairwise-comparison procedure would find, and a self-chosen step is not a "
        "discrimination threshold.",
        "5. **Stopping noise.** Each preference is a single self-tuning endpoint with no repeat "
        f"session. Late exploration still wandered by {late.min():.1f}–{late.max():.1f} %ST RMS "
        "(out/schaefer/REPORT.md §3.7). Part of the between-participant spread in G1 (b) is "
        "therefore stopping noise rather than preference, so (b) probably overstates the true "
        "timing room. That would strengthen G1 and lower G1b.",
        "6. **Not independent of the simulation.** The simulated population's personal timing "
        "shift (t₁₀, centre 137.6 ms, scale 31.6 ms) and residual landmark spread were fitted to "
        "these same 11 participants (Phase 5 spec §3.5a); the curves the ProMP priors were "
        "fitted to (20,000 per fold, §3.3) were generated with those timing factors on "
        "biological base curves (§3.2). G1 and G4 are therefore consistency checks of the room "
        "and of the change of family, not an independent test of the timing population.", ""]


# ============================================================================ entry
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the project's out/ directory")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="the Schaefer share")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    t0 = time.time()
    R = analyse(out_root, Path(args.data), args.boot)
    dest = out_root / "schaefer_grounding"
    dest.mkdir(parents=True, exist_ok=True)
    make_figure(R, dest / "figures" / "grounding.png")
    (dest / "REPORT.md").write_text(build_report(R), encoding="utf-8")
    print(f"wrote {dest / 'REPORT.md'} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
