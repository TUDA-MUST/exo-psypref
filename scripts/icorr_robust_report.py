"""ICORR robustness tests H, O and R: the planned verdicts PH1, PH2, PO1, PO2, PR1 and PR2.

"Home-advantage test at matched decisiveness" (PH1, PH2), "Out-of-generator and random-utility
wearers" (PO1, PO2, PR1, PR2) and "Decision on the robustness tests" (per-bout OOD peaks as the
primary O analysis with the common peak as a labelled sensitivity run, the PH2 scale, u_max as an
upper bound, population users only). Build and row fields: out/icorr/robust_build_report.md
(section 6 for e-ood-pb and the u_max convergence flag). No session is run.

Usage:
    python scripts/icorr_robust_report.py [--out <project>/out]
        [--boot 10000] [--report <out>/icorr/robust_report.md]

Reads, all read-only, each jsonl streamed line by line, population users only
(``user_kind == "population"`` and a ``/pop`` id, as ruled; for O, ``/ood`` ids, the only users
the O blocks hold):
  OUT/icorr/e-home.jsonl       the static (10 %) and walking (25 %) landmark wearers at matched
                               decisiveness: A, G, LineCoSpar, SortCMA on ProMP; A with Franks
                               projection (``a_lens``) and G on Franks;
  OUT/icorr/e-ood-pb.jsonl     the out-of-generator ideals, per-bout peaks (the primary O set):
                               A (EIG duels), A (random duels), G, LineCoSpar, SortCMA on ProMP,
                               plain RMS wearer at s = 1.0;
  OUT/icorr/e-ood.jsonl        the same cells on the common-peak ideals (sensitivity run);
  OUT/icorr/e-rand.jsonl       the random smooth-utility wearer: A (EIG duels), A (random duels),
                               G, LineCoSpar, SortCMA on ProMP;
  OUT/phase5e/e-main.jsonl     A, G, LineCoSpar and SortCMA on ProMP, main condition (PH2's and
                               PO2's RMS reference leads);
  OUT/phase5e/e-observer.jsonl the unmatched landmark cells (context); the wearer's own distance
                               of generic assistance is rebuilt per row as icorr_jnd_free does;
  OUT/phase5-schaefer/ood-pb/users/<fold>.json and ood/users/<fold>.json   the O users: their
                               order, answer seeds, ideals and floors (checks);
  OUT/phase5-schaefer/random_utility/<fold>.json   u_gen, u_max and ``umax_converged`` per user.
                               Read once at the start (another process may refine u_max); their
                               timestamps and hashes are recorded in the report and re-checked at
                               the end.

Estimators (phase5e_report's, as icorr_k_report uses them for its verdicts):
  medians       ``er._med``: inverted_cdf, the lower middle value;
  contrasts     the difference of two medians (distances) or of two means (win rates) on the
                same users;
  intervals     ``er.joint_bootstrap``: users resampled within folds, one draw set for every array
                of a statistic, and the same draws for every statistic (same seed, same 200 users);
                95 % ends are observed bootstrap values;
  verdicts      held when the interval lies wholly on the predicted side of the threshold, failed
                when wholly on the other side, uncertain otherwise (``er._side``).
  PO2's D       the O users are not the main condition's users, so its two halves are resampled
                independently: the O cells with the draws every other statistic uses (seed
                ``er.SEED``), the e-main cells with their own fold-stratified draws
                (``MAIN_DRAW_SEED``); D is formed draw by draw (``two_sample``).

Win rate (no JND needed): P_i(k) = lambda/2 + (1 - lambda) Phi((g_i - e_i(k)) / s), s = 1.0,
lambda = 0.1, with g_i = ``own_generic_distance`` and e_i(k) = ``own_distances[k - 1]`` (minus the
utility, on the scale the wearer answers by). For the RMS wearers (main condition, O) g_i is
``d_prior`` and e_i(k) is ``distances[k - 1]``. The win rate of an optimizer is the mean over users.
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import icorr_jnd_free as jf  # noqa: E402
import icorr_k_report as kr  # noqa: E402
import phase5e_report as er  # noqa: E402
import schaefer_grounding as G  # noqa: E402
import short_budget_report as sb  # noqa: E402

from prompref.experiment import phase5  # noqa: E402
from prompref.experiment.icorr_grid import DECISIVENESS  # noqa: E402
from prompref.experiment.metrics import prior_mean_curve  # noqa: E402
from prompref.experiment.robust import load_ood_users, ood_root  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.oracle import rms_distance  # noqa: E402

K_MAIN = 24
KS = (12, 24, 40)
KMAX = 40
N_BOOT = 10_000
RETAINED = 2.0 / 3.0          # PH2
PO1_LEVEL = 0.70              # PO1
PO2_FRACTION = 0.5            # PO2: at least half the main-condition advantage
PR2_LEVEL = 0.60              # PR2
FIRST_K_LEVELS = (0.60, 0.75)
JND1 = true_jnd(1.0)
#: PO2's D: the e-main half is resampled with its own draws (the O users are other users).
MAIN_DRAW_SEED = er.SEED + 1
#: Checks, not results: the main-condition lead the task names, the
#: unmatched landmark win rates at k = 24 printed in jnd_free.md (simulated room = plain mean),
#: and the number of users whose u_max converged (robust_build_report.md section 6.2).
MAIN_LEAD_EXPECTED = "-0.806"
REPRO_OBSERVER = {("static", "a"): "0.86", ("static", "g"): "0.79",
                  ("walking", "a"): "0.77", ("walking", "g"): "0.68"}
CONVERGED_EXPECTED = "160"
#: robust_ood_pb.json's median ideal RMS sizes, the keys of the three user sets (a check).
RMS_KEYS = {"ood-pb": "ood_pb_rms_median", "ood": "ood_common_rms_median",
            "pop": "population_rms_median"}
#: Context for the O room: A's win rate at k = 24 at
#: the literature room read as a median, in-generator users KDE-reweighted to that room.
MEDIAN_ROOM_NM, MEDIAN_ROOM_WIN = "1.465", "0.72"

#: The two landmark cells: name -> (landmark_mag_fraction, decisiveness ratio).
LANDMARK = {"static": (0.10, DECISIVENESS[None]), "walking": (0.25, DECISIVENESS[0.25])}
LANDMARK_TITLE = {"static": "static landmark (10 %)", "walking": "walking landmark (25 %)"}
#: H optimizers: key -> (family, learner, label).
H_OPTS = {"a": ("promp_s0", "a", "A"), "g": ("promp_s0", "g", "G"),
          "d_z": ("promp_s0", "d_z", "LineCoSpar"), "cmaes": ("promp_s0", "cmaes", "SortCMA"),
          "a_lens": ("franks", "a_lens", "A with Franks projection"),
          "g_franks": ("franks", "g", "G on Franks")}
H_PAIRS = {(fam, lr) for fam, lr, _ in H_OPTS.values()}
#: H contrasts (context): first minus second.
H_CONTRASTS = (("a", "g"), ("a", "d_z"), ("a", "cmaes"), ("a_lens", "g_franks"))
#: R optimizers: key -> (learner, duel mode, label).
R_OPTS = {"a": ("a", "eig", "A"), "g": ("g", "eig", "G"), "d_z": ("d_z", "eig", "LineCoSpar"),
          "cmaes": ("cmaes", "eig", "SortCMA"), "a_rand": ("a", "random", "A, random duels")}
R_COMPARATORS = ("g", "d_z", "cmaes", "a_rand")
#: O: the same five cells as R, on the out-of-generator users; the two user sets, the per-bout
#: one primary, the common-peak one a labelled sensitivity run.
O_OPTS = R_OPTS
O_SETS = {"ood-pb": "e-ood-pb", "ood": "e-ood"}
O_TITLE = {"ood-pb": "per-bout peaks (e-ood-pb, primary)",
           "ood": "common peak (e-ood, sensitivity)"}
#: PO2's comparators: the main condition's three; A with random duels is
#: A's own ablation, reported beside them and taken in only if it would lower the verdict.
O_COMPARATORS = ("g", "d_z", "cmaes")
O_COMPARATORS_ABLATION = O_COMPARATORS + ("a_rand",)
#: e-main cells read: A and the three comparators.
MAIN_LEARNERS = ("a", "g", "d_z", "cmaes")
MAIN_LABEL = {"a": "A", "g": "G", "d_z": "LineCoSpar", "cmaes": "SortCMA"}
#: Verdicts from least to most favourable to the prediction.
FAVOUR = ("failed", "uncertain", "held")

FIELDS = jf.KEEP + ("own_generic_distance", "decisiveness_ratio", "utility_scale", "duel_mode",
                    "n_basis", "jnd_scale", "user_set")
LISTS = jf.LISTS + ("own_feature_jnd",)


# ============================================================================ u_max sidecars
def read_sidecars(run: Path) -> dict:
    """Every ``random_utility/<fold>.json``, read once: per user u_gen, u_max, the search
    record and ``umax_converged`` (refused if absent: a sidecar from before the convergence run);
    per file its mtime, sha256 and the sidecar's own timestamps."""
    files, users = [], {}
    for path in sorted((run / "random_utility").glob("loso-*.json")):
        raw = path.read_bytes()
        st = path.stat()
        d = json.loads(raw.decode("utf-8"))
        files.append(dict(name=path.name, path=path, mtime=st.st_mtime,
                          mtime_iso=_dt.datetime.fromtimestamp(st.st_mtime).astimezone()
                          .isoformat(timespec="seconds"),
                          sha=hashlib.sha256(raw).hexdigest(), size=len(raw),
                          umax_when=d.get("umax_when"), refine_when=d.get("refine_when"),
                          converge_when=d.get("converge_when"), when=d.get("when")))
        for uid, u in d["users"].items():
            if not isinstance(u.get("umax_converged"), bool):
                raise SystemExit(f"{path.name}: {uid} has no umax_converged flag; run "
                                 "icorr_robust_prep.py --step converge first")
            users[uid] = dict(u_generic=float(u["u_generic"]), u_max=float(u["u_max"]),
                              n_at_best=int(u.get("umax_n_searches_at_best") or 0),
                              source=str(u.get("u_max_source")),
                              scale=float(u["scale"]), converged=bool(u["umax_converged"]))
    if len(files) != 5:
        raise SystemExit(f"expected 5 random-utility sidecars, found {len(files)}")
    return dict(files=files, users=users)


def sidecars_unchanged(side: dict) -> list[tuple[str, bool, str]]:
    out = []
    for f in side["files"]:
        raw = f["path"].read_bytes()
        same = hashlib.sha256(raw).hexdigest() == f["sha"]
        mt = _dt.datetime.fromtimestamp(f["path"].stat().st_mtime).astimezone()
        out.append((f["name"], same, mt.isoformat(timespec="seconds")))
    return out


# ============================================================================ the rows
def _slim(df):
    return None if df is None else kr.normalise(sb._n2048(sb._popn(df)))


def ood_row(r: dict) -> bool:
    """An O row of a population user (the O users are ``kind="population"`` with ``/ood`` ids;
    ``jf._pop_row`` and ``sb._popn`` would drop them for want of ``/pop``)."""
    return (not r.get("error") and r.get("user_kind") == "population"
            and "/ood" in str(r.get("user_id", "")))


def _slim_ood(df):
    return None if df is None else kr.normalise(sb._n2048(df))


def load(out: Path) -> dict:
    home = jf.stream(out / "icorr" / "e-home.jsonl", fields=FIELDS, lists=LISTS)
    rand = jf.stream(out / "icorr" / "e-rand.jsonl", fields=FIELDS, lists=LISTS)
    ood = {s: jf.stream(out / "icorr" / f"{block}.jsonl", keep=ood_row, fields=FIELDS,
                        lists=LISTS) for s, block in O_SETS.items()}
    main = jf.stream(out / "phase5e" / "e-main.jsonl",
                     keep=lambda r: (jf._pop_row(r) and r.get("family") == "promp_s0"
                                     and r.get("learner") in MAIN_LEARNERS),
                     fields=FIELDS, lists=LISTS)
    print("  e-observer landmark rows (own generic distance rebuilt per row) ...", flush=True)
    obs = jf.stream(out / "phase5e" / "e-observer.jsonl",
                    keep=lambda r: (jf._pop_row(r) and r.get("observer_metric") == "landmark"
                                    and (r.get("family"), r.get("learner")) in H_PAIRS),
                    extra=jf.own_generic_extra, fields=FIELDS, lists=LISTS)
    if any(x is None for x in (home, rand, main, obs, *ood.values())):
        raise SystemExit("a jsonl file is missing")
    obs = obs.copy()
    obs["own_generic_distance"] = obs["own_generic"].astype(float)
    return dict(home=_slim(home), rand=_slim(rand), main=_slim(main), obs=_slim(obs),
                **{s: _slim_ood(df) for s, df in ood.items()})


def ood_users(run: Path, folds=G.FOLDS) -> dict:
    """Per O user set: the users by id and their (fold, user_id) keys in fold order, then file
    order (the order the folds' users files list them), the magnitude rule and the median ideal
    size."""
    res = {}
    for s in O_SETS:
        users, keys, configs = {}, [], []
        for fold in folds:
            for uid, user in load_ood_users(str(run), fold, s).items():
                users[uid] = user
                keys.append((fold, uid))
            meta = json.loads((ood_root(str(run), s) / "users" / f"{fold}.json")
                              .read_text(encoding="utf-8"))
            configs.append({k: meta["config"].get(k) for k in
                            ("magnitude_rule", "median_peak_nm", "median_ext_peak_nm",
                             "median_flex_peak_nm", "magnitude_cv")})
        if any(cfg != configs[0] for cfg in configs):
            raise SystemExit(f"O user set {s}: the folds' magnitude rules differ")
        res[s] = dict(users=users, keys=keys, config=configs[0],
                      rms_median=median_rms([u.curve_nm for u in users.values()]))
    return res


def median_rms(curves) -> float:
    """The median RMS size of ideal curves (``rms_distance`` to zero torque), as
    out/icorr/robust_ood_pb.json reports it."""
    return float(np.median([rms_distance(np.asarray(x), np.zeros_like(np.asarray(x)))
                            for x in curves]))


def pick(df, family: str, learner: str, keys, fields: dict, duel: str = "eig") -> pd.DataFrame:
    """One cell (s = 1.0, lapse 0.1, data prior, tight arm, K = 8, phase5e_report's defaults
    overridden by ``fields``), in load_sim's 200-user order."""
    mask = ((df["family"] == family) & (df["learner"] == learner) & (df["duel_mode"] == duel)
            & (df["arm"] == "tight") & (df["prior"] == "data") & (df["n_basis"] == 8)).to_numpy()
    want = dict(er.DEFAULTS)
    want.update(s=1.0)
    want.update(fields)
    for name, value in want.items():
        mask &= er._match(df, name, value)
    if learner == "g":
        mask &= er._match(df, "jnd_scale", 1.0)
    out = df[mask]
    if len(out) != len(keys):
        raise ValueError(f"{family} {learner} {fields} {duel}: {len(out)} rows, "
                         f"expected {len(keys)}")
    return jf.in_order(out, keys)


def cells(rows: dict, keys, okeys: dict) -> dict:
    """Every cell, each in its users' order: ``keys`` (the 200 population users of load_sim) for
    H, R and the main condition, ``okeys[set]`` for the O cells."""
    c = {}
    for s in O_SETS:
        for key, (lr, duel, _) in O_OPTS.items():
            c[("ood", s, key)] = pick(rows[s], "promp_s0", lr, okeys[s], dict(user_set=s),
                                      duel=duel)
    for cell, (frac, ratio) in LANDMARK.items():
        for key, (fam, lr, _) in H_OPTS.items():
            c[("home", cell, key)] = pick(rows["home"], fam, lr, keys,
                                          dict(observer_metric="landmark",
                                               landmark_mag_fraction=frac,
                                               decisiveness_ratio=ratio))
            c[("obs", cell, key)] = pick(rows["obs"], fam, lr, keys,
                                         dict(observer_metric="landmark",
                                              landmark_mag_fraction=frac))
    for key, (lr, duel, _) in R_OPTS.items():
        c[("rand", key)] = pick(rows["rand"], "promp_s0", lr, keys,
                                dict(observer_metric="random_smooth"), duel=duel)
    for lr in MAIN_LEARNERS:
        c[("main", lr)] = pick(rows["main"], "promp_s0", lr, keys, {})
    return c


def lst(df, field: str) -> np.ndarray:
    m = np.array([v[:KMAX] for v in df[field]], dtype=float)
    if m.shape[1] < KMAX:
        raise ValueError(f"a row holds fewer than {KMAX} values of {field}")
    return m


def col(df, name: str) -> np.ndarray:
    return df[name].astype(float).to_numpy()


def own(df, k: int) -> np.ndarray:
    return lst(df, "own_distances")[:, k - 1]


def rms(df, k: int) -> np.ndarray:
    return lst(df, "distances")[:, k - 1]


def win(df, k: int) -> np.ndarray:
    """Per-user probability that the wearer prefers the identified assistance after k
    comparisons over generic assistance in one duel, in the wearer's own utility."""
    return jf.win_probability(col(df, "own_generic_distance"), own(df, k), col(df, "s"),
                              col(df, "lapse"))


def win_curve(df) -> np.ndarray:
    """(users, 40) win probabilities."""
    return jf.win_probability(col(df, "own_generic_distance")[:, None], lst(df, "own_distances"),
                              col(df, "s")[:, None], col(df, "lapse")[:, None])


def win_rms(df, k: int) -> np.ndarray:
    """``win`` for a plain RMS wearer (main condition, O): generic distance ``d_prior``,
    identified distance ``distances[k - 1]``."""
    return jf.win_probability(col(df, "d_prior"), rms(df, k), col(df, "s"), col(df, "lapse"))


# ============================================================================ statistics
med = kr.med


def mean(p):
    return p.mean(axis=1)


def diff_mean(x, y):
    return x.mean(axis=1) - y.mean(axis=1)


def draw_matrix(folds, n_boot: int, seed: int = er.SEED) -> np.ndarray:
    """The user draws er.joint_bootstrap makes, rebuilt (for statistics over all 40 k)."""
    folds = np.asarray(folds)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(folds == f) for f in pd.unique(folds)]
    return np.concatenate([g[rng.integers(0, len(g), size=(n_boot, len(g)))] for g in groups],
                          axis=1)


def first_k(P: np.ndarray, draws: np.ndarray, level: float) -> tuple:
    """The first k (1-40) at which the mean win rate meets ``level`` on the cross-sectional
    curve (as jnd_free.md section 5): estimate, 2.5 / 97.5 % order statistics over the draws,
    share of draws that reach it by k = 40."""
    point = float(jf.first_k(P.mean(axis=0), ">=", level)[0])
    curves = np.stack([P[:, k][draws].mean(axis=1) for k in range(P.shape[1])], axis=1)
    fk = jf.first_k(curves, ">=", level)
    lo, hi = jf.inverted_cdf_ci(fk)
    return point, lo, hi, float(np.mean(np.isfinite(fk)))


def side(ci: dict, threshold: float, want_low: bool) -> str:
    return kr.state(ci, threshold, want_low)


def adv_over_best(a, *others):
    """PO2's advantage: med(A) minus the smallest comparator median, the best comparator taken
    within each draw (as icorr_k_report.lead_over_best; it can only shrink A's advantage).
    Negative favours A, as every A − comparator contrast in this report."""
    return med(a) - np.min(np.stack([med(o) for o in others]), axis=0)


def best_counts(a_draws: list[np.ndarray], names) -> dict:
    """How often each comparator is the best (smallest median) within the draws: ``a_draws``
    holds one (draws, users) matrix per comparator."""
    which = np.argmin(np.stack([med(x) for x in a_draws]), axis=0)
    return {n: int(np.sum(which == i)) for i, n in enumerate(names)}


def two_sample(x_arrays, x_folds, x_stat, y_arrays, y_folds, y_stat, combine, n_boot: int,
               seeds=(er.SEED, MAIN_DRAW_SEED)) -> dict:
    """A statistic of two independent user samples, combine(x_stat(X), y_stat(Y)): each sample
    resampled within its folds with its own draws (the first with the seed every other
    statistic uses, so x_stat alone reproduces joint_bootstrap), combined draw by draw; the
    95 % ends are observed values (inverted_cdf), as joint_bootstrap takes them."""
    xs = [np.asarray(a, dtype=float) for a in x_arrays]
    ys = [np.asarray(a, dtype=float) for a in y_arrays]
    estimate = float(combine(x_stat(*[a[None, :] for a in xs]),
                             y_stat(*[a[None, :] for a in ys]))[0])
    dx, dy = draw_matrix(x_folds, n_boot, seeds[0]), draw_matrix(y_folds, n_boot, seeds[1])
    b = combine(x_stat(*[a[dx] for a in xs]), y_stat(*[a[dy] for a in ys]))
    lo, hi = np.percentile(b, [2.5, 97.5], method="inverted_cdf")
    return dict(estimate=estimate, lo=float(lo), hi=float(hi), n=int(len(xs[0])))


def po2_state(adv_main: dict, adv: dict, half: dict, half_threshold: float = 0.0) -> str:
    """PO2: A's advantage over the best comparator (A − best, negative favours A) excludes 0,
    and is at least half its main-condition size. ``half`` is the interval judged against
    ``half_threshold``: D = adv − ½·adv_main against 0, or
    adv itself against the point value ½·adv_main (the fixed-threshold reading). Held only if
    both parts hold and the main advantage excludes 0 (as PH2's gate); failed if either part
    lies wholly on the wrong side or A has no main-condition advantage; uncertain otherwise."""
    if adv_main["lo"] >= 0:
        return "failed"
    parts = [er._side(adv, 0.0, want_low=True), er._side(half, half_threshold, want_low=True)]
    state_ = kr.combine(parts)
    if state_ == "held" and not adv_main["hi"] < 0:
        return "uncertain"
    return state_


def least_favourable(*states: str) -> str:
    """The verdict least favourable to the prediction (failed < uncertain < held)."""
    return min(states, key=FAVOUR.index)


def share_of_gain(gain, u_max, u_gen) -> np.ndarray:
    """R's share of the achievable utility gain, (u_k − u_gen) / (u_max − u_gen), per user;
    ``gain`` = u_k − u_gen. u_max is a best-found lower bound, so the share is an upper bound."""
    return np.asarray(gain, dtype=float) / (np.asarray(u_max, dtype=float)
                                            - np.asarray(u_gen, dtype=float))


def ph2_state(lead_main: dict, D: dict) -> tuple[str, str]:
    """PH2: lead = med(A) - med(G) (negative favours A); D = lead_home - 2/3 lead_main, held
    when D's interval lies wholly at or below 0 (A keeps at least two thirds of its main lead).
    As QE3(b), the main lead must itself exclude 0."""
    if lead_main["lo"] >= 0:
        return "failed", "A has no lead over G in the main condition"
    if D["lo"] > 0:
        return "failed", "A keeps less than two thirds of its main-condition lead"
    if D["hi"] <= 0 and lead_main["hi"] < 0:
        return "held", ""
    return "uncertain", "an interval straddles its threshold"


# ============================================================================ formatting
def ci3(d: dict, digits: int = 3, signed: bool = True) -> str:
    """Estimate [lo, hi]; signed as short_budget_report._sg (a value rounding to 0 has no sign)."""
    g = (lambda x: sb._sg(x, digits)) if signed else (lambda x: f"{x:.{digits}f}")
    return f"{g(d['estimate'])} [{g(d['lo'])}, {g(d['hi'])}]"


def p3(d: dict) -> str:
    return ci3(d, 3, signed=False)


def fk(x) -> str:
    return "> 40" if not np.isfinite(x) else str(int(x))


def git_head() -> str:
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "scripts/icorr_robust_report.py"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f"{head}{' (script uncommitted)' if dirty else ''}, script sha256 {sha}"
    except (OSError, subprocess.CalledProcessError):
        return f"unknown, script sha256 {sha}"


# ============================================================================ checks
def checks(c: dict, side_: dict, ous: dict, run: Path) -> list[tuple[str, float, float]]:
    """(what, max |difference|, tolerance)."""
    res = []
    ref = c[("main", "a")]
    for cell, (_, ratio) in LANDMARK.items():
        for key in H_OPTS:
            h, o = c[("home", cell, key)], c[("obs", cell, key)]
            label = f"{LANDMARK_TITLE[cell]}, {H_OPTS[key][2]}"
            res.append((f"{label}: e-home own_generic_distance × ratio = e-observer's rebuilt "
                        "own generic distance",
                        float(np.max(np.abs(col(h, "own_generic_distance") * ratio
                                            - col(o, "own_generic_distance")))), 1e-9))
            res.append((f"{label}: e-home own_distances = own_feature_jnd × true_jnd(1.0) / ratio",
                        float(np.max(np.abs(lst(h, "own_distances")
                                            - lst(h, "own_feature_jnd") * JND1 / ratio))), 1e-9))
            res.append((f"{label}: e-home ratio column = {ratio!r}",
                        float(np.max(np.abs(col(h, "decisiveness_ratio") - ratio))), 0.0))
            for field in ("d_prior", "d_floor"):
                res.append((f"{label}: e-home {field} = e-observer's",
                            float(np.max(np.abs(col(h, field) - col(o, field)))), 1e-12))
            if H_OPTS[key][0] == "promp_s0":
                res.append((f"{label}: e-home d_prior = e-main A's",
                            float(np.max(np.abs(col(h, "d_prior") - col(ref, "d_prior")))), 1e-12))
            res.append((f"{label}: e-observer generic curve reproduces the stored d_prior",
                        float(o["generic_check"].astype(float).max()), 1e-9))
    for key in R_OPTS:
        r = c[("rand", key)]
        ug = np.array([side_["users"][u]["u_generic"] for u in r["user_id"]])
        res.append((f"random utility, {R_OPTS[key][2]}: own_generic_distance = −u_gen of the "
                    "sidecar (bit for bit)",
                    float(np.max(np.abs(col(r, "own_generic_distance") + ug))), 0.0))
        sc = np.array([side_["users"][u]["scale"] for u in r["user_id"]])
        res.append((f"random utility, {R_OPTS[key][2]}: utility_scale = the sidecar's scale",
                    float(np.max(np.abs(col(r, "utility_scale") - sc))), 0.0))
        res.append((f"random utility, {R_OPTS[key][2]}: d_prior = e-main A's (same users)",
                    float(np.max(np.abs(col(r, "d_prior") - col(ref, "d_prior")))), 1e-12))
    return res + ood_checks(c, ous, run)


def ood_checks(c: dict, ous: dict, run: Path, folds=G.FOLDS) -> list[tuple[str, float, float]]:
    """The O cells: the user set of every row, the same generic distance and floor in every
    cell, the floor of the users file, d_prior rebuilt from the fold's prior mean curve and the
    user's ideal, and the two user sets holding the same users (ids and answer seeds)."""
    res = []
    generic = {fold: prior_mean_curve(phase5.arm_space(phase5._prior(str(run), fold, "promp_s0"),
                                                       "tight").design)
               for fold in folds}
    for s in O_SETS:
        base, users = c[("ood", s, "a")], ous[s]["users"]
        for key in O_OPTS:
            df = c[("ood", s, key)]
            label = f"O {s}, {O_OPTS[key][2]}"
            res.append((f"{label}: rows whose user_set is not {s!r}",
                        float(np.sum(df["user_set"].astype(str).to_numpy() != s)), 0.0))
            res.append((f"{label}: d_floor = the users file's promp_s0 floor_nm",
                        float(np.max(np.abs(col(df, "d_floor") - np.array(
                            [users[u].families["promp_s0"]["floor_nm"] for u in df["user_id"]])))),
                        0.0))
            if key != "a":
                res.append((f"{label}: d_prior = A's (same users)",
                            float(np.max(np.abs(col(df, "d_prior") - col(base, "d_prior")))),
                            1e-12))
        rebuilt = np.array([rms_distance(generic[f], users[u].curve_nm)
                            for f, u in zip(base["fold"], base["user_id"])])
        res.append((f"O {s}: d_prior = RMS(prior mean curve − the user's ideal), rebuilt",
                    float(np.max(np.abs(col(base, "d_prior") - rebuilt))), 1e-12))
    pb, common = ous["ood-pb"], ous["ood"]
    res.append(("O: users in one set but not in the other, or in another order",
                float(sum(a != b for a, b in zip(pb["keys"], common["keys"]))
                      + abs(len(pb["keys"]) - len(common["keys"]))), 0.0))
    res.append(("O: users whose answer seed differs between the two sets",
                float(sum(pb["users"][u].answer_seed != common["users"][u].answer_seed
                          for u in pb["users"])), 0.0))
    return res


# ============================================================================ main analysis
def analyse_o(c: dict, folds_main: np.ndarray, n_boot: int, main: dict) -> dict:
    """PO1 and PO2 (and context) on each O user set. ``folds_main`` are the e-main users'
    folds; ``main`` holds the main condition's medians at k = 24."""
    # The main condition's best comparator by its median, fixed (not per draw): that keeps the
    # main advantage, and so PO2's bar, at its largest.
    best_main = min(O_COMPARATORS, key=lambda o: main["meds"][o])
    mA, mB = rms(c[("main", "a")], K_MAIN), rms(c[("main", best_main)], K_MAIN)
    adv_main = kr.Boot(folds_main, n_boot)([mA, mB], kr.diff_of_medians)
    res: dict = {}
    for s in O_SETS:
        cc = {key: c[("ood", s, key)] for key in O_OPTS}
        folds_o = cc["a"]["fold"].to_numpy()
        boot = kr.Boot(folds_o, n_boot)
        r: dict = {"win": {}, "err": {}, "con_win": {}, "con_err": {}}
        for key, df in cc.items():
            for k in KS:
                r["win"][(key, k)] = boot([win_rms(df, k)], mean)
                r["err"][(key, k)] = boot([rms(df, k)], lambda x: med(x))
        for o in R_COMPARATORS:
            for k in KS:
                r["con_win"][(o, k)] = boot([win_rms(cc["a"], k), win_rms(cc[o], k)], diff_mean)
                r["con_err"][(o, k)] = boot([rms(cc["a"], k), rms(cc[o], k)], kr.diff_of_medians)
        a24 = rms(cc["a"], K_MAIN)
        po2 = dict(best_main=best_main, adv_main=adv_main,
                   fixed_thr=PO2_FRACTION * adv_main["estimate"],
                   meds={key: float(er._med(rms(df, K_MAIN))) for key, df in cc.items()})
        for tag, names in (("three", O_COMPARATORS), ("four", O_COMPARATORS_ABLATION)):
            comps = [rms(cc[o], K_MAIN) for o in names]
            adv = boot([a24] + comps, adv_over_best)
            D = two_sample([a24] + comps, folds_o, adv_over_best, [mA, mB], folds_main,
                           kr.diff_of_medians, lambda x, y: x - PO2_FRACTION * y, n_boot)
            share = two_sample([a24] + comps, folds_o, adv_over_best, [mA, mB], folds_main,
                               kr.diff_of_medians, lambda x, y: x / y, n_boot)
            draws = draw_matrix(folds_o, n_boot)
            po2[tag] = dict(adv=adv, D=D, share=share,
                            counts=best_counts([x[draws] for x in comps], names),
                            point_best=min(names, key=lambda o: po2["meds"][o]),
                            state=po2_state(adv_main, adv, D),
                            fixed_state=po2_state(adv_main, adv, adv, po2["fixed_thr"]))
        # Verdict: the D rule (its interval carries the main advantage's uncertainty too);
        # random-duel A is taken in only where it lowers the verdict.
        po2["state"] = least_favourable(po2["three"]["state"], po2["four"]["state"])
        po2["fixed_state"] = least_favourable(po2["three"]["fixed_state"],
                                              po2["four"]["fixed_state"])
        po1 = dict(ci=r["win"][("a", K_MAIN)],
                   state=side(r["win"][("a", K_MAIN)], PO1_LEVEL, want_low=False))
        ad = cc["a"]
        r.update(po1=po1, po2=po2, n=len(ad),
                 room=float(er._med(col(ad, "d_prior") - col(ad, "d_floor"))),
                 floor=float(er._med(col(ad, "d_floor"))),
                 d_prior=float(er._med(col(ad, "d_prior"))))
        res[s] = r
    return res


def analyse(c: dict, folds, n_boot: int, side_: dict) -> dict:
    boot = kr.Boot(folds, n_boot)
    draws = draw_matrix(folds, n_boot)
    v: dict = {}

    # ---- the draw helper reproduces joint_bootstrap (a check)
    pa = win(c[("rand", "a")], K_MAIN)
    via = boot([pa], mean)
    b = pa[draws].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5], method="inverted_cdf")
    v["draw_check"] = max(abs(lo - via["lo"]), abs(hi - via["hi"]))

    # ---- main condition, RMS
    mA, mG = rms(c[("main", "a")], K_MAIN), rms(c[("main", "g")], K_MAIN)
    lead_main = boot([mA, mG], kr.diff_of_medians)
    ma_df = c[("main", "a")]
    v["main"] = dict(lead=lead_main, med_a=float(er._med(mA)), med_g=float(er._med(mG)),
                     room=float(er._med(col(ma_df, "d_prior") - col(ma_df, "d_floor"))),
                     gain_a=float(er._med(col(ma_df, "d_prior") - mA)),
                     gain_g=float(er._med(col(c[("main", "g")], "d_prior") - mG)),
                     win={lr: boot([win_rms(c[("main", lr)], K_MAIN)], mean)
                          for lr in MAIN_LEARNERS},
                     meds={lr: float(er._med(rms(c[("main", lr)], K_MAIN)))
                           for lr in MAIN_LEARNERS},
                     floor=float(er._med(col(ma_df, "d_floor"))),
                     d_prior=float(er._med(col(ma_df, "d_prior"))))

    # ---- O: PO1 and PO2 on each user set
    v["O"] = analyse_o(c, np.asarray(folds), n_boot, v["main"])
    v["PO1"] = v["O"]["ood-pb"]["po1"]
    v["PO2"] = v["O"]["ood-pb"]["po2"]

    # ---- H: per cell, matched (home) and unmatched (obs)
    H = {}
    for cell, (_, ratio) in LANDMARK.items():
        for src in ("home", "obs"):
            cc = {key: c[(src, cell, key)] for key in H_OPTS}
            r: dict = {"win": {}, "own": {}, "con_win": {}, "con_own": {}}
            for key, df in cc.items():
                for k in KS:
                    r["win"][(key, k)] = boot([win(df, k)], mean)
                    r["own"][(key, k)] = boot([own(df, k)], lambda x: med(x))
            for a_, b_ in H_CONTRASTS:
                for k in KS:
                    r["con_win"][(a_, b_, k)] = boot([win(cc[a_], k), win(cc[b_], k)], diff_mean)
                    r["con_own"][(a_, b_, k)] = boot([own(cc[a_], k), own(cc[b_], k)],
                                                     kr.diff_of_medians)
            # PH2 parts; for the unmatched cell, the own distance as stored and ÷ ratio
            scales = {"stored": 1.0} if src == "home" else {"stored": 1.0, "per ratio": ratio}
            r["ph2"] = {}
            for sname, div in scales.items():
                oA, oG = own(cc["a"], K_MAIN) / div, own(cc["g"], K_MAIN) / div
                lead = boot([oA, oG], kr.diff_of_medians)
                D = boot([oA, oG, mA, mG],
                         lambda a, g, ma, mg: (med(a) - med(g)) - RETAINED * (med(ma) - med(mg)))
                share = boot([oA, oG, mA, mG],
                             lambda a, g, ma, mg: (med(a) - med(g)) / (med(ma) - med(mg)))
                fixed_thr = RETAINED * lead_main["estimate"]
                st, why = ph2_state(lead_main, D)
                r["ph2"][sname] = dict(lead=lead, D=D, share=share, state=st, why=why,
                                       fixed_thr=fixed_thr,
                                       fixed_state=side(lead, fixed_thr, want_low=True),
                                       med_a=float(er._med(oA)), med_g=float(er._med(oG)))
            H[(src, cell)] = r
        # matched minus unmatched lead over G in the same metric (own distance ÷ ratio):
        # same users, same metric, only the decisiveness of the answers differs
        hA, hG = own(c[("home", cell, "a")], K_MAIN), own(c[("home", cell, "g")], K_MAIN)
        uA = own(c[("obs", cell, "a")], K_MAIN) / ratio
        uG = own(c[("obs", cell, "g")], K_MAIN) / ratio
        H[("cmp", cell)] = boot([hA, hG, uA, uG],
                                lambda a, g, ua, ug: (med(a) - med(g)) - (med(ua) - med(ug)))
    v["H"] = H

    # ---- PH1 and PH2 verdicts
    ph1 = {cell: H[("home", cell)]["con_win"][("a", "g", K_MAIN)] for cell in LANDMARK}
    ph1_states = {cell: side(ph1[cell], 0.0, want_low=False) for cell in LANDMARK}
    v["PH1"] = dict(ci=ph1, states=ph1_states,
                    state=kr.combine([None if s == "uncertain" else s
                                      for s in ph1_states.values()]),
                    a={cell: H[("home", cell)]["win"][("a", K_MAIN)] for cell in LANDMARK},
                    g={cell: H[("home", cell)]["win"][("g", K_MAIN)] for cell in LANDMARK})
    ph2 = {cell: H[("home", cell)]["ph2"]["stored"] for cell in LANDMARK}
    v["PH2"] = dict(cells=ph2, state=kr.combine([None if p["state"] == "uncertain"
                                                 else p["state"] for p in ph2.values()]))

    # ---- R
    R: dict = {"win": {}, "con": {}, "first": {}, "share": {}, "gain": {}}
    rc = {key: c[("rand", key)] for key in R_OPTS}
    users = side_["users"]
    uids = rc["a"]["user_id"].to_numpy()
    umax = np.array([users[u]["u_max"] for u in uids])
    ugen = np.array([users[u]["u_generic"] for u in uids])
    conv = np.array([users[u]["converged"] for u in uids], dtype=bool)
    folds_arr = np.asarray(folds)
    boot_conv = kr.Boot(folds_arr[conv], n_boot)
    for key, df in rc.items():
        P = win_curve(df)
        for k in KS:
            R["win"][(key, k)] = boot([P[:, k - 1]], mean)
            gain = col(df, "own_generic_distance") - own(df, k)       # u_k - u_gen
            share = share_of_gain(gain, umax, ugen)
            R["gain"][(key, k)] = boot([gain], lambda x: med(x))
            R["share"][(key, k, "all")] = boot([share], lambda x: med(x))
            R["share"][(key, k, "conv")] = boot_conv([share[conv]], lambda x: med(x))
        for level in FIRST_K_LEVELS:
            R["first"][(key, level)] = first_k(P, draws, level)
    pa24 = win(rc["a"], K_MAIN)
    for other in R_COMPARATORS:
        for k in KS:
            R["con"][(other, k)] = boot([win(rc["a"], k), win(rc[other], k)], diff_mean)
    # A minus the best comparator, the best taken within each draw (context)
    R["best"] = boot([pa24] + [win(rc[o], K_MAIN) for o in R_COMPARATORS],
                     lambda a, *others: a.mean(axis=1) - np.max(
                         np.stack([o.mean(axis=1) for o in others]), axis=0))
    R["umax"] = dict(n=len(uids), conv=int(conv.sum()),
                     conv_by_fold={f: int(np.sum(conv & (folds_arr == f)))
                                   for f in pd.unique(folds_arr)},
                     sources={s: int(np.sum([users[u]["source"] == s for u in uids]))
                              for s in sorted({users[u]["source"] for u in uids})},
                     med_umax=float(np.median(umax)), med_ugen=float(np.median(ugen)),
                     med_head=float(np.median(umax - ugen)), min_head=float(np.min(umax - ugen)),
                     med_head_conv=float(np.median((umax - ugen)[conv])),
                     med_head_unconv=float(np.median((umax - ugen)[~conv])),
                     # sessions whose best identified curve (any k) beat the stored u_max
                     above=int(sum(np.sum(col(df, "own_generic_distance")
                                          - lst(df, "own_distances").min(axis=1)
                                          > umax - ugen + 1e-9) for df in rc.values())),
                     n_sessions=len(uids) * len(rc))
    v["R"] = R
    pr1 = {o: R["con"][(o, K_MAIN)] for o in R_COMPARATORS}
    pr1_states = {o: side(pr1[o], 0.0, want_low=False) for o in R_COMPARATORS}
    v["PR1"] = dict(ci=pr1, states=pr1_states,
                    state=kr.combine([None if s == "uncertain" else s
                                      for s in pr1_states.values()]))
    v["PR2"] = dict(ci=R["win"][("a", K_MAIN)],
                    state=side(R["win"][("a", K_MAIN)], PR2_LEVEL, want_low=False))
    return v


# ============================================================================ report
def excl_text(H: dict) -> str:
    """Which cells' matched-minus-unmatched lead excludes 0, in words."""
    ex = [LANDMARK_TITLE[c].split()[0] for c in LANDMARK
          if H[("cmp", c)]["hi"] < 0 or H[("cmp", c)]["lo"] > 0]
    if not ex:
        return "Neither interval excludes 0."
    if len(ex) == len(LANDMARK):
        return "Both intervals exclude 0."
    return f"Only the {ex[0]} cell's interval excludes 0."


def verdict_lines(v: dict) -> list[str]:
    ph1, ph2, pr1, pr2 = v["PH1"], v["PH2"], v["PR1"], v["PR2"]
    po1, po2, O = v["PO1"], v["PO2"], v["O"]
    L = ["## 1. Registered verdicts", "",
         "| # | Prediction (spec §9a) | Verdict | Numbers (k = 24; 200 users per test) |",
         "|---|---|---|---|"]
    txt = "; ".join(
        f"{LANDMARK_TITLE[c]}: A − G {ci3(ph1['ci'][c])} → {ph1['states'][c]} "
        f"(A {p3(ph1['a'][c])}, G {p3(ph1['g'][c])})" for c in LANDMARK)
    L.append(f"| PH1 | A stays ahead of G in the own-utility win rate in both matched landmark "
             f"cells (interval excluding 0) | **{ph1['state']}** | {txt} |")
    m = v["main"]
    txt = "; ".join(
        f"{LANDMARK_TITLE[c]}: A − G {ci3(p['lead'])} Nm-eq., D = {ci3(p['D'])} → {p['state']}"
        for c, p in ph2["cells"].items())
    L.append(f"| PH2 | A's lead over G in the matched wearer's own distance is at least ⅔ of its "
             f"RMS main-condition lead (A − G = {ci3(m['lead'])} Nm; ⅔ = "
             f"{RETAINED * m['lead']['estimate']:+.3f}) | **{ph2['state']}** | {txt}. "
             "D = lead_home − ⅔·lead_main, held when D ≤ 0 throughout its interval |")
    oc = O["ood"]
    L.append(f"| PO1 | A's validation win rate against generic is ≥ {PO1_LEVEL:.2f} "
             f"(out-of-generator ideals, per-bout peaks) | **{po1['state']}** | A "
             f"{p3(po1['ci'])}. Common peak (sensitivity): A {p3(oc['po1']['ci'])} → "
             f"({oc['po1']['state']}) |")
    am = po2["adv_main"]
    L.append(f"| PO2 | A's error advantage over the best comparator (difference of medians) "
             f"excludes 0 and is at least half its main-condition size (A − "
             f"{MAIN_LABEL[po2['best_main']]} = {ci3(am)} Nm; ½ = {po2['fixed_thr']:+.3f}) | "
             f"**{po2['state']}** | {po2_numbers(po2)}. Common peak (sensitivity): "
             f"{po2_numbers(oc['po2'])} → ({oc['po2']['state']}). D = adv − ½·adv_main, held "
             "when adv < 0 and D ≤ 0 throughout their intervals; rulings in §3.1 |")
    txt = "; ".join(f"A − {R_OPTS[o][2]} {ci3(pr1['ci'][o])} → {pr1['states'][o]}"
                    for o in R_COMPARATORS)
    L.append(f"| PR1 | A's own-utility win rate against generic exceeds every comparator's "
             f"(every interval excluding 0) | **{pr1['state']}** | {txt} |")
    L.append(f"| PR2 | A's own-utility win rate ≥ {PR2_LEVEL:.2f} | **{pr2['state']}** | "
             f"A {p3(pr2['ci'])} |")
    H, R, mw = v["H"], v["R"], v["main"]["win"]
    st, wk = ph2["cells"]["static"], ph2["cells"]["walking"]
    un = {c: H[("obs", c)]["ph2"]["per ratio"] for c in LANDMARK}
    u = R["umax"]

    def where(p):
        """Where a cell's share kept sits against 2/3, in words, from its D interval."""
        pos = "on" if abs(p["D"]["estimate"]) < 0.02 else (
            "below" if p["D"]["estimate"] > 0 else "above")
        return f"point estimate {pos} the ⅔ threshold, D = {ci3(p['D'])}"

    below_one = all(p["share"]["hi"] < 1 for p in ph2["cells"].values())
    L += ["", "**What the numbers say.**",
          f"- **PH1 holds with margin in both cells:** A − G "
          f"{ci3(ph1['ci']['static'])} (static) and {ci3(ph1['ci']['walking'])} (walking). The "
          "unmatched landmark wearer gives "
          f"{ci3(H[('obs', 'static')]['con_win'][('a', 'g', K_MAIN)])} and "
          f"{ci3(H[('obs', 'walking')]['con_win'][('a', 'g', K_MAIN)])}, but its validation "
          "duel is more decisive too, so those win rates do not compare directly.",
          f"- **PH2 is undecided in both cells.** A keeps "
          f"{ci3(st['share'], 2, signed=False)} (static) and {ci3(wk['share'], 2, signed=False)} "
          f"(walking) of its main-condition lead (static: {where(st)}; walking: {where(wk)})."
          + (" Both share intervals exclude 1: in this metric the lead is smaller than A's RMS "
             "lead in main." if below_one else ""),
          f"- **Same metric, different decisiveness of the answers.** {er.ANALYST} read in the "
          "matched metric (own distance ÷ ratio), the unmatched, more decisive wearer's sessions "
          f"give A a lead over G of {ci3(un['static']['lead'])} (static) and "
          f"{ci3(un['walking']['lead'])} (walking), against {ci3(st['lead'])} and "
          f"{ci3(wk['lead'])} at matched decisiveness. Matched minus unmatched lead (joint "
          f"bootstrap, same users): {ci3(H[('cmp', 'static')])} (static) and "
          f"{ci3(H[('cmp', 'walking')])} (walking); negative = A further ahead at matched "
          "decisiveness. " + excl_text(H)]
    L += o_bullets(v)
    L += [f"- **PR1 is undecided because of G alone.** A beats LineCoSpar, SortCMA and its own "
          f"random-duel ablation with intervals excluding 0; A − G is "
          f"{ci3(pr1['ci']['g'])}. G's win rate is {p3(R['win'][('g', K_MAIN)])}, against "
          f"{p3(mw['g'])} in the main condition; A's is {p3(R['win'][('a', K_MAIN)])} against "
          f"{p3(mw['a'])}.",
          f"- **The random utility leaves far more room than the main condition.** {er.ANALYST} "
          "its median "
          f"headroom u_max − u_gen is {u['med_head']:.2f} scaled units (a lower bound), against "
          f"a median main-condition room d_prior − d_floor of {v['main']['room']:.2f} Nm, the "
          "scale the random utility's decisiveness is matched to. By k = 24, A gains a median "
          f"{R['gain'][('a', K_MAIN)]['estimate']:.2f} units and G "
          f"{R['gain'][('g', K_MAIN)]['estimate']:.2f} (main condition: "
          f"{v['main']['gain_a']:.2f} and {v['main']['gain_g']:.2f} Nm): A gains about as much "
          f"as in main, G about {R['gain'][('g', K_MAIN)]['estimate'] / v['main']['gain_g']:.1f}"
          "× as much. Gains of that size win most validation duels for both, while each "
          "reaches a median "
          f"{ci3(R['share'][('a', K_MAIN, 'all')], 2, signed=False)} (A) and "
          f"{ci3(R['share'][('g', K_MAIN, 'all')], 2, signed=False)} (G) of the achievable gain "
          f"over all {u['n']} users, an upper bound, and "
          f"{ci3(R['share'][('a', K_MAIN, 'conv')], 2, signed=False)} (A) and "
          f"{ci3(R['share'][('g', K_MAIN, 'conv')], 2, signed=False)} (G) over the {u['conv']} "
          "users whose u_max converged (§4.3).",
          f"- **PR2 holds with margin:** {p3(pr2['ci'])}; the main condition's A is "
          f"{p3(mw['a'])}, so the win rate is lower under R, as predicted."]
    return L + [""]


def po2_numbers(p: dict) -> str:
    t = p["three"]
    return (f"A − best comparator {ci3(t['adv'])} Nm, D = {ci3(t['D'])}, share of the main "
            f"advantage kept {ci3(t['share'], 2, signed=False)}; fixed-threshold reading "
            f"{p['fixed_state']}")


def best_win(r: dict) -> tuple[str, dict]:
    """The comparator with the highest mean win rate at k = 24 (context)."""
    o = max(O_COMPARATORS, key=lambda x: r["win"][(x, K_MAIN)]["estimate"])
    return O_OPTS[o][2], r["win"][(o, K_MAIN)]


def o_bullets(v: dict) -> list[str]:
    po1, po2, O, m = v["PO1"], v["PO2"], v["O"], v["main"]
    pb, oc = O["ood-pb"], O["ood"]
    bl, bw = best_win(pb)
    t, tc = po2["three"], oc["po2"]["three"]
    return [
        f"- **PO1: {po1['state']}.** A's win rate against generic assistance at k = 24 is "
        f"{p3(po1['ci'])} on the per-bout out-of-generator users (bar {PO1_LEVEL:.2f}), against "
        f"{p3(m['win']['a'])} in the main condition. The best comparator's is {bl} "
        f"{p3(bw)}. With the common peak (sensitivity), A's is {p3(oc['po1']['ci'])}.",
        f"- **PO2: {po2['state']}.** A's error advantage over the best comparator is "
        f"{ci3(t['adv'])} Nm (best by median: {O_OPTS[t['point_best']][2]}, "
        f"{po2['meds'][t['point_best']]:.3f} Nm against A's {po2['meds']['a']:.3f}), against "
        f"{ci3(po2['adv_main'])} Nm in the main condition: A keeps "
        f"{ci3(t['share'], 2, signed=False)} of its main advantage (D = {ci3(t['D'])}). The "
        f"fixed-threshold reading (the advantage against the point value "
        f"{po2['fixed_thr']:+.3f}) is {po2['fixed_state']}. With the common peak "
        f"(sensitivity): {ci3(tc['adv'])} Nm, {ci3(tc['share'], 2, signed=False)} kept, "
        f"D = {ci3(tc['D'])} → ({oc['po2']['state']}).",
        f"- **The per-bout users have less room.** {er.ANALYST} the per-bout rule matches the "
        f"ideals' size (median RMS {v['sizes']['ood-pb']:.2f} Nm against the population's "
        f"{v['sizes']['pop']:.2f}) but not their distance from generic assistance. Their median "
        f"generic distance is {pb['d_prior']:.2f} Nm and their room d_prior − d_floor "
        f"{pb['room']:.2f} Nm, against {m['d_prior']:.2f} and {m['room']:.2f} Nm for the main "
        f"condition's users; the common-peak users' are {oc['d_prior']:.2f} and "
        f"{oc['room']:.2f} Nm. So the per-bout test changes the room as well as the generator. "
        "W1 found A's win rate falls as the room shrinks: in-generator users reweighted to the "
        f"median-matched room of {MEDIAN_ROOM_NM} Nm gave {MEDIAN_ROOM_WIN} at k = 24 (spec "
        f"§10). The best comparator here, {O_OPTS[t['point_best']][2]}, has a median error of "
        f"{po2['meds'][t['point_best']]:.3f} Nm against a median generic distance of "
        f"{pb['d_prior']:.3f}"
        + (": it stays near generic assistance, so A's advantage over it is bounded by the room "
           "too." if po2["meds"][t["point_best"]] >= 0.9 * pb["d_prior"] else ".")]


def o_section(v: dict, ous: dict, n_boot: int) -> list[str]:
    O, m, po2 = v["O"], v["main"], v["PO2"]
    cpb, cc = ous["ood-pb"]["config"], ous["ood"]["config"]
    cv = cpb["magnitude_cv"]
    L = ["## 3. Out-of-generator ideals (O)", "",
         f"**Users.** {O['ood-pb']['n']} out-of-generator users, "
         f"{O['ood-pb']['n'] // len(G.FOLDS)} per LOSO fold (ids `<fold>/ood000`, `ood001`, …; "
         "`kind = population`): other users than the population users of H, R and the main "
         "condition. Their ideals are Schäfer's four-half-cosine profiles with timings drawn "
         "from Schäfer's distribution (build report §1.2). Two magnitude rules, on the same "
         "timings, z, answer seeds and ids:",
         f"- **per-bout peaks (e-ood-pb, primary; ruling \"O, peak magnitude\"):** each bout's "
         f"peak is its population median (extension {cpb['median_ext_peak_nm']:.3f} Nm, flexion "
         f"{cpb['median_flex_peak_nm']:.3f} Nm) × (1 + {cv:g} z), one z per user, clipped to "
         "[0, 12] Nm;",
         f"- **common peak (e-ood, sensitivity):** one peak for both bouts, "
         f"{cc['median_peak_nm']:.3f} Nm × (1 + {cc['magnitude_cv']:g} z).", "",
         "The wearer is the plain RMS wearer at s = 1.0 with lapse 0.1, so the error is the RMS "
         "distance to the user's own ideal (`distances`), generic assistance is the fold's ProMP "
         "prior mean curve (`d_prior`) and the floor is the user's own in the ProMP family "
         "(`d_floor`). The win rate is the RMS wearer's: P_i = λ/2 + (1 − λ)·Φ((d_prior − "
         "d_i(k))/s).", "",
         "| users | ideal RMS size, median | generic distance d_prior, median | floor d_floor, "
         "median | room d_prior − d_floor, median |", "|---|---|---|---|---|",
         f"| main condition (200 population users, e-main A rows) | {v['sizes']['pop']:.3f} | "
         f"{m['d_prior']:.3f} | {m['floor']:.3f} | {m['room']:.3f} |"]
    for s in O_SETS:
        r = O[s]
        L.append(f"| O, {O_TITLE[s]} | {v['sizes'][s]:.3f} | {r['d_prior']:.3f} | "
                 f"{r['floor']:.3f} | {r['room']:.3f} |")
    t4 = {s: O[s]["po2"]["four"]["counts"] for s in O_SETS}
    L += ["", "### 3.1 Rulings on PO1 and PO2 (analyst's, written before these numbers were "
          "computed)", "",
          "The registration leaves five points open. Each is ruled in the direction that makes a "
          "held verdict harder to reach:",
          f"1. **PO1's bar.** \"≥ {PO1_LEVEL:.2f}\" is held only when the whole 95 % interval lies "
          f"above {PO1_LEVEL:.2f}, failed when it lies wholly below, as PR2.",
          "2. **The best comparator** is the best of G, LineCoSpar and SortCMA, the comparators "
          "of the main condition (spec §9 table). It is taken within each bootstrap draw, as "
          "PW3b's was, which can only shrink A's advantage. A with random duels is A's own "
          "ablation: it is added as a fourth comparator only where that would lower the "
          "verdict. It is the best comparator in "
          + " and ".join(f"{t4[s]['a_rand']} of {n_boot} draws ({s})" for s in O_SETS) + ".",
          f"3. **The main-condition size** is A − {MAIN_LABEL[po2['best_main']]} in e-main at "
          f"k = 24, {ci3(po2['adv_main'])} Nm; {MAIN_LABEL[po2['best_main']]} is the main "
          "condition's best comparator by its median ("
          + ", ".join(f"{MAIN_LABEL[o]} {m['meds'][o]:.3f}" for o in O_COMPARATORS)
          + f" Nm; A {m['meds']['a']:.3f}). It is fixed, not taken per draw, which keeps the main "
          "advantage, and so PO2's bar, at its largest.",
          "4. **\"At least half\" is judged as PH2's ⅔ was:** D = adv_O − ½·adv_main, held when "
          "D's interval lies wholly at or below 0. The O users are not the main condition's "
          "users, so the two halves are resampled independently, each within its folds (O with "
          f"seed {er.SEED}, the draws of every other statistic here; e-main with seed "
          f"{MAIN_DRAW_SEED}), and D is formed draw by draw. Its interval carries the "
          "uncertainty of the main advantage as well, so it is wider than the fixed-threshold "
          "reading (adv_O's own interval against the point value ½ × adv_main), which is "
          "reported as a sensitivity.",
          "5. **Both parts must hold.** PO2 is held only if adv_O's interval lies wholly below 0 "
          "and D's wholly at or below 0, with the main advantage itself excluding 0 (PH2's gate); "
          "failed if either interval lies wholly on the other side; uncertain otherwise.",
          "", "The common-peak run gets the same statistics. Its verdicts are in parentheses: a "
          "labelled sensitivity, not the registered verdict.", "",
          "### 3.2 PO1: win rate against generic assistance at k = 24", "",
          "| users | A | best comparator | verdict |", "|---|---|---|---|"]
    for s in O_SETS:
        r = O[s]
        bl, bw = best_win(r)
        st = r["po1"]["state"] if s == "ood-pb" else f"({r['po1']['state']})"
        L.append(f"| O, {O_TITLE[s]} | {p3(r['po1']['ci'])} | {bl} {p3(bw)} | {st} |")
    mb = max(O_COMPARATORS, key=lambda o: m["win"][o]["estimate"])
    L += [f"| main condition (context) | {p3(m['win']['a'])} | {MAIN_LABEL[mb]} "
          f"{p3(m['win'][mb])} | — |", "",
          "### 3.3 PO2: A's error advantage over the best comparator at k = 24", "",
          f"Main condition (e-main, k = 24): A {m['meds']['a']:.3f} Nm, "
          + ", ".join(f"{MAIN_LABEL[o]} {m['meds'][o]:.3f}" for o in O_COMPARATORS)
          + f" Nm. adv_main = A − {MAIN_LABEL[po2['best_main']]} = {ci3(po2['adv_main'])} Nm; "
          f"half of it: {po2['fixed_thr']:+.3f} Nm.", "",
          "adv = med(A) − the smallest comparator median within each draw (negative favours A). "
          "D = adv − ½·adv_main (independent draws, ruling 4). \"Share kept\" = adv ÷ adv_main "
          "(1 = all of A's main advantage kept, 0.5 = PO2's bar), same draws as D. \"Best in "
          "draws\" counts the draws in which each comparator has the smallest median.", "",
          "| users | comparators | A, median | best by median | adv | D | share kept | best in "
          "draws | verdict (D rule) | fixed-threshold reading |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in O_SETS:
        p = O[s]["po2"]
        for tag, names in (("three", O_COMPARATORS), ("four", O_COMPARATORS_ABLATION)):
            q = p[tag]
            counts = ", ".join(f"{O_OPTS[o][2]} {q['counts'][o]}" for o in names)
            comp = "G, LineCoSpar, SortCMA" + (", A random duels" if tag == "four" else "")
            st, fx = q["state"], q["fixed_state"]
            if s != "ood-pb":
                st, fx = f"({st})", f"({fx})"
            L.append(f"| O, {O_TITLE[s]} | {comp} | {p['meds']['a']:.3f} | "
                     f"{O_OPTS[q['point_best']][2]} {p['meds'][q['point_best']]:.3f} | "
                     f"{ci3(q['adv'])} | {ci3(q['D'])} | {ci3(q['share'], 2, signed=False)} | "
                     f"{counts} | {st} | {fx} |")
    pc = O["ood"]["po2"]
    L += ["", f"**PO2, registered verdict: {po2['state']}** (the less favourable of the two "
          f"comparator sets under the D rule; fixed-threshold reading {po2['fixed_state']}). "
          f"Common peak, sensitivity: ({pc['state']}); fixed-threshold reading "
          f"({pc['fixed_state']}).", "",
          "### 3.4 Context: every optimizer, k = 12 / 24 / 40", "",
          "Win rate = mean P̄ against generic assistance; error = median RMS distance to the "
          "user's ideal (Nm). Contrasts: difference of means (win rate) and of medians (error) "
          "on the same users.", ""]
    for s in O_SETS:
        r = O[s]
        L += [f"**{O_TITLE[s]}.**", "",
              "| optimizer | win rate, k = 12 | k = 24 | k = 40 | error, k = 12 | k = 24 | "
              "k = 40 |", "|---|---|---|---|---|---|---|"]
        for key, (_, _, label) in O_OPTS.items():
            L.append(f"| {label} | " + " | ".join(p3(r["win"][(key, k)]) for k in KS) + " | "
                     + " | ".join(p3(r["err"][(key, k)]) for k in KS) + " |")
        L += ["", "| contrast | win rate, k = 12 | k = 24 | k = 40 | error, k = 12 | k = 24 | "
              "k = 40 |", "|---|---|---|---|---|---|---|"]
        for o in R_COMPARATORS:
            L.append(f"| A − {O_OPTS[o][2]} | "
                     + " | ".join(ci3(r["con_win"][(o, k)]) for k in KS) + " | "
                     + " | ".join(ci3(r["con_err"][(o, k)]) for k in KS) + " |")
        L.append("")
    return L


def h_section(v: dict, dec: dict) -> list[str]:
    H, m = v["H"], v["main"]
    L = ["## 2. Home-advantage test at matched decisiveness (H)", "",
         "**Wearer.** The landmark utility divided by its cell's decisiveness ratio (static "
         f"{LANDMARK['static'][1]:.4f}, walking {LANDMARK['walking'][1]:.4f}), so that its median "
         "|Δu| on prior-scale pairs matches the plain s = 1.0 wearer's. Re-measured as in REPORT "
         "§9 (`out/icorr/robust_decisiveness.json`, not recomputed here):"]
    for cell, key in (("static", "landmark"), ("walking", "landmark25")):
        ins = dec["in_sample"].get(f"{key}-matched", {}).get("ratio")
        oos = dec["out_of_sample"].get(f"{key}-matched", {}).get("ratio")
        L.append(f"- {LANDMARK_TITLE[cell]}: achieved ratio {ins:.6f} on the measurement pairs "
                 f"(true by construction), {oos:.3f} on 100 other users with fresh pairs.")
    L += ["", "**Scale (ruling \"H, PH2 scale\").** `own_distances` and `own_generic_distance` "
          "are on the matched scale, D × 0.674 / ratio (\"Nm-eq.\"). The win rate uses the same "
          "scale, which is the one the wearer answered by. The unmatched e-observer cells "
          "(context) are given as stored (D × 0.674, the scale that wearer answered by) and "
          "÷ ratio (the matched metric, so the only difference from e-home is how decisively "
          "the wearer answered during the session).", ""]
    # PH1
    L += ["### 2.1 PH1: own-utility win rate against generic assistance", "",
          "Mean over users of P_i(24); A − G is the difference of the two means.", "",
          "| cell | A | G | A − G | verdict |", "|---|---|---|---|---|"]
    for cell in LANDMARK:
        L.append(f"| {LANDMARK_TITLE[cell]}, matched | {p3(v['PH1']['a'][cell])} | "
                 f"{p3(v['PH1']['g'][cell])} | {ci3(v['PH1']['ci'][cell])} | "
                 f"{v['PH1']['states'][cell]} |")
    for cell in LANDMARK:
        r = H[("obs", cell)]
        L.append(f"| {LANDMARK_TITLE[cell]}, unmatched (e-observer, context) | "
                 f"{p3(r['win'][('a', K_MAIN)])} | {p3(r['win'][('g', K_MAIN)])} | "
                 f"{ci3(r['con_win'][('a', 'g', K_MAIN)])} | — |")
    # PH2
    L += ["", "### 2.2 PH2: A's lead over G in the wearer's own distance against the main "
          "condition", "",
          f"Main condition (e-main, RMS wearer, s = 1.0), k = 24: A {m['med_a']:.3f} Nm, G "
          f"{m['med_g']:.3f} Nm, A − G = {ci3(m['lead'])} Nm. Two thirds of it: "
          f"{RETAINED * m['lead']['estimate']:+.3f} Nm.", "",
          "lead = med(A) − med(G) (negative favours A). D = lead in the landmark cell − ⅔ × "
          "lead in main, all four arrays in one joint bootstrap on the same 200 users. Held when "
          "D's interval lies wholly at or below 0 and the main lead's interval excludes 0 (QE3's "
          "gate); failed when D's interval lies wholly above 0. \"Share kept\" = lead in the "
          "landmark cell ÷ lead in main (1 = all of A's main lead kept). The fixed-threshold "
          "reading compares the landmark lead's own interval with the point value ⅔ × "
          f"{m['lead']['estimate']:+.3f} (a sensitivity, not the registered rule).", "",
          "| cell | own distance | A, median | G, median | A − G | D | share kept | verdict "
          "| fixed-threshold reading |", "|---|---|---|---|---|---|---|---|---|"]
    for src, tag in (("home", "matched"), ("obs", "unmatched (context)")):
        for cell in LANDMARK:
            for sname, p in H[(src, cell)]["ph2"].items():
                scale = ("as stored (matched scale)" if src == "home" else
                         "as stored" if sname == "stored" else "÷ ratio (matched metric)")
                L.append(f"| {LANDMARK_TITLE[cell]}, {tag} | {scale} | {p['med_a']:.3f} | "
                         f"{p['med_g']:.3f} | {ci3(p['lead'])} | {ci3(p['D'])} | "
                         f"{ci3(p['share'], 2, signed=False)} | "
                         + (p["state"] if src == "home" else f"({p['state']})")
                         + f" | {p['fixed_state']} |")
    # context tables
    L += ["", "### 2.3 Context: every optimizer, k = 12 / 24 / 40", "",
          "Win rate = mean P̄ against each family's generic assistance (the prior's mean curve; "
          "for the two Franks cells, the Franks prior's). Own distance = median of "
          "`own_distances`, Nm-eq. (matched cells on the matched scale, unmatched cells as "
          "stored).", ""]
    for src, tag in (("home", "matched"), ("obs", "unmatched, e-observer")):
        for cell in LANDMARK:
            r = H[(src, cell)]
            L += [f"**{LANDMARK_TITLE[cell]}, {tag}.**", "",
                  "| optimizer | win rate, k = 12 | k = 24 | k = 40 | own distance, k = 12 | "
                  "k = 24 | k = 40 |", "|---|---|---|---|---|---|---|"]
            for key, (_, _, label) in H_OPTS.items():
                L.append(f"| {label} | " + " | ".join(p3(r["win"][(key, k)]) for k in KS) + " | "
                         + " | ".join(p3(r["own"][(key, k)]) for k in KS) + " |")
            L += ["", "| contrast | win rate, k = 12 | k = 24 | k = 40 | own distance, k = 12 | "
                  "k = 24 | k = 40 |", "|---|---|---|---|---|---|---|"]
            for a_, b_ in H_CONTRASTS:
                L.append(f"| {H_OPTS[a_][2]} − {H_OPTS[b_][2]} | "
                         + " | ".join(ci3(r["con_win"][(a_, b_, k)]) for k in KS) + " | "
                         + " | ".join(ci3(r["con_own"][(a_, b_, k)]) for k in KS) + " |")
            L.append("")
    return L


def r_section(v: dict, side_: dict, side_after) -> list[str]:
    R = v["R"]
    u = R["umax"]
    L = ["## 4. Random smooth-utility wearer (R)", "",
         "**Wearer.** u(τ) = g(x(τ)) − ‖(x(τ) − c)/ρ‖²/2 on the first 6 Fourier harmonics, "
         "scaled per user to the plain s = 1.0 wearer's decisiveness; probit answers with lapse "
         "0.1. `own_distances` = −u_k and `own_generic_distance` = −u_gen, both scaled. The RMS "
         "fields of these rows (`distances`, `d_prior`, `d_floor`) refer to the e-main ideal and "
         "are not used, except in the equality check of §5.", "",
         "### 4.1 PR1 and PR2: own-utility win rate against generic assistance", "",
         "| optimizer | k = 12 | k = 24 | k = 40 |", "|---|---|---|---|"]
    for key, (_, _, label) in R_OPTS.items():
        L.append(f"| {label} | " + " | ".join(p3(R["win"][(key, k)]) for k in KS) + " |")
    L += ["", "| contrast (difference of means) | k = 12 | k = 24 | k = 40 |", "|---|---|---|---|"]
    for o in R_COMPARATORS:
        L.append(f"| A − {R_OPTS[o][2]} | " + " | ".join(ci3(R["con"][(o, k)]) for k in KS)
                 + " |")
    L += ["", f"Not registered: A − the best comparator at k = 24, the best taken within each "
          f"draw: {ci3(R['best'])}.", "",
          "### 4.2 Comparisons to reach a win rate of 0.60 and 0.75", "",
          "The first k (1–40) at which the mean win rate meets the level, on the cross-sectional "
          "curve (not a per-user first hit, not sustained), as `jnd_free.md` §5: first k "
          "[2.5, 97.5 % order statistics over the same draws] (share of draws that reach it by "
          "k = 40). \"> 40\": not reached.", "",
          "| optimizer | " + " | ".join(f"win rate ≥ {lv:.2f}" for lv in FIRST_K_LEVELS) + " |",
          "|---|" + "---|" * len(FIRST_K_LEVELS)]
    for key, (_, _, label) in R_OPTS.items():
        cells_ = []
        for lv in FIRST_K_LEVELS:
            p, lo, hi, reach = R["first"][(key, lv)]
            cells_.append(f"{fk(p)} [{fk(lo)}, {fk(hi)}] ({reach:.2f})")
        L.append(f"| {label} | " + " | ".join(cells_) + " |")
    snap = "; ".join(f"`{f['name']}` modified {f['mtime_iso']} (sidecar `converge_when` "
                     f"{f['converge_when']}, sha256 {f['sha'][:12]})" for f in side_["files"])
    changed = [n for n, same, _ in side_after if not same]
    L += ["", "### 4.3 Share of the achievable utility gain", "",
          "(u_k − u_gen) / (u_max − u_gen), median over users, read twice:",
          f"- **over all {u['n']} users, an upper bound** (ruling \"R, u_max\"): u_max is a "
          "best-found lower bound, so where the search fell short the denominator is too small "
          "and the share too large;",
          f"- **over the {u['conv']} users whose u_max converged** (`umax_converged`: the last two "
          "of up to 10 independent searches, none started from an earlier maximum, both reached "
          "within 1 % of headroom of the best value found; build report §6.2). For them u_max is "
          "reproducible to 1 % of headroom; it is still a best-found value. They are a selected "
          f"subset: the {u['n'] - u['conv']} others are the users whose maxima a search finds "
          f"only sometimes (median headroom {u['med_head_unconv']:.2f} against "
          f"{u['med_head_conv']:.2f} for the converged users). Per fold: "
          + ", ".join(f"{f} {n}" for f, n in u["conv_by_fold"].items())
          + "; the bootstrap resamples the converged users within their folds.", "",
          "The utility gain u_k − u_gen needs no u_max.", "",
          f"u_max as read at the start of this run: {snap}. "
          + ("Unchanged at the end of the run (same sha256)." if not changed else
             f"**Changed during the run: {', '.join(changed)}**; the shares use the values read "
             "at the start."), "",
          f"Users: {u['n']}; u_max converged for {u['conv']}; u_max source: "
          + ", ".join(f"{s} {n}" for s, n in u["sources"].items())
          + f". Median u_gen {u['med_ugen']:+.2f}, median u_max {u['med_umax']:+.2f}, median "
          f"headroom u_max − u_gen {u['med_head']:.2f} (minimum {u['min_head']:.2f}), scaled "
          f"utility units. Sessions whose identified assistance at any k beat the stored u_max: "
          f"{u['above']} of {u['n_sessions']}.", "",
          f"| optimizer | k | utility gain u_k − u_gen, median | share, all {u['n']} users "
          f"(upper bound) | share, {u['conv']} users with u_max converged |",
          "|---|---|---|---|---|"]
    for key, (_, _, label) in R_OPTS.items():
        for k in KS:
            L.append(f"| {label} | {k} | {ci3(R['gain'][(key, k)], 2)} | "
                     f"{ci3(R['share'][(key, k, 'all')], 2)} | "
                     f"{ci3(R['share'][(key, k, 'conv')], 2)} |")
    return L + [""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out
    report = args.report or out / "icorr" / "robust_report.md"
    sys.stdout.reconfigure(encoding="utf-8")
    if args.boot < 2000:
        raise SystemExit("at least 2000 bootstrap draws")
    run = out / "phase5-schaefer"
    started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")

    print("reading the u_max sidecars ...", flush=True)
    side_ = read_sidecars(run)
    for f in side_["files"]:
        print(f"  {f['name']}: modified {f['mtime_iso']}, converge_when {f['converge_when']}, "
              f"sha256 {f['sha'][:12]}", flush=True)

    print("streaming rows ...", flush=True)
    rows = load(out)
    sim = G.load_sim(run)
    keys = sim["keys"]
    ous = ood_users(run)
    sizes = {s: ous[s]["rms_median"] for s in O_SETS}
    sizes["pop"] = median_rms(sim["curves"])
    c = cells(rows, keys, {s: ous[s]["keys"] for s in O_SETS})
    folds = c[("main", "a")]["fold"].to_numpy()
    for key, df in c.items():
        if not np.array_equal(df["fold"].to_numpy(), folds):
            raise RuntimeError(f"{key}: user order differs")
    chk = checks(c, side_, ous, run)
    bad = [x for x in chk if not x[1] <= x[2]]
    if bad:
        raise RuntimeError(f"checks failed: {bad}")

    print("statistics ...", flush=True)
    v = analyse(c, folds, args.boot, side_)
    if v["draw_check"] != 0.0:
        raise RuntimeError(f"draw helper differs from joint_bootstrap by {v['draw_check']}")
    got = f"{v['main']['lead']['estimate']:.3f}"
    repro = [("main condition A − G at k = 24, RMS, Nm", got, MAIN_LEAD_EXPECTED)]
    for (cell, key), want in REPRO_OBSERVER.items():
        est = v["H"][("obs", cell)]["win"][(key, K_MAIN)]["estimate"]
        repro.append((f"e-observer {LANDMARK_TITLE[cell]}, {H_OPTS[key][2]} win rate at k = 24 "
                      "(jnd_free.md)", f"{est:.2f}", want))
    repro.append(("random-utility users with umax_converged (robust_build_report.md §6.2)",
                  str(v["R"]["umax"]["conv"]), CONVERGED_EXPECTED))
    stored = json.loads((out / "icorr" / "robust_ood_pb.json").read_text(encoding="utf-8"))
    for s, name in RMS_KEYS.items():
        repro.append((f"median ideal RMS, {s} users, Nm (robust_ood_pb.json)",
                      f"{sizes[s]:.9f}", f"{stored['magnitude'][name]:.9f}"))
    v["sizes"] = sizes
    if not all(g_ == w_ for _, g_, w_ in repro):
        print(f"REPRODUCTION FAILED: {repro}; nothing written", flush=True)
        return 1
    side_after = sidecars_unchanged(side_)
    dec = json.loads((out / "icorr" / "robust_decisiveness.json").read_text(encoding="utf-8"))

    L: list[str] = []
    p = L.append
    p("# ICORR robustness tests: home advantage at matched decisiveness (H), out-of-generator "
      "ideals (O) and the random smooth-utility wearer (R)")
    p("")
    p(f"Generated by `scripts/icorr_robust_report.py`, run started {started}, at worktree commit "
      f"{git_head()}. Spec: `docs/superpowers/specs/2026-10-17-icorr-user-tuning-reframe-"
      "design.md` §9a (PH1, PH2, PO1, PO2, PR1, PR2 registered before any run; rulings recorded "
      "after the build and before any outcome; the rulings this report adds on PO1 and PO2 are "
      "in §3.1). Build: `out/icorr/robust_build_report.md` (§6 for e-ood-pb and the u_max "
      "convergence). Rows: `out/icorr/e-home.jsonl`, `out/icorr/e-ood-pb.jsonl` (O, per-bout "
      "peaks, the registered verdicts), `out/icorr/e-ood.jsonl` (O, common peak, the labelled "
      "sensitivity run), `out/icorr/e-rand.jsonl`, `out/phase5e/e-main.jsonl` (A, G, LineCoSpar "
      "and SortCMA, the PH2 and PO2 references) and `out/phase5e/e-observer.jsonl` (the "
      "unmatched landmark cells, context). No session was run.")
    p("")
    p("**Users and condition.** H, R and the main condition: the 200 population users of the "
      "five LOSO folds (ruled: population users only; e-home's 20 stress users per cell are "
      "left out). O: the 200 out-of-generator users, 40 per fold, the same in both O blocks "
      "(§3). s = 1.0, λ = 0.1, data prior, tight arm, 40 comparisons read at k = 12 / 24 / 40. "
      "Within each test every cell holds the same users in the same order.")
    p("")
    p("**Estimator.** phase5e_report's, as `icorr_k_report.py` uses it for its verdicts: "
      "medians inverted_cdf (the lower middle value); a contrast is the difference of two "
      "medians (distances) or of two means (win rates) on the same users; 95 % intervals from a "
      f"fold-stratified joint cluster bootstrap ({args.boot} draws, seed {er.SEED}; one user draw "
      "per replicate shared by every array of a statistic, and the same draws for every "
      "statistic in this report). The one statistic across two user sets, PO2's D, resamples "
      f"the main condition's users with their own draws (seed {MAIN_DRAW_SEED}; §3.1). "
      "Verdicts: held when the interval lies wholly on the predicted side of the threshold, "
      "failed when wholly on the other side, uncertain otherwise. The win rate is "
      "P_i(k) = λ/2 + (1 − λ)·Φ((g_i − e_i(k))/s) in the wearer's own utility (for the RMS "
      "wearers of O and the main condition, g_i = d_prior and e_i(k) the RMS error), averaged "
      "over users; it needs no JND. Its ceiling at λ = 0.1 is 0.95.")
    p("")
    p("**Reproduction checks (passed).** " + "; ".join(f"{w}: {g_} (expected {e_})"
                                                       for w, g_, e_ in repro) + ".")
    p("")
    L.extend(verdict_lines(v))
    L.extend(h_section(v, dec))
    L.extend(o_section(v, ous, args.boot))
    L.extend(r_section(v, side_, side_after))
    p("## 5. Checks")
    p("")
    p("Every check passed (max |Δ| ≤ tolerance). The draw matrix rebuilt for §4.2 and PO2's D "
      "reproduces `joint_bootstrap`'s interval exactly.")
    p("")
    p("| check | max \\|Δ\\| | tolerance |")
    p("|---|---|---|")
    for what, d, tol in chk:
        p(f"| {what} | {d:.1e} | {tol:.0e} |")
    p("")
    p("## 6. Caveats")
    p("")
    ph2_states = [x["state"] for x in v["PH2"]["cells"].values()]
    moot = (" Here the rule is moot: both cells are " + ph2_states[0] + " on their own, so any "
            "combination (static only, as QE3's registered cell; both; either) gives the same "
            "verdict." if len(set(ph2_states)) == 1 else
            " The two cells differ (" + ", ".join(f"{c} {x['state']}" for c, x in
                                                 v["PH2"]["cells"].items())
            + "), so the combination rule matters.")
    p("- **PH2 names no cell.** PH1's registration says \"in both matched landmark cells\"; PH2's "
      "does not. It is computed in both, and the combined verdict (held only if held in both, "
      "failed if failed in either) is the analyst's rule, not a registered one. Each cell's "
      "verdict is in §2.2." + moot)
    p("- **u_max moved before this run, not during it.** The u_max values here (median "
      f"{v['R']['umax']['med_umax']:+.2f}) are after the convergence run that ended "
      f"{side_['files'][0]['converge_when']} (build report §6.2), which raised u_max for 24 "
      "users. The version of this report generated at 16:34 read them after the 15:35 "
      "refinement, before that run, and its shares are superseded. The sidecars were unchanged "
      "between the start and the end of this run (§4.3).")
    p("- **PO2 compares two user sets.** The O advantage is measured on the out-of-generator "
      "users, the main-condition advantage on the population users; both are RMS errors in Nm "
      "against each user's own ideal, but the users' rooms differ (§3). The per-bout users' "
      "floors are lower than the population's: Schäfer's smooth profiles fit the 8-weight "
      "ProMP well (build report §1.2 and §6.1).")
    p("- **PO's win rate is the RMS wearer's** (exact here: the O wearer is the plain RMS wearer "
      "at s = 1.0), as in the main condition.")
    failed = [n for n in ("PO1", "PO2") if v[n]["state"] == "failed"]
    if failed:
        p(f"- **{' and '.join(failed)} {'is' if len(failed) == 1 else 'are'} reported as found** "
          "(spec §9a: \"If a prediction fails, it is reported as found\"), on the per-bout set as "
          "ruled. The common-peak run and the room reading in §1 are context, not a re-ruling.")
    p("- **PH2 compares two metrics.** The landmark lead is in the matched wearer's own distance "
      "(Nm-eq., ruled), the main lead in RMS Nm. They are on a like-for-like decisiveness scale, "
      "not the same distance.")
    p("- **Generic assistance differs between families.** The Franks cells' win rate is against "
      "the Franks prior's mean curve, ProMP's against ProMP's; A with Franks projection against "
      "G on Franks is a like-for-like contrast, and A on ProMP against either is not.")
    p("- **The win rate is a model probability** under the wearer that answered the session "
      "(exact for both new wearers), not a measured preference.")
    p("- **R's share of the achievable gain is an upper bound** over all users (u_max is the "
      "best found, not certified); over the converged users it rests on a u_max that "
      "independent searches reproduce, on a selected subset (§4.3). PR1 and PR2 use the win "
      "rate, which needs u_gen (exact) but not u_max.")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    for line in verdict_lines(v)[4:10]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
