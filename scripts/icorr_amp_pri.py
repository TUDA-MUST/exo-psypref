"""Test PRI: how much the results depend on the prior being right.

Usage:
  python scripts/icorr_amp_pri.py prc
  python scripts/icorr_amp_pri.py prep
  python scripts/icorr_amp_pri.py run [--workers 8]
  python scripts/icorr_amp_pri.py report
  python scripts/icorr_amp_pri.py posthoc

Three tests:
  PRC  Schaefer et al.'s 11 participants' preferred profiles under the fixed-level prior, and
       FULL's fixed-level results reweighted to their rooms (no sessions; ``prc``);
  PAM  preferred peaks not log-uniform (M1-M6, medium and wide ranges);
  PAD  a wrong shape prior (S1/S2 flexion peak later, S3 Schaefer-family shapes; S0 control)
       with static, adaptive (mean update, the prior counting as four users) and oracle priors.
``prep`` writes the shifted users' shapes and the oracle priors' moments. Outputs in
<out>/icorr/amp_free/: pri-prep-<fold>.npz, pri_sessions.jsonl (resumable), amp_pri_report.md,
amp_pri_prc.md; ``posthoc`` writes amp_pri_posthoc.md.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_full as F  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.basis import VonMisesBasis  # noqa: E402
from prompref.designs.device import DeviceDesign  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.experiment.run import run_session  # noqa: E402
from prompref.learning.arms import ARMS  # noqa: E402
from prompref.learning.space import LearnerSpace  # noqa: E402
from prompref.population.folds import loso_folds  # noqa: E402
from prompref.prior import Whitening  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.oracle import Observer, rms_distance  # noqa: E402

JND = true_jnd(AF.S)
LEVEL = F.LEVEL
#: PAM: the amplitude laws (section 21).
PAM_CONDS = {"M1 low cohort": ("lognormal", 0.2), "M2 central cohort": ("lognormal", 0.5),
             "M3 high cohort": ("lognormal", 0.8), "M4 flexion larger": ("ratio", "flex"),
             "M5 extension larger": ("ratio", "ext"), "M6 correlated": ("copula", 0.8)}
PAM_SETTINGS = ("medium", "wide")
PAM_CANDS = ("psypref", "psypref_matched", "cmaes_sch4", "g", "d_z", "cmaes")
COHORT_CV, RATIO, RHO = 0.2, 1.5, 0.8
#: PAD: the populations (flexion-peak shift in % GC; None: Schaefer-family shapes).
PAD_POPS = {"S0 users": 0.0, "S1 flexion +6": 6.0, "S2 flexion +12": 12.0, "S3 Schaefer shapes": None}
PAD_SETTINGS = ("wide", "fixed 12")
PAD_SINGLE = ("psypref_static", "psypref_oracle", "cmaes_sch4", "g")
N0 = 4                    # the prior counts as its four training subjects
N_ORACLE_SCHAEFER = 2000
BASIS = VonMisesBasis()


def _dir(out: Path) -> Path:
    return out / "icorr" / "amp_free"


# ----------------------------------------------------------------------------- shapes and laws

def draw_amps(law: tuple, lo: float, hi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` (extension, flexion) peak pairs from one of PAM's laws."""
    kind, par = law
    L, H = math.log(lo), math.log(hi)
    if kind == "lognormal":
        med, sd = L + par * (H - L), math.sqrt(math.log(1.0 + COHORT_CV ** 2))
        out = np.empty((n, 2))
        for i in range(n):
            for b in range(2):
                while True:
                    v = med + sd * rng.standard_normal()
                    if L <= v <= H:
                        out[i, b] = math.exp(v)
                        break
        return out
    if kind == "ratio":
        base = np.exp(rng.uniform(L, math.log(hi / RATIO), n))
        big = RATIO * base
        return np.column_stack([base, big]) if par == "flex" else np.column_stack([big, base])
    if kind == "copula":
        z = rng.multivariate_normal([0.0, 0.0], [[1.0, par], [par, 1.0]], n)
        return np.exp(L + norm.cdf(z) * (H - L))
    raise ValueError(kind)


def warp_flexion(curve: np.ndarray, delta: float) -> np.ndarray:
    """The curve with its flexion peak ``delta`` % GC later: a piecewise-linear phase warp
    through the extension peak (held), the flexion peak (moved) and the extension peak one
    cycle on, so the bout order and the zero crossings stay."""
    c = np.asarray(curve, dtype=float)
    base = c[:-1]
    e, f = float(np.argmin(base)), float(np.argmax(base))
    f_u = f if f > e else f + 100.0
    if not e < f_u + delta < e + 100.0:
        raise ValueError("the shift would pass the next extension peak")
    t = np.arange(100.0)
    t_u = e + (t - e) % 100.0
    s_u = np.interp(t_u, [e, f_u + delta, e + 100.0], [e, f_u, e + 100.0])
    out = np.interp(s_u % 100.0, np.arange(100.0), base, period=100.0)
    return np.r_[out, out[0]]


def _unit_rows(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neg, pos = np.minimum(curves, 0.0), np.maximum(curves, 0.0)
    return neg / -neg.min(axis=1, keepdims=True), pos / pos.max(axis=1, keepdims=True)


def _schaefer_shapes(rng, n: int):
    from prompref.designs.schaefer import SchaeferDesign
    from prompref.experiment.robust import TimingDistribution, draw_timing

    dist = TimingDistribution.load(_inputs_path())
    return [AF.unit_bouts(SchaeferDesign(1.0, 1.0).curve(draw_timing(rng, dist)[0])) for _ in range(n)]


def _inputs_path() -> Path:
    return ROOT / "out" / "schaefer" / "prior_inputs.json"


def prep(out: Path) -> None:
    """Per fold: S1/S2 users' shapes, and the oracle priors' training bouts' weights."""
    d = _dir(out)
    for fi, fold in enumerate(loso_folds()):
        z = F._users(str(d / f"full-{fold.name}.npz"))
        dd = R._fold(str(d / f"fold-{fold.name}.npz"), fi)
        train = (dd["w_minus"] + dd["w_plus"]) @ BASIS.design.T        # (n, 101) unit shapes
        save = {}
        for pop, delta in PAD_POPS.items():
            if delta in (None, 0.0):
                continue
            tag = pop.split()[0]
            users = np.array([warp_flexion(z["user_minus"][u] + z["user_plus"][u], delta)
                              for u in range(AF.FULL_USERS_PER_FOLD)])
            um, up = _unit_rows(users)
            save[f"{tag}_user_minus"], save[f"{tag}_user_plus"] = um, up
            tm, tp = _unit_rows(np.array([warp_flexion(c, delta) for c in train]))
            save[f"{tag}_w_minus"], save[f"{tag}_w_plus"] = BASIS.project(tm.T).T, BASIS.project(tp.T).T
        rng = np.random.default_rng([AF.SEED_FULL, 2300, fi])
        sh = _schaefer_shapes(rng, N_ORACLE_SCHAEFER)
        save["S3_w_minus"] = BASIS.project(np.array([s[0] for s in sh]).T).T
        save["S3_w_plus"] = BASIS.project(np.array([s[1] for s in sh]).T).T
        np.savez(d / f"pri-prep-{fold.name}.npz", **save)
        print(f"{fold.name}: prep written", flush=True)


@lru_cache(maxsize=None)
def _prep(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


@lru_cache(maxsize=None)
def _rob2_shapes(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- spaces

def _range_amps(fi: int, setting: str, n: int) -> np.ndarray:
    """R._a_moments' log-uniform training peaks (the same stream)."""
    lo, hi = F.RANGES[setting]
    return AF._loguniform(R._range_rng(fi, 200 + list(R.RANGES).index(setting)), lo, hi, (n, 2))


def moments(setting: str, fi: int, ppath: str, out: Path, kind: str = "static", pop: str | None = None,
            cond: str | None = None):
    """Prior mean and covariance: FULL's (static), the population's training bouts (oracle),
    or the training shapes at the condition's peaks (PAM's matched prior)."""
    if kind == "static":
        if setting in F.RANGES:
            return R._a_moments(ppath, fi, setting)
        dd = R._fold(ppath, fi)
        w = dd["w_minus"] + dd["w_plus"]
        return w.mean(axis=0), np.cov(w, rowvar=False)
    if kind == "oracle":
        p = _prep(str(_dir(out) / f"pri-prep-{loso_folds()[fi].name}.npz"))
        tag = pop.split()[0]
        wm, wp = p[f"{tag}_w_minus"], p[f"{tag}_w_plus"]
        if setting in F.RANGES:
            return AF.weight_moments(wm, wp, _range_amps(fi, setting, wm.shape[0]))
        w = wm + wp
        return w.mean(axis=0), np.cov(w, rowvar=False)
    if kind == "matched":
        dd = R._fold(ppath, fi)
        lo, hi = F.RANGES[setting]
        ci, si = list(PAM_CONDS).index(cond), F.SETTINGS.index(setting)
        amps = draw_amps(PAM_CONDS[cond], lo, hi, dd["w_minus"].shape[0],
                         np.random.default_rng([AF.SEED_FULL, 2200, fi, ci, si]))
        return AF.weight_moments(dd["w_minus"], dd["w_plus"], amps)
    raise ValueError(kind)


def space_for(setting: str, mu, sigma) -> LearnerSpace:
    if setting in F.RANGES:
        return AF.promp_space(mu, sigma, limit=F.RANGES[setting][1])
    k = mu.size
    design = DeviceDesign(AF.FixedLevelDesign(LEVEL, basis=BASIS), limit=LEVEL + 1e-9)
    return LearnerSpace(design, Whitening(mu, sigma, k), ARMS["tight"].radius(k))


# ----------------------------------------------------------------------------- sessions

def _shape(job: dict, out: Path):
    d = _dir(out)
    u, pop = job["index"], job.get("condition", "")
    if job["test"] in ("PAD", "PAN") and pop.startswith(("S1", "S2")):
        p = _prep(str(d / f"pri-prep-{job['fold']}.npz"))
        tag = pop.split()[0]
        return p[f"{tag}_user_minus"][u], p[f"{tag}_user_plus"][u]
    if job["test"] in ("PAD", "PAN") and pop.startswith("S3"):
        sh = _rob2_shapes(str(d / f"rob2-shapes-{job['fold']}.npz"))
        return sh["schaefer_minus"][u], sh["schaefer_plus"][u]
    z = F._users(str(d / f"full-{job['fold']}.npz"))
    return z["user_minus"][u], z["user_plus"][u]


def session(job: dict, out: Path, mu_override=None, sigma_override=None) -> tuple[dict, np.ndarray]:
    """One session; returns the row and the delivered curve."""
    d = _dir(out)
    setting, cand, fi = job["setting"], job["candidate"], job["fold_index"]
    ppath = str(d / f"fold-{job['fold']}.npz")
    run_dir = out / "phase5-schaefer"
    ideal = AF.compose(job["amps"][0], job["amps"][1], *_shape(job, out))
    dd = R._fold(ppath, fi)
    g_minus, g_plus = AF.unit_bouts(dd["generic"])
    seed = int(job["answer_seed"]) % 2**32
    if setting in F.RANGES:
        lo, hi = F.RANGES[setting]
        centre = math.sqrt(lo * hi)
    else:
        lo, hi, centre = None, LEVEL, LEVEL
    if cand.startswith("psypref"):
        kind = {"psypref": "static", "psypref_static": "static", "psypref_oracle": "oracle",
                "psypref_matched": "matched", "psypref_adaptive": "static", "psypref_pan": "static"}[cand]
        mu, sigma = moments(setting, fi, ppath, out, kind, pop=job.get("condition"), cond=job.get("condition"))
        if mu_override is not None:
            mu = mu_override
        if sigma_override is not None:
            sigma = sigma_override
        space = space_for(setting, mu, sigma)
        learner = (AF.TwoBoutGuard(AF.metric_psypref(space, seed), space, 0.5 * lo, seed=seed)
                   if setting in F.RANGES else AF.psypref(space, seed))
    else:
        mu, sigma = moments(setting, fi, ppath, out, "static")
        space = space_for(setting, mu, sigma)
        if cand.startswith("cmaes_sch"):
            space = (AF.schaefer_range_space(run_dir, job["fold"], lo, hi) if setting in F.RANGES
                     else AF.schaefer_fixed_space(run_dir, job["fold"], LEVEL))
        learner = AF.comparator(cand, space, seed)
    generic = AF.compose(centre, centre, g_minus, g_plus)
    observer = Observer(ideal_nm=ideal, s=AF.S, lapse=AF.LAPSE)
    rng = np.random.default_rng([int(job["answer_seed"]), AF.OBSERVER_SALT])
    rec = run_session(space.design, learner, observer, rng, n_comparisons=F.N_COMP,
                      randomise_order=True, d_prior=rms_distance(generic, ideal), d_floor=0.0)
    delivered = space.design.curve(np.asarray(rec["best"], dtype=float))
    row = dict({k: v for k, v in job.items() if not k.startswith("_")},
               d0=rms_distance(generic, ideal), distances=list(rec["distances"]), error=None)
    return row, delivered


def run_one(job: dict, out: str) -> list[dict]:
    out_p = Path(out)
    if job.get("chain"):
        return run_chain(job, out_p)
    try:
        return [session(job, out_p)[0]]
    except Exception as exc:  # one bad session must not stop the run
        return [dict({k: v for k, v in job.items() if not k.startswith("_")},
                     error=f"{type(exc).__name__}: {exc}")]


def run_chain(job: dict, out: Path) -> list[dict]:
    """PAD's adaptive arm: one fold's users in index order, the prior's mean updated after
    each from the identified (delivered) curves; covariance unchanged."""
    d = _dir(out)
    setting, fi = job["setting"], job["fold_index"]
    ppath = str(d / f"fold-{job['fold']}.npz")
    mu0, _ = moments(setting, fi, ppath, out, "static")
    scale = 1.0 if setting in F.RANGES else 1.0 / LEVEL
    ident, rows = [], []
    for u in range(AF.FULL_USERS_PER_FOLD):
        sj = dict(job["_users"][u], candidate="psypref_adaptive", position=u)
        mu = (N0 * mu0 + np.sum(ident, axis=0)) / (N0 + len(ident)) if ident else mu0
        try:
            row, delivered = session(sj, out, mu_override=mu)
            ident.append(BASIS.project(delivered * scale))
            row["mu_shift"] = float(np.linalg.norm(mu - mu0))
        except Exception as exc:  # the chain goes on without this user's curve
            row = dict({k: v for k, v in sj.items() if not k.startswith("_")},
                       error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def _base(fold, fi, u, setting, z):
    amps = [LEVEL, LEVEL] if setting == "fixed 12" else [float(a) for a in z[f"amps_{setting}"][u]]
    return dict(fold=fold.name, fold_index=fi, index=u, user=f"{fold.name}/f{u}", setting=setting,
                amps=amps, answer_seed=int(z["answer_seeds"][u]) + F.SETTINGS.index(setting))


def jobs(out: Path) -> list[dict]:
    d = _dir(out)
    chains, slow, fast = [], [], []
    for fi, fold in enumerate(loso_folds()):
        z = F._users(str(d / f"full-{fold.name}.npz"))
        for setting in PAD_SETTINGS:
            for pop in PAD_POPS:
                users = [dict(_base(fold, fi, u, setting, z), test="PAD", condition=pop)
                         for u in range(AF.FULL_USERS_PER_FOLD)]
                chains.append(dict(test="PAD", condition=pop, setting=setting, fold=fold.name,
                                   fold_index=fi, candidate="psypref_adaptive", chain=True, _users=users))
                if pop == "S0 users":
                    continue        # S0's static arm is FULL's; its oracle is static
                for j in users:
                    for cand in PAD_SINGLE:
                        (slow if cand.startswith("psypref") else fast).append(dict(j, candidate=cand))
        for setting in PAM_SETTINGS:
            lo, hi = F.RANGES[setting]
            si = F.SETTINGS.index(setting)
            for ci, (cond, law) in enumerate(PAM_CONDS.items()):
                amps = draw_amps(law, lo, hi, AF.FULL_USERS_PER_FOLD,
                                 np.random.default_rng([AF.SEED_FULL, 2100, fi, ci, si]))
                for u in range(AF.FULL_USERS_PER_FOLD):
                    j = dict(_base(fold, fi, u, setting, z), test="PAM", condition=cond,
                             amps=[float(a) for a in amps[u]])
                    for cand in PAM_CANDS:
                        (slow if cand.startswith("psypref") else fast).append(dict(j, candidate=cand))
    return chains + slow + fast


def _key(r):
    return r["test"], r["condition"], r["setting"], r["candidate"], r["user"]


def run(out: Path, workers: int) -> None:
    path = _dir(out) / "pri_sessions.jsonl"
    done = set()
    if path.exists():
        done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = []
    for j in jobs(out):
        if j.get("chain"):
            keys = {(j["test"], j["condition"], j["setting"], "psypref_adaptive", u["user"]) for u in j["_users"]}
            if not keys <= done:
                todo.append(j)
        elif _key(j) not in done:
            todo.append(j)
    print(f"{len(todo)} jobs to run ({sum(1 for j in todo if j.get('chain'))} chains)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            for row in fut.result():
                f.write(json.dumps(row) + "\n")
                if row.get("error"):
                    print("ERROR", _key(row), row["error"], flush=True)
            f.flush()
            if n % 200 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- PRC

def prc(out: Path) -> None:
    import schaefer_grounding as G
    from prompref.designs.schaefer import SchaeferDesign

    d = _dir(out)
    parts = json.loads(_inputs_path().read_text(encoding="utf-8"))["participants"]
    curves = [AF.compose(LEVEL, LEVEL, *AF.unit_bouts(SchaeferDesign(1.0, 1.0).curve(np.array(p["preferred_t"]))))
              for p in parts]
    r_part = np.zeros((len(parts), 5))
    d0_part = np.zeros((len(parts), 5))
    floor_part = np.array([AF.promp_floor(c, 8, level=LEVEL) for c in curves])
    r_sim, floor_sim, radius = [], [], []
    for fi, fold in enumerate(loso_folds()):
        ppath = str(d / f"fold-{fold.name}.npz")
        mu, sigma = moments("fixed 12", fi, ppath, out, "static")
        space = space_for("fixed 12", mu, sigma)
        radius.append(space.radius)
        dd = R._fold(ppath, fi)
        generic = AF.compose(LEVEL, LEVEL, *AF.unit_bouts(dd["generic"]))
        for i, c in enumerate(curves):
            r_part[i, fi] = float(np.linalg.norm(space.to_z(BASIS.project(c / LEVEL))))
            d0_part[i, fi] = rms_distance(generic, c)
        z = F._users(str(d / f"full-{fold.name}.npz"))
        for u in range(AF.FULL_USERS_PER_FOLD):
            c = AF.compose(LEVEL, LEVEL, z["user_minus"][u], z["user_plus"][u])
            r_sim.append(float(np.linalg.norm(space.to_z(BASIS.project(c / LEVEL)))))
            floor_sim.append(AF.promp_floor(c, 8, level=LEVEL))
    r_sim, floor_sim = np.array(r_sim), np.array(floor_sim)
    r_med = np.median(r_part, axis=1)
    inside = bool(np.all(r_part <= np.array(radius)[None, :]))
    q75 = float(np.percentile(r_sim, 75))
    prc1 = "held" if inside else "failed"
    prc2 = "held" if float(np.median(r_med)) <= q75 else "failed"
    # PRC3: FULL's fixed-level rows reweighted to the participants' rooms
    targets = np.median(d0_part, axis=1)
    rows = [r for r in map(json.loads, (d / "full_sessions.jsonl").read_text(encoding="utf-8").splitlines())
            if r.get("error") is None and r["setting"] == "fixed 12"]
    by = {}
    for r in rows:
        by.setdefault(r["candidate"], {})[r["user"]] = r
    users = sorted(by["psypref"])
    folds = np.array([by["psypref"][u]["fold"] for u in users])
    d0 = np.array([by["psypref"][u]["d0"] for u in users])
    pref = {c: np.array([AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((by[c][u]["d0"] - by[c][u]["distances"][23]) / AF.S)
                         for u in users]) for c in by if set(users) <= set(by[c])}

    def wmean(w, x):
        return float(np.sum(w * x) / np.sum(w))

    w0 = G.kde_weights(d0, targets)
    rng = np.random.default_rng(AF.SEED_FULL + 40)
    idx_by_fold = [np.flatnonzero(folds == f) for f in np.unique(folds)]
    draws = {c: [] for c in pref}
    diffs = {c: [] for c in pref if c != "psypref"}
    ess = []
    for _ in range(R.N_BOOT):
        idx = np.concatenate([rng.choice(ix, size=ix.size) for ix in idx_by_fold])
        tg = targets[rng.integers(targets.size, size=targets.size)]
        w = G.kde_weights(d0[idx], tg)
        ess.append(G.ess(w))
        for c in pref:
            draws[c].append(wmean(w, pref[c][idx]))
        for c in diffs:
            diffs[c].append(wmean(w, pref["psypref"][idx] - pref[c][idx]))
    ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))  # noqa: E731
    L = ["# Registration PRC: real preferences under the prior", "",
         f"Spec section 21 (e2a079f). Schäfer et al.'s {len(parts)} participants' preferred timings as "
         f"their family's curve at {LEVEL:.0f} Nm; FULL's 200 users at {LEVEL:.0f} Nm. Radius r = |z| "
         f"under the fixed-level prior (K = 8); the 99 % search ball has radius {np.mean(radius):.2f}.", "",
         "| | median r | 75th / 90th percentile r | max r | median ProMP floor (Nm) | median distance to generic (Nm) |",
         "|---|---|---|---|---|---|",
         f"| simulated users | {np.median(r_sim):.2f} | {q75:.2f} / {np.percentile(r_sim, 90):.2f} | "
         f"{r_sim.max():.2f} | {np.median(floor_sim):.2f} | {np.median(d0):.2f} |",
         f"| participants (median over folds) | {np.median(r_med):.2f} | "
         f"{np.percentile(r_med, 75):.2f} / {np.percentile(r_med, 90):.2f} | {r_med.max():.2f} | "
         f"{np.median(floor_part):.2f} | {np.median(targets):.2f} |", "",
         "Participants, r (median over folds) and its percentile among the simulated users: "
         + ", ".join(f"{p['id']} {r:.2f} ({100 * np.mean(r_sim <= r):.0f} %)" for p, r in zip(parts, r_med)), "",
         f"- PRC1 (every participant inside the 99 % ball under every fold's prior): {prc1}",
         f"- PRC2 (participants' median r at most the simulated users' 75th percentile, {q75:.2f}): {prc2}"
         f" — {np.median(r_med):.2f}", "",
         "## PRC3: fixed 12 Nm reweighted to the participants' rooms", "",
         f"Targets: the participants' distances to generic assistance at 12 Nm, {np.min(targets):.2f}–"
         f"{np.max(targets):.2f} Nm (median {np.median(targets):.2f}); simulated users' median "
         f"{np.median(d0):.2f} Nm. KDE weights (schaefer_grounding.kde_weights); effective sample size "
         f"{G.ess(w0):.0f} of {len(users)} (draws: median {np.median(ess):.0f}). Preference at k = 24.", "",
         "| optimizer | as simulated | reweighted | PsyPref − it, reweighted (points) |", "|---|---|---|---|"]
    verdicts = []
    for c in sorted(pref, key=lambda c: -np.mean(pref[c])):
        lo_, hi_ = ci(draws[c])
        cell = ""
        if c in diffs:
            dl, dh = ci(diffs[c])
            est = 100 * wmean(w0, pref["psypref"] - pref[c])
            cell = f"{est:+.1f} [{100 * dl:+.1f}, {100 * dh:+.1f}]"
            verdicts.append("held" if dl > 0 else ("failed" if dh < 0 else "uncertain"))
        L.append(f"| {F.NAMES.get(c, c)} | {np.mean(pref[c]):.3f} | {wmean(w0, pref[c]):.3f} "
                 f"[{lo_:.3f}, {hi_:.3f}] | {cell} |")
    v = "held" if all(x == "held" for x in verdicts) else ("failed" if "failed" in verdicts else "uncertain")
    L += ["", f"- PRC3 (PsyPref ahead of every comparator, reweighted): {v} {verdicts}", ""]
    (d / "amp_pri_prc.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# ----------------------------------------------------------------------------- report

def _p(r, k=24) -> float:
    return float(AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((r["d0"] - r["distances"][k - 1]) / AF.S))


def _verdict(t, side="above", bound=0.0):
    est, lo, hi = t
    if side == "above":
        return "held" if lo > bound else ("failed" if hi < bound else "uncertain")
    return "held" if hi < bound else ("failed" if lo > bound else "uncertain")


def _pairs(a: dict, b: dict, keep=lambda u: True):
    return [dict(fold=a[u]["fold"], a=a[u], b=b[u]) for u in sorted(set(a) & set(b)) if keep(a[u])]


def _diff(pairs, rng):
    return R._boot(pairs, lambda x: 100 * float(np.mean([_p(p["a"]) - _p(p["b"]) for p in x])), rng)


def report(out: Path) -> None:
    d = _dir(out)
    raw = [json.loads(x) for x in (d / "pri_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in rows]
    full = {(r["setting"], r["user"]): r for r in map(json.loads, (d / "full_sessions.jsonl").read_text(
        encoding="utf-8").splitlines()) if r.get("error") is None and r["candidate"] == "psypref"}
    rng = np.random.default_rng(AF.SEED_FULL + 41)

    def cell(test, cond, setting, cand):
        if test == "PAD" and cond == "S0 users" and cand in ("psypref_static", "psypref_oracle"):
            return {u: r for (s, u), r in full.items() if s == setting}
        return {k[4]: r for k, r in rows.items() if k[:4] == (test, cond, setting, cand)}

    L = ["# Registration PRI: PAM and PAD", "",
         f"Spec section 21 (e2a079f). {len(rows)} sessions, {len(errors)} errors. Preference over "
         f"generic assistance at k = 24 (s = {AF.S} Nm, λ = {AF.LAPSE}); paired differences in points; "
         f"95 % intervals from {R.N_BOOT} draws, users resampled within folds.", "",
         "## PAM: preferred peaks not log-uniform (prior log-uniform)", "",
         "| condition | range | PsyPref | matched prior | CMA-ES (Schäfer) | best published | PsyPref − CMA-ES (Schäfer) "
         "| PsyPref − best published | matched − PsyPref |", "|---|---|---|---|---|---|---|---|---|"]
    v1, v2, cost = [], [], []
    for cond in PAM_CONDS:
        for setting in PAM_SETTINGS:
            c = {cand: cell("PAM", cond, setting, cand) for cand in PAM_CANDS}
            if not c["psypref"]:
                continue
            mean = {k: float(np.mean([_p(r) for r in v.values()])) for k, v in c.items() if v}
            pub = max((k for k in ("g", "d_z", "cmaes") if k in mean), key=lambda k: mean[k])
            dc = _diff(_pairs(c["psypref"], c["cmaes_sch4"]), rng)
            dp = _diff(_pairs(c["psypref"], c[pub]), rng)
            dm = _diff(_pairs(c["psypref_matched"], c["psypref"]), rng)
            v1.append(_verdict(dc))
            v2.append(_verdict(dp))
            cost.append(dm[0] < 5.0)
            L.append(f"| {cond} | {setting} | {mean['psypref']:.3f} | {mean.get('psypref_matched', float('nan')):.3f} | "
                     f"{mean['cmaes_sch4']:.3f} | {F.NAMES[pub]} {mean[pub]:.3f} | {R._f(dc, 1)} | {R._f(dp, 1)} | "
                     f"{R._f(dm, 1)} |")
    agg = lambda vs: "held" if vs and all(v == "held" for v in vs) else ("failed" if "failed" in vs else "uncertain")  # noqa: E731
    L += ["", f"- PAM1 (ahead of CMA-ES on Schäfer's family in all cells): {agg(v1)} {v1}",
          f"- PAM2 (ahead of the best published in all cells): {agg(v2)} {v2}",
          f"- PAM3 (matched − PsyPref < 5 points in ≥ 10 of 12 cells): "
          f"{'held' if sum(cost) >= 10 else 'failed'} — {sum(cost)} of {len(cost)}", "",
          "## PAD: a wrong shape prior and its adaptation", "",
          "| population | setting | static | oracle | adaptive, users 21–40 | oracle − static | adaptive − static, 21–40 "
          "| static − CMA-ES (Schäfer) |", "|---|---|---|---|---|---|---|---|"]
    pad1, pad2, pad3, pad4, curves = [], [], [], [], []
    late = lambda r: r["index"] >= 20  # noqa: E731
    for pop in PAD_POPS:
        for setting in PAD_SETTINGS:
            st, orc = cell("PAD", pop, setting, "psypref_static"), cell("PAD", pop, setting, "psypref_oracle")
            ad, cm = cell("PAD", pop, setting, "psypref_adaptive"), cell("PAD", pop, setting, "cmaes_sch4")
            if not st:
                continue
            m = lambda v, keep=lambda r: True: float(np.mean([_p(r) for r in v.values() if keep(r)])) if v else float("nan")  # noqa: E731
            do = _diff(_pairs(orc, st), rng) if pop != "S0 users" else None
            da = _diff(_pairs(ad, st, late), rng) if ad else None
            dc = _diff(_pairs(st, cm), rng) if cm else None
            if pop != "S0 users":
                pad1.append(_verdict(do))
            if pop in ("S2 flexion +12", "S3 Schaefer shapes") and da:
                pad2.append(_verdict(da))
            if pop == "S0 users" and da:
                pad3.append("held" if da[0] >= -2.0 else "failed")
            if pop in ("S1 flexion +6", "S2 flexion +12") and dc:
                pad4.append(_verdict(dc))
            L.append(f"| {pop} | {setting} | {m(st):.3f} | {m(orc):.3f} | {m(ad, late):.3f} | "
                     f"{R._f(do, 1) if do else '–'} | {R._f(da, 1) if da else '–'} | {R._f(dc, 1) if dc else '–'} |")
            if ad:
                blocks = []
                for b0 in (0, 10, 20, 30):
                    pb = _pairs(ad, st, lambda r, b0=b0: b0 <= r["index"] < b0 + 10)
                    blocks.append(float(np.mean([_p(p["a"]) - _p(p["b"]) for p in pb])) * 100 if pb else float("nan"))
                curves.append(f"| {pop} | {setting} | " + " | ".join(f"{b:+.1f}" for b in blocks) + " |")
    L += ["", "Adaptive − static by the user's position in the fold (points, mean over the 5 folds' users):", "",
          "| population | setting | users 1–10 | 11–20 | 21–30 | 31–40 |", "|---|---|---|---|---|---|"] + curves
    L += ["", f"- PAD1 (static below oracle in S1–S3, both settings): {agg(pad1)} {pad1}",
          f"- PAD2 (adaptive ahead of static for users 21–40 in S2, S3): {agg(pad2)} {pad2}",
          f"- PAD3 (S0: adaptive − static ≥ −2 points for users 21–40): {agg(pad3)} {pad3}",
          f"- PAD4 (static ahead of CMA-ES on Schäfer's family in S1, S2): {agg(pad4)} {pad4}", ""]
    if errors:
        L += ["## Errors", ""] + [f"- {_key(r)}: {r['error']}" for r in errors[:30]]
    (d / "amp_pri_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def posthoc(out: Path) -> None:
    """PAD, post hoc: oracle and adaptive against static on the same users (21-40), the oracle
    against CMA-ES on Schaefer's family, and where the oracle prior differs from the static one
    (mean, in the static prior's SDs, against covariance)."""
    rows = [r for r in map(json.loads, (_dir(out) / "pri_sessions.jsonl").read_text(encoding="utf-8").splitlines())
            if r["test"] == "PAD" and r["error"] is None]
    by = {(r["condition"], r["setting"], r["candidate"], r["user"]): r for r in rows}
    rng = np.random.default_rng(AF.SEED_FULL + 31)

    def pref(r):
        return AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((r["d0"] - r["distances"][23]) / AF.S)

    def paired(cond, setting, a, b, first=0):
        items = [dict(fold=by[cond, setting, a, u]["fold"], v=pref(by[cond, setting, a, u]) - pref(by[cond, setting, b, u]))
                 for (c, s, cand, u) in by if c == cond and s == setting and cand == a
                 and (cond, setting, b, u) in by and by[cond, setting, a, u]["index"] >= first]
        return R._f(R._boot(items, lambda x: 100 * float(np.mean([i["v"] for i in x])), rng), 1)

    pops = [p for p in PAD_POPS if PAD_POPS[p] != 0.0]
    L = ["# Registration PRI, PAD: post hoc", "",
         "Not registered. Preference over generic assistance at k = 24; paired differences in points; 95 % "
         f"intervals from {R.N_BOOT} draws, users resampled within folds. Users 21–40: the adaptive prior "
         "has seen at least 20 identified users.", "",
         "| population | setting | oracle − static, 21–40 | adaptive − static, 21–40 | adaptive − oracle, 21–40 "
         "| oracle − CMA-ES (Schäfer), all |", "|---|---|---|---|---|---|"]
    for p in pops:
        for s in PAD_SETTINGS:
            L.append(f"| {p} | {s} | {paired(p, s, 'psypref_oracle', 'psypref_static', 20)} | "
                     f"{paired(p, s, 'psypref_adaptive', 'psypref_static', 20)} | "
                     f"{paired(p, s, 'psypref_adaptive', 'psypref_oracle', 20)} | "
                     f"{paired(p, s, 'psypref_oracle', 'cmaes_sch4')} |")
    L += ["", "Oracle against static prior, medians over the 5 folds: the mean difference in the static "
          "prior's SDs (Mahalanobis), the adaptive mean's shift after the fold's last user, and the "
          "covariance's trace and log-determinant ratios.", "",
          "| population | setting | mean difference (static SDs) | adaptive shift / oracle shift | trace ratio | "
          "log-det ratio |", "|---|---|---|---|---|---|"]
    for p in pops:
        for s in PAD_SETTINGS:
            mah, shift, tr, ld = [], [], [], []
            for fi, fold in enumerate(loso_folds()):
                ppath = str(_dir(out) / f"fold-{fold.name}.npz")
                mst, cst = moments(s, fi, ppath, out, "static")
                mor, cor = moments(s, fi, ppath, out, "oracle", pop=p)
                dm = mor - mst
                last = max((r for r in rows if r["condition"] == p and r["setting"] == s
                            and r["candidate"] == "psypref_adaptive" and r["fold"] == fold.name),
                           key=lambda r: r["position"])
                mah.append(float(np.sqrt(dm @ np.linalg.solve(cst, dm))))
                shift.append(last["mu_shift"] / float(np.linalg.norm(dm)))
                tr.append(float(np.trace(cor) / np.trace(cst)))
                ld.append(float(np.linalg.slogdet(cor)[1] - np.linalg.slogdet(cst)[1]))
            L.append(f"| {p} | {s} | {np.median(mah):.2f} | {np.median(shift):.2f} | {np.median(tr):.2f} | "
                     f"{np.median(ld):+.1f} |")
    (_dir(out) / "amp_pri_posthoc.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("prc", "prep", "run", "report", "posthoc"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"prc": lambda: prc(args.out), "prep": lambda: prep(args.out),
     "run": lambda: run(args.out, args.workers), "report": lambda: report(args.out),
     "posthoc": lambda: posthoc(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
