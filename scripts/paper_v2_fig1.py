"""Fig. 1 of the amplitude-free ICORR paper (tikz_v2): the numbers of every plotted curve, as CSV.

Usage:
  python scripts/paper_v2_fig1.py
      [--out <project>/out] [--csv figures/data]

The figure is the method diagram of the old Fig. 1 (fig1_method_silhouette) in the
amplitude-free design: the study sets the
peak-torque range, the walking-data prior carries the shape only, and PsyPref-T reads "no
preference" answers as data (sections 26, 26a). Its curves come from ONE new PsyPref session,
run here and checked against FULL's stored row.

The session
  * Setting: wide, 3-35 Nm. In this setting both peak torques are free over a wide range, so the
    belief snapshots show PsyPref learning shape and peak torques together; generic assistance
    (both peaks at sqrt(3 x 35) = 10.25 Nm) differs from the preferred profile in level as well
    as in shape.
  * User, by rule: among FULL's 200 PsyPref sessions in that setting
    (out/icorr/amp_free/full_sessions.jsonl), ranked by the distance to the preferred profile
    after 24 comparisons (equal distances: user order), the lower of the two middle users,
    rank 100 of 200. With an even count no user has the median distance, and the two middle
    users lie equally far from it (loso-AB31/f20 and /f32, both 0.0068 Nm), so "closest to the
    median" would not pick one; the lower middle user is the one of the two that flatters
    PsyPref less.
  * Built exactly as FULL's run_one builds it (scripts/icorr_amp_full.py: AM's space and prior
    over the range, curve metric N1, two-part guard, 40 comparisons, FULL's answer seed and
    observer), with PsyPref-T's factory: job["ties"] = "band", i.e. AF.metric_psypref(...,
    ties="band"). FULL's simulated users never answer "no preference", and PsyPref-T then equals
    PsyPref bit for bit, so the session must reproduce FULL's stored distances exactly.
    The script stops if any of the 40 distances, the identified peaks, d0, the shape error or
    the guard's counts differ from the stored row.
  * Recording: run_one is called unchanged; run_session is wrapped for the call only, and the
    wrapper hands run_session a recorder around the learner that reads the posterior and passes
    every call through. It draws no random number of the session's streams.

Written (in --csv, every file prefixed fig1_):
  fig1_curves.csv     phase (% gait cycle); preferred (the user's preferred profile, Nm); generic
                      (generic assistance); pair1, pair2 (the pair asked after six answers, no
                      longer drawn: fig1.tex takes its pair from paper_v2_fig1j.py);
                      mean_k0, mean_k6, mean_k24 (the profile PsyPref would deliver after k
                      answers: the posterior mean of z*, clipped into the trust search region, through
                      the two-part guard; after 24 answers the identified profile)
  fig1_fan_k<k>.csv   phase, d0..d19: 20 particles of the posterior after k = 0, 6, 24 answers
                      (systematic resampling by weight, seed FAN_SEED), as curves
  fig1_prior.csv      phase, s0..s29: 30 draws of the walking-data shape prior (the fold's unit
                      training shapes' weights, PsyPref's ideal-point model prior, parts at unit peaks, as
                      AF.fixed_level_space at a level of 1); generic_shape: generic assistance's
                      shape at unit peaks
  fig1_sigmoid.csv    delta = d(2) - d(1) in Nm, p = P(choose 1) of the main condition's
                      simulated users (s = 1.0 Nm, lapse 0.1)
  fig1_meta.csv       limits of the small axes, k values, the user, the setting, the check
  fig1_check.csv      k, the session's distance, FULL's stored distance, equal (1/0)
  fig1_walker_part<i>.csv, fig1_walker_parts.csv
                      the walker silhouette of fig1_method.walker at unit height
                      (paper_icorr_tikz_export.walker_geometry, unchanged geometry), its colours
                      mapped to icorr_style.tex's names

Nothing in out/ is written. design sections used: 12 (the study sets the range), 16 (FULL's users
and settings), 26/26a (PsyPref-T), 27 (PsyPref-T's factory as a named option).
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    # as FULL's workers ran: one BLAS thread, so the session reproduces the stored floats
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_full as FULL  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.learning.arms import ARMS  # noqa: E402
from prompref.user.simulated_user import rms_distance  # noqa: E402

SETTING = "wide"
K_RULE = 24              # the rule's comparison count (the paper's budget)
KS = (0, 6, 24)          # belief snapshots
K_PAIR = 6               # the pair shown to the user: the one asked after six answers
K_ID = 24                # identified profile and validation
N_FAN = 20               # particles per snapshot
N_PRIOR = 30             # prior sample shapes
FAN_SEED = 20261029
PRIOR_SEED = 20261030
S_MAIN, LAPSE = AF.S, AF.LAPSE
SIG_X = 2.8              # the sigmoid's x range, +-Nm (as the old Fig. 1)
PRIOR_PAD = 1.3          # the prior plot's y limits: +-PRIOR_PAD x the largest |torque| drawn
DEFAULT_CSV = ROOT / "figures" / "data"
#: the old Fig. 1's walker files, read only, to confirm the geometry is unchanged
OLD_WALKER = Path("figures/earlier_walker_not_included")
#: the walker's colours (fig1_method.BODY_FRONT, BODY_BACK, INK, white) as icorr_style.tex names
WALKER_COLOURS = {"9C9C9C": "icGeneric", "C9C9C9": "icContext", "222222": "icInk",
                  "FFFFFF": "white", "none": "none"}


# ---------------------------------------------------------------- helpers
def num(v) -> str:
    if isinstance(v, str):
        return v
    v = float(v)
    if v != v:
        return "nan"
    return f"{v:.10g}"


def write_csv(path: Path, header: list[str], rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([num(v) for v in r])
    return path


def systematic(weights: np.ndarray, n: int, rng) -> np.ndarray:
    """n particle indices in proportion to their weights (fig1_concepts._draws)."""
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(weights), u), weights.size - 1)


# ---------------------------------------------------------------- the user
def choose_user(out: Path) -> tuple[dict, dict]:
    """FULL's PsyPref row of the lower of the two middle users by distance after K_RULE
    comparisons (rank n/2 of n; equal distances in user order), and the rule's numbers."""
    path = FULL._dir(out) / "full_sessions.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    rows = sorted((r for r in rows if r.get("error") is None and r["candidate"] == "psypref"
                   and r["setting"] == SETTING), key=lambda r: r["user"])
    if len(rows) != 5 * AF.FULL_USERS_PER_FOLD:
        raise SystemExit(f"expected 200 PsyPref rows in {SETTING}, found {len(rows)}")
    dk = np.array([r["distances"][K_RULE - 1] for r in rows])
    med = float(np.median(dk))
    order = np.argsort(dk, kind="stable")    # stable: equal distances stay in user order
    rank = len(rows) // 2                    # the lower middle user (1-based rank)
    i, upper = int(order[rank - 1]), int(order[rank])
    if not dk[i] <= med <= dk[upper]:
        raise SystemExit("the middle users do not straddle the median")
    return rows[i], dict(n=len(rows), median=med, d=float(dk[i]), rank=rank,
                         upper_user=rows[upper]["user"], upper_d=float(dk[upper]))


# ---------------------------------------------------------------- the recorded session
class Recorder:
    """Passes every call to ``learner`` (FULL's TwoPartGuard) and records the pairs, the
    answers, the delivered parameters and the posterior after the answers in KS. It reads the
    posterior and the inner learner's deterministic best only; the guard's own stream is
    touched only by run_session's own best calls."""

    def __init__(self, learner) -> None:
        self._learner = learner
        self.k = 0
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.answers: list[int] = []
        self.delivered: dict[int, np.ndarray] = {}
        self.snap: dict[int, dict] = {}
        self._snapshot()

    def _inner(self):
        return self._learner.learner if isinstance(self._learner, AF.TwoPartGuard) else self._learner

    def _snapshot(self) -> None:
        inner = self._inner()
        post = inner.posterior
        d = inner.space.d
        self.snap[self.k] = dict(z=np.array(post.x[:, :d], dtype=float), w=post.weights().copy(),
                                 ell=post.ell_mean(), lam=post.lambda_mean(),
                                 best_inner=np.asarray(inner.best(), dtype=float).copy(),
                                 tie_active=bool(getattr(inner, "tie_active", False)))

    def propose(self):
        a, b = self._learner.propose()
        self.pairs.append((np.array(a, dtype=float), np.array(b, dtype=float)))
        return a, b

    def update(self, theta_a, theta_b, response: int) -> None:
        self._learner.update(theta_a, theta_b, response)
        self.answers.append(int(response))
        self.k += 1
        if self.k in KS:
            self._snapshot()

    def best(self):
        theta = self._learner.best()
        # the first call after k answers is the one run_session scores
        self.delivered.setdefault(self.k, np.array(theta, dtype=float))
        return theta

    def __getattr__(self, name):
        return getattr(self._learner, name)


def run_recorded(job: dict, out: Path) -> tuple[dict, dict]:
    """FULL.run_one on ``job`` with run_session wrapped; returns its row and the record."""
    captured: dict = {}
    original = FULL.run_session

    def recording(design, learner, observer, rng, **kw):
        rec = Recorder(learner)
        captured.update(rec=rec, design=design, observer=observer, learner=learner)
        return original(design, rec, observer, rng, **kw)

    FULL.run_session = recording
    try:
        row = FULL.run_one(dict(job), str(out))
    finally:
        FULL.run_session = original
    if row.get("error"):
        raise SystemExit(f"the session failed: {row['error']}")
    return row, captured


def check(row: dict, stored: dict) -> list[list]:
    """Stop unless the session reproduces FULL's stored row; the per-k table."""
    a, b = row["distances"], stored["distances"]
    table = [[k + 1, x, y, int(x == y)] for k, (x, y) in enumerate(zip(a, b))]
    bad = [k for k, x, y, e in table if not e]
    if len(a) != len(b) or bad:
        raise SystemExit(f"distances differ from FULL's stored row at k = {bad[:10]} "
                         f"({len(a)} against {len(b)} comparisons)")
    # every other field the stored row has (d_generic came after FULL's run and is not stored)
    for key in ("identified", "d0", "shape_error", "guard_fired", "guard_final"):
        if key not in stored or row.get(key) != stored[key]:
            raise SystemExit(f"{key}: {row.get(key)} against FULL's {stored.get(key)}")
    return table


# ---------------------------------------------------------------- walker
def walker(dest: Path) -> str:
    """fig1_method.walker at unit height through the old export's walker_geometry, renamed and
    its colours mapped to style names; the geometry is compared with the old Fig. 1's files."""
    import paper_icorr_tikz_export as TX

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        TX.walker_geometry(tmp, [])
        n = len(TX.WALKER_PARTS)
        same = 0
        for i in range(n):
            text = (tmp / f"walker_part{i}.csv").read_text(encoding="utf-8")
            (dest / f"fig1_walker_part{i}.csv").write_text(text, encoding="utf-8")
            old = OLD_WALKER / f"walker_part{i}.csv"
            same += int(old.exists() and old.read_text(encoding="utf-8") == text)
        with (tmp / "walker_parts.csv").open(encoding="utf-8") as fh:
            parts = list(csv.DictReader(fh))
    rows = []
    for p in parts:
        edge, fill = WALKER_COLOURS[p["edge"]], WALKER_COLOURS[p["fill"]]
        rows.append([p["part"], p["name"], p["kind"], p["lw_per_height"], edge, fill,
                     p["closed"], p["cap"]])
    write_csv(dest / "fig1_walker_parts.csv",
              ["part", "name", "kind", "lw_per_height", "edge", "fill", "closed", "cap"], rows)
    return f"walker: {n} parts; {same} of {n} geometry files equal to the old Fig. 1's"


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    dest = args.csv
    dest.mkdir(parents=True, exist_ok=True)

    stored, rule = choose_user(args.out)
    job = next(j for j in FULL.jobs(args.out) if (j["user"], j["setting"], j["candidate"])
               == (stored["user"], SETTING, "psypref"))
    job["ties"] = "band"     # PsyPref-T: AF.metric_psypref(..., ties="band") in run_one
    row, cap = run_recorded(job, args.out)
    table = check(row, stored)
    rec, design, observer, learner = cap["rec"], cap["design"], cap["observer"], cap["learner"]
    from prompref.learning.ties import TieMetricPsyPref
    if not (isinstance(learner, AF.TwoPartGuard) and isinstance(learner.learner, TieMetricPsyPref)
            and learner.learner.tie_mode == "band"):
        raise SystemExit(f"the session's learner is not PsyPref-T (band) behind the two-bout "
                         f"guard: {type(learner).__name__}/{type(getattr(learner, 'learner', None)).__name__}")
    space = learner.space
    ideal = np.asarray(observer.ideal_nm, dtype=float)
    phase = np.linspace(0.0, 100.0, ideal.size)

    def curve(theta) -> np.ndarray:
        return np.asarray(design.curve(np.asarray(theta, dtype=float)), dtype=float)

    # generic assistance as run_one builds it, checked against the stored d0
    lo, hi = FULL.RANGES[SETTING]
    centre = math.sqrt(lo * hi)
    d = R._fold(job["_parts"], job["fold_index"])
    g_minus, g_plus = AF.unit_parts(d["generic"])
    generic = AF.compose(centre, centre, g_minus, g_plus)
    if rms_distance(generic, ideal) != stored["d0"]:
        raise SystemExit("generic assistance does not give the stored d0")

    # the profile PsyPref would deliver after k answers
    mean = {}
    for k in KS:
        if k == 0:
            theta = rec.snap[0]["best_inner"]
            if min(AF.peaks(curve(theta))) < learner.min_peak:
                raise SystemExit("k = 0: the prior mean lacks a bout (the guard would fire)")
        else:
            theta = rec.delivered[k]
            if rms_distance(curve(theta), ideal) != stored["distances"][k - 1]:
                raise SystemExit(f"k = {k}: the recorded curve is not the scored one")
        mean[k] = curve(theta)
    ident_peaks = [max(x, 0.0) for x in AF.peaks(mean[K_ID])]

    # the pair asked after K_PAIR answers (comparison K_PAIR + 1)
    ta, tb = rec.pairs[K_PAIR]
    pair1, pair2 = curve(ta), curve(tb)
    answer = rec.answers[K_PAIR]
    d1, d2 = rms_distance(pair1, ideal), rms_distance(pair2, ideal)

    # posterior particles
    rng = np.random.default_rng(FAN_SEED)
    fans = {}
    for k in KS:
        s = rec.snap[k]
        idx = systematic(s["w"], N_FAN, rng)
        fans[k] = np.array([curve(space.to_theta(z)) for z in s["z"][idx]])

    # the walking-data shape prior at unit peaks
    unit = AF.fixed_level_space(d["w_minus"], d["w_plus"], 1.0)
    zp = ARMS["tight"].ideal_point_prior().sample_zstar(np.random.default_rng(PRIOR_SEED), N_PRIOR,
                                                  unit.d)
    prior = np.array([np.asarray(unit.design.curve(unit.to_theta(z)), dtype=float) for z in zp])
    generic_shape = AF.compose(1.0, 1.0, g_minus, g_plus)

    # ---- CSVs
    cols = [phase, ideal, generic, pair1, pair2] + [mean[k] for k in KS]
    write_csv(dest / "fig1_curves.csv",
              ["phase", "preferred", "generic", "pair1", "pair2"] + [f"mean_k{k}" for k in KS],
              np.column_stack(cols))
    for k in KS:
        write_csv(dest / f"fig1_fan_k{k}.csv", ["phase"] + [f"d{i}" for i in range(N_FAN)],
                  np.column_stack([phase, fans[k].T]))
    write_csv(dest / "fig1_prior.csv",
              ["phase"] + [f"s{i}" for i in range(N_PRIOR)] + ["generic_shape"],
              np.column_stack([phase, prior.T, generic_shape]))
    delta = np.round(np.arange(-SIG_X, SIG_X + 1e-9, 0.02), 10)
    from scipy.stats import norm
    p = LAPSE / 2 + (1 - LAPSE) * norm.cdf(delta / S_MAIN)
    write_csv(dest / "fig1_sigmoid.csv", ["delta", "p"], np.column_stack([delta, p]))
    write_csv(dest / "fig1_check.csv", ["k", "distance", "stored", "equal"], table)

    # limits: one for the session's small axes (every curve drawn in them), one for the prior
    drawn = np.concatenate([np.ravel(c) for c in cols[1:]] + [fans[k].ravel() for k in KS])
    pad = 0.04 * float(np.ptp(drawn))
    ymin, ymax = float(np.min(drawn)) - pad, float(np.max(drawn)) + pad
    # the prior's shapes share unit peaks: a margin of 30 % keeps the common level visibly a
    # level, not a clip, and the extension troughs off the plot's lower edge
    pmax = PRIOR_PAD * float(np.max(np.abs(np.concatenate([prior.ravel(), generic_shape]))))
    n_ties = sum(a == 2 for a in rec.answers)
    meta = dict(lapse=LAPSE, s=S_MAIN, sig_xmin=-SIG_X, sig_xmax=SIG_X, sig_ymin=-0.03,
                sig_ymax=1.03, k0=KS[0], k1=KS[1], k2=KS[2], k_pair=K_PAIR, k_id=K_ID,
                xmin=0, xmax=100, ymin=ymin, ymax=ymax, prior_ymin=-pmax, prior_ymax=pmax,
                range_lo=lo, range_hi=hi, generic_peak=centre, user=stored["user"],
                setting=SETTING, amp_ext=job["amps"][0], amp_flex=job["amps"][1],
                id_ext=ident_peaks[0], id_flex=ident_peaks[1], d0=stored["d0"],
                d_k6=stored["distances"][K_PAIR - 1], d_k24=stored["distances"][K_ID - 1],
                rule_median=rule["median"], rule_rank=rule["rank"], rule_n=rule["n"],
                pair_d1=d1, pair_d2=d2, pair_answer=answer, n_no_preference=n_ties,
                reproduced=len(table))
    write_csv(dest / "fig1_meta.csv", list(meta), [list(meta.values())])
    note = walker(dest)

    print(f"user {stored['user']} ({SETTING}, rule: the lower of the two middle users by "
          f"distance after {K_RULE} comparisons, rank {rule['rank']} of {rule['n']}): "
          f"d{K_RULE} = {rule['d']:.6f} Nm; median {rule['median']:.6f} Nm; the upper middle "
          f"user {rule['upper_user']} at {rule['upper_d']:.6f} Nm")
    print(f"preferred peaks {job['amps'][0]:.2f} / {job['amps'][1]:.2f} Nm (ext / flex); "
          f"identified at k = {K_ID}: {ident_peaks[0]:.2f} / {ident_peaks[1]:.2f} Nm; generic "
          f"{centre:.2f} Nm; d0 {stored['d0']:.3f}, d6 {stored['distances'][5]:.3f}, d24 "
          f"{stored['distances'][23]:.3f} Nm")
    print(f"reproduction: PsyPref-T ({type(learner.learner).__name__}, ties='band', behind "
          f"TwoBoutGuard) reproduced FULL's stored distances at all "
          f"{len(table)} comparisons bit for bit, and identified, d0, shape_error, "
          f"guard_fired ({row.get('guard_fired')}), guard_final")
    print(f"answers: {n_ties} 'no preference' of {len(rec.answers)}; tie model active at the "
          f"end: {rec.snap[K_ID]['tie_active']}")
    print(f"pair after {K_PAIR} answers: d(1) {d1:.3f}, d(2) {d2:.3f} Nm, answer "
          f"{['profile 1', 'profile 2', 'no preference'][answer]}")
    for k in KS:
        s = rec.snap[k]
        print(f"k = {k}: distance of the delivered curve {rms_distance(mean[k], ideal):.3f} Nm, "
              f"ell {s['ell']:.3f}, lapse {s['lam']:.3f}")
    print(f"axes: session {ymin:.2f} to {ymax:.2f} Nm, prior +-{pmax:.3f} (unit peaks)")
    print(note)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
