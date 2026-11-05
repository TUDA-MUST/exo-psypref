"""ICORR W3 and W5: the basis count K, random duels, and Ramella's learner in the main condition.

and section 4 (W5). Build and block e-k:
out/icorr/w3_build_report.md. No session is run.

Usage:
    python scripts/icorr_k_report.py [--out <project>/out]
        [--data <Schaefer share>] [--boot 2000] [--boot-verdict 10000]
        [--report <out>/icorr/k_sweep.md]

Reads, all read-only, each jsonl streamed line by line, population users only
(``user_kind == "population"``), as planned:
  OUT/icorr/e-k.jsonl        K = 4 and 12 for A, G and SortCMA; A with random duels at K = 4, 8
                             and 12; C2 (Ramella's learner on ProMP, K = 8); C1 (Ramella as
                             published, semi-trapezoid, box arm, flat prior);
  OUT/phase5e/e-main.jsonl   K = 8 for A, G and SortCMA (the stored rows, as section 9a says);
  OUT/phase5-schaefer/users/<fold>-promp_s0-k{4,12}.json, to check each row's best-profile distance;
  through icorr_jnd_free.room_setup, the simulated users and the Schaefer logs (aggregates only).

Two estimators, each as the script it comes from:
  verdicts (PW3a-PW3d)  phase5e_report's: medians are ``_med`` (inverted_cdf, the lower middle
                        value); a contrast is the difference of the two medians on the same users;
                        intervals come from ``joint_bootstrap`` (users resampled within folds, one
                        draw set for every array of a statistic, and the same draws for every
                        statistic, since every call resamples the same 200 users with the same
                        seed; 95 % ends as observed values); held / failed / uncertain by
                        ``_side`` (held when the interval lies wholly on the predicted side of the
                        threshold, failed when wholly on the other, uncertain otherwise).
  tables (W3, W5)       icorr_jnd_free's: ``Rooms.stat`` (weighted Hazen medians, means), its
                        shared draws (seed 20261009) and KDE weights. Every K and optimizer
                        carries each user's weight from the K = 8 ProMP room (e-main A's
                        d_prior - d_floor), so a room reweights the same users the same way at
                        every K. With equal weights the Hazen median of 200 users is the mean of
                        the two middle values, so it can differ from the verdicts' lower middle
                        value in the third decimal.

Per K, generic assistance is that K's prior mean (the row's d_prior) and the best-profile distance is that K's own
best-profile distance (the row's d_floor); both move with K. C1 lives in its own family: it is compared in
absolute error and in the win rate against the ProMP K = 8 generic assistance only.
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

import benchmark_report as bm  # noqa: E402
import icorr_jnd_free as jf  # noqa: E402
import phase5e_report as er  # noqa: E402
import schaefer_grounding as G  # noqa: E402
import short_budget_report as sb  # noqa: E402

KS = (12, 24, 40)
K_MAIN = 24
BASES = (4, 8, 12)
N_BOOT = jf.N_BOOT            # tables, icorr_jnd_free's draws
N_BOOT_VERDICT = 10_000       # verdicts, phase5e_report's default
TABLE_ROOMS = ("simulated", "central")
MARGIN = 0.10                 # Nm, PW3a and PW3b
WIN_LEVEL = 0.75              # PW3d and the first-k level
#: K = 8 at the simulated room must reproduce jnd_free.md section 1 as printed there (a check,
#: not a result): the estimate formatted with icorr_jnd_free's digits.
REPRO_K8 = {("win", 24): "0.82", ("err", 24): "1.712", ("jnd2", 24): "0.72", ("err", 12): "2.009"}
#: Row fields kept: icorr_jnd_free's, and W3's two point fields and G's JND scale.
FIELDS = jf.KEEP + ("n_basis", "duel_mode", "jnd_scale")
#: The four W3 optimizers: key -> (learner, duel mode, label).
OPTS = {"a": ("a", "eig", "A"), "a_rand": ("a", "random", "A, random duels"),
        "g": ("g", "eig", bm.label("ProMP", "G")), "cmaes": ("cmaes", "eig", "SortCMA")}
C2_LABEL = bm.label("ProMP", "C2")
C1_LABEL = bm.label("ProMP", "C1")
TABLE_MEASURES = ("win", "share", "err", "jnd1", "jnd2")


# ============================================================================ the rows
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """W3's two point fields with their defaults filled in: ``n_basis`` None or absent means 8
    (stored rows predate the field), ``duel_mode`` None or absent means "eig"."""
    df = df.copy()
    nb = df["n_basis"] if "n_basis" in df else pd.Series(np.nan, index=df.index)
    df["n_basis"] = pd.to_numeric(nb, errors="coerce").fillna(8).astype(int)
    dm = df["duel_mode"] if "duel_mode" in df else pd.Series(None, index=df.index, dtype=object)
    df["duel_mode"] = dm.astype(object).where(dm.notna(), "eig").astype(str)
    return df


def load(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(e-k rows, e-main ProMP rows), population users only, normalised."""
    ek = jf.stream(out / "icorr" / "e-k.jsonl", fields=FIELDS)
    em = jf.stream(out / "phase5e" / "e-main.jsonl",
                   keep=lambda r: jf._pop_row(r) and r.get("family") == "promp_s0",
                   fields=FIELDS)
    if ek is None or em is None:
        raise SystemExit("e-k.jsonl or e-main.jsonl is missing")
    return normalise(sb._n2048(ek)), normalise(sb._n2048(sb._popn(em)))


def cell(df: pd.DataFrame, family: str, learner: str, n_basis: int, duel: str, arm: str = "tight",
         prior: str = "data") -> pd.DataFrame:
    """One cell of the main condition (s = 1.0, lapse 0.1, phase5e_report's defaults)."""
    mask = ((df["family"] == family) & (df["learner"] == learner) & (df["n_basis"] == n_basis)
            & (df["duel_mode"] == duel) & (df["arm"] == arm) & (df["prior"] == prior)).to_numpy()
    want = dict(er.DEFAULTS)
    want.update(s=1.0)
    for name, value in want.items():
        mask &= er._match(df, name, value)
    if learner == "g":
        mask &= er._match(df, "jnd_scale", 1.0)
    out = df[mask]
    if out.duplicated(er.KEY).any():
        raise ValueError(f"{family} {learner} K = {n_basis} {duel}: more than one row per user")
    if not len(out):
        raise ValueError(f"{family} {learner} K = {n_basis} {duel}: no rows")
    return out


def cells(ek: pd.DataFrame, em: pd.DataFrame, keys) -> dict:
    """Every cell aligned to the 200 users of load_sim: (K, optimizer key) for W3, and "c2",
    "c1" for W5."""
    out = {}
    for K in BASES:
        for key, (learner, duel, _) in OPTS.items():
            src = em if K == 8 and duel == "eig" else ek
            out[(K, key)] = jf.in_order(cell(src, "promp_s0", learner, K, duel), keys)
    out["c2"] = jf.in_order(cell(ek, "promp_s0", "c2", 8, "eig"), keys)
    out["c1"] = jf.in_order(cell(ek, "trapezoid", "c1", 8, "eig", arm="box", prior="flat"), keys)
    return out


def errors(df: pd.DataFrame) -> np.ndarray:
    """(users, 40) RMS error of the identified assistance after each comparison."""
    d = np.array([v[:jf.KMAX] for v in df["distances"]], dtype=float)
    if d.shape[1] < jf.KMAX:
        raise ValueError("a row holds fewer than 40 comparisons")
    return d


def col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].astype(float).to_numpy()


# ============================================================================ checks
def sidecar_floors(out: Path, K: int) -> dict:
    floors = {}
    for path in sorted((out / "phase5-schaefer" / "users").glob(f"*-promp_s0-k{K}.json")):
        for uid, v in json.loads(path.read_text(encoding="utf-8")).items():
            floors[uid] = float(v["floor_nm"])
    return floors


def checks(c: dict, out: Path) -> list[tuple[str, float]]:
    """(what, max |difference| in Nm): every check must be 0 (to 1e-12)."""
    res = []
    ref = c[(8, "a")]
    for K in BASES:
        base = c[(K, "a")]
        for key in OPTS:
            if key == "a":
                continue
            for field in ("d_prior", "d_floor"):
                res.append((f"K = {K}: {field} of {OPTS[key][2]} equals A's",
                            float(np.max(np.abs(col(c[(K, key)], field) - col(base, field))))))
        if K != 8:
            side = sidecar_floors(out, K)
            res.append((f"K = {K}: d_floor equals users/<fold>-promp_s0-k{K}.json floor_nm",
                        float(np.max(np.abs(col(base, "d_floor")
                                            - np.array([side[u] for u in base["user_id"]]))))))
    for field in ("d_prior", "d_floor"):
        res.append((f"C2: {field} equals e-main A's",
                    float(np.max(np.abs(col(c["c2"], field) - col(ref, field))))))
    return res


# ============================================================================ verdicts
def med(x):
    """phase5e_report._med along the users axis of a (draws, users) matrix."""
    return er._med(x, axis=1)


def diff_of_medians(x, y):
    return med(x) - med(y)


def lead_over_best(a, g, c):
    """The best comparator's median minus A's, the best taken within each draw (the smaller of
    the two medians): positive = A ahead."""
    return np.minimum(med(g), med(c)) - med(a)


def lead_difference(a4, g4, c4, a12, g12, c12):
    """PW3b's statistic: A's lead over the best comparator at K = 12 minus that at K = 4."""
    return lead_over_best(a12, g12, c12) - lead_over_best(a4, g4, c4)


def combine(states) -> str:
    """One prediction from its parts (each "held", "failed" or None = straddles): failed if any
    part failed, else uncertain if any straddles, else held."""
    states = list(states)
    if "failed" in states:
        return "failed"
    if any(s is None for s in states):
        return "uncertain"
    return "held"


def state(ci: dict, threshold: float, want_low: bool) -> str:
    s = er._side(ci, threshold, want_low)
    return "uncertain" if s is None else s


class Boot:
    """phase5e_report.joint_bootstrap on the 200 aligned users; every call draws the same users."""

    def __init__(self, folds, n_boot: int):
        self.folds, self.n_boot = np.asarray(folds), n_boot

    def __call__(self, arrays, stat) -> dict:
        return er.joint_bootstrap(arrays, self.folds, stat, self.n_boot)


def verdicts(c: dict, boot: Boot) -> dict:
    E = {key: errors(df) for key, df in c.items()}

    def at(key, k):
        return E[key][:, k - 1]

    v = {}
    # PW3a
    a24 = boot([at((8, "a"), 24), at((4, "a"), 24)], diff_of_medians)
    a40 = boot([at((8, "a"), 40), at((4, "a"), 40)], diff_of_medians)
    a12 = boot([at((8, "a"), 12), at((4, "a"), 12)], diff_of_medians)
    k12_8 = {k: boot([at((12, "a"), k), at((8, "a"), k)], diff_of_medians) for k in KS}
    v["PW3a"] = dict(d24=a24, d40=a40, d12=a12, k12_8=k12_8,
                     s24=state(a24, MARGIN, want_low=True), s40=state(a40, 0.0, want_low=True),
                     meds={(K, k): float(er._med(at((K, "a"), k))) for K in BASES for k in KS})
    v["PW3a"]["state"] = combine([er._side(a24, MARGIN, True), er._side(a40, 0.0, True)])
    # PW3b
    leads, best, fixed = {}, {}, {}
    for K in BASES:
        leads[K] = boot([at((K, "a"), K_MAIN), at((K, "g"), K_MAIN), at((K, "cmaes"), K_MAIN)],
                        lead_over_best)
        mg, mc = float(er._med(at((K, "g"), K_MAIN))), float(er._med(at((K, "cmaes"), K_MAIN)))
        best[K] = ("g", mg, mc) if mg <= mc else ("cmaes", mg, mc)
    d = boot([at((4, "a"), K_MAIN), at((4, "g"), K_MAIN), at((4, "cmaes"), K_MAIN),
              at((12, "a"), K_MAIN), at((12, "g"), K_MAIN), at((12, "cmaes"), K_MAIN)],
             lead_difference)
    # sensitivity: the best comparator fixed at each K by its point median (phase5e's ranking)
    fixed = boot([at((4, "a"), K_MAIN), at((4, best[4][0]), K_MAIN),
                  at((12, "a"), K_MAIN), at((12, best[12][0]), K_MAIN)],
                 lambda a4, b4, a12, b12: (med(b12) - med(a12)) - (med(b4) - med(a4)))
    v["PW3b"] = dict(leads=leads, best=best, d=d, fixed=fixed,
                     state=state(d, -MARGIN, want_low=False),
                     fixed_state=state(fixed, -MARGIN, want_low=False),
                     meds={K: float(er._med(at((K, "a"), K_MAIN))) for K in BASES})
    # PW3c
    eig = {}
    for K in BASES:
        for k in KS:
            eig[(K, k)] = boot([at((K, "a"), k), at((K, "a_rand"), k)], diff_of_medians)
    v["PW3c"] = dict(c=eig, s8=state(eig[(8, K_MAIN)], 0.0, True),
                     s12=state(eig[(12, K_MAIN)], 0.0, True),
                     s4=state(eig[(4, K_MAIN)], 0.0, True),
                     meds={(K, key): float(er._med(at((K, key), K_MAIN)))
                           for K in BASES for key in ("a", "a_rand")})
    v["PW3c"]["state"] = combine([er._side(eig[(8, K_MAIN)], 0.0, True),
                                  er._side(eig[(12, K_MAIN)], 0.0, True)])
    # PW3d
    df12 = c[(12, "a")]
    s, lam = col(df12, "s"), col(df12, "lapse")
    e = at((12, "a"), K_MAIN)
    own = jf.win_probability(col(df12, "d_prior"), e, s, lam)
    common = jf.win_probability(col(c[(8, "a")], "d_prior"), e, s, lam)
    mean = lambda p: p.mean(axis=1)  # noqa: E731
    w_own, w_common = boot([own], mean), boot([common], mean)
    v["PW3d"] = dict(own=w_own, common=w_common, state=state(w_own, WIN_LEVEL, want_low=False),
                     common_state=state(w_common, WIN_LEVEL, want_low=False))
    # W5, in the verdicts' convention, for comparison with the 5c numbers (R57 at s = 0.5)
    v["W5"] = {(other, k): boot([at((8, "a"), k), at(other, k)], diff_of_medians)
               for other in ("c2", "c1") for k in KS}
    v["W5_meds"] = {(key, k): float(er._med(at(key, k))) for key in ("c2", "c1") for k in KS}
    return v


# ============================================================================ tables
class KArm(jf.Arm):
    """icorr_jnd_free.Arm for a W3 or W5 cell of the main condition. ``gen`` replaces the
    generic distance the win rate is taken against (C1, and the K = 8 generic sensitivity)."""

    def __init__(self, df: pd.DataFrame, name: str, fam_name: str = "ProMP", gen=None):
        super().__init__(er.BY_ID["main"], "promp_s0", "a", df)
        self.name, self.fam_name = name, fam_name
        if gen is not None:
            self.gen = np.asarray(gen, dtype=float)


def table_stats(c: dict, rooms) -> dict:
    res = {}
    dp8 = col(c[(8, "a")], "d_prior")
    for K in BASES:
        for key, (_, _, label) in OPTS.items():
            arm = KArm(c[(K, key)], label)
            res[(K, key)] = jf.compute_arm(arm, rooms, TABLE_ROOMS)
            if key == "a":
                res[(K, "generic")] = jf.compute_generic(arm, rooms, TABLE_ROOMS)
            # sensitivity: the win rate against the K = 8 generic assistance
            if K != 8:
                alt = KArm(c[(K, key)], label, gen=dp8)
                res[(K, key, "gen8")] = {
                    (room, k): (rooms.stat("win", alt.per_user(k)["win"], room))
                    for room in TABLE_ROOMS for k in KS}
        print(f"  K = {K} done", flush=True)
    res["c2"] = jf.compute_arm(KArm(c["c2"], C2_LABEL), rooms, TABLE_ROOMS)
    res["c1"] = jf.compute_arm(KArm(c["c1"], C1_LABEL, fam_name="semi-trapezoid", gen=dp8),
                               rooms, TABLE_ROOMS)
    res["c1_floor"] = {room: rooms.stat("err", col(c["c1"], "d_floor"), room)
                       for room in TABLE_ROOMS}
    res["c1_start"] = {room: rooms.stat("err", col(c["c1"], "d_prior"), room)
                       for room in TABLE_ROOMS}
    return res


def paired(ra: dict, rb: dict, room: str, m: str, k: int) -> tuple:
    """ra minus rb at k, paired by icorr_jnd_free's draws."""
    est = ra["at"][(room, m, k)][0] - rb["at"][(room, m, k)][0]
    b = ra["boot"][(room, m, k)].astype(float) - rb["boot"][(room, m, k)].astype(float)
    return (est, *jf.ci(b))


# ============================================================================ formatting
def nm(d: dict, digits: int = 3) -> str:
    return f"{jf.fs(d['estimate'], digits)} [{jf.fs(d['lo'], digits)}, {jf.fs(d['hi'], digits)}]"


def p2(d: dict) -> str:
    return f"{d['estimate']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def git_head() -> str:
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "scripts/icorr_k_report.py"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f"{head}{' (script uncommitted)' if dirty else ''}, script sha256 {sha}"
    except (OSError, subprocess.CalledProcessError):
        return f"unknown, script sha256 {sha}"


def room_title(setup, room: str) -> str:
    if room == "simulated":
        return "simulated room (as stored)"
    return f"central literature room, {setup['level'][room]:.2f} Nm (KDE-reweighted)"


def k_context(c: dict) -> list[str]:
    """Per K: generic distance, best-profile distance, room, and how they compare with K = 8."""
    out = ["| K | generic distance d_prior, median (Nm) | floor d_floor, median (Nm) | room "
           "d_prior − d_floor, median (Nm) | users with no room | generic further than at K = 8 "
           "| floor above K = 8's |", "|---|---|---|---|---|---|---|"]
    dp8, fl8 = col(c[(8, "a")], "d_prior"), col(c[(8, "a")], "d_floor")
    for K in BASES:
        dp, fl = col(c[(K, "a")], "d_prior"), col(c[(K, "a")], "d_floor")
        room = dp - fl
        out.append(f"| {K} | {np.median(dp):.3f} | {np.median(fl):.3f} | {np.median(room):.3f} | "
                   f"{int(np.sum(room <= bm.NO_ROOM))} | "
                   + ("—" if K == 8 else f"{int(np.sum(dp > dp8 + 1e-12))} of {len(dp)} "
                      f"(median {np.median(dp - dp8):+.3f} Nm)") + " | "
                   + ("—" if K == 8 else f"{int(np.sum(fl > fl8 + 1e-12))} of {len(fl)} "
                      f"(median {np.median(fl - fl8):+.3f} Nm)") + " |")
    return out


def verdict_section(v: dict) -> list[str]:
    a, b, cc, d = v["PW3a"], v["PW3b"], v["PW3c"], v["PW3d"]
    name = {"g": OPTS["g"][2], "cmaes": "SortCMA"}
    out = ["## 1. Planned predictions PW3a–PW3d (spec §9a)", "",
           f"Errors are RMS Nm at the simulated room. Δ = the difference of the two medians on the "
           f"same 200 users (medians inverted_cdf), 95 % fold-stratified joint bootstrap "
           f"({N_BOOT_VERDICT} draws, the same user draws for every statistic). A negative error "
           "difference favours the first. Verdicts by the planned rule: held when the "
           "interval lies wholly on the predicted side of the threshold, failed when wholly on "
           "the other side, uncertain otherwise.", ""]
    # PW3a
    m = a["meds"]
    out += [f"- **PW3a: {a['state']}.** (i) k = 24, non-inferiority: A at K = 8 minus A at K = 4 "
            f"= {nm(a['d24'])} Nm (medians {m[(8, 24)]:.3f} and {m[(4, 24)]:.3f}); predicted "
            f"≤ +{MARGIN:.2f} → {a['s24']}. (ii) k = 40: {nm(a['d40'])} Nm (medians "
            f"{m[(8, 40)]:.3f} and {m[(4, 40)]:.3f}); predicted < 0 → {a['s40']}. For reference, "
            f"k = 12: {nm(a['d12'])} Nm. Not planned, what K = 12 does: A at K = 12 minus A "
            f"at K = 8: " + "; ".join(f"k = {k} {nm(a['k12_8'][k])} Nm" for k in KS)
            + f" (K = 12 medians {m[(12, 12)]:.3f} / {m[(12, 24)]:.3f} / {m[(12, 40)]:.3f} Nm)."]
    # PW3b
    lead_txt = "; ".join(
        f"K = {K} {nm(b['leads'][K])} Nm (best comparator {name[b['best'][K][0]]}: medians "
        f"{OPTS['g'][2]} {b['best'][K][1]:.3f}, SortCMA {b['best'][K][2]:.3f}; A {b['meds'][K]:.3f})"
        for K in BASES)
    out += [f"- **PW3b: {b['state']}.** A's lead over the best of {OPTS['g'][2]} and SortCMA at "
            f"k = 24 (the best comparator's median minus A's, the smaller of the two comparator "
            f"medians taken within each draw; positive = A ahead): {lead_txt}. Lead at K = 12 "
            f"minus lead at K = 4 = {nm(b['d'])} Nm; predicted ≥ −{MARGIN:.2f} → {b['state']}. "
            f"With the best comparator fixed at each K by its point median (phase5e's ranking) "
            f"instead: {nm(b['fixed'])} Nm → {b['fixed_state']}."]
    # PW3c
    c_ = cc["c"]
    out += [f"- **PW3c: {cc['state']}.** A (EIG duels) minus A (random duels) at k = 24: "
            f"K = 8 {nm(c_[(8, 24)])} Nm → {cc['s8']}; K = 12 {nm(c_[(12, 24)])} Nm → "
            f"{cc['s12']}; predicted < 0 at both (interval excluding 0). Not planned: K = 4 "
            f"{nm(c_[(4, 24)])} Nm. Medians EIG / random: "
            + "; ".join(f"K = {K} {cc['meds'][(K, 'a')]:.3f} / {cc['meds'][(K, 'a_rand')]:.3f}"
                        for K in BASES) + "."]
    out += ["", "  | K | EIG − random, k = 12 | k = 24 | k = 40 |", "  |---|---|---|---|"]
    for K in BASES:
        out.append(f"  | {K} | " + " | ".join(nm(c_[(K, k)]) for k in KS) + " |")
    out += [""]
    out += [f"- **PW3d: {d['state']}.** A's mean win rate against generic assistance at k = 24, "
            f"K = 12, simulated room: {p2(d['own'])} (generic = the K = 12 prior mean, as the "
            f"rows store it); predicted ≥ {WIN_LEVEL:.2f} → {d['state']}. Against the K = 8 "
            f"generic assistance instead (one physical generic profile for every K): "
            f"{p2(d['common'])} → {d['common_state']}."]
    return out + [""]


def w3_tables(c: dict, res: dict, setup) -> list[str]:
    out = ["## 2. W3 measures per K", "",
           "icorr_jnd_free's measures and estimators (§0 of `jnd_free.md`): win rate = mean "
           "probability that the wearer prefers the identified assistance over generic "
           "assistance in one duel (3·P̄ = expected wins of a 3-comparison validation block); "
           "share closed = median of (d_prior − d_k)/(d_prior − d_floor); error = median d_k "
           "(Nm); within m JND = share of users with d_k − d_floor ≤ m × 0.674 Nm. Estimate [95 % "
           f"CI], {setup['rooms'].sim_draws.shape[0]} fold-stratified draws (with the Schäfer "
           "participants resampled jointly at the central room). Each K is scored against its "
           "own generic assistance (its prior mean) and its own floor.", "",
           "**Read the JND shares and the share closed with the K context below.** The floors "
           "are not nested in K: K = 4's floor is about 0.67 Nm higher than K = 8's, so \"within "
           "1 / 2 JND of the family optimum\" is an easier target at K = 4 while its absolute "
           "error is worse; its room (d_prior − d_floor) is smaller, which also inflates its "
           "share closed. The error and the win rate are the comparable measures across K.", "",
           "### 2.1 K context (200 population users, simulated room)", ""]
    out += k_context(c) + [""]
    for i, room in enumerate(TABLE_ROOMS, start=2):
        out += [f"### 2.{i} {room_title(setup, room)}", "",
                "| K | optimizer | k | win rate (3·P̄) | share closed | error, Nm | within 1 JND "
                "| within 2 JND |", "|---|---|---|---|---|---|---|---|"]
        for K in BASES:
            g = res[(K, "generic")]["at"]
            out.append(f"| {K} | *generic assistance (K = {K} prior mean)* | 0 | 0.50 (1.50) | 0 "
                       f"| {jf.fci(g[(room, 'err')], 3)} | {jf.fci(g[(room, 'jnd1')])} | "
                       f"{jf.fci(g[(room, 'jnd2')])} |")
            for key, (_, _, label) in OPTS.items():
                r = res[(K, key)]["at"]
                for k in KS:
                    w = r[(room, "win", k)]
                    out.append(f"| {K} | {label} | {k} | {jf.fci(w)} ({3 * w[0]:.2f}) | "
                               f"{jf.fci(r[(room, 'share', k)])} | "
                               f"{jf.fci(r[(room, 'err', k)], 3)} | "
                               f"{jf.fci(r[(room, 'jnd1', k)])} | {jf.fci(r[(room, 'jnd2', k)])} |")
        out.append("")
    n = len(TABLE_ROOMS) + 2
    out += [f"### 2.{n} Comparisons to reach a win rate ≥ {WIN_LEVEL:.2f}", "",
            "The first k (1–40) at which the mean win rate meets the level, on the "
            "cross-sectional curve (not a per-user first hit, not sustained), as `jnd_free.md` "
            "§5: first k [2.5, 97.5 % order statistics over the draws] (share of draws that "
            "reach it by k = 40). \"> 40\": not reached.", "",
            "| K | optimizer | " + " | ".join(room_title(setup, r) for r in TABLE_ROOMS) + " |",
            "|---|---|" + "---|" * len(TABLE_ROOMS)]
    for K in BASES:
        for key, (_, _, label) in OPTS.items():
            txt = []
            for room in TABLE_ROOMS:
                p, lo, hi, reach = res[(K, key)]["first"][(room, "win", ">=", WIN_LEVEL)]
                txt.append(f"{jf.fk(p)} [{jf.fk(lo)}, {jf.fk(hi)}] ({reach:.2f})")
            out.append(f"| {K} | {label} | " + " | ".join(txt) + " |")
    out += ["", f"### 2.{n + 1} Win rate against one generic profile (sensitivity)", "",
            "The tables above take each K's win rate against that K's own prior mean. A study "
            "would validate against one generic profile; here the K = 4 and K = 12 win rates are "
            "taken against the K = 8 prior mean (e-main's d_prior) instead. Estimate [95 % CI].",
            "", "| K | optimizer | room | k = 12 | k = 24 | k = 40 |", "|---|---|---|---|---|---|"]
    for K in (4, 12):
        for key, (_, _, label) in OPTS.items():
            for room in TABLE_ROOMS:
                cells_ = []
                for k in KS:
                    est, b = res[(K, key, "gen8")][(room, k)]
                    cells_.append(jf.fci((est, *jf.ci(b))))
                out.append(f"| {K} | {label} | {room} | " + " | ".join(cells_) + " |")
    return out + [""]


def w5_section(c: dict, res: dict, v: dict, setup) -> list[str]:
    rooms = TABLE_ROOMS
    fl, st = res["c1_floor"], res["c1_start"]
    out = ["## 3. W5: Ramella's learner (C2) and Ramella as published (C1), s = 1.0", "",
           f"Main condition, the same 200 users and answer seeds as e-main, 40 comparisons read at "
           f"12 / 24 / 40. **{C2_LABEL}** is Ramella's batch-MI learner on the ProMP family "
           "(K = 8, tight arm, data prior): same generic assistance and floor as A, so every "
           f"measure compares. **{C1_LABEL}** runs in its own family (Ramella's semi-trapezoid, "
           "its box, flat prior), so only the absolute error and the win rate compare; its win "
           "rate is taken against the ProMP K = 8 generic assistance, the same profile as A's. "
           "C1's floor in its family (median, simulated / central room): "
           + " / ".join(f"{jf.fci((fl[r][0], *jf.ci(fl[r][1])), 3)}" for r in rooms)
           + " Nm (A's ProMP K = 8 floor: 0.658 Nm pooled). Its own family's starting point (the "
           "flat prior's centre, the row's d_prior) is at a median "
           + " / ".join(f"{st[r][0]:.3f}" for r in rooms) + " Nm from the ideal, against "
           f"{np.median(col(c[(8, 'a')], 'd_prior')):.3f} Nm for the ProMP generic assistance "
           "(simulated room).", ""]
    a = {"a": res[(8, "a")], "c2": res["c2"], "c1": res["c1"]}
    labels = {"a": "A (K = 8)", "c2": C2_LABEL, "c1": C1_LABEL}
    for i, room in enumerate(rooms, start=1):
        out += [f"### 3.{i} {room_title(setup, room)}", "",
                "| optimizer | k | win rate (3·P̄) | share closed | error, Nm | within 1 JND | "
                "within 2 JND |", "|---|---|---|---|---|---|---|"]
        for key in ("a", "c2", "c1"):
            r = a[key]["at"]
            for k in KS:
                w = r[(room, "win", k)]
                rest = ([jf.fci(r[(room, "share", k)]), jf.fci(r[(room, "err", k)], 3),
                         jf.fci(r[(room, "jnd1", k)]), jf.fci(r[(room, "jnd2", k)])]
                        if key != "c1" else
                        ["— (other family)", jf.fci(r[(room, "err", k)], 3), "—", "—"])
                out.append(f"| {labels[key]} | {k} | {jf.fci(w)} ({3 * w[0]:.2f}) | "
                           + " | ".join(rest) + " |")
        out.append("")
    j = len(rooms) + 1
    out += [f"### 3.{j} Contrasts", "",
            "Table estimators (paired by icorr_jnd_free's draws). Win rate, share and JND shares: "
            "positive favours A; error: negative favours A. A difference of medians, not a "
            "median of differences.", "",
            "| contrast | room | measure | k = 12 | k = 24 | k = 40 |", "|---|---|---|---|---|---|"]
    for other, measures in (("c2", TABLE_MEASURES), ("c1", ("win", "err"))):
        for room in rooms:
            for m in measures:
                out.append(f"| A − {labels[other]} | {room} | {jf.MEASURE_NAME[m]} | "
                           + " | ".join(jf.fci(paired(a["a"], a[other], room, m, k),
                                               jf.DIGITS[m], signed=True) for k in KS) + " |")
    out += ["", "In the verdicts' convention (simulated room, medians inverted_cdf, "
            f"{N_BOOT_VERDICT} joint draws), A − the other in error:", ""]
    for other in ("c2", "c1"):
        out.append(f"- A − {labels[other]}: " + "; ".join(
            f"k = {k} {nm(v['W5'][(other, k)])} Nm (median {v['W5_meds'][(other, k)]:.3f})"
            for k in KS) + ".")
    out += ["", f"First k at which the mean win rate reaches {WIN_LEVEL:.2f}:", ""]
    for key in ("a", "c2", "c1"):
        txt = []
        for room in rooms:
            p, lo, hi, reach = a[key]["first"][(room, "win", ">=", WIN_LEVEL)]
            txt.append(f"{room} {jf.fk(p)} [{jf.fk(lo)}, {jf.fk(hi)}] ({reach:.2f})")
        out.append(f"- {labels[key]}: " + "; ".join(txt) + ".")
    return out + [""]


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--data", type=Path, default=G.DEFAULT_DATA)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-verdict", type=int, default=N_BOOT_VERDICT)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out
    report = args.report or out / "icorr" / "k_sweep.md"
    sys.stdout.reconfigure(encoding="utf-8")
    if args.boot < 2000 or args.boot_verdict < 2000:
        raise SystemExit("at least 2000 bootstrap draws")

    print("streaming rows ...", flush=True)
    ek, em = load(out)
    keys = G.load_sim(out / "phase5-schaefer")["keys"]
    c = cells(ek, em, keys)
    folds = c[(8, "a")]["fold"].to_numpy()
    for key, df in c.items():
        if not np.array_equal(df["fold"].to_numpy(), folds):
            raise RuntimeError(f"{key}: user order differs")
    chk = checks(c, out)
    bad = [x for x in chk if x[1] > 1e-12]
    if bad:
        raise RuntimeError(f"checks failed: {bad}")

    print("verdicts ...", flush=True)
    v = verdicts(c, Boot(folds, args.boot_verdict))

    print("rooms and weights ...", flush=True)
    room8 = col(c[(8, "a")], "d_prior") - col(c[(8, "a")], "d_floor")
    setup = jf.room_setup(out, args.data, args.boot, room8, folds)
    if setup["sim"]["keys"] != keys or not np.array_equal(setup["sim"]["folds"], folds):
        raise RuntimeError("user order differs between load_sim and the cells")
    rooms = setup["rooms"]
    print("tables ...", flush=True)
    res = table_stats(c, rooms)
    r8 = res[(8, "a")]["at"]
    repro = []
    for (m, k), want in REPRO_K8.items():
        got = jf.f(r8[("simulated", m, k)][0], jf.DIGITS[m])
        repro.append((m, k, got, want, got == want))
    if not all(x[4] for x in repro):
        print(f"REPRODUCTION FAILED: {repro}; nothing written", flush=True)
        return 1

    L: list[str] = []
    p = L.append
    p("# ICORR W3 and W5: basis count K, random duels, and Ramella's learner")
    p("")
    p(f"Generated by `scripts/icorr_k_report.py` on {_dt.date.today().isoformat()} at repository "
      f"commit {git_head()}. Spec: `our design notes"
      "` §9a (W3 design, PW3a–PW3d planned before any run) and §4 (W5). Rows: "
      "`out/icorr/e-k.jsonl` (block e-k, 2,420 sessions) and `out/phase5e/e-main.jsonl` (K = 8: "
      "A, G and SortCMA, the stored rows). No session was run.")
    p("")
    p("**Users and condition.** The 200 population users of the five LOSO folds "
      "(`user_kind == 'population'`, as planned; e-k's 20 stress users per cell are left "
      "out), main condition s = 1.0, λ = 0.1, data prior, tight arm, 40 comparisons read at "
      "k = 12 / 24 / 40. Every cell holds the same users with the same answer seeds.")
    p("")
    p("**Two estimators.** The planned verdicts (§1) use phase5e_report's: medians "
      "inverted_cdf (the lower middle value), the difference of medians on the same users, "
      f"fold-stratified joint bootstrap ({args.boot_verdict} draws). The tables (§2, §3) use "
      "icorr_jnd_free's: weighted Hazen medians and means, its shared draws (seed "
      f"{jf.SEED}, {args.boot} draws), and at the central room the KDE weight of each user from "
      "the K = 8 ProMP room for every K and optimizer. At n = 200 the two medians can differ in "
      "the third decimal; quote §1's numbers for the verdicts.")
    p("")
    p("**Checks** (all passed): " + "; ".join(f"{what} (max |Δ| {d:.1e} Nm)" for what, d in chk)
      + ". K = 8, simulated room, against `jnd_free.md` §1: "
      + "; ".join(f"A {jf.MEASURE_NAME[m]} at k = {k} {got} (expected {want})"
                  for m, k, got, want, _ in repro) + ".")
    p("")
    L.extend(verdict_section(v))
    L.extend(w3_tables(c, res, setup))
    L.extend(w5_section(c, res, v, setup))
    p("## 4. Caveats")
    p("")
    p("- A's settings (trust radius, duel intensities, lapse prior) were chosen at K = 8 and not "
      "retuned at K = 4 or 12 (spec §4 W3 caveat). G's GP length scale and SortCMA's population "
      "follow K by their defaults (build report §4.11), not retuned either.")
    p("- SortCMA's delivered assistance stays at about generic assistance until its first "
      "population (8, 10 and 11 at K = 4, 8 and 12) has been ranked: at K = 8 through k = 12, at "
      "K = 12 through k = 24 (share closed about 0, win rate about 0.5 in §2). The best "
      "comparator at K = 12, k = 24 in PW3b is therefore G, and SortCMA's K = 12 numbers at "
      "k ≤ 24 measure its start-up, not its search.")
    p("- The K families are not nested: a larger K is not a superset of a smaller one, so a "
      "floor can rise with K for a single user (§2.1).")
    p("- Random-duel A delivers the posterior mean unclipped; 0 of e-k's 660 random-duel sessions "
      "ended outside the trust ball at k = 40 (build report §1.5).")
    p("- The win rate is the probability under the RMS-wearer model (s = 1.0, λ = 0.1), whose "
      "ceiling is 0.95; it is a JND-free measure in the sense of spec §5, not a measured "
      "preference.")
    p("- C1 is Ramella's method as published, in its own family at 40 comparisons. Its floor is "
      "far above ProMP's, so its error mostly reflects the family, not the learner.")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
