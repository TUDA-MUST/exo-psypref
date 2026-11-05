"""ICORR: the attainable best-profile level of the preference over generic assistance.

Usage:
    python scripts/icorr_ceiling.py [--out <project>/out]
        [--data <Schaefer logs>] [--boot 2000] [--report <out>/icorr/best-profile level.md]

The paper's headline measure, the preference over generic assistance (the reports' "win rate"),
is the probability that a simulated user prefers the identified profile over generic,
population-average assistance in one blinded comparison:

    P_i(k) = lambda_i / 2 + (1 - lambda_i) Phi((g_i - e_i(k)) / s_i),

with g_i the distance of generic assistance and e_i(k) that of the identified profile after k
comparisons, in the metric the user answers by. The best profile within the parameterization
(the device-clipped family curve closest to the user's ideal, at distance f_i = d_floor) would
reach

    C_i = lambda_i / 2 + (1 - lambda_i) Phi((g_i - f_i) / s_i),

the attainable best-profile level. It does not depend on k or on the optimizer. An optimizer that delivers
a curve of the family cannot exceed it for a user (up to the best-profile distance fit's precision). No session
is run; every number comes from the stored rows.

Conventions are icorr_jnd_free's: its ``room_setup`` on e-main A's room d_prior - d_floor (KDE
reweighting to the median-matched 1.47 Nm and central 1.74 Nm rooms, seed jf.SEED, the Schaefer
participants resampled jointly), weighted means, 2.5 / 97.5 % percentile intervals, one set of
draws for every number (so differences and ratios pair by draw). Every cell on the 200
population users carries its user's K = 8 weight, at every K, as k_sweep does. Cells on other
users (out-of-generator ideals) are reported at the simulated room only.

Share of the best-profile level reached: (P - 0.5) / (C - 0.5), the part of the attainable gain above
generic assistance that an optimizer realises (the analogue of the share closed), and P / C;
both formed draw by draw from the means.

Reads, all read-only, each jsonl streamed line by line: OUT/phase5e/{e-main,e-s,e-lapse,
e-noise}.jsonl, OUT/phase5c/main-promp.jsonl and OUT/phase5d/cmaes.jsonl (the s = 0.5
reference), OUT/icorr/{e-k,e-k-hi,e-gpbald,e-gpeubo,e-ood-pb,e-rand,e-wph-pref,e-wph-bio,
e-gpeubo-ood-pb,e-gpeubo-rand,e-gpeubo-wph-pref,e-gpeubo-wph-bio}.jsonl, the user sidecars
under OUT/phase5-schaefer (users, random_utility, phase_weight) and the Schaefer logs (their
sha256 checked against OUT/icorr/median_room_data_check.json). "[conflicted]" sync copies are
never read.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import icorr_jnd_free as jf  # noqa: E402
import icorr_k_report as kr  # noqa: E402
import icorr_robust_report as rrep  # noqa: E402
import phase5e_report as er  # noqa: E402
import schaefer_grounding as G  # noqa: E402
import short_budget_report as sb  # noqa: E402

from prompref.experiment.robust import load_ood_users  # noqa: E402

KMAX = 40
KS = (12, 24, 40)
K_MAIN = 24
ROOMS = ("simulated", "median", "central")
ROOM_NAME = {"simulated": "as simulated", "median": "median-matched (1.47 Nm)",
             "central": "central (1.74 Nm)"}
FIELDS = rrep.FIELDS + ("phase_weighting",)
LISTS = ("distances", "own_distances")
MAIN_LEARNERS = ("a", "g", "d_z", "cmaes")
GP_LEARNERS = ("gp_bald", "gp_eubo")
COMPARATORS = ("gp_bald", "gp_eubo", "g", "d_z", "cmaes")
#: The reports' names, then the paper's in parentheses.
NAME = {"a": "A (PsyPref)", "gp_bald": "GP-BALD", "gp_eubo": "GP-EUBO (GP-qEUBO)",
        "g": "G (Arens' GP)", "d_z": "LineCoSpar", "cmaes": "SortCMA (CMA-ES)"}
#: Robustness cells on the population users (phase5e_report's CONDS), in Fig. 2's order.
CONDS = ("main", "s05", "s2", "het", "lapse2", "lapse3", "noise")
COND_NAME = {"main": "main (s = 1.0, λ = 0.1)", "s05": "s = 0.5 (reference rows)",
             "s2": "s = 2.0", "het": "individual s", "lapse2": "λ = 0.2", "lapse3": "λ = 0.3",
             "noise": "device noise 0.6 Nm"}
BASES = (4, 8, 12, 16, 24)
#: Report values the main condition must reproduce (point estimates at three decimals):
#: gpeubo_report.md section 3.1 (median-matched) and 7.1 (central), k = 24.
EXPECT_MAIN = {("median", "a"): 0.721, ("median", "gp_bald"): 0.650,
               ("median", "gp_eubo"): 0.543, ("median", "g"): 0.521,
               ("median", "d_z"): 0.480, ("median", "cmaes"): 0.552,
               ("central", "a"): 0.767, ("central", "gp_bald"): 0.696,
               ("central", "gp_eubo"): 0.572, ("central", "g"): 0.560,
               ("central", "d_z"): 0.517, ("central", "cmaes"): 0.587,
               ("simulated", "a"): 0.815, ("simulated", "gp_bald"): 0.755,
               ("simulated", "gp_eubo"): 0.620, ("simulated", "g"): 0.618,
               ("simulated", "d_z"): 0.578, ("simulated", "cmaes"): 0.629}


# ============================================================================ helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def col(df, name) -> np.ndarray:
    return df[name].astype(float).to_numpy()


def lst(df, name) -> np.ndarray:
    d = np.array([v[:KMAX] for v in df[name]], dtype=float)
    if d.shape[1] < KMAX:
        raise ValueError(f"a row holds fewer than {KMAX} values of {name}")
    return d


def pref(gen, e, s, lam) -> np.ndarray:
    """P = lambda/2 + (1 - lambda) Phi((gen - e)/s), per user (jf.win_probability)."""
    return jf.win_probability(gen, e, s, lam)


class Cell:
    """One condition: per user the generic distance, the best-profile distance (the best profile's distance)
    and s, lambda in the simulated user's metric, and each optimizer's identified distances."""

    def __init__(self, gen, floor, s, lam, folds, note=""):
        self.gen, self.floor = np.asarray(gen, float), (None if floor is None
                                                        else np.asarray(floor, float))
        self.s, self.lam = np.asarray(s, float), np.asarray(lam, float)
        self.folds = np.asarray(folds)
        self.E: dict[str, np.ndarray] = {}
        self.note = note

    def ceiling(self):
        return None if self.floor is None else pref(self.gen, self.floor, self.s, self.lam)

    def p(self, key, k):
        return pref(self.gen, self.E[key][:, k - 1], self.s, self.lam)


def stat(rooms, x, room):
    est, boot = rooms.stat("win", x, room)
    return float(est), np.asarray(boot, float).ravel()


def ci(est, boot) -> dict:
    lo, hi = jf.ci(boot)
    return dict(est=float(est), lo=lo, hi=hi)


def txt(d: dict | None, digits: int = 3) -> str:
    if d is None:
        return "—"
    return f"{d['est']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def evaluate(cell: Cell, rooms, room: str, ks=KS) -> dict:
    """Best-profile level, each optimizer's preference at ks, and the shares at k = 24, at one room."""
    res = {"pref": {}, "share": {}, "ratio": {}}
    C = cell.ceiling()
    c_est = c_boot = None
    if C is not None:
        c_est, c_boot = stat(rooms, C, room)
        res["ceiling"] = ci(c_est, c_boot)
        res["floor"] = ci(*rooms.stat("err", cell.floor, room)) if cell.floor is not None \
            else None
    else:
        res["ceiling"] = None
    res["generic"] = ci(*rooms.stat("err", cell.gen, room))
    for key in cell.E:
        for k in ks:
            p_est, p_boot = stat(rooms, cell.p(key, k), room)
            res["pref"][(key, k)] = ci(p_est, p_boot)
            if C is not None and k == K_MAIN:
                res["share"][key] = ci((p_est - 0.5) / (c_est - 0.5),
                                       (p_boot - 0.5) / (c_boot - 0.5))
                res["ratio"][key] = ci(p_est / c_est, p_boot / c_boot)
    comps = [c for c in COMPARATORS if c in cell.E]
    res["best"] = max(comps, key=lambda c: res["pref"][(c, K_MAIN)]["est"]) if comps else None
    return res


# ============================================================================ loading
def no_conflict(path: Path) -> Path:
    if "[conflicted" in path.name:
        raise SystemExit(f"refusing to read a sync conflict copy: {path}")
    return path


def stream(path: Path, keep=jf._pop_row):
    return jf.stream(no_conflict(path), keep=keep, fields=FIELDS, lists=LISTS)


def population_cells(out: Path, keys, folds) -> tuple[dict, dict, list[str]]:
    """The main condition and the robustness cells (jnd_free's selection), the GP cells."""
    notes: list[str] = []
    blocks = {}
    for name in ("e-main", "e-s", "e-lapse", "e-noise"):
        df = stream(out / "phase5e" / f"{name}.jsonl",
                    keep=lambda r: jf._pop_row(r) and r.get("family") == "promp_s0"
                    and r.get("learner") in MAIN_LEARNERS)
        blocks[name] = None if df is None else sb._n2048(sb._popn(df))
    blocks["5c/main-promp"] = jf.stream(out / "phase5c" / "main-promp.jsonl")
    blocks["5d/cmaes"] = jf.stream(out / "phase5d" / "cmaes.jsonl")
    ref = None
    cells: dict[str, Cell] = {}
    for cid in CONDS:
        cond = er.BY_ID[cid]
        dfs = {}
        for lr in MAIN_LEARNERS:
            df = jf.select(blocks, cond, "promp_s0", lr)
            if df is None:
                raise SystemExit(f"{cid}: no rows for {lr}")
            dfs[lr] = jf.in_order(df, keys)
        a = dfs["a"]
        if ref is None:
            ref = a
        for lr, df in dfs.items():
            for field in ("d_prior", "d_floor"):
                d = float(np.max(np.abs(col(df, field) - col(ref, field))))
                if d > 1e-12:
                    raise SystemExit(f"{cid} {lr}: {field} differs from e-main A's by {d:.2e}")
        s = col(a, "s")
        if cid == "het":
            s = col(a, "s_user")
        cell = Cell(col(a, "d_prior"), col(a, "d_floor"), s, col(a, "lapse"), folds,
                    note=jf.CELL_NOTE[cid])
        for lr, df in dfs.items():
            cell.E[lr] = lst(df, "distances")
        cells[cid] = cell
    # the GP cells, main condition (gpeubo_report's R4)
    state = {}
    for lr, block in (("gp_bald", "e-gpbald"), ("gp_eubo", "e-gpeubo")):
        path = out / "icorr" / f"{block}.jsonl"
        df = stream(path, keep=lambda r: r.get("user_kind") == "population"
                    and "/pop" in str(r.get("user_id", ""))) if path.exists() else None
        if df is None:
            state[lr] = "not run"
            continue
        if "error" in df and df["error"].notna().any():
            state[lr] = f"incomplete ({int(df['error'].notna().sum())} error rows)"
            continue
        try:
            c = rrep.pick(kr.normalise(df), "promp_s0", lr, keys, {})
        except ValueError as exc:
            state[lr] = f"incomplete ({exc})"
            continue
        for field in ("d_prior", "d_floor"):
            if not np.array_equal(col(c, field), col(ref, field)):
                raise SystemExit(f"{block}: {field} differs from e-main A's")
        cells["main"].E[lr] = lst(c, "distances")
        state[lr] = f"ok ({block}.jsonl, 200 users)"
    for twin in sorted((out / "icorr").glob("*[[]conflicted*")):
        notes.append(f"`{twin.name}` lies in out/icorr and was not read.")
    return cells, state, notes, ref


def k_cells(out: Path, keys, folds, main: Cell, ref) -> dict:
    """K = 4 and 12 from e-k, K = 16 and 24 from e-k-hi when PsyPref's cell is complete."""
    res = {8: dict(cell=main, state="ok (e-main)", blocks="e-main.jsonl")}
    ek, _ = kr.load(out)
    hi_path = out / "icorr" / "e-k-hi.jsonl"
    hi = None
    if hi_path.exists():
        hi = jf.stream(no_conflict(hi_path), fields=kr.FIELDS)
        hi = None if hi is None else kr.normalise(sb._n2048(hi))
    for K in (4, 12, 16, 24):
        src = ek if K in (4, 12) else hi
        name = "e-k.jsonl" if K in (4, 12) else "e-k-hi.jsonl"
        if src is None:
            res[K] = dict(cell=None, state="no rows", blocks=name)
            continue
        dfs = {}
        for lr in MAIN_LEARNERS:
            try:
                df = kr.cell(src, "promp_s0", lr, K, "eig")
            except ValueError as exc:
                if lr == "a":
                    dfs = None
                    res[K] = dict(cell=None, state=f"absent ({exc})", blocks=name)
                    break
                continue
            if "error" in df and df["error"].notna().any():
                continue
            have = set(zip(df["fold"], df["user_id"]))
            n = sum(k in have for k in keys)
            if n < len(keys):
                if lr == "a":
                    dfs = None
                    res[K] = dict(cell=None, state=f"incomplete (PsyPref: {n} of {len(keys)} "
                                                   "users)", blocks=name)
                    break
                continue
            dfs[lr] = jf.in_order(df, keys)
        if dfs is None:
            continue
        a = dfs["a"]
        side = kr.sidecar_floors(out, K)
        d = float(np.max(np.abs(col(a, "d_floor") - np.array([side[u] for u in a["user_id"]]))))
        if d > 1e-12:
            raise SystemExit(f"K = {K}: d_floor differs from the user sidecars by {d:.2e}")
        for lr, df in dfs.items():
            for field in ("d_prior", "d_floor"):
                if not np.array_equal(col(df, field), col(a, field)):
                    raise SystemExit(f"K = {K} {lr}: {field} differs from PsyPref's")
        cell = Cell(col(a, "d_prior"), col(a, "d_floor"), col(a, "s"), col(a, "lapse"), folds)
        for lr, df in dfs.items():
            cell.E[lr] = lst(df, "distances")
        res[K] = dict(cell=cell, state="ok", blocks=name,
                      present=[lr for lr in MAIN_LEARNERS if lr in dfs])
    return res


def gp_extra(out: Path, block: str, pick_keys, fields: dict, check: tuple, a, keep):
    """GP-EUBO in an extra-simulated user block, complete by gpeubo_report's R10, or (None, state)."""
    path = out / "icorr" / f"{block}.jsonl"
    if not path.exists():
        return None, "not run"
    df = stream(path, keep=keep)
    if df is None:
        return None, "no rows"
    if "error" in df and df["error"].notna().any():
        return None, f"incomplete ({int(df['error'].notna().sum())} error rows)"
    try:
        c = rrep.pick(kr.normalise(sb._n2048(df)), "promp_s0", "gp_eubo", pick_keys, fields)
    except ValueError as exc:
        return None, f"incomplete ({exc})"
    for field in check:
        if not np.array_equal(col(c, field), col(a, field)):
            return None, f"{field} differs from A's"
    return c, f"ok ({block}.jsonl)"


def extra_cells(out: Path, keys, folds) -> dict:
    """Other users and other simulated users, simulated room: O (per part peaks), R (u_max), W-pref,
    W-bio (own best-profile distances). H has no best-profile distance in the simulated user's own metric."""
    run = out / "phase5-schaefer"
    res = {}
    # ---- O, per part peaks: other users
    okeys = []
    for fold in G.FOLDS:
        for uid in load_ood_users(str(run), fold, "ood-pb"):
            okeys.append((fold, uid))
    ofolds = np.asarray([f for f, _ in okeys])
    if not np.array_equal(ofolds, folds):
        raise SystemExit("O users: the fold layout differs from the population users'")
    rows = rrep._slim_ood(stream(out / "icorr" / "e-ood-pb.jsonl", keep=rrep.ood_row))
    dfs = {lr: rrep.pick(rows, "promp_s0", lr, okeys, dict(user_set="ood-pb"))
           for lr in MAIN_LEARNERS}
    a = dfs["a"]
    cell = Cell(col(a, "d_prior"), col(a, "d_floor"), col(a, "s"), col(a, "lapse"), folds,
                note="RMS wearer on the out-of-generator users (per-bout peaks)")
    for lr, df in dfs.items():
        cell.E[lr] = lst(df, "distances")
    gp, st = gp_extra(out, "e-gpeubo-ood-pb", okeys, dict(user_set="ood-pb"),
                      ("d_prior", "d_floor"), a, rrep.ood_row)
    if gp is not None:
        cell.E["gp_eubo"] = lst(gp, "distances")
    res["ood"] = dict(cell=cell, gp=st)
    # ---- R, random smooth utility: own distance = -u (scaled); the best = -u_max
    side = rrep.read_sidecars(run)["users"]
    rows = rrep._slim(stream(out / "icorr" / "e-rand.jsonl"))
    dfs = {lr: rrep.pick(rows, "promp_s0", lr, keys, dict(observer_metric="random_smooth"))
           for lr in MAIN_LEARNERS}
    a = dfs["a"]
    uids = a["user_id"].to_numpy()
    ugen = np.array([side[u]["u_generic"] for u in uids])
    if float(np.max(np.abs(col(a, "own_generic_distance") + ugen))) != 0.0:
        raise SystemExit("R: own_generic_distance is not -u_gen of the sidecar")
    umax = np.array([side[u]["u_max"] for u in uids])
    conv = np.array([side[u]["converged"] for u in uids], dtype=bool)
    cell = Cell(col(a, "own_generic_distance"), -umax, col(a, "s"), col(a, "lapse"), folds,
                note="own utility; the best profile is the best-found u_max, a lower bound")
    for lr, df in dfs.items():
        cell.E[lr] = lst(df, "own_distances")
    gp, st = gp_extra(out, "e-gpeubo-rand", keys, dict(observer_metric="random_smooth"),
                      ("own_generic_distance",), a, jf._pop_row)
    if gp is not None:
        cell.E["gp_eubo"] = lst(gp, "own_distances")
    res["rand"] = dict(cell=cell, gp=st, converged=conv)
    # ---- W-pref, W-bio: own best-profile distances on the matched scale (icorr_wph_prep sidecars)
    for w, block in (("pref", "e-wph-pref"), ("bio", "e-wph-bio")):
        floors, gens = {}, {}
        for path in sorted((run / "phase_weight" / w).glob("loso-*.json")):
            for uid, u in json.loads(path.read_text(encoding="utf-8"))["users"].items():
                floors[uid], gens[uid] = float(u["own_floor"]), float(u["own_generic"])
        rows = rrep._slim(stream(out / "icorr" / f"{block}.jsonl"))
        dfs = {lr: rrep.pick(rows, "promp_s0", lr, keys, {}) for lr in MAIN_LEARNERS}
        a = dfs["a"]
        uids = a["user_id"].to_numpy()
        g_side = np.array([gens[u] for u in uids])
        d = float(np.max(np.abs(col(a, "own_generic_distance") - g_side)))
        if d > 1e-12:
            raise SystemExit(f"W-{w}: own_generic_distance differs from the sidecar by {d:.2e}")
        cell = Cell(col(a, "own_generic_distance"), np.array([floors[u] for u in uids]),
                    col(a, "s"), col(a, "lapse"), folds,
                    note="own phase-weighted distance, matched scale; own floor (wph R7)")
        for lr, df in dfs.items():
            cell.E[lr] = lst(df, "own_distances")
        gp, st = gp_extra(out, f"e-gpeubo-wph-{w}", keys, {}, ("own_generic_distance",), a,
                          jf._pop_row)
        if gp is not None:
            cell.E["gp_eubo"] = lst(gp, "own_distances")
        res[f"wph-{w}"] = dict(cell=cell, gp=st)
    return res


# ============================================================================ report
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--boot", type=int, default=jf.N_BOOT)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = args.out
    report = args.report or out / "icorr" / "ceiling.md"
    started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    check = json.loads((out / "icorr" / "median_room_data_check.json").read_text(encoding="utf-8"))
    data = args.data or Path(check["data"])
    for f in check["files"]:
        if sha256(data / f["name"]) != f["sha256"]:
            raise SystemExit(f"{f['name']}: sha256 differs from median_room_data_check.json")
    keys = G.load_sim(out / "phase5-schaefer")["keys"]
    folds = np.asarray([f for f, _ in keys])

    print("population cells ...", flush=True)
    cells, gp_state, notes, ref = population_cells(out, keys, folds)
    print(f"rooms (jnd_free.room_setup, {args.boot} draws) ...", flush=True)
    setup = jf.room_setup(out, data, args.boot, col(ref, "d_prior") - col(ref, "d_floor"), folds)
    rooms = setup["rooms"]
    D = cells["main"].E["a"]
    q1 = ((cells["main"].gen - D[:, K_MAIN - 1]) / G.JND[1.0] >= 1.0 - G.TOL).astype(float)
    gates = {r: float(rooms.stat("q1", q1, r)[0]) for r in ("median", "central")}
    if abs(gates["median"] - jf.GATE_MEDIAN[0]) > jf.GATE_TOL or \
            abs(gates["central"] - jf.GATE_CENTRAL[0]) > jf.GATE_TOL:
        raise SystemExit(f"room gate failed: {gates}")
    print("basis count K ...", flush=True)
    kc = k_cells(out, keys, folds, cells["main"], ref)
    print("other users and wearers ...", flush=True)
    xc = extra_cells(out, keys, folds)

    R = {cid: {room: evaluate(c, rooms, room) for room in ROOMS} for cid, c in cells.items()}
    RK = {K: {room: evaluate(e["cell"], rooms, room, ks=(K_MAIN,)) for room in ROOMS}
          for K, e in kc.items() if e["cell"] is not None}
    RX = {name: evaluate(e["cell"], rooms, "simulated", ks=(K_MAIN,)) for name, e in xc.items()}
    conv = xc["rand"]["converged"]

    lv = setup["level"]
    L = ["# ICORR: the attainable ceiling of the preference over generic assistance", "",
         f"Generated by `scripts/icorr_ceiling.py`, run started {started}, at repository commit "
         f"{git_head()}. No session was run; every number comes from the stored rows. Post hoc "
         "(added 2026-10-21 for the paper's results section; no verdicts).", "",
         "## 0. Definitions and conventions", "",
         "- **Preference over generic assistance** (the reports' \"win rate\"): the probability "
         "that a simulated user prefers the identified profile over generic, population-average "
         "assistance (the prior's mean curve) in one blinded comparison, P_i(k) = λ/2 + (1 − λ) "
         "Φ((g_i − e_i(k))/s), averaged over users. 0.5 = no better than generic; at most "
         "1 − λ/2 (0.95 at λ = 0.1).",
         "- **Attainable ceiling**: the same probability for the best profile within the "
         "parameterization, C_i = λ/2 + (1 − λ) Φ((g_i − f_i)/s), f_i = d_floor (the "
         "device-clipped family curve closest to the user's ideal, multi-start fit). It does not "
         "depend on k or on the optimizer.",
         "- **Share of the ceiling reached**: (P − 0.5)/(C − 0.5), the part of the attainable "
         "gain above generic assistance that an optimizer realises; and P / C. Both formed from "
         "the means draw by draw.",
         f"- **Rooms and intervals**: icorr_jnd_free's `room_setup` on e-main A's room d_prior − "
         f"d_floor ({args.boot} draws, seed {jf.SEED}; Schäfer participants resampled jointly; "
         f"data `{data}`, {len(check['files'])} files, sha256 as in "
         "median_room_data_check.json). Median-matched room level "
         f"{lv['median']:.3f} Nm, central {lv['central']:.3f} Nm. Estimate [2.5, 97.5 % "
         "percentiles over the draws]. Every cell on the 200 population users carries its "
         "user's K = 8 weight at every K (as k_sweep.md). Effective sample size "
         + "; ".join(f"{ROOM_NAME[r]} {rooms.ess[r][0]:.1f}" for r in ("median", "central"))
         + ".",
         f"- **Gate** (a check, not a result): A's Q≥1 at k = 24 is {gates['median']:.3f} at the "
         f"median-matched room (jnd_free.md §9: 0.59) and {gates['central']:.3f} at the central "
         "room (jnd_free.md §0a: 0.699).",
         "- **Medians** of distances (generic, floor) are jnd_free's weighted Hazen medians.", ""]
    L += notes and (["Files not read: " + " ".join(notes), ""]) or []

    # ---- 1. main condition
    L += ["## 1. Main condition (s = 1.0, λ = 0.1, eight basis functions)", "",
          "GP cells: " + "; ".join(f"{NAME[k]} {v}" for k, v in gp_state.items()) + ".", "",
          "| room | ceiling C | floor f, median (Nm) | generic g, median (Nm) |",
          "|---|---|---|---|"]
    for room in ROOMS:
        r = R["main"][room]
        L.append(f"| {ROOM_NAME[room]} | {txt(r['ceiling'])} | {txt(r['floor'])} | "
                 f"{txt(r['generic'])} |")
    L += ["", "Preference over generic assistance, and the share of the ceiling reached at "
          "k = 24:", "",
          "| room | optimizer | k = 12 | k = 24 | k = 40 | (P − 0.5)/(C − 0.5), k = 24 | "
          "P / C, k = 24 | report value, k = 24 |", "|---|---|---|---|---|---|---|---|"]
    problems = []
    for room in ROOMS:
        r = R["main"][room]
        for key in ("a",) + COMPARATORS:
            if key not in cells["main"].E:
                continue
            got = r["pref"][(key, K_MAIN)]["est"]
            want = EXPECT_MAIN.get((room, key))
            ok = want is not None and abs(round(got, 3) - want) < 1e-9
            if want is not None and not ok:
                problems.append(f"main {room} {key}: {got:.3f} vs report {want:.3f}")
            src = {"median": "gpeubo §3.1", "central": "gpeubo §7.1",
                   "simulated": "gpeubo §2.1"}[room]
            L.append(f"| {ROOM_NAME[room]} | {NAME[key]} | {txt(r['pref'][(key, 12)])} | "
                     f"{txt(r['pref'][(key, 24)])} | {txt(r['pref'][(key, 40)])} | "
                     f"{txt(r['share'][key], 2)} | {txt(r['ratio'][key], 2)} | "
                     f"{want if want is not None else '—'} ({src}; "
                     f"{'reproduced' if ok else 'DIFFERS'}) |")
    L.append("")

    # ---- 2. robustness cells
    L += ["## 2. Robustness conditions on the population users (k = 24)", "",
          "Each cell's own s and λ (individual s: each user's s_user). Best comparator: the "
          "highest preference among the comparators run in that cell (GP-BALD and GP-EUBO in the "
          "main condition only). Win-rate basis per cell as jnd_free.md §2.", "",
          "| condition | room | ceiling C | PsyPref P | share (P − 0.5)/(C − 0.5) | best "
          "comparator P | its share |", "|---|---|---|---|---|---|---|"]
    for cid in CONDS:
        for room in ROOMS:
            r = R[cid][room]
            b = r["best"]
            L.append(f"| {COND_NAME[cid]} | {ROOM_NAME[room]} | {txt(r['ceiling'])} | "
                     f"{txt(r['pref'][('a', K_MAIN)])} | {txt(r['share']['a'], 2)} | "
                     f"{NAME[b]} {txt(r['pref'][(b, K_MAIN)])} | {txt(r['share'][b], 2)} |")
    L += ["", "Every comparator, k = 24 (preference [95 %]):", "",
          "| condition | room | " + " | ".join(NAME[k] for k in MAIN_LEARNERS[1:]) + " |",
          "|---|---|---|---|---|"]
    for cid in CONDS:
        for room in ROOMS:
            r = R[cid][room]
            L.append(f"| {COND_NAME[cid]} | {ROOM_NAME[room]} | "
                     + " | ".join(txt(r["pref"][(k, K_MAIN)]) for k in MAIN_LEARNERS[1:]) + " |")
    L.append("")

    # ---- 3. basis count
    L += ["## 3. Basis count K (main condition, k = 24)", "",
          "Each K against its own generic assistance (its prior mean) and its own floor. "
          "K = 4 and 12: e-k.jsonl; K = 16 and 24: e-k-hi.jsonl, only when PsyPref's cell holds "
          "all 200 users.", ""]
    for K in BASES:
        e = kc.get(K)
        if e is None:
            continue
        extra = f" Present: {', '.join(NAME[k] for k in e.get('present', []))}." \
            if e.get("present") else ""
        L.append(f"- K = {K}: {e['state']} ({e['blocks']}).{extra}")
    L += ["", "| K | room | ceiling C | floor f, median (Nm) | PsyPref P | share | best "
          "comparator P | its share |", "|---|---|---|---|---|---|---|---|"]
    for K in sorted(RK):
        for room in ROOMS:
            r = RK[K][room]
            b = r["best"]
            L.append(f"| {K} | {ROOM_NAME[room]} | {txt(r['ceiling'])} | {txt(r['floor'])} | "
                     f"{txt(r['pref'][('a', K_MAIN)])} | {txt(r['share']['a'], 2)} | "
                     + (f"{NAME[b]} {txt(r['pref'][(b, K_MAIN)])} | {txt(r['share'][b], 2)} |"
                        if b else "— | — |"))
    L.append("")

    # ---- 4. other users and simulated users
    xname = {"ood": "other generator (O, per-bout peaks; other users)",
             "rand": "random utility (R; own utility, u_max a best-found bound)",
             "wph-pref": "phase-weighted, individual (W-pref; own metric)",
             "wph-bio": "phase-weighted, hip speed (W-bio; own metric)"}
    L += ["## 4. Other user models (simulated room, k = 24)", "",
          "In the wearer's own metric where it has one (R, W), with s = 1.0 and λ = 0.1 on the "
          "matched scale. Feature-judging users (H): no floor in their own metric is stored "
          "(jnd_free.md §0), so no ceiling. R: the best profile's utility u_max is a best-found "
          "lower bound, so the ceiling is a lower bound. GP-EUBO enters a cell only if its block "
          "is complete by gpeubo_report's R10 (200 users, no error row, generic distance equal "
          "to A's).", "",
          "| user model | ceiling C | generic g, median | floor f, median | PsyPref P | share | "
          "best comparator P | its share | GP-EUBO block |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, r in RX.items():
        b = r["best"]
        L.append(f"| {xname[name]} | {txt(r['ceiling'])} | {txt(r['generic'])} | "
                 f"{txt(r['floor'])} | {txt(r['pref'][('a', K_MAIN)])} | "
                 f"{txt(r['share']['a'], 2)} | {NAME[b]} {txt(r['pref'][(b, K_MAIN)])} | "
                 f"{txt(r['share'][b], 2)} | {xc[name]['gp']} |")
    cr = xc["rand"]["cell"]
    c_conv = cr.ceiling()[conv]
    L += ["", f"R, the {int(conv.sum())} users whose u_max converged: ceiling (plain mean) "
          f"{float(np.mean(c_conv)):.3f}; all 200: {float(np.mean(cr.ceiling())):.3f}; smallest "
          f"user {float(np.min(cr.ceiling())):.3f}.", "",
          "Every comparator, k = 24 (preference [95 %]):", "",
          "| user model | " + " | ".join(NAME[k] for k in ("g", "d_z", "cmaes", "gp_eubo"))
          + " |", "|---|---|---|---|---|"]
    for name, r in RX.items():
        L.append(f"| {xname[name]} | " + " | ".join(
            txt(r["pref"].get((k, K_MAIN))) for k in ("g", "d_z", "cmaes", "gp_eubo")) + " |")
    L += ["", "## 5. Reproduction", ""]
    L += ([f"- DIFFERS: {p}" for p in problems] if problems else
          ["Every main-condition preference at k = 24 reproduces gpeubo_report.md §2.1, §3.1 and "
           "§7.1 at three decimals."])
    L.append("")
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    for p in problems:
        print(f"  DIFFERS: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
