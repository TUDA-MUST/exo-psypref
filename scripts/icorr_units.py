"""ICORR: torque errors in Nm/kg and as a percentage of the preferred assistance.

Usage:
    python scripts/icorr_units.py [--out <project>/out]
        [--molinaro <time_domain root>] [--data <Schaefer logs>] [--boot 2000]
        [--report <out>/icorr/units.md]

A reusable conversion of the paper's torque numbers. Per simulated user i:

  m_i  the body mass of the Molinaro subject (and leg) whose walking hip moment the user's ideal
       comes from: ``preftune.data.loader.recover_mass(subject, side)``, the mass ``load_trial``
       hands to the generator (``Trial.mass_kg``; Nm = mass x Nm/kg,
       prompref/population/generator.py). The user's subject and leg are in the user sidecars.
  r_i  the RMS of the user's preferred assistance (the ideal tau*_i) over the 101-point gait
       cycle, the norm every distance here uses.

Every distance x_i in Nm (an error, a generic distance, a best-profile distance, a room) becomes x_i / m_i in
Nm/kg and 100 x_i / r_i in % of the preferred assistance, per user, before any median,
difference or bootstrap; no median is divided by a mean mass. Medians are icorr_jnd_free's
weighted Hazen medians at the room's weights; a difference is the difference of the two medians
on the same users, formed draw by draw; intervals are jnd_free's draws (``room_setup``, seed
jf.SEED, 2000 draws by default), the draws best-profile level.md uses. The Nm column is computed the same
way, so each row's three columns describe the same statistic. Planned verdicts in the other
reports used phase5e_report's estimators (lower middle median, 10,000 draws); their Nm values can
differ from this report's in the third decimal.

Fixed quantities with no per-user counterpart (the device limit, the JND yardsticks, the room
levels) are converted at the population's median mass and median preferred-assistance RMS, and
labelled so.

Check, before anything is written: for each subject and leg, the recovered mass rebuilds one
stored user's ideal through ``user_curve`` (the generator's own function) to 1e-9 Nm.

Reads, all read-only: OUT/phase5e/{e-main,e-lapse}.jsonl, OUT/icorr/{e-gpbald,e-gpeubo,e-k,
e-k-hi,e-rev-abl,e-rev-abl2}.jsonl, the users under OUT/phase5-schaefer, the Molinaro parquet
files (``--molinaro``, default PREFTUNE_DATA or the pCloud copy on P:) and the Schaefer logs
(sha256 checked against OUT/icorr/median_room_data_check.json).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import icorr_ceiling as ce  # noqa: E402
import icorr_jnd_free as jf  # noqa: E402
import icorr_k_report as kr  # noqa: E402
import icorr_review_report as rv  # noqa: E402
import icorr_robust_report as rrep  # noqa: E402
import phase5e_report as er  # noqa: E402
import schaefer_grounding as G  # noqa: E402
import short_budget_report as sb  # noqa: E402

from preftune.data.loader import load_trial, recover_mass  # noqa: E402
from prompref.experiment import phase5 as p5  # noqa: E402
from prompref.population.factors import FactorDraw  # noqa: E402
from prompref.population.generator import user_curve  # noqa: E402
from prompref.population.curves import TORQUE_LIMIT_NM  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.simulated_user import rms_distance  # noqa: E402

MOLINARO = Path(os.environ.get(
    "PREFTUNE_DATA",
    "data/molinaro/time_domain"))
KMAX, KS, K_MAIN = 40, (12, 24, 40), 24
ROOMS = ce.ROOMS
ROOM_NAME = ce.ROOM_NAME
MAIN_OPTS = ("a", "gp_bald", "gp_eubo", "g", "d_z", "cmaes")
NAME = dict(ce.NAME, a_lens="A with Franks projection (PsyPref, projection)",
            a_franks="A directly on the Franks parameters")
#: Ablations (icorr_review_report's cells): key -> (label, the reference cell's key).
ABLATIONS = {"a_wide": ("wide prior (z* ~ N(0, 9 I)) − A", "a"),
             "a_nolapse": ("λ = 0 in A's model − A, λ = 0.1 wearer", "a"),
             "a_nolapse_l3": ("λ = 0 in A's model − A, λ = 0.3 wearer", "a_l3"),
             "a_noprobe": ("no lapse comparisons − A, λ = 0.1 wearer", "a"),
             "a_noprobe_l3": ("no lapse comparisons − A, λ = 0.3 wearer", "a_l3")}
#: Fixed quantities in Nm (label, value, source).
FIXED = (("device limit", TORQUE_LIMIT_NM, "prompref.population.curves.TORQUE_LIMIT_NM"),
         ("JND at s = 0.5 Nm", true_jnd(0.5), "0.674 s"),
         ("JND at s = 1.0 Nm (main condition)", true_jnd(1.0), "0.674 s"),
         ("JND at s = 2.0 Nm", true_jnd(2.0), "0.674 s"),
         ("walking ankle timing JND (2.8 %GC, Peng) on our users", 1.18,
          "paper sec. Answer Model (median RMS change)"),
         ("walking magnitude JND (25 % of peak) on our users", 1.21,
          "paper sec. Answer Model (median RMS change)"),
         ("device noise of the noise condition", 0.6, "phase5e_report CONDS, noise"))


# ============================================================================ per user
def masses(users: dict, root: Path) -> tuple[dict, list[dict]]:
    """Mass per (subject, side), and the rebuild check of one stored user per leg."""
    legs = sorted({(u.subject, u.side) for u in users.values()})
    m, rows = {}, []
    for subject, side in legs:
        mass, r2 = recover_mass(subject, side, root)
        m[(subject, side)] = mass
        uid, u = next((k, v) for k, v in users.items() if (v.subject, v.side) == (subject, side))
        trial = load_trial(subject, side, root=root, mass_kg=mass)
        d = u.draw if isinstance(u.draw, dict) else u.draw.to_dict()
        draw = FactorDraw(shift_ms=float(d["shift_ms"]), ext_scale=float(d["ext_scale"]),
                          flex_scale=float(d["flex_scale"]),
                          landmark_offsets=tuple(float(x) for x in d["landmark_offsets"]))
        rebuilt = np.asarray(user_curve(trial, draw), dtype=float)
        dev = float(np.max(np.abs(rebuilt - np.asarray(u.curve_nm, dtype=float))))
        n = sum((v.subject, v.side) == (subject, side) for v in users.values())
        rows.append(dict(subject=subject, side=side, mass=mass, r2=r2, check_user=uid, dev=dev,
                         n=n))
        if dev > 1e-9:
            raise SystemExit(f"{subject}_{side}: the recovered mass does not rebuild {uid} "
                             f"(max |Δ| {dev:.2e} Nm)")
    return m, rows


class Conv:
    """Per-user divisors for the 200 population users in load_sim's order."""

    def __init__(self, run: Path, keys, root: Path):
        users = {}
        for fold in sorted({f for f, _ in keys}):
            users.update(p5._users(str(run), fold))
        self.mass_by_leg, self.legs = masses({u: users[u] for _, u in keys}, root)
        self.m = np.array([self.mass_by_leg[(users[u].subject, users[u].side)] for _, u in keys])
        self.r = np.array([rms_distance(np.asarray(users[u].curve_nm, float),
                                        np.zeros(len(users[u].curve_nm))) for _, u in keys])
        self.subject = [users[u].subject for _, u in keys]

    def three(self, x) -> dict:
        x = np.asarray(x, dtype=float)
        return {"Nm": x, "Nm/kg": x / self.m, "%": 100.0 * x / self.r}


UNITS = ("Nm", "Nm/kg", "%")
DIGITS = {"Nm": 3, "Nm/kg": 4, "%": 1}


def med(rooms, x, room) -> dict:
    est, boot = rooms.stat("err", x, room)
    return dict(est=float(est), boot=np.asarray(boot, float).ravel(),
                **dict(zip(("lo", "hi"), jf.ci(boot))))


def diff(a: dict, b: dict) -> dict:
    boot = a["boot"] - b["boot"]
    lo, hi = jf.ci(boot)
    return dict(est=a["est"] - b["est"], lo=lo, hi=hi)


def fmt(d: dict, unit: str, signed: bool = False) -> str:
    n = DIGITS[unit]
    f = (lambda v: f"{v:+.{n}f}") if signed else (lambda v: f"{v:.{n}f}")
    return f"{f(d['est'])} [{f(d['lo'])}, {f(d['hi'])}]"


def medians(conv: Conv, rooms, x, room) -> dict:
    return {u: med(rooms, v, room) for u, v in conv.three(x).items()}


# ============================================================================ cells
def load_cells(out: Path, keys) -> tuple[dict, dict]:
    """Distances (users, 40) per cell key, aligned to keys, plus d_prior / d_floor of A."""
    em = jf.stream(out / "phase5e" / "e-main.jsonl",
                   keep=lambda r: jf._pop_row(r) and r.get("learner") in
                   ("a", "g", "d_z", "cmaes", "a_lens"))
    blocks = {"e-main": sb._n2048(sb._popn(em))}
    cond = er.BY_ID["main"]
    D, info = {}, {}
    for key, fam, lr in (("a", "promp_s0", "a"), ("g", "promp_s0", "g"),
                         ("d_z", "promp_s0", "d_z"), ("cmaes", "promp_s0", "cmaes"),
                         ("a_lens", "franks", "a_lens"), ("a_franks", "franks", "a")):
        df = jf.in_order(jf.select(blocks, cond, fam, lr), keys)
        D[key] = ce.lst(df, "distances")
        if key == "a":
            info["d_prior"], info["d_floor"] = ce.col(df, "d_prior"), ce.col(df, "d_floor")
        if key == "a_lens":
            info["d_prior_franks"] = ce.col(df, "d_prior")
            info["d_floor_franks"] = ce.col(df, "d_floor")
    for lr, block in (("gp_bald", "e-gpbald"), ("gp_eubo", "e-gpeubo")):
        df = ce.stream(out / "icorr" / f"{block}.jsonl",
                       keep=lambda r: r.get("user_kind") == "population"
                       and "/pop" in str(r.get("user_id", "")))
        if "error" in df and df["error"].notna().any():
            raise SystemExit(f"{block}: error rows")
        c = rrep.pick(kr.normalise(df), "promp_s0", lr, keys, {})
        if not np.array_equal(ce.col(c, "d_prior"), info["d_prior"]):
            raise SystemExit(f"{block}: d_prior differs from A's")
        D[lr] = ce.lst(c, "distances")
    # ablations (icorr_review_report's cells and selection)
    lapse = rv.load(out / "phase5e" / "e-lapse.jsonl")
    D["a_l3"] = ce.lst(jf.in_order(rv.select(rv.population(lapse), rv.CELLS["a_l3"][2]), keys),
                       "distances")
    for block in ("e-rev-abl", "e-rev-abl2"):
        df = rv.load(out / "icorr" / f"{block}.jsonl")
        for key, c in rv.gate_block(df, block, keys).items():
            if key in ABLATIONS:
                D[key] = ce.lst(c, "distances")
    return D, info


def k_rows(out: Path, keys) -> dict:
    """Per K: A's distances (EIG and random duels), d_prior and d_floor, when complete."""
    ek, _ = kr.load(out)
    hi_path = out / "icorr" / "e-k-hi.jsonl"
    hi = jf.stream(ce.no_conflict(hi_path), fields=kr.FIELDS) if hi_path.exists() else None
    hi = None if hi is None else kr.normalise(sb._n2048(hi))
    res = {}
    for K in (4, 8, 12, 16, 24):
        src = ek if K in (4, 8, 12) else hi
        entry = {}
        for duel in ("eig", "random"):
            if K == 8 and duel == "eig":
                continue
            try:
                df = kr.cell(src, "promp_s0", "a", K, duel)
                df = jf.in_order(df, keys)
            except (ValueError, TypeError) as exc:
                entry[duel] = f"incomplete ({exc})"
                continue
            entry[duel] = ce.lst(df, "distances")
            if duel == "eig":
                entry["d_prior"], entry["d_floor"] = ce.col(df, "d_prior"), ce.col(df, "d_floor")
        res[K] = entry
    return res


# ============================================================================ report
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--molinaro", type=Path, default=MOLINARO)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--boot", type=int, default=jf.N_BOOT)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = args.out
    report = args.report or out / "icorr" / "units.md"
    started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    check = json.loads((out / "icorr" / "median_room_data_check.json").read_text(encoding="utf-8"))
    data = args.data or Path(check["data"])
    for f in check["files"]:
        if ce.sha256(data / f["name"]) != f["sha256"]:
            raise SystemExit(f"{f['name']}: sha256 differs from median_room_data_check.json")
    run = out / "phase5-schaefer"
    keys = G.load_sim(run)["keys"]
    folds = np.asarray([f for f, _ in keys])

    print("masses (recovered from the Molinaro trials, checked by rebuilding users) ...",
          flush=True)
    conv = Conv(run, keys, args.molinaro)
    print("rows ...", flush=True)
    D, info = load_cells(out, keys)
    KR = k_rows(out, keys)
    for K, e in KR.items():
        if "d_floor" in e:
            side = kr.sidecar_floors(out, K)
            d = float(np.max(np.abs(e["d_floor"] - np.array([side[u] for _, u in keys]))))
            if d > 1e-12:
                raise SystemExit(f"K = {K}: d_floor differs from the user sidecars")
    KR[8].update(eig=D["a"], d_prior=info["d_prior"], d_floor=info["d_floor"])
    print(f"rooms ({args.boot} draws) ...", flush=True)
    setup = jf.room_setup(out, data, args.boot, info["d_prior"] - info["d_floor"], folds)
    rooms = setup["rooms"]
    lv = setup["level"]

    L = ["# ICORR: torque errors in Nm/kg and in % of the preferred assistance", "",
         f"Generated by `scripts/icorr_units.py`, run started {started}, at repository commit "
         f"{ce.git_head()}. No session was run. Conventions in the script's docstring: per user "
         "x_i / m_i (Nm/kg) and 100 x_i / r_i (% of the RMS of the user's preferred assistance), "
         "then jnd_free's weighted Hazen medians at each room, differences of medians on the "
         f"same users formed draw by draw, {args.boot} draws (seed {jf.SEED}; the draws of "
         "ceiling.md). Estimate [95 % interval]. Rooms: as simulated (unit weights), "
         f"median-matched ({lv['median']:.3f} Nm) and central ({lv['central']:.3f} Nm), KDE "
         "reweighting of icorr_jnd_free.room_setup.", ""]

    # ---- 0. masses
    L += ["## 0. Masses and preferred assistance", "",
          f"Molinaro data: `{args.molinaro}`. Mass recovered per leg by "
          "`preftune.data.loader.recover_mass` (the exoskeleton command regressed on the "
          "transformed TCN estimate; R² must exceed 0.95 for `load_trial`). Check: the recovered "
          "mass rebuilds the named stored user's ideal through `user_curve` (max |Δ|).", "",
          "| subject | leg | mass (kg) | R² | users | rebuild check (user, max abs Δ, Nm) |",
          "|---|---|---|---|---|---|"]
    for r in conv.legs:
        L.append(f"| {r['subject']} | {r['side']} | {r['mass']:.2f} | {r['r2']:.4f} | {r['n']} | "
                 f"{r['check_user']}, {r['dev']:.1e} |")
    m_med, r_med = float(np.median(conv.m)), float(np.median(conv.r))
    L += ["", f"Population (200 users): median mass {m_med:.2f} kg (range {conv.m.min():.2f}–"
          f"{conv.m.max():.2f}); median RMS of the preferred assistance {r_med:.3f} Nm "
          f"({float(np.median(conv.r / conv.m)):.4f} Nm/kg).", ""]
    for room in ("median", "central"):
        mm = med(rooms, conv.m, room)
        L.append(f"- Weighted median mass at the {ROOM_NAME[room]} room: {mm['est']:.2f} kg "
                 f"[{mm['lo']:.2f}, {mm['hi']:.2f}].")
    L.append("")

    # ---- 1. main errors
    L += ["## 1. Main condition: error of the identified assistance", "",
          "Error = RMS distance of the identified assistance after k comparisons to the user's "
          "preferred profile. Generic assistance: its distance d_prior (k = 0). Best possible "
          "profile: the floor d_floor (the device-clipped family curve closest to the ideal).",
          "", "| room | optimizer | k | Nm | Nm/kg | % of preferred assistance |",
          "|---|---|---|---|---|---|"]
    M = {}
    for room in ROOMS:
        rows = [("generic assistance", 0, info["d_prior"]),
                ("best possible profile (floor)", 0, info["d_floor"])]
        for key in MAIN_OPTS:
            for k in KS:
                rows.append((NAME[key], k, D[key][:, k - 1]))
        for name, k, x in rows:
            v = medians(conv, rooms, x, room)
            M[(room, name, k)] = v
            L.append(f"| {ROOM_NAME[room]} | {name} | {k if k else '—'} | "
                     + " | ".join(fmt(v[u], u) for u in UNITS) + " |")
    L.append("")

    # ---- 2. rooms
    L += ["## 2. Room for personalization", "",
          "Per user d_prior − d_floor (ProMP, eight basis functions), its weighted median at each "
          "room.", "", "| room | Nm | Nm/kg | % of preferred assistance |", "|---|---|---|---|"]
    room_x = info["d_prior"] - info["d_floor"]
    for room in ROOMS:
        v = medians(conv, rooms, room_x, room)
        L.append(f"| {ROOM_NAME[room]} | " + " | ".join(fmt(v[u], u) for u in UNITS) + " |")
    L += ["", "Room levels (targets of the reweighting; fixed numbers, converted at the "
          f"population's median mass {m_med:.2f} kg and median preferred-assistance RMS "
          f"{r_med:.3f} Nm):", "", "| room level | Nm | Nm/kg (median mass) | % (median RMS) |",
          "|---|---|---|---|"]
    for name, val in (("Schäfer timing-only (anchor)", lv["anchor"]),
                      ("median-matched", lv["median"]), ("central", lv["central"]),
                      ("high", lv["high"])):
        L.append(f"| {name} | {val:.3f} | {val / m_med:.4f} | {100 * val / r_med:.1f} |")
    L.append("")

    # ---- 3. best-profile distances per K
    L += ["## 3. Floors and generic distances per basis count K", "",
          "Per user, each K's own floor and own generic assistance (its prior mean). K = 16 and "
          "24 only when A's cell holds all 200 users.", "",
          "| K | room | floor, Nm | floor, Nm/kg | floor, % | generic, Nm/kg | generic, % |",
          "|---|---|---|---|---|---|---|"]
    for K in sorted(KR):
        e = KR[K]
        if "d_floor" not in e:
            L.append(f"| {K} | — | {e.get('eig', 'absent')} | | | | |")
            continue
        for room in ROOMS:
            f_ = medians(conv, rooms, e["d_floor"], room)
            g_ = medians(conv, rooms, e["d_prior"], room)
            L.append(f"| {K} | {ROOM_NAME[room]} | {fmt(f_['Nm'], 'Nm')} | "
                     f"{fmt(f_['Nm/kg'], 'Nm/kg')} | {fmt(f_['%'], '%')} | "
                     f"{fmt(g_['Nm/kg'], 'Nm/kg')} | {fmt(g_['%'], '%')} |")
    L.append("")

    # ---- 4. differences
    L += ["## 4. Differences in error (difference of medians, same users)", "",
          "Negative: the first named has the smaller error.", "",
          "| contrast | room | k | Nm | Nm/kg | % of preferred assistance |",
          "|---|---|---|---|---|---|"]

    def contrast(label, xa, xb, room, k):
        va, vb = medians(conv, rooms, xa, room), medians(conv, rooms, xb, room)
        L.append(f"| {label} | {ROOM_NAME[room]} | {k} | "
                 + " | ".join(fmt(diff(va[u], vb[u]), u, signed=True) for u in UNITS) + " |")

    for key in ("gp_bald", "gp_eubo", "g", "d_z", "cmaes"):
        for room in ROOMS:
            for k in KS:
                contrast(f"A − {NAME[key]}", D["a"][:, k - 1], D[key][:, k - 1], room, k)
    for key, (label, ref) in ABLATIONS.items():
        for k in KS:
            contrast(label, D[key][:, k - 1], D[ref][:, k - 1], "simulated", k)
    for K in sorted(KR):
        e = KR[K]
        if isinstance(e.get("random"), np.ndarray) and isinstance(e.get("eig"), np.ndarray):
            for k in (24, 40):
                contrast(f"K = {K}: A − A with random comparisons", e["eig"][:, k - 1],
                         e["random"][:, k - 1], "simulated", k)
    for k1, k2 in ((8, 4), (12, 8), (16, 8), (24, 8)):
        if all(isinstance(KR.get(k, {}).get("eig"), np.ndarray) for k in (k1, k2)):
            for k in (24, 40):
                contrast(f"A at K = {k1} − A at K = {k2}", KR[k1]["eig"][:, k - 1],
                         KR[k2]["eig"][:, k - 1], "simulated", k)
    for k in (24, 40):
        contrast("A with Franks projection − A on the ProMP", D["a_lens"][:, k - 1],
                 D["a"][:, k - 1], "simulated", k)
        contrast("A with Franks projection − A directly on the Franks parameters",
                 D["a_lens"][:, k - 1], D["a_franks"][:, k - 1], "simulated", k)
    L.append("")

    # ---- 5. fixed quantities
    L += ["## 5. Fixed quantities (at the population's median mass and median preferred "
          "assistance)", "",
          f"Converted at {m_med:.2f} kg and {r_med:.3f} Nm (the medians over the 200 simulated "
          "users); labelled conversions, not per-user statistics.", "",
          "| quantity | Nm | Nm/kg (median mass) | % (median RMS) | source |",
          "|---|---|---|---|---|"]
    for name, val, src in FIXED:
        L.append(f"| {name} | {val:.3f} | {val / m_med:.4f} | {100 * val / r_med:.1f} | {src} |")
    L.append("")
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
