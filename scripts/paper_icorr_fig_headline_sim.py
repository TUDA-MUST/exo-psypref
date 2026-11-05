"""Fig. 2 (fig_headline.pdf): PsyPref needs far fewer comparisons, as simulated.

Usage:  python scripts/paper_icorr_fig_headline_sim.py [--out <main>/out] [--figdir paper/figures]
            [--boot 10000] [--check-draws]

The author's decision of 2026-10-25: one panel, the preference over generic assistance against
the comparisons k = 1..40, as simulated (the 200 population users with unit weights, no room
reweighting), four curves with 95 % bands:

  PsyPref           on the ProMP (K = 8): out/phase5e/e-main.jsonl, learner a, the main
                    condition (s = 1.0, lapse 0.1);
  PsyPref, Schäfer  each proposal projected onto Schäfer et al.'s six parameters:
                    out/icorr/e-proj.jsonl, family schaefer_peaks, learner a_lens;
  CMA-ES, Schäfer   CMA-ES searching Schäfer's six parameters directly, established practice:
                    out/icorr/e-base-cma.jsonl, learner cmaes, the setting base_tuned.json chose
                    (icorr_r2_report.load_all, cell BASE-cmaes);
  GP-BALD           on the ProMP with the same prior: out/icorr/e-gpbald.jsonl, loaded as
                    paper_icorr_figures.headline_data loads it (_head_main_cells).

Every curve is measured against the same generic assistance, e-main A's d_prior (R1 of
icorr_r2_report): the K = 8 ProMP prior mean, one physical profile whatever is searched in. The
reference lines are 50 % (generic assistance) and the attainable best-profile level of the ProMP K = 8 as
simulated (e-main A's d_prior and d_floor, as paper_icorr_figures.headline_data computes it).
The annotations give the first k at which PsyPref, and PsyPref projected onto Schäfer, reach
CMA-ES's preference after 40 comparisons, and at which PsyPref reaches GP-BALD's after 24:
jnd_free.first_k on the cross-sectional mean, the target taken in the same bootstrap draw
(paper_icorr_figures._first_reach). The other comparators (Arens' GP, LineCoSpar, SortCMA,
GP-qEUBO) are not drawn.

Statistics: jnd_free's simulated room, unit weights and its fold-stratified user draws (jnd_free's
seed, the first draws room_setup takes from it), through jnd_free.Rooms.stat, so the bands and
first-k intervals are the simulated room of gpeubo_report.md section 7 and of
paper_icorr_figures.headline_data. The Schaefer logs that the reweighted rooms need are not read;
``--check-draws`` runs jnd_free.room_setup on them and stops unless its simulated draws are the
ones used here.

Writes <figdir>/fig_headline.pdf, fig_headline.png (300 dpi) and fig_headline_numbers.md (every
plotted number, the reach values with their intervals, the sources, and the report values each
reproduces). No session is run. paper_icorr_figures.py, which drew the median-matched version of
this figure, is left unchanged; its style constants, loaders and estimators are imported.
"""

from __future__ import annotations

import os

# Four BLAS threads, set before numpy loads its pool, and only when run as a script (as
# icorr_xsource.py): an import must leave os.environ alone. Assigned, not setdefault: this
# machine's environment presets OMP_NUM_THREADS=22.
if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "4"

import argparse  # noqa: E402
import ast  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import paper_icorr_figures as P  # noqa: E402  (style, loaders, estimators; selects Agg)
import icorr_jnd_free as jf  # noqa: E402
import icorr_r2_report as R  # noqa: E402
import icorr_robust_report as rrep  # noqa: E402
import phase5e_report as er  # noqa: E402
import schaefer_grounding as G  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

KMAX = P.KMAX            # 40
KS = P.KS                # (12, 24, 40)
ROOM = "simulated"
N_BOOT = P.HEAD_BOOT     # 10,000, gpeubo_report section 7's draws

#: The four curves: key -> (right-end label, colour, dashes, width, band alpha). PsyPref and
#: GP-BALD keep fig_headline's styles; the projection takes the orange of fig_learning and Fig. 4,
#: CMA-ES a mid grey, dash-dot (check_palette.py on the four colours: min dE_cvd 19.9; blue and
#: red share L* and are told apart by the dash).
STYLE = {
    "cmaes": ("CMA-ES,\nSchäfer", "#7A7A7A", (0, (5, 1.5, 1, 1.5)), 1.1, 0.13),
    "proj": ("PsyPref,\nSchäfer", P.ORANGE, "-", 1.3, 0.15),
    "gp_bald": (P.DISPLAY["gp_bald"], *P.HEAD_STYLE["gp_bald"], 0.12),
    "a": (P.DISPLAY["a"], *P.HEAD_STYLE["a"], 0.15),
}
ORDER = ("cmaes", "proj", "gp_bald", "a")    # drawing order, back to front
NAME = {"a": "PsyPref (ProMP)", "proj": "PsyPref projected onto Schäfer (ii)",
        "cmaes": "CMA-ES on Schäfer (ii) (BASE)", "gp_bald": "GP-BALD (ProMP)"}
SOURCE = {"a": "out/phase5e/e-main.jsonl, family promp_s0, learner a",
          "proj": "out/icorr/e-proj.jsonl, family schaefer_peaks, learner a_lens",
          "cmaes": "out/icorr/e-base-cma.jsonl, learner cmaes, setting of base_tuned.json",
          "gp_bald": "out/icorr/e-gpbald.jsonl, family promp_s0, learner gp_bald"}
#: The annotations: (curve, target curve, target k), in drawing order.
REACH = (("a", "cmaes", 40), ("a", "gp_bald", 24), ("proj", "cmaes", 40))
#: Where each annotation's label sits: (side, the curve whose band it clears, the k of its left
#: edge, or None for right after the previous label). PsyPref's two labels step up above its band
#: (as in the earlier figure); the projection's sits under CMA-ES's band, where the panel is
#: empty. The sizes are measured, and layout_check lists any overlap.
LABEL_AT = {("a", "cmaes", 40): ("above", "a", 1.4),
            ("a", "gp_bald", 24): ("above", "a", None),
            ("proj", "cmaes", 40): ("below", "cmaes", 16.5)}
LABEL_GAP = 0.6          # k between two labels side by side
LABEL_PAD = 0.008        # preference between a label and the band it clears
MARKER = {"a": ("o", P.BLUE), "proj": ("o", P.ORANGE), "cmaes": ("s", "#7A7A7A"),
          "gp_bald": ("D", P.BALD_RED)}
#: The reach values the brief of 2026-10-25 gives: the first two from the post hoc
#: family-versus-method analysis (posthoc_family_method.txt itself holds no first k; with 2,000
#: draws they reproduce exactly, with 10,000 the projection's upper end is 18), the third is
#: gpeubo_report.md section 7.2 (simulated, 10,000 draws), which headline_reports also parses.
EXPECT_REACH = {("a", "cmaes", 40): "8 [6, 9]", ("proj", "cmaes", 40): "13 [9, 17]",
                ("a", "gp_bald", 24): "13 [10, 16]"}
GPE_REACH_ROW = "A's win rate reaches GP-BALD's at k = 24"
#: posthoc_family_method.txt's names of the drawn cells.
POSTHOC_NAME = {"PsyPref ProMP": "a", "PsyPref proj Sch(ii)": "proj", "BASE cmaes": "cmaes"}
GPE_NAME = {"A": "a", "GP-BALD": "gp_bald"}


# ---------------------------------------------------------------- the simulated room

class SimRoom:
    """jnd_free.Rooms reduced to its "simulated" room: unit weights and room_setup's user draws
    (jnd_free's seed, the first draws taken from it, G.fold_draws), without the Schaefer logs the
    reweighted rooms need. ``stat`` is jnd_free.Rooms.stat itself."""

    stat = jf.Rooms.stat

    def __init__(self, folds, n_boot: int):
        self.sim_draws = G.fold_draws(np.asarray(folds), n_boot, np.random.default_rng(jf.SEED))
        self.point = {ROOM: np.ones(len(folds))}
        self.draw = {ROOM: np.ones(self.sim_draws.shape)}


def check_draws(out: Path, a, folds, room: SimRoom, notes: list) -> None:
    """Stop unless jnd_free.room_setup's simulated draws equal ``room``'s (needs the logs)."""
    data = Path(json.loads((out / "icorr" / "median_room_data_check.json")
                           .read_text(encoding="utf-8"))["data"])
    ok, data_note = P._head_hashes(out, data)
    if not ok:
        raise SystemExit(f"--check-draws: {data_note}")
    print(f"  --check-draws: jnd_free.room_setup ({room.sim_draws.shape[0]} draws) ...",
          flush=True)
    setup = jf.room_setup(out, data, room.sim_draws.shape[0],
                          rrep.col(a, "d_prior") - rrep.col(a, "d_floor"), folds)
    if not np.array_equal(setup["rooms"].sim_draws, room.sim_draws):
        raise SystemExit("--check-draws: jnd_free.room_setup's simulated draws differ")
    if not np.array_equal(setup["rooms"].point[ROOM], room.point[ROOM]):
        raise SystemExit("--check-draws: jnd_free.room_setup's simulated weights differ")
    notes.append(f"Draws checked in this run: jnd_free.room_setup on the Schaefer logs ({data_note}"
                 ") draws the same simulated-room users, draw for draw.")


# ---------------------------------------------------------------- rows

def load_proj(path: Path, keys, notes: list, problems: list):
    """e-proj's Schäfer (ii) projection rows of the 200 population users, in load_sim's order,
    refused unless every row is the main condition (s = 1.0, lapse 0.1, tight arm, data prior,
    phase5e_report's defaults), error-free and 40 comparisons long."""
    P._head_block(path, notes, problems)
    df = jf.stream(path, keep=lambda r: (P._head_pop(r) and r.get("family") == "schaefer_peaks"
                                         and r.get("learner") == "a_lens"),
                   fields=P.HEAD_FIELDS, lists=P.HEAD_LISTS)
    if df is None:
        raise SystemExit(f"{path}: no Schäfer (ii) projection rows")
    if "error" in df and df["error"].notna().any():
        raise SystemExit(f"{path}: {int(df['error'].notna().sum())} error rows")
    ok = (df["arm"] == "tight").to_numpy() & (df["prior"] == "data").to_numpy()
    want = dict(er.DEFAULTS)
    want.update(s=1.0, order_bias=0.0, overdispersion=0.0, plateau_nm=0.0)
    for name, value in want.items():
        ok &= er._match(df, name, value)
    if not ok.all():
        raise SystemExit(f"{path}: {int((~ok).sum())} projection rows are not the main condition")
    c = jf.in_order(df, keys)            # refuses a user twice or a missing user
    why = P._head_complete(c)
    if why:
        raise SystemExit(f"{path}: {why}")
    return c


def load(out: Path, notes: list, problems: list) -> dict:
    run = out / "phase5-schaefer"
    keys = G.load_sim(run)["keys"]
    folds = np.asarray([f for f, _ in keys])
    print("  e-main and the GP blocks (paper_icorr_figures._head_main_cells) ...", flush=True)
    cells, state = P._head_main_cells(out, keys, notes, problems)
    if state["gp_bald"][0] != "ok":
        raise SystemExit(f"GP-BALD {state['gp_bald'][0]} ({state['gp_bald'][1]})")
    a, bald = cells["a"], cells["gp_bald"]
    print("  BASE and e-main through icorr_r2_report.load_all ...", flush=True)
    res = R.load_all(out, run, keys)
    if res["state"].get("BASE-cmaes") != "ok":
        raise SystemExit(f"BASE-cmaes: {res['state'].get('BASE-cmaes')}")
    cma = res["cells"]["BASE-cmaes"]["cmaes"]
    main_a = res["main"]["a"]
    print("  e-proj ...", flush=True)
    proj = load_proj(out / "icorr" / "e-proj.jsonl", keys, notes, problems)

    # one user order everywhere, and e-main A the same through both loaders
    for name, ks_ in (("PsyPref", list(zip(a["fold"], a["user_id"]))),
                      ("GP-BALD", list(zip(bald["fold"], bald["user_id"]))),
                      ("projection", list(zip(proj["fold"], proj["user_id"]))),
                      ("CMA-ES", cma["keys"]), ("e-main A (load_all)", main_a["keys"])):
        if [tuple(k) for k in ks_] != list(keys):
            raise SystemExit(f"{name}: user order differs from load_sim's")
    g = rrep.col(a, "d_prior")
    same_main = (np.array_equal(R.arr(main_a, "d_prior"), g)
                 and np.array_equal(R.arr(main_a, "d_floor"), rrep.col(a, "d_floor"))
                 and np.array_equal(R.mat(main_a, "distances"), rrep.lst(a, "distances")))
    if not same_main:
        raise SystemExit("e-main A differs between _head_main_cells and load_all")
    notes.append("e-main A read through paper_icorr_figures._head_main_cells and through "
                 "icorr_r2_report.load_all: d_prior, d_floor and the 40 distances identical, "
                 "user by user.")
    d_proj = float(np.max(np.abs(rrep.col(proj, "d_prior") - g)))
    d_cma = float(np.max(np.abs(R.arr(cma, "d_prior") - g)))
    d_two = float(np.max(np.abs(rrep.col(proj, "d_prior") - R.arr(cma, "d_prior"))))
    two = "identical in the two blocks" if d_two == 0 else \
        f"the two blocks differ by up to {d_two:.2e} Nm"
    notes.append("Generic assistance for every curve: e-main A's d_prior (R1 of "
                 "icorr_r2_report; GP-BALD's rows carry the same, bit for bit). The e-proj and "
                 "BASE CMA-ES rows carry their own d_prior, the generic of the Schäfer (ii) "
                 f"family ({two}), which differs from e-main A's by up to "
                 f"{max(d_proj, d_cma):.2f} Nm; it is not used.")
    return dict(keys=keys, folds=folds, a=a, bald=bald, proj=proj, cma=cma, g=g,
                base_tuned=res["base_tuned"].get("cmaes"))


# ---------------------------------------------------------------- numbers

def compute(out: Path, n_boot: int, check: bool) -> dict:
    notes: list[str] = []
    problems: list[str] = []
    rows = load(out, notes, problems)
    a, g = rows["a"], rows["g"]
    room = SimRoom(rows["folds"], n_boot)
    if check:
        check_draws(out, a, rows["folds"], room, notes)
    print(f"  curves k = 1..{KMAX}, {n_boot} draws ...", flush=True)
    # per-user preference over generic assistance (users, 40): PsyPref and GP-BALD as
    # fig_headline (their rows' d_prior equals e-main A's bit for bit, _head_gp_cell), the
    # projection and CMA-ES as icorr_r2_report.pref with e-main A's generic
    W = {"a": P._head_win(a), "gp_bald": P._head_win(rows["bald"]),
         "proj": R.y1("rms", g[:, None], rrep.lst(rows["proj"], "distances")),
         "cmaes": R.y1("rms", g[:, None], R.mat(rows["cma"], "distances"))}
    curves = {key: P._room_curve(room, W[key], "win", ROOM) for key in ORDER}

    def at(key, k):
        est, boot = curves[key]
        return est[k - 1], boot[:, k - 1]

    reach = {r: P._first_reach(curves[r[0]], ">=", at(r[1], r[2])) for r in REACH}
    # the same first k with jnd_free's default 2,000 draws (its N_BOOT), for the md only: an
    # order-statistic end can move by one comparison with the number of draws
    room_small = SimRoom(rows["folds"], jf.N_BOOT)
    small = {key: P._room_curve(room_small, W[key], "win", ROOM) for key in ORDER}
    reach_small = {r: P._first_reach(small[r[0]], ">=", (small[r[1]][0][r[2] - 1],
                                                         small[r[1]][1][:, r[2] - 1]))
                   for r in REACH}
    band = {key: dict(est=est, lo=np.percentile(boot, 2.5, axis=0),
                      hi=np.percentile(boot, 97.5, axis=0)) for key, (est, boot) in curves.items()}
    atk = {(key, k): P._head_ci(*at(key, k)) for key in ORDER for k in KS}
    c_user = jf.win_probability(g, rrep.col(a, "d_floor"), rrep.col(a, "s"), rrep.col(a, "lapse"))
    ceiling = P._head_ci(*room.stat("win", c_user, ROOM))
    return dict(band=band, atk=atk, reach=reach, reach_small=reach_small, ceiling=ceiling,
                notes=notes, problems=problems, n_boot=n_boot, n=len(rows["keys"]),
                base_tuned=rows["base_tuned"], checked=check)


# ---------------------------------------------------------------- the reports it reproduces

def posthoc_values(path: Path) -> dict:
    """posthoc_family_method.txt: (key, k) -> its simulated estimate, lo, hi."""
    out = {}
    if not path.exists():
        return out
    pat = re.compile(r"^k=\s*(\d+) \| (.+?)\s*\| simulated (\{.*?\}) \|")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m and m.group(2) in POSTHOC_NAME:
            d = ast.literal_eval(m.group(3))
            out[(POSTHOC_NAME[m.group(2)], int(m.group(1)))] = [d["estimate"], d["lo"], d["hi"]]
    return out


def references(out: Path) -> dict:
    icorr = out / "icorr"
    rep = P.headline_reports(out)
    gpe = {}
    for name, key in GPE_NAME.items():
        for k in KS:
            v = rep.get(("win", ROOM, name, k))
            if v is not None:
                gpe[(key, k)] = v[0]
    proj_num = json.loads((icorr / "proj_numbers.json").read_text(encoding="utf-8"))
    ceil = proj_num["families"]["promp8"]["ceiling"][ROOM]
    return dict(gpe=gpe, posthoc=posthoc_values(icorr / "posthoc_family_method.txt"),
                gpe_reach=rep.get(("reach", GPE_REACH_ROW, ROOM)),
                ceiling=[ceil["est"], ceil["lo"], ceil["hi"]], ceiling_boot=proj_num.get("boot"))


def _status(c: dict, ref) -> str:
    if not ref:
        return "not in a report"
    if not P._close(c["est"], ref[0], 3):
        return f"DIFFERS (report {ref[0]:.3f})"
    if all(P._close(x, r, 3) for x, r in zip((c["lo"], c["hi"]), ref[1:])):
        return "identical at printed precision"
    return "point identical; interval from other draws"


def _reach_txt(d: dict) -> str:
    return f"{jf.fk(d['point'])} [{jf.fk(d['lo'])}, {jf.fk(d['hi'])}]"


def _script_id() -> str:
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", f"scripts/{Path(__file__).name}"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f"{head}{' (script uncommitted)' if dirty else ''}, script sha256 {sha}"
    except (OSError, subprocess.CalledProcessError):
        return f"unknown, script sha256 {sha}"


def numbers_md(hd: dict, ref: dict, out: Path) -> tuple[list[str], list[str]]:
    problems = list(hd["problems"])
    L = ["# fig_headline: plotted numbers", "",
         f"Written by `scripts/{Path(__file__).name}` at "
         f"{_dt.datetime.now().astimezone().isoformat(timespec='seconds')} (repository commit "
         f"{_script_id()}), from the stored rows under `{out}`. No session was run.", "",
         "Figure (the author's decision of 2026-10-25): the preference over generic assistance "
         f"against the comparisons k = 1..{KMAX}, **as simulated** ({hd['n']} population users, "
         "unit weights, no room reweighting), four curves with 95 % bands, 50 % (generic "
         "assistance) and the attainable ceiling of the ProMP K = 8. Every curve is measured "
         "against the same generic assistance, e-main A's d_prior (the K = 8 ProMP prior mean); "
         "the wearer is the plain RMS probit wearer, s = 1.0 Nm, lapse 0.1.", "",
         "Statistics: jnd_free's simulated room, its fold-stratified user draws "
         f"({hd['n_boot']:,} draws, jnd_free's seed {jf.SEED}, the draws room_setup takes first), "
         "through jnd_free.Rooms.stat; bands and intervals are the 2.5 / 97.5 % percentiles of the "
         "draws. First k: jnd_free.first_k, the smallest k at which the cross-sectional mean meets "
         "the target, the target taken in the same draw (paired); interval: 2.5 / 97.5 % order "
         "statistics over the draws (inverted CDF); in parentheses the share of draws that reach "
         "it by k = 40.", "",
         "## Sources", "",
         "| curve | rows |", "|---|---|"]
    L += [f"| {NAME[k]} | {SOURCE[k]} |" for k in ("a", "proj", "cmaes", "gp_bald")]
    L += ["", "Right-end labels in the figure: "
          + "; ".join(f"\"{STYLE[k][0].replace(chr(10), ' ')}\" = {NAME[k]}"
                      for k in ("a", "proj", "cmaes", "gp_bald"))
          + "; \"ceiling\" = the attainable ceiling of the ProMP K = 8; \"generic\" = 50 %."]
    bt = hd.get("base_tuned") or {}
    L += ["", f"CMA-ES's setting: {bt.get('setting')} (base_tuned.json, chosen on fold "
              f"{bt.get('fold')} by the lowest median error at k = 24; that fold's 40 users are "
              "among the 200 drawn, as in r2_report.md).", ""]
    L += [f"- {n}" for n in hd["notes"]]
    if not hd["checked"]:
        L += ["- Draws not checked against jnd_free.room_setup in this run (`--check-draws`)."]

    L += ["", "## Curves at k = 12, 24, 40 (mean over users [95 % CI])", "",
          "| curve | k | plotted | report (simulated) | source of the report value | status |",
          "|---|---|---|---|---|---|"]
    for key in ("a", "proj", "cmaes", "gp_bald"):
        for k in KS:
            c = hd["atk"][(key, k)]
            for src, refs in (("gpeubo_report.md §2.1 (phase5e_report's draws)", ref["gpe"]),
                              ("posthoc_family_method.txt (phase5e_report's draws)",
                               ref["posthoc"])):
                r = refs.get((key, k))
                if r is None:
                    continue
                st = _status(c, r)
                if st.startswith("DIFFERS"):
                    problems.append(f"{NAME[key]} at k = {k}: {P._ci_txt(c)} against {src} "
                                    f"{r[0]:.3f}")
                L.append(f"| {NAME[key]} | {k} | {P._ci_txt(c)} | {r[0]:.3f} [{r[1]:.3f}, "
                         f"{r[2]:.3f}] | {src} | {st} |")

    n_small = jf.N_BOOT
    L += ["", "## Comparisons to reach the comparator's level (the annotations)", "",
          "The figure prints the first k (the point). Its interval is an order statistic of the "
          "draws, so an end can move by one comparison with the number of draws; the column "
          f"with jnd_free's default {n_small:,} draws (same seed) shows where it does.", "",
          f"| first k at which … | plotted ({hd['n_boot']:,} draws) | {n_small:,} draws | "
          "expected (brief of 2026-10-25) | status |", "|---|---|---|---|---|"]
    for r in REACH:
        d, d_small = hd["reach"][r], hd["reach_small"][r]
        target = hd["band"][r[1]]["est"][r[2] - 1]
        what = (f"{NAME[r[0]]} reaches {NAME[r[1]]}'s preference after {r[2]} "
                f"({target:.3f})")
        got, got_small = _reach_txt(d), _reach_txt(d_small)
        exp = EXPECT_REACH[r]
        if got == exp:
            st = "identical"
        elif got_small == exp:
            st = (f"identical with {n_small:,} draws; with {hd['n_boot']:,} an interval end "
                  "differs (point identical)")
        else:
            st = "DIFFERS"
            problems.append(f"reach {r}: {got} ({got_small} with {n_small:,} draws) against the "
                            f"expected {exp}")
        if r == ("a", "gp_bald", 24) and ref["gpe_reach"]:
            rep_cell = ref["gpe_reach"].split(" (")[0]
            st += ("; gpeubo_report.md §7.2 (10,000 draws) " + (
                "identical" if rep_cell == got else f"DIFFERS ({ref['gpe_reach']})"))
            if rep_cell != got:
                problems.append(f"reach {r}: {got} against gpeubo_report.md §7.2 "
                                f"{ref['gpe_reach']}")
        L.append(f"| {what} | {P._fkci(d)} | {P._fkci(d_small)} | {exp} | {st} |")
    L += ["", "Label layout (layout_check: label boxes against each other, the bands, the "
              "reference and level lines and the axes): "
          + ("no overlap." if not hd.get("layout") else "; ".join(hd["layout"]) + ".")]

    c = hd["ceiling"]
    st = _status(c, ref["ceiling"])
    L += ["", "## Reference lines", "",
          "| line | plotted | report | status |", "|---|---|---|---|",
          f"| attainable ceiling, ProMP K = 8 (e-main A's d_prior and d_floor) | {P._ci_txt(c)} | "
          f"{ref['ceiling'][0]:.3f} [{ref['ceiling'][1]:.3f}, {ref['ceiling'][2]:.3f}] "
          f"(proj_numbers.json, promp8, simulated) | {st} |",
          "| generic assistance | 0.500 | by definition | |"]
    if st.startswith("DIFFERS"):
        problems.append(f"ceiling {P._ci_txt(c)} against proj_numbers.json {ref['ceiling'][0]:.3f}")

    L += ["", f"## Every plotted point, k = 1..{KMAX} (mean [95 % band])", "",
          "| k | " + " | ".join(NAME[k] for k in ("a", "proj", "cmaes", "gp_bald")) + " |",
          "|---|" + "---|" * 4]
    for i in range(KMAX):
        cells = [f"{hd['band'][k]['est'][i]:.3f} [{hd['band'][k]['lo'][i]:.3f}, "
                 f"{hd['band'][k]['hi'][i]:.3f}]" for k in ("a", "proj", "cmaes", "gp_bald")]
        L.append(f"| {i + 1} | " + " | ".join(cells) + " |")
    L += ["", "## Numbers that do not reproduce", ""]
    L += [f"- {p}" for p in problems] if problems else ["None."]
    return L, problems


# ---------------------------------------------------------------- the figure

def spread_h(targets, heights) -> np.ndarray:
    """paper_icorr_figures.spread with a height per label: centres as close to ``targets`` as
    blocks allow, neighbours at least (h_i + h_j) / 2 apart; each block of crowded labels is
    placed to minimise the squared shifts of its labels."""
    t = np.asarray(targets, dtype=float)
    h = np.asarray(heights, dtype=float)
    order = np.argsort(t)
    ts, hs = t[order], h[order]

    def offsets(b):
        o = [0.0]
        for i, j in zip(b[:-1], b[1:]):
            o.append(o[-1] + (hs[i] + hs[j]) / 2)
        return np.array(o)

    def base(b):
        return float(np.mean(ts[b] - offsets(b)))

    blocks = [[i] for i in range(len(ts))]
    merged = True
    while merged:
        merged = False
        for k in range(len(blocks) - 1):
            b1, b2 = blocks[k], blocks[k + 1]
            top1 = base(b1) + offsets(b1)[-1] + hs[b1[-1]] / 2
            bot2 = base(b2) - hs[b2[0]] / 2
            if bot2 < top1 - 1e-12:
                blocks[k] = b1 + b2
                del blocks[k + 1]
                merged = True
                break
    pos = np.empty(len(ts))
    for b in blocks:
        pos[b] = base(b) + offsets(b)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def fig_headline(hd: dict, figdir: Path) -> Path:
    W, H = P.FIG_W, 2.30
    left, width, bottom, height = 0.50, 2.12, 0.36, 1.86
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, width / W, height / H])
    B = hd["band"]
    ks = np.arange(1, KMAX + 1)
    ceil = hd["ceiling"]["est"]
    ax.axhline(0.5, color=P.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.axhline(ceil, color=P.INK, lw=0.9, ls=(0, (2.4, 1.3)), zorder=0)
    for key in ORDER:
        _, colour, _, _, alpha = STYLE[key]
        ax.fill_between(ks, B[key]["lo"], B[key]["hi"], color=colour, alpha=alpha, lw=0, zorder=1)
    for key in ORDER:
        _, colour, ls, lw, _ = STYLE[key]
        ax.plot(ks, B[key]["est"], color=colour, ls=ls, lw=lw, zorder=3 if key == "a" else 2,
                solid_capstyle="butt")

    # the comparison savings: the comparator's level (dotted, from the first k at which the
    # faster curve reaches it to the comparator's own k; one line per level, so that two
    # annotations on one level do not double its dots), a marker where the faster curve first
    # reaches it and one where the comparator has it
    levels: dict = {}
    for r in REACH:
        levels.setdefault((r[1], r[2]), []).append(int(hd["reach"][r]["point"]))
    for (slow, kk), firsts in levels.items():
        level = float(B[slow]["est"][kk - 1])
        ax.plot([min(firsts), kk], [level, level], color=P.INK, lw=0.6, ls=(0, (2.2, 1.4)),
                zorder=4)
        m, c = MARKER[slow]
        ax.plot([kk], [level], marker=m, ms=3.4, mfc=c, mec="white", mew=0.5, ls="none",
                zorder=5, clip_on=False)      # k = 40 lies on the axes' edge
    for fast, slow, kk in REACH:
        ka = int(hd["reach"][(fast, slow, kk)]["point"])
        m, c = MARKER[fast]
        ax.plot([ka], [float(B[fast]["est"][ka - 1])], marker=m, ms=3.8, mfc=c, mec="white",
                mew=0.5, ls="none", zorder=5)

    top = ceil + 0.035
    lo_y = min(float(np.min(B[k]["est"])) for k in ORDER)
    ax.set_xlim(1, KMAX)
    ax.set_xticks([1, 12, 24, 40])
    ax.set_ylim(np.floor(lo_y * 20) / 20, top)
    ticks = [t for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) if ax.get_ylim()[0] <= t <= top]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{100 * t:.0f}" for t in ticks])
    ax.set_xlabel("comparisons k", labelpad=1.5)
    ax.set_ylabel("preference over generic\nassistance (%)", labelpad=2)
    P._spines(ax, grid_axis="y")
    boxes = annotate(ax, hd)
    hd["layout"] = layout_check(hd, boxes, ax.get_ylim())

    # direct labels at the right end: the four curves, then the two reference lines
    # (spacing as fig_headline's, 0.125 in per one-line label; a two-line label takes its
    # measured height and a small gap)
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    line_h = 0.125 * span / height
    keys = ["a", "gp_bald", "proj", "cmaes"]
    ends = [float(B[k]["est"][-1]) for k in keys] + [ceil, 0.5]
    names = [STYLE[k][0] for k in keys] + ["ceiling", "generic"]
    heights = [max(line_h, _text_h(ax, n) + 0.3 * line_h) for n in names]
    pos = spread_h(ends, heights)
    for i, (y_end, y_lab, name) in enumerate(zip(ends, pos, names)):
        ax.plot([KMAX + 0.4, KMAX + 1.4], [y_end, y_lab], color=P.CONTEXT, lw=0.5, clip_on=False)
        if i < len(keys):
            _, colour, ls, lw, _ = STYLE[keys[i]]
        else:
            colour, ls, lw = (P.INK, (0, (2.4, 1.3)), 0.9) if name == "ceiling" else \
                (P.MUTED, (0, (3, 2)), 0.7)
        ax.plot([KMAX + 1.4, KMAX + 4.0], [y_lab, y_lab], color=colour, ls=ls, lw=lw,
                clip_on=False, solid_capstyle="butt")
        ax.text(KMAX + 4.6, y_lab, name, ha="left", va="center", color=P.INK, fontsize=8,
                linespacing=0.92, clip_on=False)
    path = figdir / "fig_headline.pdf"
    fig.savefig(path, format="pdf", metadata={"CreationDate": None})
    fig.savefig(figdir / "fig_headline.png", dpi=300)
    plt.close(fig)
    return path


def _text_h(ax, s: str) -> float:
    """The height of a right-end label (8 pt, line spacing 0.92) in data units."""
    t = ax.text(0, 0, s, fontsize=8, linespacing=0.92)
    bb = t.get_window_extent(ax.figure.canvas.get_renderer())
    t.remove()
    (_, y0), (_, y1) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return float(y1 - y0)


def _band_at(band: np.ndarray, x0: float, x1: float, fn) -> float:
    """``fn`` (max or min) of a curve k = 1..40 over [x0, x1], linear between the ks."""
    xs = np.concatenate([[x0], np.arange(np.ceil(x0), np.floor(x1) + 1), [x1]])
    return float(fn(np.interp(xs, np.arange(1, KMAX + 1), band)))


def annotate(ax, hd: dict) -> list[tuple]:
    """The labels of REACH, each placed by LABEL_AT with its size measured, and a leader from
    the faster curve's marker; returns the boxes (x0, x1, y0, y1) in data units."""
    B = hd["band"]
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    boxes, x_next = [], None
    for r in REACH:
        fast, slow, kk = r
        ka = int(hd["reach"][r]["point"])
        y_a = float(B[fast]["est"][ka - 1])
        text = f"same result in\n{ka} vs {kk}"
        t = ax.text(0, 0, text, ha="left", va="bottom", multialignment="center", fontsize=8,
                    color=P.INK, linespacing=1.0, zorder=6)
        bb = t.get_window_extent(renderer)
        (u0, v0), (u1, v1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        w, h = u1 - u0, v1 - v0
        side, clear, x0 = LABEL_AT[r]
        x0 = x_next + LABEL_GAP if x0 is None else x0
        x1 = x0 + w
        if side == "above":
            y0 = _band_at(B[clear]["hi"], x0, x1, np.max) + LABEL_PAD
            y1 = y0 + h
            leader = ((ka, y_a + 0.008), (min(max(ka, x0 + 1.0), x1 - 1.0), y0 - 0.003))
        else:
            y1 = _band_at(B[clear]["lo"], x0, x1, np.min) - LABEL_PAD
            y0 = y1 - h
            leader = ((ka, y_a - 0.008), (min(max(ka, x0 + 1.0), x1 - 1.0), y1 + 0.003))
        t.set_position((x0, y0))
        (xa, ya), (xb, yb) = leader
        ax.plot([xa, xb], [ya, yb], color=P.MUTED, lw=0.5, zorder=4)
        boxes.append((x0, x1, y0, y1))
        x_next = x1
    return boxes


def layout_check(hd: dict, boxes: list[tuple], ylim: tuple) -> list[str]:
    """Every overlap of a label box with another, a band, a curve, a reference line or a level
    line, or with the axes' edges."""
    B = hd["band"]
    ceil = hd["ceiling"]["est"]
    out = []
    for i, (x0, x1, y0, y1) in enumerate(boxes):
        name = f"label {jf.fk(hd['reach'][REACH[i]]['point'])} vs {REACH[i][2]}"
        if x0 < 1 or x1 > KMAX or y0 < ylim[0] or y1 > ylim[1]:
            out.append(f"{name} leaves the axes")
        for j, (a0, a1, b0, b1) in enumerate(boxes[:i]):
            if x0 < a1 and a0 < x1 and y0 < b1 and b0 < y1:
                out.append(f"{name} overlaps label {j + 1}")
        for key in ORDER:
            hi = _band_at(B[key]["hi"], x0, x1, np.max)
            lo = _band_at(B[key]["lo"], x0, x1, np.min)
            if not (y0 > hi or y1 < lo):
                # the box spans the band's range; check point by point
                xs = np.linspace(x0, x1, 200)
                bh = np.interp(xs, np.arange(1, KMAX + 1), B[key]["hi"])
                bl = np.interp(xs, np.arange(1, KMAX + 1), B[key]["lo"])
                if np.any((bh >= y0) & (bl <= y1)):
                    out.append(f"{name} overlaps the band of {NAME[key]}")
        for y, what in ((ceil, "the ceiling"), (0.5, "the 50 % line")):
            if y0 - 0.004 <= y <= y1 + 0.004:
                out.append(f"{name} touches {what}")
        for r in REACH:
            ka = int(hd["reach"][r]["point"])
            level = float(B[r[1]]["est"][r[2] - 1])
            if y0 <= level <= y1 and x0 < r[2] and ka < x1:
                out.append(f"{name} crosses a level line")
    return out


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out",
                    help="the sweep output directory holding phase5e/, icorr/, phase5-schaefer/")
    ap.add_argument("--figdir", type=Path, default=ROOT / "paper" / "figures")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"bootstrap draws ({N_BOOT:,} reproduces gpeubo_report section 7)")
    ap.add_argument("--check-draws", action="store_true",
                    help="run jnd_free.room_setup on the Schaefer logs and stop unless its "
                         "simulated draws are the ones used here")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    args.figdir.mkdir(parents=True, exist_ok=True)
    print("fig_headline (as simulated):")
    hd = compute(args.out, args.boot, args.check_draws)
    print(f"wrote {fig_headline(hd, args.figdir)}")
    lines, problems = numbers_md(hd, references(args.out), args.out)
    md = args.figdir / "fig_headline_numbers.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md}")
    P._print_problems(problems)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
