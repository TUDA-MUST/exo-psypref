"""Figures for the ICORR 2027 paper, read from the stored Phase 5e sweep rows.

Usage:  python scripts/paper_icorr_figures.py --out <out dir> [--figdir paper/figures]
            [--boot 10000] [--preview <dir for PNG previews>] [--data <Schaefer logs>]
            [--only headline]

``--only headline`` draws fig_headline.pdf alone and replaces its section of an existing
``figure_numbers.md``; the other sections are left as they are.

Writes four vector PDFs at IEEE single-column width (3.5 in, 8-pt type) and
``figure_numbers.md``, which lists every plotted number next to the REPORT.md,
own_generic.md or grounding value it reproduces:

  fig_session.pdf     (a) one simulated user of the main condition: ideal, generic assistance
                      and A's identified assistance after 12 and 24 comparisons; (b) the
                      psychometric function of the simulated simulated user.
  fig_learning.pdf    the main condition, k = 1..40: (a) median error of the identified
                      assistance with its IQR, (b) the share of users perceptibly better than
                      generic assistance.
  fig_robustness.pdf  one row per cell: (a) A minus the best comparator at k = 24 on ProMP,
                      the difference of the medians on the same users; (b) A's lower decile of
                      gain; with each cell's perturbation strength |Δp|.
  fig_room.pdf        the main condition at k = 24: the share of simulated users perceptibly
                      better than generic assistance as a function of their room for
                      personalization (d_prior - d_floor on ProMP), a kernel regression per
                      optimizer, with the Schaefer timing-only, literature-informed central and
                      high rooms and the simulated median marked.
  fig_axes.pdf        results along the simulation's axes, one measure in every panel: the
                      preference over generic assistance (the reports' "win rate") after 24
                      comparisons, PsyPref and the strongest comparator run in each condition,
                      with the attainable best-profile level and 50 %, in one row of three panels: (a)
                      sensitivity and (b) lapse rate and device noise at the median-matched
                      room (best-profile level.md), (c) the user models as simulated (gpeubo_report,
                      robust_report, wph_report; best-profile level from best-profile level.md). Every number is
                      read from the reports in out/icorr; nothing is recomputed. The basis
                      count K, panel (d) until 2026-10-22, moved to fig_families
                      (icorr_proj_report.py), which plots every family by parameter count.
  fig_headline.pdf    the main condition at the median-matched room (post hoc), one panel: the
                      preference over generic assistance, k = 1..40, for PsyPref, GP-BALD and
                      the other comparators, with the comparisons PsyPref needs to reach
                      GP-BALD's preference after 24 and 40, and the attainable best-profile level. Rooms,
                      draws and first k exactly as icorr_gpeubo_report.py's section 7
                      (jnd_free's room_setup, 10,000 draws). The basis-count numbers are still
                      computed and recorded in figure_numbers.md (Fig. 2 (d) draws them).

Display names: every optimizer name a figure prints comes from DISPLAY (one constant).

Memory: the rows are streamed line by line (``lean_load``, patched into the report modules'
loader). Only the blocks the figures read are kept, only population users without an error,
only the s = 0.5 reference rows of 5c/5d, and every per-comparison list cut to the first 40
comparisons; the trace fields no figure reads are dropped while parsing.

No sweep is run. Every number comes from the stored rows through phase5e_report's cells and
estimators (``Ctx``, ``align``, ``_med``, ``paired_bootstrap``), benchmark_report's levels and
lower decile, and phase5e_own_generic's own-metric gains. The one session that is run again is
the example user of fig_session, through ``run_phase5_point`` at 12 and 24 comparisons; the
script stops unless that rerun reproduces the stored answers and distances exactly. Population
users only, 200 per cell. With the default 10,000 draws and the same seeds the intervals are the
ones in REPORT.md and own_generic.md, not merely close to them.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import benchmark_report as bm  # noqa: E402
import icorr_jnd_free as jf  # noqa: E402
import icorr_k_report as kr  # noqa: E402
import icorr_robust_report as rrep  # noqa: E402
import phase5e_own_generic as og  # noqa: E402
import phase5e_report as r5  # noqa: E402
import schaefer_grounding as G  # noqa: E402
import short_budget_report as sb  # noqa: E402
from preftune.profile import PHASE_GRID  # noqa: E402

from prompref.experiment import phase5 as p5  # noqa: E402
from prompref.experiment.phase5 import Phase5Point, run_phase5_point  # noqa: E402
from prompref.experiment.phase5c_stats import paired_bootstrap  # noqa: E402
from prompref.population.curves import TORQUE_LIMIT_NM  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402
from prompref.user.simulated_user import rms_distance  # noqa: E402

K = r5.K_MAIN            # 24, the main budget
KMAX = 40                # the e-main sessions' length
KS = (12, 24, 40)
LAPSE = 0.1
JND = r5.JND_MAIN        # true_jnd(1.0) = 0.674 Nm
FIG_W = 3.5              # IEEE single column, inches
FIELDS = [f.name for f in dataclasses.fields(Phase5Point)]
TOL = 1e-12              # benchmark_report.TOL

# ---------------------------------------------------------------- memory-lean rows
#: The blocks the figures read. Every other block of 5c/5d/5e loads as missing (None), so its
#: rows never enter memory: e-tune, e-tuned and e-simulated user (no figure reads them), 5c's noise,
#: region, attribution, baseline and main-dual blocks and 5d's other blocks (the s = 0.5
#: reference cell reads 5c main-promp, 5d cmaes and 5d lens-franks only).
LEAN_BLOCKS = {"e-main", "e-s", "e-observer", "e-noise", "e-drift", "e-lapse",
               "main-promp", "cmaes", "lens-franks"}
#: Per-comparison lists the figures read: cut to the first KMAX comparisons.
LEAN_CUT = ("distances", "answers", "own_distances", "own_feature_jnd")
#: Per-comparison traces no figure reads: dropped while parsing.
LEAN_DROP = ("duel_trace", "duel_kind", "ell_trace", "lambda_trace", "duel_intensity")
LEAN_STATS: dict = {}


def lean_load(directory: Path, name: str):
    """``phase5c_report._load`` without the memory: the rows of ``<directory>/<name>.jsonl``,
    streamed line by line, as a DataFrame of population users without an error (load_sweep
    drops error rows; every consumer here keeps population users only). Rows of the 5c/5d
    blocks are kept only in the s = 0.5, lapse 0.1 reference (the cell the figures read). The
    lists in LEAN_CUT keep their first KMAX values, the traces in LEAN_DROP are dropped."""
    path = Path(directory) / f"{name}.jsonl"
    if name not in LEAN_BLOCKS or not path.exists():
        return None
    reference_only = not name.startswith("e-")
    rows, read = [], 0
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.warn(f"{path}: skipping unparseable line {number} ({exc})")
                continue
            read += 1
            if r.get("error") is not None or r.get("user_kind") != "population":
                continue
            if "/pop" not in str(r.get("user_id", "")):
                continue
            if reference_only and not (abs(float(r.get("s", -1)) - 0.5) < 1e-9
                                       and abs(float(r.get("lapse", -1)) - 0.1) < 1e-9):
                continue
            for f in LEAN_DROP:
                r.pop(f, None)
            for f in LEAN_CUT:
                if isinstance(r.get(f), list):
                    r[f] = r[f][:KMAX]
            r.pop("error", None)
            rows.append(r)
    LEAN_STATS[name] = (read, len(rows))
    if not rows:
        return None
    return pd.DataFrame(rows)


# phase5e_report.Ctx and short_budget_report.Rows call ``sb._load``; this replaces it.
sb._load = lean_load

# ---------------------------------------------------------------- display names
#: Every optimizer name any figure prints, in one place: change a name here and every legend,
#: label and figure_numbers.md table follows. The keys are the code's (and the rows') names,
#: which never change; the reports' names (A, G, SortCMA, GP-EUBO) are kept separately
#: (HEAD_GPE_NAME, HEAD_KS_NAME, AXES_REPORT_NAME) because they are data keys, not display.
#: GP names as decided by the author on 2026-10-21 (technical names, explained at first use in
#: the paper).
DISPLAY = {
    "a": "PsyPref",
    "a_lens": "PsyPref, Franks",          # A with the Franks projection ("lens" in the code)
    "a_rand": "PsyPref, random comparisons",
    "gp_bald": "GP-BALD",
    "gp_eubo": "GP-qEUBO",
    "g": "Arens' GP",
    "d_z": "LineCoSpar",
    "cmaes": "CMA-ES",
}

# ---------------------------------------------------------------- style
# A strong blue, the lens orange, the comparators three greys with their own dash patterns
# (figure-colour skill, blue anchor; check_palette.py: min dE_cvd 19.4, min dL* 9.4).
INK, MUTED, CONTEXT, RULE = "#222222", "#666666", "#BBBBBB", "#DDDDDD"
BLUE, BLUE_LIGHT, ORANGE = "#1845FB", "#8FA4FD", "#EE7733"
SERIES = {  # key: (family, learner, label, colour, dashes, width); labels from DISPLAY
    "a": ("promp_s0", "a", DISPLAY["a"], BLUE, "-", 1.6),
    # ("PsyPref, projection" is clipped at the right edge of fig_learning and fig_room)
    "a_lens": ("franks", "a_lens", DISPLAY["a_lens"], ORANGE, "-", 1.3),
    "g": ("promp_s0", "g", DISPLAY["g"], "#333333", (0, (4, 1.6)), 1.1),
    "d_z": ("promp_s0", "d_z", DISPLAY["d_z"], "#808080", (0, (5, 1.5, 1, 1.5)), 1.1),
    "cmaes": ("promp_s0", "cmaes", DISPLAY["cmaes"], "#B3B3B3", (0, (1, 1.2)), 1.5),
}
SERIES_ORDER = ("a", "a_lens", "g", "d_z", "cmaes")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "axes.unicode_minus": True,
})


def _spines(ax, grid_axis: str | None = None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#EBEBEB", linewidth=0.5)
        ax.set_axisbelow(True)


def _panel(ax, letter: str, x: float = -0.02, y: float = 1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontweight="bold", color=INK)


def _save(fig, figdir: Path, name: str, preview: Path | None) -> Path:
    path = figdir / f"{name}.pdf"
    fig.savefig(path, format="pdf", metadata={"CreationDate": None})
    if preview is not None:
        fig.savefig(preview / f"{name}.png", dpi=300)
    plt.close(fig)
    return path


def spread(targets, gap: float) -> np.ndarray:
    """Label positions at least ``gap`` apart, as close to ``targets`` as blocks allow: each
    block of crowded labels is centred on the mean of its targets."""
    t = np.asarray(targets, dtype=float)
    order = np.argsort(t)
    ts = t[order]
    blocks = [[i] for i in range(len(ts))]

    def centre(b):
        n = len(b)
        return float(np.mean([ts[i] - (j - (n - 1) / 2) * gap for j, i in enumerate(b)]))

    merged = True
    while merged:
        merged = False
        for k in range(len(blocks) - 1):
            b1, b2 = blocks[k], blocks[k + 1]
            top1 = centre(b1) + (len(b1) - 1) / 2 * gap
            bot2 = centre(b2) - (len(b2) - 1) / 2 * gap
            if bot2 - top1 < gap - 1e-12:
                blocks[k] = b1 + b2
                del blocks[k + 1]
                merged = True
                break
    pos = np.empty(len(ts))
    for b in blocks:
        c = centre(b)
        for j, i in enumerate(b):
            pos[i] = c + (j - (len(b) - 1) / 2) * gap
    out = np.empty_like(pos)
    out[order] = pos
    return out


# ---------------------------------------------------------------- REPORT.md parsing

def parse_rows(path: Path) -> list[tuple[str, list[str]]]:
    """Every markdown table row of a file as (the heading above it, its cells)."""
    out, heading = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            continue
        if not line.startswith("|") or line.startswith("|---"):
            continue
        out.append((heading, [c.strip() for c in line.strip().strip("|").split("|")]))
    return out


def parse_tables(path: Path) -> dict:
    """``{(heading, first cell): [cells]}``, the first row of each key (REPORT.md repeats the
    main-condition tables in sections 1 and 4 with the same content)."""
    out = {}
    for heading, cells in parse_rows(path):
        out.setdefault((heading, cells[0]), cells)
    return out


NUM = re.compile(r"[-+]?\d+\.\d+")


def nums(text: str) -> list[float]:
    return [float(x) for x in NUM.findall(text)]


def _close(a: float, b: float, digits: int) -> bool:
    """Equal at the printed precision: ``a`` printed as the reports print it is ``b``."""
    return float(f"{a:.{digits}f}") == float(f"{b:.{digits}f}")


def verdict(computed: list[float], reported: list[float], digits: int, tol: float = 0.01) -> str:
    """'reproduces' when the point estimate is within ``tol`` and every number agrees at the
    printed precision; 'reproduces (CI differs ...)' when only the interval differs."""
    if not reported:
        return "NOT FOUND in the report"
    if abs(computed[0] - reported[0]) > tol:
        return f"DIFFERS (point estimate off by {abs(computed[0] - reported[0]):.3f})"
    if all(_close(c, r, digits) for c, r in zip(computed, reported)):
        return "reproduces (identical at printed precision)"
    worst = max(abs(c - r) for c, r in zip(computed[1:], reported[1:]))
    return f"reproduces (point estimate; CI ends differ by up to {worst:.3f})"


# ---------------------------------------------------------------- figure 1

def _point_from_row(row) -> Phase5Point:
    kw = {}
    for f in FIELDS:
        if f not in row.index:
            continue
        v = row[f]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        kw[f] = v
    return Phase5Point(**kw)


def session_example(ctx) -> dict:
    """A's main-condition user whose error at k = 24 is the median (the lower middle value,
    inverted_cdf, as the contrasts take it), with its curves rebuilt by rerunning the session
    at 12 and 24 comparisons (exact under truncation, B3)."""
    df = ctx.df("main", "promp_s0", "a")
    d24 = r5.values_at(df, K).to_numpy()
    median = float(r5._med(d24))
    i = int(np.argmin(np.abs(d24 - median)))
    row = df.iloc[i]
    point = _point_from_row(row)
    user = p5._users(point.run_dir, point.fold)[point.user_id]
    ideal = np.asarray(user.curve_nm, dtype=float)
    design = p5.arm_space(p5._prior(point.run_dir, point.fold, point.family), point.arm).design
    generic = np.asarray(og.generic_curve(point), dtype=float)
    d_generic = rms_distance(generic, ideal)
    if abs(d_generic - float(row["d_prior"])) > 1e-9:
        raise SystemExit(f"generic assistance does not reproduce d_prior: {d_generic} vs "
                         f"{row['d_prior']}")
    identified, errors = {}, {}
    for k in (12, 24):
        rec = run_phase5_point(dataclasses.replace(point, n_comparisons=k))
        if rec.get("error"):
            raise SystemExit(f"rerun failed at k = {k}: {rec['error']}")
        stored = [float(x) for x in row["distances"][:k]]
        if rec["distances"] != stored or rec["answers"] != [int(a) for a in row["answers"][:k]]:
            raise SystemExit(f"the rerun at k = {k} does not reproduce the stored session")
        curve = np.asarray(design.curve(np.asarray(rec["best"], dtype=float)), dtype=float)
        errors[k] = rms_distance(curve, ideal)
        if abs(errors[k] - stored[-1]) > 1e-12:
            raise SystemExit(f"identified assistance at k = {k} does not reproduce "
                             f"distances[{k - 1}]")
        identified[k] = curve
    touches = int(sum(bool(np.any(np.abs(np.asarray(
        p5._users(f, fo)[u].curve_nm)) >= TORQUE_LIMIT_NM - 1e-9))
        for f, fo, u in zip(df["run_dir"], df["fold"], df["user_id"])))
    return dict(fold=point.fold, user_id=point.user_id, rank_note=f"{i}", median=median,
                usual_median=float(np.median(d24)), ideal=ideal, generic=generic,
                identified=identified, errors=errors, d_prior=float(row["d_prior"]),
                d_floor=float(row["d_floor"]),
                touches_limit=bool(np.any(np.abs(ideal) >= TORQUE_LIMIT_NM - 1e-9)),
                n_touch=touches, n=len(df),
                median_d_prior=float(np.median(df["d_prior"].astype(float))),
                median_gain_nm=float((df["d_prior"].astype(float) - r5.values_at(df, K)).median()))


def psychometric(delta, s: float, lapse: float = LAPSE):
    """P(choose the curve closer to the ideal) for a difference of RMS distances ``delta``:
    the plain observer's ``lapse / 2 + (1 - lapse) Phi(delta / s)`` (oracle.prob_prefers_a)."""
    return lapse / 2.0 + (1.0 - lapse) * norm.cdf(np.asarray(delta, dtype=float) / s)


def fig_session(ex: dict, figdir: Path, preview) -> tuple[Path, dict]:
    H = 1.9
    fig = plt.figure(figsize=(FIG_W, H))
    ax = fig.add_axes([0.42 / FIG_W, 0.40 / H, 1.42 / FIG_W, 1.30 / H])
    bx = fig.add_axes([2.30 / FIG_W, 0.40 / H, 1.15 / FIG_W, 1.30 / H])

    # (a) the session of one simulated user
    x = np.asarray(PHASE_GRID, dtype=float)
    for lim in (-TORQUE_LIMIT_NM, TORQUE_LIMIT_NM):
        ax.axhline(lim, color=CONTEXT, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
    ax.axhline(0.0, color=RULE, lw=0.5, zorder=0)
    ax.plot(x, ex["generic"], color="#808080", lw=1.1, ls=(0, (4, 1.6)), zorder=2)
    ax.plot(x, ex["identified"][12], color=BLUE_LIGHT, lw=1.2, zorder=3)
    ax.plot(x, ex["identified"][24], color=BLUE, lw=1.5, zorder=4)
    ax.plot(x, ex["ideal"], color=INK, lw=1.0, zorder=5)
    ax.text(100, TORQUE_LIMIT_NM + 0.5, f"device limit ±{TORQUE_LIMIT_NM:g} Nm", ha="right",
            va="bottom", color=MUTED, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(-14.5, 14.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([-12, -6, 0, 6, 12])
    ax.set_xlabel("gait cycle (%)")
    ax.set_ylabel("hip torque (Nm), flexion +", labelpad=1)
    _spines(ax)
    _panel(ax, "a", x=-0.03)
    handles = [Line2D([], [], color=INK, lw=1.0),
               Line2D([], [], color="#808080", lw=1.1, ls=(0, (4, 1.6))),
               Line2D([], [], color=BLUE_LIGHT, lw=1.2),
               Line2D([], [], color=BLUE, lw=1.5)]
    labels = ["ideal", "generic", f"{DISPLAY['a']}, k = 12", f"{DISPLAY['a']}, k = 24"]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(-0.01, 0.925), frameon=False,
              handlelength=1.3, handletextpad=0.35, labelspacing=0.08, borderaxespad=0.0,
              fontsize=8)

    # (b) the psychometric function of the simulated simulated user
    xmax = 2.5
    delta = np.linspace(0.0, xmax, 251)
    bx.axhline(0.5, color=RULE, lw=0.5, zorder=0)
    bx.axhline(1 - LAPSE / 2, color=CONTEXT, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
    bx.plot(delta, norm.cdf(delta / 1.0), color=CONTEXT, lw=0.8, zorder=1)
    bx.plot(delta, psychometric(delta, 0.5), color="#808080", lw=0.8, ls=(0, (3, 1.5)),
            zorder=2)
    bx.plot(delta, psychometric(delta, 2.0), color="#808080", lw=0.9, ls=(0, (1, 1)), zorder=2)
    bx.plot(delta, psychometric(delta, 1.0), color=INK, lw=1.4, zorder=3)
    jnd = true_jnd(1.0)
    p_jnd = float(psychometric(jnd, 1.0))
    bx.plot([jnd, jnd], [0.45, 0.75], color=BLUE, lw=0.8, ls=(0, (1, 1)), zorder=4)
    bx.plot([jnd], [0.75], marker="o", ms=3.2, mfc="white", mec=BLUE, mew=0.9, zorder=5)
    bx.plot([jnd], [p_jnd], marker="o", ms=3.2, color=BLUE, zorder=5)
    bx.annotate(f"JND {jnd:.3f} Nm", xy=(jnd, 0.465), xytext=(3, 0),
                textcoords="offset points", ha="left", va="bottom", color=BLUE, fontsize=8)
    # Rv15: left of the steep s = 0.5 curve the label sat on the y-axis spine
    bx.text(0.3, 0.99, "s = 0.5", ha="left", va="center", color=MUTED, fontsize=8)
    bx.text(1.12, 0.805, "s = 1.0", ha="left", va="center", color=INK, fontsize=8)
    bx.text(2.05, 0.765, "s = 2.0", ha="center", va="top", color=MUTED, fontsize=8)
    bx.text(xmax, 1.0, "no lapse", ha="right", va="bottom", color="#999999", fontsize=8)
    bx.set_xlim(0, xmax)
    bx.set_ylim(0.45, 1.05)
    bx.set_xticks([0, 1, 2])
    bx.set_yticks([0.5, 0.75, 1 - LAPSE / 2])
    bx.set_yticklabels(["0.5", "0.75", "0.95"])
    bx.set_xlabel("Δ RMS distance (Nm)")
    bx.set_ylabel("P(choose closer curve)", labelpad=1)
    _spines(bx)
    _panel(bx, "b", x=-0.03)
    numbers = dict(jnd={s: true_jnd(s) for s in (0.5, 1.0, 2.0)}, p_at_jnd=p_jnd,
                   p_at_jnd_s={s: float(psychometric(true_jnd(s), s)) for s in (0.5, 1.0, 2.0)},
                   asymptotes=(LAPSE / 2, 1 - LAPSE / 2))
    return _save(fig, figdir, "fig_session", preview), numbers


# ---------------------------------------------------------------- figure 2

def learning_data(ctx) -> dict:
    """Per series: the median and quartiles of the error at k = 1..40 (linear percentiles, as
    benchmark_report._curve takes them) and the share with gain >= 1 JND (bm.share_curve)."""
    out = {}
    for key in SERIES_ORDER:
        family, learner, *_ = SERIES[key]
        df = ctx.df("main", family, learner)
        if df is None or len(df) != r5.EXPECTED_N:
            raise SystemExit(f"main {family} {learner}: expected {r5.EXPECTED_N} users")
        matrix = np.array([d[:KMAX] for d in df["distances"]], dtype=float)
        med, q25, q75 = np.percentile(matrix, [50, 25, 75], axis=0)
        share = bm.share_curve(df, "q1", kmax=KMAX, jnd=JND)
        out[key] = dict(df=df, n=len(df), median=med, q25=q25, q75=q75, share=share,
                        d_prior_median=float(np.median(df["d_prior"].astype(float))))
    return out


def _end_labels(ax, data: dict, field: str, gap: float, x_end: float = KMAX):
    ys = [float(data[k][field][-1]) for k in SERIES_ORDER]
    pos = spread(ys, gap)
    for key, y_end, y_lab in zip(SERIES_ORDER, ys, pos):
        _, _, label, colour, ls, lw = SERIES[key]
        ax.plot([x_end + 0.4, x_end + 1.4], [y_end, y_lab], color=CONTEXT, lw=0.5,
                clip_on=False)
        ax.plot([x_end + 1.4, x_end + 4.0], [y_lab, y_lab], color=colour, ls=ls, lw=lw,
                clip_on=False, solid_capstyle="butt")
        ax.text(x_end + 4.6, y_lab, label, ha="left", va="center", color=INK, fontsize=8,
                clip_on=False)


def fig_learning(data: dict, figdir: Path, preview) -> Path:
    # v2 (page budget): 3.3 -> 2.95 in tall, same 8-pt text; end-label gaps widened to match
    H = 2.95
    fig = plt.figure(figsize=(FIG_W, H))
    left, width = 0.58 / FIG_W, 1.92 / FIG_W
    ax = fig.add_axes([left, 1.60 / H, width, 1.17 / H])
    bx = fig.add_axes([left, 0.36 / H, width, 1.05 / H], sharex=ax)
    ks = np.arange(1, KMAX + 1)

    # Rv14: interquartile bands for A and G only (five overlapping bands did not read)
    for key in ("g", "a"):
        _, _, _, colour, ls, lw = SERIES[key]
        d = data[key]
        ax.fill_between(ks, d["q25"], d["q75"], color=colour,
                        alpha=0.16 if key == "a" else 0.10, lw=0)
    for key in reversed(SERIES_ORDER):
        _, _, _, colour, ls, lw = SERIES[key]
        ax.plot(ks, data[key]["median"], color=colour, ls=ls, lw=lw)
        bx.plot(ks, data[key]["share"], color=colour, ls=ls, lw=lw)

    for a in (ax, bx):
        a.axvline(K, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
        _spines(a, grid_axis="y")
    ax.text(K + 0.4, 1.0, f"k = {K}", transform=ax.get_xaxis_transform(), ha="left",
            va="top", color=MUTED, fontsize=8)
    generic = data["a"]["d_prior_median"]
    # Rv14: a thin solid line, labelled right of k = 24 where no curve runs at that height
    ax.axhline(generic, color="#999999", lw=0.5, ls="-", zorder=0)
    ax.text(K + 0.8, generic + 0.04, "generic (median)", ha="left", va="bottom",
            color=MUTED, fontsize=8)
    bx.axhline(0.9, color=CONTEXT, lw=0.7, ls=(0, (1, 1.5)), zorder=0)
    bx.text(1.4, 0.905, "90 %", ha="left", va="bottom", color=MUTED, fontsize=8)

    ax.set_xlim(1, KMAX)
    ax.set_xticks([1, 12, 24, 40])
    ax.tick_params(labelbottom=False)
    ylo = min(float(np.min(data[k]["q25"])) for k in SERIES_ORDER)
    yhi = max(float(np.max(data[k]["q75"])) for k in SERIES_ORDER)
    ax.set_ylim(np.floor(ylo * 2) / 2, np.ceil(yhi * 2) / 2)
    ax.set_ylabel("error of identified\nassistance (Nm)")
    bx.set_ylim(0, 1)
    bx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    bx.set_ylabel("share ≥ 1 JND better\nthan generic")
    bx.set_xlabel("comparisons k")
    span_a = ax.get_ylim()[1] - ax.get_ylim()[0]
    _end_labels(ax, data, "median", gap=0.12 * span_a)
    _end_labels(bx, data, "share", gap=0.13)
    _panel(ax, "a", x=-0.12)
    _panel(bx, "b", x=-0.12)
    return _save(fig, figdir, "fig_learning", preview)


# ---------------------------------------------------------------- figure 3

ROWS = (("main", "main, s = 1.0"), ("s05", "s = 0.5"), ("s2", "s = 2.0"),
        ("het", "heterogeneous s"), ("weber", "Weber"), ("asym", "asymmetry 2"),
        ("famil", "familiarity 0.8τ$^{*}$"), ("noise", "device noise 0.6 Nm"),
        ("lapse2", "lapse 0.2"), ("lapse3", "lapse 0.3"), ("landmark", "landmark, static"),
        ("landmark25", "landmark, walking"), ("drift1", "magn. drift 1 JND"),
        ("drift2", "magn. drift 2 JND"))
#: Cells scored in the simulated user's own metric (REPORT.md section 8, own_generic.md). The Weber
#: simulated user's own distance is the RMS distance, so its two readings coincide; it stays in RMS.
OWN_CELLS = ("asym", "famil", "landmark", "landmark25")
OWN_NAMES = {"landmark": "landmark observer, static JND (10 %)",
             "landmark25": "landmark observer, walking JND (25 %)",
             "asym": "asymmetry 2", "famil": "familiarity 0.25"}
GROUP_BREAKS = (0, 3, 6, 9, 11)   # a rule below these rows


def best_rms(ctx, cid: str) -> dict:
    """A minus the best ProMP comparator at k = 24 in the RMS error, chosen as section 3 of
    phase5e_report chooses it (``_rank_family``): on the users common to every optimizer of the
    cell, the comparator with the lowest inverted_cdf median (ties to the scope order), with
    paired_bootstrap's Δ-median interval."""
    scope = r5.SCOPES[r5.BY_ID[cid].scope]["promp_s0"]
    present = [lr for lr in scope if ctx.df(cid, "promp_s0", lr) is not None]
    if "a" not in present or len(present) < 2:
        raise SystemExit(f"{cid}: cell incomplete")
    aligned, dropped = r5.align([ctx.df(cid, "promp_s0", lr) for lr in present])
    err = {lr: sb.delivered_at(d, K).to_numpy() for lr, d in zip(present, aligned)}
    meds = {lr: float(r5._med(v)) for lr, v in err.items()}
    others = [lr for lr in present if lr != "a"]
    best = min(others, key=lambda lr: (meds[lr], scope.index(lr)))
    strata = aligned[0]["fold"].to_numpy()
    diff = paired_bootstrap(err["a"], err[best], strata, n_boot=ctx.n_boot)["diff_median"]
    return dict(best=best, diff=diff, n=len(aligned[0]), dropped=dropped, med_a=meds["a"],
                med_b=meds[best], metric="RMS")


def best_own(ctx, cid: str) -> dict:
    """The same contrast in the simulated user's own distance, as section 8 takes it."""
    best = r5._best_own_comparator(ctx, cid)
    c = ctx.cell_contrast(cid, "promp_s0", "a", best, field=r5.OWN)
    return dict(best=best, diff=c["diff"], n=c["n"], dropped=c["dropped"], med_a=c["med_a"],
                med_b=c["med_b"], metric="own")


def robustness_data(ctx, dir5e: Path) -> list[dict]:
    print("  own-metric gains over generic assistance (phase5e_own_generic.collect) ...")
    own_cells = og.collect(dir5e, K)
    strength = _perturbation(ctx)
    rows = []
    for cid, label in ROWS:
        cond = r5.BY_ID[cid]
        own = cid in OWN_CELLS
        left = best_own(ctx, cid) if own else best_rms(ctx, cid)
        rms_reading = best_rms(ctx, cid) if own else None
        decile, decile_src = None, None
        if not cid.startswith("drift"):
            if own:
                pts = own_cells[(OWN_NAMES[cid], "ProMP", "A")]
                g_own = np.array([p["own_generic"] for p in pts])
                own_k = np.array([p["own"] for p in pts])
                decile = bm.lower_decile((g_own - own_k) / og.JND, [p["fold"] for p in pts],
                                         ctx.n_boot)
                decile["n"] = len(pts)
                decile_src = "own"
            else:
                df = ctx.df(cid, "promp_s0", "a")
                rel = bm.reliability(df, K, ctx.n_boot, jnd=cond.jnd)
                decile = dict(rel["decile"], n=rel["n"])
                decile_src = "rms"
        rows.append(dict(cid=cid, label=label, title=cond.title, jnd=cond.jnd, own=own,
                         left=left, rms_reading=rms_reading, decile=decile,
                         decile_src=decile_src, dp=strength[cid]["mean_abs_dp"]))
        print(f"    {cid:10s} A − {left['best']:5s} {sb._ci(left['diff'])}"
              + (f"  decile {decile['estimate']:+.2f}" if decile else ""))
    return rows


def _perturbation(ctx) -> dict:
    """The cached perturbation strength, read only: the cache must carry the parameters
    phase5e_report would compute it with (it is never recomputed or written here)."""
    path = ctx.perturbation_cache
    cached = json.loads(path.read_text(encoding="utf-8"))
    if cached.get("params") != r5.perturbation_params(ctx.run_dir):
        raise SystemExit(f"{path}: cached parameters do not match phase5e_report's")
    return cached["values"]


def fig_robustness(rows: list[dict], figdir: Path, preview) -> Path:
    W, H = FIG_W, 3.05      # v2 (page budget): 3.5 -> 3.05 in, rows 11.3 pt apart, 8-pt text
    top_in, bot_in = 0.36, 0.50
    lab_x, l_x0, l_w, r_x0, r_w, dp_x = 1.07, 1.12, 1.00, 2.29, 0.86, 3.35
    fig = plt.figure(figsize=(W, H))
    y0, h = bot_in / H, (H - top_in - bot_in) / H
    axL = fig.add_axes([l_x0 / W, y0, l_w / W, h])
    axR = fig.add_axes([r_x0 / W, y0, r_w / W, h])
    n = len(rows)
    y = np.arange(n)[::-1].astype(float)          # first row on top
    for a in (axL, axR):
        a.set_ylim(-0.6, n - 0.4)
        a.set_yticks([])
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        for b in GROUP_BREAKS:
            a.axhline(y[b] - 0.5, color="#E6E6E6", lw=0.5, zorder=0)
    axL.axvline(0, color=MUTED, lw=0.6, zorder=1)
    # (b) has no value in the drift rows, so its zero line stops above them
    defined = [yi for yi, r in zip(y, rows) if not r["cid"].startswith("drift")]
    axR.vlines(0, min(defined) - 0.5, n - 0.4, color=MUTED, lw=0.6, zorder=1)

    shapes = {"g": "o", "cmaes": "s", "d_z": "^"}
    for yi, r in zip(y, rows):
        d = r["left"]["diff"]
        mk = shapes[r["left"]["best"]]
        face = "white" if r["own"] else BLUE
        axL.plot([d["lo"], d["hi"]], [yi, yi], color=BLUE, lw=1.0, solid_capstyle="butt",
                 zorder=2)
        axL.plot([d["estimate"]], [yi], marker=mk, ms=3.6, mfc=face, mec=BLUE, mew=0.9,
                 zorder=3, ls="none")
        if r["decile"] is None:
            axR.text(0.65, yi, "n/a, ideal moves", ha="center", va="center", color="#8C8C8C",
                     fontsize=8)
        else:
            dd = r["decile"]
            axR.plot([dd["lo"], dd["hi"]], [yi, yi], color=BLUE, lw=1.0, solid_capstyle="butt",
                     zorder=2)
            axR.plot([dd["estimate"]], [yi], marker="D", ms=3.0, mfc=face, mec=BLUE, mew=0.9,
                     zorder=3, ls="none")
        label = r["label"] + ("†" if r["own"] else "")
        fig.text(lab_x / W, y0 + h * (yi + 0.6) / n, label, ha="right", va="center",
                 color=INK, fontsize=8, fontweight="bold" if r["cid"] == "main" else "normal")
        fig.text(dp_x / W, y0 + h * (yi + 0.6) / n, f"{r['dp']:.2f}", ha="center",
                 va="center", color=INK, fontsize=8)

    axL.set_xlim(-1.3, 0.12)
    axL.set_xticks([-1.0, -0.5, 0.0])
    axL.set_xticklabels(["−1", "−0.5", "0"])
    axR.set_xlim(-1.0, 2.3)
    axR.set_xticks([-1, 0, 1, 2])
    axL.set_xlabel("Δ median (Nm)\n< 0: A better", labelpad=1.5)
    axR.set_xlabel("gain (JND)", labelpad=1.5)
    top = y0 + h
    head_y = top + 0.03 / H
    fig.text((l_x0 + l_w / 2) / W, head_y, "(a) A − best\ncomparator", ha="center",
             va="bottom", fontsize=8, fontweight="bold", color=INK, linespacing=1.0)
    fig.text((r_x0 + r_w / 2) / W, head_y, "(b) A, lower\ndecile of gain", ha="center",
             va="bottom", fontsize=8, fontweight="bold", color=INK, linespacing=1.0)
    fig.text(dp_x / W, head_y, "|Δp|", ha="center", va="bottom", fontsize=8,
             fontweight="bold", color=INK)
    handles = [Line2D([], [], marker="o", ls="none", ms=3.6, mfc=BLUE, mec=BLUE),
               Line2D([], [], marker="s", ls="none", ms=3.6, mfc=BLUE, mec=BLUE),
               Line2D([], [], marker="o", ls="none", ms=3.6, mfc="white", mec=BLUE, mew=0.9)]
    fig.legend(handles, ["(a) vs G", "(a) vs SortCMA", "† own metric"],
               loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=False, handletextpad=0.2,
               labelspacing=0.12, borderaxespad=0.15, fontsize=8, handlelength=1.0)
    return _save(fig, figdir, "fig_robustness", preview)


# ---------------------------------------------------------------- figure 4

#: Gaussian kernel of the conditional share (Nm of room). The share is close to a step in the
#: room, so the bandwidth stays narrow.
ROOM_H = 0.25
ROOM_X = (0.5, 3.5)
ROOM_GRID = np.linspace(ROOM_X[0], ROOM_X[1], 301)
#: A curve is drawn only where its kernel weights amount to at least this many users.
ROOM_MIN_NEFF = 8.0
ROOM_BOOT = 2000
ROOM_SEED = 20261008
#: Bins of the cross-check table (the grounding's ROOM_BANDS, extended).
ROOM_BINS = (0.0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, np.inf)
#: (family, optimizer) as literature_room.md and optimizers_real_room.md name them.
GROUNDING_NAME = {"a": ("ProMP", "A"), "g": ("ProMP", "preference GP"),
                  "d_z": ("ProMP", "LineCoSpar"), "cmaes": ("ProMP", "SortCMA"),
                  "a_lens": ("Franks", "lens")}
LIT_HEADINGS = {"anchor": "Anchor: Schäfer timing-only room, unrescaled (reproduces §6)",
                "central": "Central literature room, 1.74 Nm",
                "high": "High literature room, 2.07 Nm"}
LIT_SUMMARY = "Summary (text for spec §7.1)"


def _nw(x_grid, room, flags, h: float = ROOM_H):
    """Nadaraya-Watson share of ``flags`` at each grid point (Gaussian kernel) and the kernel's
    effective number of users there."""
    w = np.exp(-0.5 * ((x_grid[:, None] - room[None, :]) / h) ** 2)
    sw = w.sum(axis=1)
    return (w @ flags.astype(float)) / sw, sw ** 2 / (w ** 2).sum(axis=1)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - half), float(c + half)


def room_data(ctx) -> dict:
    """Per user of the main cell: the ProMP room (A's d_prior - d_floor; the same user's room
    positions the lens), and for each optimizer Q≥1 at k = 24 ((d_prior - d_24) / JND ≥ 1 on
    its own family) and 'worse than generic' (d_24 > d_prior), the definitions of
    schaefer_grounding_optimizers.py."""
    dfs = [ctx.df("main", SERIES[k][0], SERIES[k][1]) for k in SERIES_ORDER]
    aligned, dropped = r5.align(dfs)
    if dropped or len(aligned[0]) != r5.EXPECTED_N:
        raise SystemExit(f"fig_room: {len(aligned[0])} common users, {dropped} dropped")
    a = aligned[0]
    room = (a["d_prior"].astype(float) - a["d_floor"].astype(float)).to_numpy()
    folds = a["fold"].to_numpy()
    out = dict(room=room, folds=folds, n=len(a), series={})
    for key, d in zip(SERIES_ORDER, aligned):
        dp = d["d_prior"].astype(float).to_numpy()
        dk = sb.delivered_at(d, K).to_numpy()
        q1 = (dp - dk) / JND >= 1.0 - TOL
        worse = dk > dp
        curve, neff = _nw(ROOM_GRID, room, q1)
        out["series"][key] = dict(q1=q1, worse=worse, curve=curve, neff=neff)
    # 95 % percentile band for A and the lens: users resampled within folds
    rng = np.random.default_rng(ROOM_SEED)
    groups = [np.flatnonzero(folds == f) for f in pd.unique(folds)]
    draws = np.concatenate([g[rng.integers(0, len(g), size=(ROOM_BOOT, len(g)))]
                            for g in groups], axis=1)
    for key in ("a", "a_lens"):
        q1 = out["series"][key]["q1"]
        boot = np.array([_nw(ROOM_GRID, room[ix], q1[ix])[0] for ix in draws])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        out["series"][key].update(lo=lo, hi=hi)
    return out


def grounding_numbers(out_dir: Path) -> dict:
    """The marked rooms and the KDE-reweighted shares the caption quotes, read from
    out/schaefer_grounding/literature_room.md (and optimizers_real_room.md for the anchor)."""
    lit = out_dir / "schaefer_grounding" / "literature_room.md"
    opt = out_dir / "schaefer_grounding" / "optimizers_real_room.md"
    rows = parse_rows(lit)
    summary = {c[0].split(" (")[0].split(":")[0]: c for h, c in rows if h == LIT_SUMMARY}
    rooms = dict(anchor=nums(summary["Schäfer timing-only"][2])[0],
                 central=nums(summary["central"][2]),
                 high=nums(summary["high"][2])[0])
    shares = {}
    for level, heading in LIT_HEADINGS.items():
        for h, c in rows:
            if h == heading and len(c) == 6 and c[0] in ("ProMP", "Franks"):
                shares[(level, c[0], c[1])] = dict(q1_unw=nums(c[2])[0], q1=nums(c[3]),
                                                   worse_unw=nums(c[4])[0], worse=nums(c[5]))
    sens = {c[0]: c for h, c in rows if h == "Sensitivity rooms (A and lens only)"}
    mm = sens["central, median-matched (Monte Carlo median)"]
    opt_text = opt.read_text(encoding="utf-8")
    m = re.search(r"real matched room median ([\d.]+)", opt_text)
    opt_rows = {(c[0], c[1]): c for _, c in parse_rows(opt) if c[0] in ("ProMP", "Franks")}
    return dict(rooms=rooms, shares=shares, median_matched=(nums(mm[1])[0], nums(mm[3])),
                anchor_median_opt=float(m.group(1)) if m else None, opt_rows=opt_rows,
                lit_path=lit, opt_path=opt)


def fig_room(rd: dict, gr: dict, sim_median: float, figdir: Path, preview) -> Path:
    W, H = FIG_W, 1.85
    left, bottom, width, height = 0.60, 0.36, 2.00, 1.13
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, width / W, height / H])
    x = ROOM_GRID
    rooms = gr["rooms"]
    c_lo, c_mid, c_hi = rooms["central"][1], rooms["central"][0], rooms["central"][2]
    per_in = (ROOM_X[1] - ROOM_X[0]) / width          # Nm of room per inch

    # where no ProMP optimizer can be one JND better: the gain cannot exceed the room
    ax.axvspan(ROOM_X[0], JND, facecolor="none", edgecolor="#D4D4D4", hatch="/////", lw=0,
               zorder=0)
    ax.axvspan(c_lo, c_hi, color="#EBEBEB", lw=0, zorder=0)
    marks = [(rooms["anchor"], (0, (1, 1.2)), "timing"),
             (c_mid, (0, (3, 1.5)), "central"),
             (rooms["high"], (0, (1, 1.2)), "high"),
             (sim_median, (0, (4, 1.2, 1, 1.2)), "sim.")]
    for xm, ls, _ in marks:
        ax.axvline(xm, color=MUTED, lw=0.6, ls=ls, zorder=1)

    for key in reversed(SERIES_ORDER):
        _, _, _, colour, ls, lw = SERIES[key]
        s = rd["series"][key]
        # from one JND on: below it no simulated user is one JND better (binned share 0 of 7),
        # and a kernel curve there would only carry the step's smoothing into the strip
        ok = (s["neff"] >= ROOM_MIN_NEFF) & (x >= JND)
        if "lo" in s:
            ax.fill_between(x[ok], s["lo"][ok], s["hi"][ok], color=colour,
                            alpha=0.15 if key == "a" else 0.12, lw=0, zorder=2)
        ax.plot(x[ok], s["curve"][ok], color=colour, ls=ls, lw=lw, zorder=3)

    # the text's numbers: A's KDE-reweighted share at the anchor, central and high rooms
    # (filled, 95 % CI), and the unweighted share of all 200 users at the simulated room (open)
    a_col = SERIES["a"][3]
    for level, xm in (("anchor", rooms["anchor"]), ("central", c_mid), ("high", rooms["high"])):
        p, lo, hi = gr["shares"][(level, "ProMP", "A")]["q1"][:3]
        ax.errorbar([xm], [p], yerr=[[p - lo], [hi - p]], fmt="D", ms=3.6, mfc=a_col,
                    mec="white", mew=0.5, ecolor=a_col, elinewidth=0.8, capsize=1.8,
                    capthick=0.8, zorder=6)
    ax.plot([sim_median], [gr["shares"][("anchor", "ProMP", "A")]["q1_unw"]], marker="D",
            ms=3.6, mfc="white", mec=a_col, mew=0.9, ls="none", zorder=6)

    ax.set_xlim(*ROOM_X)
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 3])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("room for personalization, ProMP (Nm)", labelpad=1.5)
    ax.set_ylabel("share ≥ 1 JND better\nthan generic", labelpad=2)
    _spines(ax, grid_axis="y")

    # mark labels above the axes, spread apart with leader lines
    xs = np.array([m[0] for m in marks])
    gap = 0.43 * per_in
    pos = spread(xs, gap)
    pos = pos + max(0.0, ROOM_X[0] + 0.25 * gap - pos.min())
    trans = ax.get_xaxis_transform()
    for (xm, _, label), xp in zip(marks, pos):
        ax.plot([xm, xm, xp], [1.0, 1.03, 1.07], color=CONTEXT, lw=0.5, transform=trans,
                clip_on=False)
        ax.text(xp, 1.08, f"{label}\n{xm:.2f}", transform=trans, ha="center", va="bottom",
                color=INK, fontsize=8, linespacing=1.0, clip_on=False)

    # direct labels at the right end
    ends = {k: float(rd["series"][k]["curve"][-1]) for k in SERIES_ORDER}
    ys = spread([ends[k] for k in SERIES_ORDER], 0.125)
    x_end = ROOM_X[1]
    for key, y_lab in zip(SERIES_ORDER, ys):
        _, _, label, colour, ls, lw = SERIES[key]
        ax.plot([x_end + 0.02 * per_in, x_end + 0.06 * per_in], [ends[key], y_lab],
                color=CONTEXT, lw=0.5, clip_on=False)
        ax.plot([x_end + 0.06 * per_in, x_end + 0.20 * per_in], [y_lab, y_lab], color=colour,
                ls=ls, lw=lw, clip_on=False, solid_capstyle="butt")
        ax.text(x_end + 0.24 * per_in, y_lab, label, ha="left", va="center", color=INK,
                fontsize=8, clip_on=False)
    # key for the markers, below the direct labels
    y_key = min(ys) - 0.24
    ax.plot([x_end + 0.13 * per_in], [y_key], marker="D", ms=3.6, mfc=a_col, mec="white",
            mew=0.5, ls="none", clip_on=False)
    ax.text(x_end + 0.24 * per_in, y_key, f"{DISPLAY['a']} over a\nroom spread", ha="left",
            va="center",
            color=INK, fontsize=8, linespacing=1.0, clip_on=False)
    return _save(fig, figdir, "fig_room", preview)


# ---------------------------------------------------------------- figure 5: the axes

#: One measure in every panel: the preference over generic assistance after AXES_K comparisons
#: (the reports' "win rate"), PsyPref and the strongest comparator run in that condition, with
#: the attainable best-profile level (the preference the best profile within the parameterization would
#: reach) and 50 % (generic assistance).
AXES_K = 24
#: The realistic room of panels (a) and (b), read from best-profile level.md (jnd_free's room_setup);
#: panel (c) is as simulated, because the other user models exist only there.
AXES_ROOM = "median"
AXES_ROOM_TITLE = {"median": "median-matched (1.47 Nm)", "central": "central (1.74 Nm)",
                   "simulated": "as simulated"}
#: The reports' optimizer names (data keys, not display) and the code's keys.
AXES_REPORT_NAME = {"A": "a", "A (PsyPref)": "a", "G": "g", "G (Arens' GP)": "g",
                    "preference GP (G)": "g", "LineCoSpar": "d_z", "SortCMA": "cmaes",
                    "SortCMA (CMA-ES)": "cmaes", "GP-BALD": "gp_bald", "GP-EUBO": "gp_eubo",
                    "GP-EUBO (GP-qEUBO)": "gp_eubo"}
AXES_SHAPE = {"gp_bald": "D", "gp_eubo": "P", "g": "v", "d_z": "^", "cmaes": "s"}
AXES_GREY = "#7A7A7A"
AXES_YLIM = (0.38, 0.97)
#: Panels (a) and (b): best-profile level.md section and row label per point (x label, row label).
AXES_POP = {
    "a": ("2", [("0.5", "s = 0.5 (reference rows)"), ("1.0", "main (s = 1.0, λ = 0.1)"),
                ("2.0", "s = 2.0"), ("indiv.", "individual s")]),
    "b": ("2", [("0.1", "main (s = 1.0, λ = 0.1)"), ("0.2", "λ = 0.2"), ("0.3", "λ = 0.3"),
                ("noise", "device noise 0.6 Nm")]),
}
#: jnd_free.md section 2 heading of each best-profile level.md condition (the cross-check).
AXES_JF_HEAD = {"s = 0.5 (reference rows)": "s = 0.5 reference",
                "main (s = 1.0, λ = 0.1)": "main (s = 1.0)", "s = 2.0": "s = 2.0",
                "individual s": "heterogeneous s (population s = 1.0)", "λ = 0.2": "lapse 0.2",
                "λ = 0.3": "lapse 0.3", "device noise 0.6 Nm": "device noise 0.6 Nm"}
#: Panel (c) tick labels (the caption spells the user models out).
AXES_C_TICK = {"main": "main", "feat_static": "feature, static",
               "feat_walking": "feature, walking", "rand": "random utility",
               "ood": "other generator", "wph_pref": "phase, individual",
               "wph_bio": "phase, hip speed"}


def md_tables(path: Path) -> list[dict]:
    """Every markdown table of a report: the heading above it, the bold label line between
    that heading and the table (robust_report.md names its cells so), its header cells and
    its rows."""
    out, heading, label, cur = [], None, None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("#"):
            heading, label, cur = s.lstrip("#").strip(), None, None
            continue
        if s.startswith("**") and s.endswith("**") and len(s) > 4:
            label, cur = s.strip("*").strip().rstrip("."), None
            continue
        if not s.startswith("|"):
            cur = None
            continue
        if s.startswith("|---"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cur is None:
            cur = dict(heading=heading, label=label, header=cells, rows=[])
            out.append(cur)
        else:
            cur["rows"].append(cells)
    return out


def _md_table(tables: list[dict], path: Path, heading: str, first: str,
              label: str | None = None) -> dict:
    for t in tables:
        if t["heading"] == heading and t["header"][0] == first and (
                label is None or t["label"] == label):
            return t
    raise SystemExit(f"fig_axes: no table '{heading}'" + (f" / '{label}'" if label else "")
                     + f" in {path}")


def _rate(cell: str) -> dict:
    """'0.815 [0.796, 0.834]': a preference and its interval (the first three numbers)."""
    v = nums(cell)
    if len(v) < 3:
        raise SystemExit(f"fig_axes: cannot read a preference from {cell!r}")
    return dict(est=v[0], lo=v[1], hi=v[2], text=cell)


def _named_rate(cell: str) -> tuple[str, dict]:
    """'GP-BALD 0.650 [0.560, 0.706]' (best-profile level.md's best-comparator cell): key and value."""
    for name in sorted(AXES_REPORT_NAME, key=len, reverse=True):
        if cell.startswith(name + " "):
            return AXES_REPORT_NAME[name], _rate(cell[len(name):].strip())
    raise SystemExit(f"fig_axes: cannot read a comparator from {cell!r}")


def _best_of(rows: dict, col: int, keys=("G", "LineCoSpar", "SortCMA")) -> tuple[str, dict]:
    """The highest preference among a report table's comparator rows."""
    vals = {AXES_REPORT_NAME[n]: _rate(rows[n][col]) for n in keys if n in rows}
    best = max(vals, key=lambda c: vals[c]["est"])
    return best, vals[best]


def _first_table(tables: list[dict], heading_start: str, first: str,
                 label: str | None = None) -> dict | None:
    for t in tables:
        if (t["heading"] or "").startswith(heading_start) and t["header"][0] == first and (
                label is None or t["label"] == label):
            return t
    return None


def axes_data(out_dir: Path) -> dict:
    """The numbers of fig_axes, read from the ICORR reports (no row is loaded): (a) and (b)
    from best-profile level.md at AXES_ROOM, each cross-checked against jnd_free.md section 2 at the rooms
    both give; (c) as simulated from the planned reports (gpeubo_report, robust_report,
    wph_report) with the best-profile level from best-profile level.md section 4. The basis count K (panel (d) until
    2026-10-22) is fig_families' now."""
    icorr = out_dir / "icorr"
    paths = dict(ceil=icorr / "ceiling.md", jnd=icorr / "jnd_free.md",
                 gpe=icorr / "gpeubo_report.md", rob=icorr / "robust_report.md",
                 wph=icorr / "wph_report.md")
    tabs = {key: md_tables(p) for key, p in paths.items()}
    problems: list[str] = []
    room_title = ce_room = {"median": "median-matched (1.47 Nm)", "central": "central (1.74 Nm)",
                            "simulated": "as simulated"}
    t2 = _first_table(tabs["ceil"], "2. Robustness", "condition")
    t2c = [t for t in tabs["ceil"] if (t["heading"] or "").startswith("2. Robustness")
           and t["header"][0] == "condition" and t["header"][2] != "ceiling C"][0]
    t4 = _first_table(tabs["ceil"], "4. Other user models", "user model")
    if t2 is None or t4 is None:
        raise SystemExit("fig_axes: ceiling.md lacks section 2 or 4 (run icorr_ceiling.py)")

    def ceil_row(t, first: str, room: str, col0: int = 0):
        for r in t["rows"]:
            if r[col0] == first and (room is None or r[1] == ce_room[room]):
                return r
        raise SystemExit(f"fig_axes: ceiling.md has no row '{first}' at {room}")

    def jf_check(cond: str, room: str, key: str, value: float) -> str:
        """best-profile level.md's value against jnd_free.md section 2 at two decimals."""
        col = {"simulated": "simulated (as stored)",
               "central": "central literature room, 1.74 Nm"}[room]
        t = _first_table(tabs["jnd"], AXES_JF_HEAD[cond], "family")
        name = {"a": "A", "g": "preference GP (G)", "d_z": "LineCoSpar", "cmaes": "SortCMA"}[key]
        row = next(r for r in t["rows"] if r[0] == "ProMP" and r[1] == name
                   and r[2] == str(AXES_K))
        want = nums(row[t["header"].index(col)])[0]
        # best-profile level.md prints three decimals, jnd_free.md two: agree within half the last digit
        return "ok" if abs(value - want) <= 0.0051 else f"DIFFERS (jnd_free {want:.2f})"

    panels: dict = {}
    for key, (_, pts) in AXES_POP.items():
        out = []
        for xlab, cond in pts:
            r = ceil_row(t2, cond, AXES_ROOM)
            best, comp = _named_rate(r[5])
            p = dict(xlab=xlab, a=_rate(r[3]), best=best, comp=comp, ceiling=_rate(r[2]),
                     src=f"ceiling.md §2, '{cond}', {room_title[AXES_ROOM]}", checks=[])
            for room in ("simulated", "central"):
                rr = ceil_row(t2, cond, room)
                st = jf_check(cond, room, "a", _rate(rr[3])["est"])
                p["checks"].append(f"A {room} {_rate(rr[3])['est']:.3f}: {st}")
                rb = ceil_row(t2c, cond, room)
                for k_, n_ in ((1, "g"), (2, "d_z"), (3, "cmaes")):
                    v = _rate(rb[1 + k_])["est"]
                    st = jf_check(cond, room, n_, v)
                    if st != "ok":
                        p["checks"].append(f"{n_} {room} {v:.3f}: {st}")
            if any("DIFFERS" in c for c in p["checks"]):
                problems.append(f"fig_axes ({key}) {xlab}: {p['checks']}")
            out.append(p)
        panels[key] = out
    # (c) user models, as simulated: the planned reports, the best-profile level from best-profile level.md 4
    ceil4 = {r[0]: r for r in t4["rows"]}
    ceil_main = ceil_row(t2, "main (s = 1.0, λ = 0.1)", "simulated")

    def ceil_of(prefix: str):
        row = next((v for k_, v in ceil4.items() if k_.startswith(prefix)), None)
        return None if row is None else _rate(row[1])

    pc = []
    t = _first_table(tabs["gpe"], "2.1 Win rate against generic assistance", "optimizer")
    rows = {x[0]: x for x in t["rows"]}
    best, comp = _best_of(rows, 2, keys=("G", "LineCoSpar", "SortCMA", "GP-EUBO", "GP-BALD"))
    pc.append(dict(xlab="main", a=_rate(rows["A"][2]), best=best, comp=comp,
                   ceiling=_rate(ceil_main[2]), src="gpeubo_report.md §2.1, k = 24", checks=[]))
    for lab, cell in (("feat_static", "static landmark (10 %), matched"),
                      ("feat_walking", "walking landmark (25 %), matched")):
        t = _first_table(tabs["rob"], "2.3 Context", "optimizer", label=cell)
        rows = {x[0]: x for x in t["rows"]}
        best, comp = _best_of(rows, 2)
        pc.append(dict(xlab=lab, a=_rate(rows["A"][2]), best=best, comp=comp, ceiling=None,
                       src=f"robust_report.md §2.3, {cell}, k = 24 (no own-metric floor: no "
                           "ceiling)", checks=[]))
    t = _first_table(tabs["rob"], "4.1 PR1 and PR2", "optimizer")
    rows = {x[0]: x for x in t["rows"]}
    best, comp = _best_of(rows, 2)
    pc.append(dict(xlab="rand", a=_rate(rows["A"][2]), best=best, comp=comp,
                   ceiling=ceil_of("random utility"),
                   src="robust_report.md §4.1, k = 24; ceiling.md §4 (a lower bound)", checks=[]))
    t = _first_table(tabs["rob"], "3.4 Context", "optimizer",
                     label="per-bout peaks (e-ood-pb, primary)")
    rows = {x[0]: x for x in t["rows"]}
    best, comp = _best_of(rows, 2)
    pc.append(dict(xlab="ood", a=_rate(rows["A"][2]), best=best, comp=comp,
                   ceiling=ceil_of("other generator"),
                   src="robust_report.md §3.4, per-bout peaks, k = 24; ceiling.md §4", checks=[]))
    for lab, head, prefix in (("wph_pref", "2.1 Win rate against generic", "phase-weighted, ind"),
                              ("wph_bio", "3.1 Win rate against generic", "phase-weighted, hip")):
        t = _first_table(tabs["wph"], head, "optimizer")
        rows = {x[0]: x for x in t["rows"]}
        best, comp = _best_of(rows, 2)
        pc.append(dict(xlab=lab, a=_rate(rows["A"][2]), best=best, comp=comp,
                       ceiling=ceil_of(prefix), src=f"wph_report.md §{head[:3]}, k = 24; "
                       "ceiling.md §4", checks=[]))
    # GP-qEUBO also ran under R, O and W (rows completed after gpeubo_report.md was written);
    # best-profile level.md section 4 gives it: check that it is never the strongest there
    t4c = [t for t in tabs["ceil"] if (t["heading"] or "").startswith("4. Other")
           and t["header"][0] == "user model" and t["header"][1] != "ceiling C"][0]
    for r in t4c["rows"]:
        gp = r[4]
        if gp != "—":
            v = _rate(gp)["est"]
            others = [_rate(x)["est"] for x in r[1:4]]
            if v >= max(others):
                problems.append(f"fig_axes (c): GP-qEUBO is the strongest comparator for "
                                f"{r[0]} ({v:.3f}); draw it")
    # checks across panels: the main condition's point is the same wherever its room is
    for k1, k2 in (("a", "b"),):
        if panels[k1][1]["a"]["text"] != panels[k2][0]["a"]["text"]:
            problems.append("fig_axes: the main condition differs between (a) and (b)")
    sim_a = _rate(ceil_main[3])["est"]
    if not _close(sim_a, pc[0]["a"]["est"], 3):
        problems.append(f"fig_axes: main as simulated differs between ceiling.md ({sim_a}) and "
                        f"gpeubo_report.md ({pc[0]['a']['est']})")
    panels["c"] = pc
    return dict(panels=panels, paths=paths, problems=problems)


def _pct_axis(ax, ylabel: str | None):
    """The preference axis, in percent; 50 % (generic assistance) dashed."""
    _spines(ax, grid_axis="y")
    ax.set_ylim(*AXES_YLIM)
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
    ax.set_yticks(ticks)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    if ylabel is None:
        ax.tick_params(labelleft=False)
    else:
        ax.set_yticklabels([f"{100 * t:.0f}" for t in ticks])
        ax.set_ylabel(ylabel, labelpad=2)


def _axes_points(ax, pts: list[dict], dx: float = 0.17, half: float = 0.40):
    """PsyPref (blue) and the strongest comparator (grey, its own marker) at each category,
    estimate and 95 % interval; the attainable best-profile level as a short dashed line over both."""
    for x, p in enumerate(pts):
        if p.get("ceiling") is not None:
            ax.plot([x - half, x + half], [p["ceiling"]["est"]] * 2, color=INK, lw=0.9,
                    ls=(0, (2.4, 1.3)), zorder=2, solid_capstyle="butt")
        for val, colour, shape, off, ms in ((p["comp"], AXES_GREY, AXES_SHAPE[p["best"]], dx,
                                             3.6),
                                            (p["a"], BLUE, "o", -dx, 4.0)):
            ax.plot([x + off] * 2, [val["lo"], val["hi"]], color=colour, lw=1.1,
                    solid_capstyle="butt", zorder=3)
            ax.plot([x + off], [val["est"]], marker=shape, ms=ms, ls="none", mec=colour,
                    mfc=colour, mew=0.6, zorder=4)
    ax.set_xlim(-0.6, len(pts) - 0.4)
    ax.set_xticks(range(len(pts)))
    ax.tick_params(axis="x", length=0)


def fig_axes(ad: dict, figdir: Path, preview) -> Path:
    """One measure everywhere: the preference over generic assistance after AXES_K
    comparisons; PsyPref against the strongest comparator run in each condition, with the
    attainable best-profile level and 50 %. One row of three panels (flatter than the 2 x 2 layout of
    2026-10-21): (a) sensitivity, (b) lapse rate and device noise, (c) user models; the
    width of each panel follows its number of categories."""
    W, H = FIG_W, 1.86
    fig = plt.figure(figsize=(W, H))
    P = ad["panels"]
    left, gap, right = 0.47, 0.07, 0.03
    h, y0 = 0.94, 0.66
    ylabel = "preference over\ngeneric (%)"
    # categories of (c) are labelled at an angle and so need less width than those of (a), (b)
    n = {"a": 1.2 * len(P["a"]), "b": 1.2 * len(P["b"]), "c": 0.8 * len(P["c"])}
    span = W - left - 2 * gap - right
    unit = span / sum(n.values())
    x = left
    for key, xlabel in (("a", "s (Nm)"), ("b", "λ; noise"), ("c", None)):
        w = unit * n[key]
        ax = fig.add_axes([x / W, y0 / H, w / W, h / H])
        _pct_axis(ax, ylabel if key == "a" else None)
        _axes_points(ax, P[key], dx=0.19, half=0.42)
        if key == "c":
            ax.set_xticklabels([AXES_C_TICK[p["xlab"]] for p in P[key]], rotation=40,
                               ha="right", rotation_mode="anchor", fontsize=7)
            ax.tick_params(axis="x", pad=2)
        else:
            ax.set_xticklabels([p["xlab"] for p in P[key]], fontsize=7)
            ax.set_xlabel(xlabel, labelpad=1.5)
        _panel(ax, key, x=-0.03)
        x += w + gap
    # legend: PsyPref, the strongest comparators that appear, the best-profile level
    used = []
    for key in "abc":
        for p in P[key]:
            if p["best"] not in used:
                used.append(p["best"])
    order = [c for c in ("gp_bald", "gp_eubo", "g", "cmaes", "d_z") if c in used]
    handles = [Line2D([], [], marker="o", ls="none", ms=4.0, mfc=BLUE, mec=BLUE)]
    labels = [DISPLAY["a"]]
    for c in order:
        handles.append(Line2D([], [], marker=AXES_SHAPE[c], ls="none", ms=3.6, mfc=AXES_GREY,
                              mec=AXES_GREY))
        labels.append(DISPLAY[c])
    handles.append(Line2D([], [], color=INK, lw=0.9, ls=(0, (2.4, 1.3))))
    labels.append("ceiling")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 1.0),
               ncol=len(labels), frameon=False, handletextpad=0.2, columnspacing=0.6,
               fontsize=8, handlelength=1.2, borderaxespad=0.1)
    return _save(fig, figdir, "fig_axes", preview)


def axes_md(ad: dict) -> list[str]:
    """The fig_axes section of figure_numbers.md: every plotted number as its report gives
    it."""
    P = ad["panels"]
    room = AXES_ROOM_TITLE[AXES_ROOM]
    L = ["## fig_axes.pdf", "",
         "One measure in every panel: the preference over generic assistance (the reports' "
         f"\"win rate\") after {AXES_K} comparisons, the mean over users of λ/2 + (1 − λ) "
         "Φ((g − e)/s), estimate [95 % interval]. PsyPref (blue) and the strongest comparator "
         "run in that condition (grey; marker per comparator, name in the table and legend); "
         "dashed: the attainable ceiling (the preference the best profile within the "
         "parameterization would reach; ceiling.md) and 50 % (generic assistance). No number is "
         "computed here: each is read from a report and plotted as printed there ("
         + ", ".join('`out/' + p.relative_to(p.parents[1]).as_posix() + '`'
                     for p in ad['paths'].values()) + "). "
         f"One row of three panels since 2026-10-22; the basis-count panel (d) moved to "
         "fig_families (icorr_proj_report.py). "
         f"(a), (b): the {room} room, ceiling.md §2 (icorr_jnd_free's room_setup "
         "and conventions, 2000 draws); each point's simulated and central values in ceiling.md "
         "are checked against jnd_free.md §2 (A and every comparator) at the two decimals that "
         "report prints. (c): as simulated (the "
         "other user models exist only there), from the planned reports (10,000 "
         "fold-stratified joint draws), the ceiling from ceiling.md §4; feature-judging users "
         "have no floor in their own metric and so no ceiling; the random-utility ceiling is a "
         "lower bound (best-found u_max). Strongest comparator: the highest preference at "
         f"k = {AXES_K} among the comparators run in that cell (GP-BALD in the main condition "
         "only; GP-qEUBO ran in the main condition and under R, O and W, where it is never the "
         "strongest, ceiling.md §4).", "",
         "| panel | point | PsyPref [95 %] | strongest comparator [95 %] | ceiling [95 %] | "
         "source | checks |", "|---|---|---|---|---|---|---|"]
    for key in "abc":
        for p in P[key]:
            cl = p["ceiling"]["text"] if p.get("ceiling") is not None else "— (none)"
            lab = AXES_C_TICK.get(p["xlab"], p["xlab"]).replace("\n", " ") if key == "c" \
                else p["xlab"]
            L.append(f"| ({key}) | {lab} | {p['a']['text']} | {DISPLAY[p['best']]} "
                     f"{p['comp']['text']} | {cl} | {p['src']} | "
                     f"{'; '.join(p['checks']) or '—'} |")
    L.append("")
    return L


def room_md(rd: dict, gr: dict, sim_median: float, ctx, problems: list[str]) -> list[str]:
    """The fig_room section of figure_numbers.md."""
    room, n = rd["room"], rd["n"]
    rooms = gr["rooms"]
    L = ["## fig_room.pdf", "",
         "Post hoc (grounding spec 2026-10-08 sections 6-7; the predictions G1b and P7c were "
         "planned, this figure was not). Main condition (s = 1.0, lapse 0.1), e-main rows, "
         f"{n} population users, k = {K}. x: each simulated user's room for personalization "
         "on ProMP, d_prior − d_floor of A's row; the lens (Franks) is placed at the same "
         "user's ProMP room, as the grounding's reweighting places it. y: share with "
         "(d_prior − d_24) / true_jnd(1.0) ≥ 1 on the optimizer's own family "
         "(`schaefer_grounding_optimizers.py`'s Q≥1). Curves: Nadaraya-Watson kernel "
         f"regression of the 0/1 outcome on the room, Gaussian kernel, bandwidth {ROOM_H:.2f} "
         f"Nm, drawn where the kernel weights amount to at least {ROOM_MIN_NEFF:.0f} users "
         f"(effective n). Bands (A, lens): 95 % percentile interval, {ROOM_BOOT} draws, users "
         f"resampled within folds, seed {ROOM_SEED}. Hatched strip: room below one JND "
         f"({JND:.3f} Nm); a ProMP optimizer cannot be one JND better there, because the gain "
         "cannot exceed d_prior − d_floor. Gray band: the central room's 95 % interval. "
         "Vertical lines: dotted, the Schäfer timing-only room and the high room; dashed, the "
         "central room; dash-dot, the simulated median room. Curves start at one JND "
         f"({JND:.3f} Nm): below it no simulated user is one JND better (binned share 0 of 7), "
         "and the kernel would only smear the step into the hatched strip. Markers (A only): "
         "the KDE-reweighted Q≥1 the text quotes, filled with its 95 % CI at the timing-only, "
         "central and high rooms, open for the unweighted share of all 200 users at the "
         "simulated median room (table 'The curve is not the caption's number', last two "
         "columns). The share worse than generic "
         "assistance is not plotted (left out so that the single panel stays readable at "
         "1.85 in); its values are in the tables below.", ""]

    # ---- the marked rooms
    st_sim = verdict([sim_median], [2.31], 2)
    if not st_sim.startswith("reproduces"):
        problems.append(f"fig_room simulated median room: {st_sim}")
    st_anchor = (verdict([rooms["anchor"]], [gr["anchor_median_opt"]], 2)
                 if gr["anchor_median_opt"] is not None else "NOT FOUND")
    if not st_anchor.startswith("reproduces"):
        problems.append(f"fig_room anchor room: {st_anchor}")
    L += ["### Marked rooms", "",
          "| mark | plotted (Nm) | source | cross-check | status |", "|---|---|---|---|---|",
          f"| Schäfer timing-only room (anchor) | {rooms['anchor']:.2f} | literature_room.md "
          f"summary, '1.26 (a median)' | optimizers_real_room.md: real matched room median "
          f"{gr['anchor_median_opt']} | {st_anchor} |",
          f"| literature-informed central room [95 % CI] | {rooms['central'][0]:.2f} "
          f"[{rooms['central'][1]:.2f}, {rooms['central'][2]:.2f}] | literature_room.md "
          "summary | spec §7.1: 1.74 [1.37, 1.97] | read from the source |",
          f"| high room | {rooms['high']:.2f} | literature_room.md summary | spec §7.1: 2.07 | "
          "read from the source |",
          f"| simulated median room | {sim_median:.3f} | median of d_prior − d_floor over the "
          f"{n} users (computed) | spec §6 G1: 2.31 | {st_sim} |", ""]

    # ---- the unweighted shares reproduce the grounding's
    L += ["### Unweighted shares (the same rows as the grounding)", "",
          "| optimizer | Q≥1, computed | grounding (anchor table) | status | worse than "
          "generic, computed | grounding | status |", "|---|---|---|---|---|---|---|"]
    for key in SERIES_ORDER:
        s = rd["series"][key]
        fam, name = GROUNDING_NAME[key]
        g = gr["shares"].get(("anchor", fam, name))
        o = gr["opt_rows"].get((fam, name))
        rep_q = [g["q1_unw"]] if g else []
        rep_w = [g["worse_unw"]] if g else []
        sq = verdict([float(s["q1"].mean())], rep_q, 2)
        sw = verdict([float(s["worse"].mean())], rep_w, 2)
        if o is not None and g is not None and (nums(o[2])[0] != g["q1_unw"]
                                                or nums(o[4])[0] != g["worse_unw"]):
            sq += "; literature_room.md and optimizers_real_room.md disagree"
        for st, what in ((sq, "Q≥1"), (sw, "worse")):
            if not st.startswith("reproduces"):
                problems.append(f"fig_room {SERIES[key][2]} unweighted {what}: {st}")
        label = SERIES[key][2] if "(" in SERIES[key][2] else f"{SERIES[key][2]} ({fam})"
        L.append(f"| {label} | {s['q1'].mean():.3f} | "
                 f"{g['q1_unw'] if g else '—'} | {sq} | {s['worse'].mean():.3f} | "
                 f"{g['worse_unw'] if g else '—'} | {sw} |")
    # the ProMP comparators share A's room (same d_prior, d_floor per user)
    a_main = r5.align([ctx.df("main", "promp_s0", lr) for lr in PROMP_KEYS])[0]
    same = all(np.allclose(d["d_prior"].astype(float), a_main[0]["d_prior"].astype(float))
               and np.allclose(d["d_floor"].astype(float), a_main[0]["d_floor"].astype(float))
               for d in a_main[1:])
    impossible = {k: int(np.sum(rd["series"][k]["q1"] & (room < JND))) for k in SERIES_ORDER}
    L += ["", f"Every ProMP optimizer's rows carry the same d_prior and d_floor per user as A's: "
          f"{'yes' if same else 'NO'}. Users with room < 1 JND that are Q≥1 anyway: "
          + ", ".join(f"{SERIES[k][2]} {v}" for k, v in impossible.items())
          + f" (of {int(np.sum(room < JND))}; the lens is scored on Franks, whose room differs).",
          ""]
    if not same:
        problems.append("fig_room: ProMP optimizers differ in d_prior/d_floor")

    # ---- conditional curve against the KDE-reweighted shares
    L += ["### The curve is not the caption's number", "",
          "The curve is a conditional share: among simulated users whose own room is x. The "
          "caption's shares at the marked rooms are the grounding's KDE-reweighted shares "
          "(`literature_room.md` §4, `optimizers_real_room.md`): the simulated users are "
          "reweighted so that their rooms spread like Schäfer's amplitude-matched rooms, "
          "rescaled to a median of 1.74 or 2.07 Nm (the anchor unrescaled). Because the share "
          "is close to a step in the room and the real rooms spread widely around their median, "
          "the reweighted share lies above the curve's value below the step and below it "
          "above the step. The figure shows where the step lies; the caption numbers are the "
          "reliability over a spread of rooms like the real one. Do not read the caption "
          "numbers off the curve: A's are drawn as markers at the marked rooms instead.", "",
          "| optimizer | mark (Nm) | curve at the mark [95 % band] | KDE-reweighted Q≥1 "
          "[95 % CI] (caption) | KDE-reweighted worse than generic [95 % CI] |",
          "|---|---|---|---|---|"]
    levels = [("anchor", rooms["anchor"]), ("central", rooms["central"][0]),
              ("high", rooms["high"]), ("sim", sim_median)]
    for key in SERIES_ORDER:
        s = rd["series"][key]
        fam, name = GROUNDING_NAME[key]
        for level, xm in levels:
            v = float(np.interp(xm, ROOM_GRID, s["curve"]))
            band = (f" [{np.interp(xm, ROOM_GRID, s['lo']):.2f}, "
                    f"{np.interp(xm, ROOM_GRID, s['hi']):.2f}]" if "lo" in s else "")
            if level == "sim":
                kq = f"{s['q1'].mean():.2f} (unweighted, all {n} users)"
                kw = f"{s['worse'].mean():.2f} (unweighted)"
            else:
                g = gr["shares"].get((level, fam, name))
                kq = (f"{g['q1'][0]:.2f} [{g['q1'][1]:.2f}, {g['q1'][2]:.2f}]" if g else "—")
                kw = (f"{g['worse'][0]:.2f} [{g['worse'][1]:.2f}, {g['worse'][2]:.2f}]"
                      if g else "—")
            L.append(f"| {SERIES[key][2]} | {level} {xm:.2f} | {v:.2f}{band} | {kq} | {kw} |")
    mm_room, mm_share = gr["median_matched"]
    L += ["", f"Median-matched reading of the central room (literature_room.md §4 sensitivity "
          f"rooms): {mm_room:.2f} Nm, A's Q≥1 {mm_share[0]:.2f} [{mm_share[1]:.2f}, "
          f"{mm_share[2]:.2f}] (not marked in the figure).", ""]

    # ---- binned shares with Wilson intervals (a check on the kernel curve)
    idx = np.digitize(room, ROOM_BINS) - 1
    labels = [f"{lo:g}–{hi:g}" if np.isfinite(hi) else f"≥ {lo:g}"
              for lo, hi in zip(ROOM_BINS[:-1], ROOM_BINS[1:])]
    L += ["### Binned shares (check on the kernel curve; not plotted)", "",
          "Q≥1 share per room bin with its 95 % Wilson interval; n = users in the bin. The "
          "grounding spec §6 (G1b) gives A's shares as 0.00 below 1.0 Nm, at most 0.30 below "
          "1.5 Nm, 0.77 at 1.5–1.75 Nm and 0.97 above.", "",
          "| room bin (Nm) | n | " + " | ".join(SERIES[k][2] for k in SERIES_ORDER) + " |",
          "|---|---|" + "---|" * len(SERIES_ORDER)]
    for b, lab in enumerate(labels):
        sel = idx == b
        nb = int(sel.sum())
        if not nb:
            continue
        cells = []
        for key in SERIES_ORDER:
            kq = int(rd["series"][key]["q1"][sel].sum())
            lo, hi = wilson(kq, nb)
            cells.append(f"{kq / nb:.2f} [{lo:.2f}, {hi:.2f}]")
        L.append(f"| {lab} | {nb} | " + " | ".join(cells) + " |")
    qa = rd["series"]["a"]["q1"]
    spec_check = dict(below_1=(float(qa[room < 1.0].mean()), 0.00),
                      max_1_to_1p5=(max(float(qa[(room >= 1.0) & (room < 1.25)].mean()),
                                        float(qa[(room >= 1.25) & (room < 1.5)].mean())), 0.30),
                      b_1p5_to_1p75=(float(qa[(room >= 1.5) & (room < 1.75)].mean()), 0.77),
                      above_1p75=(float(qa[room >= 1.75].mean()), 0.97))
    L += ["", "A against the spec's description: " + "; ".join(
        f"{k} {v[0]:.2f} (spec {v[1]:.2f}, {verdict([v[0]], [v[1]], 2)})"
        for k, v in spec_check.items()) + ".", ""]
    for k, v in spec_check.items():
        if not verdict([v[0]], [v[1]], 2).startswith("reproduces"):
            problems.append(f"fig_room binned A {k}: {v[0]:.2f} vs spec {v[1]:.2f}")

    # ---- the caption's numbers
    sh = gr["shares"]

    def q(level, fam, name, field="q1"):
        g = sh[(level, fam, name)][field]
        return f"{g[0]:.2f} [{g[1]:.2f}, {g[2]:.2f}]"

    comp_q = [sh[("central", "ProMP", nm)]["q1"][0]
              for nm in ("preference GP", "LineCoSpar", "SortCMA")]
    comp_w = [sh[(lv, f, nm)]["worse"][0] for (lv, f, nm) in sh
              if lv == "central" and nm not in ("A", "lens", "A direct")]
    L += ["### Numbers for the caption (KDE-reweighted; source literature_room.md)", "",
          f"- Schäfer timing-only room {rooms['anchor']:.2f} Nm: A {q('anchor', 'ProMP', 'A')}, "
          f"lens {q('anchor', 'Franks', 'lens')}; A worse than generic "
          f"{q('anchor', 'ProMP', 'A', 'worse')}.",
          f"- Central room {rooms['central'][0]:.2f} [{rooms['central'][1]:.2f}, "
          f"{rooms['central'][2]:.2f}] Nm: A {q('central', 'ProMP', 'A')}, lens "
          f"{q('central', 'Franks', 'lens')}; G, LineCoSpar, SortCMA (ProMP) "
          f"{min(comp_q):.2f}–{max(comp_q):.2f}; the comparators of both families leave "
          f"{min(comp_w):.2f}–{max(comp_w):.2f} worse than generic (A "
          f"{q('central', 'ProMP', 'A', 'worse')}). Median-matched: {mm_share[0]:.2f}.",
          f"- High room {rooms['high']:.2f} Nm: A {q('high', 'ProMP', 'A')}, lens "
          f"{q('high', 'Franks', 'lens')}.",
          f"- Simulated median room {sim_median:.2f} Nm: A "
          f"{rd['series']['a']['q1'].mean():.2f}, lens {rd['series']['a_lens']['q1'].mean():.2f}"
          " (all 200 users, unweighted).", ""]
    return L


PROMP_KEYS = ("a", "g", "d_z", "cmaes")


# ---------------------------------------------------------------- figure 6: the headline

#: The rooms every headline number is computed at (all recorded in figure_numbers.md), and the
#: one room drawn. Two rooms x six optimizers with two bands did not read at one column; the
#: median-matched room (1.47 Nm) is the paper's lead realistic room and the lower of the two.
HEAD_ROOMS = ("simulated", "median", "central")
HEAD_ROOM = "median"
HEAD_ROOM_NAME = {"simulated": "simulated", "median": "median-matched", "central": "central"}
#: icorr_gpeubo_report's draws (its N_BOOT): its section 7 reproduces only with these.
HEAD_BOOT = 10_000
HEAD_NAME = DISPLAY
#: The names gpeubo_report.md and k_sweep.md give the same cells.
HEAD_GPE_NAME = {"a": "A", "gp_bald": "GP-BALD", "gp_eubo": "GP-EUBO", "g": "G",
                 "d_z": "LineCoSpar", "cmaes": "SortCMA"}
HEAD_KS_NAME = {"a": "A", "g": "preference GP (G)", "cmaes": "SortCMA"}
#: Drawing order, back to front.
HEAD_OPTS = ("g", "d_z", "gp_eubo", "cmaes", "gp_bald", "a")
HEAD_COMPARATORS = ("gp_bald", "gp_eubo", "g", "d_z", "cmaes")
HEAD_GP_BLOCKS = {"gp_bald": "e-gpbald", "gp_eubo": "e-gpeubo"}
HEAD_BASES = (4, 8, 12, 16, 24)
#: The blocks of K != 8 (K = 8 is e-main and the two GP blocks), in out/icorr: e-k for K = 4
#: and 12; for K = 16 and 24 e-k-hi, and any GP block the GP chain writes for those K (its name
#: is not fixed yet; GP-BALD is planned at K = 16 only). Rows are picked by n_basis, so a
#: block that also holds other K is harmless; "[conflicted]" sync copies are never read.
HEAD_K_GLOBS = {4: ("e-k.jsonl",), 12: ("e-k.jsonl",),
                16: ("e-k-hi*.jsonl", "*gp*k16*.jsonl", "*gp*k-hi*.jsonl"),
                24: ("e-k-hi*.jsonl", "*gp*k24*.jsonl", "*gp*k-hi*.jsonl")}
#: GP-BALD's budgets whose win rate PsyPref is read against in (a).
HEAD_REACH = (24, 40)
HEAD_FIELDS = rrep.FIELDS + ("error",)
HEAD_LISTS = ("distances",)
#: gpeubo_report.md section 7.2, every row (key, first cell), all reproduced here.
HEAD_REACH_ROWS = (
    ("A to BALD win@24", "A's win rate reaches GP-BALD's at k = 24"),
    ("A to BALD win@40", "A's win rate reaches GP-BALD's at k = 40"),
    ("A to BALD err@40", "A's error reaches GP-BALD's at k = 40"),
    ("BALD to A win@12", "GP-BALD's win rate reaches A's at k = 12"),
    ("BALD to A win@24", "GP-BALD's win rate reaches A's at k = 24"),
    ("BALD to A err@12", "GP-BALD's error reaches A's at k = 12"),
    ("BALD to A err@24", "GP-BALD's error reaches A's at k = 24"),
)
#: GP-BALD: the red of the figure-colour skill's blue-anchored four-series set, not the lens
#: orange of fig_learning and fig_room. It is as dark as BLUE (ΔL* 0.6), so it is also dashed.
BALD_RED = "#BD1F01"
HEAD_STYLE = {  # key: (colour, dashes, width); the four context lines are thin greys
    "a": (BLUE, "-", 1.6),
    "gp_bald": (BALD_RED, (0, (4.0, 1.4)), 1.4),
    "g": ("#404040", (0, (4, 1.6)), 0.8),
    "d_z": ("#7A7A7A", (0, (5, 1.5, 1, 1.5)), 0.8),
    "cmaes": ("#A6A6A6", (0, (1, 1.2)), 1.1),
    "gp_eubo": ("#8C8C8C", "-", 0.7),
}
HEAD_SHAPE = {"a": "o", "gp_bald": "D", "gp_eubo": "v", "g": "o", "d_z": "^", "cmaes": "s"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _head_block(path: Path, notes: list, problems: list) -> Path:
    """The plain block file. A sync "[conflicted]" copy beside it is compared by sha256 and
    never read."""
    for twin in sorted(p for p in path.parent.glob(f"{path.stem} *.jsonl")
                       if "[conflicted" in p.name):
        same = _sha256(twin) == _sha256(path)
        notes.append(f"`{twin.name}` lies beside `{path.name}`: sha256 "
                     f"{'identical' if same else 'DIFFERS'}; the plain file is read.")
        if not same:
            problems.append(f"fig_headline: {twin.name} differs from {path.name} (the plain "
                            "file was read)")
    return path


def _head_pop(r: dict) -> bool:
    """Every row of a population user, error rows too (a cell with one is incomplete)."""
    return r.get("user_kind") == "population" and "/pop" in str(r.get("user_id", ""))


def _head_complete(c: pd.DataFrame) -> str | None:
    """Why a cell in load_sim's order is not a complete main-condition cell, or None."""
    if not ((c["n_comparisons"].astype(int) == KMAX).all()
            and (c["distances"].map(len) >= KMAX).all()):
        return f"not every row ran {KMAX} comparisons"
    if not ((c["s"].astype(float) == 1.0).all() and (c["lapse"].astype(float) == LAPSE).all()):
        return "s or lapse is not 1.0 / 0.1"
    return None


def _head_gp_cell(df, learner: str, keys, a: pd.DataFrame) -> tuple[str, object]:
    """gpeubo_report's R4: ("ok", cell) only if the block holds exactly the 200 users, once
    each, no error row, 40 comparisons, s = 1.0, λ = 0.1, and d_prior and d_floor equal to
    PsyPref's row bit for bit; else (state, reason)."""
    if df is None:
        return "not run", "no rows"
    if "error" in df and df["error"].notna().any():
        return "incomplete", f"{int(df['error'].notna().sum())} error rows"
    try:
        c = rrep.pick(kr.normalise(df), "promp_s0", learner, keys, {})
    except ValueError as exc:
        return "incomplete", str(exc)
    why = _head_complete(c)
    if why:
        return "incomplete", why
    for col in ("d_prior", "d_floor"):
        if not np.array_equal(rrep.col(c, col), rrep.col(a, col)):
            return "incomplete", f"{col} differs from PsyPref's rows"
    return "ok", c


def _head_main_cells(out: Path, keys, notes: list, problems: list) -> tuple[dict, dict]:
    """K = 8: PsyPref and the three comparators of e-main as gpeubo_report selects them
    (``rr._slim``, ``rr.pick``), and the two GP cells."""
    path = _head_block(out / "phase5e" / "e-main.jsonl", notes, problems)
    rows = jf.stream(path, keep=lambda r: (jf._pop_row(r) and r.get("family") == "promp_s0"
                                           and r.get("learner") in PROMP_KEYS),
                     fields=HEAD_FIELDS, lists=HEAD_LISTS)
    if rows is None:
        raise SystemExit("fig_headline: e-main.jsonl is missing")
    rows = rrep._slim(rows)
    cells = {lr: rrep.pick(rows, "promp_s0", lr, keys, {}) for lr in PROMP_KEYS}
    state = {lr: ("ok", "e-main") for lr in PROMP_KEYS}
    for lr, block in HEAD_GP_BLOCKS.items():
        path = out / "icorr" / f"{block}.jsonl"
        if not path.exists():
            state[lr] = ("not run", f"{block}.jsonl absent")
            continue
        _head_block(path, notes, problems)
        df = jf.stream(path, keep=_head_pop, fields=HEAD_FIELDS, lists=HEAD_LISTS)
        st, c = _head_gp_cell(df, lr, keys, cells["a"])
        state[lr] = (st, block if st == "ok" else c)
        if st == "ok":
            cells[lr] = c
    return cells, state


def _head_k_cell(df, learner: str, n_basis: int, keys) -> tuple[str, object]:
    """One main-condition cell at basis count ``n_basis`` (icorr_k_report.cell's selection:
    EIG duels, tight arm, data prior, s = 1.0, phase5e_report's defaults, G at JND scale 1),
    complete only with exactly the 200 users, no error row and 40 comparisons each."""
    if df is None:
        return "absent", "no rows"
    mask = ((df["family"] == "promp_s0") & (df["learner"] == learner)
            & (df["n_basis"] == n_basis) & (df["duel_mode"] == "eig") & (df["arm"] == "tight")
            & (df["prior"] == "data")).to_numpy()
    want = dict(r5.DEFAULTS)
    want.update(s=1.0)
    for name, value in want.items():
        mask &= r5._match(df, name, value)
    if learner == "g":
        mask &= r5._match(df, "jnd_scale", 1.0)
    c = df[mask]
    if not len(c):
        return "absent", "no rows"
    if "error" in c and c["error"].notna().any():
        return "incomplete", f"{int(c['error'].notna().sum())} error rows"
    if c.duplicated(r5.KEY).any():
        return "incomplete", "more than one row for a user"
    have = set(zip(c["fold"], c["user_id"]))
    n = sum(k in have for k in keys)
    if n < len(keys):
        return "incomplete", f"{n} of {len(keys)} population users"
    c = jf.in_order(c, keys)
    why = _head_complete(c)
    return ("incomplete", why) if why else ("ok", c)


def _head_k_cells(out: Path, keys, cells8: dict, state8: dict, notes: list,
                  problems: list, kdir: Path | None = None) -> dict:
    """Per basis count: the complete cells, why the others are missing, and why a K is left
    out (PsyPref's cell incomplete, or its best-profile distances not those of the K's user sidecars).
    ``kdir`` (default out/icorr) holds the K != 8 blocks; only a test points it elsewhere."""
    kdir = kdir or out / "icorr"
    res = {8: dict(cells=dict(cells8), blocks=["e-main.jsonl"]
                   + [f"{HEAD_GP_BLOCKS[lr]}.jsonl" for lr in HEAD_GP_BLOCKS
                      if state8[lr][0] == "ok"],
                   why={lr: f"{s}: {r}" for lr, (s, r) in state8.items() if s != "ok"},
                   left_out=None)}
    frames: dict = {}
    for nb in HEAD_BASES:
        if nb == 8:
            continue
        paths = sorted({p for pat in HEAD_K_GLOBS[nb] for p in kdir.glob(pat)
                        if "[conflicted" not in p.name})
        dfs = []
        for p in paths:
            if p not in frames:
                _head_block(p, notes, problems)
                df = jf.stream(p, keep=_head_pop, fields=HEAD_FIELDS, lists=HEAD_LISTS)
                frames[p] = None if df is None else kr.normalise(sb._n2048(df))
            if frames[p] is not None:
                dfs.append(frames[p])
        df = None if not dfs else dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
        entry = dict(cells={}, why={}, blocks=[p.name for p in paths], left_out=None)
        for key in ("a",) + HEAD_COMPARATORS:
            st, c = _head_k_cell(df, key, nb, keys)
            if st == "ok":
                entry["cells"][key] = c
            else:
                entry["why"][key] = f"{st}: {c}"
        a = entry["cells"].get("a")
        if a is None:
            entry["left_out"] = (f"no block holds rows at K = {nb}" if not paths
                                 else f"PsyPref {entry['why']['a']}")
            entry["cells"] = {}
            res[nb] = entry
            continue
        for key in [k for k in entry["cells"] if k != "a"]:
            for col in ("d_prior", "d_floor"):
                d = float(np.max(np.abs(rrep.col(entry["cells"][key], col) - rrep.col(a, col))))
                if d > 1e-12:
                    problems.append(f"fig_headline K = {nb}: {HEAD_NAME[key]}'s {col} differs "
                                    f"from PsyPref's by up to {d:.2e} Nm (left out)")
                    entry["why"][key] = f"{col} differs from PsyPref's by up to {d:.2e} Nm"
                    del entry["cells"][key]
                    break
        side = kr.sidecar_floors(out, nb)
        missing = [u for u in a["user_id"] if u not in side]
        if missing:
            entry["left_out"] = (f"users/<fold>-promp_s0-k{nb}.json give no floor for "
                                 f"{len(missing)} users")
        else:
            d = float(np.max(np.abs(rrep.col(a, "d_floor")
                                    - np.array([side[u] for u in a["user_id"]]))))
            entry["floor_check"] = d
            if d > 1e-12:
                entry["left_out"] = (f"d_floor differs from users/<fold>-promp_s0-k{nb}.json by "
                                     f"up to {d:.2e} Nm")
        if entry["left_out"]:
            problems.append(f"fig_headline K = {nb} left out: {entry['left_out']}")
            entry["cells"] = {}
        res[nb] = entry
    return res


def _head_win(df: pd.DataFrame) -> np.ndarray:
    """(users, 40) probability that the plain RMS simulated user prefers the identified assistance over
    generic assistance in one duel (gpeubo_report R2)."""
    return jf.win_probability(rrep.col(df, "d_prior")[:, None], rrep.lst(df, "distances"),
                              rrep.col(df, "s")[:, None], rrep.col(df, "lapse")[:, None])


def _room_curve(rooms, x: np.ndarray, measure: str, room: str) -> tuple[np.ndarray, np.ndarray]:
    """gpeubo_report.room_curves: the estimate at every k (40,) and the draws (draws, 40)."""
    est, boot = [], []
    for k in range(KMAX):
        e, b = rooms.stat(measure, x[:, k], room)
        est.append(e)
        boot.append(np.asarray(b, dtype=float).ravel())
    return np.array(est), np.array(boot).T


def _first_reach(curve: tuple, direction: str, target: tuple) -> dict:
    """gpeubo_report.first_reach: jnd_free.md section 5's first k (1-40) on the
    cross-sectional curve, against another statistic taken in the same draw; 2.5 / 97.5 %
    order statistics over the draws, and the share of draws that reach it by k = 40."""
    est, boot = curve
    t_est, t_boot = target
    point = float(jf.first_k(est, direction, t_est)[0])
    level = np.asarray(t_boot, dtype=float)
    draws = jf.first_k(boot, direction, level[:, None] if level.ndim else level)
    lo, hi = jf.inverted_cdf_ci(draws)
    return dict(point=point, lo=lo, hi=hi, reached=float(np.isfinite(draws).mean()))


def _fkci(d: dict) -> str:
    """gpeubo_report.fkci: first k [interval] (share of draws reaching it by k = 40)."""
    return f"{jf.fk(d['point'])} [{jf.fk(d['lo'])}, {jf.fk(d['hi'])}] ({d['reached']:.2f})"


def _head_ci(est, boot) -> dict:
    lo, hi = jf.ci(np.asarray(boot, dtype=float))
    return dict(est=float(est), lo=lo, hi=hi)


def _head_hashes(out: Path, data: Path) -> tuple[bool, str]:
    """gpeubo_report.data_hashes_ok: every Schaefer log the rooms read has the sha256 of
    median_room_data_check.json."""
    check = json.loads((out / "icorr" / "median_room_data_check.json").read_text(encoding="utf-8"))
    for f in check["files"]:
        path = data / f["name"]
        if not path.exists():
            return False, f"{path} is missing"
        if _sha256(path) != f["sha256"]:
            return False, f"{f['name']}: sha256 differs from median_room_data_check.json"
    return True, f"{len(check['files'])} files, sha256 as in median_room_data_check.json"


def headline_data(out: Path, data: Path, n_boot: int) -> dict:
    """Every number of fig_headline. The rooms are jnd_free's ``room_setup`` on e-main A's
    room with ``n_boot`` draws, the construction of gpeubo_report sections 3 and 7, shared by
    both panels (every K carries each user's weight from the K = 8 room, as k_sweep does)."""
    notes: list[str] = []
    problems: list[str] = []
    keys = G.load_sim(out / "phase5-schaefer")["keys"]
    folds = np.asarray([f for f, _ in keys])
    print("  streaming e-main and the GP blocks ...", flush=True)
    cells, state = _head_main_cells(out, keys, notes, problems)
    if state["gp_bald"][0] != "ok":
        raise SystemExit(f"fig_headline: GP-BALD {state['gp_bald'][0]} ({state['gp_bald'][1]})")
    a = cells["a"]
    for key, c in cells.items():
        if not np.array_equal(c["fold"].to_numpy(), folds):
            raise SystemExit(f"fig_headline: {key}: user order differs")
    ok, data_note = _head_hashes(out, data)
    if not ok:
        raise SystemExit(f"fig_headline: {data_note}")
    print(f"  rooms (jnd_free.room_setup, {n_boot} draws) ...", flush=True)
    setup = jf.room_setup(out, data, n_boot, rrep.col(a, "d_prior") - rrep.col(a, "d_floor"),
                          folds)
    if setup["sim"]["keys"] != keys:
        raise SystemExit("fig_headline: user order differs between load_sim and the rows")
    rooms = setup["rooms"]
    # the attainable best-profile level: the preference the best profile within the parameterization
    # (d_floor) would reach, per user, then averaged as the curves are (icorr_ceiling.py)
    c_user = jf.win_probability(rrep.col(a, "d_prior"), rrep.col(a, "d_floor"),
                                rrep.col(a, "s"), rrep.col(a, "lapse"))
    ceiling = {room: _head_ci(*rooms.stat("win", c_user, room)) for room in HEAD_ROOMS}
    D = rrep.lst(a, "distances")
    q1 = ((rrep.col(a, "d_prior") - D[:, K - 1]) / G.JND[1.0] >= 1.0 - G.TOL).astype(float)
    gates = {room: float(rooms.stat("q1", q1, room)[0]) for room in ("median", "central")}
    gate_ok = {"median": abs(gates["median"] - jf.GATE_MEDIAN[0]) <= jf.GATE_TOL
               and round(setup["level"]["median"], 2) == jf.GATE_MEDIAN_LEVEL,
               "central": abs(gates["central"] - jf.GATE_CENTRAL[0]) <= jf.GATE_TOL}
    if not all(gate_ok.values()):
        raise SystemExit(f"fig_headline: room gate failed: {gates} at {setup['level']}")

    opts = [key for key in HEAD_OPTS if key in cells]
    print("  win-rate curves k = 1..40 and first k ...", flush=True)
    W = {key: _head_win(cells[key]) for key in opts}
    E = {key: rrep.lst(cells[key], "distances") for key in ("a", "gp_bald")}
    curves, reach = {}, {}
    for room in HEAD_ROOMS:
        cw = {key: _room_curve(rooms, W[key], "win", room) for key in opts}
        ce = {key: _room_curve(rooms, E[key], "err", room) for key in E}

        def at(c, key, k):
            est, boot = c[key]
            return est[k - 1], boot[:, k - 1]

        r = {"A to BALD win@24": _first_reach(cw["a"], ">=", at(cw, "gp_bald", 24)),
             "A to BALD win@40": _first_reach(cw["a"], ">=", at(cw, "gp_bald", 40)),
             "A to BALD err@40": _first_reach(ce["a"], "<=", at(ce, "gp_bald", 40))}
        for k in (12, 24):
            r[f"BALD to A win@{k}"] = _first_reach(cw["gp_bald"], ">=", at(cw, "a", k))
            r[f"BALD to A err@{k}"] = _first_reach(ce["gp_bald"], "<=", at(ce, "a", k))
        reach[room] = r
        curves[room] = {key: dict(est=est, lo=np.percentile(boot, 2.5, axis=0),
                                  hi=np.percentile(boot, 97.5, axis=0))
                        for key, (est, boot) in cw.items()}
        del cw, ce

    print("  basis count K ...", flush=True)
    ksweep, kcells = headline_ksweep(out, keys, cells, state, rooms, notes, problems)
    return dict(curves=curves, reach=reach, ksweep=ksweep, kcells=kcells, state=state,
                opts=opts, gates=gates, level=dict(setup["level"]), ceiling=ceiling,
                ess={r: rooms.ess[r] for r in ("median", "central")}, notes=notes,
                problems=problems, n_boot=n_boot, data=data, data_note=data_note, n=len(keys))


def headline_ksweep(out: Path, keys, cells8: dict, state8: dict, rooms, notes: list,
                    problems: list, kdir: Path | None = None) -> tuple[dict, dict]:
    """Panel (b): per K and room, every complete cell's win rate after 24 comparisons, the best
    comparator, and the best-profile distance (PsyPref's d_floor, the same for every cell of that K)."""
    folds = np.asarray([f for f, _ in keys])
    kcells = _head_k_cells(out, keys, cells8, state8, notes, problems, kdir)
    ksweep = {}
    for nb, entry in kcells.items():
        if not entry["cells"]:
            continue
        kc = entry["cells"]
        for key, c in kc.items():
            if not np.array_equal(c["fold"].to_numpy(), folds):
                raise SystemExit(f"fig_headline: K = {nb} {key}: user order differs")
        w24 = {key: _head_win(c)[:, K - 1] for key, c in kc.items()}
        floor = rrep.col(kc["a"], "d_floor")
        ksweep[nb] = {}
        for room in HEAD_ROOMS:
            win = {key: _head_ci(*rooms.stat("win", w, room)) for key, w in w24.items()}
            comps = [c for c in HEAD_COMPARATORS if c in win]
            best = max(comps, key=lambda c: win[c]["est"]) if comps else None
            ksweep[nb][room] = dict(win=win, best=best,
                                    floor=_head_ci(*rooms.stat("err", floor, room)))
    summary = {nb: {k: v for k, v in e.items() if k != "cells"} | {"present": list(e["cells"])}
               for nb, e in kcells.items()}
    return ksweep, summary


# ---------------------------------------------------------------- the headline: checks

def _head_table(tables: list[dict], heading: str, first: str, label: str | None = None):
    """The first table whose heading starts with ``heading`` (and whose bold label is
    ``label``), or None."""
    for t in tables:
        if (t["heading"] or "").startswith(heading) and t["header"][0] == first and (
                label is None or t["label"] == label):
            return t
    return None


def headline_reports(out: Path) -> dict:
    """The report numbers the headline reproduces: gpeubo_report.md sections 2.1, 3.1, 7.1 and
    7.2, and k_sweep.md sections 2.1-2.3."""
    gpe, ksw = out / "icorr" / "gpeubo_report.md", out / "icorr" / "k_sweep.md"
    rep: dict = {"paths": (gpe, ksw)}
    tg = md_tables(gpe) if gpe.exists() else []
    t = _head_table(tg, "7.2 Comparisons to reach GP-BALD's level", "first k at which …")
    if t is not None:
        title_room = {v: k for k, v in HEAD_ROOM_NAME.items()}
        rooms_ = [title_room.get(h) for h in t["header"][1:]]
        for row in t["rows"]:
            for room, cell in zip(rooms_, row[1:]):
                rep[("reach", row[0], room)] = cell
    win_tables = {
        "simulated": (_head_table(tg, "2.1 Win rate against generic assistance", "optimizer"),
                      "gpeubo_report.md §2.1 (phase5e_report's draws)", False),
        "median": (_head_table(tg, "3.1 Win rate against generic assistance", "optimizer"),
                   "gpeubo_report.md §3.1", True),
        "central": (_head_table(tg, "7.1 The central literature room", "optimizer",
                                label="Win rate against generic assistance (weighted mean [95 % "
                                      "CI])"), "gpeubo_report.md §7.1", True)}
    for room, (t, src, same) in win_tables.items():
        if t is None:
            continue
        for row in t["rows"]:
            for k, cell in zip(KS, row[1:]):
                rep[("win", room, row[0], k)] = (nums(cell), src, same, 3)
    tk = md_tables(ksw) if ksw.exists() else []
    for room, head in (("simulated", "2.2 simulated room"), ("central", "2.3 central literature")):
        t = _head_table(tk, head, "K")
        if t is None:
            continue
        for row in t["rows"]:
            if len(row) > 3 and row[2].isdigit():
                rep[("ks", room, int(row[0]), row[1], int(row[2]))] = (
                    nums(row[3])[:3], f"k_sweep.md §{head.split()[0]} (2000 draws)", False, 2)
    t = _head_table(tk, "2.1 K context", "K")
    if t is not None:
        for row in t["rows"]:
            rep[("floor", int(row[0]))] = nums(row[2])[0]
    return rep


def _head_status(computed: list[float], reported, digits: int, same_draws: bool) -> str:
    """Point estimate identical at the printed precision; the interval too when the report used
    the same draws."""
    if not reported:
        return "not in a report"
    if not _close(computed[0], reported[0], digits):
        return (f"DIFFERS (point {computed[0]:.{digits}f}, report {reported[0]:.{digits}f})")
    if all(_close(c, r, digits) for c, r in zip(computed[1:], reported[1:])):
        return "identical at printed precision"
    if same_draws:
        return "DIFFERS (interval)"
    return "point identical; interval from other draws"


def _ci_txt(d: dict, digits: int = 3) -> str:
    return f"{d['est']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def headline_md(hd: dict, out: Path, problems: list[str]) -> list[str]:
    """The fig_headline section of figure_numbers.md, every plotted number with the report
    value it reproduces."""
    rep = headline_reports(out)
    gpe, ksw = rep["paths"]
    room, lv = HEAD_ROOM, hd["level"]
    bases = sorted(hd["ksweep"])
    L = ["## fig_headline.pdf", "",
         "Post hoc (gpeubo_report.md §7: asked for after the planned PF3 and PF3b were read; "
         f"no verdicts). Main condition (s = 1.0, λ = 0.1), the {hd['n']} population users of "
         "the five LOSO folds, the same users, ideals and answer seeds in every cell. PsyPref = "
         "A, Arens' GP = G and CMA-ES = SortCMA (with LineCoSpar) are e-main's stored rows; "
         "GP-BALD and GP-EUBO are `out/icorr/e-gpbald.jsonl` and `e-gpeubo.jsonl`, each checked "
         "by gpeubo_report's rule R4 (200 users, no error row, 40 comparisons, d_prior and "
         "d_floor bit-equal to A's). Win rate: P_i(k) = λ/2 + (1 − λ) Φ((d_prior,i − "
         "d_i(k))/s), the plain RMS wearer that answered the session, averaged over users with "
         "the room's weights (gpeubo R2). Rooms: `icorr_jnd_free.room_setup` on e-main A's "
         f"d_prior − d_floor (KDE reweighting, seed {jf.SEED}, {hd['n_boot']} draws: the draws "
         "of gpeubo_report §3 and §7), shared by both panels; at every K each user keeps the "
         "weight of the K = 8 room, as k_sweep.md does. Drawn: the "
         f"{HEAD_ROOM_NAME[room]} room ({lv[room]:.3f} Nm); recorded also: the central room "
         f"({lv['central']:.3f} Nm) and the simulated room (unit weights). Bands and intervals: "
         "2.5 / 97.5 % percentiles over the draws. First k: jnd_free.md §5's definition, the "
         "smallest k at which the cross-sectional statistic meets the target (not sustained, "
         "not a per-user first hit), the target being GP-BALD's statistic in the same draw "
         "(paired); interval: 2.5 / 97.5 % order statistics over the draws; in parentheses the "
         "share of draws that reach it by k = 40.", "",
         f"Gates (checks, not results): A's Q≥1 at k = 24 is {hd['gates']['median']:.3f} at the "
         f"median-matched room, level {lv['median']:.3f} Nm (expected {jf.GATE_MEDIAN[0]:.2f} "
         f"at {jf.GATE_MEDIAN_LEVEL:.2f}), and {hd['gates']['central']:.3f} at the central room "
         f"(expected {jf.GATE_CENTRAL[0]:.3f}). Effective sample size "
         + "; ".join(f"{HEAD_ROOM_NAME[r]} {e[0]:.1f} [{e[1]:.1f}, {e[2]:.1f}]"
                     for r, e in hd["ess"].items())
         + f". Schäfer logs: `{hd['data']}`, {hd['data_note']}.", ""]
    if hd["notes"]:
        L += ["Block files: " + " ".join(hd["notes"]), ""]

    # ---- the best-profile level, against best-profile level.md section 1
    ceil_md = out / "icorr" / "ceiling.md"
    want = {}
    if ceil_md.exists():
        t = _head_table(md_tables(ceil_md), "1. Main condition", "room")
        title = {"as simulated": "simulated", "median-matched (1.47 Nm)": "median",
                 "central (1.74 Nm)": "central"}
        for row in (t["rows"] if t else []):
            if row[0] in title:
                want[title[row[0]]] = nums(row[1])[:3]
    L += ["### The attainable ceiling (dashed line, drawn at the "
          f"{HEAD_ROOM_NAME[room]} room)", "",
          "Per user λ/2 + (1 − λ) Φ((d_prior − d_floor)/s): the preference the best profile "
          "within the parameterization would reach; averaged with the room's weights, the same "
          f"draws as the curves ({hd['n_boot']}). ceiling.md §1 used jnd_free's 2000 draws, so "
          "only its point estimate must agree.", "",
          "| room | computed | ceiling.md §1 | status |", "|---|---|---|---|"]
    for r in HEAD_ROOMS:
        c = hd["ceiling"][r]
        w = want.get(r)
        st = _head_status([c["est"], c["lo"], c["hi"]], w, 3, False)
        if st.startswith("DIFFERS") or w is None:
            problems.append(f"fig_headline ceiling {r}: {st}")
        L.append(f"| {HEAD_ROOM_NAME[r]}{' — drawn' if r == room else ''} | {_ci_txt(c)} | "
                 f"{'—' if w is None else f'{w[0]:.3f} [{w[1]:.3f}, {w[2]:.3f}]'} | {st} |")
    L.append("")

    # ---- (a) the first k, against section 7.2
    L += ["### (a) Comparisons to reach GP-BALD's level: gpeubo_report.md §7.2", "",
          "Every row of §7.2 is recomputed; the first two are drawn (dashed level lines, "
          "markers on the curves). Status: the computed text equals the report's character for "
          "character.", "",
          "| first k at which … | room | computed | gpeubo_report.md §7.2 | status |",
          "|---|---|---|---|---|"]
    for key, text in HEAD_REACH_ROWS:
        for r in HEAD_ROOMS:
            got = _fkci(hd["reach"][r][key])
            want = rep.get(("reach", text, r))
            st = ("not in the report" if want is None else
                  "identical" if got == want else "DIFFERS")
            if st != "identical":
                problems.append(f"fig_headline §7.2 '{text}', {r}: {got} vs {want}")
            mark = " (drawn)" if r == room and key in ("A to BALD win@24", "A to BALD win@40") \
                else ""
            L.append(f"| {text}{mark} | {HEAD_ROOM_NAME[r]} | {got} | {want or '—'} | {st} |")
    L.append("")

    # ---- (a) the curves at k = 12, 24, 40, against sections 2.1, 3.1 and 7.1
    L += ["### (a) Win rate at k = 12, 24 and 40: gpeubo_report.md §2.1, §3.1, §7.1", "",
          "Estimate [95 % interval]. §3.1 and §7.1 used the same rooms and draws, so the "
          "interval must agree too; §2.1 used phase5e_report's draws for the simulated room, so "
          "only its point estimate must.", "",
          "| optimizer | room | k | computed | report | source | status |",
          "|---|---|---|---|---|---|---|"]
    for key in reversed(hd["opts"]):
        for r in HEAD_ROOMS:
            for k in KS:
                c = hd["curves"][r][key]
                comp = [c["est"][k - 1], c["lo"][k - 1], c["hi"][k - 1]]
                got = rep.get(("win", r, HEAD_GPE_NAME[key], k))
                if got is None:
                    st, rtxt, src = "not in a report", "—", "—"
                else:
                    vals, src, same, dg = got
                    st = _head_status(comp, vals, dg, same)
                    rtxt = f"{vals[0]:.3f} [{vals[1]:.3f}, {vals[2]:.3f}]"
                if st.startswith("DIFFERS"):
                    problems.append(f"fig_headline {HEAD_NAME[key]} {r} k = {k}: {st}")
                L.append(f"| {HEAD_NAME[key]} | {HEAD_ROOM_NAME[r]} | {k} | "
                         f"{comp[0]:.3f} [{comp[1]:.3f}, {comp[2]:.3f}] | {rtxt} | {src} | {st} |")
    L.append("")

    # ---- (a) the annotation as drawn
    C = hd["curves"][room]
    L += [f"### (a) The annotation as drawn ({HEAD_ROOM_NAME[room]} room)", "",
          "| GP-BALD's budget | GP-BALD's win rate there (level line) | PsyPref's first k "
          "[interval] (share) | PsyPref's win rate at that k (marker) | label |",
          "|---|---|---|---|---|"]
    for kk in HEAD_REACH:
        r = hd["reach"][room][f"A to BALD win@{kk}"]
        ka = int(r["point"])
        L.append(f"| k = {kk} | {C['gp_bald']['est'][kk - 1]:.3f} | {_fkci(r)} | "
                 f"{C['a']['est'][ka - 1]:.3f} | same result in {ka} vs {kk} |")
    L.append("")
    L += [f"### (a) Every plotted value, k = 1..40 ({HEAD_ROOM_NAME[room]} room)", "",
          "PsyPref and GP-BALD: estimate [band]; the others: estimate (drawn without a band).",
          "", "| k | " + " | ".join(HEAD_NAME[key] for key in reversed(hd["opts"])) + " |",
          "|---|" + "---|" * len(hd["opts"])]
    for k in range(1, KMAX + 1):
        cells_ = []
        for key in reversed(hd["opts"]):
            c = C[key]
            cells_.append(f"{c['est'][k - 1]:.3f} [{c['lo'][k - 1]:.3f}, {c['hi'][k - 1]:.3f}]"
                          if key in ("a", "gp_bald") else f"{c['est'][k - 1]:.3f}")
        L.append(f"| {k} | " + " | ".join(cells_) + " |")
    L.append("")

    # ---- (b) the basis count
    L += ["### Basis count K (recorded, not drawn: the figure is one panel since 2026-10-21; "
          "Fig. 2 (d) of the paper shows the preference against K)", "",
          "Each K against its own generic assistance (its prior mean, the row's d_prior) and its "
          "own floor (the row's d_floor, the best reachable error in that K's family; the "
          "weighted Hazen median at the room). Best comparator: the complete comparator cell "
          "with the highest win rate at k = 24 at that room; which cells exist differs by K, so "
          "it is named per K (marker shape in the figure). Drawn: the "
          f"{HEAD_ROOM_NAME[room]} room. K = 8 reproduces gpeubo_report.md (same draws); K = 4 "
          "and 12 reproduce k_sweep.md's point estimates (it printed two decimals and used 2000 "
          "draws); the median-matched room at K ≠ 8 is in no report.", "",
          "Cells per K:", ""]
    for nb in HEAD_BASES:
        e = hd["kcells"].get(nb)
        if e is None:
            continue
        if e["left_out"]:
            L.append(f"- K = {nb}: left out ({e['left_out']}).")
            continue
        missing = "; ".join(f"{HEAD_NAME[k]} {v}" for k, v in e["why"].items())
        L.append(f"- K = {nb}: " + ", ".join(HEAD_NAME[k] for k in ("a",) + HEAD_COMPARATORS
                                             if k in e["present"])
                 + f" (from {', '.join(e['blocks'])})" + (f"; not present: {missing}" if missing
                                                           else "") + "."
                 + (f" Floors equal users/<fold>-promp_s0-k{nb}.json (max |Δ| "
                    f"{e['floor_check']:.1e} Nm)." if "floor_check" in e else ""))
    L += ["", f"Best comparator per K at the {HEAD_ROOM_NAME[room]} and the central "
          "room, with the runner-up; \"not separated\": the runner-up's estimate lies inside the "
          "best's 95 % interval, so the marker's identity there is fragile. A CMA-ES win rate "
          "near 0.50 is its start-up (its first population not yet ranked by k = 24; "
          "k_sweep.md §4), not its search.", ""]
    for nb in bases:
        parts = []
        for r in (room, "central"):
            win = hd["ksweep"][nb][r]["win"]
            order = sorted((c for c in HEAD_COMPARATORS if c in win),
                           key=lambda c: -win[c]["est"])
            if not order:
                continue
            b = win[order[0]]
            txt = f"{HEAD_ROOM_NAME[r]}: {HEAD_NAME[order[0]]} {_ci_txt(b)}"
            if len(order) > 1:
                ru = win[order[1]]
                close = b["lo"] <= ru["est"] <= b["hi"]
                txt += (f", runner-up {HEAD_NAME[order[1]]} {ru['est']:.3f}"
                        + (" (not separated)" if close else ""))
            parts.append(txt)
        same = len({hd["ksweep"][nb][r]["best"] for r in (room, "central")}) == 1
        L.append(f"- K = {nb}: " + "; ".join(parts)
                 + ("." if same else "; the best comparator differs between the two rooms."))
    L += ["", "| K | room | optimizer | win rate, k = 24 [95 %] | report | source | status |",
          "|---|---|---|---|---|---|---|"]
    for nb in bases:
        for r in HEAD_ROOMS:
            v = hd["ksweep"][nb][r]
            for key in ("a",) + HEAD_COMPARATORS:
                if key not in v["win"]:
                    continue
                w = v["win"][key]
                comp = [w["est"], w["lo"], w["hi"]]
                if nb == 8:
                    got = rep.get(("win", r, HEAD_GPE_NAME[key], K))
                else:
                    got = (rep.get(("ks", r, nb, HEAD_KS_NAME[key], K))
                           if key in HEAD_KS_NAME else None)
                if got is None:
                    st, rtxt, src = "not in a report", "—", "—"
                else:
                    vals, src, same, dg = got
                    st = _head_status(comp, vals, dg, same)
                    rtxt = ", ".join(f"{x:.{dg}f}" for x in vals[:1]) + (
                        f" [{vals[1]:.{dg}f}, {vals[2]:.{dg}f}]" if len(vals) >= 3 else "")
                if st.startswith("DIFFERS"):
                    problems.append(f"fig_headline (b) K = {nb} {HEAD_NAME[key]} {r}: {st}")
                tag = " (best)" if key == v["best"] else ""
                L.append(f"| {nb} | {HEAD_ROOM_NAME[r]} | {HEAD_NAME[key]}{tag} | "
                         f"{_ci_txt(w)} | {rtxt} | {src} | {st} |")
    L += ["", "| K | room | floor, Nm [95 %] | k_sweep.md §2.1 (median, simulated) | status |",
          "|---|---|---|---|---|"]
    for nb in bases:
        for r in HEAD_ROOMS:
            fl = hd["ksweep"][nb][r]["floor"]
            want = rep.get(("floor", nb)) if r == "simulated" else None
            st = _head_status([fl["est"]], [want] if want is not None else None, 3, False)
            if st.startswith("DIFFERS"):
                problems.append(f"fig_headline (b) floor K = {nb}: {st}")
            L.append(f"| {nb} | {HEAD_ROOM_NAME[r]} | {_ci_txt(fl)} | "
                     f"{want if want is not None else '—'} | {st} |")
    own = [p for p in problems if p.startswith("fig_headline")]
    L += ["", "Status of this figure: " + ("every number a report gives reproduces it."
                                           if not own else f"{len(own)} numbers do not "
                                           "reproduce (listed under 'Numbers that do not "
                                           "reproduce')."), ""]
    return L


# ---------------------------------------------------------------- the headline: figure

def fig_headline(hd: dict, figdir: Path, preview) -> Path:
    """One panel: the preference over generic assistance against the comparisons, every
    optimizer at the median-matched room, with the attainable best-profile level and 50 % dashed. (The
    basis-count panel moved to Fig. 2 (d) on 2026-10-21; its numbers stay in
    figure_numbers.md.)"""
    W, H = FIG_W, 2.30
    left, width = 0.50, 2.12
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, 0.36 / H, width / W, 1.86 / H])
    room = HEAD_ROOM
    C = hd["curves"][room]
    ks = np.arange(1, KMAX + 1)
    opts = hd["opts"]

    # the preference over generic assistance, k = 1..40; 50 % and the best-profile level dashed
    ceil = hd["ceiling"][room]["est"]
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.axhline(ceil, color=INK, lw=0.9, ls=(0, (2.4, 1.3)), zorder=0)
    for key, alpha in (("gp_bald", 0.12), ("a", 0.15)):
        ax.fill_between(ks, C[key]["lo"], C[key]["hi"], color=HEAD_STYLE[key][0], alpha=alpha,
                        lw=0, zorder=1)
    for key in opts:
        colour, ls, lw = HEAD_STYLE[key]
        ax.plot(ks, C[key]["est"], color=colour, ls=ls, lw=lw,
                zorder=3 if key in ("a", "gp_bald") else 2, solid_capstyle="butt")
    # the comparison saving: GP-BALD's preference after 24 and 40 comparisons (dashed level),
    # where PsyPref first reaches it (blue) and where GP-BALD has it (red); each label sits
    # above PsyPref's band with a leader to PsyPref's marker, below the best-profile level line
    a_hi = C["a"]["hi"]
    text_h = 0.082
    top = ceil + 0.035
    for i, kk in enumerate(HEAD_REACH):
        r = hd["reach"][room][f"A to BALD win@{kk}"]
        ka = int(r["point"])
        level = float(C["gp_bald"]["est"][kk - 1])
        y_a = float(C["a"]["est"][ka - 1])
        ax.plot([ka, kk], [level, level], color=INK, lw=0.6, ls=(0, (2.2, 1.4)), zorder=4)
        ax.plot([kk], [level], marker="D", ms=3.4, mfc=BALD_RED, mec="white", mew=0.5,
                ls="none", zorder=5)
        ax.plot([ka], [y_a], marker="o", ms=3.8, mfc=BLUE, mec="white", mew=0.5, ls="none",
                zorder=5)
        x_t = max(ka - 2.0 + 5.0 * i, 8.5)   # the two-line label is about 13 k wide
        lo_k, hi_k = max(1, int(x_t - 6)), min(KMAX, int(x_t + 6))
        y_t = min(float(np.max(a_hi[lo_k - 1:hi_k])) + 0.010, ceil - 0.008 - text_h)
        ax.plot([ka, x_t], [y_a + 0.008, y_t - 0.004], color=MUTED, lw=0.5, zorder=4)
        ax.text(x_t, y_t, f"same result in\n{ka} vs {kk}", ha="center", va="bottom",
                fontsize=8, color=INK, linespacing=1.0, zorder=6)
    ax.set_xlim(1, KMAX)
    ax.set_xticks([1, 12, 24, 40])
    lo_y = min(float(np.min(C[k]["est"])) for k in opts)
    ax.set_ylim(np.floor(lo_y * 20) / 20, top)
    ticks = [t for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) if ax.get_ylim()[0] <= t <= top]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{100 * t:.0f}" for t in ticks])
    ax.set_xlabel("comparisons k", labelpad=1.5)
    ax.set_ylabel("preference over generic\nassistance (%)", labelpad=2)
    _spines(ax, grid_axis="y")
    # direct labels at the right end: every optimizer, then the two reference lines
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    ends = [float(C[k]["est"][-1]) for k in opts] + [ceil, 0.5]
    names = [HEAD_NAME[k] for k in opts] + ["ceiling", "generic"]
    pos = spread(ends, 0.125 * span / 1.86 * 1.0)
    for i, (y_end, y_lab, name) in enumerate(zip(ends, pos, names)):
        ax.plot([KMAX + 0.4, KMAX + 1.4], [y_end, y_lab], color=CONTEXT, lw=0.5, clip_on=False)
        if i < len(opts):
            colour, ls, lw = HEAD_STYLE[opts[i]]
        else:
            colour, ls, lw = (INK, (0, (2.4, 1.3)), 0.9) if name == "ceiling" else \
                (MUTED, (0, (3, 2)), 0.7)
        ax.plot([KMAX + 1.4, KMAX + 4.0], [y_lab, y_lab], color=colour, ls=ls, lw=lw,
                clip_on=False, solid_capstyle="butt")
        ax.text(KMAX + 4.6, y_lab, name, ha="left", va="center", color=INK, fontsize=8,
                clip_on=False)
    return _save(fig, figdir, "fig_headline", preview)


# ---------------------------------------------------------------- figure_numbers.md

def numbers_md(ex: dict, psy: dict, learn: dict, rob: list[dict], report: dict, own_md: dict,
               n_boot: int, out: Path, room_lines: list[str],
               problems: list[str]) -> tuple[str, list[str]]:
    L = ["# Figure numbers", "",
         "Written by `scripts/paper_icorr_figures.py` from the stored sweep rows "
         f"(`out/phase5e`, `out/phase5c`, `out/phase5d`, read only). Population users, "
         f"200 per cell, {n_boot} bootstrap draws with phase5e_report's and benchmark_report's "
         "seeds. Each plotted number is set next to the value in "
         "`out/phase5e/REPORT.md` (REPORT) or `out/phase5e/own_generic.md` (own_generic) that it "
         "reproduces; the status compares them at the printed precision. Every difference "
         "between optimizers is the difference of the medians on the same users (lower middle "
         "values, `inverted_cdf`). fig_room is checked against "
         "`out/schaefer_grounding/literature_room.md` and `optimizers_real_room.md` instead "
         "(its own section). The rows were streamed and cut to the first 40 comparisons "
         "(`lean_load`); every check below ran on those rows.", "",
         "Planned and post hoc. The predictions QE1-QE6 were recorded before the runs. The "
         "RMS contrasts of fig_robustness (a) are the contrasts QE1 (main condition) and QE2 "
         "(the other RMS cells) were judged on, read here against the best comparator as REPORT "
         "section 3 reads them; the s = 0.5 reference and the own-metric readings of the † "
         "cells (REPORT section 8, spec change E4a) are not planned predictions. Post "
         "hoc: the quality levels (Q≥1 in fig_learning, the lower decile of gain in "
         "fig_robustness (b)) are post hoc on the stored sessions (REPORT section 12), and the "
         "own-metric gains over generic assistance were added after the planned predictions "
         "were read (own_generic). Optimizer A was developed on this simulator and was not "
         "retuned for 24 comparisons.", ""]

    # ---- fig_session
    contrast_rows = [v for (h, c), v in report.items()
                     if h == "2. Paired contrasts at k = 24" and c == "main (s = 1.0)"]
    rep_med = nums(contrast_rows[0][3])[0] if contrast_rows else None
    st = verdict([ex["median"]], [rep_med] if rep_med is not None else [], 3)
    if not st.startswith("reproduces"):
        problems.append(f"fig_session example median: {st}")
    gain_jnd = (ex["d_prior"] - ex["errors"][24]) / JND
    L += ["## fig_session.pdf", "",
          "### (a) one simulated user, main condition (s = 1.0, lapse 0.1), ProMP, A", "",
          f"Chosen user: **{ex['user_id']}** (fold {ex['fold']}), the population user of e-main "
          f"whose error after 24 comparisons equals the median of A's 200 errors (the lower "
          f"middle value, as the contrasts take it). The session was run again through "
          f"`run_phase5_point` at 12 and 24 comparisons; its answers and distances reproduce "
          f"the stored row exactly (checked in the script), so the curves are those of the "
          f"stored session.", "",
          "| plotted number | value | source | REPORT value | status |", "|---|---|---|---|---|",
          f"| A's error at k = 24 (Nm) | {ex['errors'][24]:.3f} | stored distances[23] | "
          f"main A median in the contrasts: {rep_med:.3f} (section 2, `1.710 / …`) | {st} |",
          f"| A's error at k = 12 (Nm) | {ex['errors'][12]:.3f} | stored distances[11] | — "
          f"(this user only) | matches the stored row |",
          f"| generic assistance error, d_prior (Nm) | {ex['d_prior']:.3f} | stored d_prior; "
          f"rebuilt as `prior_mean_distance` builds it (2,000 projected draws, seed 0) | — | "
          f"matches the stored row |",
          f"| optimum within the parameterization, d_floor (Nm, not drawn) | {ex['d_floor']:.3f}"
          f" | stored d_floor | — | — |",
          f"| this user's gain at k = 24 | {ex['d_prior'] - ex['errors'][24]:.3f} Nm = "
          f"{gain_jnd:.2f} JND | — | median over users {ex['median_gain_nm']:.3f} Nm (REPORT "
          f"section 1: median improvement 1.317) | context |", "",
          f"This user's generic assistance is farther from the ideal than for the median user "
          f"({ex['d_prior']:.2f} Nm against a median d_prior of {ex['median_d_prior']:.2f} Nm), "
          f"so its gain ({gain_jnd:.1f} JND) is above the median gain (1.95 JND, REPORT "
          f"section 10); the user is typical in the error it ends with, not in its gain. Its "
          f"ideal reaches the device limit of ±{TORQUE_LIMIT_NM:g} Nm "
          f"({'yes' if ex['touches_limit'] else 'no'}); {ex['n_touch']} of {ex['n']} users of "
          f"the cell do (REPORT drift tables: 68 of 200). The ideal curve is a simulated "
          f"population user, not a recorded subject.", "",
          "### (b) psychometric function", "",
          "P(choose the curve closer to the ideal) = λ/2 + (1 − λ) Φ(Δd / s), the plain "
          "observer's `prob_prefers_a`, with Δd the difference of the RMS distances of the two "
          "curves. The JND is defined without lapse (`prompref.user.jnd`): Φ(JND / s) = 0.75. "
          "With λ = 0.1 the curve passes 0.725 at the JND and levels off at 0.05 and 0.95; the "
          "open marker is the lapse-free 75 % point, the filled one the λ = 0.1 curve.", "",
          "| plotted number | value | REPORT value | status |", "|---|---|---|---|"]
    rep_jnds = [0.337, 0.674, 1.349]  # REPORT intro: "0.337, 0.674 and 1.349 Nm"
    for s, r in zip((0.5, 1.0, 2.0), rep_jnds):
        stj = verdict([psy["jnd"][s]], [r], 3)
        if not stj.startswith("reproduces"):
            problems.append(f"JND s = {s}: {stj}")
        L.append(f"| JND at s = {s:g} (Nm) | {psy['jnd'][s]:.4f} | {r:.3f} (yardstick, REPORT "
                 f"intro; drawn only for s = 1.0) | {stj} |")
    L += [f"| P at the JND, s = 1.0, λ = 0.1 | {psy['p_at_jnd']:.4f} | — (0.05 + 0.9 · 0.75) | "
          f"analytic |",
          f"| P at the JND without lapse | 0.7500 | — | analytic |",
          f"| asymptotes λ/2, 1 − λ/2 | {psy['asymptotes'][0]:.2f}, {psy['asymptotes'][1]:.2f} "
          f"| — | analytic |", ""]

    # ---- fig_learning
    L += ["## fig_learning.pdf", "",
          "Main condition (s = 1.0, lapse 0.1), e-main rows, 200 population users per "
          "optimizer; A, G, LineCoSpar and SortCMA on ProMP, the lens on Franks. (a) Median and "
          "interquartile range of the error of the identified assistance (`distances[k − 1]`, "
          "linear percentiles as `benchmark_report._curve` takes them, here up to k = 40). "
          "(b) The share of users with (d_prior − distances[k − 1]) / true_jnd(1.0) ≥ 1, "
          "`benchmark_report.share_curve(df, \"q1\", kmax=40, jnd=0.6745)`. Interquartile "
          "bands are drawn for A and G only (Rv14); every IQR is in the tables. The thin "
          f"solid gray reference in (a) is the median generic-assistance error on ProMP "
          f"({learn['a']['d_prior_median']:.3f} Nm; on Franks "
          f"{learn['a_lens']['d_prior_median']:.3f} Nm).", "",
          "| optimizer | k | median error (Nm) | REPORT B7 median error | status | IQR (Nm), "
          "not in REPORT | Q≥1 share | REPORT B7 Q≥1 | status |",
          "|---|---|---|---|---|---|---|---|---|"]
    b7_name = {"a": "A", "a_lens": "lens", "g": "preference GP (G)", "d_z": "LineCoSpar",
               "cmaes": "SortCMA"}
    for key in SERIES_ORDER:
        fam_name = "Franks" if key == "a_lens" else "ProMP"
        d = learn[key]
        for k in KS:
            cells = report.get((f"main (s = 1.0), {fam_name}, k = {k}", b7_name[key]))
            rep_med = nums(cells[2])[0] if cells else None
            rep_q1 = nums(cells[4])[0] if cells else None
            s1 = verdict([d["median"][k - 1]], [rep_med] if cells else [], 3)
            s2 = verdict([d["share"][k - 1]], [rep_q1] if cells else [], 2)
            for s_, what in ((s1, "median"), (s2, "Q≥1")):
                if not s_.startswith("reproduces"):
                    problems.append(f"fig_learning {SERIES[key][2]} k = {k} {what}: {s_}")
            L.append(f"| {SERIES[key][2]} | {k} | {d['median'][k - 1]:.3f} | "
                     f"{cells[2] if cells else '—'} | {s1} | [{d['q25'][k - 1]:.3f}, "
                     f"{d['q75'][k - 1]:.3f}] | {d['share'][k - 1]:.3f} | "
                     f"{cells[4] if cells else '—'} | {s2} |")
    L += ["", "Every plotted value, k = 1..40 (median [q25, q75] in Nm / share with gain ≥ 1 "
          "JND); only k = 12, 24 and 40 appear in REPORT:", "",
          "| k | " + " | ".join(SERIES[key][2] for key in SERIES_ORDER) + " |",
          "|---|" + "---|" * len(SERIES_ORDER)]
    for k in range(1, KMAX + 1):
        L.append(f"| {k} | " + " | ".join(
            f"{learn[key]['median'][k - 1]:.3f} [{learn[key]['q25'][k - 1]:.3f}, "
            f"{learn[key]['q75'][k - 1]:.3f}] / {learn[key]['share'][k - 1]:.3f}"
            for key in SERIES_ORDER) + " |")
    L.append("")

    # ---- fig_robustness
    L += ["## fig_robustness.pdf", "",
          "(a) A minus the best ProMP comparator at k = 24: the difference of the medians on "
          "the same users (inverted_cdf), 95 % cluster bootstrap over users stratified by fold "
          f"(`paired_bootstrap`, {n_boot} draws, seed 0). The best comparator is chosen as "
          "REPORT section 3 chooses it (lowest median on the users common to all optimizers), "
          "in the RMS error; in the cells marked † it is chosen and scored in the wearer's own "
          "distance, as REPORT section 8 does (`own_distances`, Nm-equivalent for the landmark "
          "wearer). The Weber wearer's own distance is the RMS distance, so its reading is the "
          "same in both. (b) A's lower decile of gain over generic assistance in the cell's JND "
          "(`benchmark_report.lower_decile`: linear 10th percentile, cluster bootstrap within "
          "folds, seed 20261007); the heterogeneous cell uses the population JND (0.674 Nm), as "
          "in REPORT; † cells use the own-metric gain of own_generic (post hoc); the drift cells "
          "have none (the ideal moves; d_prior belongs to the start ideal). |Δp| is the cell's "
          "perturbation strength (`perturbation_strength.json`, REPORT section 9).", "",
          "| cell | metric | best comparator (plotted / REPORT) | Δ median, Nm [95 % CI] "
          "(plotted) | REPORT | status | lower decile, JND [95 % CI] (plotted) | REPORT / "
          "own_generic | status | mean abs Δp | REPORT | status |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rob:
        cid, cond_title = r["cid"], r["title"]
        left = r["left"]
        name = r5._label("promp_s0", left["best"])
        if r["own"]:
            heading = r5.BY_ID[cid].title
            hit = [(c, v) for (h, c), v in report.items()
                   if h == heading and c.startswith("A − best comparator (")]
            rep_name = hit[0][0][len("A − best comparator ("):-len(") (ProMP)")] if hit else "—"
            rep_str = hit[0][1][3] if hit else ""
            src = "REPORT §8"
        else:
            cells = report.get(("3. Ranking at k = 24", cond_title))
            rep_str = cells[2] if cells else ""
            m = re.match(r"^(.*?) [-+]?\d", rep_str)
            rep_name = m.group(1) if m else "—"
            src = "REPORT §3"
        diff = left["diff"]
        computed = [diff["estimate"], diff["lo"], diff["hi"]]
        reported = nums(rep_str)
        st_l = verdict(computed, reported, 3)
        if rep_name != name:
            st_l = f"COMPARATOR DIFFERS ({name} vs {rep_name}); " + st_l
        if not st_l.startswith("reproduces"):
            problems.append(f"fig_robustness (a) {cid}: {st_l}")
        if r["decile"] is None:
            dec_str, rep_dec, st_d = "— (drift: none)", "—", "—"
        else:
            dd = r["decile"]
            dec_str = (f"{sb._sg(dd['estimate'], 2)} [{sb._sg(dd['lo'], 2)}, "
                       f"{sb._sg(dd['hi'], 2)}]")
            if r["decile_src"] == "own":
                cells = own_md.get((OWN_NAMES[cid], "ProMP", "A"))
                rep_dec = cells[7] if cells else ""
                rep_dec_lbl = f"own_generic: {rep_dec}"
            else:
                cells = report.get((f"{cond_title}, ProMP, k = 24", "A"))
                rep_dec = cells[7] if cells else ""
                rep_dec_lbl = f"REPORT B7: {rep_dec}"
            st_d = verdict([dd["estimate"], dd["lo"], dd["hi"]], nums(rep_dec), 2)
            if not st_d.startswith("reproduces"):
                problems.append(f"fig_robustness (b) {cid}: {st_d}")
            rep_dec = rep_dec_lbl
        pcells = report.get(("9. Perturbation strength", cond_title))
        rep_dp = pcells[1] if pcells else ""
        st_p = verdict([r["dp"]], nums(rep_dp), 3)
        if not st_p.startswith("reproduces"):
            problems.append(f"fig_robustness |Δp| {cid}: {st_p}")
        L.append(f"| {cond_title}{' †' if r['own'] else ''} | "
                 f"{'own distance' if r['own'] else 'RMS'} | {name} / {rep_name} | "
                 f"{sb._ci(diff)} | {src}: {rep_str} | {st_l} | {dec_str} | {rep_dec} | {st_d} "
                 f"| {r['dp']:.3f} (plotted {r['dp']:.2f}) | {rep_dp} | {st_p} |")
    L += ["", "The † cells read in the RMS error instead (REPORT section 3; not plotted):", "",
          "| cell | best comparator in RMS | Δ median, RMS, Nm [95 % CI] | REPORT §3 | status |",
          "|---|---|---|---|---|"]
    for r in rob:
        if not r["own"]:
            continue
        rr = r["rms_reading"]
        cells = report.get(("3. Ranking at k = 24", r["title"]))
        rep_str = cells[2] if cells else ""
        st = verdict([rr["diff"]["estimate"], rr["diff"]["lo"], rr["diff"]["hi"]],
                     nums(rep_str), 3)
        if not st.startswith("reproduces"):
            problems.append(f"fig_robustness RMS reading {r['cid']}: {st}")
        L.append(f"| {r['title']} | {r5._label('promp_s0', rr['best'])} | "
                 f"{sb._ci(rr['diff'])} | {rep_str} | {st} |")
    L += [""] + room_lines
    L += ["## Numbers that do not reproduce", ""]
    L += [f"- {p}" for p in problems] if problems else [
        "None: every plotted number above reproduces its REPORT.md, own_generic.md, "
        "grounding, gpeubo_report.md or k_sweep.md value. fig_room's curve values are not expected to equal the caption's "
        "KDE-reweighted shares (see its section); that is a difference of estimands, not a "
        "mismatch."]
    return "\n".join(L) + "\n", problems


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True,
                    help="the sweep output directory holding phase5e/, phase5c/, phase5d/")
    ap.add_argument("--figdir", type=Path, default=HERE.parent / "paper" / "figures")
    ap.add_argument("--boot", type=int, default=10_000,
                    help="bootstrap draws (10,000 reproduces REPORT.md's intervals exactly)")
    ap.add_argument("--preview", type=Path, default=None,
                    help="also write PNG previews here (not into the paper)")
    ap.add_argument("--data", type=Path, default=None,
                    help="the Schaefer self-tuning logs for the realistic rooms of "
                         "fig_headline (default: the folder median_room_data_check.json names)")
    ap.add_argument("--only", nargs="+", choices=("headline", "axes"), default=None,
                    help="draw only these figures and replace their sections of "
                         "figure_numbers.md")
    args = ap.parse_args(argv)
    if args.boot < 2000:
        raise SystemExit("--boot must be at least 2000")
    if args.boot != HEAD_BOOT:
        warnings.warn(f"fig_headline reproduces gpeubo_report section 7 only with {HEAD_BOOT} "
                      "draws")
    sys.stdout.reconfigure(encoding="utf-8")
    dir5e = args.out / "phase5e"
    args.figdir.mkdir(parents=True, exist_ok=True)
    if args.preview is not None:
        args.preview.mkdir(parents=True, exist_ok=True)
    data = args.data or Path(json.loads((args.out / "icorr" / "median_room_data_check.json")
                                        .read_text(encoding="utf-8"))["data"])
    md = args.figdir / "figure_numbers.md"
    if args.only:
        every: list[str] = []
        if "axes" in args.only:
            print("fig_axes (read from out/icorr reports):")
            ad = axes_data(args.out)
            print(f"wrote {fig_axes(ad, args.figdir, args.preview)}")
            problems = list(ad["problems"])
            splice_section(md, axes_md(ad), "## fig_axes.pdf", problems)
            print(f"wrote {md} (section fig_axes.pdf)")
            every += problems
        if "headline" in args.only:
            print("fig_headline:")
            hd = headline_data(args.out, data, args.boot)
            print(f"wrote {fig_headline(hd, args.figdir, args.preview)}")
            problems = list(hd["problems"])
            splice_section(md, headline_md(hd, args.out, problems), "## fig_headline.pdf",
                           problems)
            print(f"wrote {md} (section fig_headline.pdf)")
            every += problems
        _print_problems(every)
        return 0
    ctx = r5.Ctx(dir5e, args.out / "phase5c", args.out / "phase5d", args.boot)
    report = parse_tables(dir5e / "REPORT.md")
    own_md = {(cells[0], cells[1], cells[2]): cells
              for _, cells in parse_rows(dir5e / "own_generic.md") if len(cells) >= 8}

    print("fig_session: rerunning the median user's session ...")
    ex = session_example(ctx)
    print(f"  user {ex['user_id']}: errors {ex['errors'][12]:.3f} (k = 12), "
          f"{ex['errors'][24]:.3f} (k = 24), d_prior {ex['d_prior']:.3f}")
    p1, psy = fig_session(ex, args.figdir, args.preview)
    print(f"wrote {p1}")
    learn = learning_data(ctx)
    p2 = fig_learning(learn, args.figdir, args.preview)
    print(f"wrote {p2}")
    print("fig_robustness:")
    rob = robustness_data(ctx, dir5e)
    p3 = fig_robustness(rob, args.figdir, args.preview)
    print(f"wrote {p3}")
    print("fig_room:")
    rd = room_data(ctx)
    gr = grounding_numbers(args.out)
    sim_median = float(np.median(rd["room"]))
    p4 = fig_room(rd, gr, sim_median, args.figdir, args.preview)
    print(f"wrote {p4}")
    print("fig_axes (read from out/icorr reports):")
    ad = axes_data(args.out)
    p5 = fig_axes(ad, args.figdir, args.preview)
    print(f"wrote {p5}")
    print("fig_headline:")
    hd = headline_data(args.out, data, args.boot)
    p6 = fig_headline(hd, args.figdir, args.preview)
    print(f"wrote {p6}")
    problems: list[str] = list(ad["problems"]) + list(hd["problems"])
    room_lines = (room_md(rd, gr, sim_median, ctx, problems) + axes_md(ad)
                  + headline_md(hd, args.out, problems))
    text, problems = numbers_md(ex, psy, learn, rob, report, own_md, args.boot, args.out,
                                room_lines, problems)
    print("rows kept (read / kept): " + ", ".join(f"{k} {a}/{b}"
                                                   for k, (a, b) in LEAN_STATS.items()))
    md.write_text(text, encoding="utf-8")
    print(f"wrote {md}")
    _print_problems(problems)
    return 0


def _print_problems(problems: list[str]) -> None:
    if problems:
        print("NUMBERS THAT DO NOT REPRODUCE:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("every plotted number reproduces its report")


def splice_section(md: Path, lines: list[str], heading: str, problems: list[str]) -> None:
    """Replace the section ``heading`` of an existing figure_numbers.md (or insert it before
    'Numbers that do not reproduce'), leaving every other section as it is. Problems of the new
    section are listed under 'Numbers that do not reproduce', beside the earlier status of the
    figures this run did not recompute."""
    tail_head = "## Numbers that do not reproduce"
    old = md.read_text(encoding="utf-8").splitlines() if md.exists() else ["# Figure numbers", ""]
    kept, skip = [], False
    for line in old:
        if line.startswith("## "):
            skip = line.strip() == heading
        if not skip:
            kept.append(line)
    if tail_head in kept:
        i = kept.index(tail_head)
        body, tail = kept[:i], kept[i + 1:]
    else:
        body, tail = kept, []
    stem = heading[3:].split(".")[0]
    earlier = [t for t in tail if t.strip() and not t.startswith(f"- {stem}")
               and not t.startswith("Figures not recomputed")]
    if problems:
        new_tail = [f"- {p}" for p in problems] + [""] + (
            [f"Figures not recomputed in the run that wrote {heading[3:]} (`--only`), "
             "as written before:", ""] + earlier if earlier else [])
    else:
        new_tail = earlier or ["None."]
    md.write_text("\n".join(body + lines + [tail_head, ""] + new_tail) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
