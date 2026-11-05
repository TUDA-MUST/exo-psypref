"""Concept sketches for Fig. 1 of the ICORR 2027 paper: three directions to choose from, not final
art. Each is one IEEE column wide (3.5 in) and at most 6 cm tall.

  concept1_loop.pdf      The loop with two answerers. A wearer (photo placeholder) is asked "A or
                         B?"; the answer goes to Optimizer A, which holds a belief over the preferred
                         profile and returns the next most informative pair. Below, in a dashed box,
                         the simulated wearer of this paper stands in for the wearer: its
                         psychometric function with the lapse floor and ceiling.
  concept2_belief.pdf    A's belief narrowing: posterior draws after k = 0, 6 and 24 comparisons, the
                         preferred profile (dashed) and the pair asked from that belief.
  concept3_judgment.pdf  The answer as a perceptual judgment: two candidates, their RMS distances to
                         the hidden preferred profile, and the psychometric function that turns the
                         distance difference into P(choose A).
  contact_sheet.png      The three side by side, for choosing.

Data: one stored session, no new run. The particle snapshots of the median-room user
(loso-AB31/pop016, e-main's A session with EIG duels), written to out/icorr/video by
icorr_calibration_video.py. The fold's ProMP prior and the user's ideal are loaded with the
project's own loaders (phase5._prior, phase5.arm_space, phase5._users), and particles z* map to
curves along the learner's own route (LearnerSpace.to_theta, then design.curve: eight von Mises
bases, kappa = 9.106, 101 points, weights clipped to the box, curves to +-12 Nm). The script stops
using the data unless the loaded whitening, basis and ideal equal the arrays stored with the
snapshots. If anything fails to load, it draws schematic curves instead, says so on stdout, and
stamps "schematic curves" on every figure.

The pair drawn with the belief at snapshot k is the pair asked next from that belief (duel index
k). At k = 24 that pair is a lapse probe (A's probes, share 0.15), and it is labelled as one. The
simulated wearer is the main condition's (s = 1.0 Nm, lapse 0.1, the paper's wearer equation),
through paper_icorr_figures.psychometric. Style, colours and rcParams are paper_icorr_figures'.

Usage:
    python scripts/fig1_concepts.py [--out <project>/out]
        [--figdir paper/figures/fig1_concepts] [--png-dir <dir for the per-concept PNGs>]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

# The paper figures' rcParams (Arial 8 pt, thin grey axes), colours and wearer model.
from paper_icorr_figures import (BLUE, BLUE_LIGHT, CONTEXT, FIG_W, INK, LAPSE,  # noqa: E402
                                 MUTED, RULE, _spines, psychometric)

FOLD, USER = "loso-AB31", "loso-AB31/pop016"
SNAPSHOT = "median-room_loso-AB31_pop016_eig.npz"
KS = (0, 6, 24)          # concept 2's panels
K_PAIR = 6               # concepts 1 and 3: the belief after 6 comparisons and the pair it asked
S_MAIN = 1.0             # the main condition's sensitivity, Nm
N_FAN = 40               # posterior draws per fan
SEED = 0
MAX_H = 6.0 / 2.54       # at most about 6 cm tall
#: The brief's candidate labels. The paper writes tau_1, tau_2 and calls the optimizer "A":
#: change them here if the figure should follow the paper.
PAIR_LABELS = ("A", "B")
#: Tol high-contrast rose and gold. Neither is equiluminant with BLUE (L* 50 and 73 against 40;
#: the red #BD1F01 is, L* 41), and neither is the paper's orange, which means "A, projection".
PAIR_COLOURS = ("#BB5566", "#DDAA33")
PHOTO_GREY = "#E6E6E6"
SMALL = 7                # pt, secondary labels inside the sketches
TAU_DASH = (0, (3.0, 1.6))
YLIM = (-14.5, 14.5)     # fig_session's torque range


# ---------------------------------------------------------------- data
def _rms(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _draws(weights: np.ndarray, n: int, rng) -> np.ndarray:
    """Systematic resampling: n particle indices in proportion to their weights."""
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(weights), u), weights.size - 1)


def real_session(out: Path) -> dict:
    """Curves of the stored median-room session, through the fold's prior."""
    from prompref.experiment import phase5 as p5

    run = str(out / "phase5-schaefer")
    z = np.load(out / "icorr" / "video" / SNAPSHOT)
    space = p5.arm_space(p5._prior(run, FOLD, "promp_s0"), "tight")
    design, basis = space.design, space.design.basis
    ideal = np.asarray(p5._users(run, FOLD)[USER].curve_nm, dtype=float)
    checks = {"whitening mu": np.abs(space.whitening.mu - z["whitening_mu"]).max(),
              "whitening U": np.abs(space.whitening._u - z["whitening_u"]).max(),
              "whitening scale": np.abs(space.whitening._scale - z["whitening_scale"]).max(),
              "basis": np.abs(basis.design - z["basis"]).max(),
              "ideal": np.abs(ideal - z["ideal_curve"]).max()}
    bad = {k: float(v) for k, v in checks.items() if not v <= 1e-12}
    if bad:
        raise RuntimeError(f"loaded prior or user differs from the snapshot file: {bad}")
    if basis.n_basis != 8 or basis.design.shape[0] != 101 or abs(basis.kappa - 9.106) > 5e-4:
        raise RuntimeError(f"unexpected basis: K = {basis.n_basis}, kappa = {basis.kappa}, "
                           f"{basis.design.shape[0]} points")

    def curves(zs):
        return np.array([np.asarray(design.curve(space.to_theta(v)), dtype=float)
                         for v in np.atleast_2d(zs)])

    ks = [int(k) for k in z["ks"]]
    rng = np.random.default_rng(SEED)
    fan, mean, pair, err, spread = {}, {}, {}, {}, {}
    for k in sorted(set(KS) | {K_PAIR}):
        i = ks.index(k)
        w = np.asarray(z["weights"][i], dtype=float)
        fan[k] = curves(z["zstar"][i][_draws(w, N_FAN, rng)])
        mean[k] = curves(z["zstar_mean"][i])[0]
        if _rms(mean[k], z["delivered_curve"][i]) > 1e-9:
            raise RuntimeError(f"k = {k}: the posterior mean's curve is not the stored "
                               "identified assistance")
        every = curves(z["zstar"][i])
        centre = w @ every
        spread[k] = float(np.sqrt(w @ np.mean((every - centre) ** 2, axis=1)))
        ca, cb = curves(np.stack([z["duel_z_a"][k], z["duel_z_b"][k]]))
        pair[k] = dict(a=ca, b=cb, kind=str(z["duel_kind"][k]),
                       prefers_a=bool(z["duel_prefers_a"][k]), d_a=_rms(ca, ideal),
                       d_b=_rms(cb, ideal))
        err[k] = _rms(mean[k], ideal)
    # sanity: the stored answers follow the closer curve about as often as the wearer model says
    n = len(z["duel_prefers_a"])
    closer = [_rms(*curves(z["duel_z_a"][j]), ideal) < _rms(*curves(z["duel_z_b"][j]), ideal)
              for j in range(n)]
    agree = float(np.mean(np.array(closer) == np.asarray(z["duel_prefers_a"], dtype=bool)))
    return dict(real=True, source=f"{SNAPSHOT} (stored e-main session, {USER})",
                phase=np.asarray(z["phase"], dtype=float), ideal=ideal, fan=fan, mean=mean,
                pair=pair, err=err, spread=spread, agree=agree, n_duels=n,
                room=float(z["d_prior"] - z["d_floor"]))


def schematic_session() -> dict:
    """Stand-in curves of the same shape, used only if the stored session cannot be loaded."""
    phase = np.linspace(0.0, 100.0, 101)
    x = phase / 100.0
    ideal = (-7.5 * np.sin(2 * np.pi * (x - 0.02)) + 2.0 * np.sin(4 * np.pi * (x + 0.1))).clip(-12, 12)
    generic = 0.7 * ideal + 1.2 * np.cos(2 * np.pi * x)
    rng = np.random.default_rng(SEED)

    def wiggle(scale, n):
        a = rng.normal(size=(n, 4)) * scale
        return (a[:, :1] * np.sin(2 * np.pi * x) + a[:, 1:2] * np.cos(2 * np.pi * x)
                + 0.6 * a[:, 2:3] * np.sin(4 * np.pi * x) + 0.6 * a[:, 3:4] * np.cos(4 * np.pi * x))

    fan, mean, pair, err, spread = {}, {}, {}, {}, {}
    for k, share, sd in ((0, 0.0, 3.0), (6, 0.6, 2.2), (24, 0.85, 1.4)):
        mean[k] = (1 - share) * generic + share * ideal
        fan[k] = (mean[k] + wiggle(sd, N_FAN)).clip(-12, 12)
        d = wiggle(1.3 * sd, 1)[0]
        ca, cb = (mean[k] + d).clip(-12, 12), (mean[k] - d).clip(-12, 12)
        pair[k] = dict(a=ca, b=cb, kind="acq", prefers_a=_rms(ca, ideal) < _rms(cb, ideal),
                       d_a=_rms(ca, ideal), d_b=_rms(cb, ideal))
        err[k], spread[k] = _rms(mean[k], ideal), sd
    return dict(real=False, source="schematic curves (stored session not loaded)", phase=phase,
                ideal=ideal, fan=fan, mean=mean, pair=pair, err=err, spread=spread,
                agree=float("nan"), n_duels=0, room=float("nan"))


# ---------------------------------------------------------------- drawing helpers
class Canvas:
    """A figure with an overlay in inches, for boxes and arrows around inset axes."""

    def __init__(self, h: float):
        if h > MAX_H + 1e-9:
            raise ValueError(f"{h:.2f} in is taller than {MAX_H:.2f} in (6 cm)")
        self.w, self.h = FIG_W, h
        self.fig = plt.figure(figsize=(FIG_W, h))
        self.ov = self.fig.add_axes([0, 0, 1, 1], zorder=0)
        self.ov.set_xlim(0, FIG_W)
        self.ov.set_ylim(0, h)
        self.ov.axis("off")

    def axes(self, x, y, w, h, zorder=2):
        ax = self.fig.add_axes([x / self.w, y / self.h, w / self.w, h / self.h], zorder=zorder)
        ax.set_facecolor("none")
        return ax

    def box(self, x, y, w, h, r=0.06, **kw):
        style = dict(facecolor="white", edgecolor=MUTED, lw=0.7, zorder=1)
        style.update(kw)
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **style)
        self.ov.add_patch(p)
        return p

    def text(self, x, y, s, **kw):
        style = dict(fontsize=SMALL, color=INK, ha="left", va="center")
        style.update(kw)
        return self.ov.text(x, y, s, **style)

    def arrow(self, xy_from, xy_to, colour=MUTED, lw=0.8, rad=0.0):
        self.ov.annotate("", xy=xy_to, xytext=xy_from,
                         arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw, shrinkA=0,
                                         shrinkB=0, mutation_scale=7,
                                         connectionstyle=f"arc3,rad={rad}"), zorder=3)

    def photo(self, x, y, w, h, label, rotate=False):
        self.ov.add_patch(Rectangle((x, y), w, h, facecolor=PHOTO_GREY, edgecolor="none",
                                    zorder=1))
        self.text(x + w / 2, y + h / 2, label, ha="center", color=MUTED, linespacing=1.15,
                  rotation=90 if rotate else 0, zorder=2)

    def bubble(self, x, y, w, h, tail_base_y, tail_tip):
        """A speech bubble whose tail leaves its left edge at tail_base_y."""
        self.box(x, y, w, h, r=0.08, edgecolor=MUTED, lw=0.7, zorder=1)
        b0, b1 = (x, tail_base_y - 0.05), (x, tail_base_y + 0.05)
        self.ov.add_patch(Polygon([(x + 0.012, b0[1]), tail_tip, (x + 0.012, b1[1])],
                                  closed=True, facecolor="white", edgecolor="none", zorder=1.5))
        self.ov.plot([b0[0], tail_tip[0], b1[0]], [b0[1], tail_tip[1], b1[1]], color=MUTED,
                     lw=0.7, zorder=1.6, solid_joinstyle="miter")

    def stamp(self, session):
        if not session["real"]:
            self.text(self.w - 0.03, 0.03, "schematic curves", ha="right", va="bottom",
                      color="#C00000", fontsize=SMALL, zorder=5)

    def save(self, path: Path, png_dir: Path):
        self.fig.savefig(path, format="pdf", metadata={"CreationDate": None})
        png = png_dir / f"{path.stem}.png"
        self.fig.savefig(png, dpi=300)
        plt.close(self.fig)
        return png


def _bare(ax, x=(0, 100), y=YLIM, zero=True):
    ax.set_xlim(*x)
    ax.set_ylim(*y)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CONTEXT)
        ax.spines[side].set_linewidth(0.6)
    if zero:
        ax.axhline(0.0, color=RULE, lw=0.5, zorder=0)


def _fan(ax, s, k, mean=True, lw=0.45, alpha=0.45):
    x = s["phase"]
    for c in s["fan"][k]:
        ax.plot(x, c, color=BLUE_LIGHT, lw=lw, alpha=alpha, zorder=1, solid_capstyle="round")
    if mean:
        ax.plot(x, s["mean"][k], color=BLUE, lw=1.2, zorder=3)


def _pair(ax, s, k, lw=1.2, labels=True, fontsize=SMALL):
    x, p = s["phase"], s["pair"][k]
    for c, colour, lab, dy in ((p["a"], PAIR_COLOURS[0], PAIR_LABELS[0], 0),
                               (p["b"], PAIR_COLOURS[1], PAIR_LABELS[1], 0)):
        ax.plot(x, c, color=colour, lw=lw, zorder=4, solid_capstyle="round")
        if labels:
            ax.text(101.5, c[-1] + dy, lab, color=INK, fontsize=fontsize, ha="left",
                    va="center", fontweight="bold", clip_on=False)


def _tau(ax, s, lw=1.0):
    ax.plot(s["phase"], s["ideal"], color=INK, lw=lw, ls=TAU_DASH, zorder=5)


def _psychometric(ax, mark=None, right=False, ticks=True):
    """P(choose A) against d(B) - d(A) for the main wearer, with the lapse floor and ceiling."""
    xmax = 2.6
    delta = np.linspace(-xmax, xmax, 261)
    for p in (LAPSE / 2, 1 - LAPSE / 2):
        ax.axhline(p, color=CONTEXT, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
    ax.axvline(0.0, color=RULE, lw=0.5, zorder=0)
    ax.plot(delta, psychometric(delta, S_MAIN, LAPSE), color=INK, lw=1.3, zorder=3)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([LAPSE / 2, 0.5, 1 - LAPSE / 2])
    ax.set_yticklabels(["λ/2", "0.5", "1 − λ/2"], fontsize=SMALL)
    if ticks:
        ax.set_xticks([-2, 0, 2])
        ax.tick_params(axis="x", labelsize=SMALL, pad=1.5)
    else:
        ax.set_xticks([])
    ax.tick_params(axis="y", pad=1.5)
    if right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
    else:
        _spines(ax)
    if mark is not None:
        dx, colour = mark
        p = float(psychometric(dx, S_MAIN, LAPSE))
        ax.plot([dx, dx], [-0.02, p], color=colour, lw=0.7, ls=(0, (1, 1)), zorder=4)
        ax.plot([-xmax, dx], [p, p], color=colour, lw=0.7, ls=(0, (1, 1)), zorder=4)
        ax.plot([dx], [p], marker="o", ms=3.4, color=colour, mec="white", mew=0.5, zorder=5)
        return p
    return None


def _legend_row(canvas, x, y, items, gap=0.12):
    """Hand-placed legend entries in one row; items are (handle kwargs, label)."""
    for kw, label in items:
        canvas.ov.plot([x, x + 0.16], [y, y], zorder=3, solid_capstyle="butt", **kw)
        t = canvas.text(x + 0.20, y, label)
        canvas.fig.canvas.draw()
        bb = t.get_window_extent().transformed(canvas.ov.transData.inverted())
        x = bb.x1 + gap
    return x


# ---------------------------------------------------------------- concept 1
def concept1(s: dict, figdir: Path, png_dir: Path):
    """The loop with two answerers."""
    c = Canvas(2.30)
    W = c.w
    k, p = K_PAIR, s["pair"][K_PAIR]
    # wearer
    px, py, pw, ph = 0.04, 1.06, 0.72, 1.20
    c.photo(px, py, pw, ph, "photo:\nwearer in\nhip exo-\nskeleton")
    # speech bubble with the pair
    bx, by, bw, bh = 0.86, 1.52, 0.88, 0.74
    c.bubble(bx, by, bw, bh, tail_base_y=by + 0.50, tail_tip=(px + pw - 0.06, py + ph - 0.18))
    c.text(bx + bw / 2, by + bh - 0.11, f"{PAIR_LABELS[0]} or {PAIR_LABELS[1]}?", ha="center",
           fontsize=8, fontweight="bold")
    for j, (curve, colour, lab) in enumerate(((p["a"], PAIR_COLOURS[0], PAIR_LABELS[0]),
                                             (p["b"], PAIR_COLOURS[1], PAIR_LABELS[1]))):
        ax = c.axes(bx + 0.06 + j * 0.40, by + 0.07, 0.36, 0.42)
        _bare(ax)
        ax.plot(s["phase"], curve, color=colour, lw=1.3, solid_capstyle="round")
        ax.text(3, 13.5, lab, color=INK, fontsize=SMALL, fontweight="bold", ha="left", va="top")
    # optimizer A
    ox, oy, ow, oh = 2.10, 1.06, 1.36, 1.20
    c.box(ox, oy, ow, oh, r=0.07, edgecolor=BLUE, lw=0.9)
    c.text(ox + 0.08, oy + oh - 0.11, "Optimizer A", fontsize=8, fontweight="bold", color=BLUE)
    c.text(ox + 0.08, oy + oh - 0.27, "belief over the\npreferred profile", va="center",
           linespacing=1.1)
    ax = c.axes(ox + 0.08, oy + 0.08, ow - 0.16, 0.66)
    _bare(ax)
    _fan(ax, s, k, lw=0.45, alpha=0.5)
    ax.text(98, -13.5, f"after {k} answers", ha="right", va="bottom", color=MUTED,
            fontsize=SMALL)
    # the loop
    ya = by + 0.40
    c.arrow((bx + bw + 0.02, ya), (ox - 0.02, ya))
    c.text((bx + bw + ox) / 2, ya + 0.09, "answer", ha="center", va="bottom")
    yr = py + 0.22
    c.arrow((ox - 0.02, yr), (px + pw + 0.02, yr))
    c.text((px + pw + ox) / 2, yr + 0.06, "next most informative pair", ha="center",
           va="bottom")
    # this paper: the simulated wearer
    dx0, dy0, dw, dh = 0.04, 0.03, W - 0.08, 0.84
    c.box(dx0, dy0, dw, dh, r=0.07, edgecolor=MUTED, lw=0.8, ls=(0, (3, 2)))
    c.text(dx0 + 0.08, dy0 + dh - 0.11, "This paper: simulated wearer", fontsize=8,
           fontweight="bold")
    c.text(dx0 + 0.08, dy0 + 0.36, "answers with a\npsychometric function\nof distance to its\n"
           "preferred profile", va="center", linespacing=1.1)
    c.text(dx0 + 2.66, dy0 + 0.36, "200 simulated\nwearers from\nwalking data", va="center",
           linespacing=1.1)
    ax = c.axes(1.86, 0.30, 0.72, 0.33)
    _psychometric(ax)
    ax.set_xlabel(f"d({PAIR_LABELS[1]}) − d({PAIR_LABELS[0]})  (Nm)", fontsize=SMALL,
                  labelpad=1)
    c.text(1.86 + 0.36, 0.66, f"P(choose {PAIR_LABELS[0]})", ha="center", va="bottom")
    # the stand-in arrow
    c.arrow((px + pw / 2, dy0 + dh + 0.01), (px + pw / 2, py - 0.01), colour=MUTED)
    c.text(px + pw / 2 + 0.06, (dy0 + dh + py) / 2, "stands in for the wearer", va="center",
           color=MUTED)
    c.stamp(s)
    path = figdir / "concept1_loop.pdf"
    return path, c.save(path, png_dir)


# ---------------------------------------------------------------- concept 2
def concept2(s: dict, figdir: Path, png_dir: Path):
    """A's belief narrowing toward the preferred profile."""
    c = Canvas(1.88)
    # wearer
    c.photo(0.04, 0.34, 0.36, 1.06, "photo: wearer\n(simulated here)", rotate=True)
    # panels
    x0, gap, y0, h = 0.82, 0.08, 0.34, 1.06
    w = (FIG_W - 0.12 - x0 - 2 * gap) / 3
    axes = []
    for j, k in enumerate(KS):
        ax = c.axes(x0 + j * (w + gap), y0, w, h)
        ax.axhline(0.0, color=RULE, lw=0.5, zorder=0)
        _fan(ax, s, k)
        _pair(ax, s, k, lw=1.1, labels=False)
        _tau(ax, s)
        ax.set_xlim(0, 100)
        ax.set_ylim(*YLIM)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0", "50", "100" if j == len(KS) - 1 else ""])
        ax.set_yticks([-12, 0, 12])
        ax.tick_params(labelsize=SMALL, pad=1.5)
        if j:
            ax.set_yticklabels([])
        _spines(ax)
        ax.set_title(f"k = {k}", fontsize=8, pad=2, color=INK)
        if s["pair"][k]["kind"] == "probe":
            ax.text(97, 13.8, "pair: probe", ha="right", va="top", color=MUTED,
                    fontsize=SMALL)
        axes.append(ax)
    axes[0].set_ylabel("hip torque (Nm)", fontsize=SMALL, labelpad=1)
    c.text(x0 + 1.5 * w + gap, 0.05, "gait cycle (%)", ha="center", va="bottom")
    # comparisons run left to right
    yt = y0 + h + 0.10
    for j in range(2):
        xa = x0 + j * (w + gap) + w / 2 + 0.24
        xb = x0 + (j + 1) * (w + gap) + w / 2 - 0.24
        c.arrow((xa, yt), (xb, yt), colour=CONTEXT, lw=0.7)
    # legend, two rows
    _legend_row(c, 0.06, 1.80,
                [(dict(color=BLUE_LIGHT, lw=1.6), "A's belief (posterior draws)"),
                 (dict(color=BLUE, lw=1.2), "its estimate"),
                 (dict(color=INK, lw=1.0, ls=TAU_DASH), "preferred τ* (hidden)")], gap=0.12)
    x = _legend_row(c, 0.06, 1.66, [(dict(color=PAIR_COLOURS[0], lw=1.2), PAIR_LABELS[0]),
                                    (dict(color=PAIR_COLOURS[1], lw=1.2), PAIR_LABELS[1])],
                    gap=0.06)
    c.text(x - 0.06, 1.66, ": the pair asked next")
    c.text(FIG_W - 0.06, 1.66, "k: comparisons answered", ha="right", color=MUTED)
    c.stamp(s)
    path = figdir / "concept2_belief.pdf"
    return path, c.save(path, png_dir)


# ---------------------------------------------------------------- concept 3
def concept3(s: dict, figdir: Path, png_dir: Path):
    """The answer as a noisy perceptual judgment of distance."""
    c = Canvas(1.72)
    p = s["pair"][K_PAIR]
    delta = p["d_b"] - p["d_a"]
    # wearer
    c.photo(0.04, 0.36, 0.34, 1.10, "photo: wearer", rotate=True)
    # the simulated wearer
    bx0, by0 = 0.46, 0.03
    c.box(bx0, by0, FIG_W - 0.04 - bx0, 1.66, r=0.07, edgecolor=MUTED, lw=0.8, ls=(0, (3, 2)))
    c.text(bx0 + 0.08, 1.57, "This paper: simulated wearer", fontsize=8, fontweight="bold")
    c.arrow((0.38 + 0.01, 0.91), (bx0 + 0.07, 0.91), colour=MUTED)
    # candidates against the hidden preferred profile
    ax = c.axes(0.92, 0.38, 1.16, 1.02)
    x = s["phase"]
    ax.axhline(0.0, color=RULE, lw=0.5, zorder=0)
    for curve, colour in ((p["a"], PAIR_COLOURS[0]), (p["b"], PAIR_COLOURS[1])):
        ax.fill_between(x, curve, s["ideal"], color=colour, alpha=0.18, lw=0, zorder=1)
    _pair(ax, s, K_PAIR, lw=1.2, labels=False)
    _tau(ax, s)
    ax.set_xlim(0, 100)
    ax.set_ylim(-14.5, 21.0)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([-12, 0, 12])
    ax.tick_params(labelsize=SMALL, pad=1.5)
    ax.set_xlabel("gait cycle (%)", fontsize=SMALL, labelpad=1)
    ax.set_ylabel("hip torque (Nm)", fontsize=SMALL, labelpad=1)
    _spines(ax)
    # legend in the empty top of the torque axes
    handles = [Line2D([], [], color=PAIR_COLOURS[0], lw=1.4),
               Line2D([], [], color=PAIR_COLOURS[1], lw=1.4),
               Line2D([], [], color=INK, lw=1.0, ls=TAU_DASH)]
    labels = [f"d({PAIR_LABELS[0]}) = {p['d_a']:.2f} Nm", f"d({PAIR_LABELS[1]}) = {p['d_b']:.2f} Nm",
              "preferred τ* (hidden)"]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.04), frameon=False,
              fontsize=SMALL, handlelength=1.2, handletextpad=0.35, labelspacing=0.12,
              borderaxespad=0.0)
    # distance difference to choice probability
    c.arrow((2.14, 0.84), (2.36, 0.84), colour=MUTED)
    bx = c.axes(2.44, 0.38, 0.58, 0.92)
    p_a = _psychometric(bx, mark=(delta, PAIR_COLOURS[0]), right=True)
    bx.set_xlabel(f"d({PAIR_LABELS[1]}) − d({PAIR_LABELS[0]})  (Nm)", fontsize=SMALL,
                  labelpad=1)
    bx.text(-2.75, p_a, f"{p_a:.2f}", ha="right", va="center", fontsize=SMALL,
            color=PAIR_COLOURS[0], fontweight="bold", clip_on=False)
    c.text(2.44 + 0.29, 1.33, f"P(choose {PAIR_LABELS[0]})", ha="center", va="bottom")
    c.stamp(s)
    path = figdir / "concept3_judgment.pdf"
    return path, c.save(path, png_dir), dict(delta=delta, p_a=p_a)


# ---------------------------------------------------------------- contact sheet
def contact_sheet(pngs: list[Path], titles: list[str], path: Path):
    imgs = [plt.imread(str(p)) for p in pngs]
    hmax = max(i.shape[0] for i in imgs)
    wsum = sum(i.shape[1] for i in imgs)
    scale = 1 / 300                      # inches per pixel at the 300 dpi the PNGs were saved
    gap, top = 0.35, 0.40
    fig = plt.figure(figsize=(wsum * scale + gap * (len(imgs) + 1), hmax * scale + top + 0.15))
    W, H = fig.get_size_inches()
    x = gap
    for img, title in zip(imgs, titles):
        h, w = img.shape[0] * scale, img.shape[1] * scale
        ax = fig.add_axes([x / W, (H - top - h) / H, w / W, h / H])
        ax.imshow(img, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(RULE)
            sp.set_linewidth(0.6)
        fig.text((x + w / 2) / W, (H - 0.12) / H, title, ha="center", va="top", fontsize=9,
                 fontweight="bold", color=INK)
        x += w + gap
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--figdir", type=Path, default=ROOT / "paper" / "figures" / "fig1_concepts")
    ap.add_argument("--png-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    args.figdir.mkdir(parents=True, exist_ok=True)
    png_dir = args.png_dir or Path(tempfile.mkdtemp(prefix="fig1_concepts_"))
    png_dir.mkdir(parents=True, exist_ok=True)
    try:
        s = real_session(args.out)
    except Exception as exc:  # noqa: BLE001 -- any failure falls back, and says so
        print(f"REAL DATA NOT USED ({type(exc).__name__}: {exc}); drawing schematic curves")
        s = schematic_session()
    print(f"source: {s['source']}")
    for k in sorted(s["fan"]):
        q = s["pair"][k]
        print(f"k = {k:2d}: belief spread {s['spread'][k]:.2f} Nm, identified error "
              f"{s['err'][k]:.2f} Nm; pair asked next ({q['kind']}): d(A) {q['d_a']:.2f}, "
              f"d(B) {q['d_b']:.2f} Nm, wearer chose {'A' if q['prefers_a'] else 'B'}")
    if s["real"]:
        print(f"room {s['room']:.3f} Nm; answers that chose the closer curve: "
              f"{s['agree']:.2f} of {s['n_duels']}")
    p1, g1 = concept1(s, args.figdir, png_dir)
    p2, g2 = concept2(s, args.figdir, png_dir)
    p3, g3, n3 = concept3(s, args.figdir, png_dir)
    print(f"concept 3: d(B) - d(A) = {n3['delta']:.3f} Nm, P(choose A) = {n3['p_a']:.3f}")
    sheet = args.figdir / "contact_sheet.png"
    contact_sheet([g1, g2, g3], ["1  Loop with two answerers", "2  Belief narrowing",
                                 "3  Answer as a perceptual judgment"], sheet)
    for path in (p1, p2, p3, sheet):
        print(f"wrote {path}")
    print(f"per-concept PNGs in {png_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
