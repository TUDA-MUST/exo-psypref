"""Fig. 1 of the ICORR 2027 paper: the method as a block diagram, one IEEE column wide (3.5 in)
and at most 2.6 in tall. Illustrative; it prints no result numbers.

  Left    Walking data of other people -> the population prior over whole torque profiles
          (a fan of prior draws, generic assistance as its mean).
  Middle  The loop: PsyPref's belief over the preferred profile (k = 0, 6, 16) -> ask the most
          informative pair -> user: profile 1 or 2? -> the answer updates the belief.
  Right   After k comparisons: the identified profile -> validation against generic assistance.
  Bottom  This paper: simulated users, built from walking data, answer like a psychometric
          function with a lapse best-profile distance and best-profile level; their spread of preferences is matched to
          real participants.

Two variants: fig1_method.pdf (a photo placeholder for the user, to be replaced by the author)
and fig1_method_silhouette.pdf (a line-drawn walker with a hip exoskeleton instead). Each gets a
PNG preview next to it.

Data: every curve is real, from one stored session and no new run: the particle snapshots of
the median-room user loso-AB31/pop016 (e-main, PsyPref with informative pairs) in
out/icorr/video, mapped to curves through the fold's ProMP prior by fig1_concepts.real_session,
which checks the loaded normalization, basis and ideal against the arrays stored with the snapshots.
The k = 0 particles are draws from the population prior (z* ~ N(0, I)), so the left fan and the
first belief inset show the same prior. The pair is the one asked after six answers (an
informative pair; the user chose the closer profile 1). The identified profile is the posterior
mean after 16 comparisons. Generic assistance is the stored prior mean curve; the script checks
that its distance to the ideal is the stored d_prior. If the session cannot be loaded, the
script stops: this figure has no schematic fallback.

The sigmoid is the main condition's answer model (s = 1.0 Nm, lapse 0.1, Eq. simulated user) through
paper_icorr_figures.psychometric. Style, colours and rcParams are paper_icorr_figures';
preferred profile solid black, generic assistance dashed grey as in fig_session.

Usage:
    python scripts/fig1_method.py [--out <project>/out]
        [--figdir paper/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import fig1_concepts as fc  # noqa: E402  (also sets the Agg backend and the paper's rcParams)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
from paper_icorr_figures import (BLUE, BLUE_LIGHT, FIG_W, INK, LAPSE, MUTED, RULE,  # noqa: E402
                                 psychometric)

H = 2.60                 # figure height, in
KS = (0, 6, 16)          # the belief inset's snapshots
K_PAIR = 6               # the pair shown to the user: asked after six answers
K_ID = 16                # the identified profile shown on the right
N_FAN = 30               # posterior draws per fan
S_MAIN = 1.0             # the main condition's sensitivity, Nm
GENERIC = "#808080"      # fig_session's generic assistance
GENERIC_DASH = (0, (4, 1.6))
PRIOR_FAN = "#C4C4C4"
PAIR_COLOURS = fc.PAIR_COLOURS
BODY_FRONT, BODY_BACK = "#9C9C9C", "#C9C9C9"
FS, FS_HEAD = 7, 8       # pt: body text, the few headings
YLIM = (-13.5, 13.5)
VARIANTS = (("photo", "fig1_method"), ("silhouette", "fig1_method_silhouette"))


# ---------------------------------------------------------------- data
def session(out: Path) -> dict:
    """The stored median-room session through fig1_concepts' checked loader, plus generic."""
    # real_session draws its fans and pairs at fig1_concepts' module settings
    fc.KS, fc.K_PAIR, fc.N_FAN = KS, K_PAIR, N_FAN
    s = fc.real_session(out)
    z = np.load(out / "icorr" / "video" / fc.SNAPSHOT)
    generic = np.asarray(z["generic_curve"], dtype=float)
    d_gen = fc._rms(generic, s["ideal"])
    if abs(d_gen - float(z["d_prior"])) > 1e-9:
        raise RuntimeError(f"generic assistance is {d_gen:.6f} Nm from the ideal, the stored "
                           f"d_prior is {float(z['d_prior']):.6f} Nm")
    s["generic"] = generic
    return s


# ---------------------------------------------------------------- drawing
def mini(c, x, y, w, h):
    """Small bare axes in figure inches: gait cycle on x, torque on y, a zero line."""
    ax = c.axes(x, y, w, h)
    fc._bare(ax, y=YLIM)
    return ax


def fan(ax, s, k, colour=BLUE_LIGHT, lw=0.4, alpha=0.55):
    for curve in s["fan"][k]:
        ax.plot(s["phase"], curve, color=colour, lw=lw, alpha=alpha, zorder=1,
                solid_capstyle="round")


def preferred(ax, s, lw=0.9):
    ax.plot(s["phase"], s["ideal"], color=INK, lw=lw, zorder=5, solid_capstyle="round")


def generic(ax, s, lw=1.0):
    ax.plot(s["phase"], s["generic"], color=GENERIC, lw=lw, ls=GENERIC_DASH, zorder=4)


def estimate(ax, curve, s, lw=1.2):
    ax.plot(s["phase"], curve, color=BLUE, lw=lw, zorder=4, solid_capstyle="round")


def walker(c, x, y, h, exo=True):
    """A line-drawn walker stepping right, h tall, its hip at x; y is the ground."""
    pt = 72.0 * h          # points per unit height
    P = dict(head=(0.035, 0.905), neck=(0.02, 0.81), hip=(-0.015, 0.50),
             knee_f=(0.095, 0.27), ankle_f=(0.145, 0.035), toe_f=(0.225, 0.02),
             knee_b=(-0.075, 0.27), ankle_b=(-0.195, 0.075), toe_b=(-0.125, 0.0),
             elbow_f=(0.085, 0.615), hand_f=(0.150, 0.505),
             elbow_b=(-0.075, 0.625), hand_b=(-0.115, 0.515))

    def xy(*names):
        pts = np.array([P[n] for n in names])
        return x + pts[:, 0] * h, y + pts[:, 1] * h

    def limb(names, colour, lw):
        c.ov.plot(*xy(*names), color=colour, lw=lw * pt, solid_capstyle="round",
                  solid_joinstyle="round", zorder=6)

    shoulder = (0.02, 0.765)
    P["shoulder"] = shoulder
    limb(("shoulder", "elbow_b", "hand_b"), BODY_BACK, 0.065)
    limb(("hip", "knee_b", "ankle_b", "toe_b"), BODY_BACK, 0.080)
    limb(("neck", "hip"), BODY_FRONT, 0.105)
    limb(("hip", "knee_f", "ankle_f", "toe_f"), BODY_FRONT, 0.080)
    limb(("shoulder", "elbow_f", "hand_f"), BODY_FRONT, 0.065)
    hx, hy = xy("head")
    c.ov.add_patch(Circle((hx[0], hy[0]), 0.072 * h, facecolor=BODY_FRONT, edgecolor="none",
                          zorder=6))
    if not exo:
        return
    # hip exoskeleton: a waist belt with a back unit, the hip actuator, a thigh strut and cuff
    hipx, hipy = x + P["hip"][0] * h, y + P["hip"][1] * h
    c.ov.plot([hipx - 0.085 * h, hipx + 0.075 * h], [hipy + 0.050 * h, hipy + 0.062 * h],
              color=INK, lw=0.035 * pt, solid_capstyle="butt", zorder=7)
    c.ov.add_patch(FancyBboxPatch((hipx - 0.135 * h, hipy + 0.015 * h), 0.050 * h, 0.120 * h,
                                  boxstyle=f"round,pad=0,rounding_size={0.012 * h}",
                                  facecolor=INK, edgecolor="none", zorder=7))
    kx, ky = x + P["knee_f"][0] * h, y + P["knee_f"][1] * h
    d = np.array([kx - hipx, ky - hipy])
    u = d / np.linalg.norm(d)
    nrm = np.array([u[1], -u[0]]) if u[1] < 0 else np.array([-u[1], u[0]])
    if nrm[0] < 0:
        nrm = -nrm                      # the strut runs along the front of the thigh
    a = np.array([hipx, hipy]) + 0.025 * h * nrm
    b = np.array([hipx, hipy]) + 0.70 * d + 0.025 * h * nrm
    c.ov.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=0.030 * pt, solid_capstyle="round",
              zorder=7)
    m = np.array([hipx, hipy]) + 0.66 * d
    e0, e1 = m - 0.060 * h * nrm, m + 0.050 * h * nrm
    c.ov.plot([e0[0], e1[0]], [e0[1], e1[1]], color=INK, lw=0.050 * pt, solid_capstyle="butt",
              zorder=7)
    c.ov.add_patch(Circle((hipx, hipy), 0.052 * h, facecolor="white", edgecolor=INK,
                          lw=0.020 * pt, zorder=8))
    c.ov.add_patch(Circle((hipx, hipy), 0.016 * h, facecolor=INK, edgecolor="none", zorder=8))


def legend_row(c, y, items, gap=0.16):
    """One centred row of hand-placed legend entries; items are (line kwargs, label)."""
    seg, pad = 0.18, 0.05
    widths = []
    r = c.fig.canvas.get_renderer()
    for _, label in items:
        t = c.text(0, -1, label, fontsize=FS)
        widths.append(t.get_window_extent(r).width / c.fig.dpi)
        t.remove()
    total = sum(seg + pad + w for w in widths) + gap * (len(items) - 1)
    x = (c.w - total) / 2
    for (kw, label), w in zip(items, widths):
        c.ov.plot([x, x + seg], [y, y], solid_capstyle="butt", zorder=3, **kw)
        c.text(x + seg + pad, y, label, fontsize=FS)
        x += seg + pad + w + gap


def sigmoid(c, x, y, w, h):
    """P(choose 1) against d(2) - d(1) for the main condition, lapse best-profile distance and best-profile level marked."""
    ax = c.axes(x, y, w, h)
    xmax = 2.8
    delta = np.linspace(-xmax, xmax, 281)
    for p in (LAPSE / 2, 1 - LAPSE / 2):
        ax.axhline(p, color=fc.CONTEXT, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
    ax.axvline(0.0, color=RULE, lw=0.5, zorder=0)
    ax.plot(delta, psychometric(delta, S_MAIN, LAPSE), color=INK, lw=1.2, zorder=3)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0])
    ax.set_xticklabels([])
    ax.set_yticks([LAPSE / 2, 0.5, 1 - LAPSE / 2])
    ax.set_yticklabels(["λ/2", "0.5", "1 − λ/2"])
    ax.yaxis.tick_right()
    ax.tick_params(labelsize=FS, pad=1.5, length=2.0)
    for side in ("top", "left"):
        ax.spines[side].set_visible(False)
    return ax


# ---------------------------------------------------------------- the figure
def figure(s: dict, variant: str, path: Path) -> Path:
    fc.MAX_H = H                         # fig1_concepts' Canvas caps the height at its own MAX_H
    c = fc.Canvas(H)
    box = dict(r=0.05, edgecolor=MUTED, lw=0.6)

    # columns (in): left, middle (the loop), right; top region and the band below
    LX0, LX1 = 0.03, 0.77
    MX0, MX1 = 0.86, 2.58
    RX0, RX1 = 2.67, 3.47
    TY0, TY1 = 0.80, 2.44
    lc, rc = (LX0 + LX1) / 2, (RX0 + RX1) / 2

    # legend
    legend_row(c, 2.535, [(dict(color=BLUE, lw=1.2), "PsyPref"),
                          (dict(color=INK, lw=0.9), "preferred profile"),
                          (dict(color=GENERIC, lw=1.0, ls=GENERIC_DASH), "generic assistance")])

    # ---- left: walking data -> population prior
    l1y0 = 2.02
    c.box(LX0, l1y0, LX1 - LX0, TY1 - l1y0, **box)
    for j in range(3):
        walker(c, LX0 + 0.20 + 0.18 * j, 2.265, 0.14, exo=False)
    c.text(lc, 2.135, "Walking data of\nother people", ha="center", linespacing=1.1)
    l2y1 = 1.93
    c.arrow((lc, l1y0 - 0.005), (lc, l2y1 + 0.005))
    c.box(LX0, TY0, LX1 - LX0, l2y1 - TY0, **box)
    c.text(lc, 1.715, "Population prior\nover whole\ntorque profiles", ha="center",
           linespacing=1.1)
    ax = mini(c, LX0 + 0.15, 1.02, 0.54, 0.42)
    fan(ax, s, 0, colour=PRIOR_FAN, lw=0.4, alpha=0.9)
    generic(ax, s)
    c.text(LX0 + 0.15 + 0.27, 0.925, "gait cycle", ha="center")
    c.text(LX0 + 0.075, 1.02 + 0.21, "hip torque", ha="center", rotation=90)
    c.arrow((LX1 + 0.005, 1.89), (MX0 - 0.005, 1.89))

    # ---- middle top: PsyPref's belief
    m1y0 = 1.74
    c.box(MX0, m1y0, MX1 - MX0, TY1 - m1y0, r=0.05, edgecolor=BLUE, lw=0.9)
    c.text(MX0 + 0.07, 2.355, "PsyPref:", fontsize=FS_HEAD, fontweight="bold", color=BLUE)
    c.text(MX0 + 0.07, 2.235, "belief over the preferred profile")
    pw, pg = 0.46, 0.115
    px0 = (MX0 + MX1) / 2 - (3 * pw + 2 * pg) / 2
    for j, k in enumerate(KS):
        x = px0 + j * (pw + pg)
        ax = mini(c, x, 1.88, pw, 0.26)
        fan(ax, s, k)
        estimate(ax, s["mean"][k], s, lw=1.0)
        preferred(ax, s, lw=0.8)
        c.text(x + pw / 2, 1.81, f"k = {k}", ha="center")
        if j:
            c.arrow((x - pg + 0.025, 2.01), (x - 0.025, 2.01), colour=fc.CONTEXT, lw=0.7)

    # ---- middle bottom: the user
    m2y1 = 1.40
    c.box(MX0, TY0, MX1 - MX0, m2y1 - TY0, **box)
    phx, phy, phw, phh = MX0 + 0.06, TY0 + 0.05, 0.44, m2y1 - TY0 - 0.10
    if variant == "photo":
        c.photo(phx, phy, phw, phh, "photo:\nuser in\nhip exo-\nskeleton")
    else:
        walker(c, phx + phw / 2 - 0.02, phy + 0.02, phh - 0.04, exo=True)
    qx = phx + phw + 0.08
    c.text(qx, 1.31, "User: profile 1 or 2?", fontweight="bold")
    p = s["pair"][K_PAIR]
    qw, qg = 0.48, 0.10
    for j, (curve, colour, lab) in enumerate(((p["a"], PAIR_COLOURS[0], "1"),
                                             (p["b"], PAIR_COLOURS[1], "2"))):
        x = qx + j * (qw + qg)
        ax = mini(c, x, 0.88, qw, 0.32)
        ax.plot(s["phase"], curve, color=colour, lw=1.2, solid_capstyle="round", zorder=3)
        ax.text(4, YLIM[1] - 0.5, lab, color=INK, fontsize=FS, fontweight="bold", ha="left",
                va="top")

    # ---- the loop's two edges
    xl, xr = MX0 + 0.10, MX1 - 0.10
    c.arrow((xr, m1y0 - 0.005), (xr, m2y1 + 0.005), colour=INK)
    c.text(xr - 0.05, (m1y0 + m2y1) / 2, "Ask the most\ninformative pair", ha="right",
           linespacing=1.1)
    c.arrow((xl, m2y1 + 0.005), (xl, m1y0 - 0.005), colour=INK)
    c.text(xl + 0.05, (m1y0 + m2y1) / 2, "Answer updates\nthe belief", ha="left",
           linespacing=1.1)

    # ---- right: identified profile -> validation
    r1y0 = 1.70
    c.arrow((MX1 + 0.005, 2.01), (RX0 - 0.005, 2.01))
    c.box(RX0, r1y0, RX1 - RX0, TY1 - r1y0, **box)
    c.text(RX0 + 0.06, 2.30, "After k\ncomparisons:", linespacing=1.1)
    c.text(RX0 + 0.06, 2.135, "identified profile", color=BLUE)
    ax = mini(c, RX0 + 0.06, r1y0 + 0.05, RX1 - RX0 - 0.12, 0.31)
    preferred(ax, s)
    estimate(ax, s["mean"][K_ID], s)
    ax.text(97, YLIM[0] + 0.8, f"k = {K_ID}", color=MUTED, fontsize=FS, ha="right",
            va="bottom")
    r2y1 = 1.61
    c.arrow((rc, r1y0 - 0.005), (rc, r2y1 + 0.005))
    c.box(RX0, TY0, RX1 - RX0, r2y1 - TY0, **box)
    c.text(RX0 + 0.06, 1.515, "Validation:", fontweight="bold")
    c.text(RX0 + 0.06, 1.30, "preferred over\ngeneric\nassistance?", linespacing=1.1)
    ax = mini(c, RX0 + 0.06, TY0 + 0.05, RX1 - RX0 - 0.12, 0.27)
    generic(ax, s)
    estimate(ax, s["mean"][K_ID], s)

    # ---- bottom: this paper's simulated users
    BY0, BY1 = 0.03, 0.70
    c.box(LX0, BY0, RX1 - LX0, BY1 - BY0, r=0.06, edgecolor=MUTED, lw=0.8, ls=(0, (3, 2)))
    c.text(LX0 + 0.07, 0.625, "This paper: simulated users", fontsize=FS_HEAD,
           fontweight="bold")
    c.text(LX0 + 0.07, 0.295, "Built from walking data, they\nanswer like a psychometric\n"
           "function of the distance d\nto their preferred profile:", linespacing=1.1)
    sx, sy, sw, sh = 1.64, 0.20, 0.58, 0.27
    sigmoid(c, sx, sy, sw, sh)
    c.text(sx + sw / 2, sy + sh + 0.035, "P(choose 1)", ha="center", va="bottom")
    c.text(sx + sw / 2, 0.115, "d(2) − d(1)", ha="center")
    c.text(2.68, 0.295, "Their spread of\npreferences is\nmatched to real\nparticipants.",
           linespacing=1.1)
    c.arrow((MX0 + 0.28, BY1 + 0.005), (MX0 + 0.28, TY0 - 0.005), colour=MUTED)

    w, h = c.fig.get_size_inches()
    if w > FIG_W + 1e-9 or h > H + 1e-9:
        raise RuntimeError(f"figure is {w:.2f} x {h:.2f} in")
    c.save(path, path.parent)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--figdir", type=Path, default=ROOT / "paper" / "figures")
    args = ap.parse_args(argv)
    args.figdir.mkdir(parents=True, exist_ok=True)
    s = session(args.out)
    print(f"source: {s['source']}")
    for k in KS:
        print(f"k = {k:2d}: belief spread {s['spread'][k]:.2f} Nm, identified error "
              f"{s['err'][k]:.2f} Nm")
    q = s["pair"][K_PAIR]
    print(f"pair asked after {K_PAIR} answers ({q['kind']}): d(1) {q['d_a']:.2f}, "
          f"d(2) {q['d_b']:.2f} Nm, user chose {'1' if q['prefers_a'] else '2'}")
    print(f"generic assistance: {fc._rms(s['generic'], s['ideal']):.3f} Nm from the ideal")
    for variant, name in VARIANTS:
        path = figure(s, variant, args.figdir / f"{name}.pdf")
        print(f"wrote {path} and {path.with_suffix('.png').name} ({FIG_W:.2f} x {H:.2f} in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
