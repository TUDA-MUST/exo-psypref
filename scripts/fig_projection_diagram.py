"""Panel (a) of the ICORR 2027 paper's interpretable-families figure: how an interpretable
profile family is coupled to the ProMP, so that the same optimizer serves every family. A block
diagram one IEEE column wide (3.5 in) and 1.6 in tall, 7-8 pt type, in Fig. 1's style
(fig1_method.py: rounded boxes, thin arrows, PsyPref's blue, the pair colours). It prints no
result numbers.

  Left    PsyPref (or any optimizer) searches the ProMP's standardized coordinates z and
          proposes two profiles z1, z2: two smooth ProMP curves tau(z).
  Top     Each proposed curve is replaced by its bounded least-squares fit in the family F,
          theta = argmin ||tau_F(theta) - tau(z)||, device-clipped (prompref.learning.lens).
  Right   The user feels and compares the family curves tau_F(theta1), tau_F(theta2).
  Bottom  PsyPref is told the ProMP projection of the curves shown, z~ = B+ tau_F(theta), with
          the answer, and updates.
  Strip   The result is the family fit of PsyPref's optimum, reported in the family's own
          parameters theta: peak times, magnitudes, rise and fall.

Data: every curve is real and no session is run. The two ProMP curves are the pair PsyPref asked
after six answers in the stored median-room session (loso-AB31/pop016, e-main), the pair Fig. 1
shows, loaded through fig1_concepts.real_session (which checks the fold's whitening, basis and
ideal against the snapshot file). Each is fitted in the Franks et al. family with the lens's own
code: SurfaceFitter (multi-start bounded least squares, device-clipped) on the fold's Franks
prior box (phase5._franks_prior, arm_space "tight"), in the order the lens fits a pair. The
projection is LensLearner.update's: the ProMP weights pinv(B) @ tau_F(theta), clipped to the
ProMP box, drawn as their device-clipped ProMP curve. The dots mark the Franks nodes (onset,
extension peak, the two ends of the zero-torque period, flexion peak, end of flexion), whose
times and values are the parameters a reader can report. The stored session ran PsyPref without
the projection: these are the fits of the curves it asked, not curves a projection session
showed. If anything fails to load, the script stops; this figure has no schematic fallback.

Style: fig1_concepts' Canvas and paper_icorr_figures' colours and rcParams; the walker is
fig1_method's (imported, not copied). Math is set in Arial through matplotlib's custom mathtext
fontset, in this process only.

Usage:
    python scripts/fig_projection_diagram.py [--out <project>/out]
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
from fig1_method import walker  # noqa: E402
from paper_icorr_figures import BLUE, CONTEXT, FIG_W, INK, MUTED  # noqa: E402

plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "mathtext.sf": "Arial"})

NAME = "fig_projection_diagram"
H = 1.60                 # figure height, in
K_PAIR = 6               # the pair PsyPref asked after six answers (Fig. 1's pair)
ARM = "tight"            # fig1_concepts' arm
FS, FS_HEAD = 7, 8       # pt
YLIM = (-13.5, 13.5)     # Fig. 1's glyph range, Nm
PAIR_COLOURS = fc.PAIR_COLOURS
HALO = dict(color=CONTEXT, lw=2.2)    # the input curve behind a fit or a projection


# ---------------------------------------------------------------- data
def curves(out: Path) -> dict:
    """The stored pair as ProMP curves, their Franks fits and the fits' ProMP projections."""
    from prompref.designs.franks import nodes
    from prompref.experiment import phase5 as p5
    from prompref.learning.lens import SurfaceFitter
    from prompref.population.curves import TORQUE_LIMIT_NM, physics_clip

    fc.KS, fc.K_PAIR, fc.N_FAN = (K_PAIR,), K_PAIR, 1
    s = fc.real_session(out)
    run = str(out / "phase5-schaefer")
    promp = p5.arm_space(p5._prior(run, fc.FOLD, "promp_s0"), ARM)
    franks = p5.arm_space(p5._franks_prior(run, fc.FOLD), ARM)
    fitter = SurfaceFitter(franks.design, start_mean=franks.whitening.mu)
    pinv = np.linalg.pinv(promp.design.inner.basis.design)
    pair = s["pair"][K_PAIR]
    res = dict(phase=s["phase"], source=s["source"], promp=[], fit=[], back=[], nodes=[],
               theta=[], rms_fit=[], rms_back=[])
    for key in ("a", "b"):
        tau = np.asarray(pair[key], dtype=float)
        theta = fitter.fit(tau)
        shown = physics_clip(franks.design.curve(theta))
        if abs(fc._rms(shown, tau) - fitter.last_rms) > 1e-6:
            raise RuntimeError("the fit's reported RMS is not the RMS of the curve drawn")
        back = np.asarray(promp.design.curve(promp.design.project(pinv @ shown)), dtype=float)
        t, v = nodes(theta)
        res["promp"].append(tau)
        res["fit"].append(shown)
        res["back"].append(back)
        # the last node closes the cycle (the next onset): drop it
        res["nodes"].append((np.mod(t[:-1], 100.0),
                             np.clip(v[:-1], -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM)))
        res["theta"].append(theta)
        res["rms_fit"].append(fitter.last_rms)
        res["rms_back"].append(fc._rms(back, shown))
    return res


# ---------------------------------------------------------------- drawing
def glyph(c, x, y, w, h):
    """Small bare axes in figure inches: gait cycle on x, torque on y, a zero line."""
    ax = c.axes(x, y, w, h)
    fc._bare(ax, y=YLIM)
    return ax


def smooth(ax, d, key, j, **kw):
    style = dict(color=PAIR_COLOURS[j], lw=1.0, zorder=3, solid_capstyle="round")
    style.update(kw)
    ax.plot(d["phase"], d[key][j], **style)


def family(ax, d, j, colour=None, lw=1.0, z=3, ms=2.4):
    """A family curve with its nodes, the points its parameters set."""
    colour = colour or PAIR_COLOURS[j]
    ax.plot(d["phase"], d["fit"][j], color=colour, lw=lw, zorder=z, solid_capstyle="round")
    t, v = d["nodes"][j]
    ax.plot(t, v, ls="none", marker="o", ms=ms, mfc=colour, mec="white", mew=0.4, zorder=z + 1,
            clip_on=False)


def figure(d: dict, path: Path) -> Path:
    fc.MAX_H = H                         # fig1_concepts' Canvas caps the height at its own MAX_H
    c = fc.Canvas(H)
    box = dict(r=0.05, edgecolor=MUTED, lw=0.6)

    # columns (in): PsyPref, the coupling, the user; the main row and the strip below it
    LX0, LX1 = 0.03, 0.81
    MX0, MX1 = 1.09, 2.48
    RX0, RX1 = 2.76, 3.47
    TY0, TY1 = 0.37, 1.46
    FY0 = 0.83                       # bottom of the fit box
    PY1 = 0.77                       # top of the projection box
    ay_top, ay_bot = (FY0 + TY1) / 2 - 0.03, (TY0 + PY1) / 2
    ty = TY1 - 0.085                 # box titles
    gw, gh = 0.36, 0.22              # the coupling glyphs

    # ---- headers: where each step lives
    c.text(LX0, 1.535, r"search in ProMP $\mathbf{z}$", color=MUTED)
    c.text(RX1, 1.535, "show in family F", ha="right", color=MUTED)

    # ---- left: PsyPref in the ProMP's z
    c.box(LX0, TY0, LX1 - LX0, TY1 - TY0, r=0.05, edgecolor=BLUE, lw=0.9)
    c.text(LX0 + 0.06, ty, "PsyPref", fontsize=FS_HEAD, fontweight="bold", color=BLUE)
    c.text(LX0 + 0.06, ty - 0.085, "or any other\noptimizer,\nunchanged for\nevery family",
           va="top", linespacing=1.1)
    ax = glyph(c, LX0 + 0.06, TY0 + 0.07, LX1 - LX0 - 0.12, 0.28)
    for j in (0, 1):
        smooth(ax, d, "promp", j)

    # ---- top middle: the fit in the family
    c.box(MX0, FY0, MX1 - MX0, TY1 - FY0, **box)
    c.text(MX0 + 0.06, ty, "Fit in family F", fontweight="bold")
    c.text(MX0 + 0.06, ty - 0.12,
           r"$\theta = \arg\min\,\Vert\tau_F(\theta) - \tau(\mathbf{z})\Vert$")
    c.text(MX0 + 0.06, ty - 0.235, "bounded, device-clipped", color=MUTED)
    c.text(MX0 + 0.06, FY0 + 0.025 + gh / 2, "F: Schäfer, Franks,\nPCHIP, trapezoid",
           linespacing=1.1, color=MUTED)
    ax = glyph(c, MX1 - gw - 0.06, FY0 + 0.035, gw, gh)
    smooth(ax, d, "promp", 0, zorder=2, **HALO)
    family(ax, d, 0)

    # ---- bottom middle: the projection back to the ProMP
    c.box(MX0, TY0, MX1 - MX0, PY1 - TY0, **box)
    c.text(MX0 + 0.06, PY1 - 0.085, "Project onto ProMP", fontweight="bold")
    c.text(MX0 + 0.06, PY1 - 0.20 + 0.005, r"$\tilde{\mathbf{z}} = B^{+}\tau_F(\theta)$")
    c.text(MX0 + 0.06, TY0 + 0.075, "of the curves shown", color=MUTED)
    ax = glyph(c, MX1 - gw - 0.06, TY0 + 0.035, gw, gh)
    family(ax, d, 0, colour=HALO["color"], lw=HALO["lw"], z=1)
    smooth(ax, d, "back", 0)

    # ---- right: the user feels the family curves
    c.box(RX0, TY0, RX1 - RX0, TY1 - TY0, **box)
    c.text(RX0 + 0.06, ty, "User: 1 or 2?", fontweight="bold")
    walker(c, RX0 + 0.095, TY0 + 0.06, 0.42, exo=True)
    ux, uw, uh = RX0 + 0.23, RX1 - RX0 - 0.28, 0.27
    for j, uy in enumerate((0.96, 0.55)):
        ax = glyph(c, ux, uy, uw, uh)
        family(ax, d, j)
        ax.text(4, YLIM[1] - 0.5, str(j + 1), color=INK, fontsize=FS, fontweight="bold",
                ha="left", va="top")

    # ---- the loop's four edges
    def label(x, y, s):
        c.text(x, y + 0.075, s, ha="center")

    c.arrow((LX1 + 0.005, ay_top), (MX0 - 0.005, ay_top), colour=INK)
    label((LX1 + MX0) / 2, ay_top, r"$\mathbf{z}_1, \mathbf{z}_2$")
    c.arrow((MX1 + 0.005, ay_top), (RX0 - 0.005, ay_top), colour=INK)
    label((MX1 + RX0) / 2, ay_top, r"$\theta_1, \theta_2$")
    c.arrow((RX0 - 0.005, ay_bot), (MX1 + 0.005, ay_bot), colour=INK)
    c.arrow((MX0 - 0.005, ay_bot), (LX1 + 0.005, ay_bot), colour=INK)
    label((LX1 + MX0) / 2, ay_bot, r"$\tilde{\mathbf{z}}_1, \tilde{\mathbf{z}}_2$")

    # ---- the strip: the result, in the family's own parameters
    SY0, SY1 = 0.02, 0.26
    sy, sx = (SY0 + SY1) / 2, 0.20
    c.ov.plot([sx, sx], [TY0 - 0.005, sy], color=MUTED, lw=0.8, zorder=3, solid_capstyle="butt")
    c.arrow((sx - 0.004, sy), (MX0 - 0.005, sy))
    c.text(sx + 0.06, (TY0 + sy) / 2 - 0.005, "optimum,\nfitted in F", linespacing=1.05,
           color=MUTED)
    c.box(MX0, SY0, RX1 - MX0, SY1 - SY0, **box)
    c.text(MX0 + 0.06, sy, r"Report in $\theta$: peak times, magnitudes, rise and fall")

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
    d = curves(args.out)
    print(f"source: {d['source']}, pair asked after {K_PAIR} answers")
    for j in (0, 1):
        print(f"profile {j + 1}: Franks fit {d['rms_fit'][j]:.3f} Nm RMS from the ProMP curve, "
              f"its ProMP projection {d['rms_back'][j]:.3f} Nm from the fit")
    path = figure(d, args.figdir / f"{NAME}.pdf")
    print(f"wrote {path} and {path.with_suffix('.png').name} ({FIG_W:.2f} x {H:.2f} in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
