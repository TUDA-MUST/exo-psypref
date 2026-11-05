"""The numbers of the paper's five figures as CSV, for their pgfplots/TikZ versions.

Usage:  python scripts/paper_icorr_tikz_export.py [--out <main>/out]
            [--csv paper/figures/tikz/data] [--only headline families axes projection method]

paper/figures/tikz/ holds a pgfplots/TikZ version of each matplotlib figure of the ICORR paper,
at the size the paper prints it. This script hands them their numbers. For each figure it runs,
in a process of its own, the matplotlib script's own loaders and its unchanged drawing function,
the figure going to a temporary folder, and

  1. compares that PDF with paper/figures/<name>.pdf, byte for byte (sha256); fig_families.pdf
     is saved with its creation date, which is masked. Equal bytes mean that the numbers below
     are the ones the committed figure plots;
  2. writes the CSVs: the plotted numbers from the values the drawing function used, and the
     positions the script measured with matplotlib's text metrics (label placement), the axis
     limits and the silhouette's geometry from the drawn figure, each cross-checked against the
     other where both exist.

No script and no figure in paper/figures is changed. Sources:

  fig_headline            paper_icorr_fig_headline_sim.compute (10,000 draws) and fig_headline
  fig_families            paper_icorr_fig_families.main, --room simulated, with
                          out/icorr/proj_numbers.json and out/icorr/xsource_numbers.json
  fig_axes                paper_icorr_figures.axes_data and fig_axes (the out/icorr reports)
  fig_projection_diagram  fig_projection_diagram.curves and figure
  fig1_method_silhouette  fig1_method.session and figure (the silhouette variant)

CSV conventions: comma separated, a header row, numbers with ten significant digits, "nan" where
a value does not exist; segments of a polyline file are separated by a row of nan (pgfplots:
``unbounded coords=jump``). paper/figures/tikz/README.md lists every file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGDIR = ROOT / "paper" / "figures"
FIGURES = ("headline", "families", "axes", "projection", "method")
PDF_NAME = {"headline": "fig_headline", "families": "fig_families", "axes": "fig_axes",
            "projection": "fig_projection_diagram", "method": "fig1_method_silhouette"}


# ---------------------------------------------------------------- helpers
class Capture:
    """Record every matplotlib figure that is saved inside the block (the drawing functions
    close their figure after saving it; the object stays readable)."""

    def __enter__(self):
        import matplotlib.figure as mfig
        self._cls, self._orig, self.figs = mfig.Figure, mfig.Figure.savefig, []
        cap = self

        def savefig(fig, *a, **k):
            if not any(f is fig for f in cap.figs):
                cap.figs.append(fig)
            return cap._orig(fig, *a, **k)

        mfig.Figure.savefig = savefig
        return self

    def __exit__(self, *exc):
        self._cls.savefig = self._orig
        return False


def _bytes(path: Path, mask_date: bool) -> bytes:
    b = path.read_bytes()
    if mask_date:
        b = re.sub(rb"/CreationDate \(D:[0-9+\-']*\)", b"/CreationDate (masked)", b)
    return b


def same_pdf(made: Path, name: str, mask_date: bool = False) -> str:
    """Stop unless the PDF just drawn equals paper/figures/<name>.pdf byte for byte."""
    ref = FIGDIR / f"{name}.pdf"
    a, b = _bytes(made, mask_date), _bytes(ref, mask_date)
    sha = hashlib.sha256(ref.read_bytes()).hexdigest()[:12]
    if a != b:
        raise SystemExit(f"{name}: the PDF drawn now differs from {ref} (sha256 {sha}); the "
                         "numbers would not be the committed figure's")
    note = f"{name}.pdf redrawn: identical to paper/figures/{name}.pdf (sha256 {sha}"
    return note + (", creation date masked)" if mask_date else ")")


def num(v) -> str:
    if isinstance(v, str):
        return v
    v = float(v)
    if v != v:
        return "nan"
    return f"{v:.10g}"


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([num(v) for v in r])
    return path


def _close(a, b, tol=1e-9) -> bool:
    import numpy as np
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol,
                            rtol=0, equal_nan=True))


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise SystemExit(f"cross-check failed: {what}")


def _colour(artist_colour) -> str:
    from matplotlib.colors import to_hex
    return to_hex(artist_colour, keep_alpha=False).upper()


def _hex(artist_colour) -> str:
    """The colour for a CSV cell: hex digits without '#', which TeX reads as a parameter."""
    return _colour(artist_colour)[1:]


# ---------------------------------------------------------------- Fig. 2, fig_headline
def headline(out: Path, tmp: Path, dest: Path) -> list[str]:
    import numpy as np
    import paper_icorr_fig_headline_sim as H
    P = H.P
    hd = H.compute(out, H.N_BOOT, False)
    with Capture() as cap:
        pdf = H.fig_headline(hd, tmp)
    notes = [same_pdf(pdf, "fig_headline")]
    ax = cap.figs[0].axes[0]
    B, KMAX = hd["band"], H.KMAX
    keys = ("a", "gp_bald", "proj", "cmaes")

    # the four curves with their 95 % bands, k = 1..40
    header = ["k"] + [f"{k}{s}" for k in keys for s in ("", "_lo", "_hi")]
    rows = [[i + 1] + [B[k][f][i] for k in keys for f in ("est", "lo", "hi")]
            for i in range(KMAX)]
    for k in keys:      # the drawn curves are these numbers
        line = next(ln for ln in ax.get_lines() if len(ln.get_xdata()) == KMAX
                    and _colour(ln.get_color()) == H.STYLE[k][1].upper())
        _check(_close(line.get_ydata(), B[k]["est"]), f"headline curve {k}")
    write_csv(dest / "headline_curves.csv", header, rows)

    # limits and the two reference lines (axhline: generic 50 %, then the ceiling)
    ref = [ln for ln in ax.get_lines() if list(ln.get_xdata()) == [0, 1]]
    generic, ceiling = float(ref[0].get_ydata()[0]), float(ref[1].get_ydata()[0])
    _check(_close(ceiling, hd["ceiling"]["est"]) and generic == 0.5, "headline reference lines")
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    write_csv(dest / "headline_meta.csv",
              ["xmin", "xmax", "ymin", "ymax", "generic", "ceiling", "ceiling_lo", "ceiling_hi"],
              [[x0, x1, y0, y1, generic, ceiling, hd["ceiling"]["lo"], hd["ceiling"]["hi"]]])

    # the comparison savings: a level line per (comparator, k), the comparator's marker there
    levels: dict = {}
    for r in H.REACH:
        levels.setdefault((r[1], r[2]), []).append(int(hd["reach"][r]["point"]))
    lv_rows = []
    drawn = [ln for ln in ax.get_lines() if _colour(ln.get_color()) == P.INK.upper()
             and abs(ln.get_linewidth() - 0.6) < 1e-9 and len(ln.get_xdata()) == 2]
    for i, ((slow, kk), firsts) in enumerate(levels.items()):
        level = float(B[slow]["est"][kk - 1])
        _check(_close(drawn[i].get_xdata(), [min(firsts), kk])
               and _close(drawn[i].get_ydata(), [level, level]), f"headline level {slow}")
        lv_rows.append([i, slow, min(firsts), kk, level])
    write_csv(dest / "headline_levels.csv", ["row", "comparator", "k_from", "k_to", "level"],
              lv_rows)

    # the three annotations: where the faster curve first reaches the level, the label's
    # lower-left corner as the script placed it (sizes measured in its 8-pt type) and the leader
    texts = [t for t in ax.texts if t.get_text().startswith("same result in")]
    leaders = [ln for ln in ax.get_lines() if _colour(ln.get_color()) == P.MUTED.upper()
               and abs(ln.get_linewidth() - 0.5) < 1e-9]
    _check(len(texts) == len(H.REACH) == len(leaders), "headline: three labels, three leaders")
    re_rows = []
    for i, r in enumerate(H.REACH):
        fast, slow, kk = r
        d = hd["reach"][r]
        ka = int(d["point"])
        _check(texts[i].get_text() == f"same result in\n{ka} vs {kk}", f"headline label {r}")
        lx, ly = texts[i].get_position()
        (ax0, ax1), (ay0, ay1) = leaders[i].get_xdata(), leaders[i].get_ydata()
        re_rows.append([i, fast, slow, kk, ka, d["lo"], d["hi"], float(B[fast]["est"][ka - 1]),
                        float(B[slow]["est"][kk - 1]), lx, ly, ax0, ay0, ax1, ay1])
    write_csv(dest / "headline_reach.csv",
              ["row", "faster", "comparator", "k_comparator", "k_first", "k_first_lo",
               "k_first_hi", "y_first", "level", "label_x", "label_y", "leader_x0", "leader_y0",
               "leader_x1", "leader_y1"], re_rows)

    # the direct labels at the right end, spread by the script's measured label heights
    names = {"a": H.STYLE["a"][0], "gp_bald": H.STYLE["gp_bald"][0],
             "proj": H.STYLE["proj"][0], "cmaes": H.STYLE["cmaes"][0],
             "ceiling": "ceiling", "generic": "generic"}
    ends = {"a": B["a"]["est"][-1], "gp_bald": B["gp_bald"]["est"][-1],
            "proj": B["proj"]["est"][-1], "cmaes": B["cmaes"]["est"][-1],
            "ceiling": ceiling, "generic": generic}
    side = [ln for ln in ax.get_lines() if not ln.get_clip_on() and len(ln.get_xdata()) == 2
            and min(ln.get_xdata()) > KMAX]
    el_rows = []
    for i, key in enumerate(("a", "gp_bald", "proj", "cmaes", "ceiling", "generic")):
        t = next(t for t in ax.texts if t.get_text() == names[key])
        tx, ty = t.get_position()
        lead, sample = side[2 * i], side[2 * i + 1]
        _check(_close(lead.get_ydata(), [ends[key], ty]) and _close(sample.get_ydata(), [ty, ty]),
               f"headline end label {key}")
        el_rows.append([i, key, ends[key], ty, lead.get_xdata()[0], lead.get_xdata()[1],
                        sample.get_xdata()[1], tx])
    write_csv(dest / "headline_endlabels.csv",
              ["row", "curve", "y_end", "y_label", "x_leader0", "x_leader1", "x_sample1",
               "x_text"], el_rows)
    _check(not hd.get("layout"), f"headline layout_check: {hd.get('layout')}")
    notes.append(f"fig_headline: {len(hd['band'])} curves, {H.N_BOOT:,} draws, "
                 f"reach {', '.join(H._reach_txt(hd['reach'][r]) for r in H.REACH)}")
    return notes


# ---------------------------------------------------------------- Fig. 4b, fig_families
def families(out: Path, tmp: Path, dest: Path) -> list[str]:
    import numpy as np
    import paper_icorr_fig_families as F
    numbers = out / "icorr" / "proj_numbers.json"
    xsource = out / "icorr" / "xsource_numbers.json"
    with Capture() as cap:
        F.main(["--numbers", str(numbers), "--xsource", str(xsource), "--room", "simulated",
                "--figdir", str(tmp)])
    notes = [same_pdf(tmp / "fig_families.pdf", "fig_families", mask_date=True)]
    ax = cap.figs[0].axes[0]
    num_ = json.loads(numbers.read_text(encoding="utf-8"))
    xs = json.loads(xsource.read_text(encoding="utf-8"))["cells"]
    fams = num_["families"]
    order = [k for k in F.TICK if k in fams]
    room = "simulated"

    # the drawn artists, in drawing order: per family the range bar, the ceiling's errorbar,
    # PsyPref's errorbar (where it ran) and the floor label
    from matplotlib.container import BarContainer, ErrorbarContainer
    bars = [c for c in ax.containers if isinstance(c, BarContainer)]
    errs = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    floor_txt = [t for t in ax.texts if t.get_text() != "floor, JND"]
    rows, ie = [], 0
    for i, k in enumerate(order):
        e = fams[k]
        if not e.get("built", True) or "ceiling" not in e:
            raise SystemExit(f"fig_families: {k} not built; the TikZ version has no such slot")
        c = e["ceiling"][room]
        vals = [xs[f"{src}|{F.XS_REP[k]}"]["ceiling_a"]["est"] for src in F.XS_OTHER]
        lo_r, hi_r = min(vals), max(vals)
        rect = bars[i].patches[0]
        _check(_close([rect.get_x(), rect.get_x() + rect.get_width(), rect.get_y(),
                       rect.get_y() + rect.get_height()], [i - 0.27, i + 0.03, lo_r, hi_r]),
               f"families range bar {k}")
        _check(_close([errs[ie].lines[0].get_xdata()[0], errs[ie].lines[0].get_ydata()[0]],
                      [i - 0.12, c["est"]]), f"families ceiling {k}")
        ie += 1
        if "psypref_k24" in e:
            p = e["psypref_k24"][room]
            kind = 1 if k in F.PROMP else 2
            _check(_close([errs[ie].lines[0].get_xdata()[0], errs[ie].lines[0].get_ydata()[0]],
                          [i + 0.12, p["est"]]), f"families PsyPref {k}")
            ie += 1
            pv = [p["est"], p["lo"], p["hi"]]
        else:
            kind, pv = 0, [float("nan")] * 3
        f = e["floor_jnd"]["simulated"]
        label = f"{f['est']:.2f}" if f["est"] < 1.0 else f"{f['est']:.1f}"
        _check(floor_txt[i].get_text() == label, f"families floor label {k}")
        rows.append([i, k, e["n_params"], c["est"], c["lo"], c["hi"], lo_r, hi_r, kind, *pv,
                     f["est"], label, int(f["est"] < 1.0)])
    _check(ie == len(errs), "families: every errorbar accounted for")
    write_csv(dest / "families.csv",
              ["x", "family", "n_params", "ceiling", "ceiling_lo", "ceiling_hi", "range_lo",
               "range_hi", "psypref_kind", "psypref", "psypref_lo", "psypref_hi", "floor_jnd",
               "floor_label", "floor_bold"], rows)
    # range bars as closed rectangles (x from x - 0.27 to x + 0.03, as ax.bar drew them)
    poly = []
    for i, r in enumerate(rows):
        rect = bars[i].patches[0]
        xa, xb = rect.get_x(), rect.get_x() + rect.get_width()
        ya, yb = rect.get_y(), rect.get_y() + rect.get_height()
        poly += [[xa, ya], [xb, ya], [xb, yb], [xa, yb], [xa, ya], [float("nan")] * 2]
    write_csv(dest / "families_rangebars.csv", ["x", "y"], poly)

    ref = [ln for ln in ax.get_lines() if list(ln.get_xdata()) == [0, 1]]
    leg = ax.get_legend()
    fig = ax.figure
    fig.canvas.draw()
    bb = leg.get_frame().get_window_extent().transformed(ax.transAxes.inverted())
    floor_y = floor_txt[0].get_position()[1]
    head = next(t for t in ax.texts if t.get_text() == "floor, JND")
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    write_csv(dest / "families_meta.csv",
              ["xmin", "xmax", "ymin", "ymax", "generic", "floor_y", "floor_head_x",
               "legend_x1", "legend_y0", "legend_x0", "legend_y1", "offset_ceiling",
               "offset_psypref"],
              [[x0, x1, y0, y1, float(ref[0].get_ydata()[0]), floor_y, head.get_position()[0],
                bb.x1, bb.y0, bb.x0, bb.y1, -0.12, 0.12]])
    notes.append(f"fig_families: {len(rows)} families, room {room}, XS sources "
                 f"{', '.join(F.XS_OTHER)}")
    return notes


# ---------------------------------------------------------------- Fig. 3, fig_axes
def axes(out: Path, tmp: Path, dest: Path) -> list[str]:
    import paper_icorr_figures as P
    ad = P.axes_data(out)
    if ad["problems"]:
        raise SystemExit(f"fig_axes: {ad['problems']}")
    with Capture() as cap:
        pdf = P.fig_axes(ad, tmp, None)
    notes = [same_pdf(pdf, "fig_axes")]
    fig = cap.figs[0]
    dx, half = 0.19, 0.42          # fig_axes' call of _axes_points
    rows, meta = [], []
    for j, key in enumerate("abc"):
        ax = fig.axes[j]
        pts = ad["panels"][key]
        for x, p in enumerate(pts):
            ceil = p.get("ceiling")
            cv = [ceil["est"], ceil["lo"], ceil["hi"]] if ceil is not None else [float("nan")] * 3
            rows.append([key, x, p["xlab"], x - dx, p["a"]["est"], p["a"]["lo"], p["a"]["hi"],
                         p["best"], x + dx, p["comp"]["est"], p["comp"]["lo"], p["comp"]["hi"],
                         x - half, x + half, *cv])
        # cross-check: markers, intervals and ceilings drawn are these
        drawn = {(round(float(ln.get_xdata()[0]), 6), round(float(ln.get_ydata()[0]), 9))
                 for ln in ax.get_lines() if ln.get_marker() not in ("None", None, "")}
        for r in rows:
            if r[0] != key:
                continue
            _check((round(r[3], 6), round(r[4], 9)) in drawn, f"axes ({key}) PsyPref {r[2]}")
            _check((round(r[8], 6), round(r[9], 9)) in drawn, f"axes ({key}) comparator {r[2]}")
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        meta.append([key, len(pts), x0, x1, y0, y1, 0.5])
    write_csv(dest / "axes_points.csv",
              ["panel", "x", "condition", "x_psypref", "psypref", "psypref_lo", "psypref_hi",
               "comparator", "x_comparator", "comp", "comp_lo", "comp_hi", "x_ceiling0",
               "x_ceiling1", "ceiling", "ceiling_lo", "ceiling_hi"], rows)
    for key in "abc":       # one file per panel, for pgfplots' \addplot table
        write_csv(dest / f"axes_points_{key}.csv",
                  ["x", "condition", "x_psypref", "psypref", "psypref_lo", "psypref_hi",
                   "comparator", "x_comparator", "comp", "comp_lo", "comp_hi", "x_ceiling0",
                   "x_ceiling1", "ceiling", "ceiling_lo", "ceiling_hi"],
                  [r[1:] for r in rows if r[0] == key])
        seg = []           # the ceilings as segments, each its own dash pattern
        for r in rows:
            if r[0] == key and r[14] == r[14]:
                seg += [[r[12], r[14]], [r[13], r[14]], [float("nan")] * 2]
        write_csv(dest / f"axes_ceilings_{key}.csv", ["x", "y"], seg)
    write_csv(dest / "axes_meta.csv", ["panel", "n", "xmin", "xmax", "ymin", "ymax", "generic"],
              meta)
    notes.append("fig_axes: " + "; ".join(f"({k}) {len(ad['panels'][k])} conditions"
                                          for k in "abc"))
    return notes


# ---------------------------------------------------------------- the walker (Figs. 1, 4a)
#: The walker's parts in fig1_method.walker's drawing order: name, the colour constant.
WALKER_PARTS = ("arm_back", "leg_back", "torso", "leg_front", "arm_front", "head",
                "belt", "back_unit", "strut", "cuff", "actuator", "actuator_dot")


def walker_geometry(dest: Path, figs_check: list) -> list[str]:
    """fig1_method.walker drawn with its hip at x = 0, the ground at y = 0 and height 1: every
    part as a polyline or a closed outline (patches flattened), in units of the walker's height;
    line widths as a share of the height. ``figs_check``: (overlay axes, x, y, h, exo) of a
    walker in a drawn figure, compared with this geometry."""
    import numpy as np
    import fig1_concepts as fc
    import fig1_method as M
    from matplotlib.patches import Circle, FancyBboxPatch

    fc.MAX_H = 3.0
    c = fc.Canvas(1.0)
    M.walker(c, 0.0, 0.0, 1.0, exo=True)
    lines = list(c.ov.get_lines())
    patches = list(c.ov.patches)
    pt = 72.0
    # drawing order: five limbs, the head (patch), the belt, the back unit (patch), the strut,
    # the cuff (lines), the actuator and its dot (patches)
    _check(len(lines) == 8 and len(patches) == 4, "walker: 8 lines and 4 patches")
    parts = [("line", lines[0]), ("line", lines[1]), ("line", lines[2]), ("line", lines[3]),
             ("line", lines[4]), ("patch", patches[0]), ("line", lines[5]),
             ("patch", patches[1]), ("line", lines[6]), ("line", lines[7]),
             ("patch", patches[2]), ("patch", patches[3])]
    geo, style = [], []
    for i, ((kind, art), name) in enumerate(zip(parts, WALKER_PARTS)):
        if kind == "line":
            xy = np.column_stack([art.get_xdata(), art.get_ydata()])
            style.append([i, name, "line", art.get_linewidth() / pt, _hex(art.get_color()),
                          "none", 0.0, art.get_solid_capstyle()])
        else:
            if isinstance(art, Circle):
                th = np.linspace(0.0, 2 * np.pi, 65)
                cx, cy = art.center
                xy = np.column_stack([cx + art.radius * np.cos(th), cy + art.radius * np.sin(th)])
            elif isinstance(art, FancyBboxPatch):
                poly = art.get_path().to_polygons(art.get_transform()
                                                  - c.ov.transData)
                xy = np.asarray(poly[0])
            else:
                raise SystemExit(f"walker: unexpected patch {art}")
            edge = art.get_edgecolor()
            lw = art.get_linewidth() / pt if edge[3] > 0 else 0.0
            style.append([i, name, "patch", lw, _hex(edge) if edge[3] > 0 else "none",
                          _hex(art.get_facecolor()), 1.0, "round"])
        geo.append([[x, y] for x, y in xy])
    # one file per part, so that pgfplots reads each part's few rows only (filtering one file
    # per part cost seconds per walker)
    for i, rows in enumerate(geo):
        write_csv(dest / f"walker_part{i}.csv", ["x", "y"], rows)
    write_csv(dest / "walker_parts.csv",
              ["part", "name", "kind", "lw_per_height", "edge", "fill", "closed", "cap"], style)
    # the walkers in the drawn figures are this geometry, shifted and scaled
    for ov, x, y, h, exo in figs_check:
        got = [np.column_stack([ln.get_xdata(), ln.get_ydata()]) for ln in ov.get_lines()]
        for i, (kind, art) in enumerate(parts):
            if kind != "line" or (not exo and i >= 5):
                continue
            want = np.column_stack([x + np.asarray(art.get_xdata()) * h,
                                    y + np.asarray(art.get_ydata()) * h])
            _check(any(g.shape == want.shape and _close(g, want, 1e-9) for g in got),
                   f"walker part {WALKER_PARTS[i]} at ({x:.3f}, {y:.3f}, {h:.3f})")
    return ["walker: fig1_method.walker at unit height, 12 parts; the drawn walkers match"]


# ---------------------------------------------------------------- Fig. 4a, projection diagram
def projection(out: Path, tmp: Path, dest: Path) -> list[str]:
    import numpy as np
    import fig_projection_diagram as PD
    d = PD.curves(out)
    with Capture() as cap:
        pdf = PD.figure(d, tmp / f"{PD.NAME}.pdf")
    notes = [same_pdf(pdf, PD.NAME)]
    fig = cap.figs[0]
    cols = [d["phase"], d["promp"][0], d["promp"][1], d["fit"][0], d["fit"][1], d["back"][0],
            d["back"][1]]
    write_csv(dest / "projection_curves.csv",
              ["phase", "promp1", "promp2", "fit1", "fit2", "back1", "back2"],
              [list(r) for r in np.column_stack(cols)])
    rows = []
    for j in (0, 1):
        t, v = d["nodes"][j]
        rows += [[j + 1, a, b] for a, b in zip(t, v)]
    write_csv(dest / "projection_nodes.csv", ["profile", "t", "v"], rows)
    glyphs = fig.axes[1:]
    lims = {(a.get_xlim(), a.get_ylim()) for a in glyphs}
    _check(len(lims) == 1, "projection: every glyph has the same limits")
    (gx0, gx1), (gy0, gy1) = lims.pop()
    write_csv(dest / "projection_meta.csv", ["xmin", "xmax", "ymin", "ymax", "n_glyphs"],
              [[gx0, gx1, gy0, gy1, len(glyphs)]])
    # cross-check: every curve and node set is drawn in some glyph
    drawn = [np.asarray(ln.get_ydata(), dtype=float) for a in fig.axes[1:] for ln in a.get_lines()]
    for name, y in (("promp1", d["promp"][0]), ("promp2", d["promp"][1]), ("fit1", d["fit"][0]),
                    ("fit2", d["fit"][1]), ("back1", d["back"][0])):
        _check(any(g.shape == np.shape(y) and _close(g, y) for g in drawn), f"projection {name}")
    H = PD.H
    notes += walker_geometry(dest, [(fig.axes[0], 2.76 + 0.095, 0.37 + 0.06, 0.42, True)])
    notes.append(f"fig_projection_diagram: {d['source']}, pair asked after {PD.K_PAIR} answers, "
                 f"{len(d['phase'])} phase points; figure {PD.FIG_W} x {H} in")
    return notes


# ---------------------------------------------------------------- Fig. 1, method (silhouette)
def method(out: Path, tmp: Path, dest: Path) -> list[str]:
    import numpy as np
    import fig1_method as M
    s = M.session(out)
    with Capture() as cap:
        pdf = M.figure(s, "silhouette", tmp / "fig1_method_silhouette.pdf")
    notes = [same_pdf(pdf, "fig1_method_silhouette")]
    fig = cap.figs[0]
    ph = np.asarray(s["phase"], dtype=float)
    p = s["pair"][M.K_PAIR]
    cols = [ph, s["ideal"], s["generic"], p["a"], p["b"]] + [s["mean"][k] for k in M.KS]
    write_csv(dest / "method_curves.csv",
              ["phase", "preferred", "generic", "pair1", "pair2"]
              + [f"mean_k{k}" for k in M.KS], [list(r) for r in np.column_stack(cols)])
    for k in M.KS:     # the fans: one column per posterior draw
        fan = np.asarray(s["fan"][k], dtype=float)
        write_csv(dest / f"method_fan_k{k}.csv",
                  ["phase"] + [f"d{i}" for i in range(fan.shape[0])],
                  [list(r) for r in np.column_stack([ph, fan.T])])
    # the sigmoid, as fig1_method.sigmoid draws it
    sig = next(a for a in fig.axes[1:] if a.get_xlim()[0] < -1)
    curve = next(ln for ln in sig.get_lines() if len(ln.get_xdata()) > 10)
    delta = np.asarray(curve.get_xdata(), dtype=float)
    want = M.psychometric(delta, M.S_MAIN, M.LAPSE)
    _check(_close(curve.get_ydata(), want), "method sigmoid")
    write_csv(dest / "method_sigmoid.csv", ["delta", "p"], [[a, b] for a, b in zip(delta, want)])
    (x0, x1), (y0, y1) = sig.get_xlim(), sig.get_ylim()
    glyphs = [a for a in fig.axes[1:] if a is not sig]
    lims = {(a.get_xlim(), a.get_ylim()) for a in glyphs}
    _check(len(lims) == 1, "method: every glyph has the same limits")
    (gx0, gx1), (gy0, gy1) = lims.pop()
    write_csv(dest / "method_meta.csv",
              ["lapse", "s", "sig_xmin", "sig_xmax", "sig_ymin", "sig_ymax", "k_pair", "k_id",
               "xmin", "xmax", "ymin", "ymax"],
              [[M.LAPSE, M.S_MAIN, x0, x1, y0, y1, M.K_PAIR, M.K_ID, gx0, gx1, gy0, gy1]])
    # cross-check: the fans, means and the pair are drawn
    drawn = [np.asarray(ln.get_ydata(), dtype=float) for a in fig.axes[1:] for ln in a.get_lines()]
    for k in M.KS:
        for i, y in enumerate(s["fan"][k]):
            _check(any(g.shape == y.shape and _close(g, y) for g in drawn), f"method fan k={k} {i}")
        _check(any(g.shape == s["mean"][k].shape and _close(g, s["mean"][k]) for g in drawn)
               or k == 0, f"method mean k={k}")
    for y in (p["a"], p["b"], s["ideal"], s["generic"]):
        _check(any(g.shape == y.shape and _close(g, y) for g in drawn), "method pair/ideal")
    walkers = [(fig.axes[0], 0.03 + 0.20 + 0.18 * j, 2.265, 0.14, False) for j in range(3)]
    phx, phy, phh = 0.86 + 0.06, 0.80 + 0.05, 1.40 - 0.80 - 0.10
    walkers.append((fig.axes[0], phx + 0.44 / 2 - 0.02, phy + 0.02, phh - 0.04, True))
    notes += walker_geometry(dest, walkers)
    notes.append(f"fig1_method_silhouette: {s['source']}; fans of {len(s['fan'][0])} draws at "
                 f"k = {', '.join(str(k) for k in M.KS)}")
    return notes


EXPORT = {"headline": headline, "families": families, "axes": axes, "projection": projection,
          "method": method}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out",
                    help="the sweep output directory holding phase5e/, icorr/, phase5-schaefer/")
    ap.add_argument("--csv", type=Path, default=FIGDIR / "tikz" / "data")
    ap.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES))
    ap.add_argument("--one", choices=FIGURES, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    args.csv.mkdir(parents=True, exist_ok=True)
    if args.one is None:
        # one process per figure, as the scripts run: each sets its own rcParams at import
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = "4"           # as paper_icorr_fig_headline_sim.py sets them as a script
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
        failed = []
        for name in args.only:
            print(f"{name}:", flush=True)
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out",
                                str(args.out), "--csv", str(args.csv), "--one", name], env=env)
            if r.returncode:
                failed.append(name)
        if failed:
            print(f"FAILED: {', '.join(failed)}")
        return 1 if failed else 0
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        notes = EXPORT[args.one](args.out, Path(tmp), args.csv)
    for n in notes:
        print(f"  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
