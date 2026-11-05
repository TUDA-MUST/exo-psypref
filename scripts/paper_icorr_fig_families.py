"""fig_families (draft): interpretable families and how many parameters a profile needs.

Usage:  python scripts/paper_icorr_fig_families.py [--numbers <out>/icorr/proj_numbers.json]
            [--figdir paper/figures] [--room median]

Test P of the our design notes (section 9a), of scripts/icorr_proj_report.py, at
the size the coordinator set on 2026-10-22: one IEEE column (3.5 in) and 1.2 in tall, one compact
dot-and-interval panel. The families are ordered by parameter count on x. Per family:
  - the attainable best-profile level of the preference over generic assistance at the realistic room
    (default the median-matched 1.47 Nm room), with its 95 % interval;
  - PsyPref's achieved preference after 24 comparisons where it ran (on the ProMP, or with each
    proposal projected onto the family);
  - the median best-profile distance (best reachable RMS error, as simulated) in main JND units (0.674 Nm) as a
    small label above the column, bold where it lies below 1 JND.
Every entry is measured against the same generic assistance, the K = 8 ProMP prior mean: generic
assistance is one physical profile, whatever is searched in. With --xsource, a grey bar behind each best-profile level spans the lowest to the
highest best-profile level over XS's six other preferred-profile sources (surrogate, Schäfer (i) and (ii)
published, Franks box, Ramella, Park), against the same generic (XS baseline (a)), as simulated;
use it with --room simulated. A family that was not built (PCHIP)
keeps its slot, marked "not built". Every plotted number is read from proj_numbers.json, which
icorr_proj_report.py writes; nothing is recomputed. The numbers and the figure note are written to
<figdir>/fig_families_numbers.md. Separate from paper_icorr_figures.py so the paper lane's figure
script is left alone; the style constants are that script's.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: 1.3 in tall since 2026-10-23 (was 1.2), for the paper's sans-serif type at 6.5-7 pt.
FIG_W, FIG_H = 3.5, 1.3
# paper_icorr_figures.py's palette: PsyPref blue, the projection orange, greys for context.
INK, MUTED, CONTEXT, RULE = "#222222", "#666666", "#BBBBBB", "#DDDDDD"
BLUE, ORANGE = "#1845FB", "#EE7733"
#: Tick labels: the family, then its parameter count.
TICK = {"schaefer_i": "Sch.\ni 4", "schaefer_ii": "Sch.\nii 6", "trapezoid": "Trap.\n6",
        "franks": "Franks\n8", "pchip_pub": "PCHIP\ni 5", "pchip": "PCHIP\nii 6",
        "promp4": "ProMP\n4", "promp8": "\n8",
        "promp12": "\n12", "promp16": "\n16", "promp24": "\n24"}
PROMP = ("promp4", "promp8", "promp12", "promp16", "promp24")
#: This figure's family keys to XS's representation keys (xsource_numbers.json).
XS_REP = {"schaefer_i": "schaefer_fixed", "schaefer_ii": "schaefer_peaks",
          "trapezoid": "trapezoid_surface", "franks": "franks", "pchip_pub": "pchip_pub",
          "pchip": "pchip", **{k: k for k in PROMP}}
XS_OTHER = ("surrogate", "schaefer_i", "schaefer", "franks", "ramella", "park")
NOTE = ("Every entry is measured against the same generic assistance, the K = 8 ProMP prior "
        "mean: generic assistance is one physical profile, whatever is searched in. Ceiling: the "
        "preference of the best profile of the family over generic assistance (s = 1.0 Nm, "
        "lambda = 0.1); PsyPref: its preference after 24 comparisons, on the ProMP or with each "
        "proposal projected onto the family; labels: the median floor in main JND units (0.674 "
        "Nm), bold below 1 JND. 95 % intervals.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--numbers", type=Path,
                    default=ROOT / "out" / "icorr" / "proj_numbers.json")
    ap.add_argument("--figdir", type=Path, default=ROOT / "paper" / "figures")
    ap.add_argument("--room", default="median", choices=("simulated", "median", "central"))
    ap.add_argument("--xsource", type=Path, default=None,
                    help="xsource_numbers.json: draw the range over other preferred-profile sources")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    num = json.loads(args.numbers.read_text(encoding="utf-8"))
    xs = json.loads(args.xsource.read_text(encoding="utf-8"))["cells"] if args.xsource else None
    if xs is not None and args.room != "simulated":
        raise SystemExit("--xsource ranges are as simulated; use --room simulated")
    fams = num["families"]
    order = [k for k in TICK if k in fams]
    room = args.room

    # Type as paper_icorr_figures.py (2026-10-23; was serif at 5-6 pt): sans-serif, 6.5-7 pt.
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.labelcolor": INK, "ytick.labelcolor": INK,
                         "axes.linewidth": 0.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = {k: i for i, k in enumerate(order)}
    lines = ["# fig_families: plotted numbers", "",
             f"Read from `{args.numbers}` (generated {num['generated']}, commit {num['commit']}, "
             f"{num['boot']} draws). Room: {num['rooms'][room]}. Estimates with 95 % intervals.",
             "", f"Figure note: {NOTE}", "",
             "| family | parameters | ceiling | PsyPref, k = 24 | median floor (Nm, as simulated) "
             "| median floor (main JND) |", "|---|---|---|---|---|---|"]
    err = dict(elinewidth=0.6, capsize=1.2, capthick=0.5)
    for k in order:
        e = fams[k]
        xi = x[k]
        if not e.get("built", True) or "ceiling" not in e:
            ax.axvspan(xi - 0.4, xi + 0.4, color=RULE, alpha=0.6, lw=0)
            ax.text(xi, 0.5, "not\nbuilt", ha="center", va="center", fontsize=5, color=MUTED)
            ax.text(xi, 1.06, "–", ha="center", va="bottom", fontsize=5.2, color=MUTED,
                    transform=ax.get_xaxis_transform())
            lines.append(f"| {e['label']} | — | not built | — | — | — |")
            continue
        c = e["ceiling"][room]
        if xs is not None:
            vals = [xs[f"{src}|{XS_REP[k]}"]["ceiling_a"]["est"] for src in XS_OTHER]
            lo_r, hi_r = min(vals), max(vals)
            ax.bar(xi - 0.12, hi_r - lo_r, bottom=lo_r, width=0.3, color=RULE, lw=0, zorder=2)
            lines.append(f"| {e['label']} | range over other sources (XS baseline (a)) | "
                         f"{lo_r:.3f}–{hi_r:.3f} | | | |")
        ax.errorbar(xi - 0.12, c["est"], yerr=[[c["est"] - c["lo"]], [c["hi"] - c["est"]]],
                    fmt="D", ms=2.8, color=INK, mfc=CONTEXT, mec=INK, mew=0.5, zorder=3, **err)
        p_txt = "—"
        if "psypref_k24" in e:
            p = e["psypref_k24"][room]
            colour, marker = (BLUE, "o") if k in PROMP else (ORANGE, "s")
            ax.errorbar(xi + 0.12, p["est"], yerr=[[p["est"] - p["lo"]], [p["hi"] - p["est"]]],
                        fmt=marker, ms=2.6, color=colour, mfc="white", mew=0.8, zorder=4, **err)
            p_txt = f"{p['est']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]"
        f = e["floor_jnd"]["simulated"]
        below = f["est"] < 1.0
        ax.text(xi, 1.06, f"{f['est']:.2f}" if f["est"] < 1.0 else f"{f['est']:.1f}",
                ha="center", va="bottom", fontsize=6.5, color=INK if below else MUTED,
                fontweight="bold" if below else "normal", transform=ax.get_xaxis_transform())
        fn = e["floor_nm"]["simulated"]
        lines.append(f"| {e['label']} | {e['n_params']} | {c['est']:.3f} [{c['lo']:.3f}, "
                     f"{c['hi']:.3f}] | {p_txt} | {fn['est']:.3f} [{fn['lo']:.3f}, {fn['hi']:.3f}] "
                     f"| {f['est']:.2f} [{f['lo']:.2f}, {f['hi']:.2f}] |")
    ax.text(-0.62, 1.06, "floor, JND", ha="right", va="bottom", fontsize=6, color=MUTED,
            transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=":", zorder=1)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("Preference\nover generic", labelpad=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([TICK[k] for k in order])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=1.5, width=0.4, pad=1.5)
    handles = [Line2D([], [], marker="D", ls="none", color=INK, mfc=CONTEXT, ms=2.8, mew=0.5,
                      label="ceiling = best profile of the family"),
               Line2D([], [], marker="o", ls="none", color=BLUE, mfc="white", ms=2.6, mew=0.8,
                      label="PsyPref, k = 24"),
               Line2D([], [], marker="s", ls="none", color=ORANGE, mfc="white", ms=2.6, mew=0.8,
                      label="with projection")]
    if xs is not None:
        from matplotlib.patches import Patch
        handles.insert(1, Patch(color=RULE, lw=0, label="range, other preferred profiles"))
    leg = ax.legend(handles=handles, loc="lower right", frameon=xs is not None,
                    handletextpad=0.1, borderaxespad=0.1, labelspacing=0.15, handlelength=1.0,
                    borderpad=0.2)
    if xs is not None:   # opaque, so the 50 % line does not run through the labels
        leg.get_frame().set(facecolor="white", edgecolor="none", alpha=1.0)
    fig.subplots_adjust(left=0.155, right=0.995, top=0.87, bottom=0.235)
    args.figdir.mkdir(parents=True, exist_ok=True)
    pdf = args.figdir / "fig_families.pdf"
    fig.savefig(pdf)
    fig.savefig(args.figdir / "fig_families.png", dpi=300)
    lines += ["", "Verdicts (proj_report.md): " + "; ".join(f"{k} {v}" for k, v in
                                                             num["verdicts"].items()), ""]
    (args.figdir / "fig_families_numbers.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
