"""The README's animation and figures, in a light and a dark version, from the library and the paper's figure data.

Usage: python -B scripts/make_readme_assets.py [--data figures/data] [--dest assets]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from psypref import PHASE, Prior, PsyPref, SimulatedUser, demo_profile  # noqa: E402

THEMES = {
    "light": dict(bg="#ffffff", ink="#222222", muted="#666666", rule="#dddddd", blue="#1845FB", band="#8FA4FD",
                  orange="#F89C20", orange_text="#B86E00", generic="#9C9CA1", a="#BB5566", b="#0F9BD0", family="#333333"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9da7b3", rule="#30363d", blue="#8FA4FD", band="#3F57C8",
                 orange="#F89C20", orange_text="#F8B860", generic="#8b949e", a="#E58A9A", b="#33BBEE", family="#c9d1d9"),
}
#: Cost of a wider prior after 24 comparisons, 3-12 Nm, against the walking prior (Table I and scripts/icorr_amp_pw.py).
WIDTH_COST = {1.0: 0.0, 1.5: -2.2, 2.0: -5.8, 3.0: -21.4}
SUBJECT, PEAKS, SEED = "AB30", (8.0, 6.0), 1
NOTE = "One simulated session. The preferred profile is an assumption, known only in the simulation."


def read_csv(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) for v in vals])
        except ValueError:
            out[k] = vals
    return out


def style(ax, c) -> None:
    ax.set_facecolor(c["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(c["muted"])
    ax.tick_params(colors=c["muted"], labelsize=9)
    ax.xaxis.label.set_color(c["ink"])
    ax.yaxis.label.set_color(c["ink"])


def curve_axes(ax, c, ylim=(-14, 14)) -> None:
    style(ax, c)
    ax.axhline(0, color=c["rule"], lw=0.8, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Gait cycle (%)", fontsize=10)
    ax.set_ylabel("Hip torque (Nm), flexion up", fontsize=10)


def grab(fig) -> Image.Image:
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def belief(tuner: PsyPref, rng, n: int = 40) -> np.ndarray:
    post = tuner.learner.posterior
    idx = rng.choice(len(post.x), size=n, p=post.weights())
    return np.array([tuner.design.curve(tuner.space.to_theta(z)) for z in post.x[idx, :tuner.space.d]])


def run_session(n: int = 24) -> list[dict]:
    preferred = demo_profile(SUBJECT, PEAKS)
    prior = Prior.from_walking(peak_range=(3, 12), leave_out=SUBJECT)
    user = SimulatedUser(preferred, seed=SEED)
    tuner = PsyPref(prior, seed=SEED)
    rng = np.random.default_rng(SEED)
    steps = [dict(k=0, a=None, b=None, answer=None, band=belief(tuner, rng), best=tuner.identified().copy())]
    for k in range(1, n + 1):
        a, b = tuner.ask()
        answer = user.answer(a, b)
        tuner.tell(answer)
        steps.append(dict(k=k, a=a.copy(), b=b.copy(), answer=answer, band=belief(tuner, rng), best=tuner.identified().copy()))
    return [dict(s, preferred=preferred, generic=prior.generic, p=user.would_choose(s["best"], prior.generic)) for s in steps]


def draw_session(fig, axes, c, step, show, level) -> None:
    left, right = axes
    left.clear()
    right.clear()
    curve_axes(left, c)
    left.plot(PHASE, step["band"].T, color=c["band"], lw=0.5, alpha=0.35)
    left.plot(PHASE, step["generic"], color=c["generic"], lw=1.3, ls=(0, (4, 2)))
    left.plot(PHASE, step["preferred"], color=c["ink"], lw=1.6)
    if show != "update" and step["a"] is not None:
        wa = 3.0 if show == "answer" and step["answer"] == "A" else 1.5
        wb = 3.0 if show == "answer" and step["answer"] == "B" else 1.5
        left.plot(PHASE, step["a"], color=c["a"], lw=wa)
        left.plot(PHASE, step["b"], color=c["b"], lw=wb)
    left.plot(PHASE, step["best"], color=c["blue"], lw=2.4)
    k = step["k"]
    if k == 0:
        title = "Before the first comparison: the prior's mean"
    elif show == "pair":
        title = f"Comparison {k}: which feels better, A or B?"
    elif show == "answer":
        title = f"Comparison {k}: the user chooses {step['answer'] or 'neither'}"
    else:
        title = f"After {k} comparison{'s' if k > 1 else ''}, about {k * 50 / 60:.0f} min" if k > 1 else "After 1 comparison"
    left.set_title(title, color=c["ink"], fontsize=12, loc="left", fontweight="bold")
    h = [plt.Line2D([], [], color=c["blue"], lw=2.4), plt.Line2D([], [], color=c["ink"], lw=1.6),
         plt.Line2D([], [], color=c["generic"], lw=1.3, ls=(0, (4, 2))), plt.Line2D([], [], color=c["band"], lw=1.5)]
    labels = ["Identified assistance", "Preferred profile", "Generic assistance", "PsyPref's belief"]
    if show != "update" and step["a"] is not None:
        h += [plt.Line2D([], [], color=c["a"], lw=2), plt.Line2D([], [], color=c["b"], lw=2)]
        labels += ["A", "B"]
    left.legend(h, labels, loc="lower right", fontsize=7.5, frameon=False, labelcolor=c["ink"], ncol=2)

    style(right, c)
    xs = np.arange(41)
    right.plot(xs, level["best"], color=c["orange"], lw=2)
    right.plot(xs, level["psypref"], color=c["blue"], lw=2.4)
    right.axhline(level["best"][40], color=c["orange"], lw=0.8, ls=":")
    if k > 0:
        right.plot([k], [level["psypref"][k]], "o", color=c["blue"], ms=7)
        right.plot([k], [level["best"][k]], "o", color=c["orange"], ms=6)
    right.set_xlim(0, 40)
    right.set_ylim(40, 100)
    right.set_xlabel("Comparisons", fontsize=10)
    right.set_ylabel("Chosen over generic assistance (%)", fontsize=10)
    right.set_title("Means of 200 simulated users, 3–12 Nm", color=c["ink"], fontsize=10, loc="left")
    right.text(40, level["psypref"][40] + 2, "PsyPref", color=c["blue"], ha="right", fontsize=10, fontweight="bold")
    right.text(40, level["best"][40] - 7, "Strongest other optimizer", color=c["orange_text"], ha="right", fontsize=10, fontweight="bold")
    if k >= 3:
        right.annotate("3 comparisons reach\nits level after 40", xy=(3, level["best"][40]), xytext=(9, 47), color=c["ink"], fontsize=9,
                       arrowprops=dict(arrowstyle="->", color=c["muted"]))


def session_frames(fig, axes, c, steps, level) -> list[tuple[Image.Image, int]]:
    frames = []
    draw_session(fig, axes, c, steps[0], "update", level)
    frames.append((grab(fig), 1400))
    for s in steps[1:]:
        if s["k"] <= 5:
            prev_best = dict(s, best=steps[s["k"] - 1]["best"], band=steps[s["k"] - 1]["band"])
            draw_session(fig, axes, c, prev_best, "pair", level)
            frames.append((grab(fig), 800))
            draw_session(fig, axes, c, prev_best, "answer", level)
            frames.append((grab(fig), 700))
        draw_session(fig, axes, c, s, "update", level)
        frames.append((grab(fig), 700 if s["k"] <= 5 else 240))
    frames[-1] = (frames[-1][0], 2200)
    return frames


def prior_frames(fig, axes, c) -> list[tuple[Image.Image, int]]:
    left, right = axes
    z = np.random.default_rng(3).standard_normal((30, 8))
    frames = []

    def page(prior, width, title, note, ylim=(-14, 14), hold=1500, cost=None):
        space = prior.widened(width)._space()
        left.clear()
        right.clear()
        curve_axes(left, c, ylim)
        curves = np.array([space.design.curve(space.to_theta(space.clip_to_search_region(width * v))) for v in z])
        left.plot(PHASE, curves.T, color=c["band"], lw=0.9, alpha=0.7)
        left.set_title(title, color=c["ink"], fontsize=12, loc="left", fontweight="bold")
        style(right, c)
        widths = list(WIDTH_COST)
        for i, w in enumerate(widths):
            on = cost == w
            right.barh(i, WIDTH_COST[w], color=c["blue"] if on else c["rule"], height=0.6)
            right.text(0.6, i, f"{w:g}×", va="center", color=c["ink"], fontsize=10, fontweight="bold" if on else "normal")
            if w != 1.0:
                right.text(WIDTH_COST[w] - 0.6, i, f"{WIDTH_COST[w]:+.1f}".replace("-", "−"), va="center", ha="right", color=c["ink"], fontsize=10)
        right.set_xlim(-30, 5)
        right.set_ylim(len(widths) - 0.5, -0.5)
        right.set_yticks([])
        right.spines["left"].set_visible(False)
        right.set_xlabel("Points after 24 comparisons, 3–12 Nm", fontsize=10)
        right.set_title(note, color=c["ink"], fontsize=10, loc="left")
        frames.append((grab(fig), hold))

    base = Prior.from_walking(peak_range=(3, 12))
    page(base, 1.0, "What you know goes into the prior: walking data, 3–12 Nm", "What a wider prior costs", cost=1.0)
    page(base, 1.5, "Some doubt is cheap: 1.5 times wider", "What a wider prior costs", cost=1.5)
    page(base, 2.0, "Twice as wide", "What a wider prior costs", cost=2.0)
    page(base, 3.0, "Three times wider costs much", "What a wider prior costs", cost=3.0)
    page(Prior.from_walking(peak_range=(3, 35)), 1.0, "Device and protocol set the peaks: 3–35 Nm", "What a wider prior costs", ylim=(-37, 37))
    page(Prior.from_walking(level=12.0), 1.0, "Or fix them: only the shape is tuned, 12 Nm", "What a wider prior costs", hold=1800)
    return frames


def projection_frames(fig, axes, c) -> list[tuple[Image.Image, int]]:
    left, right = axes
    preferred = demo_profile(SUBJECT, PEAKS)
    prior = Prior.from_walking(peak_range=(3, 12), leave_out=SUBJECT)
    user = SimulatedUser(preferred, seed=SEED)
    tuner = PsyPref(prior, family="schaefer", seed=SEED)
    lens = tuner.learner
    frames = []
    for k in range(1, 5):
        wa, wb = lens.inner.propose()
        raw_a, raw_b = lens.inner_design.curve(wa), lens.inner_design.curve(wb)
        ta, tb = lens._fit(raw_a), lens._fit(raw_b)
        a, b = tuner.design.curve(ta), tuner.design.curve(tb)
        left.clear()
        right.clear()
        curve_axes(left, c)
        left.plot(PHASE, raw_a, color=c["a"], lw=1.0, ls=":")
        left.plot(PHASE, raw_b, color=c["b"], lw=1.0, ls=":")
        left.plot(PHASE, a, color=c["a"], lw=2.2)
        left.plot(PHASE, b, color=c["b"], lw=2.2)
        left.set_title(f"Comparison {k}, shown in Schäfer et al.'s family", color=c["ink"], fontsize=12, loc="left", fontweight="bold")
        h = [plt.Line2D([], [], color=c["ink"], lw=2.2), plt.Line2D([], [], color=c["ink"], lw=1.0, ls=":")]
        left.legend(h, ["The fit the user feels", "PsyPref's proposal"], loc="lower right", fontsize=8, frameon=False, labelcolor=c["ink"])
        right.axis("off")
        right.set_facecolor(c["bg"])
        for i, line in enumerate(("A study may need named parameters,", "such as a peak's timing.", "",
                                  "PsyPref proposes in its own weights.", "The user feels the best fit in the family.",
                                  "PsyPref learns from the fit.", "", "The result comes in the family's parameters.")):
            right.text(0, 0.92 - 0.1 * i, line, color=c["ink"], fontsize=10.5, transform=right.transAxes)
        frames.append((grab(fig), 1500))
        response = {"A": 0, "B": 1, None: 2}[user.answer(a, b)]
        lens.update(ta, tb, response)
    return frames


def write_gif(path: Path, frames: list[tuple[Image.Image, int]]) -> None:
    w, h = frames[0][0].size
    sheet = Image.new("RGB", (w, h * len(frames)))
    for i, (f, _) in enumerate(frames):
        sheet.paste(f, (0, h * i))
    pal = sheet.convert("P", palette=Image.ADAPTIVE, colors=160)
    imgs = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f, _ in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=[d for _, d in frames], loop=0, optimize=True, disposal=1)
    print(f"{path.name}: {len(frames)} frames, {sum(d for _, d in frames) / 1000:.0f} s, {path.stat().st_size / 2 ** 20:.1f} MB")


def animation(dest: Path, data: Path, steps: list[dict]) -> None:
    a = read_csv(data / "fig2_curves_a.csv")
    level = {"psypref": a["psypref"], "best": a["best"]}
    for theme, c in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[1.45, 1]), facecolor=c["bg"])
        fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.145, wspace=0.2)
        fig.text(0.06, 0.012, NOTE, color=c["muted"], fontsize=8)
        frames = session_frames(fig, axes, c, steps, level) + prior_frames(fig, axes, c) + projection_frames(fig, axes, c)
        write_gif(dest / f"session_{theme}.gif", frames)
        plt.close(fig)


def how_it_works(dest: Path, steps: list[dict]) -> None:
    for theme, c in THEMES.items():
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True, facecolor=c["bg"])
        fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.17, wspace=0.08)
        s0, s3, s24 = steps[0], steps[3], steps[24]
        titles = ("1  Prior from walking data", "2  Which feels better, A or B?", "3  The belief narrows", "4  Identified assistance")
        for ax, t in zip(axes, titles):
            curve_axes(ax, c)
            ax.set_title(t, color=c["ink"], fontsize=11, loc="left", fontweight="bold")
            ax.set_ylabel("")
        axes[0].set_ylabel("Torque (Nm)", fontsize=10)
        axes[0].plot(PHASE, s0["band"].T, color=c["band"], lw=0.6, alpha=0.6)
        axes[1].plot(PHASE, s3["a"], color=c["a"], lw=2.2, label="A")
        axes[1].plot(PHASE, s3["b"], color=c["b"], lw=2.2, label="B")
        axes[1].legend(loc="lower right", fontsize=9, frameon=False, labelcolor=c["ink"])
        axes[2].plot(PHASE, s24["band"].T, color=c["band"], lw=0.6, alpha=0.6)
        axes[3].plot(PHASE, s24["generic"], color=c["generic"], lw=1.3, ls=(0, (4, 2)), label="Generic")
        axes[3].plot(PHASE, s24["preferred"], color=c["ink"], lw=1.6, label="Preferred (simulation only)")
        axes[3].plot(PHASE, s24["best"], color=c["blue"], lw=2.4, label="Identified after 24")
        axes[3].legend(loc="lower right", fontsize=7, frameon=False, labelcolor=c["ink"])
        fig.savefig(dest / f"how_it_works_{theme}.png", dpi=150, facecolor=c["bg"])
        plt.close(fig)


def speed(dest: Path, data: Path) -> None:
    panels = read_csv(data / "fig2d_panels.csv")
    names = (("a", "3–12 Nm peaks"), ("b", "3–35 Nm peaks"), ("c", "Fixed 12 Nm peaks"))
    for theme, c in THEMES.items():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True, facecolor=c["bg"])
        fig.subplots_adjust(left=0.06, right=0.99, top=0.81, bottom=0.135, wspace=0.07)
        for i, (ax, (key, title)) in enumerate(zip(axes, names)):
            d = read_csv(data / f"fig2_curves_{key}.csv")
            style(ax, c)
            ax.fill_between(d["k"], d["psypref_lo"], d["psypref_hi"], color=c["band"], alpha=0.3, lw=0)
            ax.fill_between(d["k"], d["best_lo"], d["best_hi"], color=c["orange"], alpha=0.25, lw=0)
            ax.plot(d["k"], d["sch"], color=c["family"], lw=1.3, ls=(0, (3, 2)), label="CMA-ES on a family of 4–6 parameters")
            ax.plot(d["k"], d["gpbald"], color=c["muted"], lw=1.3, ls=(0, (6, 2)), label="GP-BALD (no answer model)")
            ax.plot(d["k"], d["best"], color=c["orange"], lw=2, label="Strongest other optimizer")
            ax.plot(d["k"], d["psypref"], color=c["blue"], lw=2.6, label="PsyPref")
            fk, lev = panels["first_k"][i], d["best"][-1]
            ax.annotate("", xy=(fk, lev), xytext=(40, lev), arrowprops=dict(arrowstyle="->", color=c["ink"], lw=1))
            ax.plot([fk, fk], [lev, 96], color=c["ink"], lw=0.8, ls=":")
            ax.text(fk + 0.8, 96, f"{fk:.0f} comparisons reach the level at 40", color=c["ink"], fontsize=9.5, fontweight="bold", va="center")
            ax.axhline(50, color=c["rule"], lw=0.8, zorder=0)
            ax.set_xlim(0, 40)
            ax.set_ylim(40, 100)
            ax.set_xlabel("Comparisons (about 50 s each)", fontsize=10)
            ax.set_title(title, color=c["ink"], fontsize=11, loc="left", fontweight="bold")
        axes[0].set_ylabel("Chosen over generic\nassistance (%)", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=4, fontsize=9.5, frameon=False, labelcolor=c["ink"])
        fig.savefig(dest / f"speed_{theme}.png", dpi=150, facecolor=c["bg"])
        plt.close(fig)


def robustness(dest: Path, data: Path) -> None:
    rows = read_csv(data / "fig4m_rows.csv")
    index = {k: i for i, k in enumerate(rows["key"])}
    show = (("MAIN", "Main condition"), ("R2", "Noise σ = 2 and 3 Nm (smaller lead)"), ("A7", "Fatigue"), ("B4", "Preferred profile drifts"),
            ("B1", "Weber's law"), ("R6", "Users from other datasets"), ("SO", "Shapes unlike the prior"),
            ("SC", "Shapes from the 4–6-parameter family"), ("C4T", "Six deviations at once"), ("B5", "Judging single features"),
            ("B8", "No single preferred profile"))

    def value(key, s):
        i = index[key]
        for part in ("in", "out"):
            v = rows[f"{part}_{s}"][i]
            if not np.isnan(v):
                return v, rows[f"{part}_lo_{s}"][i], rows[f"{part}_hi_{s}"][i], part == "in"
        return None

    for theme, c in THEMES.items():
        fig, ax = plt.subplots(figsize=(9.4, 4.6), facecolor=c["bg"])
        fig.subplots_adjust(left=0.33, right=0.985, top=0.87, bottom=0.125)
        style(ax, c)
        ax.axvline(0, color=c["muted"], lw=1)
        for y, (key, _) in enumerate(show):
            for s, off, marker in (("w", -0.14, "o"), ("f", 0.14, "s")):
                v = value(key, s)
                if v is None:
                    continue
                est, lo, hi, clear = v
                col = c["blue"] if clear else c["orange"]
                ax.plot([lo, hi], [y + off] * 2, color=col, lw=2, solid_capstyle="butt")
                ax.plot(est, y + off, marker, color=col, ms=5.5)
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels([label for _, label in show], fontsize=9.5, color=c["ink"])
        ax.set_ylim(len(show) - 0.5, -0.6)
        ax.set_xlim(-8, 22)
        ax.set_xlabel("PsyPref's lead over the strongest other optimizer after 24 comparisons (points)", fontsize=9.5)
        fig.text(0.012, 0.955, "Circles: 3–35 Nm. Squares: fixed 12 Nm. Bars: 95 % per row. Orange: not secure across all 75 cases.",
                 color=c["muted"], fontsize=9)
        fig.text(0.012, 0.91, "No single preferred profile: at 3–35 Nm, PsyPref and Arens' GP both reach the measure's maximum.",
                 color=c["muted"], fontsize=9)
        fig.savefig(dest / f"robustness_{theme}.png", dpi=150, facecolor=c["bg"])
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=ROOT / "figures" / "data")
    ap.add_argument("--dest", type=Path, default=ROOT / "assets")
    ap.add_argument("--no-animation", action="store_true")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    steps = run_session()
    how_it_works(args.dest, steps)
    speed(args.dest, args.data)
    robustness(args.dest, args.data)
    if not args.no_animation:
        animation(args.dest, args.data, steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
