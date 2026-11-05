"""Gain over generic assistance in the wearer's own metric, for the Phase 5e observer cells.

Usage:  python scripts/phase5e_own_generic.py --dir5e out/phase5e [--k 24] [--boot 10000]
            [--out out/phase5e/own_generic.md]

Section 10 of the Phase 5e report gives the gain over generic assistance in the RMS distance to
the ideal, A's own model class. A landmark, asymmetric or familiar wearer does not feel that
distance. Here the generic assistance (the prior's mean curve, built exactly as
``prior_mean_distance`` builds it: 2,000 projected prior draws, seed 0) is scored by each
wearer's own distance, ``observer.own_distance``, and compared with ``own_distances[k - 1]`` of
the identified assistance. No session is run; the observers are rebuilt from the stored points.

Population users, as in the B7 tables. Gains are in true_jnd(1.0) = 0.674 Nm, the observer
cells' yardstick; for the landmark wearer one feature-JND answers like 0.674 Nm.
The lower decile and its CI are benchmark_report's (linear percentile, cluster bootstrap over
users within folds). Post hoc: added after the planned predictions were read.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_report import lower_decile  # noqa: E402
from prompref.experiment import phase5 as p5  # noqa: E402
from prompref.experiment.phase5 import Phase5Point  # noqa: E402
from prompref.user.jnd import true_jnd  # noqa: E402

FIELDS = [f.name for f in dataclasses.fields(Phase5Point)]
JND = true_jnd(1.0)
NAMES = {"a": "A", "g": "preference GP (G)", "d_z": "LineCoSpar", "cmaes": "SortCMA"}


def _design(point: Phase5Point):
    if point.family == "franks":
        return p5.arm_space(p5._franks_prior(point.run_dir, point.fold), point.arm).design
    return p5.arm_space(p5._prior(point.run_dir, point.fold, point.family), point.arm).design


_GENERIC: dict = {}


def generic_curve(point: Phase5Point) -> np.ndarray:
    """The prior's mean curve over projected draws, as ``prior_mean_distance`` averages it."""
    key = (point.run_dir, point.fold, point.family, point.arm)
    if key not in _GENERIC:
        design = _design(point)
        theta = design.sample_prior(np.random.default_rng(0), 2000)
        _GENERIC[key] = np.array([design.curve(design.project(t)) for t in theta]).mean(axis=0)
    return _GENERIC[key]


def cell_name(r: dict) -> str:
    if r["observer_metric"] == "landmark":
        pct = round(100 * (r["landmark_mag_fraction"] or 0.10))
        return f"landmark observer, {'static' if pct == 10 else 'walking'} JND ({pct} %)"
    if r["observer_metric"] == "weber":
        return "Weber observer"
    if r["asymmetry"] != 1.0:
        return f"asymmetry {r['asymmetry']:g}"
    return f"familiarity {r['familiarity_weight']:g}"


def optimizer_name(point: Phase5Point) -> str:
    if point.family == "franks" and point.learner in ("a_lens", "a"):
        return "lens" if point.learner == "a_lens" else "A direct"
    return NAMES.get(point.learner, point.learner)


def collect(dir5e: Path, k: int) -> dict:
    cells = defaultdict(list)
    with open(dir5e / "e-observer.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("error") or r["user_kind"] != "population":
                continue
            point = Phase5Point(**{f: r[f] for f in FIELDS if f in r})
            user = p5._users(point.run_dir, point.fold)[point.user_id]
            observer, _ = p5.build_observer(point, user, s_user=r["s_user"],
                                            plateau=point.plateau_nm)
            family = "Franks" if point.family == "franks" else "ProMP"
            cells[(cell_name(r), family, optimizer_name(point))].append(dict(
                fold=point.fold, own_generic=observer.own_distance(generic_curve(point)),
                own=r["own_distances"][k - 1], rms_generic=r["d_prior"],
                rms=r["distances"][k - 1]))
    return cells


def table(cells: dict, k: int, n_boot: int) -> list[str]:
    lines = [f"| cell | family | optimizer | users | median own distance of generic assistance "
             f"(Nm-eq.) | median own gain (Nm-eq.) | own Q≥1 | lower decile of own gain, JND "
             f"[95 % CI] | share worse than generic, own | share worse than generic, RMS |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells, key=lambda c: (c[0], c[1] != "ProMP", c[2] not in ("A", "lens"), c[2])):
        rows = cells[key]
        g_own = np.array([x["own_generic"] for x in rows])
        own = np.array([x["own"] for x in rows])
        gain = (g_own - own) / JND
        dec = lower_decile(gain, [x["fold"] for x in rows], n_boot)
        worse_rms = np.mean([x["rms"] > x["rms_generic"] for x in rows])
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(rows)} | {np.median(g_own):.3f} | "
                     f"{np.median(g_own - own):.3f} | {np.mean(gain >= 1.0 - 1e-12):.2f} | "
                     f"{dec['estimate']:+.2f} [{dec['lo']:+.2f}, {dec['hi']:+.2f}] | "
                     f"{np.mean(gain < 0):.2f} | {worse_rms:.2f} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir5e", type=Path, required=True)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    head = [f"# Gain over generic assistance in the wearer's own metric, k = {args.k}", "",
            "Post hoc (after the registered predictions were read). Population users. Generic "
            "assistance is the prior's mean curve; gains are in true_jnd(1.0) = 0.674 Nm "
            "(one feature-JND for the landmark wearer). Weber's own distance is the RMS "
            "distance, so its two last columns agree. Lower decile: linear percentile, cluster "
            "bootstrap over users within folds.", ""]
    text = "\n".join(head + table(collect(args.dir5e, args.k), args.k, args.boot)) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
