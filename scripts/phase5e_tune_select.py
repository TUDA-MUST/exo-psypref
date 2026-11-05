"""Pick each comparator's setting for 24 comparisons.

Usage:
    python scripts/phase5e_tune_select.py --dir out/phase5e

Reads ``<dir>/e-tune.jsonl`` (loso-AB30, s = 1.0, ProMP). Per optimizer (SortCMA, G, D_z) the
winner is the setting with the lowest median ``distances[23]`` over the population users
(user_id contains "/pop"; the stress users are excluded); a tie goes to the lower P75. Writes
``<dir>/tuned.json`` (optimizer -> point fields), which ``--block e-tuned`` reads, and prints
the table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from prompref.experiment.phase5e_grid import TUNED_FIELDS
from prompref.experiment.sweep import _parse_rows

K = 24      # the short budget the comparators are tuned for: distances[23]


def _setting(row: dict) -> tuple:
    fields = TUNED_FIELDS[row["learner"]]
    return (row["learner"],) + tuple(float(row[f]) for f in fields)


def select(rows: list[dict]) -> tuple[dict, list[dict]]:
    """(tuned, table) from e-tune rows. ``table`` has one entry per setting: learner, the
    setting's fields, n, median, p75, and ``chosen``."""
    errors = sum(1 for r in rows if r.get("error"))
    if errors:
        print(f"ignoring {errors} error rows")
    errors_free = {}
    for row in rows:        # a retried point keeps its latest row
        if row.get("error") or row["learner"] not in TUNED_FIELDS:
            continue
        if "/pop" not in row["user_id"] or len(row["distances"]) < K:
            continue
        errors_free[(_setting(row), row["user_id"])] = float(row["distances"][K - 1])
    by_setting: dict[tuple, list[float]] = {}
    for (setting, _), value in errors_free.items():
        by_setting.setdefault(setting, []).append(value)
    table = [dict(learner=s[0], **dict(zip(TUNED_FIELDS[s[0]], s[1:])), n=len(v),
                  median=float(np.median(v)), p75=float(np.percentile(v, 75)), chosen=False)
             for s, v in by_setting.items()]
    tuned = {}
    for learner in TUNED_FIELDS:
        mine = [t for t in table if t["learner"] == learner]
        if not mine:
            raise SystemExit(f"no population rows for {learner}: run e-tune first")
        counts = {t["n"] for t in mine}
        if len(counts) != 1:
            print(f"warning: {learner}'s settings have different user counts {sorted(counts)}")
        best = min(mine, key=lambda t: (t["median"], t["p75"]))
        best["chosen"] = True
        tuned[learner] = {f: best[f] for f in TUNED_FIELDS[learner]}
        for f in ("cma_popsize",):
            if f in tuned[learner]:
                tuned[learner][f] = int(tuned[learner][f])
    return tuned, table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.dir)
    rows = _parse_rows(out / "e-tune.jsonl")
    if not rows:
        ap.error(f"no rows in {out / 'e-tune.jsonl'}: run --block e-tune first")
    tuned, table = select(rows)
    (out / "tuned.json").write_text(json.dumps(tuned, indent=2, sort_keys=True), encoding="utf-8")
    print(f"distances[{K - 1}] over population users, Nm")
    for t in sorted(table, key=lambda t: (t["learner"], t["median"], t["p75"])):
        fields = ", ".join(f"{f}={t[f]:g}" for f in TUNED_FIELDS[t["learner"]])
        print(f"  {t['learner']:>5} {fields:<34} n={t['n']:>3} median={t['median']:.4f} "
              f"p75={t['p75']:.4f} {'<- chosen' if t['chosen'] else ''}")
    print(f"wrote {out / 'tuned.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
