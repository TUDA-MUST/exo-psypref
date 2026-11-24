"""Fig. 4 of the ICORR paper (v2), variant 4C: a compact Fig. S1 (4M) for the paper's page budget.

Usage:
  python scripts/paper_v2_fig4c.py [--dest figures/data]

Post hoc, and no new computation: reads 4M's fig4m_rows.csv and fig4m_meta.csv (scripts/paper_v2_fig4m.py) and
keeps its values, intervals, colours and label positions. Changes the layout only:
1. One row per group of assumptions, with one marker per 4M row of that group whose cells are all clear
   (light: simultaneous 95 % interval above zero). These rows carry no bars; Fig. S1 has every interval.
2. 4M's rows with a cell that is not clear (blue, or both at 95 %) stay as rows of their own, indented under
   their group, with 4M's bars and value labels; the main condition and the pessimistic user stay bold.
The horizontal layout (axis limits, label column, value-label x positions) is 4M's, so its labels stay valid;
where a row has both value labels on the left, they move apart vertically.

Writes to <dest>: fig4c_rows.csv (4M's columns, new y, is_header = 1 for group rows), fig4c_pts.csv (the group
rows' markers: y, x_w, x_f), fig4c_meta.csv (4M's meta with the new ymax, n_rows and height).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "figures" / "data"
GROUPS = ("G1", "G2", "G3", "G4")
ROW = 1.05      # distance between rows (row units), as in 4M
GROUP_GAP = 0.15  # extra space above a group row
END = 0.72      # space below the last row, as in 4M
LEGEND_ROWS = 1.3  # one legend line above the key
OWN_GAP = 0.45  # extra space above a row of its own, so that its value labels clear the row above


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(x: str) -> float:
    return float(x) if x not in ("", "nan", "-") else math.nan


def clear(row: dict) -> bool:
    """All existing cells light: no blue cell and no cell at the cap."""
    return all(row[f"blue_{s}"] in ("0", "") and math.isnan(num(row[f"cap_{s}"])) for s in ("w", "f"))


def split_labels(row: dict, meta: dict) -> dict:
    """Both value labels left of their bars stack too closely in 4C's denser layout: move them apart vertically
    (3-35 Nm up, fixed 12 Nm down, in pt); OWN_GAP leaves the room."""
    if row["pos_w"] == "3" and row["pos_f"] == "3":
        row = {**row, "vdy_w": "1.6", "vdy_f": "-1.6"}
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()
    rows = read(args.dest / "fig4m_rows.csv")
    meta = read(args.dest / "fig4m_meta.csv")[0]

    # 4M's order: MAIN, then each group header followed by its rows; the last row is the pessimistic user
    groups, order, cur = {}, [], None
    for r in rows:
        if r["is_header"] == "1":
            cur = r["key"]
            groups[cur] = []
            order.append(("group", r))
        elif cur is None or r is rows[-1]:
            order.append(("solo", r))
        else:
            groups[cur].append(r)

    out, pts, y = [], [], 0.0
    for kind, r in order:
        if kind == "solo":
            if out:
                y += ROW + (GROUP_GAP + OWN_GAP if r is rows[-1] else 0.0)
            out.append({**r, "y": f"{y:g}"})
            continue
        y += ROW + GROUP_GAP
        out.append({**r, "y": f"{y:g}"})
        for m in groups[r["key"]]:
            if clear(m):
                pts.append({"key": m["key"], "y": f"{y:g}", "x_w": m["in_w"], "x_f": m["in_f"]})
        for m in groups[r["key"]]:
            if not clear(m):
                y += ROW + OWN_GAP
                out.append(split_labels({**m, "y": f"{y:g}"}, meta))

    head = list(rows[0].keys())
    with (args.dest / "fig4c_rows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, head)
        w.writeheader()
        w.writerows(out)
    with (args.dest / "fig4c_pts.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["key", "y", "x_w", "x_f"])
        w.writeheader()
        w.writerows(pts)
    ymax = y + END
    pitch = num(meta["pitch"])
    meta["ymin"] = f"{num(meta['ymin']) - LEGEND_ROWS:g}"  # one legend line above 4M's key
    meta.update(ymax=f"{ymax:g}", n_rows=str(len(out)),
                height=f"{num(meta['bottom']) + (ymax - num(meta['ymin'])) * pitch:.3f}")
    with (args.dest / "fig4c_meta.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(meta))
        w.writeheader()
        w.writerow(meta)
    print(f"fig4c: {len(out)} rows, {len(pts)} group markers, ymax {ymax:g}; "
          + "; ".join(f"{r['key']} y={r['y']}" for r in out))


if __name__ == "__main__":
    main()
