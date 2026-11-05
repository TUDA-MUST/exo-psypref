"""Benchmark report: learning curves, reliability at a budget and N90.

Usage:  python scripts/benchmark_report.py [--dir5c out/phase5c] [--dir5d out/phase5d]
            [--out out/benchmark/REPORT.md] [--csv out/benchmark/learning_curves.csv]
            [--boot 10000]

No session is run. Every number is read from the stored rows: the identified assistance after k
comparisons is ``distances[k - 1]`` (exact under truncation, B3). Population users;
JND_ref = 0.337 Nm in every cell. The cell selection is short_budget_report's.

Per user, with start = d_prior and floor = d_floor:
  gain_k = (start - delivered_k) / JND_ref;  Q>=1: gain >= 1;  Q>=2: gain >= 2;
  Q_opt: delivered_k - floor <= 2 JND_ref;
  share of the achievable improvement = (start - delivered_k) / (start - floor).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import short_budget_report as sb  # noqa: E402
from short_budget_report import JND_REF, Rows, _num, _sg, delivered_at  # noqa: E402

KMAX = 150
CURVE_KS = (1, 2, 4, 8, 12, 16, 20, 24, 30, 40, 150)
RELIABILITY_KS = (12, 24, 40, 150)
ROBUST_K = 24
LEVELS = (("q1", "Q≥1"), ("q2", "Q≥2"), ("qopt", "Q_opt"))
RELIABLE = 0.90
NO_ROOM = 1e-9
SEED = 20261007
TOL = 1e-12
CSV_COLUMNS = ["family", "optimizer", "label", "k", "n", "median", "q25", "q75"]
NOT_RUN = "not run yet"

#: Display names. Ids stay internal, including the CSV ``optimizer`` column.
LABELS = {"D_z": "LineCoSpar", "G": "preference GP (G)", "C2": "Ramella (C2)",
          "C1": "Ramella as published (C1)", "lens": "lens", "SortCMA": "SortCMA", "A": "A"}


def label(family: str, optimizer: str) -> str:
    """The paper's name for an optimizer; A is "A direct" outside ProMP."""
    if optimizer == "A" and family != "ProMP":
        return "A direct"
    return LABELS.get(optimizer, optimizer)


def robust_label(name: str) -> str:
    """The paper's name for one of A's robustness cells."""
    if name.startswith("arm `"):
        return "search region " + name[5:-1]
    if name == "baseline population":
        return "second population (not Schäfer-centred)"
    if name.startswith(("lapse", "s ", "order bias", "overdispersion")):
        return "measurement noise, " + name
    return name


# --------------------------------------------------------------------------- the levels

def _flags(delivered, start, floor, jnd: float = JND_REF) -> dict:
    """Gain, level flags and share for arrays of any matching shape (NaN share where
    start - floor <= 1e-9). ``jnd`` is the yardstick in Nm (Phase 5e passes each cell's own)."""
    gain = (start - delivered) / jnd
    room = start - floor
    ok = room > NO_ROOM
    share = np.where(ok, (start - delivered) / np.where(ok, room, 1.0), np.nan)
    return dict(gain=gain, q1=gain >= 1.0 - TOL, q2=gain >= 2.0 - TOL,
                qopt=(delivered - floor) <= 2 * jnd + TOL, share=share)


def levels_at(df, k: int, jnd: float = JND_REF) -> pd.DataFrame:
    """Per user at comparison k: gain (in ``jnd``), the three level flags and the share of the
    achievable improvement (NaN where start - floor <= 1e-9)."""
    delivered = delivered_at(df, k)
    f = _flags(delivered.to_numpy(), df["d_prior"].astype(float).to_numpy(),
               df["d_floor"].astype(float).to_numpy(), jnd)
    return pd.DataFrame({"delivered": delivered, **f}, index=df.index)


def lower_decile(gains, folds, n_boot: int, seed: int = SEED) -> dict:
    """10th percentile of the gains (np.percentile, linear) with a 95 % CI from a cluster
    bootstrap over users, resampled within each fold."""
    gains, folds = np.asarray(gains, dtype=float), np.asarray(folds)
    estimate = float(np.percentile(gains, 10))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(folds == f) for f in pd.unique(folds)]
    draws = np.concatenate([idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))]
                            for idx in groups], axis=1)
    boot = np.percentile(gains[draws], 10, axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(estimate=estimate, lo=float(lo), hi=float(hi))


def reliability(df, k: int, n_boot: int, jnd: float = JND_REF) -> dict:
    """Shares at the three levels, the lower decile of gain with its CI and the median share of
    the achievable improvement at comparison k."""
    o = levels_at(df, k, jnd)
    share = o["share"].dropna()
    return dict(n=len(df), q1=float(o["q1"].mean()), q2=float(o["q2"].mean()),
                qopt=float(o["qopt"].mean()),
                decile=lower_decile(o["gain"], df["fold"], n_boot),
                median_share=float(share.median()) if len(share) else float("nan"),
                n_excluded=int(len(o) - len(share)))


# --------------------------------------------------------------------------- N90

def share_curve(df, level: str, kmax: int = KMAX, jnd: float = JND_REF) -> np.ndarray:
    """Cross-sectional share of users meeting ``level`` ("q1", "q2" or "qopt") at k = 1…kmax."""
    lengths = df["distances"].map(len)
    if (lengths < kmax).any():
        raise ValueError(f"{int((lengths < kmax).sum())} rows hold fewer than {kmax} comparisons")
    matrix = np.array([d[:kmax] for d in df["distances"]], dtype=float)
    flags = _flags(matrix, df["d_prior"].astype(float).to_numpy()[:, None],
                   df["d_floor"].astype(float).to_numpy()[:, None], jnd)
    return flags[level].mean(axis=0)


def n90(shares) -> int | None:
    """Smallest k (1-based) whose cross-sectional share is at least 0.90, else None. It is not a
    per-user first hit and not sustained: a later dip below 0.9 does not move it."""
    for k, value in enumerate(shares, start=1):
        if value >= RELIABLE - TOL:
            return k
    return None


def fmt_n90(value) -> str:
    return f"> {KMAX}" if value is None else str(value)


# --------------------------------------------------------------------------- cells

def family_cells(r: Rows):
    """(family, [(optimizer, rows or None)]) for the claim-1 cells on the three families."""
    return [
        ("ProMP", [("A", r.main("a")), ("G", r.main("g")), ("D_z", r.main("d_z")),
                   ("SortCMA", r.sortcma()), ("C2", r.main("c2"))]),
        ("Franks", [("lens", r.lens("a_lens")), ("A", r.lens("a")), ("G", r.lens("g")),
                    ("D_z", r.lens("d_z")), ("SortCMA", r.sortcma("franks"))]),
        ("dual-Gaussian", [("lens", r.lens("a_lens", "lens-dual")),
                           ("A", r.main("a", family="dual_gauss")),
                           ("G", r.main("g", family="dual_gauss")),
                           ("D_z", r.main("d_z", family="dual_gauss")),
                           ("C2", r.main("c2", family="dual_gauss"))]),
    ]


def _present(df) -> bool:
    return df is not None and len(df) > 0


def _curve(df) -> np.ndarray:
    """Median, q25, q75 of the delivered distance at k = 1…KMAX: shape (KMAX, 3)."""
    lengths = df["distances"].map(len)
    if (lengths < KMAX).any():
        raise ValueError(f"{int((lengths < KMAX).sum())} rows hold fewer than {KMAX} comparisons")
    matrix = np.array([d[:KMAX] for d in df["distances"]], dtype=float)
    return np.percentile(matrix, [50, 25, 75], axis=0).T


def _curve_rows(family: str, optimizer: str, df) -> list[list]:
    curve = _curve(df)
    shown = label(family, optimizer)
    return [[family, optimizer, shown, k, len(df), *curve[k - 1]] for k in range(1, KMAX + 1)]


def learning_curves(dir5c, dir5d, rows: Rows | None = None) -> pd.DataFrame:
    """Median and quartiles of the delivered distance at every k = 1…150, per family and
    optimizer; C1 (its own family, absolute Nm) is included."""
    r = rows or Rows(dir5c, dir5d)
    out = []
    for family, cells in family_cells(r):
        out += [row for name, df in cells if _present(df) for row in _curve_rows(family, name, df)]
    c1 = r.c1()
    if _present(c1):
        out += _curve_rows("C1 (trapezoid)", "C1", c1)
    return pd.DataFrame(out, columns=CSV_COLUMNS)


# --------------------------------------------------------------------------- sections

def _band(median, q25, q75) -> str:
    return f"{median:.3f} [{q25:.3f}, {q75:.3f}]"


def _curve_table(family, cells) -> list[str]:
    head = ["optimizer", "n"] + [f"k = {k}" for k in CURVE_KS]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for optimizer, df in cells:
        name = label(family, optimizer)
        if not _present(df):
            out.append(f"| {name} | {NOT_RUN} |" + " |" * (len(head) - 2))
            continue
        curve = _curve(df)
        out.append(f"| {name} | {len(df)} | "
                   + " | ".join(_band(*curve[k - 1]) for k in CURVE_KS) + " |")
    return out + [""]


def _learning_section(r: Rows) -> list[str]:
    out = ["## 1. Learning curves", "",
           "The identified assistance is the profile the optimizer believes optimal after k "
           "pairwise comparisons. The error of the identified assistance is its RMS distance in "
           "Nm to the simulated user's ideal. Each cell gives the median error after k "
           "comparisons with the interquartile range [q25, q75] over the population users, one "
           "row per optimizer and one column per k; the file `learning_curves.csv` holds every "
           "k from 1 to 150.", ""]
    for family, cells in family_cells(r):
        out += [f"### {family}", ""] + _curve_table(family, cells)
    c1 = r.c1()
    out += ["### Ramella as published (C1)", "", "C1 runs in its own family, the semi-trapezoid "
            "box with a flat prior, so its generic assistance and optimum differ from the other "
            "rows'. Only the absolute error of the identified assistance in Nm is comparable; "
            "it has no gain, level or N90 here.", ""]
    out += _curve_table("C1 (trapezoid)", [("C1", c1)])
    return out


RELIABILITY_HEAD = ("| {first} | n | Q≥1 | Q≥2 | Q_opt | lower decile of gain, JND [95 % CI] "
                    "| median share of the achievable improvement |\n|---|---|---|---|---|---|---|")


def _reliability_row(name: str, df, k: int, n_boot: int) -> str:
    if not _present(df):
        return f"| {name} | {NOT_RUN} | — | — | — | — | — |"
    s = reliability(df, k, n_boot)
    d = s["decile"]
    excluded = f" ({s['n_excluded']} excluded)"
    return (f"| {name} | {s['n']} | {s['q1']:.2f} | {s['q2']:.2f} | {s['qopt']:.2f} | "
            f"{_sg(d['estimate'], 2)} [{_sg(d['lo'], 2)}, {_sg(d['hi'], 2)}] | "
            f"{_num(s['median_share'], 2)}{excluded} |")


def _reliability_section(r: Rows, n_boot: int) -> list[str]:
    out = ["## 2. Reliability at budget", "",
           "Per optimizer after k pairwise comparisons: the share of users at each quality "
           "level, the lower decile of JNDs gained (10th percentile, linear interpolation) "
           "with a 95 % cluster-bootstrap interval over users resampled within each fold, and "
           "the median share of the achievable improvement, (start − identified) / (start − "
           "optimum), where the start is the generic assistance (population-average start) "
           "and the optimum is the optimum within the parameterization. A user with start − "
           "optimum ≤ 1e-9 is left out of that last column only, and the count of excluded "
           "users is printed in the cell. Reading of the decile: \"after k comparisons, 90 % "
           "of users gain at least X JND\".", ""]
    for family, cells in family_cells(r):
        for k in RELIABILITY_KS:
            out += [f"### {family}, k = {k}", "", RELIABILITY_HEAD.format(first="optimizer")]
            out += [_reliability_row(label(family, name), df, k, n_boot)
                    for name, df in cells] + [""]
    return out


def _n90_section(r: Rows) -> list[str]:
    out = ["## 3. N90: comparisons to reliability", "",
           "N90 is the smallest k from 1 to 150 at which at least 90 % of users meet the level, "
           "read on the cross-sectional share at each k; it is not a per-user first hit and not "
           "sustained, so a share that dips back below 90 % later does not move it. \"> 150\" "
           "means no k reaches the level.", ""]
    for family, cells in family_cells(r):
        head = ["optimizer", "n"] + [f"N90 {shown}" for _, shown in LEVELS]
        out += [f"### {family}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for optimizer, df in cells:
            name = label(family, optimizer)
            if not _present(df):
                out.append(f"| {name} | {NOT_RUN} | — | — | — |")
                continue
            out.append(f"| {name} | {len(df)} | " + " | ".join(
                fmt_n90(n90(share_curve(df, key))) for key, _ in LEVELS) + " |")
        out.append("")
    return out


def _robust_section(r: Rows, n_boot: int) -> list[str]:
    out = [f"## 4. A's robustness cells at k = {ROBUST_K}", "",
           "A on ProMP in short_budget_report's robustness cells (measurement noise, "
           "search-region constraints, a second population), the reference first; the same "
           "columns as section 2. Each cell uses its own generic assistance and optimum.", "",
           RELIABILITY_HEAD.format(first="cell")]
    named = sb.robust_cells(r)
    if named is None:
        return out + [f"| reference | {NOT_RUN} | — | — | — | — | — |", ""]
    return out + [_reliability_row(robust_label(name), df, ROBUST_K, n_boot)
                  for name, df in named] + [""]


def _methods_note() -> list[str]:
    return ["## 5. Methods note", "",
            "The benchmark scores the identified assistance prospectively. In the words of spec "
            "B7: \"All are computed on the *identified assistance*: the profile the optimizer "
            "believes optimal after N comparisons, scored against the simulated user's ground "
            "truth\", with \"no post hoc selection of the best evaluated condition\".",
            "",
            f"Q≥1 and Q≥2 are relative to generic assistance (population-average start, "
            f"d_prior); Q_opt is relative to the optimum within the parameterization (the "
            f"user's floor). The simulated user answers with measurement noise (lapse and s); "
            f"JND_ref = {JND_REF:.3f} Nm, the just-noticeable difference of the reference "
            f"wearer, in every cell.",
            "",
            "Everything here is post hoc on cells fixed for 150 comparisons (B6). The drift block "
            "is the only unseen part, and this report does not cover it.", ""]


def build_report(dir5c, dir5d, n_boot: int = 10_000, rows: Rows | None = None) -> str:
    r = rows or Rows(dir5c, dir5d)
    lines = ["# Benchmark: learning curves, reliability at budget and N90", "",
             "Spec B7, population users, read from the stored sessions. The identified assistance "
             "after k comparisons is the curve the optimizer would deliver now; its error is "
             "`distances[k − 1]`, the RMS distance in Nm to the user's ideal.", ""]
    lines += _learning_section(r)
    lines += _reliability_section(r, n_boot)
    lines += _n90_section(r)
    lines += _robust_section(r, n_boot)
    lines += _methods_note()
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir5c", default="out/phase5c")
    ap.add_argument("--dir5d", default="out/phase5d")
    ap.add_argument("--out", default="out/benchmark/REPORT.md")
    ap.add_argument("--csv", default="out/benchmark/learning_curves.csv")
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    rows = Rows(Path(args.dir5c), Path(args.dir5d))
    out, csv = Path(args.out), Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(args.dir5c, args.dir5d, n_boot=args.boot, rows=rows),
                   encoding="utf-8")
    learning_curves(args.dir5c, args.dir5d, rows=rows).to_csv(csv, index=False)
    print(f"wrote {out} and {csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
