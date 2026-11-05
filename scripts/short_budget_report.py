"""Short-budget report: every learner at 12, 24 and 40 comparisons.

Usage:  python scripts/short_budget_report.py [--dir5c out/phase5c] [--dir5d out/phase5d]
            [--out out/short-budget/REPORT.md] [--boot 10000]

The stored sessions run 150 comparisons and every learner is exact under truncation (B3), so
the outcome at k is read from each row's ``distances[k - 1]``: the RMS distance to the user's
ideal of the curve the learner would deliver after k comparisons. The exception is drift, which
moves with the session length; its short sessions come from the ``drift-short`` block.

Outcomes (B2), population users, JND_ref = 0.674 * 0.5 Nm in every cell:
  JNDs gained = (d_prior - delivered_k) / JND_ref; perceptibly better = gain >= 1;
  perceptibly worse = delivered_k - d_prior >= 1 JND_ref; near the best possible = delivered_k - floor <= 2 JND_ref;
  within tau = delivered_k - floor <= 0.5 Nm (the secondary threshold, not sustained).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase5c_report import JND, MISSING, _cell, _label, _load, _pop, _reference  # noqa: E402
from phase5d_report import EXPECTED_N, ROBUST, _match, _verdict  # noqa: E402

from prompref.experiment.phase5c_stats import BUDGET, paired_bootstrap  # noqa: E402

JND_REF = float(JND)
TAU = 0.5
KEY = ["fold", "user_id"]
BLOCKS_5C = ("main-promp", "main-dual", "noise", "attribution", "region", "baseline")
BLOCKS_5D = ("lens-franks", "lens-dual", "lens-noise", "stopping", "cmaes", "cmaes-robust",
             "drift-short")
DRIFT_LEARNERS = ("a", "g", "cmaes")
DRIFT_MULTS = (2, 4)
ARMS_OTHER = ("t8", "t4", "t2", "wide")
NAMES = {"a": "A", "g": "G", "d_z": "D_z", "c2": "C2", "cmaes": "SortCMA", "c1": "C1",
         "a_lens": "lens"}


# --------------------------------------------------------------------------- formatting

def _sg(value, digits: int = 3) -> str:
    """A signed number; a value that rounds to zero prints without a sign, never "+0.00"."""
    if value is None or not np.isfinite(value):
        if value is not None and np.isinf(value):
            return "+inf" if value > 0 else "-inf"
        return "n/a"
    if round(float(value), digits) == 0:
        return f"{0.0:.{digits}f}"
    return f"{value:+.{digits}f}"


def _num(value, digits: int = 3) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _ci(stat: dict, digits: int = 3) -> str:
    if not np.isfinite(stat["estimate"]):
        return "n/a"
    return f"{_sg(stat['estimate'], digits)} [{_sg(stat['lo'], digits)}, {_sg(stat['hi'], digits)}]"


# --------------------------------------------------------------------------- outcomes at k

def _popn(df):
    """Population users only: ``user_kind`` population and a ``/pop`` user id."""
    if df is None:
        return None
    df = _pop(df)
    return df[df["user_id"].str.contains("/pop", regex=False)]


def _n2048(df):
    """G's published candidate count; rows of other learners carry no count and pass."""
    if df is None or "n_candidates" not in df:
        return df
    count = pd.to_numeric(df["n_candidates"], errors="coerce").fillna(2048)
    return df[(count == 2048).to_numpy()]


def delivered_at(df, k: int) -> pd.Series:
    """The delivered distance after comparison k: ``distances[k - 1]`` of every row."""
    lengths = df["distances"].map(len)
    if (lengths < k).any():
        raise ValueError(f"{int((lengths < k).sum())} rows hold fewer than {k} comparisons")
    return pd.Series([float(d[k - 1]) for d in df["distances"]], index=df.index, dtype=float)


def outcomes(df, k: int) -> pd.DataFrame:
    """Per-row delivered distance at k and the B2 outcomes (fixed yardstick JND_ref)."""
    d_k = delivered_at(df, k)
    start = df["d_prior"].astype(float)
    gain = (start - d_k) / JND_REF
    over = d_k - df["d_floor"].astype(float)
    return pd.DataFrame({"d_k": d_k, "gain": gain, "better": gain >= 1.0 - 1e-12,
                         "worse": (d_k - start) >= JND_REF - 1e-12, "near": over <= 2 * JND_REF + 1e-12,
                         "tau": over <= TAU + 1e-12}, index=df.index)


def cell_stats(df, k: int) -> dict:
    n = len(df)
    if not n:
        nan = float("nan")
        return dict(n=0, median_start=nan, median_delivered=nan, median_gain=nan,
                    share_better=nan, share_worse=nan, share_near=nan, share_tau=nan)
    o = outcomes(df, k)
    return dict(n=n, median_start=float(df["d_prior"].median()),
                median_delivered=float(o["d_k"].median()), median_gain=float(o["gain"].median()),
                share_better=float(o["better"].mean()), share_worse=float(o["worse"].mean()),
                share_near=float(o["near"].mean()), share_tau=float(o["tau"].mean()))


STATS_HEAD = ("| cell | k | n | median delivered (Nm) | median JNDs gained | perceptibly better "
              "| perceptibly worse | within 2 JND of floor | within 0.5 Nm of floor |\n"
              "|---|---|---|---|---|---|---|---|---|")


def stats_row(label: str, df, k: int) -> str:
    s = cell_stats(df, k)
    return (f"| {label} | {k} | {s['n']} | {_num(s['median_delivered'])} | "
            f"{_num(s['median_gain'], 2)} | {_num(s['share_better'], 2)} | "
            f"{_num(s['share_worse'], 2)} | {_num(s['share_near'], 2)} | "
            f"{_num(s['share_tau'], 2)} |")


# --------------------------------------------------------------------------- paired contrasts

def _flags(mask) -> np.ndarray:
    """A boolean outcome in the form paired_bootstrap counts as a share: 1 = yes, nan = no."""
    return np.where(np.asarray(mask, dtype=bool), 1.0, np.nan)


def _share_ci(flag_a, flag_b, strata, n_boot) -> dict:
    return paired_bootstrap(_flags(flag_a), _flags(flag_b), strata, n_boot=n_boot)["diff_share"]


def _common(a, b):
    keys_a, keys_b = set(zip(a["fold"], a["user_id"])), set(zip(b["fold"], b["user_id"]))
    for name, df in (("first", a), ("second", b)):
        if df.duplicated(KEY).any():
            raise ValueError(f"the {name} cell holds more than one row per user")
    common = keys_a & keys_b

    def pick(df):
        keep = np.array([key in common for key in zip(df["fold"], df["user_id"])], dtype=bool)
        return df[keep].sort_values(KEY).reset_index(drop=True)

    return pick(a), pick(b), len(keys_a ^ keys_b)


def contrast(a, b, k: int, n_boot: int):
    """First cell minus second at comparison k, on the users both hold (None when none).

    ``diff_median``: difference of median delivered Nm with its 95 % cluster-bootstrap CI;
    ``diff_share``: difference in the share perceptibly better, same bootstrap.
    ``dropped`` counts users present in only one of the two cells."""
    a, b, dropped = _common(a, b)
    if not len(a):
        return None
    oa, ob = outcomes(a, k), outcomes(b, k)
    strata = a["fold"].to_numpy()
    return dict(n=len(a), dropped=dropped, med_a=float(oa["d_k"].median()),
                med_b=float(ob["d_k"].median()), share_a=float(oa["better"].mean()),
                share_b=float(ob["better"].mean()),
                diff_median=paired_bootstrap(oa["d_k"].to_numpy(), ob["d_k"].to_numpy(), strata,
                                             n_boot=n_boot)["diff_median"],
                diff_share=_share_ci(oa["better"], ob["better"], strata, n_boot))


CONTRAST_HEAD = ("| contrast | users | median delivered, Nm (first / second) | Δ median, Nm "
                 "[95 % CI] | perceptibly better (first / second) | Δ share [95 % CI] |\n"
                 "|---|---|---|---|---|---|")


def contrast_row(label: str, c) -> str:
    if c is None:
        return f"| {label} | no paired users | n/a | n/a | n/a | n/a |"
    users = f"{c['n']}" + (f" ({c['dropped']} dropped)" if c["dropped"] else "")
    return (f"| {label} | {users} | {c['med_a']:.3f} / {c['med_b']:.3f} | "
            f"{_ci(c['diff_median'])} | {c['share_a']:.2f} / {c['share_b']:.2f} | "
            f"{_ci(c['diff_share'], 2)} |")


def _contrast_rows(label, a, b, ks, n_boot) -> list[str]:
    if a is None or b is None or not len(a) or not len(b):
        missing = "first cell" if a is None or not len(a) else "second cell"
        return [f"| {label}, k = {k} | not run yet ({missing} missing) | n/a | n/a | n/a | n/a |"
                for k in ks]
    return [contrast_row(f"{label}, k = {k}", contrast(a, b, k, n_boot)) for k in ks]


# --------------------------------------------------------------------------- cell selection

class Rows:
    """Every block's rows, loaded once, with the cell filters 5c and 5d report on."""

    def __init__(self, dir5c, dir5d):
        self.c5 = {name: _load(Path(dir5c), name) for name in BLOCKS_5C}
        self.d5 = {name: _load(Path(dir5d), name) for name in BLOCKS_5D}

    def main(self, learner, family="promp_s0", arm="tight"):
        """5c's reference cell: ProMP (or dual-Gaussian), data prior, lapse 0.1, s 0.5."""
        block = self.c5["main-dual" if family == "dual_gauss" else "main-promp"]
        if block is None:
            return None
        cell = _n2048(_popn(_reference(_cell(block, learner=learner, family=family, arm=arm,
                                             prior="data"))))
        return cell if len(cell) else None

    def sortcma(self, family="promp_s0"):
        block = self.d5["cmaes"]
        if block is None:
            return None
        cell = _popn(_reference(_cell(block, family=family, prior="data")))
        return cell if len(cell) else None

    def lens(self, learner, block="lens-franks"):
        df = self.d5[block]
        if df is None:
            return None
        cell = _n2048(_popn(_reference(_cell(df, learner=learner))))
        return cell if len(cell) else None

    def c1(self):
        df = self.c5["attribution"]
        if df is None:
            return None
        cell = _popn(_cell(df, learner="c1", family="trapezoid", arm="box"))
        return cell if len(cell) else None

    def flat(self, learner):
        df = self.c5["attribution"]
        if df is None:
            return None
        cell = _n2048(_popn(_cell(df, learner=learner, family="promp_s0", arm="tight",
                                  prior="flat")))
        cell = cell[~_has(cell, "plateau_nm") & ~_has(cell, "drift_nm")] if len(cell) else cell
        return cell if len(cell) else None

    def sortcma_flat(self):
        df = self.d5["cmaes-robust"]
        if df is None:
            return None
        cell = _popn(_reference(_cell(df, family="promp_s0", prior="flat")))
        return cell if len(cell) else None


def _has(df, name):
    """Rows with a positive value in ``name`` (a column that is absent or null counts as 0)."""
    if name not in df:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0) > 0


# --------------------------------------------------------------------------- sections

def _claim1(r: Rows, ks) -> list[str]:
    out = ["## 1. Claim 1 on ProMP at k comparisons", "",
           "ProMP, `tight`, data prior, lapse 0.1, s = 0.5, population users. Truncated stored "
           "sessions (B3); a post-hoc analysis of cells fixed for 150 comparisons (B6).", ""]
    cells = [("A", r.main("a")), ("G", r.main("g")), ("D_z", r.main("d_z")),
             ("SortCMA", r.sortcma()), ("C2", r.main("c2"))]
    present = [(label, df) for label, df in cells if df is not None]
    c1 = r.c1()
    if not present and c1 is None:
        return out + [MISSING, ""]
    out += [STATS_HEAD]
    for label, df in present:
        out += [stats_row(label, df, k) for k in ks]
    if c1 is not None:
        # C1 runs in its own family (the semi-trapezoid box): its start and floor are the
        # box's, so only the absolute delivered distance is comparable.
        out += [f"| C1 | {k} | {len(c1)} | {float(delivered_at(c1, k).median()):.3f} | — | — | — "
                f"| — | — |" for k in ks]
    out += [""]
    if c1 is not None:
        out += ["C1 (Ramella as published) runs in its own family with a flat prior, so its start "
                "and floor differ from the other rows: read its median delivered Nm only, in "
                "absolute Nm; the gain and share columns are left blank for it.", ""]
    return out


def _lens_rows(r: Rows, ks, n_boot):
    franks, dual, promp_a = r.lens("a_lens"), r.lens("a_lens", "lens-dual"), r.main("a")
    out = []
    comparators = [("A direct", r.lens("a")), ("G", r.lens("g")), ("D_z", r.lens("d_z")),
                   ("SortCMA", r.sortcma("franks"))]
    for name, df in comparators:
        out += _contrast_rows(f"Franks: lens − {name}", franks, df, ks, n_boot)
    for name, learner in (("A direct", "a"), ("G", "g"), ("D_z", "d_z"), ("C2", "c2")):
        out += _contrast_rows(f"dual-Gaussian: lens − {name}", dual,
                              r.main(learner, family="dual_gauss"), ks, n_boot)
    out += _contrast_rows("ProMP: lens (Franks) − A", franks, promp_a, ks, n_boot)
    return out


def _contrasts(r: Rows, ks, n_boot) -> list[str]:
    out = ["## 2. Paired contrasts at k", "",
           "First cell minus second on the users both hold; \"(k dropped)\" counts users present "
           "in only one cell. The medians shown are the usual medians; Δ median and its interval "
           "come from the bootstrap, which takes the lower of the two middle values when the "
           "count is even, so for an even count Δ can differ from the shown medians' difference "
           "by a rounding step. Δ median is the difference of the two cells' median delivered "
           "distance (negative favours the first cell) with a 95 % cluster-bootstrap interval "
           "over users, stratified by fold. Δ share is the difference in the share perceptibly "
           "better (gain ≥ 1 JND_ref).", ""]
    a = r.main("a")
    cells = [(f"Franks: {name}", df) for name, df in (
        ("lens", r.lens("a_lens")), ("A direct", r.lens("a")), ("G", r.lens("g")),
        ("D_z", r.lens("d_z")), ("SortCMA", r.sortcma("franks")))]
    cells += [(f"dual-Gaussian: {name}", df) for name, df in (
        ("lens", r.lens("a_lens", "lens-dual")), ("A direct", r.main("a", family="dual_gauss")),
        ("G", r.main("g", family="dual_gauss")), ("D_z", r.main("d_z", family="dual_gauss")),
        ("C2", r.main("c2", family="dual_gauss")))]
    out += ["Outcomes per cell (the same columns as section 1; each family has its own floor "
            "and start):", "", STATS_HEAD]
    for label, df in cells:
        out += [stats_row(label, df, k) for k in ks] if df is not None else [
            f"| {label} | — | 0 | not run yet | — | — | — | — | — |"]
    out += ["", "Contrasts:", ""]
    rows = []
    for name, df in (("G", r.main("g")), ("D_z", r.main("d_z")), ("SortCMA", r.sortcma()),
                     ("C2", r.main("c2"))):
        rows += _contrast_rows(f"ProMP: A − {name}", a, df, ks, n_boot)
    rows += _lens_rows(r, ks, n_boot)
    return out + [CONTRAST_HEAD] + rows + [""]


def robust_cells(r: Rows):
    """A's robustness cells: (label, rows) pairs, the reference first, then the noise cells,
    the other arms and the baseline population (None when the reference is not run)."""
    ref, noise = r.main("a"), r.c5["noise"]
    if ref is None:
        return None
    named = [("reference", ref)]
    if noise is not None:
        cells = _popn(_cell(_cell(noise, learner="a", arm="tight"), family="promp_s0",
                            prior="data"))
        cells = cells[~(_has(cells, "drift_nm") | _has(cells, "plateau_nm")).to_numpy()]
        if len(cells):
            cells = cells.assign(_cell=cells.apply(_label, axis=1))
            named += [(label, cells[cells["_cell"] == label]) for label in cells["_cell"].unique()]
    named += [(f"arm `{arm}`", r.main("a", arm=arm)) for arm in ARMS_OTHER]
    base = r.c5["baseline"]
    if base is not None:
        named.append(("baseline population",
                      _n2048(_popn(_cell(base, learner="a", arm="tight", family="promp_s0")))))
    return named


def _robust_a(r: Rows, ks, n_boot) -> list[str]:
    out = ["## 3. Robustness of A at k", "",
           "Every cell is A on ProMP, `tight` arm unless the row says otherwise, paired against "
           "A's reference at the same k (Δ = cell − reference). Drift cells are not here: the "
           "drift path spreads the shift over the session length, so they are rerun at short "
           "budgets (section 8). 5c's S16 plateau cells (a region around the ideal, which this "
           "family cannot reach) are omitted: their sessions are identical to the reference, so "
           "every contrast is exactly zero.", ""]
    named = robust_cells(r)
    if named is None:
        return out + [MISSING, ""]
    ref = named[0][1]
    base = r.c5["baseline"]
    base_cell = named[-1][1] if base is not None else None
    out += ["Outcomes per cell:", "", STATS_HEAD]
    for label, df in named:
        out += [stats_row(label, df, k) for k in ks] if df is not None and len(df) else [
            f"| {label} | — | 0 | not run yet | — | — | — | — | — |"]
    rows = []
    for label, df in named[1:]:
        if label.startswith("baseline"):
            continue
        suffix = " (reference: `tight`)" if label.startswith("arm") else ""
        rows += _contrast_rows(label + suffix, df, ref, ks, n_boot)
    out += ["", "Paired contrasts (cell − reference):", "", CONTRAST_HEAD] + rows
    if base is None:
        out += ["", "Baseline population: not run yet."]
    else:
        out += ["", "Baseline population (block 5): different people under the same user ids, so "
                "it is **not paired** with the reference; the two cells' medians are compared as "
                "they stand, with no interval.", "",
                "| cell | k | users | median delivered, Nm | Δ median vs reference, Nm | "
                "perceptibly better | Δ share vs reference |", "|---|---|---|---|---|---|---|"]
        if len(base_cell):
            for k in ks:
                sb, sr = cell_stats(base_cell, k), cell_stats(ref, k)
                out.append(f"| baseline, A | {k} | {sb['n']} | {_num(sb['median_delivered'])} | "
                           f"{_sg(sb['median_delivered'] - sr['median_delivered'])} | "
                           f"{_num(sb['share_better'], 2)} | "
                           f"{_sg(sb['share_better'] - sr['share_better'], 2)} |")
        else:
            out.append("| baseline, A | — | 0 | — | — | — | — |")
    return out + [""]


def _region(r: Rows, ks, n_boot) -> list[str]:
    out = ["## 4. Preferred region (S16a) at k", "",
           "The region is every curve within `plateau_nm_used` = floor + j JND of the ideal "
           "(the region block's own field, per user). *Inside* is delivered(k) ≤ that radius; "
           "*within 0.5 Nm* is 5c's own criterion at k (the curve's distance to the region "
           "≤ τ), not sustained. The reference columns apply the same per-user radius to "
           "A's (or G's) reference session on the same users; the reference observer has no "
           "plateau, so the two sessions differ.", ""]
    region = r.c5["region"]
    if region is None:
        return out + [MISSING, ""]
    region = _popn(_cell(region, arm="tight", family="promp_s0"))
    levels = sorted(pd.to_numeric(region["plateau_over_floor_jnd"], errors="coerce").dropna()
                    .unique()) if len(region) else []
    levels = [v for v in levels if v > 0]
    if not levels:
        return out + [MISSING, ""]
    for level in levels:
        for k in ks:
            out += [f"### Region floor + {level:g} JND, k = {k}", "",
                    "| learner | n | inside | within 0.5 Nm | reference inside | reference within "
                    "0.5 Nm | Δ inside [95 % CI] | users dropped |", "|---|---|---|---|---|---|---|---|"]
            for learner in ("a", "g"):
                cell = region[(region["learner"] == learner).to_numpy()
                              & np.isclose(pd.to_numeric(region["plateau_over_floor_jnd"]), level)]
                ref = r.main(learner)
                if not len(cell) or ref is None:
                    continue
                cell, ref, dropped = _common(cell, ref)
                if not len(cell):
                    continue
                radius = cell["plateau_nm_used"].astype(float).to_numpy()
                d_cell, d_ref = delivered_at(cell, k).to_numpy(), delivered_at(ref, k).to_numpy()
                inside, near = d_cell <= radius, d_cell <= radius + TAU
                ref_inside, ref_near = d_ref <= radius, d_ref <= radius + TAU
                ci = _share_ci(inside, ref_inside, cell["fold"].to_numpy(), n_boot)
                out.append(f"| {NAMES[learner]} | {len(cell)} | {inside.mean():.2f} | "
                           f"{near.mean():.2f} | {ref_inside.mean():.2f} | {ref_near.mean():.2f} "
                           f"| {_ci(ci, 2)} | {dropped} |")
            out += [""]
    return out


def _sortcma_robust(r: Rows, ks, n_boot) -> list[str]:
    out = ["## 5. SortCMA robustness at k", "",
           "SortCMA on ProMP, data prior, paired against SortCMA's own reference at the same k "
           "(Δ = cell − reference). Drift is in section 8.", ""]
    rob, ref = r.d5["cmaes-robust"], r.sortcma()
    if rob is None or ref is None:
        return out + [MISSING, ""]
    data = _popn(_cell(rob, family="promp_s0", prior="data"))
    rows = []
    for label, fields, drift in ROBUST:
        if drift:
            continue
        cell = _match(data, fields)
        cell = cell[~_has(cell, "drift_nm").to_numpy()]
        rows += _contrast_rows(label, cell, ref, ks, n_boot)
    if not rows:
        return out + [MISSING, ""]
    return out + [CONTRAST_HEAD] + rows + [""]


def _lens_noise(r: Rows, ks, n_boot) -> list[str]:
    out = ["## 6. The lens under lapse 0.3, against G, at k", "",
           "Franks family, tight arm, lapse 0.3, same users.", ""]
    noise = r.d5["lens-noise"]
    if noise is None:
        return out + [MISSING, ""]
    cells = _popn(_cell(noise, family="franks", arm="tight"))
    cells = _n2048(cells[np.isclose(pd.to_numeric(cells["lapse"]), 0.3)])
    lens, g = cells[cells["learner"] == "a_lens"], cells[cells["learner"] == "g"]
    if not len(lens) or not len(g):
        return out + [MISSING, ""]
    out += [STATS_HEAD] + [stats_row(f"{NAMES[name]}, lapse 0.3", df, k)
                           for name, df in (("a_lens", lens), ("g", g)) for k in ks]
    return out + ["", CONTRAST_HEAD] + _contrast_rows("lens − G, lapse 0.3", lens, g, ks, n_boot) + [""]


def _prior(r: Rows, ks, n_boot) -> list[str]:
    out = ["## 7. Prior data at k: data − flat", "",
           "**T5 caveat:** the two priors' starts differ. A flat cell starts from the flat box's "
           "mean curve, which is far from the data prior's, so JNDs gained and the shares are "
           "measured from different starts (the *start* column) and both priors' floors are "
           "their own. Read the prior effect within a learner and the absolute delivered Nm "
           "beside it. A and G: 5c attribution, ProMP, `tight` (data cells from the main "
           "reference); SortCMA: 5d.", ""]
    pairs = [("A", r.main("a"), r.flat("a")), ("G", r.main("g"), r.flat("g")),
             ("SortCMA", r.sortcma(), r.sortcma_flat())]
    pairs = [(name, d, f) for name, d, f in pairs if d is not None and f is not None]
    if not pairs:
        return out + [MISSING, ""]
    for k in ks:
        out += [f"### k = {k}", "",
                "| learner | prior | n | start (Nm, median) | median delivered (Nm) | "
                "median JNDs gained | perceptibly better | perceptibly worse | "
                "within 2 JND of floor | within 0.5 Nm of floor |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, data, flat in pairs:
            for prior, df in (("data", data), ("flat", flat)):
                s = cell_stats(df, k)
                out.append(f"| {name} | {prior} | {s['n']} | {s['median_start']:.3f} | "
                           f"{s['median_delivered']:.3f} | {s['median_gain']:.2f} | "
                           f"{s['share_better']:.2f} | {s['share_worse']:.2f} | "
                           f"{s['share_near']:.2f} | {s['share_tau']:.2f} |")
        out += ["", CONTRAST_HEAD]
        out += [contrast_row(f"{name}: data − flat, k = {k}", contrast(data, flat, k, n_boot))
                for name, data, flat in pairs] + [""]
    return out


# ---- drift

def _drift_rows(r: Rows, learner, mult, k):
    ds = r.d5["drift-short"]
    if ds is None:
        return None
    cell = _popn(_cell(ds, learner=learner, family="promp_s0", arm="tight", prior="data"))
    cell = _n2048(cell)
    keep = (pd.to_numeric(cell["n_comparisons"]) == k).to_numpy() & np.isclose(
        pd.to_numeric(cell["drift_nm"]), mult * JND_REF)
    cell = cell[keep]
    return cell if len(cell) else None


def _drift_ref(r: Rows, learner):
    return r.sortcma() if learner == "cmaes" else r.main(learner)


def _drift_pair(r: Rows, learner, mult, k):
    """Tracking error at k, the learner's reference tracking error at the same k and the
    number of drift users with no reference: both indexed by (fold, user_id), common users."""
    cell, ref = _drift_rows(r, learner, mult, k), _drift_ref(r, learner)
    if cell is None or ref is None:
        return None
    cell, ref, dropped = _common(cell, ref)
    if not len(cell):
        return None
    index = pd.MultiIndex.from_frame(cell[KEY])
    track = pd.Series(cell["tracking_error_nm"].astype(float).to_numpy(), index=index)
    ref_track = pd.Series((delivered_at(ref, k) - ref["d_floor"].astype(float)).to_numpy(),
                          index=index)
    return track, ref_track, dropped


def _drift(r: Rows, ks) -> list[str]:
    out = ["## 8. Drift at k (drift-short)", "",
           "Sessions of n = k comparisons rerun with the whole shift spread over them (B4). "
           "Tracking error: the delivered distance after the last comparison minus the floor of "
           "the final, drifted ideal. Excess: tracking error minus the same learner's reference "
           "tracking error at the same k, `distances[k − 1] − d_floor` of the truncated "
           "150-comparison reference row of the same user; paired median.", ""]
    if r.d5["drift-short"] is None:
        return out + [MISSING, ""]
    out += ["| learner | drift (JND) | k | users | median tracking error (Nm) | "
            "median reference tracking (Nm) | excess over reference (Nm, paired median) | "
            "status |", "|---|---|---|---|---|---|---|---|"]
    for learner in DRIFT_LEARNERS:
        for mult in DRIFT_MULTS:
            for k in ks:
                cell = _drift_rows(r, learner, mult, k)
                if cell is None:
                    out.append(f"| {learner} | {mult} | {k} | — | — | — | — | not run yet |")
                    continue
                status = ("complete" if len(cell) >= EXPECTED_N
                          else f"partial: {len(cell)} of {EXPECTED_N}")
                pair = _drift_pair(r, learner, mult, k)
                if pair is None:
                    out.append(f"| {learner} | {mult} | {k} | 0 | — | — | — | {status}; "
                               f"no reference row for these users |")
                    continue
                track, ref_track, _ = pair
                if len(track) < len(cell):
                    status += f"; {len(cell) - len(track)} without a reference"
                out.append(f"| {learner} | {mult} | {k} | {len(track)} | {track.median():.3f} | "
                           f"{ref_track.median():.3f} | {_sg((track - ref_track).median())} | "
                           f"{status} |")
    return out + [""]


def _q_not_run(tag, text) -> str:
    return f"- **{tag}** ({text}): not run yet."


def _predictions(r: Rows) -> list[str]:
    out = ["## 9. Predictions recorded before drift-short ran (spec §4)", ""]
    q1 = ("Q1", "k = 24, 2 JND drift: A's paired median excess tracking error over its own "
                f"reference is below 1 JND_ref = {JND_REF:.3f} Nm")
    q2 = ("Q2", "k = 24, 2 and 4 JND drift: A's tracking error is below G's and below "
                "SortCMA's, paired median")
    q3 = ("Q3", f"k = 12, 4 JND drift: A's excess over its reference exceeds {JND_REF:.3f} Nm")

    a24 = _drift_pair(r, "a", 2, 24)
    if a24 is None:
        out.append(_q_not_run(*q1))
    else:
        track, ref_track, _ = a24
        excess = float((track - ref_track).median())
        out.append(f"- **Q1** ({q1[1]}): n = {len(track)}, excess {_sg(excess)} Nm "
                   f"(A {track.median():.3f}, reference {ref_track.median():.3f}); "
                   f"{_verdict(excess < JND_REF, len(track))}")

    parts, ns, missing = [], [], False
    for mult in DRIFT_MULTS:
        a = _drift_pair(r, "a", mult, 24)
        for other in ("g", "cmaes"):
            b = _drift_pair(r, other, mult, 24)
            if a is None or b is None:
                missing = True
                continue
            diff = (a[0] - b[0]).dropna()
            if not len(diff):
                missing = True
                continue
            parts.append((mult, other, float(diff.median()), len(diff)))
            ns.append(len(diff))
    if missing or not parts:
        out.append(_q_not_run(*q2))
    else:
        ok = all(d < 0 for _, _, d, _ in parts)
        text = "; ".join(f"{m} JND vs {NAMES[o]}: {_sg(d)} Nm (n = {n})" for m, o, d, n in parts)
        out.append(f"- **Q2** ({q2[1]}): A − other, paired median: {text}; {_verdict(ok, min(ns))}")

    a12 = _drift_pair(r, "a", 4, 12)
    if a12 is None:
        out.append(_q_not_run(*q3))
    else:
        track, ref_track, _ = a12
        excess = float((track - ref_track).median())
        out.append(f"- **Q3** ({q3[1]}): n = {len(track)}, excess {_sg(excess)} Nm "
                   f"(A {track.median():.3f}, reference {ref_track.median():.3f}); "
                   f"{_verdict(excess > JND_REF, len(track))}")
    return out + [""]


def _stopping(r: Rows, ks) -> list[str]:
    out = ["## 10. Stopping: sessions that stop before k", "",
           "Share of the stopping block's sessions whose `n_comparisons_run` is below k, among "
           "all sessions and among the sessions that stopped at all (fewer than "
           f"{BUDGET} comparisons). Stopping is not truncation-exact (the rule reads the "
           "session), so this is the only short-budget reading of it.", ""]
    stop = r.d5["stopping"]
    if stop is None:
        return out + [MISSING, ""]
    stop = _popn(stop)
    if not len(stop):
        return out + [MISSING, ""]
    head = ["learner", "family", "stop_sd", "sessions", f"stopped before {BUDGET}"]
    head += [f"below {k}" for k in ks] + [f"of stopped: below {k}" for k in ks]
    out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    stop = stop.assign(_sd=pd.to_numeric(stop["stop_sd"], errors="coerce"))
    for (learner, family, sd), g in stop.groupby(["learner", "family", "_sd"], sort=True):
        run = pd.to_numeric(g["n_comparisons_run"]).to_numpy()
        stopped = run < BUDGET
        cells = [f"{np.mean(run < k):.2f}" for k in ks]
        cells += [f"{np.mean(run[stopped] < k):.2f}" if stopped.any() else "—" for k in ks]
        out.append(f"| {learner} | {family} | {sd:g} | {len(g)} | {int(stopped.sum())} | "
                   + " | ".join(cells) + " |")
    return out + [""]


# --------------------------------------------------------------------------- the report

def _start_note(r: Rows) -> str:
    """How far the learners' own first curve is from d_prior, from SortCMA's stored rows (its
    delivered curve does not move during its first generation)."""
    cm = r.sortcma()
    if cm is None:
        return "The learners' own first curve differs from it slightly (not measured: no SortCMA rows)."
    gap = delivered_at(cm, 1) - cm["d_prior"].astype(float)
    return (f"The learners' own first curve is not exactly that curve (clipping and the trust "
            f"ball): SortCMA's distances[0] − d_prior has a median of {_sg(gap.median())} Nm on "
            f"ProMP (n = {len(gap)}), which is why \"worse\" is defined as 1 JND_ref above "
            f"the start, not any increase.")


def build_report(dir5c, dir5d, n_boot: int = 10_000, ks=(12, 24, 40, 150)) -> str:
    """Render the short-budget report. ``ks`` all appear in claim 1; the contrast, robustness,
    prior and stopping sections use the ks below the 150-comparison budget."""
    r = Rows(dir5c, dir5d)
    ks = tuple(int(k) for k in ks)
    short = tuple(k for k in ks if k < BUDGET)
    lines = ["# Short budgets: the study at 12, 24 and 40 comparisons", "",
             "Every learner is evaluated after k comparisons on the stored sessions: delivered "
             "distance = `distances[k − 1]`, the RMS distance of the curve it would deliver "
             "now to the user's ideal. Truncation is exact for every cell except drift (B3). "
             "Population users. Numbers from existing cells are a post-hoc analysis (B6); only "
             "the drift-short block is unseen (Q1–Q3).", "",
             f"JND_ref = {JND_REF:.3f} Nm (the reference wearer, s = 0.5) in every cell. JNDs "
             "gained = (start − delivered) / JND_ref; perceptibly better = gain ≥ 1; perceptibly "
             "worse = delivered at least 1 JND_ref above the start; near the best possible = within 2 JND_ref of the user's "
             "floor; within 0.5 Nm of the floor is the secondary threshold, not sustained. "
             f"Start = d_prior, the distance of the prior-mean curve. {_start_note(r)}", ""]
    lines += _claim1(r, ks)
    lines += _contrasts(r, short, n_boot)
    lines += _robust_a(r, short, n_boot)
    lines += _region(r, short, n_boot)
    lines += _sortcma_robust(r, short, n_boot)
    lines += _lens_noise(r, short, n_boot)
    lines += _prior(r, short, n_boot)
    lines += _drift(r, short)
    lines += _predictions(r)
    lines += _stopping(r, short)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir5c", default="out/phase5c")
    ap.add_argument("--dir5d", default="out/phase5d")
    ap.add_argument("--out", default="out/short-budget/REPORT.md")
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(Path(args.dir5c), Path(args.dir5d), n_boot=args.boot),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
