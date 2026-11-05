"""Phase 5c report: the two claims, in tables.

Usage:  python scripts/phase5c_report.py [--dir out/phase5c] [--out out/phase5c/REPORT.md]
            [--boot 10000]

Outcome everywhere: comparisons until the delivered curve is within 0.5 Nm of the user's
best-profile distance and stays there (R25), censored at 150. Population users unless a table says stress.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prompref.experiment.phase5c_stats import (BUDGET, align_pairs, atypicality_fit,
                                               cell_summary, fmt_count, fmt_signed,
                                               paired_bootstrap, select_arm, survival)
from prompref.experiment.sweep import load_sweep
from prompref.user.jnd import true_jnd

OUTCOME = "sustained_050"
PAIR = ("tight", "t4")
ARMS = ("tight", "t8", "t4", "t2", "wide")
JND = true_jnd(0.5)
KNOBS = ("order_bias", "overdispersion", "plateau_nm", "drift_nm", "plateau_over_floor_jnd")
MISSING = "_not run yet._"


def _load(directory: Path, name: str) -> pd.DataFrame | None:
    path = directory / f"{name}.jsonl"
    if not path.exists():
        return None
    df = load_sweep(path)
    return df if len(df) else None


def _pop(df):
    return df[df["user_kind"] == "population"]


def _reference(df):
    mask = np.isclose(df["lapse"], 0.1) & np.isclose(df["s"], 0.5)
    for name in KNOBS:
        if name in df:
            mask &= df[name].fillna(0.0).to_numpy() == 0.0
    return df[mask]


def _cell(df, **fields):
    mask = np.ones(len(df), dtype=bool)
    for name, value in fields.items():
        mask &= (df[name] == value).to_numpy()
    return df[mask]


def _summary_row(label: str, group: pd.DataFrame, outcome: str = OUTCOME) -> str:
    s = cell_summary(group[outcome])
    if "g_candidate_floor" in group and group["g_candidate_floor"].notna().any():
        g_floor = f"{group['g_candidate_floor'].median():.3f}"
    else:
        g_floor = "—"
    return (f"| {label} | {s['n']} | {s['share_reached']:.2f} | {fmt_count(s['median'])} | "
            f"{fmt_count(s['p95'])} | {group['delivered'].median():.3f} | {g_floor} |")


SUMMARY_HEAD = ("| cell | n | reached | median | P95 | delivered (Nm, median) | "
                "G floor (Nm, median) |\n"
                "|---|---|---|---|---|---|---|")


def _summaries(df, by, outcome=OUTCOME) -> str:
    lines = [SUMMARY_HEAD]
    for keys, group in df.groupby(list(by), sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        lines.append(_summary_row(" / ".join(str(k) for k in keys), group, outcome))
    return "\n".join(lines)


def _ci(stat: dict, digits: int = 0) -> str:
    return (f"{fmt_signed(stat['estimate'], digits)} "
            f"[{fmt_signed(stat['lo'], digits)}, {fmt_signed(stat['hi'], digits)}]")


def _censor_note(stat: dict, n_boot: int) -> str:
    """The suffix " (k % of resamples censored)" when the estimate is finite but some
    bootstrap resamples were not (final-fix brief section 1); empty otherwise."""
    n_nonfinite = stat.get("n_nonfinite", 0)
    if not n_boot or not n_nonfinite:
        return ""
    return f" ({100.0 * n_nonfinite / n_boot:.0f} % of resamples censored)"


def _gated_ci(stat: dict, both_finite: bool, n_boot: int, digits: int = 0) -> str:
    """The estimate and CI, only when both sides' own point values are finite -- else
    "—". A censored side makes the difference/ratio itself meaningless (inf - x, inf -
    inf, or x / inf reading as "+0.00", i.e. "infinitely faster"; final-fix brief
    section 1)."""
    if not both_finite:
        return "—"
    return _ci(stat, digits) + _censor_note(stat, n_boot)


CONTRAST_HEAD = ("| contrast | users | median (first / second) | Δ median [95 % CI] | "
                 "P95 (first / second) | Δ P95 [95 % CI] | median ratio [95 % CI] | "
                 "Δ share reached [95 % CI] | Δ delivered, Nm [95 % CI] |\n"
                 "|---|---|---|---|---|---|---|---|---|")


def _contrast(label, df_a, df_b, n_boot, outcome=OUTCOME) -> str:
    # A cell can be incomplete (mid-sweep, or an error row dropped by load_sweep), so the
    # two frames are restricted to their common (fold, user_id) pairs before align_pairs --
    # which stays strict, raising on anything it is handed that still disagrees.
    keys_a = set(zip(df_a["fold"], df_a["user_id"]))
    keys_b = set(zip(df_b["fold"], df_b["user_id"]))
    common = keys_a & keys_b
    dropped = len(keys_a ^ keys_b)
    if not common:
        return f"| {label} | no paired users | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    a_df = df_a[[k in common for k in zip(df_a["fold"], df_a["user_id"])]]
    b_df = df_b[[k in common for k in zip(df_b["fold"], df_b["user_id"])]]
    a, b, strata = align_pairs(a_df, b_df, outcome)
    r = paired_bootstrap(a, b, strata, n_boot=n_boot)
    sa, sb = cell_summary(a), cell_summary(b)
    # Delivered Nm is never censored (it is the curve's own distance, not a reach count),
    # so it is paired and bootstrapped the same way but always renders a number.
    d_a, d_b, d_strata = align_pairs(a_df, b_df, "delivered")
    rd = paired_bootstrap(d_a, d_b, d_strata, n_boot=n_boot)
    users = f"{len(a)}" + (f" ({dropped} dropped)" if dropped else "")
    median_ok = bool(np.isfinite(sa["median"]) and np.isfinite(sb["median"]))
    p95_ok = bool(np.isfinite(sa["p95"]) and np.isfinite(sb["p95"]))
    return (f"| {label} | {users} | {fmt_count(sa['median'])} / {fmt_count(sb['median'])} | "
            f"{_gated_ci(r['diff_median'], median_ok, n_boot)} | "
            f"{fmt_count(sa['p95'])} / {fmt_count(sb['p95'])} | "
            f"{_gated_ci(r['diff_p95'], p95_ok, n_boot)} | "
            f"{_gated_ci(r['ratio_median'], median_ok, n_boot, 2)} | "
            f"{_ci(r['diff_share'], 2)} | "
            f"{_ci(rd['diff_median'], 3)} |")


def _claim1(main, n_boot) -> list[str]:
    out = ["## Claim 1: faster than the state of the art", ""]
    if main is None:
        return out + [MISSING, ""]
    out += ["G runs at one JND level (1.0). In 5b-ii the gate blocked nothing at all three "
            "R34 levels on five users at tight, and delivered curves were identical across "
            "levels (S4); that sample is five users, not this study's.", ""]
    ref = _pop(_reference(main))
    for arm in PAIR:
        cells = _cell(ref, arm=arm, family="promp_s0", prior="data")
        out += [f"### Arm `{arm}`, ProMP data prior", "",
                _summaries(cells, ["learner"]), ""]
        g = _cell(cells, learner="g")
        if len(g) and "g_candidate_floor" in g:
            reachable = float(np.mean(g["g_candidate_floor"] <= g["d_floor"] + 0.5))
            out += [f"G's best possible candidate (never report G without it): median "
                    f"{g['g_candidate_floor'].median():.3f} Nm; within floor + 0.5 Nm for "
                    f"{reachable:.2f} of users.", ""]
        a = _cell(cells, learner="a")
        lines = [CONTRAST_HEAD]
        for other in ("g", "d_z", "c2"):
            b = _cell(cells, learner=other)
            if len(a) and len(b):
                lines.append(_contrast(f"A − {other}", a, b, n_boot))
        out += lines + [""]
        share = {learner: survival(_cell(cells, learner=learner)[OUTCOME])
                 for learner in ("a", "g", "d_z", "c2") if len(_cell(cells, learner=learner))}
        out += ["Share not yet reached after n comparisons:", "",
                "| learner | 25 | 50 | 100 | 150 |", "|---|---|---|---|---|"]
        out += [f"| {learner} | " + " | ".join(f"{s[k]:.2f}" for k in (25, 50, 100, BUDGET)) + " |"
                for learner, s in share.items()]
        out += [""]
    return out


def _claim2(main) -> list[str]:
    out = ["## Claim 2: reliable convergence across users", ""]
    if main is None:
        return out + [MISSING, ""]
    ref = _reference(main)
    pop = _pop(ref)
    out += ["### Arms (A, ProMP data prior)", "",
            _summaries(_cell(pop, learner="a"), ["arm"]), "",
            "D_z keeps Phase 4's tuned multipliers in every arm (S13); it may be mistuned "
            "in t2 and wide.", ""]
    out += ["### R31 arm selection (select on 4 folds, evaluate on the fifth)", "",
            "| learner | recommended | per held-out fold (arm, P95) |", "|---|---|---|"]
    tie_notes = []
    for learner in ("a", "g", "d_z"):
        cells = _cell(pop, learner=learner)
        if len(cells["fold"].unique()) < 2:
            continue
        sel = select_arm(cells)
        folds = "; ".join(f"{fold}: {v['arm']} {fmt_count(v['p95'])}"
                          for fold, v in sel["per_fold"].items())
        out.append(f"| {learner} | `{sel['recommended']}` | {folds} |")
        arms_present = list(cells["arm"].unique())
        if arms_present and all(
                np.isinf(cell_summary(cells[cells["arm"] == arm][OUTCOME])["p95"])
                for arm in arms_present):
            tie_notes.append(f"`{learner}`: P95 > 150 in every arm — the selection is "
                             f"decided by the tie-breaks (median, then arm order), not by P95.")
    out += [""]
    for note in tie_notes:
        out += [note, ""]
    stress = ref[ref["user_kind"] == "stress"]
    if len(stress):
        out += ["### Stress users (comparisons to τ = 0.5 Nm; > 150 = never reached within "
                "budget, blank = no run for that cell)", "",
                "| learner | user | " + " | ".join(ARMS) + " | G floor (Nm) |",
                "|---|---|" + "---|" * len(ARMS) + "---|"]
        for (learner, uid), group in stress.groupby(["learner", "user_id"]):
            cells = []
            for arm in ARMS:
                v = group[group["arm"] == arm][OUTCOME]
                if v.empty:
                    cells.append("")
                else:
                    value = np.inf if pd.isna(v.iloc[0]) else float(v.iloc[0])
                    cells.append(fmt_count(value))
            if learner == "g" and "g_candidate_floor" in group and group["g_candidate_floor"].notna().any():
                g_floor = f"{group['g_candidate_floor'].median():.3f}"
            else:
                g_floor = ""
            out.append(f"| {learner} | {uid} | " + " | ".join(cells) + f" | {g_floor} |")
        out += [""]
    out += ["### Cost versus atypicality (A; x = reach radius R_0.5; fit on sessions "
            "that reached τ only; population and stress users)", "",
            "| arm | reached | censored | R² quadratic | R² log |", "|---|---|---|---|---|"]
    a = _cell(ref, learner="a")
    for arm in ARMS:
        cells = _cell(a, arm=arm)
        if len(cells):
            fit = atypicality_fit(cells["user_reach_050"], cells[OUTCOME])
            out.append(f"| {arm} | {fit['n']} | {fit['n_censored']} | {fit['quad_r2']:.2f} | "
                       f"{fit['log_r2']:.2f} |")
    return out + [""]


def _knob(row, name: str) -> float:
    """A knob value with NaN/None (column present, this row unset) treated as absent."""
    value = row.get(name, 0.0)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)


def _label(row) -> str:
    parts = []
    if not np.isclose(row["lapse"], 0.1):
        parts.append(f"lapse {row['lapse']:g}")
    if not np.isclose(row["s"], 0.5):
        parts.append(f"s {row['s']:g}")
    if _knob(row, "order_bias"):
        parts.append(f"order bias {_knob(row, 'order_bias'):g}·s")
    if _knob(row, "overdispersion"):
        parts.append(f"overdispersion {_knob(row, 'overdispersion'):g}")
    if _knob(row, "plateau_nm"):
        parts.append(f"plateau {_knob(row, 'plateau_nm') / JND:.0f} JND")
    if _knob(row, "plateau_over_floor_jnd"):
        parts.append(f"region floor + {_knob(row, 'plateau_over_floor_jnd'):g} JND")
    if _knob(row, "drift_nm"):
        parts.append(f"drift {_knob(row, 'drift_nm') / JND:.0f} JND")
    return ", ".join(parts) or "reference"


def _median_excess(drift_df, ref_df) -> float:
    """The median over paired users of ``tracking_error_jnd`` minus the reference cell's own
    ``(delivered - d_floor) / JND``, matching rows by ``(fold, user_id)``. ``ref_df`` is
    already restricted to the drift rows' own arm and learner. ``nan`` with no pairs."""
    ref_map = {(r.fold, r.user_id): (r.delivered - r.d_floor) / JND for r in ref_df.itertuples()}
    diffs = [r.tracking_error_jnd - ref_map[(r.fold, r.user_id)]
             for r in drift_df.itertuples() if (r.fold, r.user_id) in ref_map]
    return float(np.median(diffs)) if diffs else np.nan


def _noise(main, noise, region, n_boot) -> list[str]:
    out = ["## Noise, region and drift (claim 2's named assumptions)", ""]
    if noise is None or main is None:
        return out + [MISSING, ""]
    ref = _pop(_reference(_cell(main, family="promp_s0", prior="data")))
    ref = ref[ref["learner"].isin(["a", "g"]) & ref["arm"].isin(PAIR)]
    rows = pd.concat([ref, _pop(noise)] + ([_pop(region)] if region is not None else []),
                     ignore_index=True)
    rows["cell"] = rows.apply(_label, axis=1)
    out += [_summaries(rows, ["cell", "arm", "learner"]), ""]
    out += ["Paired against the reference cell, A:", "", CONTRAST_HEAD]
    for arm in PAIR:
        base = _cell(rows, cell="reference", arm=arm, learner="a")
        for cell in rows["cell"].unique():
            if cell == "reference":
                continue
            other = _cell(rows, cell=cell, arm=arm, learner="a")
            if len(other) and len(base):
                out.append(_contrast(f"{cell}, {arm}: cell − reference", other, base, n_boot))
    out += [""]
    # S16's region sat around the ideal, which the family cannot reach (change,
    # 2026-10-01): report it as inert rather than with a criterion that measures nothing.
    plateau = rows[rows["plateau_nm"].fillna(0.0) > 0]
    if len(plateau):
        for (cell, arm, learner), group in plateau.groupby(["cell", "arm", "learner"], sort=False):
            ref_map = {(r.fold, r.user_id): r.answers
                       for r in ref[(ref["arm"] == arm) & (ref["learner"] == learner)].itertuples()}
            matches = sum(1 for r in group.itertuples()
                         if ref_map.get((r.fold, r.user_id)) == r.answers)
            out += [f"`{cell}`, `{arm}`, `{learner}`: answers identical to the reference for "
                    f"{matches} of {len(group)} users — the region around the ideal is "
                    f"unreachable in this family (S16a).", ""]
    # S16a anchors the region on the family's own best instead.
    anchored = rows[rows["plateau_over_floor_jnd"].fillna(0.0) > 0] if (
        "plateau_over_floor_jnd" in rows) else rows.iloc[0:0]
    if len(anchored):
        out += ["### Preferred region anchored on the floor (S16a): region criterion", "",
                _summaries(anchored, ["cell", "arm", "learner"], outcome="region_sustained_050"),
                "", "R25 for the same sessions:", "",
                _summaries(anchored, ["cell", "arm", "learner"]), ""]
    drift = rows[rows["drift_nm"].fillna(0.0) > 0]
    if len(drift):
        out += ["### Drift (S17): final tracking error", "",
                "| cell | arm | learner | tracking error (JND, median) | "
                "tracking error ≤ 1 JND (absolute) | capped | "
                "excess over own reference (JND, median) |",
                "|---|---|---|---|---|---|---|"]
        for (cell, arm, learner), group in drift.groupby(["cell", "arm", "learner"]):
            ref_cell = ref[(ref["arm"] == arm) & (ref["learner"] == learner)]
            excess = _median_excess(group, ref_cell)
            out.append(f"| {cell} | {arm} | {learner} | {group['tracking_error_jnd'].median():.2f} | "
                       f"{float(np.mean(group['tracking_error_jnd'] <= 1.0)):.2f} | "
                       f"{int(group['drift_capped'].sum())} | "
                       f"{'—' if np.isnan(excess) else fmt_signed(excess, 2)} |")
        out += [""]
    return out


def _attribution(main, attribution, n_boot) -> list[str]:
    out = ["## Attribution (tight): family × prior × learner", "",
           "Comparisons to τ count from each cell's own floor, and flat and data cells measure "
           "from different floors and prior baselines (spec 4.6, T5): read the prior effect "
           "within a learner and the learner effect within a prior. Absolute delivered error "
           "is beside every row.", ""]
    if attribution is None or main is None:
        return out + [MISSING, ""]
    shared = _reference(_cell(main, family="promp_s0", prior="data", arm="tight"))
    shared = shared[shared["learner"].isin(["a", "g"])]
    new = attribution[attribution["learner"].isin(["a", "g"]) & (attribution["arm"] == "tight")]
    new = new[new["n_candidates"].fillna(2048) == 2048]
    rows = _pop(pd.concat([shared, new], ignore_index=True))
    out += [_summaries(rows, ["family", "prior", "learner"]), "", CONTRAST_HEAD]
    for family in ("promp_s0", "schaefer"):
        for learner in ("a", "g"):
            data = _cell(rows, family=family, prior="data", learner=learner)
            flat = _cell(rows, family=family, prior="flat", learner=learner)
            if len(data) and len(flat):
                out.append(_contrast(f"prior: data − flat ({family}, {learner})", data, flat, n_boot))
        for prior in ("data", "flat"):
            a = _cell(rows, family=family, prior=prior, learner="a")
            g = _cell(rows, family=family, prior=prior, learner="g")
            if len(a) and len(g):
                out.append(_contrast(f"learner: A − G ({family}, {prior})", a, g, n_boot))
    for learner in ("a", "g"):
        for prior in ("data", "flat"):
            p = _cell(rows, family="promp_s0", prior=prior, learner=learner)
            s = _cell(rows, family="schaefer", prior=prior, learner=learner)
            if len(p) and len(s):
                out.append(_contrast(f"family: ProMP − Schäfer ({learner}, {prior})", p, s, n_boot))
    return out + [""]


def _candidates(main, attribution, n_boot) -> list[str]:
    out = ["## G's candidate count (S7)", ""]
    if attribution is None or main is None:
        return out + [MISSING, ""]
    big = _pop(attribution[attribution["n_candidates"].fillna(0) > 2048])
    small = _pop(_reference(_cell(main, learner="g", family="promp_s0", prior="data")))
    ref = _pop(_reference(main))
    out += [SUMMARY_HEAD]
    for arm in PAIR:
        for label, df in (("2048", small), ("32768", big)):
            cells = _cell(df, arm=arm)
            if len(cells):
                out.append(_summary_row(f"{arm} / {label} (candidate floor "
                                        f"{cells['g_candidate_floor'].median():.3f} Nm)", cells))
    out += ["", CONTRAST_HEAD]
    for arm in PAIR:
        a, b = _cell(big, arm=arm), _cell(small, arm=arm)
        if len(a) and len(b):
            out.append(_contrast(f"{arm}: 32768 − 2048", a, b, n_boot))
    # Claim 1's secondary contrast (final-fix brief section 2): A against G at its largest
    # candidate count, not only at the published 2048.
    for arm in PAIR:
        a = _cell(ref, arm=arm, family="promp_s0", prior="data", learner="a")
        g = _cell(big, arm=arm, learner="g")
        if len(a) and len(g):
            out.append(_contrast(f"A − G(32768), {arm}", a, g, n_boot))
    return out + [""]


def _c1(main, attribution, n_boot) -> list[str]:
    out = ["## C1: Ramella as published (semi-trapezoid box)", ""]
    if attribution is None:
        return out + [MISSING, ""]
    c1 = _pop(_cell(attribution, learner="c1"))
    if not len(c1):
        return out + [MISSING, ""]
    out += [SUMMARY_HEAD, _summary_row("c1", c1), "",
            f"Median floor of the box: {c1['d_floor'].median():.3f} Nm — comparisons to τ "
            f"count from it, so read the absolute delivered error.", ""]
    # Claim 1's secondary contrast (final-fix brief section 2): A against C1 directly, at
    # tight, the arm C1 has no equivalent of.
    if main is not None:
        a = _cell(_pop(_reference(main)), arm="tight", family="promp_s0", prior="data",
                  learner="a")
        if len(a):
            out += [CONTRAST_HEAD, _contrast("A (tight) − C1", a, c1, n_boot), ""]
    return out


def _landmarks(main) -> list[str]:
    out = ["## Delivered curves with two bouts in order (S14)", ""]
    if main is None or "delivered_two_bout" not in main:
        return out + [MISSING, ""]
    shares = _pop(_reference(main)).groupby("learner")["delivered_two_bout"].mean()
    return out + ["| learner | share |", "|---|---|"] + [
        f"| {learner} | {share:.3f} |" for learner, share in shares.items()] + [
        "", "The torque-rate limit of R29 is not applied (decision P4).", ""]


def _other(title, df, by) -> list[str]:
    out = [f"## {title}", ""]
    if df is None:
        return out + [MISSING, ""]
    return out + [_summaries(_pop(_reference(df)), by), ""]


def _cost(main) -> list[str]:
    out = ["## Session cost (8 workers on a shared machine: ratios only)", ""]
    if main is None:
        return out + [MISSING, ""]
    medians = _pop(_reference(main)).groupby("learner")["session_s"].median()
    base = medians.get("a", np.nan)
    return out + ["| learner | median s / A's |", "|---|---|"] + [
        f"| {learner} | {value / base:.2f} |" for learner, value in medians.items()] + [""]


def _verdict(ok: bool) -> str:
    return "held." if ok else "failed."


def _predictions(main, noise, region) -> list[str]:
    out = ["## Predictions recorded before running (spec 5.4)", ""]
    if main is None or noise is None:
        return out + [MISSING, ""]
    ref = _pop(_reference(_cell(main, learner="a", arm="tight", family="promp_s0", prior="data")))
    a = _pop(_cell(noise, learner="a", arm="tight"))

    def cell(**fields):
        return _cell(a, **fields)

    def share(df, outcome=OUTCOME):
        return cell_summary(df[outcome])["share_reached"]

    def median(df):
        return cell_summary(df[OUTCOME])["median"]

    lapse03 = cell(lapse=0.3)
    s1 = cell(s=1.0)
    if len(lapse03):
        diff = share(ref) - share(lapse03)
        out.append(f"- **P4** (share falls ≤ 0.10 from lapse 0.1 to 0.3): "
                   f"{share(ref):.2f} → {share(lapse03):.2f}; {_verdict(diff <= 0.10)}")
    if len(s1):
        s1_extra = median(s1) - median(ref)
        extra = {name: median(cell(**{name: value})) - median(ref)
                 for name, value in (("order_bias", 0.3), ("overdispersion", 0.4))
                 if len(cell(**{name: value}))}
        held = bool(extra) and all(s1_extra > v for v in extra.values())
        out.append(f"- **P5** (each model break costs less than doubling s, "
                   f"{fmt_signed(s1_extra)}): "
                   + ", ".join(f"{k} {fmt_signed(v)}" for k, v in extra.items())
                   + f"; {_verdict(held)}")
    if region is None:
        out.append("P6: not run yet.")
    else:
        region_a = _pop(_cell(region, learner="a", arm="tight"))
        s16a = region_a[np.isclose(region_a["plateau_over_floor_jnd"].fillna(0.0), 1.0)]
        if len(s16a):
            region_share = share(s16a, "region_sustained_050")
            r25_share = share(s16a)
            held = region_share >= share(ref) and r25_share >= share(ref) - 0.10
            out.append(f"- **P6** (region share ≥ reference share; R25 share falls ≤ 0.10): "
                       f"region {region_share:.2f}, R25 {r25_share:.2f}, reference "
                       f"{share(ref):.2f}; {_verdict(held)}")
        else:
            out.append("P6: not run yet.")
    ref_error = float(np.median((ref["delivered"] - ref["d_floor"]) / JND))
    for multiple in (2, 4):
        drift = a[np.isclose(a["drift_nm"].fillna(0.0), multiple * JND)]
        if len(drift):
            excess = _median_excess(drift, ref)
            held = excess < 1.0 if multiple == 2 else excess > 1.0
            out.append(f"- **P7** ({multiple} JND drift): tracking error "
                       f"{drift['tracking_error_jnd'].median():.2f} JND against the reference's "
                       f"{ref_error:.2f} JND, excess {fmt_signed(excess, 2)}; {_verdict(held)}")
    return out + [""]


def build_report(directory: Path, n_boot: int = 10_000) -> str:
    directory = Path(directory)
    main = _load(directory, "main-promp")
    noise = _load(directory, "noise")
    region = _load(directory, "region")
    attribution = _load(directory, "attribution")
    lines = ["# Phase 5c report", "",
             "Outcome: comparisons until the delivered curve is within 0.5 Nm RMS of the "
             "user's floor and stays there, censored at 150 (\"> 150\"). Population users "
             "unless stated. Brackets are 95 % cluster-bootstrap intervals over users, "
             "stratified by fold.", "",
             "\"(k dropped)\" counts users present in one cell of a contrast but not the "
             "other; contrasts use only users present in both.", ""]
    lines += _claim1(main, n_boot)
    lines += _claim2(main)
    lines += _noise(main, noise, region, n_boot)
    lines += _attribution(main, attribution, n_boot)
    lines += _candidates(main, attribution, n_boot)
    lines += _c1(main, attribution, n_boot)
    lines += _landmarks(main)
    lines += _other("Dual-Gaussian family (block 4)", _load(directory, "main-dual"),
                    ["arm", "learner"])
    lines += _other("Baseline population (block 5; different users, not paired)",
                    _load(directory, "baseline"), ["arm", "learner"])
    lines += _cost(main)
    lines += _predictions(main, noise, region)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/phase5c")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.dir) / "REPORT.md"
    out.write_text(build_report(Path(args.dir), n_boot=args.boot), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
