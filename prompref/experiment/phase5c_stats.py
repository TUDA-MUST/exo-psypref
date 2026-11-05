"""Statistics for Phase 5c.

A session's outcome is the number of comparisons until the delivered curve reached its
target and stayed there; ``None`` means it never did within the budget. Every session runs
the full budget, so censoring happens only at the budget: Kaplan-Meier then reduces to the
empirical share not yet reached, and a censored session is simply "more than 150" (R31).
Quantiles therefore treat it as infinity and pick an observed value (``inverted_cdf``), so
a quantile is either a real session count or infinity -- never an interpolation into it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

BUDGET = 150
ARM_ORDER = ("tight", "t8", "t4", "t2", "wide")


def times_array(values) -> np.ndarray:
    out = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out.append(np.inf)
        else:
            out.append(float(value))
    return np.asarray(out, dtype=float)


def _q(t: np.ndarray, q: float) -> float:
    return float(np.quantile(t, q, method="inverted_cdf"))


def _share(t: np.ndarray) -> float:
    return float(np.mean(np.isfinite(t)))


def cell_summary(values) -> dict:
    t = times_array(values)
    if t.size == 0:
        return dict(n=0, share_reached=np.nan, median=np.nan, p95=np.nan,
                    p95_over_median=np.nan)
    median, p95 = _q(t, 0.5), _q(t, 0.95)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = p95 / median if (np.isfinite(median) and median) else (np.inf if p95 else np.nan)
    return dict(n=int(t.size), share_reached=_share(t), median=median, p95=p95,
                p95_over_median=float(ratio))


def survival(values, budget: int = BUDGET) -> np.ndarray:
    t = times_array(values)
    return np.array([float(np.mean(t > k)) for k in range(budget + 1)])


def align_pairs(df_a: pd.DataFrame, df_b: pd.DataFrame, outcome: str):
    """Two cells' outcomes paired by (fold, user), with each pair's fold as its stratum."""
    key = ["fold", "user_id"]
    a = df_a.set_index(key)[outcome]
    b = df_b.set_index(key)[outcome]
    if a.index.duplicated().any() or b.index.duplicated().any():
        raise ValueError("a cell holds more than one row per user")
    if set(a.index) != set(b.index):
        raise ValueError(f"the two cells hold different users: {len(set(a.index) - set(b.index))} "
                         f"only in the first, {len(set(b.index) - set(a.index))} only in the second")
    b = b.loc[a.index]
    return (a.to_numpy(dtype=object), b.to_numpy(dtype=object),
            np.array([fold for fold, _ in a.index]))


def _pair_stats(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mx, my = _q(x, 0.5), _q(y, 0.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.array([mx - my, _q(x, 0.95) - _q(y, 0.95),
                         mx / my if my != 0 else np.nan, _share(x) - _share(y)])


def paired_bootstrap(a, b, strata, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Cluster bootstrap over users, stratified by fold: 95 % intervals for
    the difference in median and in P95, the ratio of medians and the difference in share
    reached, first cell minus (or over) second."""
    x, y = times_array(a), times_array(b)
    strata = np.asarray(strata)
    if not (x.shape == y.shape == strata.shape):
        raise ValueError("a, b and strata must have one entry per user")
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    rng = np.random.default_rng(seed)
    estimate = _pair_stats(x, y)
    boots = np.empty((int(n_boot), 4))
    for i in range(int(n_boot)):
        idx = np.concatenate([rng.choice(g, g.size) for g in groups])
        boots[i] = _pair_stats(x[idx], y[idx])
    out = {}
    for j, name in enumerate(("diff_median", "diff_p95", "ratio_median", "diff_share")):
        column = boots[:, j]
        lo, hi = np.nanpercentile(column, [2.5, 97.5], method="inverted_cdf") if np.any(~np.isnan(column)) else (np.nan, np.nan)
        out[name] = dict(estimate=float(estimate[j]), lo=float(lo), hi=float(hi),
                         n_nonfinite=int(np.sum(~np.isfinite(column))))
    return out


def select_arm(df: pd.DataFrame, outcome: str = "sustained_050") -> dict:
    """R31: for each held-out fold, the arm with the smallest P95 over the other folds (ties:
    smaller median, then arm order), and its P95 on the held-out fold. The recommended arm
    is the one chosen most often; a tie goes to the smaller pooled P95."""
    arms = [arm for arm in ARM_ORDER if arm in set(df["arm"])]

    def rank(frame, arm):
        s = cell_summary(frame[frame["arm"] == arm][outcome])
        return (s["p95"], s["median"], arms.index(arm))

    per_fold = {}
    for held in sorted(df["fold"].unique()):
        train = df[df["fold"] != held]
        best = min(arms, key=lambda arm: rank(train, arm))
        test = df[(df["fold"] == held) & (df["arm"] == best)]
        per_fold[held] = dict(arm=best, p95=cell_summary(test[outcome])["p95"])
    counts = Counter(choice["arm"] for choice in per_fold.values())
    top = max(counts.values())
    tied = [arm for arm in arms if counts.get(arm, 0) == top]
    return dict(per_fold=per_fold, recommended=min(tied, key=lambda arm: rank(df, arm)))


def _r2(design: np.ndarray, y: np.ndarray) -> float:
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    total = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum(residual**2) / total) if total > 0 else np.nan


def atypicality_fit(x, y) -> dict:
    """our design notes's cost-versus-atypicality test: comparisons to tau against the user's reach
    radius, fitted as a + b x^2 (Gaussian growth) and as a + b log x (t growth), on the
    sessions that reached it."""
    x = np.asarray(x, dtype=float)
    t = times_array(y)
    keep = np.isfinite(t) & (x > 0)
    xs, ts = x[keep], t[keep]
    ones = np.ones_like(xs)
    return dict(n=int(keep.sum()), n_censored=int(np.sum(~np.isfinite(t))),
                quad_r2=_r2(np.column_stack([ones, xs**2]), ts) if xs.size > 2 else np.nan,
                log_r2=_r2(np.column_stack([ones, np.log(xs)]), ts) if xs.size > 2 else np.nan)


def fmt_count(value) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"> {BUDGET}" if np.isinf(value) else f"{value:.0f}"


def fmt_signed(value, digits: int = 0) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    if np.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:+.{digits}f}"
