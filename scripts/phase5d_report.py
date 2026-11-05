"""Phase 5d report: the lens, stopping and SortCMA, beside 5c's reference rows.

Usage:  python scripts/phase5d_report.py [--dir out/phase5d] [--dir5c out/phase5c]
            [--out out/phase5d/REPORT.md] [--boot 10000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase5c_report import (CONTRAST_HEAD, JND, MISSING, OUTCOME, _cell, _contrast,  # noqa: E402
                            _load, _pop, _reference, _summaries)

from prompref.designs.franks import FRANKS_NAMES, franks_parameters  # noqa: E402
from prompref.experiment.phase5c_stats import cell_summary, fmt_signed  # noqa: E402

BLOCKS_5D = ("lens-franks", "lens-dual", "lens-noise", "stopping", "cmaes", "cmaes-robust")
ROBUST = (("lapse 0.3", {"lapse": 0.3}, False), ("s 1.0", {"s": 1.0}, False),
          ("overdispersion 0.4", {"overdispersion": 0.4}, False),
          ("order bias 0.3·s", {"order_bias": 0.3}, False),
          ("drift 2 JND", {"drift_nm": 2 * JND}, True), ("drift 4 JND", {"drift_nm": 4 * JND}, True))
KEY = ["fold", "user_id"]
#: franks_parameters reports peak times unwrapped (extension may be < 0, flexion > 100).
WRAPPED = ("ext_peak_time", "flex_peak_time", "zero_mid")
#: Population users a complete cell holds (5 folds x 40); a verdict on fewer is provisional (M3).
EXPECTED_N = 200


def _circular_median(values, period: float = 100.0) -> float:
    """The value, on the circle of length ``period``, with the smallest summed arc distance to
    all ``values`` (taken among the values themselves), in [0, period)."""
    v = np.mod(np.asarray(values, dtype=float), period)
    d = np.abs(v[:, None] - v[None, :])
    return float(v[int(np.argmin(np.minimum(d, period - d).sum(axis=1)))])


def _wrap_centred(values, period: float = 100.0) -> np.ndarray:
    """``values`` wrapped into the ``period``-wide window centred on their circular median (M4).
    The centre is the circular median's representative nearest the plain median of the
    unwrapped values, so extension peaks near heel strike read as -3 .. 4, not 97 .. 4."""
    v = np.asarray(values, dtype=float)
    centre = _circular_median(v, period)
    centre += period * np.round((np.median(v) - centre) / period)
    start = centre - period / 2.0
    return start + np.mod(v - start, period)


def _match(df, fields):
    mask = np.ones(len(df), dtype=bool)
    for name, value in fields.items():
        mask &= np.isclose(pd.to_numeric(df[name]).to_numpy(float), value)
    return df[mask]


def _paired(a, b, col_a, col_b=None):
    """Per-user a[col_a] - b[col_b] on the users both cells hold."""
    col_b = col_b or col_a
    m = a.set_index(KEY)[[col_a]].join(b.set_index(KEY)[[col_b]], lsuffix="_a", rsuffix="_b", how="inner")
    return m.iloc[:, 0].to_numpy(float) - m.iloc[:, 1].to_numpy(float)


def _median(x) -> float:
    return float(np.median(x)) if len(x) else np.nan


def _share(df) -> float:
    return cell_summary(df[OUTCOME])["share_reached"]


#: The columns _summaries and _contrast read; everything else is dropped before stacking cells.
_STACK_COLS = KEY + ["learner", "family", OUTCOME, "delivered", "d_floor"]


def _stack(frames):
    """Stack cells on the columns the summaries use. Outcome columns that are all-missing in a
    cell (SortCMA never reaches) become float NaN so the stack has one dtype; g_candidate_floor
    is kept only on frames that carry it (G rows)."""
    parts = []
    for f in frames:
        if f is None or not len(f):
            continue
        part = f[[c for c in _STACK_COLS if c in f]].copy()
        part[OUTCOME] = pd.to_numeric(part[OUTCOME], errors="coerce").astype(float)
        if "g_candidate_floor" in f and f["g_candidate_floor"].notna().any():
            part["g_candidate_floor"] = pd.to_numeric(f["g_candidate_floor"], errors="coerce").astype(float)
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_STACK_COLS)


def _common(a, b):
    """Both cells restricted to the users they share, in the same order."""
    keys = set(zip(a["fold"], a["user_id"])) & set(zip(b["fold"], b["user_id"]))

    def pick(df):
        keep = df[np.array([k in keys for k in zip(df["fold"], df["user_id"])], dtype=bool)]
        return keep.sort_values(KEY).reset_index(drop=True)

    return pick(a), pick(b)


def _cmp(a, b):
    """Paired comparison of two cells on their common users, or None when none are shared."""
    a, b = _common(a, b)
    if not len(a):
        return None
    return dict(n=len(a), share_a=_share(a), share_b=_share(b),
                med_a=float(a["delivered"].median()), med_b=float(b["delivered"].median()))


def _finite(*values) -> bool:
    return all(v is not None and np.isfinite(v) for v in values)


def _ref5c(c5, learner, family="promp_s0"):
    main = c5.get("main-promp" if family == "promp_s0" else "main-dual")
    if main is None:
        return None
    return _pop(_reference(_cell(main, learner=learner, arm="tight", family=family, prior="data")))


def _lens_franks(d5, n_boot):
    out = ["## Lens on Franks (L2, L3)", ""]
    lens, cm = d5.get("lens-franks"), d5.get("cmaes")
    if lens is None:
        return out + [MISSING, ""]
    rows = _stack([_pop(_reference(lens)),
                   None if cm is None else _pop(_reference(_cell(cm, family="franks")))])
    out += [_summaries(rows, ["learner"]), "", CONTRAST_HEAD]
    a = _cell(rows, learner="a_lens")
    for other in ("a", "g", "d_z", "cmaes"):
        b = _cell(rows, learner=other)
        if len(a) and len(b):
            out.append(_contrast(f"a_lens − {other}", a, b, n_boot))
    return out + [""]


def _cost(d5, c5, n_boot):
    out = ["## Cost of interpretability (P8): a_lens on Franks against A on ProMP, same users", ""]
    lens, promp = d5.get("lens-franks"), _ref5c(c5, "a")
    if lens is None or promp is None:
        return out + [MISSING, ""]
    a = _pop(_reference(_cell(lens, learner="a_lens")))
    return out + [CONTRAST_HEAD, _contrast("a_lens (Franks) − A (ProMP)", a, promp, n_boot), "",
                  _ceiling_text(d5), ""]


def _lens_ceiling(d5):
    """(median, n) over the population a_lens users of the lens best-profile level minus the Franks best-profile distance,
    both read from the Franks sidecars at the rows' run_dir; None when no sidecar holds it.

    The lens best-profile level (``scripts/phase5d_franks.py``) is the distance from the ideal of the Franks
    fit to the user's best ProMP curve: what the lens delivers if A holds its best curve."""
    lens = d5.get("lens-franks")
    if lens is None or "run_dir" not in lens:
        return None
    rows = _pop(_reference(_cell(lens, learner="a_lens"))).drop_duplicates(["run_dir"] + KEY)
    gaps = []
    for (run_dir, fold), group in rows.groupby(["run_dir", "fold"]):
        path = Path(run_dir) / "users" / f"{fold}-franks.json"
        if not path.exists():
            continue
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        for uid in group["user_id"]:
            entry = sidecar.get(uid, {})
            if "lens_ceiling_nm" in entry:
                gaps.append(float(entry["lens_ceiling_nm"]) - float(entry["floor_nm"]))
    return (float(np.median(gaps)), len(gaps)) if gaps else None


def _ceiling_text(d5) -> str:
    ceiling = _lens_ceiling(d5)
    if ceiling is None:
        return ("Lens ceiling: not available (no Franks sidecar with `lens_ceiling_nm` at the rows' "
                "run_dir; rerun scripts/phase5d_franks.py).")
    return (f"Lens ceiling: if A held each user's best ProMP curve, the lens would deliver its "
            f"Franks fit, a median {fmt_signed(ceiling[0], 3)} Nm above the user's Franks floor "
            f"(n = {ceiling[1]} population users). Part of any P8 gap is this, not learning.")


def _franks_params(d5):
    out = ["## Delivered Franks parameters (a_lens, population users)", ""]
    lens = d5.get("lens-franks")
    if lens is None:
        return out + [MISSING, ""]
    rows = _pop(_reference(_cell(lens, learner="a_lens")))
    if not len(rows):
        return out + [MISSING, ""]
    params = pd.DataFrame([franks_parameters(np.asarray(b, dtype=float)) for b in rows["best"]])
    for name in WRAPPED:
        params[name] = _wrap_centred(params[name])
    out += ["| parameter | median | IQR | spread |", "|---|---|---|---|"]
    for name in FRANKS_NAMES:
        q1, med, q3 = np.percentile(params[name], [25, 50, 75])
        if name in WRAPPED:
            spread = f"{q3 - q1:.1f} %GC"
        else:
            spread = f"{(q3 - q1) / abs(med):.2f} × |median|" if med else "—"
        out.append(f"| {name} | {med:.2f} | {q1:.2f} – {q3:.2f} | {spread} |")
    return out + ["", "Peak times and the zero-torque midpoint are wrapped, per row, into the "
                  "100 %GC window centred on that row's circular median, so a timing that "
                  "straddles heel strike reads as, say, -3 – 4 %GC rather than 97 – 4; values may "
                  "therefore lie below 0 or above 100 %GC.", "",
                  "Spread is the plain IQR in %GC for the three timing rows and IQR / |median| for "
                  "magnitudes and durations. Franks et al. (2021) found optimal timings consistent "
                  "across people and magnitudes varying; compare timing spreads only with one "
                  "another, and relative spreads only with one another, because the two kinds do "
                  "not share a scale.", ""]


def _lens_fitting(d5):
    out = ["## Lens fitting (a_lens; spec §6)", ""]
    lens, dual, noise, stop = (d5.get(k) for k in ("lens-franks", "lens-dual", "lens-noise", "stopping"))
    cells = []
    if lens is not None:
        cells.append(("Franks", _pop(_reference(_cell(lens, learner="a_lens")))))
    if dual is not None:
        cells.append(("dual-Gaussian", _pop(_reference(_cell(dual, learner="a_lens")))))
    if noise is not None:
        cells.append(("Franks, lapse 0.3", _pop(_cell(noise, learner="a_lens"))))
    if stop is not None:
        cells.append(("Franks, stop_sd 0.3", _pop(_match(_cell(stop, family="franks", learner="a_lens"),
                                                         {"stop_sd": 0.3}))))
    cells = [(label, df) for label, df in cells if len(df) and "lens_n_fits" in df]
    if not cells:
        return out + [MISSING, ""]
    out += ["| cell | sessions | shown residual, median (Nm) | shown residual, P95 (Nm) | "
            "fits per session (median) | fit time per session (median, s) | "
            "proposed vs shown, median (Nm) | proposed vs shown, P95 (Nm) | "
            "clipped projections (total; sessions with any) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for label, df in cells:
        col = lambda name: (pd.to_numeric(df[name], errors="coerce").dropna()  # noqa: E731
                            if name in df else pd.Series(dtype=float))
        resid, fits, secs = col("lens_shown_residual_median"), col("lens_n_fits"), col("lens_fit_s")
        gap_med, gap_p95 = col("lens_proposed_shown_rms_median"), col("lens_proposed_shown_rms_p95")
        clipped = col("lens_projection_clipped")
        med = lambda x, d: f"{np.median(x):.{d}f}" if len(x) else "—"  # noqa: E731
        p95 = f"{np.percentile(resid, 95):.3f}" if len(resid) else "—"
        clip = f"{int(clipped.sum())}; {int((clipped > 0).sum())}" if len(clipped) else "—"
        out.append(f"| {label} | {len(df)} | {med(resid, 3)} | {p95} | {med(fits, 0)} | {med(secs, 2)} | "
                   f"{med(gap_med, 3)} | {med(gap_p95, 3)} | {clip} |")
    return out + ["", "Shown residual: the RMS distance of each shown curve from the ProMP span; the "
                  "table gives the median over sessions of each session's median, and the 95th "
                  "percentile over sessions of that. Proposed vs shown: the RMS between the ProMP "
                  "curve A proposed and the surface curve shown, over every shown duel end; the "
                  "table gives the median over sessions of each session's median and of each "
                  "session's 95th percentile. Clipped projections: projections of shown curves that "
                  "the ProMP box clipped before A was told them (spec §8).", ""]


def _lens_dual(d5, c5, n_boot):
    out = ["## Lens on dual-Gaussian, against block 4", ""]
    lens = d5.get("lens-dual")
    if lens is None or c5.get("main-dual") is None:
        return out + [MISSING, ""]
    a = _pop(_reference(lens))
    out += [CONTRAST_HEAD]
    for other in ("a", "g", "d_z"):
        b = _ref5c(c5, other, family="dual_gauss")
        if b is not None and len(b):
            out.append(_contrast(f"a_lens − {other} (dual-Gaussian)", a, b, n_boot))
    return out + [""]


def _lens_noise(d5, n_boot):
    out = ["## Lens under noise (lapse 0.3)", ""]
    noise, lens = d5.get("lens-noise"), d5.get("lens-franks")
    if noise is None or lens is None:
        return out + [MISSING, ""]
    out += [CONTRAST_HEAD]
    for learner in ("a_lens", "g"):
        hi = _pop(_cell(noise, learner=learner))
        ref = _pop(_reference(_cell(lens, learner=learner)))
        if len(hi) and len(ref):
            out.append(_contrast(f"{learner}: lapse 0.3 − lapse 0.1", hi, ref, n_boot))
    return out + [""]


def _stop_cells(d5, c5):
    stop, lens = d5.get("stopping"), d5.get("lens-franks")
    cells = []
    if stop is None:
        return cells
    for sd in sorted(pd.to_numeric(stop["stop_sd"]).dropna().unique()):
        for family, learner in (("promp_s0", "a"), ("franks", "a_lens")):
            full = _ref5c(c5, "a") if learner == "a" else (
                None if lens is None else _pop(_reference(_cell(lens, learner="a_lens"))))
            s = _pop(_match(_cell(stop, family=family, learner=learner), {"stop_sd": sd}))
            if len(s) and full is not None and len(full) and len(_common(s, full)[0]):
                cells.append((family, learner, float(sd), s, full))
    return cells


def _stop_stats(s, full) -> dict:
    m = s.set_index(KEY)[["n_comparisons_run", "delivered", "d_floor"]].join(
        full.set_index(KEY)[["delivered"]], rsuffix="_full", how="inner")
    premature = (m["delivered"] > m["d_floor"] + 0.5) & (m["delivered_full"] <= m["d_floor"] + 0.5)
    used = float(m["n_comparisons_run"].median())
    return dict(n=len(m), used=used, saved=1.0 - used / 150.0,
                delta=float((m["delivered"] - m["delivered_full"]).median()),
                premature=float(premature.mean()))


def _stopping(d5, c5):
    out = ["## Stopping (L9; Slade et al. 2024: \"convergence criteria represent a substantial "
           "open problem\")", ""]
    cells = _stop_cells(d5, c5)
    if not cells:
        absent = [name for name, got in (("5d stopping.jsonl", d5.get("stopping")),
                                         ("5c main-promp.jsonl", c5.get("main-promp")),
                                         ("5d lens-franks.jsonl", d5.get("lens-franks"))) if got is None]
        if d5.get("stopping") is not None and not absent:
            return out + ["_not run yet: the stopping block shares no users with the full-run cells._", ""]
        return out + [f"_not run yet: missing {', '.join(absent)}._", ""]
    out += ["| cell | users | comparisons used (median) | saved | Δ delivered vs full (Nm, median) "
            "| premature stops |", "|---|---|---|---|---|---|"]
    for family, learner, sd, s, full in cells:
        st = _stop_stats(s, full)
        out.append(f"| {learner} on {family}, stop_sd {sd:g} | {st['n']} | {st['used']:.0f} | "
                   f"{st['saved']:.0%} | {fmt_signed(st['delta'], 3)} | {st['premature']:.2f} |")
    return out + [""]


def _cmaes_field(d5, c5, n_boot):
    out = ["## SortCMA against the field (ProMP, tight; L10)", ""]
    cm = d5.get("cmaes")
    if cm is None or c5.get("main-promp") is None:
        return out + [MISSING, ""]
    c = _pop(_reference(_cell(cm, family="promp_s0")))
    refs = {learner: _ref5c(c5, learner) for learner in ("a", "g", "d_z", "c2")}
    rows = _stack([c] + list(refs.values()))
    out += [_summaries(rows, ["learner"]), "", CONTRAST_HEAD]
    for learner, ref in refs.items():
        if ref is not None and len(ref):
            out.append(_contrast(f"{learner} − cmaes", ref, c, n_boot))
    return out + [""]


def _excess(cell, ref, drift) -> tuple[float, int]:
    """Paired median increase of ``cell`` over ``ref``, and the number of paired users."""
    if drift:
        r = ref.assign(_err=ref["delivered"] - ref["d_floor"])
        d = _paired(cell, r, "tracking_error_nm", "_err")
    else:
        d = _paired(cell, ref, "delivered")
    return _median(d), len(d)


def _robust_deltas(d5, c5) -> dict:
    """Per condition: (SortCMA increase, A increase, SortCMA users, A users)."""
    rob, cm, noise = d5.get("cmaes-robust"), d5.get("cmaes"), c5.get("noise")
    a_ref = _ref5c(c5, "a")
    if rob is None or cm is None:
        return {}
    c_ref = _pop(_reference(_cell(cm, family="promp_s0")))
    data = _cell(rob, prior="data")
    out = {}
    for label, fields, drift in ROBUST:
        c_cell = _pop(_match(data, fields))
        a = (np.nan, 0)
        if noise is not None and a_ref is not None:
            a_cell = _pop(_match(_cell(noise, learner="a", arm="tight"), fields))
            if len(a_cell):
                a = _excess(a_cell, a_ref, drift)
        c = _excess(c_cell, c_ref, drift) if len(c_cell) else (np.nan, 0)
        out[label] = (c[0], a[0], c[1], a[1])
    return out


def _cmaes_robust(d5, c5):
    out = ["## SortCMA under noise and drift (L12; Slade et al.: CMA-ES \"is tolerant of both "
           "measurement noise and human adaptation\")", ""]
    deltas = _robust_deltas(d5, c5)
    if not deltas:
        return out + [MISSING, ""]
    out += ["Paired median increase over each learner's own reference cell (delivered Nm; drift "
            "cells: final tracking error against the reference's delivered − floor).", "",
            "| condition | SortCMA | A | paired users (SortCMA / A) |", "|---|---|---|---|"]
    out += [f"| {label} | {fmt_signed(c, 3)} | {fmt_signed(a, 3)} | {cn} / {an} |"
            for label, (c, a, cn, an) in deltas.items()]
    return out + [""]


def _start(d5, n_boot):
    out = ["## Informed versus uninformed start (SortCMA; L12)", ""]
    cm, rob = d5.get("cmaes"), d5.get("cmaes-robust")
    if cm is None or rob is None or not len(_cell(rob, prior="flat")):
        return out + [MISSING, ""]
    informed = _pop(_reference(_cell(cm, family="promp_s0")))
    flat = _pop(_cell(rob, prior="flat"))
    return out + ["Flat cells count τ from the flat box's floor (T5); read the delivered column.", "",
                  CONTRAST_HEAD, _contrast("informed − uninformed", informed, flat, n_boot), ""]


def _verdict(ok, n=None) -> str:
    """held. / failed., marked provisional when it rests on fewer than EXPECTED_N users (M3)."""
    if ok is None:
        return "not run yet."
    text = "held." if ok else "failed."
    return text + f" (provisional, n < {EXPECTED_N})" if n is not None and n < EXPECTED_N else text


def _pair_text(c, b_name) -> str:
    return (f"vs {b_name}: n = {c['n']}, reached {c['share_a']:.2f} / {c['share_b']:.2f}, "
            f"delivered {c['med_a']:.3f} / {c['med_b']:.3f} Nm")


def _beats(c):
    """The comparison's first cell reaches more often and delivers less, on common users."""
    if c is None or not _finite(c["share_a"], c["share_b"], c["med_a"], c["med_b"]):
        return None
    return bool(c["share_a"] > c["share_b"] and c["med_a"] < c["med_b"])


def _predictions(d5, c5):
    lines = ["## Predictions recorded before running (spec §7)", ""]
    lens, noise, cm = d5.get("lens-franks"), d5.get("lens-noise"), d5.get("cmaes")
    promp_a = _ref5c(c5, "a")

    a_lens = None if lens is None else _pop(_reference(_cell(lens, learner="a_lens")))
    c8 = None if a_lens is None or promp_a is None else _common(a_lens, promp_a)
    d8 = _median(_paired(*c8, "delivered")) if c8 is not None and len(c8[0]) else np.nan
    if _finite(d8):
        lines.append(f"- **P8** (a_lens within 0.10 Nm of A on ProMP): n = {len(c8[0])}, "
                     f"{fmt_signed(d8, 3)} Nm; {_verdict(abs(d8) <= 0.10, len(c8[0]))}")
    else:
        lines.append("- **P8** (a_lens within 0.10 Nm of A on ProMP): not run yet.")
    if lens is not None:
        lines.append(f"  - {_ceiling_text(d5)}")

    p9 = "- **P9** (a_lens beats g and d_z on Franks): not run yet."
    p10 = "- **P10** (a direct ≥ 0.3 Nm worse than a_lens): not run yet."
    if a_lens is not None:
        rows = _pop(_reference(lens))
        cmps = [(x, _cmp(a_lens, _cell(rows, learner=x))) for x in ("g", "d_z")]
        if all(c is not None for _, c in cmps):
            verdicts = [_beats(c) for _, c in cmps]
            ok = None if any(v is None for v in verdicts) else all(verdicts)
            n9 = min(c["n"] for _, c in cmps)
            p9 = ("- **P9** (a_lens beats g and d_z on Franks, paired users): "
                  + "; ".join(_pair_text(c, x) for x, c in cmps) + f"; {_verdict(ok, n9)}")
        a_direct = _cell(rows, learner="a")
        if len(a_direct):
            c10 = _common(a_direct, a_lens)
            d10 = _median(_paired(*c10, "delivered")) if len(c10[0]) else np.nan
            if _finite(d10):
                p10 = (f"- **P10** (a direct ≥ 0.3 Nm worse than a_lens): n = {len(c10[0])}, "
                       f"{fmt_signed(d10, 3)} Nm; {_verdict(d10 >= 0.3, len(c10[0]))}")
    lines += [p9, p10]

    p11 = "- **P11** (a_lens ahead of g at lapse 0.3): not run yet."
    if noise is not None:
        c = _cmp(_pop(_cell(noise, learner="a_lens")), _pop(_cell(noise, learner="g")))
        if c is not None and _finite(c["med_a"], c["med_b"]):
            p11 = (f"- **P11** (a_lens ahead of g at lapse 0.3, paired users): n = {c['n']}, delivered "
                   f"{c['med_a']:.3f} / {c['med_b']:.3f} Nm; "
                   f"{_verdict(c['med_a'] < c['med_b'], c['n'])}")
    lines.append(p11)

    p12 = "- **P12** (stop_sd 0.3: ≥ 20 % saved, ≤ +0.05 Nm, ≤ 10 % premature): not run yet."
    cells = [c for c in _stop_cells(d5, c5) if c[0] == "promp_s0" and np.isclose(c[2], 0.3)]
    if cells:
        st = _stop_stats(cells[0][3], cells[0][4])
        if st["n"] and _finite(st["saved"], st["delta"], st["premature"]):
            ok = st["saved"] >= 0.20 and st["delta"] <= 0.05 and st["premature"] <= 0.10
            p12 = (f"- **P12** (stop_sd 0.3: ≥ 20 % saved, ≤ +0.05 Nm, ≤ 10 % premature): n = {st['n']}, "
                   f"saved {st['saved']:.0%}, {fmt_signed(st['delta'], 3)} Nm, premature "
                   f"{st['premature']:.2f}; {_verdict(ok, st['n'])}")
    lines.append(p12)

    p13 = "- **P13** (A beats SortCMA on ProMP): not run yet."
    if cm is not None and promp_a is not None:
        c = _cmp(promp_a, _pop(_reference(_cell(cm, family="promp_s0"))))
        if c is not None:
            p13 = (f"- **P13** (A beats SortCMA on ProMP, paired users): "
                   f"{_pair_text(c, 'SortCMA')}; {_verdict(_beats(c), c['n'])}")
    lines.append(p13)

    deltas = _robust_deltas(d5, c5)
    pair = [deltas.get("lapse 0.3"), deltas.get("drift 4 JND")]
    if all(p is not None and _finite(p[0], p[1]) for p in pair):
        ok = all(c > a for c, a, _, _ in pair)
        n14 = min(min(cn, an) for _, _, cn, an in pair)
        lines.append(f"- **P14** (SortCMA loses more than A under lapse 0.3 and 4 JND drift; paired "
                     f"increases, SortCMA / A): lapse 0.3 (n = {pair[0][2]} / {pair[0][3]}) "
                     f"{fmt_signed(pair[0][0], 3)} / {fmt_signed(pair[0][1], 3)} Nm, drift 4 JND "
                     f"(n = {pair[1][2]} / {pair[1][3]}) {fmt_signed(pair[1][0], 3)} / "
                     f"{fmt_signed(pair[1][1], 3)} Nm; {_verdict(ok, n14)}")
    else:
        lines.append("- **P14** (SortCMA loses more than A under lapse 0.3 and 4 JND drift): not run yet.")

    p15 = "- **P15** (informed start ≥ 0.2 Nm better): not run yet."
    rob = d5.get("cmaes-robust")
    if cm is not None and rob is not None and len(_cell(rob, prior="flat")):
        c15 = _common(_pop(_reference(_cell(cm, family="promp_s0"))), _pop(_cell(rob, prior="flat")))
        d = _median(_paired(*c15, "delivered")) if len(c15[0]) else np.nan
        if _finite(d):
            p15 = (f"- **P15** (informed start ≥ 0.2 Nm better): n = {len(c15[0])}, "
                   f"{fmt_signed(d, 3)} Nm; {_verdict(d <= -0.2, len(c15[0]))}")
    lines.append(p15)
    return lines + [""]


def build_report(dir5d: Path, dir5c: Path, n_boot: int = 10_000) -> str:
    d5 = {name: _load(Path(dir5d), name) for name in BLOCKS_5D}
    c5 = {name: _load(Path(dir5c), name) for name in ("main-promp", "main-dual", "noise")}
    lines = ["# Phase 5d report", "",
             "Outcome: comparisons until the delivered curve is within 0.5 Nm RMS of the user's "
             "floor in the cell's family and stays there, censored at 150 (\"> 150\"). Population "
             "users, tight arm. Brackets are 95 % cluster-bootstrap intervals over users, "
             "stratified by fold.", ""]
    lines += _lens_franks(d5, n_boot)
    lines += _cost(d5, c5, n_boot)
    lines += _lens_fitting(d5)
    lines += _franks_params(d5)
    lines += _lens_dual(d5, c5, n_boot)
    lines += _lens_noise(d5, n_boot)
    lines += _stopping(d5, c5)
    lines += _cmaes_field(d5, c5, n_boot)
    lines += _cmaes_robust(d5, c5)
    lines += _start(d5, n_boot)
    lines += _predictions(d5, c5)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/phase5d")
    ap.add_argument("--dir5c", default="out/phase5c")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.dir) / "REPORT.md"
    out.write_text(build_report(Path(args.dir), Path(args.dir5c), n_boot=args.boot), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
