"""Test AR: the study sets the peak-torque range; run the sessions and the report.

Usage:
  python scripts/icorr_amp_range.py run    [--out <repository>/out] [--workers 8]
  python scripts/icorr_amp_range.py report [--out <repository>/out]

Reuses the AF pilot's fold files (training parts,
the 40 user shapes, answer seeds) from scripts/icorr_amp_free.py prep. Outputs in
<out>/icorr/amp_free/: range_sessions.jsonl (resumable), amp_range_report.md.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.population.folds import loso_folds  # noqa: E402
from prompref.user.oracle import Observer, rms_distance  # noqa: E402

#: Section 12: the ranges (per bout, Nm) and the fixed levels.
RANGES = {"narrow": (5.0, 8.0), "medium": (3.0, 12.0), "wide": (3.0, 35.0)}
LEVELS = (6.0, 12.0)
#: AR's candidates and, since test AM (section 13), those with the metric N1.
RANGE_CANDIDATES = ("a", "b", "am", "bm", "bmp", "am12")   # "am12": test K12
FIXED_CANDIDATES = ("b_fixed", "b_fixed_m", "am_fixed", "b_fixed12")   # FP, K12
METRIC_CANDIDATES = ("am", "bm", "bmp", "b_fixed_m")
N_BOOT = 2000


def _range_rng(fold_index: int, offset: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([AF.SEED, int(fold_index), int(offset)]))


@lru_cache(maxsize=None)
def _fold(path: str, fold_index: int) -> dict:
    return AF.load_fold_data(Path(path), fold_index)


@lru_cache(maxsize=None)
def _a_moments(path: str, fold_index: int, name: str):
    """A_range's prior: the unit training shapes at log-uniform range amplitudes per bout."""
    d = _fold(path, fold_index)
    lo, hi = RANGES[name]
    amps = AF._loguniform(_range_rng(fold_index, 200 + list(RANGES).index(name)), lo, hi,
                          (d["w_minus"].shape[0], 2))
    return AF.weight_moments(d["w_minus"], d["w_plus"], amps)


@lru_cache(maxsize=None)
def _parts12(path: str) -> dict:
    p = Path(path)
    z = np.load(p.with_name(p.stem + "-K12.npz"))
    return dict(w_minus=z["w_minus"], w_plus=z["w_plus"])


@lru_cache(maxsize=None)
def _a12_moments(path: str, fold_index: int, name: str):
    """AM12's prior: the K = 12 unit training shapes at AR's range amplitudes."""
    d = _parts12(path)
    lo, hi = RANGES[name]
    amps = AF._loguniform(_range_rng(fold_index, 200 + list(RANGES).index(name)), lo, hi,
                          (d["w_minus"].shape[0], 2))
    return AF.weight_moments(d["w_minus"], d["w_plus"], amps)


@lru_cache(maxsize=None)
def _parts_k(path: str, n_basis: int) -> dict:
    """Test KS (section 28): the training shapes projected onto ``n_basis`` von Mises
    bases (``fold-<name>-K<n>.npz`` next to the fold file); :func:`_parts12` at K = 12."""
    p = Path(path)
    z = np.load(p.with_name(f"{p.stem}-K{int(n_basis)}.npz"))
    return dict(w_minus=z["w_minus"], w_plus=z["w_plus"])


@lru_cache(maxsize=None)
def _ak_moments(path: str, fold_index: int, name: str, n_basis: int):
    """Test KS: the K-basis unit training shapes at AR's range amplitudes;
    :func:`_a12_moments` at K = 12."""
    d = _parts_k(path, n_basis)
    lo, hi = RANGES[name]
    amps = AF._loguniform(_range_rng(fold_index, 200 + list(RANGES).index(name)), lo, hi,
                          (d["w_minus"].shape[0], 2))
    return AF.weight_moments(d["w_minus"], d["w_plus"], amps)


def jobs(out: Path) -> list[dict]:
    d = out / "icorr" / "amp_free"
    js = []
    for i, fold in enumerate(loso_folds()):
        path = str(d / f"fold-{fold.name}.npz")
        data = _fold(path, i)
        for r, name in enumerate(RANGES):
            lo, hi = RANGES[name]
            amps = AF._loguniform(_range_rng(i, 100 + r), lo, hi, (AF.USERS_PER_FOLD, 2))
            for u in range(AF.USERS_PER_FOLD):
                for cand in RANGE_CANDIDATES:
                    js.append(dict(fold=fold.name, fold_index=i, index=u, user=f"{fold.name}/u{u}",
                                   mode="range", setting=name, candidate=cand,
                                   amps=[float(x) for x in amps[u]], _path=path,
                                   answer_seed=int(data["answer_seeds"][u]) + 10 + r))
        for v, level in enumerate(LEVELS):
            for u in range(AF.USERS_PER_FOLD):
                for cand in FIXED_CANDIDATES:
                    js.append(dict(fold=fold.name, fold_index=i, index=u, user=f"{fold.name}/u{u}",
                                   mode="fixed", setting=f"fixed {level:g}", candidate=cand,
                                   amps=[level, level], _path=path,
                                   answer_seed=int(data["answer_seeds"][u]) + 20 + v))
    return js


def run_one(job: dict) -> dict:
    path = job.pop("_path")
    try:
        d = _fold(path, job["fold_index"])
        u = job["index"]
        ideal = AF.compose(job["amps"][0], job["amps"][1], d["user_minus"][u], d["user_plus"][u])
        g_minus, g_plus = AF.unit_bouts(d["generic"])
        if job["mode"] == "range":
            lo, hi = RANGES[job["setting"]]
            centre = math.sqrt(lo * hi)
            if job["candidate"] in ("a", "am"):
                mu, sigma = _a_moments(path, job["fold_index"], job["setting"])
                space = AF.promp_space(mu, sigma, limit=hi)
            elif job["candidate"] == "am12":
                mu, sigma = _a12_moments(path, job["fold_index"], job["setting"])
                space = AF.promp_space(mu, sigma, limit=hi, basis=AF.VonMisesBasis(12))
            elif job["candidate"] == "bmp":
                space = AF.probit_space(d["w_minus"], d["w_plus"], (lo, hi))
            else:
                space = AF.factorized_space(d["w_minus"], d["w_plus"], band=(lo, hi),
                                            peak_box=(lo, hi), limit=hi)
        else:
            centre = job["amps"][0]
            if job["candidate"] == "b_fixed12":
                p12 = _parts12(path)
                space = AF.fixed_level_space(p12["w_minus"], p12["w_plus"], centre,
                                             basis=AF.VonMisesBasis(12))
            else:
                space = AF.fixed_level_space(d["w_minus"], d["w_plus"], centre)
        generic = AF.compose(centre, centre, g_minus, g_plus)
        observer = Observer(ideal_nm=ideal, s=AF.S, lapse=AF.LAPSE)
        rng = np.random.default_rng([int(job["answer_seed"]), AF.OBSERVER_SALT])
        seed = int(job["answer_seed"]) % 2**32
        design = space.design
        if job["candidate"] == "am_fixed":
            # Test FP (section 14): AM on the linear ProMP through the projection
            design, learner = AF.projected_fixed(d["w_minus"], d["w_plus"], centre, seed)
        elif job["candidate"] == "am12":
            # Test K12 (section 15): AM at K = 12 behind the two-bout guard
            learner = AF.TwoBoutGuard(AF.metric_psypref(space, seed), space,
                                      0.5 * RANGES[job["setting"]][0], seed=seed)
        elif job["candidate"] in METRIC_CANDIDATES:
            # Test AM (section 13): the candidates with N1 use the fixed curve metric
            learner = AF.metric_psypref(space, seed)
        else:
            learner = AF.psypref(space, seed)
        rec = AF._session(design, learner, observer, rng, AF.K, generic)
        delivered = design.curve(np.asarray(rec["best"], dtype=float))
        ident = AF.peaks(delivered)
        try:   # a delivered curve without one of the bouts has no unit-peak shape
            shape_error = rms_distance(AF.compose(1, 1, *AF.unit_bouts(delivered)),
                                       AF.compose(1, 1, d["user_minus"][u], d["user_plus"][u]))
        except AF.OrderViolation:
            shape_error = None
        guard = ({} if not isinstance(learner, AF.TwoBoutGuard)
                 else dict(guard_fired=int(learner.fired), guard_final=bool(learner.fired_last)))
        return dict(job, identified=[max(ident[0], 0.0), max(ident[1], 0.0)],
                    d0=rms_distance(generic, ideal), distances=list(rec["distances"]),
                    shape_error=shape_error, error=None, **guard)
    except Exception as exc:  # one bad session must not stop the pilot
        return dict(job, error=f"{type(exc).__name__}: {exc}")


def _key(row):
    return row["user"], row["setting"], row["candidate"]


def run(out: Path, workers: int) -> None:
    path = out / "icorr" / "amp_free" / "range_sessions.jsonl"
    done = set()
    if path.exists():
        done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = [j for j in jobs(out) if _key(j) not in done]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        for n, fut in enumerate(as_completed([pool.submit(run_one, j) for j in todo]), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", _key(row), row["error"], flush=True)
            if n % 40 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- report

def _pref(rows):
    d0 = np.array([r["d0"] for r in rows])
    dk = np.array([r["distances"][-1] for r in rows])
    return float(np.mean(AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((d0 - dk) / AF.S)))


def _logs(rows, b):
    return (np.log([r["amps"][b] for r in rows]), np.log([max(r["identified"][b], 1e-3) for r in rows]))


def _r2(rows, b):
    x, y = _logs(rows, b)
    return float(1.0 - np.mean((y - x) ** 2) / np.var(x))


def _slope(rows, b):
    x, y = _logs(rows, b)
    return float(np.polyfit(x, y, 1)[0])


def _end_bias(rows, b, lo, hi):
    """Mean signed log error of the users in the range's lowest and highest third (log scale)."""
    x, y = _logs(rows, b)
    a, c = math.log(lo), math.log(hi)
    low, high = x <= a + (c - a) / 3, x >= c - (c - a) / 3
    return float(np.mean((y - x)[low])), float(np.mean((y - x)[high]))


def _boot(rows, stat, rng):
    by_fold: dict[str, list] = {}
    for r in rows:
        by_fold.setdefault(r["fold"], []).append(r)
    draws = []
    for _ in range(N_BOOT):
        sample = []
        for rs in by_fold.values():
            sample += [rs[i] for i in rng.integers(len(rs), size=len(rs))]
        draws.append(stat(sample))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return stat(rows), float(lo), float(hi)


def _f(t, d=2):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def _pref_curve(rows):
    d0 = np.array([r["d0"] for r in rows])[:, None]
    dk = np.array([r["distances"] for r in rows])
    return np.mean(AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((d0 - dk) / AF.S), axis=0)


def _paired(rows, setting, c1, c2, rng):
    u1 = {r["user"]: r for (u, s, c), r in rows.items() if s == setting and c == c1}
    u2 = {r["user"]: r for (u, s, c), r in rows.items() if s == setting and c == c2}
    pairs = [dict(fold=u1[u]["fold"], diff=_pref([u1[u]]) - _pref([u2[u]]))
             for u in sorted(set(u1) & set(u2))]
    if len(pairs) < 8:
        return None
    return _boot(pairs, lambda x: float(np.mean([p["diff"] for p in x])), rng)


def _first_k(rows, setting, cand, rng):
    """First k at which ``cand``'s mean preference reaches B's at k = 24; users paired."""
    uc = {r["user"]: r for (u, s, c), r in rows.items() if s == setting and c == cand}
    ub = {r["user"]: r for (u, s, c), r in rows.items() if s == setting and c == "b"}
    pairs = [dict(fold=uc[u]["fold"], c=uc[u], b=ub[u]) for u in sorted(set(uc) & set(ub))]
    if len(pairs) < 8:
        return None

    def stat(x):
        target = _pref_curve([p["b"] for p in x])[AF.K - 1]
        reach = np.nonzero(_pref_curve([p["c"] for p in x]) >= target)[0]
        return float(reach[0] + 1) if len(reach) else float(AF.K + 1)
    return _boot(pairs, stat, rng)


def _am_section(rows, handles, fixed, rng) -> list[str]:
    """Test AM (section 13): paired differences, speed, PAM1-PAM6 and the rule."""
    L = ["", "## Registration AM (section 13)", "",
         "Paired differences in preference over generic assistance (points) and speed (first k "
         f"reaching B's k = {AF.K} mean preference; {AF.K + 1} = not within {AF.K}):", ""]
    diffs, speed = {}, {}
    for name in RANGES:
        for cand in ("bm", "bmp", "am"):
            ref = "a" if cand == "am" else "b"
            t = _paired(rows, name, cand, ref, rng)
            if t is not None:
                diffs[name, cand] = t
                L.append(f"- {name}, {cand} − {ref}: {100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]")
            s = _first_k(rows, name, cand, rng)
            if s is not None:
                speed[name, cand] = s
                L.append(f"  - {name}, {cand}: first k {s[0]:.0f} [{s[1]:.0f}, {s[2]:.0f}]")
    for level in LEVELS:
        for cand in ("b_fixed_m", "am_fixed"):
            t = _paired(rows, f"fixed {level:g}", cand, "b_fixed", rng)
            if t is not None:
                diffs[f"fixed {level:g}", cand] = t
                L.append(f"- fixed {level:g} Nm, {cand} − b_fixed: {100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]")

    def verdict(held, failed):
        return "held" if held else ("failed" if failed else "uncertain")

    L += ["", "Predictions:", ""]
    if ("wide", "bm") in handles:
        r2f = handles["wide", "bm"]["r2"][1][0]
        L.append(f"- PAM1 (wide: BM's flexion R² ≥ 0.43): {'held' if r2f >= 0.43 else 'failed'} — {r2f:.2f}")
    if ("wide", "bm") in diffs:
        t = diffs["wide", "bm"]
        L.append(f"- PAM2 (wide: BM − B > 0): {verdict(t[1] > 0, t[2] < 0)}")
    if ("narrow", "bm") in diffs:
        t = diffs["narrow", "bm"]
        L.append(f"- PAM3 (narrow: BM − B ≥ 2 points): {'held' if t[0] >= 0.02 else 'failed'} — {100 * t[0]:+.1f}")
    if all((n, c) in handles for n in ("medium", "wide") for c in ("bm", "bmp")):
        def mab(n, c):
            return float(np.mean([abs(v) for b in (0, 1) for v in handles[n, c]["eb"][b]]))
        ok = all(mab(n, "bmp") < mab(n, "bm") for n in ("medium", "wide"))
        L.append(f"- PAM4 (BMP's mean |end bias| below BM's in medium and wide): {'held' if ok else 'failed'} — "
                 + ", ".join(f"{n} {mab(n, 'bmp'):.2f} vs {mab(n, 'bm'):.2f}" for n in ("medium", "wide")))
    if ("fixed 6", "b_fixed_m") in diffs:
        t = diffs["fixed 6", "b_fixed_m"]
        L.append(f"- PAM5 (fixed 6 Nm: B_fixed + N1 − B_fixed > 0): {verdict(t[1] > 0, t[2] < 0)}")
    if ("wide", "bm") in speed:
        s = speed["wide", "bm"]
        L.append(f"- PAM6 (wide: BM reaches B's k = 24 preference by k = 18): {'held' if s[0] <= 18 else 'failed'} — k {s[0]:.0f}")
    # Test FP (section 14)
    fp = [diffs.get((f"fixed {level:g}", "am_fixed")) for level in LEVELS]
    if all(t is not None for t in fp):
        L.append(f"- PFP1 (AM_fixed − B_fixed > 0 at 6 and 12 Nm): "
                 f"{verdict(all(t[1] > 0 for t in fp), any(t[2] < 0 for t in fp))}")
        med = {(lv, c): float(np.median([r["shape_error"] for r in fixed[lv, c].values()
                                         if r["shape_error"] is not None]))
               for lv in LEVELS for c in ("am_fixed", "b_fixed") if (lv, c) in fixed}
        if len(med) == 4:
            ok = all(med[lv, "am_fixed"] < med[lv, "b_fixed"] for lv in LEVELS)
            L.append(f"- PFP2 (AM_fixed's median shape error below B_fixed's at both levels): "
                     f"{'held' if ok else 'failed'} — "
                     + ", ".join(f"{lv:g} Nm {med[lv, 'am_fixed']:.3f} vs {med[lv, 'b_fixed']:.3f}" for lv in LEVELS))
    L += _k12_lines(rows, handles, fixed, rng, verdict)
    count = {c: sum(handles.get((n, c), {}).get("ok", False) for n in RANGES) for c in RANGE_CANDIDATES}
    mean_pref = {c: float(np.mean([handles[n, c]["pref"][0] for n in RANGES if (n, c) in handles]))
                 for c in RANGE_CANDIDATES if any((n, c) in handles for n in RANGES)}
    chosen = max(mean_pref, key=lambda c: (count[c], mean_pref[c]))
    L += ["", "Selection (rule of section 13, over all five candidates): ranges handled "
          + ", ".join(f"{c.upper()} {count[c]}" for c in mean_pref) + "; mean preference "
          + ", ".join(f"{c.upper()} {mean_pref[c]:.3f}" for c in mean_pref) + f". Chosen: {chosen.upper()}."]
    return L


def _k12_lines(rows, handles, fixed, rng, verdict) -> list[str]:
    """Test K12 (section 15): PK1-PK5."""
    L = ["", "Registration K12 (section 15):", ""]
    pk1, pk2, pk3 = [], [], []
    for n in ("medium", "wide"):
        if (n, "am12") in handles and (n, "am") in handles:
            a, b = handles[n, "am12"], handles[n, "am"]
            pk1.append((n, a["r2"][1][0] - b["r2"][1][0]))
            pk2.append((n, abs(a["eb"][1][1]), abs(b["eb"][1][1])))
            t = _paired(rows, n, "am12", "am", rng)
            pk3.append((n, t))
            L.append(f"- {n}, am12 − am: {100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}] points")
    if len(pk1) == 2:
        L.append(f"- PK1 (flexion R² + 0.1 in medium and wide): {'held' if all(v >= 0.1 for _, v in pk1) else 'failed'} — "
                 + ", ".join(f"{n} {v:+.2f}" for n, v in pk1))
        L.append(f"- PK2 (smaller top-end flexion bias): {'held' if all(a < b for _, a, b in pk2) else 'failed'} — "
                 + ", ".join(f"{n} {a:.2f} vs {b:.2f}" for n, a, b in pk2))
        L.append(f"- PK3 (not below AM by more than 2 points): "
                 f"{verdict(all(t[1] > -0.02 for _, t in pk3), any(t[2] < -0.02 for _, t in pk3))}")
    pk4 = [_paired(rows, f"fixed {lv:g}", "b_fixed12", "b_fixed", rng) for lv in LEVELS]
    if all(t is not None for t in pk4):
        L += [f"- fixed {lv:g} Nm, b_fixed12 − b_fixed: {100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"
              for lv, t in zip(LEVELS, pk4)]
        L.append(f"- PK4 (B_fixed12 not below B_fixed by more than 2 points): "
                 f"{verdict(all(t[1] > -0.02 for t in pk4), any(t[2] < -0.02 for t in pk4))}")
    g = [r for (u, s, c), r in rows.items() if c == "am12"]
    if g:
        share = float(np.mean([r.get("guard_fired", 0) > 0 for r in g]))
        final = int(sum(r.get("guard_final", False) for r in g))
        L.append(f"- PK5 (guard fires in fewer than 5 % of AM12 sessions): {'held' if share < 0.05 else 'failed'} — "
                 f"fired in {100 * share:.1f} % of {len(g)} sessions, on the final curve in {final}")
    return L


def report(out: Path) -> None:
    d = out / "icorr" / "amp_free"
    rows = [r for r in map(json.loads, (d / "range_sessions.jsonl").read_text(encoding="utf-8").splitlines())]
    ok_rows = {_key(r): r for r in rows if r.get("error") is None}
    # an error counts only while its session has no successful rerun
    errors = [r for r in rows if r.get("error") and _key(r) not in ok_rows]
    rows = ok_rows
    rng = np.random.default_rng(AF.SEED)
    L = ["# Registration AR: the study sets the peak-torque range", "",
         f"Spec section 12 (bfea0f7). {len(rows)} sessions, {len(errors)} errors. k = {AF.K}, "
         f"s = {AF.S} Nm, λ = {AF.LAPSE}. 40 users per setting (8 per LOSO fold). 95 % intervals: "
         f"{N_BOOT} draws, users resampled within folds.", "",
         "| range | candidate | R² ext | R² flex | slope ext / flex | end bias ext (low, high) | "
         "end bias flex (low, high) | median abs log error ext / flex | preference over generic | "
         "handles |", "|---|---|---|---|---|---|---|---|---|---|"]
    handles = {}
    for name, (lo, hi) in RANGES.items():
        for cand in RANGE_CANDIDATES:
            rs = [r for (u, s, c), r in rows.items() if s == name and c == cand]
            if len(rs) < 8:
                continue
            r2 = [_boot(rs, lambda x, b=b: _r2(x, b), rng) for b in (0, 1)]
            eb = [_end_bias(rs, b, lo, hi) for b in (0, 1)]
            err = [float(np.median(np.abs(np.subtract(*_logs(rs, b)[::-1])))) for b in (0, 1)]
            pr = _boot(rs, _pref, rng)
            ok = all(r2[b][0] >= 0.5 and abs(eb[b][0]) <= 0.10 and abs(eb[b][1]) <= 0.10 for b in (0, 1))
            handles[name, cand] = dict(ok=ok, pref=pr, r2=r2, eb=eb)
            L.append(f"| {name} {lo:g}–{hi:g} | {cand} | {_f(r2[0])} | {_f(r2[1])} | "
                     f"{_slope(rs, 0):.2f} / {_slope(rs, 1):.2f} | {eb[0][0]:+.2f}, {eb[0][1]:+.2f} | "
                     f"{eb[1][0]:+.2f}, {eb[1][1]:+.2f} | {err[0]:.2f} / {err[1]:.2f} | {_f(pr, 3)} | "
                     f"{'yes' if ok else 'no'} |")
    L += ["", "| fixed level | candidate | n | shape error (unit peaks, RMS) median | preference over generic |",
          "|---|---|---|---|---|"]
    fixed = {}
    for level in LEVELS:
        name = f"fixed {level:g}"
        for cand in FIXED_CANDIDATES:
            rs = [r for (u, s, c), r in rows.items() if s == name and c == cand]
            if not rs:
                continue
            fixed[level, cand] = {r["user"]: r for r in rs}
            L.append(f"| {level:g} Nm | {cand} | {len(rs)} | {np.median([r['shape_error'] for r in rs if r['shape_error'] is not None]):.3f} | "
                     f"{_f(_boot(rs, _pref, rng), 3)} |")
    L += ["", "## Predictions", ""]
    if ("medium", "b") in handles and ("wide", "b") in handles:
        ok = handles["medium", "b"]["ok"] and handles["wide", "b"]["ok"]
        L.append(f"- PAR1 (B handles medium and wide): {'held' if ok else 'failed'}")
    if ("wide", "a") in handles:
        L.append(f"- PAR2 (A does not handle wide, an end bias beyond ±0.10): "
                 f"{'held' if any(abs(x) > 0.10 for b in (0, 1) for x in handles['wide', 'a']['eb'][b]) else 'failed'}")
    if all(("narrow", c) in handles for c in ("a", "b")):
        ok = all(not all(handles["narrow", c]["r2"][b][0] >= 0.5 for b in (0, 1)) for c in ("a", "b"))
        L.append(f"- PAR3 (narrow: neither reaches R² ≥ 0.5 for both bouts): {'held' if ok else 'failed'}")
    if (6.0, "b_fixed") in fixed and (12.0, "b_fixed") in fixed:
        f6, f12 = fixed[6.0, "b_fixed"], fixed[12.0, "b_fixed"]
        users = sorted(set(f6) & set(f12))
        pairs = [dict(fold=f12[u]["fold"], diff=_pref([f12[u]]) - _pref([f6[u]])) for u in users]
        t = _boot(pairs, lambda x: float(np.mean([p["diff"] for p in x])), rng)
        verdict = "held" if t[0] >= 0.03 and t[1] > 0 else ("failed" if t[2] < 0.03 else "uncertain")
        L.append(f"- PAR4 (fixed: 12 Nm at least 3 points above 6 Nm): {verdict} — "
                 f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}] points")
    L += _am_section(rows, handles, fixed, rng)
    count = {c: sum(handles.get((n, c), {}).get("ok", False) for n in RANGES) for c in ("a", "b")}
    mean_pref = {c: np.mean([handles[n, c]["pref"][0] for n in RANGES if (n, c) in handles]) for c in ("a", "b")}
    chosen = max(("a", "b"), key=lambda c: (count[c], mean_pref[c]))
    L += ["", "## Selection (rule of section 12)", "",
          f"Ranges handled: A {count['a']}, B {count['b']}; mean preference over generic A "
          f"{mean_pref['a']:.3f}, B {mean_pref['b']:.3f}. Chosen: {chosen.upper()}."]
    if errors:
        L += ["", "## Errors", ""] + [f"- {_key(e)}: {e['error']}" for e in errors[:20]]
    (d / "amp_range_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    run(args.out, args.workers) if args.command == "run" else report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
