"""Post hoc: the fold priors' dimensions and every optimizer's parameter count (amplitude-free design).

Usage:
  python scripts/icorr_posthoc_dims.py [--out <repository>/out]

Post hoc: no verdict of any test plan is changed or re-judged. design
our design notes section 28 (KS)
asks for each prior's effective dimension and the search dimension; the title claims more
parameters. This script reads them from the stored fold files and from the spaces and designs
the run scripts build, never from memory:

- **Priors.** Fixed level: w = w_minus + w_plus of the unit training shapes (fold-<name>.npz at
  K = 8, fold-<name>-K<K>.npz at K = 12, 16, 24 when present). Ranges: icorr_amp_range's
  ``_a_moments`` (medium, wide; K = 8) and ``_a12_moments`` (wide; K = 12); wide at K = 16 / 24
  is built the same way from the K-file when it exists (checked against KS's ``_ak_moments``
  where that code exists). Per prior and fold: the participation
  ratio (sum ev)^2 / sum ev^2 of the weight covariance's eigenvalues, the directions holding
  90 % and 95 % of the variance, the same in curve space (Phi Sigma Phi^T on the 101-point grid),
  and the learner's search dimension d and search region radius (the ``LearnerSpace`` the runs build).
  Intervals: R._boot's resampling (2,000 draws, percentile) over the fold's 20,000 training
  rows, vectorized; they measure Monte Carlo precision of the generator's population, not
  between-subject spread, which the five folds' min-max shows.
- **Parameter counts.** Every cell of FULL, CMP and FAM's projection is built by its own
  ``run_one`` (user 0 of each fold); ``run_session`` is intercepted before the first comparison,
  and the evaluated family's ``n_params``, the learner space's d and radius and each inner
  optimizer's own dimension are read from the objects. FAM's family table is rebuilt with
  ``amp_free.family_design`` and compared with FAM's reported column (fam_numbers.json).

Outputs in <out>/icorr/amp_free/: amp_dims.md, amp_dims.json.
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_cmp as CMP  # noqa: E402
import icorr_amp_fam as FAM  # noqa: E402
import icorr_amp_full as F  # noqa: E402
import icorr_amp_range as R  # noqa: E402
from prompref.basis import VonMisesBasis  # noqa: E402
from prompref.designs.device import DeviceDesign  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.learning.space import LearnerSpace  # noqa: E402
from prompref.population.folds import loso_folds  # noqa: E402

#: The fixed-level priors' K (K = 16 and 24 only where another job has written the parts).
FIXED_K = (8, 12, 16, 24)
#: The range priors: (range, K). K = 8 and wide K = 12 are the runs'; wide K = 16 / 24 only
#: where the parts exist (built as ``_a12_moments``).
RANGE_PRIORS = (("medium", 8), ("wide", 8), ("wide", 12), ("wide", 16), ("wide", 24))
SHARES = (0.90, 0.95)
BOOT_SALT = 97
SPEC = ROOT.parent / "icorr" / "docs" / "superpowers" / "specs" / "2026-10-27-amplitude-free-prior-design.md"
USED_BY = {("fixed 12", 8): "fixed 12 Nm: PsyPref (B_fixed) and the ProMP comparators of FULL, CMP, FAM",
           ("fixed 12", 12): "fixed 12 Nm: ROB R5 (PsyPref, K = 12); KS",
           ("fixed 12", 16): "fixed 12 Nm: KS", ("fixed 12", 24): "fixed 12 Nm: KS",
           ("medium", 8): "medium 3-12 Nm: PsyPref (AM) and the ProMP comparators of FULL, CMP",
           ("wide", 8): "wide 3-35 Nm: PsyPref (AM) and the ProMP comparators of FULL, CMP, FAM",
           ("wide", 12): "wide 3-35 Nm: ROB R5 (PsyPref, K = 12); KS",
           ("wide", 16): "wide 3-35 Nm: KS (AR's amplitudes, as _a12_moments)",
           ("wide", 24): "wide 3-35 Nm: KS (AR's amplitudes, as _a12_moments)"}


def _dir(out: Path) -> Path:
    return out / "icorr" / "amp_free"


# ----------------------------------------------------------------------------- spectra

def _spectrum(cov: np.ndarray) -> np.ndarray:
    """Eigenvalues, largest first, clipped at 0 (round-off)."""
    return np.clip(np.linalg.eigvalsh(cov)[::-1], 0.0, None)


def _pr(ev: np.ndarray) -> float:
    return float(ev.sum() ** 2 / np.square(ev).sum())


def _n_share(ev: np.ndarray, share: float) -> int:
    """The fewest leading directions whose variance reaches ``share`` of the total."""
    frac = np.cumsum(ev) / ev.sum()
    return int(np.argmax(frac >= share) + 1)


def _cov_counts(w: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``np.cov`` of the rows of ``w`` repeated ``counts`` times (a bootstrap sample)."""
    n = counts.sum()
    m = counts @ w / n
    return ((w * counts[:, None]).T @ w / n - np.outer(m, m)) * n / (n - 1)


def _stats(cov: np.ndarray, chol_gram: np.ndarray) -> np.ndarray:
    """PR, n90, n95 of the weight covariance, then the same of the curve covariance
    Phi Sigma Phi^T, whose non-zero eigenvalues are those of L^T Sigma L (Phi^T Phi = L L^T)."""
    ev = _spectrum(cov)
    evc = _spectrum(chol_gram.T @ cov @ chol_gram)
    return np.array([_pr(ev)] + [_n_share(ev, s) for s in SHARES]
                    + [_pr(evc)] + [_n_share(evc, s) for s in SHARES], dtype=float)


STAT_NAMES = ("pr", "n90", "n95", "pr_curve", "n90_curve", "n95_curve")


def _boot_counts(n: int, stat, rng) -> tuple[np.ndarray, np.ndarray]:
    """R._boot's resampling for one fold's ``n`` rows (``rng.integers(n, size=n)`` per draw,
    R.N_BOOT draws, 2.5 / 97.5 percentiles), vectorized: ``stat`` takes the draw's counts."""
    draws = np.array([stat(np.bincount(rng.integers(n, size=n), minlength=n))
                      for _ in range(R.N_BOOT)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return lo, hi


# ----------------------------------------------------------------------------- priors

def _parts_path(d: Path, fold: str, k: int) -> Path:
    return d / (f"fold-{fold}.npz" if k == 8 else f"fold-{fold}-K{k}.npz")


def _range_amps(fold_index: int, name: str, n: int) -> np.ndarray:
    """AR's range amplitudes, drawn exactly as ``R._a_moments`` and ``R._a12_moments`` draw them."""
    lo, hi = R.RANGES[name]
    return AF._loguniform(R._range_rng(fold_index, 200 + list(R.RANGES).index(name)), lo, hi, (n, 2))


def _prior(d: Path, fold_index: int, fold: str, setting: str, k: int, checks: list):
    """The prior's training rows, its LearnerSpace as the runs build it, or None if absent."""
    p = _parts_path(d, fold, k)
    if not p.exists():
        return None
    z = np.load(p)
    wm, wp = z["w_minus"], z["w_plus"]
    basis = None if k == 8 else VonMisesBasis(k)
    if setting == "fixed 12":
        rows = wm + wp
        space = AF.fixed_level_space(wm, wp, F.LEVEL, basis=basis)
        mu, sigma = rows.mean(axis=0), np.cov(rows, rowvar=False)
    else:
        lo, hi = R.RANGES[setting]
        amps = _range_amps(fold_index, setting, wm.shape[0])
        rows = amps[:, :1] * wm + amps[:, 1:] * wp
        mu, sigma = AF.weight_moments(wm, wp, amps)
        refs = {}
        p8 = str(_parts_path(d, fold, 8))
        if k == 8:
            refs["R._a_moments"] = R._a_moments(p8, fold_index, setting)
        elif k == 12:
            refs["R._a12_moments"] = R._a12_moments(p8, fold_index, setting)
        if k != 8 and hasattr(R, "_ak_moments"):   # KS's K-basis prior, where its code exists
            refs["R._ak_moments"] = R._ak_moments(p8, fold_index, setting, k)
        for name, ref in refs.items():
            checks.append(dict(check=f"{setting} K = {k}, {fold}: the bootstrap rows give {name} exactly",
                               ok=bool(np.array_equal(ref[0], mu) and np.array_equal(ref[1], sigma))))
        space = AF.promp_space(mu, sigma, limit=hi, basis=basis)
    if not np.allclose(space.normalization.mu, mu, rtol=0, atol=0):
        raise RuntimeError(f"{fold} {setting} K = {k}: the space's mean is not the prior's")
    return rows, space, sigma, (VonMisesBasis(k) if basis is None else basis)


def priors(out: Path, checks: list, missing: list) -> list[dict]:
    d = _dir(out)
    res = []
    keys = [("fixed 12", k) for k in FIXED_K] + list(RANGE_PRIORS)
    for setting, k in keys:
        for i, fold in enumerate(loso_folds()):
            got = _prior(d, i, fold.name, setting, k, checks)
            if got is None:
                missing.append(f"{_parts_path(d, fold.name, k).name} ({setting}, K = {k})")
                continue
            rows, space, sigma, basis = got
            chol = np.linalg.cholesky(basis.design.T @ basis.design / basis.design.shape[0])
            est = _stats(sigma, chol)
            # the space's own participation ratio (Normalization.d_eff) must be the one computed here
            checks.append(dict(check=f"{setting} K = {k}, {fold.name}: Whitening.d_eff equals the "
                                     "participation ratio", ok=bool(abs(space.normalization.d_eff - est[0]) < 1e-9 * est[0])))
            rng = np.random.default_rng([AF.SEED_FULL, BOOT_SALT, zlib.crc32(f"{setting}|{k}".encode()), i])
            lo, hi = _boot_counts(rows.shape[0], lambda c: _stats(_cov_counts(rows, c), chol), rng)
            row = dict(setting=setting, K=k, fold=fold.name, fold_index=i, n_rows=int(rows.shape[0]),
                       d=int(space.d), radius=float(space.radius), used_by=USED_BY[setting, k],
                       space_family=_family(space.design), **{
                           s: dict(est=float(e), lo=float(a), hi=float(b))
                           for s, e, a, b in zip(STAT_NAMES, est, lo, hi)})
            if setting != "fixed 12":
                # AM's metric N1 is J^T J at the centre, J the curve's Jacobian in z: for the
                # linear ProMP its spectrum is the curve covariance's, up to scale
                m = AF.curve_metric_at_centre(space)
                row["pr_metric_n1"] = _pr(_spectrum(0.5 * (m + m.T)))
            res.append(row)
            print(f"prior {setting} K = {k} {fold.name}: PR {est[0]:.2f} [{lo[0]:.2f}, {hi[0]:.2f}], "
                  f"n90 {est[1]:.0f}, d {space.d}, radius {space.radius:.2f}", flush=True)
    return res


def boot_equivalence(out: Path) -> dict:
    """The vectorized resampling against R._boot itself, on one prior (fixed K = 8, fold 0)."""
    d = _dir(out)
    fold = loso_folds()[0].name
    rows, space, sigma, basis = _prior(d, 0, fold, "fixed 12", 8, [])
    n = rows.shape[0]
    chol = np.linalg.cholesky(basis.design.T @ basis.design / basis.design.shape[0])
    seed = [AF.SEED_FULL, BOOT_SALT, 0]
    lo, hi = _boot_counts(n, lambda c: _stats(_cov_counts(rows, c), chol)[:1], np.random.default_rng(seed))
    recs = [dict(fold=fold, i=j) for j in range(n)]
    ref = R._boot(recs, lambda x: float(_stats(_cov_counts(rows, np.bincount([r["i"] for r in x], minlength=n)),
                                               chol)[0]), np.random.default_rng(seed))
    return dict(check=f"vectorized resampling equals R._boot (participation ratio, fixed K = 8, {fold})",
                ok=bool(ref[1] == lo[0] and ref[2] == hi[0]),
                detail=f"R._boot [{ref[1]:.6f}, {ref[2]:.6f}], vectorized [{lo[0]:.6f}, {hi[0]:.6f}]")


# ----------------------------------------------------------------------------- parameter counts

class _Built(Exception):
    """Raised in place of the first comparison: the session's design and learner are built."""


def _build(module, run_one, job: dict, out: Path):
    got = {}

    def intercept(design, learner, *args, **kwargs):
        got.update(design=design, learner=learner)
        raise _Built

    saved = module.run_session
    module.run_session = intercept
    try:
        row = run_one(dict(job), str(out))
    finally:
        module.run_session = saved
    if "design" not in got:
        raise RuntimeError(f"{job['setting']} {job['candidate']}: not built ({row.get('error')})")
    return got["design"], got["learner"]


def _family(design) -> str:
    if isinstance(design, DeviceDesign):
        return f"{_family(design.inner)} behind a {design.limit:.4g} Nm device"
    inner = getattr(design, "inner", None) if type(design).__name__ == "FixedPeaksDesign" else None
    if inner is not None:
        return f"{type(inner).__name__} with {inner.n_params - design.n_params} parameters held"
    return type(design).__name__


def _reach(root) -> list:
    """Objects reachable from ``root`` through its own attributes (not ``__getattr__``)."""
    seen, todo, found = set(), [root], []
    while todo:
        obj = todo.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        found.append(obj)
        attrs = getattr(obj, "__dict__", {})
        for name in ("learner", "inner", "inner_design", "family", "design", "space"):
            if name in attrs:
                todo.append(attrs[name])
    return found


def _describe(design, learner) -> dict:
    objs = _reach(learner)
    spaces = [o for o in objs if isinstance(o, LearnerSpace)]
    info = dict(learner=" > ".join(type(o).__name__ for o in objs if callable(getattr(type(o), "propose", None))),
                family=_family(design), family_params=int(design.n_params),
                family_param_names=list(design.param_names))
    if spaces:
        sp = spaces[0]
        info.update(d=int(sp.d), radius=float(sp.radius), space_family=_family(sp.design),
                    space_params=int(sp.design.n_params))
    for o in objs:
        a = getattr(o, "__dict__", {})
        if "es" in a:   # SortCMALearner: pycma on the search region as a box design
            info.update(cma_dim=int(o.design.n_params), cma_popsize=int(o.es.popsize),
                        cma_sigma0=float(o.es.sigma0))
        if type(o).__name__ == "GPEUBOLearner":
            info.update(gp_dim=int(o.d), gp_acquisition=o.acquisition)
        if "candidates" in a and "curves" in a:   # Arens' GP: a fixed candidate set in z
            info.update(gp_candidates=int(np.asarray(o.candidates).shape[0]),
                        gp_dim=int(np.asarray(o.candidates).shape[1]))
        if type(o).__name__ == "LineCoSparLearner":
            info.update(linecospar_dim=int(o.design.n_params))
        if "batch" in a:   # Ramella et al.: a fixed batch in the design's parameters
            info.update(batch_size=int(o.batch.shape[0]), batch_dim=int(o.batch.shape[1]))
        if "posterior" in a and hasattr(o.posterior, "x"):
            info.update(posterior_state=int(np.asarray(o.posterior.x).shape[1]))
    return info


def _cell_name(reg: str, cand: str) -> str:
    if reg == "FULL":
        return F.NAMES[cand]
    if reg == "CMP":
        return CMP.NAMES[cand]
    return f"PsyPref projected onto {FAM.LABEL[cand[len('proj_'):]]}"


def param_counts(out: Path) -> list[dict]:
    jobs = ([("FULL", F, F.run_one, j) for j in F.jobs(out)]
            + [("CMP", F, F.run_one, j) for j in CMP.jobs(out)]
            + [("FAM", FAM, FAM.run_one, j) for j in FAM.proj_jobs(out)])
    jobs = [t for t in jobs if t[3]["index"] == 0]
    cells: dict = {}
    for reg, module, run_one, job in jobs:
        design, learner = _build(module, run_one, job, out)
        info = _describe(design, learner)
        cells.setdefault((job["setting"], reg, job["candidate"]), {})[job["fold"]] = info
    def order(key):   # each test plan's own order of its cells
        s, reg, cand = key
        if reg == "FULL":
            return 0, F.CANDIDATES.index(cand)
        if reg == "CMP":
            return 1, CMP.CELLS.index(cand)
        return 2, (AF.PROJ_RANGE if s in F.RANGES else AF.PROJ_LEVEL).index(cand[len("proj_"):])

    res = []
    for setting in F.SETTINGS:
        for (s, reg, cand), by_fold in sorted(cells.items(), key=lambda kv: order(kv[0])):
            if s != setting:
                continue
            infos = list(by_fold.values())
            same = all(json.dumps(x, sort_keys=True) == json.dumps(infos[0], sort_keys=True) for x in infos)
            res.append(dict(setting=setting, registration=reg, cell=cand, name=_cell_name(reg, cand),
                            folds=sorted(by_fold), identical_across_folds=same, **infos[0]))
            print(f"{setting} {reg} {cand}: family {infos[0]['family_params']}, d {infos[0].get('d')}",
                  flush=True)
    return res


def fam_families(out: Path, checks: list) -> list[dict]:
    """FAM's family table (section 19): n_params of each family as FAM builds it, per fold."""
    run_dir = out / "phase5-schaefer"
    reported = {}
    path = _dir(out) / "fam_numbers.json"
    if path.exists():
        reported = json.loads(path.read_text(encoding="utf-8"))["settings"]
    res = []
    for setting in FAM.SETTINGS:
        for fam in FAM._families(setting):
            if fam.startswith("promp"):
                k = int(fam[len("promp"):])
                # promp_floor: the K weights in a range; FixedLevelDesign's K weights at the level
                ns = {k if setting in F.RANGES else AF.FixedLevelDesign(F.LEVEL, basis=VonMisesBasis(k)).n_params}
                kind = ("VonMisesBasis weights" if setting in F.RANGES else "FixedLevelDesign")
            else:
                designs = [AF.family_design(fam, run_dir, f.name, **FAM._kw(setting)) for f in loso_folds()]
                ns = {int(x.n_params) for x in designs}
                kind = _family(designs[0])
            n = ns.pop() if len(ns) == 1 else None
            rep = reported.get(setting, {}).get(fam, {}).get("n_params")
            res.append(dict(setting=setting, family=fam, label=FAM.LABEL[fam], built=kind, n_params=n,
                            fold_invariant=n is not None, fam_report=rep, fam_table=FAM.N_PARAMS[setting][fam]))
    checks.append(dict(check="FAM's reported parameter column (fam_numbers.json) equals the built families' n_params",
                       ok=all(r["n_params"] is not None and r["n_params"] == r["fam_report"] for r in res)))
    return res


# ----------------------------------------------------------------------------- report

def _summary(prs: list[dict]) -> list[dict]:
    out = []
    for setting, k in [("fixed 12", k) for k in FIXED_K] + list(RANGE_PRIORS):
        rs = [r for r in prs if r["setting"] == setting and r["K"] == k]
        if not rs:
            continue
        e = dict(setting=setting, K=k, n_folds=len(rs), used_by=USED_BY[setting, k],
                 d=sorted({r["d"] for r in rs}), radius=sorted({round(r["radius"], 6) for r in rs}))
        for s in STAT_NAMES:
            v = [r[s]["est"] for r in rs]
            e[s] = dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)))
        if "pr_metric_n1" in rs[0]:
            v = [r["pr_metric_n1"] for r in rs]
            e["pr_metric_n1"] = dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)))
        out.append(e)
    return out


def _spec_checks(summary: list[dict], checks: list) -> None:
    """our design notes: 'participation ratio ... is 3.2-4.1 at K = 8 and 3.9-4.8 at K = 12'."""
    if not SPEC.exists():
        checks.append(dict(check=f"spec not found at {SPEC}", ok=None))
        return
    text = re.sub(r"\s+", " ", SPEC.read_text(encoding="utf-8"))
    num = r"(\d+(?:\.\d+)?)"
    m = re.search(rf"participation ratio of the training weights' covariance\) is {num}–{num} at "
                  rf"K = 8 and {num}–{num} at K = 12", text)
    if m is None:
        checks.append(dict(check="design note 28's participation ratios: sentence not found", ok=None))
        return
    rep = [float(x) for x in m.groups()]
    got = {e["K"]: e["pr"] for e in summary if e["setting"] == "fixed 12"}
    comp = [round(got[8]["min"], 1), round(got[8]["max"], 1), round(got[12]["min"], 1), round(got[12]["max"], 1)]
    checks.append(dict(check="design note 28: the fixed-level priors' participation ratio, min-max over "
                             "folds, K = 8 and K = 12", ok=comp == rep,
                       detail=f"reported {rep[0]}–{rep[1]} / {rep[2]}–{rep[3]}; computed {got[8]['min']:.3f}–"
                              f"{got[8]['max']:.3f} / {got[12]['min']:.3f}–{got[12]['max']:.3f}"))


def _radius_checks(out: Path, summary: list[dict], checks: list) -> None:
    """LONG's best-profile distance table and PRC's header report the K = 8 search search region's radius."""
    r8 = [e for e in summary if e["setting"] == "fixed 12" and e["K"] == 8][0]["radius"][0]
    d = _dir(out)
    long_md = d / "amp_long_report.md"
    if long_md.exists():
        lines = [x for x in long_md.read_text(encoding="utf-8").splitlines()
                 if x.startswith("| ") and x.split("|")[1].strip() in ("medium", "wide", "fixed 12")
                 and x.count("|") == 10]
        rep = sorted({float(x.split("|")[-2]) for x in lines})
        checks.append(dict(check="amp_long_report.md, floors table: the search ball's radius (K = 8)",
                           ok=rep == [round(r8, 2)], detail=f"reported {rep}; computed {r8:.4f}"))
    prc = d / "amp_pri_prc.md"
    if prc.exists():
        m = re.search(r"search ball has radius (\d+(?:\.\d+)?)", prc.read_text(encoding="utf-8"))
        if m:
            checks.append(dict(check="amp_pri_prc.md: the 99 % search ball's radius (fixed level, K = 8)",
                               ok=float(m.group(1)) == round(r8, 2), detail=f"reported {m.group(1)}; computed {r8:.4f}"))


def _k8_file_check(out: Path, checks: list) -> None:
    d = _dir(out)
    for fold in loso_folds():
        p = d / f"fold-{fold.name}-K8.npz"
        if p.exists():
            a, b = np.load(p), np.load(_parts_path(d, fold.name, 8))
            checks.append(dict(check=f"{p.name} (build_fold_parts at K = 8) equals the stored parts",
                               ok=bool(np.array_equal(a["w_minus"], b["w_minus"]) and np.array_equal(a["w_plus"], b["w_plus"]))))


def _ci(t: dict, dec: int = 2) -> str:
    return f"{t['est']:.{dec}f} [{t['lo']:.{dec}f}, {t['hi']:.{dec}f}]"


def _ci_int(t: dict) -> str:
    return (f"{t['est']:.0f}" if t["lo"] == t["hi"] == t["est"]
            else f"{t['est']:.0f} [{t['lo']:.0f}, {t['hi']:.0f}]")


def _mm(t: dict, dec: int = 2) -> str:
    return f"{t['median']:.{dec}f} ({t['min']:.{dec}f}–{t['max']:.{dec}f})"


def _mm_int(t: dict) -> str:
    return f"{t['min']:.0f}" if t["min"] == t["max"] else f"{t['min']:.0f}–{t['max']:.0f}"


def _head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # provenance only
        return "unknown"


def report(out: Path) -> None:
    d = _dir(out)
    checks, missing = [], []
    prs = priors(out, checks, missing)
    summary = _summary(prs)
    checks.append(boot_equivalence(out))
    _spec_checks(summary, checks)
    _radius_checks(out, summary, checks)
    _k8_file_check(out, checks)
    fams = fam_families(out, checks)
    counts = param_counts(out)

    # one line per kind of check in the report; the per-fold checks are summarised
    grouped: dict = {}
    for c in checks:
        key = re.sub(r"loso-AB\d+", "<fold>", re.sub(r"^(fixed 12|medium|wide) K = \d+, ", "<prior>, ", c["check"]))
        g = grouped.setdefault(key, dict(check=key, n=0, ok=0, detail=[]))
        g["n"] += 1
        g["ok"] += int(c["ok"] is True)
        if c.get("detail"):
            g["detail"].append(c["detail"])

    L = ["# Post hoc: the fold priors' dimensions and every optimizer's parameter count", "",
         f"Post hoc (amp-free {_head()}); no verdict is changed or re-judged. Spec "
         "section 28 (KS) asks for each prior's effective dimension and the search dimension. Every "
         "number below is computed by scripts/icorr_posthoc_dims.py from the stored fold files and from "
         "the spaces and designs the run scripts build.", "",
         "Definitions. Participation ratio PR = (Σ ev)² / Σ ev² of the prior covariance's eigenvalues "
         "ev. n90 / n95: the fewest leading eigen-directions holding at least 90 % / 95 % of the "
         "variance. Weight space: the covariance of the K ProMP weights (Whitening.d_eff). Curve space: "
         "Φ Σ Φᵀ on the 101-point phase grid, independent of the basis's overlap. d and radius: the "
         "LearnerSpace the runs build (whitened, full rank; the tight arm's 99 % ball). Intervals: "
         f"{R.N_BOOT} draws resampling the fold's {prs[0]['n_rows'] if prs else 0:,} training rows "
         "(R._boot's resampling, vectorized; checked against R._boot below), a Monte Carlo precision "
         "of the generator's population; the spread between walking subjects is the min–max over the "
         "five LOSO folds.", "",
         "## 1. The priors (median over the five folds, min–max)", "",
         "| prior | K | used by | d | radius | PR, weights | n90 / n95, weights | PR, curve | n90 / n95, curve |",
         "|---|---|---|---|---|---|---|---|---|"]
    for e in summary:
        L.append(f"| {e['setting']} | {e['K']} | {e['used_by']} | {', '.join(map(str, e['d']))} | "
                 f"{', '.join(f'{x:.2f}' for x in e['radius'])} | {_mm(e['pr'])} | {_mm_int(e['n90'])} / "
                 f"{_mm_int(e['n95'])} | {_mm(e['pr_curve'])} | {_mm_int(e['n90_curve'])} / {_mm_int(e['n95_curve'])} |")
    if missing:
        L += ["", "Not present at run time, skipped: " + ", ".join(sorted(set(missing))) + "."]
    late = sorted(p for k in FIXED_K if k not in (8, 12) for p in d.glob(f"fold-*-K{k}.npz"))
    if late:
        stamps = [datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") for p in late]
        L += ["", f"K = 16 / 24 parts ({len(late)} files, written by KS's prep) read as of their write times "
              f"{min(stamps)} to {max(stamps)}; rerun this script if they are rewritten."]
    n1 = [e for e in summary if "pr_metric_n1" in e]
    if n1:
        L += ["", "AM's curve metric N1 (J^T J at the centre, the ranges' PsyPref): PR of its eigenvalues "
              + "; ".join(f"{e['setting']} K = {e['K']} {_mm(e['pr_metric_n1'])}" for e in n1)
              + " (equals the curve-space PR: for the linear ProMP J^T J has the curve covariance's spectrum)."]
    L += ["", "### Per fold", "",
          "| prior | K | fold | d | radius | PR, weights | n90 | n95 | PR, curve | n90, curve | n95, curve |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in prs:
        L.append(f"| {r['setting']} | {r['K']} | {r['fold']} | {r['d']} | {r['radius']:.2f} | {_ci(r['pr'])} | "
                 f"{_ci_int(r['n90'])} | {_ci_int(r['n95'])} | {_ci(r['pr_curve'])} | {_ci_int(r['n90_curve'])} | "
                 f"{_ci_int(r['n95_curve'])} |")
    L += ["", "## 2. Parameter counts of every optimizer's search space, as run", "",
          "Built by each test plan's own run_one for user 0 of every fold and stopped before the "
          "first comparison. Family: the curve family the wearer is shown and its parameter count. "
          "d / radius: the learner's whitened search space. Inner: the optimizer's own search "
          "dimension (CMA-ES: pycma's dimension, population and σ₀; Arens' GP: candidates × dimension; "
          "GP-BALD / GP-qEUBO: the GP's input dimension; Ramella et al.: the fixed batch × parameters; "
          "PsyPref: the posterior's particle state, z* plus its noise scale and lapse).", ""]
    for setting in F.SETTINGS:
        L += [f"### {setting}", "", "| test plan | optimizer | family shown (parameters) | d | radius | inner | "
              "identical in all folds |", "|---|---|---|---|---|---|---|"]
        for c in [c for c in counts if c["setting"] == setting]:
            inner = []
            if "cma_dim" in c:
                inner.append(f"CMA-ES dim {c['cma_dim']}, population {c['cma_popsize']}, σ₀ {c['cma_sigma0']:g}")
            if "gp_candidates" in c:
                inner.append(f"GP over {c['gp_candidates']} candidates × {c['gp_dim']}")
            elif "gp_dim" in c:
                inner.append(f"GP input dim {c['gp_dim']} ({c['gp_acquisition']})")
            if "linecospar_dim" in c:
                inner.append(f"LineCoSpar dim {c['linecospar_dim']}")
            if "batch_size" in c:
                inner.append(f"batch {c['batch_size']} × {c['batch_dim']}")
            if "posterior_state" in c:
                inner.append(f"posterior state {c['posterior_state']}")
            L.append(f"| {c['registration']} | {c['name']} | {c['family']} ({c['family_params']}) | {c.get('d', '–')} | "
                     f"{c['radius']:.2f} | {'; '.join(inner) or '–'} | "
                     f"{'yes' if c['identical_across_folds'] else 'no'} ({len(c['folds'])}) |")
        L.append("")
    L += ["### FAM's families (section 19), rebuilt", "",
          "| setting | family | built as | parameters (built) | FAM's reported column |", "|---|---|---|---|---|"]
    for r in fams:
        L.append(f"| {r['setting']} | {r['label']} | {r['built']} | {r['n_params']} | {r['fam_report']} |")
    L += ["", "## 3. Self-checks", ""]
    for g in grouped.values():
        status = "ok" if g["ok"] == g["n"] else f"FAILED ({g['n'] - g['ok']} of {g['n']})"
        L.append(f"- {g['check']}: {status}" + (f" ({g['n']} cases)" if g["n"] > 1 else "")
                 + (f" — {g['detail'][0]}" if len(g["detail"]) == 1 else ""))
    numbers = dict(generated=datetime.datetime.now().isoformat(timespec="minutes"), head=_head(),
                   post_hoc=True, n_boot=R.N_BOOT, shares=list(SHARES), missing=sorted(set(missing)),
                   priors_summary=summary, priors=prs, param_counts=counts, fam_families=fams, checks=checks)
    (d / "amp_dims.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (d / "amp_dims.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
