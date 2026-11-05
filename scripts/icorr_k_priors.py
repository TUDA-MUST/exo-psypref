"""ProMP data priors and best-profile distances at other basis counts K.

Usage:
    python scripts/icorr_k_priors.py --run <main>/out/phase5-schaefer [--k 4,12]
        [--workers 8] [--folds all|loso] [--report <main>/out/icorr/k_priors_check.json]

W3 ran ``--k 4,12`` (report ``k_priors_check.json``). The higher-parameter test of section 9a
("More parameters: K = 16 and 24") ran ``--k 16,24 --workers 4 --report
<main>/out/icorr/k_hi_priors_check.json`` with ``PREFTUNE_DATA`` set to a readable copy of the
Molinaro trials; the K = 8 checks below are what make that copy acceptable. Pass ``--report``
for a new K set, or the default overwrites W3's check file.

For every fold of a Phase 5a run (the Molinaro trials are needed, as for 5a itself; the root is
the loader's, ``PREFTUNE_DATA`` if set, else the S: share):
1. Regenerate the fold's training curves from its own "training" stream and the population
   recorded in ``RUN/folds/<fold>.json``, exactly as ``phase5_prior_diagnostics.py`` does
   when it loads saved priors.
2. **Check K = 8.** Refit the ProMP prior with ``VonMisesBasis(8)`` through the unchanged
   ``fit_promp_prior`` and require it to equal ``RUN/priors/<fold>-promp_s0`` bit for bit
   (mu, sigma, fit statistics, design curves). Recompute the stored K = 8 best-profile distances with 5a's own
   best-profile distance stream and require them to equal the exported ones bit for bit. A fold that fails
   either check gets no K files.
3. For each K, fit the K prior the same way and save it as
   ``RUN/priors/<fold>-promp_s0-k<K>`` (an existing file must hold the same prior, or the
   script stops), then each user's best-profile distance and reach radii in ``RUN/users/<fold>-promp_s0-k<K>.json``,
   cached in ``RUN/best-profile distances/<fold>-promp_s0-k<K>.json`` (``floors5c.basis_sidecar``, seeded as
   ``phase5c_floors.py`` seeds its ProMP sidecars).

No existing file is written. The check results go to ``--report``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from preftune.data.loader import load_trial

from prompref.experiment.floors5c import basis_sidecar, fold_seed
from prompref.population.export import load_fold_users, population_from_config
from prompref.population.folds import (SIDES, check_fold, child_sequence, fold_streams,
                                       in_sample_fold, loso_folds)
from prompref.population.users import user_floors
from prompref.priors.basis_prior import (BasisDataPrior, basis_label, basis_prior_stem,
                                         common_fields, fit_basis_prior)
from prompref.priors.data_prior import (FAMILIES, DataPrior, fold_training_set,
                                        population_source_hashes)

MAX_WORKERS = 8
FLOOR_FIELDS = (("floor_nm", "floor_nm"), ("reach_050", "reach_050"),
                ("reach_025", "reach_025"), ("radius", "radius_floor"))

_TRIALS: dict = {}


def _trials(subjects) -> list:
    """``folds.load_trials(subjects)``, in its order, loading each leg from S: only once."""
    out = []
    for subject in subjects:
        for side in SIDES:
            if (subject, side) not in _TRIALS:
                _TRIALS[subject, side] = load_trial(subject, side)
            out.append(_TRIALS[subject, side])
    return out


def _same_prior(a: DataPrior, b: DataPrior) -> dict:
    """Field-by-field equality of two promp_s0 priors, arrays by ``np.array_equal``."""
    theta = np.random.default_rng(0).normal(size=(5, a.mu.size)) * 5.0
    curves_a = np.array([a.design().curve(t) for t in theta])
    curves_b = np.array([b.design().curve(t) for t in theta])
    return dict(mu=bool(np.array_equal(a.mu, b.mu)), sigma=bool(np.array_equal(a.sigma, b.sigma)),
                n_fit_failures=a.n_fit_failures == b.n_fit_failures,
                fit_rms_median_nm=a.fit_rms_median_nm == b.fit_rms_median_nm,
                share_at_bound=a.share_at_bound == b.share_at_bound,
                provenance=common_fields(a) == common_fields(b),
                design_curves=bool(np.array_equal(curves_a, curves_b)))


def _floor_check(entries, users) -> dict:
    exact = {name: all(e[src] == u.families["promp_s0"][name] for e, u in zip(entries, users))
             for name, src in FLOOR_FIELDS}
    diff = {name: max(abs(e[src] - u.families["promp_s0"][name]) for e, u in zip(entries, users))
            for name, src in FLOOR_FIELDS}
    return dict(exact=exact, max_abs_diff=diff)


def _prior_summary(prior: DataPrior) -> dict:
    eig = np.linalg.eigvalsh(prior.sigma)
    return dict(d_eff=float(prior.normalization().d_eff), n_fit_failures=int(prior.n_fit_failures),
                fit_rms_median_nm=float(prior.fit_rms_median_nm),
                eig_min=float(eig.min()), eig_max=float(eig.max()),
                share_at_bound_max=float(max(prior.share_at_bound.values())))


def run_fold(run: Path, fold, ks, *, workers: int) -> dict:
    started = time.perf_counter()
    config = json.loads((run / "folds" / f"{fold.name}.json").read_text(encoding="utf-8"))["config"]
    shift, switches, timing = population_from_config(config)
    seed, n_curves, starts = int(config["seed"]), int(config["n_curves"]), int(config["floor_starts"])
    streams = fold_streams(seed, fold.name)
    train = _trials(fold.train)
    check_fold(fold, [t.subject for t in train])
    training = fold_training_set(train, n_curves, seed, switches, timing,
                                 seed_sequence=streams["training"], shift=shift)
    stored = DataPrior.load(run / "priors" / f"{fold.name}-promp_s0")
    result = dict(fold=fold.name, n_curves=int(len(training.curves)),
                  n_rejected=int(training.n_rejected),
                  hashes_current=stored.source_hashes == population_source_hashes(),
                  n_rejected_matches=stored.n_rejected_draws == training.n_rejected)

    # Check K = 8: the recipe, then the best-profile distances.
    refit = fit_basis_prior(training.curves, 8, common_fields(stored))
    result["k8_prior"] = _same_prior(refit, stored)
    users = list(load_fold_users(run, fold.name).values())
    floors_5a = user_floors(stored.design(), stored.normalization(), users, n_starts=starts,
                            seed_sequence=child_sequence(streams["floors"],
                                                         FAMILIES.index("promp_s0")),
                            workers=workers)
    result["k8_floors_5a_stream"] = _floor_check(floors_5a, users)
    # The K-best-profile distance recipe (phase5c_floors-style seed) at K = 8, against the exported best-profile distances:
    # the same best-profile distances up to which multi-start wins, which is what a K best-profile distance differs by.
    floors_k = user_floors(stored.design(), stored.normalization(), users, n_starts=starts,
                           seed_sequence=fold_seed(fold.name, basis_label(8)), workers=workers)
    result["k8_floors_k_recipe"] = _floor_check(floors_k, users)
    result["k8_summary"] = _prior_summary(stored)
    ok = (result["hashes_current"] and result["n_rejected_matches"]
          and all(result["k8_prior"].values())
          and all(result["k8_floors_5a_stream"]["exact"].values()))
    result["k8_reproduced"] = bool(ok)
    if not ok:
        result["seconds"] = round(time.perf_counter() - started, 1)
        return result

    result["k"] = {}
    for k in ks:
        prior = fit_basis_prior(training.curves, k, common_fields(stored))
        stem = basis_prior_stem(run, fold.name, k)
        if Path(f"{stem}.npz").exists() or Path(f"{stem}.json").exists():
            existing = BasisDataPrior.load(stem)
            same = _same_prior(prior, existing)
            if not all(same.values()) or existing.n_basis != k:
                raise RuntimeError(f"{stem} exists and holds a different prior ({same}); "
                                   f"refusing to overwrite it")
            written = False
        else:
            prior.save(stem)
            written = True
        sidecar = basis_sidecar(run, fold.name, users, BasisDataPrior.load(stem),
                                n_starts=starts, workers=workers)
        floors = np.array([sidecar[u.user_id]["floor_nm"] for u in users
                           if u.kind == "population"])
        result["k"][str(k)] = dict(_prior_summary(prior), prior_written=written,
                                   floor_median_population=float(np.median(floors)))
    result["seconds"] = round(time.perf_counter() - started, 1)
    return result


def pooled_floor_medians(run: Path, folds, ks) -> dict:
    """Median population best-profile distance over the leave-one-subject-out folds, per K (8 = exported)."""
    out = {}
    for k in (8, *ks):
        values = []
        for fold in folds:
            if not fold.name.startswith("loso-"):
                continue
            users = load_fold_users(run, fold.name)
            if k == 8:
                values += [u.families["promp_s0"]["floor_nm"] for u in users.values()
                           if u.kind == "population"]
                continue
            path = run / "users" / f"{fold.name}-{basis_label(k)}.json"
            if not path.exists():
                break
            sidecar = json.loads(path.read_text(encoding="utf-8"))
            values += [sidecar[uid]["floor_nm"] for uid, u in users.items()
                       if u.kind == "population"]
        else:
            out[str(k)] = dict(n=len(values), median=float(np.median(values)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a Phase 5a output directory")
    ap.add_argument("--k", default="4,12", help="basis counts other than 8")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--folds", choices=("all", "loso"), default="all")
    ap.add_argument("--report", default=None,
                    help="check results (default: <run>/../icorr/k_priors_check.json)")
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        ap.error(f"--workers must lie in 1..{MAX_WORKERS}: the machine is shared")
    ks = tuple(int(k) for k in args.k.split(",") if k.strip())
    if 8 in ks or any(k < 2 for k in ks):
        ap.error("--k takes basis counts of at least 2 other than 8 (K = 8 is the run's own)")
    run = Path(args.run).resolve()
    report = Path(args.report) if args.report else run.parent / "icorr" / "k_priors_check.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    folds = loso_folds() + ([in_sample_fold()] if args.folds == "all" else [])

    results, failed = [], []
    for fold in folds:
        result = run_fold(run, fold, ks, workers=args.workers)
        results.append(result)
        if not result["k8_reproduced"]:
            failed.append(fold.name)
        print(f"{fold.name}: K=8 reproduced {result['k8_reproduced']} "
              f"(prior {all(result['k8_prior'].values())}, floors "
              f"{all(result['k8_floors_5a_stream']['exact'].values())}); "
              + ", ".join(f"K={k} floor median {v['floor_median_population']:.3f} Nm, "
                          f"d_eff {v['d_eff']:.2f}" for k, v in result.get("k", {}).items())
              + f" [{result['seconds']} s]", flush=True)
        report.write_text(json.dumps(dict(run=str(run), ks=ks, folds=results), indent=1),
                          encoding="utf-8")
    pooled = pooled_floor_medians(run, folds, ks)
    report.write_text(json.dumps(dict(run=str(run), ks=ks, folds=results,
                                      pooled_population_floor=pooled), indent=1),
                      encoding="utf-8")
    print("pooled LOSO population floor medians: "
          + ", ".join(f"K={k} {v['median']:.3f} Nm (n={v['n']})" for k, v in pooled.items()))
    if failed:
        print(f"K = 8 NOT reproduced for {failed}: no K files were written for them",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
