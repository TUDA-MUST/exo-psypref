"""Test FULL: PsyPref with the amplitude-free methods against the comparators.

Usage:
  python scripts/icorr_amp_full.py prep
  python scripts/icorr_amp_full.py run [--workers 8]
  python scripts/icorr_amp_full.py report

200 new users (40 per LOSO fold, seed 20261028);
settings medium 3-12 Nm, wide 3-35 Nm, fixed 12 Nm; 40 comparisons. The priors reuse the pilot's
training parts (fold-<name>.npz). Outputs in <out>/icorr/amp_free/: full-<fold>.npz,
full_sessions.jsonl (resumable), amp_full_report.md.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_range as R  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.experiment.run import run_session  # noqa: E402
from prompref.population.folds import loso_folds  # noqa: E402
from prompref.user.simulated_user import Observer, rms_distance  # noqa: E402

N_COMP = 40
RANGES = {"medium": (3.0, 12.0), "wide": (3.0, 35.0)}
LEVEL = 12.0
SETTINGS = ("medium", "wide", "fixed 12")
#: "cmaes_sch4" since a change: CMA-ES on Schaefer's family with its earlier tuning
CANDIDATES = ("psypref", "gp_bald", "g", "d_z", "cmaes", "cmaes_sch", "cmaes_sch4")
NAMES = {"psypref": "PsyPref", "gp_bald": "GP-BALD", "g": "Arens' GP", "d_z": "LineCoSpar",
         "cmaes": "CMA-ES (ProMP)", "cmaes_sch": "CMA-ES (Schäfer, default population)",
         "cmaes_sch4": "CMA-ES (Schäfer, tuned)"}
KS = (12, 24, 40)


def _dir(out: Path) -> Path:
    return out / "icorr" / "amp_free"


def prep(out: Path) -> None:
    run_dir = out / "phase5-schaefer"
    with ProcessPoolExecutor(max_workers=5) as pool:
        for p in pool.map(AF.build_full_users, range(5), [run_dir] * 5, [_dir(out)] * 5, [RANGES] * 5):
            print("wrote", p, flush=True)


@lru_cache(maxsize=None)
def _users(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def jobs(out: Path) -> list[dict]:
    js = []
    for i, fold in enumerate(loso_folds()):
        upath = str(_dir(out) / f"full-{fold.name}.npz")
        ppath = str(_dir(out) / f"fold-{fold.name}.npz")
        z = _users(upath)
        for si, setting in enumerate(SETTINGS):
            for u in range(AF.FULL_USERS_PER_FOLD):
                amps = ([LEVEL, LEVEL] if setting == "fixed 12"
                        else [float(x) for x in z[f"amps_{setting}"][u]])
                for cand in CANDIDATES:
                    js.append(dict(fold=fold.name, fold_index=i, index=u, user=f"{fold.name}/f{u}",
                                   setting=setting, candidate=cand, amps=amps,
                                   answer_seed=int(z["answer_seeds"][u]) + si,
                                   _users=upath, _parts=ppath))
    # the long GP-BALD sessions first, so they do not trail at the end
    return sorted(js, key=lambda j: j["candidate"] != "gp_bald")


def _k_space(ppath: str, fold_index: int, setting: str, n_basis: int, arm: str = "tight"):
    """Test KS (section 28): PsyPref's space with ``n_basis`` von Mises bases, the prior
    from the training shapes projected onto them (``fold-<name>-K<n>.npz``, as K12's): in a
    range at AR's amplitudes, at the fixed level the unit shapes."""
    basis = AF.VonMisesBasis(n_basis)
    if setting in RANGES:
        mu, sigma = R._ak_moments(ppath, fold_index, setting, n_basis)
        return AF.promp_space(mu, sigma, limit=RANGES[setting][1], basis=basis, arm=arm)
    p = R._parts_k(ppath, n_basis)
    return AF.fixed_level_space(p["w_minus"], p["w_plus"], LEVEL, basis=basis, arm=arm)


def run_one(job: dict, out: str) -> dict:
    upath, ppath = job.pop("_users"), job.pop("_parts")
    try:
        z, d = _users(upath), R._fold(ppath, job["fold_index"])
        u, setting, cand = job["index"], job["setting"], job["candidate"]
        run_dir = Path(out) / "phase5-schaefer"
        shape = (z["user_minus"][u], z["user_plus"][u])
        ideal = AF.compose(job["amps"][0], job["amps"][1], *shape)
        g_minus, g_plus = AF.unit_parts(d["generic"])
        seed = int(job["answer_seed"]) % 2**32
        arm = job.get("arm", "tight")   # test ABL: PsyPref with the wide arm
        if arm != "tight" and cand != "psypref":
            raise ValueError(f"arm {arm} is for PsyPref only")
        duels = job.get("duels", "eig")   # test RND: PsyPref with random pairs
        if duels != "eig" and cand != "psypref":
            raise ValueError(f"duels {duels} is for PsyPref only")
        # test LONG (section 27): the budget, and PsyPref-T's tie handling by name
        n_comp, ties = int(job.get("n_comp", N_COMP)), job.get("ties")
        if ties is not None and cand != "psypref":
            raise ValueError(f"ties {ties} is for PsyPref only")
        # passed only when set: TIE's script swaps in factories without the keyword
        tkw = {} if ties is None else dict(ties=ties)
        # test KS (section 28): the K-basis prior and space; 8 is FULL's (unchanged)
        n_basis = int(job.get("n_basis", 8))
        if n_basis != 8 and cand.startswith("cmaes_sch"):
            raise ValueError(f"n_basis {n_basis} is for the ProMP space only")
        if setting in RANGES:
            lo, hi = RANGES[setting]
            centre = math.sqrt(lo * hi)
            if n_basis == 8:
                mu, sigma = R._a_moments(ppath, job["fold_index"], setting)
                space = AF.promp_space(mu, sigma, limit=hi, arm=arm)
            else:
                space = _k_space(ppath, job["fold_index"], setting, n_basis, arm=arm)
            if cand == "psypref":
                learner = AF.TwoPartGuard(AF.metric_psypref(space, seed, arm=arm, **tkw), space,
                                          0.5 * lo, seed=seed)
            elif cand.startswith("cmaes_sch"):
                space = AF.schaefer_range_space(run_dir, job["fold"], lo, hi)
                learner = AF.comparator(cand, space, seed)
            else:
                learner = AF.comparator(cand, space, seed)
        else:
            centre = LEVEL
            space = (AF.fixed_level_space(d["w_minus"], d["w_plus"], LEVEL, arm=arm) if n_basis == 8
                     else _k_space(ppath, job["fold_index"], setting, n_basis, arm=arm))
            if cand == "psypref":
                learner = AF.psypref(space, seed, arm=arm, **tkw)
            elif cand.startswith("cmaes_sch"):
                space = AF.schaefer_fixed_space(run_dir, job["fold"], LEVEL)
                learner = AF.comparator(cand, space, seed)
            else:
                learner = AF.comparator(cand, space, seed)
        if duels != "eig":
            # Set after construction: PsyPref-M's constructor fixes EIG pairs, and the mode is
            # only read when a pair is proposed and the curve delivered.
            getattr(learner, "learner", learner).duels = duels
        generic = AF.compose(centre, centre, g_minus, g_plus)
        observer = Observer(ideal_nm=ideal, s=AF.S, lapse=AF.LAPSE)
        rng = np.random.default_rng([int(job["answer_seed"]), AF.OBSERVER_SALT])
        try:
            rec = run_session(space.design, learner, observer, rng, n_comparisons=n_comp,
                              randomise_order=True, d_prior=rms_distance(generic, ideal), d_floor=0.0)
        finally:
            inner = getattr(learner, "inner", None) or getattr(learner, "learner", None)
            inner = getattr(inner, "inner", inner)
            if inner is not None and hasattr(inner, "close"):
                inner.close()
        delivered = space.design.curve(np.asarray(rec["best"], dtype=float))
        ident = AF.peaks(delivered)
        try:
            shape_error = rms_distance(AF.compose(1, 1, *AF.unit_parts(delivered)), AF.compose(1, 1, *shape))
        except AF.OrderViolation:
            shape_error = None
        extra = ({} if not isinstance(learner, AF.TwoPartGuard)
                 else dict(guard_fired=int(learner.fired), guard_final=bool(learner.fired_last)))
        if "n_basis" in job:   # test KS: the search dimension, only when K is named
            extra["d"] = int(space.d)
        return dict(job, identified=[max(ident[0], 0.0), max(ident[1], 0.0)],
                    d0=rms_distance(generic, ideal), distances=list(rec["distances"]),
                    d_generic=rms_distance(delivered, generic), shape_error=shape_error, error=None,
                    **extra)
    except Exception as exc:  # one bad session must not stop the run
        return dict(job, error=f"{type(exc).__name__}: {exc}")


def _key(row):
    return row["user"], row["setting"], row["candidate"]


def run(out: Path, workers: int) -> None:
    path = _dir(out) / "full_sessions.jsonl"
    done = set()
    if path.exists():
        done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = [j for j in jobs(out) if _key(j) not in done]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        futures = [pool.submit(run_one, j, str(out)) for j in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", _key(row), row["error"], flush=True)
            if n % 100 == 0:
                print(f"{n}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------- report

def _pref_k(rows, k):
    d0 = np.array([r["d0"] for r in rows])
    dk = np.array([r["distances"][k - 1] for r in rows])
    return float(np.mean(AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((d0 - dk) / AF.S)))


def _curve(rows):
    d0 = np.array([r["d0"] for r in rows])[:, None]
    dk = np.array([r["distances"] for r in rows])
    return np.mean(AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((d0 - dk) / AF.S), axis=0)


def _pairs(rows, setting, c1, c2):
    a = {r["user"]: r for (u, s, c), r in rows.items() if s == setting and c == c1}
    b = {r["user"]: r for (u, s, c), r in rows.items() if s == setting and c == c2}
    return [dict(fold=a[u]["fold"], a=a[u], b=b[u]) for u in sorted(set(a) & set(b))]


def report(out: Path) -> None:
    d = _dir(out)
    raw = [json.loads(x) for x in (d / "full_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = {_key(r): r for r in raw if r.get("error") is None}
    errors = [r for r in raw if r.get("error") and _key(r) not in rows]
    rng = np.random.default_rng(AF.SEED_FULL)
    L = ["# Test plan FULL: PsyPref with the amplitude-free methods against the comparators", "",
         f"Design note 16 (699108c). {len(rows)} sessions, {len(errors)} errors. 200 new users per "
         f"setting, s = {AF.S} Nm, λ = {AF.LAPSE}, {N_COMP} comparisons. Preference over generic "
         f"assistance, 95 % intervals from {R.N_BOOT} draws, users resampled within folds.", ""]
    diffs, reach = {}, {}
    for setting in SETTINGS:
        L += [f"## {setting}", "", "| optimizer | n | k = 12 | k = 24 | k = 40 |", "|---|---|---|---|---|"]
        for cand in CANDIDATES:
            rs = [r for (u, s, c), r in rows.items() if s == setting and c == cand]
            if len(rs) < 8:
                continue
            cells = [R._f(R._boot(rs, lambda x, k=k: _pref_k(x, k), rng), 3) for k in KS]
            L.append(f"| {NAMES[cand]} | {len(rs)} | " + " | ".join(cells) + " |")
        L += ["", "PsyPref minus each comparator (points, paired) and the first k at which PsyPref "
              f"reaches the comparator's {N_COMP}-comparison preference ({N_COMP + 1}: not within {N_COMP}):", ""]
        for cand in CANDIDATES[1:]:
            ps = _pairs(rows, setting, "psypref", cand)
            if len(ps) < 8:
                continue
            for k in (24, 40):
                t = R._boot(ps, lambda x, k=k: _pref_k([p["a"] for p in x], k) - _pref_k([p["b"] for p in x], k), rng)
                diffs[setting, cand, k] = t
            def first(x):
                target = _curve([p["b"] for p in x])[N_COMP - 1]
                hit = np.nonzero(_curve([p["a"] for p in x]) >= target)[0]
                return float(hit[0] + 1) if len(hit) else float(N_COMP + 1)
            reach[setting, cand] = R._boot(ps, first, rng)
            t24, t40, f = diffs[setting, cand, 24], diffs[setting, cand, 40], reach[setting, cand]
            L.append(f"- {NAMES[cand]}: k = 24 {100 * t24[0]:+.1f} [{100 * t24[1]:+.1f}, {100 * t24[2]:+.1f}], "
                     f"k = 40 {100 * t40[0]:+.1f} [{100 * t40[1]:+.1f}, {100 * t40[2]:+.1f}]; "
                     f"reaches its k = 40 level at k = {f[0]:.0f} [{f[1]:.0f}, {f[2]:.0f}]")
        if setting in RANGES:
            lo, hi = RANGES[setting]
            L += ["", "| optimizer | R² ext | R² flex | slope ext / flex | end bias ext (low, high) | "
                  "end bias flex (low, high) |", "|---|---|---|---|---|---|"]
            for cand in CANDIDATES:
                rs = [r for (u, s, c), r in rows.items() if s == setting and c == cand]
                if len(rs) < 8:
                    continue
                r2 = [R._r2(rs, b) for b in (0, 1)]
                eb = [R._end_bias(rs, b, lo, hi) for b in (0, 1)]
                L.append(f"| {NAMES[cand]} | {r2[0]:.2f} | {r2[1]:.2f} | {R._slope(rs, 0):.2f} / {R._slope(rs, 1):.2f} | "
                         f"{eb[0][0]:+.2f}, {eb[0][1]:+.2f} | {eb[1][0]:+.2f}, {eb[1][1]:+.2f} |")
                if cand == "psypref":
                    diffs[setting, "r2"] = r2
        else:
            L += ["", "Median shape error (unit peaks, RMS): " + ", ".join(
                f"{NAMES[c]} {np.median([r['shape_error'] for (u, s, cc), r in rows.items() if s == setting and cc == c and r['shape_error'] is not None]):.3f}"
                for c in CANDIDATES if any(s == setting and cc == c for (u, s, cc) in rows))]
        g = [r for (u, s, c), r in rows.items() if s == setting and c == "psypref" and "guard_fired" in r]
        if g:
            L.append(f"\nGuard: fired in {100 * np.mean([r['guard_fired'] > 0 for r in g]):.1f} % of PsyPref's "
                     f"sessions, on the final curve in {sum(r['guard_final'] for r in g)}.")
        L.append("")

    def ahead(key):
        t = diffs.get(key)
        return None if t is None else ("held" if t[1] > 0 else ("failed" if t[2] < 0 else "uncertain"))

    L += ["## Predictions", ""]
    L.append(f"- PF1 (wide: AM ahead of GP-BALD): {ahead(('wide', 'gp_bald', 24))}")
    v = [ahead((s, "cmaes_sch4", 24)) for s in RANGES]
    L.append(f"- PF2 (medium and wide: AM ahead of CMA-ES on Schäfer (ii)): {v}")
    L.append(f"- PF3 (fixed 12: B_fixed ahead of CMA-ES on Schäfer (i)): {ahead(('fixed 12', 'cmaes_sch4', 24))}")
    v = {(s, c): ahead((s, c, 24)) for s in SETTINGS for c in ("g", "d_z", "cmaes")}
    L.append(f"- PF4 (ahead of Arens' GP, LineCoSpar, CMA-ES on the ProMP in every setting): "
             f"{'held' if all(x == 'held' for x in v.values()) else v}")
    if ("wide", "cmaes_sch4") in reach:
        f = reach["wide", "cmaes_sch4"]
        L.append(f"- PF5 (wide: AM reaches CMA-ES Schäfer's k = 40 level by k = 15): {'held' if f[0] <= 15 else 'failed'} — k {f[0]:.0f}")
    if ("wide", "r2") in diffs:
        r2 = diffs["wide", "r2"]
        L.append(f"- PF6 (wide: AM's R² ≥ 0.6 for both bouts): {'held' if min(r2) >= 0.6 else 'failed'} — {r2[0]:.2f} / {r2[1]:.2f}")
    L.append(f"- PF7 (fixed 12: B_fixed ahead of GP-BALD): {ahead(('fixed 12', 'gp_bald', 24))}")
    if errors:
        L += ["", "## Errors", ""] + [f"- {_key(e)}: {e['error']}" for e in errors[:30]]
    (d / "amp_full_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("prep", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"prep": lambda: prep(args.out), "run": lambda: run(args.out, args.workers),
     "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
