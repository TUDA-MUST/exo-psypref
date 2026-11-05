"""Test NP: users who also dislike negative exoskeleton power (prep, run, report).

Usage:
  python scripts/icorr_amp_np.py velocity [--out <repository>/out]   # per-trial hip angular velocity
  python scripts/icorr_amp_np.py run      [--out <repository>/out] [--workers 8]
  python scripts/icorr_amp_np.py report   [--out <repository>/out]

The users, amplitudes and answer seeds are AR's
(scripts/icorr_amp_range.py): the wide range with AM and the fixed 12 Nm level with AM_fixed.
Each user's theta_dot is the mean cycle of their own trial's hip_flexion_rad, differentiated
as scripts/icorr_wph_bio_weight.py does. Outputs in <out>/icorr/amp_free/: hip_velocity.npz,
np_sessions.jsonl (resumable), amp_np_report.md.
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
from prompref.user.simulated_user import rms_distance  # noqa: E402

SETTINGS = {"wide": "am", "fixed 12": "am_fixed"}


def velocity(out: Path) -> None:
    import pyarrow.parquet as pq
    from preftune.data import loader

    import icorr_wph_bio_weight as wb
    from prompref.user.phase_weight import periodic_derivative

    curves = {}
    for subject in wb.SUBJECTS:
        for side in wb.SIDES:
            path = Path(loader.subject_path(subject, side))
            column = wb.detect_column(pq.read_schema(path).names, None)
            trial = loader.load_trial(subject, side, wb.SPEED_MPS)
            frame = loader.load_walking(subject, side, wb.SPEED_MPS,
                                        columns=["activity", "speed_mps", "time_s", column])
            angle = wb.align(frame["time_s"].to_numpy(float), frame[column].to_numpy(float),
                             trial.time_s)
            curve, _ = wb.mean_cycle_curve(trial.gait_phase, angle, trial.cycles)
            curves[f"{subject}_{side}"] = periodic_derivative(curve)
            print(f"{subject}_{side}: theta_dot range {curves[f'{subject}_{side}'].min():.2f} to "
                  f"{curves[f'{subject}_{side}'].max():.2f} rad per cycle", flush=True)
    dest = out / "icorr" / "amp_free" / "hip_velocity.npz"
    np.savez(dest, **curves)
    print("wrote", dest)


def jobs(out: Path) -> list[dict]:
    js = []
    for j in R.jobs(out):
        if SETTINGS.get(j["setting"]) == j["candidate"]:
            kappa = AF.np_kappas(j["fold_index"], AF.USERS_PER_FOLD)[j["index"]]
            js.append(dict(j, kappa=float(kappa)))
    return js


def run_one(job: dict, out: str) -> dict:
    path = job.pop("_path")
    try:
        d = R._fold(path, job["fold_index"])
        u = job["index"]
        theta_dot = np.load(Path(out) / "icorr" / "amp_free" / "hip_velocity.npz")[str(d["user_subject"][u])]
        ideal = AF.compose(job["amps"][0], job["amps"][1], d["user_minus"][u], d["user_plus"][u])
        g_minus, g_plus = AF.unit_parts(d["generic"])
        seed = int(job["answer_seed"]) % 2**32
        if job["setting"] == "wide":
            lo, hi = R.RANGES["wide"]
            centre = math.sqrt(lo * hi)
            mu, sigma = R._a_moments(path, job["fold_index"], "wide")
            space = AF.promp_space(mu, sigma, limit=hi)
            design, learner = space.design, AF.metric_psypref(space, seed)
        else:
            centre = job["amps"][0]
            design, learner = AF.projected_fixed(d["w_minus"], d["w_plus"], centre, seed)
        generic = AF.compose(centre, centre, g_minus, g_plus)
        observer = AF.NegativePowerObserver(ideal_nm=ideal, s=AF.S, lapse=AF.LAPSE,
                                            theta_dot=theta_dot, kappa=job["kappa"])
        rng = np.random.default_rng([int(job["answer_seed"]), AF.OBSERVER_SALT])
        rec = run_session(design, learner, observer, rng, n_comparisons=AF.K, randomise_order=True,
                          d_prior=rms_distance(generic, ideal), d_floor=0.0,
                          scores={"own": observer.own_distance})
        delivered = design.curve(np.asarray(rec["best"], dtype=float))
        row = dict(job, own_d0=observer.own_distance(generic), own=list(rec["own"]),
                   distances=list(rec["distances"]),
                   neg_power_delivered=AF.negative_power_nm(delivered, theta_dot),
                   neg_power_ideal=AF.negative_power_nm(ideal, theta_dot),
                   identified=list(AF.peaks(delivered)), error=None)
        if job["setting"] == "wide":
            row["own_floor"] = AF.np_floor(ideal, theta_dot, job["kappa"])
        return row
    except Exception as exc:  # one bad session must not stop the pilot
        return dict(job, error=f"{type(exc).__name__}: {exc}")


def _key(row):
    return row["user"], row["setting"]


def run(out: Path, workers: int) -> None:
    path = out / "icorr" / "amp_free" / "np_sessions.jsonl"
    done = set()
    if path.exists():
        done = {_key(r) for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
                if r.get("error") is None}
    todo = [j for j in jobs(out) if _key(j) not in done]
    print(f"{len(todo)} sessions to run ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a", encoding="utf-8") as f:
        for fut in as_completed([pool.submit(run_one, j, str(out)) for j in todo]):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if row.get("error"):
                print("ERROR", _key(row), row["error"], flush=True)


def _own_pref(rows):
    d0 = np.array([r["own_d0"] for r in rows])
    dk = np.array([r["own"][-1] for r in rows])
    return float(np.mean(AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((d0 - dk) / AF.S)))


def report(out: Path) -> None:
    d = out / "icorr" / "amp_free"
    rows = [json.loads(x) for x in (d / "np_sessions.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = {_key(r): r for r in rows if r.get("error") is None}
    errors = [r for r in rows if r.get("error") and _key(r) not in ok]
    rng = np.random.default_rng(AF.SEED)
    L = ["# Test plan NP: users who also dislike negative exoskeleton power", "",
         f"Design note 14 (6599a14). {len(ok)} sessions, {len(errors)} errors. k = {AF.K}, "
         f"s = {AF.S} Nm, λ = {AF.LAPSE}; κ = 0 for about half the users, else log-uniform "
         "[0.5, 2]. Preference over generic assistance in the user's own utility. 95 % intervals: "
         f"{R.N_BOOT} draws, users resampled within folds.", "",
         "| setting | users | n | median κ | preference over generic (own) | median P⁻ delivered / P⁻ of τ* "
         "| median excess own distance over the best ProMP curve (Nm) |", "|---|---|---|---|---|---|---|"]
    res = {}
    for setting in SETTINGS:
        for group, sel in (("κ = 0", lambda r: r["kappa"] == 0.0), ("κ > 0", lambda r: r["kappa"] > 0.0)):
            rs = [r for (u, s), r in ok.items() if s == setting and sel(r)]
            if not rs:
                continue
            pr = R._boot(rs, _own_pref, rng)
            ratio = float(np.median([r["neg_power_delivered"] / max(r["neg_power_ideal"], 1e-9) for r in rs]))
            excess = (float(np.median([r["own"][-1] - r["own_floor"] for r in rs]))
                      if all("own_floor" in r for r in rs) else None)
            res[setting, group] = dict(pref=pr, ratio=ratio, excess=excess, n=len(rs))
            L.append(f"| {setting} | {group} | {len(rs)} | {np.median([r['kappa'] for r in rs]):.2f} | "
                     f"{R._f(pr, 3)} | {ratio:.2f} | {'—' if excess is None else f'{excess:.2f}'} |")
    L += ["", "## Predictions", ""]
    if ("wide", "κ > 0") in res:
        r = res["wide", "κ > 0"]["ratio"]
        L.append(f"- PNP1 (wide, κ > 0: median P⁻ ratio below 0.9): {'held' if r < 0.9 else 'failed'} — {r:.2f}")
    gaps = []
    for setting in SETTINGS:
        if (setting, "κ = 0") in res and (setting, "κ > 0") in res:
            gaps.append((setting, res[setting, "κ = 0"]["pref"][0] - res[setting, "κ > 0"]["pref"][0]))
    if len(gaps) == 2:
        L.append(f"- PNP2 (κ > 0 within 5 points of κ = 0 in both settings): "
                 f"{'held' if all(abs(g) <= 0.05 for _, g in gaps) else 'failed'} — "
                 + ", ".join(f"{s} {100 * g:+.1f}" for s, g in gaps))
    if ("wide", "κ = 0") in res and ("wide", "κ > 0") in res:
        e0, e1 = res["wide", "κ = 0"]["excess"], res["wide", "κ > 0"]["excess"]
        L.append(f"- PNP3 (wide: excess for κ > 0 at most 1.5 times that for κ = 0): "
                 f"{'held' if e1 <= 1.5 * e0 else 'failed'} — {e1:.2f} vs {e0:.2f} Nm")
    if errors:
        L += ["", "## Errors", ""] + [f"- {_key(e)}: {e['error']}" for e in errors[:20]]
    (d / "amp_np_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("velocity", "run", "report"))
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if args.workers > 8:
        raise SystemExit("at most 8 workers on this machine")
    {"velocity": lambda: velocity(args.out), "run": lambda: run(args.out, args.workers),
     "report": lambda: report(args.out)}[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
