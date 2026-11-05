"""W-bio's weight curve from the Molinaro trials (our design notes,
"Phase-weighted wearers", cell W-bio; decision of 2026-10-19: W-bio runs as soon as the hip-angle
data can be read, with the weighting exactly as planned and no substitute).

Usage (the data root is the loader's: ``PREFTUNE_DATA`` if set, else the S: share):
    $env:PREFTUNE_DATA = "<root holding molinaro/subjects/AB3x_{l,r}.parquet>"
    python scripts/icorr_wph_bio_weight.py --run <main>/out/phase5-schaefer [--column NAME]

Planned: w(phi) ∝ |theta_dot_hip(phi)| + 0.2 mean |theta_dot_hip|, from the mean hip angular
velocity of the data the population was generated from (Molinaro AB30-AB34, 1.1 m/s), the same
weight for every user, mean 1. Steps:

1. **The data must be the population's.** Every trial file must be readable, and for each LOSO
   fold the 20,000 training ideals are regenerated from this copy and must refit the stored
   8-basis prior bit for bit (``icorr_robust_prep._training_curves``, as the R prep checked its
   standardisation). Otherwise nothing is written.
2. **The hip angle.** For each subject and side, ``load_trial`` gives the walking trial at 1.1 m/s
   trimmed to its longest valid stretch, with its gait phase and heel-strike cycles. The angle
   column (``--column``, or the one column whose name matches ``hip...angle``) is read with the
   loader's own row filter and aligned to the trial by its time stamps, which must match exactly.
3. **Per trial**, each cycle's angle is interpolated onto the 101-point phase grid by gait phase,
   as ``preftune.data.loader.cycle_average`` does; a cycle whose samples do not reach phase 0 or
   1 borrows the neighbouring sample of the previous or next cycle, so the grid's ends are
   interpolated, not clamped (:func:`mean_cycle_curve`); cycles with a missing angle sample are
   dropped and counted. The trial's curve is the mean over its cycles.
4. **theta_dot** of each trial's curve is the periodic central difference on the 100 distinct
   points (``phase_weight.periodic_derivative``); the mean over the ten trials (equal weight) is
   the mean hip angular velocity. Then ``phase_weight.bio_weight``: |theta_dot| + 0.2 mean
   |theta_dot|, normalised to mean 1 over the 101-point grid. Units and sign cancel.

Written: ``<out>/wph_bio_weight.json`` (default ``<run>/../icorr``): the 101-vector and its
provenance (files with size and sha256, column, cycles used and dropped per trial, the
verification, the commit). No subject data is stored: no per-trial or per-subject curve.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preftune.profile import N_PHASE  # noqa: E402

from prompref.user.phase_weight import (BIO_FLOOR_FRACTION, BIO_SCHEMA, bio_weight,  # noqa: E402
                                        periodic_derivative)

SUBJECTS = ("AB30", "AB31", "AB32", "AB33", "AB34")
SIDES = ("l", "r")
SPEED_MPS = 1.1
ANGLE_PATTERN = re.compile(r"^hip_flexion_(rad|deg)$|hip.*angle", re.IGNORECASE)
#: Channels that are not the wearer's own hip angle: the exoskeleton's encoder, the other leg,
#: derivatives and kinetics.
NOT_ANGLE = re.compile(r"exo|contra|moment|torque|vel|acc", re.IGNORECASE)
#: The file's own hip angular velocity, used only as a cross-check of the derivative.
CHECK_COLUMN = "hip_flexion_vel_rad_s"
#: A cycle-averaged trial curve needs at least this many complete cycles (load_trial's own rule).
MIN_CYCLES = 5


def stamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def detect_column(names, requested: str | None = None) -> str:
    """The hip-angle column: ``requested`` if it is one of ``names``, else the one name that is
    ``hip_flexion_rad`` (or ``_deg``) or matches ``hip...angle``, and is not the exoskeleton's
    encoder, the other leg, a derivative or a kinetic channel. Refuses with the candidates listed
    when there is none or more than one. In the Molinaro trials this is ``hip_flexion_rad``, the
    wearer's own hip flexion angle beside the hip flexion moment the population was generated
    from (``exo_hip_angle_rad`` is the device's encoder)."""
    names = list(names)
    if requested is not None:
        if requested not in names:
            raise SystemExit(f"column {requested!r} is not in the trial file; its hip-angle "
                             f"candidates: {[n for n in names if ANGLE_PATTERN.search(n)]}")
        return requested
    found = [n for n in names if ANGLE_PATTERN.search(n) and not NOT_ANGLE.search(n)]
    if len(found) != 1:
        raise SystemExit(f"cannot pick the hip-angle column: candidates {found}; name it with "
                         "--column")
    return found[0]


def align(times_all: np.ndarray, values_all: np.ndarray, trial_times: np.ndarray) -> np.ndarray:
    """``values_all`` on the trial's stretch: the run of ``times_all`` that equals ``trial_times``
    exactly (both come from the loader's own row filter, in the file's order)."""
    times_all, trial_times = np.asarray(times_all, float), np.asarray(trial_times, float)
    starts = np.flatnonzero(times_all == trial_times[0])
    for a in starts:
        b = a + trial_times.size
        if b <= times_all.size and np.array_equal(times_all[a:b], trial_times):
            return np.asarray(values_all, dtype=float)[a:b]
    raise RuntimeError("the angle's time stamps do not contain the trial's stretch")


def mean_cycle_curve(gait_phase: np.ndarray, angle: np.ndarray, cycles) -> tuple[np.ndarray, int]:
    """The mean over cycles of the angle on the 101-point phase grid, and how many cycles were
    dropped for a missing angle sample.

    Each cycle is interpolated by gait phase as ``cycle_average`` does. Where a cycle's samples
    do not reach the grid's ends (first phase above 0, last below 1), the neighbouring sample of
    the previous cycle (phase - 1) or of the next heel strike (phase + 1) is included, so the ends
    are interpolated rather than clamped: a clamped end would put a kink at the heel strike, and
    the periodic derivative would turn it into a spurious velocity there. The Molinaro trials'
    ``gait_phase_hs`` runs from exactly 0 to exactly 1 within each cycle, so there no neighbour is
    needed and the interpolation is ``cycle_average``'s own."""
    grid = np.linspace(0.0, 1.0, N_PHASE)
    gait_phase = np.asarray(gait_phase, dtype=float)
    angle = np.asarray(angle, dtype=float)
    curves, dropped = [], 0
    for a, b in cycles:
        lo, hi = a, b
        phase = gait_phase[a:b]
        if phase[0] > 0.0:
            if a < 1:
                raise ValueError(f"cycle ({a}, {b}) starts above phase 0 with no sample before it")
            lo, phase = a - 1, np.concatenate([[gait_phase[a - 1] - 1.0], phase])
        if phase[-1] < 1.0:
            if b >= gait_phase.size:
                raise ValueError(f"cycle ({a}, {b}) ends below phase 1 with no sample after it")
            hi, phase = b + 1, np.concatenate([phase, [gait_phase[b] + 1.0]])
        segment = angle[lo:hi]
        if not np.all(np.isfinite(segment)):
            dropped += 1
            continue
        if not np.all(np.diff(phase) > 0.0):
            raise ValueError(f"cycle ({a}, {b}): the gait phase does not increase")
        curves.append(np.interp(grid, phase, segment))
    if len(curves) < MIN_CYCLES:
        raise RuntimeError(f"only {len(curves)} cycles with a complete angle ({dropped} dropped)")
    return np.mean(curves, axis=0), dropped


def weight_from_curves(curves) -> tuple[np.ndarray, np.ndarray]:
    """``(weight, mean theta_dot)`` from the trials' mean angle curves: each curve's periodic
    derivative, their mean (equal weight per trial), then the planned weighting."""
    theta_dot = np.mean([periodic_derivative(c) for c in curves], axis=0)
    return bio_weight(theta_dot, BIO_FLOOR_FRACTION), theta_dot


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def readable_files(subject_path) -> list[Path]:
    """Every trial file, refused loudly unless each can be opened and read to its end."""
    files, bad = [], []
    for subject in SUBJECTS:
        for side in SIDES:
            path = Path(subject_path(subject, side))
            try:
                with path.open("rb") as fh:
                    fh.read(16)
                    fh.seek(-16, 2)
                    fh.read(16)
                files.append(path)
            except OSError as exc:
                bad.append(f"{path}: {exc}")
    if bad:
        raise SystemExit("the Molinaro trials cannot be read, so W-bio's weight cannot be built "
                         "(no substitute is used):\n  " + "\n  ".join(bad))
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/../icorr")
    ap.add_argument("--column", default=None, help="the hip-angle column (default: detected)")
    args = ap.parse_args(argv)
    run = args.run.resolve()
    out = args.out or run.parent / "icorr"

    import pyarrow.parquet as pq
    from preftune.data import loader

    files = readable_files(loader.subject_path)
    print(f"data root {loader.DEFAULT_ROOT}: {len(files)} trial files readable", flush=True)

    # 1. the copy must be the data the population was generated from
    import icorr_robust_prep as rp
    from prompref.experiment.phase5c_grid import LOSO

    verified = []
    for fold in LOSO:
        rp._training_curves(run, fold)           # raises unless the stored prior refits exactly
        verified.append(fold)
        print(f"{fold}: training ideals regenerated from this copy refit the stored prior bit "
              "for bit", flush=True)

    # 2-4. the angle, per trial, then the weight
    column = None
    curves, check_curves, trials = [], [], []
    for subject in SUBJECTS:
        for side in SIDES:
            path = Path(loader.subject_path(subject, side))
            column = detect_column(pq.read_schema(path).names, args.column)
            trial = loader.load_trial(subject, side, SPEED_MPS)
            names = pq.read_schema(path).names
            check = CHECK_COLUMN if CHECK_COLUMN in names else None
            frame = loader.load_walking(subject, side, SPEED_MPS,
                                        columns=["activity", "speed_mps", "time_s", column]
                                        + ([check] if check else []))
            times = frame["time_s"].to_numpy(float)
            angle = align(times, frame[column].to_numpy(float), trial.time_s)
            curve, dropped = mean_cycle_curve(trial.gait_phase, angle, trial.cycles)
            curves.append(curve)
            if check:
                velocity = align(times, frame[check].to_numpy(float), trial.time_s)
                check_curves.append(mean_cycle_curve(trial.gait_phase, velocity, trial.cycles)[0])
            trials.append(dict(subject=subject, side=side, cycles=len(trial.cycles),
                               cycles_dropped_missing_angle=int(dropped)))
            print(f"{subject}_{side}: {len(trial.cycles) - dropped} cycles", flush=True)
    weight, _ = weight_from_curves(curves)
    cross_check = None
    if len(check_curves) == len(curves):
        # The file's own velocity (rad/s), cycle-averaged the same way, through the same rule.
        velocity = np.mean(check_curves, axis=0)
        velocity[-1] = velocity[0]
        other = bio_weight(velocity, BIO_FLOOR_FRACTION)
        cross_check = dict(column=CHECK_COLUMN, used_for_the_weight=False,
                           pearson_r=float(np.corrcoef(weight[:-1], other[:-1])[0, 1]),
                           max_abs_difference=float(np.max(np.abs(weight - other))))
        print(f"cross-check against {CHECK_COLUMN}: r = {cross_check['pearson_r']:.4f}, max "
              f"|difference| {cross_check['max_abs_difference']:.3f}", flush=True)
    payload = dict(
        schema=BIO_SCHEMA, when=stamp(), commit=git_head(),
        spec="ICORR reframe section 9a, phase-weighted wearers, W-bio",
        rule="w ∝ |theta_dot_hip| + 0.2 mean |theta_dot_hip|, mean 1 over the 101-point grid",
        floor_fraction=BIO_FLOOR_FRACTION, weight=[float(v) for v in weight],
        provenance=dict(
            data_root=str(loader.DEFAULT_ROOT), speed_mps=SPEED_MPS, subjects=list(SUBJECTS),
            sides=list(SIDES), column=column,
            files=[dict(name=p.name, bytes=p.stat().st_size, sha256=sha256(p)) for p in files],
            trials=trials,
            verification=dict(folds=verified, check="icorr_robust_prep._training_curves: the "
                              "fold's training ideals regenerated from this copy refit the stored "
                              "8-basis prior (mu, sigma) bit for bit"),
            cycle_curve="per trial: each load_trial cycle interpolated onto the 101-point grid by "
                        "gait phase (cycle_average); a cycle not reaching phase 0 or 1 borrows "
                        "the neighbouring cycle's sample so the ends are not clamped; cycles "
                        "with a missing angle dropped; mean over cycles",
            derivative="periodic central difference on the 100 distinct points, per unit phase",
            averaging="mean theta_dot over the ten trials, equal weight per trial",
            velocity_cross_check=cross_check))
    out.mkdir(parents=True, exist_ok=True)
    target = out / "wph_bio_weight.json"
    target.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {target}: weight min {weight.min():.3f}, max {weight.max():.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
