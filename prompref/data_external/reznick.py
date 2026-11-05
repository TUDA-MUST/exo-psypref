"""Reznick et al. (2021): treadmill walking at three speeds and five inclines.

Source
    E. Reznick, K. R. Embry, R. Neuman, E. Bolivar-Nieto, N. P. Fey and R. D. Gregg,
    "Lower-limb kinematics and kinetics during continuously varying human locomotion",
    Scientific Data 8:282, 2021, doi:10.1038/s41597-021-01057-9. Data: figshare
    collection doi:10.6084/m9.figshare.c.5175254 (v1), item "Normalized Dataset",
    doi:10.6084/m9.figshare.14558184 (v1), file ``Normalized.mat`` (MATLAB v7.3,
    md5 e62fb9cd3aa8dba5b81a5c9f59e5c910). Licence of the data: CC0 1.0.

What the dataset gives, and what this converter makes of it
    * ``Normalized.<AB>.Walk.<speed>.<incline>.jointMoments.HipMoment`` holds every
      accepted stride, heel strike to heel strike (from the treadmill force plates),
      interpolated to 150 points; left strides first, then right. Vicon Plug-in Gait
      inverse dynamics, internal moment in Nm/kg; component ``x`` (index 0) is the
      sagittal moment, hip flexion positive (paper, "Methods"; it correlates positively
      with Molinaro's mean in every condition).
    * ``events.StrideDetails`` gives each stride's first and last frame at 100 Hz, its
      duration and its leg (1 left, 2 right). Strides whose last frame is the next one's
      first frame are contiguous; the converter lays out each leg's longest contiguous run
      on the 200 Hz clock (``curve_source = "strides"``). Strides were removed by the
      authors' outlier screen, so runs are shorter than the condition.
    * The speed label is checked against the belt speed measured from the heel marker in
      mid-stance (15-35 % of the stride, when the foot moves with the belt). In this
      release AB01's 0.8 and 1.0 m/s labels at +5 degrees are swapped; the converter
      relabels by the measured speed and says so in ``LoadResult.notes``.
      ``speed_mps`` is the measured belt speed. It runs about 4-5 % above the commanded,
      leg-length-normalised speeds the README lists.
    * Body mass, age and sex come from ``ParticipantDetails``. The authors call all ten
      participants able-bodied; AB09 (60 y) and AB10 (59 y) are labelled "older" here by
      Fukuchi's split (young 21-37 y, older 50-84 y).

Conditions: ``s0x8``/``s1``/``s1x2`` (0.8/1.0/1.2 m/s nominal) x ``in10``/``in5``/``i0``/
``i5``/``i10`` (-10 to +10 degrees). The acceleration tasks are not steady walking and
are left out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import MIN_CYCLES, LoadResult, Skipped, data_root, trial_from_strides

DATASET = "reznick2021"
CITATION = ("Reznick E, Embry KR, Neuman R, Bolivar-Nieto E, Fey NP, Gregg RD (2021). "
            "Lower-limb kinematics and kinetics during continuously varying human "
            "locomotion. Sci Data 8:282. doi:10.1038/s41597-021-01057-9")
DATA_DOI = "10.6084/m9.figshare.14558184 (v1), in collection 10.6084/m9.figshare.c.5175254"
LICENCE = "CC0 1.0"

NORMALIZED = Path("raw") / "Normalized.mat"

MOCAP_RATE_HZ = 100.0
HIP_FE_COMPONENT = 0
HIP_FE_SIGN = 1.0

SPEEDS = {"s0x8": 0.8, "s1": 1.0, "s1x2": 1.2}
INCLINES = {"in10": -10.0, "in5": -5.0, "i0": 0.0, "i5": 5.0, "i10": 10.0}
LEGS = {1: ("l", "LHEE"), 2: ("r", "RHEE")}

#: Mid-stance window (share of the stride) in which the heel moves with the belt.
STANCE_WINDOW = (0.15, 0.35)

#: Ages at or above this are "older", as in Fukuchi et al. (2018).
OLDER_FROM_YEARS = 50.0


def dataset_dir(root: Path | str | None = None) -> Path:
    return data_root(root) / DATASET


def open_normalized(root: Path | str | None = None):
    import h5py
    return h5py.File(dataset_dir(root) / NORMALIZED, "r")


def participant(f, ab: str) -> dict:
    """Sex, age (y), body mass (kg) and height (mm) from ``ParticipantDetails``."""
    cell = f["Normalized"][ab]["ParticipantDetails"][()]
    names = ["".join(chr(c) for c in np.ravel(f[cell[0, j]][()])) for j in range(cell.shape[1])]
    values = {}
    for j, name in enumerate(names):
        v = f[cell[1, j]][()]
        if f[cell[1, j]].attrs.get("MATLAB_class", b"") == b"char":
            values[name] = "".join(chr(c) for c in np.ravel(v))
        else:
            values[name] = float(np.ravel(v)[0])
    return {"sex": {1.0: "F", 2.0: "M"}.get(values["Sex"], "?"), "age": values["Age"],
            "mass_kg": values["Bodymass"], "height_mm": values["Height"]}


def condition_group(f, ab: str, speed: str, incline: str):
    return f["Normalized"][ab]["Walk"][speed][incline]


def stride_details(group) -> np.ndarray:
    """``(n, 4)``: first frame, last frame, duration in frames, leg (1 left, 2 right)."""
    return np.atleast_2d(group["events"]["StrideDetails"][()].T)


def belt_speed(group, leg: int) -> np.ndarray:
    """Per-stride speed (m/s) of the heel marker in mid-stance, one value per stride."""
    details = stride_details(group)
    n = int(np.sum(details[:, 3] == leg))
    heel = np.asarray(group["markers"][LEGS[leg][1]][()], dtype=float)
    heel = heel[None] if heel.ndim == 2 else heel
    heel = heel[:n]
    dur = details[details[:, 3] == leg, 2] / MOCAP_RATE_HZ
    m = heel.shape[-1] - 1
    i0, i1 = int(STANCE_WINDOW[0] * m), int(STANCE_WINDOW[1] * m)
    dist = np.linalg.norm(heel[:, :3, i1] - heel[:, :3, i0], axis=1)
    return dist / ((i1 - i0) * dur[: heel.shape[0]] / m)


def leg_strides(group, leg: int):
    """One leg's hip moment strides, flexion positive, with their frames, in time order.

    Returns ``(curves (n, 150), first_frame (n,), last_frame (n,))``; strides with missing
    samples are dropped.
    """
    details = stride_details(group)
    moments = np.asarray(group["jointMoments"]["HipMoment"][()], dtype=float)
    moments = moments[None] if moments.ndim == 2 else moments
    rows = np.flatnonzero(details[:, 3] == leg)
    curves = HIP_FE_SIGN * moments[rows, HIP_FE_COMPONENT, :]
    first, last = details[rows, 0], details[rows, 1]
    order = np.argsort(first)
    curves, first, last = curves[order], first[order], last[order]
    ok = np.all(np.isfinite(curves), axis=1)
    return curves[ok], first[ok], last[ok]


def longest_contiguous_run(first, last) -> slice:
    """The longest run of strides in which each one ends where the next begins."""
    first, last = np.asarray(first), np.asarray(last)
    if first.size == 0:
        return slice(0, 0)
    best, run_start = (0, 1), 0
    for k in range(1, first.size + 1):
        if k == first.size or first[k] != last[k - 1]:
            if k - run_start > best[1] - best[0]:
                best = (run_start, k)
            run_start = k
    return slice(*best)


def speed_classes(measured: dict, labels=SPEEDS) -> dict:
    """The nominal speed each condition's measured belt speed belongs to.

    ``measured`` maps ``(speed_label, incline)`` to a belt speed. The subject's reference
    is the median measured speed of its 1.0 m/s labels; each condition goes to the
    nominal speed nearest ``measured / reference``.
    """
    ref = np.nanmedian([v for (s, _), v in measured.items() if labels[s] == 1.0])
    out = {}
    for key, v in measured.items():
        nominal = min(labels, key=lambda s: abs(v / ref - labels[s]))
        out[key] = nominal
    return out


def load_trials(root: Path | str | None = None, subjects=None, speeds=None,
                inclines=None) -> LoadResult:
    """Every usable subject-condition-leg of the walking tasks as an ``ExternalTrial``."""
    f = open_normalized(root)
    try:
        subjects = list(f["Normalized"].keys()) if subjects is None else list(subjects)
        speeds = list(SPEEDS) if speeds is None else list(speeds)
        inclines = list(INCLINES) if inclines is None else list(inclines)
        trials, skipped, notes = [], [], []
        for ab in subjects:
            info = participant(f, ab)
            walk = f["Normalized"][ab]["Walk"]
            present = [(s, i) for s in SPEEDS for i in INCLINES
                       if s in walk and i in walk[s]]
            measured = {}
            for s, i in present:
                g = walk[s][i]
                v = np.concatenate([belt_speed(g, leg) for leg in LEGS])
                measured[(s, i)] = float(np.median(v)) if v.size else float("nan")
            classes = speed_classes(measured)
            for i in inclines:
                taken = [classes[(s, i)] for s in SPEEDS if (s, i) in classes]
                for s in speeds:
                    cond = f"{s}_{i}"
                    if (s, i) not in classes:
                        skipped.append(Skipped(ab, cond, "-", "condition not recorded"))
                        continue
                    true_speed = classes[(s, i)]
                    if taken.count(true_speed) > 1:
                        skipped.append(Skipped(ab, cond, "-", "speed label ambiguous"))
                        continue
                    if true_speed != s:
                        notes.append(f"{ab} {cond}: belt speed {measured[(s, i)]:.2f} m/s "
                                     f"belongs to {true_speed}; relabelled {true_speed}_{i}")
                    label = f"{true_speed}_{i}"
                    g = walk[s][i]
                    for leg, (side, _) in LEGS.items():
                        curves, first, last = leg_strides(g, leg)
                        run = longest_contiguous_run(first, last)
                        n = run.stop - run.start
                        if n < MIN_CYCLES + 2:
                            skipped.append(Skipped(ab, label, side,
                                                   f"longest contiguous run {n} strides"))
                            continue
                        durations = (last[run] - first[run]) / MOCAP_RATE_HZ
                        trials.append(trial_from_strides(
                            ab, side, info["mass_kg"], measured[(s, i)], curves[run],
                            durations, dataset=DATASET, condition=label,
                            environment="treadmill", incline_deg=INCLINES[i],
                            nominal_speed=f"{SPEEDS[true_speed]:.1f} m/s",
                            age_group="older" if info["age"] >= OLDER_FROM_YEARS else "young",
                            age_years=info["age"], sex=info["sex"],
                            stride_source="force-plate heel strikes (dataset events)",
                            measured_speed_mps=measured[(s, i)]))
        return LoadResult(trials, skipped, notes)
    finally:
        f.close()
