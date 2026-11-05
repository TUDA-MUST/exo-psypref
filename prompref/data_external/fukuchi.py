"""Fukuchi, Fukuchi & Duarte (2018): young and older adults walking on a treadmill and overground.

Source
    C. A. Fukuchi, R. K. Fukuchi and M. Duarte, "A public dataset of overground and
    treadmill walking kinematics and kinetics in healthy individuals", PeerJ 6:e4640,
    2018, doi:10.7717/peerj.4640. Data: figshare, doi:10.6084/m9.figshare.5722711,
    **version 1** (2017-12-28): ``WBDSascii.zip`` (md5 ad9e6311d9b84b53acaf58d114d51c6d)
    and ``WBDSinfo.xlsx``. Licence CC BY 4.0.

Why version 1. The current version 6 (2026-09-13) adds nine older adults, but its
left-side treadmill files at T02-T06 for subjects 25-48 are value-identical copies of
subjects 1-24's (all left-side columns and both GRFs); version 1, the release the paper
describes (24 young, 18 older adults), has no duplicated curve.

What the dataset gives, and what this converter makes of it
    * Hip moments only as each participant's ensemble-mean cycle per condition and side
      (``WBDSnnwalk<cond>knt.txt``, 101 points from heel strike to heel strike; Visual3D,
      gait events from the force plates, ``JOINT_MOMENT`` of thigh on pelvis resolved in
      the pelvis frame, ``DEFAULT_NORMALIZATION`` = divided by body mass). So the unit is
      Nm/kg, internal moment. The sagittal component is ``{R,L}HipMomentZ`` and it is
      already flexion positive: it is negative after heel strike and positive around
      toe-off, and correlates with Molinaro's 1.1 m/s mean at r ~ 0.9 on both sides; ``X``
      is the frontal-plane (abductor) moment and ``Y`` the small rotation moment.
    * There are no per-stride moments, so each trial tiles the ensemble mean
      (``curve_source = "ensemble-mean"``): no stride-to-stride variability.
    * The stride time is measured from heel strikes in the marker data (Zeni et al. 2008:
      the heel's most forward position relative to the sacrum, the mean of the two PSIS
      markers), pooled over both sides and, overground, over every trial of the condition.

Conditions: treadmill ``T01``-``T08`` at 40-145 % of the participant's comfortable
Froude speed (30 s recorded per speed), and overground ``OS``/``OC``/``OF`` at slow,
comfortable and fast (comfortable -/+ 30 %).
"""

from __future__ import annotations

import io
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .common import (DEFAULT_TILED_CYCLES, LoadResult, Skipped, data_root,
                     trial_from_mean_cycle)

DATASET = "fukuchi2018"
CITATION = ("Fukuchi CA, Fukuchi RK, Duarte M (2018). A public dataset of overground and "
            "treadmill walking kinematics and kinetics in healthy individuals. PeerJ 6:e4640. "
            "doi:10.7717/peerj.4640")
DATA_DOI = "10.6084/m9.figshare.5722711 (version 1)"
LICENCE = "CC BY 4.0"

#: Files of the 2018 release, relative to the dataset folder.
ARCHIVE = Path("raw") / "v1_2018" / "WBDSascii.zip"
INFO = Path("raw") / "v1_2018" / "WBDSinfo.xlsx"

#: The sagittal hip moment column of the processed files, and the factor that makes it
#: flexion positive (it already is).
HIP_FE_COLUMN = "{side}HipMomentZ"
HIP_FE_SIGN = 1.0

MKR_RATE_HZ = 150.0

#: Condition code -> (environment, nominal speed label).
CONDITIONS = {
    "T01": ("treadmill", "40 %"), "T02": ("treadmill", "55 %"), "T03": ("treadmill", "70 %"),
    "T04": ("treadmill", "85 %"), "T05": ("treadmill", "100 %"), "T06": ("treadmill", "115 %"),
    "T07": ("treadmill", "130 %"), "T08": ("treadmill", "145 %"),
    "OS": ("overground", "slow"), "OC": ("overground", "comfortable"),
    "OF": ("overground", "fast"),
}

SIDES = {"R": "r", "L": "l"}

#: Stride times outside this range are detection errors, not strides.
STRIDE_RANGE_S = (0.6, 2.5)

#: Fewer overground strides than this and the stride time comes from the treadmill fit.
MIN_OVERGROUND_STRIDES = 3


def dataset_dir(root: Path | str | None = None) -> Path:
    return data_root(root) / DATASET


def subject_id(number: int) -> str:
    return f"WBDS{int(number):02d}"


def _member(cond: str, subject: int, kind: str, trial: int | None = None) -> str:
    """The archive name of a file: kind is 'knt', 'ang', 'mkr' or 'grf'."""
    sid = subject_id(subject)
    if cond.startswith("T"):
        return f"{sid}walk{cond}{kind}.txt"
    speed = cond[1]
    if trial is None:
        return f"{sid}walkO{speed}{kind}.txt"
    return f"{sid}walkO{trial:02d}{speed}{kind}.txt"


@lru_cache(maxsize=4)
def read_info(root: Path | str | None = None) -> pd.DataFrame:
    """The metadata table, one row per file, with a numeric ``speed_mps`` column."""
    info = pd.read_excel(dataset_dir(root) / INFO)
    info = info.loc[:, [c for c in info.columns if not str(c).startswith("Unnamed")]]
    info["speed_mps"] = pd.to_numeric(info["GaitSpeed(m/s)"], errors="coerce")
    return info


def subject_table(root: Path | str | None = None) -> pd.DataFrame:
    """One row per participant: Mass (kg), Age, AgeGroup, Gender, Height, LegLength."""
    info = read_info(root)
    cols = ["Subject", "AgeGroup", "Age", "Mass", "Gender", "Height", "LegLength"]
    return info.drop_duplicates("Subject")[cols].set_index("Subject").sort_index()


def condition_speed(info: pd.DataFrame, subject: int, cond: str) -> float:
    """Walking speed from the metadata: the treadmill speed, or the mean over the
    condition's overground trials."""
    sid = subject_id(subject)
    if cond.startswith("T"):
        pattern = rf"^{sid}walk{cond}(\.c3d|mkr\.txt)$"
    else:
        pattern = rf"^{sid}walkO\d\d{cond[1]}(\.c3d|mkr\.txt)$"
    rows = info[info["FileName"].astype(str).str.match(pattern)]
    if rows.empty:
        return float("nan")
    trial = rows["FileName"].str.replace(r"(\.c3d|mkr\.txt)$", "", regex=True)
    per_trial = rows.assign(trial=trial).groupby("trial")["speed_mps"].mean()
    return float(per_trial.mean())


def heel_strike_times(mkr: pd.DataFrame, side: str, fs: float = MKR_RATE_HZ) -> np.ndarray:
    """Heel strikes (s from the first frame) of one side, after Zeni et al. (2008).

    The heel's position along the direction of progression relative to the sacrum peaks
    at heel strike. The foot's own direction (first metatarsal minus heel) gives the
    direction of progression, so treadmill and overground trials in either direction are
    handled alike.
    """
    sacrum = 0.5 * (mkr["R.PSISX"].to_numpy(float) + mkr["L.PSISX"].to_numpy(float))
    heel = mkr[f"{side}.HeelX"].to_numpy(float)
    toe = mkr[f"{side}.MT1X"].to_numpy(float)
    facing = np.sign(np.nanmedian(toe - heel))
    rel = facing * (heel - sacrum)
    finite = np.isfinite(rel)
    if facing == 0.0 or finite.sum() < int(fs):
        return np.array([])
    lo, hi = np.nanpercentile(rel, [5.0, 95.0])
    filled = np.where(finite, rel, np.nanmin(rel))
    peaks, _ = find_peaks(filled, distance=int(0.6 * fs), prominence=0.4 * (hi - lo))
    peaks = peaks[finite[peaks]]
    return peaks / fs


def stride_times(mkr: pd.DataFrame, fs: float = MKR_RATE_HZ) -> np.ndarray:
    """Heel-strike-to-heel-strike times of both sides, within ``STRIDE_RANGE_S``."""
    out = []
    for side in SIDES:
        d = np.diff(heel_strike_times(mkr, side, fs))
        out.append(d[(d >= STRIDE_RANGE_S[0]) & (d <= STRIDE_RANGE_S[1])])
    return np.concatenate(out)


_MKR_COLUMNS = ["R.PSISX", "L.PSISX", "R.HeelX", "L.HeelX", "R.MT1X", "L.MT1X"]


def _read_txt(zf: zipfile.ZipFile, name: str, usecols=None) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(name)), sep="\t", usecols=usecols)


def measured_strides(zf: zipfile.ZipFile, subject: int, cond: str) -> np.ndarray:
    """All stride times of a subject in a condition, from the marker files."""
    names = set(zf.namelist())
    if cond.startswith("T"):
        files = [_member(cond, subject, "mkr")]
    else:
        pattern = re.compile(rf"^{subject_id(subject)}walkO\d\d{cond[1]}mkr\.txt$")
        files = sorted(n for n in names if pattern.match(n))
    out = [stride_times(_read_txt(zf, f, usecols=_MKR_COLUMNS)) for f in files if f in names]
    return np.concatenate(out) if out else np.array([])


def treadmill_fit(speeds, strides):
    """Per-subject power law ``stride = a * speed**b`` through the treadmill conditions."""
    v, t = np.asarray(speeds, float), np.asarray(strides, float)
    ok = np.isfinite(v) & np.isfinite(t) & (v > 0) & (t > 0)
    if ok.sum() < 3:
        return None
    b, log_a = np.polyfit(np.log(v[ok]), np.log(t[ok]), 1)
    return lambda speed: float(np.exp(log_a) * speed ** b)


def hip_curve(knt: pd.DataFrame, side: str) -> tuple[np.ndarray | None, str]:
    """The sagittal hip moment of one side, flexion positive, or ``None`` and why not.

    Some overground files miss only the first point (heel strike); it is set to the last
    point, the next heel strike of the same cycle average.
    """
    col = HIP_FE_COLUMN.format(side=side)
    if col not in knt:
        return None, f"no column {col}"
    c = HIP_FE_SIGN * knt[col].to_numpy(float)
    if c.size != 101:
        return None, f"{c.size} points instead of 101"
    bad = np.flatnonzero(~np.isfinite(c))
    if bad.size == 0:
        return c, ""
    if bad.tolist() == [0] and np.isfinite(c[-1]):
        c = c.copy()
        c[0] = c[-1]
        return c, "first point missing, set to the last"
    return None, f"{bad.size} missing points"


def duplicate_hip_curves(archive: Path | str) -> list[list[str]]:
    """Groups of processed files whose hip moment curve (one side) is value-identical
    across *different* subjects -- a release defect, not a coincidence. Each entry reads
    ``"WBDSnnwalk<cond>:<side>"``."""
    groups: dict[bytes, list[str]] = {}
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(zf.namelist()):
            base = name.rsplit("/", 1)[-1]
            if not base.endswith("knt.txt"):
                continue
            knt = _read_txt(zf, name)
            for side in SIDES:
                c = knt[HIP_FE_COLUMN.format(side=side)].to_numpy(float)
                if np.all(np.isfinite(c)):
                    groups.setdefault(np.round(c, 6).tobytes(), []).append(
                        f"{base[:-7]}:{side}")
    return [g for g in groups.values() if len({x[:6] for x in g}) > 1]


def load_trials(root: Path | str | None = None, subjects=None, conditions=None,
                n_cycles: int = DEFAULT_TILED_CYCLES) -> LoadResult:
    """Every usable subject-condition-side of the 2018 release as an ``ExternalTrial``."""
    info = read_info(root)
    table = subject_table(root)
    subjects = list(table.index) if subjects is None else [int(s) for s in subjects]
    conditions = list(CONDITIONS) if conditions is None else list(conditions)
    zf = zipfile.ZipFile(dataset_dir(root) / ARCHIVE)
    names = set(zf.namelist())
    trials, skipped, notes = [], [], []
    seen: dict[bytes, str] = {}
    for subject in subjects:
        meta = table.loc[subject]
        sid = subject_id(subject)
        # Treadmill stride times first: the overground fallback needs them.
        measured: dict[str, np.ndarray] = {}
        speeds: dict[str, float] = {}
        for cond in conditions:
            if _member(cond, subject, "knt") in names:
                measured[cond] = measured_strides(zf, subject, cond)
                speeds[cond] = condition_speed(info, subject, cond)
        tm = [c for c in measured if c.startswith("T") and measured[c].size]
        fit = treadmill_fit([speeds[c] for c in tm], [np.median(measured[c]) for c in tm])
        for cond in conditions:
            environment, nominal = CONDITIONS[cond]
            knt_name = _member(cond, subject, "knt")
            if knt_name not in names:
                skipped.append(Skipped(sid, cond, "-", "condition not recorded"))
                continue
            strides = measured[cond]
            if strides.size >= (MIN_OVERGROUND_STRIDES if environment == "overground" else 5):
                stride_s, how = float(np.median(strides)), f"markers, {strides.size} strides"
            elif fit is not None and np.isfinite(speeds[cond]):
                stride_s, how = fit(speeds[cond]), "treadmill speed-stride fit"
            else:
                skipped.append(Skipped(sid, cond, "-", "no stride time"))
                continue
            knt = _read_txt(zf, knt_name)
            for side, leg in SIDES.items():
                curve, why = hip_curve(knt, side)
                if curve is None:
                    skipped.append(Skipped(sid, cond, leg, why))
                    continue
                key = np.round(curve, 6).tobytes()
                if key in seen and not seen[key].startswith(sid + ":"):
                    skipped.append(Skipped(sid, cond, leg, f"identical to {seen[key]}"))
                    continue
                seen[key] = f"{sid}:{cond}:{leg}"
                if why:
                    notes.append(f"{sid} {cond} {leg}: {why}")
                trials.append(trial_from_mean_cycle(
                    sid, leg, float(meta["Mass"]), speeds[cond], curve, stride_s,
                    n_cycles=n_cycles, dataset=DATASET, condition=cond,
                    environment=environment, incline_deg=0.0, nominal_speed=nominal,
                    age_group=str(meta["AgeGroup"]).lower(), age_years=float(meta["Age"]),
                    sex=str(meta["Gender"]), stride_source=how))
    return LoadResult(trials, skipped, notes)
