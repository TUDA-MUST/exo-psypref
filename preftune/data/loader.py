"""Loading Molinaro et al. (2024) time-domain trials and recovering subject mass.

The archive stores joint moments in Nm/kg but records no subject mass anywhere, and
Ramella's feature space is in Nm. Mass is recovered from the exoskeleton itself: the
recorded command is the TCN moment estimate scaled by 20 %, delayed 100 ms and
low-pass filtered at 10 Hz, so regressing command against that transform gives a gain
of ``assist_scale * mass``. See :func:`recover_mass` and PLAN.md section 1.2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

DEFAULT_ROOT = Path(
    os.environ.get(
        "PREFTUNE_DATA",
        "data/molinaro/time_domain",
    )
)

NATIVE_RATE_HZ = 200.0

# Molinaro's deployed hip controller (paper, Methods).
MOLINARO_SCALE = 0.20
MOLINARO_DELAY_MS = 100.0  # relative to the TCN output
MOLINARO_LPF_HZ = 10.0
MOLINARO_LPF_ORDER = 2
TCN_TRAINING_DELAY_MS = 50.0  # the TCN estimates moments already delayed by this much

#: Lag of the delivered assistance relative to biology: 50 ms (TCN) + 100 ms (controller).
EFFECTIVE_LAG_MS = TCN_TRAINING_DELAY_MS + MOLINARO_DELAY_MS

WALK_SPEED_MPS = 1.1  # Ramella's treadmill speed


def subject_path(subject: str, side: str, root: Path | None = None) -> Path:
    root = DEFAULT_ROOT if root is None else Path(root)
    return root / "molinaro" / "subjects" / f"{subject}_{side}.parquet"


def load_walking(subject: str, side: str = "r", speed_mps: float | None = WALK_SPEED_MPS,
                 root: Path | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Treadmill walking rows for one subject and limb, optionally at one speed."""
    df = pd.read_parquet(subject_path(subject, side, root), columns=columns)
    df = df[df["activity"] == "walk_treadmill"]
    if speed_mps is not None:
        df = df[np.isclose(df["speed_mps"], speed_mps)]
    if df.empty:
        raise ValueError(f"no walk_treadmill rows for {subject}_{side} at speed {speed_mps}")
    return df.reset_index(drop=True)


def causal_lowpass(x: np.ndarray, cutoff_hz: float = MOLINARO_LPF_HZ,
                   order: int = MOLINARO_LPF_ORDER, fs: float = NATIVE_RATE_HZ) -> np.ndarray:
    """Causal Butterworth low-pass. Deliberately not ``filtfilt``: the lag is the point."""
    b, a = butter(order, cutoff_hz / (fs * 0.5), "low")
    return lfilter(b, a, np.asarray(x, dtype=float))


def delay(x: np.ndarray, delay_ms: float, fs: float = NATIVE_RATE_HZ) -> np.ndarray:
    """Shift a signal later in time by whole samples, holding the first value."""
    x = np.asarray(x, dtype=float)
    n = int(round(delay_ms * 1e-3 * fs))
    if n <= 0:
        return x.copy()
    return np.concatenate([np.full(n, x[0]), x[:-n]])


def recover_mass(subject: str, side: str = "r", root: Path | None = None,
                 assist_scale: float = MOLINARO_SCALE) -> tuple[float, float]:
    """Recover subject mass in kg by identifying the exoskeleton controller.

    Returns ``(mass_kg, r_squared)``. An R^2 well below ~0.98 means the assumed
    controller does not describe this subject's data and the mass is not trustworthy.
    """
    df = load_walking(
        subject, side, speed_mps=None, root=root,
        columns=["activity", "speed_mps", "tcn_hip_moment_estimate_nmkg", "exo_torque_command_nm"],
    )
    tcn = df["tcn_hip_moment_estimate_nmkg"].to_numpy(float)
    cmd = df["exo_torque_command_nm"].to_numpy(float)
    ok = np.isfinite(tcn) & np.isfinite(cmd)
    tcn, cmd = tcn[ok], cmd[ok]

    y = delay(causal_lowpass(tcn), MOLINARO_DELAY_MS)
    gain = float(y @ cmd / (y @ y))
    resid = cmd - gain * y
    r2 = float(1.0 - resid @ resid / ((cmd - cmd.mean()) @ (cmd - cmd.mean())))
    return gain / assist_scale, r2


@dataclass(frozen=True)
class Trial:
    """One contiguous, fully-valid stretch of a walking trial, on the native clock."""

    subject: str
    side: str
    speed_mps: float
    mass_kg: float
    time_s: np.ndarray
    gait_phase: np.ndarray  # in [0, 1)
    biological_nmkg: np.ndarray  # total moment minus measured exoskeleton torque
    total_nmkg: np.ndarray
    tcn_nmkg: np.ndarray
    exo_measured_nm: np.ndarray
    cycles: list[tuple[int, int]]

    @property
    def stride_s(self) -> float:
        return float(np.median([self.time_s[b] - self.time_s[a] for a, b in self.cycles]))

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)


def _longest_valid_run(mask: np.ndarray) -> tuple[int, int]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if starts.size == 0:
        raise ValueError("no valid samples")
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


def find_cycles(gait_phase: np.ndarray) -> list[tuple[int, int]]:
    """Heel-strike-to-heel-strike index pairs, detected by wraps in ``gait_phase``."""
    wraps = np.flatnonzero(np.diff(gait_phase) < -0.5) + 1
    return [(int(wraps[i]), int(wraps[i + 1])) for i in range(len(wraps) - 1)]


def load_trial(subject: str, side: str = "r", speed_mps: float = WALK_SPEED_MPS,
               root: Path | None = None, mass_kg: float | None = None) -> Trial:
    """Load one walking trial, trimmed to its longest gap-free stretch.

    Missing inverse-dynamics samples are excluded, never interpolated, so that the
    continuous-time filtering in :mod:`preftune.data.ideal` is never applied across a
    fabricated gap.
    """
    if mass_kg is None:
        mass_kg, r2 = recover_mass(subject, side, root)
        if r2 < 0.95:
            raise ValueError(f"mass recovery for {subject}_{side} is unreliable (R2={r2:.3f})")

    df = load_walking(subject, side, speed_mps, root)
    if df["trial_uid"].nunique() != 1:
        raise ValueError(f"expected one trial, found {df['trial_uid'].nunique()}")

    total = df["hip_flexion_moment_nmkg"].to_numpy(float)
    exo = df["exo_torque_measured_nm"].to_numpy(float)
    phase = df["gait_phase_hs"].to_numpy(float)
    bio = total - exo / mass_kg

    a, b = _longest_valid_run(np.isfinite(bio) & np.isfinite(phase))
    sl = slice(a, b)
    phase = phase[sl]
    cycles = find_cycles(phase)
    if len(cycles) < 5:
        raise ValueError(f"only {len(cycles)} complete cycles in the valid run")

    return Trial(
        subject=subject, side=side, speed_mps=speed_mps, mass_kg=mass_kg,
        time_s=df["time_s"].to_numpy(float)[sl],
        gait_phase=phase,
        biological_nmkg=bio[sl],
        total_nmkg=total[sl],
        tcn_nmkg=df["tcn_hip_moment_estimate_nmkg"].to_numpy(float)[sl],
        exo_measured_nm=exo[sl],
        cycles=cycles,
    )


def cycle_average(signal: np.ndarray, trial: Trial, n: int = 101) -> np.ndarray:
    """Average a native-clock signal over gait cycles onto an ``n``-point %GC grid."""
    grid = np.linspace(0.0, 1.0, n)
    curves = [
        np.interp(grid, trial.gait_phase[a:b], np.asarray(signal, float)[a:b])
        for a, b in trial.cycles
    ]
    return np.mean(curves, axis=0)
