"""A simulated user's preferred torque curve.

The pipeline follows ``preftune.data.ideal.build_ideal`` step by step, on the native
200 Hz clock: scale the biological moment (separately for extension and flexion), apply
the causal 10 Hz low-pass, shift by the user's personal lag, cut into gait cycles, and
only then move the landmarks and clip to the device. With every factor at its centre the
result *is* ``build_ideal(trial, delay_ms=100)``, clipped, whenever no cycle is lost to
the shift or the filter's settling (true of the synthetic trial; a real trial may lose one).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from preftune.data.loader import MOLINARO_LPF_HZ, NATIVE_RATE_HZ, Trial, causal_lowpass

from .curves import physics_clip, shift_signal, stride_curves, warp_landmarks
from .factors import FactorDraw, FactorSwitches, ShiftSpread, TimingSpread, sample_draw

#: The causal filter's settling time on the 200 Hz clock; earlier samples are not used.
SETTLE_SAMPLES = 40


class OrderViolation(ValueError):
    """A draw whose landmark offsets cannot be applied to this curve."""


def assistance_series(trial: Trial, draw: FactorDraw) -> np.ndarray:
    """Scaled, filtered and shifted assistance on the native clock."""
    bio_nm = trial.mass_kg * np.asarray(trial.biological_nmkg, dtype=float)
    scaled = np.where(bio_nm < 0.0, draw.ext_scale * bio_nm, draw.flex_scale * bio_nm)
    filtered = causal_lowpass(scaled, cutoff_hz=MOLINARO_LPF_HZ, fs=NATIVE_RATE_HZ)
    return shift_signal(filtered, draw.shift_ms, NATIVE_RATE_HZ)


def clear_cycles(trial: Trial, shift_ms: float) -> list[tuple[int, int]]:
    """Cycles untouched by the shift's edge padding and by the filter's settling."""
    n = int(round(shift_ms * 1e-3 * NATIVE_RATE_HZ))
    first = SETTLE_SAMPLES + max(n, 0)
    last = len(trial.time_s) - max(-n, 0)
    return [(a, b) for a, b in trial.cycles if a >= first and b <= last]


def _finish(curve: np.ndarray, draw: FactorDraw) -> np.ndarray:
    offsets = np.asarray(draw.landmark_offsets, dtype=float)
    if np.any(offsets != 0.0):
        try:
            curve = warp_landmarks(curve, offsets)
        except ValueError as exc:
            raise OrderViolation(str(exc)) from exc
    return physics_clip(curve)


def user_curve(trial: Trial, draw: FactorDraw) -> np.ndarray:
    """The user's preferred curve: mean over clear cycles, landmark warp, device clip."""
    cycles = clear_cycles(trial, draw.shift_ms)
    if not cycles:
        raise ValueError(f"{trial.subject}_{trial.side}: no cycle clear of a {draw.shift_ms} ms shift")
    series = assistance_series(trial, draw)
    return _finish(stride_curves(series, trial.gait_phase, cycles).mean(axis=0), draw)


#: A leg stuck this many consecutive rejections without an accepted curve is a stalled
#: timing spread for that leg, not bad luck.
MAX_CONSECUTIVE_REJECTIONS = 200


@dataclass
class TrainingSet:
    curves: np.ndarray
    draws: list[FactorDraw]
    sources: list[tuple[str, str, int]]
    n_rejected: int
    n_rejected_by_source: dict[str, int] = field(default_factory=dict)


def generate_training_curves(trials: list[Trial], n: int, rng: np.random.Generator,
                             switches: FactorSwitches = FactorSwitches(),
                             timing: TimingSpread | None = None,
                             shift: ShiftSpread | None = None) -> TrainingSet:
    """``n`` per-stride curves for prior fitting.

    The output leg (subject-side) is picked uniformly once per curve -- so a short trial
    (AB32 has 6-7 valid cycles) is not under-weighted, and so that a leg whose landmark
    geometry rejects more often is not under-represented in the accepted population
    either. Factors and stride are then redrawn *within that leg* until one curve is
    accepted; a leg stuck past ``MAX_CONSECUTIVE_REJECTIONS`` straight rejections raises
    rather than looping forever on a timing spread that leg's curves cannot take.
    """
    if not trials:
        raise ValueError("no trials to generate from")
    timing = timing or TimingSpread.published()
    curves, draws, sources = [], [], []
    n_rejected = 0
    n_rejected_by_source: dict[str, int] = {}
    for _ in range(n):
        trial = trials[int(rng.integers(len(trials)))]
        key = f"{trial.subject}_{trial.side}"
        attempts = 0
        while True:
            draw = sample_draw(rng, switches, timing, shift)
            cycles = clear_cycles(trial, draw.shift_ms)
            if cycles:
                stride = int(rng.integers(len(cycles)))
                raw = stride_curves(assistance_series(trial, draw), trial.gait_phase,
                                    [cycles[stride]])[0]
                try:
                    curve = _finish(raw, draw)
                    curves.append(curve)
                    draws.append(draw)
                    sources.append((trial.subject, trial.side, trial.cycles.index(cycles[stride])))
                    break
                except OrderViolation:
                    pass
            attempts += 1
            n_rejected += 1
            n_rejected_by_source[key] = n_rejected_by_source.get(key, 0) + 1
            if attempts > MAX_CONSECUTIVE_REJECTIONS:
                raise RuntimeError(
                    f"{trial.subject}_{trial.side}: {attempts} consecutive draws rejected "
                    "without an accepted curve")
    return TrainingSet(np.array(curves), draws, sources, n_rejected, n_rejected_by_source)
