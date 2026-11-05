"""Laying external gait data onto the generator's native clock.

``prompref.population.generator`` scales the biological moment, low-pass filters it and
shifts it in time on the native 200 Hz clock, then cuts it into heel-strike cycles by the
wraps of ``gait_phase``. A converted trial therefore needs the same three things a
Molinaro ``Trial`` has: a continuous moment series on that clock, a gait phase in [0, 1)
that wraps at every heel strike, and the cycles between those wraps.

Two sources are supported.

* **Per-stride curves with their measured durations** (``trial_from_strides``). Each
  stride, heel strike to heel strike, is laid out over its own duration and consecutive
  strides are concatenated, so the phase wraps exactly at every recorded heel strike. The
  strides must be contiguous in the recording (the end of one is the start of the next).
* **An ensemble-mean cycle with a measured stride time** (``trial_from_mean_cycle``),
  for datasets that publish only per-condition means. The mean cycle is tiled. Such a
  trial has no stride-to-stride variability; ``ExternalTrial.curve_source`` records which
  kind a trial is, so a prior fitted per stride can tell them apart.

As in ``preftune.data.loader.load_trial``, the first and the last stride of the series are
not cycles: the phase only wraps *into* a stride, so the first stride has no opening wrap
and the last has no closing one. They serve as lead-in for the causal filter and as room
for the generator's time shift. All moments are in Nm/kg, hip flexion positive.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from preftune.data.loader import NATIVE_RATE_HZ, Trial, cycle_average, find_cycles

#: Where the raw downloads live (outside every repository).
DEFAULT_DATA_ROOT = Path("data/external_gait_data")

#: A trial needs at least this many cycles, as ``load_trial`` demands of a Molinaro leg.
MIN_CYCLES = 5

#: Tiles of an ensemble-mean cycle: 12 usable cycles plus lead-in and lead-out. Enough
#: for every shift the generator draws (stress users reach +350 ms, i.e. under 0.5 cycle).
DEFAULT_TILED_CYCLES = 12


def data_root(root: Path | str | None = None) -> Path:
    """``root``, else ``$EXTERNAL_GAIT_DATA``, else the default download folder."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get("EXTERNAL_GAIT_DATA", DEFAULT_DATA_ROOT))


@dataclass(frozen=True)
class ExternalTrial(Trial):
    """A ``Trial`` from another dataset, with the recording's condition attached.

    ``biological_nmkg`` is the recorded net hip moment (no exoskeleton was worn), so
    ``total_nmkg`` and ``tcn_nmkg`` are the same series and ``exo_measured_nm`` is zero.
    ``speed_mps`` is the measured walking speed of this subject in this condition;
    ``nominal_speed`` is the dataset's own label for the speed condition.
    """

    dataset: str = ""
    condition: str = ""
    environment: str = ""  # "treadmill" or "overground"
    incline_deg: float = 0.0
    nominal_speed: str = ""
    age_group: str = ""
    age_years: float = float("nan")
    sex: str = ""
    curve_source: str = ""  # "strides" or "ensemble-mean"
    n_source_strides: int = 0
    stride_source: str = ""  # how the stride time was obtained
    measured_speed_mps: float = float("nan")  # speed measured from the data, if done


@dataclass
class Skipped:
    """A subject-condition-side that could not become a trial, and why."""

    subject: str
    condition: str
    side: str
    reason: str


@dataclass
class LoadResult:
    """What a converter produced: the trials, what it skipped, and data notes worth reporting."""

    trials: list
    skipped: list
    notes: list


def lay_out_strides(curves, durations_s, fs: float = NATIVE_RATE_HZ):
    """Concatenate heel-strike-to-heel-strike curves on a uniform clock.

    ``curves`` is ``(n_strides, n_points)``, each row sampled uniformly over its stride
    *including both heel strikes*; ``durations_s`` are the strides' durations. Returns
    ``(time_s, gait_phase, moment)``: samples at ``k / fs`` for every ``k / fs`` before
    the end of the last stride, the phase within the stride a sample falls in, and the
    stride's curve interpolated at that phase.
    """
    curves = np.atleast_2d(np.asarray(curves, dtype=float))
    durations = np.asarray(durations_s, dtype=float).ravel()
    if curves.shape[0] != durations.size:
        raise ValueError(f"{curves.shape[0]} curves but {durations.size} durations")
    if curves.shape[1] < 2:
        raise ValueError("each stride curve needs at least two points")
    if np.any(~np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("stride durations must be positive and finite")
    if not np.all(np.isfinite(curves)):
        raise ValueError("stride curves contain non-finite values")
    edges = np.concatenate([[0.0], np.cumsum(durations)])
    n = int(np.ceil(edges[-1] * fs - 1e-9))
    time_s = np.arange(n) / fs
    stride = np.clip(np.searchsorted(edges, time_s, side="right") - 1, 0, durations.size - 1)
    phase = np.clip((time_s - edges[stride]) / durations[stride], 0.0, np.nextafter(1.0, 0.0))
    grid = np.linspace(0.0, 1.0, curves.shape[1])
    moment = np.empty(n)
    for i in range(durations.size):
        sel = stride == i
        moment[sel] = np.interp(phase[sel], grid, curves[i])
    return time_s, phase, moment


def trial_from_strides(subject: str, side: str, mass_kg: float, speed_mps: float, curves,
                       durations_s, *, curve_source: str = "strides",
                       min_cycles: int = MIN_CYCLES, **condition) -> ExternalTrial:
    """An ``ExternalTrial`` from contiguous strides (see the module docstring)."""
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError(f"{subject}_{side}: body mass must be positive, got {mass_kg}")
    time_s, phase, moment = lay_out_strides(curves, durations_s)
    cycles = find_cycles(phase)
    if len(cycles) < min_cycles:
        raise ValueError(f"{subject}_{side}: only {len(cycles)} complete cycles "
                         f"(need {min_cycles})")
    condition.setdefault("n_source_strides", int(np.atleast_2d(curves).shape[0]))
    return ExternalTrial(
        subject=subject, side=side, speed_mps=float(speed_mps), mass_kg=float(mass_kg),
        time_s=time_s, gait_phase=phase, biological_nmkg=moment, total_nmkg=moment.copy(),
        tcn_nmkg=moment.copy(), exo_measured_nm=np.zeros_like(moment), cycles=cycles,
        curve_source=curve_source, **condition)


#: Share of the cycle next to each heel strike over which an end mismatch is spread.
CLOSE_BLEND = 0.10


def close_cycle(curve, blend: float = CLOSE_BLEND) -> tuple[np.ndarray, float]:
    """Make a mean cycle periodic, changing it only near the heel strike.

    The two ends of an ensemble mean are adjacent heel strikes, so they differ; tiled as
    is, the series would jump at every heel strike. Both ends are moved to their average,
    and the correction fades out linearly over ``blend`` of the cycle on each side, so the
    rest of the curve (the flexion peak in particular) is untouched. Returns the corrected
    curve and the end mismatch ``curve[-1] - curve[0]`` (same units as ``curve``).
    """
    c = np.asarray(curve, dtype=float)
    gap = float(c[-1] - c[0])
    x = np.linspace(0.0, 1.0, c.size)
    rise_start = np.clip(1.0 - x / blend, 0.0, 1.0)  # 1 at the first point, 0 after blend
    rise_end = np.clip((x - (1.0 - blend)) / blend, 0.0, 1.0)  # 0 before 1 - blend, 1 at end
    return c + 0.5 * gap * rise_start - 0.5 * gap * rise_end, gap


def trial_from_mean_cycle(subject: str, side: str, mass_kg: float, speed_mps: float,
                          curve, stride_s: float, *, n_cycles: int = DEFAULT_TILED_CYCLES,
                          **condition) -> ExternalTrial:
    """An ``ExternalTrial`` that repeats one ensemble-mean cycle at a measured stride time.

    The trial has ``n_cycles`` cycles: ``n_cycles + 2`` tiles minus lead-in and lead-out.
    ``n_source_strides`` should say how many recorded strides the mean came from, when the
    dataset reports it; it defaults to 0 (unknown).
    """
    closed, _ = close_cycle(curve)
    tiles = np.tile(closed, (n_cycles + 2, 1))
    condition.setdefault("n_source_strides", 0)
    return trial_from_strides(subject, side, mass_kg, speed_mps, tiles,
                              np.full(n_cycles + 2, float(stride_s)),
                              curve_source="ensemble-mean", **condition)


def mean_cycle(trial: Trial, n: int = 101) -> np.ndarray:
    """The trial's biological moment averaged over its cycles, Nm/kg on an ``n``-point grid."""
    return cycle_average(trial.biological_nmkg, trial, n=n)
