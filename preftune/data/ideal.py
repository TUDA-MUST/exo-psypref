"""The simulated user's ideal torque profile.

Molinaro et al. (2024) transform a hip moment into assistance by scaling it, delaying
it, and low-pass filtering it. That transform was verified against this archive: the
recorded command matches ``0.20 * mass * delay_100ms(LPF_10Hz(tcn))`` at R^2 >= 0.989
for every subject and side, with the delay optimum landing on exactly 100 ms.

The lag that matters here is relative to *biology*, not to the TCN output. The network
was trained to estimate moments already delayed by 50 ms, so the delivered assistance
lagged biology by about 150 ms. ``delay_ms`` below therefore means lag versus biology,
and its default of 100 ms is the value consistent with Ramella's subjects, who chose
extension timing earlier than the device actually delivered (PLAN.md section 1.3).

Order matters: scale, filter and delay are applied on the 200 Hz series, and only then
is the result averaged over gait cycles. Doing it the other way round would destroy the
millisecond semantics that the whole study turns on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loader import (
    MOLINARO_LPF_HZ,
    MOLINARO_SCALE,
    NATIVE_RATE_HZ,
    Trial,
    causal_lowpass,
    cycle_average,
    delay,
)

#: Lag versus biology consistent with Ramella's observed feature preferences.
DEFAULT_DELAY_MS = 100.0

MOMENT_SOURCES = ("biological", "total", "tcn")


@dataclass(frozen=True)
class IdealProfile:
    """The torque curve the simulated user would most prefer, in Nm, flexion positive."""

    torque_nm: np.ndarray  # on the 101-point %GC grid
    trial: Trial
    assist_scale: float
    delay_ms: float
    lpf_hz: float
    moment_source: str

    @property
    def stride_s(self) -> float:
        return self.trial.stride_s

    def peaks(self) -> dict[str, float]:
        """Extension and flexion peak magnitude (Nm) and location (%GC)."""
        phase = np.linspace(0.0, 100.0, self.torque_nm.size)
        lo, hi = int(np.argmin(self.torque_nm)), int(np.argmax(self.torque_nm))
        return {
            "ext_peak_nm": float(abs(self.torque_nm[lo])),
            "ext_peak_pct_gc": float(phase[lo]),
            "flex_peak_nm": float(self.torque_nm[hi]),
            "flex_peak_pct_gc": float(phase[hi]),
        }


def build_ideal(trial: Trial, *, assist_scale: float = MOLINARO_SCALE,
                delay_ms: float = DEFAULT_DELAY_MS, lpf_hz: float = MOLINARO_LPF_HZ,
                moment_source: str = "biological", n_phase: int = 101) -> IdealProfile:
    """Apply the Molinaro transform on the native clock, then cycle-average."""
    if moment_source not in MOMENT_SOURCES:
        raise ValueError(f"moment_source must be one of {MOMENT_SOURCES}, got {moment_source!r}")
    source = {
        "biological": trial.biological_nmkg,
        "total": trial.total_nmkg,
        "tcn": trial.tcn_nmkg,
    }[moment_source]

    scaled_nm = assist_scale * trial.mass_kg * np.asarray(source, dtype=float)
    filtered = causal_lowpass(scaled_nm, cutoff_hz=lpf_hz, fs=NATIVE_RATE_HZ) if lpf_hz else scaled_nm
    shifted = delay(filtered, delay_ms, fs=NATIVE_RATE_HZ)

    return IdealProfile(
        torque_nm=cycle_average(shifted, trial, n=n_phase),
        trial=trial,
        assist_scale=assist_scale,
        delay_ms=delay_ms,
        lpf_hz=lpf_hz,
        moment_source=moment_source,
    )
