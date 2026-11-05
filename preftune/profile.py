"""The eWalk parameterised assistive hip torque profile.

Geometry follows Manzoori et al. (2024), *Front. Bioeng. Biotechnol.* 12:1324587,
section 2.1 -- the controller Ramella et al. (2025) tune against:

    each bout has "a semi-trapezoidal profile defined by four parameters: peak
    amplitude, peak time (timing of the middle of the peak), total duration and
    rise time. The fall time ... was held constant at 5 %GC."

so, writing the plateau width as ``T_p = T_tot - T_r - T_f``:

    rise     [t_peak - T_p/2 - T_r,  t_peak - T_p/2]      0 -> amplitude
    plateau  [t_peak - T_p/2,        t_peak + T_p/2]      amplitude
    fall     [t_peak + T_p/2,        t_peak + T_p/2 + T_f]  amplitude -> 0

``t_peak`` is the *centre* of the plateau, not its start. The axis is circular:
under Ramella's feature ranges the extension bout always begins in the previous
gait cycle, which is why their subjects could prefer assistance starting before
heel strike.

Sign convention matches the dataset contract: hip flexion positive, so the
extension bout enters negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Ramella et al. (2025), Table I -------------------------------------------------
FEATURE_NAMES = (
    "ext_peak_torque",
    "ext_peak_time",
    "ext_rise_time",
    "flex_peak_torque",
    "flex_peak_time",
    "flex_rise_time",
)
FEATURE_UNITS = ("Nm", "%GC", "%GC", "Nm", "%GC", "%GC")
FEATURE_LOW = np.array([5.0, 10.0, 10.0, 5.0, 55.0, 10.0])
FEATURE_HIGH = np.array([8.0, 20.0, 20.0, 8.0, 65.0, 20.0])
FEATURE_STEP = 0.1  # "the features were discretized with decimal precision"

# Familiarisation profile, Ramella section II-C (grey line in their Fig. 3).
FAMILIARISATION = np.array([7.0, 10.0, 15.0, 7.0, 60.0, 15.0])

# Validation perturbation, Ramella section II-C: +/-2.0 Nm and +/-7.0 %GC.
PERTURBATION = np.array([2.0, 7.0, 7.0, 2.0, 7.0, 7.0])

# --- geometry constants from Manzoori et al. (2024) ---------------------------------
FALL_TIME = 5.0  # %GC, held constant
TOTAL_DURATION = 40.0  # %GC; Ramella does not restate it, see PLAN.md section 9.1
SMOOTH_HZ = 20.0  # first-order IIR low-pass on the torque command
TRANSITION = "smoothstep"  # shape of the rise and fall; see _transition

N_PHASE = 101
PHASE_GRID = np.linspace(0.0, 100.0, N_PHASE)


def _transition(x: np.ndarray, kind: str = "smoothstep") -> np.ndarray:
    """Shape of a rise from 0 to 1 over ``x`` in [0, 1].

    Ramella describe "smooth trapezoidal interpolation curves", and Manzoori's
    Fig. 3B draws both transitions as S-curves that ease out of zero and into the
    plateau rather than as straight ramps. ``smoothstep`` reproduces that: it is
    C1-continuous, with zero slope at both ends.
    """
    if kind == "linear":
        return x
    if kind == "smoothstep":
        return x * x * (3.0 - 2.0 * x)
    if kind == "cosine":
        return 0.5 * (1.0 - np.cos(np.pi * x))
    raise ValueError(f"unknown transition {kind!r}")


def _bout(phase: np.ndarray, amplitude: float, peak_time: float, rise_time: float,
          total_duration: float = TOTAL_DURATION, fall_time: float = FALL_TIME,
          transition: str = TRANSITION) -> np.ndarray:
    """One semi-trapezoidal bout, non-negative, on a circular %GC axis."""
    plateau = total_duration - rise_time - fall_time
    if plateau < 0.0:
        raise ValueError(
            f"rise_time={rise_time} + fall_time={fall_time} exceeds "
            f"total_duration={total_duration}; plateau would be negative"
        )
    # Position within the bout, measured from the start of the rise, wrapped to [0, 100).
    start = peak_time - plateau / 2.0 - rise_time
    u = np.mod(np.asarray(phase, dtype=float) - start, 100.0)

    y = np.zeros_like(u)
    if rise_time > 0:
        m = u < rise_time
        y[m] = amplitude * _transition(u[m] / rise_time, transition)
    # With rise_time == 0 there is no rise segment and the plateau simply starts at
    # u == 0, which the next branch already handles. Unreachable under Table I.
    m = (u >= rise_time) & (u < rise_time + plateau)
    y[m] = amplitude
    m = (u >= rise_time + plateau) & (u < rise_time + plateau + fall_time)
    y[m] = amplitude * (1.0 - _transition((u[m] - rise_time - plateau) / fall_time, transition))
    return y


OVERSAMPLE = 20  # samples per %GC used internally when smoothing


def _smooth_periodic(y: np.ndarray, stride_s: float, cutoff_hz: float) -> np.ndarray:
    """Steady-state response of a first-order low-pass to a periodic input.

    The device applies a causal IIR filter in real time. On a signal that repeats
    every ``stride_s`` seconds the steady state is obtained by scaling each harmonic
    by the filter's complex frequency response, which keeps the phase lag that a
    zero-phase filter would discard.

    ``y`` must be a periodic sequence without a duplicated endpoint.
    """
    core = np.asarray(y, dtype=float)
    n = core.size
    freqs = np.fft.rfftfreq(n, d=stride_s / n)  # Hz
    response = 1.0 / (1.0 + 1j * freqs / cutoff_hz)
    return np.fft.irfft(np.fft.rfft(core) * response, n=n)


@dataclass(frozen=True)
class TorqueProfile:
    """A candidate assistive profile, parameterised by Ramella's six features."""

    features: np.ndarray
    total_duration: float = TOTAL_DURATION
    fall_time: float = FALL_TIME
    transition: str = TRANSITION

    def __post_init__(self) -> None:
        f = np.asarray(self.features, dtype=float)
        if f.shape != (6,):
            raise ValueError(f"expected 6 features, got shape {f.shape}")
        object.__setattr__(self, "features", f)

    @classmethod
    def from_dict(cls, **kwargs: float) -> "TorqueProfile":
        return cls(np.array([kwargs[name] for name in FEATURE_NAMES], dtype=float))

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.features))

    def in_bounds(self) -> bool:
        return bool(np.all(self.features >= FEATURE_LOW) and np.all(self.features <= FEATURE_HIGH))

    def torque(self, phase: np.ndarray | None = None, *, stride_s: float | None = None,
               smooth_hz: float = SMOOTH_HZ) -> np.ndarray:
        """Commanded torque in Nm, flexion positive.

        With ``stride_s`` given, the device's first-order ``smooth_hz`` low-pass is
        applied; without it the raw trapezoid is returned.
        """
        phase = PHASE_GRID if phase is None else np.asarray(phase, dtype=float)
        if stride_s is None or not smooth_hz:
            return self._raw(phase)

        # Filter on an oversampled circular grid, then sample the result at the
        # requested phases. Filtering the 101-point curve directly would ring: a
        # trapezoid's corners carry energy well past that grid's 48 Hz Nyquist.
        n_fine = 100 * OVERSAMPLE
        fine = np.linspace(0.0, 100.0, n_fine, endpoint=False)
        smoothed = _smooth_periodic(self._raw(fine), stride_s, smooth_hz)
        return np.interp(np.mod(phase, 100.0), fine, smoothed, period=100.0)

    def _raw(self, phase: np.ndarray) -> np.ndarray:
        f1, f2, f3, f4, f5, f6 = self.features
        extension = _bout(phase, f1, f2, f3, self.total_duration, self.fall_time,
                          self.transition)
        flexion = _bout(phase, f4, f5, f6, self.total_duration, self.fall_time,
                        self.transition)
        return flexion - extension


def sample_batch(n: int = 40, rng: np.random.Generator | None = None) -> np.ndarray:
    """Ramella's batch: ``n`` profiles sampled uniformly, to decimal precision.

    Returns an ``(n, 6)`` array of feature vectors.
    """
    rng = np.random.default_rng() if rng is None else rng
    raw = rng.uniform(FEATURE_LOW, FEATURE_HIGH, size=(n, 6))
    steps = np.round(raw / FEATURE_STEP) * FEATURE_STEP
    return np.round(np.clip(steps, FEATURE_LOW, FEATURE_HIGH), 1)


def torque_matrix(batch: np.ndarray, *, stride_s: float | None = None,
                  smooth_hz: float = SMOOTH_HZ, **profile_kw) -> np.ndarray:
    """Torque curves for a batch of feature vectors: ``(n, N_PHASE)`` in Nm.

    ``profile_kw`` forwards the geometry that the six features do not carry
    (``total_duration``, ``fall_time``, ``transition``) to every profile, so a caller
    sweeping plateau width gets curves that actually differ.
    """
    return np.array([
        TorqueProfile(f, **profile_kw).torque(stride_s=stride_s, smooth_hz=smooth_hz)
        for f in np.atleast_2d(batch)
    ])
