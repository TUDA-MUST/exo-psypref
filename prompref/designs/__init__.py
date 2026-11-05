"""Profile families. Every one maps parameters to a torque curve on the %GC grid."""

from .base import BoxDesign, Design
from .bump import BumpDesign  # noqa: F401
from .fourier import FourierDesign  # noqa: F401
from .promp import PrompS0, PrompS1, PrompS2  # noqa: F401
from .schaefer import PREFERRED_MEAN, SchaeferDesign  # noqa: F401
from .trapezoid import TrapezoidDesign

__all__ = [
    "Design",
    "BoxDesign",
    "TrapezoidDesign",
    "SchaeferDesign",
    "BumpDesign",
    "FourierDesign",
    "PrompS0",
    "PrompS1",
    "PrompS2",
]
