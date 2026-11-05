"""PsyPref: tune an assistance profile from A/B comparisons, with Bayesian adaptive methods from psychophysics.

    from psypref import Prior, PsyPref

    tuner = PsyPref(Prior.from_walking(peak_range=(3, 12)))
    for _ in range(24):
        a, b = tuner.ask
        tuner.tell(show_to_user(a, b))      # "A", "B" or None
    curve = tuner.identified
"""

from .prior import Prior
from .simulate import SimulatedUser, choice_probability, demo_profile
from .tuner import PHASE, Family, PsyPref

__all__ = ["PHASE", "Family", "Prior", "PsyPref", "SimulatedUser", "choice_probability", "demo_profile"]
__version__ = "0.1.0"
