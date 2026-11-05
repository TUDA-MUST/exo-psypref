"""Show every pair in an interpretable family, and report the result in that family's parameters."""

import numpy as np

from psypref import PHASE, Family, Prior, PsyPref, SimulatedUser, demo_profile

preferred = demo_profile("AB31", peaks=(9.0, 7.0))
prior = Prior.from_walking(peak_range=(3, 12), leave_out="AB31")

for name in ("schaefer", "franks", "pchip", None):
    user = SimulatedUser(preferred, seed=2)
    tuner = PsyPref(prior, family=name, seed=2)
    for _ in range(24):
        tuner.tell(user.answer(*tuner.ask()))
    print(f"{str(name):9s} chosen over generic assistance: {100 * user.would_choose(tuner.identified(), prior.generic):3.0f} %, "
          f"parameters: {np.round(tuner.parameters(), 2)}")


# Your own family: a function from parameters to the 101-point curve, with bounds.
def two_bumps(theta):
    t_ext, t_flex, width, a_ext, a_flex = theta
    return (a_flex * np.exp(-0.5 * ((PHASE - t_flex) / width) ** 2)
            - a_ext * np.exp(-0.5 * ((PHASE - t_ext) / width) ** 2))


family = Family(two_bumps, low=[0, 40, 4, 3, 3], high=[40, 90, 25, 12, 12],
                names=("t_ext", "t_flex", "width", "a_ext", "a_flex"))
user = SimulatedUser(preferred, seed=3)
tuner = PsyPref(prior, family=family, seed=3)
for _ in range(24):
    tuner.tell(user.answer(*tuner.ask()))
print("own family:", dict(zip(family.param_names, np.round(tuner.parameters(), 1))))
