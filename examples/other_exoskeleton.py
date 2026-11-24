"""PsyPref on another device, here a made-up ankle exoskeleton that assists plantarflexion only.

The curves below are synthetic and only show the calls. For a real device, use measured joint moments or curves from
earlier studies, and check the setup with simulated users before anyone walks. The paper tested the hip only.
"""

import numpy as np

from psypref import PHASE, Prior, PsyPref, SimulatedUser

rng = np.random.default_rng(0)


def plantarflexion(peak_time, width, height):
    return height * np.exp(-0.5 * ((PHASE - peak_time) / width) ** 2)


curves = np.array([plantarflexion(rng.normal(50, 2), rng.normal(7, 1), rng.uniform(15, 30)) for _ in range(200)])

# One-sided assistance: no peak range, so the curves count as they are. The device's torque limit goes in `limit`.
prior = Prior.from_curves(curves, limit=40.0)
print(prior)

preferred = plantarflexion(54, 6, 24)   # an assumption: only a simulation knows it
user = SimulatedUser(preferred, noise=2.0, seed=4)
session = PsyPref(prior, seed=4)
for k in range(24):
    a, b = session.ask()
    session.tell(user.answer(a, b))
print(f"distance to the assumed preferred profile: {user.distance(prior.generic):.1f} Nm before, "
      f"{user.distance(session.identified()):.1f} Nm after {session.n_comparisons} comparisons")
print(f"identified peak: {session.identified().max():.1f} Nm at {PHASE[session.identified().argmax()]:.0f} % of the gait cycle")
