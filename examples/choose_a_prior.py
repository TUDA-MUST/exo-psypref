"""What a prior costs, on simulated users: run this before anyone walks, with your own prior and your own assumptions."""

import numpy as np

from psypref import Prior, PsyPref, SimulatedUser, demo_profile

SUBJECTS = ("AB30", "AB31", "AB32", "AB33", "AB34")
PEAKS = ((8.0, 6.0), (5.0, 10.0), (10.0, 4.0))


def score(make_prior, n_comparisons=24):
    """How often simulated users would choose the identified profile over generic assistance."""
    out = []
    for i, subject in enumerate(SUBJECTS):
        for j, peaks in enumerate(PEAKS):
            prior = make_prior(subject)
            user = SimulatedUser(demo_profile(subject, peaks), seed=100 * i + j)
            tuner = PsyPref(prior, seed=100 * i + j)
            for _ in range(n_comparisons):
                tuner.tell(user.answer(*tuner.ask()))
            out.append(user.would_choose(tuner.identified(), prior.generic))
    return 100 * float(np.mean(out))


for label, width in (("walking data", 1.0), ("1.5 times wider", 1.5), ("2 times wider", 2.0), ("3 times wider", 3.0)):
    p = score(lambda s, w=width: Prior.from_walking(peak_range=(3, 12), leave_out=s, width=w))
    print(f"{label:16s} {p:5.1f} %  (15 sessions only: a rough guide, the paper's numbers come from 200 users)")
