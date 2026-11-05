"""A tuning session with a simulated user. On a real device, replace ``user.answer`` by asking the person."""

from psypref import Prior, PsyPref, SimulatedUser, demo_profile

# The simulated user comes from subject AB30, so the prior leaves that subject out.
preferred = demo_profile("AB30", peaks=(8.0, 6.0))
prior = Prior.from_walking(peak_range=(3, 12), leave_out="AB30")
user = SimulatedUser(preferred, noise=1.0, lapse=0.1, seed=1)

tuner = PsyPref(prior, seed=1)
print("comparisons | distance to the preferred profile (Nm) | chosen over generic assistance")
for k in range(1, 25):
    a, b = tuner.ask()
    tuner.tell(user.answer(a, b))
    if k in (1, 3, 8, 12, 24):
        curve = tuner.identified()
        print(f"{k:11d} | {user.distance(curve):38.2f} | {100 * user.would_choose(curve, prior.generic):.0f} %")

print("\nThe distance is known only in a simulation. With a real user, compare the result with generic assistance:")
print("identified peaks:", round(-tuner.identified().min(), 1), "Nm extension,", round(tuner.identified().max(), 1), "Nm flexion")
