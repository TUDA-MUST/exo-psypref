# Using PsyPref on another exoskeleton

**We tested PsyPref only in simulation, and only for hip assistance during walking.** The library's interface works for other curves, but
none of the paper's numbers transfer to them. What follows is how we would go about it.

## What to change

| Your device | What to do |
|---|---|
| Another hip exoskeleton | Set `peak_range` or `level` to its limits. The walking prior may carry over |
| Another joint | Build the prior with `Prior.from_curves` from that joint's moments or from earlier studies' curves |
| Assistance in one direction only, such as plantarflexion | `Prior.from_curves(curves, limit=...)`, without a peak range |
| A device that can only produce certain shapes | A projection onto a [`Family`](projections.md) |
| Other walking speeds or tasks | A prior from those conditions. Preferences shift with walking speed and exposure |

[examples/other_exoskeleton.py](../examples/other_exoskeleton.py) shows the calls for an ankle exoskeleton, with made-up curves.

For the prior's curves, we took hip moments, low-pass filtered them at 10 Hz, and delayed them by about 138 ms. Torque
splines optimized for walking effort resemble biological moments, and users preferred peak assistance later than the biological peak.
Whether this holds for your joint and users is your assumption to check.

## Before anyone walks

1. **Run simulated users.** `SimulatedUser` answers by a curve distance with noise. Try several preferred profiles, also ones your prior does not expect, and a larger `noise`.
2. **Look at the curves.** Draw samples: run `ask()` a few times and check that you would be willing to apply them.
3. **Bound what can be applied.** PsyPref clips curves at `peak_range`'s top, `level` or `limit`. The library does not know your device's dynamics or safety limits, so keep your controller's own limits in place.

## What may break

The paper's robustness tests point to what matters most. PsyPref's lead survived other answer rules, changes over time and users from other datasets, but not users who judge single features instead of whole curves, or users with no single preferred profile. Nobody has measured these for assistance curves. In particular:

- **JNDs of assistance curves are largely unknown.** PsyPref learns its discrimination scale during the session, so it needs none, but how fast real users get there is open.
- **Patients.** We did not test them, and a prior from able-bodied walking cannot be assumed to fit them.
- **Parameters of other controllers,** such as stiffness or gains instead of a torque curve. PsyPref would need a prior over those parameters and a measure of how strongly a change is felt, which walking data do not provide directly. The library does not cover this.

If you try PsyPref on another device, we would like to hear how it went.
