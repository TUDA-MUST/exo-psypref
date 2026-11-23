# PsyPref

**Tune an exoskeleton's assistance curve from A/B comparisons, with few of them.**

Assistance that fits the person often works best, and users can say which of two torque profiles feels better. PsyPref decides
which two profiles to compare next and what to make of each answer, using Bayesian adaptive methods from psychophysics: prior
knowledge, the most informative comparison, and a model of noisy answers. In simulation, this reaches after 3–8 comparisons what the
strongest other optimizer reaches after 40.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/session_dark.gif">
  <img alt="A simulated tuning session: pairs of torque curves, the belief narrowing around the preferred profile, the effect of the prior's width and peak range, and a projection onto a curve family" src="assets/session_light.gif">
</picture>

> **Everything here is simulated.** No person has used PsyPref yet. It rests on assumptions that only experiments can test, above all that people judge whole curves and have one preferred profile. See [what the results depend on](#what-the-results-depend-on).

## Quick start

```bash
git clone https://github.com/TUDA-MUST/exo-psypref.git
cd exo-psypref
pip install -e .
python examples/quickstart.py
```

```python
from psypref import Prior, PsyPref

prior = Prior.from_walking(peak_range=(3, 12))   # hip torque peaks your device and protocol allow, in Nm
tuner = PsyPref(prior)

for k in range(24):
    a, b = tuner.ask()                  # two torque curves, 101 points over the gait cycle
    tuner.tell(show_to_user(a, b))      # your code: "A", "B", or None for no preference

curve = tuner.identified()              # the identified assistance
```

Show each profile long enough to feel it. With 20 s per profile, as in Ramella et al.'s study, a comparison takes about 50 s, so 24
comparisons take about 20 min. Present each pair in random order, and keep your controller's own torque limits in place.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/how_it_works_dark.png">
  <img alt="Four panels: curves from the prior, a pair A and B, a narrower belief, and the identified assistance against generic assistance and the preferred profile" src="assets/how_it_works_light.png">
</picture>

PsyPref describes the profile with eight weights of a probabilistic movement primitive (ProMP), a sum of basis functions over the gait cycle, and keeps a belief about the
profile the user prefers. It begins with a prior built from other people's walking data, asks the pair expected to reveal most, and
treats each answer as a noisy choice. It learns how sharply the user discriminates during the session, so nobody has to measure a
just-noticeable difference (JND) first.

## Make it fit your study

| Your question | Where to look |
|---|---|
| Which prior? How wide? Which peak range? How many weights? | [Choosing a prior](docs/choosing_a_prior.md) |
| My study needs named parameters, or my device only produces certain shapes | [Projection onto a family](docs/projections.md) |
| Another joint, device or population | [Using PsyPref on another exoskeleton](docs/other_exoskeletons.md) |

| You want | Call |
|---|---|
| Walking data, peaks free in a range | `Prior.from_walking(peak_range=(3, 12))` |
| Walking data, peaks fixed | `Prior.from_walking(level=12.0)` |
| Your own curves | `Prior.from_curves(curves, peak_range=(3, 12))` |
| One-sided assistance, or curves that count as they are | `Prior.from_curves(curves, limit=40.0)` |
| Some doubt about the prior | `prior.widened(2.0)` |
| Pairs shown in an interpretable family | `PsyPref(prior, family="schaefer")`, or your own `Family` |
| A session continued after a break | `PsyPref.replay(prior, tuner.answers, seed=seed)` |
| A dry run before anyone walks | `SimulatedUser(preferred, noise=1.0)`, see [examples](examples/) |

## What we found in simulation

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speed_dark.png">
  <img alt="Preference over generic assistance against comparisons, in three peak settings: PsyPref reaches the strongest other optimizer's 40-comparison level after 3, 5 and 8 comparisons" src="assets/speed_light.png">
</picture>

Means of 200 simulated users per setting, from 5 walking subjects, with 95 % intervals.

- **Fewer comparisons.** After 3–8 comparisons, users would choose PsyPref's profile over generic assistance as often as the
  profile that the strongest other optimizer reaches after 40. All optimizers share the prior here, except CMA-ES on a family.
- **Prior and pairs carry the speed.** After 24 comparisons, a prior three times wider costs 15.4–21.4 points and random pairs 14.3–16.6,
  against 2.5–5.8 for replacing the answer model.
- **More weights do not slow it down,** but within one session they do not help either. They pay off in longer tuning.
- **Not there yet.** After 24 comparisons, users stay 2.1–3.8 JND from their preferred profile, and they would still prefer profiles found later.

### What the results depend on

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/robustness_dark.png">
  <img alt="PsyPref's lead over the strongest other optimizer under broken assumptions: positive in most cases, but not secure for users who judge single features or have no single preferred profile" src="assets/robustness_light.png">
</picture>

PsyPref stays ahead in 68 of 75 robustness cases. Its lead does not survive users who judge single features or have no single preferred
profile, and at fixed 12 Nm it trails on shapes from the comparator's own family and for users who break six assumptions at once. Where users
discriminate more coarsely, the lead remains but the benefit shrinks: at fixed 12 Nm and a noise of 3 Nm, only 28 % of the users would
benefit. Our simulated users answer by a model close to PsyPref's, and the prior fits them by construction, which favors PsyPref.
How people judge assistance curves has not been measured, and that is what we consider the most useful next step.

## Reproducing the paper

The code, the logs and every figure's data are here. [REPRODUCE.md](REPRODUCE.md) leads from each figure to its script and data,
and [CONDITIONS.md](CONDITIONS.md) lists every simulated condition with its size and source. The paper's Supplementary
Fig. S1, every robustness test against the best published optimizer, is [figures/figS1_robustness.pdf](figures/figS1_robustness.pdf).

```bash
pytest tests/test_psypref_library.py   # with the data bundle: this library reruns the paper's sessions, to within 1e-12 Nm
```

## How to cite

The paper's reference will follow.

## License

The code is under the [MIT License](LICENSE). The data are under CC BY 4.0, subject to the terms of the source datasets.
