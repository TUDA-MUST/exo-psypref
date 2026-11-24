# Projection onto a profile family

By default, PsyPref optimizes the whole curve through the weights of a ProMP. Those weights are hard to report, and a study may
need interpretable parameters, such as a peak's timing, to constrain or compare them. A projection keeps both: PsyPref holds its belief
over the ProMP, the user feels each profile of a pair as its least-squares fit in the family, and the result comes in the family's parameters.

```python
session = PsyPref(prior, family="schaefer")
...
session.parameters()      # the family's parameters
```

## Which to use

- **No projection** (`family=None`) if your question allows it. It reaches the highest preference in our simulations.
- **A family** if your study reports or constrains its parameters, or if your device can only produce such curves.

Showing the pairs in a family costs some preference. Its size depends on the family and on whether the peaks are free:

| Family | Parameters | Change against no projection, 3–35 Nm, after 24 comparisons |
|---|---|---|
| `"schaefer"` (Schäfer et al., four half-cosines) | 4 timings, 2 peaks | −6.1 [−7.8, −4.6] points |
| `"franks"` (Franks et al.) | 8, with 2 peaks | −0.6 points |
| `"pchip"` (Park et al.) | timings, 2 peaks | −1.1 points |

At a fixed 12 Nm, the three families lie 3.4 to 9.8 points lower than no projection.

The projection still beats searching the family directly. With its pairs shown in Schäfer et al.'s family, users would choose
PsyPref's profile over generic assistance more often than that of CMA-ES searching this family, by 18.8 [14.7, 22.8] points at
3–35 Nm and 9.0 [6.1, 11.9] at fixed 12 Nm. In that family, the search limits CMA-ES more than the family does.

Two cautions. The timing ranges of `"franks"` and `"pchip"` are fitted to our data, and with their published ranges their best
profiles reach 91.8 and 93.9 % at 3–35 Nm, 74.2 and 89.1 % at fixed 12 Nm. And all of this holds under our model of preferred
profiles: a family's best profile is known only in the simulation.

## Your own family

```python
from psypref import Family

family = Family(curve, low=[...], high=[...], names=[...])   # curve(theta) gives 101 points
session = PsyPref(prior, family=family)
```

See [examples/projection.py](../examples/projection.py). Each pair needs a least-squares fit, so the family's `curve` should be fast. A good `initial_guess(target)` makes the fits more reliable. We tested only the three families above.
