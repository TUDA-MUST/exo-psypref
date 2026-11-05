# Choosing a prior

The prior says what you know before the first comparison. In our simulations it carries much of PsyPref's speed, and it is
also the largest assumption. All numbers below come from simulated users (the paper, and [REPRODUCE.md](../REPRODUCE.md)).

| What you have | Call | Note |
|---|---|---|
| A hip exoskeleton, able-bodied users | `Prior.from_walking(peak_range=(3, 12))` | Hip moments of five subjects at 1.1 m/s, delayed and shifted like self-tuned profiles |
| Peaks that must stay fixed | `Prior.from_walking(level=12.0)` | Only the shape is tuned |
| Curves of your own, with both directions | `Prior.from_curves(curves, peak_range=(3, 12))` | Only their shapes count, the peaks come from the range |
| Curves of your own, one direction, or torques you trust | `Prior.from_curves(curves, limit=40.0)` | The curves count as they are |
| A mean and covariance of the weights | `Prior.from_moments(mu, sigma, peak_range=(3, 12))` | |

Curves have 101 points over the gait cycle, with flexion torque positive.

## The peak range

Set it to what your device and protocol allow. Published preferred and optimized hip peaks span 0.04–0.5 Nm/kg and disagree on
whether flexion or extension is larger, so PsyPref assumes nothing else. It draws both peaks log-uniformly and independently
in the range. A fixed level also costs something: users whose preferred peaks differ from it cannot get them.

## How wide

If you doubt that the prior fits your users, widen it with `width` or `prior.widened(2.0)`. The prior and its search region grow together.

| Prior | Preference over generic assistance after 24 comparisons, against the walking prior |
|---|---|
| 1.5 times wider | −1.7 to −2.5 points |
| 2 times wider | −3.8 to −6.9 points |
| 3 times wider | −15.4 to −21.4 points |

So some doubt is cheap, up to about twice the width. After 40 comparisons, a prior three times wider reaches only what the
walking prior reaches after 5–7. These costs hold for users who fit the prior, as our simulated users do. For users who do
not fit it, a wider prior may be worth more than the table shows.

## Another population, or added knowledge

- A prior from a population other than the users' cost PsyPref at most 2.1 and 5.6 points at 3–35 Nm and at fixed 12 Nm.
- Where users' shapes came from Schäfer et al.'s family at fixed 12 Nm, PsyPref trailed CMA-ES on that family by 3.1 points.
  With 30 of that family's curves added to the prior, it led by 1.9 [0.5, 3.3] points.

We have not tested a prior for patients. A prior from able-bodied walking cannot be assumed to fit them: after stroke, for
example, assistance proportional to the natural joint moment would give the weaker leg less support.

## How many weights

`n_weights` is the ProMP's number of weights: 8 by default, and walking priors exist for 8, 12, 16 and 24. In one session, more weights
make no measurable difference to PsyPref (after 24 comparisons with 24 instead of 8 weights: −0.4 [−1.8, 0.6] and
+1.5 [−0.6, 3.5] points). They pay off only in longer tuning: users would choose a 16-weight profile over an 8-weight profile in
50–53 % of cases after 24 and 40 comparisons, 55–57 % after 80, and 57–61 % after 150. This is not a test of larger problems, because
the prior's curves still vary in about the same few directions.

## Try it first

[examples/choose_a_prior.py](../examples/choose_a_prior.py) runs simulated users through your prior. Change the simulated users'
assumptions too, for example `noise`: at 3 Nm, the end that walking JNDs imply, PsyPref kept its lead, but at fixed 12 Nm only 28 % of the
users would benefit.
