"""One tuning session: a Design, a learner and an observer, for N comparisons.

Presentation order is randomised by default. Without it, the observer's order bias
aliases directly onto the preference and the recovered optimum is silently shifted --
the classic paired-comparison confound.
"""

from __future__ import annotations

import numpy as np

from ..user.simulated_user import Observer, rms_distance
from .metrics import summarise_run


def run_session(design, learner, observer: Observer, rng: np.random.Generator,
                n_comparisons: int = 24, randomise_order: bool = True, *,
                d_prior: float, d_floor: float, scores: dict | None = None) -> dict:
    """Run one session and score what it delivered after every comparison.

    ``scores`` maps a record name to a function of a curve. Each is applied, after every
    comparison, to the same curve ``distances`` scores -- the one the learner would deliver
    now -- and recorded as a list under its name. Without it the session runs and records exactly as before.

    ``d_prior`` and ``d_floor`` are **required**, keyword-only, and have no defaults on
    purpose. Both are properties of the (design, ideal) pair rather than of the session,
    both are expensive enough to want caching by the caller, and a wrong value for either
    silently produces a complete, plausible, wrong record: ``d_prior`` is ``N90``'s
    denominator and ``d_floor`` its offset, so a default would not fail loudly, it would
    quietly mis-score every row. ``d_prior`` in particular must be the distance of the
    *mean curve* (:func:`~.metrics.prior_mean_distance`); the earlier default computed
    the curve at the box centre instead, which differs by a factor of 2.1 for the
    semi-trapezoid and whose error flips sign by family.
    """
    distances: list[float] = []
    answers: list[int] = []
    scores = dict(scores or {})
    traces: dict[str, list[float]] = {name: [] for name in scores}

    for _ in range(n_comparisons):
        theta_a, theta_b = learner.propose()
        curve_a = design.curve(theta_a)
        curve_b = design.curve(theta_b)
        order = 1 if (not randomise_order or rng.random() < 0.5) else -1
        response = observer.respond(curve_a, curve_b, rng, order=order)
        learner.update(theta_a, theta_b, response)
        answers.append(int(response))
        delivered = design.curve(learner.best())
        distances.append(rms_distance(delivered, observer.ideal_nm))
        for name, score in scores.items():
            traces[name].append(float(score(delivered)))

        # our design notes / a learner may end the session once its posterior is
        # tighter than the simulated user can feel. `n_comparisons_run` records where it stopped.
        should_stop = getattr(learner, "should_stop", None)
        if should_stop is not None and should_stop():
            break

    record = summarise_run(distances, d_prior=d_prior, d_floor=d_floor)
    record.update(distances=distances, answers=answers,
                  best=np.asarray(learner.best(), dtype=float).tolist())
    record.update(traces)
    record.update(learner.posterior_summary().get("extras", {}))
    return record
