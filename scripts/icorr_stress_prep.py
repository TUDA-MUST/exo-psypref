"""ICORR stress test S5: the link scales matched to the main simulated user.

Usage:
    python scripts/icorr_stress_prep.py --run <main>/out/phase5-schaefer [--out <main>/out/icorr]

S5 lets the simulated user answer by a logistic link (S5a) or by Student-t noise with nu = 3 (S5b) instead of
the probit, "matched to the main user's median |dp| on prior-scale pairs". Here:

- **Pairs.** REPORT section 9's prior-scale pairs (``phase5e_perturbation.duel_pairs``, 200 per user,
  seeded from the user id), for all 200 population users of the five LOSO folds.
- **|dp|** is |p - 1/2| of the attentive simulated user (lapse 0; the lapse multiplies every |dp| by the
  same 1 - lambda and drops out), with ``du`` the RMS utility difference of the pair against the
  user's ideal. The main simulated user's p is Phi(du / 1.0).
- **One scale per link, pooled.** kappa solves median over all 40,000 pairs of
  |F(|du| / kappa) - 1/2| = the same median for the probit, by Brent's method (both sides are
  monotone in |du|, so this matches the link at the median pair). A link is a property of the answer
  model, not of a user's metric, so one pooled scale serves every user; the per-user achieved
  ratios are reported.
- **Check.** The achieved median |dp| through ``build_observer``, as a session builds the simulated user,
  on the measurement pairs and on fresh pairs (``icorr_robust_prep.OOS_SEED``).

Writes ``OUT/stress_prep_links.json``; ``prompref.experiment.icorr_grid.LINK_SCALES`` carries the
scales at full precision, and the stress report checks every row against this file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import icorr_robust_prep as rp  # noqa: E402
import phase5e_perturbation as pert  # noqa: E402
from prompref.experiment import phase5 as p5  # noqa: E402
from prompref.experiment.phase5 import STRESS_LINKS, Phase5Point, build_observer  # noqa: E402
from prompref.experiment.phase5c_grid import LOSO  # noqa: E402
from prompref.user.simulated_user import Observer, link_cdf  # noqa: E402

SCHEMA = "stress-links-v1"


def stamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def population(run: str, fold: str) -> list[str]:
    users = p5._users(run, fold)
    return sorted(uid for uid, u in users.items() if u.kind == "population" and "/pop" in uid)


def abs_du(run: str, seed: int) -> tuple[np.ndarray, list[tuple[str, str]], list]:
    """|du| of every pair (pooled, in user order), the users, and each user's pairs."""
    values, users, all_pairs = [], [], []
    for fold in LOSO:
        for uid in population(run, fold):
            user = p5._users(run, fold)[uid]
            duels = (pert.duel_pairs(run, fold, uid, pert.N_PAIRS) if seed == pert.SEED
                     else rp.pairs(run, fold, uid, pert.N_PAIRS, seed))
            plain = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.0)
            values.append([abs(plain.utility(a) - plain.utility(b)) for a, b in duels])
            users.append((fold, uid))
            all_pairs.append(duels)
    return np.asarray(values, dtype=float), users, all_pairs


def matched_scale(du: np.ndarray, link: str) -> float:
    """kappa with median |F(|du| / kappa) - 1/2| = median |Phi(|du|) - 1/2| over the pooled pairs."""
    x = np.abs(np.asarray(du, dtype=float)).ravel()
    target = float(np.median(np.abs(norm.cdf(x) - 0.5)))

    def gap(kappa: float) -> float:
        return float(np.median(np.abs(link_cdf(link, x / kappa) - 0.5))) - target

    return float(brentq(gap, 1e-3, 1e3, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500))


def achieved(run: str, users, all_pairs, link: str, kappa: float) -> dict:
    """The median |dp| of the S5 simulated user built as a session builds it (lapse 0), pooled and per user,
    against the plain s = 1.0 probit simulated user on the same pairs."""
    cell, plain_dp, per_user = [], [], []
    for (fold, uid), duels in zip(users, all_pairs):
        user = p5._users(run, fold)[uid]
        point = Phase5Point(run, fold, "promp_s0", "tight", "a", uid, s=1.0, lapse=0.0,
                            n_comparisons=40, link=link, link_scale=kappa)
        observer, _ = build_observer(point, user, s_user=1.0, plateau=0.0)
        plain = Observer(ideal_nm=user.curve_nm, s=1.0, lapse=0.0)
        mine = [abs(observer.prob_prefers_a(a, b) - 0.5) for a, b in duels]
        ref = [abs(plain.prob_prefers_a(a, b) - 0.5) for a, b in duels]
        cell += mine
        plain_dp += ref
        per_user.append(float(np.median(mine) / np.median(ref)))
    return dict(median_abs_dp=float(np.median(cell)), main_median_abs_dp=float(np.median(plain_dp)),
                ratio=float(np.median(cell) / np.median(plain_dp)),
                per_user_ratio_quantiles=[float(q) for q in np.quantile(
                    per_user, [0.0, 0.1, 0.5, 0.9, 1.0])],
                mean_abs_dp=float(np.mean(cell)), main_mean_abs_dp=float(np.mean(plain_dp)),
                n_users=len(users), n_pairs=len(cell))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/../icorr")
    args = ap.parse_args(argv)
    run = str(Path(args.run).resolve())
    out = args.out or Path(run).parent / "icorr"
    started = time.perf_counter()
    du, users, measure_pairs = abs_du(run, pert.SEED)
    _, _, fresh_pairs = abs_du(run, rp.OOS_SEED)
    result = dict(schema=SCHEMA, when=stamp(), run=run,
                  spec="ICORR reframe section 9a, stress test S5",
                  pairs=f"phase5e_perturbation.duel_pairs, {pert.N_PAIRS} per user, seed "
                        f"[{pert.SEED}, crc32(user id)]; fresh pairs seed [{rp.OOS_SEED}, "
                        "crc32(user id)]",
                  users=f"{len(users)} population users (5 LOSO folds x 40)",
                  du_quantiles=[float(q) for q in np.quantile(du, [0.0, 0.1, 0.5, 0.9, 1.0])],
                  main_median_abs_dp=float(np.median(np.abs(norm.cdf(du) - 0.5))),
                  links={})
    for link in STRESS_LINKS:
        kappa = matched_scale(du, link)
        result["links"][link] = dict(
            link_scale=kappa, link_scale_repr=repr(kappa),
            measurement_pairs=achieved(run, users, measure_pairs, link, kappa),
            fresh_pairs=achieved(run, users, fresh_pairs, link, kappa))
        m = result["links"][link]
        print(f"{link}: kappa {kappa!r}; achieved ratio {m['measurement_pairs']['ratio']:.12f} "
              f"(measurement), {m['fresh_pairs']['ratio']:.4f} (fresh)", flush=True)
    result["seconds"] = round(time.perf_counter() - started, 1)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "stress_prep_links.json"
    target.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
