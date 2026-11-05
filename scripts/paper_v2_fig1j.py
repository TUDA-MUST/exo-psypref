"""Fig. 1 (figures/fig1.tex): the pair of the loop and of the worked answer in the users' box, as CSV.

Usage:
  python scripts/paper_v2_fig1j.py
      [--out <project>/out] [--csv figures/data] [--pair-k K]

Fig. 1's session (paper_v2_fig1.py: same user, same run, checked bit for bit against FULL's stored
row), with every pair it asked. The figure shows one pair in the loop and again in the bottom box,
with the preferred profile, both distances, and the probability the answer model gives.

The pair, by rule: among the first K_RULE comparisons, with d(A) < d(B) (so that P(choose A) reads
above 0.5), the pair whose d(B) - d(A) lies closest to 1 JND. --pair-k overrides the rule for
looking at another pair and is recorded in the meta file. Nothing here changes the session, and no
result of the paper depends on this choice.

Written (in --csv, every file prefixed fig1j_; fig1_*.csv stay untouched):
  fig1j_pair.csv   phase, preferred, pair1, pair2
  fig1j_pairs.csv  k (answers before the pair), d1, d2, delta, p, answer: all pairs of the session
  fig1j_meta.csv   the pair's k, distances, P(choose 1), the rule, the box's y limits, printed values
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    # as FULL's workers ran: one BLAS thread, so the session reproduces the stored floats
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import paper_v2_fig1 as F1  # noqa: E402
from prompref.user.oracle import rms_distance  # noqa: E402

K_RULE = 24              # the paper's budget: the pair comes from the comparisons it reports


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=F1.ROOT / "out")
    ap.add_argument("--csv", type=Path, default=F1.DEFAULT_CSV)
    ap.add_argument("--pair-k", type=int, default=None)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    stored, _ = F1.choose_user(args.out)
    job = next(j for j in F1.FULL.jobs(args.out) if (j["user"], j["setting"], j["candidate"])
               == (stored["user"], F1.SETTING, "psypref"))
    job["ties"] = "band"     # PsyPref-T, as paper_v2_fig1.py
    row, cap = F1.run_recorded(job, args.out)
    F1.check(row, stored)
    rec, design, observer = cap["rec"], cap["design"], cap["observer"]
    ideal = np.asarray(observer.ideal_nm, dtype=float)
    phase = np.linspace(0.0, 100.0, ideal.size)

    def curve(theta) -> np.ndarray:
        return np.asarray(design.curve(np.asarray(theta, dtype=float)), dtype=float)

    def prob(delta: float) -> float:
        return float(F1.LAPSE / 2 + (1 - F1.LAPSE) * norm.cdf(delta / F1.S_MAIN))

    jnd = float(norm.ppf(0.75)) * F1.S_MAIN
    table = []
    for k, (a, b) in enumerate(rec.pairs):
        d1, d2 = rms_distance(curve(a), ideal), rms_distance(curve(b), ideal)
        table.append([k, d1, d2, d2 - d1, prob(d2 - d1), rec.answers[k]])
    F1.write_csv(args.csv / "fig1j_pairs.csv", ["k", "d1", "d2", "delta", "p", "answer"], table)

    by_rule = min((r for r in table[:K_RULE] if r[3] > 0), key=lambda r: abs(r[3] - jnd))
    pick = table[args.pair_k] if args.pair_k is not None else by_rule
    k, d1, d2, delta, p, answer = pick
    ta, tb = rec.pairs[k]
    pair1, pair2 = curve(ta), curve(tb)
    F1.write_csv(args.csv / "fig1j_pair.csv", ["phase", "preferred", "pair1", "pair2"],
                 np.column_stack([phase, ideal, pair1, pair2]))

    drawn = np.concatenate([ideal, pair1, pair2])
    pad = 0.04 * float(np.ptp(drawn))
    meta = dict(k_pair=k, k_rule=by_rule[0], d1=d1, d2=d2, delta=delta, p=p, answer=answer, jnd=jnd,
                ymin=float(np.min(drawn)) - pad, ymax=float(np.max(drawn)) + pad,
                d1_txt=f"{d1:.2f}", d2_txt=f"{d2:.2f}", p_txt=f"{p:.2f}")
    F1.write_csv(args.csv / "fig1j_meta.csv", list(meta), [list(meta.values())])

    print(f"pair after {k} answers (rule: k = {by_rule[0]}): d(A) {d1:.3f}, d(B) {d2:.3f}, "
          f"d(B) - d(A) {delta:.3f} Nm = {delta / jnd:.2f} JND, P(choose A) {p:.3f}; the session's "
          f"answer: {['profile A', 'profile B', 'no preference'][answer]}")
    print(f"box limits {meta['ymin']:.2f} to {meta['ymax']:.2f} Nm; wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
