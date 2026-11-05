"""ICORR review response: comparator fairness, ablations and a sanity check (PS, PF1, PF2, PA1,
PA2).

"Review
response: comparator fairness, ablations and a sanity check". Build: commit c5d76ae (``prompref/experiment/review.py``); blocks e-rev-fair and e-rev-abl
(``out/icorr/run_chain_review.py``). GP-EUBO and GP-BALD (PF3, PF3b) run in their own lane and are
not read here. No session is run.

Usage:
    python scripts/icorr_review_report.py [--out <project>/out]
        [--data <Schaefer share>] [--boot 2000] [--boot-verdict 10000]
        [--report <out>/icorr/review_report.md]

Reads, all read-only, each jsonl streamed line by line:
  OUT/icorr/e-rev-fair.jsonl   PS (C2, s = 0.5, lapse 0), SortCMA popsize 4 and 6, G with a generic
                               start with and without the gate;
  OUT/icorr/e-rev-abl.jsonl    A-wide (the wide arm), A-no-lapse;
  OUT/phase5e/e-main.jsonl     A, G, LineCoSpar and SortCMA (popsize 10) on ProMP, main condition;
  OUT/icorr/e-k.jsonl          C2 (Ramella's learner on ProMP) at s = 1.0 (W5);
  OUT/phase5e/e-tune.jsonl, e-oracle.jsonl   SortCMA popsize 4 and 6 at sigma multiplier 1.0, the
                               cross-check of the new SortCMA rows (sigma0 x 1.0 is sigma0);
  the simulated users, and through icorr_jnd_free.room_setup the Schaefer logs (aggregates only);
  OUT/icorr/review_c2_inspection.md, if present: the C2 inspection of.

The decision this report applies are in :data:`RULINGS`; they were fixed and committed before any
outcome of e-rev-fair or e-rev-abl was read, and the report prints them before the numbers.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import icorr_jnd_free as jf  # noqa: E402
import icorr_k_report as kr  # noqa: E402
import icorr_robust_report as rr  # noqa: E402
import phase5e_report as er  # noqa: E402
import schaefer_grounding as G  # noqa: E402
import short_budget_report as sb  # noqa: E402

K_MAIN = 24
KS = (12, 24, 40)
KMAX = 40
N_BOOT = jf.N_BOOT              # tables: icorr_jnd_free's draws
N_BOOT_VERDICT = 10_000         # verdicts: phase5e_report's joint bootstrap
TABLE_ROOMS = ("simulated", "central", "median")
PS_LEVEL = 0.5
FAVOUR = ("failed", "uncertain", "held")
INSPECTION = "review_c2_inspection.md"

#: Cells: key -> (block, label, the fields that select it). Every cell is ProMP, data prior,
#: 8 bases, EIG duels where A, 40 comparisons, the 200 population users; a field given as None
#: must be absent (its default) in the row.
CELLS = {
    "a": ("e-main", "A", dict(learner="a", arm="tight", s=1.0, lapse=0.1, a_no_lapse=None,
                              a_no_probes=None)),
    "g": ("e-main", "G (random start, gate)",
          dict(learner="g", arm="tight", s=1.0, lapse=0.1, jnd_scale=1.0, g_start=None,
               g_kappa0=None)),
    "d_z": ("e-main", "LineCoSpar", dict(learner="d_z", arm="tight", s=1.0, lapse=0.1)),
    "cmaes": ("e-main", "SortCMA (popsize 10, pycma default)",
              dict(learner="cmaes", arm="tight", s=1.0, lapse=0.1, cma_popsize=None,
                   cma_sigma_mult=None)),
    "c2": ("e-k", "Ramella (C2)", dict(learner="c2", arm="tight", s=1.0, lapse=0.1)),
    "cma4": ("e-rev-fair", "SortCMA (popsize 4)",
             dict(learner="cmaes", arm="tight", s=1.0, lapse=0.1, cma_popsize=4,
                  cma_sigma_mult=None)),
    "cma6": ("e-rev-fair", "SortCMA (popsize 6)",
             dict(learner="cmaes", arm="tight", s=1.0, lapse=0.1, cma_popsize=6,
                  cma_sigma_mult=None)),
    "g_gen": ("e-rev-fair", "G (generic start, gate)",
              dict(learner="g", arm="tight", s=1.0, lapse=0.1, jnd_scale=1.0, g_start="generic")),
    "g_gen_off": ("e-rev-fair", "G (generic start, no gate)",
                  dict(learner="g", arm="tight", s=1.0, lapse=0.1, jnd_scale=None,
                       g_start="generic")),
    "a_wide": ("e-rev-abl", "A-wide (z* ~ N(0, 9 I))",
               dict(learner="a", arm="wide", s=1.0, lapse=0.1, a_no_lapse=None,
                    a_no_probes=None)),
    "a_nolapse": ("e-rev-abl", "A-no-lapse (λ = 0 in A's model, no probes)",
                  dict(learner="a", arm="tight", s=1.0, lapse=0.1, a_no_lapse=True,
                       a_no_probes=None)),
    "ps": ("e-rev-fair", "Ramella (C2), easy wearer (s = 0.5, λ = 0)",
           dict(learner="c2", arm="tight", s=0.5, lapse=0.0)),
    # PA3 and PA4.
    "a_l3": ("e-lapse", "A, λ = 0.3 wearer (stored, e-lapse)",
             dict(learner="a", arm="tight", s=1.0, lapse=0.3, a_no_lapse=None, a_no_probes=None)),
    "a_nolapse_l3": ("e-rev-abl2", "A-no-lapse, λ = 0.3 wearer",
                     dict(learner="a", arm="tight", s=1.0, lapse=0.3, a_no_lapse=True,
                          a_no_probes=None)),
    "a_noprobe": ("e-rev-abl2", "A-no-probes (λ inferred, no probes)",
                  dict(learner="a", arm="tight", s=1.0, lapse=0.1, a_no_lapse=None,
                       a_no_probes=True)),
    "a_noprobe_l3": ("e-rev-abl2", "A-no-probes, λ = 0.3 wearer",
                     dict(learner="a", arm="tight", s=1.0, lapse=0.3, a_no_lapse=None,
                          a_no_probes=True)),
}
NEW = ("ps", "cma4", "cma6", "g_gen", "g_gen_off", "a_wide", "a_nolapse", "a_nolapse_l3",
       "a_noprobe", "a_noprobe_l3")
BLOCK_CELLS = {"e-rev-fair": ("ps", "cma4", "cma6", "g_gen", "g_gen_off"),
               "e-rev-abl": ("a_wide", "a_nolapse"),
               "e-rev-abl2": ("a_nolapse_l3", "a_noprobe", "a_noprobe_l3")}
#: PA4's non-inferiority margin, Nm.
PA4_MARGIN = 0.10
#: Every main-condition comparator (R8, R9): e-main's three, W5's C2 and the four new cells.
COMPARATORS = ("g", "d_z", "cmaes", "c2", "cma4", "cma6", "g_gen", "g_gen_off")
#: The comparators the paper's claim was made against (e-main's).
PAPER_COMPARATORS = ("g", "d_z", "cmaes")
ABLATIONS = ("a_wide", "a_nolapse")
#: The five W1 measures, and the claim's two.
MEASURES = ("win", "share", "err", "jnd1", "jnd2")
CLAIM_MEASURES = ("win", "err")
#: The direction that favours A in a contrast A minus comparator.
FAVOURS_A = {"win": +1, "share": +1, "err": -1, "jnd1": +1, "jnd2": +1}

BASE_FIELDS = dict(family="promp_s0", prior="data", n_basis=8, duel_mode="eig",
                   observer_metric="rms", s_user_logsd=0.0, asymmetry=1.0, familiarity_weight=0.0,
                   device_noise_nm=0.0, drift_nm=0.0, user_set=None, phase_weighting=None,
                   calibration_log=None)
FIELDS = jf.KEEP + ("n_basis", "duel_mode", "jnd_scale", "cma_popsize", "cma_sigma_mult",
                    "g_kappa0", "g_start", "a_no_lapse", "a_no_probes", "user_set",
                    "phase_weighting",
                    "calibration_log", "g_generic_distance", "g_generic_gap_nm",
                    "n_gate_blocked", "n_gate_exhausted", "cmaes_generations", "lambda_mean",
                    "error")
LISTS = ("distances",)

#: Analyst decision, fixed and committed before any outcome of e-rev-fair or e-rev-abl was read.
#: Where the test plan leaves a choice, each takes the reading that makes "held" harder.
RULINGS = (
    ("R1. Users and completeness.",
     "Population users only: the 200 of the five LOSO folds, in load_sim's order. Every new cell "
     "must hold exactly these users, once each, with no error row, 40 comparisons and its "
     "registered fields (PS: C2, s = 0.5, λ = 0; SortCMA: popsize 4 or 6, σ0 unchanged; G: generic "
     "start, jnd_scale 1.0 or none; A-wide: the wide arm; A-no-lapse: the flag, against the "
     "λ = 0.1 wearer). Every cell's d_prior and d_floor must equal e-main A's bit for bit, and "
     "every new SortCMA row must equal the stored e-tune or e-oracle row of the same user and "
     "popsize at σ multiplier 1.0 in every field but session_s and cma_sigma_mult. Otherwise the "
     "report refuses to write; there is no verdict on partial data."),
    ("R2. The win rate and the error.",
     "Per user, P_i(k) = λ/2 + (1 − λ)·Φ((d_prior,i − d_i(k))/s), the probability that the "
     "session's own wearer prefers the identified assistance after k comparisons over generic "
     "assistance (the prior mean curve, `d_prior`) in one duel, with the row's s and λ: s = 1.0 "
     "and λ = 0.1 in every cell but PS (s = 0.5, λ = 0). A-no-lapse is scored against its "
     "λ = 0.1 wearer: only A's model has λ = 0. The error is d_i(k) = distances[k − 1], RMS Nm to "
     "the ideal."),
    ("R3. Statistics.",
     "As for H, O, R and PW: verdicts at the simulated room, by a fold-stratified joint cluster "
     "bootstrap with 10,000 draws and phase5e_report's seed. Every array of a statistic uses the "
     "same draw of users, and every statistic the same draws. The 95 % ends are observed "
     "bootstrap values (inverted_cdf); medians are the lower middle value. A verdict is held only "
     "if its interval lies wholly and strictly on the predicted side of its threshold, failed if "
     "wholly on the other side, uncertain otherwise."),
    ("R4. PS.",
     "Held only if the lower end of C2's mean win rate at k = 24 lies strictly above 0.5; failed "
     "if the upper end lies below 0.5; uncertain otherwise. A failed PS carries the registered "
     "consequence: W5 is withdrawn from the paper until C2 has been checked. A PS that is not held "
     "(failed or uncertain) is stated in the report's first line and triggers an inspection of C2, "
     "fixed here: (a) an audit of preftune's `numpy_core.py` and prompref's `batch_mi.py` against "
     "the algorithm they document (APReL as Ramella et al. used it: reward linear in the "
     "batch-normalised features, softmax answers with β = 1, a uniform prior on ‖w‖ ≤ 1, "
     "Metropolis-Hastings with step 0.05, 200 burn-in, thinning 20, 100 samples, exhaustive "
     "mutual-information search over all pairs of the batch); (b) the equivalence evidence the "
     "code cites; (c) a diagnostic with a synthetic wearer whose utility is linear in C2's own "
     "normalised features, where a correct C2 must deliver a better candidate than its batch's "
     "average. The report states which explanation the evidence supports, a code fault or a "
     "model that cannot represent a bowl-shaped preference. No C2 cell is rerun or changed."),
    ("R5. PF1.",
     "A minus the better of SortCMA (popsize 4) and SortCMA (popsize 6), the better taken within "
     "each draw (the larger of the two mean win rates at k = 24), so the comparator is never "
     "fixed in advance. Held only if the interval lies wholly above 0."),
    ("R6. PF2.",
     "Two contrasts, A minus G with a generic start with the gate, and A minus G with a generic "
     "start without it, difference of mean win rates at k = 24. Held only if both lie wholly "
     "above 0; failed if either lies wholly below 0; uncertain otherwise."),
    ("R7. PA1 and PA2.",
     "The error at k = 24, difference of medians on the same users, ablation minus A (positive: "
     "the ablation is worse). Held only if the interval lies wholly above 0."),
    ("R8. \"A after 12 beats every comparator after 40\".",
     "Re-evaluated with icorr_jnd_free's cross-budget contrast (A at k = 12 minus the comparator "
     "at k = 40, paired by that script's draws) on the win rate and the error, at the three rooms "
     "the paper cites (simulated, central literature room, median-matched room), against every "
     "main-condition comparator: e-main's G, LineCoSpar and SortCMA (popsize 10), W5's C2, and "
     "the new SortCMA (popsize 4 and 6) and G with a generic start (gate on and off). The claim "
     "stands only if every one of these intervals excludes 0 in A's favour; otherwise it is "
     "dropped, and the report lists where it fails. The paper's longer version (all five W1 "
     "measures at the simulated and central rooms, and all but the share within 1 JND at the "
     "median-matched room) is judged the same way beside it."),
    ("R9. The best comparator.",
     "Among the same comparators, the one with the highest mean win rate at k = 24, named per "
     "room; the abstract's \"best published optimizer\" is the best at the room each of its "
     "numbers quotes (central and median-matched). The share of draws in which each comparator is "
     "the best is reported beside it. GP-EUBO and GP-BALD are not read here, so the name is "
     "provisional until their cells are in."),
    ("R10. Two estimators.",
     "The verdicts (§1) use phase5e_report's estimator (R3). The tables use icorr_jnd_free's: "
     "weighted Hazen medians and means, its shared draws (seed 20261009) and the KDE weight of each "
     "user from e-main A's room d_prior − d_floor, at the simulated, central and median-matched "
     "rooms. With equal weights the two medians can differ in the third decimal; quote §1 for "
     "the verdicts."),
    ("R11. G's generic start.",
     "Candidate 0 of G's set is z = 0, the prior mean in the arm's whitened coordinates, where A, "
     "LineCoSpar and SortCMA start. Its device-clipped curve is not exactly generic assistance "
     "(the mean of device-clipped prior curves); the gap is reported (`g_generic_gap_nm`), and the "
     "win rate is taken against generic assistance itself."),
    ("R12. A-no-lapse.",
     "λ = 0 exactly in A's likelihood (logit −∞: Φ alone in the update, the acquisition and the "
     "one-step predictive), and no intensity probes (p_probe = 0). The wearer lapses at 0.1."),
    ("R13. A failed prediction",
     "is reported as found (spec §9a)."),
    ("R14. PA3.",
     "At λ = 0.3, the error at k = 24 of A-no-lapse (λ = 0 in A's model, no probes) minus A's, "
     "difference of medians on the same 200 users, both against the λ = 0.3 wearer (s = 1.0). "
     "Held only if the interval lies wholly and strictly above 0; failed if wholly below 0; "
     "uncertain otherwise. A at λ = 0.3 is the stored e-lapse cell (R16)."),
    ("R15. PA4.",
     "Non-inferiority of A-no-probes (λ still inferred, probe share 0): its error at k = 24 minus "
     "A's, difference of medians, at λ = 0.1 (against e-main's A) and at λ = 0.3 (against "
     "e-lapse's A). Each part is held only if the upper end of its interval lies strictly below "
     "+0.10 Nm, failed if the lower end lies above +0.10 Nm, uncertain otherwise. PA4 is held only "
     "if both parts are held, failed if either part failed, uncertain otherwise. A part that is "
     "held says nothing about superiority in either direction."),
    ("R16. A at λ = 0.3.",
     "No new A cell runs at λ = 0.3: e-lapse (Phase 5e) stored A on ProMP in the main condition "
     "(s = 1.0, data prior, tight arm, 40 comparisons) at lapse 0.3 for every exported user, and "
     "the e-rev-abl2 cells are that point with one flag changed. Its 200 population rows are "
     "read as stored; one of them was re-run bit for bit under this build "
     "(`out/icorr/review_identity_check_abl2.json`). It is context and PA3's and PA4's "
     "reference, with no verdict of its own. At λ = 0.3 the win rate uses that wearer's lapse "
     "(R2)."),
)

#: What the analyst could see before the decision were committed (printed after them).
BLINDING_NOTE = (
    "The rulings were written before any e-rev row was read. Before the script was committed, it "
    "was tested end to end on synthetic rows built from stored rows only (e-main's A and G "
    "relabelled as the new A and G cells, W5's C2 relabelled as PS). The synthetic SortCMA rows "
    "were the stored e-tune and e-oracle sessions at popsize 4 and 6 and σ multiplier 1.0, which "
    "the cross-check of R1 shows are the very sessions the new cells reproduce. PF1's outcome, "
    "the SortCMA rows of the 12-versus-40 table and of the best-comparator table were therefore "
    "visible to the analyst at that test, after the rulings were written and before they were "
    "committed. No ruling was changed after the test. PA3 and PA4 were registered after PA2 and "
    "every other verdict above had been read; their rulings (R14–R16) were committed after that "
    "and before any e-rev-abl2 row existed. The stored e-lapse A rows were known from Phase 5e.")


# ============================================================================ rows
def load(path: Path, keep=None) -> pd.DataFrame | None:
    """Rows of a jsonl, error rows too unless ``keep`` filters them, normalised (n_basis 8,
    duel_mode "eig" filled in)."""
    df = jf.stream(path, keep=keep or (lambda r: True), fields=FIELDS, lists=LISTS)
    return None if df is None else kr.normalise(df)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def select(df: pd.DataFrame, want: dict) -> pd.DataFrame:
    """The rows matching ``BASE_FIELDS`` and ``want``: None means the field is absent or null
    (its default), a bool or a string must match exactly, a number must equal it."""
    fields = dict(BASE_FIELDS, n_comparisons=KMAX, **want)
    mask = np.ones(len(df), dtype=bool)
    for name, value in fields.items():
        col = _column(df, name)
        if value is None:
            mask &= col.isna().to_numpy()
        elif isinstance(value, bool):
            mask &= np.array([isinstance(x, (bool, np.bool_)) and bool(x) == value for x in col],
                             dtype=bool)
        elif isinstance(value, str):
            mask &= np.array([isinstance(x, str) and x == value for x in col], dtype=bool)
        else:
            mask &= (pd.to_numeric(col, errors="coerce") == float(value)).to_numpy()
    return df[mask]


def population(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["user_kind"] == "population").to_numpy()
              & df["user_id"].str.contains("/pop", regex=False).to_numpy()]


def gate_block(df: pd.DataFrame | None, block: str, keys) -> dict:
    """R1 for one new block: no error row, only the planned cells, each the 200 users once."""
    if df is None:
        raise SystemExit(f"{block}.jsonl is missing")
    if "error" in df and df["error"].notna().any():
        raise SystemExit(f"{block}: {int(df['error'].notna().sum())} error rows")
    df = population(df)
    out, n = {}, 0
    for key in BLOCK_CELLS[block]:
        c = select(df, CELLS[key][2])
        if len(c) != len(keys):
            raise SystemExit(f"{block} {key}: {len(c)} rows, expected {len(keys)}")
        out[key] = jf.in_order(c, keys)
        n += len(c)
    if n != len(df):
        raise SystemExit(f"{block}: {len(df) - n} population rows outside the registered cells")
    return out


def stored_cells(main: pd.DataFrame, ek: pd.DataFrame, keys, lapse: pd.DataFrame) -> dict:
    out = {}
    for key in ("a", "g", "d_z", "cmaes"):
        out[key] = jf.in_order(select(sb._n2048(population(main)), CELLS[key][2]), keys)
    out["c2"] = jf.in_order(select(population(ek), CELLS["c2"][2]), keys)
    out["a_l3"] = jf.in_order(select(population(lapse), CELLS["a_l3"][2]), keys)
    return out


def sortcma_cross_check(out: Path, keys, rows_dir: Path | None = None) -> dict:
    """R1: every new SortCMA row (popsize 4, 6; sigma multiplier None) against the stored e-tune or
    e-oracle row of the same user and popsize at sigma multiplier 1.0, every field but session_s
    and cma_sigma_mult."""
    want = set(keys)

    def rows(path, pred):
        found = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if pred(r) and (r["fold"], r["user_id"]) in want:
                    found[(r["fold"], r["user_id"], int(r["cma_popsize"]))] = r
        return found

    new = rows((rows_dir or out / "icorr") / "e-rev-fair.jsonl",
               lambda r: r.get("learner") == "cmaes" and r.get("cma_popsize") in (4, 6)
               and r.get("cma_sigma_mult") is None)
    old = {}
    for name in ("e-tune", "e-oracle"):
        old.update(rows(out / "phase5e" / f"{name}.jsonl",
                        lambda r: r.get("learner") == "cmaes" and r.get("cma_popsize") in (4, 6)
                        and r.get("cma_sigma_mult") == 1.0 and not r.get("error")))
    differ, missing = {}, []
    for k, r in new.items():
        if k not in old:
            missing.append(k)
            continue
        s = old[k]
        names = (set(r) | set(s)) - {"session_s", "cma_sigma_mult"}
        bad = sorted(n for n in names if r.get(n, "<absent>") != s.get(n, "<absent>"))
        if bad:
            differ[k] = bad
    return dict(n_new=len(new), n_matched=len(new) - len(missing), missing=missing, differ=differ)


# ============================================================================ per-user values
def errors(df: pd.DataFrame) -> np.ndarray:
    return rr.lst(df, "distances")


def win_curve(df: pd.DataFrame) -> np.ndarray:
    """(users, 40) win probabilities against generic assistance, the row's own wearer (R2)."""
    return jf.win_probability(rr.col(df, "d_prior")[:, None], errors(df), rr.col(df, "s")[:, None],
                              rr.col(df, "lapse")[:, None])


# ============================================================================ verdicts
def state(ci: dict, threshold: float = 0.0) -> str:
    """R3: held only if the interval lies wholly and strictly above ``threshold``."""
    return kr.state(ci, threshold, want_low=False)


def ps_state(ci: dict) -> str:
    """R4."""
    return state(ci, PS_LEVEL)


def both_state(states) -> str:
    """R6: held only if every part is held, failed if any part failed, else uncertain."""
    return kr.combine([None if s == "uncertain" else s for s in states])


def noninferior_state(ci: dict, margin: float = PA4_MARGIN) -> str:
    """R15: held only if the upper end lies strictly below the margin, failed if the lower end
    lies above it, uncertain otherwise."""
    if ci["hi"] < margin:
        return "held"
    if ci["lo"] > margin:
        return "failed"
    return "uncertain"


def minus_better(a, b, c):
    """R5: A's mean minus the larger of two comparators' means, per draw."""
    return a.mean(axis=1) - np.maximum(b.mean(axis=1), c.mean(axis=1))


def verdicts(c: dict, folds, n_boot: int) -> dict:
    boot = kr.Boot(folds, n_boot)
    W = {key: win_curve(df) for key, df in c.items()}
    E = {key: errors(df) for key, df in c.items()}
    v: dict = {"win": {}, "err": {}, "con": {}}
    for key in c:
        for k in KS:
            v["win"][(key, k)] = boot([W[key][:, k - 1]], rr.mean)
            v["err"][(key, k)] = boot([E[key][:, k - 1]], kr.med)
    w24 = {key: W[key][:, K_MAIN - 1] for key in c}
    e24 = {key: E[key][:, K_MAIN - 1] for key in c}
    ci = v["win"][("ps", K_MAIN)]
    v["PS"] = dict(ci=ci, state=ps_state(ci))
    ci = boot([w24["a"], w24["cma4"], w24["cma6"]], minus_better)
    v["PF1"] = dict(ci=ci, state=state(ci),
                    parts={o: boot([w24["a"], w24[o]], rr.diff_mean) for o in ("cma4", "cma6")})
    parts = {o: boot([w24["a"], w24[o]], rr.diff_mean) for o in ("g_gen", "g_gen_off")}
    v["PF2"] = dict(ci=parts, states={o: state(x) for o, x in parts.items()})
    v["PF2"]["state"] = both_state(v["PF2"]["states"].values())
    for name, key, ref in (("PA1", "a_wide", "a"), ("PA2", "a_nolapse", "a"),
                           ("PA3", "a_nolapse_l3", "a_l3")):
        ci = boot([e24[key], e24[ref]], kr.diff_of_medians)
        v[name] = dict(ci=ci, state=state(ci))
    parts = {key: boot([e24[key], e24[ref]], kr.diff_of_medians)
             for key, ref in (("a_noprobe", "a"), ("a_noprobe_l3", "a_l3"))}
    v["PA4"] = dict(ci=parts, states={key: noninferior_state(x) for key, x in parts.items()})
    v["PA4"]["state"] = both_state(v["PA4"]["states"].values())
    for o in COMPARATORS:
        v["con"][o] = boot([w24["a"], w24[o]], rr.diff_mean)
    v["best"] = boot([w24["a"]] + [w24[o] for o in COMPARATORS],
                     lambda a, *others: a.mean(axis=1) - np.max(
                         np.stack([x.mean(axis=1) for x in others]), axis=0))
    return v


# ============================================================================ tables
def table_stats(c: dict, rooms) -> dict:
    res = {}
    for key, df in c.items():
        res[key] = jf.compute_arm(kr.KArm(df, CELLS[key][1]), rooms, TABLE_ROOMS)
        print(f"  {key} done", flush=True)
    res["generic"] = jf.compute_generic(kr.KArm(c["a"], "A"), rooms, TABLE_ROOMS)
    return res


def cross(res: dict, ours: str, other: str, m: str, k_ours: int, k_other: int, room: str) -> tuple:
    """ours at k_ours minus other at k_other, paired by icorr_jnd_free's draws."""
    est = res[ours]["at"][(room, m, k_ours)][0] - res[other]["at"][(room, m, k_other)][0]
    b = (res[ours]["boot"][(room, m, k_ours)].astype(float)
         - res[other]["boot"][(room, m, k_other)].astype(float))
    return (est, *jf.ci(b))


def favours_a(t: tuple, m: str) -> bool:
    """The interval excludes 0 on A's side."""
    return t[1] > 0 if FAVOURS_A[m] > 0 else t[2] < 0


def claim(res: dict, comparators, measures_by_room: dict) -> dict:
    """R8: every (comparator, room, measure) contrast A(12) − X(40) and whether it favours A."""
    rows, fails = {}, []
    for o in comparators:
        for room, measures in measures_by_room.items():
            for m in measures:
                t = cross(res, "a", o, m, 12, 40, room)
                rows[(o, room, m)] = t
                if not favours_a(t, m):
                    fails.append((o, room, m))
    return dict(rows=rows, fails=fails, stands=not fails)


def best_comparator(res: dict, room: str) -> dict:
    """R9: the comparator with the highest mean win rate at k = 24 at ``room``, and how often each
    is the best within icorr_jnd_free's draws."""
    est = {o: res[o]["at"][(room, "win", K_MAIN)][0] for o in COMPARATORS}
    draws = np.stack([res[o]["boot"][(room, "win", K_MAIN)].astype(float) for o in COMPARATORS])
    which = np.argmax(draws, axis=0)
    share = {o: float(np.mean(which == i)) for i, o in enumerate(COMPARATORS)}
    best = max(est, key=est.get)
    a = res["a"]["boot"][(room, "win", K_MAIN)].astype(float)
    lead = res["a"]["at"][(room, "win", K_MAIN)][0] - est[best]
    lead_draw = a - np.max(draws, axis=0)
    return dict(best=best, est=est, share=share,
                lead_fixed=(lead, *jf.ci(a - draws[COMPARATORS.index(best)])),
                lead_within=(float(res["a"]["at"][(room, "win", K_MAIN)][0]
                                   - max(est.values())), *jf.ci(lead_draw)))


# ============================================================================ formatting
def ci3(d: dict, digits: int = 3, signed: bool = True) -> str:
    g = jf.fs if signed else jf.f
    return f"{g(d['estimate'], digits)} [{g(d['lo'], digits)}, {g(d['hi'], digits)}]"


def p3(d: dict) -> str:
    return ci3(d, 3, signed=False)


def git_head() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                                "scripts/icorr_review_report.py"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    digest = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]
    return f"{sha}{' (script uncommitted)' if dirty else ''}, script sha256 {digest}"


def room_title(setup, room: str) -> str:
    if room == "simulated":
        return "simulated room (as stored)"
    if room == "central":
        return f"central literature room, {setup['level']['central']:.2f} Nm"
    return f"median-matched room, {setup['level']['median']:.2f} Nm"


ROOM_SHORT = {"simulated": "simulated", "central": "central", "median": "median-matched"}


def label(key: str) -> str:
    return CELLS[key][1]


def verdict_section(v: dict) -> list[str]:
    out = ["## 1. Registered verdicts", "",
           "Simulated room, k = 24, the 200 population users; estimate [95 % CI] by the "
           "verdict estimator (R3).", "",
           "| # | Prediction (spec §9a) | Verdict | Numbers |", "|---|---|---|---|"]
    ps = v["PS"]
    out.append(f"| PS | C2's win rate against generic under the easy wearer (s = 0.5, λ = 0) "
               f"exceeds 0.5 (interval excluding 0.5) | **{ps['state']}** | C2 {p3(ps['ci'])} |")
    pf1 = v["PF1"]
    parts = "; ".join(f"A − {label(o)} {ci3(x)}" for o, x in pf1["parts"].items())
    out.append(f"| PF1 | A's win rate exceeds the better of SortCMA (popsize 4) and (popsize 6), "
               f"interval excluding 0 | **{pf1['state']}** | A − the better, within each draw, "
               f"{ci3(pf1['ci'])} ({parts}) |")
    pf2 = v["PF2"]
    parts = "; ".join(f"A − {label(o)} {ci3(x)} ({pf2['states'][o]})" for o, x in pf2["ci"].items())
    out.append(f"| PF2 | A's win rate exceeds G's with a generic start, with and without the "
               f"gate (both intervals excluding 0) | **{pf2['state']}** | {parts} |")
    for name, key, text in (("PA1", "a_wide", "A-wide has a higher error than A"),
                            ("PA2", "a_nolapse", "A-no-lapse has a higher error than A against "
                                                 "the λ = 0.1 wearer")):
        x = v[name]
        out.append(f"| {name} | {text} (difference of medians, interval excluding 0) | "
                   f"**{x['state']}** | {label(key)} − A {ci3(x['ci'])} Nm (medians "
                   f"{v['err'][(key, K_MAIN)]['estimate']:.3f} and "
                   f"{v['err'][('a', K_MAIN)]['estimate']:.3f} Nm) |")
    x = v["PA3"]
    out.append(f"| PA3 | At λ = 0.3, A-no-lapse has a higher error than A (interval excluding 0) | "
               f"**{x['state']}** | {label('a_nolapse_l3')} − {label('a_l3')} {ci3(x['ci'])} Nm "
               f"(medians {v['err'][('a_nolapse_l3', K_MAIN)]['estimate']:.3f} and "
               f"{v['err'][('a_l3', K_MAIN)]['estimate']:.3f} Nm) |")
    pa4 = v["PA4"]
    parts = "; ".join(
        f"{label(key)} − {label(ref)} {ci3(pa4['ci'][key])} Nm ({pa4['states'][key]})"
        for key, ref in (("a_noprobe", "a"), ("a_noprobe_l3", "a_l3")))
    out.append(f"| PA4 | A-no-probes is not worse than A by more than 0.10 Nm, at λ = 0.1 and at "
               f"λ = 0.3 (non-inferiority, interval below +0.10) | **{pa4['state']}** | {parts} |")
    out += ["", "PF3 and PF3b (GP-EUBO, GP-BALD) run in their own lane; they are not part of "
            "this report.", ""]
    if ps["state"] == "failed":
        out += ["**PS failed.** By the registration, our C2 implementation is suspect, and W5 is "
                "withdrawn from the paper until it has been checked (§6).", ""]
    elif ps["state"] == "uncertain":
        out += ["**PS is uncertain.** Its interval straddles 0.5. The registered consequence "
                "(W5 withdrawn) attaches to a failed PS; the inspection of R4 runs anyway (§6).",
                ""]
    return out


def verdict_tables(v: dict) -> list[str]:
    out = ["### 1.1 Every cell by the verdict estimator", "",
           "Mean win rate against generic and median error (Nm), simulated room, estimate "
           "[95 % CI]. PS's win rate is under its own easy wearer (s = 0.5, λ = 0), the three "
           "λ = 0.3 cells' under the λ = 0.3 wearer (s = 1.0), all others under the main wearer "
           "(s = 1.0, λ = 0.1).", "",
           "| cell | win k = 12 | win k = 24 | win k = 40 | error k = 12 | error k = 24 | "
           "error k = 40 |", "|---|---|---|---|---|---|---|"]
    for key in CELLS:
        out.append(f"| {label(key)} | " + " | ".join(p3(v["win"][(key, k)]) for k in KS) + " | "
                   + " | ".join(p3(v["err"][(key, k)]) for k in KS) + " |")
    out += ["", "A minus each comparator, mean win rate at k = 24 (positive favours A):", ""]
    for o in COMPARATORS:
        out.append(f"- A − {label(o)}: {ci3(v['con'][o])}")
    out.append(f"- A − the best comparator, the best taken within each draw: {ci3(v['best'])}")
    return out + [""]


def room_tables(res: dict, setup) -> list[str]:
    out = ["## 2. Win rate and error at k = 12 / 24 / 40, by room", "",
           "icorr_jnd_free's estimator (R10): mean win rate against generic assistance, median "
           "error in Nm, estimate [95 % CI]. The rooms reweight the same 200 users by e-main A's "
           "room d_prior − d_floor. New cells are marked †.", ""]
    for room in TABLE_ROOMS:
        ess = setup["rooms"].ess[room]
        out += [f"### {room_title(setup, room)}", "",
                f"Effective sample size {ess[0]:.0f}." if room != "simulated" else
                "Equal weights.", "",
                "| cell | win k = 12 | win k = 24 | win k = 40 | error k = 12 | error k = 24 | "
                "error k = 40 |", "|---|---|---|---|---|---|---|"]
        g = res["generic"]["at"][(room, "err")]
        out.append("| *generic assistance* | 0.50 | 0.50 | 0.50 | " + " | ".join(
            [jf.fci(g, 3)] * 3) + " |")
        for key in CELLS:
            r = res[key]["at"]
            mark = " †" if key in NEW else ""
            out.append(f"| {label(key)}{mark} | "
                       + " | ".join(jf.fci(r[(room, "win", k)]) for k in KS) + " | "
                       + " | ".join(jf.fci(r[(room, "err", k)], 3) for k in KS) + " |")
        out.append("")
    return out


def claim_section(res: dict, main_claim: dict, body_claim: dict, paper_claim: dict) -> list[str]:
    status = "**stands**" if main_claim["stands"] else "**is dropped**"
    out = ["## 3. \"A after 12 comparisons beats every comparator after 40\" (R8)", "",
           f"In win rate and error, at the simulated, central and median-matched rooms, against "
           f"every main-condition comparator, the claim {status}.", ""]
    if main_claim["fails"]:
        out.append("Where an interval does not exclude 0 in A's favour: " + "; ".join(
            f"{label(o)}, {ROOM_SHORT[room]}, {jf.MEASURE_NAME[m]} "
            f"{jf.fci(main_claim['rows'][(o, room, m)], jf.DIGITS[m], signed=True)}"
            for o, room, m in main_claim["fails"]) + ".")
        out.append("")
    out.append("Against e-main's three comparators alone (the paper's version), it "
               + ("still holds." if paper_claim["stands"] else "fails at: " + "; ".join(
                   f"{label(o)}, {ROOM_SHORT[r]}, {jf.MEASURE_NAME[m]}"
                   for o, r, m in paper_claim["fails"]) + "."))
    out.append("")
    out.append("The paper's longer version (all five W1 measures at the simulated and central "
               "rooms, all but the share within 1 JND at the median-matched room), against every "
               "comparator: " + ("holds." if body_claim["stands"] else "fails at " + "; ".join(
                   f"{label(o)}, {ROOM_SHORT[r]}, {jf.MEASURE_NAME[m]} "
                   f"{jf.fci(body_claim['rows'][(o, r, m)], jf.DIGITS[m], signed=True)}"
                   for o, r, m in body_claim["fails"]) + "."))
    out += ["", "A (k = 12) minus each comparator (k = 40), paired by icorr_jnd_free's draws. Win "
            "rate: positive favours A; error: negative favours A.", "",
            "| comparator | room | win rate | error, Nm |", "|---|---|---|---|"]
    for o in COMPARATORS:
        for room in TABLE_ROOMS:
            out.append(f"| {label(o)} | {ROOM_SHORT[room]} | "
                       f"{jf.fci(main_claim['rows'][(o, room, 'win')], 2, signed=True)} | "
                       f"{jf.fci(main_claim['rows'][(o, room, 'err')], 3, signed=True)} |")
    return out + [""]


def best_section(bests: dict) -> list[str]:
    out = ["## 4. The best comparator (R9)", "",
           "The comparator with the highest mean win rate at k = 24, per room (icorr_jnd_free's "
           "estimator). Share of draws: how often each comparator is the best within the draws.",
           "", "| room | best | its win rate | A's win rate | A − it [95 % CI] | A − the best "
           "within each draw | share of draws best |", "|---|---|---|---|---|---|---|"]
    for room, b in bests.items():
        shares = ", ".join(f"{label(o)} {s:.2f}" for o, s in
                           sorted(b["share"].items(), key=lambda t: -t[1]) if s > 0)
        out.append(f"| {ROOM_SHORT[room]} | {label(b['best'])} | {b['est'][b['best']]:.3f} | "
                   f"{b['a']:.3f} | {jf.fci(b['lead_fixed'], 3, signed=True)} | "
                   f"{jf.fci(b['lead_within'], 3, signed=True)} | {shares} |")
    out += ["", "GP-EUBO and GP-BALD are not read here: the name is provisional until their "
            "cells are in (R9).", ""]
    return out


def diagnostics_section(c: dict, cross_check: dict) -> list[str]:
    out = ["## 5. What the new cells did", ""]
    for key in ("g_gen", "g_gen_off"):
        df = c[key]
        E = errors(df)
        gen = rr.col(df, "g_generic_distance")
        at = {k: float(np.mean(E[:, k - 1] == gen)) for k in KS}
        gap = rr.col(df, "g_generic_gap_nm")
        blocked = rr.col(df, "n_gate_blocked")
        exhausted = rr.col(df, "n_gate_exhausted")
        out.append(f"- **{label(key)}.** Share of users whose G still delivers its generic "
                   "candidate: " + ", ".join(f"k = {k} {at[k]:.2f}" for k in KS)
                   + f". The generic candidate's curve lies {np.median(gap):.3f} Nm RMS (median; "
                   f"max {np.max(gap):.3f}) from generic assistance. Gate blocks per session: "
                   f"median {np.median(blocked):.0f}, max {np.max(blocked):.0f}; exhausted: "
                   f"median {np.median(exhausted):.0f}, max {np.max(exhausted):.0f}.")
    for key in ("cmaes", "cma4", "cma6"):
        gens = rr.col(c[key], "cmaes_generations")
        out.append(f"- **{label(key)}.** Completed generations after 40 comparisons: median "
                   f"{np.median(gens):.0f} (range {np.min(gens):.0f}–{np.max(gens):.0f}).")
    for key in ("c2", "ps"):
        E = errors(c[key])
        dp = rr.col(c[key], "d_prior")
        out.append(f"- **{label(key)}.** Share of users whose identified assistance is closer to "
                   "the ideal than generic assistance: " + ", ".join(
                       f"k = {k} {np.mean(E[:, k - 1] < dp):.2f}" for k in KS) + ".")
    for key in ("a_nolapse", "a_nolapse_l3"):
        lam = rr.col(c[key], "lambda_mean")
        out.append(f"- **{label(key)}.** λ in A's model: {np.max(np.abs(lam)):.1f} in every "
                   "session (pinned).")
    for key in ("a", "a_noprobe", "a_l3", "a_noprobe_l3"):
        lam = rr.col(c[key], "lambda_mean")
        lo, mid, hi = np.percentile(lam, [10, 50, 90])
        out.append(f"- **{label(key)}.** A's posterior mean λ after 40 comparisons: median "
                   f"{mid:.3f} (10–90 %: {lo:.3f}–{hi:.3f}); the wearer's λ is "
                   f"{rr.col(c[key], 'lapse')[0]:.1f}.")
    out.append(f"- **SortCMA cross-check (R1).** {cross_check['n_matched']} of "
               f"{cross_check['n_new']} new SortCMA rows have a stored e-tune / e-oracle "
               f"counterpart at σ multiplier 1.0; {len(cross_check['differ'])} differ in any "
               "field but session_s and cma_sigma_mult.")
    return out + [""]


def inspection_section(out_dir: Path, ps_state_: str) -> list[str]:
    out = ["## 6. C2 inspection (R4)", ""]
    if ps_state_ == "held":
        return out + ["PS is held; R4's inspection was not triggered.", ""]
    path = out_dir / "icorr" / INSPECTION
    if not path.exists():
        return out + [f"PS is {ps_state_}: the inspection of R4 is pending "
                      f"(`out/icorr/{INSPECTION}` is not written yet).", ""]
    text = path.read_text(encoding="utf-8").strip().splitlines()
    return out + [f"From `out/icorr/{INSPECTION}` (written after the verdicts, by the analyst; "
                  "no C2 cell was rerun or changed):", ""] + text + [""]


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--data", type=Path, default=G.DEFAULT_DATA)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-verdict", type=int, default=N_BOOT_VERDICT)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--rows", type=Path, default=None,
                    help="the directory holding e-rev-fair.jsonl and e-rev-abl.jsonl (default "
                         "OUT/icorr; another one only to test the script on synthetic rows)")
    args = ap.parse_args(argv)
    out = args.out
    rows_dir = args.rows or out / "icorr"
    report = args.report or out / "icorr" / "review_report.md"
    sys.stdout.reconfigure(encoding="utf-8")
    if args.boot < 2000 or args.boot_verdict < 2000:
        raise SystemExit("at least 2000 bootstrap draws")
    started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    sim = G.load_sim(out / "phase5-schaefer")
    keys = sim["keys"]
    folds = np.asarray([f for f, _ in keys])

    print("streaming rows ...", flush=True)
    blocks = {b: load(rows_dir / f"{b}.jsonl") for b in BLOCK_CELLS}
    counts = {b: (0 if df is None else len(df)) for b, df in blocks.items()}
    c = {}
    for b, df in blocks.items():
        c.update(gate_block(df, b, keys))
    main_rows = load(out / "phase5e" / "e-main.jsonl",
                     keep=lambda r: jf._pop_row(r) and r.get("family") == "promp_s0")
    ek = load(out / "icorr" / "e-k.jsonl",
              keep=lambda r: jf._pop_row(r) and r.get("learner") == "c2")
    lapse_rows = load(out / "phase5e" / "e-lapse.jsonl",
                      keep=lambda r: (jf._pop_row(r) and r.get("family") == "promp_s0"
                                      and r.get("learner") == "a"))
    c.update(stored_cells(main_rows, ek, keys, lapse_rows))
    c = {key: c[key] for key in CELLS}
    checks = []
    for key, df in c.items():
        if not np.array_equal(df["fold"].to_numpy(), folds):
            raise SystemExit(f"{key}: user order differs")
        for field in ("d_prior", "d_floor"):
            d = float(np.max(np.abs(rr.col(df, field) - rr.col(c["a"], field))))
            if d != 0.0:
                raise SystemExit(f"{key}: {field} differs from e-main A's by {d}")
        checks.append(key)
    cross_check = sortcma_cross_check(out, keys, rows_dir)
    if cross_check["missing"] or cross_check["differ"] or cross_check["n_new"] != 2 * len(keys):
        raise SystemExit(f"SortCMA cross-check failed: {cross_check}")

    print("verdicts ...", flush=True)
    v = verdicts(c, folds, args.boot_verdict)

    print("rooms and weights ...", flush=True)
    room = rr.col(c["a"], "d_prior") - rr.col(c["a"], "d_floor")
    setup = jf.room_setup(out, args.data, args.boot, room, folds)
    if setup["sim"]["keys"] != keys or not np.array_equal(setup["sim"]["folds"], folds):
        raise SystemExit("user order differs between load_sim and the room setup")
    rooms = setup["rooms"]
    print("tables ...", flush=True)
    res = table_stats(c, rooms)
    repro = []
    for (m, k), want in kr.REPRO_K8.items():
        got = jf.f(res["a"]["at"][("simulated", m, k)][0], jf.DIGITS[m])
        repro.append((m, k, got, want, got == want))
    gate_median = jf.gate(kr.KArm(c["a"], "A"), rooms, ("median",))["median"]
    median_ok = jf.f(gate_median[0], 2) == f"{jf.GATE_MEDIAN[0]:.2f}"
    if not all(x[4] for x in repro) or not median_ok:
        print(f"REPRODUCTION FAILED: {repro}, median room {gate_median}; nothing written")
        return 1

    main_claim = claim(res, COMPARATORS, {room: CLAIM_MEASURES for room in TABLE_ROOMS})
    paper_claim = claim(res, PAPER_COMPARATORS, {room: CLAIM_MEASURES for room in TABLE_ROOMS})
    body_claim = claim(res, COMPARATORS, {"simulated": MEASURES, "central": MEASURES,
                                          "median": tuple(m for m in MEASURES if m != "jnd1")})
    bests = {}
    for room_ in TABLE_ROOMS:
        bests[room_] = best_comparator(res, room_)
        bests[room_]["a"] = res["a"]["at"][(room_, "win", K_MAIN)][0]

    L: list[str] = []
    p = L.append
    ps = v["PS"]
    if ps["state"] != "held":
        p(f"**PS {ps['state']}: Ramella's learner (C2) under the easy wearer (s = 0.5, λ = 0) "
          f"reaches a mean win rate of {p3(ps['ci'])} against generic assistance at k = 24"
          + (" — W5 is withdrawn until C2 has been checked (§6)."
             if ps["state"] == "failed" else "; the C2 inspection of R4 runs (§6).") + "**")
        p("")
    p("# ICORR review response: comparator fairness, ablations and a sanity check")
    p("")
    p(f"Generated by `scripts/icorr_review_report.py`, run started {started}, at worktree commit "
      f"{git_head()}. Spec: `docs/superpowers/specs/2026-10-17-icorr-user-tuning-reframe-design.md`"
      " §9a, \"Review response: comparator fairness, ablations and a sanity check\" (PS, PF1, "
      "PF2, PA1, PA2 registered 2026-10-20 before any code or run). Build: commit c5d76ae "
      "(`prompref/experiment/review.py`). Rows: " + ", ".join(
          f"`out/icorr/{b}.jsonl` ({n} rows)" for b, n in counts.items())
      + "; `out/phase5e/e-main.jsonl`, `out/icorr/e-k.jsonl` and `out/phase5e/e-lapse.jsonl` "
      "(stored cells). No session was run. PA3 and PA4 (registered 2026-10-20 after PA2; build "
      "commit 86b5a9c, block e-rev-abl2) are judged with the same estimator.")
    p("")
    p("**Cells.** Main condition (s = 1.0, λ = 0.1), ProMP, data prior, the 200 population users, "
      "40 comparisons read at k = 12 / 24 / 40. New: SortCMA with pycma popsize 4 and 6 (e-main's "
      "is pycma's default, 10); G with a generic start, with the true-JND gate and without a gate; "
      "A in the wide arm (z* ~ N(0, 9 I)); A with λ = 0 in its model and no probes; and, for PS, "
      "Ramella's learner (C2) under an easy wearer (s = 0.5, λ = 0). Stored: e-main's A, G, "
      "LineCoSpar and SortCMA, and W5's C2 at s = 1.0. Every cell holds the same users with the "
      "same answer seeds. PA3/PA4 add A-no-lapse against a λ = 0.3 wearer and A-no-probes (λ still "
      "inferred, probe share 0) against the λ = 0.1 and λ = 0.3 wearers; A itself at λ = 0.3 is "
      "e-lapse's stored cell.")
    p("")
    p("## 0. Rulings (fixed and committed before any outcome was read)")
    p("")
    for head, text in RULINGS:
        p(f"- **{head}** {text}")
    p("")
    p(f"**What was visible before the commit.** {BLINDING_NOTE}")
    p("")
    L.extend(verdict_section(v))
    L.extend(verdict_tables(v))
    L.extend(room_tables(res, setup))
    L.extend(claim_section(res, main_claim, body_claim, paper_claim))
    L.extend(best_section(bests))
    L.extend(diagnostics_section(c, cross_check))
    L.extend(inspection_section(out, ps["state"]))
    p("## 7. Checks")
    p("")
    p("Every check passed; otherwise nothing would have been written (R1).")
    p("- " + ", ".join(f"{b}: {n} rows" for b, n in counts.items()) + ", 0 error rows; each new "
      "cell holds the 200 population users once, in load_sim's order, with its registered "
      "fields. The stored A at λ = 0.3 (e-lapse) holds the same 200 users once each.")
    p("- d_prior and d_floor of every cell equal e-main A's bit for bit (" + ", ".join(
        label(k) for k in checks if k != "a") + ").")
    p(f"- SortCMA cross-check: all {cross_check['n_new']} new SortCMA rows equal their stored "
      "e-tune (loso-AB30) or e-oracle (the other four folds) rows at σ multiplier 1.0 in every "
      "field but session_s and cma_sigma_mult.")
    p("- Reproduction (icorr_jnd_free's estimator, A at the simulated room, against `jnd_free.md` "
      "§1): " + "; ".join(f"{jf.MEASURE_NAME[m]} at k = {k} {got} (expected {want})"
                          for m, k, got, want, _ in repro)
      + f"; A's Q≥1 at k = 24 at the median-matched room {jf.fci(gate_median)} (expected "
      f"{jf.GATE_MEDIAN[0]:.2f} [{jf.GATE_MEDIAN[1]:.2f}, {jf.GATE_MEDIAN[2]:.2f}]).")
    p("- The build's bit identity (stored keys and re-runs) is recorded in "
      "`out/icorr/review_identity_check.json`.")
    p("")
    p("## 8. Caveats")
    p("")
    p("- **The win rate is a model probability** under the wearer that answered the session "
      "(R2), exact for that wearer, with a ceiling of 0.95 at λ = 0.1 (1.0 for PS's wearer). It "
      "is not a measured preference.")
    p("- **PS's wearer is not the main wearer.** Its win rate is under s = 0.5 and λ = 0, so it "
      "compares with C2's own s = 1.0 row only in direction, not in size.")
    p("- **SortCMA tells pycma the ball-clipped offspring it showed**, as in every earlier "
      "SortCMA cell; at small popsizes pycma warns about these injected solutions "
      "(`InjectionWarning` in `runs_review.log`). The cross-check shows the new rows are the "
      "e-tune / e-oracle sessions, so this is the stored behaviour, not a new one.")
    p("- **G's generic candidate is z = 0** (R11), not generic assistance's own curve.")
    p("- **A-wide also widens the trust ball** (the arm's 99 % radius, three times the tight "
      "one): the arm removes the data prior's information and lets A search further, as the "
      "existing arm always has.")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    for name in ("PS", "PF1", "PF2", "PA1", "PA2", "PA3", "PA4"):
        print(name, v[name]["state"])
    print("claim stands:", main_claim["stands"], "best:", {r: b["best"] for r, b in bests.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
