"""Fig. 3 of the ICORR 2027 paper, new version (track fig3p): the projection onto an interpretable
family, as CSVs for figures/fig3p.tex.

Usage:
  python -B scripts/paper_v2_fig3p.py
      [--out <project>/out] [--csv figures/data]

Test FAM. Nothing in out/ is
written; no stored row is changed.

(a) The projection diagram: ONE projection session, rerun here and checked against FAM's stored row
  * User: Fig. 1's (paper_v2_fig1.choose_user: among FULL's 200 PsyPref sessions in 3-35 Nm, the
    lower of the two middle users by distance after 24 comparisons); the script stops unless it
    equals data/fig1_meta.csv's user.
  * Session: FAM's "PsyPref with projection" onto Schaefer et al.'s family (ii) (four timings and
    two free peaks, 6 parameters) in 3-35 Nm, built and run exactly as icorr_amp_fam.run_one
    builds it (AM with the two-part guard on the K = 8 ProMP; AF.FamilyLens: each proposal shown
    as its device-clipped bounded least-squares fit from the family's initial guess, the previous
    fit and the box centre; PsyPref told the ProMP projection of the curve shown; the delivered
    curve the fit to PsyPref's identified curve). run_session is wrapped for the call only: the
    wrapper records the inner learner's proposals and the projections it is told, and the lens's
    fits and delivered parameters, passing every call through (no extra call, no random draw).
    The script stops unless all 40 distances, identified peaks, d0 and both lens-gap statistics
    equal fam_sessions.jsonl's row bit for bit.
  * Drawn: one pair, by rule the first one asked after at least six answers (Fig. 1 shows the pair
    after six answers) whose two fit gaps (RMS between a proposal and its family fit) both lie
    inside the interquartile range of all the session's 121 fits, the fits FAM's lens-gap
    statistics summarize, so that the diagram shows a typical fit. (The pair after exactly six
    answers is atypical in this session by its fit gaps, both above the session's 75th
    percentile; the script prints them. Touching the family's box bounds is not what sets it
    apart: most of the session's fits touch the published timing box, the drawn pair's too.) Of
    that pair: its two ProMP curves tau(z1), tau(z2), their family fits tau_F(theta1),
    tau_F(theta2) (the curves the user feels), the fits' ProMP projections tau(z~1), tau(z~2)
    (what PsyPref is told), the family's nodes (theta: zero crossing, extension peak, zero
    crossing, flexion peak); and the delivered curve after 24 comparisons (the fit of PsyPref's
    identified curve). fig3p.tex draws the pair in the ProMP and as shown, with the nodes.

(b) Chosen over generic assistance after 24 comparisons, % (100 x mean over the 200 users of
  lam/2 + (1 - lam) Phi((d0 - d24) / s), s = 1 Nm, lam = 0.1), per setting (3-35 Nm = "wide",
  fixed 12 Nm; FAM has no 3-12 Nm) and row:
  * Schaefer et al.'s family ((ii), 6 parameters, in 3-35 Nm; (i), 4, at fixed 12 Nm), Franks et
    al.'s (8 / 6), PCHIP ((ii) 6 / 4): the family's best profile (best-profile level: mean preference at each
    user's best-profile distance, fam_floors.jsonl) and PsyPref with projection onto it (fam_sessions.jsonl). The
    semi-trapezoid, also in FAM, is not drawn: our implementation does not reproduce the torques
    measured by Ramella et al. (scripts/provenance/trap_*.py, paper/notes/trapezoid_check_2026-11-02.md);
  * ProMP (8 weights): the best 8-weight profile and PsyPref on the ProMP itself (FULL's rows,
    full_sessions.jsonl), the values of Fig. 2 and Table I;
  * CMA-ES on Schaefer et al.'s family, planned configuration, the dash-dot line of Fig. 2, drawn on the Schaefer row.
  Point estimates are recomputed here from the rows. Intervals (95 %, 2000 draws, users resampled
  within their walking subject): the families' from fam_numbers.json, the
  ProMP row's and CMA-ES's from amp_table1.json (Table I, Fig. 2). Families without a projection
  run (Schaefer (i) and PCHIP (i) with their published fixed peaks in 3-35 Nm; the ProMP with
  K = 4, 12-24) are not drawn: they carry no PsyPref result. Their best-profile levels in 3-35 Nm are written to fig3p_text.tex for the caption and the Results text, with the
  paired best-profile level differences to Schaefer (ii) as fam_report.md prints them.
  Paired differences at k = 24 (projection minus CMA-ES on Schaefer's family, minus PsyPref on the
  ProMP): point estimates recomputed here, intervals as fam_report.md prints them.
  Axis: one scale (pt per point) for both panels, 45-100 %; the script stops if any drawn value or
  interval end leaves its panel.

Self-checks (the script stops if one fails):
  * icorr_amp_fam.report rerun on copies of the four row files in a temporary directory
    reproduces fam_report.md character for character and fam_numbers.json's numbers and verdicts
    exactly;
  * every recomputed estimate equals fam_numbers.json's (1e-12), and the ProMP row and CMA-ES equal
    amp_table1.json's estimates (1e-12) and Fig. 2's data at k = 24 (fig2_curves_b/c.csv,
    fig2_panels.csv's best-profile levels; 1e-6 points);
  * every recomputed paired difference equals fam_report.md's printed value (0.1 point), the
    paired best-profile level differences of Schaefer (ii) minus Schaefer (i) and minus PCHIP (i) in 3-35 Nm
    too; PFM1's and PFM5's thresholds are read from fam_report.md's prediction lines;
  * every number in fig3p_text.tex is formatted from the checked values (no typed number);
  * the rows' users are the same 200 in every cell, and the paired cells share user, fold, d0;
  * the example session as above.

Outputs (figures/data/, every file prefixed fig3p_):
  fig3p_curves.csv     phase (% gait cycle); promp1, promp2 (tau(z1), tau(z2), Nm); shown1,
                       shown2 (tau_F(theta1), tau_F(theta2)); back1, back2 (tau(z~1), tau(z~2));
                       preferred (the user's preferred profile); delivered (after 24 comparisons)
  fig3p_nodes.csv      profile (1, 2, 3 = delivered), node (1-4), t (% gait cycle), v (Nm)
  fig3p_rows.csv       (b): one row per drawn row, top to bottom: key, parameter counts (np_w,
                       np_f), per setting (suffix _w 3-35 Nm, _f fixed 12 Nm) the best-profile level and
                       PsyPref (%, with lo/hi)
  fig3p_meta.csv       axis ranges (xmin_w, xmin_f, xmax), k, n, s, lapse, CMA-ES (% and
                       interval) per setting, the printed paired differences, the example's
                       user, setting, k values, its glyph limits, thetas, fit gaps and distances
  fig3p_text.tex       TeX def macros (names starting "pj") for every number in the caption and
                       the draft Results paragraph
  fig3p_diffs.csv      every paired difference at k = 24 (points): recomputed estimate, the
                       report's printed estimate and interval, and the verdicts (text)
  fig3p_check.csv      k, the rerun's distance, FAM's stored distance, equal (1/0)
  fig3p_provenance.csv item, source
"""

from __future__ import annotations

import os

if __name__ == "__main__":
    # as FAM's workers ran: one BLAS thread, so the session reproduces the stored floats
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import icorr_amp_fam as FAM  # noqa: E402
import icorr_amp_full as FULL  # noqa: E402
import paper_v2_fig1 as FIG1  # noqa: E402
from prompref.experiment import amp_free as AF  # noqa: E402
from prompref.user.simulated_user import rms_distance  # noqa: E402

DEFAULT_CSV = ROOT / "figures" / "data"
SETTINGS = (("w", "wide"), ("f", "fixed 12"))
#: (b)'s rows, top to bottom: key, family per setting
ROWS = (("schaefer", {"wide": "schaefer_ii", "fixed 12": "schaefer_i"}),
        ("franks", {"wide": "franks", "fixed 12": "franks"}),
        ("pchip", {"wide": "pchip_ii", "fixed 12": "pchip"}),
        ("promp", {"wide": "promp8", "fixed 12": "promp8"}))
CMA = "cmaes_sch4"           # CMA-ES on Schaefer's family, planned configuration (16a)
K = 24
N_USERS = 200
#: (b)'s axis, %: one scale, 45-100 in both panels
XMIN = {"wide": 45.0, "fixed 12": 45.0}
XMAX = 100.0
PUBLISHED = ("schaefer_i", "pchip_i")   # 3-35 Nm, published fixed peaks: best-profile levels only (PFM1)
EX_SETTING, EX_FAMILY = "wide", "schaefer_ii"
K_PAIR_MIN, K_ID = 6, 24     # the pair: asked after at least six answers (rule below); delivered after 24
GLYPH_PAD = 0.06             # the glyphs' y limits: the drawn curves' range +- 6 %
REPORT_NAMES = {"psypref": "PsyPref", CMA: "CMA-ES (Schäfer, tuned)"}


# ---------------------------------------------------------------- helpers
def num(v) -> str:
    if isinstance(v, str):
        return v
    v = float(v)
    if v != v:
        return "nan"
    return f"{v:.10g}"


def write_csv(path: Path, header: list[str], rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([num(v) for v in r])
    return path


def fail(msg: str) -> None:
    raise SystemExit(f"SELF-CHECK FAILED: {msg}")


def same(a: float, b: float, tol: float, what: str) -> None:
    if not abs(float(a) - float(b)) <= tol:
        fail(f"{what}: {a!r} against {b!r} (tolerance {tol})")


def load_rows(path: Path, key) -> dict:
    """As icorr_amp_fam._load: rows without error, keyed, the last one winning."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("error") is None:
            rows[key(r)] = r
    return rows


def pref(d0, dk) -> np.ndarray:
    return AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((np.asarray(d0) - np.asarray(dk)) / AF.S)


def one(v: float) -> str:
    """One decimal, as the reports print (Python's correct rounding of the value)."""
    return f"{float(v):.1f}"


def mag(txt: str) -> str:
    """A printed difference without its sign ("-6.1" -> "6.1")."""
    return txt.lstrip("+-")


def tex_num(txt: str) -> str:
    """A printed signed difference for running text: "+18.8" -> "18.8", "-6.1" -> "$-6.1$"."""
    return f"${txt}$" if txt.startswith("-") else txt.lstrip("+")


def tex_ci(lo: str, hi: str) -> str:
    return f"[{tex_num(lo)}, {tex_num(hi)}]"


# ---------------------------------------------------------------- the planned report, rerun
def replay_report(d: Path) -> tuple[str, dict]:
    """icorr_amp_fam.report on copies of its four row files in a temporary directory; returns
    the regenerated fam_report.md and fam_numbers.json."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = FULL._dir(Path(tmp))
        dest.mkdir(parents=True)
        for name in ("fam_floors.jsonl", "fam_sessions.jsonl", "full_sessions.jsonl",
                     "robust_sessions.jsonl"):
            shutil.copyfile(d / name, dest / name)
        with contextlib.redirect_stdout(io.StringIO()):
            FAM.report(Path(tmp))
        md = (dest / "fam_report.md").read_text(encoding="utf-8")
        nums = json.loads((dest / "fam_numbers.json").read_text(encoding="utf-8"))
    return md, nums


# ---------------------------------------------------------------- the recorded session
class InnerRec:
    """Passes every call to the lens's inner learner (AM behind the two-part guard) and records
    its proposals (ProMP parameters), what it is told (the projections) and its best."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.proposals: list[tuple[np.ndarray, np.ndarray]] = []
        self.updates: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.bests: list[np.ndarray] = []

    def propose(self):
        a, b = self._inner.propose()
        self.proposals.append((np.array(a, dtype=float), np.array(b, dtype=float)))
        return a, b

    def update(self, theta_a, theta_b, response: int) -> None:
        self.updates.append((np.array(theta_a, dtype=float), np.array(theta_b, dtype=float),
                             int(response)))
        self._inner.update(theta_a, theta_b, response)

    def best(self):
        theta = self._inner.best()
        self.bests.append(np.array(theta, dtype=float))
        return theta

    def __getattr__(self, name):
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class LensRec:
    """Passes every call to the FamilyLens and records the pairs shown (family parameters), the
    answers and the delivered parameters after k answers (run_session's own best calls)."""

    def __init__(self, lens) -> None:
        self._lens = lens
        self.k = 0
        self.pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.answers: list[int] = []
        self.delivered: dict[int, np.ndarray] = {}

    def propose(self):
        a, b = self._lens.propose()
        self.pairs.append((np.array(a, dtype=float), np.array(b, dtype=float)))
        return a, b

    def update(self, theta_a, theta_b, response: int) -> None:
        self._lens.update(theta_a, theta_b, response)
        self.answers.append(int(response))
        self.k += 1

    def best(self):
        theta = self._lens.best()
        self.delivered.setdefault(self.k, np.array(theta, dtype=float))
        return theta

    def __getattr__(self, name):
        if name == "_lens":
            raise AttributeError(name)
        return getattr(self._lens, name)


def run_example(out: Path, user: str) -> tuple[dict, dict]:
    job = [j for j in FAM.proj_jobs(out) if (j["user"], j["setting"], j["candidate"])
           == (user, EX_SETTING, f"proj_{EX_FAMILY}")]
    if len(job) != 1:
        fail(f"{len(job)} FAM jobs for {user} {EX_SETTING} proj_{EX_FAMILY}")
    captured: dict = {}
    original = FAM.run_session

    def recording(design, learner, observer, rng, **kw):
        if not isinstance(learner, AF.FamilyLens):
            fail(f"the session's learner is {type(learner).__name__}, not AF.FamilyLens")
        inner = InnerRec(learner.inner)
        learner.inner = inner
        rec = LensRec(learner)
        captured.update(rec=rec, inner=inner, lens=learner, family=design, observer=observer)
        return original(design, rec, observer, rng, **kw)

    FAM.run_session = recording
    try:
        row = FAM.run_one(dict(job[0]), str(out))
    finally:
        FAM.run_session = original
    if row.get("error"):
        fail(f"the example session failed: {row['error']}")
    return row, captured


def schaefer_nodes(theta: np.ndarray) -> list[tuple[float, float]]:
    """Schaefer et al.'s family (ii): zero crossing t1, extension peak t2 (-ext), zero crossing
    t3, flexion peak t4 (+flex); times mod 100 (% gait cycle)."""
    t1, t2, t3, t4, ext, flex = (float(x) for x in theta)
    return [(t1 % 100.0, 0.0), (t2 % 100.0, -ext), (t3 % 100.0, 0.0), (t4 % 100.0, flex)]


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    d = FULL._dir(args.out)
    dest = args.csv
    if not dest.is_dir():
        fail(f"no data folder {dest}")
    checks: list[str] = []

    # ---- (b) 1. the planned report regenerated from the rows
    md_stored = (d / "fam_report.md").read_text(encoding="utf-8")
    nums = json.loads((d / "fam_numbers.json").read_text(encoding="utf-8"))
    md_new, nums_new = replay_report(d)
    if md_new != md_stored:
        fail("icorr_amp_fam.report() on the stored rows does not reproduce fam_report.md")
    if nums_new["settings"] != nums["settings"] or nums_new["verdicts"] != nums["verdicts"]:
        fail("icorr_amp_fam.report() on the stored rows does not reproduce fam_numbers.json")
    checks.append("fam_report.md and fam_numbers.json regenerated from the rows, identical")

    # ---- (b) 2. the rows
    floors = load_rows(d / "fam_floors.jsonl", lambda r: (r["user"], r["setting"], r["family"]))
    fam = load_rows(d / "fam_sessions.jsonl", FULL._key)
    full = load_rows(d / "full_sessions.jsonl", FULL._key)
    table = json.loads((d / "amp_table1.json").read_text(encoding="utf-8"))
    if (table["s_nm"], table["lapse"]) != (AF.S, AF.LAPSE):
        fail("amp_table1.json's s or lapse differ from the answer model's")

    def cell_floor(setting, family):
        rs = sorted((r for (u, s, f), r in floors.items() if s == setting and f == family),
                    key=lambda r: r["user"])
        return rs

    def cell_sess(setting, cand):
        src = fam if cand.startswith("proj_") else full
        return sorted((r for (u, s, c), r in src.items() if s == setting and c == cand),
                      key=lambda r: r["user"])

    users_ref = None
    est: dict = {}
    cells: dict = {}
    for _, setting in SETTINGS:
        for key, fams in ROWS:
            family = fams[setting]
            fl = cell_floor(setting, family)
            ps = cell_sess(setting, "psypref" if family == "promp8" else f"proj_{family}")
            for name, rs in (("floors", fl), ("sessions", ps)):
                us = [r["user"] for r in rs]
                if users_ref is None:
                    users_ref = us
                if us != users_ref or len(us) != N_USERS:
                    fail(f"{setting} {family} {name}: not the same {N_USERS} users")
            if [r["d0"] for r in fl] != [r["d0"] for r in ps]:
                fail(f"{setting} {family}: floors and sessions disagree on d0")
            cells[setting, key] = (fl, ps)
            ceil = float(np.mean(pref([r["d0"] for r in fl], [r["floor"] for r in fl])))
            psy = float(np.mean(pref([r["d0"] for r in ps], [r["distances"][K - 1] for r in ps])))
            e = nums["settings"][setting][family]
            same(ceil, e["ceiling"]["est"], 1e-12, f"{setting} {family} ceiling vs fam_numbers")
            same(psy, e[f"psypref_k{K}"]["est"], 1e-12, f"{setting} {family} PsyPref vs fam_numbers")
            if FAM.N_PARAMS[setting][family] != e["n_params"]:
                fail(f"{setting} {family}: parameter count")
            est[setting, key] = dict(
                ceil=100 * ceil, ceil_lo=100 * e["ceiling"]["lo"], ceil_hi=100 * e["ceiling"]["hi"],
                psy=100 * psy, psy_lo=100 * e[f"psypref_k{K}"]["lo"],
                psy_hi=100 * e[f"psypref_k{K}"]["hi"], n_params=e["n_params"])
        cm = cell_sess(setting, CMA)
        if [r["user"] for r in cm] != users_ref:
            fail(f"{setting} {CMA}: not the same users")
        cma = float(np.mean(pref([r["d0"] for r in cm], [r["distances"][K - 1] for r in cm])))
        cells[setting, "cma"] = cm
        rows_t = table["settings"][setting]["rows"]
        same(cma, rows_t[CMA]["k"][str(K)]["est"], 1e-12, f"{setting} CMA-ES vs amp_table1.json")
        t_psy = rows_t["psypref"]["k"][str(K)]
        same(est[setting, "promp"]["psy"], 100 * t_psy["est"], 1e-10, f"{setting} PsyPref vs amp_table1.json")
        same(est[setting, "promp"]["ceil"], 100 * rows_t["ceiling"]["k"][str(K)]["est"], 1e-10,
             f"{setting} ceiling vs amp_table1.json")
        # the ProMP row and CMA-ES carry Table I's (and Fig. 2's) intervals
        est[setting, "promp"].update(psy_lo=100 * t_psy["lo"], psy_hi=100 * t_psy["hi"])
        t_cma = rows_t[CMA]["k"][str(K)]
        est[setting, "cma"] = dict(est=100 * cma, lo=100 * t_cma["lo"], hi=100 * t_cma["hi"])
        # Fig. 2's data at k = 24
        panel, letter = {"wide": (1, "b"), "fixed 12": (2, "c")}[setting]
        with open(dest / f"fig2_curves_{letter}.csv", encoding="utf-8") as fh:
            k24 = next(r for r in csv.DictReader(fh) if int(float(r["k"])) == K)
        same(est[setting, "promp"]["psy"], float(k24["psypref"]), 1e-6, f"{setting} PsyPref vs Fig. 2")
        same(est[setting, "cma"]["est"], float(k24["sch"]), 1e-6, f"{setting} CMA-ES vs Fig. 2")
        with open(dest / "fig2_panels.csv", encoding="utf-8") as fh:
            prow = [r for r in csv.DictReader(fh)][panel]
        if int(prow["sch_code"]) != 2:
            fail(f"Fig. 2's panel {panel} does not draw the tuned CMA-ES on Schaefer's family")
        same(est[setting, "promp"]["ceil"], float(prow["ceiling"]), 1e-6, f"{setting} ceiling vs Fig. 2")
    checks.append("estimates recomputed from the rows equal fam_numbers.json, amp_table1.json "
                  "and Fig. 2's data")

    # ---- (b) 2a. the published fixed-peak families in 3-35 Nm: best-profile levels only (PFM1)
    pub = {}
    for family in PUBLISHED:
        fl = cell_floor("wide", family)
        if [r["user"] for r in fl] != users_ref:
            fail(f"wide {family} floors: not the same {N_USERS} users")
        if [r["d0"] for r in fl] != [r["d0"] for r in cells["wide", "schaefer"][0]]:
            fail(f"wide {family}: floors disagree with Schaefer (ii)'s on d0")
        ceil = float(np.mean(pref([r["d0"] for r in fl], [r["floor"] for r in fl])))
        e = nums["settings"]["wide"][family]
        same(ceil, e["ceiling"]["est"], 1e-12, f"wide {family} ceiling vs fam_numbers")
        if FAM.N_PARAMS["wide"][family] != e["n_params"]:
            fail(f"wide {family}: parameter count")
        pub[family] = dict(ceil=100 * ceil, lo=100 * e["ceiling"]["lo"], hi=100 * e["ceiling"]["hi"],
                           n_params=e["n_params"], rows=fl)
    checks.append("the published fixed-peak ceilings in 3-35 Nm recomputed from the floors equal "
                  "fam_numbers.json")

    # ---- (b) 3. paired differences, k = 24: recomputed estimates against the report's print
    pat = re.compile(r"^- (.+?) − (PsyPref|CMA-ES \(Schäfer, tuned\)): ([+-]\d+\.\d) "
                     r"\[([+-]\d+\.\d), ([+-]\d+\.\d)\]$")
    printed: dict = {}
    section = None
    for line in md_stored.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        m = pat.match(line)
        if m:
            printed[section, m.group(1), m.group(2)] = tuple(m.group(i) for i in (3, 4, 5))
    diffs = []
    for _, setting in SETTINGS:
        for key, fams in ROWS[:3]:
            family = fams[setting]
            ps = cells[setting, key][1]
            for other, rs in (("psypref", cells[setting, "promp"][1]), (CMA, cells[setting, "cma"])):
                for a, b in zip(ps, rs):
                    if (a["user"], a["fold"], a["d0"], a["answer_seed"]) != \
                            (b["user"], b["fold"], b["d0"], b["answer_seed"]):
                        fail(f"{setting} {family} vs {other}: rows not paired")
                diff = 100 * float(np.mean(pref([r["d0"] for r in ps], [r["distances"][K - 1] for r in ps])
                                           - pref([r["d0"] for r in rs], [r["distances"][K - 1] for r in rs])))
                txt = printed.get((setting, FAM.LABEL[family], REPORT_NAMES[other]))
                if txt is None:
                    fail(f"fam_report.md prints no {FAM.LABEL[family]} − {REPORT_NAMES[other]} in {setting}")
                same(diff, float(txt[0]), 0.05 + 1e-9, f"{setting} {family} − {other} vs fam_report.md")
                diffs.append([setting, family, other, diff, *txt])
    checks.append("paired differences recomputed from the rows equal fam_report.md's to 0.1 point")
    verdicts = {m.group(1): m.group(2) for m in
                re.finditer(r"^- (PFM\d) \(.*?\): (held|failed|uncertain)", md_stored, re.M)}
    if set(verdicts) != {f"PFM{i}" for i in range(1, 7)}:
        fail(f"fam_report.md's verdicts: {verdicts}")
    dtext = {(s, f, o): (e, lo, hi) for s, f, o, _, e, lo, hi in diffs}

    # ---- (b) 3a. paired best-profile level differences in 3-35 Nm (fam_report.md's "Best-profile level differences")
    cpat = re.compile(r"^- \('(\w+)', '(\w+)'\): ([+-]\d+\.\d) \[([+-]\d+\.\d), ([+-]\d+\.\d)\]$")
    cprinted = {}
    for line in md_stored.split("## Ceiling differences", 1)[-1].splitlines():
        m = cpat.match(line)
        if m:
            cprinted[m.group(1), m.group(2)] = tuple(m.group(i) for i in (3, 4, 5))
    wide_floor = {"schaefer_ii": cells["wide", "schaefer"][0], **{f: pub[f]["rows"] for f in PUBLISHED}}
    cdiff = {}
    for a, b in (("schaefer_ii", "schaefer_i"), ("schaefer_ii", "pchip_i")):
        ra, rb = wide_floor[a], wide_floor[b]
        if [(r["user"], r["d0"]) for r in ra] != [(r["user"], r["d0"]) for r in rb]:
            fail(f"ceiling difference {a} - {b}: rows not paired")
        x = 100 * float(np.mean(pref([r["d0"] for r in ra], [r["floor"] for r in ra])
                                - pref([r["d0"] for r in rb], [r["floor"] for r in rb])))
        txt = cprinted.get((a, b))
        if txt is None:
            fail(f"fam_report.md prints no ceiling difference ({a}, {b})")
        same(x, float(txt[0]), 0.05 + 1e-9, f"ceiling difference {a} - {b} vs fam_report.md")
        cdiff[a, b] = (x, *txt)
    checks.append("paired ceiling differences in 3-35 Nm recomputed from the floors equal "
                  "fam_report.md's to 0.1 point")
    # the planned thresholds, as fam_report.md states the predictions
    m1 = re.search(r"^- PFM1 \(.*?ceilings < (0\.\d+) and ≥ (\d+) points below", md_stored, re.M)
    m5 = re.search(r"^- PFM5 \(.*?within (\d+) points of PsyPref", md_stored, re.M)
    if not (m1 and m5):
        fail("fam_report.md's PFM1 / PFM5 lines changed")
    pfm1_ceil, pfm1_pts, pfm5_pts = 100 * float(m1.group(1)), int(m1.group(2)), int(m5.group(1))
    if verdicts["PFM1"] != "failed" or verdicts["PFM5"] != "failed":
        fail("the text below states PFM1 and PFM5 as failed")
    for p in ("PFM2", "PFM3", "PFM4", "PFM6"):
        if verdicts[p] != "held":
            fail(f"the text below states {p} as held")
    # PFM5 failed on Schaefer (ii) alone; the text says so
    for fam5, ok in (("schaefer_ii", False), ("franks", True), ("pchip_ii", True)):
        if (float(dtext["wide", fam5, "psypref"][0]) >= -pfm5_pts) != ok:
            fail(f"PFM5's sub-verdict for {fam5} is not as the text states")

    # ---- (a) the example session
    stored_f1, rule = FIG1.choose_user(args.out)
    with open(dest / "fig1_meta.csv", encoding="utf-8") as fh:
        f1_user = next(csv.DictReader(fh))["user"]
    user = stored_f1["user"]
    if user != f1_user:
        fail(f"Fig. 1's rule picks {user}, fig1_meta.csv holds {f1_user}")
    stored = fam[user, EX_SETTING, f"proj_{EX_FAMILY}"]
    row, cap = run_example(args.out, user)
    check = [[k + 1, x, y, int(x == y)] for k, (x, y) in enumerate(zip(row["distances"], stored["distances"]))]
    bad = [k for k, x, y, e in check if not e]
    if len(row["distances"]) != len(stored["distances"]) or bad:
        fail(f"the example's distances differ from FAM's stored row at k = {bad[:10]}")
    for key in ("identified", "d0", "lens_gap_median", "lens_gap_p95"):
        if row.get(key) != stored.get(key):
            fail(f"the example's {key}: {row.get(key)} against FAM's {stored.get(key)}")
    checks.append(f"example session ({user}, 3-35 Nm, projection onto Schaefer (ii)) reproduced "
                  f"FAM's stored row bit for bit ({len(check)} distances, identified, d0, gaps)")
    rec, inner, lens, family = cap["rec"], cap["inner"], cap["lens"], cap["family"]
    observer = cap["observer"]
    if not (len(rec.pairs) == len(inner.proposals) == len(inner.updates) == FULL.N_COMP):
        fail("the recorder missed calls")
    if len(lens.gaps) != 3 * FULL.N_COMP + 1:
        fail(f"the lens fitted {len(lens.gaps)} curves, not 3 per comparison plus the last best()")
    ideal = np.asarray(observer.ideal_nm, dtype=float)
    phase = np.linspace(0.0, 100.0, ideal.size)
    promp = lambda th: np.asarray(lens.inner_design.curve(np.asarray(th, dtype=float)), dtype=float)  # noqa: E731
    fam_c = lambda th: np.asarray(family.curve(np.asarray(th, dtype=float)), dtype=float)  # noqa: E731

    # the pair drawn, by rule: the first one asked after at least K_PAIR_MIN answers whose two fit
    # gaps both lie inside the interquartile range of all the session's fits (lens.gaps, the
    # fits FAM's lens_gap statistics summarize), so that the diagram shows a typical fit
    gaps = np.asarray(lens.gaps, dtype=float)
    q1, q3 = (float(x) for x in np.percentile(gaps, [25, 75]))
    if float(np.median(gaps)) != stored["lens_gap_median"]:
        fail("the session's fit gaps do not give FAM's stored median")
    k_pair = next((k for k in range(K_PAIR_MIN, FULL.N_COMP)
                   if all(q1 <= gaps[3 * k + j] <= q3 for j in (0, 1))), None)
    if k_pair is None:
        fail("no pair satisfies the rule")
    six = (float(gaps[3 * K_PAIR_MIN]), float(gaps[3 * K_PAIR_MIN + 1]))
    # descriptive only: how many of the shown fits touch the family's box bounds (any parameter)
    low, high = np.asarray(family.low, dtype=float), np.asarray(family.high, dtype=float)
    on_bounds = [bool(np.any(np.isclose(th, low, atol=1e-6) | np.isclose(th, high, atol=1e-6)))
                 for pair in rec.pairs for th in pair]
    za, zb = inner.proposals[k_pair]
    ta, tb = rec.pairs[k_pair]
    pa, pb, resp = inner.updates[k_pair]
    promp1, promp2 = promp(za), promp(zb)
    shown1, shown2 = fam_c(ta), fam_c(tb)
    back1, back2 = promp(pa), promp(pb)
    gap1, gap2 = rms_distance(shown1, promp1), rms_distance(shown2, promp2)
    same(gap1, lens.gaps[3 * k_pair], 1e-9, "the fit gap of profile 1")
    same(gap2, lens.gaps[3 * k_pair + 1], 1e-9, "the fit gap of profile 2")
    # what PsyPref is told is the ProMP projection of the curve shown, recomputed
    for th, told in ((ta, pa), (tb, pb)):
        again = lens.to_inner(family.curve(th))
        if not np.array_equal(np.asarray(again, dtype=float), told):
            fail("the recorded projection is not to_inner of the curve shown")
    theta_id = rec.delivered[K_ID]
    delivered = fam_c(theta_id)
    if rms_distance(delivered, ideal) != stored["distances"][K_ID - 1]:
        fail(f"the delivered curve after {K_ID} comparisons is not the scored one")
    # the lens fits three curves per comparison (the pair, then best): the delivered curve's
    # fit gap to PsyPref's identified ProMP curve is the one the lens recorded for that best
    if len(inner.bests) != FULL.N_COMP + 1:
        fail("the recorder missed best() calls")
    same(rms_distance(delivered, promp(inner.bests[K_ID - 1])), lens.gaps[3 * (K_ID - 1) + 2], 1e-9,
         "the delivered curve is the fit of PsyPref's identified ProMP curve")
    if not np.all(np.abs(np.concatenate([shown1, shown2, delivered])) <= 35.0 + 1e-9):
        fail("a shown curve exceeds the device clip")
    nodes = {1: schaefer_nodes(ta), 2: schaefer_nodes(tb), 3: schaefer_nodes(theta_id)}
    drawn = np.concatenate([promp1, promp2, shown1, shown2, back1, back2]
                           + [np.array([v for _, v in nodes[j]]) for j in (1, 2)])
    pad = GLYPH_PAD * float(np.ptp(drawn))
    gmin, gmax = float(drawn.min()) - pad, float(drawn.max()) + pad
    back_gap1, back_gap2 = rms_distance(back1, shown1), rms_distance(back2, shown2)

    # ---- CSVs
    write_csv(dest / "fig3p_curves.csv",
              ["phase", "promp1", "promp2", "shown1", "shown2", "back1", "back2", "preferred", "delivered"],
              np.column_stack([phase, promp1, promp2, shown1, shown2, back1, back2, ideal, delivered]))
    write_csv(dest / "fig3p_nodes.csv", ["profile", "node", "t", "v"],
              [[j, i + 1, t, v] for j in (1, 2, 3) for i, (t, v) in enumerate(nodes[j])])
    write_csv(dest / "fig3p_check.csv", ["k", "distance", "stored", "equal"], check)
    head = ["row", "key", "np_w", "np_f"]
    for sfx, _ in SETTINGS:
        head += [f"{c}_{sfx}" for c in ("ceil", "ceil_lo", "ceil_hi", "psy", "psy_lo", "psy_hi")]
    out_rows = []
    for i, (key, _) in enumerate(ROWS):
        r = [i, key, est["wide", key]["n_params"], est["fixed 12", key]["n_params"]]
        for sfx, setting in SETTINGS:
            e = est[setting, key]
            # everything drawn lies inside its panel: the bar (generic assistance, 50 %, to the
            # best-profile level), PsyPref and its interval
            for what in ("ceil", "psy", "psy_lo", "psy_hi"):
                if not XMIN[setting] <= e[what] <= XMAX:
                    fail(f"{setting} {key} {what} {e[what]:.2f} outside {XMIN[setting]}-{XMAX}")
            r += [e["ceil"], e["ceil_lo"], e["ceil_hi"], e["psy"], e["psy_lo"], e["psy_hi"]]
        out_rows.append(r)
    write_csv(dest / "fig3p_rows.csv", head, out_rows)
    for _, setting in SETTINGS:
        if not XMIN[setting] <= 50.0 <= XMAX:
            fail(f"{setting}: generic assistance (50 %) outside the panel")
        for what in ("est", "lo", "hi"):
            if not XMIN[setting] <= est[setting, "cma"][what] <= XMAX:
                fail(f"{setting}: CMA-ES's {what} leaves the panel")
    gw = dtext["wide", "schaefer_ii", CMA]
    gf = dtext["fixed 12", "schaefer_i", CMA]
    meta = dict(xmin_w=XMIN["wide"], xmin_f=XMIN["fixed 12"], xmax=XMAX, k=K, n_users=N_USERS,
                s_nm=AF.S, lapse=AF.LAPSE,
                cma_w=est["wide", "cma"]["est"], cma_lo_w=est["wide", "cma"]["lo"],
                cma_hi_w=est["wide", "cma"]["hi"], cma_f=est["fixed 12", "cma"]["est"],
                cma_lo_f=est["fixed 12", "cma"]["lo"], cma_hi_f=est["fixed 12", "cma"]["hi"],
                gap_w=gw[0], gap_lo_w=gw[1], gap_hi_w=gw[2], gap_f=gf[0], gap_lo_f=gf[1], gap_hi_f=gf[2],
                ex_user=user, ex_setting=EX_SETTING, ex_family=EX_FAMILY, k_pair=k_pair,
                k_pair_rule_min=K_PAIR_MIN, gap_q1=q1, gap_q3=q3, k_id=K_ID,
                ex_answer=["profile 1", "profile 2", "no preference"][resp],
                gmin=gmin, gmax=gmax, fit_gap1=gap1, fit_gap2=gap2, back_gap1=back_gap1,
                back_gap2=back_gap2, d0=stored["d0"], d_kpair=stored["distances"][k_pair - 1],
                d_kid=stored["distances"][K_ID - 1], amp_ext=stored["amps"][0], amp_flex=stored["amps"][1],
                **{f"theta1_{n}": v for n, v in zip(("t1", "t2", "t3", "t4", "ext", "flex"), ta)},
                **{f"theta2_{n}": v for n, v in zip(("t1", "t2", "t3", "t4", "ext", "flex"), tb)},
                **{f"thetaid_{n}": v for n, v in zip(("t1", "t2", "t3", "t4", "ext", "flex"), theta_id)})
    write_csv(dest / "fig3p_meta.csv", list(meta), [list(meta.values())])
    vtxt = "; ".join(f"{k} {v}" for k, v in sorted(verdicts.items()))
    write_csv(dest / "fig3p_diffs.csv",
              ["setting", "family", "minus", "recomputed", "report_est", "report_lo", "report_hi"],
              [[s, f, o, x, e, lo, hi] for s, f, o, x, e, lo, hi in diffs]
              + [["verdicts", vtxt, "", "nan", "", "", ""]])
    # ---- fig3p_text.tex: every number of the caption and the draft Results paragraph
    def ci(e):          # an estimate's interval, %, one decimal
        return f"[{one(e['lo'])}, {one(e['hi'])}]"

    def fixed_cost(fam):
        return dtext["fixed 12", fam, "psypref"][0]

    fix_costs = sorted((fixed_cost(f), f) for f in ("schaefer_i", "franks", "pchip"))
    fix_mags = sorted(float(mag(c)) for c, _ in fix_costs)
    for c, f in fix_costs:
        if not c.startswith("-"):
            fail(f"fixed 12: the projection onto {f} is not below PsyPref on the ProMP")
    e_s = est["wide", "schaefer"]
    # the statements the caption and the text make in words, checked
    for s, f in (("wide", "schaefer_ii"), ("fixed 12", "schaefer_i")):
        if float(dtext[s, f, CMA][1]) <= 0:
            fail(f"{s}: PsyPref with projection is not ahead of CMA-ES (paired interval)")
    # PFM1 as the text words it: both published best-profile levels above the threshold, PCHIP (i) less and
    # Schaefer (i) at least the predicted points below Schaefer (ii)
    if not (pub["schaefer_i"]["ceil"] > pfm1_ceil and pub["pchip_i"]["ceil"] > pfm1_ceil):
        fail("the text says both published fixed-peak ceilings lie above PFM1's threshold")
    if not (float(cdiff["schaefer_ii", "pchip_i"][1]) < pfm1_pts <= float(cdiff["schaefer_ii", "schaefer_i"][1])):
        fail("the text says PCHIP (i), not Schaefer (i), lies less than PFM1's points below")
    folds = {r["fold"] for r in cells["wide", "schaefer"][1]}
    # Franks' and PCHIP's published timing ranges (post hoc; scripts/provenance/family_readings.py,
    # rows in out/icorr/amp_free/family_checks/); its flat-onset reading must equal the fixed code
    readings = [json.loads(x) for x in (d / "family_checks" / "family_readings.jsonl").read_text(encoding="utf-8").splitlines()]

    def reading_ceiling(variant, setting):
        rs = [r for r in readings if r["variant"] == variant and r["setting"] == setting and not r.get("error")]
        if len(rs) != N_USERS:
            fail(f"family_readings: {len(rs)} rows of {variant} in {setting}")
        return 100 * float(np.mean(pref([r["d0"] for r in rs], [r["floor"] for r in rs])))

    for r in readings:
        if r["variant"] == "franks_flat" and abs(r["floor"] - floors[r["user"], r["setting"], "franks"]["floor"]) > 1e-9:
            fail(f"family_readings: the flat onset differs from the fixed Franks floor ({r['user']}, {r['setting']})")
    checks.append("family_readings' flat-onset floors equal the fixed code's stored Franks floors")
    macros = {
        "pjK": str(K), "pjNusers": str(N_USERS), "pjNsubj": str(len(folds)),
        "pjSigma": f"{AF.S:g}", "pjLapse": f"{AF.LAPSE:g}",
        # PsyPref with projection onto Schaefer et al.'s family minus CMA-ES on it, paired
        "pjGapW": tex_num(gw[0]), "pjGapWci": tex_ci(gw[1], gw[2]),
        "pjGapF": tex_num(gf[0]), "pjGapFci": tex_ci(gf[1], gf[2]),
        # Schaefer et al.'s family: best profile, PsyPref with projection, CMA-ES
        "pjSchCeilW": one(e_s["ceil"]), "pjSchCeilF": one(est["fixed 12", "schaefer"]["ceil"]),
        "pjSchPsyW": one(e_s["psy"]), "pjSchPsyF": one(est["fixed 12", "schaefer"]["psy"]),
        "pjCmaW": one(est["wide", "cma"]["est"]), "pjCmaWci": ci(est["wide", "cma"]),
        "pjCmaF": one(est["fixed 12", "cma"]["est"]), "pjCmaFci": ci(est["fixed 12", "cma"]),
        # PFM1: the published fixed peaks in 3-35 Nm
        "pjPubSch": one(pub["schaefer_i"]["ceil"]), "pjPubSchci": ci(pub["schaefer_i"]),
        "pjPubPchip": one(pub["pchip_i"]["ceil"]), "pjPubPchipci": ci(pub["pchip_i"]),
        "pjPubSchDiff": tex_num(cdiff["schaefer_ii", "schaefer_i"][1]),
        "pjPubSchDiffci": tex_ci(*cdiff["schaefer_ii", "schaefer_i"][2:]),
        "pjPubPchipDiff": tex_num(cdiff["schaefer_ii", "pchip_i"][1]),
        "pjPubPchipDiffci": tex_ci(*cdiff["schaefer_ii", "pchip_i"][2:]),
        # projection minus PsyPref on the ProMP, 3-35 Nm
        "pjProjSchW": mag(dtext["wide", "schaefer_ii", "psypref"][0]),
        "pjProjSchWci": tex_ci(*dtext["wide", "schaefer_ii", "psypref"][1:]),
        "pjProjSchWsigned": tex_num(dtext["wide", "schaefer_ii", "psypref"][0]),
        "pjProjFranksW": tex_num(dtext["wide", "franks", "psypref"][0]),
        "pjProjPchipW": tex_num(dtext["wide", "pchip_ii", "psypref"][0]),
        "pjProjFixLo": f"{fix_mags[0]:.1f}", "pjProjFixHi": f"{fix_mags[-1]:.1f}",
        "pjPrompPsyW": one(est["wide", "promp"]["psy"]),
        # best profiles with the published timing ranges
        "pjPubTimeFranksW": one(reading_ceiling("franks_pub", "wide")),
        "pjPubTimeFranksF": one(reading_ceiling("franks_pub", "fixed 12")),
        "pjPubTimePchipW": one(reading_ceiling("pchip_pub", "wide")),
        "pjPubTimePchipF": one(reading_ceiling("pchip_pub", "fixed 12")),
        # the example (a)
        "pjExKpair": str(k_pair),
    }
    text_lines = ["% fig3p_text.tex: written by scripts/paper_v2_fig3p.py (repository); do not edit "
                  "by hand.",
                  "% Every number of Fig. 3's caption and the draft Results paragraph, formatted from "
                  "values the script checked",
                  "% (fam_report.md, fam_numbers.json, amp_table1.json, Fig. 2's data). Differences as "
                  "fam_report.md prints them."]
    text_lines += [f"\\def\\{k}{{{v}}}" for k, v in macros.items()]
    (dest / "fig3p_text.tex").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    prov = [
        ["(b) ceilings and PsyPref with projection, estimates",
         "recomputed from out/icorr/amp_free/fam_floors.jsonl and fam_sessions.jsonl; equal to fam_numbers.json"],
        ["(b) their 95 % intervals", "fam_numbers.json (test plan FAM, report regenerated identically here)"],
        ["(b) ProMP row and CMA-ES on Schaefer's family",
         "recomputed from full_sessions.jsonl (psypref, cmaes_sch4) and fam_floors.jsonl (promp8); "
         "intervals amp_table1.json (Table I); equal to Fig. 2's fig2_curves_b/c.csv and fig2_panels.csv"],
        ["(b) paired differences", "recomputed estimates; printed estimate and interval fam_report.md"],
        ["text: published fixed peaks in 3-35 Nm (PFM1)",
         "ceilings recomputed from fam_floors.jsonl (schaefer_i, pchip_i), equal to fam_numbers.json, "
         "intervals fam_numbers.json; paired ceiling differences recomputed, printed as fam_report.md"],
        ["text: thresholds of PFM1 and PFM5", "fam_report.md's prediction lines"],
        ["fig3p_text.tex", "every number of the caption and the draft Results paragraph, formatted "
                           "from the checked values above"],
        ["(a) example", f"user {user} (Fig. 1's rule), {EX_SETTING}, PsyPref with projection onto "
                        f"Schaefer (ii), rerun with icorr_amp_fam.run_one and checked against "
                        f"fam_sessions.jsonl bit for bit; the pair asked after {k_pair} answers (the "
                        f"first after at least {K_PAIR_MIN} whose two fit gaps lie in the interquartile "
                        f"range of the session's fits); delivered after {K_ID}"],
        ["script", "amp-free scripts/paper_v2_fig3p.py"],
    ]
    write_csv(dest / "fig3p_provenance.csv", ["item", "source"], prov)

    # ---- printout
    print("\n".join("check: " + c for c in checks))
    print(f"\n(b) chosen over generic assistance after {K} comparisons, % (95 % interval)")
    for sfx, setting in SETTINGS:
        print(f"  {setting}: CMA-ES on Schaefer's family {est[setting, 'cma']['est']:.1f} "
              f"[{est[setting, 'cma']['lo']:.1f}, {est[setting, 'cma']['hi']:.1f}]")
        for key, fams in ROWS:
            e = est[setting, key]
            print(f"    {key:10s} ({e['n_params']}): ceiling {e['ceil']:.1f}, PsyPref {e['psy']:.1f} "
                  f"[{e['psy_lo']:.1f}, {e['psy_hi']:.1f}]")
    print("\npaired differences at k = 24, points (recomputed; report: estimate [interval])")
    for s, f, o, x, e, lo, hi in diffs:
        print(f"  {s:8s} {f:12s} − {o:10s} {x:+.2f}  report {e} [{lo}, {hi}]")
    print("verdicts:", vtxt)
    print(f"\n(a) user {user} (Fig. 1's: rank {rule['rank']} of {rule['n']}), {EX_SETTING}, "
          f"projection onto {EX_FAMILY}; preferred peaks {stored['amps'][0]:.2f} / "
          f"{stored['amps'][1]:.2f} Nm; d0 {stored['d0']:.3f}, d{k_pair} "
          f"{stored['distances'][k_pair - 1]:.3f}, "
          f"d{K_ID} {stored['distances'][K_ID - 1]:.3f} Nm")
    print(f"  pair after {k_pair} answers: fit gaps {gap1:.3f} / {gap2:.3f} Nm, projection gaps "
          f"{back_gap1:.3f} / {back_gap2:.3f} Nm; answer {meta['ex_answer']}")
    print(f"  (the pair after {K_PAIR_MIN} answers: fit gaps {six[0]:.2f} / {six[1]:.2f} Nm; session "
          f"interquartile range {q1:.2f}-{q3:.2f} Nm; shown fits touching a box bound "
          f"{sum(on_bounds)} of {len(on_bounds)}, the drawn pair's {on_bounds[2 * k_pair]} / "
          f"{on_bounds[2 * k_pair + 1]})")
    print(f"PFM1 (3-35 Nm, published fixed peaks): Schaefer (i) ceiling {pub['schaefer_i']['ceil']:.1f}, "
          f"PCHIP (i) {pub['pchip_i']['ceil']:.1f} %; Schaefer (ii) minus them "
          f"{cdiff['schaefer_ii', 'schaefer_i'][1]} / {cdiff['schaefer_ii', 'pchip_i'][1]} points "
          f"(predicted: below {pfm1_ceil:.0f} % and at least {pfm1_pts} points below)")
    print(f"wrote {dest / 'fig3p_text.tex'} ({len(macros)} macros)")
    print("  theta1", np.round(ta, 2), "theta2", np.round(tb, 2), "delivered", np.round(theta_id, 2))
    print(f"  glyph limits {gmin:.2f} to {gmax:.2f} Nm")
    print(f"wrote {dest} (fig3p_*.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
