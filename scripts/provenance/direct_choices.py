"""Post hoc, read-only: what a study could observe, as blinded A/B choices between tested profiles.
P(choose X over Y) = lapse/2 + (1 - lapse) Phi((d_Y - d_X) / s), from the stored distances, with the
main condition's answer model; means over users, intervals as in the reports."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
R0 = ROOT
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import icorr_amp_full as F
import icorr_amp_range as R

AF = F.AF
d = F._dir(R0 / "out")
SETTINGS = F.SETTINGS
rng = np.random.default_rng(20261105)


def read(name):
    return [json.loads(x) for x in (d / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def choose(dx, dy):
    return AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((np.asarray(dy) - np.asarray(dx)) / AF.S)


def boot(rows, f):
    vals = [dict(fold=r["fold"], v=float(f(r))) for r in rows]
    t = R._boot(vals, lambda x: 100 * float(np.mean([r["v"] for r in x])), rng)
    return f"{t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]"


full = {(r["candidate"], r["user"], r["setting"]): r for r in read("full_sessions.jsonl") if r.get("error") is None}
cmp_ = {(r["candidate"], r["user"], r["setting"]): r for r in read("cmp_sessions.jsonl") if r.get("error") is None}
long_ = {(r["user"], r["setting"]): r for r in read("long_sessions.jsonl") if r.get("error") is None and r.get("cell") == "main"}
t1 = json.loads((d / "amp_table1.json").read_text(encoding="utf-8"))["settings"]
both = {**full, **cmp_}
floors = {(r["user"], r["setting"]): r["floor"] for r in read("fam_floors.jsonl") if r.get("error") is None and r["family"] == "promp8"}
floors.update({(u, "medium"): v for u, v in t1["medium"]["promp8_floors_nm"].items()})

for st in SETTINGS:
    strongest = t1[st]["best_published"]["candidate"]
    users = sorted(u for (c, u, s) in full if c == "psypref" and s == st)
    rows = [dict(fold=both["psypref", u, st]["fold"], p=both["psypref", u, st], s=both[strongest, u, st],
                 g=both["gp_bald", u, st], c=both["cmaes_sch4", u, st], best=floors[u, st]) for u in users]
    print(f"\n== {st} (n = {len(rows)}), strongest: {strongest}")
    print(" head to head, P(PsyPref's profile over the other's), k = 24:")
    print("   over the strongest's:", boot(rows, lambda r: choose(r["p"]["distances"][23], r["s"]["distances"][23])))
    print("   over GP-BALD's:      ", boot(rows, lambda r: choose(r["p"]["distances"][23], r["g"]["distances"][23])))
    print("   over CMA-ES 4-6:     ", boot(rows, lambda r: choose(r["p"]["distances"][23], r["c"]["distances"][23])))
    print(" within PsyPref, P(the later profile over the earlier):")
    print("   k = 24 over k = 12:  ", boot(rows, lambda r: choose(r["p"]["distances"][23], r["p"]["distances"][11])))
    print("   k = 40 over k = 24:  ", boot(rows, lambda r: choose(r["p"]["distances"][39], r["p"]["distances"][23])))
    lr = [dict(fold=r["fold"], r=r) for k, r in long_.items() if k[1] == st]
    print(f"   k = 150 over k = 24 (n = {len(lr)}):", boot(lr, lambda x: choose(x["r"]["distances"][149], x["r"]["distances"][23])))
    print(" model only, not observable: P(best 8-weight profile over PsyPref's at k = 24):",
          boot(rows, lambda r: choose(r["best"], r["p"]["distances"][23])))
    print(" same for the strongest's profile, k = 40 over k = 24:", boot(rows, lambda r: choose(r["s"]["distances"][39], r["s"]["distances"][23])))
