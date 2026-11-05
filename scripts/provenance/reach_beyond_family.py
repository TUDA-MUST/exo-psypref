"""Post hoc, read-only: can PsyPref get closer to the preference than any curve of a fixed family could?
(a) Share of users whose identified profile is closer than their family's best curve, by K and k.
(b) How much perceivable shape lies outside the family, and how much the search ball cuts off.
(c) A yardstick from rate-distortion: bits needed for a distance, under a Gaussian source."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R0 = ROOT
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import icorr_amp_fam as FAM
import icorr_posthoc_dims as DIMS

F, R, AF = FAM.F, FAM.R, FAM.AF
out = R0 / "out"
d = F._dir(out)
JND = 0.6744897501960817
SETTINGS = ("wide", "fixed 12")
SCH = {"wide": "schaefer_ii", "fixed 12": "schaefer_i"}


def read(name):
    return [json.loads(x) for x in (d / name).read_text(encoding="utf-8").splitlines() if x.strip()]


floors = {}
for r in read("fam_floors.jsonl"):
    if r.get("error") is None:
        floors.setdefault((r["user"], r["setting"]), {})[r["family"]] = r["floor"]

long_rows = [r for r in read("long_sessions.jsonl") if r.get("error") is None]
print("LONG cells:", sorted({str(r.get("cell")) for r in long_rows}), "lengths", sorted({len(r["distances"]) for r in long_rows}))
ks_rows = [r for r in read("ks_sessions.jsonl") if r.get("error") is None]
print("KS:", sorted({(r["candidate"], r["n_basis"], len(r["distances"])) for r in ks_rows}))
main_cell = [r for r in long_rows if r.get("candidate") == "psypref" and r.get("s", None) in (None, 1.0) and "s2" not in str(r.get("cell"))]
print("LONG main rows per setting:", {s: sum(r["setting"] == s for r in main_cell) for s in ("medium", "wide", "fixed 12")}, "keys", [k for k in long_rows[0] if k in ("s", "noise", "lapse")])

curves = {}
for r in main_cell:
    if r["setting"] in SETTINGS:
        curves.setdefault((8, r["setting"]), []).append(r)
for r in ks_rows:
    if r["candidate"] == "psypref" and r["setting"] in SETTINGS:
        curves.setdefault((r["n_basis"], r["setting"]), []).append(r)

print("\n(a) Share of users closer than the family's best curve (Schaefer: 6 param. in the range, 4 at the level), and median distance in JND")
for s in SETTINGS:
    print(f" {s}: family's best, median {np.median([v[SCH[s]] for (u, ss), v in floors.items() if ss == s]) / JND:.2f} JND")
    for K in (8, 12, 16, 24):
        rs = curves.get((K, s), [])
        if not rs:
            continue
        n = min(len(r["distances"]) for r in rs)
        parts = []
        for k in (12, 24, 40, 80, 150):
            if k > n:
                continue
            ds = np.array([r["distances"][k - 1] for r in rs])
            fl = np.array([floors[r["user"], s][SCH[s]] for r in rs])
            best = np.array([min(v for fam, v in floors[r["user"], s].items() if not fam.startswith("promp")) for r in rs])
            parts.append(f"k={k}: {100 * np.mean(ds < fl):3.0f}% (any family {100 * np.mean(ds < best):3.0f}%), {np.median(ds) / JND:.2f} JND")
        print(f"  K={K:2d} n={len(rs):3d} | " + " | ".join(parts))

print("\n(b) Representation: share of the 200 users whose best K-weight curve beats the family's best; shape outside the family")
for s in SETTINGS:
    us = [v for (u, ss), v in floors.items() if ss == s]
    print(f" {s}: Schaefer's best {np.median([v[SCH[s]] for v in us]):.2f} Nm = {np.median([v[SCH[s]] for v in us]) / JND:.2f} JND; users with >= 1 JND outside it: {100 * np.mean([v[SCH[s]] >= JND for v in us]):.0f}%")
    for K in (4, 8, 12, 16, 24):
        print(f"   K={K:2d}: best closer than the family for {100 * np.mean([v[f'promp{K}'] < v[SCH[s]] for v in us]):3.0f}%; by >= 1 JND for {100 * np.mean([v[SCH[s]] - v[f'promp{K}'] >= JND for v in us]):3.0f}%")

print("\n(c) Yardstick: Gaussian rate-distortion of the users around the prior's mean; answers carry at most 1 - H(lapse/2) bits")
lam = AF.LAPSE / 2
cap = 1 + lam * np.log2(lam) + (1 - lam) * np.log2(1 - lam)
print(f" capacity per answer at lapse {AF.LAPSE}: {cap:.3f} bit")
jobs = FAM._base_jobs(out)
checks = []
starts = {}
for s in SETTINGS:
    err = []
    for j in jobs:
        if j["setting"] != s:
            continue
        key = (j["fold_index"], s)
        if key not in starts:
            space = DIMS._prior(d, j["fold_index"], j["fold"], s, 8, checks)[1]
            starts[key] = np.asarray(space.design.curve(space.to_theta(np.zeros(space.d))), float)
        err.append(np.asarray(FAM._user(j)[0], float) - starts[key])
    err = np.array(err)
    ev = np.linalg.eigvalsh(err.T @ err / len(err) / err.shape[1])[::-1]
    ev = ev[ev > 1e-12]

    def bits(D):
        lo, hi = 0.0, ev.max()
        for _ in range(200):
            th = 0.5 * (lo + hi)
            if np.minimum(th, ev).sum() > D:
                hi = th
            else:
                lo = th
        return float(0.5 * np.log2(ev / max(th, 1e-15)).clip(min=0).sum())

    def row(name, rmsd):
        b = bits(rmsd ** 2)
        print(f"   {name:34s} RMS d {rmsd:5.2f} Nm = {rmsd / JND:5.2f} JND: {b:6.1f} bit, at least about {b / cap:6.1f} comparisons")

    print(f" {s}: start at the prior's mean, RMS d {np.sqrt(ev.sum()):.2f} Nm; PR of the users around it {ev.sum() ** 2 / (ev ** 2).sum():.2f}")
    us = [v for (u, ss), v in floors.items() if ss == s]
    row("family's best", float(np.sqrt(np.mean([v[SCH[s]] ** 2 for v in us]))))
    for K in (8, 16, 24):
        row(f"best {K}-weight curve", float(np.sqrt(np.mean([v[f'promp{K}'] ** 2 for v in us]))))
    row("1 JND", JND)
    rs = curves[8, s]
    for k in (12, 24, 40, 80, 150):
        row(f"PsyPref K = 8 reached, k = {k}", float(np.sqrt(np.mean([r["distances"][k - 1] ** 2 for r in rs]))))
