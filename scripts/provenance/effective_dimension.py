"""Post hoc, read-only: how many dimensions do the users need, and what does the same measure say
for the published family? No sessions are run, nothing under out/ is written."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
R0 = ROOT
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import icorr_amp_fam as FAM
import icorr_posthoc_dims as DIMS

F, R, AF = FAM.F, FAM.R, FAM.AF
out = R0 / "out"
d = F._dir(out)
JND = 0.6744897501960817
LAPSE, S = AF.LAPSE, AF.S
SETTINGS = ("wide", "fixed 12")
run_dir = out / "phase5-schaefer"


def pref(d0, dk):
    return 100 * float(np.mean(LAPSE / 2 + (1 - LAPSE) * norm.cdf((np.asarray(d0) - np.asarray(dk)) / S)))


# A. Stored floors per family
print("A. Best profile per family (stored floors, n = 200): median Nm, median JND, P_g at the floor")
rows = [json.loads(x) for x in (d / "fam_floors.jsonl").read_text(encoding="utf-8").splitlines()]
for s in SETTINGS:
    fams = sorted({r["family"] for r in rows if r["setting"] == s and r.get("error") is None})
    for fam in fams:
        rs = [r for r in rows if r["setting"] == s and r["family"] == fam and r.get("error") is None]
        fl = np.array([r["floor"] for r in rs])
        print(f"  {s:9s} {fam:12s} n={len(rs)} {np.median(fl):5.2f} Nm {np.median(fl) / JND:5.2f} JND  P_g {pref([r['d0'] for r in rs], fl):5.1f}")

# B. Floors in the prior's top-d directions
print("\nB. Best profile in the prior's d leading directions (unconstrained least squares): median Nm, JND, P_g")
jobs = FAM._base_jobs(out)
checks = []
cache = {}
for K in (8, 24):
    for s in SETTINGS:
        res = {}
        d0s = []
        for j in jobs:
            if j["setting"] != s:
                continue
            ideal, generic, _ = FAM._user(j)
            key = (j["fold_index"], s, K)
            if key not in cache:
                pri = DIMS._prior(d, j["fold_index"], j["fold"], s, K, checks)
                _, space, sigma, basis = pri
                mu = np.asarray(space.whitening.mu, float)
                vals, vecs = np.linalg.eigh(np.asarray(sigma, float))
                cache[key] = (mu, vecs[:, np.argsort(vals)[::-1]], np.asarray(basis.design, float))
            mu, vecs, phi = cache[key]
            scale = 1.0 if s != "fixed 12" else float(F.LEVEL)
            target = np.asarray(ideal, float) / scale
            d0s.append(float(np.sqrt(np.mean((np.asarray(generic) - np.asarray(ideal)) ** 2))))
            for m in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24):
                if m > K:
                    continue
                A = phi @ vecs[:, :m]
                a, *_ = np.linalg.lstsq(A, target - phi @ mu, rcond=None)
                res.setdefault(m, []).append(scale * float(np.sqrt(np.mean((A @ a + phi @ mu - target) ** 2))))
        for m, v in res.items():
            print(f"  K={K:2d} {s:9s} d={m:2d}: {np.median(v):5.2f} Nm {np.median(v) / JND:5.2f} JND  P_g {pref(d0s, v):5.1f}  (n={len(v)})")

# C. The same measure for ProMP and Schaefer's family: curves as the design shows them
print("\nC. Spectrum of the curves the prior generates, as shown (N(0, I) in z, 4000 draws per fold, median over folds)")


def spectrum(space, seed=20261104):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((4000, space.d))
    curves = np.array([space.design.curve(space.to_theta(x)) for x in z])
    ev = np.linalg.eigvalsh(np.cov(curves, rowvar=False))[::-1]
    ev = ev[ev > 0]
    share = np.cumsum(ev) / ev.sum()
    return float(ev.sum() ** 2 / (ev ** 2).sum()), int(np.searchsorted(share, 0.90) + 1), int(np.searchsorted(share, 0.95) + 1), int(np.searchsorted(share, 0.99) + 1), share[:6]


folds = sorted({(j["fold_index"], j["fold"]) for j in jobs})
for s in SETTINGS:
    for name in ("ProMP 8", "ProMP 12", "ProMP 16", "ProMP 24", "Schaefer"):
        out_rows = []
        for fi, fold in folds:
            if name.startswith("ProMP"):
                K = int(name.split()[1])
                pri = DIMS._prior(d, fi, fold, s, K, checks)
                space = pri[1]
            elif s == "wide":
                space = AF.schaefer_range_space(run_dir, fold, *F.RANGES[s])
            else:
                space = AF.schaefer_fixed_space(run_dir, fold, F.LEVEL)
            out_rows.append(spectrum(space))
        a = np.array([x[:4] for x in out_rows], float)
        sh = np.median([x[4] for x in out_rows], axis=0)
        print(f"  {s:9s} {name:9s} PR {np.median(a[:, 0]):4.2f} ({a[:, 0].min():.2f}-{a[:, 0].max():.2f})  n90 {np.median(a[:, 1]):.0f}  n95 {np.median(a[:, 2]):.0f}  n99 {np.median(a[:, 3]):.0f}  "
              f"cum. share of first 6: {' '.join(f'{100 * v:.0f}' for v in sh)}")

# D. The held-out users' own spread, in curve space
print("\nD. The users' preferred curves (n = 200): their own spectrum")
for s in SETTINGS:
    cs = np.array([FAM._user(j)[0] for j in jobs if j["setting"] == s])
    ev = np.linalg.eigvalsh(np.cov(cs, rowvar=False))[::-1]
    ev = ev[ev > 0]
    share = np.cumsum(ev) / ev.sum()
    print(f"  {s:9s} PR {ev.sum() ** 2 / (ev ** 2).sum():4.2f}  n90 {np.searchsorted(share, 0.9) + 1}  n95 {np.searchsorted(share, 0.95) + 1}  n99 {np.searchsorted(share, 0.99) + 1}  cum. share of first 6: {' '.join(f'{100 * v:.0f}' for v in share[:6])}")
