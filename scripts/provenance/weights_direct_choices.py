"""Post hoc, read-only: would users choose PsyPref's profile with K weights over its profile with 8?
P = lapse/2 + (1 - lapse) Phi((d_8 - d_K) / s), same user, same k, n = 100 per setting."""
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
rng = np.random.default_rng(20261105)


def read(name):
    return [json.loads(x) for x in (d / name).read_text(encoding="utf-8").splitlines() if x.strip()]


base = {(r["user"], r["setting"]): r for r in read("long_sessions.jsonl") if r.get("error") is None and r.get("cell") == "main"}
ks = {(r["n_basis"], r["user"], r["setting"]): r for r in read("ks_sessions.jsonl")
      if r.get("error") is None and r["candidate"] == "psypref"}
for s in ("wide", "fixed 12"):
    print(f"== {s}")
    for K in (12, 16, 24):
        parts = []
        for k in (12, 24, 40, 80, 150):
            rows = []
            for (kk, u, ss), r in ks.items():
                if kk == K and ss == s and (u, s) in base:
                    v = AF.LAPSE / 2 + (1 - AF.LAPSE) * norm.cdf((base[u, s]["distances"][k - 1] - r["distances"][k - 1]) / AF.S)
                    rows.append(dict(fold=r["fold"], v=float(v)))
            t = R._boot(rows, lambda x: 100 * float(np.mean([y["v"] for y in x])), rng)
            parts.append(f"k={k}: {t[0]:.1f} [{t[1]:.1f}, {t[2]:.1f}]")
        print(f"  K={K} over K=8 (n={len(rows)}): " + " | ".join(parts))
