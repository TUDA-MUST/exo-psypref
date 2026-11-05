"""Does the 'fewer comparisons' claim survive the noise levels? PsyPref's first k at the best level any 8-weight
comparator reaches at k = 40, and PsyPref's distance gain, at s = 1, 2, 3, against every 8-weight comparator."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
AMP = ROOT / "out" / "icorr" / "amp_free"


def load(name, cond=None):
    rows = [json.loads(x) for x in open(AMP / name, encoding="utf-8")]
    return [r for r in rows if r.get("error") is None and (cond is None or r.get("condition") == cond)]


SETS = {1.0: load("full_sessions.jsonl") + [r for r in load("cmp_sessions.jsonl") if r["candidate"] in ("g_generic_nogate", "cmaes4")],
        2.0: load("robust_sessions.jsonl", "R2 s=2.0") + load("var_sessions.jsonl", "R2 s=2.0"),
        3.0: load("s3_sessions.jsonl") + load("var_sessions.jsonl", "R2b s=3.0")}
for s, rows in SETS.items():
    for st in ("wide", "fixed 12"):
        by = {}
        for r in rows:
            if r["setting"] == st:
                by.setdefault(r["candidate"], []).append(r)

        def curve(c):
            d0 = np.array([r["d0"] for r in by[c]])
            d = np.array([r["distances"] for r in by[c]])
            return 100 * np.mean(0.05 + 0.9 * norm.cdf((d0[:, None] - d) / s), axis=0)

        psy = curve("psypref")
        others = {c: curve(c)[39] for c in ("g", "d_z", "cmaes", "g_generic_nogate", "cmaes4") if c in by}
        best = max(others, key=others.get)
        hit = np.flatnonzero(psy >= others[best])
        k = int(hit[0]) + 1 if hit.size else None
        d0 = np.array([r["d0"] for r in by["psypref"]])
        d24 = np.array([r["distances"][23] for r in by["psypref"]])
        print(f"s={s:g} {st:9s} PsyPref k=3 {psy[2]:5.1f} k=12 {psy[11]:5.1f} k=24 {psy[23]:5.1f} k=40 {psy[39]:5.1f}; best 8-weight at k=40: {best} {others[best]:5.1f}; "
              f"PsyPref passes it at k={k}; median d0 {np.median(d0):.2f}, d24 {np.median(d24):.2f}, closer than generic by >= 1 JND: "
              f"{100 * np.mean(d0 - d24 >= 0.6745 * s):.0f} %, worse than generic: {100 * np.mean(d24 > d0):.0f} %")
