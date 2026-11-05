"""Post hoc, read-only: at sigma = 3 Nm, PsyPref's preference over generic assistance after 24 comparisons, the share of
users who would benefit, and the share who would choose generic assistance, from the stored sessions."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
AMP = ROOT / "out" / "icorr" / "amp_free"
sys.stdout.reconfigure(encoding="utf-8")

rows = [json.loads(x) for x in open(AMP / "s3_sessions.jsonl", encoding="utf-8")]
rows = [r for r in rows if r.get("error") is None and r["candidate"] == "psypref"]
for st in ("wide", "fixed 12"):
    sel = [r for r in rows if r["setting"] == st]
    d0 = np.array([r["d0"] for r in sel])
    d24 = np.array([r["distances"][23] for r in sel])
    p = 0.05 + 0.9 * norm.cdf((d0 - d24) / 3.0)
    print(f"{st}: n {len(sel)}, P_g {100 * p.mean():.1f} %, benefit (>= 72.5 %) {100 * np.mean(p >= 0.725):.1f} %, "
          f"generic more often than not (< 50 %) {100 * np.mean(p < 0.5):.1f} %, to the same degree (<= 27.5 %) "
          f"{100 * np.mean(p <= 0.275):.1f} %")
