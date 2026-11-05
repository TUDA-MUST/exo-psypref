"""Post hoc, for the paper's outline: how close each optimizer comes to the preferred profile, by k.
Read-only on out/icorr/amp_free. Medians over users; intervals as in the report (users within folds)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R0 = ROOT
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import icorr_amp_ks2 as K

F, KS, R, AF = K.F, K.KS, K.R, K.AF
d = F._dir(R0 / "out")
JND = 0.6744897501960817


def read(name):
    p = d / name
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


full, cmp_ = read("full_sessions.jsonl"), read("cmp_sessions.jsonl")
print("full keys", sorted(full[0].keys())[:30])
print("full candidates", sorted({r["candidate"] for r in full}), "settings", sorted({r["setting"] for r in full}))
print("cmp candidates", sorted({r["candidate"] for r in cmp_}))

rows = {}
for r in full + cmp_:
    if r.get("error") is None:
        rows.setdefault((r["setting"], r["candidate"]), []).append(r)

order = ["psypref", "gp_bald", "cmaes4", "g_generic_nogate", "g_generic", "g", "cmaes", "gp_eubo", "d_z", "ramella", "cmaes_sch4", "cmaes_sch"]
t1 = json.loads((d / "amp_table1.json").read_text(encoding="utf-8"))
rng = np.random.default_rng(20261104)
for s in ("medium", "wide", "fixed 12"):
    sub = {c: rows[s, c] for c in order if (s, c) in rows}
    any_rows = next(iter(sub.values()))
    print(f"\n== {s}: median generic {np.median([r['d0'] for r in sub['psypref']]):.2f} Nm, "
          f"floor {t1['settings'][s]['median_floor_nm']:.2f} Nm; n per optimizer {sorted({len(v) for v in sub.values()})}")
    print("optimizer | d(12) | d(24) | d(40) Nm | d(24)/generic % | half 1 JND better: first k | better/worse at 24 %")
    for c, rs in sub.items():
        ds = np.array([r["distances"] for r in rs])
        d0 = np.array([r["d0"] for r in rs])
        med = [np.median(ds[:, k - 1]) for k in (12, 24, 40)]
        boot = R._boot(rs, lambda x: float(np.median([r["distances"][23] for r in x])), rng)
        share = np.mean(d0[:, None] - ds >= JND, axis=0)
        fk = next((i + 1 for i, v in enumerate(share) if v >= 0.5), None)
        print(f"{c} | {med[0]:.2f} | {med[1]:.2f} [{boot[1]:.2f}, {boot[2]:.2f}] | {med[2]:.2f} | {100 * med[1] / np.median(d0):.0f} | {fk} | "
              f"{100 * share[23]:.0f} / {100 * np.mean(ds[:, 23] - d0 >= JND):.0f}")

# The 100 users of the weight sweep: the 4-6-parameter family and P_g by K.
ks2 = {(r["candidate"], r["n_basis"], r["user"], r["setting"]): r for r in read("ks2_sessions.jsonl")}
users = {(u, s) for (_, _, u, s) in ks2}
print("\nCMA-ES 4-6 param on the sweep's 100 users, P_g:")
for s in ("wide", "fixed 12"):
    rs = [r for r in rows[s, "cmaes_sch4"] if (r["user"], s) in users]
    print(s, len(rs), {k: round(100 * F._pref_k(rs, k), 1) for k in (12, 24, 40)})

# Long sessions.
for name in ("long_sessions.jsonl",):
    long_rows = read(name)
    print(f"\n{name}: {len(long_rows)} rows", sorted(long_rows[0].keys())[:25] if long_rows else "")
    if long_rows:
        groups = {}
        for r in long_rows:
            if r.get("error") is None:
                groups.setdefault((r.get("setting"), r.get("candidate"), r.get("n_basis"), r.get("s", r.get("noise"))), []).append(r)
        for key, rs in sorted(groups.items(), key=str):
            lens = {len(r["distances"]) for r in rs}
            if max(lens) >= 150:
                print(key, len(rs), {k: round(float(np.median([r["distances"][k - 1] for r in rs])), 2) for k in (12, 24, 40, 80, 150) if k <= min(lens)})
