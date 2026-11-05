"""Post hoc: which widths does PsyPref's acquisition pick in the final sessions? Reruns FULL's PsyPref
for 2 users per fold and setting, records the pairs only, writes nothing under out/."""
import collections
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R0 = ROOT
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
import icorr_amp_full as F

_orig = F.run_session
_CAP = {}


def _rs(*a, **k):
    rec = _orig(*a, **k)
    _CAP.update(kind=rec.get("duel_kind"), trace=rec.get("duel_trace"))
    return rec


F.run_session = _rs


def one(job):
    _CAP.clear()
    row = F.run_one(dict(job), str(R0 / "out"))
    return dict(setting=job["setting"], error=row.get("error"), same=row.get("distances"), kind=_CAP.get("kind"), trace=_CAP.get("trace"))


if __name__ == "__main__":
    jobs = [j for j in F.jobs(R0 / "out") if j["candidate"] == "psypref" and j["index"] < 2]
    with ProcessPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(one, jobs))
    print(len(res), "sessions; errors", sum(1 for r in res if r["error"]))
    print("sample trace entry:", res[0]["trace"][:2])
    for s in F.SETTINGS:
        kinds, widths = collections.Counter(), collections.Counter()
        for r in res:
            if r["setting"] != s:
                continue
            for kind, t in zip(r["kind"], r["trace"]):
                kinds[kind] += 1
                if kind == "acq":
                    m = t.get("multiple", t.get("width")) if isinstance(t, dict) else (t[0] if isinstance(t, (list, tuple)) else t)
                    widths[round(float(m), 2)] += 1
        n = sum(widths.values())
        print(s, dict(kinds), "acquisition widths (share %):", {k: round(100 * v / n, 1) for k, v in sorted(widths.items())})
