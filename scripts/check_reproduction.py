"""Rerun a sample of the stored sessions and compare them with the logs, number for number.

Usage:
  python -B scripts/check_reproduction.py [--out <dir>] [--per-cell 1] [--workers 8] [--sweeps full,rob2,...]
         [--seed N] [--skip-candidates gp_bald,gp_eubo,psypref_adaptive] [--audit <file>]

One randomly drawn session per cell (sweep, condition, setting, optimizer) by default. A session counts as reproduced only if
every number of its row equals the stored row exactly, apart from timings. GP-BALD, GP-qEUBO and the adaptive-prior
chains are skipped by default because one run takes minutes or more. ``--audit`` writes the files each session opened below --out.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

#: Sweep -> (module, stored log, the function that runs one job, in that module or in FULL's).
SWEEPS = {
    "full": ("icorr_amp_full", "full_sessions.jsonl"),
    "cmp": ("icorr_amp_cmp", "cmp_sessions.jsonl"),
    "abl": ("icorr_amp_abl", "abl_sessions.jsonl"),
    "rnd": ("icorr_amp_rnd", "rnd_sessions.jsonl"),
    "robust": ("icorr_amp_robust", "robust_sessions.jsonl"),
    "rob2": ("icorr_amp_rob2", "rob2_sessions.jsonl"),
    "s3": ("icorr_amp_s3", "s3_sessions.jsonl"),
    "tie": ("icorr_amp_tie", "tie_sessions.jsonl"),
    "pri": ("icorr_amp_pri", "pri_sessions.jsonl"),
    "fam": ("icorr_amp_fam", "fam_sessions.jsonl"),
    "long": ("icorr_amp_long", "long_sessions.jsonl"),
    "ks": ("icorr_amp_ks", "ks_sessions.jsonl"),
    "ks2": ("icorr_amp_ks2", "ks2_sessions.jsonl"),
    "pw": ("icorr_amp_pw", "pw_sessions.jsonl"),
    "np": ("icorr_amp_np", "np_sessions.jsonl"),
    "var": ("icorr_amp_var", "var_sessions.jsonl"),
}
#: Row fields that may differ between runs.
VOLATILE = ("second", "time", "elapsed", "wall", "cpu", "host", "pid", "stamps")
#: Fields a job and its row share that say which session it is.
IDENT = ("condition", "fold", "user", "setting", "candidate", "n_basis", "variant", "family", "budget", "arm",
         "cohort", "test", "k")

_audit_path = None
_in_hook = False  # writing the audit line opens a file too


def _hook(event, args):
    global _in_hook
    if event == "open" and _audit_path and not _in_hook and isinstance(args[0], (str, os.PathLike)):
        _in_hook = True
        try:
            with open(_audit_path, "a", encoding="utf-8") as f:
                f.write(f"{args[1]}\t{os.fspath(args[0])}\n")
        finally:
            _in_hook = False


def _init(audit):
    global _audit_path
    if audit:
        _audit_path = f"{audit}.{os.getpid()}"
        sys.addaudithook(_hook)


def _runner(module):
    mod = importlib.import_module(module)
    if hasattr(mod, "run_one"):
        return mod.run_one
    return importlib.import_module("icorr_amp_full").run_one


def _one(sweep, job, out):
    module, _ = SWEEPS[sweep]
    try:
        return sweep, job, _runner(module)(job, out), None
    except Exception as e:  # noqa: BLE001 -- reported per session, the check goes on
        return sweep, job, None, f"{type(e).__name__}: {e}"


def _ident(row):
    return tuple((k, row[k]) for k in IDENT if k in row and not isinstance(row[k], (list, dict)))


def _differs(new, old, path=""):
    """The first difference between two rows, or None. Exact on numbers."""
    if isinstance(new, dict) and isinstance(old, dict):
        for k in sorted(set(new) & set(old)):
            if any(v in k.lower() for v in VOLATILE):
                continue
            d = _differs(new[k], old[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(new, (list, tuple)) and isinstance(old, (list, tuple)):
        if len(new) != len(old):
            return f"{path}: length {len(new)} against {len(old)}"
        for i, (a, b) in enumerate(zip(new, old)):
            d = _differs(a, b, f"{path}[{i}]")
            if d:
                return d
        return None
    if hasattr(new, "tolist"):
        return _differs(new.tolist(), old, path)
    return None if new == old else f"{path}: {new!r} against {old!r}"


def _sample(js, per_cell, skip, seed):
    """``per_cell`` jobs per cell, drawn with a fixed seed so that folds and users vary across cells."""
    cells: dict = {}
    for j in js:
        if j.get("candidate") in skip:
            continue
        key = (j.get("condition"), j.get("setting"), j.get("candidate"), j.get("n_basis"), j.get("variant"),
               j.get("family"), j.get("test"))
        cells.setdefault(key, []).append(j)
    rng = random.Random(seed)
    picked = [j for group in cells.values() for j in rng.sample(group, min(per_cell, len(group)))]
    return picked, len(cells)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "out")
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sweeps", default=",".join(SWEEPS))
    ap.add_argument("--skip-candidates", default="gp_bald,gp_eubo,psypref_adaptive")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--summary", type=Path, default=None)
    args = ap.parse_args(argv)
    skip = set(filter(None, args.skip_candidates.split(",")))
    _init(args.audit)

    import preftune
    import prompref
    print("prompref:", prompref.__file__, "\npreftune:", preftune.__file__, "\nout:", args.out, flush=True)

    d = args.out / "icorr" / "amp_free"
    todo, stored = [], {}
    for name in args.sweeps.split(","):
        module, log = SWEEPS[name]
        rows = {}
        for line in (d / log).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("error") is None:
                rows[_ident(r)] = r
        stored[name] = rows
        mod = importlib.import_module(module)
        picked, ncells = _sample((mod.jobs if hasattr(mod, "jobs") else mod.proj_jobs)(args.out), args.per_cell, skip, args.seed)
        print(f"{name}: {len(rows)} stored sessions, {ncells} cells, sampling {len(picked)}", flush=True)
        todo += [(name, j) for j in picked]

    tally = {n: dict(same=0, differ=0, error=0, nolog=0) for n in stored}
    problems = []
    # Fresh workers per sweep: some sweeps wrap another's run_session on import (icorr_amp_tie, icorr_amp_long), and
    # such a wrapper must not reach the next sweep's sessions.
    for sweep in stored:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.audit,)) as pool:
            futs = [pool.submit(_one, n, j, str(args.out)) for n, j in todo if n == sweep]
            for fut in as_completed(futs):
                name, job, got, err = fut.result()
                if err:
                    tally[name]["error"] += 1
                    problems.append((name, _ident(job), err))
                    continue
                for row in got if isinstance(got, list) else [got]:
                    if row.get("error"):
                        tally[name]["error"] += 1
                        problems.append((name, _ident(row), row["error"]))
                        continue
                    old = stored[name].get(_ident(row))
                    if old is None:
                        tally[name]["nolog"] += 1
                        problems.append((name, _ident(row), "no stored row with this identity"))
                        continue
                    diff = _differs(row, old)
                    tally[name]["differ" if diff else "same"] += 1
                    if diff:
                        problems.append((name, _ident(row), diff))
        print(f"{sweep}: {tally[sweep]}", flush=True)

    print("\nsweep   same differ error nolog")
    for n, t in tally.items():
        print(f"{n:7s} {t['same']:4d} {t['differ']:6d} {t['error']:5d} {t['nolog']:5d}")
    for p in problems[:40]:
        print("  ", p[0], p[1], str(p[2])[:200])
    if len(problems) > 40:
        print(f"   ... and {len(problems) - 40} more")
    if args.summary:
        args.summary.write_text(json.dumps(dict(tally=tally, problems=[[p[0], list(map(list, p[1])), str(p[2])]
                                                                       for p in problems]), indent=1), encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
