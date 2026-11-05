"""The Phase 3 gate's stamp: which source the gate last passed on.

The gate takes minutes, so it is a script and not a unit test -- which meant nothing
re-ran it when the code it validates changed. The stamp closes that gap cheaply: every
full gate run records a hash of each gated file, and a fast test fails whenever one no
longer matches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAMP = Path(__file__).with_name("gate_stamp.json")

#: The files whose behaviour the gate establishes. The gate script is one of them: its
#: thresholds are the gate.
#: arms.py since Phase 5b: the gate takes its priors and duel radii from it.
GATED = (
    "prompref/learning/smc.py",
    "prompref/validation/reference_posterior.py",
    "scripts/recovery_gate.py",
    "prompref/learning/arms.py",
)


def source_hash(path: Path) -> str:
    """SHA-256 of a text file with CRLF normalised to LF -- checkouts here convert endings."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current_hashes(root: Path = ROOT, gated: tuple[str, ...] = GATED) -> dict[str, str]:
    return {rel: source_hash(Path(root) / rel) for rel in gated}


def write_stamp(report: dict, lines: list[str], path: Path = STAMP, root: Path = ROOT,
                gated: tuple[str, ...] = GATED) -> None:
    """Record the verdict, pass or fail, against the source it was reached on."""
    stamp = {"passed": bool(report["passed"]), "hashes": current_hashes(root, gated),
             "verdict": list(lines)}
    Path(path).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")


#: The Phase 4 gate: the acquisition, the likelihood it uses, the independent
#: exact-information grid, and the script that holds the thresholds -- and, since Phase 5b,
#: the arms whose priors and radii it checks.
ACQ_STAMP = Path(__file__).with_name("acquisition_gate_stamp.json")
ACQ_GATED = (
    "prompref/learning/smc.py",
    "prompref/learning/acquisition.py",
    "prompref/validation/exact_mi_1d.py",
    "scripts/acquisition_gate.py",
    "prompref/learning/arms.py",
)
