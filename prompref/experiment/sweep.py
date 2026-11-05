"""Run many points across processes, appending results as they land.

Every claim in the study comes from here rather than from a single session. Results
append to JSONL and existing *successful* rows are skipped, so a sweep can be
interrupted and resumed. Failed rows are deliberately retried: a point can fail for
reasons that have nothing to do with the point.
"""

from __future__ import annotations

import itertools
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..designs.bump import BumpDesign
from ..designs.fourier import FourierDesign
from ..designs.promp import PrompS0, PrompS1, PrompS2
from ..designs.schaefer import SchaeferDesign
from ..designs.trapezoid import TrapezoidDesign
from ..learning.batch_mi import BatchMILearner
from ..learning.linecospar import LENGTHSCALE, STEP, LineCoSparLearner
from ..learning.psychometric import PsychometricLearner
from ..learning.whitened import WhitenedLearner
from ..user.oracle import Observer
from .metrics import design_floor, prior_mean_distance
from .run import run_session

DESIGNS = {
    "trapezoid": TrapezoidDesign,
    "schaefer": SchaeferDesign,
    "bump": BumpDesign,
    "fourier": FourierDesign,
    "promp_s0": PrompS0,
    "promp_s1": PrompS1,
    "promp_s2": PrompS2,
}

#: Where the simulated wearer's preferred curve comes from. The two answer different
#: questions and must not be mixed in one table.
#:
#: ``in_family`` draws a plausible smooth two-bout curve from the family's own prior, so
#: the target is exactly reachable and every shortfall measured is the LEARNER's. This is
#: the right oracle for "did the search work".
#:
#: ``biological`` uses the shared Molinaro-transformed hip moment, identical for every
#: family, so a six-parameter semi-trapezoid and a sixteen-weight ProMP are scored against
#: one target. This is the only valid oracle for CROSS-FAMILY comparison, and what it
#: measures is family fit as much as learning.
IDEALS = ("in_family", "biological")


def _linecospar_constants(point) -> dict:
    return dict(lengthscale=LENGTHSCALE * point.lengthscale_mult,
                step=STEP * point.step_mult)


#: Every learner is built from its point, because A and D_z read ``eps`` and ``d``.
#:
#: C1 and C2 are the same learner on different design spaces -- that is the point of
#: the pairing, since it isolates the search space from the learning rule. The key is
#: kept distinct so sweep rows stay readable.
#: D is LineCoSpar on the family's own box.
#: A_random is learner A in Phase 3: the SMC posterior driven by random duels.
#: D_z is LineCoSpar inside A's whitened trust ball -- the comparator
#: that holds the search space fixed, so A versus D_z tests duel direction alone.
#: A is learner A in Phase 4: duels chosen by expected information gain.
LEARNERS = {
    "c1": lambda design, point: BatchMILearner(design, seed=point.seed),
    "c2": lambda design, point: BatchMILearner(design, seed=point.seed),
    "d": lambda design, point: LineCoSparLearner(
        design, seed=point.seed, **_linecospar_constants(point)),
    "a_random": lambda design, point: PsychometricLearner(
        design, seed=point.seed, eps=point.eps, d=point.d, duel_space=point.duel_space),
    "d_z": lambda design, point: WhitenedLearner.linecospar(
        design, seed=point.seed, eps=point.eps, d=point.d, **_linecospar_constants(point)),
    "a": lambda design, point: PsychometricLearner(
        design, seed=point.seed, eps=point.eps, d=point.d, duels="eig",
        duel_space=point.duel_space, p_probe=point.p_probe, max_mult=point.max_mult,
        r_step=point.r_step, duel_directions=point.duel_directions, stop_sd=point.stop_sd),
}

#: Learners that reason in a LearnerSpace, and so are the only ones that read eps and d.
SPACE_LEARNERS = frozenset({"a", "a_random", "d_z"})

#: Learners built on LineCoSpar, and so the only ones that read its constant multipliers.
LINECOSPAR_LEARNERS = frozenset({"d", "d_z"})


@dataclass(frozen=True)
class Point:
    design: str
    learner: str
    lapse: float
    s: float
    seed: int
    n_comparisons: int
    ideal: str = "in_family"
    randomise_order: bool = True
    order_bias: float = 0.0
    #: The learner's prior mixture. Never reaches the design.
    eps: float = 0.0
    #: Whitened dimension; ``None`` means every parameter.
    d: int | None = None
    #: Where learner A draws duels. The default reproduces every earlier row.
    duel_space: str = "reference"
    #: Multipliers on LineCoSpar's published lengthscale and step, for D and D_z only
    #:. 1.0 passes the published constants exactly.
    lengthscale_mult: float = 1.0
    step_mult: float = 1.0
    #: Learner A's Phase 4 options; read only by learner "a". The defaults are
    #: the learner's, so every earlier row resolves to them.
    p_probe: float = 0.15
    max_mult: float = 4.0
    r_step: float | None = None
    duel_directions: str = "free"
    stop_sd: float | None = None

    def key(self) -> str:
        fields = dict(self.__dict__)
        # json.dumps distinguishes 1 from 1.0; normalise so an int eps or a float d cannot
        # fork a resume key and double-count a seed.
        fields["eps"] = float(fields["eps"])
        if fields["d"] is not None:
            fields["d"] = int(fields["d"])
        fields["lengthscale_mult"] = float(fields["lengthscale_mult"])
        fields["step_mult"] = float(fields["step_mult"])
        fields["p_probe"] = float(fields["p_probe"])
        fields["max_mult"] = float(fields["max_mult"])
        for name in ("r_step", "stop_sd"):
            if fields[name] is not None:
                fields[name] = float(fields[name])
        return json.dumps(fields, sort_keys=True)


#: The Point fields only learner "a" reads, with the values every other learner must carry.
PHASE4_FIELDS = ("p_probe", "max_mult", "r_step", "duel_directions", "stop_sd")
_PHASE4_DEFAULTS = {name: Point.__dataclass_fields__[name].default for name in PHASE4_FIELDS}


def expand(grid: dict[str, list], **fixed) -> list[Point]:
    """Cartesian product of the grid, as :class:`Point` objects."""
    names = list(grid)
    return [Point(**dict(zip(names, values)), **fixed)
            for values in itertools.product(*(grid[n] for n in names))]


_IDEAL_CACHE: dict[tuple, tuple] = {}


def ideal_curve(design_name: str, ideal: str) -> np.ndarray:
    """The simulated wearer's preferred curve. Cheap: no floor is computed.

    See :data:`IDEALS` for why the two oracles are not interchangeable. Takes no
    ``eps``, by design -- the target must be identical at every epsilon.
    """
    design = DESIGNS[design_name]()
    if ideal == "in_family":
        return design.curve(design.sample_prior(np.random.default_rng(12345), 1)[0])
    if ideal == "biological":
        from preftune.data import build_ideal, load_trial

        return build_ideal(load_trial("AB30", "r"), delay_ms=100.0).torque_nm
    raise ValueError(f"ideal must be one of {IDEALS}, got {ideal!r}")


def _ideal_for(point: Point) -> tuple[np.ndarray, float, float]:
    """Ideal curve, family floor and prior-mean distance, for this design and oracle.

    Cached per (design, oracle) because the floor costs a multi-start optimisation.
    """
    key = (point.design, point.ideal)
    if key not in _IDEAL_CACHE:
        design = DESIGNS[point.design]()
        ideal = ideal_curve(point.design, point.ideal)
        _IDEAL_CACHE[key] = (
            ideal,
            design_floor(design, ideal, n_starts=20),
            prior_mean_distance(design, ideal),
        )
    return _IDEAL_CACHE[key]


def _check_point(point: Point, design) -> None:
    """Refuse configurations that would otherwise produce a complete, plausible, wrong row."""
    if point.lengthscale_mult <= 0.0 or point.step_mult <= 0.0:
        # LineCoSpar's kernel squares the lengthscale-scaled difference, and its line
        # search tries both signs of the step at every anchor, so flipping either sign
        # produces exactly the same candidates as its positive counterpart: a negative
        # multiplier would silently repeat the positive setting under a different key
        # rather than fail. Checked for every learner, ahead of "does not read", so a
        # non-LineCoSpar point with a bad multiplier is refused for the right reason.
        raise ValueError(f"lengthscale_mult and step_mult must be positive; got "
                         f"{point.lengthscale_mult}, {point.step_mult}")
    if point.learner not in SPACE_LEARNERS:
        if point.eps != 0.0 or point.d is not None:
            raise ValueError(f"learner {point.learner!r} does not read eps or d; got "
                             f"eps={point.eps}, d={point.d}")
    elif point.d is not None and point.d < design.n_params:
        raise ValueError(f"d={point.d} < n_params={design.n_params} needs a subspace "
                         f"floor, which is not implemented (spec Ruling R9)")
    if point.learner not in ("a", "a_random") and point.duel_space != "reference":
        raise ValueError(f"learner {point.learner!r} does not read duel_space; got "
                         f"{point.duel_space!r}")
    if point.learner == "d_z" and point.eps != 0.0:
        raise ValueError(f"D_z is defined at eps = 0 only (Ruling R15): LineCoSpar's "
                         f"unit-cube constants lose their meaning in an anisotropic ball; "
                         f"got eps={point.eps}")
    if point.learner not in LINECOSPAR_LEARNERS and (
            point.lengthscale_mult != 1.0 or point.step_mult != 1.0):
        raise ValueError(f"learner {point.learner!r} does not read lengthscale_mult or "
                         f"step_mult; got {point.lengthscale_mult}, {point.step_mult}")
    if point.learner != "a":
        changed = [name for name in PHASE4_FIELDS
                   if getattr(point, name) != _PHASE4_DEFAULTS[name]]
        if changed:
            raise ValueError(f"learner {point.learner!r} does not read {', '.join(changed)}")


def run_point(point: Point) -> dict:
    """One session. Never raises, so one bad point cannot abort a long sweep."""
    try:
        design = DESIGNS[point.design]()
        _check_point(point, design)
        ideal, floor, d_prior = _ideal_for(point)
        observer = Observer(ideal_nm=ideal, s=point.s, lapse=point.lapse,
                            order_bias=point.order_bias)
        learner = LEARNERS[point.learner](design, point)
        record = run_session(design, learner, observer,
                             np.random.default_rng(point.seed),
                             n_comparisons=point.n_comparisons,
                             randomise_order=point.randomise_order,
                             d_prior=d_prior, d_floor=floor)
        return dict(point.__dict__, **record, error=None)
    except Exception as exc:  # pragma: no cover - defensive
        return dict(point.__dict__, error=f"{type(exc).__name__}: {exc}")


def _parse_rows(path: Path) -> list[dict]:
    """Every parseable JSONL row, warning about and skipping the ones that are not.

    A hard kill mid-write leaves a torn final line. Nothing about that line is
    recoverable, but it must not be *fatal*: parsing it unguarded raised from both the
    resume scan and the loader, so a single torn line permanently broke every future
    resume of that sweep **and** made the results unreadable -- losing an entire sweep's
    compute to one interrupted write. Skip it loudly and carry on.
    """
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            warnings.warn(
                f"{path}: skipping unparseable line {number} ({exc}). A torn final line "
                f"is the expected cause -- that point will simply be re-run.",
                stacklevel=2,
            )
    return rows


def _row_key(row: dict, point_cls=Point) -> str | None:
    """The :meth:`Point.key` of the point that produced this row, or ``None``.

    Reconstructing the ``Point`` rather than filtering the row's own keys is what lets a
    resume survive **adding a field to** :class:`Point`. Comparing filtered row keys
    against ``p.key`` breaks the moment a field is added: every existing row's key
    omits the new name while ``p.key`` carries its default, so no key matches, every
    completed point is re-run, and its second result is *appended beside* the first --
    silently double-counting that seed in every mean the analysis takes. Building a
    ``Point`` fills absent optional fields from their defaults, which is exactly the
    value ``expand`` would have used.

    ``None`` means the row cannot be attributed to any point -- a field with no default is
    missing, a name is unknown (``TypeError``), or the point class refuses the combination
    the row describes (``ValueError``) -- so it is not treated as done.

    ``ValueError`` is caught because a point class may refuse a field combination at
    construction, not merely a wrong signature:
    :class:`~prompref.experiment.phase5.Phase5Point` refuses a G point whose ``jnd_scale``
    was never stated. No writer here produces such a row -- ``run_phase5_point`` serialises
    ``point.__dict__`` for success and error rows alike, so ``jnd_scale`` is always present
    -- but a hand-edited, truncated or third-party row could, and letting it escape would
    abort the resume scan of a 13,400-session sweep instead of being skipped. Widening what
    is caught cannot change the outcome for any row that already round-tripped: those return
    a key and never reach the handler.

    ``point_cls`` is the sweep's point class; Phase 5 sweeps pass
    :class:`~prompref.experiment.phase5.Phase5Point`.
    """
    fields = set(point_cls.__dataclass_fields__)
    try:
        return point_cls(**{k: v for k, v in row.items() if k in fields}).key()
    except (TypeError, ValueError):
        return None


def _done_keys(path: Path, point_cls=Point) -> set[str]:
    """Points already recorded *successfully*. Failed rows are retried."""
    keys = set()
    for row in _parse_rows(path):
        if row.get("error"):
            continue
        key = _row_key(row, point_cls)
        if key is not None:
            keys.add(key)
    return keys


def _repair_trailing_newline(path: Path) -> None:
    """Terminate a file whose last write was cut short, before appending to it.

    A torn line has no trailing newline, so the next appended row would be glued onto
    the fragment -- turning one unparseable line into one unparseable line *that also
    swallows a good result*, on every resume, forever.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) != b"\n":
            with path.open("a", encoding="utf-8") as out:
                out.write("\n")


def run_sweep(points: list, path: str | Path, workers: int | None = None,
              resume: bool = True, *, runner=None) -> Path:
    """Run ``points`` with ``runner`` (default :func:`run_point`), appending rows to ``path``.

    Rows are matched to points by the points' own class, so a Phase 5 sweep resumes exactly
    as the Phase 3 and 4 sweeps do.
    """
    runner = runner or run_point
    point_cls = type(points[0]) if points else Point
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(path, point_cls) if resume else set()
    _repair_trailing_newline(path)
    todo = [p for p in points if p.key() not in done]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(points)} points, {len(points) - len(todo)} already done, "
          f"{len(todo)} to run on {workers} workers")

    # Flush after every row. A sweep is hours of compute and is expected to be
    # interrupted; anything still sitting in the buffer when it is killed is lost, and a
    # partially flushed buffer is exactly what tears a line in the first place.
    if workers == 1:
        with path.open("a", encoding="utf-8") as fh:
            for p in todo:
                fh.write(json.dumps(runner(p)) + "\n")
                fh.flush()
        return path

    with path.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(runner, p): p for p in todo}
        for future in as_completed(futures):
            fh.write(json.dumps(future.result()) + "\n")
            fh.flush()
    return path


def load_sweep(path: str | Path):
    """Read a sweep into a DataFrame, dropping failed points."""
    import pandas as pd

    df = pd.DataFrame(_parse_rows(Path(path)))
    if "error" in df:
        bad = df["error"].notna()
        if bad.any():
            print(f"dropping {int(bad.sum())} failed runs")
        df = df[~bad].drop(columns=["error"])
    return df.reset_index(drop=True)
