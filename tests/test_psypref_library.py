"""The library runs the paper's PsyPref: its sessions equal the stored logs, number for number."""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"   # as the reproduction check: more threads change the last digits

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from psypref import Family, Prior, PsyPref, SimulatedUser, choice_probability, demo_profile

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (ROOT / "out", ROOT / "out")
OUT = next((p for p in CANDIDATES if (p / "icorr" / "amp_free" / "full_sessions.jsonl").exists()), None)
needs_logs = pytest.mark.skipif(OUT is None, reason="needs the data bundle")
OBSERVER_SALT = 0x5E55105
RANGES = {"medium": (3.0, 12.0), "wide": (3.0, 35.0)}


def _rows(name: str, setting: str, candidate: str, n: int = 2) -> list[dict]:
    rows = []
    with open(OUT / "icorr" / "amp_free" / f"{name}_sessions.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") is None and r["setting"] == setting and r["candidate"] == candidate:
                rows.append(r)
                if len(rows) == n:
                    break
    return rows


def _user_and_fold(row: dict) -> tuple[np.ndarray, dict]:
    amp = OUT / "icorr" / "amp_free"
    fold = row["fold"]
    with np.load(amp / f"full-{fold}.npz") as z:
        u = row["index"]
        minus, plus = z["user_minus"][u], z["user_plus"][u]
    with np.load(amp / f"fold-{fold}.npz") as z:
        parts = dict(w_minus=z["w_minus"], w_plus=z["w_plus"])
    return row["amps"][0] * minus + row["amps"][1] * plus, parts


def _replay(row: dict, prior: Prior, family=None) -> list[float]:
    ideal, _ = _user_and_fold(row)
    rng = np.random.default_rng([int(row["answer_seed"]), OBSERVER_SALT])
    user = SimulatedUser(ideal, noise=1.0, lapse=0.1, rng=rng)
    session = PsyPref(prior, family, seed=int(row["answer_seed"]) % 2 ** 32)
    out = []
    for _ in row["distances"]:
        session.tell(user.answer(*session.ask()))
        out.append(user.distance(session.identified()))
    return out


@needs_logs
def test_fixed_level_equals_the_log():
    for row in _rows("full", "fixed 12", "psypref"):
        _, parts = _user_and_fold(row)
        w = parts["w_minus"] + parts["w_plus"]
        prior = Prior.from_moments(w.mean(axis=0), np.cov(w, rowvar=False), level=12.0)
        assert _replay(row, prior) == pytest.approx(row["distances"], abs=1e-12)


@needs_logs
@pytest.mark.parametrize("setting", ["medium", "wide"])
def test_range_equals_the_log(setting):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import icorr_amp_range as R

    for row in _rows("full", setting, "psypref"):
        path = str(OUT / "icorr" / "amp_free" / f"fold-{row['fold']}.npz")
        mu, sigma = R._a_moments(path, row["fold_index"], setting)
        prior = Prior.from_moments(mu, sigma, peak_range=RANGES[setting])
        assert _replay(row, prior) == pytest.approx(row["distances"], abs=1e-12)


@needs_logs
@pytest.mark.parametrize("candidate,family", [("proj_schaefer_ii", "schaefer"), ("proj_pchip_ii", "pchip"), ("proj_franks", "franks")])
def test_projection_equals_the_log(candidate, family):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import icorr_amp_range as R

    with open(OUT / "icorr" / "amp_free" / "fam_sessions.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    row = next(r for r in rows if r.get("error") is None and r["setting"] == "wide" and r["candidate"] == candidate
               and r["fold"] == "loso-AB30")
    path = str(OUT / "icorr" / "amp_free" / f"fold-{row['fold']}.npz")
    mu, sigma = R._a_moments(path, row["fold_index"], "wide")
    prior = Prior.from_moments(mu, sigma, peak_range=RANGES["wide"])
    assert _replay(row, prior, family) == pytest.approx(row["distances"], abs=1e-9)


def test_closed_form_prior_matches_a_large_sample():
    p = Prior.from_walking(peak_range=(3, 35), leave_out="AB30")
    assert p.n_weights == 8 and p.sigma.shape == (8, 8)
    assert np.allclose(p.sigma, p.sigma.T)
    assert np.all(np.linalg.eigvalsh(p.sigma) > 0)


@needs_logs
def test_closed_form_is_close_to_the_paper_prior():
    with np.load(OUT / "icorr" / "amp_free" / "fold-loso-AB30.npz") as z:
        w_minus, w_plus = z["w_minus"], z["w_plus"]
    rng = np.random.default_rng(1)
    amps = np.exp(rng.uniform(math.log(3), math.log(35), (len(w_minus), 2)))
    w = amps[:, :1] * w_minus + amps[:, 1:] * w_plus
    p = Prior.from_walking(peak_range=(3, 35), leave_out="AB30")
    # the closed form against 20,000 draws: their sampling error is the size of these bounds
    assert np.abs(p.mu - w.mean(axis=0)).max() < 0.02 * np.abs(p.mu).max()
    assert np.abs(p.sigma - np.cov(w, rowvar=False)).max() < 0.03 * np.abs(p.sigma).max()


def test_session_beats_generic_assistance():
    preferred = demo_profile("AB30")
    prior = Prior.from_walking(peak_range=(3, 12), leave_out="AB30")
    user = SimulatedUser(preferred, seed=3)
    session = PsyPref(prior, seed=3)
    for _ in range(24):
        session.tell(user.answer(*session.ask()))
    assert user.distance(session.identified()) < user.distance(prior.generic)
    assert choice_probability(session.identified(), prior.generic, preferred) > 0.5


def test_ask_is_stable_and_tell_needs_ask():
    session = PsyPref(Prior.from_walking(level=12.0), seed=1)
    with pytest.raises(RuntimeError):
        session.tell("A")
    a1, b1 = session.ask()
    a2, b2 = session.ask()
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert a1.shape == (101,)
    with pytest.raises(ValueError):
        session.tell("left")


def test_no_preference_is_an_answer():
    session = PsyPref(Prior.from_walking(peak_range=(3, 12)), seed=2)
    for answer in ("A", None, "B", None):
        session.ask()
        session.tell(answer)
    assert session.n_comparisons == 4
    assert session.identified().shape == (101,)


def test_replay_continues_a_session():
    prior = Prior.from_walking(peak_range=(3, 12))
    a = PsyPref(prior, seed=5)
    for answer in ("A", "B", "B", "A"):
        a.ask()
        a.tell(answer)
        a.identified()
    b = PsyPref.replay(prior, a.answers, seed=5)
    assert np.array_equal(a.identified(), b.identified())
    assert all(np.array_equal(x, y) for x, y in zip(a.ask(), b.ask()))


def test_wider_prior_has_a_larger_search_region():
    prior = Prior.from_walking(peak_range=(3, 12))
    wide = prior.widened(2.0)
    assert wide._space().radius == pytest.approx(2.0 * prior._space().radius)
    assert np.array_equal(wide.sigma, prior.sigma)


@pytest.mark.parametrize("n_weights", [8, 12, 16, 24])
def test_more_weights(n_weights):
    session = PsyPref(Prior.from_walking(peak_range=(3, 12), n_weights=n_weights), seed=1)
    a, b = session.ask()
    session.tell("A")
    assert a.shape == b.shape == session.identified().shape == (101,)


def test_prior_from_own_curves():
    x = np.linspace(0, 2 * np.pi, 101)
    rng = np.random.default_rng(0)
    curves = np.array([np.sin(x + rng.normal(0, 0.15)) * rng.uniform(5, 9) for _ in range(60)])
    shaped = Prior.from_curves(curves, peak_range=(3, 12))
    assert shaped.peak_range == (3.0, 12.0) and shaped.generic.shape == (101,)
    as_is = Prior.from_curves(np.maximum(curves, 0.0), limit=12.0)
    session = PsyPref(as_is, seed=0)
    a, _ = session.ask()
    session.tell("B")
    assert a.max() <= 12.0 and session.identified().shape == (101,)
    with pytest.raises(ValueError):
        Prior.from_curves(curves[:5], limit=12.0)


@pytest.mark.parametrize("family", ["schaefer", "franks", "pchip"])
def test_projection_shows_family_curves(family):
    session = PsyPref(Prior.from_walking(peak_range=(3, 35)), family, seed=1)
    a, b = session.ask()
    assert a.shape == (101,)
    session.tell("A")
    assert session.identified().shape == (101,) and session.parameters().size == session.design.n_params


def test_own_family():
    x = np.linspace(0, 1, 101)

    def curve(theta):
        peak, width, height = theta
        return height * np.exp(-0.5 * ((x - peak) / width) ** 2) - height * np.exp(-0.5 * ((x - peak - 0.5) / width) ** 2)

    session = PsyPref(Prior.from_walking(peak_range=(3, 12)), Family(curve, [0.0, 0.03, 3.0], [0.5, 0.2, 12.0]), seed=1)
    a, _ = session.ask()
    session.tell("B")
    assert np.max(np.abs(a)) <= 12.0 + 1e-6 and session.parameters().size == 3


def test_projection_needs_a_range_or_level():
    prior = Prior.from_moments(np.zeros(8), np.eye(8), limit=12.0)
    with pytest.raises(ValueError):
        PsyPref(prior, "schaefer")
