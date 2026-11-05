# exo-psypref: evaluation code for PsyPref

Supplement to our ICORR 2027 paper on speeding up user preference tuning of exoskeleton assistance with Bayesian adaptive
methods from psychophysics. PsyPref is the optimizer the paper proposes. This repository holds the simulation that
produced the paper's results. It contains no new human data, and everything in it was run in simulation.

- [CONDITIONS.md](CONDITIONS.md): how each simulated user and robustness condition is modeled, the size we chose, and
  where that size comes from.
- `prompref/`: the Python package, whose name comes from ProMP preference tuning. It holds PsyPref, the comparators, the
  simulated users and the profile families.
- `preftune/`: the gait-phase grid, the loader for the walking data and an earlier learner core that `prompref` builds on.
- `scripts/`: the sweeps behind each result and the scripts that turn their logs into figures.
- `figures/`: the paper's figures as TikZ, with the data they plot.

A guide to using PsyPref with other data, projections and priors will follow.

## From a figure to its data

| In the paper | Script | Logs it reads |
|---|---|---|
| Fig. 1 | `scripts/paper_v2_fig1.py`, `scripts/paper_v2_fig1j.py` | `full` |
| Fig. 2 | `scripts/paper_v2_fig2.py --variant v1`, then `c` and `d` | `full`, `cmp`, `abl`, `rnd` |
| Fig. 3 | `scripts/paper_v2_fig3p.py` | `fam_floors`, `fam`, `full`, `robust` |
| Fig. 4 | `scripts/paper_v2_fig4l.py`, then `scripts/paper_v2_fig4m.py` | `full`, `robust`, `rob2`, `tie`, `s3`, `var`, `pri`, `cmp` |
| Table I | `scripts/paper_v2_table_main.py --dest figures/tables` | `full`, `cmp`, `abl`, `rnd`, `pw`, `ks`, `ks2` |
| Absolute benefit by noise level | `scripts/provenance/s3_headline.py`, `scripts/provenance/noise_benefit_shares.py` | `full`, `cmp`, `robust`, `s3`, `var` |
| Direct choices between profiles | `scripts/provenance/direct_choices.py` | `full`, `cmp`, `long` |
| Choices between 16 and 8 weights | `scripts/provenance/weights_direct_choices.py` | `long`, `ks`, `ks2` |
| Distance removed, JND remaining | `scripts/provenance/distance_by_k.py` | `full`, `cmp` |
| Participation ratio, best profile per family | `scripts/provenance/effective_dimension.py`, `scripts/provenance/reach_beyond_family.py` | `fam_floors`, `long`, `ks`, `ks2` |
| Share of the widest pairs (reruns 30 sessions) | `scripts/provenance/pair_width_share.py` | `full` |

Each log is `out/icorr/amp_free/<name>_sessions.jsonl`, written by `scripts/icorr_amp_<name>.py`, with one line per
session. A line records the user, the setting, the optimizer, the distance to the preferred profile after each
comparison, and the seed.

## Three levels of reproduction

Python 3.12 with the versions in `requirements.txt`. numpy must stay below 2.

```bash
pip install -r requirements.txt
```

The data bundle (`exo-psypref-data.zip`, attached to the release, with a DOI to follow) unpacks at the repository's root
into `out/` and holds the logs, the users and the priors.

1. **Figures from the logs.** Needs the bundle and takes minutes.
   ```bash
   python scripts/paper_v2_fig4l.py
   python scripts/paper_v2_fig4m.py
   cd figures && pdflatex test_fig4m.tex
   ```
2. **Sessions from the stored users.** Needs the bundle. This reruns one randomly drawn session per condition, setting and
   optimizer and compares every number with the log, timings aside.
   ```bash
   python scripts/check_reproduction.py --workers 8
   ```
   A whole sweep reruns with `python scripts/icorr_amp_rob2.py run --workers 8`, after moving its log aside.
3. **Users and priors from raw data.** Needs the public gait data listed below, unpacked under
   `data/external_gait_data` and `data/molinaro/time_domain`. These are the `prep` commands of the sweeps.

### What we checked

From this repository and its bundle alone, on Windows with Python 3.12.3 and numpy 1.26.4:

- Level 1: all figure data and Table I came out identical to what the paper prints.
- Level 2: 658 sessions across 16 sweeps equal their logs exactly, one drawn per condition, setting and optimizer.
- Not rerun by the check: GP-BALD and GP-qEUBO (minutes per session), the adaptive-prior chains, the sweep with priors
  from other users (`pan`), and the floors.
- Level 3 was not run from this repository.

Another platform or BLAS may differ in the last digits.

## Data

| Data | Used for | In this repository |
|---|---|---|
| Molinaro et al., doi:10.35090/gatech/75759 | Prior and main users | Derived curves in the bundle, raw data not |
| Fukuchi et al. 2018, Reznick et al. 2021 | Users from other datasets | Derived curves in the bundle, raw data not |
| Schäfer et al. 2026, 11 participants | Centre and spread of the personal lag, spread of personal timing, and the family's timing distribution | Aggregates only: the mean and SD of the four preferred timings, inside the stored priors of CMA-ES, 4–6 param. (`out/phase5-schaefer/priors/*schaefer_peaks.json`). No raw or per-participant data |

Level 3 is therefore incomplete from this repository: the `prep` commands that draw new users also need the personal
lag's statistics from Schäfer et al.'s participants. `icorr_amp_pri.py prc`, which places the participants' own preferred
profiles on the simulated scale, needs their per-participant values, and its report is not included. The scripts that
read Schäfer et al.'s raw logs (`schaefer_grounding.py`, `icorr_jnd_free.py`) take the location with `--data`.

## Code comments

The code is the code that produced the logs, apart from file paths. Comments and docstrings were edited for this release,
and the build compares syntax trees to confirm nothing else changed. Some comments keep short labels of tests, such as K12,
from our design notes, which are not part of this repository.

## How to cite

The paper's reference will follow.

## License

The code is under the MIT License ([LICENSE](LICENSE)). The data bundle is under CC BY 4.0, subject to the terms of the
source datasets.
