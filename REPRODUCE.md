# Reproducing the paper

Supplement to our ICORR 2027 paper *Accelerating Hip Exoskeleton Preference Optimization: A Study with Simulated Users*, which speeds up preference-based optimization of exoskeleton
assistance with Bayesian adaptive methods from psychophysics. This part of the repository holds the simulation that produced the paper's results. It contains no new
human data, and everything in it was run in simulation. For using PsyPref, see the [README](README.md).

- [CONDITIONS.md](CONDITIONS.md): how each simulated user and robustness condition is modeled, the size we chose, and
  where that size comes from.
- [supplement/supplement.pdf](supplement/supplement.pdf): the paper's supplement. It holds PsyPref step by step (Alg. S1),
  a session with a simulated user (Alg. S2), the simulated users and conditions (Tables S1–S3), and Fig. S1, PsyPref
  against the best published optimizer under each violated assumption. Built from `supplement/supplement.tex` with
  `figures/fig4m.tex`.
- `prompref/`: the Python package, whose name comes from ProMP preference tuning. It holds PsyPref, the comparators, the
  simulated users and the profile families.
- `preftune/`: the gait-phase grid, the loader for the walking data and an earlier learner core that `prompref` builds on.
- `scripts/`: the sweeps behind each result and the scripts that turn their logs into figures.
- `figures/`: the paper's figures as TikZ, with the data they plot.

## From a figure to its data

| In the paper | Script | Logs it reads |
|---|---|---|
| Fig. 1 | `scripts/paper_v2_fig1.py`, `scripts/paper_v2_fig1j.py`; README version: `scripts/make_readme_fig1.py` (needs pdflatex) | `full` |
| Fig. 2 | `scripts/paper_v2_fig2.py --variant v1`, then `c` and `d` | `full`, `cmp`, `abl`, `rnd` |
| Fig. 3 | `scripts/paper_v2_fig3p.py` | `fam_floors`, `fam`, `full`, `robust` |
| Fig. 4 (compact Fig. S1) | `scripts/paper_v2_fig4c.py`, after `paper_v2_fig4m.py` | Fig. S1's data (`figures/data/fig4m_*.csv`) |
| Supplementary Fig. S1 | `scripts/paper_v2_fig4l.py`, then `scripts/paper_v2_fig4m.py` | `full`, `robust`, `rob2`, `tie`, `s3`, `var`, `pri`, `cmp` |
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
   python scripts/paper_v2_fig4c.py
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
  from other users (`pan`), and the distances of the best profiles (`fam_floors`).
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

## Terms and names

The code runs as it did for the logs. For this release, comments, docstrings and report wording were edited, and some
identifiers were renamed to match the paper. The build compares syntax trees and stops if anything else changed, and the
reproduction check above ran on this renamed code. Some comments keep short labels of tests, such as K12, from our design
notes, which are not part of this repository.

Names that stand in the logs or enter the seeds keep their original spelling:

| In the paper | In the code and logs |
|---|---|
| 3--12 Nm, 3--35 Nm, fixed 12 Nm | `medium`, `wide`, `fixed 12` |
| PsyPref | `psypref` |
| Arens' GP, from the prior's mean, with and without its gate | `g`, `g_generic`, `g_generic_nogate` |
| LineCoSpar | `d_z` |
| CMA-ES, with population 4 | `cmaes`, `cmaes4` |
| CMA-ES, 4--6 param. | `cmaes_sch4` |
| GP-qEUBO | `gp_eubo` |
| Noise level $\sigma$ | `s` |
| The best published optimizer, or the comparator | `best`, `best_published` |
| Distance of the best profile, its preference | `floor`, `ceiling` and names with them |
| Projection onto a family | `lens` and names with it |
| A profile on offer | `arm` |
| A planned test, in a stored key | `registered` |

## License

The code is under the MIT License ([LICENSE](LICENSE)). The data bundle is under CC BY 4.0, subject to the terms of the
source datasets.
