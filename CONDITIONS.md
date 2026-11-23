# Simulated users and robustness conditions

This table is the supplement to the paper's robustness figure (Fig. 4). It states how each condition is modeled, the
size we chose, and where that size comes from.

Hip data bear only on device noise and the spread of peak torques, so the other sizes rest on other joints and tasks, or
on our choice. Of the 42 values below, 4 are stated by a source, 1 lies inside a measured range, 14 are transferred from
other joints or tasks, and 23 are our own choices. Read the "our choice" rows as stress tests, not as estimates of real
users.

## The main simulated user

| Part | Model | Value | Source |
|---|---|---|---|
| Users | Each fold holds out one of five walking subjects. The prior comes from the other four, the users from the held-out one. | 5 folds × 40 users = 200 users | Molinaro et al., AB30–AB34 |
| Preferred profile | $\tau^* = a\,u^- + b\,u^+$, with $u^-$ and $u^+$ the unit extension and flexion parts of a user's shape | Peaks log-uniform in 3–35 Nm, or both at 12 Nm | Our choice |
| Personal shape | The held-out subject's moment curve, with a personal lag and personal landmark offsets | Lag centre and spreads from the summary statistics of Schäfer et al.'s 11 participants, Student-t with $\nu = 10$ | Schäfer et al., summary only |
| Distance | $d(c) = \mathrm{RMS}(c - \tau^*)$ over the gait cycle | – | – |
| Answer | $P(A \succ B) = \lambda/2 + (1-\lambda)\,\Phi\big((d(B) - d(A))/\sigma\big)$ | $\sigma = 1$ Nm, $\lambda = 0.1$ | $\sigma$: see below. $\lambda$: our choice |
| JND | The distance difference answered correctly in 75 % of attentive answers | $0.674\,\sigma$ | Follows from the model |

**Noise level.** No JND is published for hip assistance in walking. Measured on the 200 users (median RMS 7.36 and
7.03 Nm at 3–35 and 12 Nm), the anchors give:

| Anchor | Source | Curve change (Nm RMS) | Implied $\sigma$ (Nm) |
|---|---|---|---|
| Static force, 7 % | Jones 1989†, Pang 1991† | 0.49–0.52 | 0.73–0.77 |
| Static force, 10 % (our round-up) | none states 10 % | 0.71–0.74 | 1.05–1.09 |
| Walking stiffness, 25 % (a multiplier of about 3 from ankle stiffness) | Maberry & Martin 2026† | 1.76–1.84 | 2.6–2.7 |
| Ankle timing, 2.8 %GC | Peng 2022 | 1.88–2.01 | 2.8–3.0 |

The main condition sits near the static end, and the tests reach $\sigma = 3$ Nm. Peng et al. report that their JND is
biased upward, so $\sigma = 3$ Nm is closer to an upper end than to an expected value.

**Rating.** The measure $P_\mathrm{g}$ is the probability of choosing the identified profile over generic assistance,
$100\,[\lambda/2 + (1-\lambda)\,\Phi((d_0 - d_k)/\sigma)]$ %, with $d_0$ the distance of generic assistance, $d_k$ the
distance of the profile after $k$ comparisons, and $k = 24$ in Fig. 4. Each user is rated with their own $\sigma$ and
$\lambda$ at the session's end. The maximum is therefore 95 % at $\lambda = 0.1$ and 85 % at $\lambda = 0.3$. Where a
condition changes what the user judges, $d$ is that user's own distance.

## Conditions

PsyPref keeps its model and prior in every condition. The one exception is the tie handling in the two rows marked ‡.
Each row changes only what it names, relative to the main user.

**Classes.** *Stated*: a source gives this value for this quantity. *Range*: inside a measured range. *Transferred*: from
another joint, task or quantity. *Ours*: no source gives the size.

### How they answer

| Fig. 4 label | Condition in the code | Model | Value | Class | Source and limits |
|---|---|---|---|---|---|
| Noise level σ = 0.5 Nm | `R1 s=0.5` | $\sigma$ replaced | 0.5 Nm | Ours | Below every anchor above |
| Noise level σ = 2, 3 Nm | `R2 s=2.0`, `R2b s=3.0` | $\sigma$ replaced | 2, 3 Nm | Transferred | Peng 2022 (ankle timing), Maberry & Martin 2026† (ankle stiffness), converted to Nm RMS by us |
| Lapse rate λ = 0.2, 0.3 | `A2 lapse 0.2`, `R3 lapse=0.3` | $\lambda$ replaced | 0.2, 0.3 | Transferred, ours | 0.2: indirect bound from validation rates of Arens 2025, Lee 2023, Alili 2023 (other joints). 0.3: stress case above that bound |
| σ differs per user | `A1 individual s` | $\sigma_u = \sigma\,e^{0.35 z}$, $z \sim N(0,1)$ | log SD 0.35 | Transferred | Shepherd 2018 (prosthesis stiffness, n = 8), Miller 2026† (ankle timing): about fourfold range. Shepherd alone implies about 0.45 |
| Other answer-curve shapes | `A6a`, `A6b`, `A6c` | $\Phi$ replaced by a logistic, Student-t or Cauchy curve, scaled to match the decisiveness of $\Phi$ on an earlier simulated population | Scales 0.5916, 0.8100, 0.4800; $\nu = 3$, $\nu = 1$ | Transferred (logistic), ours | Logistic: fitting convention in Shepherd 2018. Heavy tails: stress cases |
| Device noise | `A3 device noise` | Each presented curve gets its own Fourier series over harmonics 1–4, scaled to this RMS | 0.6 Nm RMS | Stated | Franks 2021, p. e16-5, at the hips of a tethered emulator. The spectrum is ours |
| Bias toward first profile | `A4 order bias` | $\beta\sigma$ added in favour of the curve presented first | $\beta = 0.3$ | Ours | No published estimate found |
| "No preference" below 0.5 JND ‡ | `A5 no preference` | The user answers "no preference" whenever $|d(A) - d(B)| < 0.5$ JND. A lapse is a coin flip between A and B | 0.5 JND | Ours | Arens 2025 and Tucker 2020 allow such answers, and neither gives a frequency |

### How they change

Paths are linear over the 40 comparisons of a session. Familiarity is the exception: it holds from the first comparison.

| Fig. 4 label | Condition in the code | Model | Value | Class | Source and limits |
|---|---|---|---|---|---|
| Fatigue | `A7 fatigue` | $\lambda$ rises | 0.05 to 0.30 | Ours | Ingraham 2022 rules out fatigue for their drift |
| σ 1.5 to 0.75 Nm, and reverse | `A8a`, `A8b` | $\sigma$ falls or rises | 1.5 to 0.75 Nm, and reverse | Ours | Direction only: Ingraham 2022, El Husaini 2023 |
| Preferred profile drifts | `B4a`, `B4b` | $\tau^*$ grows by a factor, so that it ends this far from the start | 1, 2 JND | Transferred | Direction and size: Arens 2025 (rise not significant, P = 0.181), Ingraham 2022. 2 JND exceeds the derived 1.0–1.6 JND |
| Familiarity | `B3 familiarity` | Target becomes $(\tau^* + w\,\tau_h)/(1 + w)$, with $\tau_h = 0$, so it sits 20 % closer to zero torque | $w = 0.25$ | Ours | Direction only: Díaz 2026 (n = 3, "possibly") |

### What they judge and prefer

| Fig. 4 label | Condition in the code | Model | Value | Class | Source and limits |
|---|---|---|---|---|---|
| Weber's law | `B1 Weber` | $\sigma_\text{eff} = \sigma\,(\mathrm{RMS}_\text{pair} + a)/(\mathrm{RMS}_\text{typ} + a)$ | $a = 1$ Nm, $\mathrm{RMS}_\text{typ} = 4.83$ Nm | Transferred (form), ours (values) | Hatzfeld 2016 (fingertip vibration). $\mathrm{RMS}_\text{typ}$ stems from an earlier population, so the median user here is also 1.38–1.43 times noisier |
| Over-assistance counts double | `B2 asymmetry` | Squared error doubled where $|c| > |\tau^*|$, about 1.41 times in RMS | Factor 2 | Ours | No source. Clites 2021 points the other way |
| Judging single features | `B5a`, `B5b` | Distance over two peak magnitudes and two peak timings: $\sqrt{\sum (\Delta_i/\mathrm{JND}_i)^2}$, with magnitude JND $\max(f\,|\text{peak}|,\,0.5\text{ Nm})$, timing JND 2.8 %GC, and the scale matched to the main user's decisiveness | $f = 0.10, 0.25$ | Transferred | Jones 1989†, Pang 1991† state 7 %, static. Maberry & Martin 2026†: walking about 3 times larger. Peng 2022 for timing. The 0.5 Nm floor is ours |
| Gait phases weighted unequally | `B6a`, `B6b`, `B7` | Weighted RMS, $w(\varphi)$ with mean 1, on a per-user scale matched to the main user's decisiveness. Personal: $\log w$ a random Fourier series over harmonics 1–3, scaled to this SD over the cycle. Hip speed: $w \propto |\dot\theta| + 0.2\,\overline{|\dot\theta|}$ | log SD 0.5, 1 | Ours | Rationale for the hip-speed form: power change $\Delta\tau\,\dot\theta$. No measurement of phase-dependent perception found |
| No single preferred profile | `B8 random utility` | A random smooth function of a curve's first six harmonics, from random Fourier features, inside an envelope, on a matched per-user scale | Length scale 1.5, envelope 3 SD | Ours | Deliberately not in PsyPref's coordinates |
| Negative power disliked | `R4 negative power` | $d(c) + \kappa\,\overline{\max(0, -c\,\dot\theta)}/\overline{|\dot\theta|}$ | $\kappa = 0$ with probability ½, else log-uniform in 0.5–2 | Ours | Direction only: Ramella 2025 |

### How they differ from the prior

| Fig. 4 label | Condition in the code | Model | Value | Class | Source and limits |
|---|---|---|---|---|---|
| Users from other datasets | `R6 other datasets` | Shapes from other people, 40 users per group | Young and older adults on a treadmill at comfortable speed; level and inclined walking at 1.0 m/s | Stated | Fukuchi 2018, Reznick 2021. These are biological moments, and treating them as preferred shapes is our assumption |
| Peaks unlike the prior | `M1`–`M6` | Peak laws: three narrow cohorts, flexion or extension larger, correlated | Medians at 0.2, 0.5, 0.8 of the log range with CV 0.2; ratio 1.5; correlation 0.8 | Range (CV), ours | CV: Ramella 2025 (0.13–0.15), Arens 2025 (0.30–0.34), Shepherd 2018 (0.30). Rest is ours |
| Flexion peak later | `S1`, `S2` | Piecewise-linear phase warp that moves the flexion peak and keeps the extension peak | 6, 12 %GC | Transferred | Kim 2022 (soft flexion exosuit, metabolic optimum, n = 8) |
| Other shapes unlike the prior | `R7 surrogates` | The preferred curve's magnitudes kept, its phases randomized, the result clipped to the setting's top | Harmonics 1–50 | Transferred | Theiler 1992, a method for testing time series. It serves here as a source of curves without the prior's shape |
| Other shapes unlike the prior | `C2 heavy tails` | Personal lag and landmarks with $\nu = 1.5$ instead of 10, same scales | $\nu = 1.5$ | Ours | Stress case: infinite variance before invalid draws are rejected |
| Other shapes unlike the prior | `C3 family mixture` | A share of users replaced, half from Schäfer et al.'s family, half from Franks et al.'s | 20 % | Ours | – |
| Shapes from the 4–6-param. family | `C1 Schaefer shapes` | Four timings from a Student-t distribution in Schäfer et al.'s family. Draws outside the device limits are rejected | Mean and diagonal covariance of 11 participants, $\nu = 10$ | Stated (family), ours ($\nu$) | Schäfer 2026. CMA-ES, 4–6 param., searches this family |

### Pessimistic user

| Fig. 4 label | Condition in the code | Model | Value | Class | Source and limits |
|---|---|---|---|---|---|
| Pessimistic user (six at once) ‡ | `C4 pessimistic` | Six of the above at once | $\sigma = 2$, fatigue, bias 0.3 σ, no preference at 0.5 JND, phase weight log SD 1, Schäfer-family shapes | Ours | Combination of the rows above |

‡ PsyPref uses "no preference" answers as data (PsyPref-T, `prompref/learning/ties.py`). These two rows use users 8–39 of
each fold, n = 160 (`scripts/icorr_amp_tie.py`).

† Read as abstract only.

## Optimizers

Fig. 4 measures PsyPref against the strongest of the first six, in every row and for every bootstrap draw.

| Name in the paper | Name in the code | Searches | Where |
|---|---|---|---|
| PsyPref | `psypref` | 8 ProMP weights, in the prior's search region from walking data. At 3–35 Nm, as `metric_psypref` inside `TwoPartGuard` | `prompref/learning/smc.py`, `acquisition.py`, `psychometric.py`; `prompref/experiment/stress.py`, `amp_free.py` |
| Arens' GP | `g` | 8 ProMP weights, in the prior's search region | `prompref/learning/gp_ucb.py`, `preference_gp.py` |
| Arens' GP from the prior's mean, no gate | `g_generic_nogate` | The same, started from the prior's mean, without its gate. Fig. 2's comparator at 3–35 Nm | `prompref/experiment/review.py` |
| LineCoSpar | `d_z` | 8 ProMP weights, in the prior's search region | `prompref/learning/linecospar.py` |
| CMA-ES | `cmaes` | 8 ProMP weights, in the prior's search region. pycma's default population, initial step 1.0 of the prior's scale | `prompref/learning/sortcma.py` |
| CMA-ES, population 4 | `cmaes4` | The same, with population 4 and initial step 0.5. Fig. 2's comparator at fixed 12 Nm | `prompref/learning/sortcma.py` |
| CMA-ES, 4–6 param. | `cmaes_sch4` | Schäfer et al.'s family: four timings, plus both peaks in 3–35 Nm. At fixed 12 Nm, the four timings only. Population 4, initial step 0.5 of the prior's scale | `prompref/learning/sortcma.py`, `prompref/experiment/amp_free.py` |
| GP-BALD | `gp_bald` | 8 ProMP weights. Main condition, $\sigma = 2$ Nm and users from other datasets. Not part of Fig. 4's leads | `prompref/learning/gp_eubo.py` |

## Method details left out of the paper

The paper's Methods are short because of its page limit. These details complete them.

### Prior and sessions

- Each fold's prior comes from 20,000 per-stride curves, with a leave-one-subject-out design over the five walking subjects.
- Published peaks span an order of magnitude, which is why a range draws both peaks log-uniformly.
- The filter and the delay follow moment-based control, which also filters and delays (Molinaro et al.).
- Each setting has 200 users, 40 per fold. The 150-comparison runs and the runs with 12, 16 and 24 weights use 100 of them, 20 per fold.
- We read each 40-comparison session at k = 12, 24 and 40. These readings match sessions of that length, since no optimizer uses the session's length.

### PsyPref

- **Metric.** In a range, M = J'J/101 is scaled to trace d. A step then counts by how much it changes the curve in Nm, not by how much the prior lets it vary.
- **Candidate pairs.** c ± m u / 2 combines four centers (the posterior mean and three posterior draws), 11 directions (the posterior's eight principal axes and three random ones) and five widths m in {0.25, 0.5, 1, 2, 4} times the posterior mean of l, which gives 220 candidates.
- **Width cap.** In an earlier pilot, caps of 1, 2, 4 and 8 discrimination scales left the identified curve 0.640, 0.456, 0.388 and 0.375 Nm from the target.
- **Probe pairs.** With probability 0.15, a pair at a set distance from the posterior mean, in multiples of the posterior mean of l.
- **Delivered curve.** Each delivered curve must assist both extension and flexion. If one of its peaks falls below half of the range's lower limit, PsyPref replaces it with the posterior draw closest to the posterior mean, among 500 draws, that meets this limit. In ranges this fires in 4 and 18 % of sessions, never at the session's end.

### Other optimizers

- **GP-BALD.** Searches up to PsyPref's radius. The prior mean falls quadratically from the prior's center, with learned curvature.
- **Arens' GP.** As published, it learns the user's preference with a Gaussian process.
- **LineCoSpar.** Reimplemented. It searches a random line through its best point by Thompson sampling.
- **CMA-ES.** pycma 4.5.0. With population 4 and initial step 0.5, sorting a generation costs about five comparisons.
- **GP-qEUBO.** On GP-BALD's model, it asks the pair whose better option has the highest expected utility.
- **Ramella et al.'s learner.** Fits a linear reward and asks by mutual information among 40 candidates drawn from the prior.
- **CMA-ES on Schäfer et al.'s family.** Population 4, initial step 0.5, and a prior of the population's timings and uniform peaks.
  Post hoc, it also starts at generic assistance's fit in the family, with everything else unchanged
  (`scripts/icorr_amp_cmastart.py`, 3–35 and fixed 12 Nm).

### Answer rules and statistics

- Users who judge single features, users with phase weighting and users without a single preferred profile answer as decisively as the main users.
- Other shapes are phase-randomized surrogates (Theiler et al., 1992), heavy-tailed timing, or a mixture with 20 % from other families.
- Bootstrap intervals use 2000 draws, with users resampled within their walking subject.
- Fig. 4 shows 75 cases (condition x peak-torque setting). With a 95 % interval per case, about four would miss the true value by chance. Its thin bars therefore contain all cases at once in 95 % of 20,000 bootstrap draws, as a rank envelope (Besag et al., 1995). Each draw takes the strongest other optimizer in that draw and, in a row that merges conditions, the smallest lead.

### Further settings

- Curves are clipped at the range's top.
- The delay rests on two findings: a delay of 50–150 ms aligns assistance with higher joint velocities and can increase its positive work (Shepherd et al., Molinaro et al.).
- 3–12 Nm is the range on a 12 Nm device, and 3–35 Nm the published band at the median mass of 68.9 kg.
- PsyPref's answer model assumes squared distances in z, whereas the simulated users answer by RMS distances on the curve.
- The entropy proxy is H = log det Cov(z*) / 2.
- CMA-ES with population 4: about four generations fit into 24 comparisons.
- Speed is the first k at which PsyPref reaches the P_g that another optimizer reaches after 40 comparisons.
- Fig. 4's leads are computed with each user's sigma and lambda at the session's end. With fixed sigma and lambda, no lead changes sign, and leads move by at most 8.6 points.

### Numbers left out of the paper

All at k = 24 unless stated, in the order 3–12 Nm, 3–35 Nm, fixed 12 Nm where three values are given.

- Prior 1.5 times wider: −2.2, −1.7, −2.5 points.
- Best 8-weight profile: P_g 90.4, 94.6, 92.6 %, median distance 0.7, 1.4, 1.2 Nm.
- First k at which PsyPref reaches GP-BALD's 40-comparison level: 19 [15, 25], 12 [8, 15], 25 [18, 30].
- First k at which half of the users benefit, per setting: PsyPref 4, 2, 4; the strongest other optimizer none within 40, 5, 13 (at fixed 12 Nm, Arens' GP without its gate needs 8).
- Users who would prefer generic assistance to the same degree, pooled: PsyPref 0, GP-BALD 1, the strongest other optimizer 4, CMA-ES on Schäfer et al.'s family 12 %.
- PsyPref's profile chosen over that of CMA-ES on Schäfer et al.'s family: 75.8, 83.4, 71.6 %.
- 24 instead of 8 weights (3–35, fixed 12 Nm): GP-BALD −3.1 [−6.2, 0.0], −1.4 [−4.5, 1.4]; CMA-ES with population 4 −6.9 [−12.2, −1.7], +1.6 [−2.3, 5.5].
- At 8, 12, 16 and 24 weights, PsyPref leads GP-BALD by 2.1–7.9 points and CMA-ES with population 4 by 9.0–17.4, with all intervals above zero.
- After 150 comparisons, the 16-weight profile lies 0.35 [0.20, 0.51] and 0.55 [0.41, 0.71] JND closer to the preferred profile than the 8-weight profile (3–35, fixed 12 Nm), and 24 weights are not closer than 16.
- A different law of preferred peaks costs PsyPref at most 1.3 points at 3–35 Nm and 2.0 at 3–12 Nm.
- With 30 curves of Schäfer et al.'s family added to the prior, PsyPref leads CMA-ES on that family by 1.9 [0.5, 3.3] points on its shapes at fixed 12 Nm, where it trailed before.
- PsyPref's lead over the strongest other optimizer is positive for each of the five walking subjects, from 5.4 to 20.6 points.
- A prior from a population other than the users' costs PsyPref at most 2.1 and 5.6 points (users with Schäfer-family shapes, 3–35 and fixed 12 Nm).
