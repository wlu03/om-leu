# What choice-modelling reviewers expect vs what OM-LEU 2 does — survey (2026-09-05)

[unverified] marks details not opened at the primary source.

## 0. Two protocol facts that will dominate any review

- **Swissmetro**: each respondent's 9 SP scenarios are split *within person* across
  train / val / test; rows with an unavailable mode were dropped (16 % of rows, 188
  respondents).
- **LPMC**: 600 persons with ≥ 5 trips (a tiny, heavily selected sample of heavy
  travellers), chronological split *within one diary day*: the outbound trip is in
  train and the return trip in test. Hillel (STRC 2021, https://strc.ethz.ch/2021/Hillel.pdf):
  95.2 % of LPMC return trips share the chosen mode, 71.9 % of the data sits in
  matching sets with the same mode, and under trip-wise sampling > 50 % of validation
  trips have a same-mode match in training; the bias is largest for flexible non-linear
  classifiers. Our person random effects and person tastes are built to exploit exactly
  this. The LPMC and Swissmetro headline gaps are not interpretable until re-run under
  grouped splits.

## 1. Evaluation protocols

- **Grouped sampling (biggest gap).** Hillel et al. 2021 JOCM systematic review (five
  pitfalls: technique application, partitioning, performance estimation, hyper-parameter
  optimisation, interpretation); Martín-Baos et al. 2023 TR-C (arXiv:2301.04404): a
  person's responses are exclusively in train or test, also inside CV folds. Fix:
  respondent-grouped splits on Swissmetro (FE-RUMBoost uses 70/15/15 by individual,
  arXiv:2509.18047), household-grouped 5-fold on LPMC; keep the within-person split as a
  labelled "warm-start / panel" secondary condition. Effort 1 day + reruns. Impact high.
- **Canonical splits.** LPMC: train 2012/13–2013/14, test 2014/15, household-grouped CV
  inside train, full 81,086 trips (Hillel 2018; Martín-Baos; RUMBoost arXiv:2401.11954).
  Swissmetro: keep the full 10,728 rows with availability masks (FE-RUMBoost 10,692;
  Wang et al. 2026 10,719; RUMnet ~10k) or Biogeme's PURPOSE ∈ {1,3} filter (6,768);
  RUMnet 10-fold 80/10/10, FE-RUMBoost 70/15/15 by individual. Optima: Martín-Baos 70/30
  grouped by individual. Effort 2–4 days (LLM outcomes for 81k trips, cache per unique
  profile × alternative). Impact high.
- **Folds / seeds.** RUMnet 10-fold with paired t-tests; Wang 2026 10 bootstrap refits +
  McNemar + sign test; Martín-Baos 5-fold + held-out. Fix: 5 grouped folds × 2 seeds,
  paired bootstrap over persons, not events.
- **Metrics.** Cross-entropy (RUMBoost), GMPCA = exp(mean log-lik) (Martín-Baos eq. 26),
  ρ̄², LL / AIC / BIC for parametric parts, test market-share recovery (Martín-Baos Table
  10: all within 1 %), Brier (Lin, Yin, Liu 2026 arXiv:2602.21376), ECE with K = 15 after
  temperature scaling (Wang 2026). Our LLM-ranker NLLs (2.7–3.9) are uncalibrated and
  will be called unfair. Fix: add GMPCA, ρ̄², market shares, temperature-scale every
  probabilistic baseline before NLL / ECE. Effort 0.5–1 day.
- **External validation.** LPMC year-3 hold-out; suggestion: Swissmetro train on
  rail-recruited (SURVEY = 0), test on car-recruited (SURVEY = 1).

## 2. Behavioural validation

- **VOT / WTP.** Report per mode with bootstrap CIs against published ranges: Schmid et
  al. 2021 TR-A 150 (Zurich median VTTS car 30.6, PT 14.8 CHF/h); Axhausen, Hess et al.
  2008; Martín-Baos test-set MNL VOT Optima PT 12.1 / private 48.8 CHF/h, LPMC PT 42.4 /
  car 41.6 £/h, RF / XGBoost VOT invalid on 50–57 % of Optima rows; Wang 2026 Swissmetro
  84.4 CHF/h, LPMC 1.8 / 16.6 £/h; RUMBoost VoT distributions with 100 bootstrap refits;
  UK TAG data book [values unverified]. Our Optima non-identification is a specification
  issue: Martín-Baos identify it with TimePT, MarginalCostPT, distance_km (PT) and
  TimeCar, CostCarCHF, distance_km (car) and note the sign flips if distance is omitted
  from the car utility. Effort 1–2 days. Impact high.
- **Monotonicity audit.** Per-row monotonicity rate under cost / time perturbations
  (Wang 2026: TFMs violate in up to half of rows, negative VOT 40–65 %); Kim & Bansal
  2024 TR-B lattice networks. Report for structural, + residual, + mixture. Effort 0.5
  day. Impact high.
- **Elasticities, substitution, policy.** Aggregate arc elasticities by mode × attribute
  (Alt-GNN tables); cross-substitution under a fare change (RUMnet Fig. 5); Wang, Wang,
  Zhao 2020 (arXiv:1812.04528: aggregate DNN quantities reliable, disaggregate not);
  fare −20 % / PT time −10 % scenarios vs NL / mixed logit. Effort 1–2 days.
- **Interpretability.** TasteNet's "interpretability condition" verified on synthetic
  data with known tastes before real data; Martín-Baos validate WTP recovery on 12
  synthetic sets. Fix: semi-synthetic recovery experiment; report slot agreement / head
  correlation as an explicit metric. Effort 2 days.

## 3. Baselines we lack

| baseline | why | effort |
|---|---|---|
| RUMBoost / nested / FE-RUMBoost (pip `rumboost`, github.com/big-ucl/rumboost, functional-effects-model) | LPMC SOTA 0.673 CEL; FE runs Swissmetro + LPMC with individual splits — direct competitor to our person structure | 1–2 d |
| tuned LightGBM / XGBoost (Hyperopt 1,000 evals, grouped CV) | "baselines not tuned" is a desk-reject risk | 1 d compute |
| RUMnet, TasteNet, DeepMNL under RUMnet's 10-fold protocol (github.com/antoinedesir/rumnet, `choice-learn`) | Swissmetro log-loss MNL 0.830, TasteNet 0.554, RUMnet 0.546, RF 0.527 | 1 d |
| TabPFN v2 + Wang 2026 two-stage adapter (arXiv:2606.26432) | Swissmetro 63.6 → 76.3 % acc with VOT preserved; their row-level splits are leaky by Hillel's standard (a point we can make) | 1 d |
| cross-nested logit, latent class, panel mixed logit, ICLV (Biogeme examples 11a–c, 7/15/16, 12–13) | Krueger et al. 2021 JOCM: mixed logit gives no unconditional OOS gain over MNL, so beating them is not news | 1–2 d |
| Bayesian mixed logit (Bansal et al. 2020 TR-B, arXiv:1904.03647) | credible intervals on VOT for free | 1 d |
| Alt-GNN, ASU-DNN, L-MNL, ResLogit with SEs | elasticity tables, parameter SEs | 1 d |

## 4. Inference for ML–DCM hybrids

Farrell, Liang, Misra 2021 Econometrica (arXiv:1809.09953); Hetzenecker & Osterhaus 2024
(arXiv:2408.09560: robust SEs invalid for DNN-heterogeneous DCM, repeated sample
splitting stabilises); Chernozhukov et al. 2018 DML; Villarraga & Daziano 2025
(arXiv:2505.18077: SGLD Bayesian DL, 97 % coverage of MRS intervals vs 75–81 %); Huch &
Keane 2026 (arXiv:2603.24705). Practical precedent: RUMBoost 100-refit bootstrap on VoT;
Wang 2026 10 refits. "Acharya et al. 2026" DML for discrete choice: not found
[unverified]. Fix: respondent-cluster bootstrap (100 refits) for β and VOT; residual and
mixture weight fitted on out-of-fold structural predictions; coverage check on
semi-synthetic Swissmetro. Effort 1–2 days.

## 5. Datasets beyond mode choice

Expedia ICDM-2013 (Kaggle; 39 alternatives; RUMnet 10-fold log-loss 2.018; in
`choice-learn`); Amazon Reviews 2023; Twin-2K-500 (Marketing Science 2025, DOI
10.1287/mksc.2025.0262); conjoint with text (Goli & Singh; Wang et al. 2024); retail
scanner panels (Berbeglia et al. 2022 MS; Liu & Zhang 2026 SSRN TFM for choice, +8 % over
hierarchical Bayes [abstract only]); `choice-learn` bundle (Swissmetro, ModeCanada,
Train, Heating, Electricity, TaFeng, Expedia, LPMC, Bakery).

## 6. 2024–2026 papers a reviewer will name

Wang, Sun, Li, Fan, Zhuang 2026 (TFM adapter, Swissmetro / LPMC); Salvadé & Hillel
RUMBoost and FE-RUMBoost; Aouad & Désir RUMnet; Zhou et al. 2025 Alt-GNN; Liu, Li, Yin
2025 persona-embedding LLM (arXiv:2505.19003; Transportation Science venue [unverified]);
Lin, Yin, Liu 2026 Fenchel–Young perturbed utility; Alsaleh & Farooq 2025 LiTransMC;
Nishida et al. 2025 TRR; Mo et al. 2023; Xu & Jiao RAG (Travel Behaviour and Society
2027 [unverified]); SAPA; Villarraga & Daziano 2025; Kim & Bansal 2024; Hernández et al.
2024 ASS-NN; Bagheri et al. 2025 RUM-NN. Cautionary: Zhai et al. 2024 on LPMC / Optima
was withdrawn for a preprocessing error — reviewers are primed for leakage.

## Ranked top-10 gaps

1. Within-person splits on Swissmetro and LPMC (return-trip leakage; person effects
   trained on test respondents). Re-run grouped; report warm-start separately. High.
2. LPMC on 600 heavy travellers instead of the canonical 81k-trip year split. High.
3. Missing SOTA baselines: RUMBoost / FE-RUMBoost, tuned LightGBM (1,000 TPE evals),
   TabPFN + adapter, RUMnet under its protocol. High.
4. No VOT anywhere; Optima VOT unidentified (specification fix). High.
5. No monotonicity audit of the combined model. High, cheap.
6. Uncalibrated LLM-ranker NLLs; temperature-scale all probabilistic baselines. Medium-high.
7. No parameter / VOT uncertainty; residual fitted on in-sample predictions. Medium-high.
8. Missing GMPCA, ρ̄², market shares, elasticities, one policy scenario. Medium-high.
9. No head-to-head with 2025–26 LLM + choice papers on the same data. Medium-high.
10. "General choice model" claim rests on three mode-choice datasets. Medium-high.
