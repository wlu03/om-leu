# Optima and hybrid choice / ICLV models: literature survey

Compiled 2026-09-05 (web survey; most primary sources read from the EPFL
TRANSP-OR technical-report PDFs). "Unverified" marks details behind paywalls.

## 0. The dataset

`optima_dataset` — Bierlaire (2018) *Mode choice in Switzerland (Optima)*, case
study description, https://transp-or.epfl.ch/documents/technicalReports/CS_OptimaDescription.pdf.
Survey: Bierlaire et al. (2011), TRANSP-OR 110704, RP mail survey for CarPostal in
low-density French- and German-speaking Switzerland, 2009–2010. Trips aggregated
into home-based loops with a main mode: 1,906 loops in `optima.dat` (2,265-tour
version in the 2011–2014 papers). Choice 0 = PT, 1 = private motorised, 2 = soft;
rows with Choice = −1 (359) are excluded by every Biogeme example (1,547 kept).
Attributes: TimePT, WalkingTimePT, WaitingTimePT, NbTransf, TimeCar, CostPT,
MarginalCostPT (0 with GA, half with half-fare card), CostCarCHF, distance_km,
TripPurpose; socio-economics; childhood mobility; 54 Likert items (Envir01–06,
Mobil01–27, ResidCh01–07, LifSty01–14; 6 = n/a, −1/−2 missing). Times and costs of
unchosen alternatives were imputed from SBB and ViaMichelin.

## 1. Papers using Optima

### `iclv_procar_env_optima`, `latent_class_attitudinal_optima`
Atasoy, Glerum, Bierlaire (2013). "Attitudes towards mode choice in Switzerland."
*disP* 49(2):101–117, DOI 10.1080/02513625.2013.827518; report TRANSP-OR 110502.
Two latent attitudes from factor analysis of 17 statements: pro-car (indicators:
hard to take PT with children / luggage, dislike changing modes) and environmental
concern (increase fuel price, more PT even with taxes, global-warming concern, act
on GHG). Structural: A_car = c + θ·Ncars − θ·Educ + region dummies + η;
A_env = c + θ·Educ + θ·Nbikes + θ·Age·1(Age>45) + η. Linear-normal measurement
I_k = α_k + λ_k A + υ_k. Both latent variables enter the PT utility additively.
Full-information ML in Biogeme. Second model: 2-class latent class (independent vs
dependent) with item-response probabilities for three indicators.
Data: 2,265 tours / 1,763 respondents. Results: choice LL base logit −1067.4
(ρ² 0.490), ICLV −1069.8 (0.489), latent class −1032.5 (0.507); β_Acar −0.574
(t −3.5), β_Aenv +0.393 (t 3.0); VOT ≈ 30 CHF/h car, 12 CHF/h PT; 80/20 hold-out
share of P(chosen) > 0.5: 72.9 % base, 73.7 % ICLV, 75.0 % LC.
**ICLV does not improve choice fit over MNL; the latent-class model does.**

### `hcm_semiopen_comfort`
Glerum, Atasoy, Bierlaire (2014). "Using semi-open questions to integrate
perceptions in choice models." *J. Choice Modelling* 10:11–33; report TRANSP-OR
120325. Latent "perception of PT comfort" measured by free-text adjectives rated
−2…+2 by 25 external evaluators; enters as an interaction with PT travel time.
Logit LL −1153 (ρ̄² 0.443) vs HCM −1190…−1199 (0.422–0.427): logit fits slightly
better; identical 80/20 prediction. VOT ≈ 30 / 12 CHF/h.

### `latent_class_indicator_measurement`
Hurtubia, Nguyen, Glerum, Bierlaire (2014). *TR-A* 64:135–146, DOI
10.1016/j.tra.2014.03.010. Class-specific ordinal-logit measurement equations whose
latent response depends on socio-economics. Choice LL LCM1 (no indicators) −994.7,
LCM2 −1032.5, LCM3 −1006.7; better indicator LL and more plausible VOT with LCM3.

### `mis_endogeneity_interaction` / `iclv_carloving_benchmark`
Fernández-Antolín, Guevara, de Lapparent, Bierlaire (2016). *J. Choice Modelling*
20:1–15; report TRANSP-OR 160405. Omitted car-loving attitude interacting with car
time → endogeneity; Multiple Indicator Solution extended to interactions
(indicators Mobil10 "difficult to take PT with my children", Mobil13 "with my car
I can go wherever and whenever"), benchmarked against an ICLV with CarLoving ×
TimeCar. Data: 1,686 loops (27 % PT / 67 % car / 6 % soft). Logit LL −880.35
(ρ̄² 0.332) vs MIS −864.9 (0.342), LR 30.9; car-time elasticity −0.37 logit,
−0.48 MIS, −0.43 ICLV. MIS < 1 s vs ICLV ≈ 5 min.

### `biogeme_carlover_hybrid`
Bierlaire (2018). *Estimating choice models with latent variables with
PandasBiogeme*, TRANSP-OR 181227, https://transp-or.epfl.ch/documents/technicalReports/Bier18b.pdf;
examples b01–b07 (latent gallery). Factor analysis → "car lover" latent variable
loading on Envir01(−), Envir02(−), Envir03, Mobil11, Mobil14, Mobil16, Mobil17.
Structural regressors: age65+, >1 car, >1 bike, individual house, male, children,
GA, high education, piecewise income. Measurement: continuous (Envir01 normalised
β = −1) or ordered probit with symmetric thresholds. Choice: logit where the latent
variable scales the time coefficients, β_t^PT = β̂ exp(β_CL^PT x*), integrated
numerically over ε. Data: 1,547 rows. Results: choice-only mixture LL −1077.83
(ρ̄² 0.474); sequential LL −1092.59; full-information joint LL −18,406.15 (45
params); agent-effect model insignificant. Explicitly "not good specification
practice", numerically fragile.

### `biogeme_carcentric_hybrid`
Bierlaire, Ben-Akiva, Walker (2026). *Estimating hybrid choice models with
Biogeme*, TRANSP-OR 260814; Biogeme 3.3.5 gallery h01–h08. New "car-centric
attitude" (regressors: high/low education, top manager, car-oriented parents,
went to school by car) measured by Envir01/02/06, Mobil03/05/08/09/10, LifSty07;
enters the car utility additively; cost coefficient fixed to −1 with an estimated
scale so time coefficients are VOTs. 50,000 MLHS antithetic draws. Data: 889
loops. Baseline LL −508.62; sequential β_LV 3.80 (p 0.10); simultaneous Gaussian
β_LV 9.79 (t 5.0), ordered logit 5.30. Time coefficients essentially unchanged.

### `nl_optima_indicators`
Bierlaire (2018). *Calculating indicators with PandasBiogeme*, TRANSP-OR 181223.
Nested logit (PT + soft nested, µ = 1.53), LL −1298.5, ρ̄² 0.376; weighted shares
car 65.3 %, PT 28.1 %, soft 6.6 %; PT time elasticity −0.27, car cost −0.09.

### `lmnl_representation_learning`
Sifringer, Lurkin, Alahi (2020). *TR-B* 140:236–261; arXiv:1812.09747; code
github.com/BSifringer/EnhancedDCM. L-MNL: linear expert term on X plus a dense-NN
representation on disjoint Q. Optima (1,376 obs, 1,089/287): test accuracy Logit
76.7 %, L-MNL 79.2 %, ICLV 77.7 %, NN40 81.3 % (high variance).

### Others
- `iebsm_drift_ensemble` — Golik, Grzenda, Sienkiewicz (2024), IDA 2024,
  arXiv:2404.14017: stream/batch ensembles; Optima as 4-class problem, F1-macro
  0.36–0.44.
- `llm_textual_mode_choice` — Nishida et al. (2025) TRR: LLM across alternative
  sets incl. Optima (numbers unverified).
- `llm_prompt_learning_withdrawn` — Zhai et al. (2024), withdrawn.
- Checked and NOT using Optima: Alsaleh & Farooq 2025; Lederrey et al. 2021;
  Arkoudi et al. 2021; Fernández-Antolín et al. car-market papers.

## 2. Methodological papers on hybrid choice / ICLV

- `iclv_framework_benakiva_walker` — Ben-Akiva, Walker, Bernardino, Gopinath,
  Morikawa, Polydoropoulou (2002), in *In Perpetual Motion*, pp. 431–470.
  Structural X* = h(X; γ) + η, U = V(X, X*; β) + ε; measurement I = g(X, X*; α) + υ;
  f(y, I | X) = ∫ P(y | X, X*) f(I | X, X*) f(X* | X) dX*. Indicators directly in
  utility are non-causal; sequential without integration inconsistent; simultaneous
  ML consistent and efficient; three-step identification rule; forecasting uses
  ∫ P(y | X, X*) f(X* | X) dX* only.
- `hcm_progress_challenges` — Ben-Akiva, McFadden, Train, Walker, Bhat, Bierlaire,
  Bolduc et al. (2002), *Marketing Letters* 13(3):163–175.
- `grum_walker_benakiva` — Walker, Ben-Akiva (2002), *Math. Soc. Sci.*
  43(3):303–343: generalised RUM (logit kernel + latent variables + latent classes
  + RP/SP) by MSL.
- `hcm_logit_kernel_bolduc2005` — Bolduc, Ben-Akiva, Walker, Michaud (2005).
- `hcm_review_kim2014` — Kim, Rasouli, Timmermans (2014), *Procedia Env. Sci.*
  22:20–34.
- `hcm_handbook_abouzeid2014` — Abou-Zeid, Ben-Akiva (2014), *Handbook of Choice
  Modelling* ch. 17.
- `iclv_reduced_form_vij_walker` — Vij, Walker (2016), *TR-B* 90:192–217: every
  ICLV has a reduced-form mixed logit on the same observables with at least as good
  choice fit; ICLV is useful for efficiency, measurement-error / omitted-variable
  correction, structural identification, and policies that act on the latent
  construct.

## 3. Cross-cutting findings

1. On Optima every published latent-variable model fits the choice data about as
   well as, or slightly worse than, a plain logit with the same observables
   (Atasoy 2013, Glerum 2014, Fernández-Antolín 2016, Sifringer 2020), as Vij &
   Walker 2016 predict. Gains are interpretability, VOT / elasticity heterogeneity,
   endogeneity correction.
2. The canonical "car lover / car-centric" latent variable uses Envir01–03 and
   Mobil11/14/16/17 (2018 examples) or Envir01/02/06, Mobil03/05/08/09/10, LifSty07
   (2026 examples); the 2016 endogeneity paper uses Mobil10 and Mobil13.
3. Sample sizes vary (2,265 / 1,906 / 1,686 / 1,547 / 1,376 / 889), so cross-paper
   log-likelihoods are not comparable without re-estimation on a common subset.

Implementation in this repo: `methods/iclv_hybrid_choice` follows the 2018
indicator set with a Gaussian structural equation, linear-normal measurement, and
the latent variable entering the non-reference utilities additively (as in the 2026
gallery); prediction on held-out respondents uses the structural equation only.
